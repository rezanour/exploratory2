#include "Precomp.h"
#include "Pipeline.h"
#include "PipelineThread2.h"
#include "VertexBuffer.h"
#include "Texture2D.h"

//#define RENDER_WIREFRAME

using namespace Microsoft::WRL::Wrappers;

TRPipelineThread2::TRPipelineThread2(int id, TRPipeline* pipeline, SharedPipelineData* sharedData)
    : ID(id)
    , Pipeline(pipeline)
    , SharedData(sharedData)
{
}

TRPipelineThread2::~TRPipelineThread2()
{
    if (TheThread.IsValid())
    {
        // If the thread was created, assert that it's exited by now.
        // It's the responsibility of the Pipeline object to shut these down
        assert(WaitForSingleObject(TheThread.Get(), 0) == WAIT_OBJECT_0);
    }

    delete[] VertexMemory;
    delete[] PipelineTriangles;
}

bool TRPipelineThread2::Initialize()
{
    // Initialize the CommandReady event
    CommandReady.Attach(CreateEvent(nullptr, FALSE, FALSE, nullptr));
    if (!CommandReady.IsValid())
    {
        assert(false);
        return false;
    }

    // Create the thread
    TheThread.Attach(CreateThread(nullptr, 0, s_ThreadProc, this, 0, nullptr));
    if (!TheThread.IsValid())
    {
        assert(false);
        return false;
    }

    return true;
}

void TRPipelineThread2::QueueCommand(const std::shared_ptr<RenderCommand>& command)
{
    {
        auto lock = CommandsLock.Lock();
        Commands.push_back(command);
    }
    SetEvent(CommandReady.Get());
}

std::shared_ptr<RenderCommand> TRPipelineThread2::GetNextCommand()
{
    auto lock = CommandsLock.Lock();

    std::shared_ptr<RenderCommand> command;

    if (!Commands.empty())
    {
        command = Commands.front();
        Commands.pop_front();
    }

    return command;
}

DWORD CALLBACK TRPipelineThread2::s_ThreadProc(PVOID context)
{
    TRPipelineThread2* pThis = static_cast<TRPipelineThread2*>(context);
    pThis->ThreadProc();
    return 0;
}

void TRPipelineThread2::ThreadProc()
{
    const HANDLE hSignals[] = { SharedData->ShutdownEvent, CommandReady.Get() };
    DWORD result = WaitForMultipleObjects(_countof(hSignals), hSignals, FALSE, INFINITE);
    while (result != WAIT_OBJECT_0)
    {
        std::shared_ptr<RenderCommand> command_ptr = GetNextCommand();

        while (command_ptr)
        {
            // Cache a few things for easy access
            CurrentCommand = command_ptr.get();
            OutputVertexStride = CurrentCommand->OutputVertexStride;
            PositionOffset = -1;
            for (auto& attr : CurrentCommand->OutputVertexLayout)
            {
                if (strcmp(attr.Semantic, "SV_POSITION") == 0)
                {
                    PositionOffset = attr.ByteOffset;
                    break;
                }
            }
            assert(PositionOffset >= 0);

            // This vertex processing step includes:
            //   1. transform vertex (run vertex shader)
            //   2. triangle setup:
            //     a. clip to viewport
            //     b. transform to render target coordinates
            //   3. binning of triangles into screen space tiles
            ProcessVertices();

            // Join, and have last thread to complete signal fence
            if (++SharedData->JoinBarrier == SharedData->NumThreads)
            {
                // Last one through the barrier, reset the barrier
                SharedData->JoinBarrier = 0;

                // Signal fence
                Pipeline->NotifyCompletion(CurrentCommand->FenceValue);
            }

            command_ptr = GetNextCommand();
        }

        // Wait for either shutdown, or more work.
        result = WaitForMultipleObjects(_countof(hSignals), hSignals, FALSE, INFINITE);
    }
}

void TRPipelineThread2::ProcessVertices()
{
    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->VertexStartTime[ID] = time.QuadPart;
    }

    // Determine how many triangles our thread should process,
    // and determine base vertex
    int64_t triangleCount = CurrentCommand->NumTriangles / SharedData->NumThreads;
    if ((CurrentCommand->NumTriangles % SharedData->NumThreads) != 0)
    {
        ++triangleCount;
    }

    int64_t baseVertex = triangleCount * 3 * ID;

    // If total triangles was not evenly dividable by num threads,
    // we may have partial list to process.
    if ((baseVertex + triangleCount * 3) > CurrentCommand->NumVertices)
    {
        triangleCount = (CurrentCommand->NumVertices - baseVertex) / 3;
    }

    if (triangleCount == 0)
    {
        // Nothing to do
        return;
    }
        
    // Ensure the local memory pools are large enough

    // Allow for up to 10x vertices (to leave room for clipping, etc...)
    int64_t vertexCount = triangleCount * 3;
    if (vertexCount * 10 * OutputVertexStride > VertexMemoryCapacity)
    {
        VertexMemoryCapacity = vertexCount * 10 * OutputVertexStride;
        delete[] VertexMemory;
        VertexMemory = new uint8_t[VertexMemoryCapacity];
    }

    // Allow for up to 10x triangles (just index tuples)
    if (triangleCount * 10 > PipelineTriangleCapacity)
    {
        PipelineTriangleCapacity = triangleCount * 10;
        delete[] PipelineTriangles;
        PipelineTriangles = new PipelineTriangle[PipelineTriangleCapacity];
    }

    // Reset our master lists
    PipelineTriangleCount = 0;
    VertexMemoryOffset = 0;

    // blast through all vertices first, keeping high cache coherence and allowing
    // for SSE processing of 4 verts at once by the shader (if they choose to do so)
    CurrentCommand->VertexShader3(
        CurrentCommand->VSConstantBuffer,
        (const uint8_t*)CurrentCommand->VertexBuffer->GetVertices() + (baseVertex * CurrentCommand->InputVertexStride),   // input stream
        VertexMemory,                                               // output stream
        vertexCount);

    // Set current offset to the end of where the VS should have written to
    VertexMemoryOffset = vertexCount * OutputVertexStride;

    // perspective divide
    vertexCount = VertexMemoryOffset / OutputVertexStride;
    uint8_t* pVert = VertexMemory + PositionOffset;
    for (int64_t i = 0; i < vertexCount; ++i, pVert += OutputVertexStride)
    {
        float4* v = (float4*)pVert;

        // Divide by w
        *v /= v->w;
    }

    // Preclip and initialize PipelineTriangle list
    for (int64_t iTriangle = 0; iTriangle < triangleCount; ++iTriangle)
    {
        int64_t i1 = iTriangle * 3;
        int64_t i2 = iTriangle * 3 + 1;
        int64_t i3 = iTriangle * 3 + 2;

        // Find 3 SV_POSITION values
        float4* v1 = (float4*)(VertexMemory + i1 * OutputVertexStride + PositionOffset);
        float4* v2 = (float4*)(VertexMemory + i2 * OutputVertexStride + PositionOffset);
        float4* v3 = (float4*)(VertexMemory + i3 * OutputVertexStride + PositionOffset);

        int clipResult = PreClipTriangle(v1, v2, v3);
        if (clipResult == 0)
        {
            // Clip against viewport edges (appends resulting triangles to PipelineTriangles list)
            ClipTriangle(i1, i2, i3);
        }
        else if (clipResult == 1)
        {
            // trivial reject
        }
        else if (clipResult == 2)
        {
            // trivial accept
            AppendTriangle(i1, i2, i3);
        }
    }

    // Transform to render target coordinates
    float rtWidth = (float)CurrentCommand->RenderTarget->GetWidth();
    float rtHeight = (float)CurrentCommand->RenderTarget->GetHeight();
    vertexCount = VertexMemoryOffset / OutputVertexStride;
    pVert = VertexMemory + PositionOffset;
    for (int64_t i = 0; i < vertexCount; ++i, pVert += OutputVertexStride)
    {
        float4* v = (float4*)pVert;

        // Transform to render target coordinates
        v->x = (v->x * 0.5f + 0.5f) * rtWidth;
        v->y = (1.f - (v->y * 0.5f + 0.5f)) * rtHeight;
    }

    // TEMP: Rasterize
    uint32_t* renderTarget = (uint32_t*)CurrentCommand->RenderTarget->GetData();
    int rtPitch = CurrentCommand->RenderTarget->GetPitchInPixels();

    PipelineTriangle* triangle = PipelineTriangles;
    for (int64_t i = 0; i < PipelineTriangleCount; ++i, ++triangle)
    {
        // Find 3 SV_POSITION values so that we can transform to render target coords
        float4* v1 = (float4*)(VertexMemory + triangle->i1 * OutputVertexStride + PositionOffset);
        float4* v2 = (float4*)(VertexMemory + triangle->i2 * OutputVertexStride + PositionOffset);
        float4* v3 = (float4*)(VertexMemory + triangle->i3 * OutputVertexStride + PositionOffset);

        DDARastTriangle(v1, v2, v3, renderTarget, rtPitch);
    }

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->VertexStopTime[ID] = time.QuadPart;
    }
}

int TRPipelineThread2::PreClipTriangle(const float4* v1, const float4* v2, const float4* v3)
{
    uint32_t mask1 = 0;
    if (v1->x <= -1.f) mask1 |= 1;
    if (v1->x >= 1.f) mask1 |= 2;
    if (v1->y <= -1.f) mask1 |= 4;
    if (v1->y >= 1.f) mask1 |= 8;

    uint32_t mask2 = 0;
    if (v2->x <= -1.f) mask2 |= 1;
    if (v2->x >= 1.f) mask2 |= 2;
    if (v2->y <= -1.f) mask2 |= 4;
    if (v2->y >= 1.f) mask2 |= 8;

    uint32_t mask3 = 0;
    if (v3->x <= -1.f) mask3 |= 1;
    if (v3->x >= 1.f) mask3 |= 2;
    if (v3->y <= -1.f) mask3 |= 4;
    if (v3->y >= 1.f) mask3 |= 8;

    // if all 3 vertices failed the same plane, then
    // we can trivially reject the entire triangle
    if ((mask1 & mask2 & mask3) != 0)
    {
        // trivial reject
        return 1;
    }
    else if ((mask1 | mask2 | mask3) == 0)
    {
        // trivial accept
        return 2;
    }
    else
    {
        // need to clip
        return 0;
    }
}

void TRPipelineThread2::ClipTriangle(int64_t in_i1, int64_t in_i2, int64_t in_i3)
{
    int64_t indices[2][8];
    int iRead = 0, iWrite = 1, numIndices = 0, count = 0;

    indices[0][0] = in_i1;
    indices[0][1] = in_i2;
    indices[0][2] = in_i3;
    count = 3;

    // To avoid being right on the edge of the viewport,
    // which can lead to 1 pixel outside being rasterized,
    // we clip using 0.9999f instead of 1.f
    const float clipDist = 0.9999f;

    for (int axis = 0; axis < 3; ++axis)
    {
        // negative axis
        for (int i = 0; i < count; ++i)
        {
            int64_t i1 = indices[iRead][i];
            int64_t i2 = indices[iRead][(i + 1) % count];
            float* v1 = (float*)(VertexMemory + i1 * OutputVertexStride + PositionOffset);
            float* v2 = (float*)(VertexMemory + i2 * OutputVertexStride + PositionOffset);
            float d1 = -clipDist - v1[axis];
            float d2 = -clipDist - v2[axis];
            if (d1 < 0 && d2 < 0)
            {
                // both inside, keep #2
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 < 0 && d2 >= 0)
            {
                // clip & keep clipped point
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
            }
            else if (d1 >= 0 && d2 < 0)
            {
                // clip & keep clipped + i2
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
                indices[iWrite][numIndices++] = i2;
            }
            else if (d2 >= 0 && d2 >= 0)
            {
                // both outside, don't keep anything
            }
        }

        iRead = 1;
        iWrite = 0;
        count = numIndices;
        numIndices = 0;

        // positive axis
        for (int i = 0; i < count; ++i)
        {
            int64_t i1 = indices[iRead][i];
            int64_t i2 = indices[iRead][(i + 1) % count];
            float* v1 = (float*)(VertexMemory + i1 * OutputVertexStride + PositionOffset);
            float* v2 = (float*)(VertexMemory + i2 * OutputVertexStride + PositionOffset);
            float d1 = v1[axis] - clipDist;
            float d2 = v2[axis] - clipDist;
            if (d1 < 0 && d2 < 0)
            {
                // both inside, keep #2
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 < 0 && d2 >= 0)
            {
                // clip & keep clipped point
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
            }
            else if (d1 >= 0 && d2 < 0)
            {
                // clip & keep clipped + i2
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
                indices[iWrite][numIndices++] = i2;
            }
            else if (d2 >= 0 && d2 >= 0)
            {
                // both outside, don't keep anything
            }
        }

        iRead = 0;
        iWrite = 1;
        count = numIndices;
        numIndices = 0;
    }

    // indices[iRead] should now have a set of correctly ordered indices
    // which we can triangulate
    for (int i = 1; i < count - 1; ++i)
    {
        AppendTriangle(indices[iRead][0], indices[iRead][i], indices[iRead][i + 1]);
    }
}

int64_t TRPipelineThread2::ClipEdge(int64_t i1, int64_t i2, float d1, float d2)
{
    d1 = fabsf(d1);
    d2 = fabsf(d2);
    float lerp2 = d1 / (d1 + d2);
    float lerp1 = 1.f - lerp2;

    uint8_t* v1 = VertexMemory + i1 * OutputVertexStride;
    uint8_t* v2 = VertexMemory + i2 * OutputVertexStride;

    assert(VertexMemoryOffset + OutputVertexStride < VertexMemoryCapacity);

    int64_t iOutput = (VertexMemoryOffset / OutputVertexStride);
    VertexMemoryOffset += OutputVertexStride;
    uint8_t* output = VertexMemory + iOutput * OutputVertexStride;

    // interpolate attributes
    for (auto& attr : CurrentCommand->OutputVertexLayout)
    {
        // TODO: should avoid using switch here
        switch (attr.Type)
        {
        case VertexAttributeType::Float:
        {
            float* x1 = (float*)(v1 + attr.ByteOffset);
            float* x2 = (float*)(v2 + attr.ByteOffset);
            float* a = (float*)(output + attr.ByteOffset);
            *a = *x1 * lerp1 + *x2 * lerp2;
            break;
        }
        case VertexAttributeType::Float2:
        {
            float2* x1 = (float2*)(v1 + attr.ByteOffset);
            float2* x2 = (float2*)(v2 + attr.ByteOffset);
            float2* a = (float2*)(output + attr.ByteOffset);
            *a = *x1 * lerp1 + *x2 * lerp2;
            break;
        }
        case VertexAttributeType::Float3:
        {
            float3* x1 = (float3*)(v1 + attr.ByteOffset);
            float3* x2 = (float3*)(v2 + attr.ByteOffset);
            float3* a = (float3*)(output + attr.ByteOffset);
            *a = *x1 * lerp1 + *x2 * lerp2;
            break;
        }
        case VertexAttributeType::Float4:
        {
            float4* x1 = (float4*)(v1 + attr.ByteOffset);
            float4* x2 = (float4*)(v2 + attr.ByteOffset);
            float4* a = (float4*)(output + attr.ByteOffset);
            *a = *x1 * lerp1 + *x2 * lerp2;
            break;
        }

        default:
            assert(false);
            break;
        }
    }

    return iOutput;
}

void TRPipelineThread2::AppendTriangle(int64_t i1, int64_t i2, int64_t i3)
{
    assert(PipelineTriangleCount + 1 < PipelineTriangleCapacity);
    PipelineTriangles[PipelineTriangleCount++] = { i1, i2, i3 };
}

// rasterize a 2D triangle, binning the spans
void TRPipelineThread2::DDARastTriangle(const float4* v1, const float4* v2, const float4* v3, uint32_t* renderTarget, int pitch)
{
    struct rast_span
    {
        float x1, x2;
    };

    rast_span spans[2048];  // supports up to 2k height render target
    int next_span = 0;

    assert(CurrentCommand->RenderTarget->GetHeight() < _countof(spans));

    float2 v[] =
    {
        float2(v1->x, v1->y),
        float2(v2->x, v2->y),
        float2(v3->x, v3->y)
    };

    // Ensure properly clipped vertices
    assert(v[0].y >= 0 && v[1].y >= 0 && v[2].y >= 0);
    assert(v[0].y < CurrentCommand->RenderTarget->GetHeight() && v[1].y < CurrentCommand->RenderTarget->GetHeight() && v[2].y < CurrentCommand->RenderTarget->GetHeight());

    assert(v[0].x >= 0 && v[1].x >= 0 && v[2].x >= 0);
    assert(v[0].x < CurrentCommand->RenderTarget->GetWidth() && v[1].x < CurrentCommand->RenderTarget->GetWidth() && v[2].x < CurrentCommand->RenderTarget->GetWidth());

    // first, sort the vertices based on y
    int top = 0, mid = 0, bottom = 0;

    for (int i = 1; i < 3; ++i)
    {
        if (v[i].y > v[bottom].y) bottom = i;
        if (v[i].y < v[top].y) top = i;
    }

    mid = 3 - top - bottom;

    // get y value at each level
    float ytop = v[top].y;
    float ymid = v[mid].y;
    float ybottom = v[bottom].y;

    // first, rasterize from top to mid.
    // determine between mid & bottom which is left-most and right-most
    int left = (v[mid].x < v[bottom].x) ? mid : bottom;
    int right = mid + bottom - left;

    // determine slope step for each side that we'll be walking down.
    float step1 = (v[left].x - v[top].x) / (v[left].y - v[top].y);
    float step2 = (v[right].x - v[top].x) / (v[right].y - v[top].y);

    // store starting x value for both sides. They start at same location.
    float x1 = v[top].x;
    float x2 = v[top].x;

    int y1 = (int)ytop;
    int y2 = (int)ymid;

    for (int y = y1; y < y2; ++y)
    {
        rast_span* span = &spans[next_span++];
        span->x1 = x1;
        span->x2 = x2;

        x1 += step1;
        x2 += step2;
    }

    // next, we rasterize lower half of triangle from mid to bottom
    left = (v[top].x < v[mid].x) ? top : mid;
    right = top + mid - left;

    step1 = (v[bottom].x - v[left].x) / (v[bottom].y - v[left].y);
    step2 = (v[bottom].x - v[right].x) / (v[bottom].y - v[right].y);

    // x1 and x2 are left at their current values, since we're continuing
    // down the same triangle

    y1 = (int)ymid;
    y2 = (int)ybottom;

    for (int y = y1; y < y2; ++y)
    {
        rast_span* span = &spans[next_span++];
        span->x1 = x1;
        span->x2 = x2;

        x1 += step1;
        x2 += step2;
    }

#if 0
    // Spans determined, now set up lerping parameters (shared by all spans)
    vs_output v1 = GetSSEVertexAttributes(iFirstVertex);
    vs_output v2 = GetSSEVertexAttributes(iFirstVertex + 1);
    vs_output v3 = GetSSEVertexAttributes(iFirstVertex + 2);
    vec2 ab{ _mm_sub_ps(v2.Position.x, v1.Position.x),_mm_sub_ps(v2.Position.y, v1.Position.y) };
    vec2 bc{ _mm_sub_ps(v3.Position.x, v2.Position.x),_mm_sub_ps(v3.Position.y, v2.Position.y) };
    vec2 ac{ _mm_sub_ps(v3.Position.x, v1.Position.x),_mm_sub_ps(v3.Position.y, v1.Position.y) };
#endif

    int y = (int)ytop;

    rast_span* span = spans;
    for (; next_span > 0; --next_span, ++span)
    {
#ifdef RENDER_WIREFRAME

        renderTarget[y * pitch + (int)span->x1] = 0xFFFF0000;
        renderTarget[y * pitch + (int)span->x2] = 0xFFFF0000;

#else // RENDER_WIREFRAME

#ifdef ENABLE_LERPED_PATH
        for (int x = (int)span->x1; x < (int)span->x2; x += 4)
        {
            lerp_result lerp = sseLerp(
                v1, v2, v3, ab, bc, ac,
                vec2{ _mm_set_ps((float)x, x + 1.f, x + 2.f, x + 3.f), _mm_set1_ps((float)y) });

            int imask = _mm_movemask_ps(lerp.mask);

            vs_output input{ lerp.position, lerp.color };
            vec4 frags = command->PixelShader(command->PixelShader, input);

            uint32_t colors[4];
            ConvertFragsToColors(frags, colors);

            int index = y * pitch + x + 3;
            for (int i = 0; i < 4; ++i, --index)
            {
                if ((imask & 0x01) == 0)
                {
                    renderTarget[index] = colors[i];
                }
                imask >>= 1;
            }
        }
#else // ENABLE_LERPED_PATH
        for (int x = (int)span->x1; x < (int)span->x2; ++x)
        {
            renderTarget[y * pitch + x] = 0xFFFF0000;
        }
#endif // ENABLE_LERPED_PATH

#endif // RENDER_WIREFRAME
        ++y;
    }
}
