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

            RenderTarget = (uint32_t*)CurrentCommand->RenderTarget->GetData();
            RTWidth = CurrentCommand->RenderTarget->GetWidth();
            RTHeight = CurrentCommand->RenderTarget->GetHeight();
            RTPitch = CurrentCommand->RenderTarget->GetPitchInPixels();

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

        // Appends to the pipeline triangle list
        ClipTriangle(i1, i2, i3);
    }

    // Transform to render target coordinates
    float rtWidth = (float)RTWidth;
    float rtHeight = (float)RTHeight;
    vertexCount = VertexMemoryOffset / OutputVertexStride;
    pVert = VertexMemory + PositionOffset;
    for (int64_t i = 0; i < vertexCount; ++i, pVert += OutputVertexStride)
    {
        float4* v = (float4*)pVert;

        // Transform to render target coordinates
        v->x = (v->x * 0.5f + 0.5f) * rtWidth;
        v->y = (1.f - (v->y * 0.5f + 0.5f)) * rtHeight;
    }

    // Rasterize
    // TODO: Since rasterization is an expensive operation, and the final triangle and
    // resulting pixel load per thread may be wildly uneven, it's probably better to bin
    // these triangles into a large shared list, and then have each thread pull & rasterize
    // from that list. Or, alternatively, we could bin the triangles into screen tiles (before clipping)
    // and then the threads process the tiles in parallel.
    PipelineTriangle* triangle = PipelineTriangles;
    for (int64_t i = 0; i < PipelineTriangleCount; ++i, ++triangle)
    {
        // Find 3 SV_POSITION values so that we can transform to render target coords
        float4* v1 = GetVertexPosition<float4>(triangle->i1);
        float4* v2 = GetVertexPosition<float4>(triangle->i2);
        float4* v3 = GetVertexPosition<float4>(triangle->i3);

        DDARastTriangle(v1, v2, v3);
    }

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->VertexStopTime[ID] = time.QuadPart;
    }
}

TRPipelineThread2::PreClipResult TRPipelineThread2::PreClipTriangle(const float4* v1, const float4* v2, const float4* v3)
{
#if 0
    // trivial reject back-facing
    float3 ab = *(const float3*)v2 - *(const float3*)v1;
    float3 ac = *(const float3*)v3 - *(const float3*)v1;
    float3 norm = cross(ab, ac);
    if (dot(norm, float3(0, 0, -1)) <= 0)
    {
        return 1;
    }
#endif

    // now check for completely outside any plane
    uint32_t mask1 = 0;
    if (v1->x <= -1.f) mask1 |= 0x01;
    if (v1->x >= 1.f) mask1 |= 0x02;
    if (v1->y <= -1.f) mask1 |= 0x04;
    if (v1->y >= 1.f) mask1 |= 0x08;
    if (v1->z <= -1.f) mask1 |= 0x10;
    if (v1->z >= 1.f) mask1 |= 0x20;

    uint32_t mask2 = 0;
    if (v2->x <= -1.f) mask2 |= 0x01;
    if (v2->x >= 1.f) mask2 |= 0x02;
    if (v2->y <= -1.f) mask2 |= 0x04;
    if (v2->y >= 1.f) mask2 |= 0x08;
    if (v2->z <= -1.f) mask2 |= 0x10;
    if (v2->z >= 1.f) mask2 |= 0x20;

    uint32_t mask3 = 0;
    if (v3->x <= -1.f) mask3 |= 0x01;
    if (v3->x >= 1.f) mask3 |= 0x02;
    if (v3->y <= -1.f) mask3 |= 0x04;
    if (v3->y >= 1.f) mask3 |= 0x08;
    if (v3->z <= -1.f) mask3 |= 0x10;
    if (v3->z >= 1.f) mask3 |= 0x20;

    if ((mask1 & mask2 & mask3) != 0)
    {
        // if all 3 vertices failed the same plane, then
        // we can trivially reject the entire triangle
        return PreClipResult::TrivialReject;
    }
    else if ((mask1 | mask2 | mask3) == 0)
    {
        // if all 3 vertices are completely inside
        // all of the clip planes, we can just keep the whole triangle
        return PreClipResult::TrivialAccept;
    }
    else
    {
        // Needs some amount of clipping
        return PreClipResult::NeedsClipping;
    }
}

void TRPipelineThread2::ClipTriangle(int64_t in_i1, int64_t in_i2, int64_t in_i3)
{
    // Find 3 SV_POSITION values
    float4* in_v1 = GetVertexPosition<float4>(in_i1);
    float4* in_v2 = GetVertexPosition<float4>(in_i2);
    float4* in_v3 = GetVertexPosition<float4>(in_i3);

    PreClipResult preClipResult = PreClipTriangle(in_v1, in_v2, in_v3);
    switch (preClipResult)
    {
    case PreClipResult::TrivialAccept:
        AppendTriangle(in_i1, in_i2, in_i3); // append the whole triangle & we're done
        return;

    default:
    case PreClipResult::TrivialReject:
        // Not keeping any of it. Return
        return;

    case PreClipResult::NeedsClipping:
        // break and continue this function
        break;
    }

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
            float* v1 = GetVertexPosition<float>(i1);
            float* v2 = GetVertexPosition<float>(i2);
            float d1 = -clipDist - v1[axis];
            float d2 = -clipDist - v2[axis];
            if (d1 < 0 && d2 < 0)
            {
                assert(numIndices < _countof(indices[iWrite]));
                // both inside, keep #2
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 < 0 && d2 >= 0)
            {
                assert(numIndices < _countof(indices[iWrite]));
                // clip & keep clipped point
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
            }
            else if (d1 >= 0 && d2 < 0)
            {
                // clip & keep clipped + i2
                assert(numIndices + 1 < _countof(indices[iWrite]));
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 >= 0 && d2 >= 0)
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
            float* v1 = GetVertexPosition<float>(i1);
            float* v2 = GetVertexPosition<float>(i2);
            float d1 = v1[axis] - clipDist;
            float d2 = v2[axis] - clipDist;
            if (d1 < 0 && d2 < 0)
            {
                assert(numIndices < _countof(indices[iWrite]));
                // both inside, keep #2
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 < 0 && d2 >= 0)
            {
                assert(numIndices < _countof(indices[iWrite]));
                // clip & keep clipped point
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
            }
            else if (d1 >= 0 && d2 < 0)
            {
                assert(numIndices + 1 < _countof(indices[iWrite]));
                // clip & keep clipped + i2
                int64_t iClipped = ClipEdge(i1, i2, d1, d2);
                indices[iWrite][numIndices++] = iClipped;
                indices[iWrite][numIndices++] = i2;
            }
            else if (d1 >= 0 && d2 >= 0)
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
        // TODO: should avoid using switch here?
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

static inline int rzround(float f)
{
    return (int)(f + 0.5f);
}

// rasterize a 2D triangle, binning the spans
void TRPipelineThread2::DDARastTriangle(const float4* v1, const float4* v2, const float4* v3)
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

#if 0
    // Ensure properly clipped vertices
    assert(v[0].y >= 0 && v[1].y >= 0 && v[2].y >= 0);
    assert(v[0].y < CurrentCommand->RenderTarget->GetHeight() && v[1].y < CurrentCommand->RenderTarget->GetHeight() && v[2].y < CurrentCommand->RenderTarget->GetHeight());

    assert(v[0].x >= 0 && v[1].x >= 0 && v[2].x >= 0);
    assert(v[0].x < CurrentCommand->RenderTarget->GetWidth() && v[1].x < CurrentCommand->RenderTarget->GetWidth() && v[2].x < CurrentCommand->RenderTarget->GetWidth());
#endif

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
    float step1 = (v[left].x - v[top].x) / (v[left].y - ytop);
    float step2 = (v[right].x - v[top].x) / (v[right].y - ytop);

    // store starting x value for both sides. They start at same location.
    float x1 = v[top].x;
    float x2 = v[top].x;

    int y1 = rzround(ytop);
    int y2 = rzround(ymid);

    for (int y = y1; y < y2; ++y)
    {
        rast_span* span = &spans[next_span++];
        span->x1 = x1;
        span->x2 = x2;

        // only advance if we have another span,
        // otherwise, we'll compute our new steps to take first
        if (y < y2 - 1)
        {
            x1 += step1;
            x2 += step2;
        }
    }

    // next, we rasterize lower half of triangle from mid to bottom
    // unless there is no lower half :)
    if (ymid + 0.001f < ybottom)
    {
        step1 = (v[bottom].x - x1) / (v[bottom].y - ymid);
        step2 = (v[bottom].x - x2) / (v[bottom].y - ymid);

        // x1 and x2 are left at their current values, since we're continuing
        // down the same triangle

        y1 = rzround(ymid);
        y2 = rzround(ybottom);

        for (int y = y1; y < y2; ++y)
        {
            // step first, then record
            x1 += step1;
            x2 += step2;

            rast_span* span = &spans[next_span++];
            span->x1 = x1;
            span->x2 = x2;
        }
    }

#define ENABLE_LERPED_PATH

#ifdef ENABLE_LERPED_PATH
    // Spans determined, now set up lerping parameters (shared by all spans)
    uint8_t* v1Base = (uint8_t*)v1 - PositionOffset;
    uint8_t* v2Base = (uint8_t*)v2 - PositionOffset;
    uint8_t* v3Base = (uint8_t*)v3 - PositionOffset;
    vec2 a = { _mm_set1_ps(v1->x), _mm_set1_ps(v1->y) };
    vec2 b = { _mm_set1_ps(v2->x), _mm_set1_ps(v2->y) };
    vec2 c = { _mm_set1_ps(v3->x), _mm_set1_ps(v3->y) };
    vec2 ab = { _mm_sub_ps(b.x, a.x), _mm_sub_ps(b.y, a.y) };
    vec2 bc = { _mm_sub_ps(c.x, b.x), _mm_sub_ps(c.y, b.y) };
    vec2 ac = { _mm_sub_ps(c.x, a.x), _mm_sub_ps(c.y, a.y) };
    uint8_t* scratchVertex = VertexMemory + VertexMemoryOffset;
#endif

    int y = rzround(ytop);

    rast_span* span = spans;
    for (; next_span > 0; --next_span, ++span)
    {
        if (y < 0) continue;
        if (y >= RTHeight) break;

        int start = rzround(span->x1);
        int end = rzround(span->x2);

#ifdef RENDER_WIREFRAME

        if (start >= 0)
            RenderTarget[y * RTPitch + start] = 0xFFFF0000;

        if (end < RTWidth)
            RenderTarget[y * RTPitch + end] = 0xFFFF0000;

#else // RENDER_WIREFRAME

#ifdef ENABLE_LERPED_PATH

        __m128 vecy = _mm_set1_ps((float)y);

        for (int x = start; x < end; x += 4)
        {
            if (x < 0) continue;
            if (y >= RTWidth) break;

            vec2 p{ _mm_set_ps((float)x, x + 1.f, x + 2.f, x + 3.f), vecy };

            bary_result bary = ComputeBarycentricCoords(a, b, ab, bc, ac, p);
            int imask = _mm_movemask_ps(bary.mask);

            float w1[4], w2[4], w3[4];
            _mm_storeu_ps(w1, bary.xA);
            _mm_storeu_ps(w2, bary.xB);
            _mm_storeu_ps(w3, bary.xC);

            int index = y * RTPitch + x + 3;
            for (int i = 0; i < 4; ++i, --index)
            {
                if ((imask & 0x01) == 0)
                {
                    Lerp(v1Base, v2Base, v3Base, w1[i], w2[i], w3[i], scratchVertex);
                    float4 color = CurrentCommand->PixelShader3(CurrentCommand->PSConstantBuffer, scratchVertex);
                    RenderTarget[index] =
                        (uint32_t)(uint8_t)(color.w * 255.f) << 24 |
                        (uint32_t)(uint8_t)(color.z * 255.f) << 16 |
                        (uint32_t)(uint8_t)(color.y * 255.f) << 8 |
                        (uint32_t)(uint8_t)(color.x * 255.f);
                }
                imask >>= 1;
            }
        }
#else // ENABLE_LERPED_PATH
        for (int x = start; x < end; ++x)
        {
            if (x < 0) continue;
            if (x >= RTWidth) break;

            RenderTarget[y * RTPitch + x] = 0xFFFF0000;
        }
#endif // ENABLE_LERPED_PATH

#endif // RENDER_WIREFRAME
        ++y;
    }
}

template <typename T>
inline T* TRPipelineThread2::GetVertexAttribute(int64_t i, int byteOffset)
{
    return (T*)(VertexMemory + byteOffset + (i * OutputVertexStride));
}

template <typename T>
inline T* TRPipelineThread2::GetVertexPosition(int64_t i)
{
    return (T*)(VertexMemory + PositionOffset + (i * OutputVertexStride));
}
                                                    
// Compute barycentric coordinates (lerp weights) for 4 samples at once.
bary_result __vectorcall TRPipelineThread2::ComputeBarycentricCoords(
    const vec2 a, const vec2 b,                     // first 2 vertices in vectorized form
    const vec2 ab, const vec2 bc, const vec2 ac,    // edges of triangle, in vectorized form
    const vec2 p)                                   // 4 pixels to compute lerp values for
{
    // Find barycentric coordinates of P (wA, wB, wC)
    vec2 ap, bp;
    ap.x = _mm_sub_ps(p.x, a.x);
    ap.y = _mm_sub_ps(p.y, a.y);
    bp.x = _mm_sub_ps(p.x, b.x);
    bp.y = _mm_sub_ps(p.y, b.y);

    // float3 wC = cross(ab, ap);
    // expand out to:
    //    wC.x = ab.y * ap.z - ap.y * ab.z;
    //    wC.y = ab.z * ap.x - ap.z * ab.x;
    //    wC.z = ab.x * ap.y - ap.x * ab.y;
    // since we are doing in screen space, z is always 0 so simplify:
    //    wC.x = 0
    //    wC.y = 0
    //    wC.z = ab.x * ap.y - ap.x * ab.y
    // or, simply:
    //    wC = abx * apy - apx * aby;
    __m128 wC = _mm_sub_ps(_mm_mul_ps(ab.x, ap.y), _mm_mul_ps(ap.x, ab.y));
    __m128 mask1 = _mm_cmplt_ps(wC, _mm_setzero_ps());

    // Use same reduction for wB & wA
    __m128 wB = _mm_sub_ps(_mm_mul_ps(ap.x, ac.y), _mm_mul_ps(ac.x, ap.y));
    __m128 mask2 = _mm_cmplt_ps(wB, _mm_setzero_ps());

    __m128 wA = _mm_sub_ps(_mm_mul_ps(bc.x, bp.y), _mm_mul_ps(bp.x, bc.y));
    __m128 mask3 = _mm_cmplt_ps(wA, _mm_setzero_ps());

    bary_result result;
    result.mask = _mm_or_ps(mask1, _mm_or_ps(mask2, mask3));

    // Use a similar reduction for cross of ab x ac (to find unnormalized normal)
    __m128 norm = _mm_sub_ps(_mm_mul_ps(ab.x, ac.y), _mm_mul_ps(ac.x, ab.y));
    norm = _mm_rcp_ps(norm);

    // to find length of this cross product, which already know is purely in the z
    // direction, is just the length of the z component, which is the exactly the single
    // channel norm we computed above. Similar logic is used for lengths of each of
    // the weights, since they are all single channel vectors, the one channel is exactly
    // the length.

    result.xA = _mm_mul_ps(wA, norm);
    result.xB = _mm_mul_ps(wB, norm);
    result.xC = _mm_mul_ps(wC, norm);

    return result;
}


void TRPipelineThread2::Lerp(
    uint8_t* v1, uint8_t* v2, uint8_t* v3,          // triangle vertex (top of each vertex struct)
    float w1, float w2, float w3,                   // lerping weights
    uint8_t* output)                                // scratch memory to write lerped values to
{
    // interpolate attributes
    for (auto& attr : CurrentCommand->OutputVertexLayout)
    {
        // TODO: should avoid using switch here?
        switch (attr.Type)
        {
        case VertexAttributeType::Float:
        {
            float* x1 = (float*)(v1 + attr.ByteOffset);
            float* x2 = (float*)(v2 + attr.ByteOffset);
            float* x3 = (float*)(v3 + attr.ByteOffset);
            float* out = (float*)(output + attr.ByteOffset);
            *out = *x1 * w1 + *x2 * w2 + *x3 * w3;
            break;
        }
        case VertexAttributeType::Float2:
        {
            float2* x1 = (float2*)(v1 + attr.ByteOffset);
            float2* x2 = (float2*)(v2 + attr.ByteOffset);
            float2* x3 = (float2*)(v3 + attr.ByteOffset);
            float2* out = (float2*)(output + attr.ByteOffset);
            *out = *x1 * w1 + *x2 * w2 + *x3 * w3;
            break;
        }
        case VertexAttributeType::Float3:
        {
            float3* x1 = (float3*)(v1 + attr.ByteOffset);
            float3* x2 = (float3*)(v2 + attr.ByteOffset);
            float3* x3 = (float3*)(v3 + attr.ByteOffset);
            float3* out = (float3*)(output + attr.ByteOffset);
            *out = *x1 * w1 + *x2 * w2 + *x3 * w3;
            break;
        }
        case VertexAttributeType::Float4:
        {
            float4* x1 = (float4*)(v1 + attr.ByteOffset);
            float4* x2 = (float4*)(v2 + attr.ByteOffset);
            float4* x3 = (float4*)(v3 + attr.ByteOffset);
            float4* out = (float4*)(output + attr.ByteOffset);
            *out = *x1 * w1 + *x2 * w2 + *x3 * w3;
            break;
        }

        default:
            assert(false);
            break;
        }
    }
}

