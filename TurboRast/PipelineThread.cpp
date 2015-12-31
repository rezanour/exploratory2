#include "Precomp.h"
#include "Pipeline.h"
#include "PipelineThread.h"
#include "VertexBuffer.h"
#include "Texture2D.h"

using namespace Microsoft::WRL::Wrappers;

// Compute barycentric coordinates (lerp weights) for 4 samples at once.
// The computation is done in 2 dimensions (screen space).
// in: a (ax, ay), b (bx, by) and c (cx, cy) are the 3 vertices of the triangle.
//     p (px, py) is the point to compute barycentric coordinates for
// out: wA, wB, wC are the weights at vertices a, b, and c
//      mask will contain a 0 (clear) if the value is computed. It will be 0xFFFFFFFF (set) if invalid
__forceinline bary_result __vectorcall sseBary2D(
    const vec4 a, const vec4 b,
    const vec2 ab, const vec2 bc, const vec2 ac,
    const vec2 p)
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

struct alignas(16) lerp_result
{
    __m128 mask;
    vec4 position;
    vec3 color;
};

__forceinline lerp_result __vectorcall sseLerp(
    const vs_output v1, const vs_output v2, const vs_output v3,
    const vec2 ab, const vec2 bc, const vec2 ac,
    const vec2 p)
{
    bary_result bary = sseBary2D(
        v1.Position, v2.Position,
        ab, bc, ac,
        p);

    lerp_result result;
    result.mask = bary.mask;

    // Interpolate all the attributes for these 4 pixels
    result.position.x = _mm_add_ps(_mm_mul_ps(v1.Position.x, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Position.x, bary.xB), _mm_mul_ps(v3.Position.x, bary.xC)));
    result.position.y = _mm_add_ps(_mm_mul_ps(v1.Position.y, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Position.y, bary.xB), _mm_mul_ps(v3.Position.y, bary.xC)));
    result.position.z = _mm_add_ps(_mm_mul_ps(v1.Position.z, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Position.z, bary.xB), _mm_mul_ps(v3.Position.z, bary.xC)));
    result.position.w = _mm_set1_ps(1.f);

    result.color.x = _mm_add_ps(_mm_mul_ps(v1.Color.x, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Color.x, bary.xB), _mm_mul_ps(v3.Color.x, bary.xC)));
    result.color.y = _mm_add_ps(_mm_mul_ps(v1.Color.y, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Color.y, bary.xB), _mm_mul_ps(v3.Color.y, bary.xC)));
    result.color.z = _mm_add_ps(_mm_mul_ps(v1.Color.z, bary.xA), _mm_add_ps(_mm_mul_ps(v2.Color.z, bary.xB), _mm_mul_ps(v3.Color.z, bary.xC)));

    return result;
}

TRPipelineThread::TRPipelineThread(int id, TRPipeline* pipeline, SharedPipelineData* sharedData)
    : ID(id)
    , Pipeline(pipeline)
    , SharedData(sharedData)
{
}

TRPipelineThread::~TRPipelineThread()
{
    if (TheThread.IsValid())
    {
        // If the thread was created, assert that it's exited by now.
        // It's the responsibility of the Pipeline object to shut these down
        assert(WaitForSingleObject(TheThread.Get(), 0) == WAIT_OBJECT_0);
    }
}

bool TRPipelineThread::Initialize()
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

void TRPipelineThread::QueueCommand(const std::shared_ptr<RenderCommand>& command)
{
    {
        auto lock = CommandsLock.Lock();
        Commands.push_back(command);
    }
    SetEvent(CommandReady.Get());
}

DWORD CALLBACK TRPipelineThread::s_ThreadProc(PVOID context)
{
    TRPipelineThread* pThis = static_cast<TRPipelineThread*>(context);
    pThis->ThreadProc();
    return 0;
}

void TRPipelineThread::ThreadProc()
{
    const HANDLE hSignals[] = { SharedData->ShutdownEvent, CommandReady.Get() };
    DWORD result = WaitForMultipleObjects(_countof(hSignals), hSignals, FALSE, INFINITE);
    while (result != WAIT_OBJECT_0)
    {
        std::shared_ptr<RenderCommand> command = GetNextCommand();

        while (command)
        {
            // Initialization
            JoinAndDoSerialInitialization(command);

            // Stage 1: Vertex processing
            ProcessVertices(command);

            // Stage 2: Triangle processing

            // TODO: There are multiple strategies we can consider here. Should make this code modular
            // enough to be able to experiment with each strategy here (or switch between them at runtime based
            // on some criteria). The strategies are:
            //   1. Process each triangle all the way down to pixels. Do this in parallel for different triangles on each thread
            //   2. Bin triangles into screen space tiles, and then process tiles in parallel across threads.
            //
            //   * When sort order matters (blending, no depth reject, etc...), we need to be careful to maintain draw order
            //   * Should pixel processing be lifted out separately from triangle processing? How can we efficiently do this?

            switch (SharedData->CurrentRastStrategy)
            {
            case RastStrategy::OneTrianglePerThread:
                ProcessOneTrianglePerThread(command);
                break;

            case RastStrategy::OneTilePerThread:
                ProcessOneTilePerThread(command);
                break;

            default:
                assert(false);
                break;
            }

            // Stage 3: Fragment/pixel processing (if applicable)

            // Completion
            JoinAndDoSerialCompletion(command);

            command = GetNextCommand();
        }

        // Wait for either shutdown, or more work.
        result = WaitForMultipleObjects(_countof(hSignals), hSignals, FALSE, INFINITE);
    }
}

std::shared_ptr<RenderCommand> TRPipelineThread::GetNextCommand()
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

void TRPipelineThread::ProcessVertices(const std::shared_ptr<RenderCommand>& command)
{
    uint64_t numBlocks = command->VertexBuffer->GetNumBlocks();
    int rtWidth = command->RenderTarget->GetWidth();
    int rtHeight = command->RenderTarget->GetHeight();
    SSEVSOutput* VSOutputs = SharedData->VSOutputs;

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->VertexStartTime[ID] = time.QuadPart;
    }

    uint64_t iVertexBlock = SharedData->CurrentVertex++;
    while (iVertexBlock < numBlocks)
    {
        auto& vin = command->VertexBuffer->GetBlocks()[iVertexBlock];
        vs_input input
        {
            vec3{ _mm_load_ps(vin.Position_x), _mm_load_ps(vin.Position_y), _mm_load_ps(vin.Position_z) },
            vec3{ _mm_load_ps(vin.Color_x), _mm_load_ps(vin.Color_y), _mm_load_ps(vin.Color_z) },
        };
        vs_output output = command->VertexShader(command->VSConstantBuffer, input);

        // Divide by w
        output.Position.x = _mm_div_ps(output.Position.x, output.Position.w);
        output.Position.y = _mm_div_ps(output.Position.y, output.Position.w);
        output.Position.z = _mm_div_ps(output.Position.z, output.Position.w);
        output.Position.w = _mm_set1_ps(1.f);

        // Scale to viewport
        __m128 point5 = _mm_set1_ps(0.5f);
        output.Position.x = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(output.Position.x, point5), point5), _mm_set1_ps((float)rtWidth));
        output.Position.y = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.f), _mm_add_ps(_mm_mul_ps(output.Position.y, point5), point5)), _mm_set1_ps((float)rtHeight));

        // Store back result
        _mm_store_ps(VSOutputs[iVertexBlock].Position_x, output.Position.x);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_y, output.Position.y);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_z, output.Position.z);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_w, output.Position.w);

        _mm_store_ps(VSOutputs[iVertexBlock].Color_x, output.Color.x);
        _mm_store_ps(VSOutputs[iVertexBlock].Color_y, output.Color.y);
        _mm_store_ps(VSOutputs[iVertexBlock].Color_z, output.Color.z);

        iVertexBlock = SharedData->CurrentVertex++;
    }

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->VertexStopTime[ID] = time.QuadPart;
    }
}

void TRPipelineThread::ProcessOneTrianglePerThread(const std::shared_ptr<RenderCommand>& command)
{
    int rtWidth = command->RenderTarget->GetWidth();
    int rtHeight = command->RenderTarget->GetHeight();
    int rtPitchInPixels = command->RenderTarget->GetPitchInPixels();
    uint32_t* renderTarget = (uint32_t*)command->RenderTarget->GetData();

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->TriangleStartTime[ID] = time.QuadPart;
    }

    Triangle triangle;

    uint64_t iTriangle = SharedData->CurrentTriangle++;
    while (iTriangle < command->NumTriangles)
    {
        uint64_t iFirstVertex = iTriangle * 3;

        float4 verts[3];
        GetVertexAttributes(iFirstVertex, &verts[0], nullptr);
        GetVertexAttributes(iFirstVertex + 1, &verts[1], nullptr);
        GetVertexAttributes(iFirstVertex + 2, &verts[2], nullptr);

        DDARastTriangle(command, iFirstVertex, verts, renderTarget, rtWidth, rtHeight, rtPitchInPixels);

        iTriangle = SharedData->CurrentTriangle++;
    }

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->TriangleStopTime[ID] = time.QuadPart;
    }
}

void TRPipelineThread::ProcessOneTilePerThread(const std::shared_ptr<RenderCommand>& command)
{
    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->TriangleStartTime[ID] = time.QuadPart;
    }

    // Parallel Bin the triangles first

    Triangle triangle;

    uint64_t iTriangle = SharedData->CurrentTriangle++;
    while (iTriangle < command->NumTriangles)
    {
        float4 p1, p2, p3;
        GetVertexAttributes(iTriangle * 3, &p1, nullptr);
        GetVertexAttributes(iTriangle * 3 + 1, &p2, nullptr);
        GetVertexAttributes(iTriangle * 3 + 2, &p3, nullptr);

        triangle.iTriangle = iTriangle;
        triangle.Next = nullptr;

        triangle.p1 = float2(p1.x, p1.y);
        triangle.p2 = float2(p2.x, p2.y);
        triangle.p3 = float2(p3.x, p3.y);

        // edge equation Bx + Cy = 0, where B & C are computed from slope as B = (y1 - y0) and C = -(x1 - x0) or (x0 - x1).
        triangle.e1 = float2(triangle.p2.y - triangle.p1.y, triangle.p1.x - triangle.p2.x);
        triangle.e2 = float2(triangle.p3.y - triangle.p2.y, triangle.p2.x - triangle.p3.x);
        triangle.e3 = float2(triangle.p1.y - triangle.p3.y, triangle.p3.x - triangle.p1.x);

        // compute corner offset x & y to add to top left corner to find
        // trivial reject corner for each edge
        triangle.o1.x = (triangle.e1.x < 0) ? 1.f : 0.f;
        triangle.o1.y = (triangle.e1.y < 0) ? 1.f : 0.f;
        triangle.o2.x = (triangle.e2.x < 0) ? 1.f : 0.f;
        triangle.o2.y = (triangle.e2.y < 0) ? 1.f : 0.f;
        triangle.o3.x = (triangle.e3.x < 0) ? 1.f : 0.f;
        triangle.o3.y = (triangle.e3.y < 0) ? 1.f : 0.f;

        // determine overlapped bins by bounding box
        float2 bb_min = min(triangle.p1, min(triangle.p2, triangle.p3));
        float2 bb_max = max(triangle.p1, max(triangle.p2, triangle.p3));
        int x1 = (int)bb_min.x / SharedData->TileSize;
        int y1 = (int)bb_min.y / SharedData->TileSize;
        int x2 = (int)bb_max.x / SharedData->TileSize;
        int y2 = (int)bb_max.y / SharedData->TileSize;
        x1 = std::max(std::min(x1, SharedData->NumHorizBins - 1), 0);
        y1 = std::max(std::min(y1, SharedData->NumVertBins - 1), 0);
        x2 = std::max(std::min(x2, SharedData->NumHorizBins - 1), 0);
        y2 = std::max(std::min(y2, SharedData->NumVertBins - 1), 0);

        for (int r = y1; r <= y2; ++r)
        {
            for (int c = x1; c <= x2; ++c)
            {
                auto& bin = SharedData->Bins[r * SharedData->NumHorizBins + c];

                Triangle* triangleBin = &SharedData->TriangleMemory[SharedData->CurrentTriangleBin++];
                *triangleBin = triangle;

                triangleBin->Next = bin.Head;
                while (!bin.Head.compare_exchange_strong(triangleBin->Next, triangleBin))
                {
                    triangleBin->Next = bin.Head;
                }
            }
        }

        iTriangle = SharedData->CurrentTriangle++;
    }

    // Now wait for all threads to reach this barrier (indicating that all triangles are done being binned)
    // before we start processing screen tiles/bins
    if (++SharedData->BinningJoinBarrier == SharedData->NumThreads)
    {
        // Last one through the barrier, signal the rest

        // Signal the wait barrier
        SharedData->BinningWaitBarrier = true;
    }
    else
    {
        while (!SharedData->BinningWaitBarrier)
        {
            // Allow other threads to make progress
            _mm_pause();
        }
    }

    // Process tiles
    int rtWidth = command->RenderTarget->GetWidth();
    int rtHeight = command->RenderTarget->GetHeight();
    int rtPitchInPixels = command->RenderTarget->GetPitchInPixels();
    uint32_t* renderTarget = (uint32_t*)command->RenderTarget->GetData();

    int iBin = SharedData->CurrentBin++;
    while (iBin < SharedData->NumTotalBins)
    {
        auto& bin = SharedData->Bins[iBin];

        int x = (iBin % SharedData->NumHorizBins) * SharedData->TileSize;
        int y = (iBin / SharedData->NumHorizBins) * SharedData->TileSize;

        Triangle* t = bin.Head;
        while (t != nullptr)
        {
            sseProcessBlock(command, *t,
                x, y, SharedData->TileSize,
                renderTarget, rtWidth, rtHeight, rtPitchInPixels);

            t = t->Next;
        }

        iBin = SharedData->CurrentBin++;
    }

    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER time;
        QueryPerformanceCounter(&time);
        SharedData->TriangleStopTime[ID] = time.QuadPart;
    }
}

void TRPipelineThread::JoinAndDoSerialInitialization(const std::shared_ptr<RenderCommand>& command)
{
    // Everyone increments the join barrier and then waits on InitWaitBarrier
    // Last thread through barrier resets it, then does the initialization before
    // signaling the InitWaitBarrier
    if (++SharedData->JoinBarrier == SharedData->NumThreads)
    {
        // Last one through the barrier, reset & do serial work
        SharedData->JoinBarrier = 0;

        // Everyone is blocked on InitWaitBarrier right now,
        // so we can safely reset CompletionWaitBarrier
        SharedData->CompletionWaitBarrier = false;

        // Do the serial init
        SerialInitialization(command);

        // Signal the wait barrier
        SharedData->InitWaitBarrier = true;
    }
    else
    {
        while (!SharedData->InitWaitBarrier)
        {
            // Allow other threads to make progress
            _mm_pause();
        }
    }
}

void TRPipelineThread::JoinAndDoSerialCompletion(const std::shared_ptr<RenderCommand>& command)
{
    // Join, then do serial work (also resets wait barrier 2)
    if (++SharedData->JoinBarrier == SharedData->NumThreads)
    {
        // Last one through the barrier, reset & do serial work
        SharedData->JoinBarrier = 0;

        // Everyone is blocked on CompletionWaitBarrier right now,
        // so we can safely reset InitWaitBarrier
        SharedData->InitWaitBarrier = false;

        // Do the serial completion
        SerialCompletion(command);

        // Signal the wait barrier
        SharedData->CompletionWaitBarrier = true;
    }
    else
    {
        while (!SharedData->CompletionWaitBarrier)
        {
            // Allow other threads to make progress
            _mm_pause();
        }
    }
}

void TRPipelineThread::SerialInitialization(const std::shared_ptr<RenderCommand>& command)
{
    // Reset some internal shared state
    SharedData->CurrentVertex = 0;
    SharedData->CurrentTriangle = 0;
    SharedData->CurrentTriangleBin = 0;
    SharedData->BinningJoinBarrier = 0;
    SharedData->BinningWaitBarrier = false;

    // Ensure we have the right number of bins
    int targetHorizBins = (command->RenderTarget->GetWidth() + (SharedData->TileSize - 1)) / SharedData->TileSize;
    int targetVertBins = (command->RenderTarget->GetHeight() + (SharedData->TileSize - 1)) / SharedData->TileSize;
    if (targetHorizBins != SharedData->NumHorizBins || targetVertBins != SharedData->NumVertBins)
    {
        SharedData->NumHorizBins = targetHorizBins;
        SharedData->NumVertBins = targetVertBins;
        SharedData->NumTotalBins = SharedData->NumHorizBins * SharedData->NumVertBins;
        SharedData->Bins.reset(new Bin[SharedData->NumTotalBins]);
    }

    // Reset bins
    for (int i = 0; i < SharedData->NumTotalBins; ++i)
    {
        SharedData->Bins[i].Head = nullptr;
    }

    SharedData->CurrentBin = 0;
}

void TRPipelineThread::SerialCompletion(const std::shared_ptr<RenderCommand>& command)
{
    // Log stats
    ProcessAndLogStats();

    // Signal completion of this work
    Pipeline->NotifyCompletion(command->FenceValue);
}

void TRPipelineThread::ProcessAndLogStats()
{
    if (SharedData->StatsEnabled)
    {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);

        // find absolute start & end across all threads
        uint64_t minVertexStart = SharedData->VertexStartTime[0];
        uint64_t maxVertexEnd = SharedData->VertexStopTime[0];
        uint64_t totalVertexElapsed = maxVertexEnd - minVertexStart;
        uint64_t minTriangleStart = SharedData->TriangleStartTime[0];
        uint64_t maxTriangleEnd = SharedData->TriangleStopTime[0];
        uint64_t totalTriangleElapsed = maxTriangleEnd - minTriangleStart;
        for (int i = 1; i < SharedData->NumThreads; ++i)
        {
            // Vertex processing
            if (SharedData->VertexStartTime[i] < minVertexStart)
            {
                minVertexStart = SharedData->VertexStartTime[i];
            }
            if (SharedData->VertexStopTime[i] > maxVertexEnd)
            {
                maxVertexEnd = SharedData->VertexStopTime[i];
            }
            totalVertexElapsed += SharedData->VertexStopTime[i] - SharedData->VertexStartTime[i];

            // Triangle processing
            if (SharedData->TriangleStartTime[i] < minTriangleStart)
            {
                minTriangleStart = SharedData->TriangleStartTime[i];
            }
            if (SharedData->TriangleStopTime[i] > maxTriangleEnd)
            {
                maxTriangleEnd = SharedData->TriangleStopTime[i];
            }
            totalTriangleElapsed += SharedData->TriangleStopTime[i] - SharedData->TriangleStartTime[i];
        }

        wchar_t message[1024];
        swprintf_s(message, L"Vertex Processing End to End: %3.2fms\n", 1000.0 * ((maxVertexEnd - minVertexStart) / (double)freq.QuadPart));
        OutputDebugString(message);
        swprintf_s(message, L"Vertex Processing Avg per thread: %3.2fms\n", 1000.0 * ((totalVertexElapsed / (double)SharedData->NumThreads) / (double)freq.QuadPart));
        OutputDebugString(message);
        swprintf_s(message, L"Triangle Processing End to End: %3.2fms\n", 1000.0 * ((maxTriangleEnd - minTriangleStart) / (double)freq.QuadPart));
        OutputDebugString(message);
        swprintf_s(message, L"Triangle Processing Avg per thread: %3.2fms\n", 1000.0 * ((totalTriangleElapsed / (double)SharedData->NumThreads) / (double)freq.QuadPart));
        OutputDebugString(message);
    }
}


void TRPipelineThread::sseProcessBlock(
    const std::shared_ptr<RenderCommand>& command,
    const Triangle& triangle,
    int top_left_x, int top_left_y, int tileSize,
    uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels)
    
{
    const int size = tileSize >> 2;

    assert(size > 0);

    float2 off1 = triangle.o1 * (float)size;
    float2 off2 = triangle.o2 * (float)size;
    float2 off3 = triangle.o3 * (float)size;

    if (size == 1)
    {
        // for pixel level, use center of pixels
        off1 = off2 = off3 = float2(0.5f, 0.5f);
    }

    __m128 indices = _mm_set_ps(0.f, 1.f, 2.f, 3.f);
    __m128 sizes = _mm_set1_ps((float)size);
    __m128 xoff = _mm_mul_ps(indices, sizes);
    __m128 base_corner_x = _mm_add_ps(_mm_set1_ps((float)top_left_x), xoff);

    int maxY = std::min(top_left_y + tileSize, rtHeight);

    for (int y = top_left_y; y < maxY; y += size)
    {
        __m128 base_corner_y = _mm_set1_ps((float)y);

        // trivial reject against edge1

        // side_of_edge:
        // float2 diff = point - base_vert;
        // return dot(diff, edge_equation);

        // float2 diff part
        // break down to all edge1-adjusted x's, then edge1-adjusted y's
        // base_vert is p1
        __m128 adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off1.x));
        __m128 diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p1.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off1.y));
        __m128 diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p1.y));

        // dot part is broken into muls & adds
        // m1 = diff.x * edge.x
        // m2 = diff.y * edge.y
        // m1 + m2
        __m128 m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e1.x));
        __m128 m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e1.y));
        __m128 dots = _mm_add_ps(m1, m2);

        // if dot is > 0.f, those tiles are rejected.
        __m128 e1mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // Now, repeat for edge 2
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off2.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off2.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e2.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e2.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e2mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // And edge3
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off3.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off3.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e3.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e3.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e3mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // only elements we keep are the ones that passed all three filters. ie:
        // mask1 | mask2 | mask3 == 0
        __m128 mask = _mm_or_ps(e1mask, _mm_or_ps(e2mask, e3mask));

        // mask out any of the tiles/pixels that are off screen
        __m128 screenclipmask = _mm_cmpge_ps(base_corner_x, _mm_set1_ps((float)rtWidth));

        // convert to integer mask for easier testing below
        int imask = _mm_movemask_ps(_mm_or_ps(mask, screenclipmask));

        if (size > 1)
        {
            // recurse sub tiles
            if ((imask & 0x08) == 0)
            {
                sseProcessBlock(command, triangle,
                    top_left_x, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels);
            }
            if ((imask & 0x04) == 0)
            {
                sseProcessBlock(command, triangle,
                    top_left_x + size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels);
            }
            if ((imask & 0x02) == 0)
            {
                sseProcessBlock(command, triangle,
                    top_left_x + 2 * size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels);
            }
            if ((imask & 0x01) == 0)
            {
                sseProcessBlock(command, triangle,
                    top_left_x + 3 * size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels);
            }
        }
        else
        {
            vs_output v1 = GetSSEVertexAttributes(triangle.iTriangle * 3);
            vs_output v2 = GetSSEVertexAttributes(triangle.iTriangle * 3 + 1);
            vs_output v3 = GetSSEVertexAttributes(triangle.iTriangle * 3 + 2);

            // rasterize the pixels!
            lerp_result lerp = sseLerp(
                v1, v2, v3,
                vec2{ _mm_sub_ps(v2.Position.x, v1.Position.x),_mm_sub_ps(v2.Position.y, v1.Position.y) },
                vec2{ _mm_sub_ps(v3.Position.x, v2.Position.x),_mm_sub_ps(v3.Position.y, v2.Position.y) },
                vec2{ _mm_sub_ps(v3.Position.x, v1.Position.x),_mm_sub_ps(v3.Position.y, v1.Position.y) },
                vec2{ base_corner_x, base_corner_y });

                imask |= _mm_movemask_ps(lerp.mask);

            vs_output input{ lerp.position, lerp.color };
            vec4 frags = command->PixelShader(command->PixelShader, input);

            uint32_t colors[4];
            ConvertFragsToColors(frags, colors);

            if ((imask & 0x08) == 0)
            {
                renderTarget[y * rtPitchPixels + top_left_x] = colors[3];
            }
            if ((imask & 0x04) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + size)] = colors[2];
            }
            if ((imask & 0x02) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + 2 * size)] = colors[1];
            }
            if ((imask & 0x01) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + 3 * size)] = colors[0];
            }
        }
    }
}

void TRPipelineThread::ConvertFragsToColors(const vec4 frags, uint32_t colors[4])
{
    __m128 x = _mm_mul_ps(frags.x, _mm_set1_ps(255.f));
    __m128 y = _mm_mul_ps(frags.y, _mm_set1_ps(255.f));
    __m128 z = _mm_mul_ps(frags.z, _mm_set1_ps(255.f));
    __m128 w = _mm_mul_ps(frags.w, _mm_set1_ps(255.f));

    __m128i r = _mm_cvtps_epi32(x);
    r = _mm_max_epi32(_mm_min_epi32(r, _mm_set1_epi32(255)), _mm_setzero_si128());
    __m128i g = _mm_cvtps_epi32(y);
    g = _mm_max_epi32(_mm_min_epi32(g, _mm_set1_epi32(255)), _mm_setzero_si128());
    __m128i b = _mm_cvtps_epi32(z);
    b = _mm_max_epi32(_mm_min_epi32(b, _mm_set1_epi32(255)), _mm_setzero_si128());
    __m128i a = _mm_cvtps_epi32(w);
    a = _mm_max_epi32(_mm_min_epi32(a, _mm_set1_epi32(255)), _mm_setzero_si128());

    for (int i = 0; i < 4; ++i)
    {
        colors[i] = a.m128i_u32[i] << 24 | b.m128i_u32[i] << 16 | g.m128i_u32[i] << 8 | r.m128i_u32[i];
    }
}



void TRPipelineThread::GetVertexAttributes(uint64_t iVertex, float4* position, float3* color)
{
    uint64_t iBase = iVertex / 4;
    uint64_t iOff = iVertex % 4;

    SSEVSOutput* v = SharedData->VSOutputs;

    if (position)
    {
        position->x = v[iBase].Position_x[iOff];
        position->y = v[iBase].Position_y[iOff];
        position->z = v[iBase].Position_z[iOff];
        position->w = v[iBase].Position_w[iOff];
    }

    if (color)
    {
        color->x = v[iBase].Color_x[iOff];
        color->y = v[iBase].Color_y[iOff];
        color->z = v[iBase].Color_z[iOff];
    }
}

vs_output TRPipelineThread::GetSSEVertexAttributes(uint64_t iVertex)
{
    uint64_t iBase = iVertex / 4;
    uint64_t iOff = iVertex % 4;

    SSEVSOutput* v = SharedData->VSOutputs;

    vs_output output;

    output.Position.x = _mm_set1_ps(v[iBase].Position_x[iOff]);
    output.Position.y = _mm_set1_ps(v[iBase].Position_y[iOff]);
    output.Position.z = _mm_set1_ps(v[iBase].Position_z[iOff]);
    output.Position.w = _mm_set1_ps(v[iBase].Position_w[iOff]);

    output.Color.x = _mm_set1_ps(v[iBase].Color_x[iOff]);
    output.Color.y = _mm_set1_ps(v[iBase].Color_y[iOff]);
    output.Color.z = _mm_set1_ps(v[iBase].Color_z[iOff]);

    return output;
}

// rasterize a 2D triangle, binning the spans
void TRPipelineThread::DDARastTriangle(
    const std::shared_ptr<RenderCommand>& command,
    uint64_t iFirstVertex,              // to get attributes from later
    float4 v[3],                        // input position of each vertex
    uint32_t* renderTarget, int rtWidth, int rtHeight, int pitch)  // output pixels here
{
    struct rast_span
    {
        float x1, x2;
    };

    rast_span spans[2048];  // supports up to 2k height render target
    int next_span = 0;

    assert(rtHeight < _countof(spans));

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

    int y1 = std::max(0, (int)ytop);
    int y2 = std::min((int)ymid, rtHeight);

    for (int y = y1; y < y2; ++y)
    {
        rast_span* span = &spans[next_span++];
        span->x1 = std::max(0.f, x1);
        span->x2 = std::min(x2, (float)rtWidth);

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

    y1 = std::max(0, (int)ymid);
    y2 = std::min((int)ybottom, rtHeight);

    for (int y = y1; y < y2; ++y)
    {
        rast_span* span = &spans[next_span++];
        span->x1 = std::max(0.f, x1);
        span->x2 = std::min(x2, (float)rtWidth);

        x1 += step1;
        x2 += step2;
    }

    // Spans determined, now set up lerping parameters (shared by all spans)
    vs_output v1 = GetSSEVertexAttributes(iFirstVertex);
    vs_output v2 = GetSSEVertexAttributes(iFirstVertex + 1);
    vs_output v3 = GetSSEVertexAttributes(iFirstVertex + 2);
    vec2 ab{ _mm_sub_ps(v2.Position.x, v1.Position.x),_mm_sub_ps(v2.Position.y, v1.Position.y) };
    vec2 bc{ _mm_sub_ps(v3.Position.x, v2.Position.x),_mm_sub_ps(v3.Position.y, v2.Position.y) };
    vec2 ac{ _mm_sub_ps(v3.Position.x, v1.Position.x),_mm_sub_ps(v3.Position.y, v1.Position.y) };

    int y = std::max(0, (int)ytop);

    rast_span* span = spans;
    for (; next_span > 0; --next_span, ++span)
    {
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

        ++y;
    }
}
