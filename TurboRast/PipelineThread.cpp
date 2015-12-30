#include "Precomp.h"
#include "Pipeline.h"
#include "PipelineThread.h"
#include "VertexBuffer.h"
#include "Texture2D.h"

using namespace Microsoft::WRL::Wrappers;

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
        command->VertexShader(command->VSConstantBuffer, command->VertexBuffer->GetBlocks()[iVertexBlock], VSOutputs[iVertexBlock]);

        // Load result to work on it
        __m128 x = _mm_load_ps(VSOutputs[iVertexBlock].Position_x);
        __m128 y = _mm_load_ps(VSOutputs[iVertexBlock].Position_y);
        __m128 z = _mm_load_ps(VSOutputs[iVertexBlock].Position_z);
        __m128 w = _mm_load_ps(VSOutputs[iVertexBlock].Position_w);

        // Divide by w
        x = _mm_div_ps(x, w);
        y = _mm_div_ps(y, w);
        z = _mm_div_ps(z, w);

        // Scale to viewport
        x = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0.5f)), _mm_set1_ps(0.5f)), _mm_set1_ps((float)rtWidth));
        y = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.f), _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(0.5f)), _mm_set1_ps(0.5f))), _mm_set1_ps((float)rtHeight));

        // Store back result
        _mm_store_ps(VSOutputs[iVertexBlock].Position_x, x);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_y, y);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_z, z);
        _mm_store_ps(VSOutputs[iVertexBlock].Position_w, _mm_set1_ps(1.f));

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
    SSEVSOutput* VSOutputs = SharedData->VSOutputs;

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
        uint64_t iP1 = iTriangle * 3;
        uint64_t iP2 = iP1 + 1;
        uint64_t iP3 = iP2 + 1;

        uint64_t iP1base = iP1 / 4;
        uint64_t iP1off = iP1 % 4;
        uint64_t iP2base = iP2 / 4;
        uint64_t iP2off = iP2 % 4;
        uint64_t iP3base = iP3 / 4;
        uint64_t iP3off = iP3 % 4;

        triangle.iTriangle = iTriangle;
        triangle.p1 = float2(VSOutputs[iP1base].Position_x[iP1off], VSOutputs[iP1base].Position_y[iP1off]);
        triangle.p2 = float2(VSOutputs[iP2base].Position_x[iP2off], VSOutputs[iP2base].Position_y[iP2off]);
        triangle.p3 = float2(VSOutputs[iP3base].Position_x[iP3off], VSOutputs[iP3base].Position_y[iP3off]);

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

        for (int y = 0; y < rtHeight; y += SharedData->TileSize)
        {
            for (int x = 0; x < rtWidth; x += SharedData->TileSize)
            {
                sseProcessBlock(
                    triangle,
                    x, y, SharedData->TileSize,
                    renderTarget, rtWidth, rtHeight, rtPitchInPixels,
                    SharedData->VSOutputs);
            }
        }

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
    SSEVSOutput* VSOutputs = SharedData->VSOutputs;

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
        uint64_t iP1 = iTriangle * 3;
        uint64_t iP2 = iP1 + 1;
        uint64_t iP3 = iP2 + 1;

        uint64_t iP1base = iP1 / 4;
        uint64_t iP1off = iP1 % 4;
        uint64_t iP2base = iP2 / 4;
        uint64_t iP2off = iP2 % 4;
        uint64_t iP3base = iP3 / 4;
        uint64_t iP3off = iP3 % 4;

        triangle.iTriangle = iTriangle;
        triangle.p1 = float2(VSOutputs[iP1base].Position_x[iP1off], VSOutputs[iP1base].Position_y[iP1off]);
        triangle.p2 = float2(VSOutputs[iP2base].Position_x[iP2off], VSOutputs[iP2base].Position_y[iP2off]);
        triangle.p3 = float2(VSOutputs[iP3base].Position_x[iP3off], VSOutputs[iP3base].Position_y[iP3off]);

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

                // claim an index
                int64_t index = bin.CurrentTriangle++;
                assert(index < _countof(bin.Triangles));
                bin.Triangles[index] = triangle;
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

        for (iTriangle = 0; iTriangle < bin.CurrentTriangle; ++iTriangle)
        {
            sseProcessBlock(
                bin.Triangles[iTriangle],
                x, y, SharedData->TileSize,
                renderTarget, rtWidth, rtHeight, rtPitchInPixels,
                VSOutputs);
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
        SharedData->Bins[i].CurrentTriangle = 0;
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
    const Triangle& triangle,
    int top_left_x, int top_left_y, int tileSize,
    uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels,
    SSEVSOutput* VSOutputs)
    
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
                sseProcessBlock(
                    triangle,
                    top_left_x, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels,
                    VSOutputs);
            }
            if ((imask & 0x04) == 0)
            {
                sseProcessBlock(
                    triangle,
                    top_left_x + size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels,
                    VSOutputs);
            }
            if ((imask & 0x02) == 0)
            {
                sseProcessBlock(
                    triangle,
                    top_left_x + 2 * size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels,
                    VSOutputs);
            }
            if ((imask & 0x01) == 0)
            {
                sseProcessBlock(
                    triangle,
                    top_left_x + 3 * size, y, size,
                    renderTarget, rtWidth, rtHeight, rtPitchPixels,
                    VSOutputs);
            }
        }
        else
        {
            // rasterize the pixels!
            __m128 lerpMask;
            SSEVSOutput output;
            sseLerp(triangle, base_corner_x, base_corner_y, lerpMask, VSOutputs, &output);

            imask |= _mm_movemask_ps(lerpMask);

            if ((imask & 0x08) == 0)
            {
                renderTarget[y * rtPitchPixels + top_left_x] = 
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.Color_z[3] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.Color_y[3] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.Color_x[3] * 255.f));
            }
            if ((imask & 0x04) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + size)] = 
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.Color_z[2] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.Color_y[2] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.Color_x[2] * 255.f));
            }
            if ((imask & 0x02) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + 2 * size)] = 
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.Color_z[1] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.Color_y[1] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.Color_x[1] * 255.f));
            }
            if ((imask & 0x01) == 0)
            {
                renderTarget[y * rtPitchPixels + (top_left_x + 3 * size)] = 
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.Color_z[0] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.Color_y[0] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.Color_x[0] * 255.f));
            }
        }
    }
}

// Compute barycentric coordinates (lerp weights) for 4 samples at once.
// The computation is done in 2 dimensions (screen space).
// in: a (ax, ay), b (bx, by) and c (cx, cy) are the 3 vertices of the triangle.
//     p (px, py) is the point to compute barycentric coordinates for
// out: wA, wB, wC are the weights at vertices a, b, and c
//      mask will contain a 0 (clear) if the value is computed. It will be 0xFFFFFFFF (set) if invalid
void TRPipelineThread::sseBary2D(
    const __m128& ax, const __m128& ay, const __m128& bx, const __m128& by, const __m128& cx, const __m128& cy,
    const __m128& px, const __m128& py, __m128& xA, __m128& xB, __m128& xC, __m128& mask)
{
    __m128 abx = _mm_sub_ps(bx, ax);
    __m128 aby = _mm_sub_ps(by, ay);
    __m128 acx = _mm_sub_ps(cx, ax);
    __m128 acy = _mm_sub_ps(cy, ay);

    // Find barycentric coordinates of P (wA, wB, wC)
    __m128 bcx = _mm_sub_ps(cx, bx);
    __m128 bcy = _mm_sub_ps(cy, by);
    __m128 apx = _mm_sub_ps(px, ax);
    __m128 apy = _mm_sub_ps(py, ay);
    __m128 bpx = _mm_sub_ps(px, bx);
    __m128 bpy = _mm_sub_ps(py, by);

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
    __m128 wC = _mm_sub_ps(_mm_mul_ps(abx, apy), _mm_mul_ps(apx, aby));
    __m128 mask1 = _mm_cmplt_ps(wC, _mm_setzero_ps());

    // Use same reduction for wB & wA
    __m128 wB = _mm_sub_ps(_mm_mul_ps(apx, acy), _mm_mul_ps(acx, apy));
    __m128 mask2 = _mm_cmplt_ps(wB, _mm_setzero_ps());

    __m128 wA = _mm_sub_ps(_mm_mul_ps(bcx, bpy), _mm_mul_ps(bpx, bcy));
    __m128 mask3 = _mm_cmplt_ps(wA, _mm_setzero_ps());

    mask = _mm_or_ps(mask1, _mm_or_ps(mask2, mask3));

    // Use a similar reduction for cross of ab x ac (to find unnormalized normal)
    __m128 norm = _mm_sub_ps(_mm_mul_ps(abx, acy), _mm_mul_ps(acx, aby));
    norm = _mm_rcp_ps(norm);

    // to find length of this cross product, which already know is purely in the z
    // direction, is just the length of the z component, which is the exactly the single
    // channel norm we computed above. Similar logic is used for lengths of each of
    // the weights, since they are all single channel vectors, the one channel is exactly
    // the length.

    xA = _mm_mul_ps(wA, norm);
    xB = _mm_mul_ps(wB, norm);
    xC = _mm_mul_ps(wC, norm);
}

void TRPipelineThread::sseLerp(
    const Triangle& triangle,
    const __m128& px, const __m128& py, __m128& mask,
    SSEVSOutput* VSOutputStream, SSEVSOutput* outputs)
{
    __m128 ax = _mm_set1_ps(triangle.p1.x);
    __m128 ay = _mm_set1_ps(triangle.p1.y);
    __m128 bx = _mm_set1_ps(triangle.p2.x);
    __m128 by = _mm_set1_ps(triangle.p2.y);
    __m128 cx = _mm_set1_ps(triangle.p3.x);
    __m128 cy = _mm_set1_ps(triangle.p3.y);

    __m128 xA, xB, xC;
    sseBary2D(ax, ay, bx, by, cx, cy, px, py, xA, xB, xC, mask);

    // Interpolate all the attributes for these 4 pixels
    uint64_t iP1 = triangle.iTriangle * 3;
    uint64_t iP2 = iP1 + 1;
    uint64_t iP3 = iP2 + 1;

    uint64_t iP1base = iP1 / 4;
    uint64_t iP1off = iP1 % 4;
    uint64_t iP2base = iP2 / 4;
    uint64_t iP2off = iP2 % 4;
    uint64_t iP3base = iP3 / 4;
    uint64_t iP3off = iP3 % 4;

    __m128 posx = _mm_add_ps(_mm_mul_ps(ax, xA), _mm_add_ps(_mm_mul_ps(bx, xB), _mm_mul_ps(cx, xC)));
    __m128 posy = _mm_add_ps(_mm_mul_ps(ay, xA), _mm_add_ps(_mm_mul_ps(by, xB), _mm_mul_ps(cy, xC)));
    __m128 posz = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(VSOutputStream[iP1base].Position_z[iP1off]), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(VSOutputStream[iP2base].Position_z[iP2off]), xB), _mm_mul_ps(_mm_set1_ps(VSOutputStream[iP3base].Position_z[iP3off]), xC)));
    __m128 posw = _mm_set1_ps(1.f);

    float3 c1(VSOutputStream[iP1base].Color_x[iP1off], VSOutputStream[iP1base].Color_y[iP1off], VSOutputStream[iP1base].Color_z[iP1off]);
    float3 c2(VSOutputStream[iP2base].Color_x[iP2off], VSOutputStream[iP2base].Color_y[iP2off], VSOutputStream[iP2base].Color_z[iP2off]);
    float3 c3(VSOutputStream[iP3base].Color_x[iP3off], VSOutputStream[iP3base].Color_y[iP3off], VSOutputStream[iP3base].Color_z[iP3off]);

    __m128 colx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.x), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.x), xB), _mm_mul_ps(_mm_set1_ps(c3.x), xC)));
    __m128 coly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.y), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.y), xB), _mm_mul_ps(_mm_set1_ps(c3.y), xC)));
    __m128 colz = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.z), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.z), xB), _mm_mul_ps(_mm_set1_ps(c3.z), xC)));

    _mm_store_ps(outputs->Position_x, posx);
    _mm_store_ps(outputs->Position_y, posy);
    _mm_store_ps(outputs->Position_z, posz);
    _mm_store_ps(outputs->Position_w, posw);
    _mm_store_ps(outputs->Color_x, colx);
    _mm_store_ps(outputs->Color_y, coly);
    _mm_store_ps(outputs->Color_z, colz);
}

