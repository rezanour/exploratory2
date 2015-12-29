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
            // Stage 1: Vertex processing

            uint64_t numBlocks = command->VertexBuffer->GetNumBlocks();
            int rtWidth = command->RenderTarget->GetWidth();
            int rtHeight = command->RenderTarget->GetHeight();
            int rtPitchInPixels = command->RenderTarget->GetPitchInPixels();
            uint32_t* renderTarget = (uint32_t*)command->RenderTarget->GetData();
            SSEVSOutput* VSOutputs = SharedData->VSOutputs;

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

            // Stage 2: Triangle processing

            // TODO: There are multiple strategies we could consider here. Should make this code modular
            // enough to be able to experiment with each strategy here (or switch between them at runtime based
            // on some criteria). The strategies are:
            //   1. Bin triangles into screen space tiles, and then process tiles in parallel across threads.
            //   2. Process each triangle all the way down to pixels. Do this in parallel for different triangles on each thread
            //   * When sort order matters (blending, no depth reject, etc...), we need to be careful to maintain draw order
            //   * Should pixel processing be lifted out separately from triangle processing? How can we efficiently do this?

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

                float2 p1(VSOutputs[iP1base].Position_x[iP1off], VSOutputs[iP1base].Position_y[iP1off]);
                float2 p2(VSOutputs[iP2base].Position_x[iP2off], VSOutputs[iP2base].Position_y[iP2off]);
                float2 p3(VSOutputs[iP3base].Position_x[iP3off], VSOutputs[iP3base].Position_y[iP3off]);

                // edge equation Bx + Cy = 0, where B & C are computed from slope as B = (y1 - y0) and C = -(x1 - x0) or (x0 - x1).
                float2 e1 = float2(p2.y - p1.y, p1.x - p2.x);
                float2 e2 = float2(p3.y - p2.y, p2.x - p3.x);
                float2 e3 = float2(p1.y - p3.y, p3.x - p1.x);

                // compute corner offset x & y to add to top left corner to find
                // trivial reject corner for each edge
                float2 off1, off2, off3;
                off1.x = (e1.x < 0) ? 1.f : 0.f;
                off1.y = (e1.y < 0) ? 1.f : 0.f;
                off2.x = (e2.x < 0) ? 1.f : 0.f;
                off2.y = (e2.y < 0) ? 1.f : 0.f;
                off3.x = (e3.x < 0) ? 1.f : 0.f;
                off3.y = (e3.y < 0) ? 1.f : 0.f;

                static const int tileSize = 64;

                for (int y = 0; y < rtHeight; y += tileSize)
                {
                    for (int x = 0; x < rtWidth; x += tileSize)
                    {
                        sseProcessBlock(p1, p2, p3, e1, e2, e3, off1, off2, off3,
                            renderTarget, rtWidth, rtHeight,
                            rtPitchInPixels, x, y, tileSize);
                    }
                }

                iTriangle = SharedData->CurrentTriangle++;
            }

            // Stage 3: Fragment/pixel processing (if applicable)


            // Stage 4: Synchronize & do some serial work from last thread through, then unblock other threads
            SyncAndDoSerialWork(command->FenceValue);

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

void TRPipelineThread::SyncAndDoSerialWork(uint64_t fenceValue)
{
    // Join, then do serial work (also resets wait barrier 2)
    if (++SharedData->JoinBarrier == SharedData->NumThreads)
    {
        // Last one through the barrier, reset & do serial work
        SharedData->JoinBarrier = 0;

        // Everyone is blocked on wait barrier 1 right now, so we can reset 2
        SharedData->WaitBarrier2 = false;

        // Signal completion of this work
        Pipeline->NotifyCompletion(fenceValue);

        // Reset some internal shared state
        SharedData->CurrentVertex = 0;
        SharedData->CurrentTriangle = 0;

        // Signal the wait barrier
        SharedData->WaitBarrier1 = true;
    }
    else
    {
        while (!SharedData->WaitBarrier1)
        {
            // Allow other threads to make progress
            _mm_pause();
        }
    }

    // Since we need to reset wait barrier 1, but can't until we know nobody
    // can reach it, we have to join & block on wait barrier 2 now and then
    // reset barrier 1
    if (++SharedData->JoinBarrier == SharedData->NumThreads)
    {
        // Last one through the barrier, reset & do serial work
        SharedData->JoinBarrier = 0;

        // Everyone is blocked on wait barrier 2 right now, so we can reset 1
        SharedData->WaitBarrier1 = false;

        // Signal the wait barrier
        SharedData->WaitBarrier2 = true;
    }
    else
    {
        while (!SharedData->WaitBarrier2)
        {
            // Allow other threads to make progress
            _mm_pause();
        }
    }
}

void TRPipelineThread::sseProcessBlock(
    const float2& p1, const float2& p2, const float2& p3,   // three triangle vertices
    const float2& e1, const float2& e2, const float2& e3,   // three edge equations
    const float2& o1, const float2& o2, const float2& o3,   // three rejection corner offsets
    uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels,
    int top_left_x, int top_left_y, int tileSize)          // in pixels
{
    const int size = tileSize / 4;

    assert(size > 0);

    float2 off1 = o1 * (float)size;
    float2 off2 = o2 * (float)size;
    float2 off3 = o3 * (float)size;

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
        __m128 diffx = _mm_sub_ps(adj, _mm_set1_ps(p1.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off1.y));
        __m128 diffy = _mm_sub_ps(adj, _mm_set1_ps(p1.y));

        // dot part is broken into muls & adds
        // m1 = diff.x * edge.x
        // m2 = diff.y * edge.y
        // m1 + m2
        __m128 m1 = _mm_mul_ps(diffx, _mm_set1_ps(e1.x));
        __m128 m2 = _mm_mul_ps(diffy, _mm_set1_ps(e1.y));
        __m128 dots = _mm_add_ps(m1, m2);

        // if dot is > 0.f, those tiles are rejected.
        __m128 e1mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // Now, repeat for edge 2
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off2.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(p2.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off2.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(p2.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(e2.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(e2.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e2mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // And edge3
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off3.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(p3.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off3.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(p3.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(e3.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(e3.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e3mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // only elements we keep are the ones that passed all three filters. ie:
        // mask1 | mask2 | mask3 == 0
        __m128 mask = _mm_or_ps(e1mask, _mm_or_ps(e2mask, e3mask));

        // mask out any of the tiles/pixels that are off screen
        __m128 screenclipmask = _mm_cmpge_ps(base_corner_x, _mm_set1_ps((float)rtWidth));

        // convert to integer mask for easier testing below
        __m128i imask = _mm_cvtps_epi32(_mm_or_ps(mask, screenclipmask));

        if (size > 1)
        {
            // recurse sub tiles
            if (_mm_testz_si128(imask, _mm_set_epi32(0xFFFFFFFF, 0, 0, 0)))
            {
                sseProcessBlock(p1, p2, p3, e1, e2, e3, o1, o2, o3, renderTarget, rtWidth, rtHeight, rtPitchPixels, top_left_x, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0xFFFFFFFF, 0, 0)))
            {
                sseProcessBlock(p1, p2, p3, e1, e2, e3, o1, o2, o3, renderTarget, rtWidth, rtHeight, rtPitchPixels, top_left_x + size, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0xFFFFFFFF, 0)))
            {
                sseProcessBlock(p1, p2, p3, e1, e2, e3, o1, o2, o3, renderTarget, rtWidth, rtHeight, rtPitchPixels, top_left_x + 2 * size, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0, 0xFFFFFFFF)))
            {
                sseProcessBlock(p1, p2, p3, e1, e2, e3, o1, o2, o3, renderTarget, rtWidth, rtHeight, rtPitchPixels, top_left_x + 3 * size, y, size);
            }
        }
        else
        {
            // rasterize the pixels!
            renderTarget[y * rtPitchPixels + (top_left_x)] = 0xFFFF0000;
            renderTarget[y * rtPitchPixels + (top_left_x + 1)] = 0xFFFF0000;
            renderTarget[y * rtPitchPixels + (top_left_x + 2)] = 0xFFFF0000;
            renderTarget[y * rtPitchPixels + (top_left_x + 3)] = 0xFFFF0000;
#if 0
            SSEVSOutput output;
            __m128 barymask;
            rz_lerp(triangle, base_corner_x, base_corner_y, barymask, &output);
            imask = _mm_cvtps_epi32(_mm_or_ps(mask, barymask));

            if (_mm_testz_si128(imask, _mm_set_epi32(0xFFFFFFFF, 0, 0, 0)))
            {
                RenderTarget[y * RTPitchPixels + top_left_x] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[3] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[3] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[3] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0xFFFFFFFF, 0, 0)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 1)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[2] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[2] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[2] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0xFFFFFFFF, 0)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 2)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[1] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[1] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[1] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0, 0xFFFFFFFF)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 3)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[0] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[0] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[0] * 255.f));
            }
#endif
        }
    }
}
