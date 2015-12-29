#pragma once

// One (of potentially many) render threads. Each render thread
// processes as much of the work in the pipeline as possible
class RenderThread
{
public:
    RenderThread(uint32_t id);
    ~RenderThread();

    bool Initialize();
    void SignalShutdown();

    const HANDLE ThreadHandle() const { return TheThread.Get(); }

    void QueueRendering(SharedRenderData* renderData);

private:
    RenderThread(const RenderThread&) = delete;
    RenderThread& operator= (const RenderThread&) = delete;

    static DWORD CALLBACK s_ThreadProc(PVOID context);
    void ThreadProc();

    static void __vectorcall sseProcessBlock(
        const float2& p1, const float2& p2, const float2& p3,   // three triangle vertices
        const float2& e1, const float2& e2, const float2& e3,   // three edge equations
        const float2& o1, const float2& o2, const float2& o3,   // three rejection corner offsets
        uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels,
        int top_left_x, int top_left_y, int tileSize);          // in pixels

private:
    static const uint32_t ShutdownTimeoutMilliseconds = 30000;
    static const uint32_t MaxInFlightRenderJobs = 32;
    static uint32_t NumThreads;

    // Shared scratch memory that threads work with
    static SSEVSOutput VSOutputs[];
    static SSEPSOutput PSOutputs[];

    uint32_t ID;
    Thread TheThread;

    std::mutex RenderJobMutex;
    std::queue<SharedRenderData*> RenderJobs;

    Microsoft::WRL::Wrappers::Event RenderJobReady;

    // TODO: We could use 1 manual reset event for all threads, unless
    // there's a use case for shutting just a subset of threads down?
    Microsoft::WRL::Wrappers::Event ShutdownEvent;
};
