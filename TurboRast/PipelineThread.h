#pragma once

struct RenderCommand;
class TRPipeline;

// One (of potentially many) pipeline threads.
class TRPipelineThread
{
    NON_COPYABLE(TRPipelineThread);

public:
    TRPipelineThread(int id, TRPipeline* pipeline, SharedPipelineData* sharedData);
    ~TRPipelineThread();

    bool Initialize();

    const HANDLE GetThreadHandle() const { return TheThread.Get(); }

    void QueueCommand(const std::shared_ptr<RenderCommand>& command);

private:
    static DWORD CALLBACK s_ThreadProc(PVOID context);
    void ThreadProc();

    std::shared_ptr<RenderCommand> GetNextCommand();
    void SyncAndDoSerialWork(uint64_t fenceValue);

    static void __vectorcall sseProcessBlock(
        const float2& p1, const float2& p2, const float2& p3,   // three triangle vertices
        const float2& e1, const float2& e2, const float2& e3,   // three edge equations
        const float2& o1, const float2& o2, const float2& o3,   // three rejection corner offsets
        uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels,
        int top_left_x, int top_left_y, int tileSize);          // in pixels

private:
    int ID;
    TRPipeline* Pipeline;
    SharedPipelineData* SharedData;
    Thread TheThread;

    Microsoft::WRL::Wrappers::CriticalSection CommandsLock;
    Microsoft::WRL::Wrappers::Event CommandReady;
    std::deque<std::shared_ptr<RenderCommand>> Commands;
};
