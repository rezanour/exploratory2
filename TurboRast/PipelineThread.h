#pragma once

struct RenderCommand;
class TRPipeline;
enum class RastStrategy;
struct Triangle;
struct Bin;

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
    void ProcessVertices(const std::shared_ptr<RenderCommand>& command);

    void ProcessOneTrianglePerThread(const std::shared_ptr<RenderCommand>& command);
    void ProcessOneTilePerThread(const std::shared_ptr<RenderCommand>& command);

    void JoinAndDoSerialInitialization(const std::shared_ptr<RenderCommand>& command);
    void JoinAndDoSerialCompletion(const std::shared_ptr<RenderCommand>& command);

    void SerialInitialization(const std::shared_ptr<RenderCommand>& command);
    void SerialCompletion(const std::shared_ptr<RenderCommand>& command);

    void ProcessAndLogStats();

    static void sseProcessBlock(
        const Triangle& triangle,
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
