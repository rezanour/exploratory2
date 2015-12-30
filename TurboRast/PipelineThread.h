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

    void sseProcessBlock(
        const std::shared_ptr<RenderCommand>& command,
        const Triangle& triangle,
        int top_left_x, int top_left_y, int tileSize,
        uint32_t* renderTarget, int rtWidth, int rtHeight, int rtPitchPixels);

    void sseBary2D(
        const __m128& ax, const __m128& ay, const __m128& bx, const __m128& by, const __m128& cx, const __m128& cy,
        const __m128& px, const __m128& py, __m128& xA, __m128& xB, __m128& xC, __m128& mask);

    void sseLerp(
        const Triangle& triangle,
        const __m128& px, const __m128& py, __m128& mask,
        SSEVSOutput* output);

private:
    int ID;
    TRPipeline* Pipeline;
    SharedPipelineData* SharedData;
    Thread TheThread;

    Microsoft::WRL::Wrappers::CriticalSection CommandsLock;
    Microsoft::WRL::Wrappers::Event CommandReady;
    std::deque<std::shared_ptr<RenderCommand>> Commands;
};
