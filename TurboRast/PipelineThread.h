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

    void __vectorcall ConvertFragsToColors(const vec4 frags, uint32_t colors[4]);

    void GetVertexAttributes(uint64_t iVertex, float4* position, float3* color);
    vs_output __vectorcall GetSSEVertexAttributes(uint64_t iVertex);

    void DDARastTriangle(
        const std::shared_ptr<RenderCommand>& command,
        uint64_t iFirstVertex,              // to get attributes from later
        float4 v[3],                        // input position of each vertex
        uint32_t* renderTarget, int rtWidth, int rtHeight, int pitch);

private:
    int ID;
    TRPipeline* Pipeline;
    SharedPipelineData* SharedData;
    Thread TheThread;

    Microsoft::WRL::Wrappers::CriticalSection CommandsLock;
    Microsoft::WRL::Wrappers::Event CommandReady;
    std::deque<std::shared_ptr<RenderCommand>> Commands;
};
