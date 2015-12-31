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

    bary_result __vectorcall sseBary2D(const vec2 a, const vec2 b, const vec2 c, const vec2 p);

    struct alignas(16) lerp_result
    {
        __m128 mask;
        vec4 position;
        vec3 color;
    };

    lerp_result __vectorcall sseLerp(
        const vec4 p1, const vec4 p2, const vec4 p3,
        const vec3 c1, const vec3 c2, const vec3 c3,
        const vec2 p);

    void GetTriangleVerts(uint64_t iTriangle, float4* p1, float4* p2, float4* p3, float3* c1, float3* c2, float3* c3);

    void __vectorcall ConvertFragsToColors(const vec4 frags, uint32_t colors[4]);

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
