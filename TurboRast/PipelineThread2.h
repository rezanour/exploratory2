#pragma once

struct RenderCommand;
class TRPipeline;
struct Triangle;

// One (of potentially many) pipeline threads.
class TRPipelineThread2
{
    NON_COPYABLE(TRPipelineThread2);

public:
    TRPipelineThread2(int id, TRPipeline* pipeline, SharedPipelineData* sharedData);
    ~TRPipelineThread2();

    bool Initialize();

    const HANDLE GetThreadHandle() const { return TheThread.Get(); }

    void QueueCommand(const std::shared_ptr<RenderCommand>& command);

private:
    std::shared_ptr<RenderCommand> GetNextCommand();

    static DWORD CALLBACK s_ThreadProc(PVOID context);
    void ThreadProc();

    void ProcessVertices();

    struct PipelineTriangle
    {
        int64_t i1, i2, i3;
    };

    // returns 1 if the entire triangle can be trivially rejected.
    // returns 2 if the entier triangle can be trivially accepted.
    // returns 0 if further clipping is needed
    int PreClipTriangle(const float4* v1, const float4* v2, const float4* v3);

    // clips the triangle against the viewport edges and appends the results
    // to the PipelineTriangle list
    void ClipTriangle(int64_t i1, int64_t i2, int64_t i3);
    int64_t ClipEdge(int64_t i1, int64_t i2, float d1, float d2);

    void AppendTriangle(int64_t i1, int64_t i2, int64_t i3);

    void DDARastTriangle(const float4* v1, const float4* v2, const float4* v3, uint32_t* renderTarget, int pitch);

private:
    int ID;
    TRPipeline* Pipeline;
    SharedPipelineData* SharedData;
    Thread TheThread;

    // For convenience, a pointer to the command being currently
    // processed is stored here, avoiding the need to pass to every function
    RenderCommand* CurrentCommand = nullptr;

    // Also cache the SV_POSITION offset within the OutputVertices
    // and output vertex stride
    int64_t PositionOffset = 0;
    int64_t OutputVertexStride = 0;

    uint8_t* VertexMemory = nullptr;
    int64_t VertexMemoryOffset = 0;
    int64_t VertexMemoryCapacity = 0;

    PipelineTriangle* PipelineTriangles = nullptr;
    int64_t PipelineTriangleCount = 0;
    int64_t PipelineTriangleCapacity = 0;

    Microsoft::WRL::Wrappers::CriticalSection CommandsLock;
    Microsoft::WRL::Wrappers::Event CommandReady;
    std::deque<std::shared_ptr<RenderCommand>> Commands;
};

