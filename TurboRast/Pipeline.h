#pragma once

class TRPipelineThread;

// Data shared between pipeline threads
struct SharedPipelineData
{
    int NumThreads;
    HANDLE ShutdownEvent;

    // Needs to be reset from a pipeline thread in between each
    // render packet. Requires synch barriers
    std::atomic_uint64_t CurrentVertex;
    std::atomic_uint64_t CurrentTriangle;

    SSEVSOutput* VSOutputs;

    // synchronization barriers

    // Starts as 0. Each thread increments it as it passes barrier.
    // When == NumThreads, all threads have passed by barrier. Also, if
    // incrementing it returns NumThreads, that is last thread through barrier.
    // Barrier needs to be reset to 0 when it's safe to do so
    std::atomic_int32_t JoinBarrier;

    // Starts as 0. Each thread that waits on this tries to compare/exchange it
    // from 1 to 1, which will fail when it's 0. When owning thread wants to signal it,
    // it sets it to 1. Needs to be reset to 0 when it's safe to do so
    std::atomic_bool WaitBarrier1;
    std::atomic_bool WaitBarrier2;
};

// A single render command (ie. a Draw() call)
struct RenderCommand
{
    // Assigned by pipeline
    uint64_t FenceValue;

    //========================================
    // Pipeline configuration
    //========================================

    // Input (currently only support triangle list)
    std::shared_ptr<const TRVertexBuffer> VertexBuffer;
    uint64_t NumVertices;
    uint64_t NumTriangles;

    // Vertex Stage
    pfnSSEVertexShader VertexShader;
    void* VSConstantBuffer;

    // Pixel Stage
    pfnSSEPixelShader PixelShader;
    void* PSConstantBuffer;

    // Output (no blend ops or z buffer support yet)
    std::shared_ptr<const TRTexture2D> RenderTarget;
};

class TRPipeline
{
    NON_COPYABLE(TRPipeline);

public:
    TRPipeline();
    virtual ~TRPipeline();

    bool Initialize();

    bool Render(const std::shared_ptr<RenderCommand>& command);
    void FlushAndWait();

    // Called by pipeline thread #0 when all threads have completed processing a command
    void NotifyCompletion(uint64_t fenceValue);

private:
    static const int RenderQueueDepth = 32;
    static const DWORD ThreadShutdownTimeoutMs = 10000;

    SharedPipelineData SharedData;
    std::vector<uint8_t> Scratch;
    Microsoft::WRL::Wrappers::Event ShutdownEvent;
    std::vector<std::unique_ptr<TRPipelineThread>> Threads;

    std::atomic_uint64_t LastCompletedFenceValue = 0;
    uint64_t CurrentFenceValue = 0;
};
