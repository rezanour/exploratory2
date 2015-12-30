#pragma once

class TRPipelineThread;

enum class RastStrategy
{
    OneTrianglePerThread = 0,
    OneTilePerThread,
    NumStrategies,
};

// Binning
struct Triangle
{
    float2 p1, p2, p3;
    float2 e1, e2, e3;
    float2 o1, o2, o3;
    uint64_t iTriangle;
};

struct Bin
{
    static const size_t MaxTrianglesPerBin = 64 * 1024; // 128K
    Triangle Triangles[MaxTrianglesPerBin];
    std::atomic_uint64_t CurrentTriangle;
};

// Data shared between pipeline threads
struct SharedPipelineData
{
    int NumThreads = 1;
    HANDLE ShutdownEvent = nullptr;

    // Needs to be reset from a pipeline thread in between each
    // render packet. Requires synch barriers
    std::atomic_uint64_t CurrentVertex = 0;
    std::atomic_uint64_t CurrentTriangle = 0;

    SSEVSOutput* VSOutputs = nullptr;

    // synchronization barriers

    // Starts as 0. Each thread increments it as it passes barrier.
    // When == NumThreads, all threads have passed by barrier. Also, if
    // incrementing it returns NumThreads, that is last thread through barrier.
    // Barrier needs to be reset to 0 when it's safe to do so
    std::atomic_int32_t JoinBarrier = 0;

    // Starts as false. Each thread waits on this to change to true.
    // When a thread wants to signal it, they switch it to true, releasing
    // the other threads. It needs to be reset to false when it is safe to do so.
    std::atomic_bool InitWaitBarrier = false;
    std::atomic_bool CompletionWaitBarrier = false;

    // Stats
    bool StatsEnabled = false;
    uint64_t* VertexStartTime = nullptr;
    uint64_t* VertexStopTime = nullptr;
    uint64_t* TriangleStartTime = nullptr;
    uint64_t* TriangleStopTime = nullptr;

    RastStrategy CurrentRastStrategy = RastStrategy::OneTrianglePerThread;

    static const int TileSize = 64;
    int NumHorizBins = 0;
    int NumVertBins = 0;
    int NumTotalBins = 0;
    std::unique_ptr<Bin[]> Bins;
    std::atomic_int CurrentBin;

    std::atomic_int32_t BinningJoinBarrier = 0;
    std::atomic_bool BinningWaitBarrier = false;
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
