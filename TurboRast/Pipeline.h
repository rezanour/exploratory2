#pragma once

class TRPipelineThread;
class TRPipelineThread2;

enum class RastStrategy
{
    OneTrianglePerThread = 0,
    OneTilePerThread,
    ScreenTileDDAPerThread,
    NumStrategies,
};

// Binning
struct Triangle
{
    uint64_t iTriangle;
    float2 p1, p2, p3;
    float2 e1, e2, e3;
    float2 o1, o2, o3;
    Triangle* Next = nullptr;
};

struct Bin
{
    std::atomic<Triangle*> Head;
};

// Data shared between pipeline threads
struct SharedPipelineData
{
    int NumThreads = 1;
    HANDLE ShutdownEvent = nullptr;

    // Needs to be reset from a pipeline thread in between each
    // render packet. Requires synch barriers
    std::atomic_int64_t CurrentVertex = 0;
    std::atomic_int64_t CurrentTriangle = 0;

    SSEVSOutput* VSOutputs = nullptr;
    int64_t MaxVSOutputs = 0;
    Triangle* TriangleMemory = nullptr;
    int64_t MaxTriangles = 0;
    std::atomic_int64_t CurrentTriangleBin = 0;

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

enum class VertexAttributeType
{
    Float = 0,
    Float2,
    Float3,
    Float4,
};

// Description of single vertex attribute
struct VertexAttributeDesc
{
    int ByteOffset;
    VertexAttributeType Type;
    const char* Semantic;
};

// A single render command (ie. a Draw() call)
struct RenderCommand
{
    // Assigned by pipeline
    uint64_t FenceValue;

    //========================================
    // Vertex layout information
    //========================================

    std::vector<VertexAttributeDesc> InputVertexLayout;
    std::vector<VertexAttributeDesc> OutputVertexLayout;
    int64_t InputVertexStride;
    int64_t OutputVertexStride;

    //========================================
    // Pipeline configuration
    //========================================

    // Input (currently only support triangle list)
    std::shared_ptr<const TRVertexBuffer> VertexBuffer;
    int64_t NumVertices;
    int64_t NumTriangles;

    // Vertex Stage
    pfnSSEVertexShader VertexShader;
    pfnVertexShader VertexShader2;
    pfnStreamVertexShader VertexShader3;
    void* VSConstantBuffer;

    // Pixel Stage
    pfnSSEPixelShader PixelShader;
    pfnPixelShader PixelShader2;
    pfnPixelShader3 PixelShader3;
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
    Microsoft::WRL::Wrappers::Event ShutdownEvent;
    std::vector<std::unique_ptr<TRPipelineThread2>> Threads;

    std::atomic_uint64_t LastCompletedFenceValue = 0;
    uint64_t CurrentFenceValue = 0;
};
