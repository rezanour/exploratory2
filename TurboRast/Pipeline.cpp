#include "Precomp.h"
#include "Pipeline.h"
#include "PipelineThread.h"

static int GetNumLogicalCores()
{
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION procInfos[64];
    DWORD sizeOfProcInfos = sizeof(procInfos);
    if (GetLogicalProcessorInformation(procInfos, &sizeOfProcInfos))
    {
        int numProcInfos = sizeOfProcInfos / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        int numLogicalCores = 0;
        for (int i = 0; i < numProcInfos; ++i)
        {
            if (procInfos[i].Relationship == RelationProcessorCore)
            {
                numLogicalCores += (int)_mm_popcnt_u64(procInfos[i].ProcessorMask);
            }
        }
        return numLogicalCores;
    }
    else
    {
        assert(false);
        return 1;
    }
}

TRPipeline::TRPipeline()
{
}

TRPipeline::~TRPipeline()
{
    std::vector<HANDLE> threadHandles;
    for (auto& thread : Threads)
    {
        threadHandles.push_back(thread->GetThreadHandle());
    }

    if (!threadHandles.empty())
    {
        assert(ShutdownEvent.IsValid());
        SetEvent(ShutdownEvent.Get());
        DWORD result = WaitForMultipleObjects((DWORD)threadHandles.size(), threadHandles.data(), TRUE, ThreadShutdownTimeoutMs);
        assert(result == WAIT_OBJECT_0);
        UNREFERENCED_PARAMETER(result);
    }
    Threads.clear();

    // Don't delete scratch until after all threads are done running
    Scratch.clear();

    delete[] SharedData.VertexStartTime;
    delete[] SharedData.VertexStopTime;
    delete[] SharedData.TriangleStartTime;
    delete[] SharedData.TriangleStopTime;
}

bool TRPipeline::Initialize()
{
    ShutdownEvent.Attach(CreateEvent(nullptr, TRUE, FALSE, nullptr));
    if (!ShutdownEvent.IsValid())
    {
        assert(false);
        return false;
    }

    // Note that matching (or exceeding) the number of logical CPUs on the system
    // reduces performance due to thread starvation and spin contention.
    // Remember that the main render thread (app render thread) is still running
    // and submitting more commands, flushing, etc... so it will claim at least 1
    // thread from the # of logical CPUs. And since we aren't doing any IO at the
    // moment, we are purely CPU bound on the worker threads. So, a decent number
    // to pick here is 1 or 2 less than the total # of logical CPUs.
    // NOTE: It may also be useful to look into using Windows UMS to micro-manage the threads.
    SharedData.NumThreads = std::max(GetNumLogicalCores() - 1, 1);

    SharedData.ShutdownEvent = ShutdownEvent.Get();

    SharedData.StatsEnabled = true;
    if (SharedData.StatsEnabled)
    {
        SharedData.VertexStartTime = new uint64_t[SharedData.NumThreads];
        SharedData.VertexStopTime = new uint64_t[SharedData.NumThreads];
        SharedData.TriangleStartTime = new uint64_t[SharedData.NumThreads];
        SharedData.TriangleStopTime = new uint64_t[SharedData.NumThreads];
    }

    // Configure default scratch space for pipeline threads
    Scratch.resize(4 * 1024 * 1024);    // 4 MB
    SharedData.VSOutputs = (SSEVSOutput*)Scratch.data();

    for (int i = 0; i < SharedData.NumThreads; ++i)
    {
        std::unique_ptr<TRPipelineThread> thread = std::make_unique<TRPipelineThread>(i, this, &SharedData);
        if (!thread->Initialize())
        {
            assert(false);
            return false;
        }

        Threads.push_back(std::move(thread));
    }

    return true;
}

bool TRPipeline::Render(const std::shared_ptr<RenderCommand>& command)
{
    // If the current scratch space isn't large enough, we need to resize.
    // However, we can't resize while there is outstanding work using the
    // current scratch address, so we need to stall until oustanding work
    // completes
    if (command->NumVertices * sizeof(SSEVSOutput) > Scratch.size())
    {
        FlushAndWait();

        Scratch.resize(command->NumVertices * sizeof(SSEVSOutput));

        // Update the shared data pointer
        SharedData.VSOutputs = (SSEVSOutput*)Scratch.data();
    }

    command->FenceValue = ++CurrentFenceValue;

    for (auto& thread : Threads)
    {
        thread->QueueCommand(command);
    }

    return true;
}

void TRPipeline::FlushAndWait()
{
    while (LastCompletedFenceValue < CurrentFenceValue)
    {
        ::Sleep(1);
    }
}

void TRPipeline::NotifyCompletion(uint64_t fenceValue)
{
    assert(fenceValue > LastCompletedFenceValue);
    LastCompletedFenceValue = fenceValue;
}
