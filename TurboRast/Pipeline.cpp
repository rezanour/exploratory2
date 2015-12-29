#include "Precomp.h"
#include "Pipeline.h"
#include "PipelineThread.h"

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
}

bool TRPipeline::Initialize()
{
    ShutdownEvent.Attach(CreateEvent(nullptr, TRUE, FALSE, nullptr));
    if (!ShutdownEvent.IsValid())
    {
        assert(false);
        return false;
    }

    // TODO: formalize numThreads determination
    SharedData.NumThreads = 8;
    SharedData.ShutdownEvent = ShutdownEvent.Get();
    SharedData.JoinBarrier = 0;
    SharedData.WaitBarrier = 0;

    SharedData.CurrentVertex = 0;
    SharedData.CurrentTriangle = 0;

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
        // Allow other threads & hyperthreads to make progress
        // before checking again
        _mm_pause();
        _mm_pause();
        _mm_pause();
        _mm_pause();
        _mm_pause();
    }
}

void TRPipeline::NotifyCompletion(uint64_t fenceValue)
{
    assert(fenceValue > LastCompletedFenceValue);
    LastCompletedFenceValue = fenceValue;
}
