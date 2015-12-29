#include "Precomp.h"
#include "Renderer.h"
#include "RenderThread.h"
#include "VertexBuffer.h"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
    std::vector<HANDLE> handles;
    for (auto& thread : RenderThreads)
    {
        thread->SignalShutdown();
        handles.push_back(thread->ThreadHandle());
    }
    // TODO: don't wait INFINITE
    WaitForMultipleObjects((DWORD)handles.size(), handles.data(), TRUE, INFINITE);
}

bool Renderer::Initialize(uint32_t numThreads)
{
    for (uint32_t i = 0; i < numThreads; ++i)
    {
        std::unique_ptr<RenderThread> thread = std::make_unique<RenderThread>(i);
        if (!thread->Initialize())
        {
            assert(false);
            return false;
        }

        RenderThreads.push_back(std::move(thread));
    }

    return true;
}

// TODO: Should do this processing offline during vertex buffer object creation
void Renderer::IASetVertexBuffer(const VertexBuffer* const vertexBuffer)
{
    TheVertexBuffer = vertexBuffer;
}

void Renderer::Draw(uint64_t vertexCount, uint64_t baseVertex)
{
    UNREFERENCED_PARAMETER(baseVertex);

    std::unique_ptr<SharedRenderData> renderData = std::make_unique<SharedRenderData>();
    renderData->JobID = NextJobID++;

    renderData->RenderTarget = RenderTarget;
    renderData->TheVertexBuffer = TheVertexBuffer;
    renderData->VertexShader = VertexShader;
    renderData->PixelShader = PixelShader;
    renderData->VSConstantBuffer = VSConstantBuffer;
    renderData->PSConstantBuffer = PSConstantBuffer;

    renderData->ProcessedVertices = 0;
    renderData->NumVertices = vertexCount;
    renderData->ProcessedTriangles = 0;
    renderData->NumTriangles = vertexCount / 3;

    for (auto& thread : RenderThreads)
    {
        thread->QueueRendering(renderData.get());
    }

    InFlightRenderJobs.push(std::move(renderData));
}
