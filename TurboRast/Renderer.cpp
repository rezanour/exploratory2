#include "Precomp.h"
#include "Renderer.h"
#include "RenderThread.h"

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
    static const uint32_t DefaultScratchSize = 65536;
    VSOutputScratch.resize(DefaultScratchSize);
    PSOutputScratch.resize(DefaultScratchSize);

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
void Renderer::IASetVertexBuffer(Vertex* vertexBuffer, uint64_t numVerts)
{
    // TODO: Need to double buffer or something, since this buffer is still in use by
    // outstanding render jobs!

    // Restructure input data into SSE friendly layout (AOSOA)
    VertexBuffer.clear();
    for (uint64_t i = 0; i < numVerts; i += 4)
    {
        VertexBuffer.push_back(SSEVertexBlock(&vertexBuffer[i], std::min(numVerts - i, 4ull)));
    }
}

void Renderer::Draw(uint64_t vertexCount, uint64_t baseVertex)
{
    UNREFERENCED_PARAMETER(baseVertex);

    if (vertexCount > VSOutputScratch.size())
    {
        // TODO: Need to flush all outstanding work before resizing!!!
        VSOutputScratch.resize(vertexCount);
        PSOutputScratch.resize(vertexCount);
    }

    std::unique_ptr<SharedRenderData> renderData = std::make_unique<SharedRenderData>();
    renderData->JobID = NextJobID++;

    renderData->RenderTarget = RenderTarget;
    renderData->RTWidth = RTWidth;
    renderData->RTHeight = RTHeight;
    renderData->RTPitchInPixels = RTPitchInPixels;

    renderData->VertexBuffer = VertexBuffer.data();
    renderData->VertexShader = VertexShader;
    renderData->PixelShader = PixelShader;
    renderData->VSConstantBuffer = VSConstantBuffer;
    renderData->PSConstantBuffer = PSConstantBuffer;
    renderData->VSOutputs = VSOutputScratch.data();
    renderData->PSOutputs = PSOutputScratch.data();

    renderData->ProcessedVertices = 0;
    renderData->NumVertices = vertexCount;
    renderData->NumVertexBlocks = VertexBuffer.size();
    renderData->ProcessedTriangles = 0;
    renderData->NumTriangles = vertexCount / 3;

    for (auto& thread : RenderThreads)
    {
        thread->QueueRendering(renderData.get());
    }

    InFlightRenderJobs.push(std::move(renderData));
}
