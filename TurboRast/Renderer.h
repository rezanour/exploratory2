#pragma once

class RenderThread;
class VertexBuffer;
class Texture;

// The entry point into TurboRast renderer

class Renderer
{
public:
    Renderer();
    ~Renderer();

    bool Initialize(uint32_t numThreads);

    void OMSetRenderTarget(const Texture* texture) { RenderTarget = texture; }
    void IASetVertexBuffer(const VertexBuffer* const vertexBuffer);
    void VSSetShader(pfnSSEVertexShader vertexShader) { VertexShader = vertexShader; }
    void VSSetConstantBuffer(void* constantBuffer) { VSConstantBuffer = constantBuffer; }
    void PSSetShader(pfnSSEPixelShader pixelShader) { PixelShader = pixelShader; }
    void PSSetConstantBuffer(void* constantBuffer) { PSConstantBuffer = constantBuffer; }

    void Draw(uint64_t vertexCount, uint64_t baseVertex);

private:
    uint64_t NextJobID = 0;

    std::vector<std::unique_ptr<RenderThread>> RenderThreads;
    std::queue<std::unique_ptr<SharedRenderData>> InFlightRenderJobs;

    std::vector<SSEVSOutput> VSOutputScratch;
    std::vector<SSEPSOutput> PSOutputScratch;

    // Current pipeline state
    const Texture* RenderTarget = nullptr;
    const VertexBuffer* TheVertexBuffer = nullptr;
    pfnSSEVertexShader VertexShader = nullptr;
    pfnSSEPixelShader PixelShader = nullptr;
    void* VSConstantBuffer = nullptr;
    void* PSConstantBuffer = nullptr;
};
