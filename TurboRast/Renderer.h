#pragma once

class RenderThread;

// The entry point into TurboRast renderer

class Renderer
{
public:
    Renderer();
    ~Renderer();

    bool Initialize(uint32_t numThreads);

    void OMSetRenderTarget(void* pRenderTarget, int width, int height, int pitchInBytes)
    {
        RenderTarget = (uint32_t*)pRenderTarget;
        RTWidth = width;
        RTHeight = height;
        RTPitchInPixels = pitchInBytes / sizeof(uint32_t);
    }

    void IASetVertexBuffer(Vertex* vertexBuffer, uint64_t numVerts);
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
    uint32_t* RenderTarget = nullptr;
    int RTWidth = 0;
    int RTHeight = 0;
    int RTPitchInPixels = 0;
    std::vector<SSEVertexBlock> VertexBuffer;
    pfnSSEVertexShader VertexShader = nullptr;
    pfnSSEPixelShader PixelShader = nullptr;
    void* VSConstantBuffer = nullptr;
    void* PSConstantBuffer = nullptr;
};
