#pragma once

#include "Pipeline.h"
class TRVertexBuffer;
class TRTexture2D;

// The entry point into TurboRast renderer

class TRDevice
{
    NON_COPYABLE(TRDevice);

public:
    TRDevice();
    virtual ~TRDevice();

    bool Initialize();

    void IASetVertexBuffer(const std::shared_ptr<const TRVertexBuffer>& vertexBuffer) { VertexBuffer = vertexBuffer; }
    void VSSetShader(pfnSSEVertexShader vertexShader) { VertexShader = vertexShader; }
    void VSSetConstantBuffer(void* constantBuffer) { VSConstantBuffer = constantBuffer; }
    void PSSetShader(pfnSSEPixelShader pixelShader) { PixelShader = pixelShader; }
    void PSSetConstantBuffer(void* constantBuffer) { PSConstantBuffer = constantBuffer; }
    void OMSetRenderTarget(const std::shared_ptr<const TRTexture2D>& renderTarget) { RenderTarget = renderTarget; }

    void Draw(uint64_t vertexCount, uint64_t baseVertex);

    void FlushAndWait() { Pipeline->FlushAndWait(); }

private:
    uint64_t NextJobID = 0;

    std::unique_ptr<TRPipeline> Pipeline;

    std::vector<SSEVSOutput> VSOutputScratch;
    std::vector<SSEPSOutput> PSOutputScratch;

    // Current pipeline state
    std::shared_ptr<const TRTexture2D> RenderTarget;
    std::shared_ptr<const TRVertexBuffer> VertexBuffer;
    pfnSSEVertexShader VertexShader = nullptr;
    pfnSSEPixelShader PixelShader = nullptr;
    void* VSConstantBuffer = nullptr;
    void* PSConstantBuffer = nullptr;
};
