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

    void SetInputLayout(const std::vector<VertexAttributeDesc>& layout, int64_t stride) { InputVertexLayout = layout; InputVertexStride = stride; }
    void SetVSOutputLayout(const std::vector<VertexAttributeDesc>& layout, int64_t stride) { OutputVertexLayout = layout; OutputVertexStride = stride; }
    void IASetVertexBuffer(const std::shared_ptr<const TRVertexBuffer>& vertexBuffer) { VertexBuffer = vertexBuffer; }
    void VSSetShader(pfnSSEVertexShader vertexShader) { VertexShader = vertexShader; }
    void VSSetShader2(pfnVertexShader vertexShader) { VertexShader2 = vertexShader; }
    void VSSetShader3(pfnStreamVertexShader vertexShader) { VertexShader3 = vertexShader; }
    void VSSetConstantBuffer(void* constantBuffer) { VSConstantBuffer = constantBuffer; }
    void PSSetShader(pfnSSEPixelShader pixelShader) { PixelShader = pixelShader; }
    void PSSetShader2(pfnPixelShader pixelShader) { PixelShader2 = pixelShader; }
    void PSSetConstantBuffer(void* constantBuffer) { PSConstantBuffer = constantBuffer; }
    void OMSetRenderTarget(const std::shared_ptr<const TRTexture2D>& renderTarget) { RenderTarget = renderTarget; }
    void ClearRenderTarget(const std::shared_ptr<TRTexture2D>& renderTarget) const;

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
    pfnVertexShader VertexShader2 = nullptr;
    pfnStreamVertexShader VertexShader3 = nullptr;
    pfnPixelShader PixelShader2 = nullptr;
    void* VSConstantBuffer = nullptr;
    void* PSConstantBuffer = nullptr;
    std::vector<VertexAttributeDesc> InputVertexLayout;
    std::vector<VertexAttributeDesc> OutputVertexLayout;
    int64_t InputVertexStride;
    int64_t OutputVertexStride;
};
