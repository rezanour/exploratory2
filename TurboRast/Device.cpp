#include "Precomp.h"
#include "Device.h"
#include "Pipeline.h"
#include "PipelineThread.h"
#include "VertexBuffer.h"
#include "Texture2D.h"

TRDevice::TRDevice()
{
}

TRDevice::~TRDevice()
{
}

bool TRDevice::Initialize()
{
    Pipeline = std::make_unique<TRPipeline>();
    if (!Pipeline->Initialize())
    {
        assert(false);
        return false;
    }

    return true;
}

void TRDevice::ClearRenderTarget(const std::shared_ptr<TRTexture2D>& renderTarget) const
{
    uint32_t size = renderTarget->GetPitchInPixels() * renderTarget->GetHeight();
    memset(renderTarget->GetData(), 0, size * sizeof(uint32_t));
}

void TRDevice::Draw(uint64_t vertexCount, uint64_t baseVertex)
{
    UNREFERENCED_PARAMETER(baseVertex);

    std::shared_ptr<RenderCommand> command = std::make_shared<RenderCommand>();

    // Input
    command->VertexBuffer = VertexBuffer;
    command->NumVertices = vertexCount;
    command->NumTriangles = vertexCount / 3;

    // Vertex Shader
    command->VertexShader = VertexShader;
    command->VSConstantBuffer = VSConstantBuffer;

    // Pixel Shader
    command->PixelShader = PixelShader;
    command->PSConstantBuffer = PSConstantBuffer;

    // Output
    command->RenderTarget = RenderTarget;

    Pipeline->Render(command);
}
