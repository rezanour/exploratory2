#include "Precomp.h"
#include "Renderer.h"
#include "DebugUtil.h"

// Shaders
#include "DrawQuad_vs.h"
#include "DrawQuad_ps.h"

//=============================================================================
// Renderer
//=============================================================================

Renderer::Renderer(HWND targetWindow)
{
    ZeroMemory(&Viewport, sizeof(Viewport));
    InitializeGraphics(targetWindow);
}

Renderer::~Renderer()
{
}

std::shared_ptr<Image> Renderer::CreateColorImage(uint32_t width, uint32_t height, const uint32_t* optionalSourceData)
{
    return CreateImageInternal(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, ImageType::Color, optionalSourceData, sizeof(uint32_t));
}

std::shared_ptr<Image> Renderer::CreateDepthImage(uint32_t width, uint32_t height, const float* optionalSourceData)
{
    return CreateImageInternal(width, height, DXGI_FORMAT_R32_FLOAT, ImageType::Depth, optionalSourceData, sizeof(float));
}

std::shared_ptr<Image> Renderer::CreateLuminanceImage(uint32_t width, uint32_t height, const float* optionalSourceData)
{
    return CreateImageInternal(width, height, DXGI_FORMAT_R32_FLOAT, ImageType::Luminance, optionalSourceData, sizeof(float));
}

std::shared_ptr<Image> Renderer::CreateNormalsImage(uint32_t width, uint32_t height, const uint32_t* optionalSourceData)
{
    return CreateImageInternal(width, height, DXGI_FORMAT_R8G8B8A8_UNORM, ImageType::Normals, optionalSourceData, sizeof(uint32_t));
}

void Renderer::Clear()
{
    static const float ClearColor[]{ 0.f, 0.f, 0.f, 1.f };
    Context->ClearRenderTargetView(BackBufferRTV.Get(), ClearColor);
}

void Renderer::FillColorImage(const uint32_t* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY)
{
    assert(dest->Type == ImageType::Color);
    FillImageInternal(sourceData, width, height, sizeof(uint32_t), dest, destX, destY);
}

void Renderer::FillDepthImage(const float* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY)
{
    assert(dest->Type == ImageType::Depth);
    FillImageInternal(sourceData, width, height, sizeof(float), dest, destX, destY);
}

void Renderer::FillLuminanceImage(const float* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY)
{
    assert(dest->Type == ImageType::Luminance);
    FillImageInternal(sourceData, width, height, sizeof(float), dest, destX, destY);
}

void Renderer::FillNormalsImage(const uint32_t* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY)
{
    assert(dest->Type == ImageType::Normals);
    FillImageInternal(sourceData, width, height, sizeof(uint32_t), dest, destX, destY);
}

void Renderer::CopyImage(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest)
{
    // if easy path is available, take it
    if (source->Format == dest->Format &&
        source->Width == dest->Width &&
        source->Height == dest->Height)
    {
        Context->CopyResource(dest->Texture.Get(), source->Texture.Get());
        return;
    }

    // Otherwise, we need to use a shader

    FAIL(L"%s Not implemented!", __FUNCTIONW__);
}

void Renderer::ColorToLum(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest)
{
    UNREFERENCED_PARAMETER(source);
    UNREFERENCED_PARAMETER(dest);

    FAIL(L"%s Not implemented!", __FUNCTIONW__);
}

void Renderer::LumToNormals(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest)
{
    UNREFERENCED_PARAMETER(source);
    UNREFERENCED_PARAMETER(dest);

    FAIL(L"%s Not implemented!", __FUNCTIONW__);
}

void Renderer::DepthToNormals(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest)
{
    UNREFERENCED_PARAMETER(source);
    UNREFERENCED_PARAMETER(dest);

    FAIL(L"%s Not implemented!", __FUNCTIONW__);
}

void Renderer::DrawImage(const std::shared_ptr<Image>& image, int x, int y, uint32_t width, uint32_t height)
{
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(DrawQuadPS.Get(), nullptr, 0);

    static const uint32_t stride = sizeof(DrawQuadVertex);
    uint32_t offset = 0;
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetInputLayout(DrawQuadIL.Get());
    Context->IASetVertexBuffers(0, 1, DrawQuadVB.GetAddressOf(), &stride, &offset);

    DrawQuadVSConstants constants{};
    constants.InvViewportSize = XMFLOAT2(1.f / Viewport.Width, 1.f / Viewport.Height);
    constants.Offset = XMINT2(x, y);
    constants.Size = XMUINT2(width, height);
    Context->UpdateSubresource(DrawQuadVS_CB.Get(), 0, nullptr, &constants, sizeof(constants), 0);
    Context->VSSetConstantBuffers(0, 1, DrawQuadVS_CB.GetAddressOf());

    Context->PSSetShaderResources(0, 1, image->SRV.GetAddressOf());
    Context->PSSetSamplers(0, 1, LinearSampler.GetAddressOf());

    Context->OMSetRenderTargets(1, BackBufferRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &Viewport);

    Context->Draw(6, 0);
}

void Renderer::Present()
{
    HRESULT hr = SwapChain->Present(1, 0);
    CHECKHR(hr, L"Present failed. hr = 0x%08x.", hr);
}

void Renderer::InitializeGraphics(HWND targetWindow)
{
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&Factory));
    CHECKHR(hr, L"CreateDXGIFactory1 failed. hr = 0x%08x.", hr);

    hr = Factory->EnumAdapters(0, &Adapter);
    CHECKHR(hr, L"EnumAdapters failed. hr = 0x%08x.", hr);

    UINT flags = 0;
#ifdef _DEBUG
    flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

    hr = D3D11CreateDevice(Adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags,
        &featureLevel, 1, D3D11_SDK_VERSION, &Device, nullptr, &Context);
    CHECKHR(hr, L"D3D11CreateDevice failed. hr = 0x%08x.", hr);

    DXGI_SWAP_CHAIN_DESC1 scd{};
    scd.BufferCount = 2;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.Width = 1280;
    scd.Height = 720;
    scd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;

    hr = Factory->CreateSwapChainForHwnd(Device.Get(), targetWindow, &scd, nullptr, nullptr, &SwapChain);
    CHECKHR(hr, L"CreateSwapChainForHwnd failed. hr = 0x%08x.", hr);

    hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
    CHECKHR(hr, L"GetBuffer failed. hr = 0x%08x.", hr);

    hr = Device->CreateRenderTargetView(BackBuffer.Get(), nullptr, &BackBufferRTV);
    CHECKHR(hr, L"CreateRenderTargetView failed. hr = 0x%08x.", hr);

    D3D11_SAMPLER_DESC sd{};
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;

    hr = Device->CreateSamplerState(&sd, &LinearSampler);
    CHECKHR(hr, L"CreateSamplerState failed. hr = 0x%08x.", hr);

    Viewport.Width = (float)scd.Width;
    Viewport.Height = (float)scd.Height;
    Viewport.MaxDepth = 1.f;

    // DrawQuad
    hr = Device->CreateVertexShader(DrawQuad_vs, sizeof(DrawQuad_vs), nullptr, &DrawQuadVS);
    CHECKHR(hr, L"CreateVertexShader failed. hr = 0x%08x.", hr);

    hr = Device->CreatePixelShader(DrawQuad_ps, sizeof(DrawQuad_ps), nullptr, &DrawQuadPS);
    CHECKHR(hr, L"CreatePixelShader failed. hr = 0x%08x.", hr);

    D3D11_INPUT_ELEMENT_DESC elems[2]{};
    elems[0].Format = DXGI_FORMAT_R32G32_FLOAT;
    elems[0].SemanticName = "POSITION";
    elems[1].AlignedByteOffset = sizeof(XMFLOAT2);
    elems[1].Format = DXGI_FORMAT_R32G32_FLOAT;
    elems[1].SemanticName = "TEXCOORD";

    hr = Device->CreateInputLayout(elems, _countof(elems), DrawQuad_vs, sizeof(DrawQuad_vs), &DrawQuadIL);
    CHECKHR(hr, L"CreateInputLayout failed. hr = 0x%08x.", hr);

    DrawQuadVertex drawQuadVerts[]
    {
        { XMFLOAT2(-1, 1), XMFLOAT2(0, 0) },
        { XMFLOAT2(1, 1), XMFLOAT2(1, 0) },
        { XMFLOAT2(1, -1), XMFLOAT2(1, 1) },
        { XMFLOAT2(-1, 1), XMFLOAT2(0, 0) },
        { XMFLOAT2(1, -1), XMFLOAT2(1, 1) },
        { XMFLOAT2(-1, -1), XMFLOAT2(0, 1) },
    };

    D3D11_BUFFER_DESC bd{};
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = sizeof(drawQuadVerts);
    bd.StructureByteStride = sizeof(DrawQuadVertex);

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = drawQuadVerts;
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = bd.ByteWidth;

    hr = Device->CreateBuffer(&bd, &init, &DrawQuadVB);
    CHECKHR(hr, L"CreateBuffer failed. hr = 0x%08x.", hr);

    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.ByteWidth = sizeof(DrawQuadVSConstants);
    bd.StructureByteStride = bd.ByteWidth;

    hr = Device->CreateBuffer(&bd, nullptr, &DrawQuadVS_CB);
    CHECKHR(hr, L"CreateBuffer failed. hr = 0x%08x.", hr);
}

std::shared_ptr<Image> Renderer::CreateImageInternal(uint32_t width, uint32_t height, DXGI_FORMAT format, ImageType type, const void* optionalSourceData, uint32_t sourceStride)
{
    std::shared_ptr<Image> image(new Image);
    image->Type = type;
    image->Width = width;
    image->Height = height;
    image->Format = format;

    D3D11_TEXTURE2D_DESC td{};
    td.ArraySize = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.Format = format;
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.SampleDesc.Count = 1;

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = optionalSourceData;
    init.SysMemPitch = sourceStride * td.Width;
    init.SysMemSlicePitch = sourceStride * td.Width * td.Height;

    HRESULT hr = Device->CreateTexture2D(&td, optionalSourceData ? &init : nullptr, &image->Texture);
    CHECKHR(hr, L"CreateTexture2D failed. hr = 0x%08x.", hr);

    hr = Device->CreateShaderResourceView(image->Texture.Get(), nullptr, &image->SRV);
    CHECKHR(hr, L"CreateShaderResourceView failed. hr = 0x%08x.", hr);

    hr = Device->CreateRenderTargetView(image->Texture.Get(), nullptr, &image->RTV);
    CHECKHR(hr, L"CreateRenderTargetView failed. hr = 0x%08x.", hr);

    return image;
}

void Renderer::FillImageInternal(const void* sourceData, uint32_t width, uint32_t height, uint32_t sourceStride, const std::shared_ptr<Image>& dest, int destX, int destY)
{
    D3D11_BOX box{};
    box.left = destX;
    box.top = destY;
    box.right = destX + width;
    box.bottom = destY + height;
    box.back = 1;

    Context->UpdateSubresource(dest->Texture.Get(), 0, &box, sourceData, sourceStride * width, sourceStride * width * height);
}
