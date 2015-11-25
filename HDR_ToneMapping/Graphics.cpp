#include "Precomp.h"
#include "Graphics.h"
#include "Debug.h"

// Shaders
#include "DrawQuad_vs.h"
#include "DrawTexture_ps.h"
#include "HighPass_ps.h"
#include "Blur_ps.h"

static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> Context;
static ComPtr<IDXGISwapChain> SwapChain;
static ComPtr<ID3D11Texture2D> BackBuffer;
static ComPtr<ID3D11RenderTargetView> BackBufferRTV;
static ComPtr<ID3D11RenderTargetView> HighPassRTV;
static ComPtr<ID3D11ShaderResourceView> HighPassSRV;
static ComPtr<ID3D11RenderTargetView> BlurRTV;
static ComPtr<ID3D11ShaderResourceView> BlurSRV;
static ComPtr<ID3D11ShaderResourceView> FilmLutSRV;
static ComPtr<ID3D11InputLayout> InputLayout;
static ComPtr<ID3D11VertexShader> DrawQuadVS;
static ComPtr<ID3D11PixelShader> DrawTexturePS;
static ComPtr<ID3D11PixelShader> HighPassPS;
static ComPtr<ID3D11PixelShader> BlurPS;
static ComPtr<ID3D11Buffer> FullscreenQuadVB;
static ComPtr<ID3D11Buffer> QuadVB;
static ComPtr<ID3D11Buffer> DrawTextureCB;
static ComPtr<ID3D11Buffer> HighPassCB;
static ComPtr<ID3D11Buffer> BlurCB;
static ComPtr<ID3D11SamplerState> Sampler;
static D3D11_VIEWPORT BackBufferVP;
static D3D11_VIEWPORT HighPassVP;
static D3D11_VIEWPORT BlurVP;

struct Vertex
{
    float x, y;
    float u, v;
};

static const uint32_t VertexStride = sizeof(Vertex);
static const uint32_t VertexOffset = 0;


struct DrawTextureConstants
{
    uint32_t Operator;
    float Exposure;
    uint32_t PerformGamma;
    uint32_t Pad;
};

static DrawTextureConstants TheDrawTextureConstants;

struct HighPassConstants
{
    float YThreshold;
    float Pad[3];
};

static HighPassConstants TheHighPassConstants;

struct BlurConstants
{
    uint32_t Direction; // 0 = horiz, 1 = vert
    float Pad[3];
};

static BlurConstants TheBlurConstants;
static bool HighPassBlurEnabled = true;

static void InitResourcesAndShaders();
static void BindHighPass();
static void BindFirstBlur();
static void BindSecondBlur();
static void BindDrawTexture();

void GraphicsStartup(HWND window)
{
    RECT rc{};
    GetClientRect(window, &rc);

    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = rc.right - rc.left;
    scd.BufferDesc.Height = rc.bottom - rc.top;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = window;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    UINT flags = 0;

#ifdef _DEBUG
    //flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        flags, &featureLevel, 1, D3D11_SDK_VERSION, &scd, &SwapChain, &Device, nullptr, &Context);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"D3D11CreateDeviceAndSwapChain failed. 0x%08x", hr);

    hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
    FAIL_IF_FALSE(SUCCEEDED(hr), L"SwapChain GetBuffer failed. 0x%08x", hr);

    hr = Device->CreateRenderTargetView(BackBuffer.Get(), nullptr, &BackBufferRTV);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"RenderTarget creation failed. 0x%08x", hr);

    BackBufferVP.Width = (float)scd.BufferDesc.Width;
    BackBufferVP.Height = (float)scd.BufferDesc.Height;
    BackBufferVP.MaxDepth = 1.f;

    InitResourcesAndShaders();

    TheDrawTextureConstants.Operator = 0;
    TheDrawTextureConstants.Exposure = 16.f;
    TheDrawTextureConstants.PerformGamma = 1;

    TheHighPassConstants.YThreshold = 0.1f;
}

void GraphicsShutdown()
{
    Sampler = nullptr;
    BlurCB = nullptr;
    HighPassCB = nullptr;
    DrawTextureCB = nullptr;
    QuadVB = nullptr;
    FullscreenQuadVB = nullptr;
    BlurPS = nullptr;
    HighPassPS = nullptr;
    DrawTexturePS = nullptr;
    DrawQuadVS = nullptr;
    InputLayout = nullptr;
    FilmLutSRV = nullptr;
    BlurSRV = nullptr;
    BlurRTV = nullptr;
    HighPassRTV = nullptr;
    HighPassSRV = nullptr;
    BackBufferRTV = nullptr;
    BackBuffer = nullptr;
    SwapChain = nullptr;
    Context = nullptr;
    Device = nullptr;
}

const ComPtr<ID3D11Device>& GraphicsGetDevice()
{
    return Device;
}

void GraphicsClear()
{
    static const float clearColor[] = { 0, 0, 0, 1 };
    Context->ClearRenderTargetView(BackBufferRTV.Get(), clearColor);
}

void GraphicsPresent()
{
    HRESULT hr = SwapChain->Present(1, 0);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Present failed. 0x%08x", hr);
}

void GraphicsSetOperator(ToneMappingOperator op)
{
    TheDrawTextureConstants.Operator = (uint32_t)op;
}

void GraphicsSetFilmLut(const ComPtr<ID3D11ShaderResourceView>& filmLut)
{
    FilmLutSRV = filmLut;
    Context->PSSetShaderResources(2, 1, FilmLutSRV.GetAddressOf());
}

void GraphicsEnableGamma(bool enable)
{
    TheDrawTextureConstants.PerformGamma = enable ? 1 : 0;
}

bool GraphicsGammaEnabled()
{
    return TheDrawTextureConstants.PerformGamma ? true : false;
}

void GraphicsSetExposure(float exposure)
{
    TheDrawTextureConstants.Exposure = exposure;
}

void GraphicsSetHighPassThreshold(float threshold)
{
    TheHighPassConstants.YThreshold = threshold;
}

void GraphicsEnableHighPassBlur(bool enable)
{
    HighPassBlurEnabled = enable;
}

bool GraphicsHighPassBlurEnabled()
{
    return HighPassBlurEnabled;
}

void GraphicsDrawQuad(const RECT* dest, const ComPtr<ID3D11ShaderResourceView>& source)
{
    if (HighPassBlurEnabled)
    {
        // Run high pass filter
        BindHighPass();
        Context->UpdateSubresource(HighPassCB.Get(), 0, nullptr, &TheHighPassConstants, sizeof(TheHighPassConstants), 0);
        Context->PSSetShaderResources(0, 1, source.GetAddressOf());
        Context->Draw(6, 0);

        // Run horizontal blur
        BindFirstBlur();
        Context->Draw(6, 0);

        // Run vertical blur
        BindSecondBlur();
        Context->Draw(6, 0);
    }

    // Render texture w/ high pass data alongside it
    BindDrawTexture();
    Context->GenerateMips(HighPassSRV.Get());

    const float invWidth = 1.0f / BackBufferVP.Width;
    const float invHeight = 1.0f / BackBufferVP.Height;

    float x = (float)dest->left * invWidth;
    float y = (float)dest->top * invHeight;
    float x2 = (float)dest->right * invWidth;
    float y2 = (float)dest->bottom * invHeight;

    y = 1.f - y;
    y2 = 1.f - y2;

    x = x * 2 - 1;
    x2 = x2 * 2 - 1;
    y = y * 2 - 1;
    y2 = y2 * 2 - 1;

    Vertex verts[] = 
    {
        { x, y, 0, 0 },
        { x2, y, 1, 0 },
        { x2, y2, 1, 1 },
        { x, y, 0, 0 },
        { x2, y2, 1, 1 },
        { x, y2, 0, 1 },
    };

    Context->UpdateSubresource(QuadVB.Get(), 0, nullptr, verts, sizeof(verts), 0);
    Context->UpdateSubresource(DrawTextureCB.Get(), 0, nullptr, &TheDrawTextureConstants, sizeof(TheDrawTextureConstants), 0);
    Context->PSSetShaderResources(0, 1, source.GetAddressOf());
    Context->Draw(6, 0);
}

void InitResourcesAndShaders()
{
    HRESULT hr = Device->CreateVertexShader(DrawQuad_vs, sizeof(DrawQuad_vs), nullptr, &DrawQuadVS);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create vertex shader. 0x%08x", hr);

    hr = Device->CreatePixelShader(DrawTexture_ps, sizeof(DrawTexture_ps), nullptr, &DrawTexturePS);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create pixel shader. 0x%08x", hr);

    hr = Device->CreatePixelShader(HighPass_ps, sizeof(HighPass_ps), nullptr, &HighPassPS);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create pixel shader. 0x%08x", hr);

    hr = Device->CreatePixelShader(Blur_ps, sizeof(Blur_ps), nullptr, &BlurPS);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create pixel shader. 0x%08x", hr);

    D3D11_INPUT_ELEMENT_DESC elems[2] {};
    elems[0].Format = DXGI_FORMAT_R32G32_FLOAT;
    elems[0].SemanticName = "POSITION";
    elems[1].AlignedByteOffset = sizeof(float) * 2;
    elems[1].Format = DXGI_FORMAT_R32G32_FLOAT;
    elems[1].SemanticName = "TEXCOORD";

    hr = Device->CreateInputLayout(elems, _countof(elems), DrawQuad_vs, sizeof(DrawQuad_vs), &InputLayout);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create input layout. 0x%08x", hr);

    D3D11_BUFFER_DESC bd{};
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = sizeof(Vertex) * 6;
    bd.StructureByteStride = sizeof(Vertex);
    bd.Usage = D3D11_USAGE_DEFAULT;

    Vertex verts[] =
    {
        { -1, 1, 0, 0 },
        { 1, 1, 1, 0 },
        { 1, -1, 1, 1 },
        { -1, 1, 0, 0 },
        { 1, -1, 1, 1 },
        { -1, -1, 0, 1 },
    };

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = verts;
    init.SysMemPitch = sizeof(verts);
    init.SysMemSlicePitch = init.SysMemPitch;

    hr = Device->CreateBuffer(&bd, &init, &FullscreenQuadVB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create vertex buffer. 0x%08x", hr);

    hr = Device->CreateBuffer(&bd, nullptr, &QuadVB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create vertex buffer. 0x%08x", hr);

    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.ByteWidth = sizeof(DrawTextureConstants);
    bd.StructureByteStride = sizeof(DrawTextureConstants);

    hr = Device->CreateBuffer(&bd, nullptr, &DrawTextureCB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create constant buffer. 0x%08x", hr);

    bd.ByteWidth = sizeof(HighPassConstants);
    bd.StructureByteStride = sizeof(HighPassConstants);

    hr = Device->CreateBuffer(&bd, nullptr, &HighPassCB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create constant buffer. 0x%08x", hr);

    bd.ByteWidth = sizeof(BlurConstants);
    bd.StructureByteStride = sizeof(BlurConstants);

    hr = Device->CreateBuffer(&bd, nullptr, &BlurCB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create constant buffer. 0x%08x", hr);

    D3D11_SAMPLER_DESC sd{};
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    hr = Device->CreateSamplerState(&sd, &Sampler);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create texture sampler. 0x%08x", hr);

    D3D11_TEXTURE2D_DESC td{};
    td.ArraySize = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.Width = (uint32_t)(BackBufferVP.Width * 0.5f);
    td.Height = (uint32_t)(BackBufferVP.Height * 0.5f);
    td.MipLevels = 0;
    td.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;

    ComPtr<ID3D11Texture2D> texture;
    hr = Device->CreateTexture2D(&td, nullptr, &texture);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create high pass texture. 0x%08x", hr);

    hr = Device->CreateRenderTargetView(texture.Get(), nullptr, &HighPassRTV);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create high pass RTV. 0x%08x", hr);

    D3D11_SHADER_RESOURCE_VIEW_DESC srvd{};
    srvd.Format = td.Format;
    srvd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    srvd.Texture2D.MipLevels = 8;
    srvd.Texture2D.MostDetailedMip = 0;

    hr = Device->CreateShaderResourceView(texture.Get(), &srvd, &HighPassSRV);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create high pass SRV. 0x%08x", hr);

    HighPassVP.Width = (float)td.Width;
    HighPassVP.Height = (float)td.Height;
    HighPassVP.MaxDepth = 1.f;

    hr = Device->CreateTexture2D(&td, nullptr, texture.ReleaseAndGetAddressOf());
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create blur texture. 0x%08x", hr);

    hr = Device->CreateRenderTargetView(texture.Get(), nullptr, &BlurRTV);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create blur RTV. 0x%08x", hr);

    hr = Device->CreateShaderResourceView(texture.Get(), nullptr, &BlurSRV);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create blur SRV. 0x%08x", hr);

    BlurVP.Width = (float)td.Width;
    BlurVP.Height = (float)td.Height;
    BlurVP.MaxDepth = 1.f;
}

void BindHighPass()
{
    ID3D11ShaderResourceView* srvs[] = { nullptr, nullptr };
    Context->PSSetShaderResources(0, _countof(srvs), srvs);

    Context->OMSetRenderTargets(1, HighPassRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &HighPassVP);
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetVertexBuffers(0, 1, FullscreenQuadVB.GetAddressOf(), &VertexStride, &VertexOffset);
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(HighPassPS.Get(), nullptr, 0);
    Context->PSSetConstantBuffers(0, 1, HighPassCB.GetAddressOf());
    Context->PSSetSamplers(0, 1, Sampler.GetAddressOf());
}

void BindFirstBlur()
{
    ID3D11ShaderResourceView* srvs[] = { nullptr, nullptr };
    Context->PSSetShaderResources(0, _countof(srvs), srvs);

    Context->OMSetRenderTargets(1, BlurRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &BlurVP);
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetVertexBuffers(0, 1, FullscreenQuadVB.GetAddressOf(), &VertexStride, &VertexOffset);
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(BlurPS.Get(), nullptr, 0);
    Context->PSSetConstantBuffers(0, 1, BlurCB.GetAddressOf());
    Context->PSSetShaderResources(0, 1, HighPassSRV.GetAddressOf());
    Context->PSSetSamplers(0, 1, Sampler.GetAddressOf());

    TheBlurConstants.Direction = 0;
    Context->UpdateSubresource(BlurCB.Get(), 0, nullptr, &TheBlurConstants, sizeof(TheBlurConstants), 0);
}

void BindSecondBlur()
{
    ID3D11ShaderResourceView* srvs[] = { nullptr, nullptr };
    Context->PSSetShaderResources(0, _countof(srvs), srvs);

    Context->OMSetRenderTargets(1, HighPassRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &BlurVP);
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetVertexBuffers(0, 1, FullscreenQuadVB.GetAddressOf(), &VertexStride, &VertexOffset);
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(BlurPS.Get(), nullptr, 0);
    Context->PSSetConstantBuffers(0, 1, BlurCB.GetAddressOf());
    Context->PSSetShaderResources(0, 1, BlurSRV.GetAddressOf());
    Context->PSSetSamplers(0, 1, Sampler.GetAddressOf());

    TheBlurConstants.Direction = 1;
    Context->UpdateSubresource(BlurCB.Get(), 0, nullptr, &TheBlurConstants, sizeof(TheBlurConstants), 0);
}

void BindDrawTexture()
{
    ID3D11ShaderResourceView* srvs[] = { nullptr, nullptr };
    Context->PSSetShaderResources(0, _countof(srvs), srvs);

    Context->OMSetRenderTargets(1, BackBufferRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &BackBufferVP);
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetVertexBuffers(0, 1, QuadVB.GetAddressOf(), &VertexStride, &VertexOffset);
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(DrawTexturePS.Get(), nullptr, 0);
    Context->PSSetConstantBuffers(0, 1, DrawTextureCB.GetAddressOf());
    Context->PSSetSamplers(0, 1, Sampler.GetAddressOf());

    if (HighPassBlurEnabled)
    {
        Context->PSSetShaderResources(1, 1, HighPassSRV.GetAddressOf());
    }
}
