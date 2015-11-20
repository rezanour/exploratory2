#include "Precomp.h"
#include "Graphics.h"
#include "Debug.h"

// Shaders
#include "DrawQuad_vs.h"
#include "DrawTexture_ps.h"

static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> Context;
static ComPtr<IDXGISwapChain> SwapChain;
static ComPtr<ID3D11Texture2D> BackBuffer;
static ComPtr<ID3D11RenderTargetView> BackBufferRTV;
static ComPtr<ID3D11InputLayout> InputLayout;
static ComPtr<ID3D11VertexShader> DrawQuadVS;
static ComPtr<ID3D11PixelShader> DrawTexturePS;
static ComPtr<ID3D11Buffer> QuadVB;
static ComPtr<ID3D11Buffer> ConstantBuffer;
static ComPtr<ID3D11SamplerState> Sampler;
static D3D11_VIEWPORT Viewport;

struct Vertex
{
    float x, y;
    float u, v;
};

struct Constants
{
    uint32_t Operator;
    float Exposure;
    uint32_t PerformGamma;
    uint32_t Pad;
};

static Constants TheConstants;

static void InitResourcesAndShaders();
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

    Viewport.Width = (float)scd.BufferDesc.Width;
    Viewport.Height = (float)scd.BufferDesc.Height;
    Viewport.MaxDepth = 1.f;

    InitResourcesAndShaders();
    BindDrawTexture();

    TheConstants.Operator = 0;
    TheConstants.Exposure = 16.f;
    TheConstants.PerformGamma = 1;
}

void GraphicsShutdown()
{
    Sampler = nullptr;
    ConstantBuffer = nullptr;
    QuadVB = nullptr;
    DrawTexturePS = nullptr;
    DrawQuadVS = nullptr;
    InputLayout = nullptr;
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
    TheConstants.Operator = (uint32_t)op;
}

void GraphicsEnableGamma(bool enable)
{
    TheConstants.PerformGamma = enable ? 1 : 0;
}

bool GraphicsGammaEnabled()
{
    return TheConstants.PerformGamma ? true : false;
}

void GraphicsSetExposure(float exposure)
{
    TheConstants.Exposure = exposure;
}

void GraphicsDrawQuad(const RECT* dest, const ComPtr<ID3D11ShaderResourceView>& source)
{
    float invWidth = 1.0f / Viewport.Width;
    float invHeight = 1.0f / Viewport.Height;

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

    Context->UpdateSubresource(ConstantBuffer.Get(), 0, nullptr, &TheConstants, sizeof(TheConstants), 0);

    Context->PSSetShaderResources(0, 1, source.GetAddressOf());
    Context->Draw(6, 0);
}

void InitResourcesAndShaders()
{
    HRESULT hr = Device->CreateVertexShader(DrawQuad_vs, sizeof(DrawQuad_vs), nullptr, &DrawQuadVS);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create vertex shader. 0x%08x", hr);

    hr = Device->CreatePixelShader(DrawTexture_ps, sizeof(DrawTexture_ps), nullptr, &DrawTexturePS);
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

    hr = Device->CreateBuffer(&bd, nullptr, &QuadVB);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create vertex buffer. 0x%08x", hr);

    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.ByteWidth = sizeof(Constants);
    bd.StructureByteStride = sizeof(Constants);

    hr = Device->CreateBuffer(&bd, nullptr, &ConstantBuffer);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create constant buffer. 0x%08x", hr);

    D3D11_SAMPLER_DESC sd{};
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    hr = Device->CreateSamplerState(&sd, &Sampler);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create texture sampler. 0x%08x", hr);
}

void BindDrawTexture()
{
    const uint32_t stride = sizeof(Vertex), offset = 0;

    Context->OMSetRenderTargets(1, BackBufferRTV.GetAddressOf(), nullptr);
    Context->RSSetViewports(1, &Viewport);
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->IASetVertexBuffers(0, 1, QuadVB.GetAddressOf(), &stride, &offset);
    Context->VSSetShader(DrawQuadVS.Get(), nullptr, 0);
    Context->PSSetShader(DrawTexturePS.Get(), nullptr, 0);
    Context->PSSetConstantBuffers(0, 1, ConstantBuffer.GetAddressOf());
    Context->PSSetSamplers(0, 1, Sampler.GetAddressOf());
}
