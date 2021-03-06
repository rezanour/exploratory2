#include "Precomp.h"
#include <d3d11.h>      // Device/swapchain for final framebuffer

#include "Device.h"
#include "VertexBuffer.h"
#include "Texture2D.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

static const wchar_t WinClassName[] = L"TurboRastDemoApp";
static const wchar_t WinTitle[] = L"TurboRast Demo App";
static const uint32_t OutputWidth = 1280;
static const uint32_t OutputHeight = 720;
static const uint32_t MaxFramesInFlight = 3;

static HINSTANCE Instance;
static HWND Window;

static uint32_t FrameIndex;
static ComPtr<IDXGIFactory1> Factory;
static ComPtr<IDXGIAdapter> Adapter;
static ComPtr<IDXGISwapChain> SwapChain;
static ComPtr<ID3D11Device> Device;
static ComPtr<ID3D11DeviceContext> Context;
static ComPtr<ID3D11Texture2D> BackBuffer;
static ComPtr<ID3D11Texture2D> CPUBuffer[MaxFramesInFlight];

static std::shared_ptr<TRDevice> TheDevice;
static std::shared_ptr<TRVertexBuffer> VertBuffer;
#pragma warning(push)
#pragma warning(disable: 4592)
static std::shared_ptr<TRTexture2D> RenderTargets[MaxFramesInFlight];
#pragma warning(pop)

struct SimpleConstants
{
    matrix4x4 WorldMatrix;
    matrix4x4 ViewProjectionMatrix;
};

static SimpleConstants ShaderConstants;

static bool WinStartup();
static void WinShutdown();

static bool DXStartup();
static void DXShutdown();

static bool AppStartup();
static void AppShutdown();
static bool DoFrame();

static vs_output __vectorcall SimpleVertexShader(const void* const constants, const vs_input input);
static vec4 __vectorcall SimplePixelShader(const void* const constants, const vs_output input);

static VertexOut __vectorcall SimpleVertexShader2(const void* const constants, const Vertex& input);
static float4 __vectorcall SimplePixelShader2(const void* const constants, const VertexOut& input);

static void SimpleVertexShader3(const void* const constants, const void* const input, void* output, int64_t vertexCount);
static float4 __vectorcall SimplePixelShader3(const void* const constants, const uint8_t* input);

static LRESULT CALLBACK AppWinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    Instance = instance;

#ifdef _DEBUG
    uint32_t flag = _crtDbgFlag;
    flag |= _CRTDBG_LEAK_CHECK_DF;
    _CrtSetDbgFlag(flag);

    _Atexit([]()
    {
        assert(!_CrtDumpMemoryLeaks());
    });
#endif

    if (!WinStartup())
    {
        return -1;
    }

    if (!DXStartup())
    {
        WinShutdown();
        return -2;
    }

    if (!AppStartup())
    {
        DXShutdown();
        WinShutdown();
        return -3;
    }

    ShowWindow(Window, SW_SHOW);
    UpdateWindow(Window);

    MSG msg{};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            if (!DoFrame())
            {
                break;
            }
        }
    }

    AppShutdown();
    DXShutdown();
    WinShutdown();
    return 0;
}

bool WinStartup()
{
    WNDCLASSEX wcx{};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = Instance;
    wcx.lpfnWndProc = AppWinProc;
    wcx.lpszClassName = WinClassName;

    if (RegisterClassEx(&wcx) == INVALID_ATOM)
    {
        assert(false);
        return false;
    }

    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);

    RECT rc{};
    rc.right = OutputWidth;
    rc.bottom = OutputHeight;
    AdjustWindowRect(&rc, style, FALSE);

    Window = CreateWindow(WinClassName, WinTitle, style, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, Instance, nullptr);

    if (!Window)
    {
        assert(false);
        return false;
    }

    return true;
}

void WinShutdown()
{
    DestroyWindow(Window);
    Window = nullptr;
}

bool DXStartup()
{
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&Factory));
    assert(SUCCEEDED(hr));
    if (SUCCEEDED(hr))
    {
        hr = Factory->EnumAdapters(0, &Adapter);
        assert(SUCCEEDED(hr));
    }

    if (SUCCEEDED(hr))
    {
        D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
        UINT flags = 0;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

        hr = D3D11CreateDevice(Adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
            flags, &featureLevel, 1, D3D11_SDK_VERSION, &Device, nullptr, &Context);

        if (hr == DXGI_ERROR_SDK_COMPONENT_MISSING && (flags & D3D11_CREATE_DEVICE_DEBUG) != 0)
        {
            // If it failed & we were trying to create debug device, try again without debug
            flags &= ~D3D11_CREATE_DEVICE_DEBUG;
            hr = D3D11CreateDevice(Adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                flags, &featureLevel, 1, D3D11_SDK_VERSION, &Device, nullptr, &Context);
        }

        assert(SUCCEEDED(hr));
    }

    if (SUCCEEDED(hr))
    {
        DXGI_SWAP_CHAIN_DESC scd{};
        scd.BufferCount = MaxFramesInFlight;
        scd.BufferDesc.Width = OutputWidth;
        scd.BufferDesc.Height = OutputHeight;
        scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        scd.OutputWindow = Window;
        scd.SampleDesc.Count = 1;
        scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        scd.Windowed = TRUE;

        hr = Factory->CreateSwapChain(Device.Get(), &scd, &SwapChain);
        assert(SUCCEEDED(hr));
    }

    if (SUCCEEDED(hr))
    {
        hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
        assert(SUCCEEDED(hr));
    }

    if (SUCCEEDED(hr))
    {
        D3D11_TEXTURE2D_DESC desc{};
        BackBuffer->GetDesc(&desc);

#if 0
        // TODO: Is it faster to write into our own buffer and then blit it in? Needs testing
        desc.MiscFlags = 0;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.Usage = D3D11_USAGE_DYNAMIC;
#endif

        desc.MiscFlags = 0;
        desc.BindFlags = 0;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.Usage = D3D11_USAGE_STAGING;

        for (int i = 0; i < _countof(CPUBuffer) && SUCCEEDED(hr); ++i)
        {
            hr = Device->CreateTexture2D(&desc, nullptr, &CPUBuffer[i]);
            assert(SUCCEEDED(hr));
        }
    }

    return SUCCEEDED(hr);
}

void DXShutdown()
{
    for (int i = 0; i < _countof(CPUBuffer); ++i)
    {
        CPUBuffer[i] = nullptr;
    }

    BackBuffer = nullptr;
    SwapChain = nullptr;
    Context = nullptr;
    Device = nullptr;
    Adapter = nullptr;
    Factory = nullptr;
}

LRESULT CALLBACK AppWinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CLOSE:
        PostQuitMessage(0);
        break;

    case WM_KEYUP:
        if (wParam == VK_ESCAPE)
        {
            PostQuitMessage(0);
        }
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}

#include <DirectXMath.h>
using namespace DirectX;

bool AppStartup()
{
    TheDevice = std::make_shared<TRDevice>();
    if (!TheDevice->Initialize())
    {
        assert(false);
        return false;
    }

    std::vector<Vertex> vertices;
    // Fill in vertices for triangle
#define RENDER_MANY
#ifdef RENDER_MANY
    for (float z = 5.f; z >= -5.f; z -= 1.f)
    {
        for (float y = 5.f; y >= -5.f; y -= 0.25f)
        {
            for (float x = -5.f; x < 5.f; x += 0.25f)
            {
                vertices.push_back(Vertex(float3(x - 0.125f, y - 0.125f, z), float3(0.f, 0.f, 1.f)));
                vertices.push_back(Vertex(float3(x + 0.f, y + 0.125f, z), float3(0.f, 1.f, 0.f)));
                vertices.push_back(Vertex(float3(x + 0.125f, y - 0.125f, z), float3(1.f, 0.f, 0.f)));
            }
        }
    }
#else
    for (int i = 0; i < 1; ++i)
    {
        vertices.push_back(Vertex(float3(-0.5f, -0.5f, 0.f), float3(0.f, 0.f, 1.f)));
        vertices.push_back(Vertex(float3(0.f, 0.5f, 0.f), float3(0.f, 1.f, 0.f)));
        vertices.push_back(Vertex(float3(0.5f, -0.5f, 0.f), float3(1.f, 0.f, 0.f)));
    }
#endif

    VertBuffer = std::make_shared<TRVertexBuffer>();
    VertBuffer->Update(vertices.data(), vertices.size());

    std::vector<VertexAttributeDesc> layout(2);
    layout[0].ByteOffset = 0;
    layout[0].Type = VertexAttributeType::Float3;
    layout[0].Semantic = "POSITION";
    layout[1].ByteOffset = sizeof(float3);
    layout[1].Type = VertexAttributeType::Float3;
    layout[1].Semantic = "COLOR";

    TheDevice->SetInputLayout(layout, sizeof(Vertex));

    layout.resize(2);
    layout[0].ByteOffset = 0;
    layout[0].Type = VertexAttributeType::Float4;
    layout[0].Semantic = "SV_POSITION";
    layout[1].ByteOffset = sizeof(float4);
    layout[1].Type = VertexAttributeType::Float3;
    layout[1].Semantic = "COLOR";

    TheDevice->SetVSOutputLayout(layout, sizeof(VertexOut));

    TheDevice->IASetVertexBuffer(VertBuffer);
    TheDevice->VSSetShader(SimpleVertexShader);
    TheDevice->PSSetShader(SimplePixelShader);
    TheDevice->VSSetShader2(SimpleVertexShader2);
    TheDevice->PSSetShader2(SimplePixelShader2);
    TheDevice->VSSetShader3(SimpleVertexShader3);
    TheDevice->PSSetShader3(SimplePixelShader3);
    TheDevice->VSSetConstantBuffer(&ShaderConstants);

    XMStoreFloat4x4((XMFLOAT4X4*)&ShaderConstants.WorldMatrix, XMMatrixIdentity());
    XMStoreFloat4x4((XMFLOAT4X4*)&ShaderConstants.ViewProjectionMatrix, XMMatrixMultiply(XMMatrixLookAtLH(XMVectorSet(0.f, 0, -8.f, 1), XMVectorSet(0, 0, 0, 1), XMVectorSet(0, 1, 0, 0)), XMMatrixPerspectiveFovLH(XMConvertToRadians(90.f), OutputWidth / (float)OutputHeight, 0.1f, 100.f)));

    return true;
}

void AppShutdown()
{
    VertBuffer = nullptr;
    for (int i = 0; i < _countof(RenderTargets); ++i)
    {
        RenderTargets[i] = nullptr;
    }
    TheDevice = nullptr;
}

bool DoFrame()
{
#define ENABLE_ANIMATION
#ifdef ENABLE_ANIMATION
    XMFLOAT4X4 transform;

    static int totalFrameIndex = 0;
    static float angle = -0.5f;
    static float dir = -1.f;

#ifdef RENDER_MANY
    if (totalFrameIndex++ % 1000 == 0) dir *= -1;
#else
    if (totalFrameIndex++ % 1500 == 0) dir *= -1;
#endif
    static bool animating = true;
    static bool wasDown = false;
    if (GetAsyncKeyState(VK_SPACE) & 0x8000)
    {
        wasDown = true;
    }
    else
    {
        if (wasDown) animating = !animating;
        wasDown = false;
    }
    if (animating)
    {
        angle += dir * 0.00125f;
    }
    XMStoreFloat4x4(&transform, XMMatrixRotationY(angle));
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &transform, sizeof(transform));
#endif

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = Context->Map(CPUBuffer[FrameIndex].Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (FAILED(hr))
    {
        assert(false);
        return false;
    }

    if (!RenderTargets[FrameIndex] || (RenderTargets[FrameIndex]->GetData() != mapped.pData))
    {
        RenderTargets[FrameIndex] = std::make_shared<TRTexture2D>(mapped.pData, (int)OutputWidth, (int)OutputHeight, (int)mapped.RowPitch / (int)sizeof(uint32_t));
    }

    TheDevice->ClearRenderTarget(RenderTargets[FrameIndex]);
   
    TheDevice->OMSetRenderTarget(RenderTargets[FrameIndex]);
    TheDevice->Draw(VertBuffer->GetNumVertices(), 0);

    TheDevice->FlushAndWait();
    Context->Unmap(CPUBuffer[FrameIndex].Get(), 0);

    // Copy the CPU buffer to the back buffer
    Context->CopyResource(BackBuffer.Get(), CPUBuffer[FrameIndex].Get());

    // Present
    hr = SwapChain->Present(1, 0);
    if (FAILED(hr))
    {
        assert(false);
        return false;
    }

    // Advance frame index
    ++FrameIndex;
    assert(FrameIndex <= MaxFramesInFlight);
    if (FrameIndex == MaxFramesInFlight)
    {
        FrameIndex = 0;
    }

    return true;
}

vs_output __vectorcall SimpleVertexShader(const void* const constants, const vs_input input)
{
    const SimpleConstants* const vsConstants = (const SimpleConstants* const)constants;

    vs_output output;

    vec4 position{ input.Position.x, input.Position.y, input.Position.z, _mm_set1_ps(1.f) };

    const matrix4x4* matrices[] = { &vsConstants->WorldMatrix, &vsConstants->ViewProjectionMatrix };

    vec4 pos2 = position;
    for (int i = 0; i < _countof(matrices); ++i)
    {
        // expanded multiply of all 4 positions by matrix
        // dot(float4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]), v),
        // dot(float4(m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]), v),
        // dot(float4(m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]), v),
        // dot(float4(m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]), v));
        // Resulting 4 dots are the components of the result vector
        __m128 mx = _mm_set1_ps(matrices[i]->m[0][0]);
        __m128 my = _mm_set1_ps(matrices[i]->m[1][0]);
        __m128 mz = _mm_set1_ps(matrices[i]->m[2][0]);
        __m128 mw = _mm_set1_ps(matrices[i]->m[3][0]);
        pos2.x = _mm_add_ps(_mm_mul_ps(mx, position.x), _mm_add_ps(_mm_mul_ps(my, position.y), _mm_add_ps(_mm_mul_ps(mz, position.z), _mm_mul_ps(mw, position.w))));
        mx = _mm_set1_ps(matrices[i]->m[0][1]);
        my = _mm_set1_ps(matrices[i]->m[1][1]);
        mz = _mm_set1_ps(matrices[i]->m[2][1]);
        mw = _mm_set1_ps(matrices[i]->m[3][1]);
        pos2.y = _mm_add_ps(_mm_mul_ps(mx, position.x), _mm_add_ps(_mm_mul_ps(my, position.y), _mm_add_ps(_mm_mul_ps(mz, position.z), _mm_mul_ps(mw, position.w))));
        mx = _mm_set1_ps(matrices[i]->m[0][2]);
        my = _mm_set1_ps(matrices[i]->m[1][2]);
        mz = _mm_set1_ps(matrices[i]->m[2][2]);
        mw = _mm_set1_ps(matrices[i]->m[3][2]);
        pos2.z = _mm_add_ps(_mm_mul_ps(mx, position.x), _mm_add_ps(_mm_mul_ps(my, position.y), _mm_add_ps(_mm_mul_ps(mz, position.z), _mm_mul_ps(mw, position.w))));
        mx = _mm_set1_ps(matrices[i]->m[0][3]);
        my = _mm_set1_ps(matrices[i]->m[1][3]);
        mz = _mm_set1_ps(matrices[i]->m[2][3]);
        mw = _mm_set1_ps(matrices[i]->m[3][3]);
        pos2.w = _mm_add_ps(_mm_mul_ps(mx, position.x), _mm_add_ps(_mm_mul_ps(my, position.y), _mm_add_ps(_mm_mul_ps(mz, position.z), _mm_mul_ps(mw, position.w))));
        // assign over to x,y,z,w so we can do next iteration back into vx,vy,vz,vw
        position = pos2;
    }

    output.Position = position;
    output.Color = input.Color;

    return output;
}

vec4 __vectorcall SimplePixelShader(const void* const constants, const vs_output input)
{
    UNREFERENCED_PARAMETER(constants);
    return vec4{ input.Color.x, input.Color.y, input.Color.z, _mm_set1_ps(1.f) };
}


VertexOut __vectorcall SimpleVertexShader2(const void* const constants, const Vertex& input)
{
    const SimpleConstants* const vsConstants = (const SimpleConstants* const)constants;

    VertexOut output;

    float4 pos = mul(vsConstants->WorldMatrix, float4(input.Position, 1.f));
    output.Position = mul(vsConstants->ViewProjectionMatrix, pos);
    output.Color = input.Color;

    return output;
}

float4 __vectorcall SimplePixelShader2(const void* const constants, const VertexOut& input)
{
    UNREFERENCED_PARAMETER(constants);
    return float4(input.Color, 1.f);
}

void SimpleVertexShader3(
    const void* const constants,
    const void* const input,
    void* output,
    int64_t vertexCount)
{
    const SimpleConstants* const vsConstants = (const SimpleConstants* const)constants;

    const Vertex* const vertices = (const Vertex* const)input;
    VertexOut* outputVertices = (VertexOut*)output;

    for (int64_t i = 0; i < vertexCount; ++i)
    {
        float4 pos = mul(vsConstants->WorldMatrix, float4(vertices[i].Position, 1.f));
        outputVertices[i].Position = mul(vsConstants->ViewProjectionMatrix, pos);
        outputVertices[i].Color = vertices[i].Color;
    }
}

float4 __vectorcall SimplePixelShader3(const void* const constants, const uint8_t* input)
{
    UNREFERENCED_PARAMETER(constants);
    const VertexOut* v = (const VertexOut*)input;
    return float4(v->Color, 1.f);
}

