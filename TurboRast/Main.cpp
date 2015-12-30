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
static const uint32_t MaxFramesInFlight = 2;

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
static std::shared_ptr<TRTexture2D> RenderTargets[MaxFramesInFlight];

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

static void __vectorcall SimpleVertexShader(const void* const constants, const SSEVertexBlock& input, SSEVSOutput& output);
static void __vectorcall SimplePixelShader(const void* const constants, const SSEVSOutput& input, SSEPSOutput& output);

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

        // TODO: Is it faster to write into our own buffer and then blit it in? Needs testing
        desc.MiscFlags = 0;
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        desc.Usage = D3D11_USAGE_DYNAMIC;

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
    vertices.push_back(Vertex(float3(-0.5f, -0.5f, 0.f), float3(0.f, 0.f, 1.f)));
    vertices.push_back(Vertex(float3(0.f, 0.5f, 0.f), float3(0.f, 1.f, 0.f)));
    vertices.push_back(Vertex(float3(0.5f, -0.5f, 0.f), float3(1.f, 0.f, 0.f)));
#endif

    VertBuffer = std::make_shared<TRVertexBuffer>();
    VertBuffer->Update(vertices.data(), vertices.size());

    TheDevice->IASetVertexBuffer(VertBuffer);
    TheDevice->VSSetShader(SimpleVertexShader);
    TheDevice->PSSetShader(SimplePixelShader);
    TheDevice->VSSetConstantBuffer(&ShaderConstants);

    XMStoreFloat4x4((XMFLOAT4X4*)&ShaderConstants.WorldMatrix, XMMatrixIdentity());
    XMStoreFloat4x4((XMFLOAT4X4*)&ShaderConstants.ViewProjectionMatrix, XMMatrixMultiply(XMMatrixLookAtLH(XMVectorSet(0, 0, -10, 1), XMVectorSet(0, 0, 0, 1), XMVectorSet(0, 1, 0, 0)), XMMatrixPerspectiveFovLH(XMConvertToRadians(90.f), OutputWidth / (float)OutputHeight, 0.1f, 100.f)));

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
    if (totalFrameIndex++ % 100 == 0) dir *= -1;
#else
    if (totalFrameIndex++ % 150 == 0) dir *= -1;
#endif
    angle += dir * 0.0125f;
    XMStoreFloat4x4(&transform, XMMatrixRotationY(angle));
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &transform, sizeof(transform));
#endif

    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = Context->Map(CPUBuffer[FrameIndex].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
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

void __vectorcall SimpleVertexShader(const void* const constants, const SSEVertexBlock& input, SSEVSOutput& output)
{
    const SimpleConstants* const vsConstants = (const SimpleConstants* const)constants;

    __m128 x = _mm_load_ps(input.Position_x);
    __m128 y = _mm_load_ps(input.Position_y);
    __m128 z = _mm_load_ps(input.Position_z);
    __m128 w = _mm_set1_ps(1.f);

    const matrix4x4* matrices[] = { &vsConstants->WorldMatrix, &vsConstants->ViewProjectionMatrix };

    __m128 vx = x;
    __m128 vy = y;
    __m128 vz = z;
    __m128 vw = w;
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
        vx = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][1]);
        my = _mm_set1_ps(matrices[i]->m[1][1]);
        mz = _mm_set1_ps(matrices[i]->m[2][1]);
        mw = _mm_set1_ps(matrices[i]->m[3][1]);
        vy = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][2]);
        my = _mm_set1_ps(matrices[i]->m[1][2]);
        mz = _mm_set1_ps(matrices[i]->m[2][2]);
        mw = _mm_set1_ps(matrices[i]->m[3][2]);
        vz = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][3]);
        my = _mm_set1_ps(matrices[i]->m[1][3]);
        mz = _mm_set1_ps(matrices[i]->m[2][3]);
        mw = _mm_set1_ps(matrices[i]->m[3][3]);
        vw = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        // assign over to x,y,z,w so we can do next iteration back into vx,vy,vz,vw
        x = vx;
        y = vy;
        z = vz;
        w = vw;
    }

    _mm_store_ps(output.Position_x, vx);
    _mm_store_ps(output.Position_y, vy);
    _mm_store_ps(output.Position_z, vz);
    _mm_store_ps(output.Position_w, vw);

    for (int i = 0; i < 4; ++i)
    {
        output.Color_x[i] = input.Color_x[i];
        output.Color_y[i] = input.Color_y[i];
        output.Color_z[i] = input.Color_z[i];
    }
}

void __vectorcall SimplePixelShader(const void* const constants, const SSEVSOutput& input, SSEPSOutput& output)
{
    UNREFERENCED_PARAMETER(constants);
    UNREFERENCED_PARAMETER(input);
    UNREFERENCED_PARAMETER(output);
}
