#include "precomp.h"

#include <Windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl.h>

#include "rasterizer.h"

using namespace Microsoft::WRL;

static const wchar_t WinClassName[] = L"SoftRast";
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

static bool WinStartup();
static void WinShutdown();

static bool DXStartup();
static void DXShutdown();

static bool DXDoFrame();

static LRESULT CALLBACK AppWinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    Instance = instance;

    if (!WinStartup())
    {
        return -1;
    }

    if (!DXStartup())
    {
        WinShutdown();
        return -2;
    }

    if (!RastStartup())
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
            if (!DXDoFrame())
            {
                break;
            }
        }
    }

    RastShutdown();
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

    Window = CreateWindow(WinClassName, L"Software Rasterizer", style, CW_USEDEFAULT, CW_USEDEFAULT,
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

        // Attempt to get write-combine by making a dynamic, write-only texture
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

bool DXDoFrame()
{
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = Context->Map(CPUBuffer[FrameIndex].Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr))
    {
        assert(false);
        return false;
    }

    if (!RenderScene(mapped.pData, OutputWidth, OutputHeight, mapped.RowPitch))
    {
        Context->Unmap(CPUBuffer[FrameIndex].Get(), 0);
        assert(false);
        return false;
    }

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
