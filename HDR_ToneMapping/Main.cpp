#include "Precomp.h"
#include "Debug.h"
#include "HDRFile.h"

static const wchar_t AppName[] = L"HDR_ToneMapping";
static const uint32_t ScreenWidth = 1280;
static const uint32_t ScreenHeight = 720;

static HWND Window;
ComPtr<ID3D11Device> Device;
ComPtr<ID3D11DeviceContext> Context;
ComPtr<IDXGISwapChain> SwapChain;
ComPtr<ID3D11Texture2D> BackBuffer;

static void AppStartup(HINSTANCE instance);
static void AppShutdown();
static LRESULT CALLBACK WinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

static void GraphicsStartup(HWND hwnd);
static void GraphicsShutdown();

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    AppStartup(instance);
    GraphicsStartup(Window);

    ShowWindow(Window, SW_SHOW);
    UpdateWindow(Window);

    ComPtr<ID3D11Texture2D> image = HDRLoadImage(Device, L"memorial.hdr");

    HRESULT hr = S_OK;
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
            // Idle
            Context->CopySubresourceRegion(BackBuffer.Get(), 0, 0, 0, 0,
                image.Get(), 0, nullptr);

            hr = SwapChain->Present(1, 0);
            FAIL_IF_FALSE(SUCCEEDED(hr), L"Present failed. 0x%08x", hr);
        }
    }

    GraphicsShutdown();
    AppShutdown();
}

void AppStartup(HINSTANCE instance)
{
    WNDCLASSEX wcx{};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = instance;
    wcx.lpfnWndProc = WinProc;
    wcx.lpszClassName = AppName;

    if (RegisterClassEx(&wcx) == INVALID_ATOM)
    {
        FAIL(L"RegisterClass failed. %d", GetLastError());
    }

    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    RECT rc{};
    rc.right = ScreenWidth;
    rc.bottom = ScreenHeight;
    AdjustWindowRect(&rc, style, FALSE);

    Window = CreateWindow(AppName, AppName, style, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, instance, nullptr);

    FAIL_IF_NULL(Window, L"CreateWindow failed. %d", GetLastError());
}

void AppShutdown()
{
    DestroyWindow(Window);
    Window = nullptr;
}

LRESULT CALLBACK WinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

void GraphicsStartup(HWND hwnd)
{
    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = ScreenWidth;
    scd.BufferDesc.Height = ScreenHeight;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    UINT flags = 0;

#ifdef DEBUG
    flags = D3D11_CREATE_DEVICE_DEBUG;
#endif

    HRESULT hr = D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        flags, &featureLevel, 1, D3D11_SDK_VERSION, &scd, &SwapChain, &Device, nullptr, &Context);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"D3D11CreateDeviceAndSwapChain failed. 0x%08x", hr);

    hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
    FAIL_IF_FALSE(SUCCEEDED(hr), L"SwapChain GetBuffer failed. 0x%08x", hr);
}

void GraphicsShutdown()
{
    BackBuffer = nullptr;
    SwapChain = nullptr;
    Context = nullptr;
    Device = nullptr;
}
