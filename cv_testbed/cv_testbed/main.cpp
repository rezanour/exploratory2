#include "precomp.h"
#include "imagefile.h"
#include "edge_detect.h"
#include "util.h"

#include <d3d11.h>
#include <wrl.h>
using namespace Microsoft::WRL;

static const wchar_t ClassName[] = L"cv_testbed";
static const int ClientWidth = 1280;
static const int ClientHeight = 720;

static LRESULT CALLBACK AppWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    CoInit coInit(COINIT_MULTITHREADED);
    if (FAILED(coInit.hr))
    {
        assert(false);
        return -1;
    }

    WNDCLASSEX wcx{};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = instance;
    wcx.lpfnWndProc = AppWindowProc;
    wcx.lpszClassName = ClassName;
    if (RegisterClassEx(&wcx) == INVALID_ATOM)
    {
        assert(false);
        return -2;
    }

    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    RECT rc = { 0, 0, ClientWidth, ClientHeight };
    AdjustWindowRect(&rc, style, FALSE);

    HWND hwnd = CreateWindow(ClassName, ClassName, style, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, instance, nullptr);
    if (!hwnd)
    {
        assert(false);
        return -3;
    }

    ComPtr<IDXGIFactory> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        assert(false);
        return -4;
    }

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        &featureLevel, 1, D3D11_SDK_VERSION, &device, nullptr, &context);
    if (FAILED(hr))
    {
        assert(false);
        return -5;
    }

    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = ClientWidth;
    scd.BufferDesc.Height = ClientHeight;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = hwnd;
    scd.SampleDesc.Count = 1;
    scd.Windowed = TRUE;

    ComPtr<IDXGISwapChain> swapChain;
    hr = factory->CreateSwapChain(device.Get(), &scd, &swapChain);
    if (FAILED(hr))
    {
        assert(false);
        return -6;
    }

    ComPtr<ID3D11Texture2D> gpu_back_buffer;
    hr = swapChain->GetBuffer(0, IID_PPV_ARGS(&gpu_back_buffer));
    if (FAILED(hr))
    {
        assert(false);
        return -6;
    }

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    std::unique_ptr<uint32_t[]> cpu_back_buffer(new uint32_t[ClientWidth * ClientHeight]);
    if (!cpu_back_buffer)
    {
        assert(false);
        return -7;
    }

    int width = 0, height = 0;
    std::unique_ptr<uint32_t[]> image = load_image(L"car2.jpg", true, &width, &height);
    if (!image)
    {
        assert(false);
        return -4;
    }

#if 0 // Show luminance

    std::unique_ptr<float[]> lum = convert_to_luminance(image, true, width, height);
    if (!lum)
    {
        assert(false);
        return -5;
    }

    std::unique_ptr<uint32_t[]> edges(new uint32_t[width * height]);
    if (!edges)
    {
        assert(false);
        return -5;
    }

    for (int i = 0; i < width * height; ++i)
    {
        uint32_t byte = (uint32_t)(uint8_t)(lum[i] * 256.f);
        edges[i] = 0xFF000000 | (byte << 16) | (byte << 8) | byte;
    }

#else // Show edge filter

    std::unique_ptr<uint32_t[]> edges = detect_edges(image, true, width, height);
    if (!edges)
    {
        assert(false);
        return -5;
    }

#endif

    for (int y = 0; y < std::min(ClientHeight, height); ++y)
    {
        for (int x = 0; x < std::min(ClientWidth, width); ++x)
        {
            cpu_back_buffer[y * ClientWidth + x] = edges[y * width + x];
        }
    }

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
            // Render image into the client area
            context->UpdateSubresource(gpu_back_buffer.Get(), 0, nullptr, cpu_back_buffer.get(), ClientWidth * sizeof(uint32_t), ClientWidth * ClientHeight * sizeof(uint32_t));
            swapChain->Present(1, 0);
        }
    }

    DestroyWindow(hwnd);
    return 0;
}

LRESULT CALLBACK AppWindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
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
