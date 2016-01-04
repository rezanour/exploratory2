#include "Precomp.h"
#include <d3d11.h>      // Device/swapchain for final framebuffer

#include "TurboTrace.h"

using Microsoft::WRL::ComPtr;

static const wchar_t WinClassName[] = L"TurboTraceDemoApp";
static const wchar_t WinTitle[] = L"TurboTrace Demo App";
static const uint32_t OutputWidth = 640;
static const uint32_t OutputHeight = 480;
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

static raytracer_config TracerConfig;
static std::vector<sphere_data> Spheres;
static std::vector<triangle_data> Triangles;
static std::vector<box_data> Boxes;
static aabb_node* AabbHeap;
static aabb_node* AabbRoot;

static bool WinStartup();
static void WinShutdown();

static bool DXStartup();
static void DXShutdown();

static bool AppStartup();
static void AppShutdown();
static bool DoFrame();

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
    // Fill in vertices for triangle
#define RENDER_MANY
#ifdef RENDER_MANY
    for (float z = 5.f; z >= -5.f; z -= 1.f)
    {
        for (float y = 5.f; y >= -5.f; y -= 0.25f)
        {
            for (float x = -5.f; x < 5.f; x += 0.25f)
            {
                triangle_data triangle;
                triangle.v1[0] = x - 0.125f;
                triangle.v1[1] = y - 0.125f;
                triangle.v1[2] = z;
                triangle.v2[0] = x;
                triangle.v2[1] = y + 0.125f;
                triangle.v2[2] = z;
                triangle.v3[0] = x + 0.125f;
                triangle.v3[1] = y - 0.125f;
                triangle.v3[2] = z;
                triangle.normal[0] = 0.f;
                triangle.normal[1] = 0.f;
                triangle.normal[2] = -1.f;
                Triangles.push_back(triangle);

                sphere_data sphere;
                sphere.center[0] = x;
                sphere.center[1] = y;
                sphere.center[2] = z;
                sphere.radius_squared = 0.25f * 0.25f;
                Spheres.push_back(sphere);

                box_data box;
                box.min[0] = x - 0.125f;
                box.min[1] = y - 0.125f;
                box.min[2] = z - 0.125f;
                box.max[0] = x + 0.125f;
                box.max[1] = y + 0.125f;
                box.max[2] = z + 0.125f;
                Boxes.push_back(box);
            }
        }
    }
#else
    for (int i = 0; i < 1; ++i)
    {
        triangle_data triangle;
        triangle.v1[0] = -0.5f;
        triangle.v1[1] = -0.5f;
        triangle.v1[2] = 0.f;
        triangle.v2[0] = 0.f;
        triangle.v2[1] = 0.5f;
        triangle.v2[2] = 0.f;
        triangle.v3[0] = 0.5f;
        triangle.v3[1] = -0.5f;
        triangle.v3[2] = 0.f;
        triangle.normal[0] = 0.f;
        triangle.normal[1] = 0.f;
        triangle.normal[2] = -1.f;
        Triangles.push_back(triangle);

        sphere_data sphere;
        sphere.center[0] = 0.f;
        sphere.center[1] = 0.f;
        sphere.center[2] = 0.f;
        sphere.radius_squared = 0.5f * 0.5f;
        Spheres.push_back(sphere);

        box_data box;
        box.min[0] = -0.5f;
        box.min[1] = -0.5f;
        box.min[2] = -0.5f;
        box.max[0] = 0.5f;
        box.max[1] = 0.5f;
        box.max[2] = 0.5f;
        Boxes.push_back(box);
}
#endif

    tt_build_aabb_tree(Triangles.data(), (int)Triangles.size(),
        &AabbHeap, &AabbRoot);

    return true;
}

void AppShutdown()
{
    // HUGE LEAK (need to recurse and delete children).
    // Better: need to actually allocate all nodes out of
    // contiguous block and use either pointers or indices
    // to reference children. Then can be deleted in 1 shot
    delete [] AabbHeap;
}

bool DoFrame()
{
    D3D11_MAPPED_SUBRESOURCE mapped{};
    HRESULT hr = Context->Map(CPUBuffer[FrameIndex].Get(), 0, D3D11_MAP_WRITE, 0, &mapped);
    if (FAILED(hr))
    {
        assert(false);
        return false;
    }

    float cameraPosition[] = { 0.f, 0.f, -8.f };
    tt_setup(&TracerConfig, (uint32_t*)mapped.pData, OutputWidth, OutputHeight, mapped.RowPitch / sizeof(uint32_t),
        XMConvertToRadians(90.f), cameraPosition);

    //tt_trace(&TracerConfig,
    //    Spheres.data(), 0,//(int)Spheres.size(),
    //    Triangles.data(), 0,// (int)Triangles.size());
    //    Boxes.data(), (int)Boxes.size());

    tt_trace(&TracerConfig, AabbRoot);

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
