#include "Precomp.h"
#include "Graphics.h"
#include "HDRFile.h"
#include "Debug.h"

using namespace DirectX;

static const wchar_t AppName[] = L"HDR_ToneMapping";
static const uint32_t ScreenWidth = 1280;
static const uint32_t ScreenHeight = 720;

static HWND Window;
static float Exposure = 16.f;
static float HighPassThreshold = 0.1f;

static void AppStartup(HINSTANCE instance);
static void AppShutdown();
static LRESULT CALLBACK WinProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    AppStartup(instance);
    GraphicsStartup(Window);

    ShowWindow(Window, SW_SHOW);
    UpdateWindow(Window);

    TexMetadata metadata;
    ScratchImage scratchImage;
    HRESULT hr = LoadFromDDSFile(L"Habib_House_Med.dds", DDS_FLAGS_NONE, &metadata, scratchImage);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to open HDR file. 0x%08x", hr);

    ComPtr<ID3D11Texture2D> image;// = HDRLoadImage(GraphicsGetDevice(), L"memorial.hdr");
    ComPtr<ID3D11Resource> resource;
    hr = CreateTexture(GraphicsGetDevice().Get(), scratchImage.GetImages(), scratchImage.GetImageCount(), metadata, &resource);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to open HDR file. 0x%08x", hr);

    resource.As(&image);

    ComPtr<ID3D11ShaderResourceView> srv;
    hr = GraphicsGetDevice()->CreateShaderResourceView(image.Get(), nullptr, &srv);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create SRV to image. 0x%08x", hr);

    RECT dest{};
    //D3D11_TEXTURE2D_DESC desc{};
    //image->GetDesc(&desc);
    //dest.right = desc.Width;
    //dest.bottom = desc.Height;
    dest.right = ScreenWidth;
    dest.bottom = ScreenHeight;

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
            GraphicsClear();
            GraphicsDrawQuad(&dest, srv);
            GraphicsPresent();
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
        else if (wParam == '0')
        {
            GraphicsSetOperator(ToneMappingOperator::None);
        }
        else if (wParam == '1')
        {
            GraphicsSetOperator(ToneMappingOperator::Linear);
        }
        else if (wParam == '2')
        {
            GraphicsSetOperator(ToneMappingOperator::ReinhardRGB);
        }
        else if (wParam == '3')
        {
            GraphicsSetOperator(ToneMappingOperator::ReinhardYOnly);
        }
        else if (wParam == 'G')
        {
            GraphicsEnableGamma(!GraphicsGammaEnabled());
        }
        else if (wParam == 'B')
        {
            GraphicsEnableHighPassBlur(!GraphicsHighPassBlurEnabled());
        }
        else if (wParam == VK_UP)
        {
            Exposure += 1.f;
            GraphicsSetExposure(Exposure);
        }
        else if (wParam == VK_DOWN)
        {
            Exposure -= 1.f;
            GraphicsSetExposure(Exposure);
        }
        else if (wParam == VK_LEFT)
        {
            HighPassThreshold -= 0.025f;
            GraphicsSetHighPassThreshold(HighPassThreshold);
        }
        else if (wParam == VK_RIGHT)
        {
            HighPassThreshold += 0.025f;
            GraphicsSetHighPassThreshold(HighPassThreshold);
        }
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}
