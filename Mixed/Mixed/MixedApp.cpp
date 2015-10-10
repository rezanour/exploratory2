#include "Precomp.h"
#include "MixedApp.h"
#include "Renderer.h"
#include "DebugUtil.h"

const wchar_t MixedApp::ClassName[] = L"MixedApp";

//=============================================================================
// Main entry point
//=============================================================================

int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    try
    {
        std::unique_ptr<MixedApp> app(new MixedApp(instance));
        return app->Run();
    }
    catch (const std::exception&)
    {
        // Logging happens at time exception is generated
    }

    return -1;
}

//=============================================================================
// MixedApp
//=============================================================================

MixedApp::MixedApp(HINSTANCE instance)
    : Window(nullptr)
{
#if HACK_GENERATE_GAUSSIAN_KERNEL // Move this somewhere else!
    float o = 0.9f;     // scale
    float matrix[7]{};  // -3 to 3
    float sum = 0.f;
    for (int i = -3; i <= 3; ++i)
    {
        matrix[i + 3] = exp(-(i * i) / (2 * o * o)) / sqrtf(2 * XM_PI * (o * o));
        sum += matrix[i + 3];
    }

    // Normalize
    for (int i = -3; i <= 3; ++i)
    {
        matrix[i + 3] = matrix[i + 3] / sum;
    }
#endif

    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&WicFactory));
    CHECKHR(hr, L"Failed to create WIC factory. hr = 0x%08x.", hr);

    InitializeWindow(instance);
    Renderer.reset(new ::Renderer(Window));

    uint32_t width = 0, height = 0;
    std::unique_ptr<uint8_t[]> pixels = LoadImageFile(L"car2.jpg", &width, &height);

    Color = Renderer->CreateColorImage(width, height, (const uint32_t*)pixels.get());
    Lum = Renderer->CreateLuminanceImage(width, height, nullptr);
    Norm = Renderer->CreateNormalsImage(width, height, nullptr);
    Blurred = Renderer->CreateColorImage(width, height, nullptr);
    Edges1 = Renderer->CreateLuminanceImage(width, height, nullptr);
    Edges2 = Renderer->CreateLuminanceImage(width, height, nullptr);

    Renderer->ColorToLum(Color, Lum);
    Renderer->LumToNormals(Lum, Norm);
    Renderer->Gaussian(Color, Blurred);
    Renderer->EdgeDetect(Color, Edges1);
    Renderer->EdgeDetect(Blurred, Edges2);
}

MixedApp::~MixedApp()
{
    if (Window)
    {
        DestroyWindow(Window);
        Window = nullptr;
    }

    // Release before shutting down COM
    WicFactory = nullptr;

    CoUninitialize();
}

int MixedApp::Run()
{
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
            Update();
        }
    }
    return 0;
}

void MixedApp::Update()
{
    Renderer->Clear();

    Renderer->DrawImage(Color, 0, 0, 640, 360);
    Renderer->DrawImage(Lum, 640, 0, 640, 360);
    Renderer->DrawImage(Edges1, 0, 360, 640, 360);
    Renderer->DrawImage(Edges2, 640, 360, 640, 360);

    Renderer->Present();
}

void MixedApp::InitializeWindow(HINSTANCE instance)
{
    WNDCLASSEX wcx{};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = instance;
    wcx.lpfnWndProc = s_WindowProc;
    wcx.lpszClassName = ClassName;

    if (RegisterClassEx(&wcx) == INVALID_ATOM)
    {
        FAIL(L"RegisterClassEx failed.");
    }

    DWORD style = WS_OVERLAPPEDWINDOW;
    RECT rc{ 0, 0, 1280, 720 };
    AdjustWindowRect(&rc, style, FALSE);

    Window = CreateWindow(ClassName, ClassName, style, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, instance, nullptr);

    FAIL_IF_NULL(Window, L"Failed to create window. Error %d", GetLastError());

    ShowWindow(Window, SW_SHOW);
    UpdateWindow(Window);
}

LRESULT CALLBACK MixedApp::s_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
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

std::unique_ptr<uint8_t[]> MixedApp::LoadImageFile(const wchar_t* filename, uint32_t* width, uint32_t* height)
{
    ComPtr<IWICBitmapDecoder> decoder;
    HRESULT hr = WicFactory->CreateDecoderFromFilename(filename, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder);
    CHECKHR(hr, L"Failed to create image decoder for file. %s, hr = 0x%08x.", filename, hr);

    ComPtr<IWICBitmapFrameDecode> frame;
    hr = decoder->GetFrame(0, &frame);
    CHECKHR(hr, L"Failed to decode image frame. hr = 0x%08x.", hr);

    ComPtr<IWICFormatConverter> converter;
    hr = WicFactory->CreateFormatConverter(&converter);
    CHECKHR(hr, L"Failed to create image format converter. hr = 0x%08x.", hr);

    hr = converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom);
    CHECKHR(hr, L"Failed to initialize image format converter. hr = 0x%08x.", hr);

    frame->GetSize(width, height);
    std::unique_ptr<uint8_t[]> pixels(new uint8_t[(*width) * (*height) * sizeof(uint32_t)]);
    hr = converter->CopyPixels(nullptr, sizeof(uint32_t) * (*width), sizeof(uint32_t) * (*width) * (*height), pixels.get());
    CHECKHR(hr, L"Failed to decode image pixels. hr = 0x%08x.", hr);

    return pixels;
}
