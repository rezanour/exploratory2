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
    InitializeWindow(instance);
    Renderer.reset(new ::Renderer(Window));

    uint32_t red[10000];
    for (int i = 0; i < _countof(red); ++i) { red[i] = 0xFF0000FF; }

    uint32_t blue[400];
    for (int i = 0; i < _countof(blue); ++i) { blue[i] = 0xFFFF0000; }

    ScratchColor = Renderer->CreateColorImage(100, 100, red);
    Renderer->FillColorImage(blue, 20, 20, ScratchColor, 40, 40);
}

MixedApp::~MixedApp()
{
    if (Window)
    {
        DestroyWindow(Window);
        Window = nullptr;
    }
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

    Renderer->DrawImage(ScratchColor, 128, 128, 64, 64);

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
