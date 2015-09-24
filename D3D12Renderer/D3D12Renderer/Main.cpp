#include "Precomp.h"
#include "Errors.h"

// Constants
static const wchar_t ApplicationName[] = L"D3D12Renderer";
static const uint32_t DefaultWinWidth = 1280;
static const uint32_t DefaultWinHeight = 720;

// Forward declares
static HRESULT InitializeApp(HINSTANCE instance, HWND* window);
static LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Entry point
int WINAPI WinMain(HINSTANCE instance, HINSTANCE prevInstance, LPSTR commandLine, int showCommand)
{
    UNREFERENCED_PARAMETER(prevInstance);
    UNREFERENCED_PARAMETER(commandLine);
    UNREFERENCED_PARAMETER(showCommand);

    HWND appWindow = nullptr;
    if (FAILED(InitializeApp(instance, &appWindow)))
    {
        return -1;
    }

    ComPtr<ID3D12Device> device;
    CHECKHR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)));

    ShowWindow(appWindow, SW_SHOW);
    UpdateWindow(appWindow);

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
        }
    }

    DestroyWindow(appWindow);
    return 0;
}

HRESULT InitializeApp(HINSTANCE instance, HWND* window)
{
    WNDCLASSEX wcx{};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = instance;
    wcx.lpfnWndProc = AppWndProc;
    wcx.lpszClassName = ApplicationName;

    CHECKGLE(RegisterClassEx(&wcx));

    DWORD style = WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    RECT clientRect = { 0, 0, DefaultWinWidth, DefaultWinHeight };
    AdjustWindowRect(&clientRect, style, FALSE);

    *window = CreateWindow(ApplicationName, ApplicationName, style, CW_USEDEFAULT, CW_USEDEFAULT,
        clientRect.right - clientRect.left, clientRect.bottom - clientRect.top, nullptr, nullptr,
        instance, nullptr);

    CHECKGLE(*window != nullptr);

    return true;
}

LRESULT CALLBACK AppWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CLOSE:
        PostQuitMessage(0);
        break;

    case WM_KEYUP:
        // For quick testing, handle ESC key to quit
        if (wParam == VK_ESCAPE)
        {
            PostQuitMessage(0);
        }
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}
