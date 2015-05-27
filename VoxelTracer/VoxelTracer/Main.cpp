#include "Precomp.h"
#include "Debug.h"
#include "Renderer.h"

// Constants
static const wchar_t ClassName[] = L"VoxelTracer Test Application";
static const uint32_t ScreenWidth = 640;
static const uint32_t ScreenHeight = 480;
static const float Fov = XMConvertToRadians(70.f);
static const float NearClip = 0.1f;
static const float FarClip = 100.f;
static const float CameraMoveSpeedPerSecond = 1.f;
static const float CameraTurnSpeedPerSecond = 0.1f;
static const bool VSyncEnabled = true;

// Application variables
static HINSTANCE Instance;
static HWND Window;

// Local methods
static bool Initialize();
static void Shutdown();
static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Entry point
_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE instance, HINSTANCE, LPSTR, int)
{
    Instance = instance;
    if (!Initialize())
    {
        assert(false);
        return -1;
    }

    // Initialize graphics
    std::unique_ptr<Renderer> renderer(Renderer::Create(Window));
    if (!renderer)
    {
        assert(false);
        return -4;
    }

    ShowWindow(Window, SW_SHOW);
    UpdateWindow(Window);

    // Timing info
    LARGE_INTEGER lastTime {};
    LARGE_INTEGER currTime {};
    LARGE_INTEGER frequency {};
    QueryPerformanceFrequency(&frequency);
    float invFreq = 1.0f / frequency.QuadPart;

    // TODO: Replace with something better as needed

    // Camera info
    XMVECTOR position = XMVectorSet(0.f, 1.f, -2.f, 1.f);
    XMMATRIX cameraWorld = XMMatrixIdentity();
    cameraWorld.r[3] = position;

    renderer->SetFov(Fov);

    XMMATRIX projection = XMMatrixPerspectiveFovLH(
        Fov,
        ScreenWidth / (float)ScreenHeight,  // Aspect ratio of window client (rendering) area
        NearClip,
        FarClip);

    wchar_t caption[200] = {};

    // Main loop
    MSG msg {};
    while (msg.message != WM_QUIT)
    {
        if (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        else
        {
            // Idle, measure time and produce a frame
            QueryPerformanceCounter(&currTime);
            if (lastTime.QuadPart == 0)
            {
                // First frame, skip so we have a good lastTime to compute from
                lastTime.QuadPart = currTime.QuadPart;
                continue;
            }

            // Compute time step from last frame until now
            float timeStep = (currTime.QuadPart - lastTime.QuadPart) * invFreq;

            // Compute fps
            float frameRate = 1.0f / (float)timeStep;
            lastTime = currTime;

            // Handle input
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000)
            {
                // Exit
                break;
            }

            // TODO: Replace with something better as needed

            //renderer->Render(XMMatrixLookToLH(position, XMVectorSet(0.f, 0.f, 1.f, 0.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)), projection, VSyncEnabled);
            renderer->Render(cameraWorld, VSyncEnabled);

            swprintf_s(caption, L"%s (%dx%d) - FPS: %3.2f", ClassName, ScreenWidth, ScreenHeight, frameRate);
            SetWindowText(Window, caption);
        }
    }

    //renderer.reset();
    Shutdown();
    return 0;
}

bool Initialize()
{
    WNDCLASSEX wcx {};
    wcx.cbSize = sizeof(wcx);
    wcx.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wcx.hInstance = Instance;
    wcx.lpfnWndProc = WndProc;
    wcx.lpszClassName = ClassName;

    if (!RegisterClassEx(&wcx))
    {
        LogError(L"Failed to initialize window class.");
        return false;
    }

    DWORD style { WS_OVERLAPPEDWINDOW & ~(WS_THICKFRAME | WS_MAXIMIZEBOX) };

    RECT rc {};
    rc.right = ScreenWidth;
    rc.bottom = ScreenHeight;
    AdjustWindowRect(&rc, style, FALSE);

    Window = CreateWindow(ClassName, ClassName, style,
        CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
        nullptr, nullptr, Instance, nullptr);

    if (!Window)
    {
        LogError(L"Failed to create window.");
        return false;
    }

    return true;
}

void Shutdown()
{
    DestroyWindow(Window);
    Window = nullptr;
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_CLOSE:
        PostQuitMessage(0);
        break;
    }

    return DefWindowProc(hwnd, msg, wParam, lParam);
}
