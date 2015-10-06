#include "precomp.h"
#include "imagefile.h"
#include "edge_detect.h"
#include "util.h"

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

    HDC hdc = GetDC(hwnd);
    HDC hMemDC = CreateCompatibleDC(hdc);
    assert(hMemDC);
    ReleaseDC(hwnd, hdc);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = ClientWidth;
    bmi.bmiHeader.biHeight = -ClientHeight; // DIB is bottom up
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biSizeImage = ClientWidth * ClientHeight * sizeof(uint32_t);

    uint32_t* frame_buffer = nullptr;
    HBITMAP dib = CreateDIBSection(hMemDC, &bmi, DIB_RGB_COLORS, (void**)&frame_buffer, nullptr, 0);
    assert(dib && frame_buffer);

    SelectObject(hMemDC, dib);

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    int width = 0, height = 0;
    std::unique_ptr<uint32_t[]> image = load_image(L"simple_shapes.png", true, &width, &height);
    if (!image)
    {
        assert(false);
        return -4;
    }

    std::unique_ptr<uint32_t[]> edges = detect_edges(image, true, width, height);
    if (!edges)
    {
        assert(false);
        return -5;
    }

    // Copy the result into the mem DC
    for (int y = 0; y < std::min(height, ClientHeight); ++y)
    {
        for (int x = 0; x < std::min(width, ClientWidth); ++x)
        {
            frame_buffer[y * ClientWidth + x] = edges[y * width + x];
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
            hdc = GetDC(hwnd);
            StretchBlt(hdc, 0, 0, ClientWidth, ClientHeight, hMemDC, 0, 0, width, height, SRCCOPY);
            ReleaseDC(hwnd, hdc);
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
