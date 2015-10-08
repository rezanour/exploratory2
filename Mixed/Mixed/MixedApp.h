#pragma once

class Renderer;
class Image;

// Basic application level things (entry point, window, message pump, etc...)
class MixedApp
{
public:
    MixedApp(HINSTANCE instance);
    virtual ~MixedApp();

    // Call blocks until application is exited
    int Run();

private:
    // Main logic
    void Update();

    // Windows/Application stuff
    void InitializeWindow(HINSTANCE instance);
    static LRESULT CALLBACK s_WindowProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

    // Image loading
    std::unique_ptr<uint8_t[]> LoadImageFile(const wchar_t* filename, uint32_t* width, uint32_t* height);

private:
    static const wchar_t ClassName[];

    HWND Window;
    ComPtr<IWICImagingFactory> WicFactory;
    std::unique_ptr<Renderer> Renderer;
    std::shared_ptr<Image> Color;
    std::shared_ptr<Image> Lum;
    std::shared_ptr<Image> Norm;
};
