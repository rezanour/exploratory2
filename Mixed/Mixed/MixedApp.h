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

private:
    static const wchar_t ClassName[];

    HWND Window;
    std::unique_ptr<Renderer> Renderer;
    std::shared_ptr<Image> ScratchColor;
};
