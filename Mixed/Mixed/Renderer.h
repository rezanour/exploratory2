#pragma once

class Renderer;

enum class ImageType
{
    Unknown = 0,
    Color,
    Depth,
    Luminance,
    Normals,
};

class Image
{
public:
    virtual ~Image() {}

    ImageType GetType() const { return Type; }
    uint32_t GetWidth() const { return Width; }
    uint32_t GetHeight() const { return Height; }

private:
    friend class Renderer;

    Image() : Type(ImageType::Unknown), Width(0), Height(0), Format(DXGI_FORMAT_UNKNOWN)
    {
    }

private:
    ImageType Type;
    uint32_t Width;
    uint32_t Height;
    DXGI_FORMAT Format;
    ComPtr<ID3D11Texture2D> Texture;
    ComPtr<ID3D11ShaderResourceView> SRV;
    ComPtr<ID3D11RenderTargetView> RTV;
    ComPtr<ID3D11UnorderedAccessView> UAV;
};

class Renderer
{
public:
    Renderer(HWND targetWindow);
    virtual ~Renderer();

    // Create images
    std::shared_ptr<Image> CreateColorImage(uint32_t width, uint32_t height, const uint32_t* optionalSourceData);
    std::shared_ptr<Image> CreateDepthImage(uint32_t width, uint32_t height, const float* optionalSourceData);
    std::shared_ptr<Image> CreateLuminanceImage(uint32_t width, uint32_t height, const float* optionalSourceData);
    std::shared_ptr<Image> CreateNormalsImage(uint32_t width, uint32_t height, const uint32_t* optionalSourceData);

    // Clears the output to black
    void Clear();

    // Copies data from the CPU to the image
    void FillColorImage(const uint32_t* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY);
    void FillDepthImage(const float* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY);
    void FillLuminanceImage(const float* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY);
    void FillNormalsImage(const uint32_t* sourceData, uint32_t width, uint32_t height, const std::shared_ptr<Image>& dest, int destX, int destY);

    // Copies the source image to the dest image
    void CopyImage(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, converts from color to luminance, and stores it in the dest image
    void ColorToLum(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, converts from luminance to normals, and stores it in the dest image
    void LumToNormals(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, converts from depth to normals, and stores it in the dest image
    void DepthToNormals(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Draws the image to the specified location and size on the screen (in pixels)
    void DrawImage(const std::shared_ptr<Image>& image, int x, int y, uint32_t width, uint32_t height);

    // Finalizes and presents output to the window
    void Present();

private:
    void InitializeGraphics(HWND targetWindow);
    std::shared_ptr<Image> CreateImageInternal(uint32_t width, uint32_t height, DXGI_FORMAT format, ImageType type, const void* optionalSourceData, uint32_t sourceStride);
    void FillImageInternal(const void* sourceData, uint32_t width, uint32_t height, uint32_t sourceStride, const std::shared_ptr<Image>& dest, int destX, int destY);

private:
    ComPtr<IDXGIFactory2> Factory;
    ComPtr<IDXGIAdapter> Adapter;
    ComPtr<IDXGISwapChain1> SwapChain;
    ComPtr<ID3D11Device> Device;
    ComPtr<ID3D11DeviceContext> Context;
    ComPtr<ID3D11Texture2D> BackBuffer;
    ComPtr<ID3D11RenderTargetView> BackBufferRTV;
    ComPtr<ID3D11SamplerState> LinearSampler;
    D3D11_VIEWPORT Viewport;

    // DrawQuad
    ComPtr<ID3D11VertexShader> DrawQuadVS;
    ComPtr<ID3D11PixelShader> DrawQuadPS;
    ComPtr<ID3D11InputLayout> DrawQuadIL;
    ComPtr<ID3D11Buffer> DrawQuadVB;
    ComPtr<ID3D11Buffer> DrawQuadVS_CB;

    struct DrawQuadVSConstants
    {
        XMINT2 Offset;              // in pixels
        XMUINT2 Size;               // in pixels
        XMFLOAT2 InvViewportSize;   // in pixels
        int pad[2];
    };

    struct DrawQuadVertex
    {
        XMFLOAT2 Position;
        XMFLOAT2 TexCoord;
    };
};
