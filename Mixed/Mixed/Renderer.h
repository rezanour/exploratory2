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

    // Perform Gaussian blur on image
    void Gaussian(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, converts from color to luminance, and stores it in the dest image
    void ColorToLum(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, converts from luminance to normals, and stores it in the dest image
    void LumToNormals(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

    // Reads the source image, detects edges, and stores it in the dest image
    void EdgeDetect(const std::shared_ptr<Image>& source, const std::shared_ptr<Image>& dest);

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

    void BindFullScreenQuad(const ComPtr<ID3D11PixelShader>& pixelShader, const std::shared_ptr<Image>& dest);
    void BindDrawQuad(const ComPtr<ID3D11PixelShader>& pixelShader, const std::shared_ptr<Image>& dest);
    void BindQuadRendering(const ComPtr<ID3D11VertexShader>& vertexShader, const ComPtr<ID3D11InputLayout>& inputLayout, const ComPtr<ID3D11PixelShader>& pixelShader, const std::shared_ptr<Image>& dest);

private:
    ComPtr<IDXGIFactory2> Factory;
    ComPtr<IDXGIAdapter> Adapter;
    ComPtr<IDXGISwapChain1> SwapChain;
    ComPtr<ID3D11Device> Device;
    ComPtr<ID3D11DeviceContext> Context;
    ComPtr<ID3D11Texture2D> BackBuffer;
    ComPtr<ID3D11RenderTargetView> BackBufferRTV;

    // Shared
    ComPtr<ID3D11SamplerState> LinearSampler;
    D3D11_VIEWPORT Viewport;
    ComPtr<ID3D11VertexShader> FullScreenQuadVS;
    ComPtr<ID3D11PixelShader> DrawFloatTexPS;
    ComPtr<ID3D11InputLayout> FullScreenQuadIL;
    ComPtr<ID3D11Buffer> QuadVB;

    struct QuadVertex
    {
        XMFLOAT2 Position;
        XMFLOAT2 TexCoord;
    };

    // DrawQuad
    ComPtr<ID3D11VertexShader> DrawQuadVS;
    ComPtr<ID3D11PixelShader> DrawQuadPS;
    ComPtr<ID3D11InputLayout> DrawQuadIL;
    ComPtr<ID3D11Buffer> DrawQuadVS_CB;

    struct DrawQuadVSConstants
    {
        XMINT2 Offset;              // in pixels
        XMUINT2 Size;               // in pixels
        XMFLOAT2 InvViewportSize;   // in pixels
        int pad[2];
    };

    // ColorToLum
    ComPtr<ID3D11PixelShader> ColorToLumPS;

    // LumToNorm
    ComPtr<ID3D11PixelShader> LumToNormPS;

    // EdgeDetectColor
    ComPtr<ID3D11PixelShader> EdgeDetectPS;

    // Gaussian
    struct GaussianPSConstants
    {
        int Direction;  // 0 = horiz, 1 = vert
        XMINT3 Padding0;
    };

    ComPtr<ID3D11PixelShader> GaussianPS;
    ComPtr<ID3D11Buffer> GaussianPS_CB;
};
