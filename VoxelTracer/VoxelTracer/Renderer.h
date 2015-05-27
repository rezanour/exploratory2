#pragma once

class Renderer
{
    NONCOPYABLE(Renderer);

public:
    static std::unique_ptr<Renderer> Create(HWND window);
    ~Renderer();

    // For CPU ray tracing path
    void SetFov(float horizFovRadians);
    void Render(FXMMATRIX cameraWorldTransform, bool vsyncEnabled);

private:
    Renderer(HWND window);

    bool Initialize();

    // Test to see if ray hits anything in scene
    bool Raycast(FXMVECTOR start, FXMVECTOR dir);

    static uint32_t UintFromColor(const XMFLOAT3& color);

private:
    HWND                                Window;
    int32_t                             Width;
    int32_t                             Height;
    ComPtr<ID3D11Device>                Device;
    ComPtr<ID3D11DeviceContext>         Context;
    ComPtr<IDXGISwapChain>              SwapChain;
    ComPtr<ID3D11Texture2D>             BackBuffer;
    ComPtr<ID3D11RenderTargetView>      BackBufferRT;
    ComPtr<ID3D11UnorderedAccessView>   BackBufferUAV;
    ComPtr<ID3D11ComputeShader>         ComputeShader;

    // Temporary CPU ray tracing path for bring up...
    std::unique_ptr<uint32_t[]>         PixelBuffer;
    // Just points to the underlying buffer in PixelBuffer (to avoid constant unique_ptr accessors)
    uint32_t*                           Pixels;
    // For computing eye rays
    float                               HalfWidth;
    float                               HalfHeight;
    float                               HorizFov;
    float                               DistToProjPlane;

    // Constants to pass shader
    struct CameraData
    {
        XMFLOAT4X4 CameraWorldTransform;
        XMFLOAT2 HalfSize;
        float DistToProjPlane;
        float Padding;
    };
    ComPtr<ID3D11Buffer>                CameraDataCB;

    // Scene
    struct AABB
    {
        XMFLOAT3 Center;
        XMFLOAT3 HalfWidths;
    };
    std::vector<AABB>                   Scene;

    ComPtr<ID3D11Buffer>                Blocks;
    ComPtr<ID3D11ShaderResourceView>    BlocksSRV;
};
