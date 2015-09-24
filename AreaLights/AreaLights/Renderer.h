#pragma once

class Renderer
{
public:
    static std::unique_ptr<Renderer> Create(HWND window);
    virtual ~Renderer();

    void Render(FXMVECTOR cameraPosition, FXMMATRIX worldToView, CXMMATRIX projection);

private:
    Renderer(HWND window);

    bool Initialize();
    bool InitScene();

private:
    HWND                                Window;
    int32_t                             Width;
    int32_t                             Height;
    ComPtr<ID3D11Device>                Device;
    ComPtr<ID3D11DeviceContext>         Context;
    ComPtr<IDXGISwapChain>              SwapChain;
    ComPtr<ID3D11Texture2D>             BackBuffer;
    ComPtr<ID3D11RenderTargetView>      BackBufferRTV;

    // Pipeline
    ComPtr<ID3D11InputLayout>           InputLayout;
    ComPtr<ID3D11VertexShader>          VertexShader;
    ComPtr<ID3D11PixelShader>           PixelShader;
    ComPtr<ID3D11PixelShader>           DrawLightPixelShader;
    ComPtr<ID3D11Buffer>                VSConstantBuffer;
    ComPtr<ID3D11Buffer>                PSConstantBuffer;

    // Rendering data (TODO: currently single, static scene. Make this into modifyable model objects)
    ComPtr<ID3D11Buffer>                VertexBuffer;
    ComPtr<ID3D11Buffer>                IndexBuffer;
    uint32_t                            IndexCount;

    // Light geometry for visualizing
    ComPtr<ID3D11Buffer>                LightVertexBuffer;
    ComPtr<ID3D11Buffer>                LightIndexBuffer;
    uint32_t                            LightIndexCount;

    // Vertex type
    struct Vertex
    {
        XMFLOAT3 Position;
        XMFLOAT3 Normal;

        Vertex(const XMFLOAT3& position, const XMFLOAT3& normal) : Position(position), Normal(normal) {}
    };

    // Constants for vertex shader
    struct VSConstants
    {
        XMFLOAT4X4 WorldToView;
        XMFLOAT4X4 Projection;
    };

    // Constants for pixel shader
    struct PSConstants
    {
        // Test area light
        XMFLOAT4 AreaLightCorners[4];
        XMFLOAT3 AreaLightNormal;   float pad1;
        XMFLOAT3 AreaLightColor;    float pad2;
    };
};
