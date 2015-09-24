#include "Precomp.h"
#include "Renderer.h"
#include "Debug.h"
#include "TheVertexShader.h"
#include "ThePixelShader.h"
#include "DrawLightPS.h"

std::unique_ptr<Renderer> Renderer::Create(HWND window)
{
    std::unique_ptr<Renderer> renderer(new Renderer(window));
    if (renderer)
    {
        if (renderer->Initialize())
        {
            return renderer;
        }
    }
    return nullptr;
}

Renderer::Renderer(HWND hwnd)
    : Window(hwnd)
    , IndexCount(0)
{
    assert(Window);

    RECT clientRect = {};
    GetClientRect(Window, &clientRect);

    Width = clientRect.right - clientRect.left;
    Height = clientRect.bottom - clientRect.top;
}

Renderer::~Renderer()
{
}

void Renderer::Render(FXMVECTOR cameraPosition, FXMMATRIX worldToView, CXMMATRIX projection)
{
    UNREFERENCED_PARAMETER(cameraPosition);

    // Bind the RTV
    Context->OMSetRenderTargets(1, BackBufferRTV.GetAddressOf(), nullptr);

    D3D11_VIEWPORT vp{};
    vp.Width = (float)Width;
    vp.Height = (float)Height;
    vp.MaxDepth = 1.f;
    Context->RSSetViewports(1, &vp);

    // Update constants
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = Context->Map(VSConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr))
    {
        LogError(L"Failed to map constant buffer for writing.");
        return;
    }

    VSConstants* vsConstants = (VSConstants*)mapped.pData;
    XMStoreFloat4x4(&vsConstants->WorldToView, worldToView);
    XMStoreFloat4x4(&vsConstants->Projection, projection);
    Context->Unmap(VSConstantBuffer.Get(), 0);

    hr = Context->Map(PSConstantBuffer.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr))
    {
        LogError(L"Failed to map constant buffer for writing.");
        return;
    }

    PSConstants* psConstants = (PSConstants*)mapped.pData;
    psConstants->AreaLightCorners[0] = XMFLOAT4(-0.75f, 2.f, -2.f, 0.f);
    psConstants->AreaLightCorners[1] = XMFLOAT4(0.75f, 2.f, -2.f, 0.f);
    psConstants->AreaLightCorners[2] = XMFLOAT4(0.75f, 2.f, 2.f, 0.f);
    psConstants->AreaLightCorners[3] = XMFLOAT4(-0.75f, 2.f, 2.f, 0.f);
    psConstants->AreaLightNormal = XMFLOAT3(0.f, -1.f, 0.f);
    psConstants->AreaLightColor = XMFLOAT3(1.f, 1.f, 1.f);
    Context->Unmap(PSConstantBuffer.Get(), 0);

    // Bind scene
    uint32_t stride = sizeof(Vertex);
    uint32_t offset = 0;
    Context->IASetVertexBuffers(0, 1, VertexBuffer.GetAddressOf(), &stride, &offset);
    Context->IASetIndexBuffer(IndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    Context->PSSetShader(PixelShader.Get(), nullptr, 0);

    // Draw
    Context->DrawIndexed(IndexCount, 0, 0);

    // Bind light visualization
    Context->IASetVertexBuffers(0, 1, LightVertexBuffer.GetAddressOf(), &stride, &offset);
    Context->IASetIndexBuffer(LightIndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
    Context->PSSetShader(DrawLightPixelShader.Get(), nullptr, 0);

    // Draw
    Context->DrawIndexed(LightIndexCount, 0, 0);



    SwapChain->Present(1, 0);
}

bool Renderer::Initialize()
{
    ComPtr<IDXGIFactory1> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        LogError(L"Failed to create DXGI factory. hr = 0x%08x.\n", hr);
        return false;
    }

    // Find first enumerated HW adapter
    ComPtr<IDXGIAdapter1> adapter;
    UINT iAdapter = 0;
    while (SUCCEEDED(hr = factory->EnumAdapters1(iAdapter++, adapter.ReleaseAndGetAddressOf())))
    {
        DXGI_ADAPTER_DESC1 adapterDesc{};
        adapter->GetDesc1(&adapterDesc);

        if ((adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0)
        {
            // HW adapter
            break;
        }
    }

    if (FAILED(hr))
    {
        LogError(L"Failed to find any hardware graphics adapters. hr = 0x%08x.\n", hr);
        return false;
    }

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    UINT d3dFlags = 0;
#if defined(_DEBUG)
    d3dFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, d3dFlags,
        &featureLevel, 1, D3D11_SDK_VERSION, &Device, nullptr, &Context);
    if (FAILED(hr))
    {
        LogError(L"Failed to create d3d11 device. hr = 0x%08x.\n", hr);
        return false;
    }

    DXGI_SWAP_CHAIN_DESC scd{};
    scd.BufferCount = 2;
    scd.BufferDesc.Width = Width;
    scd.BufferDesc.Height = Height;
    scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.OutputWindow = Window;
    scd.SampleDesc.Count = 1;
    scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    scd.Windowed = TRUE;

    hr = factory->CreateSwapChain(Device.Get(), &scd, &SwapChain);
    if (FAILED(hr))
    {
        LogError(L"Failed to create DXGI swapchain. hr = 0x%08x.\n", hr);
        return false;
    }

    hr = SwapChain->GetBuffer(0, IID_PPV_ARGS(&BackBuffer));
    if (FAILED(hr))
    {
        LogError(L"Failed to get back buffer texture. hr = 0x%08x.\n", hr);
        return false;
    }

    hr = Device->CreateRenderTargetView(BackBuffer.Get(), nullptr, &BackBufferRTV);
    if (FAILED(hr))
    {
        LogError(L"Failed to create back buffer render target. hr = 0x%08x.\n", hr);
        return false;
    }

    // Create scene
    if (!InitScene())
    {
        LogError(L"Failed to create the static scene.");
        return false;
    }

    // Load shaders and build up pipeline
    hr = Device->CreateVertexShader(TheVertexShader, sizeof(TheVertexShader), nullptr, &VertexShader);
    if (FAILED(hr))
    {
        LogError(L"Failed to create vertex shader. hr = 0x%08x.\n", hr);
        return false;
    }

    hr = Device->CreatePixelShader(ThePixelShader, sizeof(ThePixelShader), nullptr, &PixelShader);
    if (FAILED(hr))
    {
        LogError(L"Failed to create pixel shader. hr = 0x%08x.\n", hr);
        return false;
    }

    hr = Device->CreatePixelShader(DrawLightPS, sizeof(DrawLightPS), nullptr, &DrawLightPixelShader);
    if (FAILED(hr))
    {
        LogError(L"Failed to create pixel shader. hr = 0x%08x.\n", hr);
        return false;
    }

    D3D11_INPUT_ELEMENT_DESC elems[2]{};
    elems[0].Format = DXGI_FORMAT_R32G32B32_FLOAT;
    elems[0].SemanticName = "POSITION";
    elems[1].AlignedByteOffset = sizeof(XMFLOAT3);
    elems[1].Format = DXGI_FORMAT_R32G32B32_FLOAT;
    elems[1].SemanticName = "NORMAL";
    hr = Device->CreateInputLayout(elems, _countof(elems), TheVertexShader, sizeof(TheVertexShader), &InputLayout);
    if (FAILED(hr))
    {
        LogError(L"Failed to create input layout. hr = 0x%08x.\n", hr);
        return false;
    }

    // Create the constant buffers
    D3D11_BUFFER_DESC bd = {};
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.ByteWidth = sizeof(VSConstants);
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.StructureByteStride = bd.ByteWidth;
    bd.Usage = D3D11_USAGE_DYNAMIC;

    hr = Device->CreateBuffer(&bd, nullptr, &VSConstantBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create constant buffer.");
        return false;
    }

    bd.ByteWidth = sizeof(PSConstants);
    bd.StructureByteStride = bd.ByteWidth;

    hr = Device->CreateBuffer(&bd, nullptr, &PSConstantBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create constant buffer.");
        return false;
    }

    // Create scene
    if (!InitScene())
    {
        LogError(L"Failed to create the static scene.");
        return false;
    }

    // Bind the pipeline state
    Context->IASetInputLayout(InputLayout.Get());
    Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    Context->VSSetShader(VertexShader.Get(), nullptr, 0);
    Context->PSSetShader(PixelShader.Get(), nullptr, 0);

    // Bind the constant buffers
    Context->VSSetConstantBuffers(0, 1, VSConstantBuffer.GetAddressOf());
    Context->PSSetConstantBuffers(0, 1, PSConstantBuffer.GetAddressOf());

    return true;
}

bool Renderer::InitScene()
{
    std::vector<Vertex> vertices;

    // front
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, -0.5f), XMFLOAT3(0.f, 0.f, -1.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, -0.5f), XMFLOAT3(0.f, 0.f, -1.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, -0.5f), XMFLOAT3(0.f, 0.f, -1.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, -0.5f), XMFLOAT3(0.f, 0.f, -1.f)));

    // back
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, 0.5f), XMFLOAT3(0.f, 0.f, 1.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, 0.5f), XMFLOAT3(0.f, 0.f, 1.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, 0.5f), XMFLOAT3(0.f, 0.f, 1.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, 0.5f), XMFLOAT3(0.f, 0.f, 1.f)));

    // left
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, 0.5f), XMFLOAT3(-1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, -0.5f), XMFLOAT3(-1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, -0.5f), XMFLOAT3(-1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, 0.5f), XMFLOAT3(-1.f, 0.f, 0.f)));

    // right
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, -0.5f), XMFLOAT3(1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, 0.5f), XMFLOAT3(1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, 0.5f), XMFLOAT3(1.f, 0.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, -0.5f), XMFLOAT3(1.f, 0.f, 0.f)));

    // top
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, 0.5f), XMFLOAT3(0.f, 1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, 0.5f), XMFLOAT3(0.f, 1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, 0.5f, -0.5f), XMFLOAT3(0.f, 1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, 0.5f, -0.5f), XMFLOAT3(0.f, 1.f, 0.f)));

    // bottom
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, -0.5f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, -0.5f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(2.5f, -0.5f, 0.5f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-2.5f, -0.5f, 0.5f), XMFLOAT3(0.f, -1.f, 0.f)));

    uint32_t indices[] =
    {
        0, 1, 2,
        0, 2, 3,
        4, 5, 6,
        4, 6, 7,
        8, 9, 10,
        8, 10, 11,
        12, 13, 14,
        12, 14, 15,
        16, 17, 18,
        16, 18, 19,
        20, 21, 22,
        20, 22, 23,
    };

    IndexCount = _countof(indices);

    // Create buffers
    D3D11_BUFFER_DESC bd = {};
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(Vertex) * (int)vertices.size();
    bd.StructureByteStride = sizeof(Vertex);

    D3D11_SUBRESOURCE_DATA init = {};
    init.pSysMem = vertices.data();
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = init.SysMemPitch;

    HRESULT hr = Device->CreateBuffer(&bd, &init, &VertexBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create vertex buffer.");
        return false;
    }

    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.ByteWidth = sizeof(uint32_t) * _countof(indices);
    bd.StructureByteStride = sizeof(uint32_t);

    init.pSysMem = indices;
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = init.SysMemPitch;

    hr = Device->CreateBuffer(&bd, &init, &IndexBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create index buffer.");
        return false;
    }

    // Bind them
    uint32_t stride = sizeof(Vertex);
    uint32_t offset = 0;
    Context->IASetVertexBuffers(0, 1, VertexBuffer.GetAddressOf(), &stride, &offset);
    Context->IASetIndexBuffer(IndexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);

    // Geometry for visualizing the light(s)
    vertices.clear();

    // light
    vertices.push_back(Vertex(XMFLOAT3(-0.75f, 2.f, -2.f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(0.75f, 2.f, -2.f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(0.75f, 2.f, 2.f), XMFLOAT3(0.f, -1.f, 0.f)));
    vertices.push_back(Vertex(XMFLOAT3(-0.75f, 2.f, 2.f), XMFLOAT3(0.f, -1.f, 0.f)));

    uint32_t lightIndices[] = { 0, 1, 2, 0, 2, 3 };
    LightIndexCount = _countof(lightIndices);

    // Create buffers
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    bd.ByteWidth = sizeof(Vertex) * (int)vertices.size();
    bd.StructureByteStride = sizeof(Vertex);

    init.pSysMem = vertices.data();
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = init.SysMemPitch;

    hr = Device->CreateBuffer(&bd, &init, &LightVertexBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create vertex buffer.");
        return false;
    }

    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    bd.ByteWidth = sizeof(uint32_t) * _countof(lightIndices);
    bd.StructureByteStride = sizeof(uint32_t);

    init.pSysMem = lightIndices;
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = init.SysMemPitch;

    hr = Device->CreateBuffer(&bd, &init, &LightIndexBuffer);
    if (FAILED(hr))
    {
        LogError(L"Failed to create index buffer.");
        return false;
    }

    return true;
}
