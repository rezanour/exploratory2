#include "Precomp.h"
#include "Renderer.h"
#include "Debug.h"
#include "RaytraceCS.h"

//#define USE_FPU_CPU_TEST
//#define USE_CPU

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
    , Pixels(nullptr)
    , HorizFov(0.f)
    , DistToProjPlane(0.f)
{
    assert(Window);

    RECT clientRect = {};
    GetClientRect(Window, &clientRect);

    Width = clientRect.right - clientRect.left;
    Height = clientRect.bottom - clientRect.top;
    HalfWidth = Width * 0.5f;
    HalfHeight = Height * 0.5f;
}

Renderer::~Renderer()
{
    Pixels = nullptr;
}

void Renderer::SetFov(float horizFovRadians)
{
    HorizFov = horizFovRadians;
    float denom = tanf(HorizFov * 0.5f);
    assert(!isnan(denom) && fabsf(denom) > 0.00001f);
    DistToProjPlane = HalfWidth / denom;
}

void Renderer::Render(FXMMATRIX cameraWorldTransform, bool vsyncEnabled)
{
    static const uint32_t ClearColor = 0xFF000000;

#if defined(USE_CPU)
    // Render scene via CPU raycasting
    for (int y = 0; y < Height; ++y)
    {
        for (int x = 0; x < Width; ++x)
        {
            // Compute ray direction
            XMVECTOR dir = XMVectorScale(cameraWorldTransform.r[2], DistToProjPlane);
            dir = XMVectorAdd(dir, XMVectorScale(cameraWorldTransform.r[0], (float)x - HalfWidth));
            dir = XMVectorAdd(dir, XMVectorScale(cameraWorldTransform.r[1], HalfHeight - (float)y));
            dir = XMVector3Normalize(dir);

            if (Raycast(cameraWorldTransform.r[3], dir))
            {
                Pixels[y * Width + x] = UintFromColor(XMFLOAT3(0.f, 0.25f, 0.75f));
            }
            else
            {
                Pixels[y * Width + x] = ClearColor;
            }
        }
    }

    // BLT CPU buffer to backbuffer
    Context->UpdateSubresource(BackBuffer.Get(), 0, nullptr, Pixels, Width * sizeof(uint32_t), Width * Height * sizeof(uint32_t));

#else

    // Bind the UAV
    UINT count = 0;
    Context->CSSetUnorderedAccessViews(0, 1, BackBufferUAV.GetAddressOf(), &count);

    // Update constants
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = Context->Map(CameraDataCB.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr))
    {
        LogError(L"Failed to map constant buffer for writing.");
        return;
    }

    CameraData* camera = (CameraData*)mapped.pData;
    XMStoreFloat4x4(&camera->CameraWorldTransform, cameraWorldTransform);
    camera->HalfSize = XMFLOAT2(HalfWidth, HalfHeight);
    camera->DistToProjPlane = DistToProjPlane;
    Context->Unmap(CameraDataCB.Get(), 0);

    Context->Dispatch(Width / 4, Height / 4, 1);

#endif

    SwapChain->Present(vsyncEnabled ? 1 : 0, 0);
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
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | DXGI_USAGE_UNORDERED_ACCESS;
    scd.OutputWindow = Window;
    scd.SampleDesc.Count = 1;
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
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

    hr = Device->CreateRenderTargetView(BackBuffer.Get(), nullptr, &BackBufferRT);
    if (FAILED(hr))
    {
        LogError(L"Failed to create back buffer render target. hr = 0x%08x.\n", hr);
        return false;
    }

#if !defined(USE_CPU)
    // Create UAV for compute access
    hr = Device->CreateUnorderedAccessView(BackBuffer.Get(), nullptr, &BackBufferUAV);
    if (FAILED(hr))
    {
        LogError(L"Failed to create back buffer uav.");
        return false;
    }

    // Create compute shader
    hr = Device->CreateComputeShader(RaytraceCS, _countof(RaytraceCS), nullptr, &ComputeShader);
    if (FAILED(hr))
    {
        LogError(L"Failed to create compute shader.");
        return false;
    }

    // Bind the shader
    Context->CSSetShader(ComputeShader.Get(), nullptr, 0);

    // Create the constant buffer
    D3D11_BUFFER_DESC bd = {};
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.ByteWidth = sizeof(CameraData);
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.StructureByteStride = bd.ByteWidth;
    bd.Usage = D3D11_USAGE_DYNAMIC;

    hr = Device->CreateBuffer(&bd, nullptr, &CameraDataCB);
    if (FAILED(hr))
    {
        LogError(L"Failed to create constant buffer.");
        return false;
    }

    // Bind the constant buffer
    Context->CSSetConstantBuffers(0, 1, CameraDataCB.GetAddressOf());

#endif

    // Create scene
    AABB block{};

#if 0
    block.Center = XMFLOAT3(0.f, 0.f, 0.f);
    block.HalfWidths = XMFLOAT3(0.5f, 0.5f, 0.5f);
    Scene.push_back(block);

    block.Center = XMFLOAT3(0.f, 0.75f, 0.f);
    block.HalfWidths = XMFLOAT3(0.25f, 0.25f, 0.25f);
    Scene.push_back(block);

    block.Center = XMFLOAT3(0.f, 1.125f, 0.f);
    block.HalfWidths = XMFLOAT3(0.125f, 0.125f, 0.125f);
    Scene.push_back(block);
#else
    for (int i = 0; i < 150; ++i)
    {
        block.HalfWidths = XMFLOAT3(rand() / (float)RAND_MAX * 0.25f, rand() / (float)RAND_MAX * 0.25f, rand() / (float)RAND_MAX * 0.25f);
        block.Center = XMFLOAT3(rand() / (float)RAND_MAX * 10 - 5, rand() / (float)RAND_MAX * 10 - 5, rand() / (float)RAND_MAX * 10);
        Scene.push_back(block);
    }
#endif

#if defined(USE_CPU)

    PixelBuffer.reset(new uint32_t[Width * Height]);
    Pixels = PixelBuffer.get();

#else

    // Create buffer to hold the scene's sphere
    bd = {};
    bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    bd.ByteWidth = sizeof(AABB) * (int)Scene.size();
    bd.StructureByteStride = sizeof(AABB);
    bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
    bd.Usage = D3D11_USAGE_DEFAULT;

    // Initialize the sphere buffer with the spheres
    D3D11_SUBRESOURCE_DATA init = {};
    init.pSysMem = Scene.data();
    init.SysMemPitch = bd.ByteWidth;
    init.SysMemSlicePitch = init.SysMemPitch;

    hr = Device->CreateBuffer(&bd, &init, &Blocks);
    if (FAILED(hr))
    {
        LogError(L"Failed to create blocks buffer.");
        return false;
    }

    // Create shader resource so we can read from it in the shader
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
    srvDesc.Format = DXGI_FORMAT_UNKNOWN;
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
    srvDesc.Buffer.ElementWidth = sizeof(AABB);
    srvDesc.Buffer.NumElements = (int)Scene.size();

    hr = Device->CreateShaderResourceView(Blocks.Get(), &srvDesc, &BlocksSRV);
    if (FAILED(hr))
    {
        LogError(L"Failed to create blocks buffer srv.");
        return false;
    }

    // Bind it at slot 0
    Context->CSSetShaderResources(0, 1, BlocksSRV.GetAddressOf());

#endif

    return true;
}

bool Renderer::Raycast(FXMVECTOR start, FXMVECTOR dir)
{
#if defined(USE_FPU_CPU_TEST)
    XMFLOAT3 sf, df;
    XMStoreFloat3(&sf, start);
    XMStoreFloat3(&df, dir);
    float* s = &sf.x;
    float* d = &df.x;
#endif

    for (auto& block : Scene)
    {
#if defined(USE_FPU_CPU_TEST)   // FPU version

        // test the 3 axes
        for (int i = 0; i < 3; ++i)
        {
            float blockCenter = (&block.Center.x)[i];
            float blockHalfWidth = (&block.HalfWidths.x)[i];

            if ((s[i] < blockCenter && d[i] > 0) ||
                (s[i] > blockCenter && d[i] < 0))
            {
                float dist = fabsf(s[i] - blockCenter);
                if (dist > blockHalfWidth)
                {
                    dist -= blockHalfWidth;
                    dist = fabsf(dist / d[i]);
                    XMFLOAT3 pf = XMFLOAT3(s[0] + d[0] * dist, s[1] + d[1] * dist, s[2] + d[2] * dist);
                    float* p = &pf.x;

                    bool hit = true;
                    for (int j = 0; j < 3; ++j)
                    {
                        if (i == j) continue;

                        if (fabsf(p[j] - (&block.Center.x)[j]) > (&block.HalfWidths.x)[j])
                        {
                            // no hit
                            hit = false;
                            break;
                        }
                    }
                    if (hit)
                    {
                        return true;
                    }
                }
            }
        }

#else   // Vector version
        XMVECTOR blockCenter = XMLoadFloat3(&block.Center);
        XMVECTOR blockHalfWidths = XMLoadFloat3(&block.HalfWidths);
        XMVECTOR nonAbsDist = XMVectorSubtract(start, blockCenter);
        XMVECTOR totalDist = XMVectorAbs(nonAbsDist);
        XMVECTOR dist = XMVectorSubtract(totalDist, blockHalfWidths);
        XMVECTOR startDirCheck = XMVectorMultiply(nonAbsDist, dir);
        XMVECTOR scaleValue = XMVectorAbs(XMVectorMultiply(dist, XMVectorReciprocal(dir)));
        uint32_t testResult = 0;
        // x
        if (XMVectorGetX(startDirCheck) < 0 && XMVectorGetX(dist) > 0)
        {
            XMVECTOR p = XMVectorAdd(start, XMVectorScale(dir, XMVectorGetX(scaleValue)));
            testResult = XMVector3GreaterR(XMVectorAbs(XMVectorSubtract(p, blockCenter)), blockHalfWidths);
            if (XMComparisonAllFalse(testResult))
            {
                // hit!
                return true;
            }
        }
        // y
        if (XMVectorGetY(startDirCheck) < 0 && XMVectorGetY(dist) > 0)
        {
            XMVECTOR p = XMVectorAdd(start, XMVectorScale(dir, XMVectorGetY(scaleValue)));
            testResult = XMVector3GreaterR(XMVectorAbs(XMVectorSubtract(p, blockCenter)), blockHalfWidths);
            if (XMComparisonAllFalse(testResult))
            {
                // hit!
                return true;
            }
        }
        // z
        if (XMVectorGetZ(startDirCheck) < 0 && XMVectorGetZ(dist) > 0)
        {
            XMVECTOR p = XMVectorAdd(start, XMVectorScale(dir, XMVectorGetZ(scaleValue)));
            testResult = XMVector3GreaterR(XMVectorAbs(XMVectorSubtract(p, blockCenter)), blockHalfWidths);
            if (XMComparisonAllFalse(testResult))
            {
                // hit!
                return true;
            }
        }
#endif
    }

    return false;
}

uint32_t Renderer::UintFromColor(const XMFLOAT3& color)
{
    return 0xFF000000 |
        (min(255, (int)(color.z * 255.0f)) << 16) |
        (min(255, (int)(color.y * 255.0f)) << 8) |
         min(255, (int)(color.x * 255.0f));
}
