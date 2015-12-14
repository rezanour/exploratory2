#include "precomp.h"
#include "rasterizer.h"

#include <vector>
#include <DirectXMath.h>
using namespace DirectX;

// Programmable stages
struct Constants
{
    XMFLOAT4X4 WorldMatrix;
    XMFLOAT4X4 ViewMatrix;
    XMFLOAT4X4 ProjectionMatrix;
};

struct alignas(16) VertexInput
{
    XMFLOAT3 Position;
    XMFLOAT3 Normal;
    XMFLOAT2 TexCoord;
};

struct alignas(16) VertexOutput
{
    XMFLOAT4 PostProjectionPosition;
    XMFLOAT3 WorldPosition;
    XMFLOAT3 WorldNormal;
    XMFLOAT2 TexCoord;
};

static VertexOutput VertexShader(VertexInput input);
static XMFLOAT4 PixelShader(VertexOutput input);


// Other stuff
struct Vertex
{
    XMFLOAT3 Position;
    XMFLOAT3 Normal;
    XMFLOAT2 TexCoord;

    Vertex(const XMFLOAT3& pos, const XMFLOAT3& norm, const XMFLOAT2& tex)
        : Position(pos), Normal(norm), TexCoord(tex)
    {}
};
static const uint32_t Stride = sizeof(Vertex);

// Variables
static Constants LocalConstants, ShaderConstants;
static std::vector<Vertex> Vertices;
static std::vector<VertexOutput> VertOutput;
static uint32_t FrameIndex;

static bool LerpFragment(float x, float y, const VertexOutput* a, const VertexOutput* b, const VertexOutput* c, VertexOutput* frag);

bool RastStartup()
{
    // Fill in vertices for triangle
    Vertices.push_back(Vertex(XMFLOAT3(-0.5f, -0.5f, 0.f), XMFLOAT3(0.f, 0.f, 1.f), XMFLOAT2(0.f, 1.f)));
    Vertices.push_back(Vertex(XMFLOAT3(0.f, 0.5f, 0.f), XMFLOAT3(0.f, 1.f, 0.f), XMFLOAT2(0.5f, 0.f)));
    Vertices.push_back(Vertex(XMFLOAT3(0.5f, -0.5f, 0.f), XMFLOAT3(1.f, 0.f, 0.f), XMFLOAT2(1.f, 1.f)));

    VertOutput.resize(Vertices.size());

    XMStoreFloat4x4(&LocalConstants.WorldMatrix, XMMatrixIdentity());
    XMStoreFloat4x4(&LocalConstants.ViewMatrix, XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -1.f, 1.f), XMVectorSet(0.f, 0.f, 0.f, 1.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)));
    XMStoreFloat4x4(&LocalConstants.ProjectionMatrix, XMMatrixPerspectiveFovLH(XMConvertToRadians(90.f), 1280.f / 720.f, 0.1f, 100.f));

    return true;
}

void RastShutdown()
{
}


bool RenderScene(void* const pOutput, uint32_t width, uint32_t height, uint32_t rowPitch)
{
    uint32_t pitch = rowPitch / sizeof(uint32_t);

    memset(pOutput, 0, rowPitch * height);

    XMStoreFloat4x4(&LocalConstants.WorldMatrix, XMMatrixRotationY(FrameIndex++ * 0.25f));

    // Serial, simple impl first just to try some ideas. Then will do this right
    ShaderConstants = LocalConstants;

    uint32_t numVerts = (uint32_t)Vertices.size();
    Vertex* v = Vertices.data();
    VertexOutput* out = VertOutput.data();

    // Vertex Shader Stage
    for (uint32_t i = 0; i < numVerts; ++i, ++v, ++out)
    {
        VertexInput input;
        input.Position = v->Position;
        input.Normal = v->Normal;
        input.TexCoord = v->TexCoord;
        *out = VertexShader(input);

        // w divide & convert to viewport (pixels)
        float invW = 1.f / out->PostProjectionPosition.w;
        out->PostProjectionPosition.x = (out->PostProjectionPosition.x * invW * 0.5f + 0.5f) * width;
        out->PostProjectionPosition.y = (1.f - (out->PostProjectionPosition.y * invW * 0.5f + 0.5f)) * height;
        out->PostProjectionPosition.z = out->PostProjectionPosition.z * invW;
        out->PostProjectionPosition.w = 1.f;
    }

    // TODO: Clip

    // Rasterize
    out = VertOutput.data();
    for (uint32_t i = 0; i < numVerts / 3; ++i, out += 3)
    {
        VertexOutput* verts[3] = { out, out + 1, out + 2 };
        uint32_t top = 0, bottom = 0, mid = 0;

        for (uint32_t j = 0; j < 3; ++j)
        {
            if (verts[j]->PostProjectionPosition.y > verts[bottom]->PostProjectionPosition.y) bottom = j;
            if (verts[j]->PostProjectionPosition.y < verts[top]->PostProjectionPosition.y) top = j;
        }

        for (uint32_t j = 0; j < 3; ++j)
        {
            if (j != top && j != bottom)
            {
                mid = j;
                break;
            }
        }

        // first, rasterize from top to other
        float ytop = verts[top]->PostProjectionPosition.y;
        float ymid = verts[mid]->PostProjectionPosition.y;
        float ybottom = verts[bottom]->PostProjectionPosition.y;

        uint32_t left = (verts[mid]->PostProjectionPosition.x < verts[bottom]->PostProjectionPosition.x) ? mid : bottom;
        uint32_t right = (left == mid) ? bottom : mid;

        float step1 =
            (verts[left]->PostProjectionPosition.x - verts[top]->PostProjectionPosition.x) /
            (verts[left]->PostProjectionPosition.y - verts[top]->PostProjectionPosition.y);

        float step2 =
            (verts[right]->PostProjectionPosition.x - verts[top]->PostProjectionPosition.x) /
            (verts[right]->PostProjectionPosition.y - verts[top]->PostProjectionPosition.y);

        float x1 = verts[top]->PostProjectionPosition.x;
        float x2 = x1;

        for (int y = (int)ytop; y < (int)ymid; ++y)
        {
            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                XMFLOAT4 fragColor = PixelShader(frag);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * pitch + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
            }

            x1 += step1;
            x2 += step2;
        }

        // then, from other to bottom
        left = (verts[top]->PostProjectionPosition.x < verts[mid]->PostProjectionPosition.x) ? top : mid;
        right = (left == top) ? mid: top;

        step1 =
            (verts[bottom]->PostProjectionPosition.x - verts[left]->PostProjectionPosition.x) /
            (verts[bottom]->PostProjectionPosition.y - verts[left]->PostProjectionPosition.y);

        step2 =
            (verts[bottom]->PostProjectionPosition.x - verts[right]->PostProjectionPosition.x) /
            (verts[bottom]->PostProjectionPosition.y - verts[right]->PostProjectionPosition.y);

        for (int y = (int)ymid; y < (int)ybottom; ++y)
        {
            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                XMFLOAT4 fragColor = PixelShader(frag);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * pitch + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
            }

            x1 += step1;
            x2 += step2;
        }

    }

    // Pixel Shader Stage

    return true;
}



VertexOutput VertexShader(VertexInput input)
{
    XMVECTOR position = XMVectorSetW(XMLoadFloat3(&input.Position), 1);
    XMVECTOR normal = XMLoadFloat3(&input.Normal);
    XMMATRIX worldMatrix = XMLoadFloat4x4(&ShaderConstants.WorldMatrix);
    XMMATRIX viewMatrix = XMLoadFloat4x4(&ShaderConstants.ViewMatrix);
    XMMATRIX projMatrix = XMLoadFloat4x4(&ShaderConstants.ProjectionMatrix);

    VertexOutput output;
    XMVECTOR pos = XMVector4Transform(position, worldMatrix);
    XMStoreFloat3(&output.WorldPosition, pos);
    //XMStoreFloat3(&output.WorldNormal, XMVector3TransformNormal(normal, worldMatrix));
    output.WorldNormal = input.Normal;
    XMStoreFloat4(&output.PostProjectionPosition, XMVector4Transform(pos, XMMatrixMultiply(viewMatrix, projMatrix)));
    output.TexCoord = input.TexCoord;

    return output;
}

XMFLOAT4 PixelShader(VertexOutput input)
{
    //UNREFERENCED(input);
    return XMFLOAT4(input.WorldNormal.x, input.WorldNormal.y, input.WorldNormal.z, 1.f);// XMFLOAT4(0.f, 0.6f, 0.8f, 1.f);
}



bool LerpFragment(float x, float y, const VertexOutput* inA, const VertexOutput* inB, const VertexOutput* inC, VertexOutput* frag)
{
    XMVECTOR a = XMVectorSetZ(XMLoadFloat4(&inA->PostProjectionPosition), 0);
    XMVECTOR b = XMVectorSetZ(XMLoadFloat4(&inB->PostProjectionPosition), 0);
    XMVECTOR c = XMVectorSetZ(XMLoadFloat4(&inC->PostProjectionPosition), 0);
    XMVECTOR ab = XMVectorSubtract(b, a);
    XMVECTOR ac = XMVectorSubtract(c, a);

    // TODO: Check for degenerecy.
    XMVECTOR n = XMVector3Cross(ab, ac);

    XMVECTOR p = XMVectorSet(x, y, 0, 0);

    // Find barycentric coordinates of P (wA, wB, wC)
    XMVECTOR bc = XMVectorSubtract(c, b);
    XMVECTOR ap = XMVectorSubtract(p, a);
    XMVECTOR bp = XMVectorSubtract(p, b);

    XMVECTOR wC = XMVector3Cross(ab, ap);
    if (XMVectorGetX(XMVector3Dot(wC, n)) < 0.f)
    {
        return false;
    }

    XMVECTOR wB = XMVector3Cross(ap, ac);
    if (XMVectorGetX(XMVector3Dot(wB, n)) < 0.f)
    {
        return false;
    }

    XMVECTOR wA = XMVector3Cross(bc, bp);
    if (XMVectorGetX(XMVector3Dot(wA, n)) < 0.f)
    {
        return false;
    }

    float invN = XMVectorGetX(XMVector3ReciprocalLength(n));

    float xA = XMVectorGetX(XMVector3Length(wA)) * invN;
    float xB = XMVectorGetX(XMVector3Length(wB)) * invN;
    float xC = XMVectorGetX(XMVector3Length(wC)) * invN;

    XMStoreFloat4(&frag->PostProjectionPosition, XMLoadFloat4(&inA->PostProjectionPosition) * xA + XMLoadFloat4(&inB->PostProjectionPosition) * xB + XMLoadFloat4(&inC->PostProjectionPosition) * xC);
    XMStoreFloat3(&frag->WorldPosition, XMLoadFloat3(&inA->WorldPosition) * xA + XMLoadFloat3(&inB->WorldPosition) * xB + XMLoadFloat3(&inC->WorldPosition) * xC);
    XMStoreFloat3(&frag->WorldNormal, XMLoadFloat3(&inA->WorldNormal) * xA + XMLoadFloat3(&inB->WorldNormal) * xB + XMLoadFloat3(&inC->WorldNormal) * xC);
    XMStoreFloat2(&frag->TexCoord, XMLoadFloat2(&inA->TexCoord) * xA + XMLoadFloat2(&inB->TexCoord) * xB + XMLoadFloat2(&inC->TexCoord) * xC);

    return true;
}
