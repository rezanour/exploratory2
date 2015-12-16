#include "precomp.h"
#include "rz_rast.h"
#include "rz_math.h"

#include <vector>
#include <DirectXMath.h>
using namespace DirectX;

// Programmable stages
struct Constants
{
    matrix4x4 WorldMatrix;
    matrix4x4 ViewMatrix;
    matrix4x4 ProjectionMatrix;
};

struct alignas(16) VertexInput
{
    float3 Position;
    float3 Color;
};

struct alignas(16) VertexOutput
{
    float4 Position; // SV_POSITION
    float3 Color;
};

static void VertexShader(const VertexInput& input, VertexOutput& output);
static void PixelShader(const VertexOutput& input, float4& output);


// Other stuff
struct Vertex
{
    float3 Position;
    float3 Color;

    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {}
};
static const uint32_t Stride = sizeof(Vertex);

// Variables
static Constants ShaderConstants;
static std::vector<Vertex> Vertices;
static std::vector<VertexOutput> VertOutput;
static uint32_t FrameIndex;

static bool LerpFragment(float x, float y, const VertexOutput* a, const VertexOutput* b, const VertexOutput* c, VertexOutput* frag);

bool RastStartup()
{
    // Fill in vertices for triangle
    Vertices.push_back(Vertex(float3(-0.5f, -0.5f, 0.f), float3(0.f, 0.f, 1.f)));
    Vertices.push_back(Vertex(float3(0.f, 0.5f, 0.f), float3(0.f, 1.f, 0.f)));
    Vertices.push_back(Vertex(float3(0.5f, -0.5f, 0.f), float3(1.f, 0.f, 0.f)));

    VertOutput.resize(Vertices.size());

    XMFLOAT4X4 temp;
    XMStoreFloat4x4(&temp, XMMatrixIdentity());
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    XMStoreFloat4x4(&temp, XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -1.f, 1.f), XMVectorSet(0.f, 0.f, 0.f, 1.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)));
    memcpy_s(&ShaderConstants.ViewMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    XMStoreFloat4x4(&temp, XMMatrixPerspectiveFovLH(XMConvertToRadians(90.f), 1280.f / 720.f, 0.1f, 100.f));
    memcpy_s(&ShaderConstants.ProjectionMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    return true;
}

void RastShutdown()
{
}


bool RenderScene(void* const pOutput, uint32_t width, uint32_t height, uint32_t rowPitch)
{
    uint32_t pitch = rowPitch / sizeof(uint32_t);

    memset(pOutput, 0, rowPitch * height);

    XMFLOAT4X4 transform;
    XMStoreFloat4x4(&transform, XMMatrixRotationY(FrameIndex++ * 0.25f));
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &transform, sizeof(transform));
    
    // Serial, simple impl first just to try some ideas. Then will do this right

    uint32_t numVerts = (uint32_t)Vertices.size();
    Vertex* v = Vertices.data();
    VertexOutput* out = VertOutput.data();

    // Vertex Shader Stage
    for (uint32_t i = 0; i < numVerts; ++i, ++v, ++out)
    {
        VertexInput input;
        input.Position = v->Position;
        input.Color = v->Color;
        VertexShader(input, *out);

        // w divide & convert to viewport (pixels)
        out->Position /= out->Position.w;
        out->Position.x = (out->Position.x * 0.5f + 0.5f) * width;
        out->Position.y = (1.f - (out->Position.y * 0.5f + 0.5f)) * height;
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
            if (verts[j]->Position.y > verts[bottom]->Position.y) bottom = j;
            if (verts[j]->Position.y < verts[top]->Position.y) top = j;
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
        float ytop = verts[top]->Position.y;
        float ymid = verts[mid]->Position.y;
        float ybottom = verts[bottom]->Position.y;

        uint32_t left = (verts[mid]->Position.x < verts[bottom]->Position.x) ? mid : bottom;
        uint32_t right = (left == mid) ? bottom : mid;

        float step1 =
            (verts[left]->Position.x - verts[top]->Position.x) /
            (verts[left]->Position.y - verts[top]->Position.y);

        float step2 =
            (verts[right]->Position.x - verts[top]->Position.x) /
            (verts[right]->Position.y - verts[top]->Position.y);

        float x1 = verts[top]->Position.x;
        float x2 = x1;

        float4 fragColor;

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

                PixelShader(frag, fragColor);

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
        left = (verts[top]->Position.x < verts[mid]->Position.x) ? top : mid;
        right = (left == top) ? mid: top;

        step1 =
            (verts[bottom]->Position.x - verts[left]->Position.x) /
            (verts[bottom]->Position.y - verts[left]->Position.y);

        step2 =
            (verts[bottom]->Position.x - verts[right]->Position.x) /
            (verts[bottom]->Position.y - verts[right]->Position.y);

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

                PixelShader(frag, fragColor);

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



void VertexShader(const VertexInput& input, VertexOutput& output)
{
    output.Position = mul(ShaderConstants.WorldMatrix, float4(input.Position, 1.f));
    output.Position = mul(ShaderConstants.ViewMatrix, output.Position);
    output.Position = mul(ShaderConstants.ProjectionMatrix, output.Position);
    output.Color = input.Color;
}

void PixelShader(const VertexOutput& input, float4& output)
{
    output = float4(input.Color, 1.f);
}


bool LerpFragment(float x, float y, const VertexOutput* inA, const VertexOutput* inB, const VertexOutput* inC, VertexOutput* frag)
{
    float3 a = float3(inA->Position.x, inA->Position.y, 0);
    float3 b = float3(inB->Position.x, inB->Position.y, 0);
    float3 c = float3(inC->Position.x, inC->Position.y, 0);
    float3 ab = b - a;
    float3 ac = c - a;

    // TODO: Check for degenerecy.
    float3 unnormalizedNormal = cross(ab, ac);

    float3 p(x, y, 0);

    // Find barycentric coordinates of P (wA, wB, wC)
    float3 bc = c - b;
    float3 ap = p - a;
    float3 bp = p - b;

    float3 wC = cross(ab, ap);
    if (dot(wC, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float3 wB = cross(ap, ac);
    if (dot(wB, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float3 wA = cross(bc, bp);
    if (dot(wA, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float invLenN = 1.f / unnormalizedNormal.length();

    float xA = wA.length() * invLenN;
    float xB = wB.length() * invLenN;
    float xC = wC.length() * invLenN;

    frag->Position = inA->Position * xA + inB->Position * xB + inC->Position * xC;
    frag->Color = inA->Color * xA + inB->Color * xB + inC->Color * xC;

    return true;
}
