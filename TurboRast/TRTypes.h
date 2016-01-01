#pragma once

#include "TRMath.h"

class TRVertexBuffer;
class TRTexture2D;

// TODO: Only support one type of vertex currently. Need to find a good way to efficiently handle arbitrary vertices
struct Vertex
{
    float3 Position;    // POSITION
    float3 Color;       // COLOR

    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {
    }
};

struct VertexOut
{
    float4 Position;    // SV_POSITION
    float3 Color;       // COLOR
};

// Block of 4 restructured vertices, optimized for SSE processing together
struct alignas(16) SSEVertexBlock
{
    // POSITION
    float Position_x[4];
    float Position_y[4];
    float Position_z[4];

    // COLOR
    float Color_x[4];
    float Color_y[4];
    float Color_z[4];

    SSEVertexBlock() {}
    SSEVertexBlock(const Vertex* verts, uint64_t numVerts)
    {
        assert(numVerts <= 4);
        const float3* pos = &verts[0].Position;
        const float3* color = &verts[0].Color;
        int stride = sizeof(Vertex) / sizeof(float3);
        for (int i = 0; i < numVerts; ++i, pos += stride, color += stride)
        {
            Position_x[i] = pos->x;
            Position_y[i] = pos->y;
            Position_z[i] = pos->z;
            Color_x[i] = color->x;
            Color_y[i] = color->y;
            Color_z[i] = color->z;
        }
    }
};

struct alignas(16) SSEVSOutput
{
    // SV_POSITION
    float Position_x[4];
    float Position_y[4];
    float Position_z[4];
    float Position_w[4];

    // COLOR
    float Color_x[4];
    float Color_y[4];
    float Color_z[4];
};

struct alignas(16) SSEPSOutput
{
    // SV_TARGET
    float R[4];
    float G[4];
    float B[4];
    float A[4];
};

// 4 float2's packed together
struct alignas(16) vec2
{
    __m128 x;
    __m128 y;
};

// 4 float3's packed together
struct alignas(16) vec3
{
    __m128 x;
    __m128 y;
    __m128 z;
};

// 4 float4's packed together
struct alignas(16) vec4
{
    __m128 x;
    __m128 y;
    __m128 z;
    __m128 w;
};

// 4 barycentric coord results
struct alignas(16) bary_result
{
    __m128 xA;
    __m128 xB;
    __m128 xC;
    __m128 mask;
};

struct alignas(16) vs_input
{
    vec3 Position;
    vec3 Color;
};

struct alignas(16) vs_output
{
    vec4 Position;
    vec3 Color;
};

// Process 4 vertices at a time
typedef vs_output(__vectorcall * pfnSSEVertexShader)(const void* const constantBuffer, const vs_input input);

// Process 4 pixels at a time
typedef vec4(__vectorcall * pfnSSEPixelShader)(const void* const constantBuffer, const vs_output input);

// Process 1 vertices at a time
typedef VertexOut (__vectorcall * pfnVertexShader)(const void* const constantBuffer, const Vertex& input);

// Process 1 pixels at a time
typedef float4 (__vectorcall * pfnPixelShader)(const void* const constantBuffer, const VertexOut& input);

typedef void (*pfnStreamVertexShader)(
    const void* const constantBuffer,
    const void* const input,    // input stream
    void* output,               // output stream
    int64_t vertexCount);
