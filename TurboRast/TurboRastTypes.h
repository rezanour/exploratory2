#pragma once

#include "TurboRastMath.h"

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

// Process 4 vertices at a time
typedef void(__vectorcall * pfnSSEVertexShader)(const void* const constantBuffer, const SSEVertexBlock& input, SSEVSOutput& output);

// Process 4 pixels at a time
typedef void(__vectorcall * pfnSSEPixelShader)(const void* const constantBuffer, const SSEVSOutput& input, SSEPSOutput& output);
