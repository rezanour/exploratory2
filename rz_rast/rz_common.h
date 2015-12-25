#pragma once

#include "rz_math.h"
#include <vector>

// constant buffer available to the vertex shader
struct VSConstants
{
    matrix4x4 WorldMatrix;
    matrix4x4 ViewMatrix;
    matrix4x4 ProjectionMatrix;
};

// application provided vertex
struct Vertex
{
    float3 Position;
    float3 Color;

    Vertex() {}
    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {}
};

struct VertexOutput
{
    float4 Position; // SV_POSITION
    float3 Color;
};

//=================================================================================================
// AOSOA (array of structs of array). Each element is a group, or bundle,
// of vertices optimized for concurrent SIMD processing (currently only SSE, later AVX1/2/512)
//=================================================================================================

// SSE version. Designed for doing 4 vertices at once
struct alignas(16) sse_Vertex
{
    float position_x[4];
    float position_y[4];
    float position_z[4];
    float color_x[4];
    float color_y[4];
    float color_z[4];

    sse_Vertex() {}
    sse_Vertex(const Vertex* verts, int numVerts)
    {
        assert(numVerts <= 4);
        const float3* pos = &verts[0].Position;
        const float3* color = &verts[0].Color;
        int stride = sizeof(Vertex) / sizeof(float3);
        for (int i = 0; i < numVerts; ++i, pos += stride, color += stride)
        {
            position_x[i] = pos->x;
            position_y[i] = pos->y;
            position_z[i] = pos->z;
            color_x[i] = color->x;
            color_y[i] = color->y;
            color_z[i] = color->z;
        }
    }
};

struct alignas(16) sse_VSOutput
{
    float position_x[4];
    float position_y[4];
    float position_z[4];
    float position_w[4];
    float color_x[4];
    float color_y[4];
    float color_z[4];
};

//=================================================================================================
// Bins used for tiled rasterization
//=================================================================================================

struct alignas(16) Triangle
{
    float2 p1, p2, p3;  // vertices of the triangle (assumed to be in clockwise order)
    float2 e1, e2, e3;  // edge equation for each edge
    float2 off1, off2, off3;    // offset direction from top-left corner of cell to get trivial reject for edge
    int iFirstVertex;   // Index into vertex stream for first vertex,
                        // followed sequentially by next two.
};

struct TileBin
{
    int row, column;
    std::vector<Triangle> triangles;
};

