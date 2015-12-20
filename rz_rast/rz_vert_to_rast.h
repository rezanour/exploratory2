#pragma once

// These functions handle the vertices from app -> rasterizer

#include "rz_math.h"

// application provided vertex
struct Vertex
{
    float3 Position;
    float3 Color;

    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {}
};

static const uint32_t VertexStride = sizeof(Vertex);

// constant buffer available to the vertex shader
struct VSConstants
{
    matrix4x4 WorldMatrix;
    matrix4x4 ViewMatrix;
    matrix4x4 ProjectionMatrix;
};

struct VertexOutput
{
    float4 Position; // SV_POSITION
    float3 Color;
};

// Begins the graphics pipeline by issuing a draw of the provided vertices
void rz_draw(const VSConstants& constants, const Vertex* vertices, uint32_t num_verts);
