#pragma once

#include "rz_math.h"

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

    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {}
};

struct VertexOutput
{
    float4 Position; // SV_POSITION
    float3 Color;
};

