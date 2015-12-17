// Vertex Shader, written to be compiled by ISPC

struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

struct matrix4x4
{
    float m00, m01, m02, m03,
          m10, m11, m12, m13,
          m20, m21, m22, m23,
          m30, m31, m32, m33;
};

inline struct float4 mul(const uniform struct matrix4x4& m, varying struct float4& v)
{
    struct float4 result;
    result.x = m.m00 * v.x + m.m10 * v.y + m.m20 * v.z + m.m30 * v.w;
    result.y = m.m01 * v.x + m.m11 * v.y + m.m21 * v.z + m.m31 * v.w;
    result.z = m.m02 * v.x + m.m12 * v.y + m.m22 * v.z + m.m32 * v.w;
    result.w = m.m03 * v.x + m.m13 * v.y + m.m23 * v.z + m.m33 * v.w;
    return result;
}

struct Constants
{
    struct matrix4x4 WorldMatrix;
    struct matrix4x4 ViewMatrix;
    struct matrix4x4 ProjectionMatrix;
};

struct VertexInput
{
    struct float3 Position;
    struct float3 Color;
};

struct VertexOutput
{
    struct float4 Position; // SV_POSITION
    struct float3 Color;
};

export void vertex_shader(
    uniform struct VertexInput input[],
    uniform struct VertexOutput output[],
    uniform int count,
    const uniform struct Constants& constants)
{
    foreach (index = 0 ... count )
    {
        VertexInput v = input[index];

        struct float4 wpos = { v.Position.x, v.Position.y, v.Position.z, 1.f };
        struct float4 wpos2 = mul(constants.WorldMatrix, wpos);
        wpos = mul(constants.ViewMatrix, wpos2);
        wpos2 = mul(constants.ProjectionMatrix, wpos);

        VertexOutput out = { wpos2, v.Color };
        output[index] = out;
    }
}