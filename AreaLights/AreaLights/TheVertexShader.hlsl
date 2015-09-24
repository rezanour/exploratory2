cbuffer VSConstants
{
    float4x4 WorldToView;
    float4x4 Projection;
};

struct Vertex
{
    float3 Position : POSITION;
    float3 Normal   : NORMAL;
};

struct VSOutput
{
    float4 Position         : SV_POSITION;
    float3 WorldPosition    : POSITION;
    float3 WorldNormal      : NORMAL;
};

VSOutput main(Vertex input)
{
    VSOutput output;
    output.Position = mul(Projection, mul(WorldToView, float4(input.Position, 1)));
    // Input is currently already in world space (note lack of objectToWorld matrix)
    output.WorldPosition = input.Position;
    output.WorldNormal = input.Normal;
    return output;
}