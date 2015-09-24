cbuffer PSConstants
{
    // Test area light
    float4 AreaLightCorners[4];
    float3 AreaLightNormal;
    float3 AreaLightColor;
};

struct PSInput
{
    float4 Position         : SV_POSITION;
    float3 WorldPosition    : POSITION;
    float3 WorldNormal      : NORMAL;
};

float4 main(PSInput input) : SV_TARGET
{
    return float4(AreaLightColor, 1.0f);
}