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
    float3 accumFactor = float3(0, 0, 0);

    static const uint gridSize = 8;
    static const float invGridSize = 1.f / (gridSize - 1);
    for (int i = 0; i < gridSize; ++i)
    {
        for (int j = 0; j < gridSize; ++j)
        {
            float3 P = lerp(
                lerp(AreaLightCorners[0].xyz, AreaLightCorners[1].xyz, i * invGridSize),
                lerp(AreaLightCorners[3].xyz, AreaLightCorners[2].xyz, i * invGridSize),
                j * invGridSize);
            float3 L = P - input.WorldPosition;
            float dist = length(L) - 1;
            L = normalize(L);
            float nDotL1 = saturate(dot(L, input.WorldNormal));
            float nDotL2 = saturate(dot(-L, AreaLightNormal));
            // TODO: use dist to attenuate
            accumFactor += nDotL1 * nDotL2 * rcp(dist);
        }
    }

    static const float3 baseColor = float3(0.f, 0.f, 1.f);
    return float4(baseColor * AreaLightColor * accumFactor * 0.075f, 1.0f);
}