Texture2D Image;
SamplerState Sampler;

float4 main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET
{
    static const float heightScale = 20.f;

    float self = Image.Sample(Sampler, TexCoord).x;
    float left = Image.Sample(Sampler, TexCoord, int2(-1, 0)).x;
    float right = Image.Sample(Sampler, TexCoord, int2(1, 0)).x;
    float up = Image.Sample(Sampler, TexCoord, int2(0, -1)).x;
    float down = Image.Sample(Sampler, TexCoord, int2(0, 1)).x;

    float3 normals[] =
    {
        float3((self - left) * heightScale, 0, -1),
        float3((right - self) * heightScale, 0, 1),
        float3(0, (self - up) * heightScale, -1),
        float3(0, (down - self) * heightScale, 1),
    };

    float3 averageN = 0.25f * (normalize(normals[0]) + normalize(normals[1]) + normalize(normals[2]) + normalize(normals[3]));

    // Convert to [0,1] range
    averageN = averageN * 0.5f + 0.5f;

    return float4(averageN.x, averageN.y, averageN.z, 1);
}
