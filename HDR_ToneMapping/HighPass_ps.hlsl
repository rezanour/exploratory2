Texture2D SourceTexture;
sampler SourceSampler;

struct Vertex
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer Constants
{
    float YThreshold;
};

static float3 RGBtoYUV(float3 input)
{
    static const float3x3 convMat = float3x3(
        float3(0.2126, 0.7152, 0.0722),
        float3(-0.09991, -0.33609, 0.436),
        float3(0.615, -0.55861, -0.05639)
        );

    return mul(input, convMat);
}

static float3 YUVtoRGB(float3 input)
{
    static const float3x3 convMat = float3x3(
        float3(1, 0, 1.28033),
        float3(1, -0.21482, -0.38059),
        float3(1, 2.12798, 0)
        );

    return mul(input, convMat);
}

float4 main(Vertex input) : SV_TARGET
{
    float4 texColor = SourceTexture.Sample(SourceSampler, input.TexCoord);
    float3 yuv = RGBtoYUV(texColor.rgb);
    clip(texColor.r - YThreshold);
    return float4(texColor.rgb, 1);
}