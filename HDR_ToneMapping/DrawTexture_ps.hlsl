Texture2D SourceTexture : register(t0);
Texture2D HighPassTexture : register(t1);
Texture2D FilmLut : register(t2);
sampler SourceSampler;

struct Vertex
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

cbuffer Constants
{
    // TONE_MAPPING_OPERATORs
    // 0 = None, simply display texture as-is
    // 1 = Hard coded fixed exposure
    // 2 = Reinhard RGB
    // 3 = Reinhard Y only
    // 4 = Haarm-Peter Dulker (Filmic)
    uint Operator;
    float Exposure;
    uint PerformGamma;
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
    float3 highPass = HighPassTexture.SampleLevel(SourceSampler, input.TexCoord, 8).xyz;

    highPass *= 3;

    if (Operator == 1)      // Linear, hard coded exposure
    {
        texColor.rgb *= Exposure;
    }
    else if (Operator ==  2) // Reinhard RGB
    {
        texColor.rgb *= Exposure;
        texColor.rgb = texColor.rgb / (1 + texColor.rgb);
    }
    else if (Operator == 3) // Reinhard Y Only
    {
        texColor.rgb *= Exposure;
        texColor.rgb = RGBtoYUV(texColor.rgb);
        texColor.r = texColor.r / (1 + texColor.r);
        texColor.rgb = YUVtoRGB(texColor.rgb);
    }
    else if (Operator == 4) // Haarm-Peter Dulker's Curve (Filmic)
    {
        texColor.rgb *= Exposure;

        float3 ld = 0.002;
        float linReference = 0.18;
        float logReference = 444;
        float logGamma = 0.45;
        float3 LogColor = (log10(0.4*texColor.rgb / linReference) / ld*logGamma + logReference) / 1023.f;
        LogColor = saturate(LogColor);

        float FilmLutWidth = 256;
        float Padding = 0.5 / FilmLutWidth;

        // Apply response and color grading for target display
        float3 finalColor;
        finalColor.r = FilmLut.Sample(SourceSampler, float2(lerp(Padding, 1 - Padding, LogColor.r), 0.5)).x;
        finalColor.g = FilmLut.Sample(SourceSampler, float2(lerp(Padding, 1 - Padding, LogColor.g), 0.5)).x;
        finalColor.b = FilmLut.Sample(SourceSampler, float2(lerp(Padding, 1 - Padding, LogColor.b), 0.5)).x;
        texColor.rgb = finalColor;
    }

    float3 result = texColor.rgb + highPass;
    if (Operator != 4 && PerformGamma)
    {
        result = pow(result, 1 / 2.2);    // Gamma
    }

    return float4(result, 1);
}