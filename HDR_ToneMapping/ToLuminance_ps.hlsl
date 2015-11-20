Texture2D SourceTexture;
sampler SourceSampler;

struct Vertex
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

static float3 RGBtoYUV(float3 input)
{
}

float4 main(Vertex input) : SV_TARGET
{
    return SourceTexture.Sample(SourceSampler, input.TexCoord);
}