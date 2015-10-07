Texture2D Image;
SamplerState Sampler;

float4 main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET
{
    return Image.Sample(Sampler, TexCoord);
}