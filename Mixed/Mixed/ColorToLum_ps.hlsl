Texture2D Image;
SamplerState Sampler;

float main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET
{
    static const float3 conv = float3(0.2126f, 0.7152f, 0.0722f);
    float4 color = Image.Sample(Sampler, TexCoord);
    return dot(color.rgb, conv);
}
