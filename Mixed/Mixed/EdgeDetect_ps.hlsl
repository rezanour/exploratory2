Texture2D Image;
SamplerState Sampler;

float main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET
{
    float4 self = Image.Sample(Sampler, TexCoord);
    float4 right = Image.Sample(Sampler, TexCoord, int2(1, 0));
    float4 down = Image.Sample(Sampler, TexCoord, int2(0, 1));

    float3 dist1 = right.rgb - self.rgb;
    float3 dist2 = down.rgb - self.rgb;

    if (length(dist1) > 0.1f || length(dist2) > 0.1f)
    {
        return 1.f;
    }

    return 0.f;
}