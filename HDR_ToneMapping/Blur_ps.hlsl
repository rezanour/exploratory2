Texture2D Image;
SamplerState Sampler;

#if 1
// Kernel generated with scale of 0.9f. 3 pixels back to 3 pixels forward
static const float GaussianKernel[7] =
{
    0.00171372085f,
    0.0375279821f,
    0.239113614f,
    0.443289369f,
    0.239113614f,
    0.0375279821f,
    0.00171372085f,
};
#else
// Kernel generated with scale of 2.f
static const float GaussianKernel[7] =
{
    0.0701593310f,
    0.131074890f,
    0.190712824f,
    0.216105953f,
    0.190712824f,
    0.131074890f,
    0.0701593310f,
};

#endif

cbuffer Constants
{
    int Direction;  // 0 = horiz, 1 = vert
};

float4 main(float4 Position : SV_POSITION, float2 TexCoord : TEXCOORD) : SV_TARGET
{
    float4 samples[7];
    if (Direction == 0)
    {
        [unroll]
        for (int i = 0; i < 7; ++i)
        {
            samples[i] = Image.Sample(Sampler, TexCoord, int2(i - 3, 0));
        }
    }
    else
    {
        [unroll]
        for (int i = 0; i < 7; ++i)
        {
            samples[i] = Image.Sample(Sampler, TexCoord, int2(0, i - 3));
        }
    }

    float4 final = float4(0, 0, 0, 0);

    for (int i = 0; i < 7; ++i)
    {
        final += samples[i] * GaussianKernel[i];
    }

    return float4(final.rgb, 1);
}
