struct Input
{
    float2 Position : POSITION;
    float2 TexCoord : TEXCOORD;
};

struct Output
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

Output main(Input input)
{
    Output output;
    output.Position = float4(input.Position, 0, 1);
    output.TexCoord = input.TexCoord;
    return output;
}
