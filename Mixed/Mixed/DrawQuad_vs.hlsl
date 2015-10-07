cbuffer Constants
{
    int2 Offset;            // in pixels
    uint2 Size;             // in pixels
    float2 InvViewportSize; // in pixels
};

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
    // 2x because viewport goes from -1 -> 1
    float2 scale = Size * InvViewportSize;
    float2 offset = Offset * InvViewportSize;

    // Convert from viewport to normalized 0->1
    float2 pos = input.Position * 0.5f + 0.5f;
    pos.y = 1 - pos.y;

    // Compute location of quad
    pos = pos * scale + offset;

    // Convert back to viewport coords
    pos.y = 1 - pos.y;
    pos = pos * 2 - 1;

    Output output;
    output.Position = float4(pos, 0, 1);
    output.TexCoord = input.TexCoord;

    return output;
}