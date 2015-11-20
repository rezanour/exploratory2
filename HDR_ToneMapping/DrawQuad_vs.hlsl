struct Vertex
{
    float2 Position : POSITION;
    float2 TexCoord : TEXCOORD;
};

struct VertexOut
{
    float4 Position : SV_POSITION;
    float2 TexCoord : TEXCOORD;
};

VertexOut main(Vertex input)
{
    VertexOut output;
    output.Position = float4(input.Position, 0, 1);
    output.TexCoord = input.TexCoord;
    return output;
}