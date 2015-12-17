// Vertex Shader, written to be compiled by ISPC

inline void mul(const float m[4][4], const float v[4], float result[4])
{
    result[0] = m[0][0] * v[0] + v[1][0] * v[1] + v[2][0] * v[2] + v[3][0] * v[3];
    result[0] = m[0][1] * v[0] + v[1][1] * v[1] + v[2][1] * v[2] + v[3][1] * v[3];
    result[0] = m[0][2] * v[0] + v[1][2] * v[1] + v[2][2] * v[2] + v[3][2] * v[3];
    result[0] = m[0][3] * v[0] + v[1][3] * v[1] + v[2][3] * v[2] + v[3][3] * v[3];
    return result;
}

struct Constants
{
    float WorldMatrix[4][4];
    float ViewMatrix[4][4];
    float ProjectionMatrix[4][4];
};

struct VertexInput
{
    float Position[4];
    float Color[4];
};

struct VertexOutput
{
    float Position[4]; // SV_POSITION
    float Color[4];
};

export void vertex_shader(
    uniform struct VertexInput input[],
    uniform struct VertexOutput output[],
    uniform int count,
    const uniform struct Constants& constants)
{
    foreach (index = 0 ... count )
    {
        VertexInput v = input[index];

        float wpos[] = { v.Position.x, v.Position.y, v.Position.z, 1.f };
        float wpos2[4];
        mul(constants.WorldMatrix, wpos, wpos2);
        mul(constants.ViewMatrix, wpos2, wpos);
        mul(constants.ProjectionMatrix, wpos, wpos2);
        output[index].Position = wpos2;
        output[index].Color = v.Color;
    }
}