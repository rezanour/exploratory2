#include "precomp.h"
#include "rz_vert_to_rast.h"

#include <vector>

// 4 vectors across x, y, z, w, multiplied by m, returned as a set of 4 vectors again
struct alignas(16) sse_mul_result
{
    __m128 x;
    __m128 y;
    __m128 z;
    __m128 w;
};

static inline sse_mul_result __vectorcall sse_mul(const matrix4x4& m, const __m128& x, const __m128& y, const __m128& z, const __m128& w)
{
    sse_mul_result result;

    // expanded multiply of all 4 positions by matrix
    // dot(float4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]), v),
    // dot(float4(m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]), v),
    // dot(float4(m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]), v),
    // dot(float4(m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]), v));
    // Resulting 4 dots are the components of the result vector

    __m128 mx = _mm_set1_ps(m.m[0][0]);
    __m128 my = _mm_set1_ps(m.m[1][0]);
    __m128 mz = _mm_set1_ps(m.m[2][0]);
    __m128 mw = _mm_set1_ps(m.m[3][0]);
    result.x = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(m.m[0][1]);
    my = _mm_set1_ps(m.m[1][1]);
    mz = _mm_set1_ps(m.m[2][1]);
    mw = _mm_set1_ps(m.m[3][1]);
    result.y = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(m.m[0][2]);
    my = _mm_set1_ps(m.m[1][2]);
    mz = _mm_set1_ps(m.m[2][2]);
    mw = _mm_set1_ps(m.m[3][2]);
    result.z = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(m.m[0][3]);
    my = _mm_set1_ps(m.m[1][3]);
    mz = _mm_set1_ps(m.m[2][3]);
    mw = _mm_set1_ps(m.m[3][3]);
    result.w = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));

    return result;
}

// Non vectorized version. Just takes one input and makes one output
static void VertexShader(const Vertex* inputs, VertexOutput* outputs, const VSConstants& constants)
{
    const Vertex& input = inputs[0];
    VertexOutput& output = outputs[0];
    output.Position = float4(input.Position, 1.f);
    output.Position = mul(constants.WorldMatrix, output.Position);
    output.Position = mul(constants.ViewMatrix, output.Position);
    output.Position = mul(constants.ProjectionMatrix, output.Position);
    output.Color = input.Color;
}

// Vectorized version. Points to a stream of inputs and outputs. Does 4 at a time (must have 4 inputs available)
static void sse_VertexShader(const Vertex* inputs, VertexOutput* outputs, const VSConstants& constants)
{
    // Multiply positions out by matrices
    __m128 posx = _mm_set_ps(inputs[0].Position.x, inputs[1].Position.x, inputs[2].Position.x, inputs[3].Position.x);
    __m128 posy = _mm_set_ps(inputs[0].Position.y, inputs[1].Position.y, inputs[2].Position.y, inputs[3].Position.y);
    __m128 posz = _mm_set_ps(inputs[0].Position.z, inputs[1].Position.z, inputs[2].Position.z, inputs[3].Position.z);
    __m128 posw = _mm_set1_ps(1.f);

    sse_mul_result result = sse_mul(constants.WorldMatrix, posx, posy, posz, posw);
    result = sse_mul(constants.ViewMatrix, result.x, result.y, result.z, result.w);
    result = sse_mul(constants.ProjectionMatrix, result.x, result.y, result.z, result.w);

    // TODO: Consider outputing in aligned vectorized form for the rest of pipeline
    float xs[4], ys[4], zs[4], ws[4];
    _mm_storeu_ps(xs, result.x);
    _mm_storeu_ps(ys, result.y);
    _mm_storeu_ps(zs, result.z);
    _mm_storeu_ps(ws, result.w);
    for (int i = 0; i < 4; ++i)
    {
        outputs[i].Position.x = xs[i];
        outputs[i].Position.y = ys[i];
        outputs[i].Position.z = zs[i];
        outputs[i].Position.w = ws[i];
        outputs[i].Color = inputs[i].Color;
    }
}

static std::vector<VertexOutput> outputs;

#include <Windows.h>

void rz_draw(const VSConstants& constants, const Vertex* vertices, uint32_t num_verts)
{
    if (outputs.size() < num_verts)
    {
        outputs.resize(num_verts);
    }

    LARGE_INTEGER start, stop, freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    uint32_t i = 0;
    const Vertex* v = vertices;
    VertexOutput* o = outputs.data();

#ifdef USE_SSE
    for (; i + 4 <= num_verts; i += 4, o += 4, v += 4)
    {
        sse_VertexShader(v, o, constants);
    }
    // any remainder that didn't fit into groups of 4
    for (; i < num_verts; ++i, ++o, ++v)
    {
        VertexShader(v, o, constants);
    }
#else
    for (; i < num_verts; ++i, ++o, ++v)
    {
        VertexShader(v, o, constants);
    }
#endif
    QueryPerformanceCounter(&stop);

    wchar_t message[100];
    swprintf_s(message, L"vert_to_rast: %3.3fms\n", 1000.0 * ((stop.QuadPart - start.QuadPart) / (double)freq.QuadPart));
    OutputDebugString(message);
}
