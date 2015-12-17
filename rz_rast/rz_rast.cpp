#include "precomp.h"
#include "rz_rast.h"
#include "rz_math.h"

#include <vector>
#include <DirectXMath.h>
using namespace DirectX;

#define USE_SSE
#define USE_FULL_PS

// Programmable stages
struct Constants
{
    matrix4x4 WorldMatrix;
    matrix4x4 ViewMatrix;
    matrix4x4 ProjectionMatrix;
};

struct alignas(16) VertexInput
{
    float3 Position;
    float3 Color;
};

struct alignas(16) VertexOutput
{
    float4 Position; // SV_POSITION
    float3 Color;
};

static void VertexShader(const VertexInput& input, VertexOutput& output);
static void PixelShader(const VertexOutput& input, float4& output);


// Other stuff
struct Vertex
{
    float3 Position;
    float3 Color;

    Vertex(const float3& pos, const float3& color)
        : Position(pos), Color(color)
    {}
};
static const uint32_t Stride = sizeof(Vertex);

// Variables
static Constants ShaderConstants;
static std::vector<Vertex> Vertices;
static std::vector<VertexOutput> VertOutput;
static uint32_t FrameIndex;

static bool LerpFragment(float x, float y, const VertexOutput* a, const VertexOutput* b, const VertexOutput* c, VertexOutput* frag);








//=================================================================================================
// Trying out Larrabee style of rasterizing here....

// edge equation Bx + Cy = 0, where B & C are computed from slope as B = (y1 - y0) and C = -(x1 - x0) or (x0 - x1).
static float2 rz_edge_equation(const float2& v1, const float2& v2)
{
    return float2(v2.y - v1.y, v1.x - v2.x);
}

#ifndef USE_SSE
// function returns negative number of point behind edge. Positive if in front, and 0 if on the edge
static float rz_side_of_edge(const float2& base_vert, const float2& edge_equation, const float2& point)
{
    float2 diff = point - base_vert;
    return dot(diff, edge_equation);
}
#endif

// binning and rasterization

static const int MAX_TRIANGLES_PER_BIN = 512;
static int stride = sizeof(Vertex);
static int position_offset;

struct tile_bin
{
    // Each element is the start vertex position for the triangle. The next two
    // vertices are continguous at the pointer location, spaced by 'stride'
    // The position is offset from top of the vertex struct by 'position_offset'
    const float3* triangles[MAX_TRIANGLES_PER_BIN];
    int num_triangles;
};

static const int target_tile_size = 256; // must break down to 4x4 evenly
static int bin_hsize;
static int bin_vsize;
static int num_hbins;
static int num_vbins;
static tile_bin* bins;

static int render_target_width;
static int render_target_height;
static uint32_t* render_target;
static int render_target_pitch_in_pixels;

static void rz_init_bins()
{
    if (bins != nullptr)
    {
        delete[] bins;
    }

    bins = new tile_bin[num_vbins * num_hbins];
}

static void rz_destroy_bins()
{
    delete[] bins;
    bins = nullptr;
}

static void rz_clear_bins()
{
    for (int r = 0; r < num_vbins; ++r)
        for (int c = 0; c < num_hbins; ++c)
            bins[r * num_hbins + c].num_triangles = 0;
}

static void rz_bin_triangles(const float3* triangles, int num_triangles)
{
    const uint8_t* p = (const uint8_t*)triangles;
    for (int i = 0; i < num_triangles; ++i)
    {
        const float3* first = (const float3*)p;

        float2 p1 = *(const float2*)p;      p += stride;
        float2 p2 = *(const float2*)p;      p += stride;
        float2 p3 = *(const float2*)p;      p += stride;

        // determine overlapped bins by bounding box
        float2 bb_min = min(p1, min(p2, p3));
        float2 bb_max = max(p1, max(p2, p3));

        for (int r = ((int)bb_min.y / bin_vsize); r <= ((int)bb_max.y / bin_vsize); ++r)
        {
            if (r < 0 || r >= num_vbins)
                continue;

            for (int c = ((int)bb_min.x / bin_hsize); c <= ((int)bb_max.x / bin_hsize); ++c)
            {
                if (c < 0 || c >= num_hbins)
                    continue;

                auto& bin = bins[r * num_hbins + c];

                // We should do something intelligent here, like flush the tile
                assert(bin.num_triangles < MAX_TRIANGLES_PER_BIN);

                bin.triangles[bin.num_triangles++] = first;
            }
        }
    }
}

static void rz_process_sub_tile(
    const float3* triangle,
    int top_left_x, int top_left_y, int width, int height, // in pixels
    const float2& p1, const float2& edge1, int off_x1, int off_y1,
    const float2& p2, const float2& edge2, int off_x2, int off_y2,
    const float2& p3, const float2& edge3, int off_x3, int off_y3)
{
    const int hsize = width / 4;
    const int vsize = height / 4;

    if (hsize == 0 || vsize == 0) return;

    float off_h1 = (float)off_x1 * hsize;
    float off_v1 = (float)off_y1 * vsize;
    float off_h2 = (float)off_x2 * hsize;
    float off_v2 = (float)off_y2 * vsize;
    float off_h3 = (float)off_x3 * hsize;
    float off_v3 = (float)off_y3 * vsize;

    if (hsize == 1)
    {
        off_h1 = 0.5f;
        off_v1 = 0.5f;
        off_h2 = 0.5f;
        off_v2 = 0.5f;
        off_h3 = 0.5f;
        off_v3 = 0.5f;
    }

#ifdef USE_SSE // vectorized version. Does the tile row by row using SSE (4 wide)

    __m128 indices = _mm_set_ps(0.f, 1.f, 2.f, 3.f);
    __m128 hsizes = _mm_set1_ps((float)hsize);
    __m128 xoff = _mm_mul_ps(indices, hsizes);
    __m128 base_corner_x = _mm_add_ps(_mm_set1_ps((float)top_left_x), xoff);

    for (int y = top_left_y; y < top_left_y + height; y += vsize)
    {
        if (y >= render_target_height)
            break;

        __m128 base_corner_y = _mm_set1_ps((float)y);

        // trivial reject against edge1

        // side_of_edge:
        // float2 diff = point - base_vert;
        // return dot(diff, edge_equation);

        // float2 diff part
        // break down to all edge1-adjusted x's, then edge1-adjusted y's
        // base_vert is p1
        __m128 adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h1));
        __m128 diffx = _mm_sub_ps(adj, _mm_set1_ps(p1.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v1));
        __m128 diffy = _mm_sub_ps(adj, _mm_set1_ps(p1.y));

        // dot part is broken into muls & adds
        // m1 = diff.x * edge.x
        // m2 = diff.y * edge.y
        // m1 + m2
        __m128 m1 = _mm_mul_ps(diffx, _mm_set1_ps(edge1.x));
        __m128 m2 = _mm_mul_ps(diffy, _mm_set1_ps(edge1.y));
        __m128 dots = _mm_add_ps(m1, m2);

        // if dot is > 0.f, those tiles are rejected.
        __m128 e1mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // Now, repeat for edge 2
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h2));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(p2.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v2));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(p2.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(edge2.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(edge2.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e2mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // And edge3
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h3));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(p3.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v3));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(p3.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(edge3.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(edge3.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e3mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // only elements we keep are the ones that passed all three filters. ie:
        // mask1 | mask2 | mask3 == 0
        __m128 mask = _mm_or_ps(e1mask, _mm_or_ps(e2mask, e3mask));

        float fmasks[4]{};
        _mm_storeu_ps(fmasks, mask);

        if (hsize > 1 && vsize > 1)
        {
            // recurse sub tiles

            for (int i = 0; i < 4; ++i) // TODO: vectorize this too
            {
                if (fmasks[3 - i] == 0.f)
                {
                    rz_process_sub_tile(
                        triangle,
                        top_left_x + (hsize * i), y, hsize, vsize,
                        p1, edge1, off_x1, off_y1,
                        p2, edge2, off_x2, off_y2,
                        p3, edge3, off_x3, off_y3);
                }
            }
        }
        else
        {
            // rasterize the pixels!

            for (int i = 0; i < 4; ++i) // TODO: vectorize this too
            {
                if (fmasks[3 - i] == 0.f)
                {
                    int x = top_left_x + (i * hsize);

                    if (x >= render_target_width)
                        break;

#ifndef USE_FULL_PS // quick test
                    render_target[y * render_target_pitch_in_pixels + x] = 0xFFFF0000; // blue
#else
                    VertexOutput frag;
                    if (!LerpFragment((float)x, (float)y, (const VertexOutput*)triangle, (const VertexOutput*)((const uint8_t*)triangle + stride), (const VertexOutput*)((const uint8_t*)triangle + 2 * stride), &frag))
                    {
                        continue;
                    }

                    float4 fragColor;
                    PixelShader(frag, fragColor);

                    // convert to RGBA
                    render_target[y * render_target_pitch_in_pixels + x] =
                        (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                        (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                        (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                        (uint32_t)((uint8_t)(fragColor.x * 255.f));
#endif // USE_FULL_PS
                }
            }
        }
    }

#else // USE_SSE

    for (int y = top_left_y; y < top_left_y + height; y += vsize)
    {
        if (y >= render_target_height)
            break;

        for (int x = top_left_x; x < top_left_x + width; x += hsize)
        {
            if (x >= render_target_width)
                break;

            // trivial reject against edge1?
            if (rz_side_of_edge(p1, edge1, float2(x + off_h1, y + off_v1)) > 0.f)
            {
                // outside of edge1, early reject
                continue;
            }

            // TODO: trivial accept test? worth it?

            // trivial reject against edge2?
            if (rz_side_of_edge(p2, edge2, float2(x + off_h2, y + off_v2)) > 0.f)
            {
                // outside of edge2, early reject
                continue;
            }

            // TODO: trivial accept test? worth it?

            // trivial reject against edge3?
            if (rz_side_of_edge(p3, edge3, float2(x + off_h3, y + off_v3)) > 0.f)
            {
                // outside of edge3, early reject
                continue;
            }

            if (hsize > 1 && vsize > 1)
            {
                // recurse sub tile
                rz_process_sub_tile(
                    triangle,
                    x, y, hsize, vsize,
                    p1, edge1, off_x1, off_y1,
                    p2, edge2, off_x2, off_y2,
                    p3, edge3, off_x3, off_y3);
            }
            else
            {
                // rasterize the pixel!

#ifndef USE_FULL_PS // quick test
                render_target[y * render_target_pitch_in_pixels + x] = 0xFFFF0000; // blue
#else
                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, (const VertexOutput*)triangle, (const VertexOutput*)((const uint8_t*)triangle + stride), (const VertexOutput*)((const uint8_t*)triangle + 2 * stride), &frag))
                {
                    continue;
                }

                float4 fragColor;
                PixelShader(frag, fragColor);

                // convert to RGBA
                render_target[y * render_target_pitch_in_pixels + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
#endif // USE_FULL_PS
            }
        }
    }
#endif // USE_SSE
}

static void rz_rasterize_tile(int row, int col)
{
    auto& bin = bins[row * num_hbins + col];

    for (int i = 0; i < bin.num_triangles; ++i)
    {
        const uint8_t* p = (const uint8_t*)bin.triangles[i];

        float2 p1 = *(const float2*)p;      p += stride;
        float2 p2 = *(const float2*)p;      p += stride;
        float2 p3 = *(const float2*)p;      p += stride;

        // recursive descent approach (see Larrabee rasterization article)

        float2 edge1 = rz_edge_equation(p1, p2);
        float2 edge2 = rz_edge_equation(p2, p3);
        float2 edge3 = rz_edge_equation(p3, p1);

        // we need to determine 'trivial reject' corner. compute corner offset x & y
        // to add to top left corner to find trivial reject corner for each edge
        int off_x1 = 0, off_x2 = 0, off_x3 = 0, off_y1 = 0, off_y2 = 0, off_y3 = 0;

        if (edge1.x < 0)    off_x1 = 1;
        if (edge2.x < 0)    off_x2 = 1;
        if (edge3.x < 0)    off_x3 = 1;
        if (edge1.y < 0)    off_y1 = 1;
        if (edge2.y < 0)    off_y2 = 1;
        if (edge3.y < 0)    off_y3 = 1;

        rz_process_sub_tile(
            bin.triangles[i],
            col * bin_hsize, row * bin_vsize,
            bin_hsize, bin_vsize,
            p1, edge1, off_x1, off_y1,
            p2, edge2, off_x2, off_y2,
            p3, edge3, off_x3, off_y3);
    }
}

static void rz_rasterize_bins()
{

    for (int r = 0; r < num_vbins; ++r)
        for (int c = 0; c < num_hbins; ++c)
            rz_rasterize_tile(r, c);
}

// End of Larrabee style rasterizing (except below where we call this instead of other rasterization + pixel shader code)
//=================================================================================================

bool RastStartup(uint32_t width, uint32_t height)
{
    // Fill in vertices for triangle
    Vertices.push_back(Vertex(float3(-0.5f, -0.5f, 0.f), float3(0.f, 0.f, 1.f)));
    Vertices.push_back(Vertex(float3(0.f, 0.5f, 0.f), float3(0.f, 1.f, 0.f)));
    Vertices.push_back(Vertex(float3(0.5f, -0.5f, 0.f), float3(1.f, 0.f, 0.f)));

    VertOutput.resize(Vertices.size());

    XMFLOAT4X4 temp;
    XMStoreFloat4x4(&temp, XMMatrixIdentity());
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    XMStoreFloat4x4(&temp, XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -1.f, 1.f), XMVectorSet(0.f, 0.f, 0.f, 1.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)));
    memcpy_s(&ShaderConstants.ViewMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    XMStoreFloat4x4(&temp, XMMatrixPerspectiveFovLH(XMConvertToRadians(90.f), (float)width / height, 0.1f, 100.f));
    memcpy_s(&ShaderConstants.ProjectionMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    render_target_width = (int)width;
    render_target_height = (int)height;
    bin_hsize = target_tile_size;
    bin_vsize = target_tile_size;
    num_hbins = render_target_width / bin_hsize + (render_target_width % bin_hsize ? 1 : 0);
    num_vbins = render_target_height / bin_vsize + (render_target_height % bin_vsize ? 1 : 0);

    rz_init_bins();
    return true;
}

void RastShutdown()
{
    rz_destroy_bins();
}


bool RenderScene(void* const pOutput, uint32_t rowPitch)
{
    render_target_pitch_in_pixels = rowPitch / sizeof(uint32_t);

    memset(pOutput, 0, rowPitch * render_target_height);

#ifdef USE_FULL_PS
    XMFLOAT4X4 transform;
    XMStoreFloat4x4(&transform, XMMatrixRotationY(FrameIndex++ * 0.25f));
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &transform, sizeof(transform));
#endif

    // Serial, simple impl first just to try some ideas. Then will do this right

    uint32_t numVerts = (uint32_t)Vertices.size();
    Vertex* v = Vertices.data();
    VertexOutput* out = VertOutput.data();

    // Vertex Shader Stage
    for (uint32_t i = 0; i < numVerts; ++i, ++v, ++out)
    {
        VertexInput input;
        input.Position = v->Position;
        input.Color = v->Color;
        VertexShader(input, *out);

        // w divide & convert to viewport (pixels)
        out->Position /= out->Position.w;
        out->Position.x = (out->Position.x * 0.5f + 0.5f) * render_target_width;
        out->Position.y = (1.f - (out->Position.y * 0.5f + 0.5f)) * render_target_height;
    }

    out = VertOutput.data();

    // TODO: Clip

    // Rasterize

    render_target = (uint32_t*)pOutput;

    rz_clear_bins();
    rz_bin_triangles((float3*)&out->Position, 1);
    rz_rasterize_bins();

#if 0
    for (uint32_t i = 0; i < numVerts / 3; ++i, out += 3)
    {
        VertexOutput* verts[3] = { out, out + 1, out + 2 };
        uint32_t top = 0, bottom = 0, mid = 0;

        for (uint32_t j = 0; j < 3; ++j)
        {
            if (verts[j]->Position.y > verts[bottom]->Position.y) bottom = j;
            if (verts[j]->Position.y < verts[top]->Position.y) top = j;
        }

        for (uint32_t j = 0; j < 3; ++j)
        {
            if (j != top && j != bottom)
            {
                mid = j;
                break;
            }
        }

        // first, rasterize from top to other
        float ytop = verts[top]->Position.y;
        float ymid = verts[mid]->Position.y;
        float ybottom = verts[bottom]->Position.y;

        uint32_t left = (verts[mid]->Position.x < verts[bottom]->Position.x) ? mid : bottom;
        uint32_t right = (left == mid) ? bottom : mid;

        float step1 =
            (verts[left]->Position.x - verts[top]->Position.x) /
            (verts[left]->Position.y - verts[top]->Position.y);

        float step2 =
            (verts[right]->Position.x - verts[top]->Position.x) /
            (verts[right]->Position.y - verts[top]->Position.y);

        float x1 = verts[top]->Position.x;
        float x2 = x1;

        float4 fragColor;

        for (int y = (int)ytop; y < (int)ymid; ++y)
        {
            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                PixelShader(frag, fragColor);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * pitch + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
            }

            x1 += step1;
            x2 += step2;
        }

        // then, from other to bottom
        left = (verts[top]->Position.x < verts[mid]->Position.x) ? top : mid;
        right = (left == top) ? mid: top;

        step1 =
            (verts[bottom]->Position.x - verts[left]->Position.x) /
            (verts[bottom]->Position.y - verts[left]->Position.y);

        step2 =
            (verts[bottom]->Position.x - verts[right]->Position.x) /
            (verts[bottom]->Position.y - verts[right]->Position.y);

        for (int y = (int)ymid; y < (int)ybottom; ++y)
        {
            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                PixelShader(frag, fragColor);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * pitch + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
            }

            x1 += step1;
            x2 += step2;
        }
    }
#endif

    return true;
}



void VertexShader(const VertexInput& input, VertexOutput& output)
{
    output.Position = mul(ShaderConstants.WorldMatrix, float4(input.Position, 1.f));
    output.Position = mul(ShaderConstants.ViewMatrix, output.Position);
    output.Position = mul(ShaderConstants.ProjectionMatrix, output.Position);
    output.Color = input.Color;
}

#ifdef USE_FULL_PS
void PixelShader(const VertexOutput& input, float4& output)
{
    output = float4(input.Color, 1.f);
}


bool LerpFragment(float x, float y, const VertexOutput* inA, const VertexOutput* inB, const VertexOutput* inC, VertexOutput* frag)
{
    float3 a = float3(inA->Position.x, inA->Position.y, 0);
    float3 b = float3(inB->Position.x, inB->Position.y, 0);
    float3 c = float3(inC->Position.x, inC->Position.y, 0);
    float3 ab = b - a;
    float3 ac = c - a;

    // TODO: Check for degenerecy.
    float3 unnormalizedNormal = cross(ab, ac);

    float3 p(x, y, 0);

    // Find barycentric coordinates of P (wA, wB, wC)
    float3 bc = c - b;
    float3 ap = p - a;
    float3 bp = p - b;

    float3 wC = cross(ab, ap);
    if (dot(wC, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float3 wB = cross(ap, ac);
    if (dot(wB, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float3 wA = cross(bc, bp);
    if (dot(wA, unnormalizedNormal) < 0.f)
    {
        return false;
    }

    float invLenN = 1.f / unnormalizedNormal.length();

    float xA = wA.length() * invLenN;
    float xB = wB.length() * invLenN;
    float xC = wC.length() * invLenN;

    frag->Position = inA->Position * xA + inB->Position * xB + inC->Position * xC;
    frag->Color = inA->Color * xA + inB->Color * xB + inC->Color * xC;

    return true;
}
#endif
