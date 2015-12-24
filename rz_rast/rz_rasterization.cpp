#include "precomp.h"
#include "rz_common.h"
#include "rz_vertex_pipeline.h"
#include "rz_rasterization.h"

#include <vector>
#include <DirectXMath.h>
using namespace DirectX;

#ifdef USE_SSE_LERP_AND_PS
struct ps_result
{
    __m128 r;
    __m128 g;
    __m128 b;
    __m128 a;
};
static ps_result PixelShader(const __m128& posx, const __m128& posy, const __m128& posz, const __m128& posw, const __m128& colx, const __m128& coly, const __m128& colz);
#else
static void PixelShader(const VertexOutput& input, float4& output);
#endif


// Other stuff

// Variables
static VSConstants ShaderConstants;
static std::vector<Vertex> Vertices;
static std::vector<VertexOutput> VertOutput;
static uint32_t FrameIndex;

#ifdef USE_SSE_LERP_AND_PS
// lerps 4 pixels at a time, writing out 4 lerped frag values
static void LerpFragment(const __m128& x, const __m128& y, __m128& mask, const VertexOutput* first, __m128& posx, __m128& posy, __m128& posz, __m128& posw, __m128& colx, __m128& coly, __m128& colz);
#else
static bool LerpFragment(float x, float y, const VertexOutput* a, const VertexOutput* b, const VertexOutput* c, VertexOutput* frag);
#endif


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

static const int MAX_TRIANGLES_PER_BIN = 65536;
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

static void rz_rasterize_tile(int row, int col);

#ifdef USE_MT_TILES

#define NOMINMAX
#include <Windows.h>
HANDLE signal_work;
HANDLE signal_shutdown;
HANDLE* threads;
HANDLE* signal_done;
int num_threads;
long current_tile;
long num_tiles;

static DWORD CALLBACK rasterizer_thread(PVOID context)
{
    HANDLE my_signal_done = (HANDLE)context;

    HANDLE signals[]{ signal_shutdown, signal_work };

    DWORD result = WaitForMultipleObjects(_countof(signals), signals, FALSE, INFINITE);
    while (result != WAIT_OBJECT_0)
    {
        long my_tile = InterlockedIncrement(&current_tile);
        while (my_tile < num_tiles)
        {
            rz_rasterize_tile(my_tile / num_hbins, my_tile % num_hbins);
            my_tile = InterlockedIncrement(&current_tile);
        }

        SetEvent(my_signal_done);

        result = WaitForMultipleObjects(_countof(signals), signals, FALSE, INFINITE);
    }

    return 0;
}

#endif



#ifdef USE_LRB

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
#ifdef USE_SSE_LERP_AND_PS
            __m128 posx, posy, posz, posw, colx, coly, colz;
            __m128 fragMask;
            LerpFragment(base_corner_x, base_corner_y, fragMask, (const VertexOutput*)triangle, posx, posy, posz, posw, colx, coly, colz);
            ps_result pixels = PixelShader(posx, posy, posz, posw, colx, coly, colz);

            float fFragMask[4];
            _mm_storeu_ps(fFragMask, fragMask);

            // convert to RGBA
            __m128 scale = _mm_set1_ps(255.f);
            __m128 r = _mm_mul_ps(pixels.r, scale);
            __m128 g = _mm_mul_ps(pixels.g, scale);
            __m128 b = _mm_mul_ps(pixels.b, scale);
            __m128 a = _mm_mul_ps(pixels.a, scale);

            float fr[4], fg[4], fb[4], fa[4];
            _mm_storeu_ps(fr, r);
            _mm_storeu_ps(fg, g);
            _mm_storeu_ps(fb, b);
            _mm_storeu_ps(fa, a);

            for (int i = 0; i < 4; ++i)
            {
                int maskIndex = 3 - i;
                if (fmasks[maskIndex] == 0.f && fFragMask[maskIndex] == 0.f)
                {
                    // can vectorize this comparison
                    int x = top_left_x + (i * hsize);
                    if (x >= render_target_width)
                        break;

                    // convert to RGBA
                    render_target[y * render_target_pitch_in_pixels + x] =
                        (uint32_t)((uint8_t)fa[maskIndex]) << 24 |
                        (uint32_t)((uint8_t)fb[maskIndex]) << 16 |
                        (uint32_t)((uint8_t)fg[maskIndex]) << 8 |
                        (uint32_t)((uint8_t)fr[maskIndex]);
                }
            }
#else
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
#endif // USE_SSE_LERP_AND_PS
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

void rz_rasterize_tile(int row, int col)
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

#ifndef USE_MT_TILES
static void rz_rasterize_bins()
{
    for (int r = 0; r < num_vbins; ++r)
        for (int c = 0; c < num_hbins; ++c)
            rz_rasterize_tile(r, c);
}
#endif

// End of Larrabee style rasterizing (except below where we call this instead of other rasterization + pixel shader code)
//=================================================================================================

#endif // USE_LRB

bool RastStartup(uint32_t width, uint32_t height)
{
    // Fill in vertices for triangle
#ifdef RENDER_MANY
    for (float z = 5.f; z >= -5.f; z -= 1.f)
    {
        for (float y = 5.f; y >= -5.f; y -= 0.25f)
        {
            for (float x = -5.f; x < 5.f; x += 0.25f)
            {
                Vertices.push_back(Vertex(float3(x - 0.125f, y - 0.125f, z), float3(0.f, 0.f, 1.f)));
                Vertices.push_back(Vertex(float3(x + 0.f, y + 0.125f, z), float3(0.f, 1.f, 0.f)));
                Vertices.push_back(Vertex(float3(x + 0.125f, y - 0.125f, z), float3(1.f, 0.f, 0.f)));
            }
        }
    }
#else
    Vertices.push_back(Vertex(float3(-0.5f, -0.5f, 0.f), float3(0.f, 0.f, 1.f)));
    Vertices.push_back(Vertex(float3(0.f, 0.5f, 0.f), float3(0.f, 1.f, 0.f)));
    Vertices.push_back(Vertex(float3(0.5f, -0.5f, 0.f), float3(1.f, 0.f, 0.f)));
#endif

    VertOutput.resize(Vertices.size());

    XMFLOAT4X4 temp;
    XMStoreFloat4x4(&temp, XMMatrixIdentity());
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

#ifdef RENDER_MANY
    XMStoreFloat4x4(&temp, XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -10.f, 1.f), XMVectorSet(0.f, 0.f, 0.f, 1.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)));
#else
    XMStoreFloat4x4(&temp, XMMatrixLookAtLH(XMVectorSet(0.f, 0.f, -1.f, 1.f), XMVectorSet(0.f, 0.f, 0.f, 1.f), XMVectorSet(0.f, 1.f, 0.f, 0.f)));
#endif
    memcpy_s(&ShaderConstants.ViewMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    XMStoreFloat4x4(&temp, XMMatrixPerspectiveFovLH(XMConvertToRadians(60.f), (float)width / height, 0.1f, 100.f));
    memcpy_s(&ShaderConstants.ProjectionMatrix, sizeof(matrix4x4), &temp, sizeof(XMFLOAT4X4));

    render_target_width = (int)width;
    render_target_height = (int)height;
    bin_hsize = target_tile_size;
    bin_vsize = target_tile_size;
    num_hbins = render_target_width / bin_hsize + (render_target_width % bin_hsize ? 1 : 0);
    num_vbins = render_target_height / bin_vsize + (render_target_height % bin_vsize ? 1 : 0);

#ifdef USE_LRB
    rz_init_bins();

#ifdef USE_MT_TILES

    num_tiles = num_hbins * num_vbins;

    signal_work = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    signal_shutdown = CreateEvent(nullptr, TRUE, FALSE, nullptr);

    num_threads = 8;
    threads = new HANDLE[num_threads];
    signal_done = new HANDLE[num_threads];
    for (int i = 0; i < num_threads; ++i)
    {
        signal_done[i] = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        threads[i] = CreateThread(nullptr, 0, rasterizer_thread, (PVOID)signal_done[i], 0, nullptr);
    }

#endif

#endif // USE_LRB
    return true;
}

void RastShutdown()
{
#ifdef USE_LRB
#ifdef USE_MT_TILES

    SetEvent(signal_shutdown);
    WaitForMultipleObjects(num_threads, threads, TRUE, INFINITE);

    for (int i = 0; i < num_threads; ++i)
    {
        CloseHandle(threads[i]);
        CloseHandle(signal_done[i]);
    }
    delete[] threads;
    delete[] signal_done;

    CloseHandle(signal_work);
    CloseHandle(signal_shutdown);

#endif

    rz_destroy_bins();
#endif // USE_LRB
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


bool RenderScene(void* const pOutput, uint32_t rowPitch)
{
    render_target_pitch_in_pixels = rowPitch / sizeof(uint32_t);

    memset(pOutput, 0, rowPitch * render_target_height);

#ifdef ENABLE_ANIMATION
    XMFLOAT4X4 transform;

    static float angle = -0.5f;
    static float dir = -1.f;

#ifdef RENDER_MANY
    if (FrameIndex++ % 100 == 0) dir *= -1;
#else
    if (FrameIndex++ % 150 == 0) dir *= -1;
#endif
    angle += dir * 0.0125f;
    XMStoreFloat4x4(&transform, XMMatrixRotationY(angle));
    memcpy_s(&ShaderConstants.WorldMatrix, sizeof(matrix4x4), &transform, sizeof(transform));
#endif

    extern void rz_SetRenderTarget(uint32_t* const pRenderTarget, int width, int height, int pitchPixels);
    extern void rz_Draw(const VSConstants& constants, const Vertex* vertices, int numVerts);

    rz_SetRenderTarget((uint32_t*)pOutput, render_target_width, render_target_height, rowPitch / sizeof(uint32_t));
    rz_Draw(ShaderConstants, Vertices.data(), (int)Vertices.size());

#if 0
    // Serial, simple impl first just to try some ideas. Then will do this right

    uint32_t numVerts = (uint32_t)Vertices.size();
    Vertex* v = Vertices.data();
    VertexOutput* out = VertOutput.data();

    // Vertex Shader Stage
    for (uint32_t i = 0; i < numVerts; ++i, ++v, ++out)
    {
        VertexShader(v, out, ShaderConstants);

        // w divide & convert to viewport (pixels)
        out->Position /= out->Position.w;
        out->Position.x = (out->Position.x * 0.5f + 0.5f) * render_target_width;
        out->Position.y = (1.f - (out->Position.y * 0.5f + 0.5f)) * render_target_height;
    }

    out = VertOutput.data();

    // TODO: Clip

    // Rasterize

    render_target = (uint32_t*)pOutput;

#ifdef USE_LRB
    rz_clear_bins();
    rz_bin_triangles((float3*)&out->Position, numVerts / 3);

#ifdef USE_MT_TILES
    InterlockedExchange(&current_tile, -1);  // Looks weird huh? That's because InterlockedIncrement returns the POST-incremented value above. Yes, really. WTF
    SetEvent(signal_work);

    while (InterlockedCompareExchange(&current_tile, 0, 0) < 0)
    {
        YieldProcessor();
    }

    ResetEvent(signal_work);
    WaitForMultipleObjects(num_threads, signal_done, TRUE, INFINITE);
#else
    rz_rasterize_bins();
#endif

#else // USE_LRB

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
            if (y < 0 || y >= render_target_height) continue;

            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                if (x < 0 || x >= render_target_width) continue;

                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                PixelShader(frag, fragColor);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * render_target_pitch_in_pixels + x] =
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
            if (y < 0 || y >= render_target_height) continue;

            // rasterize span from (x1,y) to (x2,y)
            for (int x = (int)x1; x < (int)x2; ++x)
            {
                if (x < 0 || x >= render_target_width) continue;

                VertexOutput frag;
                if (!LerpFragment((float)x, (float)y, out, out + 1, out + 2, &frag))
                {
                    continue;
                }

                PixelShader(frag, fragColor);

                // convert to RGBA
                ((uint32_t* const)pOutput)[y * render_target_pitch_in_pixels + x] =
                    (uint32_t)((uint8_t)(fragColor.w * 255.f)) << 24 |
                    (uint32_t)((uint8_t)(fragColor.z * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(fragColor.y * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(fragColor.x * 255.f));
            }

            x1 += step1;
            x2 += step2;
        }
    }
#endif  // USE_LRB
#endif

    return true;
}

#ifdef USE_FULL_PS

void PixelShader(const VertexOutput& input, float4& output)
{
    output = float4(input.Color, 1.f);
}


#ifdef USE_SSE_LERP_AND_PS

ps_result PixelShader(const __m128& posx, const __m128& posy, const __m128& posz, const __m128& posw, const __m128& colx, const __m128& coly, const __m128& colz)
{
    ps_result result;
    result.r = colx;
    result.g = coly;
    result.b = colz;
    result.a = _mm_set1_ps(1.f);
    return result;
}



// lerps 4 pixels at a time, writing out 4 lerped frag values
void LerpFragment(const __m128& x, const __m128& y, __m128& mask, const VertexOutput* first, __m128& posx, __m128& posy, __m128& posz, __m128& posw, __m128& colx, __m128& coly, __m128& colz)
{
    const VertexOutput* inA = first;
    const VertexOutput* inB = (const VertexOutput*)((const uint8_t*)inA + stride);
    const VertexOutput* inC = (const VertexOutput*)((const uint8_t*)inB + stride);

    __m128 ax = _mm_set1_ps(inA->Position.x);
    __m128 ay = _mm_set1_ps(inA->Position.y);
    __m128 bx = _mm_set1_ps(inB->Position.x);
    __m128 by = _mm_set1_ps(inB->Position.y);
    __m128 cx = _mm_set1_ps(inC->Position.x);
    __m128 cy = _mm_set1_ps(inC->Position.y);

    __m128 abx = _mm_sub_ps(bx, ax);
    __m128 aby = _mm_sub_ps(by, ay);
    __m128 acx = _mm_sub_ps(cx, ax);
    __m128 acy = _mm_sub_ps(cy, ay);

    // Find barycentric coordinates of P (wA, wB, wC)
    __m128 bcx = _mm_sub_ps(cx, bx);
    __m128 bcy = _mm_sub_ps(cy, by);
    __m128 apx = _mm_sub_ps(x, ax);
    __m128 apy = _mm_sub_ps(y, ay);
    __m128 bpx = _mm_sub_ps(x, bx);
    __m128 bpy = _mm_sub_ps(y, by);

    // float3 wC = cross(ab, ap);
    // expand out to:
    //    wC.x = ab.y * ap.z - ap.y * ab.z;
    //    wC.y = ab.z * ap.x - ap.z * ab.x;
    //    wC.z = ab.x * ap.y - ap.x * ab.y;
    // since we are doing in screen space, z is always 0 so simplify:
    //    wC.x = 0
    //    wC.y = 0
    //    wC.z = ab.x * ap.y - ap.x * ab.y
    // or, simply:
    //    wC = abx * apy - apx * aby;
    __m128 wC = _mm_sub_ps(_mm_mul_ps(abx, apy), _mm_mul_ps(apx, aby));
    __m128 mask1 = _mm_cmplt_ps(wC, _mm_setzero_ps());

    // Use same reduction for wB & wA
    __m128 wB = _mm_sub_ps(_mm_mul_ps(apx, acy), _mm_mul_ps(acx, apy));
    __m128 mask2 = _mm_cmplt_ps(wB, _mm_setzero_ps());

    __m128 wA = _mm_sub_ps(_mm_mul_ps(bcx, bpy), _mm_mul_ps(bpx, bcy));
    __m128 mask3 = _mm_cmplt_ps(wA, _mm_setzero_ps());

    mask = _mm_or_ps(mask1, _mm_or_ps(mask2, mask3));

    // Use a similar reduction for cross of ab x ac (to find unnormalized normal)
    __m128 norm = _mm_sub_ps(_mm_mul_ps(abx, acy), _mm_mul_ps(acx, aby));
    norm = _mm_rcp_ps(norm);

    // to find length of this cross product, which already know is purely in the z
    // direction, is just the length of the z component, which is the exactly the single
    // channel norm we computed above. Similar logic is used for lengths of each of
    // the weights, since they are all single channel vectors, the one channel is exactly
    // the length.

    __m128 xA = _mm_mul_ps(wA, norm);
    __m128 xB = _mm_mul_ps(wB, norm);
    __m128 xC = _mm_mul_ps(wC, norm);

    // Interpolate all the attributes for these 4 pixels
    posx = _mm_add_ps(_mm_mul_ps(ax, xA), _mm_add_ps(_mm_mul_ps(bx, xB), _mm_mul_ps(cx, xC)));
    posy = _mm_add_ps(_mm_mul_ps(ay, xA), _mm_add_ps(_mm_mul_ps(by, xB), _mm_mul_ps(cy, xC)));
    posz = _mm_setzero_ps();
    posw = _mm_set1_ps(1.f);

    colx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inA->Color.x), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inB->Color.x), xB), _mm_mul_ps(_mm_set1_ps(inC->Color.x), xC)));
    coly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inA->Color.y), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inB->Color.y), xB), _mm_mul_ps(_mm_set1_ps(inC->Color.y), xC)));
    colz = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inA->Color.z), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(inB->Color.z), xB), _mm_mul_ps(_mm_set1_ps(inC->Color.z), xC)));
}

#else

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

    frag->Position.x = inA->Position.x * xA + inB->Position.x * xB + inC->Position.x * xC;
    frag->Position.y = inA->Position.y * xA + inB->Position.y * xB + inC->Position.y * xC;
    frag->Position.z = inA->Position.z * xA + inB->Position.z * xB + inC->Position.z * xC;
    frag->Color.x = inA->Color.x * xA + inB->Color.x * xB + inC->Color.x * xC;
    frag->Color.y = inA->Color.y * xA + inB->Color.y * xB + inC->Color.y * xC;
    frag->Color.z = inA->Color.z * xA + inB->Color.z * xB + inC->Color.z * xC;

    return true;
}
#endif // USE_SSE_LERP_AND_PS

#endif // USE_FULL_PS
