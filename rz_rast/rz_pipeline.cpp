#include "precomp.h"
#include "rz_common.h"
#include "rz_rast.h"
#include <vector>
#include <algorithm>

static const int TILE_SIZE = 256; // top level tile size. must break down to 4x4 evenly

// RenderTarget data
static uint32_t* RenderTarget;
static int RTWidth;
static int RTHeight;
static int RTPitchPixels;

// Input data
static int NumTriangles;
static std::vector<sse_Vertex> VSInputStream;
static std::vector<sse_VSOutput> VSOutputStream;

// Binning data
static int NumHorizBins;
static int NumVertBins;
static int NumTotalBins;    // NumHorizBins * NumVertBins
static std::vector<TileBin> Bins;


//=================================================================================================
// Function prototypes
//=================================================================================================

static void sse_ProcessVertices(const VSConstants& constants);

static void sse_BinTriangles();
static void sse_ProcessSubTile(const Triangle& triangle, int top_left_x, int top_left_y, int tile_size);

static void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output);

static void rz_lerp(const Triangle& triangle, const __m128& px, const __m128& py, __m128& mask, sse_VSOutput* outputs);
static void sse_PixelShader(const sse_VSOutput& input, __m128& r, __m128& g, __m128& b, __m128& a);

//=================================================================================================
// Public methods
//=================================================================================================

void rz_SetRenderTarget(uint32_t* const pRenderTarget, int width, int height, int pitchPixels)
{
    if (RenderTarget == pRenderTarget)
    {
        // No change
        return;
    }

    if (!pRenderTarget)
    {
        RenderTarget = nullptr;
        Bins.clear();
        RTWidth = RTHeight = RTPitchPixels = 0;
        NumHorizBins = NumVertBins = NumTotalBins = 0;
    }
    else
    {
        RenderTarget = pRenderTarget;
        RTWidth = width;
        RTHeight = height;
        RTPitchPixels = pitchPixels;

        NumHorizBins = RTWidth / TILE_SIZE + (RTWidth % TILE_SIZE ? 1 : 0);
        NumVertBins = RTHeight / TILE_SIZE + (RTHeight % TILE_SIZE ? 1 : 0);
        NumTotalBins = NumHorizBins * NumVertBins;
        Bins.resize(NumTotalBins);

        for (int row = 0; row < NumVertBins; ++row)
        {
            for (int column = 0; column < NumHorizBins; ++column)
            {
                auto& bin = Bins[row * NumHorizBins + column];
                bin.row = row;
                bin.column = column;
            }
        }
    }
}


void rz_Draw(const VSConstants& constants, const Vertex* vertices, int numVerts)
{
    assert(numVerts % 3 == 0);
    NumTriangles = numVerts / 3;

    // clear bins
    for (auto& bin : Bins)
    {
        bin.triangles.clear();
    }

    // Restructure input data into SSE friendly layout
    VSInputStream.clear();
    for (int i = 0; i < numVerts; i += 4)
    {
        VSInputStream.push_back(sse_Vertex(&vertices[i], std::min(numVerts - i, 4)));
    }
    VSOutputStream.resize(VSInputStream.size());

    // Vertex shading step
    sse_ProcessVertices(constants);

    // Bin triangles to top level tiles
    sse_BinTriangles();

    // rasterize & shade (recursive descent approach. see Larrabee rasterization article)
    for (auto& bin : Bins)
    {
        for (auto& triangle : bin.triangles)
        {
            sse_ProcessSubTile(triangle, bin.column * TILE_SIZE, bin.row * TILE_SIZE, TILE_SIZE);
        }
    }
}

//=================================================================================================
// Internal methods
//=================================================================================================

// Process vertex input stream, invoking vertex shader and filling in output stream
void sse_ProcessVertices(const VSConstants& constants)
{
    assert(VSInputStream.size() == VSOutputStream.size());
    sse_Vertex* input = VSInputStream.data();
    sse_VSOutput* output = VSOutputStream.data();
    for (size_t i = 0; i < VSInputStream.size(); ++i, ++input, ++output)
    {
        sse_VertexShader(constants, *input, *output);

        // Load result to work on it
        __m128 x = _mm_load_ps(output->position_x);
        __m128 y = _mm_load_ps(output->position_y);
        __m128 z = _mm_load_ps(output->position_z);
        __m128 w = _mm_load_ps(output->position_w);

        // Divide by w
        x = _mm_div_ps(x, w);
        y = _mm_div_ps(y, w);
        z = _mm_div_ps(z, w);

        // Scale to viewport
        x = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0.5f)), _mm_set1_ps(0.5f)), _mm_set1_ps((float)RTWidth));
        y = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(1.f), _mm_add_ps(_mm_mul_ps(y, _mm_set1_ps(0.5f)), _mm_set1_ps(0.5f))), _mm_set1_ps((float)RTHeight));

        // Store back result
        _mm_store_ps(output->position_x, x);
        _mm_store_ps(output->position_y, y);
        _mm_store_ps(output->position_z, z);
        _mm_store_ps(output->position_w, _mm_set1_ps(1.f));
    }
}

void sse_BinTriangles()
{
    Triangle triangle;
    sse_VSOutput* v = VSOutputStream.data();
    for (int i = 0; i < NumTriangles; ++i)
    {
        int iP1 = i * 3;
        int iP2 = iP1 + 1;
        int iP3 = iP2 + 1;

        int iP1base = iP1 / 4;
        int iP1off = iP1 % 4;
        int iP2base = iP2 / 4;
        int iP2off = iP2 % 4;
        int iP3base = iP3 / 4;
        int iP3off = iP3 % 4;

        float2 p1(v[iP1base].position_x[iP1off], v[iP1base].position_y[iP1off]);
        float2 p2(v[iP2base].position_x[iP2off], v[iP2base].position_y[iP2off]);
        float2 p3(v[iP3base].position_x[iP3off], v[iP3base].position_y[iP3off]);

        // determine overlapped bins by bounding box
        float2 bb_min = min(p1, min(p2, p3));
        float2 bb_max = max(p1, max(p2, p3));
        int x1 = (int)bb_min.x / TILE_SIZE;
        int y1 = (int)bb_min.y / TILE_SIZE;
        int x2 = (int)bb_max.x / TILE_SIZE;
        int y2 = (int)bb_max.y / TILE_SIZE;
        x1 = std::max(std::min(x1, NumHorizBins - 1), 0);
        y1 = std::max(std::min(y1, NumVertBins - 1), 0);
        x2 = std::max(std::min(x2, NumHorizBins - 1), 0);
        y2 = std::max(std::min(y2, NumVertBins - 1), 0);

        for (int r = y1; r <= y2; ++r)
        {
            for (int c = x1; c <= x2; ++c)
            {
                auto& bin = Bins[r * NumHorizBins + c];

                triangle.iFirstVertex = iP1;
                triangle.p1 = p1;
                triangle.p2 = p2;
                triangle.p3 = p3;

                // edge equation Bx + Cy = 0, where B & C are computed from slope as B = (y1 - y0) and C = -(x1 - x0) or (x0 - x1).
                triangle.e1 = float2(p2.y - p1.y, p1.x - p2.x);
                triangle.e2 = float2(p3.y - p2.y, p2.x - p3.x);
                triangle.e3 = float2(p1.y - p3.y, p3.x - p1.x);

                // compute corner offset x & y to add to top left corner to find
                // trivial reject corner for each edge
                triangle.off1.x = (triangle.e1.x < 0) ? 1.f : 0.f;
                triangle.off1.y = (triangle.e1.y < 0) ? 1.f : 0.f;
                triangle.off2.x = (triangle.e2.x < 0) ? 1.f : 0.f;
                triangle.off2.y = (triangle.e2.y < 0) ? 1.f : 0.f;
                triangle.off3.x = (triangle.e3.x < 0) ? 1.f : 0.f;
                triangle.off3.y = (triangle.e3.y < 0) ? 1.f : 0.f;

                bin.triangles.push_back(triangle);
            }
        }
    }
}

void sse_ProcessSubTile(const Triangle& triangle, int top_left_x, int top_left_y, int tile_size)   // in pixels
{
    const int size = tile_size / 4;

    assert(size > 0);

    float2 off1 = triangle.off1 * (float)size;
    float2 off2 = triangle.off2 * (float)size;
    float2 off3 = triangle.off3 * (float)size;

    if (size == 1)
    {
        // for pixel level, use center of pixels
        off1 = off2 = off3 = float2(0.5f, 0.5f);
    }

    __m128 indices = _mm_set_ps(0.f, 1.f, 2.f, 3.f);
    __m128 sizes = _mm_set1_ps((float)size);
    __m128 xoff = _mm_mul_ps(indices, sizes);
    __m128 base_corner_x = _mm_add_ps(_mm_set1_ps((float)top_left_x), xoff);

    int maxY = std::min(top_left_y + tile_size, RTHeight);

    for (int y = top_left_y; y < maxY; y += size)
    {
        __m128 base_corner_y = _mm_set1_ps((float)y);

        // trivial reject against edge1

        // side_of_edge:
        // float2 diff = point - base_vert;
        // return dot(diff, edge_equation);

        // float2 diff part
        // break down to all edge1-adjusted x's, then edge1-adjusted y's
        // base_vert is p1
        __m128 adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off1.x));
        __m128 diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p1.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off1.y));
        __m128 diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p1.y));

        // dot part is broken into muls & adds
        // m1 = diff.x * edge.x
        // m2 = diff.y * edge.y
        // m1 + m2
        __m128 m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e1.x));
        __m128 m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e1.y));
        __m128 dots = _mm_add_ps(m1, m2);

        // if dot is > 0.f, those tiles are rejected.
        __m128 e1mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // Now, repeat for edge 2
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off2.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off2.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e2.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e2.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e2mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // And edge3
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off3.x));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off3.y));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e3.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e3.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e3mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // only elements we keep are the ones that passed all three filters. ie:
        // mask1 | mask2 | mask3 == 0
        __m128 mask = _mm_or_ps(e1mask, _mm_or_ps(e2mask, e3mask));

        // mask out any of the tiles/pixels that are off screen
        __m128 screenclipmask = _mm_cmpge_ps(base_corner_x, _mm_set1_ps((float)RTWidth));

        // convert to integer mask for easier testing below
        __m128i imask = _mm_cvtps_epi32(_mm_or_ps(mask, screenclipmask));

        if (size > 1)
        {
            // recurse sub tiles
            if (_mm_testz_si128(imask, _mm_set_epi32(0xFFFFFFFF, 0, 0, 0)))
            {
                sse_ProcessSubTile(triangle, top_left_x, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0xFFFFFFFF, 0, 0)))
            {
                sse_ProcessSubTile(triangle, top_left_x + size, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0xFFFFFFFF, 0)))
            {
                sse_ProcessSubTile(triangle, top_left_x + 2 * size, y, size);
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0, 0xFFFFFFFF)))
            {
                sse_ProcessSubTile(triangle, top_left_x + 3 * size, y, size);
            }
        }
        else
        {
            // rasterize the pixels!
            sse_VSOutput output;
            __m128 barymask;
            rz_lerp(triangle, base_corner_x, base_corner_y, barymask, &output);
            imask = _mm_cvtps_epi32(_mm_or_ps(mask, barymask));

            if (_mm_testz_si128(imask, _mm_set_epi32(0xFFFFFFFF, 0, 0, 0)))
            {
                RenderTarget[y * RTPitchPixels + top_left_x] = 
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[3] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[3] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[3] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0xFFFFFFFF, 0, 0)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 1)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[2] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[2] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[2] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0xFFFFFFFF, 0)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 2)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[1] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[1] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[1] * 255.f));
            }
            if (_mm_testz_si128(imask, _mm_set_epi32(0, 0, 0, 0xFFFFFFFF)))
            {
                RenderTarget[y * RTPitchPixels + (top_left_x + 3)] =
                    0xFF000000 |
                    (uint32_t)((uint8_t)(output.color_z[0] * 255.f)) << 16 |
                    (uint32_t)((uint8_t)(output.color_y[0] * 255.f)) << 8 |
                    (uint32_t)((uint8_t)(output.color_x[0] * 255.f));
            }
        }
    }
}

void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output)
{
    __m128 x = _mm_load_ps(input.position_x);
    __m128 y = _mm_load_ps(input.position_y);
    __m128 z = _mm_load_ps(input.position_z);
    __m128 w = _mm_set1_ps(1.f);

    const matrix4x4* matrices[] = { &constants.WorldMatrix, &constants.ViewMatrix, &constants.ProjectionMatrix };

    __m128 vx, vy, vz, vw;
    for (int i = 0; i < _countof(matrices); ++i)
    {
        // expanded multiply of all 4 positions by matrix
        // dot(float4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]), v),
        // dot(float4(m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]), v),
        // dot(float4(m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]), v),
        // dot(float4(m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]), v));
        // Resulting 4 dots are the components of the result vector
        __m128 mx = _mm_set1_ps(matrices[i]->m[0][0]);
        __m128 my = _mm_set1_ps(matrices[i]->m[1][0]);
        __m128 mz = _mm_set1_ps(matrices[i]->m[2][0]);
        __m128 mw = _mm_set1_ps(matrices[i]->m[3][0]);
        vx = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][1]);
        my = _mm_set1_ps(matrices[i]->m[1][1]);
        mz = _mm_set1_ps(matrices[i]->m[2][1]);
        mw = _mm_set1_ps(matrices[i]->m[3][1]);
        vy = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][2]);
        my = _mm_set1_ps(matrices[i]->m[1][2]);
        mz = _mm_set1_ps(matrices[i]->m[2][2]);
        mw = _mm_set1_ps(matrices[i]->m[3][2]);
        vz = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        mx = _mm_set1_ps(matrices[i]->m[0][3]);
        my = _mm_set1_ps(matrices[i]->m[1][3]);
        mz = _mm_set1_ps(matrices[i]->m[2][3]);
        mw = _mm_set1_ps(matrices[i]->m[3][3]);
        vw = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
        // assign over to x,y,z,w so we can do next iteration back into vx,vy,vz,vw
        x = vx;
        y = vy;
        z = vz;
        w = vw;
    }

    _mm_store_ps(output.position_x, vx);
    _mm_store_ps(output.position_y, vy);
    _mm_store_ps(output.position_z, vz);
    _mm_store_ps(output.position_w, vw);

    for (int i = 0; i < 4; ++i)
    {
        output.color_x[i] = input.color_x[i];
        output.color_y[i] = input.color_y[i];
        output.color_z[i] = input.color_z[i];
    }
}

// Compute barycentric coordinates (lerp weights) for 4 samples at once.
// The computation is done in 2 dimensions (screen space).
// in: a (ax, ay), b (bx, by) and c (cx, cy) are the 3 vertices of the triangle.
//     p (px, py) is the point to compute barycentric coordinates for
// out: wA, wB, wC are the weights at vertices a, b, and c
//      mask will contain a 0 (clear) if the value is computed. It will be 0xFFFFFFFF (set) if invalid
static inline void rz_bary2d(
    const __m128& ax, const __m128& ay, const __m128& bx, const __m128& by, const __m128& cx, const __m128& cy,
    const __m128& px, const __m128& py, __m128& xA, __m128& xB, __m128& xC, __m128& mask)
{
    __m128 abx = _mm_sub_ps(bx, ax);
    __m128 aby = _mm_sub_ps(by, ay);
    __m128 acx = _mm_sub_ps(cx, ax);
    __m128 acy = _mm_sub_ps(cy, ay);

    // Find barycentric coordinates of P (wA, wB, wC)
    __m128 bcx = _mm_sub_ps(cx, bx);
    __m128 bcy = _mm_sub_ps(cy, by);
    __m128 apx = _mm_sub_ps(px, ax);
    __m128 apy = _mm_sub_ps(py, ay);
    __m128 bpx = _mm_sub_ps(px, bx);
    __m128 bpy = _mm_sub_ps(py, by);

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

    xA = _mm_mul_ps(wA, norm);
    xB = _mm_mul_ps(wB, norm);
    xC = _mm_mul_ps(wC, norm);
}

void rz_lerp(const Triangle& triangle, const __m128& px, const __m128& py, __m128& mask, sse_VSOutput* outputs)
{
    __m128 ax = _mm_set1_ps(triangle.p1.x);
    __m128 ay = _mm_set1_ps(triangle.p1.y);
    __m128 bx = _mm_set1_ps(triangle.p2.x);
    __m128 by = _mm_set1_ps(triangle.p2.y);
    __m128 cx = _mm_set1_ps(triangle.p3.x);
    __m128 cy = _mm_set1_ps(triangle.p3.y);

    __m128 xA, xB, xC;
    rz_bary2d(ax, ay, bx, by, cx, cy, px, py, xA, xB, xC, mask);

    // Interpolate all the attributes for these 4 pixels
    __m128 posx = _mm_add_ps(_mm_mul_ps(ax, xA), _mm_add_ps(_mm_mul_ps(bx, xB), _mm_mul_ps(cx, xC)));
    __m128 posy = _mm_add_ps(_mm_mul_ps(ay, xA), _mm_add_ps(_mm_mul_ps(by, xB), _mm_mul_ps(cy, xC)));
    __m128 posz = _mm_setzero_ps();
    __m128 posw = _mm_set1_ps(1.f);

    int iP1 = triangle.iFirstVertex;
    int iP2 = iP1 + 1;
    int iP3 = iP2 + 1;

    int iP1base = iP1 / 4;
    int iP1off = iP1 % 4;
    int iP2base = iP2 / 4;
    int iP2off = iP2 % 4;
    int iP3base = iP3 / 4;
    int iP3off = iP3 % 4;

    float3 c1(VSOutputStream[iP1base].color_x[iP1off], VSOutputStream[iP1base].color_y[iP1off], VSOutputStream[iP1base].color_z[iP1off]);
    float3 c2(VSOutputStream[iP2base].color_x[iP2off], VSOutputStream[iP2base].color_y[iP2off], VSOutputStream[iP2base].color_z[iP2off]);
    float3 c3(VSOutputStream[iP3base].color_x[iP3off], VSOutputStream[iP3base].color_y[iP3off], VSOutputStream[iP3base].color_z[iP3off]);

    __m128 colx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.x), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.x), xB), _mm_mul_ps(_mm_set1_ps(c3.x), xC)));
    __m128 coly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.y), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.y), xB), _mm_mul_ps(_mm_set1_ps(c3.y), xC)));
    __m128 colz = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c1.z), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(c2.z), xB), _mm_mul_ps(_mm_set1_ps(c3.z), xC)));

    _mm_store_ps(outputs->position_x, posx);
    _mm_store_ps(outputs->position_y, posy);
    _mm_store_ps(outputs->position_z, posz);
    _mm_store_ps(outputs->position_w, posw);
    _mm_store_ps(outputs->color_x, colx);
    _mm_store_ps(outputs->color_y, coly);
    _mm_store_ps(outputs->color_z, colz);
}

void sse_PixelShader(const sse_VSOutput& input, __m128& r, __m128& g, __m128& b, __m128& a)
{
    r = _mm_load_ps(input.color_x);
    g = _mm_load_ps(input.color_y);
    b = _mm_load_ps(input.color_z);
    a = _mm_set1_ps(1.f);
}
