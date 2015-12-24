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
static int NumVertices;
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

static void sse_StreamInVertices(const Vertex* const vertices, int numVerts);
static void sse_ProcessVertices(const VSConstants& constants);

static void rz_ClearBins();
static void sse_BinTriangles();

static void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output);
static void rz_RasterizeTile(int row, int col);
static float2 rz_edge_equation(const float2& v1, const float2& v2);


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
    }
}


void rz_Draw(const VSConstants& constants, const Vertex* vertices, int numVerts)
{
    NumVertices = numVerts;
    NumTriangles = NumVertices / 3;
    assert(NumTriangles * 3 == NumVertices);

    rz_ClearBins();
    sse_StreamInVertices(vertices, numVerts);
    sse_ProcessVertices(constants);
    sse_BinTriangles();

    for (int r = 0; r < NumVertBins; ++r)
    {
        for (int c = 0; c < NumHorizBins; ++c)
        {
            rz_RasterizeTile(r, c);
        }
    }
}

//=================================================================================================
// Internal methods
//=================================================================================================

// Prepare and fill input stream to vertex shader from app vertex source data
void sse_StreamInVertices(const Vertex* const vertices, int numVerts)
{
    // number of sse_Vertex that we need. Add 1 if input vertices don't fit into exact multiple
    // of 4. That last one is a partial sse_Vertex.
    int sse_NumVerts = (numVerts / 4) + (numVerts % 4 ? 1 : 0);
    VSInputStream.resize(sse_NumVerts);
    VSOutputStream.resize(sse_NumVerts);

    const Vertex* v = vertices;
    for (int i = 0; i < numVerts; ++i, ++v)
    {
        int sse_vertex_index = i / 4;
        int sse_vertex_comp = i % 4;

        VSInputStream[sse_vertex_index].position_x[sse_vertex_comp] = v->Position.x;
        VSInputStream[sse_vertex_index].position_y[sse_vertex_comp] = v->Position.y;
        VSInputStream[sse_vertex_index].position_z[sse_vertex_comp] = v->Position.z;
        VSInputStream[sse_vertex_index].color_x[sse_vertex_comp] = v->Color.x;
        VSInputStream[sse_vertex_index].color_y[sse_vertex_comp] = v->Color.y;
        VSInputStream[sse_vertex_index].color_z[sse_vertex_comp] = v->Color.z;
    }
}

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

void rz_ClearBins()
{
    TileBin* bin = Bins.data();
    for (size_t i = 0; i < Bins.size(); ++i, ++bin)
    {
        bin->triangles.clear();
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
                triangle.e1 = rz_edge_equation(p1, p2);
                triangle.e2 = rz_edge_equation(p2, p3);
                triangle.e3 = rz_edge_equation(p3, p1);

                bin.triangles.push_back(triangle);
            }
        }
    }
}

static void sse_ProcessSubTile(
    const Triangle& triangle,
    int top_left_x, int top_left_y, // in pixels
    int width, int height,          // in pixels
    int off_x1, int off_y1,
    int off_x2, int off_y2,
    int off_x3, int off_y3)
{
    const int hsize = width / 4;
    const int vsize = height / 4;

    assert(hsize > 0 && vsize > 0);

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

    __m128 indices = _mm_set_ps(0.f, 1.f, 2.f, 3.f);
    __m128 hsizes = _mm_set1_ps((float)hsize);
    __m128 xoff = _mm_mul_ps(indices, hsizes);
    __m128 base_corner_x = _mm_add_ps(_mm_set1_ps((float)top_left_x), xoff);

    int maxY = std::min(top_left_y + height, RTHeight);

    for (int y = top_left_y; y < maxY; y += vsize)
    {
        __m128 base_corner_y = _mm_set1_ps((float)y);

        // trivial reject against edge1

        // side_of_edge:
        // float2 diff = point - base_vert;
        // return dot(diff, edge_equation);

        // float2 diff part
        // break down to all edge1-adjusted x's, then edge1-adjusted y's
        // base_vert is p1
        __m128 adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h1));
        __m128 diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p1.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v1));
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
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h2));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v2));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p2.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e2.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e2.y));
        dots = _mm_add_ps(m1, m2);
        __m128 e2mask = _mm_cmpgt_ps(dots, _mm_setzero_ps());

        // And edge3
        adj = _mm_add_ps(base_corner_x, _mm_set1_ps(off_h3));
        diffx = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.x));
        adj = _mm_add_ps(base_corner_y, _mm_set1_ps(off_v3));
        diffy = _mm_sub_ps(adj, _mm_set1_ps(triangle.p3.y));
        m1 = _mm_mul_ps(diffx, _mm_set1_ps(triangle.e3.x));
        m2 = _mm_mul_ps(diffy, _mm_set1_ps(triangle.e3.y));
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
                    sse_ProcessSubTile(
                        triangle,
                        top_left_x + (hsize * i), y,
                        hsize, vsize,
                        off_x1, off_y1,
                        off_x2, off_y2,
                        off_x3, off_y3);
                }
            }
        }
        else
        {
            // rasterize the pixels!
            for (int i = 0; i < 4; ++i)
            {
                int maskIndex = 3 - i;
                if (fmasks[maskIndex] == 0.f)
                {
                    // can vectorize this comparison
                    int x = top_left_x + (i * hsize);
                    if (x >= RTWidth)
                        break;

                    RenderTarget[y * RTPitchPixels + x] = 0xFFFF0000;
                }
            }
        }
    }
}

// recursive descent approach (see Larrabee rasterization article)
void rz_RasterizeTile(int row, int col)
{
    auto& bin = Bins[row * NumHorizBins + col];

    for (size_t i = 0; i < bin.triangles.size(); ++i)
    {
        const auto& triangle = bin.triangles[i];

        // we need to determine 'trivial reject' corner. compute corner offset x & y
        // to add to top left corner to find trivial reject corner for each edge
        int off_x1 = 0, off_x2 = 0, off_x3 = 0, off_y1 = 0, off_y2 = 0, off_y3 = 0;

        if (triangle.e1.x < 0)    off_x1 = 1;
        if (triangle.e2.x < 0)    off_x2 = 1;
        if (triangle.e3.x < 0)    off_x3 = 1;
        if (triangle.e1.y < 0)    off_y1 = 1;
        if (triangle.e2.y < 0)    off_y2 = 1;
        if (triangle.e3.y < 0)    off_y3 = 1;

        sse_ProcessSubTile(
            triangle,
            col * TILE_SIZE, row * TILE_SIZE,
            TILE_SIZE, TILE_SIZE,
            off_x1, off_y1,
            off_x2, off_y2,
            off_x3, off_y3);
    }
}

void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output)
{
    __m128 x = _mm_load_ps(input.position_x);
    __m128 y = _mm_load_ps(input.position_y);
    __m128 z = _mm_load_ps(input.position_z);
    __m128 w = _mm_set1_ps(1.f);

    // expanded multiply of all 4 positions by matrix
    // dot(float4(m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0]), v),
    // dot(float4(m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1]), v),
    // dot(float4(m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2]), v),
    // dot(float4(m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]), v));
    // Resulting 4 dots are the components of the result vector

    __m128 mx = _mm_set1_ps(constants.WorldMatrix.m[0][0]);
    __m128 my = _mm_set1_ps(constants.WorldMatrix.m[1][0]);
    __m128 mz = _mm_set1_ps(constants.WorldMatrix.m[2][0]);
    __m128 mw = _mm_set1_ps(constants.WorldMatrix.m[3][0]);
    __m128 vx = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.WorldMatrix.m[0][1]);
    my = _mm_set1_ps(constants.WorldMatrix.m[1][1]);
    mz = _mm_set1_ps(constants.WorldMatrix.m[2][1]);
    mw = _mm_set1_ps(constants.WorldMatrix.m[3][1]);
    __m128 vy = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.WorldMatrix.m[0][2]);
    my = _mm_set1_ps(constants.WorldMatrix.m[1][2]);
    mz = _mm_set1_ps(constants.WorldMatrix.m[2][2]);
    mw = _mm_set1_ps(constants.WorldMatrix.m[3][2]);
    __m128 vz = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.WorldMatrix.m[0][3]);
    my = _mm_set1_ps(constants.WorldMatrix.m[1][3]);
    mz = _mm_set1_ps(constants.WorldMatrix.m[2][3]);
    mw = _mm_set1_ps(constants.WorldMatrix.m[3][3]);
    __m128 vw = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));

    x = vx;
    y = vy;
    z = vz;
    w = vw;
    mx = _mm_set1_ps(constants.ViewMatrix.m[0][0]);
    my = _mm_set1_ps(constants.ViewMatrix.m[1][0]);
    mz = _mm_set1_ps(constants.ViewMatrix.m[2][0]);
    mw = _mm_set1_ps(constants.ViewMatrix.m[3][0]);
    vx = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ViewMatrix.m[0][1]);
    my = _mm_set1_ps(constants.ViewMatrix.m[1][1]);
    mz = _mm_set1_ps(constants.ViewMatrix.m[2][1]);
    mw = _mm_set1_ps(constants.ViewMatrix.m[3][1]);
    vy = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ViewMatrix.m[0][2]);
    my = _mm_set1_ps(constants.ViewMatrix.m[1][2]);
    mz = _mm_set1_ps(constants.ViewMatrix.m[2][2]);
    mw = _mm_set1_ps(constants.ViewMatrix.m[3][2]);
    vz = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ViewMatrix.m[0][3]);
    my = _mm_set1_ps(constants.ViewMatrix.m[1][3]);
    mz = _mm_set1_ps(constants.ViewMatrix.m[2][3]);
    mw = _mm_set1_ps(constants.ViewMatrix.m[3][3]);
    vw = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));

    x = vx;
    y = vy;
    z = vz;
    w = vw;
    mx = _mm_set1_ps(constants.ProjectionMatrix.m[0][0]);
    my = _mm_set1_ps(constants.ProjectionMatrix.m[1][0]);
    mz = _mm_set1_ps(constants.ProjectionMatrix.m[2][0]);
    mw = _mm_set1_ps(constants.ProjectionMatrix.m[3][0]);
    vx = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ProjectionMatrix.m[0][1]);
    my = _mm_set1_ps(constants.ProjectionMatrix.m[1][1]);
    mz = _mm_set1_ps(constants.ProjectionMatrix.m[2][1]);
    mw = _mm_set1_ps(constants.ProjectionMatrix.m[3][1]);
    vy = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ProjectionMatrix.m[0][2]);
    my = _mm_set1_ps(constants.ProjectionMatrix.m[1][2]);
    mz = _mm_set1_ps(constants.ProjectionMatrix.m[2][2]);
    mw = _mm_set1_ps(constants.ProjectionMatrix.m[3][2]);
    vz = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));
    mx = _mm_set1_ps(constants.ProjectionMatrix.m[0][3]);
    my = _mm_set1_ps(constants.ProjectionMatrix.m[1][3]);
    mz = _mm_set1_ps(constants.ProjectionMatrix.m[2][3]);
    mw = _mm_set1_ps(constants.ProjectionMatrix.m[3][3]);
    vw = _mm_add_ps(_mm_mul_ps(mx, x), _mm_add_ps(_mm_mul_ps(my, y), _mm_add_ps(_mm_mul_ps(mz, z), _mm_mul_ps(mw, w))));

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

// edge equation Bx + Cy = 0, where B & C are computed from slope as B = (y1 - y0) and C = -(x1 - x0) or (x0 - x1).
float2 rz_edge_equation(const float2& v1, const float2& v2)
{
    return float2(v2.y - v1.y, v1.x - v2.x);
}

