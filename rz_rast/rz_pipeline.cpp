#include "precomp.h"
#include "rz_common.h"
#include <atomic>
#include <vector>

//=================================================================================================
// AOSOA (array of structs of array). Each element is a group, or bundle,
// of vertices optimized for concurrent SIMD processing (currently only SSE, later AVX1/2/512)
//=================================================================================================

// SSE version. Designed for doing 4 vertices at once
struct alignas(16) sse_Vertex
{
    float position_x[4];
    float position_y[4];
    float position_z[4];
    float color_x[4];
    float color_y[4];
    float color_z[4];
};

struct alignas(16) sse_VSOutput
{
    float position_x[4];
    float position_y[4];
    float position_z[4];
    float position_w[4];
    float color_x[4];
    float color_y[4];
    float color_z[4];
};

static std::vector<sse_Vertex> sse_vs_input;
static std::vector<sse_VSOutput> sse_vs_output;
static int num_vertices;
static int num_triangles;

//=================================================================================================
// Bins used for tiled rasterization
//=================================================================================================

// Each triangle is packed into the first 3 components of an sse_VSOutput for convenience
struct tile_bin
{
    std::vector<sse_VSOutput> triangles;
};

static const int TILE_SIZE = 256; // top level tile size. must break down to 4x4 evenly

static int num_hbins;
static int num_vbins;
static int num_total_bins;  // num_hbins * num_vbins
static tile_bin* bins;

static int render_target_width;
static int render_target_height;
static uint32_t* render_target;
static int render_target_pitch_in_pixels;

//=================================================================================================
// Function prototypes
//=================================================================================================

static void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output);
static void sse_DivideByW(sse_VSOutput& output);
static void rz_rasterize_tile(int row, int col);

//=================================================================================================
// Implementation
//=================================================================================================

// Prepare and fill input stream to vertex shader from app vertex source data
static void sse_StreamInVertices(const Vertex* vertices, int numVerts)
{
    num_vertices = numVerts;
    num_triangles = num_vertices / 3;
    assert(num_triangles * 3 == num_vertices);

    // number of sse_Vertex that we need. Add 1 if input vertices don't fit into exact multiple
    // of 4. That last one is a partial sse_Vertex.
    int num_sse_verts = (numVerts / 4) + (numVerts % 4 ? 1 : 0);
    sse_vs_input.resize(num_sse_verts);
    sse_vs_output.resize(num_sse_verts);

    // TODO: Consider starting up vertex processing async while this
    // stream gets filled in.

    const Vertex* v = vertices;
    for (int i = 0; i < numVerts; ++i, ++v)
    {
        int sse_vertex_index = i / 4;
        int sse_vertex_comp = i % 4;

        sse_vs_input[sse_vertex_index].position_x[sse_vertex_comp] = v->Position.x;
        sse_vs_input[sse_vertex_index].position_y[sse_vertex_comp] = v->Position.y;
        sse_vs_input[sse_vertex_index].position_z[sse_vertex_comp] = v->Position.z;
        sse_vs_input[sse_vertex_index].color_x[sse_vertex_comp] = v->Color.x;
        sse_vs_input[sse_vertex_index].color_y[sse_vertex_comp] = v->Color.y;
        sse_vs_input[sse_vertex_index].color_z[sse_vertex_comp] = v->Color.z;
    }
}

// Process vertex input stream, invoking vertex shader and filling in output stream
static void sse_ProcessVertices()
{
    VSConstants constants;

    // TODO: parallelize this
    assert(sse_vs_input.size() == sse_vs_output.size());
    sse_Vertex* input = sse_vs_input.data();
    sse_VSOutput* output = sse_vs_output.data();
    for (int i = 0; i < sse_vs_input.size(); ++i, ++input, ++output)
    {
        sse_VertexShader(constants, *input, *output);
        sse_DivideByW(*output);
    }
}

// Clear bins
static void rz_clear_bins()
{
    tile_bin* bin = bins;
    for (int i = 0; i < num_total_bins; ++i, ++bin)
    {
        bin->triangles.clear();
    }
}

// Bin triangles
static void sse_BinTriangles()
{
    // TODO: Consider making this parallel
    sse_VSOutput triangle;
    sse_VSOutput* v = sse_vs_output.data();
    for (int i = 0; i < num_triangles; ++i)
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

        for (int r = ((int)bb_min.y / TILE_SIZE); r <= ((int)bb_max.y / TILE_SIZE); ++r)
        {
            if (r < 0 || r >= num_vbins)
                continue;

            for (int c = ((int)bb_min.x / TILE_SIZE); c <= ((int)bb_max.x / TILE_SIZE); ++c)
            {
                if (c < 0 || c >= num_hbins)
                    continue;

                auto& bin = bins[r * num_hbins + c];

                triangle.position_x[0] = v[iP1base].position_x[iP1off];
                triangle.position_y[0] = v[iP1base].position_y[iP1off];
                triangle.position_z[0] = v[iP1base].position_z[iP1off];
                triangle.position_w[0] = v[iP1base].position_w[iP1off];
                triangle.color_x[0] = v[iP1base].color_x[iP1off];
                triangle.color_y[0] = v[iP1base].color_y[iP1off];
                triangle.color_z[0] = v[iP1base].color_z[iP1off];

                triangle.position_x[1] = v[iP2base].position_x[iP2off];
                triangle.position_y[1] = v[iP2base].position_y[iP2off];
                triangle.position_z[1] = v[iP2base].position_z[iP2off];
                triangle.position_w[1] = v[iP2base].position_w[iP2off];
                triangle.color_x[1] = v[iP2base].color_x[iP2off];
                triangle.color_y[1] = v[iP2base].color_y[iP2off];
                triangle.color_z[1] = v[iP2base].color_z[iP2off];

                triangle.position_x[2] = v[iP3base].position_x[iP3off];
                triangle.position_y[2] = v[iP3base].position_y[iP3off];
                triangle.position_z[2] = v[iP3base].position_z[iP3off];
                triangle.position_w[2] = v[iP3base].position_w[iP3off];
                triangle.color_x[2] = v[iP3base].color_x[iP3off];
                triangle.color_y[2] = v[iP3base].color_y[iP3off];
                triangle.color_z[2] = v[iP3base].color_z[iP3off];

                bin.triangles.push_back(triangle);
            }
        }
    }
}

void sse_VertexShader(const VSConstants& constants, const sse_Vertex& input, sse_VSOutput& output)
{
    UNREFERENCED(constants);
    UNREFERENCED(input);
    UNREFERENCED(output);
}

void sse_DivideByW(sse_VSOutput& output)
{
    __m128 x = _mm_load_ps(output.position_x);
    __m128 y = _mm_load_ps(output.position_y);
    __m128 z = _mm_load_ps(output.position_z);
    __m128 w = _mm_load_ps(output.position_w);
    _mm_store_ps(output.position_x, _mm_div_ps(x, w));
    _mm_store_ps(output.position_y, _mm_div_ps(y, w));
    _mm_store_ps(output.position_z, _mm_div_ps(z, w));
    _mm_store_ps(output.position_w, _mm_set1_ps(1.f));
}
