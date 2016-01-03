#include "Precomp.h"
#include "TurboTrace.h"

//#define CULL_ENABLED
#define MT_ENABLED

// private implementation types

struct vec3
{
    __m128 x;
    __m128 y;
    __m128 z;
};

struct test_result
{
    __m128 hit;
    __m128 dist;
};

struct thread_context
{
    int id;
    HANDLE work_event;
};

struct thread_work_item
{
    const raytracer_config* config;
    const sphere_data* spheres;
    int sphere_count;
    const triangle_data* triangles;
    int triangle_count;
    const box_data* boxes;
    int box_count;
    const aabb_node* scene;
    // block to work on
    int x, y;
    int width, height;
};

// forward declaration of private implementation methods

static __forceinline vec3 __vectorcall add(vec3 v1, vec3 v2);
static __forceinline vec3 __vectorcall sub(vec3 v1, vec3 v2);
static __forceinline __m128 __vectorcall dot(vec3 v1, vec3 v2);
static __forceinline vec3 __vectorcall cross(vec3 v1, vec3 v2);
static __forceinline __m128 __vectorcall length_squared(vec3 v);
static __forceinline vec3 __vectorcall normalize(vec3 v);
static __forceinline vec3 __vectorcall expand(__m128 v);

static __forceinline float dot(const float v1[3], const float v2[3]);
static __forceinline void cross(const float v1[3], const float v2[3], float result[3]);
static __forceinline float length_squared(const float v[3]);
static __forceinline void normalize(float v[3]);


static test_result __vectorcall test_sphere(
    vec3 start, vec3 dir,
    __m128 sphere);

static test_result __vectorcall test_triangle(
    vec3 start, vec3 dir,
    __m128 v1, __m128 v2, __m128 v3, __m128 in_norm);

static test_result __vectorcall test_box(
    vec3 start, vec3 dir,
    __m128 in_min, __m128 in_max);

static void __vectorcall trace_rays(
    const raytracer_config* config,
    vec3 start, vec3 dir,
    __m128i output_x, __m128i output_y,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count,
    const box_data* boxes, int box_count);

static __forceinline uint32_t compute_color(const float norm[3]);

static void cull_objects(
    const raytracer_config* config,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count);

static void cull_spheres(
    vec3 clip_planes, __m128 clip_dists,
    const sphere_data* spheres, int sphere_count);

static void cull_triangles(
    vec3 clip_planes, __m128 clip_dists,
    const triangle_data* triangles, int triangle_count);

static void mt_init();

static void __vectorcall trace_rays(
    const raytracer_config* config,
    vec3 start, vec3 dir,
    __m128i output_x, __m128i output_y,
    const aabb_node* scene);


// Results after culling pass
// TODO: needs to be thread local?
static sphere_data* final_spheres;
static int final_sphere_count;
static int final_sphere_capacity;
static triangle_data* final_triangles;
static int final_triangle_count;
static int final_triangle_capacity;

static const int num_threads = 8;
static HANDLE shutdown_event;
static HANDLE complete_event;
static HANDLE* thread_handles;
static thread_context* thread_contexts;
static thread_work_item work_items[16]; // 4x4 division of screen
static long work_items_count = _countof(work_items);
static long current_work_item;
static long work_complete_gate;
static DWORD CALLBACK thread_proc(void* context);

//=============================================================================
// public methods
//=============================================================================

void __stdcall tt_setup(
    raytracer_config* config,
    uint32_t* render_target, int width, int height, int pitch,
    float fov, const float position[3])
{
#ifdef MT_ENABLED
    // no-ops if already called
    mt_init();
#endif

    config->render_target = render_target;
    config->render_target_width = width;
    config->render_target_height = height;
    config->render_target_pitch = pitch;
    config->half_render_target_width = 0.5f * width;
    config->half_render_target_height = 0.5f * height;
    memcpy_s(config->view_position, sizeof(config->view_position), position, sizeof(config->view_position));
    config->view_forward[0] = 0.f;
    config->view_forward[1] = 0.f;
    config->view_forward[2] = 1.f;
    config->view_up[0] = 0.f;
    config->view_up[1] = 1.f;
    config->view_up[2] = 0.f;

    float half_fov = fov * 0.5f;
    float inv_tan_half_fov = 1.f / tanf(half_fov);

    config->dist_to_plane =
        config->half_render_target_width * inv_tan_half_fov;
}

void __stdcall tt_trace(
    const raytracer_config* config,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count,
    const box_data* boxes, int box_count)
{
    assert(config);
    assert(spheres || sphere_count <= 0);
    assert(triangles || triangle_count <= 0);
    assert(boxes || box_count <= 0);
    if (sphere_count <= 0 && triangle_count <= 0 && box_count <= 0)
    {
        return;
    }

#ifdef CULL_ENABLED
    cull_objects(config, spheres, sphere_count, triangles, triangle_count);
#endif

#ifdef MT_ENABLED

    // divide rt into 4x4
    int width = (config->render_target_width + 3) / 4;
    int height = (config->render_target_height + 3) / 4;
    // ensure they are even (since we do 2x2s)
    if (width % 2 != 0) ++width;
    if (height % 2 != 0) ++height;
    for (int i = 0; i < 16; ++i)
    {
        work_items[i].config = config;
        work_items[i].x = (i % 4) * width;
        work_items[i].y = (i / 4) * height;
        work_items[i].width = width;
        work_items[i].height = height;
        work_items[i].spheres = spheres;
        work_items[i].sphere_count = sphere_count;
        work_items[i].triangles = triangles;
        work_items[i].triangle_count = triangle_count;
        work_items[i].scene = nullptr;
    }

    current_work_item = 0;
    work_complete_gate = 0;
    for (int i = 0; i < num_threads; ++i)
    {
        SetEvent(thread_contexts[i].work_event);
    }

    WaitForSingleObject(complete_event, INFINITE);

#else
    // process in 2x2 blocks of rays
    for (int y = 0; y < config->render_target_height; y += 2)
    {
        __m128i output_y = _mm_set_epi32(y + 1, y + 1, y, y);

        for (int x = 0; x < config->render_target_width; x += 2)
        {
            __m128i output_x = _mm_set_epi32(x + 1, x, x + 1, x);

            vec3 start, dir;

            // TODO: don't support view_forward/up yet
            start.x = _mm_set1_ps(config->view_position[0]);
            start.y = _mm_set1_ps(config->view_position[1]);
            start.z = _mm_set1_ps(config->view_position[2]);

            float base_x = x - config->half_render_target_width;
            float base_y = config->half_render_target_height - y;

            dir.x = _mm_set_ps(base_x + 1, base_x, base_x + 1, base_x);
            dir.y = _mm_set_ps(base_y - 1, base_y - 1, base_y, base_y);
            dir.z = _mm_set1_ps(config->dist_to_plane);

            dir = normalize(dir);

#ifdef CULL_ENABLED
            trace_rays(config,
                start, dir,
                output_x, output_y,
                final_spheres, final_sphere_count,
                final_triangles, final_triangle_count);
#else
            trace_rays(config,
                start, dir,
                output_x, output_y,
                spheres, sphere_count,
                triangles, triangle_count,
                boxes, box_count);
#endif // CULL_ENABLED
        }
    }

#endif  // MT_ENABLED
}


//=============================================================================
// private methods
//=============================================================================

__forceinline vec3 __vectorcall add(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = _mm_add_ps(v1.x, v2.x);
    result.y = _mm_add_ps(v1.y, v2.y);
    result.z = _mm_add_ps(v1.z, v2.z);
    return result;
}

__forceinline vec3 __vectorcall sub(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = _mm_sub_ps(v1.x, v2.x);
    result.y = _mm_sub_ps(v1.y, v2.y);
    result.z = _mm_sub_ps(v1.z, v2.z);
    return result;
}

__forceinline __m128 __vectorcall dot(vec3 v1, vec3 v2)
{
    __m128 m1 = _mm_mul_ps(v1.x, v2.x);
    __m128 m2 = _mm_mul_ps(v1.y, v2.y);
    __m128 m3 = _mm_mul_ps(v1.z, v2.z);
    __m128 a1 = _mm_add_ps(m1, m2);
    return _mm_add_ps(a1, m3);
}

__forceinline vec3 __vectorcall cross(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = _mm_sub_ps(_mm_mul_ps(v1.y, v2.z), _mm_mul_ps(v2.y, v1.z));
    result.y = _mm_sub_ps(_mm_mul_ps(v1.z, v2.x), _mm_mul_ps(v2.z, v1.x));
    result.z = _mm_sub_ps(_mm_mul_ps(v1.x, v2.y), _mm_mul_ps(v2.x, v1.y));
    return result;
}

__forceinline __m128 __vectorcall length_squared(vec3 v)
{
    return dot(v, v);
}

__forceinline vec3 __vectorcall normalize(vec3 v)
{
    __m128 len_sq = length_squared(v);
    __m128 inv_len = _mm_div_ps(_mm_set1_ps(1.f), _mm_sqrt_ps(len_sq));
    vec3 result;
    result.x = _mm_mul_ps(v.x, inv_len);
    result.y = _mm_mul_ps(v.y, inv_len);
    result.z = _mm_mul_ps(v.z, inv_len);
    return result;
}

__forceinline vec3 __vectorcall expand(__m128 v)
{
    vec3 result;
    result.x = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
    result.y = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
    result.z = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
    return result;
}

__forceinline float dot(const float v1[3], const float v2[3])
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

__forceinline void cross(const float v1[3], const float v2[3], float result[3])
{
    result[0] = v1[1] * v2[2] - v2[1] * v1[2];
    result[1] = v1[2] * v2[0] - v2[2] * v1[0];
    result[2] = v1[0] * v2[1] - v2[0] * v1[1];
}

__forceinline float length_squared(const float v[3])
{
    return dot(v, v);
}

__forceinline void normalize(float v[3])
{
    float len_sq = length_squared(v);
    float inv_len = 1.f / sqrtf(len_sq);
    v[0] *= inv_len;
    v[1] *= inv_len;
    v[2] *= inv_len;
}

test_result __vectorcall test_sphere(
    vec3 start, vec3 dir,
    __m128 sphere)
{
    vec3 center = expand(sphere);
    __m128 radius_squared =
        _mm_shuffle_ps(sphere, sphere, _MM_SHUFFLE(3, 3, 3, 3));

    // diff = center - start;
    vec3 diff = sub(center, start);

    // d = dot(diff, dir);
    __m128 d = dot(diff, dir);
    __m128 d_squared = _mm_mul_ps(d, d);

    test_result result;
    result.hit = _mm_cmpgt_ps(d, _mm_setzero_ps());

    // diff_squared = dot(diff, diff);
    __m128 diff_squared = dot(diff, diff);

    // x_squared = diff_squared - d_squared;
    __m128 x_squared = _mm_sub_ps(diff_squared, d_squared);

    // hit = x_squared < radius_squared;
    result.hit = _mm_and_ps(result.hit, _mm_cmplt_ps(x_squared, radius_squared));

    // d2_squared = radius_squared - x_squared;
    // dist = d - sqrt(d2_squared);
    __m128 d2_squared = _mm_sub_ps(radius_squared, x_squared);
    result.dist = _mm_sub_ps(d, _mm_sqrt_ps(d2_squared));

    return result;
}

test_result __vectorcall test_triangle(
    vec3 start, vec3 dir,
    __m128 v1, __m128 v2, __m128 v3, __m128 in_norm)
{
    vec3 a = expand(v1);
    vec3 b = expand(v2);
    vec3 c = expand(v3);
    vec3 norm = expand(in_norm);

    __m128 zero = _mm_setzero_ps();
    // we negate the norm before doing the dot
    // this gives us a positive dot for dir approaching
    // plane, but more importantly it gives us cosA
    vec3 neg_norm = sub(expand(zero), norm);

    __m128 d = dot(neg_norm, dir);

    // test that we're pointing towards the plane
    test_result result;
    result.hit = _mm_cmpgt_ps(d, zero);

    // test that our start is in front of the plane
    vec3 diff = sub(start, a);

    __m128 d2 = dot(norm, diff);
    result.hit = _mm_and_ps(result.hit, _mm_cmpgt_ps(d2, zero));

    // dist = d2 / cosA
    result.dist = _mm_div_ps(d2, d);

    // scale dir by r & add to start to get point on plane
    vec3 point;
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, result.dist));
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, result.dist));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, result.dist));

    // test if point is in triangle
    vec3 c1 = cross(sub(b, a), sub(point, a));
    vec3 c2 = cross(sub(c, b), sub(point, b));
    vec3 c3 = cross(sub(a, c), sub(point, c));
    __m128 dot1 = dot(c1, norm);
    __m128 dot2 = dot(c2, norm);
    __m128 dot3 = dot(c3, norm);
    result.hit = _mm_and_ps(result.hit, _mm_cmpge_ps(dot1, zero));
    result.hit = _mm_and_ps(result.hit, _mm_cmpge_ps(dot2, zero));
    result.hit = _mm_and_ps(result.hit, _mm_cmpge_ps(dot3, zero));

    return result;
}

test_result __vectorcall test_box(
    vec3 start, vec3 dir, 
    __m128 in_min, __m128 in_max)
{
    vec3 min = expand(in_min);
    vec3 max = expand(in_max);

    test_result result;
    result.hit = _mm_setzero_ps();
    result.dist = _mm_set1_ps(FLT_MAX);

    __m128 zero = _mm_setzero_ps();

    vec3 dist1 = sub(min, start);
    vec3 dist2 = sub(start, max);

    // enter from -x
    __m128 mask = _mm_and_ps(_mm_cmpgt_ps(dist1.x, zero), _mm_cmpgt_ps(dir.x, zero));
    __m128 r = _mm_div_ps(dist1.x, dir.x);
    vec3 point;
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, r));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.y, min.y));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.y, max.y));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.z, min.z));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.z, max.z));
    result.hit = _mm_or_ps(result.hit, mask);
    result.dist = r;

    // enter from +x
    mask = _mm_and_ps(_mm_cmpgt_ps(dist2.x, zero), _mm_cmplt_ps(dir.x, zero));
    r = _mm_div_ps(dist2.x, _mm_sub_ps(zero, dir.x));
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, r));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.y, min.y));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.y, max.y));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.z, min.z));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.z, max.z));
    result.hit = _mm_or_ps(result.hit, mask);
    __m128 comp = _mm_cmplt_ps(r, result.dist);
    mask = _mm_and_ps(mask, comp);
    result.dist = _mm_or_ps(_mm_and_ps(mask, r), _mm_andnot_ps(mask, result.dist));

    // enter from -y
    mask = _mm_and_ps(_mm_cmpgt_ps(dist1.y, zero), _mm_cmpgt_ps(dir.y, zero));
    r = _mm_div_ps(dist1.y, dir.y);
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, r));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.x, min.x));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.x, max.x));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.z, min.z));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.z, max.z));
    result.hit = _mm_or_ps(result.hit, mask);
    comp = _mm_cmplt_ps(r, result.dist);
    mask = _mm_and_ps(mask, comp);
    result.dist = _mm_or_ps(_mm_and_ps(mask, r), _mm_andnot_ps(mask, result.dist));

    // enter from +y
    mask = _mm_and_ps(_mm_cmpgt_ps(dist2.y, zero), _mm_cmplt_ps(dir.y, zero));
    r = _mm_div_ps(dist2.y, _mm_sub_ps(zero, dir.y));
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, r));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.x, min.x));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.x, max.x));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.z, min.z));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.z, max.z));
    result.hit = _mm_or_ps(result.hit, mask);
    comp = _mm_cmplt_ps(r, result.dist);
    mask = _mm_and_ps(mask, comp);
    result.dist = _mm_or_ps(_mm_and_ps(mask, r), _mm_andnot_ps(mask, result.dist));

    // enter from -z
    mask = _mm_and_ps(_mm_cmpgt_ps(dist1.z, zero), _mm_cmpgt_ps(dir.z, zero));
    r = _mm_div_ps(dist1.z, dir.z);
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, r));
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.x, min.x));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.x, max.x));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.y, min.y));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.y, max.y));
    result.hit = _mm_or_ps(result.hit, mask);
    comp = _mm_cmplt_ps(r, result.dist);
    mask = _mm_and_ps(mask, comp);
    result.dist = _mm_or_ps(_mm_and_ps(mask, r), _mm_andnot_ps(mask, result.dist));

    // enter from +z
    mask = _mm_and_ps(_mm_cmpgt_ps(dist2.z, zero), _mm_cmplt_ps(dir.z, zero));
    r = _mm_div_ps(dist2.z, _mm_sub_ps(zero, dir.z));
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, r));
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, r));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.x, min.x));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.x, max.x));
    mask = _mm_and_ps(mask, _mm_cmpgt_ps(point.y, min.y));
    mask = _mm_and_ps(mask, _mm_cmplt_ps(point.y, max.y));
    result.hit = _mm_or_ps(result.hit, mask);
    comp = _mm_cmplt_ps(r, result.dist);
    mask = _mm_and_ps(mask, comp);
    result.dist = _mm_or_ps(_mm_and_ps(mask, r), _mm_andnot_ps(mask, result.dist));

    return result;
}

void __vectorcall trace_rays(
    const raytracer_config* config,
    vec3 start, vec3 dir,
    __m128i output_x, __m128i output_y,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count,
    const box_data* boxes, int box_count)
{
    __m128 hit = _mm_setzero_ps();
    __m128 nearest = _mm_set1_ps(1000.f);
    __m128i nearest_index = _mm_setzero_si128();
    __m128i nearest_type = _mm_setzero_si128();

    for (int i = 0; i < sphere_count; ++i)
    {
        __m128i index = _mm_set1_epi32(i);
        __m128 sphere = _mm_load_ps((const float*)(spheres + i));

        // TODO: evaluate: _mm_prefetch((const char*)(spheres + i + 1), _MM_HINT_T0);

        test_result result = test_sphere(start, dir, sphere);

        // check if any of the dists were less than the previous nearest.
        // combine that with the hit mask though, so we ignore results that
        // didn't hit.
        __m128 mask = _mm_cmplt_ps(result.dist, nearest);
        mask = _mm_and_ps(result.hit, mask);

        __m128i imask = _mm_castps_si128(mask);

        // now keep the old nearests for items that didn't pass the combined mask,
        // and add in the new nearests for the elements that did pass the mask
        nearest =
            _mm_or_ps(
                _mm_and_ps(mask, result.dist),
                _mm_andnot_ps(mask, nearest));

        nearest_index =
            _mm_or_si128(
                _mm_and_si128(imask, index),
                _mm_andnot_si128(imask, nearest_index));

        // update our overall hit bit
        hit = _mm_or_ps(result.hit, hit);
    }

    __m128i type = _mm_set1_epi32(1); // triangle
    for (int i = 0; i < triangle_count; ++i)
    {
        __m128i index = _mm_set1_epi32(i);
        const triangle_data* triangle = triangles + i;

        __m128 v1 = _mm_load_ps(triangle->v1);
        __m128 v2 = _mm_load_ps(triangle->v2);
        __m128 v3 = _mm_load_ps(triangle->v3);
        __m128 norm = _mm_load_ps(triangle->normal);

        // TODO: evaluate: _mm_prefetch((const char*)(triangles + i + 1), _MM_HINT_T0);

        test_result result = test_triangle(start, dir, v1, v2, v3, norm);

        // check if any of the dists were less than the previous nearest.
        // combine that with the hit mask though, so we ignore results that
        // didn't hit.
        __m128 mask = _mm_cmplt_ps(result.dist, nearest);
        mask = _mm_and_ps(result.hit, mask);

        __m128i imask = _mm_castps_si128(mask);

        // now keep the old nearests for items that didn't pass the combined mask,
        // and add in the new nearests for the elements that did pass the mask
        nearest =
            _mm_or_ps(
                _mm_and_ps(mask, result.dist),
                _mm_andnot_ps(mask, nearest));

        nearest_index =
            _mm_or_si128(
                _mm_and_si128(imask, index),
                _mm_andnot_si128(imask, nearest_index));

        nearest_type =
            _mm_or_si128(
                _mm_and_si128(imask, type),
                _mm_andnot_si128(imask, nearest_type));

        // update our overall hit bit
        hit = _mm_or_ps(result.hit, hit);
    }

    type = _mm_set1_epi32(2); // box
    for (int i = 0; i < box_count; ++i)
    {
        __m128i index = _mm_set1_epi32(i);
        const box_data* box = boxes + i;

        __m128 min = _mm_load_ps(box->min);
        __m128 max = _mm_load_ps(box->max);

        // TODO: evaluate: _mm_prefetch((const char*)(triangles + i + 1), _MM_HINT_T0);

        test_result result = test_box(start, dir, min, max);

        // check if any of the dists were less than the previous nearest.
        // combine that with the hit mask though, so we ignore results that
        // didn't hit.
        __m128 mask = _mm_cmplt_ps(result.dist, nearest);
        mask = _mm_and_ps(result.hit, mask);

        __m128i imask = _mm_castps_si128(mask);

        // now keep the old nearests for items that didn't pass the combined mask,
        // and add in the new nearests for the elements that did pass the mask
        nearest =
            _mm_or_ps(
                _mm_and_ps(mask, result.dist),
                _mm_andnot_ps(mask, nearest));

        nearest_index =
            _mm_or_si128(
                _mm_and_si128(imask, index),
                _mm_andnot_si128(imask, nearest_index));

        nearest_type =
            _mm_or_si128(
                _mm_and_si128(imask, type),
                _mm_andnot_si128(imask, nearest_type));

        // update our overall hit bit
        hit = _mm_or_ps(result.hit, hit);
    }

    vec3 point;
    point.x = _mm_add_ps(start.x, _mm_mul_ps(dir.x, nearest));
    point.y = _mm_add_ps(start.y, _mm_mul_ps(dir.y, nearest));
    point.z = _mm_add_ps(start.z, _mm_mul_ps(dir.z, nearest));

    float point_x[4], point_y[4], point_z[4];
    _mm_storeu_ps(point_x, point.x);
    _mm_storeu_ps(point_y, point.y);
    _mm_storeu_ps(point_z, point.z);

    int x[4], y[4];
    _mm_storeu_si128((__m128i*)x, output_x);
    _mm_storeu_si128((__m128i*)y, output_y);

    int nearest_i[4];
    _mm_storeu_si128((__m128i*)nearest_i, nearest_index);

    int nearest_t[4];
    _mm_storeu_si128((__m128i*)nearest_t, nearest_type);

    int hit_mask = _mm_movemask_ps(hit);
    for (int bit = 0; bit < 4; ++bit)
    {
        if (hit_mask & 0x01)
        {
            float norm[3];
            bool skip = false;
                
            // if it was a sphere that was hit, compute normal
            if (nearest_t[bit] == 0)
            {
                // get the sphere that was hit
                const sphere_data* sphere = spheres + nearest_i[bit];

                // compute normal
                norm[0] = point_x[bit] - sphere->center[0];
                norm[1] = point_y[bit] - sphere->center[1];
                norm[2] = point_z[bit] - sphere->center[2];

                normalize(norm);
            }
            else if (nearest_t[bit] == 1)
            {
                // triangle, just read the normal
                const triangle_data* triangle = triangles + nearest_i[bit];
                norm[0] = triangle->normal[0];
                norm[1] = triangle->normal[1];
                norm[2] = triangle->normal[2];
            }
            else
            {
                // box, just use temp one for now
                int output_index = y[bit] * config->render_target_pitch + x[bit];
                config->render_target[output_index] = 0xFFFF0000;
                skip = true;
            }

            if (!skip)
            {
                // rasterize this ray
                int output_index = y[bit] * config->render_target_pitch + x[bit];
                config->render_target[output_index] = compute_color(norm);
            }
        }
        hit_mask >>= 1;
    }
}

__forceinline uint32_t compute_color(const float norm[3])
{
    // cheesy n dot l lighting for a hard coded light
    static const float light_dir[3] = { -0.577f, 0.577f, -0.577f };
    static const float light_color[3] = { 0.f, 0.f, 1.f };

    float n_dot_l = dot(norm, light_dir);
    if (n_dot_l < 0.f) n_dot_l = 0.f;
    if (n_dot_l > 1.f) n_dot_l = 1.f;
    float color[3] = { light_color[0] * n_dot_l, light_color[1] * n_dot_l, light_color[2] * n_dot_l };

    return 0xFF000000 |
        ((uint32_t)(uint8_t)(color[2] * 255.f)) << 16 |
        ((uint32_t)(uint8_t)(color[1] * 255.f)) << 8 |
        ((uint32_t)(uint8_t)(color[0] * 255.f));
}

void cull_objects(
    const raytracer_config* config,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count)
{
    // ensure we have enough room
    if (final_sphere_capacity < sphere_count)
    {
        final_sphere_capacity = sphere_count;
        delete[] final_spheres;
        final_spheres = new sphere_data[final_sphere_capacity];
    }

    if (final_triangle_capacity < triangle_count)
    {
        final_triangle_capacity = triangle_count;
        delete[] final_triangles;
        final_triangles = new triangle_data[final_triangle_capacity];
    }

    final_sphere_count = 0;
    final_triangle_count = 0;

    // compute the clip planes

    float view_right[3];
    cross(config->view_up, config->view_forward, view_right);

    float right_dir[3];
    float left_dir[3];
    for (int i = 0; i < 3; ++i)
    {
        right_dir[i] =
            view_right[i] * config->half_render_target_width +
            config->view_up[i] * config->half_render_target_height +
            config->view_forward[i] * config->dist_to_plane;
        left_dir[i] =
            -view_right[i] * config->half_render_target_width +
            config->view_up[i] * config->half_render_target_height +
            config->view_forward[i] * config->dist_to_plane;
    }

    float right_plane[3], left_plane[3];
    cross(config->view_up, right_dir, right_plane);
    cross(left_dir, config->view_up, left_plane);
    normalize(right_plane);
    normalize(left_plane);

    // pack 4 clip planes together. we skip top/bottom for now
    vec3 clip_planes; // far, near, left, right
    __m128 clip_dists;

    clip_planes.x = _mm_set_ps(right_plane[0], left_plane[0], -config->view_forward[0], config->view_forward[0]);
    clip_planes.y = _mm_set_ps(right_plane[1], left_plane[1], -config->view_forward[1], config->view_forward[1]);
    clip_planes.z = _mm_set_ps(right_plane[2], left_plane[2], -config->view_forward[2], config->view_forward[2]);

    vec3 pos;
    pos.x = _mm_set1_ps(config->view_position[0]);
    pos.y = _mm_set1_ps(config->view_position[1]);
    pos.z = _mm_set1_ps(config->view_position[2]);

    clip_dists = dot(clip_planes, pos);
    __m128 offsets = _mm_set_ps(0.f, 0.f, -1.f, 1000.f);
    clip_dists = _mm_add_ps(clip_dists, offsets);

    cull_spheres(clip_planes, clip_dists, spheres, sphere_count);
    cull_triangles(clip_planes, clip_dists, triangles, triangle_count);
}

void cull_spheres(
    vec3 clip_planes, __m128 clip_dists,
    const sphere_data* spheres, int sphere_count)
{
    for (int i = 0; i < sphere_count; ++i)
    {
        __m128 sphere = _mm_load_ps((const float*)(spheres + i));
        vec3 center = expand(sphere);
        __m128 radius_squared =
            _mm_shuffle_ps(sphere, sphere, _MM_SHUFFLE(3, 3, 3, 3));
        __m128 r = _mm_sqrt_ps(radius_squared);

        __m128 d = dot(clip_planes, center);
        d = _mm_sub_ps(d, clip_dists);
        __m128 mask = _mm_cmple_ps(d, r);

        // if outside of any plane, mask will have at least some 0s
        if (_mm_test_all_ones(_mm_castps_si128(mask)))
        {
            final_spheres[final_sphere_count++] = *(spheres + i);
        }
    }
}

void cull_triangles(
    vec3 clip_planes, __m128 clip_dists,
    const triangle_data* triangles, int triangle_count)
{
    for (int i = 0; i < triangle_count; ++i)
    {
        const triangle_data* triangle = triangles + i;
        __m128 v1 = _mm_load_ps(triangle->v1);
        __m128 v2 = _mm_load_ps(triangle->v2);
        __m128 v3 = _mm_load_ps(triangle->v3);
        vec3 a = expand(v1);
        vec3 b = expand(v2);
        vec3 c = expand(v3);

        __m128 d1 = dot(clip_planes, a);
        __m128 d2 = dot(clip_planes, b);
        __m128 d3 = dot(clip_planes, c);

        d1 = _mm_sub_ps(d1, clip_dists);
        d2 = _mm_sub_ps(d1, clip_dists);
        d3 = _mm_sub_ps(d1, clip_dists);

        __m128 zero = _mm_setzero_ps();
        __m128 mask = _mm_cmple_ps(d1, zero);
        mask = _mm_and_ps(mask, _mm_cmple_ps(d2, zero));
        mask = _mm_and_ps(mask, _mm_cmple_ps(d3, zero));

        // if outside of any plane, mask will have at least some 0s
        if (_mm_test_all_ones(_mm_castps_si128(mask)))
        {
            final_triangles[final_triangle_count++] = *(triangles + i);
        }
    }
}

void mt_init()
{
    if (shutdown_event)
    {
        // already initialized
        return;
    }

    shutdown_event = CreateEvent(nullptr, TRUE, FALSE, nullptr);
    complete_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    thread_handles = new HANDLE[num_threads];
    thread_contexts = new thread_context[num_threads];

    for (int i = 0; i < num_threads; ++i)
    {
        thread_contexts[i].id = i;
        thread_contexts[i].work_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        thread_handles[i] = CreateThread(nullptr, 0, thread_proc, thread_contexts + i, 0, nullptr);
    }
}

DWORD CALLBACK thread_proc(void* context)
{
    thread_context* self = (thread_context*)context;
    HANDLE handles[] = { shutdown_event, self->work_event };
    DWORD result = WaitForMultipleObjects(_countof(handles), handles, FALSE, INFINITE);
    while (result != WAIT_OBJECT_0)
    {
        long i = InterlockedExchangeAdd(&current_work_item, 1);
        while (i < work_items_count)
        {
            thread_work_item* work = (thread_work_item*)&work_items[i];

            int y1 = work->y;
            int y2 = work->y + work->height;
            if (y2 > work->config->render_target_height) y2 = work->config->render_target_height;

            int x1 = work->x;
            int x2 = work->x + work->width;
            if (x2 > work->config->render_target_width) x2 = work->config->render_target_width;

            // process in 2x2 blocks of rays
            for (int y = y1; y < y2; y += 2)
            {
                __m128i output_y = _mm_set_epi32(y + 1, y + 1, y, y);

                for (int x = x1; x < x2; x += 2)
                {
                    __m128i output_x = _mm_set_epi32(x + 1, x, x + 1, x);

                    vec3 start, dir;

                    // TODO: don't support view_forward/up yet
                    start.x = _mm_set1_ps(work->config->view_position[0]);
                    start.y = _mm_set1_ps(work->config->view_position[1]);
                    start.z = _mm_set1_ps(work->config->view_position[2]);

                    float base_x = x - work->config->half_render_target_width;
                    float base_y = work->config->half_render_target_height - y;

                    dir.x = _mm_set_ps(base_x + 1, base_x, base_x + 1, base_x);
                    dir.y = _mm_set_ps(base_y - 1, base_y - 1, base_y, base_y);
                    dir.z = _mm_set1_ps(work->config->dist_to_plane);

                    dir = normalize(dir);

                    if (work->scene)
                    {
                        trace_rays(work->config,
                            start, dir,
                            output_x, output_y,
                            work->scene);
                    }
                    else
                    {
#ifdef CULL_ENABLED
                        trace_rays(work->config,
                            start, dir,
                            output_x, output_y,
                            final_spheres, final_sphere_count,
                            final_triangles, final_triangle_count);
#else
                        trace_rays(work->config,
                            start, dir,
                            output_x, output_y,
                            work->spheres, work->sphere_count,
                            work->triangles, work->triangle_count,
                            work->boxes, work->box_count);
#endif
                    }
                }
            }

            i = InterlockedExchangeAdd(&current_work_item, 1);
        }

        i = InterlockedIncrement(&work_complete_gate);
        if (i == num_threads)
        {
            // last thread through the gate, signal completion
            SetEvent(complete_event);
        }

        result = WaitForMultipleObjects(_countof(handles), handles, FALSE, INFINITE);
    }

    return 0;
}

bool test_box(const float start[3], const float dir[3], const float min[3], const float max[3], float* out_dist)
{
    // for each axis, see if we enter the box
    float point[3];
    float nearest = FLT_MAX;
    bool hit = false;
    for (int i = 0; i < 3; ++i)
    {
        float dist = min[i] - start[i];
        if (dist > 0 && dir[i] > 0)
        {
            float r = dist / dir[i];
            point[0] = start[0] + dir[0] * r;
            point[1] = start[1] + dir[1] * r;
            point[2] = start[2] + dir[2] * r;

            // check other axes
            for (int j = 0; j < 3; ++j)
            {
                if (i == j) continue;
                if (point[j] > min[j] && point[j] < max[j])
                {
                    // hit
                    hit = true;
                    if (r < nearest)
                    {
                        nearest = r;
                    }
                }
            }
        }

        dist = start[i] - max[i];
        if (dist > 0 && dir[i] < 0)
        {
            float r = dist / -dir[i];
            point[0] = start[0] + dir[0] * r;
            point[1] = start[1] + dir[1] * r;
            point[2] = start[2] + dir[2] * r;

            // check other axes
            for (int j = 0; j < 3; ++j)
            {
                if (i == j) continue;
                if (point[j] > min[j] && point[j] < max[j])
                {
                    // hit
                    hit = true;
                    if (r < nearest)
                    {
                        nearest = r;
                    }
                }
            }
        }
    }

    *out_dist = nearest;
    return hit;
}

static void get_aabb(const triangle_data* triangle, float min[3], float max[3])
{
    for (int i = 0; i < 3; ++i)
    {
        min[i] = std::min(triangle->v1[i], std::min(triangle->v2[i], triangle->v3[i]));
        max[i] = std::max(triangle->v1[i], std::max(triangle->v2[i], triangle->v3[i]));
    }
}

static float get_growth_volume(
    const float min[3], const float max[3],
    const triangle_data* triangle)
{
    float growth[3];
    for (int i = 0; i < 3; ++i)
    {
        growth[i] = std::max(0.f, min[i] - std::min(triangle->v1[i], std::min(triangle->v2[i], triangle->v3[i])));
        growth[i] += std::max(0.f, std::max(triangle->v1[i], std::max(triangle->v2[i], triangle->v3[i])) - max[i]);
    }
    return growth[0] * growth[1] * growth[2];
}

static aabb_node* insert_triangle(
    aabb_node* node, const triangle_data* triangle)
{
    if (!node)
    {
        aabb_node* new_node = new aabb_node;
        new_node->children[0] = nullptr;
        new_node->triangles[0] = *triangle;
        new_node->num_triangles = 1;
        get_aabb(triangle, new_node->min, new_node->max);
        return new_node;
    }
    else if (node->children[0] == nullptr)
    {
        // leaf, does it have room?
        if (node->num_triangles < _countof(node->triangles))
        {
            node->triangles[node->num_triangles++] = *triangle;
            float min[3], max[3];
            get_aabb(triangle, min, max);
            for (int i = 0; i < 3; ++i)
            {
                node->min[i] = std::min(node->min[i], min[i]);
                node->max[i] = std::max(node->max[i], max[i]);
            }

            return node;
        }
        
        // no? split it
        aabb_node* new_leaf = new aabb_node;
        new_leaf->children[0] = nullptr;
        new_leaf->triangles[0] = *triangle;
        new_leaf->num_triangles = 1;
        get_aabb(triangle, new_leaf->min, new_leaf->max);

        aabb_node* new_inner = new aabb_node;
        new_inner->children[0] = node;
        new_inner->children[1] = new_leaf;
        for (int i = 0; i < 3; ++i)
        {
            new_inner->min[i] = std::min(node->min[i], new_leaf->min[i]);
            new_inner->max[i] = std::max(node->max[i], new_leaf->max[i]);
        }
        return new_inner;
    }
    else
    {
        // inner node, whichever child grows least
        if (get_growth_volume(node->children[0]->min, node->children[0]->max, triangle)
            < get_growth_volume(node->children[1]->min, node->children[1]->max, triangle))
        {
            node->children[0] = insert_triangle(node->children[0], triangle);
        }
        else
        {
            node->children[1] = insert_triangle(node->children[1], triangle);
        }

        for (int i = 0; i < 3; ++i)
        {
            node->min[i] = std::min(node->children[0]->min[i], node->children[1]->min[i]);
            node->max[i] = std::max(node->children[0]->max[i], node->children[1]->max[i]);
        }
        return node;
    }
}

aabb_node* __stdcall tt_build_aabb_tree(
    const triangle_data* triangles, int triangle_count)
{
    aabb_node* root = nullptr;
    for (int i = 0; i < triangle_count; ++i)
    {
        root = insert_triangle(root, &triangles[i]);
    }
    return root;
}


static void __vectorcall trace_rays(
    const raytracer_config* config,
    vec3 start, vec3 dir,
    __m128i output_x, __m128i output_y,
    const aabb_node* scene);

void __stdcall tt_trace(
    const raytracer_config* config,
    const aabb_node* scene)
{
    assert(config && scene);

#ifdef MT_ENABLED

    // divide rt into 4x4
    int width = (config->render_target_width + 3) / 4;
    int height = (config->render_target_height + 3) / 4;
    // ensure they are even (since we do 2x2s)
    if (width % 2 != 0) ++width;
    if (height % 2 != 0) ++height;
    for (int i = 0; i < 16; ++i)
    {
        work_items[i].config = config;
        work_items[i].x = (i % 4) * width;
        work_items[i].y = (i / 4) * height;
        work_items[i].width = width;
        work_items[i].height = height;
        work_items[i].spheres = nullptr;
        work_items[i].sphere_count = 0;
        work_items[i].triangles = nullptr;
        work_items[i].triangle_count = 0;
        work_items[i].scene = scene;
    }

    current_work_item = 0;
    work_complete_gate = 0;
    for (int i = 0; i < num_threads; ++i)
    {
        SetEvent(thread_contexts[i].work_event);
    }

    WaitForSingleObject(complete_event, INFINITE);

#else
    // process in 2x2 blocks of rays
    for (int y = 0; y < config->render_target_height; y += 2)
    {
        __m128i output_y = _mm_set_epi32(y + 1, y + 1, y, y);

        for (int x = 0; x < config->render_target_width; x += 2)
        {
            __m128i output_x = _mm_set_epi32(x + 1, x, x + 1, x);

            vec3 start, dir;

            // TODO: don't support view_forward/up yet
            start.x = _mm_set1_ps(config->view_position[0]);
            start.y = _mm_set1_ps(config->view_position[1]);
            start.z = _mm_set1_ps(config->view_position[2]);

            float base_x = x - config->half_render_target_width;
            float base_y = config->half_render_target_height - y;

            dir.x = _mm_set_ps(base_x + 1, base_x, base_x + 1, base_x);
            dir.y = _mm_set_ps(base_y - 1, base_y - 1, base_y, base_y);
            dir.z = _mm_set1_ps(config->dist_to_plane);

            dir = normalize(dir);

            trace_rays(config,
                start, dir,
                output_x, output_y,
                scene);
        }
    }
#endif // MT_ENABLED
}

static test_result __vectorcall test_node(vec3 start, vec3 dir, const aabb_node* node)
{
    test_result total;
    total.hit = _mm_setzero_ps();
    total.dist = _mm_set1_ps(FLT_MAX);

    if (node->children[0] == nullptr)
    {
        // leaf, process triangles
        for (int i = 0; i < node->num_triangles; ++i)
        {
            const triangle_data* triangle = node->triangles + i;

            __m128 v1 = _mm_load_ps(triangle->v1);
            __m128 v2 = _mm_load_ps(triangle->v2);
            __m128 v3 = _mm_load_ps(triangle->v3);
            __m128 norm = _mm_load_ps(triangle->normal);

            test_result result = test_triangle(start, dir, v1, v2, v3, norm);

            // check if any of the dists were less than the previous nearest.
            // combine that with the hit mask though, so we ignore results that
            // didn't hit.
            __m128 mask = _mm_cmplt_ps(result.dist, total.dist);
            mask = _mm_and_ps(result.hit, mask);

            // now keep the old nearests for items that didn't pass the combined mask,
            // and add in the new nearests for the elements that did pass the mask
            total.dist =
                _mm_or_ps(
                    _mm_and_ps(mask, result.dist),
                    _mm_andnot_ps(mask, total.dist));

            // update our overall hit bit
            total.hit = _mm_or_ps(result.hit, total.hit);
        }
    }
    else
    {
        // inner node. check any child that the ray hits
        __m128 min1 = _mm_load_ps(node->children[0]->min);
        __m128 max1 = _mm_load_ps(node->children[0]->max);
        test_result res1 = test_box(start, dir, min1, max1);
        int mask1 = _mm_movemask_ps(res1.hit);
        if (mask1)
        {
            res1 = test_node(start, dir, node->children[0]);
        }

        __m128 min2 = _mm_load_ps(node->children[1]->min);
        __m128 max2 = _mm_load_ps(node->children[1]->max);
        test_result res2 = test_box(start, dir, min2, max2);
        int mask2 = _mm_movemask_ps(res2.hit);
        if (mask2)
        {
            res2 = test_node(start, dir, node->children[1]);
        }

        __m128 mask = _mm_cmplt_ps(res1.dist, total.dist);
        mask = _mm_and_ps(res1.hit, mask);
        total.dist =
            _mm_or_ps(
                _mm_and_ps(mask, res1.dist),
                _mm_andnot_ps(mask, total.dist));

        mask = _mm_cmplt_ps(res2.dist, total.dist);
        mask = _mm_and_ps(res2.hit, mask);
        total.dist =
            _mm_or_ps(
                _mm_and_ps(mask, res2.dist),
                _mm_andnot_ps(mask, total.dist));

        // update our overall hit bit
        total.hit = _mm_or_ps(res1.hit, total.hit);
        total.hit = _mm_or_ps(res2.hit, total.hit);
    }

    return total;
}

void __vectorcall trace_rays(
    const raytracer_config* config,
    vec3 start, vec3 dir,
    __m128i output_x, __m128i output_y,
    const aabb_node* scene)
{
    test_result result = test_node(start, dir, scene);

    int x[4], y[4];
    _mm_storeu_si128((__m128i*)x, output_x);
    _mm_storeu_si128((__m128i*)y, output_y);

    int hit_mask = _mm_movemask_ps(result.hit);
    for (int bit = 0; bit < 4; ++bit)
    {
        if (hit_mask & 0x01)
        {
            // box, just use temp one for now
            int output_index = y[bit] * config->render_target_pitch + x[bit];
            config->render_target[output_index] = 0xFFFF0000;
        }
        hit_mask >>= 1;
    }
}

