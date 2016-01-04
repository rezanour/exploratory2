#pragma once

#include <stdint.h>

struct alignas(16) sphere_data
{
    float center[3];
    float radius_squared;
};

struct alignas(16) triangle_data
{
    float v1[3]; float pad1;
    float v2[3]; float pad2;
    float v3[3]; float pad3;
    float normal[3]; float pad4;
};

struct alignas(16) box_data
{
    float min[3]; float pad1;
    float max[3]; float pad2;
};

struct alignas(16) aabb_node
{
    float min[3]; float pad1;
    float max[3]; float pad2;

    aabb_node* children[2];
    triangle_data triangles[32];
    int num_triangles;
};

struct raytracer_config
{
    // render target info
    uint32_t* render_target;
    int render_target_width;
    int render_target_height;
    int render_target_pitch;    // in pixels
    float half_render_target_width;
    float half_render_target_height;

    // view info
    float view_position[3];
    float dist_to_plane;
    float view_forward[3];
    float view_up[3];
};

void __stdcall tt_build_aabb_tree(
    const triangle_data* triangles, int triangle_count,
    aabb_node** node_heap, aabb_node** root_node);

void __stdcall tt_setup(
    raytracer_config* config,
    uint32_t* render_target, int width, int height, int pitch,
    float fov, const float position[3]);

void __stdcall tt_trace(
    const raytracer_config* config,
    const sphere_data* spheres, int sphere_count,
    const triangle_data* triangles, int triangle_count,
    const box_data* boxes, int box_count);

void __stdcall tt_trace(
    const raytracer_config* config,
    const aabb_node* scene);
