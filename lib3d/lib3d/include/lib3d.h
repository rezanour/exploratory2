// lib3d.h
//
// Simple C library containing basic 3D math and geometry functionality.
//
// Reza Nourai

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

// NOTE: To enable SSE 4.1 optimized implementations where available, define
// LIB3D_ENABLE_SSE before including this header file.

#ifdef LIB3D_ENABLE_SSE
#include <nmmintrin.h>
#endif // LIB3D_ENABLE_SSE

// Since this header may be included by multiple source compilation units,
// it is necessary to add the inline keyword in front of each function. This is
// more to assist in compilation than it is to try and force the compiler to
// actually inline these.
#define LIB3D_INLINE inline

//
// Basic (boolean) intersection tests.
//

// Do the spheres [c1, r1] and [c2, r2] overlap?
// c1 and c2 are expected to be float triplets for x, y, and z
LIB3D_INLINE bool lib3d_basic_sphere_sphere(const float* c1, float r1, const float* c2, float r2)
{
    // find squared distance between two centers
    float diff[3];
    diff[0] = c1[0] - c2[0];
    diff[1] = c1[1] - c2[1];
    diff[2] = c1[2] - c2[2];
    float length_squared = diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2];
    // is it less than combined radius squared?
    float r = r1 + r2;
    return (length_squared <= (r * r));
}

// Does the ray [start, dir], intersect the sphere [center, r]?
// start, dir, and center are expected to be float triplets for x, y, and z
LIB3D_INLINE bool lib3d_basic_ray_sphere(const float* start, const float* dir, const float* center, float r)
{
    // find vector from start to center of sphere
    float start_to_center[3];
    start_to_center[0] = center[0] - start[0];
    start_to_center[1] = center[1] - start[1];
    start_to_center[2] = center[2] - start[2];
    // find squared length of this vector
    float length_squared = start_to_center[0] * start_to_center[0] + 
        start_to_center[1] * start_to_center[1] + start_to_center[2] * start_to_center[2];
    if (length_squared <= (r * r))
    {
        // inside sphere
        return true;
    }

    // projected distance of start_to_center in ray direction
    float proj_dist = start_to_center[0] * dir[0] + start_to_center[1] * dir[1] + start_to_center[2] * dir[2];
    if (proj_dist <= 0.f)
    {
        // ray pointing away from sphere
        return false;
    }

    // find squared distance from sphere center to ray
    float dist_to_ray_squared = length_squared - (proj_dist * proj_dist);
    return (dist_to_ray_squared <= (r * r));
}

// Does the ray [start, dir], intersect any of the spheres [c, r]?
LIB3D_INLINE bool lib3d_basic_ray_spheres(const float* start, const float* dir,
    const float* c, const float* r, int num_spheres)
{
#ifndef LIB3D_ENABLE_SSE

    for (int i = 0; i < num_spheres; ++i)
    {
        if (lib3d_basic_ray_sphere(start, dir, c + 3 * i, r[i]))
        {
            return true;
        }
    }
    return false;

#else

    __m128 sx = _mm_set_ps1(start[0]);
    __m128 sy = _mm_set_ps1(start[1]);
    __m128 sz = _mm_set_ps1(start[2]);
    __m128 dx = _mm_set_ps1(dir[0]);
    __m128 dy = _mm_set_ps1(dir[1]);
    __m128 dz = _mm_set_ps1(dir[2]);
    int i;
    for (i = 0; i < num_spheres - 4; i += 4)
    {
        const float* c1 = c + 3 * i;
        const float* c2 = c + 3 * (i + 1);
        const float* c3 = c + 3 * (i + 2);
        const float* c4 = c + 3 * (i + 3);
        __m128 center_x = _mm_set_ps(c1[0], c2[0], c3[0], c4[0]);
        __m128 center_y = _mm_set_ps(c1[1], c2[1], c3[1], c4[1]);
        __m128 center_z = _mm_set_ps(c1[2], c2[2], c3[2], c4[2]);
        __m128 radius = _mm_set_ps(r[i], r[i + 1], r[i + 2], r[i + 3]);

        // find vector from start to center of sphere
        __m128 start_to_center_x = _mm_sub_ps(center_x, sx);
        __m128 start_to_center_y = _mm_sub_ps(center_y, sy);
        __m128 start_to_center_z = _mm_sub_ps(center_z, sz);

        // find squared length of this vector
        __m128 length_squared = _mm_mul_ps(start_to_center_x, start_to_center_x);
        length_squared = _mm_add_ps(length_squared, _mm_mul_ps(start_to_center_y, start_to_center_y));
        length_squared = _mm_add_ps(length_squared, _mm_mul_ps(start_to_center_z, start_to_center_z));

        __m128 r_squared = _mm_mul_ps(radius, radius);
        __m128i comp = _mm_cvtps_epi32(_mm_cmple_ps(length_squared, r_squared));
        if (!_mm_testz_si128(comp, comp))
        {
            // at least one was inside
            return true;
        }

        // projected distance of start_to_center in ray direction
        __m128 proj_dist = _mm_mul_ps(start_to_center_x, dx);
        proj_dist = _mm_add_ps(proj_dist, _mm_mul_ps(start_to_center_y, dy));
        proj_dist = _mm_add_ps(proj_dist, _mm_mul_ps(start_to_center_z, dz));

        // mask only the ones that are moving towards sphere
        __m128 keep_mask = _mm_cmpgt_ps(proj_dist, _mm_setzero_ps());

        // find squared distance from sphere center to ray
        __m128 dist_to_ray_squared = _mm_sub_ps(length_squared, _mm_mul_ps(proj_dist, proj_dist));
        __m128 keep_mask2 = _mm_cmple_ps(dist_to_ray_squared, r_squared);
        comp = _mm_cvtps_epi32(_mm_and_ps(keep_mask, keep_mask2));
        if (!_mm_testz_si128(comp, comp))
        {
            // at least one hit
            return true;
        }
    }

    for (; i < num_spheres; ++i)
    {
        if (lib3d_basic_ray_sphere(start, dir, c + 3 * i, r[i]))
        {
            return true;
        }
    }

    return false;

#endif // LIB3D_ENABLE_SSE
}
