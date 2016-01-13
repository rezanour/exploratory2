#include <stdio.h>
#include <wchar.h>

//#define LIB3D_ENABLE_SSE
#include "lib3d.h"

int wmain(int argc, wchar_t* argv[])
{
    argc;
    argv;

    float c1[3] = { -1.f, 0.f, 0.f };
    float c2[3] = { 1.f,  0.f, 0.f };
    float r1 = 1.f;
    float r2 = 1.f;

    if (lib3d_basic_sphere_sphere(c1, r1, c2, r2))
    {
        wprintf_s(L"spheres overlap.\n");
    }

    float start[3] = { 0, 0, 0 };
    float dir[3] = { 0, 0, 1 };

    float centers[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, };
    float radius[] = { 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, };

    if (lib3d_basic_ray_spheres(start, dir, centers, radius, 9))
    {
        wprintf_s(L"ray hit a sphere.\n");
    }

    return 0;
}
