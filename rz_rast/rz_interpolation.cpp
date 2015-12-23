#include "precomp.h"
#include "rz_common.h"
#include "rz_interpolation.h"

// Compute barycentric coordinates (lerp weights) for 4 samples at once.
// The computation is done in 2 dimensions (screen space).
// in: a (ax, ay), b (bx, by) and c (cx, cy) are the 3 vertices of the triangle.
//     p (px, py) is the point to compute barycentric coordinates for
// out: wA, wB, wC are the weights at vertices a, b, and c
//      mask will contain a 0 (clear) if the value is computed. It will be 0xFFFFFFFF (set) if invalid
static inline void __vectorcall rz_bary2d(
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

void __vectorcall rz_lerp(const VertexOutput* first_vertex, const __m128& px, const __m128& py, __m128& mask, VertexOutput* outputs)
{
    const float3* vA = (const float3*)&first_vertex->Position;
    const float3* vB = (const float3*)((const uint8_t*)vA + sizeof(VertexOutput));
    const float3* vC = (const float3*)((const uint8_t*)vA + 2 * sizeof(VertexOutput));

    __m128 ax = _mm_set1_ps(vA->x);
    __m128 ay = _mm_set1_ps(vA->y);
    __m128 bx = _mm_set1_ps(vB->x);
    __m128 by = _mm_set1_ps(vB->y);
    __m128 cx = _mm_set1_ps(vC->x);
    __m128 cy = _mm_set1_ps(vC->y);

    __m128 xA, xB, xC;
    rz_bary2d(ax, ay, bx, by, cx, cy, px, py, xA, xB, xC, mask);

    // Interpolate all the attributes for these 4 pixels
    __m128 posx = _mm_add_ps(_mm_mul_ps(ax, xA), _mm_add_ps(_mm_mul_ps(bx, xB), _mm_mul_ps(cx, xC)));
    __m128 posy = _mm_add_ps(_mm_mul_ps(ay, xA), _mm_add_ps(_mm_mul_ps(by, xB), _mm_mul_ps(cy, xC)));
    __m128 posz = _mm_setzero_ps();
    __m128 posw = _mm_set1_ps(1.f);

    const float3* cA = (const float3*)&first_vertex->Color;
    const float3* cB = (const float3*)((const uint8_t*)vA + sizeof(VertexOutput));
    const float3* cC = (const float3*)((const uint8_t*)vA + 2 * sizeof(VertexOutput));

    __m128 colx = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cA->x), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cB->x), xB), _mm_mul_ps(_mm_set1_ps(cC->x), xC)));
    __m128 coly = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cB->y), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cB->y), xB), _mm_mul_ps(_mm_set1_ps(cC->y), xC)));
    __m128 colz = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cC->z), xA), _mm_add_ps(_mm_mul_ps(_mm_set1_ps(cB->z), xB), _mm_mul_ps(_mm_set1_ps(cC->z), xC)));


}

