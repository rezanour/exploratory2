#pragma once

// Interpolate vertex attributes.
// in: first_vertex and next two subsequent vertices (by stride) make up triangle.
//     p (px, py) are the points to interpolate.
// out: writes 4 lerped attributes to outputs
void __vectorcall rz_lerp(const VertexOutput* first_vertex, const __m128& px, const __m128& py, __m128& mask, VertexOutput* outputs);
