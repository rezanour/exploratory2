#pragma once

// These functions handle the vertices from app -> rasterizer

#include "rz_math.h"

// Begins the graphics pipeline by issuing a draw of the provided vertices
void rz_draw(const VSConstants& constants, const Vertex* vertices, uint32_t num_verts);
