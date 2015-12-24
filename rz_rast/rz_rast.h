#pragma once

// Pipeline setup and drawing
void rz_SetRenderTarget(uint32_t* const pRenderTarget, int width, int height, int pitchPixels);
void rz_Draw(const VSConstants& constants, const Vertex* vertices, int numVerts);
