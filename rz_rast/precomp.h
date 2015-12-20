#pragma once

#include <memory.h>
#include <stdint.h>
#include <assert.h>
#include <nmmintrin.h>

#define USE_LRB
#define USE_SSE
#define USE_SSE_LERP_AND_PS
#define USE_MT_TILES // multithreaded tile processing
#define USE_FULL_PS
#define RENDER_MANY
#define ENABLE_ANIMATION

#define UNREFERENCED(x) (x)