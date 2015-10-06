#include "precomp.h"
#include "edge_detect.h"

std::unique_ptr<uint32_t[]> detect_edges(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height)
{
    std::unique_ptr<uint32_t[]> output(new uint32_t[width * height]);
    if (!output)
    {
        assert(false);
        return nullptr;
    }
    memset(output.get(), 0, width * height * sizeof(uint32_t));

#if 0 // Use luminance
    std::unique_ptr<float[]> lum = convert_to_luminance(input, isBGRA, width, height);
    if (!lum)
    {
        assert(false);
        return nullptr;
    }

    // naive brute force, look for high variance between neighboring pixels
    for (int y = 0; y < height - 1; ++y)
    {
        for (int x = 0; x < width - 1; ++x)
        {
            float self = lum[y * width + x];
            float right = lum[y * width + (x + 1)];
            float down = lum[(y + 1) * width + x];

            if (fabs(down - self) > 0.02f || fabs(right - self) > 0.02f)
            {
                output[y * width + x] = 0xFFFFFFFF;
            }
        }
    }

#else // use color distance
    UNREFERENCED_PARAMETER(isBGRA);

#if MEASURE_PERF
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    for (int i = 0; i < 100; ++i)
    {
#endif

    __m128i mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0);
    float* lengths = (float*)_aligned_malloc(4 * sizeof(float), 16);

    // naive brute force, look for high variance between neighboring pixels
    for (int y = 0; y < height - 1; ++y)
    {
        for (int x = 0; x < width - 1; ++x)
        {
            uint32_t self = input[y * width + x];
            uint32_t right = input[y * width + (x + 1)];
            uint32_t down = input[(y + 1) * width + x];

#if 0 // FPU
            UNREFERENCED_PARAMETER(mask);
            UNREFERENCED_PARAMETER(lengths);

            float distRightX = (float)(right & 0x000000FF) - (float)(self & 0x000000FF);
            float distRightY = (float)((right & 0x0000FF00) >> 8) - (float)((self & 0x0000FF00) >> 8);
            float distRightZ = (float)((right & 0x00FF0000) >> 16) - (float)((self & 0x00FF0000) >> 16);
            float distDownX = (float)(down & 0x000000FF) - (float)(self & 0x000000FF);
            float distDownY = (float)((down & 0x0000FF00) >> 8) - (float)((self & 0x0000FF00) >> 8);
            float distDownZ = (float)((down & 0x00FF0000) >> 16) - (float)((self & 0x00FF0000) >> 16);

            float distRightSq = distRightX * distRightX + distRightY * distRightY + distRightZ * distRightZ;
            float distDownSq = distDownX * distDownX + distDownY * distDownY + distDownZ * distDownZ;

            if (sqrt(distDownSq) > 10.f || sqrt(distRightSq) > 10.f)
            {
                output[y * width + x] = 0xFFFFFFFF;
            }
#else // SSE
            __m128i selfi = _mm_shuffle_epi8(_mm_set1_epi32(self), mask);
            __m128i righti = _mm_shuffle_epi8(_mm_set1_epi32(right), mask);
            __m128i downi = _mm_shuffle_epi8(_mm_set1_epi32(down), mask);

            __m128 selff = _mm_cvtepi32_ps(selfi);
            __m128 rightf = _mm_cvtepi32_ps(righti);
            __m128 downf = _mm_cvtepi32_ps(downi);

            __m128 diffRight = _mm_sub_ps(rightf, selff);
            __m128 diffDown = _mm_sub_ps(downf, selff);

            __m128 lengthRight = _mm_mul_ps(diffRight, diffRight);
            __m128 lengthDown = _mm_mul_ps(diffDown, diffDown);
            __m128 adder = _mm_hadd_ps(lengthRight, lengthDown);
            adder = _mm_hadd_ps(adder, adder);
            __m128 length = _mm_sqrt_ps(adder);
            _mm_store_ps(lengths, length);

            if (lengths[0] > 10.f || lengths[1] > 10.f)
            {
                output[y * width + x] = 0xFFFFFFFF;
            }
#endif
        }
    }
    _aligned_free(lengths);
#endif
#if MEASURE_PERF
    }
    QueryPerformanceCounter(&end);

    float totalTime = (end.QuadPart - start.QuadPart) / (float)freq.QuadPart;
    wchar_t message[100];
    swprintf_s(message, L"Time: %3.2fms", totalTime * 1000.f);
    MessageBox(nullptr, message, L"Perf", MB_OK);
#endif

    return output;
}

std::unique_ptr<float[]> convert_to_luminance(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height)
{
    static const float inv256 = 1.0f / 256.0f;

    std::unique_ptr<float[]> lum(new float[width * height]);
    if (!lum)
    {
        assert(false);
        return nullptr;
    }

#ifdef MEASURE_LUMINANCE_PERF
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    for (int i = 0; i < 1000; ++i)
    {
#endif
#if 0

        uint32_t* p = input.get();
        float* l = lum.get();

        if (isBGRA)
        {
            for (int i = 0; i < width * height; ++i, ++p, ++l)
            {
                uint32_t c = *p;
                float b = (float)(c & 0x000000FF) * inv256;
                float g = (float)((c & 0x0000FF00) >> 8) * inv256;
                float r = (float)((c & 0x00FF0000) >> 16) * inv256;
                *l = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            }
        }
        else
        {
            for (int i = 0; i < width * height; ++i, ++p, ++l)
            {
                uint32_t c = *p;
                float r = (float)(c & 0x000000FF) * inv256;
                float g = (float)((c & 0x0000FF00) >> 8) * inv256;
                float b = (float)((c & 0x00FF0000) >> 16) * inv256;
                *l = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            }
        }

#else
        __m128i extract_r = _mm_set_epi8(-1, -1, -1, 12, -1, -1, -1, 8, -1, -1, -1, 4, -1, -1, -1, 0);
        __m128i extract_g = _mm_set_epi8(-1, -1, -1, 13, -1, -1, -1, 9, -1, -1, -1, 5, -1, -1, -1, 1);
        __m128i extract_b = _mm_set_epi8(-1, -1, -1, 14, -1, -1, -1, 10, -1, -1, -1, 6, -1, -1, -1, 2);

        __m128 r_factor = _mm_set1_ps(0.2126f * inv256);
        __m128 g_factor = _mm_set1_ps(0.7152f * inv256);
        __m128 b_factor = _mm_set1_ps(0.0722f * inv256);

        const __m128i* src = (const __m128i*)input.get();
        float* dst = lum.get();

        // stream 4 pixels at a time
        const int num_blocks = width * height / 4;
        for (int i = 0; i < num_blocks; ++i)
        {
            // read in block of 4 pixels
            __m128i block = _mm_loadu_si128(src++);

            // extract out r, g, and b channels into 3 separate registers
            __m128i r_int = _mm_shuffle_epi8(block, isBGRA ? extract_b : extract_r);
            __m128i g_int = _mm_shuffle_epi8(block, extract_g);
            __m128i b_int = _mm_shuffle_epi8(block, isBGRA ? extract_r : extract_b);

            // convert from int to float
            __m128 r = _mm_cvtepi32_ps(r_int);
            __m128 g = _mm_cvtepi32_ps(g_int);
            __m128 b = _mm_cvtepi32_ps(b_int);

            // multiply by the factors (with the div by 256 baked in)
            r = _mm_mul_ps(r, r_factor);
            g = _mm_mul_ps(g, g_factor);
            b = _mm_mul_ps(b, b_factor);

            // sum them up to get the 4 luminance values
            __m128 l = _mm_add_ps(r, g);
            l = _mm_add_ps(l, b);

            // write them out
            _mm_storeu_ps(dst, l);
            dst += 4;
        }
#endif
#ifdef MEASURE_LUMINANCE_PERF
    }
    QueryPerformanceCounter(&end);

    float totalTime = (end.QuadPart - start.QuadPart) / (float)freq.QuadPart;
    wchar_t message[100];
    swprintf_s(message, L"Time: %3.2fms", totalTime * 1000.f);
    MessageBox(nullptr, message, L"Perf", MB_OK);
#endif

    return lum;
}