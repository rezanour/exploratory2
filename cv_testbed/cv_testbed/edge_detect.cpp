#include "precomp.h"
#include "edge_detect.h"

static std::unique_ptr<float[]> convert_to_luminance(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height);

std::unique_ptr<uint32_t[]> detect_edges(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height)
{
    std::unique_ptr<float[]> lum = convert_to_luminance(input, isBGRA, width, height);
    if (!lum)
    {
        assert(false);
        return nullptr;
    }

    std::unique_ptr<uint32_t[]> output(new uint32_t[width * height]);
    if (!output)
    {
        assert(false);
        return nullptr;
    }
    memset(output.get(), 0, width * height * sizeof(uint32_t));

    // naive brute force, look for high variance between neighboring pixels
    for (int y = 1; y < height - 1; ++y)
    {
        for (int x = 1; x < width - 1; ++x)
        {
            float left = lum[y * height + (x - 1)];
            float right = lum[y * height + (x + 1)];
            float up = lum[(y - 1) * height + x];
            float down = lum[(y + 1) * height + x];

#if 0
            float self = lum[y * height + x];
            if (fabs(down - self) > 0.2f || fabs(right - self) > 0.2f ||
                fabs(up - self) > 0.2f || fabs(left - self) > 0.2f)
            {
                output[y * height + x] = 0xFFFFFFFF;
            }
#else
            if (fabs(down - up) > 0.2f || fabs(right - left) > 0.2f)
            {
                output[y * height + x] = 0xFFFFFFFF;
            }
#endif
        }
    }

    return output;
}

std::unique_ptr<float[]> convert_to_luminance(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height)
{
#if 1
    static const float inv256 = 1.0f / 256.0f;

    std::unique_ptr<float[]> lum(new float[width * height]);
    if (!lum)
    {
        assert(false);
        return nullptr;
    }

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

    return lum;

#else // Not done
    // Our math is based on the data being in RGBA format, so use a shuffle to byte swap if
    // BGRA is passed in
    __m128i mask = (isBGRA ?
        _mm_set_epi8(15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2) :
        _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0));

    __m128i zero = _mm_setzero_si128();

    // stream 4 pixels at a time
    const int num_blocks = width * height / 4;
    for (int i = 0; i < num_blocks; ++i)
    {
        __m128i block = _mm_loadu_si128((const __m128i*)&input[i * 4]);
        __m128i swapped = _mm_shuffle_epi8(block, mask);

        __m128i ab = _mm_unpacklo_epi8(swapped, zero);
        __m128i a = _mm_unpacklo_epi16(ab, zero);
        __m128i b = _mm_unpackhi_epi16(ab, zero);
        __m128i cd = _mm_unpackhi_epi8(swapped, zero);
        __m128i c = _mm_unpacklo_epi16(cd, zero);
        __m128i d = _mm_unpackhi_epi16(cd, zero);

        __m128 fa = _mm_cvtepi32_ps(a);
        __m128 fb = _mm_cvtepi32_ps(b);
        __m128 fc = _mm_cvtepi32_ps(c);
        __m128 fd = _mm_cvtepi32_ps(d);

        _mm_stream_si128(dst++, _mm_or_si128(expanded, alpha));
    }
#endif
}