#pragma once

// Assumed to be 0xFFBBGGRR 32bpp RGBA unless isBGRA is specified
std::unique_ptr<float[]> convert_to_luminance(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height);

std::unique_ptr<uint32_t[]> detect_edges(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height);

std::unique_ptr<uint32_t[]> detect_interest_points(const std::unique_ptr<uint32_t[]>& input, bool isBGRA, int width, int height);

