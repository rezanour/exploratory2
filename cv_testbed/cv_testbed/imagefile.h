#pragma once

// loads image into standard 0xFFBBGGRR 32bit RGBA format unless asBGRA is specified as true
std::unique_ptr<uint32_t[]> load_image(const wchar_t* filename, bool asBGRA, int* out_width, int* out_height);
