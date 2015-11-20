#include "Precomp.h"
#include "HDRFile.h"
#include "Debug.h"

using Microsoft::WRL::Wrappers::FileHandle;

static const char* ReadLine(const char* p, const char* end, char* buffer, size_t bufferSize);
static void DecodeRLEOld(const char* pCompressed, const char* pEnd, uint32_t* pixels, uint32_t width, uint32_t height);
static void DecodeRLENew(const char* pCompressed, const char* pEnd, uint32_t* pixels, uint32_t width, uint32_t height);

ComPtr<ID3D11Texture2D> HDRLoadImage(const ComPtr<ID3D11Device>& device, const wchar_t* filename)
{
    UNREFERENCED_PARAMETER(device);

    FileHandle file(CreateFile(filename, GENERIC_READ, FILE_SHARE_READ,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr));
    FAIL_IF_FALSE(file.IsValid(), L"Failed to open %s. %d", filename, GetLastError());

    LARGE_INTEGER fileSize{};
    FAIL_IF_FALSE(GetFileSizeEx(file.Get(), &fileSize), L"Failed to get file size. %d", GetLastError());

    assert(fileSize.HighPart == 0); // ReadFile below doesn't handle 64-bit size

    std::unique_ptr<char[]> buffer(new char[fileSize.QuadPart]);

    DWORD bytesRead = 0;
    FAIL_IF_FALSE(ReadFile(file.Get(), buffer.get(), fileSize.LowPart, &bytesRead, nullptr), L"Failed to read from file. %d", GetLastError());

    const char* p = buffer.get();
    const char* pEnd = p + fileSize.LowPart;
    char line[512]{};

    p = ReadLine(p, pEnd, line, _countof(line));
    FAIL_IF_FALSE(strncmp(line, "#?RADIANCE", 10) == 0, L"File doesn't appear to a valid RADIANCE .hdr file.");

    bool rle = false;

    // All text lines up to first blank line are header info
    OutputDebugString(L"HDR File Header:\n");
    while (strlen(line) > 0)
    {
        if (_strnicmp(line, "FORMAT=", 7) == 0)
        {
            // Parse the format
            if (_strnicmp(line + 7, "32-bit_rle_rgbe", 15) == 0)
            {
                rle = true;
            }
            else
            {
                // Don't support other types yet
                FAIL(L"Unsupported HDR file color format: %s", line + 7);
            }
        }

        OutputDebugStringA(line);
        OutputDebugString(L"\n");
        p = ReadLine(p, pEnd, line, _countof(line));
    }

    // Resolution string
    p = ReadLine(p, pEnd, line, _countof(line));
    int32_t xAxis = 1, yAxis = 1;
    uint32_t width = 0, height = 0;
    char xAxisString[3]{};
    char yAxisString[3]{};
    sscanf_s(line, "%s %u %s %u", yAxisString, _countof(yAxisString), &height, xAxisString, _countof(xAxisString), &width);
    xAxis = xAxisString[0] == '-' ? -xAxis : xAxis;
    yAxis = yAxisString[0] == '-' ? -yAxis : yAxis;

    FAIL_IF_FALSE(width > 0 && height > 0, L"Invalid image dimensions. %ux%u", width, height);

    std::unique_ptr<uint32_t[]> pixels(new uint32_t[width * height]);
    memset(pixels.get(), 0, width * height * sizeof(uint32_t));

    if (rle)
    {
        // if next 2 bytes are = 0x0202, then this is new style RLE
        if (*reinterpret_cast<const uint16_t*>(p) == 0x0202)
        {
            DecodeRLENew(p, pEnd, pixels.get(), width, height);
        }
        else
        {
            DecodeRLEOld(p, pEnd, pixels.get(), width, height);
        }
    }
    else
    {
        FAIL(L"Non RLE path not supported yet.");
    }

#pragma pack (push, 1)
    struct float4
    {
        float r, g, b, a;
    };
#pragma pack (pop)

    std::unique_ptr<float4[]> expanded(new float4[width * height]);
    memset(expanded.get(), 0, sizeof(float4) * width * height);
    for (uint32_t i = 0; i < width * height; ++i)
    {
        uint32_t exp = (pixels[i] & 0xFF000000) >> 24;
        if (exp != 0)
        {
            static const int exposure_hack = 6; // default 8 in ray sample
            double f = ldexp(1.0, (int32_t)exp - (128 + exposure_hack));
            expanded[i].r = (float)(((pixels[i] & 0x000000FF) + 0.5f) * f);
            expanded[i].g = (float)((((pixels[i] & 0x0000FF00) >> 8) + 0.5f) * f);
            expanded[i].b = (float)((((pixels[i] & 0x00FF0000) >> 16) + 0.5f) * f);
        }
    }

    D3D11_TEXTURE2D_DESC td{};
    td.ArraySize = 1;
    td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;

    D3D11_SUBRESOURCE_DATA init{};
    init.pSysMem = expanded.get();
    init.SysMemPitch = width * sizeof(float4);
    init.SysMemSlicePitch = init.SysMemPitch * height;

    ComPtr<ID3D11Texture2D> texture;
    HRESULT hr = device->CreateTexture2D(&td, &init, &texture);
    FAIL_IF_FALSE(SUCCEEDED(hr), L"Failed to create texture. 0x%08x", hr);

    return texture;
}

const char* ReadLine(const char* p, const char* end, char* buffer, size_t bufferSize)
{
    const char* current = p;
    size_t count = 0;

    assert(bufferSize > 0);
    buffer[0] = 0;

    while (count < bufferSize && current < end && *current != '\n')
    {
        buffer[count] = *current;
        ++count;
        ++current;
    }

    if (*current != '\n')
    {
        if (count == bufferSize)
        {
            FAIL(L"Ran out of buffer space parsing line in HDR file");
        }
        else if (current == end)
        {
            FAIL(L"HDR file appears to be truncated");
        }
        else
        {
            assert(false); // shouldn't be able to get here
        }
    }

    if (count < bufferSize)
    {
        buffer[count] = 0;
    }

    return (current + 1);
}

void DecodeRLEOld(const char* pCompressed, const char* pEnd, uint32_t* pixels, uint32_t width, uint32_t height)
{
    uint32_t i = 0;
    uint32_t size = width * height;
    const uint32_t* p = reinterpret_cast<const uint32_t*>(pCompressed);
    const uint32_t* pEnd32 = reinterpret_cast<const uint32_t*>(pEnd);

    // WARNING: Untested!
    assert(false);
    while (p < pEnd32 && i < size)
    {
        // Fetch a value
        uint32_t value = *p;
        ++p;

        if ((value & 0x00FFFFFF) == 0x00FFFFFF)
        {
            // RLE value
            FAIL_IF_FALSE(i > 0, L"Invalid image data. First value can't be RLE coding.");
            int32_t repeat = (value & 0xFF000000) >> 24;
            value = pixels[i - 1];

            // Repease last value that many times
            for (int32_t r = 0; r < repeat; ++r)
            {
                pixels[i + r] = value;
            }

            i += repeat;
        }
        else
        {
            pixels[i] = value;
            ++i;
        }
    }

    FAIL_IF_FALSE(i == size, L"Truncated image data.");
}

void DecodeRLENew(const char* pCompressed, const char* pEnd, uint32_t* pixels, uint32_t width, uint32_t height)
{
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pCompressed);
    const uint8_t* pEnd8 = reinterpret_cast<const uint8_t*>(pEnd);

    for (uint32_t row = 0; row < height; ++row)
    {
        uint32_t shifts[] = { 0, 8, 16, 24 };

        // ensure each scan line is encoded as "new" RLE
        FAIL_IF_FALSE(*reinterpret_cast<const uint16_t*>(p) == 0x0202, L"Non new-style RLE mixed in.");
        p += 2;

        uint32_t scanlineLength = (uint32_t)*p << 8;
        ++p;
        scanlineLength |= (uint32_t)*p;
        ++p;

        FAIL_IF_FALSE(scanlineLength == width, L"Invalid scanline size");

        for (int32_t channel = 0; channel < 4; ++channel)
        {
            uint32_t iScan = 0;
            while (p < pEnd8 && iScan < scanlineLength)
            {
                // Fetch a value
                uint8_t value = *p;
                ++p;

                if (value > 128)
                {
                    // RLE value
                    int32_t repeat = value & 127;
                    value = *p;
                    ++p;;

                    for (int32_t rep = 0; rep < repeat; ++rep)
                    {
                        pixels[row * width + iScan + rep] |= (uint32_t)value << shifts[channel];
                    }

                    iScan += repeat;
                }
                else
                {
                    int32_t repeat = value;

                    for (int32_t rep = 0; rep < repeat; ++rep)
                    {
                        pixels[row * width + iScan + rep] |= (uint32_t)*p << shifts[channel];
                        ++p;
                    }

                    iScan += repeat;
                }
            }

            FAIL_IF_FALSE(iScan == scanlineLength, L"Truncated image data.");
        }
    }
}

