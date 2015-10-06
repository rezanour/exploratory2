#include "precomp.h"
#include "imagefile.h"
#include <wincodec.h>
#include <wrl.h>

using namespace Microsoft::WRL;
using namespace Microsoft::WRL::Wrappers;

std::unique_ptr<uint32_t[]> load_image(const wchar_t* filename, bool asBGRA, int* out_width, int* out_height)
{
    assert(out_width && out_height);
    *out_width = 0;
    *out_height = 0;

    ComPtr<IWICImagingFactory2> factory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory2, nullptr,
        CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    ComPtr<IWICBitmapDecoder> decoder;
    hr = factory->CreateDecoderFromFilename(filename, nullptr, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    ComPtr<IWICBitmapFrameDecode> bitmapFrame;
    hr = decoder->GetFrame(0, &bitmapFrame);
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    ComPtr<IWICFormatConverter> converter;
    hr = factory->CreateFormatConverter(&converter);
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    hr = converter->Initialize(bitmapFrame.Get(), asBGRA ? GUID_WICPixelFormat32bppBGRA : GUID_WICPixelFormat32bppRGBA,
        WICBitmapDitherTypeNone, nullptr, 0, WICBitmapPaletteTypeCustom);
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    uint32_t width = 0, height = 0;
    hr = bitmapFrame->GetSize(&width, &height);
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    std::unique_ptr<uint32_t[]> pixels(new uint32_t[width * height]);
    hr = converter->CopyPixels(nullptr, width * sizeof(uint32_t), width * height * sizeof(uint32_t), (uint8_t*)pixels.get());
    if (FAILED(hr))
    {
        assert(false);
        return nullptr;
    }

    *out_width = (int)width;
    *out_height = (int)height;
    return pixels;
}
