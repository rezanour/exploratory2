#pragma once

ComPtr<ID3D11Texture2D> HDRLoadImage(const ComPtr<ID3D11Device>& device, const wchar_t* filename);
