#pragma once

enum class ToneMappingOperator
{
    None = 0,
    Linear,
    ReinhardRGB,
    ReinhardYOnly
};

void GraphicsStartup(HWND window);
void GraphicsShutdown();

const ComPtr<ID3D11Device>& GraphicsGetDevice();

void GraphicsClear();
void GraphicsPresent();

void GraphicsSetOperator(ToneMappingOperator op);
void GraphicsEnableGamma(bool enable);
bool GraphicsGammaEnabled();
void GraphicsSetExposure(float exposure);
void GraphicsSetHighPassThreshold(float threshold);
void GraphicsEnableHighPassBlur(bool enable);
bool GraphicsHighPassBlurEnabled();

void GraphicsDrawQuad(const RECT* dest, const ComPtr<ID3D11ShaderResourceView>& texture);
