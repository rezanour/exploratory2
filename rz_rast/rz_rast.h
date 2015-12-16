#pragma once

bool RastStartup();
void RastShutdown();

bool RenderScene(void* const pOutput, uint32_t width, uint32_t height, uint32_t rowPitch);
