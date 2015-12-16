#pragma once

bool RastStartup(uint32_t width, uint32_t height);
void RastShutdown();

bool RenderScene(void* const pOutput, uint32_t rowPitch);
