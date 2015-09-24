#pragma once

#include "Util.h"

#define CHECKHR(x) { HRESULT hr##_LINE_ = (x); if (FAILED(hr##_LINE_)) { LogError(L#x L" failed (0x%08x)\n", hr##_LINE_); return hr##_LINE_; } }
#define CHECKGLE(x) { if (!(x)) { HRESULT hr##_LINE_ = HRESULT_FROM_WIN32(GetLastError()); LogError(L#x L" failed (0x%08x)\n", hr##_LINE_); return hr##_LINE_; } }
