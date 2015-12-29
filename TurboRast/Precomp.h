#pragma once

#define NOMINMAX
#include <Windows.h>
#include <wrl.h>

#include <stdint.h>
#include <assert.h>
#include <nmmintrin.h>

#include <atomic>
#include <memory>
#include <vector>
#include <queue>
#include <mutex>

typedef Microsoft::WRL::Wrappers::HandleT<Microsoft::WRL::Wrappers::HandleTraits::HANDLENullTraits> Thread;

#include "TurboRastTypes.h"
