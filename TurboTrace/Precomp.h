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
#include <deque>
#include <algorithm>

typedef Microsoft::WRL::Wrappers::HandleT<Microsoft::WRL::Wrappers::HandleTraits::HANDLENullTraits> Thread;

#define NON_COPYABLE(className) \
    className(const className&) = delete;   \
    className& operator= (const className&) = delete;

#include "TTMath.h"
