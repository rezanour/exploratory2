#pragma once

#include <Windows.h>
#include <d3d11.h>
#include <wrl.h>
#include <DirectXMath.h>

#include <stdint.h>
#include <stdarg.h>
#include <assert.h>

#include <memory>
#include <vector>

// Yes, this is evil. This is a minimal hobby project so I don't care
using namespace Microsoft::WRL;
using namespace Microsoft::WRL::Wrappers;
using namespace DirectX;

// Place this immediately inside the opening brace, before any access modifiers, of the class to make it noncopyable
#define NONCOPYABLE(className) \
className(const className&); \
className& operator= (const className&);

