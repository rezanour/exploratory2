#pragma once

void __cdecl DbgOut(const wchar_t* formatString, ...);

#ifdef _DEBUG

#define FAIL(formatString, ...) { DbgOut(L"ERROR: " formatString L"\n", __VA_ARGS__); assert(false); throw std::exception(); }

#else // RELEASE

#define FAIL(formatString, ...) { throw std::exception(); }

#endif // DEBUG/RELEASE

#define FAIL_IF_NULL(x, formatString, ...) if (!(x)) { FAIL(formatString, __VA_ARGS__); }
#define CHECKHR(x, formatString, ...) if (FAILED(x)) { FAIL(formatString, __VA_ARGS__); }
