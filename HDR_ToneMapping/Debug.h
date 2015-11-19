#pragma once

void __cdecl FailFast();
void __cdecl FailFast(const wchar_t* format, ...);

#ifdef _DEBUG
#define FAIL(format, ...) FailFast(format, __VA_ARGS__);
#else
#define FAIL(format, ...) FailFast();
#endif

#define FAIL_IF_NULL(x, format, ...) { if (!(x)) { FAIL(format, __VA_ARGS__); }}
#define FAIL_IF_FALSE(x, format, ...) { if (!(x)) { FAIL(format, __VA_ARGS__); }}
