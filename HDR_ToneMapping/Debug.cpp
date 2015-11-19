#include "Precomp.h"
#include "Debug.h"

void __cdecl FailFast()
{
    assert(false);
    exit(0);
}

void __cdecl FailFast(const wchar_t* format, ...)
{
    wchar_t message[1024]{};

    va_list args;
    va_start(args, format);
    vswprintf_s(message, format, args);
    va_end(args);

    OutputDebugString(message);
    OutputDebugString(L"\n");

    FailFast();
}
