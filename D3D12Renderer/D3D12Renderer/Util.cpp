#include "Precomp.h"
#include "Util.h"

#define DBGOUT(format) \
    va_list args; \
    va_start(args, format); \
    wchar_t message[1024]{}; \
    vswprintf_s(message, format, args); \
    OutputDebugString(message); \
    va_end(args); \

void __cdecl Log(const wchar_t* format, ...)
{
    DBGOUT(format);
}

void __cdecl LogError(const wchar_t* format, ...)
{
    DBGOUT(format);
    assert(false);
}
