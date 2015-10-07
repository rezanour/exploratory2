#include "Precomp.h"
#include "DebugUtil.h"

void __cdecl DbgOut(const wchar_t* formatString, ...)
{
    wchar_t message[1024]{};

    va_list args;
    va_start(args, formatString);
    vswprintf_s(message, formatString, args);
    va_end(args);

    OutputDebugString(message);
}

