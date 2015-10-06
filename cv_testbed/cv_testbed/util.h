#pragma once

// RAII helper for CoInitializeEx/CoUninitialize
struct CoInit
{
    HRESULT hr;
    CoInit(DWORD coInit) { hr = CoInitializeEx(nullptr, coInit); }
    ~CoInit() { if (SUCCEEDED(hr)) { CoUninitialize(); } }
};

