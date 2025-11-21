#include "wmi_helper.h"
#include <windows.h>
#include <comdef.h>
#include <wbemidl.h>
#include <stdexcept>

#pragma comment(lib, "wbemuuid.lib")

static std::string queryCpuNameFromRegistry() {
    HKEY hKey;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
        "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
        0, KEY_READ, &hKey) != ERROR_SUCCESS) {
        return "Unknown CPU";
    }

    char buf[256];
    DWORD bufSize = sizeof(buf);
    if (RegGetValueA(hKey, nullptr, "ProcessorNameString",
        RRF_RT_REG_SZ, nullptr, buf, &bufSize) != ERROR_SUCCESS) {
        RegCloseKey(hKey);
        return "Unknown CPU";
    }

    RegCloseKey(hKey);
    return std::string(buf);
}

WmiHelper::WmiHelper() {
    initCom_();
}

WmiHelper::~WmiHelper() {
    uninitCom_();
}

void WmiHelper::initCom_() {
    HRESULT hr = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE) {
        initialized_ = true;
    }
}

void WmiHelper::uninitCom_() {
    if (initialized_) {
        CoUninitialize();
        initialized_ = false;
    }
}

CpuInfo WmiHelper::queryCpuInfo() {
    CpuInfo info;
    info.modelName = queryCpuNameFromRegistry();

    SYSTEM_INFO si{};
    GetSystemInfo(&si);

    info.logicalProcessors = static_cast<int>(si.dwNumberOfProcessors);
    info.physicalCores = info.logicalProcessors;
    info.baseClockMHz = 0.0;

    return info;
}

MemoryInfo WmiHelper::queryMemoryInfo() {
    MemoryInfo mi;

    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);

    if (GlobalMemoryStatusEx(&ms)) {
        mi.totalRamMB = static_cast<long long>(ms.ullTotalPhys / (1024ull * 1024ull));
    }

    return mi;
}
