#pragma once
#include <string>

struct CpuInfo {
    std::string modelName;
    int physicalCores   = 0;
    int logicalProcessors = 0;
    double baseClockMHz = 0.0;
};

struct MemoryInfo {
    long long totalRamMB = 0;
};

class WmiHelper {
public:
    WmiHelper();
    ~WmiHelper();

    CpuInfo   queryCpuInfo();
    MemoryInfo queryMemoryInfo();

private:
    bool initialized_ = false;
    void initCom_();
    void uninitCom_();
};
