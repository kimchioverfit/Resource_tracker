#pragma once
#include <string>

struct NvmlGpuMetrics {
    std::string name;
    double temperatureC   = 0.0;
    double gpuUtilPercent = 0.0;
    double memUtilPercent = 0.0;
    double memUsedMB      = 0.0;
    double memTotalMB     = 0.0;
    double powerDrawW     = 0.0;
    double pcieRxKBs      = 0.0;
    double pcieTxKBs      = 0.0;
};

class NvmlHelper {
public:
    bool init();
    bool queryFirstGpu(NvmlGpuMetrics& out);
};
