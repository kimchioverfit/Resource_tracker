#pragma once
#include <string>

struct DiskInfo {
    std::string systemDiskModel;
    std::string systemDiskInterface;
    long long   systemDiskSizeGB = 0;
};

class DiskHelper {
public:
    DiskHelper() = default;
    DiskInfo querySystemDiskInfo();
};
