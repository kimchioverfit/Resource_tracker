#include "disk_helper.h"
#include <windows.h>

DiskInfo DiskHelper::querySystemDiskInfo() {
    DiskInfo di;

    ULARGE_INTEGER totalBytes{};
    ULARGE_INTEGER freeBytes{};
    ULARGE_INTEGER caller{};

    if (GetDiskFreeSpaceExA("C:\\", &caller, &totalBytes, &freeBytes)) {
        di.systemDiskSizeGB =
            totalBytes.QuadPart / (1024ull * 1024ull * 1024ull);
    }

    di.systemDiskModel = "SystemDisk";
    di.systemDiskInterface = "Unknown";

    return di;
}
