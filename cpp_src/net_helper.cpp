#include "net_helper.h"
#include <windows.h>
#include <iphlpapi.h>

#pragma comment(lib, "iphlpapi.lib")

NicInfo NetHelper::queryPrimaryNicInfo() {
    NicInfo ni;
    ni.name = "PrimaryNIC";
    return ni;
}
