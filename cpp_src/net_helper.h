#pragma once
#include <string>

struct NicInfo {
    std::string name;
};

class NetHelper {
public:
    NetHelper() = default;
    NicInfo queryPrimaryNicInfo();
};
