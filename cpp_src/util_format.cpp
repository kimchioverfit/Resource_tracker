#include "util_format.h"
#include <sstream>
#include <iomanip>

namespace util {

std::string formatDouble(double v, int precision) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss << std::setprecision(precision) << v;
    return oss.str();
}

std::string formatInt(long long v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

} // namespace util
