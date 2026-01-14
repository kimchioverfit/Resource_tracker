#include "util.h"

#include <fstream>
#include <regex>
#include <sstream>

std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_csv_line(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            in_quotes = !in_quotes;
            continue;
        }
        if (c == ',' && !in_quotes) {
            out.push_back(trim(cur));
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(trim(cur));
    return out;
}

std::string read_entire_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

std::vector<std::string> parse_string_array(const std::string &json, const std::string &key) {
    std::vector<std::string> out;
    auto pos = json.find(key);
    if (pos == std::string::npos) return out;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return out;
    auto end = json.find(']', pos);
    if (end == std::string::npos) return out;
    std::string inner = json.substr(pos + 1, end - pos - 1);
    std::regex re("\"([^\"]+)\"");
    std::sregex_iterator it(inner.begin(), inner.end(), re);
    std::sregex_iterator endit;
    for (; it != endit; ++it) out.push_back((*it)[1].str());
    return out;
}

std::vector<double> parse_number_array(const std::string &json, const std::string &key) {
    std::vector<double> out;
    auto pos = json.find(key);
    if (pos == std::string::npos) return out;
    pos = json.find('[', pos);
    if (pos == std::string::npos) return out;
    auto end = json.find(']', pos);
    if (end == std::string::npos) return out;
    std::string inner = json.substr(pos + 1, end - pos - 1);
    std::regex re(R"([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::sregex_iterator it(inner.begin(), inner.end(), re);
    std::sregex_iterator endit;
    for (; it != endit; ++it) out.push_back(std::stod((*it)[0].str()));
    return out;
}
