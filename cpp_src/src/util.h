#pragma once

#include <string>
#include <vector>

std::string trim(const std::string &s);
std::vector<std::string> split_csv_line(const std::string &line);
std::string read_entire_file(const std::string &path);
std::vector<std::string> parse_string_array(const std::string &json, const std::string &key);
std::vector<double> parse_number_array(const std::string &json, const std::string &key);
