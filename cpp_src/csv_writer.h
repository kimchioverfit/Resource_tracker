#pragma once
#include <string>
#include <vector>
#include <fstream>

class CsvWriter {
public:
    CsvWriter(const std::string& path, bool append);
    ~CsvWriter();

    void writeHeader(const std::vector<std::string>& cols);
    void writeRow(const std::vector<std::string>& cols);

private:
    std::ofstream ofs_;
    bool wroteHeader_ = false;

    void writeLine(const std::vector<std::string>& cols);
};
