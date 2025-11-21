#include "csv_writer.h"
#include <stdexcept>

CsvWriter::CsvWriter(const std::string& path, bool append) {
    std::ios::openmode mode = std::ios::out;
    if (append) mode |= std::ios::app;
    ofs_.open(path, mode);
    if (!ofs_.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + path);
    }
}

CsvWriter::~CsvWriter() {
    if (ofs_.is_open()) {
        ofs_.flush();
        ofs_.close();
    }
}

void CsvWriter::writeLine(const std::vector<std::string>& cols) {
    bool first = true;
    for (auto& c : cols) {
        if (!first) ofs_ << ",";
        first = false;

        bool needQuote =
            (c.find(',') != std::string::npos ||
             c.find('"') != std::string::npos);

        if (needQuote) {
            ofs_ << '"';
            for (char ch : c) {
                if (ch == '"') ofs_ << "\"\"";
                else ofs_ << ch;
            }
            ofs_ << '"';
        } else {
            ofs_ << c;
        }
    }
    ofs_ << "\n";
}

void CsvWriter::writeHeader(const std::vector<std::string>& cols) {
    if (!wroteHeader_) {
        writeLine(cols);
        wroteHeader_ = true;
    }
}

void CsvWriter::writeRow(const std::vector<std::string>& cols) {
    writeLine(cols);
}
