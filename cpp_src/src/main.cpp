#include <torch/script.h>
#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static inline std::string trim(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::vector<std::string> split_csv_line(const std::string &line) {
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

static std::string read_entire_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static std::vector<std::string> parse_string_array(const std::string &json, const std::string &key) {
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

static std::vector<double> parse_number_array(const std::string &json, const std::string &key) {
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

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <meta.json> <model.pt> <input.csv> [output.csv]\n";
        return 1;
    }

    std::string meta_path = argv[1];
    std::string model_path = argv[2];
    std::string input_csv = argv[3];
    std::string output_csv = (argc >= 5) ? argv[4] : "";

    std::string meta_txt = read_entire_file(meta_path);
    if (meta_txt.empty()) {
        std::cerr << "Failed to read meta JSON: " << meta_path << "\n";
        return 1;
    }

    auto feature_cols = parse_string_array(meta_txt, "feature_cols");
    auto scaler_mean = parse_number_array(meta_txt, "scaler_mean");
    auto scaler_scale = parse_number_array(meta_txt, "scaler_scale");

    if (feature_cols.empty() || scaler_mean.empty() || scaler_scale.empty()) {
        std::cerr << "meta.json missing required fields (feature_cols/scaler_mean/scaler_scale)\n";
        return 1;
    }

    if (scaler_mean.size() != feature_cols.size() || scaler_scale.size() != feature_cols.size()) {
        std::cerr << "meta arrays length mismatch\n";
        return 1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
    } catch (const c10::Error &e) {
        std::cerr << "Error loading the model: " << e.what() << "\n";
        return 1;
    }

    std::ifstream ifs(input_csv);
    if (!ifs) {
        std::cerr << "Failed to open input CSV: " << input_csv << "\n";
        return 1;
    }

    std::ofstream ofs;
    if (!output_csv.empty()) {
        ofs.open(output_csv);
        if (!ofs) {
            std::cerr << "Failed to open output CSV: " << output_csv << "\n";
            return 1;
        }
    }

    std::string header_line;
    if (!std::getline(ifs, header_line)) {
        std::cerr << "Empty CSV input\n";
        return 1;
    }

    auto headers = split_csv_line(header_line);
    std::unordered_map<std::string, size_t> hdr_idx;
    for (size_t i = 0; i < headers.size(); ++i) hdr_idx[headers[i]] = i;

    std::vector<size_t> feature_idx;
    for (auto &fn : feature_cols) {
        if (hdr_idx.find(fn) == hdr_idx.end()) {
            std::cerr << "Feature column not found in CSV: " << fn << "\n";
            return 1;
        }
        feature_idx.push_back(hdr_idx[fn]);
    }

    if (ofs) {
        ofs << header_line << ",Prediction\n";
    }

    std::string line;
    std::vector<double> values(feature_cols.size());
    std::vector<float> input_buf(feature_cols.size());
    while (std::getline(ifs, line)) {
        if (trim(line).empty()) continue;
        auto cols = split_csv_line(line);
        if (cols.size() < headers.size()) continue;

        bool skip = false;
        for (size_t i = 0; i < feature_idx.size(); ++i) {
            size_t idx = feature_idx[i];
            std::string v = cols[idx];
            if (v.empty()) { skip = true; break; }
            try { values[i] = std::stod(v); } catch (...) { skip = true; break; }
            double scaled = (values[i] - scaler_mean[i]) / scaler_scale[i];
            input_buf[i] = static_cast<float>(scaled);
        }
        if (skip) continue;

        auto tensor = torch::from_blob(input_buf.data(), {1, (long)input_buf.size()}, torch::kFloat).clone();
        std::vector<torch::IValue> inputs;
        inputs.push_back(tensor);
        at::Tensor out;
        try {
            out = module.forward(inputs).toTensor();
        } catch (const c10::Error &e) {
            std::cerr << "Model forward error: " << e.what() << "\n";
            return 1;
        }

        float pred = 0.0f;
        if (out.numel() >= 1) pred = out.flatten()[0].item<float>();

        if (ofs) {
            ofs << line << "," << pred << "\n";
        } else {
            std::cout << pred << "\n";
        }
    }

    return 0;
}
