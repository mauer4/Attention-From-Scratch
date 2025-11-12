#include "custom_engine/utils/model_paths.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace custom_engine::utils {

namespace {

std::string trim(const std::string& input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    const auto end = input.find_last_not_of(" \t\r\n");
    if (begin == std::string::npos || end == std::string::npos) {
        return {};
    }
    return input.substr(begin, end - begin + 1);
}

std::string parse_value(const std::filesystem::path& config_path, const std::string& key) {
    std::ifstream file(config_path);
    if (!file) {
        return {};
    }
    std::string line;
    const std::string prefix = key + ":";
    while (std::getline(file, line)) {
        const auto hash = line.find('#');
        if (hash != std::string::npos) {
            line.erase(hash);
        }
        line = trim(line);
        if (line.rfind(prefix, 0) == 0) {
            std::string value = trim(line.substr(prefix.size()));
            if (!value.empty() && (value.front() == '"' || value.front() == '\'')) {
                value.erase(value.begin());
            }
            if (!value.empty() && (value.back() == '"' || value.back() == '\'')) {
                value.pop_back();
            }
            return value;
        }
    }
    return {};
}

}  // namespace

std::filesystem::path determine_project_root() {
#ifdef PROJECT_ROOT_DIR
    return std::filesystem::path(PROJECT_ROOT_DIR);
#else
    return std::filesystem::current_path();
#endif
}

std::filesystem::path config_yaml_path(const std::filesystem::path& project_root) {
    return project_root / "config" / "config.yaml";
}

std::filesystem::path resolve_snapshot_dir(const std::filesystem::path& project_root) {
    const auto config_path = config_yaml_path(project_root);
    const std::string variant = parse_value(config_path, "model_variant");
    if (variant.empty()) {
        return {};
    }
    std::filesystem::path snapshot = project_root / "weights";
    snapshot /= std::filesystem::path(variant);
    return snapshot;
}

}  // namespace custom_engine::utils
