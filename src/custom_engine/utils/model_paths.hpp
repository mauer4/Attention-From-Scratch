#pragma once

#include <filesystem>

namespace custom_engine::utils {

std::filesystem::path determine_project_root();
std::filesystem::path config_yaml_path(const std::filesystem::path& project_root);
std::filesystem::path resolve_snapshot_dir(const std::filesystem::path& project_root);

}  // namespace custom_engine::utils

