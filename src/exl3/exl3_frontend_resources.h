#pragma once

#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include <filesystem>

namespace ninfer::exl3 {

// Pinned frontend binding: checks metadata bytes, not model-shard payload identity.
// Ordinary chat_template.jinja is required; .fixed is never a fallback.
[[nodiscard]] targets::qwen3_6::FrontendResources
load_pinned_frontend_resources(const std::filesystem::path& target_directory);

} // namespace ninfer::exl3
