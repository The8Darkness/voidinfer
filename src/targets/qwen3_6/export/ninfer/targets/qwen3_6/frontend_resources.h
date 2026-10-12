#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>

#include <string>

namespace ninfer::targets::qwen3_6 {

// The pinned frontend files of a model directory (tokenizer, chat template,
// generation and media preprocessing configuration), read verbatim.
struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
};

} // namespace ninfer::targets::qwen3_6
