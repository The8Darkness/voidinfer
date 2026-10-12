#pragma once

#include <voidinfer/targets/qwen3_6/frontend.h>

#include <string>

namespace voidinfer::targets::qwen3_6 {

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

} // namespace voidinfer::targets::qwen3_6
