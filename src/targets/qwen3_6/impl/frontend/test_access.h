#pragma once

#include <voidinfer/targets/qwen3_6/frontend.h>
#include <voidinfer/targets/qwen3_6/frontend_resources.h>
#include <voidinfer/targets/qwen3_6/prepared_prompt.h>

namespace voidinfer::targets::qwen3_6 {

class FrontendTestAccess {
public:
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   bool vision_enabled = true);
    [[nodiscard]] static Frontend create_component(const FrontendResources& resources,
                                                   FrontendOptions options);
    [[nodiscard]] static const PreparedPromptData& inspect(const PreparedPrompt& prompt);
};

} // namespace voidinfer::targets::qwen3_6
