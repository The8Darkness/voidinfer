#include "voidinfer/engine.h"

#include <type_traits>

static_assert(std::is_move_constructible_v<voidinfer::PreparedPrompt>);
static_assert(!std::is_copy_constructible_v<voidinfer::PreparedPrompt>);
static_assert(std::is_move_constructible_v<voidinfer::Engine>);
static_assert(!std::is_copy_constructible_v<voidinfer::Engine>);

int main() {
    const voidinfer::EngineOptions options;
    return options.vision_enabled() ? 1 : 0;
}
