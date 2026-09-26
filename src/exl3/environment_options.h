#pragma once

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::exl3 {

// Strict opt-in controls: absent means false; only literal 0 and 1 are valid.
// The caller supplies its existing diagnostic and chooses when to read the
// environment. Presence-only and permissive legacy controls are separate contracts.
inline bool parse_binary_option(const char* value, const char* diagnostic) {
    if (!value) return false;
    if (std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::invalid_argument(diagnostic);
}

inline bool read_binary_option(const char* name, const char* diagnostic) {
    return parse_binary_option(std::getenv(name), diagnostic);
}

} // namespace ninfer::exl3
