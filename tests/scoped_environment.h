#pragma once

#include <cstdlib>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test {

// Windows qualification fixtures mutate process-local controls synchronously.
// Capture before any mutation and restore in the original key order, including
// removal of a previously absent variable (_putenv_s(name, "")).
class ScopedEnvironmentRestore {
public:
    explicit ScopedEnvironmentRestore(std::initializer_list<const char*> names) {
        saved_.reserve(names.size());
        for (const char* name : names) {
            const char* value = std::getenv(name);
            saved_.emplace_back(name, value ? value : "");
        }
    }
    ScopedEnvironmentRestore(const ScopedEnvironmentRestore&) = delete;
    ScopedEnvironmentRestore& operator=(const ScopedEnvironmentRestore&) = delete;
    ~ScopedEnvironmentRestore() {
        for (const auto& [name, value] : saved_)
            _putenv_s(name.c_str(), value.c_str());
    }

private:
    std::vector<std::pair<std::string, std::string>> saved_;
};

} // namespace ninfer::test
