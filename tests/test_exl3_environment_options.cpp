#include "exl3/environment_options.h"
#include "scoped_environment.h"

#include <iostream>
#include <string_view>

namespace {
void need(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void set(const char* name, const char* value) {
    need(_putenv_s(name, value) == 0, "test environment update failed");
}
}

int main() try {
    using ninfer::exl3::parse_binary_option;
    using ninfer::exl3::read_binary_option;
    using ninfer::test::ScopedEnvironmentRestore;
    constexpr const char* diagnostic = "original option diagnostic";
    need(!parse_binary_option(nullptr, diagnostic), "absent opt-in enabled");
    need(!parse_binary_option("0", diagnostic), "explicit zero enabled");
    need(parse_binary_option("1", diagnostic), "explicit one disabled");
    // Do not turn strict options into presence-only, integer or permissive flags.
    for (const char* value : {"", "00", "01", "2", "-1", "true", "false", " 1", "1 ", "1x"}) {
        bool refused = false;
        try { (void)parse_binary_option(value, diagnostic); }
        catch (const std::invalid_argument& error) {
            need(std::string_view(error.what()) == diagnostic, "diagnostic changed");
            refused = true;
        }
        need(refused, "malformed binary option accepted");
    }
    constexpr const char* key = "VOIDINFER_HOST_TEST_BINARY_OPTION";
    constexpr const char* absent = "VOIDINFER_HOST_TEST_ABSENT_OPTION";
    ScopedEnvironmentRestore original{key, absent};
    set(key, "original");
    set(absent, "");
    {
        ScopedEnvironmentRestore outer{key, absent};
        set(key, "1");
        set(absent, "created");
        need(read_binary_option(key, diagnostic), "environment enable not observed");
        bool observed_disabled = false;
        try {
            ScopedEnvironmentRestore inner{key};
            set(key, "0");
            observed_disabled = !read_binary_option(key, diagnostic);
            throw std::runtime_error("unwind fixture");
        } catch (const std::runtime_error&) {}
        need(observed_disabled, "environment change was cached");
        need(read_binary_option(key, diagnostic), "nested unwind lost outer value");
        set(key, "invalid");
        bool refused = false;
        try {
            // Setup validates before applying owner/dependency predicates.
            const bool requested = read_binary_option(key, diagnostic);
            const bool selected = false && requested;
            (void)selected;
        } catch (const std::invalid_argument&) { refused = true; }
        need(refused, "inactive route suppressed malformed-option validation");
    }
    need(std::string_view(std::getenv(key)) == "original", "scope lost saved value");
    need(std::getenv(absent) == nullptr, "scope failed to remove newly created variable");
    need(!read_binary_option(absent, diagnostic), "absent environment enabled");
    std::cout << "exl3_environment_options PASS\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
