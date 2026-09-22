#include "exl3/mtp_proposal_economics.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int check(bool value, const char* message) {
    if (value) return 0;
    std::cerr << message << '\n';
    return 1;
}

std::filesystem::path fixture_directory() {
    const char* value = std::getenv("NINFER_COMPACT_PERF_DIR");
    return value != nullptr && *value != '\0' ? std::filesystem::path(value)
                                               : std::filesystem::path("<NINFER_COMPACT_PERF_DIR-unset>");
}

} // namespace

int main() {
    using namespace ninfer::exl3;

    const auto directory = fixture_directory();
    const auto code       = Exl3MtpFixtureIdentity::code(directory);
    const auto prose      = Exl3MtpFixtureIdentity::prose(directory);
    int failures          = 0;
    failures += check(code.valid() && prose.valid(), "MTP economics fixture identities are invalid");
    failures += check(code.kind_name() == "code" && prose.kind_name() == "prose",
                      "MTP economics fixture kinds are not explicit");
    failures += check(code.ids_path.filename() == std::filesystem::path("code.ids") &&
                          prose.ids_path.filename() == std::filesystem::path("prose.ids"),
                      "MTP economics fixture filenames changed");
    failures += check(code.prompt_tokens == 4'096 && prose.prompt_tokens == 4'096 &&
                          code.output_tokens == 128 && prose.output_tokens == 128,
                      "MTP economics fixture extent is not the selected DFlash2 4K/128 screen");

    // This contract target deliberately has no numeric executor or target
    // oracle.  It must produce a visible NOT_RUN record rather than turning
    // unavailable timing into zero or claiming a microbenchmark result.
    const Exl3MtpExecutionBinding binding{};
    const Exl3MtpStepInput input{};
    const Exl3MtpStepOutput output{};
    for (const auto& fixture : {code, prose}) {
        const auto result = Exl3MtpProposalEconomicsScreen::measure(
            fixture, binding, input, output, nullptr, nullptr);
        failures += check(result.status == Exl3MtpEconomicsStatus::NotRunMissingExecutor,
                          "missing MTP executor did not fail closed");
        failures += check(!result.measured() && !result.timing.proposal_cost_ms() &&
                              !result.timing.complete_round_ms(),
                          "unavailable MTP economics were reported as measured timing");
        std::cout << "MTP_PROPOSAL_ECONOMICS fixture=" << fixture.label
                  << " ids_path=" << fixture.ids_path.string()
                  << " prompt_tokens=" << fixture.prompt_tokens
                  << " output_tokens=" << fixture.output_tokens
                  << " status=" << result.status_name()
                  << " measurement_scope=one_step_fixture"
                  << " target_authority=required"
                  << " complete_ms_per_useful_token=null\n";
    }

    if (failures != 0) return 1;
    std::cout << "exl3_mtp_proposal_economics_contract: PASS executor_required=1"
                 " target_oracle_required=1 measured_rows=0\n";
    return 0;
}
