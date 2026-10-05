#pragma once
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ninfer::exl3 {
// Campaign experiments. Snapshot once per owner before allocation/capture;
// never read environment or allocate from a captured forward path.
enum class Gopt : unsigned {
    GdnSmallResident, GdnConvTrace, GdnOutputPack, GdnVerifierResidualNorm,
    DraftRopePair, DraftHeadNormRope, DraftConvResidual, DraftResidualNorm,
    GreedyWarp, DraftDenseRowPair, GdnControlRowPair, DraftRingNormRope, Count
};
inline constexpr std::array<const char*,static_cast<unsigned>(Gopt::Count)> gopt_flags{{
    "NINFER_GOPT_001", "NINFER_GOPT_002", "NINFER_GOPT_003", "NINFER_GOPT_004",
    "NINFER_GOPT_005", "NINFER_GOPT_006", "NINFER_GOPT_007", "NINFER_GOPT_008",
    "NINFER_GOPT_009", "NINFER_GOPT_010", "NINFER_GOPT_011",
    "NINFER_GOPT_012"
}};
using GoptSubmissions = std::array<std::uint64_t,static_cast<unsigned>(Gopt::Count)>;
struct Exl3GamingOptions {
    std::array<bool,static_cast<unsigned>(Gopt::Count)> enabled{};
    bool operator[](Gopt id) const noexcept { return enabled[static_cast<unsigned>(id)]; }
    static bool parse(const char* value) {
        if(!value || std::strcmp(value,"0")==0) return false;
        if(std::strcmp(value,"1")==0) return true;
        throw std::invalid_argument("GOPT flags must be unset, 0 or 1");
    }
    static Exl3GamingOptions from_environment() {
        Exl3GamingOptions result;
        for(unsigned i=0;i<gopt_flags.size();++i) {
            const char* value=std::getenv(gopt_flags[i]);
            // GOPT-002 (convolution trace published from the convolution
            // kernel, bit-identical) is the measured default.
            result.enabled[i]=!value && i==static_cast<unsigned>(Gopt::GdnConvTrace)
                ? true : parse(value);
        }
        // Both consume the same attn-finish residual. Test them separately.
        if(result[Gopt::DraftConvResidual] && result[Gopt::DraftResidualNorm])
            throw std::invalid_argument("GOPT-007 and GOPT-008 are exclusive");
        return result;
    }
};
inline bool gopt_small_gdn(int rows,bool preserve_m1,bool wide,bool storage) noexcept {
    return rows>=2 && rows<=8 && preserve_m1 && !wide && storage;
}
inline void gopt_record(GoptSubmissions& counters,Gopt id) noexcept {
    ++counters[static_cast<unsigned>(id)];
}
// Counts successful host launch submissions, including capture submissions;
// never a claim of graph replays, device completion, or useful output tokens.
}
