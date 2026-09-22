#pragma once
#include <array>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

namespace ninfer::exl3 {
// The compact device argmax is the qualified greedy-text default. Explicit 0
// retains the host-logit control and rollback path; explicit 1 remains
// accepted for pinned configurations.
inline bool exl3_device_greedy_enabled() {
    const auto* flag=std::getenv("NINFER_EXL3_DEVICE_GREEDY");
    if(flag && std::string_view(flag)!="0" && std::string_view(flag)!="1")
        throw std::invalid_argument("device greedy must be 0 or 1");
    return !flag || std::string_view(flag)=="1";
}

// Internal synchronous readback ABI, not a serialized root or sampling result.
struct Exl3GreedyRow {
    std::uint64_t serial = 0;
    std::int32_t token = -1;
    std::uint32_t nonfinite = 0;
};
struct Exl3GreedyPacket {
    static constexpr std::uint32_t version = 1;
    std::uint32_t abi = version, rows = 0;
    std::uint64_t generation = 0, serial = 0;
    int position = 0;
    bool ready = false;
    std::array<Exl3GreedyRow,16> decisions{};

    void validate(std::uint32_t expected_rows, std::uint64_t expected_generation,
                  int expected_position, std::uint64_t expected_serial,
                  int vocabulary = 248320) const {
        if(abi!=version || !ready || rows!=expected_rows || rows==0 || rows>16 ||
           generation!=expected_generation || position!=expected_position ||
           serial==0 || serial!=expected_serial || vocabulary<=0)
            throw std::runtime_error("stale/incomplete greedy packet");
        // Includes the final produced row, even when it cannot authorize a
        // tentative token. Matches the full-score host refusal contract.
        for(std::uint32_t i=0;i<rows;++i)
            if(decisions[i].serial!=serial || decisions[i].nonfinite ||
               decisions[i].token<0 || decisions[i].token>=vocabulary)
                throw std::runtime_error("invalid/nonfinite greedy packet row");
    }
};

// A sampled decision needs the complete per-row distribution and cannot be
// represented by the compact argmax packet above. This descriptor is only an
// authority boundary; production EXL3 dispatch remains greedy-only.
struct Exl3SampledDistributionAuthority {
    std::shared_ptr<const void> owner;
    const float* probabilities=nullptr;
    std::uint32_t rows=0,vocabulary=0,row_stride=0;
    std::uint64_t generation=0,serial=0;
    int position=0;
    bool ready=false;
};

enum class Exl3SelectionAuthorityKind : std::uint8_t {
    greedy_argmax,
    sampled_distribution
};

class Exl3SelectionAuthorityRequirement {
    Exl3SelectionAuthorityKind kind_=Exl3SelectionAuthorityKind::greedy_argmax;
    std::uint32_t rows_=0,vocabulary_=0;
    std::uint64_t generation_=0,serial_=0;
    int position_=0;
public:
    static Exl3SelectionAuthorityRequirement greedy(std::uint32_t rows,
        std::uint64_t generation,int position,std::uint64_t serial,
        std::uint32_t vocabulary=248320) {
        if(!rows || rows>16 || !vocabulary)
            throw std::invalid_argument("invalid greedy authority requirement");
        return {Exl3SelectionAuthorityKind::greedy_argmax,rows,vocabulary,
            generation,serial,position};
    }
    static Exl3SelectionAuthorityRequirement sampled(std::uint32_t rows,
        std::uint64_t generation,int position,std::uint64_t serial,
        std::uint32_t vocabulary=248320) {
        if(!rows || !vocabulary)
            throw std::invalid_argument("invalid sampled authority requirement");
        return {Exl3SelectionAuthorityKind::sampled_distribution,rows,vocabulary,
            generation,serial,position};
    }
    static Exl3SelectionAuthorityRequirement request(float temperature,
        float presence_penalty,float frequency_penalty) {
        return (temperature!=0 || presence_penalty!=0 || frequency_penalty!=0)?
            sampled(1,0,0,0):greedy(1,0,0,0);
    }
    Exl3SelectionAuthorityKind kind() const noexcept {return kind_;}
    bool needs_sampled_distribution() const noexcept {
        return kind_==Exl3SelectionAuthorityKind::sampled_distribution;
    }
    void validate(const Exl3GreedyPacket& packet) const {
        if(kind_!=Exl3SelectionAuthorityKind::greedy_argmax)
            throw std::invalid_argument(
                "compact argmax packet cannot satisfy sampled authority");
        packet.validate(rows_,generation_,position_,serial_,
            static_cast<int>(vocabulary_));
    }
    void validate(const Exl3SampledDistributionAuthority& distribution) const {
        if(kind_!=Exl3SelectionAuthorityKind::sampled_distribution)
            throw std::invalid_argument(
                "sampled distribution cannot satisfy greedy authority");
        if(!distribution.owner || !distribution.probabilities || !distribution.ready ||
           distribution.rows!=rows_ || distribution.vocabulary!=vocabulary_ ||
           distribution.row_stride<vocabulary_ ||
           distribution.generation!=generation_ || distribution.position!=position_ ||
           distribution.serial!=serial_)
            throw std::invalid_argument("missing/incomplete sampled distribution authority");
    }
private:
    Exl3SelectionAuthorityRequirement(Exl3SelectionAuthorityKind kind,
        std::uint32_t rows,std::uint32_t vocabulary,std::uint64_t generation,
        std::uint64_t serial,int position)
        :kind_(kind),rows_(rows),vocabulary_(vocabulary),generation_(generation),
         serial_(serial),position_(position) {}
};

// Seed belongs to the pre-forward root. Row r-1 authorizes tentative row r.
inline void exl3_greedy_path(std::int64_t seed, const Exl3GreedyPacket& packet,
                            std::span<std::int64_t> path) {
    if(path.empty() || path.size()>16)throw std::invalid_argument("greedy path extent");
    Exl3SelectionAuthorityRequirement::greedy(
        static_cast<std::uint32_t>(path.size()),packet.generation,
        packet.position,packet.serial).validate(packet);
    if(seed<0 || seed>=248320) throw std::invalid_argument("greedy packet seed");
    path[0]=seed;
    for(std::size_t i=1;i<path.size();++i) path[i]=packet.decisions[i-1].token;
}
} // namespace ninfer::exl3
