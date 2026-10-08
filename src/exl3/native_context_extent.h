#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer::exl3 {

// One logical request extent. This contract is deliberately independent from
// aggregate queue/admission tokens: a native64K configuration means that one
// exact request may own positions [0,65536), with all positions represented as
// signed 32-bit values consumed by the existing CUDA kernels.
struct Exl3NativeContextExtent {
    using Position = std::int32_t;
    static constexpr std::uint32_t ordinary_limit = 32768;
    static constexpr std::uint32_t native64k_tokens = 65536;
    static constexpr std::uint32_t candidate128k_tokens = 131072;
    // L0 OSCAR contexts keep FP16 K/V in mapped host memory and only INT2
    // history plus exact windows on the device (l0_oscar.cuh).
    static constexpr std::uint32_t l0_oscar_tokens = 262144;
    // L0 OSCAR per-row buffers (hidden states, taps) hold one layer-major prefill block.
    static constexpr int l0_prefill_block_rows = 8192;
    static bool l0_oscar_enabled() {
        const char* value=std::getenv("NINFER_EXL3_L0_OSCAR");
        return value && value[0]=='1' && value[1]==0;
    }
    static constexpr std::uint64_t exact_kv_bytes_per_token = 65536;
    static constexpr std::uint32_t exact_kv_page_tokens = 64;

    struct Candidate128KPlan {
        Position maximum_context = 0;
        Position last_position = 0;
        std::uint32_t exact_kv_pages = 0;
        std::uint64_t exact_kv_payload_bytes = 0;
        std::uint64_t minimum_host_reservation_bytes = 0;
    };

    static std::uint32_t configuration_limit(bool native64k,bool candidate128k) {
        if(native64k && candidate128k)
            throw std::invalid_argument("extended 64K and 128K contexts are separate controls");
        if(l0_oscar_enabled() && !native64k && !candidate128k) return l0_oscar_tokens;
        return candidate128k?candidate128k_tokens:(native64k?native64k_tokens:ordinary_limit);
    }

    static Position checked_max_context(std::uint64_t requested,
        bool native64k,bool candidate128k) {
        const auto limit=configuration_limit(native64k,candidate128k);
        if(!requested || requested>limit || requested>
                static_cast<std::uint64_t>(std::numeric_limits<Position>::max()))
            throw std::invalid_argument("native context maximum is outside its selected extent");
        return static_cast<Position>(requested);
    }

    static Position checked_input_end(std::size_t tokens,Position maximum) {
        if(maximum<=0 || tokens>static_cast<std::size_t>(maximum) ||
            tokens>static_cast<std::size_t>(std::numeric_limits<Position>::max()))
            throw std::invalid_argument("single request input is outside native context extent");
        return static_cast<Position>(tokens);
    }

    static Position checked_position(std::uint64_t position,Position maximum) {
        if(maximum<=0 || position>=static_cast<std::uint64_t>(maximum) ||
            position>static_cast<std::uint64_t>(std::numeric_limits<Position>::max()))
            throw std::invalid_argument("logical position is outside native context extent");
        return static_cast<Position>(position);
    }

    static std::uint64_t exact_kv_payload_bytes(std::uint64_t tokens) {
        if(tokens>std::numeric_limits<std::uint64_t>::max()/exact_kv_bytes_per_token)
            throw std::overflow_error("exact host KV payload extent overflow");
        return tokens*exact_kv_bytes_per_token;
    }

    // Source planning for the next fixed native extent is intentionally
    // separate from execution enablement.  This establishes the integer and
    // unavoidable exact-KV reserve contract without claiming that the current
    // kernels, cache menus or lifecycle have been qualified at 128K.
    static Candidate128KPlan candidate128k_plan(std::uint32_t maximum,
        bool native64k_enabled,bool candidate128k_enabled,bool exact_host,
        bool oscar_only,std::uint32_t concurrency,std::uint64_t host_kv_budget,
        bool device_prefix_enabled) {
        if(maximum!=candidate128k_tokens || !candidate128k_enabled || native64k_enabled)
            throw std::invalid_argument("128K candidate requires its separate fixed-extent control");
        if(!exact_host || oscar_only)
            throw std::invalid_argument("128K candidate requires ordinary exact-host KV ownership");
        if(concurrency!=1)
            throw std::invalid_argument("128K candidate reserve model is single-request C1 only");
        if(device_prefix_enabled)
            throw std::invalid_argument("128K candidate has no supported represented device-prefix cache menu");
        const auto maximum_position=checked_position(candidate128k_tokens-1,
            static_cast<Position>(candidate128k_tokens));
        const auto payload=exact_kv_payload_bytes(candidate128k_tokens);
        if(host_kv_budget<payload)
            throw std::invalid_argument("128K candidate host KV budget is below one exact request payload");
        return {static_cast<Position>(candidate128k_tokens),maximum_position,
            candidate128k_tokens/exact_kv_page_tokens,payload,payload};
    }

    static void refuse_candidate128k_execution(std::uint32_t maximum,
        bool native64k_enabled,bool candidate128k_enabled,bool exact_host,
        bool oscar_only,std::uint32_t concurrency,std::uint64_t host_kv_budget,
        bool device_prefix_enabled) {
        if(maximum!=candidate128k_tokens && !candidate128k_enabled)return;
        (void)candidate128k_plan(maximum,native64k_enabled,candidate128k_enabled,
            exact_host,oscar_only,concurrency,host_kv_budget,device_prefix_enabled);
        throw std::invalid_argument(
            "128K is a static configuration candidate only; native Engine execution is unsupported");
    }

    // This is a necessary lower bound, not the complete root footprint:
    // recurrent state, convolution history, taps, token history and owner
    // metadata remain separately accounted by their authoritative owners.
    static void require_native64k_engine(std::uint32_t maximum,
        bool native64k_enabled,bool candidate128k_enabled,bool exact_host,
        bool oscar_only,std::uint32_t concurrency,std::uint64_t host_kv_budget) {
        if(maximum!=native64k_tokens)return;
        if(!native64k_enabled || candidate128k_enabled)
            throw std::invalid_argument("native64K Engine requires its separate 64K configuration control");
        if(!exact_host || oscar_only)
            throw std::invalid_argument("native64K Engine requires ordinary exact-host KV ownership");
        if(concurrency!=1)
            throw std::invalid_argument("native64K Engine is a genuine single-request C1 configuration");
        if(host_kv_budget<exact_kv_payload_bytes(native64k_tokens))
            throw std::invalid_argument("native64K Engine host KV budget is below one exact request payload");
    }
};

static_assert(sizeof(Exl3NativeContextExtent::Position)==4);
static_assert(Exl3NativeContextExtent::native64k_tokens-1<=
    static_cast<std::uint32_t>(std::numeric_limits<Exl3NativeContextExtent::Position>::max()));
static_assert(Exl3NativeContextExtent::candidate128k_tokens-1<=
    static_cast<std::uint32_t>(std::numeric_limits<Exl3NativeContextExtent::Position>::max()));
static_assert(Exl3NativeContextExtent::candidate128k_tokens%
    Exl3NativeContextExtent::exact_kv_page_tokens==0);

} // namespace ninfer::exl3
