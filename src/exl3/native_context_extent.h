#pragma once

#include "exl3/attention_profile.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace voidinfer::exl3 {

// One logical request extent, independent from aggregate queue/admission tokens. Positions are
// signed 32-bit values consumed by the CUDA kernels.
struct Exl3NativeContextExtent {
    using Position = std::int32_t;
    // FP16 device KV (VOIDINFER_EXL3_L0_OSCAR=0, diagnostic only).
    static constexpr std::uint32_t ordinary_limit = 32768;
    // L0 OSCAR contexts keep FP16 K/V in mapped host memory and only INT4
    // history plus exact windows on the device (l0_oscar.cuh).
    static constexpr std::uint32_t l0_oscar_tokens = 262144;
    // L0 OSCAR per-row buffers (hidden states, taps) hold one layer-major prefill block.
    static constexpr int l0_prefill_block_rows_max = 8192;
    // Rows of one layer-major L0 prompt block (VOIDINFER_EXL3_L0_PREFILL_BLOCK_ROWS,
    // 2048..8192 in steps of 1024, default 8192). Per-context row buffers (hidden,
    // taps, embedding trace) are sized by it; concurrent lanes trade a little
    // prefill speed for that memory.
    static int l0_prefill_block_rows() {
        static const int rows=[] {
            const char* value=std::getenv("VOIDINFER_EXL3_L0_PREFILL_BLOCK_ROWS");
            const int parsed=value?std::atoi(value):l0_prefill_block_rows_max;
            if(parsed<2048 || parsed>l0_prefill_block_rows_max || parsed%1024)
                throw std::invalid_argument("VOIDINFER_EXL3_L0_PREFILL_BLOCK_ROWS must be 2048..8192 in steps of 1024");
            return parsed;
        }();
        return rows;
    }
    static bool l0_oscar_enabled() { return exl3_l0_oscar_enabled(); }
    static constexpr std::uint64_t exact_kv_bytes_per_token = 65536;
    static constexpr std::uint32_t exact_kv_page_tokens = 64;

    static std::uint32_t configuration_limit() {
        return l0_oscar_enabled() ? l0_oscar_tokens : ordinary_limit;
    }

    static Position checked_max_context(std::uint64_t requested) {
        const auto limit=configuration_limit();
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

};

static_assert(sizeof(Exl3NativeContextExtent::Position)==4);
static_assert(Exl3NativeContextExtent::l0_oscar_tokens-1<=
    static_cast<std::uint32_t>(std::numeric_limits<Exl3NativeContextExtent::Position>::max()));

} // namespace voidinfer::exl3
