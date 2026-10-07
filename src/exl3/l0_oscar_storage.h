#pragma once
// Per-layer device storage of the L0 OSCAR INT2 KV history (l0_oscar.cuh).
#include <cstdint>

namespace ninfer::exl3::l0_oscar {

// Codes [capacity][4 KV heads][64] bytes and meta [capacity][4][scale0, zero0,
// scale1, zero1] floats per K and V; state[0] encoded watermark, state[1] last
// committed base. Query and history scratch serve up to `rows` verifier rows.
struct LayerStorage {
    std::uint8_t* k_codes = nullptr;
    std::uint8_t* v_codes = nullptr;
    float* k_meta = nullptr;
    float* v_meta = nullptr;
    int* state = nullptr;
    std::uint16_t* q_rot = nullptr;   // FP16 [rows][24][256]
    float* q_mu = nullptr;            // [rows][24]
    float* hist_work = nullptr;       // [rows][4][segments][fused-flash stride]
    float* hist_slot = nullptr;       // [rows][4][fused-flash stride]
    float* hist_rotated = nullptr;    // [rows][24][256] rotated history numerators
    int capacity = 0, rows = 0, segment_keys = 0, segments = 0, bank = 0;
};

}  // namespace ninfer::exl3::l0_oscar
