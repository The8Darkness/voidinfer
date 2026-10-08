#pragma once
// Per-layer device storage of the L0 OSCAR INT2 KV history (l0_oscar.cuh).
#include <cstdint>

namespace ninfer::exl3::l0_oscar {

// Codes [capacity][4 KV heads][64] bytes and meta [capacity][4][scale0, zero0,
// scale1, zero1] FP16 per K and V; state[0] encoded watermark, state[1] last
// committed base. Query and history scratch serve up to `rows` verifier rows.
struct LayerStorage {
    std::uint8_t* k_codes = nullptr;
    std::uint8_t* v_codes = nullptr;
    std::uint16_t* k_meta = nullptr;   // FP16
    std::uint16_t* v_meta = nullptr;
    int* state = nullptr;
    // Device copy of the exact window: absolute row r >= kSink lives in ring
    // slot r & (kRingRows - 1), rows < kSink in the sink planes. The mapped
    // host planes stay the complete FP16 record (L2) for export and aging.
    std::uint16_t* k_ring = nullptr;
    std::uint16_t* v_ring = nullptr;
    std::uint16_t* k_sink = nullptr;
    std::uint16_t* v_sink = nullptr;
    std::uint16_t* q_rot = nullptr;   // FP16 [rows][24][256]
    float* q_mu = nullptr;            // [rows][24]
    float* hist_work = nullptr;       // [rows][4][segments][fused-flash stride]
    float* hist_slot = nullptr;       // [rows][4][fused-flash stride]
    float* hist_rotated = nullptr;    // [rows][24][256] rotated history numerators
    float* hist_hot = nullptr;        // [rows][24][256] original-basis hot numerators (normalized)
    // Prefill chunks (rows > 8): history output, then FA2 splits 0..3 plus the
    // un-rotated history as split 4.
    float* prefill_hist = nullptr;    // [prefill_rows][24][256] rotated, normalized
    float* prefill_hist_stats = nullptr;  // [prefill_rows][24][2]
    float* prefill_split = nullptr;   // [5][prefill_rows][24][256]
    float* prefill_split_stats = nullptr;  // [5][prefill_rows][24][2]
    std::uint16_t* prefill_numer = nullptr;  // FP16 [prefill_rows][24][256] merged history
    int capacity = 0, rows = 0, prefill_rows = 0, segment_keys = 0, segments = 0, bank = 0;
    // Exact hot rows (l0_hot.cuh), per layer: the history rows the verifier
    // attends to most are served from exact FP16 copies instead of their INT2 codes.
    std::uint16_t* hot_k = nullptr;   // FP16 [4][hot_slots][256] exact copies of L2 rows
    std::uint16_t* hot_v = nullptr;   // FP16 [4][hot_slots][256]
    int* hot_row = nullptr;           // [4][hot_slots] absolute row, -1 empty
    float* hot_prio = nullptr;        // [4][hot_slots] ln p + lambda * base at the last hit
    unsigned* hot_bits = nullptr;     // [4][hot_words] row is hot
    int* hot_count = nullptr;         // [4] candidates emitted this round
    int* hot_cand_row = nullptr;      // [4][kHotCandidates]
    float* hot_cand_val = nullptr;    // [4][kHotCandidates] log2 p relative to hot_ref
    float* hot_ref = nullptr;         // [24] log2 history LSE of row 0, previous round
    int* hot_won = nullptr;           // [4] won last round, [4][kHotMaxInsert] slots, [4][kHotMaxInsert] rows
    int hot_slots = 0, hot_words = 0;
};

}  // namespace ninfer::exl3::l0_oscar
