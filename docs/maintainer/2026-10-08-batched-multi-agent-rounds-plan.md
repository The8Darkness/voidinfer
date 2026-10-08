# Batched multi-agent DFlash2 rounds (plan, 2026-10-08)

Goal: serve two (later more) concurrent agents on the coherent-device L0 OSCAR route with
shared target weight reads, so the aggregate decode throughput approaches 2x of one agent
instead of the +10% that time-sliced lanes give.

## Measurements behind the decision (RTX 5090, Qwen3.8-27B EXL3 6bpw, L0 OSCAR, 8K)

* One target forward by rows (`continue_rows`, `tests/research_exl3_overlap` mode 4):
  1 row 13.37 ms, 2 rows 13.83, 4 rows 14.20, 8 rows 15.20 (about 0.26 ms per extra row).
  Rows 9..16 currently fall to the wide prefill route (37.6 ms).
* Two contexts decoding concurrently on separate streams: 24.7 ms/token each, 81 tok/s
  aggregate versus 74 tok/s for one context (+10%).
* Per-context device memory at 16K: about 3.4 GiB, mostly 1024-row prefill/wide scratch
  (hidden taps 400 MiB, shared accumulation 340, layer scratch 291, reconstruct workspace
  about 700, full-attention buffers 545, GDN buffers 336, L0 storage 256). Two lanes leave
  under 400 MiB free, below the layer-major projection cache reserve.
* The multi-row coherent GEMM producer already runs an M16 MMA tile with rows 8..15
  zero-filled, so 16 rows add little GEMM work over 8.

## Design

A batched round concatenates the verify rows of up to two contexts (each 2..8 rows) into
one 4..16-row target forward. Row-independent work (embedding, norms, every projection,
residuals, LM head, greedy) runs once on all rows; stateful work runs per context on its
own rows with its own layer state:

* full attention: Q/K norm + RoPE at the context's positions, K/V writes into its cache,
  L0 history/window/hot attention, verifier sibling layout;
* GDN: convolution state, gated delta recurrence, per-row checkpoints for repair;
* taps, greedy packets, transactions and retained-prefix repair stay per context.

## Phases

1. **16-row projection routes.** Admit 9..16 rows on every target continuation route
   (qkv, z, q, kv, o, gate/up, down, head). Check each shape against the FP64 per-row oracle in
   `tests/test_exl3_cuda_linear.cpp`, then microbenchmark 8 vs 16 rows.
2. **Layer forward split.** The multi-row continuation path of `Exl3FullAttentionLayer`
   and `Exl3GdnLayer` exposes a pre-projection / per-context core / post-projection
   sequence that accepts row segments owned by different layer objects.
3. **Batched continuation.** `Exl3TextContext` gains a two-context continuation that runs
   one layer stack over both segments, with per-context transactions, taps and greedy
   packets; output must equal two separate `continue_rows` calls within the route's
   numerical profile.
4. **Engine.** The coherent-device engine admits two lanes and drives both lanes' rounds
   in lockstep (draft A, draft B, one batched verify, settle both). The 1024-row prefill
   scratch is shared between lanes and prefill passes are serialized.
5. **Measure and extend.** Aggregate and per-agent tok/s at 4K/16K/64K; then batch the
   draft model and consider more than two agents.
