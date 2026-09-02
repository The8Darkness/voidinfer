# OSCAR D4.5 — Batched/Fused Causal Prefill

Date: 2026-09-02  
Status: **PASS**

## Scope and reconstructed baseline

D4.5 continued from the qualified calibrated OSCAR INT2 path:

- asset: `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`
- asset hash: `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`
- model SHA-256: `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`
- runtime mode: `oscar-int2-gpu-resident`
- cache policy: BF16 prefix 64, official OSCAR INT2 historical bulk, BF16 recent 256

The D1.3, D2.1/D2.2, D2.3b, D3.1, D4.1, D4.2a, D4.2b, D4.3, full-validator, and D4.4 reports were inspected before implementation. The repository directory has no `.git` metadata, so `git status`/`git diff` could not provide a repository-level baseline.

## Implementation

The resident D4.4 reader now has a multi-query launch path. A contiguous prefill block is processed by three GPU stages:

1. causal QK score generation over `(KV head, key tile, query)`;
2. per-query/per-Q-head masked softmax;
3. causal AV accumulation with the existing GQA mapping (`q_head / 6`).

Each query uses its own logical boundary, `keys <= query_position`; future rows are never read. The scratch is query-major score/softmax workspace and is not a decoded K/V cache. The resident compressed representation remains authoritative, device-side aging is unchanged, and `R_V.T` recovery remains unchanged.

`NINFER_OSCAR_D4_5_QBLOCK` accepts Q8, Q16, Q32, and Q64; the production-direction default is Q64. Only resident prefill uses batching. One-token decode remains the scalar resident path and is separately counted. QKV rotation and value recovery were not optimized, and DFlash2, MTP, and adaptive-K remain disabled.

## Correctness evidence

- `ninfer_oscar_mixed_gpu_attention_test`: **1/1 PASS**, 7.87 s.
- Q64 real-model D4.5 run covered 321, 332, 512, 2K, and 4K, including the first historical transition, recent-ring reuse, selected boundary queries, forced decode, and layers 3/35/63.
- The internal batched-vs-qualified-reference checks produced 375 PASS comparisons. Worst observed relative-L2 was `2.52443e-6` for rotated AV and `2.50990e-6` for recovered output; worst recovered absolute error was `5.14984e-5`, below the `1e-4` relative / `1e-3` absolute gates.
- The optimized FULL validator passed its self-check (`21` scalar-golden checks; contexts 64, 320, 321, 332, 512, 2048, 4096; layers 3, 35, 63) and passed all five tap sets (`30` taps each, worst relative-L2 `0`).
- Final telemetry reported `full_layer_dispatch_bitmap=1111111111111111`, `legacy_q2_dispatched=false`, `bf16_historical_shadow=false`, `fallback=false`, and zero host cache staging. Device aging/codec parity remained enabled in the correctness run.

The four candidate performance runs all produced finite outputs and the exact nine-token forced continuation (`forced_decode_agree=true`). The Q64 candidate is the one that received the full reference/FULL-validator qualification above.

## Candidate benchmark

All rows are real-model resident runs. Wall time includes the fixed forced continuation; prompt tok/s is therefore labeled as prompt tokens divided by request wall time, not decode throughput. Attention launches are prefill-only and count all 16 full-attention layers; each query block uses three GPU stage launches.

| Query block | 512 wall | 2K wall | 4K wall | 4K mixed GPU attention | 4K prefill launches | Queries/launch | Reduction vs D4.4 | Prompt tok/s (512 / 2K / 4K) | Resident cache | Workspace | Numerical result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Q8 | 505.825 ms | 2.613 s | 8.109 s | 6.471 s | 24,576 | 8 | 8.0x / 87.5% | 1,012 / 784 / 505 | 63,078,400 B | 6,316,032 B | finite; forced decode exact |
| Q16 | 432.526 ms | 2.050 s | 6.009 s | 4.386 s | 12,288 | 16 | 16.0x / 93.75% | 1,184 / 999 / 682 | 63,078,400 B | 12,632,064 B | finite; forced decode exact |
| Q32 | 408.799 ms | 1.785 s | 4.948 s | 3.338 s | 6,144 | 32 | 32.0x / 96.875% | 1,252 / 1,147 / 828 | 63,078,400 B | 25,264,128 B | finite; forced decode exact |
| Q64 | 415.599 ms | 1.647 s | 4.466 s | 2.860 s | 3,072 | 64 | 64.0x / 98.4375% | 1,232 / 1,244 / 917 | 63,078,400 B | 50,528,256 B | **qualified PASS** |

For 4K prefill, D4.4 issued 65,536 one-query calls and 196,608 mixed-attention kernel launches across the 16 full-attention layers. Q64 issues 1,024 query-block calls and 3,072 kernel launches. The launch reduction is therefore 64x while preserving per-query causal masking.

## D4.4 → D4.5 timing

| Context | D4.4 request | D4.5 Q64 request | Speedup | D4.5 mixed GPU attention (prefill) |
|---|---:|---:|---:|---:|
| 512 | 1,505.832 ms | 415.599 ms | **3.62x** | 49.678 ms |
| 2K | 13,525.424 ms | 1,646.964 ms | **8.21x** | 738.808 ms |
| 4K | 46,450.017 ms | 4,465.609 ms | **10.40x** | 2.860 s |

The optimized path also completed 8K in **14.528 s** and 16K in **51.641 s**. At 16K, resident workspace was 201,523,200 B and prefill mixed-attention time was 45.577 s.

## Prefill versus steady-state decode

The following is a separate phase measurement from Q64 telemetry. Decode values cover eight ordinary one-token attention columns from the nine-token forced continuation, aggregated across all 16 full-attention layers; they are not reported as prompt throughput or as full-engine generation throughput.

| History | Request wall | Prefill full-attention branch | Prefill mixed GPU | Decode branch total / one-token average | Decode columns |
|---|---:|---:|---:|---:|---:|
| 512 | 415.599 ms | 167.951 ms | 49.678 ms | 151.025 ms / **18.878 ms** | 8 |
| 2K | 1,646.964 ms | 1.228 s | 738.808 ms | 243.076 ms / **30.385 ms** | 8 |
| 4K | 4,465.609 ms | 3.835 s | 2.860 s | 364.137 ms / **45.517 ms** | 8 |
| 8K | 14.528 s | 13.452 s | 11.465 s | 615.885 ms / **76.986 ms** | 8 |
| 16K | 51.641 s | 49.598 s | 45.577 s | 1.125 s / **140.61 ms** | 8 |

## D4.5 conclusion

D4.5 passes: batched resident prefill is causally exact, preserves the D4.4 cache/codec semantics, passes the optimized FULL gate, substantially reduces launch count, and materially improves 4K real-model wall time without using legacy Q2, a BF16 historical shadow, CPU attention fallback, DFlash2, MTP, or adaptive-K.

The new dominant bottleneck is the remaining mixed GPU attention arithmetic inside the batched score/softmax/AV stages, especially the repeated INT2 decode and QK/AV work at long histories; launch overhead is no longer the primary limiter.

## D4.6 recommendation

Fuse the resident mixed-attention score, masked-softmax, and AV stages into a persistent tiled kernel that reuses decoded INT2 K/V tiles across a query block.
