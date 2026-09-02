# OSCAR D4.6 — Fused Persistent Mixed-Attention Kernel

Status: **PASS**
Date: 2026-09-02
Branch: `codex/oscar-d4-6-fused-mixed-attention-20260902`

## Scope and baseline

D4.5 was read in full before implementation. The D4.5 production-direction path was retained as the correctness and performance baseline:

- calibrated OSCAR INT2, official `OscarInt2G128`, group size 128;
- BF16 prefix 64, OSCAR INT2 historical bulk, BF16 recent 256;
- calibrated FP32 `R_K` query/key rotation and `R_V.T` recovery;
- GPU-resident typed cache for all 16 full-attention layers;
- device-side aging and exact host/device codec parity;
- Q64 causal prefill, 24 query heads, 4 KV heads, GQA 6:1;
- GDN layers unchanged and outside the OSCAR dispatch;
- FULL independent validator and existing numerical gates.

DFlash2, MTP, adaptive-K, calibration, quantization, vision, prefix/state reuse, and unrelated model layers remain out of scope.

The measured D4.5 mixed-attention baseline was 49.678 ms at 512, 738.808 ms at 2K, 2.860 s at 4K, 11.465 s at 8K, and 45.577 s at 16K. D4.5 used three GPU stages per Q64 prefill block: QK score, masked softmax, and AV.

## Implementation

D4.6 adds a fused resident-cache reader in `src/ops/softmax_attention/oscar_mixed/launch.cu` and wires it into the calibrated GPU-resident runtime. The production implementation is selected as:

`oscar-mixed-gpu-d4-6-fused-resident`

The cache representation and cache ownership are unchanged. The fused kernels read the logical sequence in its existing order—BF16 prefix, OSCAR INT2 history, then BF16 recent—and never materialize a complete decoded K/V cache.

### Prefill kernel

The public D4.5 Q64 path is preserved. Inside each Q64 launch, the fused kernel uses:

- internal query tiles of Q4;
- 32-token K/V tiles;
- one block per KV head and internal query tile;
- shared-memory K and V slabs reused across the six associated query heads;
- vectorized resident reads through the existing BF16/INT2 logical reader;
- exact per-query visibility checks for each tile;
- online, numerically stable softmax state `(m, l)`;
- immediate AV accumulation, with no global score or probability tensor.

The K slab is decoded once for its tile, reused by the six GQA query heads, then reused as the V slab for the same logical tile. The tile is discarded before the next tile is read. A Q64 public block therefore remains one launch, while its work is internally scheduled as Q4 tiles.

### Decode kernel

One-token decode uses a four-way split-KV fused kernel. Each split walks a disjoint logical history interval, performs the same tiled K/QK/online-softmax/AV work, and writes only fixed-size partial `(m, l, AV)` state. A second fixed-size merge kernel combines the four partial states with the stable softmax merge equation.

This is two launches per full-attention call instead of the D4.5 three-stage sequence. The decode workspace is fixed at 99,072 bytes and does not scale with history.

No TMA or tensor-core path was forced: the measured shape is dominated by packed INT2 decode, shared-memory staging, and SIMT dot/accumulate work. The selected Q4/K32 shape was the fastest qualified candidate on the RTX 5090; wider alternatives did not improve the measured path under the SM120a shared-memory/register constraints.

## Correctness evidence

The direct CUDA mixed-attention test passed after the final D4.6 rebuild and reports fused parity at the tier boundaries and long synthetic histories:

| Context | Tiers (prefix/history/recent) | Fused AV relative L2 | Result |
| ------: | ----------------------------: | -------------------: | :----- |
| 64 | 64 / 0 / 0 | 4.04e-7 | PASS |
| 65 | 64 / 0 / 1 | 3.95e-7 | PASS |
| 320 | 64 / 0 / 256 | 5.84e-7 | PASS |
| 321 | 64 / 1 / 256 | 5.85e-7 | PASS |
| 332 | 64 / 12 / 256 | 5.84e-7 | PASS |
| 512 | 64 / 192 / 256 | 8.69e-7 | PASS |
| 2K | 64 / 1,728 / 256 | 2.55e-6 | PASS |
| 4K | 64 / 3,776 / 256 | 5.28e-6 | PASS |

The direct test also exercised forced decode and recent-ring reuse through logical tokens 997, 1001, 1003, 1005, 1007, 1009, 1011, and 1013. Its final status was:

`PASS: mixed BF16-prefix + OscarInt2G128-history + BF16-recent GPU path is reference-correct and materially faster; fused decode includes split-KV and merge kernels`

The optimized FULL validator passed all serialized live taps at 321, 332, 512, 2K, and 4K. Each context supplied 30 taps across layers 3, 35, and 63. The validation set included the prefix/history boundary, history/recent boundary, first aging, recent-ring reuse, selected Q-block causal boundaries, forced decode, and all 16 full-attention dispatch bits.

The real-model run used the required model SHA-256 and calibrated runtime asset. Final telemetry confirmed:

- `full_layer_dispatch_bitmap=1111111111111111`;
- `gdn_dispatches=0`;
- `legacy_q2_dispatches=0`;
- `bf16_history_shadow=false`;
- `fallback=false`;
- runtime asset hash `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`;
- maximum observed real-model recovered relative L2 approximately `1.57e-6`, within the qualified gates.

The final CTest gate also passed:

`ninfer_oscar_mixed_gpu_attention_test: 1/1 passed`

## Prefill performance

These are real-model request measurements with the D4.6 fused resident path. Prompt tok/s is prompt tokens divided by total request wall time; it is not decode throughput.

| Context | Wall time | Mixed-attention time | Prompt tok/s |
| ------: | ---------: | -------------------: | -----------: |
| 512 | 354.136 ms | 41.886 ms | 1,445.8 |
| 2K | 1.389530 s | 587.968 ms | 1,473.9 |
| 4K | 3.748923 s | 2.285960 s | 1,092.6 |
| 8K | 11.832912 s | 9.042220 s | 692.3 |
| 16K | 41.549240 s | 36.031300 s | 394.3 |

Direct comparison with D4.5 mixed-attention time:

| Context | D4.5 attention | D4.6 attention | Speedup | D4.6 attention workspace |
| ------: | --------------: | --------------: | ------: | -----------------------: |
| 512 | 49.678 ms | 41.886 ms | 1.186x | 99,072 B |
| 2K | 738.808 ms | 587.968 ms | 1.257x | 99,072 B |
| 4K | 2.860 s | 2.285960 s | 1.251x | 99,072 B |
| 8K | 11.465 s | 9.042220 s | 1.268x | 99,072 B |
| 16K | 45.577 s | 36.031300 s | 1.265x | 99,072 B |

The complete request wall-time speedups over D4.5 were 1.174x, 1.185x, 1.191x, 1.228x, and 1.242x at 512, 2K, 4K, 8K, and 16K respectively. The 4K mixed-attention improvement is 20.1%; the 16K improvement is 20.9%.

At 4K, D4.5 launched 3,072 attention kernels for the 16 full-attention layers and Q64 prefill blocks. D4.6 launched 1,024 fused prefill kernels: one launch per Q64 block, a 3.0x reduction. Decode calls add two kernels per one-token attention call—the split kernel and fixed merge—rather than three separate D4.5 stages.

The D4.6 fused prefill path was also run at 8K and 16K. The real-model harness does not provide a lightweight “populate a 32K resident cache, then measure decode only” mode; enabling its 32K switch forces a full 32K prefill and BF16 control run. The direct fused fixture was measured through 32K, but no 32K real-model latency claim is made here.

## Decode performance

Decode was measured as eight forced one-token columns after each real-model prefill. The table reports the cumulative eight-column full-attention branch, its per-token value across all 16 full-attention layers, the average per full-attention layer, and the fused mixed-attention component. The attention-only ceiling is `1 / one-token full-attention latency`; it is not end-to-end generation tok/s.

| History | 8-token branch, all 16 layers | One-token, all 16 layers | Avg / full-attention layer | Fused mixed attention / token | Attention-only ceiling |
| ------: | ----------------------------: | -----------------------: | -------------------------: | ---------------------------: | ---------------------: |
| 512 | 124.230 ms | 15.529 ms | 0.971 ms | 2.265 ms | 64.4 tok/s |
| 2K | 169.484 ms | 21.186 ms | 1.324 ms | 7.917 ms | 47.2 tok/s |
| 4K | 230.318 ms | 28.790 ms | 1.799 ms | 15.106 ms | 34.7 tok/s |
| 8K | 350.191 ms | 43.774 ms | 2.736 ms | 30.074 ms | 22.8 tok/s |
| 16K | 602.285 ms | 75.286 ms | 4.705 ms | 61.188 ms | 13.3 tok/s |

D4.5 one-token all-layer latencies were 18.878, 30.385, 45.517, 76.986, and 140.610 ms at the same histories. D4.6 therefore improved one-token full-attention latency by 17.7% at 512, 30.3% at 2K, 36.7% at 4K, 43.1% at 8K, and 46.5% at 16K. At 4K the latency fell from 45.517 ms to 28.790 ms; at 16K it fell from 140.610 ms to 75.286 ms.

The endpoint history slope across all 16 full-attention layers fell from approximately 7.670 microseconds per history token in D4.5 to 3.765 microseconds per history token in D4.6, a 50.9% reduction. The synthetic fused fixture also completed at 32K and remained parity-correct; its 32K effective logical resident traffic was 9.08 GB/s, not a hardware DRAM-bandwidth measurement.

## Workspace and profile

D4.5 allocated history-proportional score and softmax buffers: 50,528,256 bytes at Q64/4K and 201,523,200 bytes at Q64/16K. D4.6 allocates only the fixed four-split decode partial-state buffer:

`99,072 bytes` = `96.75 KiB`

This is a reduction of 99.8039% versus the D4.5 4K workspace and 99.9508% versus the D4.5 16K workspace. The resident cache allocation itself remains capacity-dependent and authoritative; it is not counted as attention score/probability workspace and was not changed by D4.6.

The fused kernel deliberately removes the stage boundaries that previously allowed exact internal timing of QK, softmax, and AV. The runtime’s `gpu_fused_kernel_us` is the exact aggregate elapsed time around the fused stage. Internal phase accounting is therefore structural rather than fabricated from host timers:

| Fused phase | Implementation evidence |
| ----------- | ----------------------- |
| Packed K load/decode | Logical resident K reader, one 32-token prefill tile or 64-token decode tile at a time; INT2 rows decoded into shared memory. |
| QK | Six query-head dot products per KV head and visible tile row, with the shared K tile reused across GQA peers. |
| Softmax/update | Per-query online `m/l` update using stable rescaling; causal rows are excluded before the update. |
| Packed V load/decode | The same tile slab is reused for V, with immediate AV accumulation. |
| AV | Probability-weighted V accumulation stays in the fused kernel and does not write a full probability tensor. |
| Synchronization/launch | Shared-memory barriers separate tile load, QK, V reuse, and accumulation. Prefill uses one kernel launch per Q64 block; decode uses split plus merge. |

No separate Nsight hardware-utilization run was added. The unit-test effective-GB/s figure is logical resident traffic divided by fused elapsed time and must not be interpreted as DRAM bandwidth. The exact real-model profile shows the new dominant cost is still the fused mixed kernel’s history-proportional packed decode plus QK/AV inner loop—not host cache staging, not CPU aging, and no longer the three-stage launch/score/softmax traffic path.

At 16K, fused mixed attention accounts for 36.0313 s of the 40.0187 s full-attention prefill branch and 61.188 ms of the 75.286 ms one-token full-attention latency. QKV rotation and `R_V.T` recovery remain smaller downstream costs and were intentionally left for later phases.

## PASS decision

D4.6 passes all requested gates:

1. reference equivalence remains qualified by the direct CUDA parity checks, real-model taps, and optimized FULL validator;
2. causal masking is exact at Q-block boundaries and in split decode intervals;
3. D4.4 GPU-resident cache, device aging, official codec, and mixed-tier semantics are unchanged;
4. D4.5 Q64 prefill launch behavior is retained without per-query launches and improves at 4K+;
5. one-token decode materially improves, with the largest measured gain at 16K;
6. score/probability workspace is reduced from history-proportional buffers to a fixed 99,072-byte decode merge workspace;
7. telemetry confirms all 16 full-attention dispatch bits, zero GDN OSCAR dispatches, no legacy Q2, no BF16 historical shadow, and no fallback.

## D4.7 recommendation

Tune a persistent SM120a history-tile kernel for vectorized OSCAR INT2 unpack/dequantization and cross-query KV-tile residency, starting with the one-token decode path.
