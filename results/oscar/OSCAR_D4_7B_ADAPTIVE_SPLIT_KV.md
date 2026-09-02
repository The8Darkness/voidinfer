# OSCAR D4.7B — Context-Adaptive Split-KV Decode Parallelism

Status: **PASS**

Date: 2026-09-02

This phase changes only the split-KV parallelism of the already-qualified D4.6
fused resident decode path. The official OSCAR INT2 codec, resident cache
representation, fused arithmetic, prefill kernel, and GDN path are unchanged.

## Environment and qualification snapshot

- Repository: `D:\AI\voidinfer-adaptive-dflash2`
- Branch: `codex/oscar-d4-7b-adaptive-split-kv-20260902`
- D4.7A parent commit: `613ca48a474ebcb493ce55fa5a71a102337ea914`
- D4.7B implementation commit: `c7c20398`
- Build: `D:\AI\build-adaptive-dflash2`, `RelWithDebInfo`, Ninja
- Target architecture: CUDA `120a` / SM120a
- OS: Windows 11 x64, kernel `10.0.26200.9278` (25H2)
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB reported, compute capability 12.0
- Driver: `610.88`
- CUDA toolkit: `13.1.80`
- MSVC: `19.44.35225.0` compiler identification, x64 host toolchain
- Model source: `C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer`
- Model SHA-256: `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`
- OSCAR asset: `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`
- OSCAR runtime asset SHA-256: `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`

The exact D4.7A parent was also built in a separate temporary build directory
and run against the same C-drive model and runtime asset. That parent comparison
was used only to distinguish machine/run variance from a Q64 prefill regression.

## Implementation

The production resident decode caller now passes an adaptive sentinel. The
decoder selects one supported split count from the visible logical context:

| Visible tokens | Selected splits | Split grid (`KV heads x splits`) | Total split CTAs |
| --------------: | --------------: | --------------------------------: | ----------------: |
| 1–512 | 16 | `4 x 16` | 64 |
| 513–8,192 | 32 | `4 x 32` | 128 |
| 8,193+ | 64 | `4 x 64` | 256 |

The policy is a deterministic host-side lookup with negligible dispatch cost.
It uses `min(total_tokens, query_token + 1)` so a decode query never schedules
against future positions. Explicit values in `{1,2,4,8,16,32,64}` remain
available to the direct CUDA qualification harness, while production uses the
adaptive sentinel only.

The split kernel receives the selected count and partitions the same logical
causal interval as D4.6. The merge kernel reduces only the selected partial
rows using the existing FP32 online-softmax `(m,l,AV)` state. No cache tier,
codec, unpack routine, shared-memory layout, QK/AV arithmetic, or merge
reduction equation was changed.

The maximum workspace is allocated once with the resident cache and reused by
all decode calls. There is no per-token `cudaMalloc`/`cudaFree` and no
history-proportional score/probability tensor.

## Workspace

The workspace is `24,768` bytes per split for four KV heads, six GQA groups,
256 head dimensions, and FP32 partial state.

| Splits | Workspace bytes | Workspace KiB |
| -----: | --------------: | ------------: |
| 4 | 99,072 | 96.75 KiB |
| 8 | 198,144 | 193.50 KiB |
| 16 | 396,288 | 387.00 KiB |
| 32 | 792,576 | 774.00 KiB |
| 64 | 1,585,152 | 1,548.00 KiB |

D4.6 used 99,072 bytes. D4.7B increases the fixed resident allocation by
1,486,080 bytes (16x at maximum policy), or approximately 1.42 MiB. It remains
constant with history length and is allocated outside the decode hot loop.

## Split-policy search

The policy was selected from production-equivalent direct CUDA fixture runs at
321, 512, 1,024, 2,048, 4,096, 8,192, 16,384, and 32,768 tokens, with
neighboring split counts tested. Representative combined split-plus-merge
results were:

| Context | 8 splits | 16 splits | 32 splits | 64 splits | Selected |
| ------: | --------: | ---------: | ---------: | ---------: | -------: |
| 321 | 39.902 us | **33.986 us** | 35.813 us | 54.494 us | 16 |
| 512 | 49.970 us | **44.766 us** | 47.568 us | 94.162 us | 16 |
| 1,024 | 83.699 us | 57.019 us | **54.458 us** | 62.888 us | 32 |
| 2,048 | 152.469 us | 87.306 us | **60.040 us** | 77.246 us | 32 |
| 4,096 | 378.939 us | 165.174 us | 119.854 us | 116.114 us | 32 |
| 8,192 | 700.764 us | 313.852 us | **167.736 us** | 171.988 us | 32 |
| 16,384 | 1,338.820 us | 796.396 us | 403.728 us | **304.604 us** | 64 |
| 32,768 | 2,613.344 us | 1,210.360 us | 805.952 us | **771.328 us** | 64 |

At 4K the 64-way direct result was a narrow single-run win, but the targeted
Nsight Systems phase measurement favored 32 splits after accounting for the
larger 64-way merge. The final policy therefore uses 32 through 8K and 64 only
above 8K. This is a measured threshold policy, not a copied D4.7A crossover
assumption.

## Correctness

The final direct CUDA test exited 0 and passed all existing gates. It covered:

- 64, 65, 320, 321, 322, 332, 512, 2K, 4K, 8K, 16K, and 32K;
- prefix/history and history/recent boundaries;
- first aging and recent-ring reuse;
- forced decode;
- layers 3, 35, and 63;
- all 16 full-attention dispatch bits.

Direct fused relative-L2 values remained below the existing `1e-4` gate:

| Context | Fused relative L2 | Result |
| ------: | ----------------: | :----- |
| 4K | `5.231505384e-6` | PASS |
| 8K | `9.428927115e-6` | PASS |
| 16K | `2.317413055e-5` | PASS |
| 32K | `3.623968587e-5` | PASS |

The 16K and 32K values were accepted under the existing gate; no tolerance was
loosened. The optimized FULL validator also passed:

`OSCAR optimized full validator: PASS taps=120 workers=8 avx2=true`

`OSCAR D2.3b independent live/reference parity: PASS taps=120 worst_relative_l2=0 legacy_q2=false bf16_history_shadow=false`

Final telemetry confirmed:

```text
full_layer_dispatch_bitmap=1111111111111111
gdn_dispatches=0
legacy_q2_dispatches=0
bf16_historical_shadow=false
fallback=false
```

The real-model validation used the C-drive model and the required model and
asset hashes. No NaN/Inf or CPU attention fallback occurred.

## Real-model decode performance

The following D4.7B values are from the repeated clean no-oracle C-drive run.
Eight forced one-token columns were measured after each prefill. Fused and full
branch values are GPU-timer deltas divided by eight; full-branch values include
the surrounding full-attention costs across all 16 full-attention layers. They
are not end-to-end generation throughput.

| History | D4.6 fused mixed / token | D4.7B fused mixed / token | Fused speedup | D4.6 full branch / token | D4.7B full branch / token | Selected splits |
| ------: | -----------------------: | -------------------------: | ------------: | ------------------------: | -------------------------: | --------------: |
| 512 | 2.265 ms | 0.89 ms | 2.54x | 15.529 ms | 18.28 ms | 16 |
| 2K | 7.917 ms | 1.55 ms | 5.11x | 21.186 ms | 18.64 ms | 32 |
| 4K | 15.106 ms | 2.46 ms | 6.14x | 28.790 ms | 18.64 ms | 32 |
| 8K | 30.074 ms | 4.53 ms | 6.64x | 43.774 ms | 21.74 ms | 32 |
| 16K | 61.188 ms | 8.29 ms | 7.38x | 75.286 ms | 22.98 ms | 64 |
| 32K | not real-model measured | 0.762 ms synthetic fixture | n/a | not real-model measured | n/a | 64 |

The 4K, 8K, and 16K fused decode values pass the required D4.7B gates of 10,
20, and 35 ms respectively. At 16K the complete full-attention branch fell
from 75.286 ms to 22.98 ms per token. Average full-attention-layer latency at
16K is approximately 1.44 ms.

The conservative repeated-run fused history slope from 512 to 16K is
approximately `0.46 us/history-token`, versus the D4.6 slope of approximately
`3.765 us/history-token`, an 87.8% reduction. The complete full-attention
branch slope in the same repeated run is approximately `0.29 us/history-token`.

The attention-only ceiling implied by the D4.7B full branch is approximately
53.6 tok/s at 4K, 46.0 tok/s at 8K, and 43.5 tok/s at 16K. This is an
attention-only ceiling, not end-to-end generation throughput.

### Synthetic 32K diagnostic fixture

The direct fixture reported `fused_split_merge_ms=0.762416` at 32K with 64
splits, 28,418,048 bytes of fixed-capacity workspace, and 53.78 GB/s of
logical resident traffic. That GB/s number is a logical cache-traffic figure,
not measured DRAM bandwidth.

Against the D4.7A fixed-four-split Nsight Systems fixture baseline of 6.088043
ms, the combined split-plus-merge fixture time is approximately 7.98x lower.

## Split and merge phase timing

Nsight Systems was used for phase separation; tracing adds perturbation, so the
following values are phase diagnostics rather than the primary real-model
timing source.

| Context | Splits | Split kernel | Merge kernel | Merge share of split+merge |
| ------: | -----: | -----------: | -----------: | -------------------------: |
| 4K | 16 | 138.560 us | 6.817 us | 4.7% |
| 4K | 32 | **71.809 us** | 12.416 us | 14.7% |
| 4K | 64 | 76.384 us | 23.393 us | 23.4% |
| 8K | 16 | 274.722 us | 6.912 us | 2.5% |
| 8K | 32 | **140.481 us** | 12.448 us | 8.1% |
| 8K | 64 | 134.017 us | 23.872 us | 15.1% |
| 16K | 16 | 622.468 us | 6.880 us | 1.1% |
| 16K | 32 | 280.130 us | 12.352 us | 4.2% |
| 16K | 64 | **304.098 us** | 23.296 us | 7.1% |
| 32K | 16 | 1,368.777 us | 6.816 us | 0.5% |
| 32K | 32 | 826.565 us | 12.320 us | 1.5% |
| 32K | 64 | **555.524 us** | 23.776 us | 4.1% |

Merge grows with split count but is not the limiting cost at the selected 16K
or 32K configurations. At 4K, 32 splits is preferred partly because 64 splits
spends a larger fraction in merge.

## Nsight Compute before/after verification

Fresh targeted Nsight Compute collection succeeded after the D4.7A counter
permission change. No `ERR_NVGPUCTRPERM` occurred. Reports contained valid,
nonempty hardware metrics and targeted the RTX 5090 / CC 12.0 device.

The profiled split kernel remained one 256-thread CTA with approximately 43
registers/thread and 73.312 KiB dynamic shared memory. Shared memory still
limits residency to one block per active SM; adaptive splitting increases grid
coverage and the number of scheduling waves rather than per-SM resident-block
occupancy.

| Metric | D4.6 fixed 4, 4K | D4.7B adaptive 32, 4K | D4.6 fixed 4, 16K | D4.7B adaptive 64, 16K |
| ------ | ----------------: | --------------------: | -----------------: | ----------------------: |
| Split grid / CTAs | `4 x 4` / 16 | `4 x 32` / 128 | `4 x 4` / 16 | `4 x 64` / 256 |
| Active-warps ceiling | 16.67% | 16.67% | 16.67% | 16.67% |
| Eligible warps/active cycle | not recorded in D4.6 run | 0.10 | not recorded in D4.6 run | 0.10 |
| Issue-active/active cycle | not recorded in D4.6 run | 0.09 | not recorded in D4.6 run | 0.09 |
| L1/TEX long-scoreboard stall | 66.97% | 66.11% | 66.88% | 65.89% |
| Barrier stall | 7.90% | 8.84% | 8.40% | 9.01% |
| Wait stall | 11.09% | 11.99% | 11.83% | 12.20% |
| DRAM utilization | 0.20% | 1.37% | 0.17% | 1.10% |
| L2 throughput | 0.21% | 0.88% | 0.18% | 0.70% |
| L1/TEX sector hit rate | not retained | 77.79% | not retained | 83.70% |
| Tensor-pipe instructions | 0 | 0 | 0 | 0 |
| Raw shared-conflict evidence | ~1.10M events | metric family/sample dependent | ~4.41M events | metric family/sample dependent |

The selected split counts improve whole-GPU work availability while leaving the
per-active-SM occupancy limit unchanged. The lower long-scoreboard percentage
and still-low DRAM/L2 utilization are consistent with latency hiding, not with
conversion to a bandwidth-bound kernel. The raw shared-conflict evidence from
D4.7A remains a real secondary issue; the D4.7B change intentionally did not
redesign the shared-memory layout.

## Prefill regression check

The Q64 batched prefill implementation was not modified. A same-machine
comparison against a separately built exact D4.7A parent showed:

| Context | Parent D4.7A prefill fused | D4.7B prefill fused | Difference |
| ------: | -------------------------: | ------------------: | ---------: |
| 4K | 2.68036 s | 2.66219 s | -0.7% |
| 8K | 10.68840 s | 10.72330 s | +0.3% |
| 16K | 43.85090 s | 43.38600 s | -1.1% |

The older D4.6 report contains faster historical runs (2.285960 s, 9.042220 s,
and 36.031300 s), but those values were not reproducible in the current
machine state even with the unchanged parent. The exact parent comparison is
therefore the valid regression control. Q64 remained selected and the D4.7B
decode-only change caused no material prefill regression.

## Remaining bottlenecks

1. The fused kernel remains latency/occupancy limited by L1/TEX dependency
   latency; long-scoreboard stalls remain approximately 66% despite increased
   grid coverage.
2. Shared-memory bank-conflict/replay behavior remains a secondary measured
   cost.
3. Scalar INT2 unpack/dequant instruction pressure remains present.
4. Approximately 14 ms of the historical D4.6 16K full-attention branch was
   outside the fused mixed kernel; the current complete branch is much lower,
   but QKV, rotations, recovery, and surrounding layer work remain outside this
   phase.

These are recorded for later work and were not bundled into D4.7B.

## PASS decision

D4.7B passes:

1. production decode selects 16, 32, or 64 splits by visible context;
2. official OSCAR cache semantics and D4.4 resident aging remain unchanged;
3. direct CUDA and optimized FULL numerical qualification pass without relaxed
   thresholds;
4. 4K, 8K, and 16K fused decode are 2.46 ms, 4.53 ms, and 8.29 ms,
   respectively, below all D4.7B gates;
5. Q64 prefill has no material regression against the exact parent build;
6. workspace remains fixed and bounded at 1,585,152 bytes maximum;
7. merge overhead is bounded and non-dominant at the selected long-context
   configurations;
8. no legacy Q2, BF16 historical shadow, CPU attention fallback, or GDN OSCAR
   dispatch was observed;
9. temporary profiling-only context/split selectors were removed before final
   rebuild and qualification.

## Recommended next phase

**D4.7C should first reduce the measured shared-memory bank-conflict/replay
cost in the fused resident decode tile layout, while preserving the adaptive
split policy and all existing numerical gates.**

D4.7C was not implemented in this phase.
