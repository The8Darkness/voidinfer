# OSCAR D4.7A — SM120a Mixed-Attention Hardware Profile

Status: PASS — profiling and diagnosis complete
Date: 2026-09-02
Branch: codex/oscar-d4-7a-hardware-profile-20260902
Profiled qualified implementation commit: 9b6b5635d2ff242241ffa144f2b00948dc3bb166

This phase profiled the D4.6 production path only. No D4.7B implementation was
merged. The temporary test-only context and split-count selectors used to collect
the isolated captures were removed before the final rebuild and qualification.

## 1. Environment

- Windows 11 x64, native Windows build.
- GPU: NVIDIA GeForce RTX 5090, 32 GiB, compute capability 12.0 / SM120a.
- Driver: 610.88; nvidia-smi reports CUDA 13.3.
- CUDA toolkit: 13.1.
- Nsight Compute: 2026.2.1.0.
- Nsight Systems: 2026.4.1.191.
- Build: RelWithDebInfo, x64, build tree D:\AI\build-adaptive-dflash2.
- Model loaded by the final real-model gate from the C-drive default:
  C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer.
- Model SHA-256: 6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e.
- Runtime asset: qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1.
- Runtime asset SHA-256: 4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560.

The qualified representation remained calibrated OSCAR INT2 (OscarInt2G128,
group 128), BF16 prefix 64, INT2 historical bulk, BF16 recent 256, calibrated
R_K/R_V.T, resident GPU cache, GQA 24Q/4KV/6, Q64 prefill, and untouched GDN.

## 2. Counter-permission verification

All Nsight Compute captures in this phase were fresh processes started after the
driver policy change. No stale ncu/nsys process was present when the fresh sanity
capture began.

The sanity command targeted the demangled
oscar_mixed_fused_decode_split_kernel with --set basic, one launch, and nine
metric passes. It returned NCU_EXIT_CODE=0, generated a new report, and emitted
no ERR_NVGPUCTRPERM. The imported report contained nonempty hardware values,
including:

- device: NVIDIA GeForce RTX 5090, CC 12.0;
- grid 16, block 256, SM count 170;
- dynamic shared memory 73.312 KiB and 41 registers/thread;
- nonzero DRAM bytes, integer instructions, FP32 instructions, and active-warp metrics.

The subsequent targeted 512, 2K, 4K, 8K, 16K, 32K, split-64, and merge captures
also completed with exit code 0. Hardware-counter collection therefore succeeded;
the previous permission blocker is closed.

Raw reports are in results/oscar/d4-7a-profile/, including
d4-7a-ncu-sanity-4k.ncu-rep, d4-7a-ncu-targeted-{512,2k,4k,8k,16k,32k}.ncu-rep,
the split-64 captures, and d4-7a-ncu-merge-4k.ncu-rep.

## 3. D4.6 baseline

The qualified D4.6 one-token decode numbers below are total across all 16
full-attention layers, with the fused mixed component shown separately.

| History | Total full-attention | Fused mixed attention |
| ------: | -------------------: | --------------------: |
| 512 | 15.529 ms | 2.265 ms |
| 2K | 21.186 ms | 7.917 ms |
| 4K | 28.790 ms | 15.106 ms |
| 8K | 43.774 ms | 30.074 ms |
| 16K | 75.286 ms | 61.188 ms |

D4.6 real-model prefill mixed-attention time was 41.886 ms, 587.968 ms,
2.285960 s, 9.042220 s, and 36.031300 s at 512, 2K, 4K, 8K, and 16K.
The fused attention workspace was fixed at 99,072 bytes.

## 4. Nsight Compute results

The following are the production four-split decode kernel. Percentages are the
Nsight Compute metric values; shared conflicts and wavefronts are raw event
counts. sm__warps_active is active-warp utilization on the SMs that receive a
CTA, not whole-GPU occupancy; the 16-CTA grid leaves most of the 170 SMs idle.

| History | Grid | Active warps | Eligible-warp raw avg | Long scoreboard | L1TEX focused retest | Barrier | Wait | DRAM read | DRAM % peak | L2 % peak | Shared conflicts | Tensor |
| ------: | ---: | -----------: | -------------------: | --------------: | --------------------: | ------: | ----: | ---------: | ---------: | --------: | ---------------: | ------: |
| 512 | 16 | 16.67% | not normalized | 68.38% | not isolated | 8.49% | 11.71% | 1.48 MB | 0.54% | 0.34% | 137,811 | 0 |
| 2K | 16 | 16.67% | not normalized | 67.34% | not isolated | 8.40% | 11.77% | 2.47 MB | 0.25% | 0.15% | 550,976 | 0 |
| 4K | 16 | 16.67% | 22,919 | 69.37% | 66.97% | 7.90% | 11.09% | 3.78 MB | 0.20% | 0.21% | 1,101,884 | 0 |
| 8K | 16 | 16.67% | not normalized | 67.23% | not isolated | 8.40% | 11.82% | 6.50 MB | 0.22% | 0.14% | 2,203,692 | 0 |
| 16K | 16 | 16.67% | 92,379 | 67.21% | 66.88% | 8.40% | 11.83% | 11.71 MB | 0.17% | 0.18% | 4,407,422 | 0 |
| 32K | 16 | 16.67% | not normalized | 67.14% | not isolated | 8.40% | 11.83% | 22.43 MB | 0.16% | 0.13% | 8,814,823 | 0 |

The raw smsp__warps_eligible.avg value scales with the measured kernel sample
duration in this Nsight section and is not a normalized per-scheduler count. It
is retained above as reported; the normalized scheduling conclusion comes from
the grid/shared-memory limits and the controlled split sweep below.

### Occupancy and resource limits

The production kernel uses block 256, 41 registers/thread, and approximately
73.31 KiB dynamic shared memory. Nsight reports the shared-memory limit as one
resident block per SM; the register limit would allow five blocks and the warp
limit six. Thus shared memory, not registers or block size, is the active
occupancy limiter. The fixed production grid is only 4 KV heads x 4 splits = 16
CTAs, versus 170 SMs.

### Instruction and math evidence

At 4K the kernel executed approximately 168.35M integer thread instructions,
92.47M FP32 thread instructions, and 7.73M conversion instructions. At 16K
these were 674.32M, 373.80M, and 32.90M respectively. The integer path is real
and scales with history, but aggregate pipe activity remained only about
0.30–0.32% ALU and 0.23–0.24% FMA. Tensor instructions and tensor-pipe activity
were zero. This is not an FP/SIMT throughput roof; it is a latency/scheduling
problem with unpack/control work in the dependency chain.

### Memory and shared-memory evidence

DRAM throughput stayed between 0.16% and 0.54% of peak, and L2 throughput stayed
between 0.13% and 0.34% of peak. Global loads were well coalesced by the
sector/request metric: about 1.03 sectors/request at 4K, 1.01 at 8K, and 1.00
at 16K/32K. The kernel is therefore not genuinely DRAM-bandwidth-bound.

The shared-memory event count is substantial: at 4K there were 1,101,884 bank
conflicts over 3,137,516 shared-memory wavefronts (about 35.1% as a raw event
ratio); the counts scale linearly with history. This is a material secondary
problem, but Nsight did not provide a direct elapsed-time fraction for those
replays, so no fabricated percentage of runtime is assigned to it.

### L1/TEX stall attribution

A focused fresh retest collected both
smsp__warp_issue_stalled_long_scoreboard_per_warp_active and
smsp__warp_issue_stalled_long_scoreboard_pipe_l1tex_per_warp_active. They were
identical within the report at 4K (66.97%) and 16K (66.88%). The dominant
measured stall is therefore a long L1/TEX scoreboard dependency, not a DRAM
throughput ceiling. The shared tile staging, scalar packed reads/unpack, and
subsequent QK/online-softmax/AV dependency chain are the likely sources of this
latency; the counter establishes the pipe, while the exact substage is not
separately observable without instrumenting or restructuring the fused kernel.

### Split-KV causal experiment

For diagnosis only, the same fused kernel body was launched with a temporary
workspace-capacity wrapper using 1, 2, 4, 8, 16, 32, or 64 splits. The production
path remained four splits. Every diagnostic output passed the direct reference
comparison within the existing direct-test tolerance; the 64-split 16K output
had relative L2 about 2.32e-5, so any D4.7B implementation must use the FULL
validator and should not blindly select 64 splits without requalification.

Nsight Systems GPU-kernel timings (split plus merge) were:

| History | 4 splits | 8 splits | 16 splits | 32 splits | 64 splits | Best diagnostic choice |
| ------: | -------: | -------: | --------: | --------: | --------: | :--------------------- |
| 4K | 550.089 us | 279.720 us | 145.620 us | 87.482 us | 92.723 us | 32 |
| 8K | 1.212153 ms baseline | — | 284.271 us | 168.065 us | 174.984 us | 32 |
| 16K | 3.046262 ms | 1.121409 ms | 610.093 us | 308.249 us | 298.114 us | 64, narrowly |
| 32K | 6.088043 ms baseline | — | 1.455802 ms | 723.010 us | 636.755 us | 64 |

Relative to four splits, the same-body diagnostic sweep reduced split-plus-merge
time by approximately 6.3x at 4K, 7.2x at 8K, 10.2x at 16K, and 9.6x at 32K.
These are isolated synthetic kernel results, not claims of equivalent end-to-end
real-model speedups. They directly demonstrate that grid granularity and
latency hiding are limiting the current four-split launch.

The 64-split NCU captures expanded the grid to 256 CTAs and raised DRAM activity
to 1.60% at 4K and 1.20% at 16K, while long-scoreboard stall remained about
65.7–65.9%, barrier about 8.9–9.0%, and active-warp utilization remained 16.67%
on active SMs because the shared-memory limit was unchanged. More CTAs improved
whole-GPU scheduling without making the kernel DRAM-bound.

### Merge kernel

The fixed four-split merge is four blocks, 256 threads, 33 registers/thread, and
about 1.02 KiB shared memory. At 4K its fresh NCU report showed no meaningful
resource pressure; Nsight Systems measured approximately 1.9 us merge time
against 540.4 us split time. The merge fractions at 512/2K/4K/8K/16K/32K were
2.64%, 0.69%, 0.35%, 0.16%, 0.073%, and 0.032%. It is negligible in the
production four-split path and is not the D4.7B target, although a high split
count makes it a secondary consideration.

### Phase-level limitation

The fused kernel has no production phase markers between packed K load/decode,
QK, softmax/update, packed V load/decode, and AV. Adding markers would alter
the synchronization and resource schedule being diagnosed. Nsight Compute
therefore reports the aggregate fused launch, while source mapping and
instruction/counter evidence identify the shared fused dependency chain. No
phase-time percentages are fabricated. The controlled split-count experiment is
the direct causal measurement used for the primary recommendation.

The four-KV-head logical compressed representation is 640 bytes per history token:
per KV head, K payload 64 B + K metadata 16 B + V payload 64 B + V metadata 16 B.
The D4.6 kernel decodes each disjoint split's tile once, reuses K across its six
associated Q heads, reuses the slab for V, and discards it. There is no complete
decoded K/V cache and no logical duplicate traversal between splits. The avoidable
cost is instruction-level scalar packed-byte/metadata extraction and staging
latency, not redundant logical cache rows.

## 5. Bottleneck ranking and rejected hypotheses

1. Grid-level latency hiding / fixed four-way split — highest confidence.
   The production grid is 16 CTAs on 170 SMs, with one block/SM due to shared
   memory. The split sweep directly cuts isolated kernel time by 6.3–10.2x.
2. L1/TEX dependency latency in the fused tile loop — measured secondary
   limiter. Long scoreboard is 67–69%, and the focused pipe metric attributes it
   to L1/TEX. More CTAs hide part of this latency; they do not remove it.
3. Shared-memory bank-conflict/replay behavior — measured secondary issue.
   Raw conflict events are about 35% of 4K shared wavefronts and scale with
   history. The exact elapsed-time contribution needs a controlled layout variant.
4. Scalar INT2 unpack/dequant instruction pressure — real but not first.
   Integer instructions exceed FP32 instructions, but DRAM is nearly idle and
   aggregate ALU/FMA pipe activity is low. A vectorized unpack change should be
   evaluated after scheduling/layout work.
5. Merge kernel — negligible at production split count.
6. Tensor-core QK/AV — rejected as the first change. Tensor activity is zero,
   but the whole kernel is severely underfilled; MMA operand preparation would
   add packing, registers, synchronization, and numerical risk before the grid
   problem is addressed.

Rejected as primary diagnoses: DRAM bandwidth saturation, L2 bandwidth
saturation, register-limited occupancy, and merge-kernel cost. Nsight values do
not support those explanations.

### Theoretical benefit estimates

These are ceilings or diagnostic bounds, not additive promises:

| Change | Evidence-based upper/observed benefit |
| :----- | :------------------------------------ |
| Increase useful split/grid coverage | Observed 6.3–10.2x isolated kernel reduction in the diagnostic sweep; end-to-end gain is bounded by non-attention work and numerical qualification. |
| Eliminate all measured long-scoreboard issue stalls | The issue-stall fraction gives a mathematical issue-cycle ceiling of roughly 3.0–3.3x at 4K/16K, but it is not a runtime prediction because other stalls become exposed. |
| Remove shared-memory conflicts | Raw conflict ratio is ~35.1% at 4K, but no direct time fraction was measured; expected gain is potentially meaningful but unquantified until a layout A/B exists. |
| Vectorize exact INT2 unpack/dequant | History-linear integer work is confirmed, but pipe utilization and low memory throughput prevent a defensible standalone gain estimate from current counters. Treat as a follow-on experiment. |

## 6. Roofline-style conclusion

D4.6 is primarily occupancy/latency-bound with a mixed L1/TEX dependency and
shared-memory replay component, not DRAM-bandwidth-bound and not FP/SIMT
throughput-bound. The single most important limiting fact is the 16-CTA
four-split grid on a 170-SM device under a one-block/SM shared-memory limit.

At 16K, D4.6's 61.188 ms fused mixed component remains the dominant full-attention
cost. Approximately 14.1 ms of the 75.286 ms full-attention branch is outside
that kernel (QKV, rotations/recovery, surrounding work, and dispatch overhead);
that is deferred to a later phase.

## 7. Candidate D4.7B analysis

| Candidate | Expected value | Complexity/risk | Decision |
| :-------- | :------------- | :-------------- | :------- |
| Context-adaptive split-KV grid | Directly supported by 6.3–10.2x isolated sweep; preserves resident representation | Low/medium; merge/workspace and numerical drift must be gated | First |
| Larger history tile | May reduce launch/tile overhead, but shared memory is already the occupancy limit | Medium/high register/shared-memory risk | After split sweep |
| Shared-memory layout / bank-conflict reduction | Counter evidence supports it | Medium; requires exact causal parity | Second diagnostic direction |
| Double-buffered async staging / warp specialization | Could hide L1/TEX latency and barriers | High scheduling/resource complexity | After grid/layout |
| Vectorized exact INT2 load/unpack | Targets the integer dependency chain | Medium; packed layout and metadata alignment risk | Follow-on |
| Better packed physical layout / TMA | Could improve sector behavior, but current global coalescing and DRAM use are already low | High; cache-semantic and codec risk | Not first |
| Tensor-core QK/AV | Potential arithmetic throughput | High preparation/register/numerical cost; current underfill dominates | Rejected as first |

## 8. Single recommended D4.7B change

D4.7B should first implement context-adaptive split-KV for the one-token fused
decode grid because the measured four-split path launches only 16 CTAs on 170
SMs, while the same kernel body reduced isolated 4K–32K time by 6.3–10.2x when
supplied with 32/64-way grid granularity.

The first implementation should keep the resident cache and fused arithmetic
unchanged, choose a validated split count by history length, and size the partial
workspace for the selected count. The diagnostic crossover suggests 32 splits
near 4K–8K and 64 near 16K–32K, but those thresholds are hypotheses rather than
production settings. The 16K 64-split relative-L2 drift (~2.32e-5) makes FULL
validator qualification mandatory before any setting is enabled.

### D4.7B gates

- Direct CUDA parity at 321, 332, 512, 2K, 4K, plus longer selected taps;
- optimized FULL validator on the complete serialized tap set;
- exact causal masking, prefix/history/recent boundaries, first aging, ring reuse,
  forced decode, layers 3/35/63, and all 16 full-attention dispatch bits;
- zero NaN/Inf, legacy Q2, BF16 historical shadow, CPU attention fallback, or GDN
  OSCAR dispatch;
- production resident cache and official codec byte/metadata semantics unchanged;
- 4K mixed decode no more than 10 ms, 8K no more than 20 ms, 16K no more than
  35 ms, subject to the same measurement method as D4.6;
- no regression in Q64 prefill at 4K/8K/16K;
- merge time and workspace reported for each selected split count.

## 9. Expected D4.7B performance (estimates)

These are estimates for a qualified context-adaptive split implementation, not
measurements from D4.7A. They assume the resident representation and fused
kernel math remain unchanged and that the grid benefit survives real-model
dispatch overhead.

| History | D4.6 fused decode | Conservative D4.7B estimate | Aggressive D4.7B estimate |
| ------: | ----------------: | --------------------------: | ------------------------: |
| 4K | 15.106 ms | 8–10 ms | 5–7 ms |
| 8K | 30.074 ms | 16–20 ms | 10–14 ms |
| 16K | 61.188 ms | 30–35 ms | 18–26 ms |
| 32K | not real-model qualified in D4.6 | 60–80 ms | 35–50 ms |

The estimates are intentionally lower-confidence than the isolated sweep: the
real-model branch includes QKV, rotations, recovery, dispatch, and all 16 layer
calls, and the split-count choice must pass stricter numerical gates.

## 10. Correctness and cleanup status

After profiling instrumentation was removed:

- clean rebuild completed for the mixed CUDA test, FULL validator, and runtime test;
- direct CUDA mixed-attention parity passed at 64/65/320/321/322/332/512/2K/4K,
  forced-decode/recent-ring cases, and the final all-layer bitmap
  1111111111111111;
- direct test reported gdn_dispatch=0, legacy_q2_dispatch=0, and
  cpu_fallback=0;
- optimized FULL validator passed 120 serialized taps across layers 3/35/63;
- clean C-drive real-model D4.6 runtime gate passed at 321, 332, 512, 2K, 4K,
  8K, and 16K. Sampled live/reference taps were finite and PASS;
- final real-model telemetry retained calibrated asset identity/hash, resident
  cache, gpu_resident_workspace_bytes=99072, full_layer_dispatch_bitmap=1111111111111111,
  gdn_dispatches=0, legacy_q2_dispatches=0, bf16_historical_shadow=false,
  and fallback=false;
- temporary D4.7A selectors and split-count wrappers are absent from source;
- no D4.7B, DFlash2, MTP, EXL3, cache-semantic, calibration, or unrelated-layer
  change was implemented.

## Final diagnosis

On RTX 5090, D4.6 is limited first by insufficient CTA parallelism and latency
hiding: a four-split decode grid creates only 16 CTAs, while each CTA consumes
enough shared memory to permit only one resident block per SM. The active kernel
then spends about 67–69% of issue cycles in measured L1/TEX long-scoreboard
stalls, with shared-memory conflicts as a secondary issue. It is not saturating
DRAM or L2, and it is not using enough FP/tensor throughput for a math roof to be
the first target. D4.7B should therefore begin with context-adaptive split-KV grid
granularity, followed by FULL numerical qualification.
