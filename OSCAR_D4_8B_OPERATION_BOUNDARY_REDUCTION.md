# OSCAR D4.8B — Operation-Boundary Reduction and Stable Decode Scheduling

Status: **PASS** (qualified generic scheduling/state reduction; CUDA Graphs deferred)

Date: 2026-09-02

## 1. Qualification identity

| Item | Value |
| --- | --- |
| Repository | publication source tree |
| Branch | `codex/d4-8b-operation-boundary-reduction-20260902` |
| Qualified parent | `32fe3137cf6304400229f4d72a9dedb6aa71c8af` (D4.8A qualified ancestry) |
| Target GPU | NVIDIA GeForce RTX 5090, SM120a / compute capability 12.0 |
| Driver | `616.56` |
| CUDA toolkit | `13.1.80` |
| Nsight Systems | `2026.4.1.191-264138605071v0` |
| Build | isolated RelWithDebInfo Ninja build, MSVC 19.44 |
| Model | local Qwen3.8-27B NVFP4-DFlash2 `.ninfer` artifact (hash retained below) |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| OSCAR asset SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |

The future EXL3 target and DFlash2 drafter were not loaded or integrated in D4.8B.

## 2. D4.7B / D4.8A baseline

The unchanged D4.7B clean full-attention timer, measured across all 16 full-attention
layers for eight forced one-token decodes, was:

| Visible history | Fused OSCAR | Complete full-attention branch |
| ---: | ---: | ---: |
| 512 | 0.892 ms | 17.656 ms |
| 2K | 1.551 ms | 18.282 ms |
| 4K | 2.460 ms | 18.642 ms |
| 8K | 4.525 ms | 21.738 ms |
| 16K | 8.288 ms | 22.975 ms |

At 16K, the outside-fused portion was `22.975 - 8.288 = 14.687 ms`.
The D4.8A Nsight Systems interval contained 128 isolated `verify.attention`
ranges: eight forced tokens times 16 full-attention layers.

## 3. Synchronization map

The relevant full-attention sequence is in
`src\targets\qwen3_6\impl\runtime\text_context_impl.h`; resident cache publication,
aging, and workspace ownership are in
`src\targets\qwen3_6\impl\runtime\oscar_rotated_bf16.cpp`.

| Boundary | Before | Producer → consumer | CPU/GPU consumer | D4.8B disposition |
| --- | --- | --- | --- | --- |
| Position transfer | `cudaMemcpyAsync` of the position window followed by `cudaStreamSynchronize` for every full-attention layer | device `cache_positions` → host contiguous-append check | CPU | Removed for resident one-token decode. The resident logical extent supplies the append position, and same-stream GPU work consumes the resulting host scalar without a D2H transfer. Retained for prefill and nonresident diagnostic paths that genuinely inspect a host position window. |
| Q/K/V rotation completion | Stream wait after rotation/position copies | rotation kernels and copies → host staging/validation | CPU on prefill/nonresident paths; GPU on resident path | Resident one-token path no longer waits. Same-stream append/publish ordering is authoritative. |
| Resident aging | Profile-only `cudaStreamSynchronize` after the aging kernel | aging kernel → timing readback | CPU timing only | Removed. Enqueue span is recorded separately; production consumers remain ordered on the decode stream. |
| Resident publish | Profile-only `cudaStreamSynchronize` after publish | publish kernel → timing readback | CPU timing only | Removed. Enqueue span is recorded separately; subsequent GPU work remains same-stream ordered. |
| Fused OSCAR attention | Profile-only `cudaStreamSynchronize` after split/merge | fused kernels → timing readback | CPU timing only | Removed for resident decode. |
| `R_V.T` recovery | Profile-only `cudaStreamSynchronize` after inverse rotation | inverse kernel → timing readback | CPU timing only | Removed for resident decode. |
| Reference validation | Explicit waits at layers 3, 35, and 63 before host readback | GPU tensors → independent CPU oracle | CPU validation | Retained and classified validation-only. These are not production no-oracle timings. |
| Diagnostic tensor dumps | Synchronization/readback in diagnostic dump helpers | GPU tensors → diagnostic files | CPU diagnostics | Retained only when diagnostics are explicitly enabled; profile-only diagnostic paths return without draining the decode stream. |
| Decode result egress | Outer device synchronization after the model submission | complete device decode → host result/commit | CPU API egress | Retained. It is the required outer result boundary, not an inner per-layer OSCAR boundary. |

No new stream was introduced. Same-stream CUDA ordering is used for the resident
one-token path. Cross-stream graph/event dependencies remain future work because the
current generic graph body also performs host-side live-cache bookkeeping.

## 4. Position/state architecture

Before D4.8B, each full-attention layer copied its device position window back to the
host even for a one-token resident decode, then waited before checking contiguity and
using the value for cache append. The same hot path also maintained resident cache
state in device buffers but repeatedly crossed the host/device boundary for this scalar.

D4.8B now detects the resident one-token case and obtains the logical append position
from the per-layer resident logical extent. No position D2H copy is issued for that
case, and no host wait is needed before the same-stream GPU sequence:

`rotate → resident aging/publish → fused split/merge → inverse recovery → BF16 output`.

Prefill and the nonresident diagnostic route keep their existing host position/staging
behavior because they process multi-token windows and/or intentionally perform host
validation. The device resident cache payload, metadata, ring storage, and aging
semantics remain unchanged. The logical context scalar is still host metadata for now;
promoting it to a graph-updatable device parameter is explicitly deferred with graph
capture rather than silently freezing it in a replay.

## 5. Scratch allocation architecture

The resident one-token path previously created five FP32 temporaries per full-attention
layer: Q, K, V, attention, and recovered output. They were freed at the end of each
layer. D4.8B allocates five cache-owned device buffers during resident workspace
initialization and reuses them sequentially across all 16 full-attention layers.

| Resource | Size |
| --- | ---: |
| Q | 24,576 bytes |
| K | 4,096 bytes |
| V | 4,096 bytes |
| Attention | 24,576 bytes |
| Recovered | 24,576 bytes |
| Total persistent decode scratch | **81,920 bytes** |

The maximum adaptive-split workspace therefore increases from `1,585,152` to
`1,667,072` bytes. The buffers are destroyed with the resident cache context, and the
single decode stream serializes reuse. Prefill retains its existing chunk-shaped
temporaries and was not redesigned.

## 6. Timing instrumentation

The old `gpu_qkv_rotation_us`, `gpu_mixed_kernel_us`, `gpu_recovery_us`, and full-branch
software counters were populated around explicit waits. Consequently, the elapsed
interval could include previously queued work and was not a pure kernel interval.

For resident decode, D4.8B keeps those synchronized counters at zero and adds explicit
`*_enqueue_us` counters. These are host enqueue spans only and are labeled as such in
the telemetry; they must not be interpreted as GPU execution time. The authoritative
post-change GPU interval measurements in this report come from Nsight Systems GPU
projection and kernel ranges, with no measurement-driven inner stream wait.

## 7. Incremental checkpoints

| Checkpoint | Isolated change | Result |
| --- | --- | --- |
| B0 | Unchanged D4.7B baseline | Clean branch: `17.656/18.282/18.642/21.738/22.975 ms` at 512/2K/4K/8K/16K. |
| B1 | Persistent resident FP32 decode scratch | Correctness passed; resident workspace became `1,667,072` bytes; scratch ownership moved to context lifetime. |
| B2 | Resident position path and position D2H removal | Correctness passed; one-token resident decode no longer copies a position scalar per full-attention layer. B1/B2 were captured together because the dependency-safe scratch and position changes share the resident decode route. |
| B3 | Removal of avoidable profile waits and enqueue-only telemetry | Correctness passed; synchronized stage counters were replaced by enqueue counters for resident decode. |
| B4 | Combined boundary-reduced production path | Fresh broad Nsight trace passed with zero inner sync/copy/allocation boundaries and projected branch values below all D4.8B PASS limits. |
| B5 | CUDA Graph evaluation | Not enabled. The constructor now fails closed for `use_cuda_graph=true` with resident OSCAR because the current generic capture body can freeze host-side context/append state. No graph variant is qualified. |

The B1/B2/B3 intermediate runs are retained in `results\oscar\d4-8b-b1b2.out`
and `results\oscar\d4-8b-b3-reprofile.out`. Their old synchronized timing fields
are intentionally not used as post-change GPU intervals.

## 8. Post-change branch measurements

The fresh broad trace is
`results\oscar\d4-8b-post-4816.nsys-rep`; its exported NVTX/GPU data is retained
under the matching `d4-8b-post-4816-*` files. Each after value below is the summed
Nsight GPU-projected duration of 128 `verify.attention` ranges divided by eight forced
tokens. It is an end-to-end attention-branch interval, not the retired synchronized
software timer. Nsight capture overhead is therefore expected; the measurement basis
is stated explicitly so it is not confused with the clean D4.7B baseline.

| Visible history | B0 clean full branch | B4 Nsight-projected attention branch | D4.8B PASS limit |
| ---: | ---: | ---: | ---: |
| 512 | 17.656 ms | 3.549 ms | n/a |
| 2K | 18.282 ms | 4.591 ms | n/a |
| 4K | 18.642 ms | **5.437 ms** | 15.5 ms |
| 8K | 21.738 ms | **8.666 ms** | 18.0 ms |
| 16K | 22.975 ms | **13.720 ms** | 19.5 ms |

The 512/2K/4K/8K/16K after values are from the candidate ranges in the same broad
trace. The post values meet the required 4K/8K/16K limits on this GPU-projected
branch metric. Clean real-model forced-decode qualification also completed at
321/332/512/2K/4K/8K/16K/32K with exact forced tokens and finite outputs; those
whole-engine timings are not substituted for the isolated full-attention branch.

### Stage/kernel projection at 16K

The 16K candidate `verify.attention` range contained the following operation-equivalent
GPU kernel sum over eight forced tokens:

| Component | 8-token GPU sum | Per forced token |
| --- | ---: | ---: |
| Q/K/V rotations | 6.189 ms | 0.774 ms |
| Resident encode/write | 1.012 ms | 0.127 ms |
| Fused OSCAR split | 70.356 ms | 8.795 ms |
| Fused partial merge | 3.362 ms | 0.420 ms |
| `R_V.T` inverse/recovery | 3.857 ms | 0.482 ms |
| FP32→BF16 conversion | 0.091 ms | 0.011 ms |
| Operation-equivalent live-full sum | **84.868 ms** | **10.608 ms** |

The end-to-end projected `verify.attention` interval is 13.720 ms/token because it
also includes the surrounding attention work and GPU timeline gaps. The operation
sum is useful for kernel attribution; the projected interval is the primary after
branch metric.

## 9. API boundary comparison

Counts below are for the same 128 isolated attention ranges (eight tokens times 16
full-attention layers), using interval-union attribution in the D4.8A and D4.8B
Nsight exports.

| CUDA/API boundary | D4.8A B0 | D4.8B B4 | Per forced token across 16 full layers |
| --- | ---: | ---: | ---: |
| `cudaStreamSynchronize` inside attention ranges | 640 | **0** | 5 → 0 |
| Position `cudaMemcpyAsync` | 128 | **0** | 1 → 0 |
| `cudaMalloc` | 640 | **0** | 5 → 0 |
| `cudaFree` | 640 | **0** | 5 → 0 |
| `cudaLaunchKernel` | 2,048 | 2,048 | 256 → 256 |

D4.8A captured about 104.52 ms of stream-wait API time, 84.85 ms of position-copy
API time, 14.36 ms of launch API time, 2.02 ms of malloc API time, and 3.95 ms of
free API time across those ranges. D4.8B removes the four avoidable boundary classes
from the inner ranges. The remaining 2,048 launches are the stable ungraphed launch
topology and are the next scheduling opportunity.

## 10. CUDA Graph design evaluation

The adaptive split policy still has three stable shape classes:

- 16 splits for visible context 1–512;
- 32 splits for visible context 513–8,192;
- 64 splits for visible context 8,193 and above.

In principle this supports one graph per split class. In the current runtime, however,
the generic graph capture body surrounds host-side live OSCAR cache bookkeeping. Replay
could freeze the resident logical context, append position, ring head, or other mutable
state. The production constructor therefore rejects the resident OSCAR plus graph
combination rather than permitting stale-state replay. `NINFER_OSCAR_D4_8B_USE_GRAPH`
exists only in the test harness to exercise that selection path; no graph replay was
reported as qualified.

The next graph attempt must promote mutable position/state to graph-updatable device
parameters or use explicit graph kernel-node parameter updates, then test 16→32→64
selection, 512/8192 crossings, aging, ring wrap, reset, teardown/recreation, and
workspace reuse.

## 11. Fixed floor and slope

For the B0 clean branch, a least-squares fit over 512/2K/4K/8K/16K gives an
extrapolated intercept of approximately **17.65 ms** and a history slope of
**0.353 us per visible token**. The 16K B0 outside-fused floor is **14.687 ms**.

For B4, fitting the 4K/8K/16K Nsight-projected attention-branch points gives an
extrapolated intercept of approximately **2.91 ms** and a visible-history slope of
**0.666 us per token**, reaching approximately 13.82 ms at 16K. The post-change
floor is now small enough that the history-dependent OSCAR traversal is visible;
the fit is an Nsight projection, not a claim that the retired host timer is clean.

At 16K, using the unchanged clean fused OSCAR gate as the non-regression reference,
the outside-fused branch is approximately **14.687 → 5.432 ms** on the stated mixed
measurement basis. The fresh Nsight kernel-equivalent live-full sum is 10.608 ms/token
and is reported separately because it excludes timeline gaps and outer branch work.

## 12. Prefill regression control

The Q64 batched prefill implementation and query-block policy were not redesigned.
The only resident prefill-visible resource change is the additional 81,920 bytes of
context-owned decode scratch; prefill continues to use its existing chunk-shaped
buffers and 64-token query blocks.

The established same-machine D4.7A parent control remains:

| Context | D4.7A parent Q64 prefill | D4.7B control | Difference |
| ---: | ---: | ---: | ---: |
| 4K | 2.68036 s | 2.66219 s | -0.7% |
| 8K | 10.68840 s | 10.72330 s | +0.3% |
| 16K | 43.85090 s | 43.38600 s | -1.1% |

D4.8B source changes do not touch the Q64 prefill algorithm, split policy, or
prefill kernel. The final real-model production qualification also exercised prefill
at 4K/8K/16K and completed without correctness failures. The run’s whole-engine
prefill wall values are retained in
`results\oscar\d4-8b-final-production-qualification.out`; they are not substituted
for the established parent-controlled Q64 metric because that harness includes the
full model setup and its variant-specific startup/workspace costs.

## 13. Correctness and qualification gates

Passed after the D4.8B source changes:

- direct CUDA mixed-attention parity at 64, 65, 320, 321, 322, 332, 512, 2K, 4K,
  8K, 16K, and 32K, including the adaptive split boundary sweep;
- `full_layer_dispatch_bitmap=1111111111111111`;
- `gdn_dispatches=0`;
- `legacy_q2_dispatches=0`;
- `bf16_historical_shadow=false`;
- `fallback=false` and no CPU attention fallback;
- no NaN/Inf in final real-model outputs;
- optimized FULL validator: `PASS taps=120 workers=8 avx2=true`;
- independent live/reference parity: `PASS taps=120 worst_relative_l2=0`;
- validator self-check: 21/21 contexts/layers under the original gates;
- final real-model forced continuation at 321/332/512/2K/4K/8K/16K/32K;
- first aging, recent-ring reuse, causal/prefix/history/recent boundaries, and all
  16 full-attention dispatch bits through the final sweep;
- clean rebuild target `ninfer_qwen3_6_27b_oscar_runtime_test` and the final runtime
  qualification executable exited successfully.

The official OSCAR semantics were not changed: calibrated `OscarInt2G128`, group 128,
64-token BF16 prefix, INT2 historical bulk, 256-token BF16 recent tier, calibrated
`R_K`/`R_V.T`, resident cache, device aging, codec parity, GQA 24/4, exact causal
masking, and the 16-layer full-attention dispatch remain intact. Adaptive split counts
remain 16/32/64 with the original maximum split workspace policy.

## 14. Nsight before/after timeline

| Evidence | D4.8A before | D4.8B after |
| --- | ---: | ---: |
| Isolated attention ranges | 128 | 128 |
| `cudaStreamSynchronize` | 640 | **0** |
| Position `cudaMemcpyAsync` | 128 | **0** |
| `cudaMalloc` / `cudaFree` | 640 / 640 | **0 / 0** |
| `cudaLaunchKernel` | 2,048 | 2,048 |
| Graph launches | 0 | 0 |
| GPU idle gaps | repeated host-induced gaps between small operations | boundary gaps removed; remaining gaps are stable ungraphed launch/kernel scheduling |

Fresh trace artifacts:

- `results\oscar\d4-8b-post-nsys-16k.nsys-rep`
- `results\oscar\d4-8b-post-4816.nsys-rep`
- exported `d4-8b-post-4816-api-*`, `d4-8b-post-4816-nvtx-*`, and
  `d4-8b-post-4816-nvtx-gpu-*` files under `results\oscar\`.

## 15. Remaining bottlenecks and next phase

The fixed host/API boundary floor has been removed from the inner resident attention
sequence. The dominant remaining cost at 16K is now the OSCAR history traversal itself,
with the stable ungraphed kernel-launch topology next. The next generic scheduling
candidate is a graph-updatable resident-state design with one safe graph per 16/32/64
split class. It must be qualified before enabling graphs.

After this D4.8B stop point, the highest-value project milestone is **EXL3 6 bpw H6 V6
integration**. It is the future exclusive target format and avoids spending substantial
NVFP4-specific GEMV effort that will be replaced. DFlash2 integration remains a
separate subsequent phase. No D4.7C, EXL3, DFlash2, MTP, vision, cache-format, or
architectural work was started here.

## Conclusion

D4.8B is **PASS** for operation-boundary reduction and stable resident decode
scheduling. The production path now has persistent resident one-token scratch, no
per-layer position D2H transfer, no avoidable inner stream waits, no per-layer
malloc/free, corrected non-draining telemetry, preserved OSCAR semantics, and a
fail-closed CUDA Graph boundary pending graph-updatable mutable state.
