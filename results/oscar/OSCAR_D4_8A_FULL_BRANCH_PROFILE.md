# OSCAR D4.8A — Full-Attention Decode Floor Profile

Status: **complete diagnosis; no production optimization implemented**

This report profiles the unchanged, PASS-qualified D4.7B runtime after adaptive split-KV
selection. The profiling-only context/GPU selector used to isolate an Nsight Systems trace
was removed before the final rebuild and qualification reruns.

## Executive result

At 16K, the clean D4.7B one-token measurement is:

| Quantity | Time |
| --- | ---: |
| Complete full-attention branch, all 16 full layers | **22.975 ms** |
| Fused resident OSCAR mixed attention | **8.288 ms** |
| Outside the fused mixed-attention timer | **14.687 ms** |

The remaining floor is not explained by a large non-OSCAR GEMV. The strongest fresh
evidence is Nsight Systems API sequencing: repeated explicit `cudaStreamSynchronize`,
device-to-host position copies, per-layer temporary `cudaMalloc`/`cudaFree`, and launch
gaps around small per-layer operations. The QKV and output projection kernels are real,
but together account for only about 1.18 ms per forced token across all 16 full layers in
the captured timeline.

The single recommended next implementation is **D4.8B operation-boundary reduction**:
make position/state handling and temporary FP32 scratch persistent/device-resident, remove
the per-layer host position copy and explicit synchronization where correctness permits,
then evaluate CUDA Graph capture of the stable decode sequence. This is a recommendation
only; it was not implemented in D4.8A.

## 1. Environment and exact snapshot

| Item | Value |
| --- | --- |
| Repository | `D:\AI\voidinfer-adaptive-dflash2` |
| Branch | `codex/oscar-d4-7b-adaptive-split-kv-20260902` |
| Commit profiled and qualified | `32fe3137cf6304400229f4d72a9dedb6aa71c8af` |
| Build | `RelWithDebInfo`, Ninja, native Windows x64 |
| CUDA target | `120a` / SM120a |
| OS | Windows 11 25H2, kernel `10.0.26200.9278` |
| GPU | NVIDIA GeForce RTX 5090, 32 GB, compute capability 12.0 |
| NVIDIA driver | `610.88` |
| CUDA toolkit | `13.1.80` |
| Nsight Compute | `2026.2.1.0` (build `38286902`) |
| Nsight Systems | `2026.4.1.191-264138605071v0` |
| Model | `C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| OSCAR runtime asset SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |

The C-drive model path was used for all final live qualification. The D-drive path was not
used for regular model loading or validation.

## 2. D4.7B baseline

These are the qualified D4.7B real-model one-token results used for direct comparison.
The fused value is the resident adaptive-split OSCAR mixed-attention work across all 16
full-attention layers. The full value is the existing software full-attention branch timer.

| History | Fused OSCAR mixed attention | Complete full-attention branch |
| ---: | ---: | ---: |
| 512 | 0.892 ms | 17.656 ms |
| 2K | 1.551 ms | 18.282 ms |
| 4K | 2.460 ms | 18.642 ms |
| 8K | 4.525 ms | 21.738 ms |
| 16K | 8.288 ms | 22.975 ms |

The user-facing D4.7B headline values round these to 0.89/18.28 ms at 512, 2.46/18.64
ms at 4K, and 8.29/22.98 ms at 16K.

## 3. Definition of the measured full-attention branch

In `text_context_impl.h`, the `live_full_attention_start` marker is created after:

- input RMSNorm;
- the QKV projection;
- Q and K RMSNorm;
- RoPE;
- the normal attention diagnostic dumps.

From that marker through `record_full_attention_us`, the timer includes:

- FP32 calibrated Q/K/V rotation;
- the device-to-host position copy and its stream synchronization;
- resident cache append/publish and any device aging;
- fused mixed OSCAR attention;
- inverse `R_V.T` recovery;
- optional validation/diagnostic work when enabled;
- FP32-to-BF16 conversion.

It excludes the preceding QKV projection and Q/K preparation, and it stops before the
sigmoid gate and attention output projection. The output projection is launched later in
`attn_mix` and is visible in the broader `VerifyAttention` NVTX range, but is not part of
the software `gpu_full_attention_us` number. Post-attention norm and MLP are in the
separate post-mixer range.

This is a software timer boundary over an asynchronous CUDA stream. In particular, the
rotation timing includes a synchronization that can drain earlier queued work. Therefore
the clean component counters below are an accounting decomposition, not independent,
strictly additive GPU intervals.

## 4. Clean component accounting

The following values are from the clean repeated D4.7B no-oracle run. All columns are
aggregate across the 16 full-attention layers for one forced decode token.

`qkv_rotation/position` is the existing `gpu_qkv_rotation_us` counter. It is **not** pure
QKV projection: it covers calibrated rotation, the position D2H copy, synchronization, and
any earlier queued work drained by that synchronization.

| History | Full branch | Fused OSCAR | qkv_rotation/position* | Recovery `R_V.T` | Aging | Publish | Residual accounting |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 17.656 ms | 0.892 ms | 11.443 ms | 0.708 ms | 3.626 ms | 0.334 ms | 0.653 ms |
| 2K | 18.282 ms | 1.551 ms | 11.546 ms | 0.674 ms | 3.689 ms | 0.249 ms | 0.573 ms |
| 4K | 18.642 ms | 2.460 ms | 11.638 ms | 0.823 ms | 2.781 ms | 0.192 ms | 0.748 ms |
| 8K | 21.738 ms | 4.525 ms | 11.711 ms | 1.188 ms | 3.332 ms | 0.276 ms | 0.706 ms |
| 16K | 22.975 ms | 8.288 ms | 11.235 ms | 0.722 ms | 1.815 ms | 0.157 ms | 0.758 ms |

The residual is `full - fused - qkv_rotation/position - recovery - aging - publish`.
At 16K the arithmetic is therefore 22.975 = 8.288 + 11.235 + 0.722 + 1.815 + 0.157
+ 0.758 ms, but the qkv/position value overlaps queue-drain effects and must not be
interpreted as a pure disjoint stage.

The 16K outside-fused value is directly measured as 22.975 - 8.288 = **14.687 ms**.
The non-OSCAR portion is effectively fixed over 512–16K; its variation is dominated by
synchronization and queue timing rather than a clean positive history slope.

## 5. Nsight Systems timeline and API sequencing

Fresh captures:

- `results/oscar/d4-8a-gpu-only-16k.nsys-rep`
- `results/oscar/d4-8a-gpu-only-16k.sqlite`
- `results/oscar/d4-8a-gpu-only-16k-api-kern_cuda_api_trace.csv`
- `results/oscar/d4-8a-gpu-only-16k-api-kern_cuda_kern_exec_trace.csv`
- `results/oscar/d4-8a-gpu-only-16k-gpu-trace_cuda_gpu_trace.csv`

The isolated trace contains 8 forced decode columns across 16 full-attention layers:
128 `verify.attention` instances. The exact QKV-to-following-full-output segments contain
1,920 CUDA kernel executions, approximately 15 kernels per full-layer segment. The broad
attention ranges contain 2,048 associated launch records because asynchronous work around
the range boundary can be queued or drained later.

The representative 16K decode kernel inventory is:

| Operation | Kernel time in 128 segments | Per forced token, all 16 full layers |
| --- | ---: | ---: |
| QKV projection, `fp8_gemv_kernel<14336,5120>` | 6.218 ms | 0.777 ms |
| Three FP32 Q/K/V rotations | 3.217 ms | 0.402 ms |
| Resident encode/publish work | 0.861 ms | 0.108 ms |
| Fused OSCAR split kernel | 58.015 ms | 7.252 ms |
| Fused OSCAR partial merge | 3.023 ms | 0.378 ms |
| `R_V.T` inverse/recovery | 3.094 ms | 0.387 ms |
| Attention output projection, `fp8_gemv_kernel<5120,6144>` | 3.236 ms | 0.405 ms |

These Nsight Systems kernel times are intentionally reported as a profiled timeline, not
substituted for the clean D4.7B timer. Nsight capture overhead is material, and the broad
NVTX association can contain previously queued work. Operation-specific kernel names and
the exact QKV-to-output sequencing are used for attribution.

The host/API view inside the 128 attention ranges is the decisive new evidence:

| API | Calls | Captured host time | Mean |
| --- | ---: | ---: | ---: |
| `cudaStreamSynchronize` | 640 | 104.52 ms | 163.31 us |
| `cudaMemcpyAsync` (positions) | 128 | 84.85 ms | 662.90 us |
| `cudaLaunchKernel` | 2,048 | 14.36 ms | 7.01 us |
| `cudaMalloc` | 640 | 2.02 ms | 3.16 us |
| `cudaFree` | 640 | 3.95 ms | 6.17 us |

There are five synchronization calls and five scratch allocation/free pairs per broad
attention range on this path. These are Nsight-instrumented API durations and are not
added to the clean 22.975 ms number; they show the repeated operation boundaries that the
software timer collapses into its fixed floor. No CUDA Graph coverage is present:
D4.7B uses `use_cuda_graph=false`.

The trace also shows no material memcpy volume from the historical cache. D4.4 resident
cache staging remains zero for the production path. The position D2H copy is the repeated
small state transfer that matters here.

## 6. Nsight Compute findings for material non-OSCAR kernels

### QKV projection

A fresh targeted Nsight Compute run profiled the exact QKV kernel on the RTX 5090/CC 12.0;
hardware counters were valid and nonempty. There was no `ERR_NVGPUCTRPERM`.

| Metric | Measured value |
| --- | ---: |
| Kernel | `fp8_gemv_kernel<Fp8Geometry<14336,5120>>` |
| Grid / block | 896 CTAs / 256 threads |
| Duration | 48.83 us (profiled) |
| Registers/thread | 39 |
| Dynamic/static shared | 0 / 0 bytes |
| Achieved occupancy | 84.42% |
| Active warps/SM | 40.52 |
| DRAM throughput | 1.50 TB/s, 70.07% of peak metric |
| L2 hit rate | 2.94% |
| L1/TEX hit rate | 79.56% |
| SM compute throughput | 11.65% |
| Memory pipes busy | 9.62% |
| Tensor-core activity | 0 |
| Branch efficiency | 93.20% |

Nsight Compute classifies this kernel as memory-utilized relative to its own available
compute, but its measured timeline contribution is only about 0.777 ms per forced token
across all full layers. It is not large enough to explain the 14.687 ms remaining floor.
The low L2 hit rate is relevant to a future weight-format-specific investigation, but the
QKV kernel is not the first D4.8B target.

### Fused OSCAR context from D4.7A

The prior valid D4.7A hardware-counter profile remains the correct evidence for the
unchanged fused kernel body:

| Metric | 4K / 32 splits | 16K / 64 splits |
| --- | ---: | ---: |
| Grid | 128 CTAs | 256 CTAs |
| Block / registers | 256 / 43 per thread | 256 / 43 per thread |
| Dynamic shared memory | 73.312 KiB/block | 73.312 KiB/block |
| Achieved occupancy | 16.67% | 16.67% |
| L1/TEX long-scoreboard stall | 66.11% | 65.89% |
| Barrier stall | 8.84% | 9.01% |
| Wait stall | 11.99% | 12.20% |
| DRAM throughput | 1.37% | 1.10% |
| L2 throughput | 0.88% | 0.70% |
| Tensor-core activity | 0 | 0 |

D4.7B already addressed the grid-level occupancy problem by adaptive splitting. D4.8A
does not change that kernel. The fresh evidence says that further OSCAR optimization is
now second-order compared with the fixed per-layer runtime boundary.

## 7. Fixed versus history-dependent cost

Using the clean accounting table and a two-point slope estimate:

- full branch: about **0.335 us/history-token**, with an extrapolated fixed intercept of
  about **17.48 ms**;
- fused OSCAR: about **0.466 us/history-token**, with an extrapolated fixed intercept of
  about **0.65 ms**;
- outside-fused portion: no stable positive history slope in these measurements; it stays
  in an approximately **16–17 ms fixed floor** with synchronization noise.

Thus the D4.7B adaptive split change moved the optimization question. At 16K, the fused
history traversal is 8.288 ms, but the fixed operation-boundary floor is 14.687 ms. The
remaining 16K full-branch budget is not a DRAM-saturation problem.

## 8. Amdahl analysis

At 16K, fused OSCAR is 8.288/22.975 = **36.1%** of the measured branch. The maximum
end-to-end savings from improving OSCAR alone are bounded by that component:

| OSCAR-only improvement | Saved | Resulting branch | Speedup |
| ---: | ---: | ---: | ---: |
| 10% | 0.829 ms | 22.146 ms | 1.04x |
| 25% | 2.072 ms | 20.903 ms | 1.10x |
| 50% | 4.144 ms | 18.831 ms | 1.22x |
| 100% (theoretical ceiling) | 8.288 ms | 14.687 ms | 1.56x |

For comparison, removing only 25% of the outside-fused 14.687 ms would save about
3.67 ms, and removing 50% would save about 7.34 ms. This is why D4.7C bank-conflict
optimization is no longer the highest-value immediate choice, despite the real conflicts
measured in D4.7A.

## 9. Candidate next phases

Savings below are conservative engineering estimates, not measurements of unimplemented
variants. They use the clean D4.7B budgets and the fresh timeline to rank opportunity.

| Candidate | Current cost addressed | Realistic opportunity | Complexity / risk | EXL3 survival |
| --- | --- | --- | --- | --- |
| A. D4.7C shared-memory/bank-conflict reduction | Part of 8.288 ms OSCAR | Roughly 0.8–4.1 ms at 16K for 10–50% OSCAR improvement | Medium/high; numerical semantics low risk | Partial; OSCAR-specific |
| B. Vectorized OSCAR INT2 unpack/dequant | A subset of the OSCAR history loop | Unknown without a new microbenchmark; bounded by 8.288 ms | High; codec/layout sensitive | Partial; OSCAR-specific |
| C. Fuse `R_K` into projection/cache write | About 0.402 ms profiled rotation kernel time, plus sync opportunity | Sub-ms direct kernel saving; possibly more if it removes a boundary | Medium; numerical/layout review | Yes |
| D. Fuse `R_V.T` into AV/output | About 0.387 ms profiled inverse kernel time | Sub-ms direct kernel saving; possibly more if it removes a boundary | Medium; numerical review | Yes |
| E. Optimize NVFP4 QKV projection | About 0.777 ms/token kernel time | Less than about 0.8 ms direct at 16K | Medium/high; weight-format-specific | No, likely replaced by EXL3 |
| F. Optimize NVFP4 attention output projection | About 0.405 ms/token kernel time | Less than about 0.4 ms direct | Medium; weight-format-specific | No, likely replaced by EXL3 |
| G. Reduce launches/synchronization/scratch boundaries | 14.687 ms outside fused timer; 640 syncs and 640 alloc/free pairs in trace | Estimated 25–50% of outside floor: 3.7–7.3 ms at 16K | High; requires lifetime/graph correctness work | Yes |
| H. Begin EXL3 6 bpw H6 V6 integration | Future weight backend, no current measured cost | No honest D4.8A latency estimate before integration | High; new backend/qualification scope | It is the migration itself |

Candidate G has the largest measured addressable boundary and is the only candidate whose
benefit is largely independent of the current NVFP4 weight format. It should be attempted
before spending substantial effort on E/F. Candidate H remains important, but it would
carry the current operation-boundary problem into a new backend unless that generic issue
is first addressed.

## 10. Estimated D4.8B result

These are estimates, not qualification results. They assume fused OSCAR remains at the
qualified D4.7B time and D4.8B removes a fraction of the measured outside-fused floor.

| Context | Current full branch | Conservative (25% outside-floor reduction) | Aggressive (50% reduction) |
| ---: | ---: | ---: | ---: |
| 4K | 18.642 ms | ~14.60 ms | ~10.55 ms |
| 8K | 21.738 ms | ~17.43 ms | ~13.13 ms |
| 16K | 22.975 ms | ~19.30 ms | ~15.63 ms |

The estimate is deliberately expressed as a range because the current `qkv_rotation`
timer includes queue drain and the Nsight Systems capture perturbs launch and synchronization
latency. A D4.8B implementation should replace these estimates with clean unprofiled
measurements and retain the D4.7B fused attention time as a non-regression gate.

## 11. Recommended single next phase

**D4.8B should first implement operation-boundary reduction: persistent device-side
position/state and FP32 scratch, removal of the per-layer position D2H copy and explicit
stream synchronizations where the dependency graph allows, followed by CUDA Graph capture
of the stable one-token decode sequence.**

Reason: the fresh timeline found 640 stream synchronizations, 128 position copies, 640
temporary allocations, 640 frees, and 2,048 kernel launches across only 128 isolated
attention ranges, while QKV and output projection together contributed about 1.18 ms per
forced token. This target addresses the measured fixed floor, survives the eventual EXL3
migration, and has a larger Amdahl ceiling than an OSCAR-only bank-conflict change.

## 12. Correctness and cleanup

The final production snapshot is unchanged D4.7B:

- direct CUDA mixed-attention parity passed contexts 64, 65, 320, 321, 322, 332, 512,
  2048, 4096, 8192, 16384, and 32768;
- dispatch telemetry was `full_layer_dispatch_bitmap=1111111111111111`;
- `gdn_dispatches=0`;
- `legacy_q2_dispatches=0`;
- `bf16_historical_shadow=false`;
- `fallback=false`;
- the optimized FULL validator self-check passed;
- the optimized FULL validator passed 30 taps with 8 workers;
- live D4.6 reference taps on layers 3, 35, and 63 passed through the 16K run;
- final live qualification exited 0 and reported `OSCAR D4.6 fused live GPU runtime: COMPLETE`.

The temporary D4.8A selector was removed from source. The final rebuild had no profiling-only
selector environment variables, and `git diff --check` was clean. No cache semantics,
codec, split policy, arithmetic, prefill path, GDN behavior, or model weights were changed.

## Conclusion

D4.7B's adaptive split-KV policy has made the resident OSCAR kernel a minority of the
16K full-attention branch. The next bottleneck is the fixed host/device operation boundary:
small per-layer work is repeatedly synchronized, position state is copied to the host,
and temporary device allocations are created and destroyed. The evidence does not support
calling the current floor DRAM-bound or making D4.7C the immediate next implementation.

Recommended next phase: **D4.8B operation-boundary reduction and stable decode scheduling**.

