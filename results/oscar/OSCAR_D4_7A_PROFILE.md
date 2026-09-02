# OSCAR D4.7A — SM120a Mixed-Attention Hardware Profiling

## Status: PAUSED — NVIDIA performance-counter permission required

This is an interim, non-final D4.7A record. Nsight Systems timing/sequencing
captures are complete for the requested isolated one-token decode fixtures, but
the Nsight Compute hardware-counter run is blocked by the local NVIDIA driver
profiling-security policy. No D4.7B implementation recommendation is made from
this incomplete measurement set.

## Reproducibility

- Branch: `codex/oscar-d4-7a-hardware-profile-20260902`
- Commit at pause: `17b9211874080c944d95389218b836ebbb88b067`
- Repository: `D:\AI\voidinfer-adaptive-dflash2`
- Build: `D:\AI\build-adaptive-dflash2`, `RelWithDebInfo`, native Windows x64
- GPU: NVIDIA GeForce RTX 5090, compute capability 12.0 / SM120a, 32,607 MiB
- Driver: 610.88
- CUDA toolkit: 13.1; driver-reported CUDA: 13.3
- Nsight Compute: 2026.2.1.0
- Nsight Systems: 2026.4.1.191
- Model default: C-drive artifact, SHA-256
  `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`

The temporary selector is test-only (`NINFER_OSCAR_D4_7A_PROFILE_CONTEXT` and
`NINFER_OSCAR_D4_7A_PROFILE_REPETITIONS`) and calls the qualified D4.6 fused
resident decode path. It does not change cache representation or production
dispatch. It remains present only so the counter-enabled rerun can be made
without changing the profiled workload.

## Exact counter blocker

Nsight Compute attached to the test process, but every attempted kernel capture
failed before collecting a report:

```text
==PROF== Connected to process ... (ninfer_oscar_mixed_gpu_attention_test.exe)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

The same error occurred with the normal sandboxed invocation and with an
escalated invocation. No `.ncu-rep` was produced. Therefore the following are
not yet measured and must not be inferred as measured:

- achieved occupancy, active/eligible warps, and register-limited occupancy;
- shared-memory bank conflicts and barrier/dependency stall reasons;
- DRAM/L2 throughput and hit rate, global-load efficiency, and memory stalls;
- integer unpack pressure, SIMT QK/AV utilization, instruction mix, and warp
  divergence.

Nsight Systems is not being used as a substitute for these counters.

## D4.6 baseline retained for comparison

The qualified D4.6 one-token full-attention branch, across all 16 full-attention
layers, reported:

| History | Total full-attention | Mixed component |
| ---: | ---: | ---: |
| 512 | 15.529 ms | 2.265 ms |
| 2K | 21.186 ms | 7.917 ms |
| 4K | 28.790 ms | 15.106 ms |
| 8K | 43.774 ms | 30.074 ms |
| 16K | 75.286 ms | 61.188 ms |

D4.6 real-model fused mixed-prefill timings were 41.886 ms at 512, 587.968 ms
at 2K, 2.285960 s at 4K, 9.042220 s at 8K, and 36.031300 s at 16K.

## Nsight Systems timing/sequencing captures

Command family: `nsys profile --trace=cuda --sample=none
--capture-range=cudaProfilerApi --stats=true`. Each capture is an isolated
synthetic resident fixture using the existing D4.6 decode split kernel followed
by its merge kernel. Timings below are Nsight Systems GPU-kernel averages; the
merge fraction is computed from the two measured kernel durations.

| Context | Repetitions | Split kernel | Merge kernel | Split + merge | Merge fraction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 20 | 69.651 us | 1.890 us | 71.541 us | 2.64% |
| 2K | 10 | 268.177 us | 1.875 us | 270.052 us | 0.69% |
| 4K | 10 | 540.394 us | 1.898 us | 542.292 us | 0.35% |
| 8K | 5 | 1.210239 ms | 1.914 us | 1.212153 ms | 0.16% |
| 16K | 3 | 2.674217 ms | 1.952 us | 2.676169 ms | 0.073% |
| 32K | 2 | 6.086107 ms | 1.936 us | 6.088043 ms | 0.032% |

The captures show exact kernel ordering and API sequencing: one split-kernel
launch and one merge-kernel launch per profiled decode call, with a host
`cudaDeviceSynchronize` after the call. They do not expose the unavailable
hardware counters or phase-level resource metrics. The 32K capture has only two
repetitions because the fixture is intentionally diagnostic and synthetic.

Capture files are retained locally under:
`results/oscar/d4-7a-profile/`.

## Pending work after counter access is enabled

1. Rerun Nsight Compute for isolated 4K, 8K, 16K, and 32K decode cases with
   the full requested section set.
2. Add the measured occupancy, stall, memory, instruction, unpack, QK, softmax,
   AV, and split/merge evidence to this report.
3. Rank bottlenecks only after those counters are available.
4. Remove the temporary selector/instrumentation.
5. Rerun the existing direct CUDA parity test and optimized FULL validator.
6. Complete the final report and make exactly one evidence-backed D4.7B
   recommendation.

No D4.7B code, DFlash2 work, EXL3 work, or OSCAR cache-semantic change has been
started.
