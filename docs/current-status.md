# Current status

## Status 2026-10-08

Toolchain: VS 2026 Build Tools (MSVC 14.51) + CUDA 13.3.1, `sm_120a`, RTX 5090 (driver 617.14).
CUDA 13.1 and 13.3 builds give identical greedy state hashes and tokens at 4K/16K.

| Capability | State/default | Measured (2026-10-08) |
| --- | --- | --- |
| Cold prefill, target-only | Default (layer-major, MXFP8/NVFP4 block-quantized projections, FA2 attention) | 10.1K tok/s at 4K, 9.6K at 16K |
| Base greedy decode, target-only | Default (whole-stack decode graph, fused reductions, L2 prefetch) | 79.2 tok/s at 4K, 75.7 at 16K |
| DFlash2 K=7 decode, public Engine (coherent-device C1) | Default (verification trees, ≤8 verify rows) | 211 tok/s at 4K, 186 at 16K (code-review workload); 241 tok/s geomean on NInfer's 12 scenario prompts |
| L0 OSCAR long context | Opt-in `NINFER_EXL3_L0_OSCAR=1`; required above the FP16-KV VRAM limit | DFlash2 178 / 164 / 180 tok/s at 32K / 64K / 128K; prefill 7.3K / 6.4K / 5.1K tok/s; +0.016 nats/token paired NLL vs FP16 KV |
| FP16 device KV with DFlash2 | Default | Fits 16K; the layer-major prefill headroom guard refuses 31K |
| Model quality vs NInfer (identical tokens) | — | FP16 KV better by 0.017–0.020 nats/token (95% CI excludes 0); L0 OSCAR statistically tied |

Head-to-head results against NInfer are in [ninfer-comparison-2026-10-08.md](ninfer-comparison-2026-10-08.md).
The 2026-09-22 publication table below remains as the historical record of that snapshot.

## Status 2026-09-22 (publication snapshot)
Status date: 2026-09-22. Publication source snapshot: 1,283 paths with aggregate SHA-256
`642169e8ccc96cfe4115868a399a3af0f6e22ffccd90667e1aea6a540b348425` before publication-only
documentation and portability edits.

| Capability | State/default | Source | Evidence and boundary | Next gate |
| --- | --- | --- | --- | --- |
| Native `.ninfer` CLI/server | Implemented | `apps/`, `src/serve/`, `src/runtime/` | Windows implementation and unit/integration tests; model artifact required | Clean hardware smoke on the publication commit |
| EXL3 greedy text serving, physical C1 | Qualified slice/default route | `src/exl3/`, `docs/exl3-serving.md` | Exact route tests and dated serving evidence; ordinary FP16 host L2 and FP32 recurrent state | Repeat clean serving smoke and client-visible TTFT |
| EXL3 physical C2 | Explicit bounded configuration | `src/exl3/` | Profile-specific C2 receipts; not a blanket concurrency claim | Held-out load/soak and memory-pressure qualification |
| EXL3 media/vision | Unsupported in public slice | frontend rejects unsupported combinations | No broad media qualification | Separate semantic and lifecycle campaign |
| Positive-temperature EXL3 sampling | Not broadly qualified | sampling policy/transaction tests | Greedy authority does not establish distribution preservation | Distribution and stop-semantics panel |
| Native same-weight DFlash2 C1 harness | Qualified research baseline; not server default | `tests/test_exl3_real_dflash_execution.h`, `tests/test_exl3_dflash2_accept.cpp` | R608: 2,822 checks, 12 exact rows, grouped workload wall | Reproduce from publication commit; separate server integration gate |
| Device-working-state route | Retained/default in its harness profile | `src/exl3/text_model.*` | R599 liveness and R597/R600 matched AB evidence | Broader quality panel before product promotion |
| Segmented resident prefix | Rejected/default off | `src/exl3/text_model.*` | R612 exact bounded screen, 17.08–21.15% slower | No promotion; revisit only with a new mechanism |
| WMMA32 prefill | Quality-pending/default off | EXL3 prefill experiments | Does not match the frozen exact token stream | Held-out quality and exactness gate |
| Native MTP | Unqualified/default off | MTP test and runner sources | Capture/lifetime evidence only; populated-prefix semantics/economics absent | Independent KV/logit/token differential, then economics |
| OSCAR/VeriCache/NVFP4 | Historical/experimental | `src/core/oscar_*`, target runtime, historical docs | Useful implementation and receipts, not the current EXL3 performance headline | Requalify per exact model/configuration |

## Corrections to historical pointers

R608 is a same-weight DFlash2 workload with target verification and an ordinary FP16 device-KV
state. The heading “target-only” in one immutable R608 handover is inaccurate; the receipt metadata
and current documentation are authoritative. R610–R612 are eight-output diagnostic screens and do
not supersede R608. R604's profiler-perturbed CUDA API proportions are attribution evidence, not
normal production wall-time shares.

The goals of 70 target-only tok/s, 200 DFlash2 tok/s, 3,000 cold-prefill tok/s, Mia parity, broad
quality qualification, Engine/client TTFT parity, and native MTP qualification remain unmet or
unmeasured. `null` means not measured; it is not inferred from nearby data.

The performance-optimization campaign remains paused. This repository publication consolidated
and documented its state without starting the next optimization.
