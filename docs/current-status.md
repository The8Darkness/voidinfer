# Current status

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
