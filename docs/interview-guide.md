# Interview and demonstration guide

## 30-second introduction

“VoidInfer is an experimental native Windows C++/CUDA inference engine for Qwen3.8-27B on Blackwell.
I used it to study EXL3 execution and exact-state speculative decoding under tight GPU-memory and
lifecycle constraints. The strongest part is the evidence discipline: serving and benchmark paths
are separate, every performance result is tied to source/binary/model identity and exact state gates,
and slower or numerically incomplete routes remain default-off.”

## Two-minute explanation

Start at the request/Engine boundary. The Engine owns admission, contexts, host/device resources, and
publication. Prefill materializes prompt state; decode may use a DFlash2 proposal, but the target
remains authoritative. Each round is transactional: provisional KV and recurrent GDN state are
committed only for accepted tokens and rolled back on rejection. Specialization for `sm_120a`, EXL3
weights, and fixed shapes creates speed opportunities, while exact state comparisons, ownership
records, and matched AB measurements limit what can safely become a default. R608 is a grouped
research workload—not HTTP throughput—and its best recorded arm is 47.83 useful tok/s.

## Five-minute demo

Primary route (requires a locally licensed `.ninfer` artifact and a build validated on the current
machine):

```powershell
$env:VOIDINFER_MODEL = 'C:\models\qwen3_8_27b.ninfer'
& .\build\apps\ninfer-serve.exe $env:VOIDINFER_MODEL --host 127.0.0.1 --port 8080
```

Show the startup log's effective configuration, send one temperature-zero loopback request using
[build-and-run.md](build-and-run.md), and point out that this proves only the serving smoke.

Fallback (no model/GPU): open [R608 evidence](../evidence/r608/README.md), recompute one throughput
row from `workloads.csv`, then trace the exactness/timing boundary into
`tests/test_exl3_real_dflash_execution.h`. This is a dated recorded-evidence demo, not a live result.

## Code tour

1. `include/ninfer/engine.h` and `src/runtime/engine/`: public lifecycle, admission, ownership.
2. `src/exl3/text_model.*`: EXL3 target/draft execution and state.
3. `src/ops/` and `src/exl3/kernels/`: specialized CUDA dispatch and kernels.
4. `tests/test_exl3_dflash2_accept.cpp`: test entry and qualification gates.
5. `tests/test_exl3_real_dflash_execution.h`: frozen workload, exactness, and timing boundary.
6. `src/serve/` and `apps/serve/main.cpp`: HTTP schemas, error contracts, streaming, startup.

## Technical questions and answer notes

1. **Why quantize weights?** Reduce VRAM and bandwidth; accuracy, codec overhead, and specialized
   kernels become part of the contract.
2. **What usually limits decode?** Often memory movement and launch/serialization at small batch,
   but R610 shows CPU graph submission was only about 1–2% here; measure before asserting.
3. **Why specialize for Blackwell?** It enables architecture-specific CUDA/Tensor Core paths and
   predictable tuning, at the cost of portability and a larger qualification burden.
4. **What is speculative decoding?** A draft proposes multiple tokens; the target verifies and
   publishes an accepted prefix without changing the target distribution/authority.
5. **Why is rollback difficult?** KV plus recurrent GDN state, taps, ring indices, and ownership
   must return to one consistent prefix after partial rejection.
6. **KV state versus recurrent state?** Attention stores per-token K/V history; GDN-style layers
   carry compact recurrent state with different update and rollback semantics.
7. **Why isn't token equality enough?** Hidden represented state can diverge and fail later; R49
   matched token subgroups but failed state/tap/ring equality.
8. **How is throughput defined?** Useful accepted/published outputs divided by a precisely declared
   wall boundary. R608 uses 384 outputs over grouped workload wall, not decode-column medians.
9. **Why keep instrumentation off for baselines?** Synchronization and profiling perturb launch
   overlap and phase timing. Attribution and production-style measurements answer different questions.
10. **How do you prevent use-after-free?** Explicit resource ownership, request/lane records,
    teardown ordering, and lifecycle/failure tests; asynchronous GPU work must finish before owners die.
11. **How are defaults selected?** Exactness, actual dispatch, frozen workload, matched controls,
    and provenance. An implementation can remain useful and default-off.
12. **What would you do next?** First reproduce the publication commit and complete the held-out
    quality/TTFT gates; only then profile the exact verifier and choose a bounded optimization.

## Defensible portfolio bullets

- Integrated and qualified a Windows C++/CUDA Qwen3.8 inference research stack with explicit GPU,
  host-state, rollback, and request-lifetime ownership boundaries.
- Improved a matched exact-state DFlash2 workload by 5.88–6.65% across four code/prose cold/reused
  arms while preserving 2,822 recorded correctness checks.
- Repaired benchmark instrumentation to separate synchronization-heavy attribution from production-
  style wall timing, preventing profiler artifacts from being reported as speedups.
- Built evidence and maintenance tooling that preserves negative results, verifies source/binary
  provenance, and safely reclaims generated build/result storage.

## Know before presenting

Be ready to explain EXL3 quantization, bandwidth versus compute limits, CUDA Graph tradeoffs,
speculative acceptance, KV/GDN rollback, greedy versus sampling correctness, useful-token accounting,
source/binary/model provenance, upstream lineage, and why the unfinished quality and TTFT gates matter.
