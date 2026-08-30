# Experiment registry

All numbers must be classified as `PUBLISHED_EXTERNAL_RESULT`, `LOCAL_BASELINE`, `LOCAL_MEASUREMENT`, `MODELLED_ESTIMATE`, or `STRETCH_TARGET`.

| ID | Hypothesis | Status | Owner | Gate / rollback |
| --- | --- | --- | --- | --- |
| `E000` | Native Windows build can be reproduced from the relocated checkout with local dependencies | accepted | integrator | CMake Release build + exact 94-entry CTest (`0` failures, `5` expected skips); rollback is source commit `3e2a28be` |
| `E001` | Correct root/scaffolding and artifact hashes make subsequent measurements restartable | accepted | integrator | `PROJECT_STATE.md` + baseline JSON + clean git state |
| `E002` | Windows-aware serving/benchmark defaults remove path-only smoke failures without changing runtime semantics | accepted | Windows/benchmark | Python compile/import checks, benchmark unit test, and native Windows binary-path resolution passed; Linux paths remain unchanged |
| `E003` | Canonical dev StateImage/paged-KV/Engine cutover can be integrated as one reviewable Windows worktree | accepted locally | upstream/runtime | staged integration tree builds; 94-test CTest passes with 5 expected skips; Qwen3.8 NVFP4 resource-settlement route passes; retain `9a6813a3` as rollback until review/commit |
| `E004` | NInfer parser drift fixtures improve agent tool fidelity without changing canonical templates | pending | serving | streaming/differential/security corpus; stop after three attempts |
| `E005` | The Qwen3.8 NVFP4 artifact supports native-Windows text and Vision serving through the staged Engine/resource tree | accepted locally | Windows/benchmark | separate text-only and `--vision` contract smokes pass; reports under `profiles/bench/qwen38-nvfp4-{text,vision}-smoke-20260826` |
| `E006` | The native Qwen3.8 NVFP4 path reaches the published-style 252,928/262,144 context attempt with FP8 KV without OOM | accepted locally | benchmarking | 252,928 and 262,144 token synthetic prefill complete; capacity/prefill only, not quality or C2/C4/C8 evidence |
| `E007` | The current three-coefficient transfer roofline is stable enough to publish a machine preset | rejected | context-cost | two transfer runs rejected D2H/D2D/H2D held-out p95 gates; keep compiled conservative defaults and improve model/fixture before retry |
| `E008` | The Qwen3.8 NVFP4 serving corpus exposes reproducible long-context, reasoning, code, translation, story, and structured-output behavior under MTP0/MTP3 | accepted locally | benchmarking | MTP0 20/20 and MTP3 75/75 formal requests complete with request/log counter checks; reports under `profiles/bench/qwen38-serve-corpus-20260827*`; outputs are measurements, not a quality gate |
| `E009` | Extending the fused NVFP4 SwiGLU W4A4 route from T=48 to T=64 improves a ragged prefill boundary | rejected | nvfp4-mlp | Focused correctness passed, and the isolated T=64 microbench improved `138.048`→`70.272` µs with Nsight Systems confirming two launches instead of three; real Qwen3.8 MTP3 saturation was neutral at C=1 (`168.7`→`169.2` tok/s) and slightly negative at C=4 (`544.2`→`542.8` tok/s), so the schedule was rolled back |

Rejected/closed experiments must retain their manifests, evidence, and reason here. No optimization result is accepted without correctness and task-time evidence.
