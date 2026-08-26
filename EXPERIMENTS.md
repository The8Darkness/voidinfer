# Experiment registry

All numbers must be classified as `PUBLISHED_EXTERNAL_RESULT`, `LOCAL_BASELINE`, `LOCAL_MEASUREMENT`, `MODELLED_ESTIMATE`, or `STRETCH_TARGET`.

| ID | Hypothesis | Status | Owner | Gate / rollback |
| --- | --- | --- | --- | --- |
| `E000` | Native Windows build can be reproduced from the relocated checkout with local dependencies | accepted | integrator | CMake Release build + 89-test CTest; rollback is source commit `3e2a28be` |
| `E001` | Correct root/scaffolding and artifact hashes make subsequent measurements restartable | accepted | integrator | `PROJECT_STATE.md` + baseline JSON + clean git state |
| `E002` | Windows-aware serving/benchmark defaults remove path-only smoke failures without changing runtime semantics | pending | Windows/benchmark | Python tests, `--help`/dry-run; revert if Linux defaults or command manifests regress |
| `E003` | Canonical dev StateImage/paged-KV/Engine cutover can be integrated as one reviewable Windows worktree | pending | upstream/runtime | focused state/KV tests, full CTest, Windows build; discard worktree on ownership/build failure |
| `E004` | NInfer parser drift fixtures improve agent tool fidelity without changing canonical templates | pending | serving | streaming/differential/security corpus; stop after three attempts |

Rejected/closed experiments must retain their manifests, evidence, and reason here. No optimization result is accepted without correctness and task-time evidence.
