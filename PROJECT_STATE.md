# VoidInfer project state

Updated: 2026-08-26 UTC

## Canonical state

- Project: VoidInfer / NInfer native Windows RTX 5090 engine
- Canonical checkout: `D:\AI\voidinfer`
- Branch: `master`
- HEAD: `2d6e3a35`; pushed on `origin/phase1/bootstrap-state-20260826`; PR #1 is open against protected `master`
- Last-known-good source commit: `2d6e3a35` (`docs(state): track bootstrap pull request`)
- Last-known-good build: native Visual Studio 17 2022 x64, Release, CUDA `120a`, tests enabled; build and 89-test CTest run pass; benchmark target also built
- Configured push remote: `origin` (`https://github.com/The8Darkness/voidinfer.git`)
- Last accepted pushed commit: `2d6e3a35` on `origin/phase1/bootstrap-state-20260826`; direct `master` push is blocked by required status check `secret-and-artifact-checks`
- Current phase: Phase 1, reproducible native-Windows baseline and audit scaffolding; PR #1 awaits protected-branch CI
- Active hypothesis: qualify canonical `Neroued/ninfer` `dev` context/resource implementation before designing a competing prefix cache

## Local environment

- OS: Windows 11 Pro 25H2, build 26200, x64
- GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, UUID `GPU-878c0965-6802-7d0d-e793-0c3d1962559f`
- Driver: `616.56`; CUDA Toolkit: `13.1.80`; required architecture: `sm_120a`
- MSVC: Visual Studio Build Tools 17.14.29 / MSVC 19.44 / Windows SDK 10.0.26100
- CMake: 3.31.6; Ninja: 1.12.1; Git: 2.45.1; Python: 3.12.10
- Nsight Systems: 2026.4.1; Nsight Compute: 2026.2.1
- Physical RAM: about 102.7 GB; D: NTFS has about 266 GB free at inventory time
- vcpkg toolchain: `C:\BuildTools\VC\vcpkg\scripts\buildsystems\vcpkg.cmake`; triplet `x64-windows`

## Local model artifacts

| Artifact | Bytes | SHA-256 | Classification |
| --- | ---: | --- | --- |
| `models/Qwen3.8-27B-NVFP4-Ninfer/qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` | Qwen3.8 target, NVFP4; local artifact |
| `models/Qwen3.8-27B-DSpark-NInfer/qwen3_8_27b_dspark.ninfer` | 20,929,108,994 | `a6501d2793c8b822aecd975a3d9f7a552646f58af5549c818dd5d4219236c2ce` | Identity reports `qwen3.8-27b/groupwise-int`; compatibility with DSpark remains unverified |

Metadata inspection reports 1,118 tensors + 6 resources for NVFP4 and 1,165 tensors + 6 resources for the DSpark-named artifact. No Qwen3.6, 35B, tokenizer-only, or separate DFlash2 artifact is present locally.

## Baseline evidence

Configure (after relocation):

```text
cmake --fresh -S . -B build-windows -G "Visual Studio 17 2022" -A x64
  -DCMAKE_TOOLCHAIN_FILE=C:/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=120a
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
```

Build and test:

```text
cmake --build build-windows --config Release -j
ctest --test-dir build-windows -C Release --output-on-failure
```

Result: `100% tests passed, 0 failed out of 89`; five registered real-model tests skipped because their required artifacts/fixtures are absent. Build succeeds with existing MSVC module-dependency and signed/unsigned comparison warnings; no source changes were made for those warnings. Full evidence is in `results/bootstrap-baseline-2026-08-26.json`.

The Qwen3.8 NVFP4 artifact inspects successfully and `ninfer-serve.exe --help` exposes vision, MTP, KV dtype, prefix reuse, Responses, and tool-serving options. A bounded local service run with vision + MTP3 passed `tools/smoke/serve_contract.py` and was stopped cleanly. A one-repetition local benchmark measured prefill 7,091.6 tok/s at 512 tokens, 8,027.7 at 4,096, 7,629.2 at 8,192, and MTP3 decode 91.88 output tok/s with 24.1% draft acceptance; this is a preliminary `LOCAL_MEASUREMENT`, not a stable performance baseline. Raw report: `results/qwen3.8-nvfp4-mtp3-short-baseline.json`.

## Fetched audit refs (2026-08-26)

- natpate `upstream/master`, `upstream/dev`: `b686696eebd4`
- Neroued `canonical/master`: `feaf4dd0983f`; `canonical/dev`: `59febed27ca`
- headpiece: `c1b0ad34d438`; q27 current master: `9b10e16fc568`
- SM120 overlay: `5f2fd9d76544`; vLLM: `0a5ad6f0d42c`; vLLM PR #52816: `3406ec1dae99`
- FlashInfer: `919a24e5b1d9`; PR #4346: `bce8f44e319`
- Speculators: `51f8e02f077c`; SGLang: `e7e78940168f`; KVarN: `7586257f1c63`
- Model Optimizer: `d35bf8919c52`; KVPress: `161705a68f64`; MHA2MLA-VLM: `69a3c6e3116c`; Picot: `6e9fddd453a9`

Detailed provenance and dispositions: `UPSTREAM_AUDIT.md`, `Q27_AUDIT.md`, `SM120_DFLASH_AUDIT.md`.

## Current blockers and next steps

1. Add Windows-aware defaults to benchmark/smoke tools and separate text-only from vision-enabled serving smoke.
2. Build the Phase 1 baseline matrix against the local Qwen3.8 NVFP4 artifact; classify all results as local measurements.
3. Add a real Qwen3.8 integration test and enable artifact-backed smoke only when bounded and reproducible.
4. Integrate canonical `dev` only in an isolated experiment worktree, beginning with the coupled StateImage + paged-KV + Engine/resource cutover; preserve a rollback point.
5. Do not start DFlash2, NVFP4-KV, WHT, VeriCache, or broad scheduler work before exact baseline and quality gates exist.

The Picot launcher and harness now use `D:\AI\voidinfer`; the source repository was relocated from the obsolete `D:\AI\Pi\engine` path without overwriting the existing `models` directory. The launcher/harness edits are outside this git checkout and remain local configuration.
