# VoidInfer project state

Updated: 2026-08-27 UTC

## Canonical state

- Project: VoidInfer / NInfer native Windows RTX 5090 engine
- Canonical checkout: `D:\AI\voidinfer`
- Branch: `handoff/phase2-context-resource-20260827`
- Handoff PR: `https://github.com/The8Darkness/voidinfer/pull/2`
- Orchestration prompt: `MASTERPROMPT_MUSE_SPARK_1_2_CONTRIBUTOR.md`
- State checkpoint commit: `21de38d7a8` (`feat(runtime): integrate canonical context resource runtime`); this is the committed handoff checkpoint containing the canonical `dev` runtime/resource cutover through `59febed2`, Windows fixes, qualification tests, and benchmark-harness fixes
- Handoff documentation commit: `8492f900f8` (`docs(state): record phase2 handoff`)
- Previous rollback checkpoint: `9a6813a3` (`fix(windows): complete upstream f0 build integration`); keep it available until the refreshed canonical head is separately qualified
- Last-known-good build: native Visual Studio 17 2022 x64, Release, CUDA `13.1`, `sm_120a`; `build-windows-phase7` full build and 94-test CTest run pass, with 5 expected artifact skips; benchmark and serve targets built
- Configured push remote: `origin` (`https://github.com/The8Darkness/voidinfer.git`)
- Pushed handoff branch: `origin/handoff/phase2-context-resource-20260827` at `8492f900f8`; direct `master` push was rejected by the required status check `secret-and-artifact-checks`, so review/merge through PR #2
- Current phase: Phase 2 validation/handoff; the committed context/resource integration is locally qualified on Qwen3.8 NVFP4, while the refreshed canonical head remains a separate next step
- Active hypothesis: the canonical content-indexed State/KV/resource path is the correct prefix-cache substrate; qualify it on Qwen3.8 before adding persistence, proposer routing, or codecs

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

Configure (after relocation; the recorded qualification build is `build-windows-phase7`):

```text
cmake --fresh -S . -B build-windows-phase7 -G "Visual Studio 17 2022" -A x64
  -DCMAKE_TOOLCHAIN_FILE=C:/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=120a
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
```

Build and test:

```text
cmake --build build-windows-phase7 --config Release -j
ctest --test-dir build-windows-phase7 -C Release --output-on-failure
```

Result: `100% tests passed, 0 failed out of 94`; five registered Qwen3.6/35B real-model tests skipped because their required artifacts/fixtures are absent. The same executable's opt-in Qwen3.8 NVFP4 resource-settlement route passes when `NINFER_QWEN3_8_27B_WEIGHTS` is set. Build succeeds with existing MSVC module-dependency and signed/unsigned comparison warnings; no source changes were made for those warnings. Full local evidence is in `results/qwen3.8-nvfp4-phase1-baseline-2026-08-26.json`.

The Qwen3.8 NVFP4 artifact inspects successfully and `ninfer-serve.exe --help` exposes vision, MTP, KV dtype, prefix reuse, Responses, and tool-serving options. Separate native service runs passed the text-only and Vision contract smoke; the Qwen3.8 NVFP4 public-Engine gate now covers Host State/KV restore, multimodal reuse, cancellation/eviction, and concurrent MTP settlement and passes. The 43-case C1 core matrix passed, with MTP3 graph decode ranging from 133.97 to 172.33 tok/s over the selected generation lengths and MTP5 from 107.99 to 151.18 tok/s. Separate decode-saturation runs cover C1/C2/C4/C8: MTP0 steady aggregate 80.4/157.5/292.9/538.5 tok/s and MTP3 171.7/329.8/568.2/1046.8 tok/s. The MTP0 corpus completed 20/20 long-context requests at 8k/64k/128k/256k, and the MTP3 corpus completed 75/75 long-decode, code, story, translation, and structured requests; these remain performance/behavior measurements rather than the complete quality/VLM gate. FP8-KV synthetic prefill completed at 252,928 tokens in 102.87 s (2,458.64 tok/s) and 262,144 tokens in 110.42 s (2,373.99 tok/s). Reports: `results/qwen3.8-nvfp4-phase1-baseline-2026-08-26.json` and ignored `profiles/bench/qwen38-*`.

## Fetched audit refs (2026-08-27)

- natpate `upstream/master`, `upstream/dev`: `b686696eebd4`
- Neroued `canonical/master`, `canonical/dev`: `9dbc074005a1`; committed local integration is qualified through `59febed27ca` in `21de38d7a8`
- headpiece: `c1b0ad34d438`; q27 current master: `9b10e16fc568`
- SM120 overlay: `351cf46c8910`; vLLM: `75dea9b4ae9e`; vLLM PR #52816: `3406ec1dae99`
- FlashInfer: `39b484f1ce2f`; PR #4346: deleted from the fetched remote and requires re-audit
- Speculators: `973fd49c81d0`; SGLang: `8ad76415e2db`; KVarN: `7586257f1c63`
- Model Optimizer: `449a39922b5f`; KVPress: `161705a68f64`; MHA2MLA-VLM: `69a3c6e3116c`; Picot: `6e9fddd453a9`

Detailed provenance and dispositions: `UPSTREAM_AUDIT.md`, `Q27_AUDIT.md`, `SM120_DFLASH_AUDIT.md`.

## Current blockers and next steps

1. The current single three-coefficient context-cost transfer roofline rejects two runs on fragmented D2H/D2D/H2D geometries; keep conservative compiled defaults and improve the fit or fixture before publishing a measured preset.
2. C1/core, separate C2/C4/C8 decode-saturation, MTP0/MTP3 corpus, and 252,928/262,144 FP8-KV capacity baselines exist, but closed-loop agent task time, long-generation quality, and the complete VLM suite remain outstanding.
3. The staged canonical context/resource cutover through `59febed2` has passed its local Qwen3.8 Host State/KV, media-reuse, cancellation/eviction, and MTP settlement gate; the refreshed canonical `9dbc074005a1` pressure-planner/tool/parser/anchor series still requires a separate integration worktree and Windows qualification.
4. Do not start DFlash2, NVFP4-KV, WHT, VeriCache, or broad scheduler work before those refreshed-runtime, exact-quality, and VLM gates exist.

The Picot launcher and harness now use `D:\AI\voidinfer`; the source repository was relocated from the obsolete `D:\AI\Pi\engine` path without overwriting the existing `models` directory. The launcher/harness edits are outside this git checkout and remain local configuration.

For the next agent: start with this file, `KNOWN_ISSUES.md`, `EXPERIMENTS.md`, and `UPSTREAM_AUDIT.md`; do not select an unqualified artifact by glob. The immediate next technical step is a separate worktree/branch for the refreshed canonical `9dbc074005a1` pressure-planner/tool/parser/anchor series, followed by the same native build, 94-test CTest run, direct Qwen3.8 resource gate, and focused corpus checks.
