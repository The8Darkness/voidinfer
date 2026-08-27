# VoidInfer project state

Updated: 2026-08-28 UTC

## Canonical state

- Project: VoidInfer / NInfer native Windows RTX 5090 engine
- Canonical checkout: `D:\AI\voidinfer`
- Branch: `handoff/phase2-context-resource-20260827`
- PR #2: `https://github.com/The8Darkness/voidinfer/pull/2`
- PR2 source handoff/head: `c3ca3cb9ef6d2ed6d899d57a6c56e5f4cddc3f41`
- Orchestration prompt: `MASTERPROMPT_MUSE_SPARK_1_2_CONTRIBUTOR.md`
- Canonical integration checkpoint: `21de38d7a8` (`feat(runtime): integrate canonical context resource runtime`); contains the committed context/resource cutover through `59febed2`, Windows fixes, qualification tests, and benchmark-harness fixes
- Handoff documentation commits: `8492f900f8` (`docs(state): record phase2 handoff`), `90e76f3d4c` (GitHub handoff link), and `d4fce7aa1f` (Muse Spark master prompt)
- Previous rollback checkpoint: `9a6813a3` (`fix(windows): complete upstream f0 build integration`); keep it available until the refreshed canonical head is separately qualified
- PR2 local Release configure/build: unique worker directory `build-pr2-retry`, native Visual Studio 17 2022 x64, MSVC `19.44`, CUDA `13.1.80`, `sm_120a`; configure/build passed
- PR2 exact Release CTest: `94` entries, `100%` passed, `0` failures, `5` expected skips for absent Qwen3.6/35B artifacts
- PR2 focused repair tests: `ninfer_tool_call_parser_test.exe` passed with `ok`; `ninfer_qwen3_6_context_store_test` passed `1/1` with `0` failures in `0.10` seconds; result log: `D:\AI\agent-orchestrator\data\worktrees\voidinfer\voidinfer-24\build-pr2-repair-retry\context-store-test.log`
- Configured push remote: `origin` (`https://github.com/The8Darkness/voidinfer.git`); this state update is committed and pushed to the handoff branch
- Current phase: Phase 2 validation/handoff; PR2 qualification: Release configure/build passed, exact Release CTest passed, and direct Qwen3.8 NVFP4 route passed
- Active state: direct Qwen3.8 NVFP4 route is qualified; prior code-9 runs remain retained only as superseded invocation diagnostics

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
| `D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-Ninfer\qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` | Qwen3.8 target, NVFP4; canonical direct-route artifact; exact binding variable `NINFER_QWEN3_8_27B_NVFP4_WEIGHTS` |
| `models/Qwen3.8-27B-DSpark-NInfer/qwen3_8_27b_dspark.ninfer` | 20,929,108,994 | `a6501d2793c8b822aecd975a3d9f7a552646f58af5549c818dd5d4219236c2ce` | Identity reports `qwen3.8-27b/groupwise-int`; compatibility with DSpark remains unverified |

Metadata inspection reports 1,118 tensors + 6 resources for NVFP4 and 1,165 tensors + 6 resources for the DSpark-named artifact. No Qwen3.6, 35B, tokenizer-only, or separate DFlash2 artifact is present locally.

## Baseline evidence

Release configure (unique PR2 worker directory `build-pr2-retry`):

```text
cmake --fresh -S . -B build-pr2-retry -G "Visual Studio 17 2022" -A x64
  -DCMAKE_TOOLCHAIN_FILE=C:/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=120a
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
```

Release build and exact CTest:

```text
cmake --build build-pr2-retry --config Release -j
ctest --test-dir build-pr2-retry -C Release --output-on-failure
```

Result: configure/build passed on native Visual Studio 17 2022 x64 with MSVC `19.44`, CUDA `13.1.80`, and `sm_120a`. The exact `94`-entry Release CTest passed `100%`, with `0` failures and `5` expected skips for absent Qwen3.6/35B artifacts.

Direct Qwen3.8 NVFP4 route:

```text
set NINFER_QWEN3_8_27B_NVFP4_WEIGHTS=D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-Ninfer\qwen3_8_27b_nvfp4.ninfer&&D:\AI\voidinfer\build-windows-phase7\tests\Release\ninfer_qwen3_6_27b_prefix_real_test.exe
```

Classification: `LOCAL_MEASUREMENT`.

The exact direct route used the retained canonical `build-windows-phase7` Release binary, the canonical Qwen3.8-27B NVFP4 artifact, and `NINFER_QWEN3_8_27B_NVFP4_WEIGHTS`. It ran once through `cmd.exe`, exited with code `0` in `15.54` seconds, and produced `4` bytes (`ok` plus newline) on stdout and `0` bytes on stderr. The GPU returned idle with no NInfer process or service. Result paths: `D:\AI\voidinfer\profiles\bench\pr2-route-recheck-20260827\stdout.log` and `D:\AI\voidinfer\profiles\bench\pr2-route-recheck-20260827\stderr.log`.

The earlier local `build-pr2-retry` and inherited `build-windows-phase7` code-9 / CTest `0xc0000409` exits, plus the no-environment control's expected skip code `77`, are retained as superseded invocation diagnostics. They are not current source blockers or evidence against PR2 qualification.

Known evidence paths: `build-pr2-retry/`, `build-windows-phase7/`, `results/qwen3.8-nvfp4-phase1-baseline-2026-08-26.json`, the canonical artifact path listed above, and the two exact route result paths above.

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

1. None. Release configure/build, exact Release CTest, and the direct Qwen3.8 NVFP4 route passed; prior code-9 results remain superseded invocation diagnostics only.

The Picot launcher and harness use `D:\AI\voidinfer`; the source repository was relocated from the obsolete `D:\AI\Pi\engine` path without overwriting the existing `models` directory. The launcher/harness edits are outside this git checkout and remain local configuration. Windows, WebUI, and local workflow authorities remain preserved, as does the user's dirty prompt deletion.

## AO checkpoint

- Orchestrator session: `voidinfer-6`; harness: `omp`; exact model: `openai-codex/gpt-5.6-luna`.
- Worker: session `voidinfer-24`, name `PR2 repair retry`; workspace `D:\AI\agent-orchestrator\data\worktrees\voidinfer\voidinfer-24`; ownership: PR #2 repair takeover.
- Prior qualification, integration, and monitor sessions are exited.
- PR #2 source/head now: `handoff/phase2-context-resource-20260827` / `c3ca3cb9ef6d2ed6d899d57a6c56e5f4cddc3f41`; required check passing; reviews: none.
- GPU at checkpoint: RTX 5090, 32,607 MiB; about 2.3 GiB desktop/AO use; 0% utilization. No NInfer model, service, build process, or owned service to restore; no active GPU job remains.
- Windows/WebUI/local workflow authorities and the user's dirty canonical deletion remain untouched. The direct route is qualified; prior code-9 results are superseded invocation diagnostics.
