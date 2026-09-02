# VoidInfer project state

Updated: 2026-09-02 UTC

## Canonical state

- Project: VoidInfer / NInfer native Windows RTX 5090 engine
- Canonical checkout: `D:\AI\voidinfer`
- Branch: `handoff/phase2-context-resource-20260827`
- PR #2: `https://github.com/The8Darkness/voidinfer/pull/2`
- PR2 source handoff/head: `6f45851629bee2b691a324767b6b2965955428f` (docs tip); functional repair commit: `c3ca3cb9ef6d2ed6d899d57a6c56e5f4cddc3f41`
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
- PR #2 source/head now: `handoff/phase2-context-resource-20260827` / `6f45851629bee2b691a324767b6b2965955428f` (docs tip; functional repair commit: `c3ca3cb9ef6d2ed6d899d57a6c56e5f4cddc3f41`); required check passing; reviews: none.
- GPU at checkpoint: RTX 5090, 32,607 MiB; about 2.3 GiB desktop/AO use; 0% utilization. No NInfer model, service, build process, or owned service to restore; no active GPU job remains.
- Windows/WebUI/local workflow authorities and the user's dirty canonical deletion remain untouched. The direct route is qualified; prior code-9 results are superseded invocation diagnostics.
- Planner integration checkpoint (2026-08-28): canonical planner provenance is `3e903b704ce40eb74cf3b1c8e16261978536e547` (parent `59febed27ca2beec156c056ad9a0a74537d03ac0`); accepted local chain is `da5b4e5b49`, `a2aa655244`, `47bf869d7f`, `59f2dcb045`, and `1fda1961c2`. PR #2 source/head after integration is `handoff/phase2-context-resource-20260827` / `1fda1961c208689a83b22e2c260618ee58b2855a`. AO session `voidinfer-31` qualified the absolute CMake/vcpkg build of `ninfer_resource_manager_test` and `ninfer_request_log_test`; exact CTest `2/2` passed, `0` failures, total `0.04s`; result build: `D:\AI\agent-orchestrator\data\worktrees\voidinfer\voidinfer-31\build-planner-compile-repair-v2`. No GPU/model/service run occurred for this planner slice. Direct-route and PR2 baseline evidence above remain preserved. Next ordered action: separately qualify adjacent canonical fixes `123bf1a11c`, `1d13c21336`, `fbd04729c8`, and `ab82f88603`.
- Device lifecycle checkpoint (2026-08-28): canonical provenance `1d13c21336f66b200028a0ce4e029c2b58a3d61b`; accepted candidate `d16cb2deb06c3bb012349686d7a517b123e842b5` (parent `78a4933af1bba7c57a549389bcb107fb22cb3bae`), reviewer `voidinfer-36`: ACCEPT, no findings. `build-device-v1` exact configured build of `ninfer_device_test` and `ninfer_engine` succeeded; exact Debug CTest `-R ^ninfer_device_test$` passed `1/1` in `0.15s`. Runtime evidence is limited to GPU/device 0. Planner and warmup evidence remain preserved; no empty warmup marker was added. Reverse-apply showed `123bf1a11c` is already present in `78a4933af1bba7c57a549389bcb107fb22cb3bae`, so it is not a new diff. Next ordered action: sparse-MoE sync `fbd04729c8`.
- Sparse-MoE synchronization checkpoint (2026-08-28): canonical provenance `fbd04729c8356263e53c6d16b437ec752037a684`; accepted candidate `2a07d0ad1a6cd73287d1c1c1c3e0a297c265ca50` (parent `323cfc295895ffdb0a9cea79bb7123d514e17d4f`), reviewer `voidinfer-42`: ACCEPT, no findings. `build-moe-sync-v1` Release verification passed; exact CTest `1/1` passed in `1.66s`. GPU pre/post: RTX 5090, driver `616.56`, `32,607 MiB` total, `2,328 MiB` used, `0%` utilization pre and `7%` post. Planner/device/warmup evidence remains preserved; warmup `123` remains a no-op already present in the accepted parent and adds no diff. Next ordered action: `ab82f88603`.
- Readiness-after-warmup integration checkpoint (2026-08-28): canonical readiness provenance `ab82f8860356959d10a3ebd98f70f916b853b0fb`; accepted candidate `affe93be0481e75bf7043dd7c2925e3fe08911df` (parent `93357743d000c70488325fb2e3c325893b7b4d74`), integrated on the handoff branch without changing the verified remote source. The patch is exactly `apps/serve/main.cpp`, `src/serve/generation_service.cpp`, and `src/serve/generation_service.h`; it preserves the current CLI parser/WebUI setup, gates attachment/listening and therefore readiness/traffic on successful synchronous warmup, uses an unbounded startup deadline, and leaves normal scheduling, cache participation, and OpenAI/Anthropic translation paths unchanged. `build-readiness-v1` was configured for Visual Studio 17 2022 x64 with the vcpkg manifest and CUDA `sm_120a`; `ninfer-serve` plus eight tests built successfully, and exact Release CTest passed `8/8`, `0` failures in `0.09s`. Evidence path: `D:\AI\agent-orchestrator\data\worktrees\voidinfer\voidinfer-44\build-readiness-v1`. No GPU, model, service, or benchmark run occurred. Next ordered serving slices after runtime gates: `79c292bc07`, then `780d576791`, then `9dbc074005`.

## OSCAR fidelity execution record (2026-09-01)

- The designated build source `D:\AI\voidinfer-adaptive-dflash2` now carries a typed `OscarKVLayout`
  through paged geometry, cache views, full-cache append, dense prompt/decode dispatch, and
  StateImage conversion. The old process-global transposed-layout switch no longer controls the
  writer and reader independently; contiguous remains the safe default.
- Calibration staging is documented in `docs/OSCAR_KV_FIDELITY.md` and `tools/oscar/README.md`,
  with manifest/hash validation in `tools/oscar/validate_dump.py`.
- Phase B1 established the reproducible CPU fitter environment at
  D:\AI\tools\oscar-calibration\.venv: Python 3.12.10, torch 2.13.0+cpu, deterministic .pt
  save/load, and deterministic 128x128 torch.linalg.eigh verification. The repo setup script
  is tools\oscar\setup_calibration_env.ps1; the freeze and verification report are retained
  under D:\AI\tools\oscar-calibration.
- Phase B1 verified the latest qwen3.8-27b / nvfp4-dflash2 artifact metadata and the latest
  D:\AI\build-adaptive-dflash2 load-plan test (exit 0): 64 text layers, 16 full-attention
  layers at 3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63, and 48 GDN layers; Q/KV heads
  24/4, GQA ratio 6, head dimension 256, rotary dimension 64.
- The documented future capture boundary is qn, kn, and v immediately after Q/K
  normalization plus in-place RoPE and before causal_softmax_attention/cache append in
  src\targets\qwen3_6\impl\runtime\text_context_impl.h:860-881. GDN recurrent state is out
  of scope. No QKV capture or calibration was run.
- Phase B2 then added an opt-in native smoke capture and fail-closed validator. The focused target
  in D:\AI\build-adaptive-dflash2 captured exactly 256 deterministic token positions for Q/K/V at
  every full-attention layer (3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63), with Q shape
  [256,24,256], K/V shape [256,4,256], BF16 payloads, aligned token positions, and no GDN or
  drafter tensors. Payload and manifest-sidecar validation passed; raw dump bytes are 67,108,864
  and the manifest SHA-256 is df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89.
- Phase B2 evidence and the exact provenance-bearing dump are retained in
  results\oscar\phase-b2-qkv-smoke-capture.md and
  results\oscar\captures\phase-b2-qkv-256\manifest.json. No OSCAR fitting, 10K/30K capture,
  INT2 runtime integration, DFlash2/MTP/adaptive-K work, or CUDA-kernel optimization was run.
- Phase B3 converted that validated raw dump to the official hierarchy in
  results\oscar\dumps\phase-b2-256-official using explicit chunk `1.pt` because the upstream
  `--chunk-id all` loader skips chunk 0. The unchanged official fitter is pinned at OSCAR main
  commit `41ebcdba3db5f0ce1339c3727caea80df575d437` with script SHA-256
  `f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b`.
- The official `qqt_sst` + `r_h_pbr` smoke fit ran with head dimension 256 in 2.015 seconds and
  produced separate K/V checkpoints for all 16 full-attention layers. Reloaded FP32 `[256,256]`
  assets passed finite-value, exact-layer, deterministic-reload, and orthogonality checks; worst
  `max_abs(R @ R.T - I)` was `2.011652722e-08` for K and `1.916613712e-08` for V. The independent
  8x8 R*H*Pbr fixture passed at `2.22e-16`.
- Phase B3 evidence and rotation validation are retained in
  results\oscar\phase-b3-fitter-smoke.md and
  results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr\rotation_validation.json. These are smoke
  assets only; no production runtime integration, 10K/30K calibration, or kernel optimization was
  run.
- Phase C1 proved the mathematical rotated-BF16 convention offline across all 16 full-attention
  layers: Q and K use right-side `R_K`, V uses right-side `R_V`, and the attention output applies
  `R_V.T` before any output projection. Max errors were `4.971724872e-07` for scores,
  `7.114301309e-08` for softmax, and `3.373735723e-06` for recovered post-attention output;
  the independent JSON evidence is retained in
  results\oscar\phase-c1-rotation-invariance.json.
- Phase C1 found no matrix-orientation, RoPE, Q/K/V rotation, inverse-V, layer-mapping, or GQA
  blocker. The full layer residual/output projection was not tested because those weights and
  residual activations are not present in the Phase B2 dump. No INT2/runtime integration,
  speculative-decoding change, or kernel optimization was run; smoke-data quality remains the
  production-calibration blocker.
- Phase C2 added a deliberately slow offline INT2 reference at
  `tools\oscar\test_int2_reference.py` and ran layer 3 first, then all 16 full-attention layers
  over the existing 256-token smoke data. It compares BF16, rotated BF16, the current fixed-
  Hadamard Q2 control, and calibrated OSCAR `qqt_sst` + `r_h_pbr` INT2. The calibrated path uses
  the pinned upstream `simulate_int2_asym` semantics exactly, applies inverse `R_V.T`, keeps all
  codes in `[0,3]`, and produced finite deterministic results for every layer.
- Phase C2 is PASS for offline mathematical coherence, not production quality: maximum calibrated
  score/softmax/attention-output errors were `7.058641e+00` / `8.521844e-01` / `4.403106e+01`,
  with maximum output relative L2 `4.606063e-01`; the fixed-Hadamard control was materially
  worse at maximum attention-output error `1.280092e+02`. Evidence is retained in
  `results\oscar\phase-c2-int2-reference.md` and
  `results\oscar\phase-c2-int2-reference.json`.
- The Phase B2 artifact still lacks output-projection weights, residual activations, and LM-head
  logits, so full layer-output and generated-token agreement remain unavailable. No CUDA runtime
  integration, INT2 packing, 10K/30K calibration, speculative-decoding change, or optimization
  was performed in C2; long calibration and production fidelity remain open.
- Focused OSCAR, runtime-mechanism, and StateImage executables built and exited successfully in
  `D:\AI\build-adaptive-dflash2`. Faithful production OSCAR now has a validated 30K
  production-candidate asset set; runtime loading, group-128 packing/metadata, and end-to-end
  fidelity remain the next qualification boundary.
- Phase C3 produced a separate deterministic 10,240-token pilot capture as 40 independent
  256-token requests, plus a distinct 1,024-token held-out capture. The existing post-RoPE,
  pre-cache-append boundary was unchanged; the fail-closed multi-chunk validator passed all
  640 layer/chunk records and 1,920 Q/K/V payload hashes with no GDN records.
- The unchanged official FutureMLS-Lab OSCAR fitter completed qqt_sst + r_h_pbr on official
  chunks 1..40 for all 16 full-attention layers. K/V assets are finite FP32 [256,256] with
  worst max_abs(R @ R.T - I) of 2.193682191e-08/2.131072918e-08; the independent composition
  fixture passed.
- Held-out rotated-BF16 invariance remained healthy (max score 4.062135e-07, softmax
  8.728284e-08, recovered output 3.013387e-06, 100% attention-argmax agreement). The
  calibrated 10K INT2 reference improved aggregate output max/relative-L2 versus prior 256-token
  assets (37.105069/0.303087 versus 41.614878/0.317166). C3 is KEEP for a later 30K calibration
  pilot only; group-128 runtime packing, output-projection/LM-logit quality, CUDA integration,
  and production qualification remain open.
- Phase C4 produced immutable asset set `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` from
  30,720 useful tokens (120 independent 256-token requests) at the unchanged capture boundary.
  The fail-closed raw/official conversion checks passed all 16 full-attention layers, 5,760
  calibration tensors, exact BF16 preservation, hashes, finite values, and no GDN records.
- The unchanged official FutureMLS-Lab OSCAR fitter completed `qqt_sst` + `r_h_pbr` at D=256
  over official chunks 1..120. K/V FP32 assets passed deterministic reload, exact layer
  mapping, orthogonality, and the independent composition fixture; worst errors were
  2.082505102e-08 K and 2.296734802e-08 V. Asset manifest and all hashes are archived beside
  the checkpoints.
- Fresh 1,024-token held-out reference evaluation passed. Relative output L2 improved for 30K
  versus 10K (0.302515 vs 0.305705), mean absolute output improved (0.294053 vs 0.295811),
  worst output maximum effectively matched (39.595471 vs 39.497786), while score/softmax
  maxima and attention-argmax agreement regressed. C4 is a mixed-but-favorable asset
  qualification ready for the next runtime-integration phase; no runtime, CUDA, packing,
  DFlash/MTP, or adaptive-K work was started.

## OSCAR Phase D1 — runtime rotated-BF16 qualification (2026-09-01)

- Added an explicit opt-in `oscar-rotated-bf16` runtime mode in the designated source. The
  fail-closed loader validates the immutable C4 identity, model SHA-256 declaration, asset
  manifest SHA-256, exact K/V payload hashes and sizes, FP32 finite values, topology fields, and
  the exact 16-layer mapping before uploading the banks. Telemetry reports the asset identity,
  asset hash, `calibrated=true`, 16 full-attention layers, and the rotation mode.
- The real runtime applies `Q @ R_K`, `K @ R_K`, and `V @ R_V` only at full-attention layers
  `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`, then applies `R_V.T` after weighted V and
  before the existing gate/output projection. All 48 GDN layers remain untouched; NVFP4 weights,
  INT2, DFlash2/MTP, and adaptive-K were not changed.
- The latest `D:\AI\build-adaptive-dflash2` target
  `ninfer_qwen3_6_27b_oscar_runtime_test.exe` built successfully. The deterministic 32-token
  normal-vs-rotated smoke loaded the C4 assets and produced matching pre-rotation Q/K/V, but the
  first full-attention layer (model layer 3) diverged after BF16 materialization: QK score max
  `1.48105e-02`, softmax max `2.17747e-03`, recovered attention max `1.5625e-02`, and post-MLP
  relative L2 `7.73924e-03`. Final hidden relative L2 was `5.98146e-02`, logits relative L2
  `2.04136e-02`, top-1 agreement was false, and generated-token agreement was false.
- D1 is **BLOCKED**. The independent stage audit found no layer mapping, GQA, RoPE placement,
  or inverse-transpose anomaly; the remaining blocker is BF16 round-trip precision through the
  on-the-fly Q/K/V and inverse-V transforms. The test stopped before 512/4K as required. Evidence
  is in `results\oscar\phase-d1-runtime-rotated-bf16.md`; attention diagnostics are retained
  under `results\oscar\d1-runtime-diagnostics`. Do not proceed to INT2 runtime integration or
  performance work until this equivalence blocker is resolved.

## OSCAR Phase D1.1 — rotation precision contract (2026-09-01)

- Added opt-in diagnostic precision modes and a scalar FP32 causal GQA reference route in the
  designated runtime. The target built successfully in `D:\AI\build-adaptive-dflash2`.
- On the required deterministic 32-token input, FP32 rotation reduced layer-3 Q/K/V orientation
  errors to `3.209392e-7/3.082207e-7/2.861411e-7` relative L2 and QK/softmax errors to
  `9.274475e-8/3.505521e-7`. FP32 inverse alone matched the BF16 control.
- The combined FP32 rotation+inverse diagnostic was best at the layer boundary (`5.990686e-4`
  post-attention residual relative L2; `5.917754e-3` post-MLP), but full-model 32-token hidden
  relative L2 remained `4.570910e-2`, logits `1.693360e-2`, with false top-1 and generated-token
  agreement. All variants first diverged at full-attention layer 3.
- The CUDA/Python reference algebra was near-invariant (`1.091371e-6` relative L2); remaining
  drift is a real-runtime attention-order/cache precision confound, not a new orientation,
  RoPE, GQA, or layer-mapping finding. The diagnostic FP32 reference is prefill-only and decode
  remains BF16-cache based.
- D1.1 is **BLOCKED**. Recommended contract: FP32/fused online Q/K/V rotation, FP32 attention
  accumulation, FP32 inverse-V, and one final downstream conversion. Do not begin D2 until a
  matched prefill/decode implementation passes the full-model gate. Evidence:
  `results\oscar\phase-d1-1-rotation-precision.md`.

## OSCAR Phase D1.2 — matched attention/cache control (2026-09-01)

- Added an opt-in diagnostic mode controlled by `NINFER_OSCAR_MATCHED_FP32=1`. Matched control A
  and rotated candidate B now use the same scalar FP32 causal GQA attention kernel, FP32 FMA
  accumulation, stable softmax, persistent diagnostic FP32 K/V cache, prefill append, decode
  append/read, and final FP32-to-BF16 conversion. The production `PagedKVCache` is bypassed only
  in this diagnostic mode; GDN, DFlash2/MTP, and adaptive-K remain untouched.
- The diagnostic path uses original post-RoPE Q/K/V for A and `Q@R_K`, `K@R_K`, `V@R_V` plus
  FP32 `R_V.T` recovery for B. Cache storage is separate FP32 `[D,Hkv,absolute_token]` K/V banks,
  reset for a new base-0 prefill and reused by ordinary decode. The build target succeeded in
  `D:\AI\build-adaptive-dflash2`; executable SHA-256 is
  `EC3275ABF6FDA2CD3E724192ACF97405C5CED4696E80DE5513A19193F574B6E7`.
- The required deterministic 32-token gate failed before the optional 512-token test. At layer 3,
  matched FP32 recovered attention A/B was `1.368033032e-6` relative L2; the common BF16
  conversion led to post-attention residual `2.898554914e-5` and post-MLP/layer output
  `3.130950573e-4`. The next GDN/MLP stage reached `7.657250444e-3`; final hidden/logit relative
  L2 was `7.18921733e-2`/`2.32925045e-2`, with top-1 and generated-token agreement false.
- The first differing operation is the BF16 materialization of recovered attention before the
  existing gate/output projection, not the cache path. Decode diagnostics did execute the matched
  cache, but A/B selected different prefill tokens, so their decode comparison is not a valid
  same-token cache test. D1.2 is **BLOCKED**; do not begin D2 or INT2. Evidence is
  `results\oscar\phase-d1-2-matched-attention-cache.md`.

## OSCAR Phase D1.3 — forced-token persistent-cache qualification (2026-09-01)

- Added an opt-in D1.3 teacher-forced control only for the matched FP32 ordinary backend. The
  prefill `Begin` commit is forced to seed `997`, followed by eight forced ordinary decode commits
  `1001,1003,1005,1007,1009,1011,1013,1015`. Both A and B returned the exact same nine-token
  ledger sequence; no natural greedy-token equality was used as a correctness gate.
- A and B retained identical scalar FP32 causal GQA attention, FMA/softmax order, persistent FP32
  cache, and final BF16 boundary. A stored original post-RoPE coordinates; B stored `Q@R_K`,
  `K@R_K`, `V@R_V` and recovered with FP32 `R_V.T`. The production paged BF16 cache was not the
  matched attention source; GDN, DFlash2/MTP, and adaptive-K were unchanged.
- Layer-3 read-back validation covered every cache prefix after eight appends. A/B positions were
  identical and contiguous at `0..39`; A cache read-back matched its append history exactly.
  Maximum whole-cache transform errors were K `3.08864e-7` and V `2.86285e-7` relative L2;
  maximum Q transform error was `4.09037e-7`.
- Layer-3 score, softmax, and recovered-attention maxima were respectively `1.07778e-7`,
  `5.74594e-7`, and `1.60696e-6` relative L2. Recovered attention remained below the declared
  `1e-5` tolerance with no sequence-length growth. All expected records for the 16 full-attention
  layers were present at every step and no GDN record appeared.
- D1.3 is **PASS / implementation-validated**. Exact full-model BF16 greedy-token equality is
  retired as a rotation-correctness requirement; downstream BF16/NVFP4/GDN quality remains a
  separate model-fidelity concern. D2 may proceed under its own INT2 reference and runtime gates.
  Evidence: `results\oscar\phase-d1-3-forced-decode.md` and
  `results\oscar\phase-d1-3-validation-20260901e.json`.

## OSCAR Phase D2.1 — official INT2 quantizer / codec parity (2026-09-01)

- Pinned the serving-side FutureMLS-Lab/OSCAR clipped INT2 reference at commit
  `41ebcdba3db5f0ce1339c3727caea80df575d437`; the exact `oscar_rotation_clip_int2_kv.py`
  source SHA-256 is `c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157`.
  The fitter helper's older `simulate_int2_asym` routine was not treated as the serving codec.
- Added a separate scalar `OscarInt2G128` host codec for D=256: two independent 128-wide
  groups, K/V clips `.96/.92`, FP32 scale/zero-point metadata, symbols `0..3`, official
  quarter-interleaved packed bytes, and FP32 dequantization. The prior experimental Q2 path
  remains unchanged because its clipping, row-wide grouping, BF16 metadata, zero semantics, and
  packing differ materially.
- Generated and hash-validated 15 deterministic golden cases / 63 rows, including positive-only,
  negative-only, mixed-sign, near-zero, clipping-outlier, group-boundary, multi-row, page-like,
  and real rotated K/V samples from layers 3, 35, and 63. Fixture SHA-256 is
  `a0d2bba734fcefde2999542bba559dd338487270235e1c462dcb7392eac98bfe`.
- The separate C++ parity executable passed exact clipped values, unpacked symbols, and packed
  bytes, with maximum FP32 metadata and decoded-value absolute/relative errors all zero in the
  63-row run. No live attention, mixed BF16 window, throughput, recalibration, DFlash2/MTP, or
  adaptive-K work was performed.
- D2.1 is **PASS**. A mathematically qualified official group-128 codec is now available for
  the next D2.2 live-cache phase; it is not yet wired into live attention.
  Evidence: `results\oscar\phase-d2-1-int2-codec.md` and
  `results\oscar\d2-1-int2-fixtures\cpp-parity.json`.

## OSCAR Phase D2.2a — mixed BF16 / INT2 cache representation (2026-09-01)

- Added a separate typed `OscarMixedCacheBundle`/`OscarMixedLayerCache` representation in
  `src/core/oscar_mixed_cache_layout.h/.cpp`. It uses independent region pools and a tagged
  storage variant per page: BF16 protected prefix, official `OscarInt2G128` historical bulk,
  and BF16 recent window. The legacy experimental Q2 representation is untouched.
- Every page and slot records full-attention layer, sequence, logical and physical ranges,
  storage type for K/V, OSCAR layout version, group size, and prefix/bulk/recent role. Physical
  ranges are region-local and pages never mix formats, so logical addressing does not infer type
  from environment state.
- The deterministic `ninfer_oscar_mixed_cache_layout_test` passed for 500 tokens across all 16
  full-attention layers. It verified positions `0..499` exactly once, prefix boundary `63/64`,
  unaligned recent transition `243/244`, final position `499`, K/V agreement, identical policy
  across layers, and absence of GDN state. Metadata sizes are 56-byte page headers and 48-byte
  slot records.
- Accounting covers contexts 64, 65, 320, 321, 384, 500, 512, 1024, and 4096 with full-page
  allocation, headers, and slot tables. At 500 tokens the mixed bundle is `23,328,768` bytes,
  `1.423875` bytes/value, or `11.391` bits/value; historical bulk is `0.3572265625` bytes/value
  including its partial page and metadata. `std::vector` allocator bookkeeping is explicitly
  excluded because it is implementation-dependent and was not measurable from this layout.
- D2.2a is **PASS**. No token aging/promotion, live attention dispatch, CUDA optimization,
  DFlash2/MTP, adaptive-K, or codec mathematics changes were made. The typed layout is ready for
  a separately authorized transition/attention phase.
  Evidence: `results/oscar/phase-d2-2a-mixed-layout.md` and
  `results/oscar/phase-d2-2a-layout-validation.json` (SHA-256
  `fb158c93b4f9e5b6a9c1999392928c4318df3ec13674b572b245d7afb40a2ce7`).

## OSCAR Phase D2.2b — token aging: BF16 recent to OSCAR INT2 (2026-09-01)

- Added the slow diagnostic `OscarMixedAgingCacheBundle` path in the existing mixed-layout
  implementation. It accepts already-rotated BF16 K/V rows, retains prefix `[0,64)` as BF16,
  retains the newest 256 non-prefix tokens as BF16, and converts only the current oldest recent
  token into official calibrated `OscarInt2G128` historical storage.
- The C4 contract is fail-closed for asset identity
  `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`, model SHA-256
  `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`, runtime asset-manifest
  SHA-256 `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`, calibrated=true,
  `qqt_sst+r_h_pbr`, and the verified 16-layer topology.
- Forced append tokens `0..323` covered token 63 as protected prefix, token 64 as recent, the
  first age on append token 320, and subsequent ages of 65, 66, and 67. All 16 full-attention
  layers followed the same path; no GDN layer/state entered the bundle.
- Independent test-side calls to the parity-qualified D2.1 codec matched K/V clipped values,
  FP32 metadata, symbols, and packed bytes exactly on layers 3, 35, and 63 across all four KV
  heads using K/V clips `.96/.92`.
- Fail-closed checks rejected altered asset identity, GDN topology, protected-prefix aging, and
  a second conversion. Historical rows lost their BF16 payload, every logical token resolved to
  one tier, and conversion count equaled historical token count.
- D2.2b is **PASS**. At final context 324, tiers are prefix 64 / historical 4 / recent 256;
  physical bytes across 16 layers are BF16 prefix `4,194,304`, BF16 recent `16,777,216`, INT2
  payload `524,288`, and FP32 metadata `131,072` (page-rounded). Live INT2 attention remains
  unimplemented. Evidence: `results/oscar/phase-d2-2b-aging.md` and
  `results/oscar/phase-d2-2b-aging-validation.json` (SHA-256
  `a702f53e4ea2d0c0120cb6d80312d25a545ba478e7a75f41830450ca00292477`).
- D2.2c is **PASS** for representation-stable mixed-cache transitions. Added diagnostic-only
  `OscarMixedTransitionCache` with immutable shared page blocks, branch-local metadata, complete
  logical/page/slot K/V/INT2 metadata fingerprints, copy-on-write refresh, fail-closed commit
  lineage, and deterministic StateImage-compatible snapshot/restore. The context-324 fixture
  contained prefix/historical/recent `64/4/256` and 96 pages across all 16 full-attention layers.
- Immediate fork shared 96/96 page blocks at refcount 2 without historical re-encoding. Divergent
  context-328 branches each retained 16 shared prefix pages and did not corrupt the base. Four
  speculative appends then rollback restored the exact complete fingerprint and removed stale
  positions; valid commit produced context 328 / tiers `64/8/256` without duplicates.
- A context-332 state image of 21,760,845 bytes restored in a fresh context with exact logical,
  tier, INT2 packed payload, FP32 metadata, BF16 payload, page, and slot fingerprints. The image
  is a dedicated diagnostic OSCAR snapshot; existing fixed Qwen StateImage payloads and legacy
  Q4 shadow remain unchanged. Layout, aging, and transition targets passed. Evidence:
`results/oscar/phase-d2-2c-state-transitions.md` and validation JSON SHA-256
`df1fa525a1791cf6c6463686d1b3a741901a58825d832cca24c2e96992e16784`.

## Phase D2.3a — mixed-cache reference attention parity (2026-09-01)

- Added diagnostic-only `OscarMixedAttentionReader` in
  `src/core/oscar_mixed_attention_reference.{h,cpp}`. It performs strict typed page dispatch over
  protected BF16 prefix, official `OscarInt2G128` historical rows, and BF16 recent rows; all
  temporary row conversion, GQA score/softmax/AV arithmetic, and `R_V.T` recovery are FP32.
- The independent CPU comparator reconstructs rows from the deterministic source archive without
  using the reader or page/slot resolver. Historical rows are independently encoded/decoded with
  the D2.1 official codec. This is implementation parity for the post-aging mixed representation,
  not a claim that lossy INT2 equals pre-aging BF16.
- Static 324/332-token fixtures and a persistent four-append forced-decode fixture passed exact
  logical/tier traces. Representative layer-3 tiers were `64/0/0`, `64/1/0`, `64/4/1`,
  `64/4/256`, `64/12/245`, and `64/12/256`; forced decode reached `64/5/256` through
  `64/8/256`.
- The validation executable passed 31/31 comparisons: layer 3 boundaries and forced decode,
  layers 35/63, and the all-layer set `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`.
  Maximum relative-L2 was zero at rotated Q, scores, softmax, rotated AV, and recovered output
  against the independent CPU reference; the fail-closed gate was `1e-6`.
- No GDN state, legacy Q2 path, production-cache fallback, CUDA optimization, or serving
  integration was added. D2.3a is **PASS**; D2.3b may connect the qualified reader to real
  Qwen3.8 attention. Evidence: `results/oscar/phase-d2-3a-reference-attention.md` and
  `results/oscar/phase-d2-3a-reference-attention-validation.json` (SHA-256
  `6090dab55f3822b283c39f306e2789d758eb351a137b09f9acdf824735015610`).

## OSCAR Phase D2.3b — live Qwen runtime + forced-token reference parity (2026-09-01)

- Added diagnostic-only `oscar-int2-reference-live` integration in the real Qwen3.8 runtime.
  Actual post-RMSNorm/post-RoPE Q/K and projected V feed the validated C4 rotations, typed
  mixed cache, and slow FP32 reader; no synthetic K/V, legacy Q2, GDN cache, BF16 historical
  shadow, or production StateImage path is used.
- Forced sequence `997,1001,1003,1005,1007,1009,1011,1013,1015` passed 2/2 short taps and
  30/30 mixed taps. Independent live/reference parity was exact in deterministic FP32 for
  rotated Q, scores, softmax, rotated AV, and `R_V.T` recovered attention under the
  `2e-6` relative-L2 / `1e-4` max-absolute gate.
- Context-boundary telemetry and taps covered prefix 64, historical aging, and recent 256;
  exact official codec parity passed for 192 layer-row K/V aging checks. All 16 full-attention
  dispatch bits were set; GDN and legacy-Q2 dispatch counts were zero and BF16 history shadow
  was false.
- D2.3b is **PASS** and authorizes D3 model-fidelity qualification. Informational forced-model
  drift versus BF16 is recorded in the phase report and is not a live/reference gate.
  Evidence: `results/oscar/phase-d2-3b-live-reference-runtime.md`.

## OSCAR Phase D3.1 — minimum full-model fidelity gate (2026-09-02)

- Added a deterministic D3.1 harness path that compares unchanged-weight BF16 KV control with
  calibrated OSCAR INT2 through the existing slow live/reference runtime. The 10K and 30K banks
  are both selectable through validated runtime manifests; no recalibration or production
  weight change was performed.
- Fixed-token tests at 32, 324, and 512 tokens passed for both banks. Logit relative L2 was
  `0.023341/0.017997/0.025883` for 10K and `0.024416/0.017815/0.023566` for 30K. Top-10
  overlap stayed 9/10, 10/10, 9/10 for 10K and 8/10, 10/10, 9/10 for 30K. The only fixed
  top-1 disagreement was 10K at 32 tokens, at a BF16 top-1 margin of `0.03125`; top-5
  containment remained 5/5.
- The natural suite contains 12 prompts: 3 arithmetic, 3 coding, 2 structured JSON, 2
  retrieval, and 2 copy/exactness. Both calibrated banks achieved 12/12 objective successes,
  including all four 1,145–1,151-token retrieval/copy prompts that use historical INT2.
  Natural wording/token identity was not used as a correctness gate; the structured checker
  was corrected to accept ordinary JSON whitespace.
- D3.1 is **PASS**. The 30K bank is the provisional baseline based on deeper-context fixed-token
  results and the C4 aggregate held-out relative-L2 advantage (`0.302515` vs `0.305705`), while
  10K remains an explicit preserved A/B control. The evidence authorizes optimized SM120a
  attention-kernel work, but 4K/16K/32K/64K/128K model-fidelity qualification remains pending.
  Evidence: `results/oscar/phase-d3-1-minimum-fidelity.md`; 10K natural-run log SHA-256
  `dc5425c24d3b6bf88a0a6bfdfcd8a685c6a12beda610e305a2d53f0923159551`.

## OSCAR Phase D4.1 — correct live-path profile (2026-09-02)

- Added profiling-only counters to the genuine `oscar-int2-reference-live` route. The counters
  cover runtime Q/K/V rotation and staging, recent-to-INT2 conversion, duplicate aging guards,
  reader Q rotation, INT2 K/V decode, QK, softmax, AV, `R_V.T`, reader total, and the complete
  live full-attention branch. Profiling is gated by `NINFER_OSCAR_D4_1_PROFILE=1`; no production
  kernel or mathematical behavior was changed.
- A complete 512-token live run with the C4 30K bank took `166.190 s` in the full-attention
  branch and `174.529601 s` request/verifier wall, with prefix/history/recent `64/192/256`,
  8,192 full-layer reader calls, and 3,072 aging events. Telemetry retained the full dispatch
  bitmap and reported no GDN, legacy-Q2, BF16 historical shadow, or fallback dispatch.
- The scalar path was stopped before a full 2K/4K causal sweep because the 512-token request
  already required 174.5 s. The permitted genuine mixed-cache microbenchmark measured one final
  query at 31.9101/108.7611/213.0691 ms for 512/2K/4K; reader slope was approximately 50.5 us
  per additional historical token. At 4K, QK was 56.0506 ms and AV 50.1997 ms; INT2 K/V decode
  was 34.0487/33.9776 ms. D2.3a parity was rerun and remained PASS (31/31, zero reported
  relative-L2).
- D4.1 identifies the dominant long-history cost as scalar causal QK + AV traversal, with
  INT2 K/V decode next. The single D4.2 target is one fused SM120a mixed-cache attention kernel
  that feeds INT2 K decode directly into QK and performs V decode/AV in the same traversal,
  without a persistent decoded-K/V buffer. Evidence: `results/oscar/phase-d4-1-profile.md`.

## OSCAR Phase D4.2a — fused historical INT2 attention kernel (2026-09-02)

- Added a separate SM120a historical-only CUDA path in
  `src/ops/softmax_attention/oscar_history/launch.cu`. It decodes official `OscarInt2G128`
  K directly into six shared-GQA QK accumulators and V directly into six AV accumulators;
  no persistent decoded K/V buffer, legacy Q2 path, prefix/recent integration, or runtime
  policy change was introduced.
- The RTX 5090 test passed independent CPU-oracle parity at 128/512/2K/4K. Maximum relative-L2
  was `1.105956699e-06` for softmax and `1.089803277e-06` for AV at 4K; all outputs were finite.
- The same test benchmarked 128/512/2K/4K/8K/16K/32K. GPU latency was
  `0.056451/0.161510/0.401674/0.801082/1.444493/2.968627/6.012608 ms`, with
  `119.0–301.7x` speedup over the CPU scalar historical oracle and `0.183490 us/history token`
  at 32K. CUDA attributes reported `32/38/40` registers and `6144/2048/0` static shared bytes
  for score/softmax/AV.
- D4.2a is **PASS**. D4.2b may connect the qualified historical path to BF16 prefix/recent;
  mixed-cache integration and full-model fidelity remain unqualified. Evidence:
  `results/oscar/phase-d4-2a-fused-history.md`.

## OSCAR Phase D4.2b — mixed-tier GPU attention (2026-09-02)

- Added a separate complete mixed-tier GPU path in
  `src/ops/softmax_attention/oscar_mixed/launch.cu`. It traverses BF16 protected prefix,
  official `OscarInt2G128` historical bulk, and BF16 recent rows in logical order. BF16 rows
  are converted only at arithmetic use; historical K/V are decoded directly into FP32 QK/AV.
  The one global stable FP32 softmax is the tier merge, so no separate merge kernel or decoded
  historical BF16 shadow exists.
- Boundary, aging, partial-page, forced-tap, and cache-backed D2.3 reader checks passed. Required
  contexts 512/2K/4K had worst relative-L2 `7.33e-7/2.36e-6/5.15e-6` for rotated AV;
  all outputs were finite. All 16 full-attention dispatch bits were `1111111111111111`, with
  zero GDN, legacy-Q2, and CPU fallback dispatch.
- Complete mixed attention measured `0.208710/0.667861/1.109395/2.153588/4.373304/9.361144 ms`
  at 512/2K/4K/8K/16K/32K, or `213.2–272.3x` versus the scalar mixed reader baseline.
- D4.2b is **PASS**. D4.3 real Qwen runtime integration is authorized; Q rotation, `R_V.T`,
  and end-to-end model fidelity remain subsequent-phase work. Evidence:
  `results/oscar/phase-d4-2b-mixed-gpu-attention.md`.

## OSCAR Phase D4.3 — real Qwen GPU runtime integration (2026-09-02)

- Added the explicit opt-in `oscar-int2-gpu` runtime mode. Real post-RMSNorm/post-RoPE Q/K/V
  tensors now use the validated C4 rotations, the existing BF16-prefix / official
  `OscarInt2G128` historical / BF16-recent cache, the D4.2b mixed GPU reader, and FP32
  `R_V.T` recovery. The 48 GDN layers remain untouched.
- The runtime validates the typed page metadata before staging and preserves packed historical
  INT2 rows; no legacy Q2, BF16 historical shadow, CPU serving fallback, or Q4 shadow was added.
  The scalar reader is used only as an explicitly labelled D4.3 validation oracle.
- Live/reference taps passed at requested contexts 321/332/512 and 4K for layers 3/35/63,
  with worst rotated-AV/recovered relative-L2 `2.52443e-6/2.50990e-6`; all 16 full-attention
  dispatch bits were `1111111111111111`, with zero GDN and legacy-Q2 dispatch.
- Fixed-token BF16-vs-GPU comparisons passed the forced continuation and retained top-1/top-10
  at 321/332/512; hidden/logit relative-L2 at 512 was `0.0524625/0.0246913`. A four-case
  D3.1-style GPU smoke (arithmetic, coding, JSON, and 1,148-token retrieval) achieved 4/4
  objective success.
- Real GPU verifier wall was `2.903730/2.776749/6.889555/486.346291 s` at 321/332/512/4K.
  At 4K, cumulative profile counters were QKV rotation `0.590442 s`, recent-to-INT2 append/
  aging `398.977 s`, explicit staging `0.132383 s`, fused mixed GPU kernel `46.0793 s`,
  `R_V.T` recovery `0.339310 s`, and complete full-attention `480.292 s`.
- The 16K exploratory run was stopped at logical context 3,072 after confirming correct asset
  telemetry because the first integration's causal prefill scaling was not healthy; 32K was not
  attempted. D4.3 is **PASS** for live integration/correctness and authorizes D4.4, while real
  16K/32K throughput and long-context fidelity remain pending.
  Evidence: `results/oscar/phase-d4-3-live-gpu-runtime.md`.

## OSCAR Phase D4.4 — GPU-resident incremental cache and device aging (2026-09-02)

- Added explicit `oscar-int2-gpu-resident` mode with persistent per-layer GPU storage for BF16
  prefix/recent rows, packed official `OscarInt2G128` historical rows, FP32 metadata, ring-head
  state, and reusable score/softmax workspace. The D4.2b mixed GPU attention kernel and all
  OSCAR rotation/R_V.T mathematics remain unchanged.
- Actual runtime rotated FP32 K/V rows are appended once directly to device storage. Device aging
  encodes a departing recent row before ring-slot publication, writes packed 64-byte payloads and
  FP32 metadata directly to historical storage, and never uses a persistent decoded-K/V or BF16
  historical shadow. The existing `oscar-int2-gpu` staged mode remains available as the control.
- Device-vs-host `OscarInt2G128` parity was exact for K/V symbols, packed bytes, and FP32 metadata;
  the 4K-plus-forced-decode validation reached `60,528` logical token checks (each across both
  K/V and all four KV heads). Live/reference taps at 321/332/512/2K/4K passed with maximum
  recovered relative-L2 `5.97630e-7/6.55704e-7/8.28918e-7/7.44411e-7/1.40692e-6`.
- All 16 full-attention bits remained `1111111111111111`; GDN dispatch, legacy Q2,
  BF16 historical shadow, fallback, and full-cache staging were zero/false. The only remaining
  host-device traffic at the final 4K point was `262,592` bytes of position metadata; runtime
  K/V stayed resident on the GPU.
- Clean real-model fixed-token wall time was `1.505832 s` at 512, `13.525424 s` at 2K, and
  `46.450017 s` at 4K, versus D4.3 staged `6.889555 s` and `486.346291 s` at 512/4K. This
  is `4.58x`/`10.47x` faster, with the resident 512→4K request slope reduced to about
  `12.54 ms` per added token from `133.8 ms/token`. A genuine 16K attempt reached the mixed
  regime but was stopped before a final measurement because the per-query launch loop remained
  impractical.
- D4.4 is **PASS**. The next single target is batching/fusing the per-query full-attention
  launch loop; Q rotation, R_V.T, DFlash2/MTP, adaptive-K, and model policy remain unchanged.
  Evidence: `results/oscar/phase-d4-4-gpu-resident-cache.md`.
## OSCAR Phase D4.V — full validation optimization (2026-09-02)

- Replaced the scalar-only OSCAR live-reference validator with an optimized, still independent
  post-run oracle over immutable tap snapshots. The target is built as `RelWithDebInfo` with
  `/O2 /Ob3 /Oi /Ot /arch:AVX2 /fp:precise /UNDEBUG`; assertions and fail-closed checks remain
  enabled. A bounded pool uses at most eight workers, with deterministic sorted input/result
  ordering and no nested oversubscription.
- The same real D4.4 tap archive (30 taps, layers 3/35/63 and boundary/forced-decode queries)
  passed 30/30 under unchanged `2e-6` relative-L2 and `1e-4` max-absolute gates. External wall
  time improved from `869.854 ms` to `179.441 ms` (`4.85x`, `79.37%` reduction). The final
  in-process run reported `11,819.2 us` validation wall and `15,625 us` process CPU across
  workers. Profiled worker-summed arithmetic was QK `6,201.6 us`, Q rotation `3,525.3 us`,
  inverse-V `3,607.0 us`, AV `2,723.8 us`, softmax `263.3 us`, INT2 K/V reconstruction
  `74.4/115.1 us`; file parse/I/O inclusive decode was `58,766.8 us`.
- The permanent validator-of-validator matrix passed 21/21 for contexts `64,320,321,332,512,
  2048,4096` and layers `3,35,63`. GQA-aware row reuse decodes each historical K/V row once
  per KV head and reuses it across six Q heads. Fixed workspaces and AVX2 D256 product
  formation reduce allocations without changing scalar accumulation order or thresholds.
- A reproducible all-16-layer synthetic oracle benchmark measured `12.6115/41.9779/88.2372 ms`
  at 512/2K/4K. It is explicitly a fixture benchmark because the repository's real tap archive
  currently contains 30 taps only for context 321; no 512/2K/4K real tap claim is substituted.
  No weaker FAST qualification mode was introduced; FULL validation remains the default.
- D4.V is **PASS**. The dominant remaining arithmetic cost is QK; the single next candidate is
  batching/fusing the CPU QK traversal around existing GQA reuse while retaining the scalar
  golden path. No production OSCAR runtime, DFlash2/MTP, adaptive-K, calibration, or kernel
  semantics were changed. Evidence: `results/oscar/full-validation-optimization.md` and the
  hashes recorded there.
