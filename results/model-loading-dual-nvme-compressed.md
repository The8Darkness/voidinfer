# VoidInfer dual-NVMe compressed model loading

Date: 2026-09-02  
Project: `D:\AI\voidinfer-adaptive-dflash2`  
Artifact: `qwen3_8_27b_nvfp4.ninfer`  
Profile: Release/RelWithDebInfo CUDA build, `max_context=512`, one engine concurrency, no vision, no prefix cache, no speculative backend.

## Result

The dual-source loader is correct and materially faster for this artifact. The best observed load was **2.43667 s** with dynamic two-drive scheduling, compared with **3.56250 s** for the best same-matrix single-drive run: **1.46x speedup**. The static alternating scheduler was only about 1.6% slower than dynamic on the three-run mean, so static alternating is the recommended default for this first implementation; dynamic remains available as an opt-in comparison.

The result is not 2x because the loader is already CPU/decompression and materialization constrained. The two-drive path does, however, show genuine simultaneous reads: observed maximum active reads was 2, one per drive, with approximately 10.4-10.6 GB/s aggregate planned-range read throughput.

## Identical compressed copies

The original D: file was preserved. The second complete copy is:

`C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer`

Both files were checked after copying and NTFS compression:

| Property | C: copy | D: copy |
|---|---:|---:|
| File length | 23,719,496,192 B | 23,719,496,192 B |
| SHA-256 | `6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E` | `6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E` |
| NTFS attribute | `Archive, Compressed` | `Archive, Compressed` |

The C: hash was computed after `compact /c /q` completed. No D: file was deleted or modified.

## Implementation

The opt-in API is represented by `ninfer::ArtifactReadMode`:

| Mode | Behavior |
|---|---|
| `Single` | Existing loader; default and unchanged |
| `DualStaticAlternating` | Planned aligned read spans assigned C/D/C/D/... |
| `DualDynamic` | Two drive workers claim the next unclaimed span from an atomic work index |

Relevant locations:

- `D:\AI\voidinfer-adaptive-dflash2\include\ninfer\types.h:159-177` — read mode and secondary artifact options.
- `D:\AI\voidinfer-adaptive-dflash2\src\targets\registry.cpp:23-105` — option validation and fail-closed equivalent-artifact directory validation.
- `D:\AI\voidinfer-adaptive-dflash2\src\targets\registry.cpp:292-303` — secondary reader construction and validation.
- `D:\AI\voidinfer-adaptive-dflash2\src\artifact\materializer.h:20-80` — dual materialization API and telemetry.
- `D:\AI\voidinfer-adaptive-dflash2\src\artifact\materializer.cpp:410-697` — two-reader materializer.
- `D:\AI\voidinfer-adaptive-dflash2\tests\targets\qwen3_6_27b\test_model_load_probe.cpp:34-166` — benchmark and deterministic smoke harness.

The dual materializer keeps the original materialization plan and destination offsets. It creates one independent `Reader` and worker thread per drive, two 64 MiB pinned slots per drive, and sends completed ranges to the existing transfer stream with `cudaMemcpyAsync`. There is no concatenation, tensor-offset change, persistent full-cache copy, or artifact-format change. Each worker has one active direct read; the measured aggregate depth is reported by `dual_max_parallel_reads`.

### Default/opt-out policy (follow-up)

The public option contract now makes dynamic dual loading the default whenever `secondary_artifact_path` is supplied and no explicit scheduler is selected. In addition, on Windows C:/D: model paths, the engine discovers an equivalent artifact at the same relative path on the other drive when it exists. The existing `Single` enum value remains the internal baseline when no secondary artifact is available. To explicitly opt out, set `EngineOptions::disable_dual_artifact_loading=true`; normalization clears the secondary path and uses the unchanged single-source loader. Explicit `DualStaticAlternating` and `DualDynamic` selections remain honored.

Before materialization, the engine compares file size, embedded artifact identity, directory count/order/index/name/offset/bytes, tensor shape/format/layout, and resource encoding. The benchmark preparation additionally gates both paths on the exact required SHA-256 above. A mismatch aborts dual loading; there is no silent single-source fallback.

The post-patch executable was built successfully with:

```text
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_model_load_probe -j 4"
```

Executable SHA-256: `3BD8B9D702FCE68C0D06899379BFCDF25C5D6BF5ABBDEB60E96DC715EADC14C9`  
Build configuration: `RelWithDebInfo`, CUDA SM120 configuration already present in `D:\AI\build-adaptive-dflash2`.

## Independent sequential-read controls

Command form:

```text
D:\AI\build-adaptive-dflash2\tools\model_read_probe.exe <artifact.ninfer> <buffer_MiB>
```

The probe uses `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`, reads 4 KiB-aligned ranges, and skips the final 2,560 bytes required to satisfy no-buffering alignment. These tests are on the current compressed copies.

| Drive | Buffer | Run 1 | Run 2 | Run 3 / Run 4 | Mean or median |
|---|---:|---:|---:|---:|---:|
| C: | 64 MiB | 6,791.886 MB/s | 7,674.903 MB/s | 7,645.810 MB/s | median 7,645.810 |
| C: | 256 MiB | 7,336.799 MB/s | 7,513.060 MB/s | — | mean 7,424.930 |
| D: | 64 MiB | 6,944.151 MB/s | 7,057.607 MB/s | 7,058.733 MB/s | median 7,057.607 |
| D: | 256 MiB | 6,977.228 MB/s | 6,946.578 MB/s | — | mean 6,961.903 |

The direct probe confirms that neither current drive is the former ~500 MB/s bottleneck. The engine's planned ranges are smaller than a full-file probe because retained resources are mapped/copied separately; its direct-read diagnostics are reported below.

## Engine benchmark

Each row is the mean of three timed load runs at `max_context=512`. `load_seconds` is the engine-reported load interval. C:/D: GB/s are calculated from each reader's direct-read bytes and elapsed time. For dual aggregate GB/s, the measured wall interval from the first direct read to the last direct read is used, not the sum of the two per-drive elapsed counters.

| Mode | Load time (s) | C GB/s | D GB/s | Aggregate GB/s | H2D active (s) | CPU utilization |
|---|---:|---:|---:|---:|---:|---:|
| C: compressed only | 3.603 | 6.97 | — | 6.97 | 0.391 | 99.2% |
| D: compressed only | 3.766 | — | 6.65 | 6.65 | 0.392 | 99.5% |
| C:+D: static alternating | 2.514 | 5.89 | 5.63 | 10.42 | 0.440 | 173.7% |
| C:+D: dynamic | 2.475 | 5.90 | 5.62 | 10.63 | 0.441 | 175.6% |

Representative static post-instrumentation run:

- `load_seconds=2.49183`
- `dual_direct_read_wall_seconds=1.95497`
- primary C: `10,200,547,328 B`, 152 requests, `1.73065 s`
- secondary D: `10,175,107,072 B`, 152 requests, `1.80693 s`
- aggregate planned direct bytes: `20,375,654,400 B`
- aggregate effective read rate: `10.422 GB/s`
- `dual_max_parallel_reads=2`
- `h2d_stream_seconds=1.80538`, `h2d_active_seconds=0.434228`
- `peak_staging_bytes=268,435,456` (128 MiB per drive)

Representative dynamic post-instrumentation run:

- `load_seconds=2.43667`
- `dual_direct_read_wall_seconds=1.91722`
- primary C: `10,443,542,528 B`, 156 requests, `1.77065 s`
- secondary D: `9,932,111,872 B`, 148 requests, `1.76822 s`
- aggregate planned direct bytes: `20,375,654,400 B`
- aggregate effective read rate: `10.627 GB/s`
- `dual_max_parallel_reads=2`
- `h2d_stream_seconds=1.76771`, `h2d_active_seconds=0.434119`
- `peak_staging_bytes=268,435,456`

### Repetition values

| Mode | Load run 1 (s) | Load run 2 (s) | Load run 3 (s) | Best (s) |
|---|---:|---:|---:|---:|
| C: only | 3.65186 | 3.59447 | 3.56250 | 3.56250 |
| D: only | 3.73305 | 3.80327 | 3.76271 | 3.73305 |
| Static C:+D: | 2.53295 | 2.51468 | 2.49406 | 2.49406 |
| Dynamic C:+D: | 2.46218 | 2.46129 | 2.50230 | 2.46129 |

The additional post-instrumentation runs measured 2.49183 s static and 2.43667 s dynamic while recording direct-read wall time. The dynamic-versus-static difference is small relative to run-to-run variation: 2.475 s versus 2.514 s on the three-run means.

Other engine counters were stable:

- host-to-device bytes: `20,375,587,264 B` for every mode;
- planned artifact bytes read: `20,388,491,577 B`;
- tensor/resource counts: `659 / 6`;
- per-drive direct-read maximum outstanding: 1;
- dual observed maximum active readers: 2;
- H2D stream span: approximately 1.77-1.86 s for dual runs;
- H2D active copy time: approximately 0.434-0.443 s for dual runs;
- process CPU utilization: approximately 172-176% in dual mode, consistent with two active read workers plus loader work.

## Scheduler choice

Static alternating assigned 152 planned read requests to each drive in the representative run. Dynamic work claiming assigned 156 requests to C: and 148 to D:, adapting to completion timing. Dynamic improved the three-run mean by about 39 ms (1.6%) and the best post-instrumentation run by about 55 ms compared with the representative static post-instrumentation run. This is not a meaningful enough gain to make the more complex scheduler the default yet.

Recommended behavior:

- keep normal single-file loading as the default;
- keep dual-source loading opt-in;
- use static alternating as the first documented dual-source mode;
- retain dynamic work claiming for future testing if drive/CPU asymmetry changes.

## Correctness and smoke validation

All four loader modes completed model construction with the same artifact directory plan and the same `659` tensors / `6` resources. The dual mode copied into the original planned device offsets; no artifact concatenation or offset rewrite was used.

The optional deterministic smoke command was:

```text
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_model_load_probe.exe C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer 512 D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer dynamic --smoke
```

The same fixed eight-token prompt produced one greedy token in each of:

| Mode | Smoke token |
|---|---:|
| C: only | 248046 |
| D: only | 248046 |
| C:+D: dynamic | 248046 |

The post-change default/opt-out probe was also run at `max_context=512`:

- `C: 512 D: default` reported `artifact_read_mode=dual-dynamic`, `dual_source=true`, and `dual_max_parallel_reads=2`.
- `C: 512 D: single` reported `artifact_read_mode=single`, `dual_source=false`, and an empty secondary path.
- `D: 512` with no secondary argument auto-discovered the C: counterpart and reported `artifact_read_mode=dual-dynamic`, `dual_source=true`, and `dual_max_parallel_reads=2`.
- `D: 512 --smoke` still produced deterministic token `248046`.

Thus the default switch is conditional on a configured second complete artifact; ordinary one-path callers remain single-source automatically.

The production CLI and server expose the same opt-out as `--no-dual-load`. Without that flag,
loading the D: artifact in this environment auto-discovers the hash-matched C: copy and selects
`dual-dynamic`. The C++ API equivalent is
`EngineOptions::disable_dual_artifact_loading=true`.

The final auto-discovery smoke run reported `dual_source=true`,
`dual_max_parallel_reads=2`, and generated token `248046`; the explicit `single` probe reported
`dual_source=false`. CLI and server option-parser tests both passed.

This is a construction/materialization smoke, not a quality benchmark. It demonstrates that the dual destination model is usable by the same runtime and that the deterministic one-token result matched both single-source controls.

## Source identity and scope

Changed source hashes for this experiment:

| File | SHA-256 |
|---|---|
| `include/ninfer/types.h` | `57C54A928BD980697A0071620A02E28F006FFB06B0DAE223FC713F882DA8DB28` |
| `src/artifact/materializer.cpp` | `2628C2B9C39760905C55DE6AEC627647E3CC8BECDF7B6D3F643DCB86E5CFCA40` |
| `src/targets/registry.cpp` | `D6A7A7CF41B45D640749AE1DBEC272EC7199848843FD1FE5AD78B1D86C70AE4E` |
| `src/runtime/engine/engine.cpp` | `DA15EACFB1127AD2D5907D364BE91E2798E6AE2F836321F4FAE63E24211F49B2` |
| `tests/targets/qwen3_6_27b/test_model_load_probe.cpp` | `D034E1AAADD266D7252ADD23C61E5E5B7A1E5654040D25871268431CACF45065` |
| `build-adaptive-dflash2/tests/ninfer_qwen3_6_27b_model_load_probe.exe` | `17A343FA18A784E8A6A3D1DE7597978F20392C2C53F0C796BB211C9097E822DE` |

Default-policy follow-up source/binary hashes:

| File | SHA-256 |
|---|---|
| `apps/cli/options.cpp` | `C0EF58867CE0D5C928697B779F3A9409477EF71760E1A60430CE47DCBAE30514` |
| `apps/cli/main.cpp` | `9B9911F24DC563F7E3F5A04E11622F7F4A4FAECE2391B69006474CC75C4B0437` |
| `src/serve/serve_options.cpp` | `3AD2BF3D297070A37C80B4B6F2665F0A5B32899AB37F6202B2D00929C0B87329` |
| `src/serve/generation_service.cpp` | `EF0001684A8FBB9356477EA9B1A2DB8E3AA1DC2CBA71CFE816D79A9E3139FC2A` |
| `build-adaptive-dflash2/apps/ninfer.exe` | `2EA09DC7AF7F797263856B284C920BCFE963BDAF35AAFDC9873151FA9F91E7C5` |
| `build-adaptive-dflash2/apps/ninfer-serve.exe` | `120808A80411CD4F7FE1462547321881A70A981BBB0F839C69401506F08FE49F` |

No `.ninfer` format, tensor offset, weight, CUDA attention, DFlash2, MTP, or adaptive-K changes were made.

## Final decision

- Best single-drive load: **3.56250 s**, C: compressed.
- Best dual-drive load: **2.43667 s**, dynamic C:+D:.
- Speedup: **1.46x** versus the best single-drive run; approximately **1.45x** using the three-run means.
- Actual concurrent aggregate direct-read bandwidth: **10.63 GB/s** in the best measured dynamic run.
- Current next bottleneck: CPU-side compressed-range processing/materialization and engine construction, not H2D and not a single physical drive.
- Decision: **KEEP DUAL** as an optional loader mode, with static alternating as the initial default and normal single-file loading preserved.

The next optimization should be a bounded measurement of compressed-range decompression/materialization CPU time and whether increasing per-drive queue depth beyond one active read can improve overlap without saturating CPU or increasing pinned-memory pressure. Do not use RAID0 as the next experiment.
