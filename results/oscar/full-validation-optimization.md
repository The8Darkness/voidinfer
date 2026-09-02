# OSCAR D4.V — Full-validation optimization

Date: 2026-09-02  
Status: **PASS — full validation coverage and gates preserved**

## Scope and qualification rule

This phase optimized the independent CPU validator used for the real OSCAR live-reference
taps. It did not change the Qwen3.8 capture boundary, OSCAR rotations, INT2 codec, mixed-cache
policy, GPU runtime, DFlash2/MTP, adaptive-K, or any numerical qualification threshold.

The final validator remains fail-closed at the existing gate:

- relative L2 `<= 2e-6`;
- max absolute error `<= 1e-4`;
- finite values, exact tensor shapes, exact topology/layer membership, contiguous logical
  positions, valid tier policy, asset identity/hash, and no trailing/truncated input bytes.

No weakened FAST qualification mode was introduced.

## Baseline and final artifacts

The baseline was the pre-change scalar executable in the designated build tree, run over the
same 30 real tap files. Its external PowerShell stopwatch was `869.854 ms`; its output was
30/30 PASS but contained no component timing counters.

The final optimized executable is:

`D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe`

The real tap set is the unchanged D4.4 capture at:

`D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-correctness-final\live_reference_taps\321`

It contains 30 immutable taps (layers 3/35/63; boundary and forced-decode queries) and
`34,443,396` bytes. The optimized run completed 30/30 PASS in `179.441 ms`, a `4.85x`
speedup and `79.37%` wall-time reduction. All five compared stages remained exact in this
deterministic tap set: rotated Q, scores, softmax, rotated AV, and recovered output.

The process-level timing counter reports aggregate CPU utilization, so values above 100% are
expected when worker threads overlap. The final run reported `125,000 us` of process CPU over
the validator interval and `11,819.2 us` of in-process validation time, or `132.20%` aggregate
process CPU utilization. The external invocation time includes process startup and loader cost.

## Implementation changes

The optimized validator is a separate post-run executable over immutable binary snapshots. It
does not stop the model between taps and does not call the live reader, page resolver, CUDA
runtime, or model execution path. This preserves an independent oracle boundary while allowing
the model run to finish before validation is processed.

Implementation locations in
`tests/targets/qwen3_6_27b/test_oscar_live_reference_validator.cpp`:

- lines 108–135: fixed-size `Row` storage for four KV heads and the official encoded-row shape;
- lines 136–230: fail-closed tap parsing and one-time INT2 K/V reconstruction per row/head;
- lines 252–285: rotation-bank loading, finite checks, and one-time inverse-bank transpose;
- lines 319–385: AVX2 product formation with scalar left-to-right reduction preservation;
- lines 438–594: reusable per-worker workspace and optimized QK/softmax/AV/recovery path;
- lines 596–639: deterministic synthetic scalar-golden fixtures;
- lines 660–733: permanent scalar-golden and 512/2K/4K all-layer oracle benchmark entry points;
- lines 736–849: sorted tap enumeration, bounded worker pool, deterministic result ordering,
  component timing, process CPU accounting, and fail-closed stage gates.

Specific changes:

1. `Row` uses fixed arrays rather than one heap vector per K/V row. INT2 rows are decoded once
   per KV head while loading a tap; the six associated Q heads reuse that decoded row.
2. QK traversal is KV-head-major. A K row is reused across its six GQA query heads, rather
   than decoding or rediscovering it in six independent history walks. V traversal likewise
   reuses the already decoded KV row for the six AV accumulations.
3. Each worker owns one `FastWorkspace`, resized once per tap and reused for rotated Q, scores,
   softmax, AV, and recovered output. No inner attention loop allocates vectors.
4. A bounded pool uses `min(hardware_concurrency, 8, tap_count)` workers. There is no nested
   worker pool. Paths are sorted before dispatch and result slots are indexed by sorted path,
   so output and the final verdict are deterministic.
5. AVX2 is used for D256 product formation and row-wise multi-output Q and inverse-V rotations.
   Each output still accumulates in the scalar left-to-right order required by the scalar
   golden oracle; a vector-tree reduction that failed the strict 4K golden check was removed.
6. The immutable tap format already contains the compact rows needed for independent replay.
   The validator decodes each input tap once and never serializes giant intermediate tensors.
   A cross-tap incremental cache was not enabled: tap files are independent full snapshots and
   parallel reuse of decoded state would weaken the independence boundary. No production cache
   state is shared with the validator.

## Build contract

The designated build was reconfigured as `RelWithDebInfo`:

`D:\AI\build-adaptive-dflash2\CMakeCache.txt:25` reports `CMAKE_BUILD_TYPE=RelWithDebInfo`.

The validator target has target-local optimized flags in `tests/CMakeLists.txt`:

```text
/O2 /Ob3 /Oi /Ot /arch:AVX2 /fp:precise /UNDEBUG
```

The emitted Ninja rule retains `/Zi`, uses `/MD`, and ends with `/UNDEBUG`, so the target is
optimized while assertions and explicit fail-closed checks remain enabled. The compile rule
also defines `NINFER_OSCAR_FULL_VALIDATION_OPTIMIZED=1`.

Build command:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe -S D:\AI\voidinfer-adaptive-dflash2 -B D:\AI\build-adaptive-dflash2 -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DFFMPEG_DIR=D:\AI\voidinfer\build-windows-phase7\vcpkg_installed\x64-windows\share\ffmpeg && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_live_reference_validator -j 4"
```

## Component profile

The final real-tap run reported worker-summed component times. These are CPU work totals across
the bounded worker pool, not additive wall time:

| Component | Worker-summed time |
| --- | ---: |
| File I/O inclusive of parse and INT2 decode | `58,766.8 us` |
| INT2 K reconstruction | `74.4 us` |
| INT2 V reconstruction | `115.1 us` |
| Workspace resize/allocation | `757.5 us` |
| Q rotation | `3,525.3 us` |
| QK | `6,201.6 us` |
| Stable softmax | `263.3 us` |
| AV | `2,723.8 us` |
| `R_V.T` recovery | `3,607.0 us` |
| Synchronization inside validator | `0` beyond worker join |

The largest aggregate measured input cost is file loading, while the largest arithmetic stage is
QK. The dominant remaining compute candidates are QK traversal, Q rotation, and inverse-V
rotation. INT2 reconstruction is already a small fraction of the profile after one-time row
decode and GQA reuse. The file-I/O figure includes parsing and is not a model-side synchronization
or CUDA wait. The parser's unavoidable file-read/parse/copy work is included in
`file_io_inclusive_decode`; no giant intermediate tensor copy is performed after parsing.
Workspace allocation/resize is reported separately, and synchronization is limited to the final
worker join (`synchronization=0` beyond the measured join).

The pre-change scalar program had no component counters; its exact same-coverage wall result is
preserved above rather than backfilled with estimates. Its source behavior explains the cost:
heap-backed row vectors, head-major scalar history traversal, repeated K/V access for GQA heads,
and per-tap/per-stage temporary allocations. The final profile is measured instrumentation,
not a fabricated scalar breakdown.

## Validator-of-validator gate

The final binary passed its permanent scalar-golden check:

```text
OSCAR optimized validator scalar-golden: PASS checks=21
contexts=64,320,321,332,512,2048,4096 layers=3,35,63
tolerance_rel_l2=2e-6 tolerance_max_abs=1e-4 avx2=true
```

This covers every requested context and representative layer. The scalar implementation remains
in the same executable as the `--self-check` regression path; the optimized result is compared
stage-by-stage against it. A focused `--golden-one 4096 3` check also passed after the SIMD
reduction correction.

Exact commands:

```powershell
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe' `
  --self-check `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin' `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin'
```

## Full tap validation gate

Exact final command:

```powershell
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe' `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-correctness-final\live_reference_taps\321' `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin' `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin'
```

Final result:

```text
OSCAR optimized full validator: PASS taps=30 workers=8 avx2=true
OSCAR validator timing_us: wall=11819.2 process_cpu=15625 process_cpu_utilization_pct=132.200149
OSCAR validator profile_us: file_io_inclusive_decode=58766.8 int2_k_decode=74.4 int2_v_decode=115.1 allocation_resize=757.5 q_rotation=3525.3 qk=6201.6 softmax=263.3 av=2723.8 rv_inverse=3607 synchronization=0
OSCAR validator worst stage=rotated_q max_abs=0 relative_l2=0 PASS
OSCAR validator worst stage=scores max_abs=0 relative_l2=0 PASS
OSCAR validator worst stage=softmax max_abs=0 relative_l2=0 PASS
OSCAR validator worst stage=rotated_av max_abs=0 relative_l2=0 PASS
OSCAR validator worst stage=recovered max_abs=0 relative_l2=0 PASS
OSCAR D2.3b independent live/reference parity: PASS taps=30 worst_relative_l2=0 legacy_q2=false bf16_history_shadow=false
```

The final 30-tap path therefore preserves all D2.3b stage thresholds, independent snapshot
semantics, and fail-closed legacy-Q2/BF16-history exclusions.

## 512/2K/4K oracle scaling benchmark

The repository currently has real binary tap files only for the 30-tap context-321 archive;
the pre-existing 512/2K/4K directories are empty. The following reproducible benchmark therefore
uses the validator's deterministic fixture generator, with one final query for every one of the
16 full-attention layers. It is a CPU oracle scaling measurement, not a newly captured live-model
claim. The scalar-golden matrix above remains the qualification gate for the optimized arithmetic.

| Context | Taps | Layer coverage | In-process wall | Compute total |
| ---: | ---: | --- | ---: | ---: |
| 512 | 16 | all full-attention layers | `12.6115 ms` | `12.6087 ms` |
| 2,048 | 16 | all full-attention layers | `41.9779 ms` | `41.9741 ms` |
| 4,096 | 16 | all full-attention layers | `88.2372 ms` | `88.2317 ms` |

The benchmark command is:

```powershell
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe' `
  --benchmark `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin' `
  'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin'
```

The 512/2K/4K cases exercise the same full-attention mathematical path, official INT2 decode,
mixed-tier fixture policy, Q/K/V dimensions, GQA mapping, and C4 rotation banks. They do not
replace the need for future real-model tap capture at those contexts.

## Evidence hashes

| Evidence | SHA-256 |
| --- | --- |
| Optimized validator executable | `69A3E156C758AF08965A3E05068577048FA524CEA417022A02ED711E36C4D87A` |
| Optimized validator source | `BA41EFD8DAB9830BC67915369C681E1268A54C9D7CBC3A475260DB714B0BFE60` |
| Tests CMake manifest | `935074FC012C60B8B088D58DC4D9F58D8E25C38F65024052F735DED8C55B82F2` |
| Scalar baseline log | `1397075F7A4B3307C3D06585946ACEFB4F63781D098F926AD9F0BF540467E3E7` |
| Real-tap optimized log | `46BDE192DE22C997444BE9DDB25F1B53172672F331A01003159A4F083385C290` |
| Scalar-golden log | `1B4EAFDF5DF9A10FD06E827CD233E4CDEE8B00497B087804CF5088ADEB47C1C2` |
| Oracle benchmark log | `53D7267B4702F3B3886BC6F7036224DD0ECB23BB56AB979DD0216BA4D6E50322` |

## Verdict and next target

**PASS.** Full real-tap coverage is unchanged at 30/30, all five numerical stage gates are
unchanged and pass, the optimized CPU oracle reproduces the scalar golden reference across the
requested context/layer matrix, and the measured final validation wall time improves by `4.85x`.
The optimized path is now practical for normal development and remains the default full validator;
no weaker validation mode is needed.

The dominant remaining measured arithmetic cost is QK, with Q rotation and `R_V.T` next. The
single recommended follow-up is to batch/fuse the CPU QK traversal around the existing GQA row
reuse, while retaining the scalar-golden regression and independent tap boundary.
