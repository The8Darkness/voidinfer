# OSCAR D4.4 — GPU-resident incremental cache + device aging

Date: 2026-09-02  
Status: **PASS — resident cache, device aging, and real-runtime scaling qualified**

## Scope and decision

D4.4 replaces the D4.3 per-invocation host cache flatten/stage and host recent-to-INT2
conversion in the explicit mode `oscar-int2-gpu-resident`. The persistent cache is allocated
once per the 16 full-attention layers, receives only newly produced rotated K/V rows, and is
read directly by the unchanged D4.2b mixed GPU attention path. Recent rows are encoded on the
GPU before their ring slots are reused.

The D4.4 path passes the requested runtime/reference checks at 321, 332, 512, 2K, and 4K,
including forced decode and actual Qwen K/V rows. Device-produced OSCAR bytes and FP32
metadata match the qualified host `OscarInt2G128` result exactly. The old
`oscar-int2-gpu` staged path remains available as the before comparison; the legacy Q2 path,
BF16 historical shadow, CPU attention fallback, GDN dispatch, and DFlash2/MTP/adaptive-K were
not used or changed.

## Qualified configuration

| Item | Value |
| --- | --- |
| Model | Qwen3.8-27B NVFP4 DFlash2 artifact |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Runtime asset hash | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| Topology | 64 layers; full attention `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; 48 GDN layers untouched |
| Attention shape | 24 Q heads / 4 KV heads / GQA 6 / head dimension 256 |
| Cache policy | BF16 prefix 64 + official OSCAR INT2 historical + BF16 recent 256 |
| Codec | `OscarInt2G128`, K clip `.96`, V clip `.92`, FP32 metadata |
| GPU mode | `NINFER_OSCAR_ROTATION_MODE=oscar-int2-gpu-resident` |

## Implementation

The persistent device layout is typed and layer-local:

- `prefix_k/v`: BF16 rows for logical positions 0..63;
- `historical_k/v`: packed 64-byte-per-row INT2 payloads;
- `historical_k/v_metadata`: four FP32 values per KV-head row;
- `recent_k/v`: 256 BF16 rows in a ring, with a resident ring head;
- reusable device score/softmax workspace for the mixed attention launch.

The main implementation points are:

- `src/ops/softmax_attention/oscar_mixed/launch.h:29-77` — resident view and append/encode
  API;
- `src/ops/softmax_attention/oscar_mixed/launch.cu:308-503` — BF16 publication, device
  `OscarInt2G128` encoding, ring-aware mixed reader launch;
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:613-855` — persistent allocation,
  incremental append, aging order, and validation-only parity readback;
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:1049-1062` — resident D4.2b
  dispatch;
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h:939-1187` — actual runtime Q/K/V
  rotation, append, attention, recovery, and telemetry;
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp:650-710` — D4.4 forced/performance
  harness.

The critical append ordering is device encode first, then BF16 publication. This preserves the
old recent row when the incoming append reuses the same physical ring slot. No persistent
decoded K/V buffer and no host historical mirror exists in the performance mode. The validation
mode may create a host oracle mirror only to compare the device output; it is disabled by
`NINFER_OSCAR_D4_4_PERF_NO_ORACLE=1`.

## Exact commands

Build:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test -j 4"
```

Focused regression:

```powershell
& 'D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\ctest.exe' --test-dir D:\AI\build-adaptive-dflash2 -R ninfer_oscar_mixed_gpu_attention_test --output-on-failure
```

Result: `1/1` passed, `27.92 s`. The independent D2.3a reference executable was also rerun
with its required rotation/report arguments and passed.

Correctness command:

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS='D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256='6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-correctness-final'
$env:NINFER_OSCAR_D4_4_LIVE='1'
$env:NINFER_OSCAR_D4_4_VALIDATE_REFERENCE='1'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe' 2>&1 | Tee-Object -FilePath 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-correctness-final.log'
```

Performance command (no scalar oracle/readback):

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS='D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256='6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-performance-final'
$env:NINFER_OSCAR_D4_4_LIVE='1'
$env:NINFER_OSCAR_D4_4_PERF_NO_ORACLE='1'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe' 2>&1 | Tee-Object -FilePath 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-4-performance-final.log'
```

The rebuilt runtime-test executable SHA-256 is
`CE270D09BD4237A496B4EAA6D2F46F2D2B34E5B46C47383B388FCBD41E4E63CB`.

## Device codec and runtime correctness

The validation run compared every newly historical K/V row in the resident device arrays with
the host-qualified codec. At the final 4K-plus-forced-decode point it reported:

```text
prefix_token_count=64 historical_token_count=3783 recent_token_count=256
gpu_resident_aging_events=60528 gpu_resident_codec_parity_checks=60528
gpu_cache_staging_us=0 gpu_cache_staging_bytes=0
legacy_q2_dispatched=false bf16_historical_shadow=false fallback=false
```

Each parity check covers both K and V, all four KV heads, both G128 groups, packed symbols,
64-byte payloads, and all FP32 metadata values. K/V code and metadata matched byte-for-byte and
float-for-float. The checks include first aging, repeated aging, ring-slot reuse, and ranges
that cross the existing typed-cache page boundaries. The D2.2a/2b page-boundary regressions
also remained green.

Live GPU attention was compared with the qualified scalar/reference reader at layers 3, 35,
and 63. The maximum relative-L2 values over those taps were:

| Case | Tap coverage | Rotated AV rel L2 | Recovered attention rel L2 | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 321 | 30 taps | `5.14381e-7` | `5.97630e-7` | PASS |
| 332 | 30 taps | `5.58012e-7` | `6.55704e-7` | PASS |
| 512 | 30 taps | `7.59950e-7` | `8.28918e-7` | PASS |
| 2,048 | 30 taps | `6.75212e-7` | `7.44411e-7` | PASS |
| 4,096 | 72 taps | `1.36262e-6` | `1.40692e-6` | PASS |

The declared stage gate was relative-L2 `<=1e-4` and max absolute `<=1e-3`; no unexplained
attention discrepancy occurred. The 4K forced continuation reached query 4103 with recovered
relative-L2 `1.7918e-6` at the worst selected tap and still passed. All 16 full-attention
dispatch bits were `1111111111111111`; GDN dispatch count was zero.

## Resident telemetry and traffic

The final 4K performance run reported:

```text
gpu_cache_staging_us=0
gpu_cache_staging_bytes=0
gpu_incremental_host_device_bytes=262592
gpu_resident_cache_bytes=63078400
gpu_resident_workspace_bytes=789504
gpu_qkv_rotation_us=621422
gpu_mixed_kernel_us=4.50384e+07
gpu_recovery_us=345559
gpu_full_attention_us=4.61149e+07
gpu_attention_calls=65648
gpu_resident_publish_us=2403
gpu_resident_aging_us=5277.9
gpu_resident_append_calls=368
gpu_resident_aging_events=60528
```

The `gpu_incremental_host_device_bytes` value is the cumulative device-to-host position
metadata validation traffic (`4103 positions * 4 bytes * 16 layers`). Runtime Q/K/V remain on
the device; no full K/V cache transfer occurs. The resident-cache and workspace values are
persistent allocations for all 16 layers and include both K/V tiers plus score/softmax
workspace. The publish/aging microsecond counters in this no-oracle run are enqueue-side
diagnostics; the synchronized request wall and mixed-kernel counters are the authoritative
end-to-end timing.

At 4K-plus-forced-decode, the logical resident policy was prefix/history/recent `64/3783/256`.
The telemetry's page-rounded historical records were 31,457,280 payload bytes and 7,864,320
metadata bytes. There was no BF16 historical shadow.

## Before/after real-model performance

The before values are the D4.3 `oscar-int2-gpu` staged route. The after values are the clean
D4.4 `oscar-int2-gpu-resident` run with the oracle disabled. The harness uses the same
deterministic forced continuation and reports request/verifier wall time.

| Context | BF16 control (ms) | D4.3 staged (ms) | D4.4 resident (ms) | Resident prompt tok/s | Resident speedup vs staged |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 204.413 | 6,889.555 | 1,505.832 | 340.1 | `4.58x` |
| 2,048 | 449.463 | not recorded in D4.3 | 13,525.424 |  — |
| 4,096 | 793.384 | 486,346.291 | 46,450.017 | 88.2 | `10.47x` |

The D4.3 4K profile was `398.977 s` host append/aging, `132.383 ms` explicit staging with
`1.114 GB` cumulative staged traffic, `46.079 s` fused mixed kernel, and `480.292 s` complete
full-attention branch. D4.4 reduced the corresponding resident request to `46.450 s`, removed
the full-cache staging counter entirely, and made device aging the only append conversion path.
The resident mixed-kernel counter at the final 4K point was `45.0384 s`; QKV rotation was
`621.422 ms`, FP32 recovery was `345.559 ms`, and the complete full-attention counter was
`46.1149 s`. Q rotation and `R_V.T` remain intentionally outside this phase's optimization
target.

The 512→4K resident request slope is approximately
`(46,450.017 - 1,505.832) ms / 3,584 = 12.54 ms per additional input token`, versus
approximately `133.8 ms/token` for the D4.3 staged request. This is still a causal prefill
and per-query-launch slope, not a steady-state decode microbenchmark.

## 16K attempt

A genuine performance-only 16K run was started with the resident mode and correct model/asset.
It reached at least the mixed regime with 3,776 historical tokens, 256 recent tokens, and
zero staging/fallback telemetry, but it did not reach the final 16K measurement in a practical
time under the current per-query launch loop. It was terminated before claiming a 16K latency.
The 16K log is retained at
`results/oscar/d4-4-performance-16k.log`; no synthetic or isolated-kernel number is substituted
for a real-model 16K result.

## Evidence hashes

| Evidence | SHA-256 |
| --- | --- |
| Resident runtime-test executable | `CE270D09BD4237A496B4EAA6D2F46F2D2B34E5B46C47383B388FCBD41E4E63CB` |
| Correctness log | `29B3F15C2D6A029A8172850231CC8C924F5924AFBB4EFA4613A2A4733FB88EF8` |
| Clean performance log | `04003D44985C6C5A1D093CA87C6EF154E20FADF3ED4A76760728984A0885A3DE` |
| `launch.h` | `EFD7667AB065C184BC40A956EE2FF366ED0E4CA0F3BE40060DD727216BA05318` |
| `launch.cu` | `657FE013F93D660D3F40E9FAA19CC32B2BF2352F6917160CAF382A59A921168E` |
| `oscar_rotated_bf16.cpp` | `9B5E621F286013E38E88855ECC057DF703F89ADADAE2BF25BE8B90DE89E9195F` |
| `text_context_impl.h` | `B4F64985CA902E050FAE5091C2C486F28288E9B108FF679AC90AAF461DB4934A` |

## Verdict and next step

**PASS.** D4.4 establishes a correctness-qualified GPU-resident incremental OSCAR cache:
actual runtime rows are appended once, recent rows age on device into official INT2, the
resident mixed reader remains reference-equivalent, full-cache host staging is eliminated,
and real-model 512/2K/4K wall time improves materially.

One recommended D4.5 optimization: remove the per-query full-attention launch loop by batching
or fusing the causal query-column launches while preserving the current resident cache view and
softmax/rotation contract. This is the remaining dominant real-runtime scaling cost exposed by
the resident profile; Q rotation, `R_V.T`, DFlash2/MTP, and adaptive-K remain unchanged.
