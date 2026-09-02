# OSCAR D4.1 — Correct Live-Path Profile

Date: 2026-09-02  
Status: **PROFILE COMPLETE — D4.2 TARGET IDENTIFIED**

This phase measured the genuine calibrated `oscar-int2-reference-live` path using the immutable
30K C4 runtime bank. No OSCAR mathematics, model weights, CUDA kernel, DFlash2/MTP path, or
adaptive-K behavior was changed. The path remains the deliberately slow scalar/reference route;
the timings below are diagnostic measurements, not production throughput claims.

## Configuration and reproduction

- Model artifact: `Qwen3.8-27B-NVFP4-DFlash2-NInfer`
- Model SHA-256: `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`
- Asset identity: `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`
- Runtime asset manifest SHA-256: `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`
- Topology: 16 full-attention layers, 24 Q heads, 4 KV heads, GQA 6, head dimension 256
- Cache policy: BF16 prefix 64, official `OscarInt2G128` historical bulk, BF16 recent 256
- Runtime mode: `oscar-int2-reference-live`
- Build executable: `D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe`
- Runtime executable SHA-256: `b936cbfe397160f86eb4aff169861fee5a4cf42a2427ca6867706a9c3192f6a3`
- Microbenchmark executable SHA-256: `2599bd4d05a50959e269448858d3dd3c692f92907cb343521e8bd23a31564146`

Live profile command:

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS='D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256='6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-1-profile-runtime'
$env:NINFER_OSCAR_D4_1_PROFILE='1'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe' 2>&1 | Tee-Object -FilePath 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d4-1-profile.log'
```

The live harness was intentionally stopped after the complete 512-token case: that case took
`174.529601 s` for one request, so repeating the full causal sweep at 2K and 4K would spend
substantial time in the known quadratic scalar reader. The requested larger-context measurements
therefore use the permitted isolated genuine reader microbenchmark described below.

## Instrumentation

Diagnostic timers are enabled only by `NINFER_OSCAR_D4_1_PROFILE=1`:

- `src/core/oscar_mixed_attention_reference.{h,cpp}` records reader-side Q rotation, official
  INT2 K/V row decode, QK, stable softmax, AV, `R_V.T`, and total reader time.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.{h,cpp}` aggregates all 16 live layers,
  aging events, reader traces, and full live-branch time.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h` times actual runtime Q/K/V rotation
  plus synchronized staging and the complete live full-attention branch. Profiling mode skips
  only large diagnostic tensor files; final hidden/logit files remain available.
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp` adds the 512/2K/4K profile driver and
  request wall time.
- `tests/test_oscar_mixed_attention_reference.cpp` adds the isolated mixed-cache microbenchmark.

`qkv_rotation_us` includes the CUDA FP32 rotation launch and the synchronized host staging of
the rotated rows. `reader_q_rotation_us` is the scalar reader's actual Q `@ R_K` work. `aging_us`
is the cache's actual recent-to-INT2 conversion call; the duplicate reference encode and parity
guard are reported separately so their diagnostic overhead is not hidden.

## Genuine live 512-token run

The completed live case used 512 deterministic prompt tokens and one requested output token. At
the final 512-token prefill boundary it had 8,192 full-attention calls/appends (512 positions x
16 layers), 3,072 aging events (192 historical positions x 16 layers), prefix/history/recent
counts `64/192/256`, and the full dispatch bitmap `1111111111111111`. Telemetry also reported
`legacy_q2_dispatched=false`, `bf16_historical_shadow=false`, and `fallback=false`.

All values below are aggregate wall-clock microseconds unless noted.

| Stage | 512 live aggregate | Scope |
| --- | ---: | --- |
| Runtime Q/K/V rotation + synchronized staging | 64,357.6 us | 8,192 live layer calls |
| Recent → INT2 conversion | 2,167,810 us | 3,072 actual aging events |
| Duplicate reference encode guard | 685,236 us | diagnostic parity preparation |
| Aging parity comparison | 1,842.2 us | diagnostic guard |
| Reader-side Q rotation | 26,280,900 us | 8,192 scalar reader calls |
| INT2 K decode | 6,979,470 us | historical K rows only |
| QK score calculation | 28,786,100 us | all causal reader calls |
| Stable softmax | 325,349 us | all causal reader calls |
| INT2 V decode | 7,063,650 us | historical V rows only |
| AV accumulation | 25,827,700 us | all causal reader calls |
| `R_V.T` inverse | 26,419,300 us | 8,192 scalar reader calls |
| Total mixed reader | 158,154,000 us | includes scalar allocations/checks/row traversal |
| Total live full-attention branch | 166,190,000 us | includes rotation/staging, aging, reader, BF16 output upload |
| Total request/verifier wall | 174,529.601 ms | `engine.generate`; model initialization excluded |

There is no DFlash verifier in this test configuration (`SpeculativeBackend::None`), so the final
row is the harness request wall time for the live model path, not a production speculative-round
measurement.

## Isolated genuine mixed-reader microbenchmark

For 2K and 4K, the microbenchmark constructs the real typed `OscarMixedAgingLayerCache` for layer
3, ages rows with the official `OscarInt2G128` codec, and runs one final causal query through the
same `OscarMixedAttentionReader` and C4 rotation bank. It does not use synthetic arithmetic in
place of the codec or decode path. Cache construction/setup is reported but excluded from reader
stage timings.

| Context | Historical | Recent | Q rotation | INT2 K decode | QK | Softmax | INT2 V decode | AV | `R_V.T` | Reader total | Setup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 192 | 256 | 3.2113 ms | 1.6364 ms | 7.0254 ms | 0.2549 ms | 1.7419 ms | 6.3959 ms | 3.2287 ms | 31.9101 ms | 427.0314 ms |
| 2,048 | 1,728 | 256 | 3.1805 ms | 15.1566 ms | 28.3659 ms | 0.8662 ms | 15.1740 ms | 25.5872 ms | 3.1985 ms | 108.7611 ms | 5,860.3746 ms |
| 4,096 | 3,776 | 256 | 3.1906 ms | 34.0487 ms | 56.0506 ms | 1.6400 ms | 33.9776 ms | 50.1997 ms | 3.1975 ms | 213.0691 ms | 25,373.3638 ms |

The microbenchmark's exact output is in
`results/oscar/d4-1-microbench.log` (SHA-256
`9f961107ff0e6118acbb899c1a5ff360bb1c16ff559baffeeb43419419d7d4a7`). The live aggregate output
is in `results/oscar/d4-1-profile.log` (SHA-256
`436560e22cd830993d748c51c5b03a77cf7f404f7b051991368dc1eeaca55a25`).

At 4K final-query reader time, QK is the largest single measured stage (`56.0506 ms`, 26.3%),
followed by AV (`50.1997 ms`, 23.6%). INT2 K and V decode are next at `34.0487 ms` (16.0%) and
`33.9776 ms` (15.9%). Softmax is only `0.8%`; Q rotation and inverse rotation are each about
`1.5%` for this one long query. The remaining reader time is tier discovery, BF16 row reads,
temporary allocation/copying, finite checks, and other scalar traversal overhead.

## History slope

Using adjacent final-query microbenchmark points:

| Interval | Reader-total slope | INT2 K slope | QK slope | Softmax slope | INT2 V slope | AV slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 → 2,048 | 50.0332 us / added history token | 8.8022 | 13.8936 | 0.3980 | 8.7449 | 12.4943 |
| 2,048 → 4,096 | 50.9316 us / added history token | 9.2247 | 13.5179 | 0.3778 | 9.1814 | 12.0178 |

The measured final-query reader slope is therefore approximately **50.5 us per additional
historical token**, with no material slope instability. Since the full live prefill invokes a
causal query at every position, this per-query linear history cost becomes quadratic aggregate
prefill work in the scalar path.

## Validation and interpretation

After the timing changes, the existing D2.3a reference test was rerun without profiling:

```text
OscarMixed attention reference parity: PASS
31/31 comparisons; max relative-L2 Q/scores/softmax/rotated-AV/recovered = 0
```

The resulting parity JSON SHA-256 is
`6090dab55f3822b283c39f306e2789d758eb351a137b09f9acdf824735015610`.

The dominant long-history bottleneck is the **scalar causal QK + AV traversal**, with QK the
largest individual stage and INT2 K/V decode the next major cost. At 512 aggregate runtime, the
repeated scalar Q rotation and `R_V.T` matrix products are also substantial fixed per-query
costs, but they do not grow with history as steeply as QK/AV/decode.

Top three optimization candidates for later phases:

1. Fuse mixed-cache traversal so INT2 K decode feeds QK directly, then the same logical traversal
   feeds INT2 V decode/AV without full temporary decoded K/V materialization.
2. Move Q rotation and `R_V.T` recovery into batched/fused SM120a math to remove repeated scalar
   256x256 matrix products.
3. Add a dedicated coalesced group-128 INT2 K/V decode path, preserving exact D2.1 byte,
   metadata, clipping, and dequantization semantics.

## D4.2 recommendation and stop decision

The one candidate D4.2 should implement is:

> **A fused SM120a mixed-cache attention kernel that traverses the prefix/INT2/recent regions once,
> performs INT2 K decode directly into QK, and performs the corresponding V decode/AV accumulation
> without a persistent decoded-K/V buffer.**

This targets the measured QK+AV majority while also addressing the next-order K/V decode cost;
softmax and aging are not the first targets. D4.1 is complete. No optimization was started in
this phase.
