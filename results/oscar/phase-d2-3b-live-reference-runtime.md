# OSCAR Phase D2.3b — live Qwen runtime + forced-token reference parity

Date: 2026-09-01  
Status: **PASS — correctness-qualified live calibrated OSCAR INT2 runtime**

## Scope and qualification boundary

This phase connected the qualified D2.2 mixed cache and D2.3a slow reader to the real
Qwen3.8-27B NVFP4 runtime. The opt-in mode is
`oscar-int2-reference-live`. It uses the immutable C4 asset
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` and is diagnostic/reference code, not an
optimized serving profile.

The primary gate is live runtime attention versus an independent reference reconstructed
from the live tap. BF16-versus-OSCAR model fidelity is informational only; natural greedy
token equality is not a gate. No calibration, DFlash2/MTP/adaptive-K, production StateImage,
legacy Q2, CUDA optimization, or persistent BF16 historical shadow was added.

## Runtime integration points

| Location | Responsibility |
| --- | --- |
| `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h:117` | `OscarLiveMixedReferenceCache`, per-layer live cache ownership and diagnostics |
| `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:488` | C4 contract validation, live cache construction, append/aging parity, tap emission, telemetry |
| `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cu:315` | existing FP32 Q/K/V rotation launch used by the live path |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h:269` | explicit live-mode cache setup |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h:923` | real attention dispatch; live branch takes actual runtime Q/K/V |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h:952` | actual post-RMSNorm/post-RoPE Q/K and projected V rotation |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h:984` | live K/V append into typed BF16 recent / INT2 historical cache |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h:1000` | mixed reference attention and FP32 `R_V.T` recovery |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:55` | D2.3b forced-token environment parsing |
| `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp:113` | deterministic 32/324-token live harness |
| `tests/targets/qwen3_6_27b/test_oscar_live_reference_validator.cpp:1` | independent tap decoder/reference comparator |

The 16 dispatch-eligible layers are exactly:

`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`.

The other 48 GDN layers remain on their existing path and are never represented in the OSCAR
cache.

## Real K/V append and aging path

For each full-attention invocation, the live branch consumes actual runtime tensors:

1. post-RMSNorm/post-RoPE Q and K, plus projected V, are passed to the existing C4 FP32
   rotation operation;
2. rotated K/V rows are converted to BF16 for the recent/prefix physical representation;
3. logical positions `[0,64)` remain BF16 protected prefix;
4. the newest 256 non-prefix positions remain BF16 recent;
5. when the oldest recent position leaves the window, its rotated BF16 K/V rows are encoded
   exactly once with `OscarInt2G128`, group size 128, K clip `.96`, V clip `.92`, and FP32
   metadata;
6. the slow reader decodes prefix BF16, historical INT2, and recent BF16 in logical order,
   computes FP32 scores/softmax/AV, and applies the layer-specific `R_V.T` in FP32.

The live branch does not use synthetic rows, a global BF16 history, or the experimental Q2
codec. Cache positions are the actual runtime absolute positions. K/V are stored in the
rotated coordinate system; Q is rotated at query time with the corresponding layer's
`R_K`.

## Reference tap and independent comparator

Selected live calls emit a binary tap containing the layer/query/context, C4 identity and
asset hash, original Q, every logical cache position and tier, the actual BF16 or packed
INT2 K/V row plus metadata, and the live trace vectors: rotated Q, scores, softmax, rotated
AV, and recovered output.

`ninfer_qwen3_6_27b_oscar_live_reference_validator.exe` independently reads those taps. It
does not call `OscarMixedAttentionReader`, the live cache, or the page resolver. It independently
decodes BF16/official INT2 rows, applies the layer C4 matrices, performs FP32 GQA and stable
softmax, accumulates AV, and applies `R_V.T`. The fail-closed stage gate is relative L2 `2e-6`
and max absolute error `1e-4`.

## Build and execution

Build used the latest requested build directory:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_engine -j 4"
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test ninfer_qwen3_6_27b_oscar_live_reference_validator -j 4"
```

Authoritative runtime smoke command:

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS = 'D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256 = '6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_D2_3B_LIVE = '1'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-3b-live-run-final'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe'
```

Independent validation commands:

```powershell
$k = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin'
$v = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-3b-live-run-final\live_reference_taps\32' $k $v
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_live_reference_validator.exe' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-3b-live-run-final\live_reference_taps\324' $k $v
```

The runtime was run with one deterministic seed token followed by the same forced sequence
for eight decode positions:

`997,1001,1003,1005,1007,1009,1011,1013,1015`.

Both 32-token and 324-token cases therefore use identical forced token IDs in the live and
BF16-control runs.

## Live/reference parity results

| Case | Prefill | Forced decode | Taps | Prefix / historical / recent | Worst stage rel L2 | Verdict |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| short smoke | 32 | 8 | 2 | 32 / 0 / 0 at final query | 0 | PASS |
| mixed-cache | 324 | 8 | 30 | 64 / 8 / 256 at final selected tap; run telemetry reached 64 / 12 / 256 | 0 | PASS |

The 324-token tap set covers layers 3, 35, and 63 at queries around the prefix boundary,
historical/recent boundary, and all eight forced decode appends. The tap `context` field is
reported verbatim below; the causal source trace for each row is exactly `0..query`.
Representative layer-3 traces were:

| Query / tap context | Prefix | Historical | Recent | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 63 / 256 | 64 | 0 | 0 | final protected-prefix position |
| 64 / 256 | 64 | 0 | 1 | first recent position |
| 68 / 256 | 64 | 0 | 5 | recent-only early context |
| 319 / 324 | 64 | 4 | 252 | historical/recent coexistence |
| 320 / 324 | 64 | 4 | 253 | aging-boundary progression |
| 323 / 324 | 64 | 4 | 256 | complete initial mixed cache |
| 324 / 325 | 64 | 5 | 256 | first live aging transition |
| 327 / 328 | 64 | 8 | 256 | fourth live aging transition |

The validator output was PASS for all 2 short taps and all 30 mixed-cache taps. For every
tap, rotated Q, QK scores, stable softmax, rotated-coordinate AV, and recovered attention
matched the independent reference with worst relative L2 `0` in the deterministic scalar
FP32 implementation. No first differing live/reference operation exists.

## Exact aging codec parity

At each live transition, the runtime independently re-encoded the source row and compared it
with the actual historical row after insertion. The comparison covered all four KV heads,
both K and V, packed 2-bit payload, group ordering, and all four FP32 metadata values.

| Check | Result |
| --- | ---: |
| Full-attention layers covered | 16 |
| Live aged rows per layer, context 324 → 332 | 12 |
| K/V exact packed-and-metadata parity checks | 192 layer-row checks |
| K clip / V clip | `.96` / `.92` |
| Group size | 128 (D=256, two groups) |
| Legacy Q2 dispatches | 0 |
| BF16 historical shadow | false |
| GDN dispatches | 0 |

The final layer-63 telemetry emitted the complete dispatch bitmap
`1111111111111111`, `aging_codec_parity_checks=192`,
`gdn_dispatches=0`, `legacy_q2_dispatches=0`, and
`bf16_history_shadow=false`.

## Runtime telemetry and asset validation

The loader rejected mismatched model identity, missing/incorrect K/V banks, incorrect
topology, and incorrect asset hashes. The successful run reported:

```text
asset_identity=qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1
asset_hash=4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560
calibrated=true full_attention_layers=16 rotation_mode=oscar-int2-reference-live
group_size=128 k_clip=0.96 v_clip=0.92 prefix=64 recent=256
legacy_q2_dispatches=0 bf16_history_shadow=false fallback=false
```

At the end of the complete mixed run, telemetry reached prefix `64`, historical `12`, recent
`256`; the final selected tap is query 327 and therefore contains historical `8` plus recent
`256` in its causal trace. INT2 payload and metadata increased deterministically with each
append. No production StateImage serialization was changed or used.

## Informational full-model comparison

The forced input sequence was identical, but the live OSCAR path intentionally introduces
BF16/INT2 cache representation changes. These model-level numbers are not the
implementation-correctness gate:

| Case | Hidden relative L2 | Logit relative L2 | Top-1 | Top-10 overlap | Forced output agreement |
| --- | ---: | ---: | --- | ---: | --- |
| 32 tokens | `0.0710382048` | `0.0244162833` | true | 8 | exact forced sequence |
| 324 + 8 decode | `0.0451812669` | `0.0178153973` | true | 10 | exact forced sequence |

These values are recorded for D3 model-fidelity work and do not indicate a live/reference
attention discrepancy.

## Evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| C4 runtime manifest | `c1979e86744682733a668739642ad3a945b5a6220e8b3ed983b74d33b82c3afc` |
| C4 K rotation bank | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` |
| C4 V rotation bank | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |
| Runtime test executable | `38fcee6a81a1299213822b60d107621a3dde13e143bcb2a72f5618b605278bce` |
| Independent validator executable | `5f5378c3f4a3d642265d4a0de704b901e31f0cf8e685c7d41e0f12d6f40d3001` |
| Live tap set (32 files, 34,719,936 bytes) | `9f5de2465a55ae983d0c4338657751fe10dfcccc1b35380fb75faffea374b080` |

Source hashes for the implementation are recorded here for reproducibility:

| Source | SHA-256 |
| --- | --- |
| `oscar_rotated_bf16.h` | `078af15ecd63632d40bde4c6db5d997893f27d527f2e4be587681e493b93f5bd` |
| `oscar_rotated_bf16.cpp` | `1fcaf7000c7d117cb35a88327819b6fee2d2d7c669b3f42ca3fddfdbb1ac09a1` |
| `text_context_impl.h` | `dd8a4f939d4b969fef8f869a881bfbd8379418b0b629b5f003d89fa4d47d5147` |
| `test_oscar_runtime.cpp` | `cffd46d9707e6c740b7f34152ae08b0d3cc1486fca688ae3720d23ef79bcf6eb` |
| `test_oscar_live_reference_validator.cpp` | `f0b02fb4dcab35521af35851b095a3ec8eccae4c9481e1f3db458f347e8feb31` |

## Diagnostic history

The first validator pass exposed a defect in the newly written independent checker, not in
the live runtime: V zero-point metadata was initially indexed as `group+2` instead of
`2*group+1`. Correcting the checker to the official four-value-per-group layout and rerunning
the unchanged live taps produced the complete 2/2 and 30/30 PASS results above. No runtime
cache, tier, codec, rotation, or attention discrepancy remained.

## Verdict and authorization

**PASS.** Actual Qwen3.8 Q/K/V tensors feed the calibrated C4 rotated mixed cache; live recent
aging exactly matches the official `OscarInt2G128` representation; the slow live reader matches
the independent D2.3a-style reference through rotated Q, scores, softmax, AV, and `R_V.T`
recovery for prefill and forced decode; all 16 full-attention layers are covered; and no GDN,
legacy-Q2, or hidden historical-BF16 substitution occurs.

The project now has a **correctness-qualified live calibrated OSCAR INT2 runtime**. D3 model-
fidelity qualification is authorized. Performance optimization and production persistence remain
future phases.
