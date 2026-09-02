# OSCAR Phase D1 — full-runtime rotated-BF16 integration

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and decision

This phase added an opt-in runtime loader and on-the-fly OSCAR Q/K/V rotation path for the
validated C4 assets. It did not modify NVFP4 weights, INT2 packing, DFlash2/MTP/adaptive-K,
speculative decoding, or CUDA attention kernels.

**Decision: BLOCKED.** The real runtime loads the immutable C4 assets and reaches the rotated
path, but the first full-attention layer produces material end-to-end drift in the BF16 runtime.
Per the stop condition, the 512-token and 4K tests were not started.

## Inputs and immutable asset validation

The runtime was bound to:

| Item | Value |
| --- | --- |
| Model artifact | `D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR identity | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Asset manifest SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| K asset SHA-256 | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` |
| V asset SHA-256 | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |
| K/V format | raw little-endian FP32, 16 contiguous `[256,256]` matrices per bank |

The loader requires `NINFER_OSCAR_ROTATION_MODE=oscar-rotated-bf16`, the runtime asset directory,
and the expected model SHA-256. It validates the C4 identity, manifest digest, exact model digest,
topology fields, full-layer list, file names, exact file sizes, SHA-256 payload hashes, FP32
finite values, and device upload before publishing the rotation set. Unsupported tensor shapes
or layers fail closed in the CUDA entry point. The loader emits:

```text
OSCAR telemetry: asset_identity=qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1 asset_hash=4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560 calibrated=true full_attention_layers=16 rotation_mode=oscar-rotated-bf16 model_sha256=6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e
```

The loader is runtime-only and does not use LibTorch or `.pt` deserialization.

## Verified layer mapping and runtime topology

The runtime configuration used by the test is the previously verified Qwen3.8-27B topology:

| Property | Verified value |
| --- | ---: |
| Total text layers | 64 |
| Full-attention layers | 16 |
| GDN layers | 48 |
| Query heads | 24 |
| KV heads | 4 |
| GQA ratio | 6 (`q_head // 6`) |
| Head dimension | 256 |
| Rotary dimension | 64 |
| Runtime tensor layout | contiguous `[D, heads, tokens]` |
| Serialized calibration layout | `[tokens, heads, D]`, token-major/head-major/D-fastest |
| Runtime KV dtype | BF16 |

Only these model layers use the asset bank:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

The complement is the untouched GDN set:

```text
0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25,
26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48, 49, 50,
52, 53, 54, 56, 57, 58, 60, 61, 62
```

The runtime never interprets GDN recurrent state as KV cache. The loader's fixed matrix bank
contains exactly 16 K and 16 V entries, and the attention entry point rejects non-`[256,24,T]`
Q or non-`[256,4,T]` K/V tensors.

## Implementation locations

- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h/.cpp/.cu`: fail-closed manifest/payload
  loader, immutable C4 checks, device banks, layer mapping, BF16 matrix application, and inverse
  V transform.
- `src/targets/qwen3_6/impl/runtime/text_context.h`: per-`TextContext` optional rotation-set
  ownership.
- `src/targets/qwen3_6/impl/runtime/text_context_impl.h:244`: environment-controlled loader
  activation; `:866-868`: post-RoPE Q/K and projected V diagnostic boundary; `:891-895`: Q/K/V
  rotation; `:928`: attention result; `:934-935`: inverse V before gate/output projection.
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h:406-430`: conditional BF16 scratch banks;
  Q scratch is reused for inverse V output.
- `src/targets/qwen3_6/CMakeLists.txt`: runtime source registration.
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp`: deterministic normal-vs-rotated real-model
  comparison and fail-closed comparison gate.
- `tools/oscar/analyze_runtime_attention_diagnostics.py`: independent BF16 stage, score, softmax,
  rotation-orientation, and inverse-orientation analysis.

The existing Q/K/V boundary is unchanged: Q/K are after their RMSNorm and in-place RoPE; V is
the corresponding projected BF16 value before attention/cache append. The rotated path applies
`Q' = Q @ R_K`, `K' = K @ R_K`, `V' = V @ R_V`, runs the existing causal GQA attention, then
applies `R_V.T` to the weighted V result before the existing sigmoid gate and output projection.
No RoPE is reapplied.

## Reproduction

The latest build supplied for this phase was `D:\AI\build-adaptive-dflash2`.

Build command:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test -j 4"
```

The target built successfully. Test executable SHA-256:

```text
9c40f813b85bc135b5ef07d09938c21af61d38e5fda26e7f0139ffb2dec70a29
```

Runtime command:

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS='D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256='6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\d1-runtime-diagnostics'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe'
```

The deterministic input is `token[i] = 198 + ((i * 131) mod 4096)`. The test is configured for
BF16 KV, no speculative backend, no CUDA graph, and unchanged NVFP4 weights. It has cases for
32, 512, and 4096 tokens, but the fail-closed gate stopped after the first 32-token case.

## First-divergence audit

The 32-token run produced normal and rotated diagnostics for all 64 layer outputs and all 16
full-attention stages. Layers 0, 1, and 2 are bit-identical. Layer 3 is the first full-attention
layer and the first divergence. Layers 4 onward are GDN or later full-attention layers receiving a
changed residual stream; no GDN-specific rotation was executed.

Independent stage analysis used:

```powershell
& 'D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\python\python-3.12.7-x64-1\python.exe' `
  'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\analyze_runtime_attention_diagnostics.py' `
  --normal 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d1-runtime-diagnostics\normal' `
  --rotated 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d1-runtime-diagnostics\rotated' `
  --k-bank 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin' `
  --v-bank 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin'
```

Metrics are max absolute, mean absolute, and relative L2, in that order:

| Stage at layer 3 | max abs | mean abs | relative L2 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Pre-Q | `0` | `0` | `0` | same normal/rotated input |
| Pre-K | `0` | `0` | `0` | same normal/rotated input |
| Pre-V | `0` | `0` | `0` | same normal/rotated input |
| Runtime Q vs `Q @ R_K` | `1.52416e-2` | `1.32809e-3` | `1.66955e-3` | BF16 result of the right-side matrix application |
| Runtime K vs `K @ R_K` | `1.32532e-2` | `1.33212e-3` | `1.65526e-3` | BF16 result of the right-side matrix application |
| Causal QK scores | `1.48105e-2` | `2.59536e-3` | `4.55208e-4` | score invariance lost after BF16 Q/K materialization |
| Softmax rows | `2.17747e-3` | `1.06272e-4` | `1.72080e-3` | follows Q/K BF16 round-off |
| Raw attention, different V coordinate | `4.453125` | `5.45569e-1` | `1.42944` | expected non-invariant pre-inverse diagnostic |
| Inverse orientation vs `attention_raw @ R_V.T` | `7.80139e-3` | `5.17556e-4` | `1.63859e-3` | right-side transpose orientation is correct |
| Normal attention vs recovered rotated attention | `1.5625e-2` | `9.61884e-4` | `3.26458e-3` | BF16 rotation/inverse round-trip residue |
| Layer-3 post-MLP output | `1.25e-1` | `2.27139e-3` | `7.73924e-3` | first layer-output divergence |

These diagnostics rule out a missing pre-rotation tensor, layer-3 mapping error, GQA expansion
error, RoPE reapplication, or inverse transpose direction as the primary cause. The runtime
matrix results follow the validated row-vector convention, but each transformed Q/K/V and the
inverse result is materialized as BF16. C1's earlier PASS used FP64 products specifically to
remove this intermediate BF16 rounding; it did not qualify this real CUDA BF16 round-trip.

## Full-model result at the stopping point

The test output for the only permitted case was:

```text
case_tokens=32 hidden_max_abs=1.21875 hidden_mean_abs=0.0738913286 hidden_relative_l2=0.059814581 logits_max_abs=0.60546875 logits_mean_abs=0.0761243858 logits_relative_l2=0.0204135977 top1_agree=false top10_agree_count=8 generated_agree=false
OSCAR runtime rotated-BF16: FAIL: OSCAR rotated-BF16 runtime comparison failed
```

The run produced 324 finite diagnostic BF16 files totaling 86,996,992 bytes. The runtime asset
loader telemetry and C4 hashes passed before the comparison gate. No 512-token, 4K, or 16K
comparison is claimed because the first short smoke already failed and D1 requires stopping at
the first material divergence.

## Unresolved blocker

The remaining blocker is the numerical contract for an on-the-fly BF16 rotation round-trip. The
FP32 assets and matrix orientation are valid, but the existing BF16-only attention interface
requires transformed Q/K/V and recovered output to be rounded to BF16. The resulting small first
full-attention error is amplified by the unchanged output projection/residual/GDN stack and is
not equivalent at the required full-model logit/token level.

Do not proceed to INT2 runtime integration or performance work until a follow-up experiment
defines and validates a rotation path whose intermediate precision preserves the full-model
equivalence gate.
