# OSCAR Phase B2 — 256-token Q/K/V smoke capture

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and result

Phase B2 implemented and validated a deterministic, opt-in native Q/K/V capture path for the
verified Qwen3.8-27B full-attention layers. The capture is a smoke dataset only. No OSCAR fitter,
10K/30K capture, INT2 runtime integration, DFlash2/MTP/adaptive-K work, or CUDA-kernel
optimization was run.

Final result: **PASS**.

- Useful tokens: `256`
- Prompt: 256 copies of token id `198`
- Generated output: 1 greedy token; native smoke request completed with `OutputLimit`
- Captured layers: 16, exactly the verified full-attention set
- Chunks: 1
- Q/K/V tensors: 48 raw BF16 payloads
- Raw payload bytes: `67,108,864`
- Capture directory bytes including manifest and sidecar: `67,190,687`
- Manifest SHA-256: `df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89`
- Manifest sidecar: `results/oscar/captures/phase-b2-qkv-256/manifest.sha256`

The complete provenance-bearing result is at:

```text
D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-b2-qkv-256\manifest.json
```

## Environment and input identity

The calibration tooling remains isolated from VoidInfer:

| Item | Verified value |
|---|---|
| Calibration environment | `D:\AI\tools\oscar-calibration\.venv` |
| Python | 3.12.10, 64-bit Windows |
| PyTorch | 2.13.0+cpu |
| CUDA in fitter environment | unavailable/False; CPU-only is intentional |
| Environment setup | `D:\AI\voidinfer-adaptive-dflash2\tools\oscar\setup_calibration_env.ps1` |
| Native source | `D:\AI\voidinfer-adaptive-dflash2` |
| Latest build | `D:\AI\build-adaptive-dflash2` |
| Build target | `ninfer_qwen3_6_27b_oscar_capture_test.exe` |
| Model artifact | `D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| Capture executable SHA-256 | `e7844c73276c7d551a677e283a228d5c3cdb57648d113fd028a3390888e1959e` |

The environment setup had already passed the required torch import, deterministic `.pt` save/load,
and deterministic 128x128 `torch.linalg.eigh()` checks in Phase B1. The runtime capture itself does
not import PyTorch or LibTorch.

## Reproduction commands

Build the focused target with the latest build directory:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_capture_test -j 1"
```

Run the deterministic wrapper in a native/elevated PowerShell with the isolated validator Python:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\capture_qkv_smoke.ps1' -PythonExe 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe'
```

The wrapper binds the model and executable hashes, hashes the capture source identity, sets the
256-token budget, runs the native test, validates the dump, writes `manifest.sha256`, and validates
the sidecar. It refuses to reuse an existing output directory.

## Verified Qwen3.8 topology

The capture uses the Phase B1 topology, which was derived from the loaded artifact/runtime and the
latest DFlash2 load-plan test:

| Property | Exact value |
|---|---:|
| Text transformer layers | 64 |
| Full-attention layers | 16 |
| GDN layers | 48 |
| Q heads | 24 |
| KV heads | 4 |
| GQA ratio | 6 (`24 / 4`) |
| Head dimension | 256 |
| Rotary dimension | 64 |
| Q-to-KV map | `q_head // 6` |

Full-attention model-layer indices, 0-based:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

GDN model-layer indices, 0-based:

```text
0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45,
46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62
```

The native capture rejects any layer that is not `3 + 4 * full_attention_index`, and the validator
requires exactly the 16-layer list above. No GDN recurrent state is interpreted as or written as KV.

## Capture boundary and serialized layout

In `TextContext::attn_mix`, the verified sequence is:

```text
x
  -> input RMSNorm
  -> fused Q/gate/K/V projection
  -> Q RMSNorm -> qn; K RMSNorm -> kn
  -> in-place RoPE(qn, kn)
  -> capture qn, kn, v
  -> causal_softmax_attention(...)
       -> runtime K/V cache append inside the attention wrapper
       -> attention
```

The capture hook is immediately after `ops::rope(...)` at
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:860`, before the attention dispatch at
lines 884-892. The exact hook and opt-in arming logic are at lines 861-868.

- Q: `qn`, after Q RMSNorm and after RoPE.
- K: `kn`, after K RMSNorm and after RoPE.
- V: `v`, the corresponding projected BF16 value tensor, without Q/K RMSNorm or RoPE.
- Alignment: all three tensors use the same prefill token columns; Q has 24 heads and K/V have 4
  KV heads for each of the 256 positions.
- Runtime logical views: Q `[D,Hq,T] = [256,24,256]`; K/V `[D,Hkv,T] = [256,4,256]`.
- Serialized dimensions: Q `[tokens,heads,head_dim] = [256,24,256]`; K/V
  `[256,4,256]`.
- Serialized ordering: contiguous token-major, head-major, dimension-fastest `[T,H,D]` for the
  single-batch smoke. This preserves the runtime contiguous physical order of the `[D,H,T]`
  views while making the dimensions explicit for the validator/fitter bridge.
- GQA mapping: query head `h` maps to KV head `floor(h / 6)`, represented explicitly in the
  manifest as 24 entries.

The hook never observes packed cache rows. It copies the source BF16 tensors to host memory before
the causal-attention wrapper can append or pack K/V.

## Implementation changes

- Added `src/targets/qwen3_6/impl/runtime/oscar_qkv_capture.h` and `.cpp`.
  The implementation is opt-in through `NINFER_OSCAR_QKV_CAPTURE_DIR` plus the explicit arming
  variable, copies only BF16 contiguous Q/K/V at the verified shape, rejects CUDA graph capture,
  computes payload SHA-256 values, and writes a provenance-bearing JSON manifest.
- Added the hook to `src/targets/qwen3_6/impl/runtime/text_context_impl.h:861-868` and kept the
  capture object unarmed until after Engine construction, so setup/warmup activity cannot pollute
  the smoke dump.
- Added `tests/targets/qwen3_6_27b/test_oscar_capture.cpp`, which binds the verified DFlash2
  artifact, disables speculative execution and CUDA graphs, submits the deterministic 256-token
  prefill, and finalizes the capture.
- Added the focused target to `src/targets/qwen3_6/CMakeLists.txt` and
  `tests/CMakeLists.txt`.
- Replaced `tools/oscar/validate_dump.py` with a stdlib-only fail-closed validator. It checks
  topology, exact full-layer coverage, no GDN entries, shape/dtype/token consistency, GQA mapping,
  finite BF16 values, exact payload sizes, file integrity, per-record provenance, and manifest
  sidecar hash.
- Added `tools/oscar/capture_qkv_smoke.ps1` to make model/build/output/source identity and the
  capture command reproducible.

## Validation evidence

Focused build result:

```text
[5/5] Linking CXX executable tests\ninfer_qwen3_6_27b_oscar_capture_test.exe
```

Native smoke result:

```text
OSCAR capture smoke: PASS prompt_tokens=256 generated_tokens=1
```

Final validator command, including sidecar verification:

```powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\validate_dump.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-b2-qkv-256\manifest.json' --manifest-sha256 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-b2-qkv-256\manifest.sha256'
```

Validator result:

```text
OSCAR QKV dump: PASS layers=16 chunks=1 useful_tokens=256 dump_bytes=67108864 manifest_sha256=df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89
```

All 48 payload files were non-empty and exact-sized. No NaN/Inf BF16 exponent patterns were found,
all manifest payload hashes matched, and the manifest sidecar matched the manifest hash.

Representative first four values from the first element of the first token/head/dimension tuple,
inspected independently by the validator:

| Model layer | Q | K | V |
|---:|---|---|---|
| 3 | `[-0.142578125, -0.333984375, -0.64453125, -0.99609375]` | `[0.271484375, 0.416015625, -0.66796875, -0.1767578125]` | `[-0.515625, 0.01336669921875, 0.435546875, 0.41015625]` |
| 35 | `[0.82421875, 0.01055908203125, -0.021484375, -0.57421875]` | `[0.0120849609375, -0.01202392578125, -0.022705078125, 0.006378173828125]` | `[-0.006103515625, 0.69140625, 0.328125, 0.15234375]` |
| 63 | `[-1.4140625, -0.044677734375, -2.234375, -1.890625]` | `[-0.103515625, -0.06982421875, 0.06396484375, -0.1298828125]` | `[-2.265625, 0.048828125, 1.5859375, 0.0108642578125]` |

## Mismatches and unresolved questions

The first native attempt exposed a Windows file-finalization issue: the temporary manifest stream
was still open when `manifest.json.tmp` was renamed. The implementation was corrected to close and
check the stream before rename; the clean rerun passed. This did not alter any tensor payload and
the failed temporary output was discarded before the final run.

No Q/K/V shape, layer coverage, GQA, token alignment, finite-value, capture-stage, payload-hash,
or manifest-hash mismatch remained after the fix. The final directory contains no GDN or drafter
KV tensor.

The dump is intentionally raw `.bin` plus manifest rather than the upstream fitter's `.pt` chunk
layout. A reviewed serialization bridge and the upstream-compatible 10K/30K calibration procedure
remain before fitting. The upstream reference is
[FutureMLS-Lab OSCAR `compute_kv_rotation.py`](https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/main/rotation/compute_kv_rotation.py).

## Stop condition

The Phase B2 stop condition is satisfied: approximately 256 useful tokens were captured, all 16
expected full-attention layers have aligned Q/K/V, manifest and payload validation passed, and no
GDN tensors were captured. Work stops here; OSCAR fitting is not started.
