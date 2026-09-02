# OSCAR Phase D1.3 — forced-token persistent-cache qualification

Date: 2026-09-01  
Status: **PASS — implementation-validated**

## Scope and gate

This phase qualifies only the OSCAR coordinate/cache plumbing. It does not require downstream
BF16 hidden-state, logit, top-1, or natural greedy-token equality. The comparison is:

- A: matched FP32 unrotated Q/K/V and persistent FP32 cache in original coordinates;
- B: matched FP32 `Q@R_K`, `K@R_K`, `V@R_V`, persistent FP32 cache in rotated coordinates,
  and FP32 `R_V.T` recovery;
- both paths use the same scalar FP32 causal GQA attention, FMA accumulation, stable softmax,
  cache append/read implementation, and final BF16 conversion boundary.

The C4 asset is unchanged:

| Item | Value |
| --- | --- |
| Asset identity | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Asset manifest hash | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| K runtime-bank SHA-256 | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` |
| V runtime-bank SHA-256 | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |
| Runtime executable SHA-256 | `007D3D525D66FD059EB71B7187A9F67E32D20A769C73E0390C5C9C07CB6FE050` |

## Deterministic fixture

The prompt is the existing D1/D1.1/D1.2 deterministic 32-token prompt:
`token[i] = 198 + ((i * 131) % 4096)`. Runtime options were one lane, context/KV capacity
4096, prefill chunk 256, CUDA Graphs off, ordinary backend, no prefix reuse, and BF16 production
KV configuration. The matched diagnostic branch uses its separate FP32 cache for attention.

The runtime emits one initial prefill `Begin` token before ordinary decode. The opt-in control
therefore forces seed `997` at that prefill commit and then forces eight ordinary decode commits:

```text
997,1001,1003,1005,1007,1009,1011,1013,1015
```

Both A and B returned exactly that nine-token committed sequence. The eight ordinary decode
attention steps are table positions 1–8 below. This is teacher forcing at the runtime commit
boundary: the committed forced token is the input ledger token for the next ordinary decode
round; the ordinary decode scheduler and attention path are not replaced.

## Commands

Build, using the current `D:\AI\build-adaptive-dflash2` build tree:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test -j 4"
```

Runtime smoke command (the final run used the fresh directory ending in `20260901e`):

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS='D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256='6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR='D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d1-3-diagnostics-20260901e'
$env:NINFER_OSCAR_D1_3_FORCED_DECODE='1'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe'
```

The executable exited `0` and reported the C4 calibrated asset telemetry. The closed-form
validator was:

```powershell
& 'D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\python\python-3.12.7-x64-1\python.exe' 'tools\oscar\validate_d1_3_forced_decode.py' `
  --unrotated 'results\oscar\phase-d1-3-diagnostics-20260901e\matched-fp32-unrotated' `
  --rotated 'results\oscar\phase-d1-3-diagnostics-20260901e\matched-fp32-rotated' `
  --rotation-k 'results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\k_rotation_fp32.bin' `
  --rotation-v 'results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime\v_rotation_fp32.bin' `
  --layer 3 --steps 8
```

Validator output is archived as `results/oscar/phase-d1-3-validation-20260901e.json`.

## Capture and cache coverage

The matched path produced Q/K/V/attention records for every expected full-attention layer at
every prefill/decode step. The fail-closed validator found all 16 layers
`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63` and no GDN records. GDN recurrent state is not
treated as KV.

The final diagnostic archive contains 4,216 files and 197,692,416 bytes: 2,036 files for A and
2,180 files for B. The additional B files are the FP32 recovered-attention records.

At layer 3, the persistent cache was read back after every append. Both paths used FP32 cache
banks with layout `[head_dim, kv_head, absolute_position] = [256,4,position]`; A stored original
coordinates and B stored rotated coordinates. The position vectors were identical and contiguous:

```text
prefill: 0..31
decode:  32,33,34,35,36,37,38,39
total:   40 positions
```

The A cache read-back matched its append history exactly (`relative L2 = 0`). No production
BF16-cache fallback occurred in the matched attention path: `TextContext::attn_mix` used
`matched_fp32_cache_->append()` and `matched_fp32_cache_->attention()` for each full-attention
layer, while the ordinary production paged cache was not used as the diagnostic attention source.

## Layer-3 per-decode results

Cache K/V are checked over the entire read-back cache prefix after each append against the other
run transformed by its layer-specific C4 matrix. Score and softmax are recomputed from the
captured FP32 Q/cache records with the verified GQA mapping `q_head // 6`; recovered attention is
the direct runtime A `attention_fp32` versus B `attention_recovered_fp32` comparison.

| Decode pos | Absolute pos | Cache K rel L2 | Cache V rel L2 | Score rel L2 | Softmax rel L2 | Recovered attention rel L2 | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 32 | 3.08864e-7 | 2.86285e-7 | 9.41184e-8 | 4.00075e-7 | 1.07450e-6 | PASS |
| 2 | 33 | 3.08707e-7 | 2.85501e-7 | 1.00921e-7 | 4.11430e-7 | 1.60696e-6 | PASS |
| 3 | 34 | 3.08852e-7 | 2.85298e-7 | 1.00036e-7 | 4.49973e-7 | 9.41492e-7 | PASS |
| 4 | 35 | 3.08504e-7 | 2.84865e-7 | 1.02839e-7 | 4.68753e-7 | 1.20595e-6 | PASS |
| 5 | 36 | 3.08131e-7 | 2.84792e-7 | 1.03510e-7 | 4.46232e-7 | 1.27754e-6 | PASS |
| 6 | 37 | 3.08061e-7 | 2.85100e-7 | 1.02668e-7 | 4.42251e-7 | 1.06492e-6 | PASS |
| 7 | 38 | 3.08215e-7 | 2.85146e-7 | 9.32469e-8 | 5.74594e-7 | 1.30209e-6 | PASS |
| 8 | 39 | 3.08350e-7 | 2.85187e-7 | 1.07778e-7 | 5.10012e-7 | 1.30369e-6 | PASS |

Aggregate maxima across the eight decode appends:

| Quantity | Max absolute | Max mean absolute | Max relative L2 |
| --- | ---: | ---: | ---: |
| Q transform `Q_B` vs `Q_A @ R_K` | — | — | 4.09037e-7 |
| Read-back cache K transform | 2.89865e-6 | 2.54815e-7 | 3.08864e-7 |
| Read-back cache V transform | 2.26600e-6 | 1.69730e-7 | 2.86285e-7 |
| QK score | 2.50775e-6 | 5.38611e-7 | 1.07778e-7 |
| Softmax | 2.82577e-7 | 1.22701e-8 | 5.74594e-7 |
| Recovered attention | 6.91414e-6 | 5.11797e-7 | 1.60696e-6 |

The qualification tolerance is recovered-attention relative L2 `<= 1e-5`, justified by the
D1.1/D1.2 FP32 rotation/reference scale. The observed maximum is `1.60696e-6`; it does not grow
with sequence length.

Two pre-final harness observations were corrected before this qualifying run: the runtime's
prefill `Begin` commit was included as the forced seed, and the direct cache dump was corrected to
use full-attention ordinal `fidx` rather than model-layer label `3`. The final validator passed
after both corrections; neither was a runtime coordinate defect.

## Downstream information only

The final forced A/B runs still show the previously documented downstream sensitivity: final
hidden relative L2 `7.18921733e-2` and logit relative L2 `2.32925045e-2`. These are not D1.3
correctness gates because both executions cross the common downstream BF16/model boundary and the
NVFP4/GDN network amplifies small finite FP32 differences. The forced committed token sequence
itself agrees exactly by construction and validates persistent cache plumbing rather than model
quality.

## Implementation locations

- `src/targets/qwen3_6/impl/runtime/text_context_impl.h:847-945`: matched FP32 A/B path,
  post-RoPE Q/K/V conversion, cache append/attention, inverse `R_V.T`, and final BF16 boundary.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cu:160-230`: FP32 cache append and
  causal GQA attention kernels; `:417-451`: diagnostic wrappers.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:317-374`: persistent FP32 cache,
  direct prefix read-back, and attention dispatch; `:427-437`: dump step tagging.
- `src/targets/qwen3_6/impl/runtime/program_impl.h:54-80`: fail-closed forced-token parser;
  `:11188-11202`: prefill seed commit; `:11293-11375`: ordinary decode commit forcing.
- `src/targets/qwen3_6/impl/runtime/program.h:697-700`: opt-in forced-sequence state.
- `tools/oscar/validate_d1_3_forced_decode.py`: fail-closed full-layer coverage, position,
  cache read-back, rotation, score/softmax, and recovered-attention validation.

## Decision

**PASS.** Persistent OSCAR cache coordinates remain correct through the deterministic prefill →
eight ordinary decode append sequence. A and B have identical logical positions, B is the proper
layer-specific transform of A across the entire read-back cache, and recovered FP32 attention
remains within the declared FP32 tolerance without growth. No cache-coordinate, layer-selection,
GQA, RoPE, or decode-path defect was found.

Exact full-model BF16 greedy-token equality is retired as a rotation-correctness requirement; it
belongs to a later model-fidelity/quality phase. OSCAR rotation plus persistent-cache plumbing is
implementation-validated, and D2 may proceed under its separate INT2/reference gates. No INT2,
recalibration, 512/4K expansion, DFlash/MTP/adaptive-K, or kernel optimization was performed in
D1.3.
