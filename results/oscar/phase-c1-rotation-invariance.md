# OSCAR Phase C1 — rotated-BF16 invariance

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and result

This test compares the original captured BF16 full-attention computation with the same computation
using the Phase B3 fitted K/V rotations, the corresponding query rotation, and the V inverse after
attention. No INT2 quantization, runtime integration, speculative-decoding change, or kernel
optimization was used.

**Result: PASS for all 16 verified full-attention layers.**

- Input tokens: 256, official converted chunk `1.pt`
- Q/K/V source dtype: BF16; source values were retained exactly
- Diagnostic compute dtype: FP64, to isolate matrix orientation and inverse semantics from
  intermediate BF16 rounding
- Q heads / KV heads: `24 / 4`
- GQA ratio: `6`, with query head `h` mapped to KV head `h // 6`
- Head dimension: `256`
- Fitted composition: `r_h_pbr` (`R * H_d * P_br`)
- Full-attention layers: `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`

The exact test implementation is:

```text
D:\AI\voidinfer-adaptive-dflash2\tools\oscar\test_rotation_invariance.py
```

## Reference equations and confirmed orientation

The official `.pt` dumps use row vectors in `[tokens, heads, head_dim]` order. For each layer the
test applies:

```text
Q_rot = Q @ R_K
K_rot = K @ R_K
V_rot = V @ R_V
scores_rot = causal_scores(Q_rot, K_rot)
O_rot_raw = softmax(scores_rot) @ V_rot
O_recovered = O_rot_raw @ R_V.T
```

The GQA expansion is performed before score and value products using the verified mapping
`[0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,3]`.

This confirms the correct semantics for this fitted representation:

- Q and K use the same right-side K rotation, so `Q_rot K_rot.T` preserves scores up to the
  serialized FP32 orthogonality error.
- V uses the separate V rotation and must be inverse-rotated on the right after attention.
- V rotation is not expected to preserve the raw attention output before the inverse. The test
  measures that intentional difference separately.
- RoPE is not reapplied. The Phase B2 tensors are already post-RoPE Q/K values.
- The inverse V operation belongs immediately after the weighted V sum and before the model's
  output projection.

## Reproduction command

Run in the Phase B1 isolated environment:

```powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\test_rotation_invariance.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c1-rotation-invariance.json'
```

The machine-readable result is:

```text
D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c1-rotation-invariance.json
```

The test uses strict fail-closed thresholds:

| Metric | Maximum allowed absolute error |
|---|---:|
| Causal attention score | `1.0e-05` |
| Softmax probabilities | `1.0e-06` |
| Recovered post-attention output | `1.0e-05` |

These thresholds are justified by the Phase B3 serialized FP32 rotation orthogonality maxima of
`2.011652722e-08` for K and `1.916613712e-08` for V. They are still substantially above the
observed score and recovered-output errors and do not mask an orientation-scale failure.

## Per-layer results

Errors are max absolute differences. `raw V-rot` is the intentionally non-invariant attention
output before applying `R_V.T`; `recovered` is the required post-V-inverse result.

| Layer | Score | Softmax | Raw V-rot | Recovered post-attention | Result |
|---:|---:|---:|---:|---:|:---:|
| 3 | 1.928265312e-07 | 2.499663554e-08 | 4.149036307e+00 | 9.359409475e-08 | PASS |
| 7 | 1.951365336e-07 | 3.455283926e-08 | 6.221278806e+00 | 1.088392918e-07 | PASS |
| 11 | 2.511186814e-07 | 3.246614177e-08 | 7.886345477e+00 | 2.063480817e-07 | PASS |
| 15 | 2.243033030e-07 | 3.635663887e-08 | 5.605344026e+00 | 1.559535021e-07 | PASS |
| 19 | 2.237109769e-07 | 3.290356132e-08 | 6.484682803e+00 | 1.578182922e-07 | PASS |
| 23 | 2.499755478e-07 | 5.530217129e-08 | 1.576330153e+01 | 4.508082623e-07 | PASS |
| 27 | 3.240096529e-07 | 7.114301309e-08 | 2.747916380e+01 | 1.469125495e-06 | PASS |
| 31 | 3.379992553e-07 | 5.283500731e-08 | 3.838722738e+01 | 2.024847870e-06 | PASS |
| 35 | 3.933178050e-07 | 5.860079022e-08 | 3.374178928e+01 | 1.694271496e-06 | PASS |
| 39 | 4.415737221e-07 | 4.945066640e-08 | 3.502946248e+01 | 9.086132184e-07 | PASS |
| 43 | 3.810843463e-07 | 6.232390315e-08 | 5.677834765e+01 | 2.282102813e-06 | PASS |
| 47 | 3.343878845e-07 | 6.174881528e-08 | 5.728219539e+01 | 3.080665177e-06 | PASS |
| 51 | 4.971724872e-07 | 6.477188569e-08 | 4.849794476e+01 | 3.373735723e-06 | PASS |
| 55 | 3.374257004e-07 | 3.989649178e-08 | 2.747416831e+01 | 9.307173485e-07 | PASS |
| 59 | 2.263791670e-07 | 4.886619509e-08 | 7.557940795e+01 | 1.042333349e-06 | PASS |
| 63 | 2.731841704e-07 | 3.556130568e-08 | 8.232807710e+01 | 1.710273843e-06 | PASS |
| **max** | **4.971724872e-07** | **7.114301309e-08** | **8.232807710e+01** | **3.373735723e-06** | **PASS** |

The raw V-rotated output is intentionally different, reaching `82.3280771` max absolute error;
this is evidence that omitting the V inverse would be incorrect. After `R_V.T`, the maximum
recovered attention-output error is only `3.373735723e-06`.

Additional round-trip diagnostics from the serialized FP32 matrices:

| Quantity | Maximum max absolute error |
|---|---:|
| `Q @ R_K @ R_K.T` vs Q | `2.945389328e-07` |
| `K @ R_K @ R_K.T` vs K | `3.206442791e-07` |
| `V @ R_V @ R_V.T` vs V | `1.032426029e-06` |

## Post-attention/layer-output scope

The recovered attention output is the post-attention tensor immediately before the model's output
projection. The Phase B2 activation dump does not contain the per-layer output-projection weights,
residual input, or post-projection residual state, so a full production layer-output comparison is
not practical from this artifact alone. The V inverse placement is nevertheless directly tested at
the exact boundary where it must occur; any subsequent linear output projection would receive the
same recovered tensor up to the measured error.

## Anomalies and blocker status

No orientation, row/column, RoPE, Q rotation, K rotation, V rotation, inverse-V, layer mapping, or
GQA mapping anomaly was found. Every expected full-attention layer passed all three required
metrics. No INT2 quantization or runtime rotation wiring was added.

The only remaining limitation is dataset quality: these are 256-token smoke rotations, not the
10K/30K production calibration assets. C1 proves mathematical invariance of the fitted matrices
and semantics before quantization; it does not qualify the rotations for production quality or
integrate them into CUDA runtime code.
