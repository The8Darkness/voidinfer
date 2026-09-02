# OSCAR Phase C2 — slow calibrated INT2 reference

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and classification

This is an offline, deliberately slow correctness reference over the validated Phase B2
256-token smoke capture. It does not modify or call the production CUDA runtime, run the
OSCAR fitter, or perform 10K/30K calibration.

**Calibrated OSCAR reference path: PASS.**

The PASS classification means that the capture → fitted rotations → upstream INT2
quantization → dequantization → causal GQA attention → inverse-V chain is finite,
deterministic, and mathematically coherent for this smoke dataset. It is not a
production-quality calibration or model-quality claim; the source is only 256 repeated
tokens.

C1 rotated-BF16 invariance was checked as a prerequisite for every layer before any
quantized path was evaluated.

## Implementation

The reference utility is:

    D:\AI\voidinfer-adaptive-dflash2\tools\oscar\test_int2_reference.py

It evaluates the exact verified full-attention set:

    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63

All source tensors are loaded as exact BF16 and represented as row vectors in
[tokens, heads, head_dim] order. Diagnostic matrix products and attention use FP64
to isolate quantization and orientation behavior from an additional intermediate
rounding effect. Q remains unquantized; only rotated K/V are quantized.

The four compared paths are:

1. BF16 baseline: Q, K, V in original model coordinates.
2. Rotated BF16: Q @ R_K, K @ R_K, V @ R_V, attention, then output @ R_V.T.
3. Existing fixed-Hadamard Q2 control: Q/K/V use normalized H256; K/V use the
   current runtime affine Q2 contract with per-row K clip 0.93, V clip 0.91, and
   BF16 scale/zero metadata; the weighted output uses H256.T.
4. Calibrated OSCAR INT2: Q @ R_K, K @ R_K, V @ R_V; K/V use the pinned upstream
   simulate_int2_asym semantics; attention output uses R_V.T.

The reference uses integer code symbols in [0,3] and performs the same dequantization
before attention. Bit packing is intentionally not modeled here because it is storage
only and this phase is not a runtime/storage test.

## Provenance and commands

| Item | Value |
|---|---|
| Smoke dump | D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official |
| Conversion manifest SHA-256 | D3117A51BA933B79BC1A54A2F436DBDDBE7A40E5BF46E73F3497708C3FB3D0A4 |
| C1 machine report SHA-256 | 514EAA0A7B8B7574E7628B4A8D7DB4BFDDA4C84BD11D8A0D1674D47A441B0A91 |
| K rotation SHA-256 | A5D424C835D82D055AE26B8001581B2E5389D825A4A01BD47AD1FBE12B5981FA |
| V rotation SHA-256 | CCCF5EB11910034AC05B0716DDC98D3ED6C946CFC7A2E917E0DDAF7A8041ABD7 |
| Pinned upstream source | D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py |
| Upstream source SHA-256 | F7F9F738BE5DEA75CC1D7E8D6928E4EE91DF1CBF1E1C7B7D9C7D308B55D4DA8B |
| Reference utility SHA-256 | 81A98AE8DA997752F12D0A2E4A290F113A207B25B59A758C26557B3944066187 |
| Representative JSON SHA-256 | 9904A73F0E64D91E7C93E2512A4305B4489E5DD3D4EF419CC7EE9A4FD6252491 |
| All-layer JSON SHA-256 | B838A36F53D9633EABDACE428087566F09D2C28FCDBF6AAC69D822FA810E96CB |

Representative layer command, run first:

~~~powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\test_int2_reference.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' --upstream 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --c1-json 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c1-rotation-invariance.json' --report-json 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c2-int2-reference-layer3.json' --layers 3
~~~

All-layer command:

~~~powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\test_int2_reference.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' --upstream 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --c1-json 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c1-rotation-invariance.json' --report-json 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-c2-int2-reference.json' --layers all
~~~

The actual all-layer command also recorded the representative command in the JSON
manifest. The isolated CPU environment emitted its known non-fatal missing-NumPy
warning; PyTorch, tensor loading, and all reference computations completed.

## Quantization authority

The calibrated path uses the pinned FutureMLS-Lab OSCAR implementation at:

    https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/main/rotation/compute_kv_rotation.py

Its asymmetric INT2 simulation is applied independently to each final-dimension row:

    x_fp32 = x.float()
    min/max = row extrema
    scale = clamp(max - min, 1e-8) / 3
    zero = -min / scale
    code = int32(x_fp32 / scale + zero + 0.5), clamped to [0, 3]
    dequant = (code - zero) * scale

The local reference computes those same operations and compares its result
element-for-element with upstream simulate_int2_asym. No clipping ratio, fixed
Hadamard matrix, or BF16 metadata was substituted into the calibrated OSCAR path.

The fixed-Hadamard control mirrors the existing runtime code in
src/ops/kv_cache/hadamard_d256.cuh, src/ops/kv_cache/oscar_codec.cuh,
src/ops/kv_cache/append/kernel.cuh, and src/core/paged_kv_cache.cpp. It uses normalized
H256, per-token/per-KV-head affine Q2, K/V clipping ratios 0.93/0.91, and BF16
scale/zero metadata. This is a control, not an upstream OSCAR calibration result.

## Geometry and metrics

| Property | Value |
|---|---:|
| Useful tokens | 256 |
| Q/KV heads | 24 / 4 |
| GQA ratio | 6; query head maps to KV head q_head // 6 |
| Head dimension | 256 |
| Source shapes | Q [256,24,256], K/V [256,4,256] |
| Source dtype | BF16 |
| Attention mask | causal; query token attends keys 0 through query token |
| Metric denominator | relative L2 = norm(candidate - BF16) / max(norm(BF16), 1e-24) |
| Argmax metric | causal attention-key argmax agreement over 24 × 256 = 6144 rows |

Attention score differences are the causal attention-logit differences over valid
positions. The attention-output metric is in original model coordinates, after the
required inverse H256 or R_V.T. The smoke artifact has no output projection weights,
residual input, or LM-head activations, so full layer-output error and LM
logit/token agreement are unavailable.

## Per-layer results

Values are maximum absolute error unless marked rel-L2 or agreement.

| Layer | Path | Score max | Softmax max | Attention output max | Output rel-L2 | Attn argmax |
|---:|---|---:|---:|---:|---:|---:|
| 3 | rotated BF16 | 1.928265E-007 | 2.499664E-008 | 9.359409E-008 | 3.721377E-008 | 100.00% |
| 3 | fixed-Hadamard INT2 | 1.723080E+001 | 8.835254E-001 | 4.249023E+000 | 1.473956E+000 | 2.47% |
| 3 | calibrated OSCAR INT2 | 3.437004E+000 | 3.796334E-001 | 1.721750E+000 | 2.091640E-001 | 41.98% |
| 7 | rotated BF16 | 1.951365E-007 | 3.455284E-008 | 1.088393E-007 | 3.674678E-008 | 100.00% |
| 7 | fixed-Hadamard INT2 | 2.194910E+001 | 9.574781E-001 | 7.388300E+000 | 1.449548E+000 | 8.27% |
| 7 | calibrated OSCAR INT2 | 3.645934E+000 | 6.308347E-001 | 2.400854E+000 | 2.873142E-001 | 50.29% |
| 11 | rotated BF16 | 2.511187E-007 | 3.246614E-008 | 2.063481E-007 | 3.841158E-008 | 100.00% |
| 11 | fixed-Hadamard INT2 | 2.445230E+001 | 9.968700E-001 | 1.423145E+001 | 1.405978E+000 | 3.42% |
| 11 | calibrated OSCAR INT2 | 5.024082E+000 | 7.595785E-001 | 2.315129E+000 | 2.870462E-001 | 48.23% |
| 15 | rotated BF16 | 2.243033E-007 | 3.635664E-008 | 1.559535E-007 | 3.758708E-008 | 100.00% |
| 15 | fixed-Hadamard INT2 | 2.193765E+001 | 9.915878E-001 | 5.619794E+000 | 1.620350E+000 | 22.14% |
| 15 | calibrated OSCAR INT2 | 3.796587E+000 | 5.589002E-001 | 2.395647E+000 | 3.384401E-001 | 59.49% |
| 19 | rotated BF16 | 2.237110E-007 | 3.290356E-008 | 1.578183E-007 | 3.856794E-008 | 100.00% |
| 19 | fixed-Hadamard INT2 | 2.129483E+001 | 9.824594E-001 | 1.129904E+001 | 1.463836E+000 | 24.43% |
| 19 | calibrated OSCAR INT2 | 4.307083E+000 | 6.679308E-001 | 3.752782E+000 | 3.588852E-001 | 69.69% |
| 23 | rotated BF16 | 2.499755E-007 | 5.530217E-008 | 4.508083E-007 | 4.123278E-008 | 100.00% |
| 23 | fixed-Hadamard INT2 | 2.142835E+001 | 9.986612E-001 | 2.071171E+001 | 1.525245E+000 | 13.80% |
| 23 | calibrated OSCAR INT2 | 4.549099E+000 | 5.773472E-001 | 6.899522E+000 | 3.622691E-001 | 74.40% |
| 27 | rotated BF16 | 3.240097E-007 | 7.114301E-008 | 1.469125E-006 | 4.157786E-008 | 100.00% |
| 27 | fixed-Hadamard INT2 | 2.584079E+001 | 9.984449E-001 | 6.485498E+001 | 1.414082E+000 | 9.81% |
| 27 | calibrated OSCAR INT2 | 6.222492E+000 | 6.886109E-001 | 1.800649E+001 | 4.099763E-001 | 52.10% |
| 31 | rotated BF16 | 3.379993E-007 | 5.283501E-008 | 2.024848E-006 | 4.057353E-008 | 100.00% |
| 31 | fixed-Hadamard INT2 | 3.298889E+001 | 9.986317E-001 | 4.504442E+001 | 1.490713E+000 | 7.93% |
| 31 | calibrated OSCAR INT2 | 6.129692E+000 | 8.133041E-001 | 3.495982E+001 | 4.606063E-001 | 47.82% |
| 35 | rotated BF16 | 3.933178E-007 | 5.860079E-008 | 1.694271E-006 | 3.920034E-008 | 100.00% |
| 35 | fixed-Hadamard INT2 | 2.503291E+001 | 9.937462E-001 | 4.008203E+001 | 1.386558E+000 | 4.87% |
| 35 | calibrated OSCAR INT2 | 5.970670E+000 | 7.536406E-001 | 1.870076E+001 | 3.509122E-001 | 36.52% |
| 39 | rotated BF16 | 4.415737E-007 | 4.945067E-008 | 9.086132E-007 | 3.884964E-008 | 100.00% |
| 39 | fixed-Hadamard INT2 | 2.890738E+001 | 9.991599E-001 | 3.355823E+001 | 1.514874E+000 | 6.66% |
| 39 | calibrated OSCAR INT2 | 7.058641E+000 | 8.415731E-001 | 1.585604E+001 | 3.314831E-001 | 36.98% |
| 43 | rotated BF16 | 3.810843E-007 | 6.232390E-008 | 2.282103E-006 | 4.053722E-008 | 100.00% |
| 43 | fixed-Hadamard INT2 | 3.121502E+001 | 9.998057E-001 | 6.879643E+001 | 2.071727E+000 | 4.69% |
| 43 | calibrated OSCAR INT2 | 6.465348E+000 | 8.157889E-001 | 2.749746E+001 | 3.839897E-001 | 38.57% |
| 47 | rotated BF16 | 3.343879E-007 | 6.174882E-008 | 3.080665E-006 | 4.342503E-008 | 100.00% |
| 47 | fixed-Hadamard INT2 | 2.631906E+001 | 9.966242E-001 | 6.934597E+001 | 1.562263E+000 | 3.43% |
| 47 | calibrated OSCAR INT2 | 7.046906E+000 | 8.259612E-001 | 3.743783E+001 | 3.769680E-001 | 43.91% |
| 51 | rotated BF16 | 4.971725E-007 | 6.477189E-008 | 3.373736E-006 | 4.183646E-008 | 100.00% |
| 51 | fixed-Hadamard INT2 | 2.601150E+001 | 9.959278E-001 | 1.025863E+002 | 1.459066E+000 | 2.75% |
| 51 | calibrated OSCAR INT2 | 6.235047E+000 | 7.294796E-001 | 3.119293E+001 | 3.601067E-001 | 34.05% |
| 55 | rotated BF16 | 3.374257E-007 | 3.989649E-008 | 9.307173E-007 | 3.878715E-008 | 100.00% |
| 55 | fixed-Hadamard INT2 | 2.283493E+001 | 9.775002E-001 | 5.231521E+001 | 1.704563E+000 | 9.07% |
| 55 | calibrated OSCAR INT2 | 6.825516E+000 | 6.834561E-001 | 1.372078E+001 | 3.069726E-001 | 66.88% |
| 59 | rotated BF16 | 2.263792E-007 | 4.886620E-008 | 1.042333E-006 | 3.849653E-008 | 100.00% |
| 59 | fixed-Hadamard INT2 | 2.087877E+001 | 9.541100E-001 | 7.523228E+001 | 1.447119E+000 | 2.05% |
| 59 | calibrated OSCAR INT2 | 4.815978E+000 | 6.756883E-001 | 1.354597E+001 | 2.642331E-001 | 73.23% |
| 63 | rotated BF16 | 2.731842E-007 | 3.556131E-008 | 1.710274E-006 | 3.626699E-008 | 100.00% |
| 63 | fixed-Hadamard INT2 | 2.702884E+001 | 9.983193E-001 | 1.280092E+002 | 1.534070E+000 | 5.03% |
| 63 | calibrated OSCAR INT2 | 5.749996E+000 | 8.521844E-001 | 4.403106E+001 | 2.396977E-001 | 47.71% |

## Maxima, outliers, and controls

| Path | Max score error | Max softmax error | Max attention-output error | Max output rel-L2 | Outlier layer |
|---|---:|---:|---:|---:|---:|
| Rotated BF16 | 4.971725E-007 | 7.114301E-008 | 3.373736E-006 | 4.342503E-008 | 51 / 27 / 51 / 47 |
| Fixed-Hadamard INT2 | 3.298889E+001 | 9.998057E-001 | 1.280092E+002 | 2.071727E+000 | 31 / 43 / 63 / 43 |
| Calibrated OSCAR INT2 | 7.058641E+000 | 8.521844E-001 | 4.403106E+001 | 4.606063E-001 | 39 / 63 / 63 / 31 |

The four outlier entries in each row are respectively the layers attaining score,
softmax, attention-output, and output-relative-L2 maxima.

Quantized basis reconstruction maxima:

- Fixed-Hadamard K: 2.270004E+001; fixed-Hadamard V: 9.399324E+001.
- Calibrated OSCAR K: 2.182948E+000; calibrated OSCAR V: 6.050539E+000.
- Every path produced K/V codes in the required range 0 through 3.
- All 16 layers produced finite scores, probabilities, dequantized K/V, and
  attention outputs.

The calibrated path materially improves this smoke comparison over the existing
fixed-Hadamard control: mean per-layer maximum attention-output error is
1.715218E+001 versus 4.645777E+001 for the control. This is an informative smoke
comparison only, not a production-quality acceptance target.

## Layer-output and unresolved scope

- Full output-projection/layer-residual error is unavailable because the B2 artifact
  contains Q/K/V only. The reported attention output is the tensor immediately before
  the model output projection.
- Model LM-logit and generated-token agreement are unavailable without output
  projection and LM-head artifacts. Attention-key argmax agreement is reported instead.
- The calibrated INT2 values are expected to differ materially from BF16 at 2 bits;
  the nonzero errors above are quantization quality measurements, not failed
  rotation-invariance checks.
- This phase does not qualify 10K/30K calibration, production clipping, protected
  windows, serialized runtime metadata, bit-packed CUDA storage, or performance.

## Final result

**PASS for the slow calibrated OSCAR INT2 reference chain on all 16 full-attention
layers.** No orientation, layer mapping, GQA, inverse-V, finite-value, or upstream
quantizer-semantic blocker was found. Stop here before runtime integration or
optimization.
