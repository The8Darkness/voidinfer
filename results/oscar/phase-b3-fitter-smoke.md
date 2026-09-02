# OSCAR Phase B3 — official fitter smoke

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and result

The validated Phase B2 raw BF16 capture was converted to the upstream OSCAR `.pt` hierarchy and
processed by the unchanged official FutureMLS-Lab OSCAR fitter. The requested smoke fit succeeded.

**STOP CONDITION: PASS.**

- Input: 256-token Phase B2 smoke capture
- Full-attention layers: 16 (`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`)
- Method: `qqt_sst`
- Composition: `r_h_pbr` (`R * H_d * P_br`)
- Head dimension: `256`
- Output: separate K and V rotation checkpoints, all layers present
- Rotation dtype/shape after reload: `torch.float32`, `[256,256]`
- Fitter runtime: `2.015` seconds
- No production runtime integration, 10K/30K calibration, or kernel optimization was performed

Input provenance hashes:

| Input | SHA-256 |
|---|---|
| Phase B2 `manifest.json` | `df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89` |
| Converted `conversion_manifest.json` | `d3117a51ba933b79bc1a54a2f436dbddbe7a40e5bf46e73f3497708c3fb3d0a4` |
| Bound Qwen3.8 artifact | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |

## Upstream fitter provenance

The official source was fetched unchanged from:

```text
https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/main/rotation/compute_kv_rotation.py
```

| Item | Value |
|---|---|
| Repository | `FutureMLS-Lab/OSCAR` |
| Branch | `main` |
| Commit | `41ebcdba3db5f0ce1339c3727caea80df575d437` |
| Local source | `D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py` |
| Local script SHA-256 | `f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b` |

The upstream loader documents that `--chunk-id all` excludes chunk `0` as a six-token warmup
chunk. The Phase B2 dump contains the useful data at chunk `0`, so the converter deliberately
writes it as explicit chunk `1.pt`, and the fitter is invoked with `--chunk-id 1`. No upstream
source or mathematical behavior was modified.

## Converter

Converter:

```text
D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_b2_to_official_pt.py
```

The converter first calls the existing fail-closed `validate_dump.py` against the Phase B2
manifest and its sidecar. It then reads the raw BF16 payload as uint16 bit patterns, views those
bits as `torch.bfloat16`, preserves the serialized `[tokens, heads, head_dim]` orientation, and
writes:

```text
D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official\
  layer_<id>\q\1.pt
  layer_<id>\k\1.pt
  layer_<id>\v\1.pt
```

For every one of the 48 tensors it reloads the `.pt` file and compares the BF16 uint16 values and
shape exactly. It emits `conversion_manifest.json` containing source hashes, output `.pt` hashes,
dimensions, dtype, chunk id, model metadata, and GQA mapping.

Conversion command:

```powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_b2_to_official_pt.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-b2-qkv-256\manifest.json' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' --chunk-id 1
```

Converter result:

```text
OSCAR .pt conversion: PASS layers=16 tensors=48 tokens=256 chunk_id=1
source_manifest_sha256=df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89
conversion_manifest_sha256=d3117a51ba933b79bc1a54a2f436dbddbe7a40e5bf46e73f3497708c3fb3d0a4
```

The output contains 48 `.pt` files with total `.pt` payload bytes `67,173,328`. The conversion
manifest is 24,152 bytes. Representative output hashes are:

| Layer | Tensor | Shape | `.pt` SHA-256 |
|---:|---|---|---|
| 3 | Q | `[256,24,256]` | `43fb9beaeebe579915c34970ef34f14c72707afa2745db3e4d08b8c52525523c` |
| 3 | K | `[256,4,256]` | `f44ab422d77b976f509f871f2e6332a0f27d912a672d7be20beaf5c9ee042e15` |
| 3 | V | `[256,4,256]` | `9003dd91860d16c98314995b562355187f0bda6828a53ef75d54cc5a5c650b98` |
| 35 | Q | `[256,24,256]` | `9b9079fe7e9230b951714f985d65478f8a9e84fdf0b2042ba0acc01b5e2b0db2` |
| 35 | K | `[256,4,256]` | `1c42113f4481d08a49736d008306f9d0aecbad747a411a0bf837319a2e173426` |
| 35 | V | `[256,4,256]` | `ad2f05015b150ab1e3d5709abbd0b9e15ef0170e1a4b3724e99fa0ffa7ac02ee` |
| 63 | Q | `[256,24,256]` | `eced4db3b935ff65df4095f87ba0b4cc24a856eb71edcb321292ffb90fe7c949` |
| 63 | K | `[256,4,256]` | `1f4f68f9e0b135220fb44fb6b5f1749a8e68e937ae4ac681bd456c7462a9431b` |
| 63 | V | `[256,4,256]` | `da6fbdd42c3e8a28d0482411861747d8021674a4be75edb859abb76005312517` |

## Official fitter invocation

The isolated Phase B1 environment was used with one CPU thread for reproducible timing:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --dump-path 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' --output-dir 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' --chunk-id 1 --head-dim 256 --method qqt_sst --composition r_h_pbr
```

Official stdout reported 16 layers and separate per-layer K/V diagnostics. It saved:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `k_rotation_qqt_r_h_pbr.pt` | 4,220,861 | `a5d424c835d82d055ae26b8001581b2e5389d825a4a01bd47ad1fbe12b5981fa` |
| `v_rotation_sst_r_h_pbr.pt` | 4,220,861 | `cccf5eb11910034ac05b0716ddc98d3ed6c946cfc7a2e917e0ddaf7a8041abd7` |

The K checkpoint objective is `qqt_r_h_pbr`; the V checkpoint objective is `sst_r_h_pbr`.
Neither checkpoint is a fixed-Hadamard substitute.

The fitter emitted one non-fatal warning because the minimal CPU environment does not install
NumPy:

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

The official fitter completed with exit code `0`; no fitting error occurred.

## Rotation validation

Validator:

```text
D:\AI\voidinfer-adaptive-dflash2\tools\oscar\validate_rotations.py
```

Command:

```powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\validate_rotations.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py'
```

The validator loaded each checkpoint twice and required exact tensor equality between loads. It
also required the exact full-attention layer set, `layer_id` mapping, finite FP32 rotation and
eigenvalue tensors, rotation shape `[256,256]`, eigenvalue shape `[256]`, and no missing K/V layer.

### Orthogonality

Values are `max_abs(R @ R.T - I)` after reloading the serialized FP32 checkpoint, computed in
float64. No anomalous layer was found.

| Layer | K (`qqt`) | V (`sst`) |
|---:|---:|---:|
| 3 | 1.616284218e-08 | 1.582423281e-08 |
| 7 | 1.402900152e-08 | 1.261104021e-08 |
| 11 | 1.820284234e-08 | 1.348969847e-08 |
| 15 | 1.750568290e-08 | 1.771485136e-08 |
| 19 | 1.575594588e-08 | 1.674784422e-08 |
| 23 | 1.346792233e-08 | 1.916613712e-08 |
| 27 | 1.691309981e-08 | 1.709784025e-08 |
| 31 | 2.011652722e-08 | 1.501585079e-08 |
| 35 | 1.565807661e-08 | 1.393516325e-08 |
| 39 | 1.974953556e-08 | 1.860080756e-08 |
| 43 | 1.734229971e-08 | 1.703272956e-08 |
| 47 | 1.826533968e-08 | 1.611366995e-08 |
| 51 | 1.759121537e-08 | 1.604982236e-08 |
| 55 | 1.722849107e-08 | 1.732011445e-08 |
| 59 | 1.773900493e-08 | 1.678267680e-08 |
| 63 | 1.706497410e-08 | 1.696254048e-08 |
| **max** | **2.011652722e-08** | **1.916613712e-08** |

Validation summary:

```text
OSCAR rotations: PASS layers=16 head_dim=256
K_sha256=a5d424c835d82d055ae26b8001581b2e5389d825a4a01bd47ad1fbe12b5981fa
V_sha256=cccf5eb11910034ac05b0716ddc98d3ed6c946cfc7a2e917e0ddaf7a8041abd7
```

Full machine-readable validation is retained in:

```text
D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr\rotation_validation.json
```

Its SHA-256 is `09c5c5eb917ef7d2faa1fe202ce1093319ad508bc36a7b6d9d63130af74f4718`.

### Independent bit-reversal/Hadamard fixture

The validator used a deterministic independent 8x8 fixture with explicit Sylvester Hadamard and
explicit descending-eigenvalue bit-reversal construction. It compared that expected
`rotation @ hadamard @ pbr` result with the official fitter's `r_h_pbr` composition:

```text
dimension=8
composition=r_h_pbr
composition_max_abs_error=2.220446049250313e-16
pbr_orthogonality_max_abs_error=0.0
```

The first validator version incorrectly required exact zero for the floating-point composition;
it failed closed at `2.22e-16`. The assertion was corrected to a strict `1e-12` tolerance and the
rerun passed. No official fitter code or mathematical operation was changed.

## Files and remaining boundary

- Converted dump: `results/oscar/dumps/phase-b2-256-official`
- Conversion manifest: `results/oscar/dumps/phase-b2-256-official/conversion_manifest.json`
- K asset: `results/oscar/rotations/phase-b3-qqt-sst-r-h-pbr/k_rotation_qqt_r_h_pbr.pt`
- V asset: `results/oscar/rotations/phase-b3-qqt-sst-r-h-pbr/v_rotation_sst_r_h_pbr.pt`
- Validation report: `results/oscar/rotations/phase-b3-qqt-sst-r-h-pbr/rotation_validation.json`

The 256-token smoke data and resulting rotations are not calibration-quality production assets.
The next phase may collect 10K/30K data and qualify production clipping/assets. These rotation
files are not loaded by the VoidInfer CUDA runtime in this phase.
