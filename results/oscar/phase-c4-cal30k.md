# OSCAR Phase C4 — 30K primary calibration assets

Recorded: 2026-09-01 (Europe/Berlin)

## Result

**PASS: a validated, versioned 30K OSCAR asset set now exists for the next runtime-integration phase.**

The immutable asset identity is:

    qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1

The quality comparison against the C3 10K assets is **mixed but favorable on aggregate attention-output error**:

- output mean absolute error improved from 0.2958114424 to 0.2940526709;
- output relative L2 improved from 0.3057045925 to 0.3025147417;
- worst attention-output maximum was effectively matched, changing from 39.49778622 to 39.59547076 (+0.25%);
- score maximum, softmax maximum, and attention-argmax agreement regressed relative to the 10K assets.

This is a calibration-asset qualification, not a production-quality claim.
The next phase may integrate and qualify these assets at the intended
group-128 runtime boundary. No runtime integration, CUDA work, DFlash/MTP work,
or adaptive-K work was performed in C4.

## Preconditions and topology

The explicit Phase C3 decision was KEEP for the next 30K calibration step.
The required earlier gates were PASS:

| Gate | Evidence |
|---|---|
| B2 QKV smoke capture | results/oscar/phase-b2-qkv-smoke-capture.md |
| B3 official fitter smoke | results/oscar/phase-b3-fitter-smoke.md |
| C1 rotated-BF16 invariance | results/oscar/phase-c1-rotation-invariance.md |
| C2 calibrated INT2 reference | results/oscar/phase-c2-int2-reference.md |
| C3 10K pilot | results/oscar/phase-c3-cal10k.md, explicit KEEP |

The verified Qwen3.8-27B topology was unchanged:

| Property | Value |
|---|---:|
| Total layers | 64 |
| Full-attention layers | 16 |
| GDN layers | 48 |
| Query heads / KV heads | 24 / 4 |
| GQA ratio | 6 |
| Head dimension | 256 |
| Rotary dimension | 64 |
| Q-to-KV map | q_head // 6 |

Full-attention layers, 0-based:

    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63

GDN layers, excluded from capture and fitting:

    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45,
    46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62

The unchanged capture boundary was:

    Q/K projection -> Q/K RMSNorm -> in-place RoPE -> capture qn/kn/v
    -> causal attention and cache append

Q and K are post-normalization and post-RoPE; V is the corresponding projected
BF16 tensor without RMSNorm or RoPE. Capture occurs immediately before the
attention/cache wrapper at:

    src/targets/qwen3_6/impl/runtime/text_context_impl.h:861-867

Serialized tensors retain [tokens, heads, head_dim] order:

    Q [256, 24, 256], K [256, 4, 256], V [256, 4, 256]

No GDN recurrent state or DFlash2 drafter KV was captured.

## Model and capture provenance

| Item | Value |
|---|---|
| Model artifact | D:/AI/voidinfer/models/Qwen3.8-27B-NVFP4-DFlash2-NInfer/qwen3_8_27b_nvfp4.ninfer |
| Model SHA-256 | 6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e |
| Latest build | D:/AI/build-adaptive-dflash2 |
| Capture executable SHA-256 | 8f79d0c0be7a7a7d5a72749dcd1446f42a64b8c8016844150225d19950da40b4 |
| Calibration environment | D:/AI/tools/oscar-calibration/.venv |
| Python / PyTorch | 3.12.10 / 2.13.0+cpu |
| Runtime group-size target | 128 |

The capture source identity recorded in the manifest was:

| Source | SHA-256 |
|---|---|
| src/targets/qwen3_6/impl/runtime/text_context_impl.h | 1c57f26989556549421655f62dea77e5c67d0df28f6116af658697466b3e678e |
| src/targets/qwen3_6/impl/frontend/frontend.cpp | 392464d0fc60e0ba0bd17f92960482f21d849229d7afcddc593fa22d2a318b69 |
| src/targets/qwen3_6/impl/runtime/oscar_qkv_capture.h | ab38c99c011a5f0d996658daa0fe90cd2845b215acfceec876a35988c0b3d49e |
| src/targets/qwen3_6/impl/runtime/oscar_qkv_capture.cpp | 4e18232d7b40d0cc97fbe751df9b1649bf158e69f06f16bd99530338cce9c345 |
| tests/targets/qwen3_6_27b/test_oscar_capture.cpp | 6c238fb737b82f0bb5c9a42b7b4041615488c7b5bb123f8b401fc8113f5b11be |
| tools/oscar/capture_qkv_calibration.ps1 | cd5a7edc5a0b11f8bffe92dfea645bfb0fa70d0f6f421d3132feb73592d06110 |
| tools/oscar/validate_calibration_dump.py | 52118eb1d2e85c9498bd497b537bff2a2221c5e6386cfc523453d0e60b8bd949 |

## 30K calibration capture

The deterministic independent-request formula was:

    token_id = 198 + ((30017 + request_index*7919 + position*131) mod 4096)

Exact command:

~~~powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File '.\tools\oscar\capture_qkv_calibration.ps1' -OutputDirectory 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1' -Requests 120 -PromptTokens 256 -FormulaSeed 30017 -InputDescription 'deterministic_request_formula_v1_cal30k_seed30017'
~~~

Capture validation passed:

    OSCAR calibration QKV dump: PASS layers=16 chunks=120
    chunk_tokens=256 useful_tokens=30720 dump_bytes=8053063680

| Artifact | Value |
|---|---|
| Raw capture | results/oscar/captures/qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1 |
| Raw manifest | .../manifest.json |
| Raw manifest SHA-256 | 8a5f6a6f89f2ab21d178f7644297365012bc6dcebb03388d514ddf12e0b59bd8 |
| Useful tokens | 30,720 |
| Chunks | 120 × 256 |
| Raw payload bytes | 8,053,063,680 |
| Tensor records | 5,760 |
| Input description | deterministic_request_formula_v1_cal30k_seed30017 |

The fail-closed validator checked exact layer/chunk/type coverage, no GDN
layers, Q/K/V dimensions, GQA mapping, capture-stage uniformity, BF16 finite
values, exact payload hashes and sizes, tensor-record provenance, and no
unlisted payload files. Representative numerical values from layers 3, 35,
and 63 at the first and last chunks were retained in the native validator
output and manifest.

## Held-out capture

Held-out data used a separate deterministic seed and was not passed to the fitter.

~~~powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File '.\tools\oscar\capture_qkv_calibration.ps1' -OutputDirectory 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout-1024' -Requests 4 -PromptTokens 256 -FormulaSeed 40019 -InputDescription 'deterministic_request_formula_v1_cal30k_heldout_seed40019'
~~~

| Artifact | Value |
|---|---|
| Held-out useful tokens | 1,024 |
| Held-out raw manifest SHA-256 | 605efbc46d60c50eeb4da5682a1ab03f7e065904df25d05c843f969b268a34b3 |
| Held-out raw payload bytes | 268,435,456 |
| Held-out conversion manifest SHA-256 | 7b7115c03cdab3abeb37ff21d3aa6ed595723637b5bf765691466edf59bb351a |

The held-out set passed the same fail-closed validation before conversion.

## Official conversion

The converter first validated each raw manifest and sidecar, then preserved
exact BF16 uint16 bit patterns through .pt save/reload. Raw chunks 0..119 were
mapped to official chunks 1..120; therefore upstream --chunk-id all discarded
no useful data.

~~~powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_calibration_dump_to_official_pt.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\manifest.json' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-official' --expected-useful-tokens 30720 --expected-chunks 120

& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_calibration_dump_to_official_pt.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout-1024\manifest.json' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout-1024-official' --expected-useful-tokens 1024 --expected-chunks 4
~~~

| Conversion artifact | SHA-256 | Contents |
|---|---|---|
| 30K conversion_manifest.json | 2c0a200a64b8825aef36bb14c958b7bb9bb82912f00d54692fff9d7ca0e6ba76 | 5,760 .pt tensors; 8,061,249,216 payload bytes |
| Held-out conversion_manifest.json | 7b7115c03cdab3abeb37ff21d3aa6ed595723637b5bf765691466edf59bb351a | 192 .pt tensors; 268,693,312 payload bytes |

All converted tensors reloaded with shapes Q [256,24,256], K/V [256,4,256],
dtype torch.bfloat16, finite values, and exact source BF16 bit patterns.

## Official fitter and asset identity

Official FutureMLS-Lab OSCAR source:

| Item | Value |
|---|---|
| Repository / commit | FutureMLS-Lab/OSCAR, 41ebcdba3db5f0ce1339c3727caea80df575d437 |
| Local source | tools/oscar/upstream_compute_kv_rotation.py |
| Source SHA-256 | f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b |
| Method | qqt_sst |
| Composition | r_h_pbr |
| Head dimension | 256 |
| Chunk selection | all, official chunks 1..120 |
| Fitter result | exit code 0; 85.181 seconds |

Exact fitter command:

~~~powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; & 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --dump-path 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-official' --output-dir 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1' --chunk-id all --head-dim 256 --method qqt_sst --composition r_h_pbr
~~~

The only fitter warning was the known non-fatal missing-NumPy warning in
the minimal CPU environment. No upstream fitter code or mathematics was modified.

Immutable asset directory:

    results/oscar/rotations/qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1

| Asset | Dtype / shape | Bytes | SHA-256 |
|---|---|---:|---|
| k_rotation_qqt_r_h_pbr.pt | FP32; 16 layer entries of [256,256] | 4,220,861 | f2b97b27bb3c453e4e01f1303c9ab21ef96bcf595b48569ad5cbfb2799cfda08 |
| v_rotation_sst_r_h_pbr.pt | FP32; 16 layer entries of [256,256] | 4,220,861 | 516bb00b45ba37dadc311b20de3620b0b0e792a3e6a5b76951ef4d24c87036da |
| asset_manifest.json | JSON identity/provenance manifest | 4,869 | 4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560 |
| rotation_validation.json | JSON validation report | 4,420 | 78822c39f0f09bd5fd4618a103e702c2715a578fd9a289d0be296b6cab96407a |

The independent asset manifest records the immutable identity, model/source
identity, capture/conversion hashes, fitter command, asset hashes, target
group size, and held-out evaluation hash.

## Rotation validation

Command:

~~~powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\validate_rotations.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py'
~~~

Result: **PASS**. All 16 exact full-attention layers had finite FP32
[256,256] rotations, deterministic reload, exact layer mapping, and valid
objectives. Every row below passed.

| Layer | K max_abs(R @ R.T - I) | V max_abs(R @ R.T - I) | Result |
|---:|---:|---:|:---:|
| 3 | 1.666530425e-08 | 1.570461827e-08 | PASS |
| 7 | 1.661069726e-08 | 1.568449859e-08 | PASS |
| 11 | 1.720332987e-08 | 1.701376795e-08 | PASS |
| 15 | 1.613477285e-08 | 2.196368865e-08 | PASS |
| 19 | 1.908608227e-08 | 2.296734802e-08 | PASS |
| 23 | 1.627466739e-08 | 1.554131113e-08 | PASS |
| 27 | 1.411982598e-08 | 1.524404081e-08 | PASS |
| 31 | 1.867809640e-08 | 1.408694872e-08 | PASS |
| 35 | 1.347184275e-08 | 1.803291172e-08 | PASS |
| 39 | 1.452860832e-08 | 1.917661296e-08 | PASS |
| 43 | 1.526155358e-08 | 1.661501348e-08 | PASS |
| 47 | 1.904629743e-08 | 1.578644038e-08 | PASS |
| 51 | 1.734720390e-08 | 1.642519176e-08 | PASS |
| 55 | 2.082505102e-08 | 1.591893439e-08 | PASS |
| 59 | 1.767359015e-08 | 1.609651512e-08 | PASS |
| 63 | 1.803076588e-08 | 1.872450328e-08 | PASS |
| max | 2.082505102e-08 | 2.296734802e-08 | PASS |

The independent deterministic r_h_pbr fixture passed with maximum absolute
composition error 2.2e-16.

## Held-out evaluation

The slow C4 reference evaluator is:

    tools/oscar/evaluate_cal30k_heldout.py

Its SHA-256 is 23ab5a3862a103ab6091b08323a70eebd04e9dd8d391780546440282b2bf3d4f.
It reuses the validated C3 evaluator primitives from
tools/oscar/evaluate_cal10k_heldout.py, SHA-256
66113be1e0766f6039f5a7f05cef811eeb6f84b23a1d52b4e430c7b7d9b403fc.

It revalidated the held-out conversion manifest, all 192 .pt hashes and shapes,
both rotation sets, and the pinned upstream quantizer before computing causal GQA
attention. Each held-out chunk was evaluated as an independent 256-token request.
The BF16 reference is the original-coordinate attention; rotated outputs use the
required R_V.T recovery. Layer-output/LM-logit metrics remain unavailable because
the QKV artifact does not include output projection or residual activations.

Exact command:

~~~powershell
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\evaluate_cal30k_heldout.py' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout-1024-official' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1' 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-c3-cal10k-qqt-sst-r-h-pbr' --upstream 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --report-json 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\evaluations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout.json'
~~~

Machine report SHA-256:

    6ef7ca996516473871efa7b1de1354ac7f0666e0bde43c67f4192b59dce18c98

### Aggregate metrics

The BF16 row is the zero-error reference by definition.

| Path | Score max abs | Softmax max abs | Attention-output max abs | Output mean abs | Output relative L2 | Attention argmax |
|---|---:|---:|---:|---:|---:|---:|
| Original BF16 reference | 0 | 0 | 0 | 0 | 0 | 100.00% |
| Rotated BF16, 30K assets | 4.309937e-07 | 8.126786e-08 | 3.746161e-06 | 3.387338e-08 | 3.777895e-08 | 100.00% |
| Calibrated OSCAR INT2, 30K assets | 7.722678 | 9.168378e-01 | 39.595471 | 0.294053 | 0.302515 | 46.91% |
| Calibrated OSCAR INT2, 10K assets | 6.932931 | 8.748651e-01 | 39.497786 | 0.295811 | 0.305705 | 49.55% |

### Per-layer metrics

Values are maximum absolute errors except relative L2 and agreement columns.

| Layer | Rot BF16 score | Rot BF16 output | 30K score | 30K softmax | 30K output | 30K rel-L2 | 30K argmax | 10K score | 10K softmax | 10K output | 10K rel-L2 | 10K argmax |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2.787e-07 | 2.054e-07 | 4.533 | 0.735 | 3.279 | 0.4215 | 43.20% | 5.013 | 0.716 | 3.627 | 0.4313 | 42.74% |
| 7 | 3.313e-07 | 4.976e-07 | 5.591 | 0.729 | 7.603 | 0.4475 | 49.96% | 5.299 | 0.815 | 7.476 | 0.4419 | 50.76% |
| 11 | 3.405e-07 | 3.221e-07 | 5.727 | 0.772 | 5.993 | 0.4409 | 57.70% | 5.057 | 0.776 | 6.098 | 0.4299 | 55.29% |
| 15 | 3.366e-07 | 2.695e-07 | 7.723 | 0.862 | 2.978 | 0.4350 | 66.17% | 5.737 | 0.791 | 3.655 | 0.4472 | 65.19% |
| 19 | 3.473e-07 | 4.923e-07 | 5.495 | 0.779 | 5.023 | 0.4738 | 57.13% | 5.748 | 0.769 | 6.117 | 0.4668 | 57.49% |
| 23 | 3.597e-07 | 7.211e-07 | 5.032 | 0.809 | 8.338 | 0.5171 | 61.07% | 5.511 | 0.808 | 8.138 | 0.4991 | 63.49% |
| 27 | 3.775e-07 | 1.606e-06 | 6.112 | 0.815 | 22.143 | 0.5196 | 50.16% | 6.340 | 0.807 | 24.483 | 0.5626 | 59.82% |
| 31 | 3.893e-07 | 2.126e-06 | 6.824 | 0.855 | 27.693 | 0.4642 | 47.68% | 6.811 | 0.872 | 27.115 | 0.5223 | 49.62% |
| 35 | 3.518e-07 | 2.055e-06 | 6.919 | 0.847 | 24.702 | 0.4983 | 41.61% | 6.429 | 0.810 | 24.878 | 0.4684 | 43.17% |
| 39 | 4.310e-07 | 1.275e-06 | 6.347 | 0.787 | 24.045 | 0.6261 | 47.27% | 5.956 | 0.851 | 19.210 | 0.5540 | 60.47% |
| 43 | 4.198e-07 | 2.842e-06 | 7.290 | 0.917 | 33.858 | 0.5259 | 44.63% | 6.933 | 0.875 | 38.387 | 0.4964 | 47.59% |
| 47 | 3.544e-07 | 3.746e-06 | 6.406 | 0.787 | 31.750 | 0.4826 | 43.40% | 5.593 | 0.803 | 39.498 | 0.5584 | 54.28% |
| 51 | 3.209e-07 | 1.860e-06 | 5.550 | 0.777 | 29.508 | 0.5157 | 35.03% | 6.253 | 0.784 | 32.041 | 0.5451 | 35.72% |
| 55 | 4.142e-07 | 5.510e-07 | 5.484 | 0.680 | 14.714 | 0.5552 | 33.09% | 5.415 | 0.754 | 14.549 | 0.5331 | 33.93% |
| 59 | 2.996e-07 | 1.048e-06 | 5.822 | 0.825 | 39.595 | 0.3006 | 29.00% | 5.207 | 0.705 | 20.904 | 0.3028 | 29.28% |
| 63 | 3.708e-07 | 1.154e-06 | 6.107 | 0.812 | 28.702 | 0.2198 | 43.41% | 6.044 | 0.831 | 20.263 | 0.2216 | 43.95% |

The largest 30K absolute output outlier is layer 59 (39.595471); the largest
10K outlier is layer 47 (39.497786). Relative-L2 outliers are layer 39 for
30K (0.6261) and layer 39 for 10K (0.5540). The per-layer table shows that
the 30K set is not uniformly better: it improves some layers and regresses
others, while improving aggregate output mean and relative L2.

## Independent asset manifest and hashes

The immutable asset manifest is:

    results/oscar/rotations/qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1/asset_manifest.json

It records the full model/source identity, dataset and conversion hashes,
official fitter commit/source/command, K/V asset hashes, orthogonality report,
target group size, and held-out evaluation hash. The complete per-tensor dump
hashes remain in the conversion manifest:

    results/oscar/dumps/qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-official/conversion_manifest.json

## Remaining qualification

- The QKV-only held-out artifact cannot measure output-projection, residual,
  full layer-output, LM-logit, or generated-token quality.
- The upstream fitter is covariance-calibrated at D=256 and does not expose a
  group-size option; group-128 is the recorded initial runtime target, not a
  runtime implementation in this phase.
- The calibration corpus is 120 independent 256-token requests, totaling
  30,720 tokens; it is not a single contiguous 30K-position context.
- The minimal CPU environment emits a known non-fatal missing-NumPy warning.

No blocker was found in capture coverage, tensor orientation, BF16 preservation,
manifest hashes, official fitting, layer mapping, rotation finiteness,
orthogonality, or held-out evaluator execution. The C4 asset set is therefore
ready for the next runtime-integration phase. This phase stops here.
