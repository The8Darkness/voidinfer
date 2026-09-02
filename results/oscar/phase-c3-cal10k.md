# OSCAR Phase C3 — 10K calibration pilot

Recorded: 2026-09-01 (Europe/Berlin)

## Decision

KEEP for the next 30K calibration step.

The 10K pilot produced a complete, hash-validated calibration set for every
verified full-attention layer, the unchanged official fitter produced valid K/V
assets, and the held-out rotated-BF16 path remained invariant. The 10K
calibrated INT2 reference improved the aggregate held-out attention-output error
over the previous 256-token assets. This KEEP decision is a calibration-quality
pilot gate only; it is not approval for runtime integration or a production
quality claim.

The largest calibrated 10K held-out attention-output maximum was layer 47
(37.105069), followed by layer 43 (33.425509) and layer 51 (29.855600).
These are 2-bit reference quantization outliers, not rotation, capture, layer
mapping, or finite-value failures.

## Preconditions

All required earlier gates were read and were PASS:

| Gate | Evidence |
|---|---|
| B2 QKV smoke capture | results/oscar/phase-b2-qkv-smoke-capture.md |
| B3 official fitter smoke | results/oscar/phase-b3-fitter-smoke.md |
| C1 rotated-BF16 invariance | results/oscar/phase-c1-rotation-invariance.md |
| C2 calibrated INT2 reference | results/oscar/phase-c2-int2-reference.md |

The authoritative topology remains 64 layers, full attention at
3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63, 48 GDN layers, Q/KV heads
24/4, GQA ratio 6, head dimension 256, and rotary dimension 64. No GDN
recurrent state was captured or interpreted as KV.

## Provenance and implementation

The designated latest build was D:\AI\build-adaptive-dflash2. The model artifact
was D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer
with SHA-256
6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e.

The already validated capture boundary was unchanged:

    q/k projection -> Q/K RMSNorm -> in-place RoPE -> capture qn/kn/v ->
    causal attention and cache append

The native capture continues to capture only post-normalization, post-RoPE Q/K
and projected, unnormalized/unrotated V immediately before attention/cache
append. The source boundary is
D:\AI\voidinfer-adaptive-dflash2\src\targets\qwen3_6\impl\runtime\text_context_impl.h:853-889,
with the actual capture call at lines 861-867.

The implementation changes for this pilot were:

- Parameterized the existing test defaults, without changing B2 behavior, to
  run deterministic independent requests of 256 prompt tokens. Forty requests
  provide 10,240 useful tokens; four separate requests provide 1,024 held-out
  tokens.
- Raised the opt-in capture budget ceiling from 4,096 to 1,048,576 tokens and
  made the manifest input description explicit. The Q/K/V capture class and
  boundary were not changed.
- Added the fail-closed multi-chunk validator
  tools/oscar/validate_calibration_dump.py.
- Added the exact-BF16 converter
  tools/oscar/convert_calibration_dump_to_official_pt.py.
- Added tools/oscar/capture_qkv_calibration.ps1 and
  tools/oscar/evaluate_cal10k_heldout.py.
- Added an opt-in test-only output-session mode controlled by
  NINFER_OSCAR_QKV_CAPTURE_ONLY=1. It validates generated token IDs but skips
  presentation decoding so a standalone byte-level output token cannot abort
  a completed prefill capture. It is not a production mode, does not change
  Q/K/V tensors, and was included in the executable source identity.

The source identity recorded in the calibration manifest is:

    src\targets\qwen3_6\impl\runtime\text_context_impl.h=1c57f26989556549421655f62dea77e5c67d0df28f6116af658697466b3e678e;
    src\targets\qwen3_6\impl\frontend\frontend.cpp=392464d0fc60e0ba0bd17f92960482f21d849229d7afcddc593fa22d2a318b69;
    src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.h=ab38c99c011a5f0d996658daa0fe90cd2845b215acfceec876a35988c0b3d49e;
    src\targets\qwen3_6\impl\runtime\oscar_qkv_capture.cpp=4e18232d7b40d0cc97fbe751df9b1649bf158e69f06f16bd99530338cce9c345;
    tests\targets\qwen3_6_27b\test_oscar_capture.cpp=6c238fb737b82f0bb5c9a42b7b4041615488c7b5bb123f8b401fc8113f5b11be;
    tools\oscar\capture_qkv_calibration.ps1=cd5a7edc5a0b11f8bffe92dfea645bfb0fa70d0f6f421d3132feb73592d06110;
    tools\oscar\validate_calibration_dump.py=95a00dcf8f9b795c70e8cba51da8e6ec542f6ddd8ee2187b602d4fb2c8dce017

## Calibration capture

Exact command:

    & 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File .\tools\oscar\capture_qkv_calibration.ps1 -OutputDirectory D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-c3-cal10k -Requests 40 -PromptTokens 256 -FormulaSeed 17 -InputDescription deterministic_request_formula_v1_calibration_seed17

The prompt formula is deterministic and independent-request based:

    token_id = 198 + ((17 + request_index*7919 + position*131) mod 4096)

The capture generated 40 chunks of 256 prompt tokens, exactly 10,240 useful
tokens. Each of the 16 full-attention layers produced Q, K, and V for every
chunk. Q is [256,24,256], K is [256,4,256], and V is [256,4,256], in
[tokens, heads, head_dim] order with BF16 dimension-fastest storage.

The raw manifest is
results/oscar/captures/phase-c3-cal10k/manifest.json.
Its SHA-256 is
4083f9c7c62e41dacab1ce55322362502d6a33b416eb660e4339413d3a074d6e.
The manifest sidecar was checked. Raw payload bytes are 2,684,354,560.
The fail-closed validator checked exact full-layer/chunk coverage, exact Q/K/V
dimensions, GQA mapping, stage uniformity, payload sizes and hashes, finite
BF16 values, representative values, tensor-record agreement, and no unlisted
or GDN payload files.

Representative validated raw values:

| Data | Q first four | K first four | V first four |
|---|---|---|---|
| layer 3, chunk 0 | 0.94921875, -0.2578125, -1.2265625, -2.125 | 1.609375, 1.046875, -0.7734375, -0.91015625 | -0.80078125, 0.236328125, 0.1259765625, 1.3125 |
| layer 35, chunk 0 | 0.81640625, 0.03271484375, 0.07177734375, -0.48828125 | 0.0151977539, -0.0261230469, -0.0622558594, 0.0039672852 | -0.095703125, 0.79296875, 0.283203125, 0.1259765625 |
| layer 63, chunk 39 | -0.3046875, -0.765625, -0.59765625, 0.16015625 | -0.1953125, -0.0212402344, 0.087890625, -0.259765625 | 1.171875, -0.283203125, -0.3828125, -0.53125 |

Held-out data was captured separately with four requests, seed 1009, and
input description deterministic_request_formula_v1_heldout_seed1009. It
contains 1,024 tokens and was not passed to the fitter.

Held-out raw manifest:
results/oscar/captures/phase-c3-heldout-1024/manifest.json

Held-out manifest SHA-256:
646f81413d41e151192798ac32b36a57180a7d7463157655fcc4fb28d30cc361

Held-out raw payload bytes: 268,435,456.

## Official conversion and fitter

Exact conversion commands:

    & D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_calibration_dump_to_official_pt.py D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-c3-cal10k\manifest.json D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-c3-cal10k-official --expected-useful-tokens 10240 --expected-chunks 40

    & D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe D:\AI\voidinfer-adaptive-dflash2\tools\oscar\convert_calibration_dump_to_official_pt.py D:\AI\voidinfer-adaptive-dflash2\results\oscar\captures\phase-c3-heldout-1024\manifest.json D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-c3-heldout-1024-official --expected-useful-tokens 1024 --expected-chunks 4

The official dump hierarchy is:

    results/oscar/dumps/phase-c3-cal10k-official/layer_<id>/{q,k,v}/<chunk>.pt

The converter emitted 1,920 calibration .pt tensors and 192 held-out .pt
tensors. It reloaded every file, checked shape/dtype/finite values, and
compared BF16 uint16 bit patterns exactly. Raw chunk c is mapped to official
chunk c+1, so the official --chunk-id all loader skips no useful data.

Calibration conversion manifest SHA-256:
fca80369249fa67a74e7715385fe37dc9638e742ddd7f9be3b495c07647e79f5

Held-out conversion manifest SHA-256:
106884bbfe2391896602dbbc8a3faa561cca8f7e1453d3470f96d374785b3a39

Official fitter provenance:

| Item | Value |
|---|---|
| Source | D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py |
| FutureMLS-Lab/OSCAR commit | 41ebcdba3db5f0ce1339c3727caea80df575d437 |
| Local source SHA-256 | f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b |
| Method | qqt_sst |
| Composition | r_h_pbr |
| Head dimension | 256 |
| Chunk selection | all; official chunks 1..40 |
| Intended runtime group size | 128 |

Exact fitter command:

    $env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; & D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py --dump-path D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-c3-cal10k-official --output-dir D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-c3-cal10k-qqt-sst-r-h-pbr --chunk-id all --head-dim 256 --method qqt_sst --composition r_h_pbr

The fitter completed with exit code 0 in approximately 14.7 seconds. It
reported 16 layers and per-layer K(qqt)/V(sst) numerical residuals in the
approximately 2.6e-15 to 4.4e-15 range. The only warning was the known
CPU-environment warning that NumPy is not installed; no fitting error occurred.

Assets:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| k_rotation_qqt_r_h_pbr.pt | 4,220,861 | 322b6d56a51f3948bd3fae6d0650bd916c47b3b62bb5cee73c1d5173644582dc |
| v_rotation_sst_r_h_pbr.pt | 4,220,861 | 26a1471d90788c528bb0d8ad3ea3a8e697f841fd220b6b5334083bdbc7f3b379 |
| rotation_validation.json | 4,380 | 7426bcd5fd34cd396d5fa2f225d590910e6c213fb3291295e0472478a6f231e9 |

Rotation validation passed for every exact full-attention layer: FP32
[256,256] tensors, finite values, deterministic reload, and layer mapping.
The worst max_abs(R @ R.T - I) was 2.193682191e-08 for K and
2.131072918e-08 for V. The independent r_h_pbr fixture passed with
2.2e-16. No anomalous rotation layer was found.

The official upstream quantizer has no group-size argument and its pinned
simulate_int2_asym operation is per [token, KV-head, 256] row. Group-128 is
recorded as the intended first runtime target in the manifests, but no runtime
group-128 packing or CUDA behavior was introduced in this pilot.

## Held-out evaluation

Exact evaluator command:

    & D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe D:\AI\voidinfer-adaptive-dflash2\tools\oscar\evaluate_cal10k_heldout.py D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-c3-heldout-1024-official D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-c3-cal10k-qqt-sst-r-h-pbr D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr --upstream D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py --report-json D:\AI\voidinfer-adaptive-dflash2\results\oscar\evaluations\phase-c3-cal10k-heldout.json

The evaluator checked all held-out .pt hashes, shapes, finite values, exact
layer/chunk/type coverage, both rotation checkpoints, and the pinned upstream
quantizer before evaluating. Each held-out chunk was evaluated as its own
causal 256-token request, so attention was not incorrectly allowed to cross
request boundaries. Diagnostic products used FP64 from exact BF16 source
values. V output was recovered with R_V.T.

Machine report:
results/oscar/evaluations/phase-c3-cal10k-heldout.json

Machine report SHA-256:
c4281712d3c66c1fcf6a279f40846102a87e660fc014965fc4a27a7c15c72886

### Aggregate metrics

The BF16 baseline is the reference. Rotated-BF16 means Q/K/V use the 10K
rotations, with inverse V output recovery and no quantization. The two INT2
paths use upstream simulate_int2_asym on rotated K/V.

| Path | Score max abs | Softmax max abs | Attention output max abs | Output relative L2 | Attention argmax agreement |
|---|---:|---:|---:|---:|---:|
| Rotated BF16, 10K assets | 4.062135e-7 | 8.728284e-8 | 3.013387e-6 | 3.759458e-8 | 100.00% |
| Calibrated OSCAR INT2, 10K assets | 6.777395 | 8.739543e-1 | 37.105069 | 3.030869e-1 | 51.02% |
| Calibrated OSCAR INT2, previous 256 assets | 7.202456 | 8.724983e-1 | 41.614878 | 3.171661e-1 | 51.07% |

The 10K assets therefore preserve the C1 mathematical convention on unseen
tokens. Their INT2 output error is material, as expected at two bits, but the
10K fit improves the held-out aggregate output maximum, mean absolute error,
and relative L2 over the previous 256-token fit. This is enough to KEEP the
calibration line for 30K data collection, not enough to claim runtime quality.

### Per-layer metrics

Values are maximum absolute errors except relative L2 and agreement columns.

| Layer | Rot BF16 score | Rot BF16 output | 10K score | 10K softmax | 10K output | 10K output rel L2 | 10K argmax | Prior 256 output | Prior 256 rel L2 | Prior 256 argmax |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 2.900e-7 | 2.055e-7 | 4.198 | 7.039e-1 | 3.672 | .4225 | 41.80% | 4.869 | .4297 | 42.31% |
| 7 | 3.401e-7 | 4.299e-7 | 5.177 | 7.814e-1 | 8.200 | .4487 | 51.71% | 5.999 | .4619 | 51.48% |
| 11 | 3.320e-7 | 4.106e-7 | 5.413 | 7.639e-1 | 4.714 | .4278 | 56.83% | 4.583 | .4444 | 55.53% |
| 15 | 3.382e-7 | 2.219e-7 | 5.473 | 8.161e-1 | 4.176 | .4605 | 63.93% | 4.169 | .4657 | 67.90% |
| 19 | 3.247e-7 | 3.349e-7 | 6.084 | 8.094e-1 | 5.550 | .4674 | 57.31% | 6.422 | .4826 | 57.70% |
| 23 | 3.667e-7 | 6.482e-7 | 5.636 | 7.799e-1 | 7.909 | .4979 | 63.41% | 8.673 | .4981 | 67.79% |
| 27 | 3.767e-7 | 2.373e-6 | 6.689 | 8.092e-1 | 23.105 | .5698 | 61.24% | 21.625 | .5542 | 52.73% |
| 31 | 3.517e-7 | 2.219e-6 | 5.827 | 7.792e-1 | 24.106 | .5238 | 50.03% | 23.952 | .5025 | 53.37% |
| 35 | 3.589e-7 | 2.898e-6 | 5.389 | 8.634e-1 | 25.906 | .4600 | 42.13% | 24.300 | .4756 | 45.31% |
| 39 | 3.789e-7 | 1.273e-6 | 6.215 | 7.628e-1 | 19.642 | .5627 | 62.57% | 16.211 | .5246 | 55.61% |
| 43 | 3.877e-7 | 2.823e-6 | 6.466 | 8.740e-1 | 33.426 | .4861 | 49.10% | 32.455 | .4880 | 42.29% |
| 47 | 3.387e-7 | 3.013e-6 | 6.777 | 8.485e-1 | 37.105 | .5649 | 54.13% | 41.615 | .5234 | 49.24% |
| 51 | 3.395e-7 | 1.916e-6 | 6.200 | 8.502e-1 | 29.856 | .5478 | 33.35% | 25.996 | .5005 | 35.94% |
| 55 | 3.077e-7 | 9.915e-7 | 5.169 | 7.192e-1 | 16.041 | .5241 | 41.17% | 18.429 | .5454 | 41.53% |
| 59 | 2.913e-7 | 1.436e-6 | 5.173 | 7.361e-1 | 24.248 | .2929 | 42.74% | 23.928 | .2830 | 54.29% |
| 63 | 4.062e-7 | 1.828e-6 | 5.820 | 7.997e-1 | 20.585 | .2209 | 44.86% | 25.508 | .2578 | 44.08% |

## Unresolved qualification

- The held-out artifact lacks output-projection, residual, and LM-head
  activations, so post-layer output and generated-token/logit agreement remain
  unavailable.
- The official fitter and reference quantizer operate on D=256 rows. The
  group-128 runtime packing/metadata contract remains a later runtime
  qualification and was not implemented here.
- The 10K prompts are deterministic independent 256-token requests, not a
  long contiguous 10K-position context. This is a token-volume pilot, not a
  long-context quality result.
- No CUDA, INT2 packing, runtime asset loading, speculative decoding,
  adaptive-K, DFlash2, or 30K calibration work was performed.

The next permitted step is a separately archived 30K calibration pilot after
reviewing this KEEP decision. This phase stops here.
