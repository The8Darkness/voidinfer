# OSCAR Phase D1.1 — rotation precision contract

**Decision: BLOCKED for D2.** The rotation-only mathematics is correct at FP32 precision, but no
current real-runtime variant satisfies the full-model 32-token equivalence gate. The evidence
isolates the required contract and the remaining blocker; no INT2, recalibration, DFlash2/MTP,
adaptive-K, weight baking, or general kernel optimization was performed.

## Scope and immutable inputs

This pass used the deterministic 32-token prompt already used by D1:

```text
prompt[i] = 198 + ((i * 131) mod 4096), i = 0..31
```

The model and C4 assets were unchanged:

| Item | Value |
| --- | --- |
| Model | `qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| Asset identity | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Asset manifest SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| K bank SHA-256 | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` |
| V bank SHA-256 | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |
| Topology | 64 layers; full layers `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; GDN complement 48 |
| Attention geometry | Q/K/V `[head_dim, heads, tokens]`; Q `[256,24,T]`, K/V `[256,4,T]`; GQA `q_head//6` |

## Implementation

The diagnostic implementation adds four opt-in precision modes in
`src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h:14` and dispatches them from
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:892`:

- `bf16-materialized`: the D1 control path.
- `fp32-rotation`: BF16 source Q/K/V, FP32 rotation output, scalar FP32 reference attention,
  then the existing BF16 inverse path.
- `fp32-inverse`: existing BF16 rotation/attention, FP32 inverse accumulation from BF16 output.
- `fp32-rotation+inverse`: FP32 rotation, FP32 scalar reference attention, FP32 inverse, and one
  final BF16 conversion before the existing gate/output projection.

The CUDA diagnostic kernels are in
`src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cu:42-150` and include FP32 row-right
rotation, causal GQA reference attention, BF16-to-FP32 inverse, FP32 inverse, and FP32-to-BF16
conversion. The reference attention is intentionally limited to prefill `T <= 64`; it is a
diagnostic path, not a serving implementation.

The exact post-attention tap is at
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1011`. It records the result of the
existing fused `linear_add`, so `post_attention_linear_add` means output projection plus residual;
the unfused projection tensor is not exposed by that API. The post-MLP/layer tap remains at
`text_context_impl.h:1209`. The test harness is
`tests/targets/qwen3_6_27b/test_oscar_runtime.cpp:108-212`.

Build succeeded for `ninfer_qwen3_6_27b_oscar_runtime_test` in
`D:\AI\build-adaptive-dflash2`. The executable SHA-256 was
`45d45b81f06c823c4d082c4ba2a5b7f2ba31b27c5970e2de0f8c6a719f9baff5`.

## Exact precision boundaries

| Variant | Q/K/V rotation | Attention | V inverse | First materialized rotation boundary |
| --- | --- | --- | --- | --- |
| BF16-materialized | FP32 FMA, BF16 output | existing BF16 interface/output | BF16 input, FP32 FMA, BF16 output | immediately after each rotation |
| FP32-rotation | FP32 output | FP32 diagnostic reference | FP32 attention is cast to BF16, then existing BF16 inverse | before existing BF16 interface/inverse |
| FP32-inverse | FP32 FMA, BF16 output | existing BF16 interface/output | BF16 input, FP32 FMA, FP32 output, then BF16 | immediately after each rotation |
| FP32-rotation+inverse | FP32 output | FP32 diagnostic reference/output | FP32 input, FP32 FMA, FP32 output, then one BF16 conversion | immediately before gate/output projection |

All variants start from identical post-RMSNorm, post-RoPE Q/K/V. The row-vector semantics remain
`Q'=Q@R_K`, `K'=K@R_K`, `V'=V@R_V`, followed by `O=O'@R_V.T`; no RoPE is reapplied. The normal
runtime and all GDN layers remain unchanged.

## Layer-3 boundary measurements

Values below are `(max absolute, mean absolute, relative L2)`. Q/K/V orientation errors compare
the runtime tensor with a Python double-precision row-right reference. `recovered attention` is
the normal runtime attention output versus the recovered candidate output. `post-attention` is
the fused output-projection-plus-residual tap.

| Variant | Q rotation rel L2 | K rotation rel L2 | V rotation rel L2 | QK score rel L2 | Softmax rel L2 | Recovered attention rel L2 | Post-attention rel L2 | Post-MLP rel L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16-materialized | 1.669553e-3 | 1.655263e-3 | 1.654389e-3 | 4.552080e-4 | 1.720801e-3 | 3.264583e-3 | 1.744987e-3 | 7.739243e-3 |
| FP32-rotation | 3.209392e-7 | 3.082207e-7 | 2.861411e-7 | 9.274475e-8 | 3.505521e-7 | 2.626651e-3 | 1.974898e-3 | 7.705390e-3 |
| FP32-inverse | 1.669553e-3 | 1.655263e-3 | 1.654389e-3 | 4.552080e-4 | 1.720801e-3 | 3.264583e-3 | 1.744987e-3 | 7.739243e-3 |
| FP32-rotation+inverse | 3.209392e-7 | 3.082207e-7 | 2.861411e-7 | 9.274475e-8 | 3.505521e-7 | 1.563658e-3 | 5.990686e-4 | 5.917754e-3 |

The complete layer-3 `(max absolute, mean absolute, relative L2)` triples are recorded below to
avoid hiding large-value effects behind relative-only summaries.

| Variant | Stage | Max abs | Mean abs | Relative L2 |
| --- | --- | ---: | ---: | ---: |
| BF16-materialized | rotated Q | 1.524161e-2 | 1.328089e-3 | 1.669553e-3 |
| BF16-materialized | rotated K | 1.325317e-2 | 1.332117e-3 | 1.655263e-3 |
| BF16-materialized | rotated V | 7.810749e-3 | 9.305885e-4 | 1.654389e-3 |
| BF16-materialized | QK scores | 1.481049e-2 | 2.595363e-3 | 4.552080e-4 |
| BF16-materialized | softmax | 2.177467e-3 | 1.062719e-4 | 1.720801e-3 |
| BF16-materialized | recovered attention | 1.562500e-2 | 9.618841e-4 | 3.264583e-3 |
| BF16-materialized | post-attention projection+residual | 1.250000e-1 | 2.239402e-4 | 1.744987e-3 |
| BF16-materialized | post-MLP/layer output | 1.250000e-1 | 2.271394e-3 | 7.739243e-3 |
| FP32-rotation | rotated Q | 3.758931e-6 | 2.606909e-7 | 3.209392e-7 |
| FP32-rotation | rotated K | 2.898654e-6 | 2.547000e-7 | 3.082207e-7 |
| FP32-rotation | rotated V | 2.265995e-6 | 1.666287e-7 | 2.861411e-7 |
| FP32-rotation | QK scores | 2.961189e-6 | 5.299080e-7 | 9.274475e-8 |
| FP32-rotation | softmax | 4.989892e-7 | 2.155731e-8 | 3.505521e-7 |
| FP32-rotation | recovered attention | 1.562500e-2 | 6.848914e-4 | 2.626651e-3 |
| FP32-rotation | post-attention projection+residual | 1.250000e-1 | 2.217843e-4 | 1.974898e-3 |
| FP32-rotation | post-MLP/layer output | 1.250000e-1 | 2.227557e-3 | 7.705390e-3 |
| FP32-inverse | rotated Q | 1.524161e-2 | 1.328089e-3 | 1.669553e-3 |
| FP32-inverse | rotated K | 1.325317e-2 | 1.332117e-3 | 1.655263e-3 |
| FP32-inverse | rotated V | 7.810749e-3 | 9.305885e-4 | 1.654389e-3 |
| FP32-inverse | QK scores | 1.481049e-2 | 2.595363e-3 | 4.552080e-4 |
| FP32-inverse | softmax | 2.177467e-3 | 1.062719e-4 | 1.720801e-3 |
| FP32-inverse | recovered attention | 1.562500e-2 | 9.618841e-4 | 3.264583e-3 |
| FP32-inverse | recovered FP32 attention | 1.611304e-2 | 1.130470e-3 | 2.982433e-3 |
| FP32-inverse | post-attention projection+residual | 1.250000e-1 | 2.239402e-4 | 1.744987e-3 |
| FP32-inverse | post-MLP/layer output | 1.250000e-1 | 2.271394e-3 | 7.739243e-3 |
| FP32-rotation+inverse | rotated Q | 3.758931e-6 | 2.606909e-7 | 3.209392e-7 |
| FP32-rotation+inverse | rotated K | 2.898654e-6 | 2.547000e-7 | 3.082207e-7 |
| FP32-rotation+inverse | rotated V | 2.265995e-6 | 1.666287e-7 | 2.861411e-7 |
| FP32-rotation+inverse | QK scores | 2.961189e-6 | 5.299080e-7 | 9.274475e-8 |
| FP32-rotation+inverse | softmax | 4.989892e-7 | 2.155731e-8 | 3.505521e-7 |
| FP32-rotation+inverse | recovered attention | 1.562500e-2 | 2.299016e-4 | 1.563658e-3 |
| FP32-rotation+inverse | recovered FP32 attention | 1.014996e-2 | 5.713681e-4 | 1.749785e-3 |
| FP32-rotation+inverse | post-attention projection+residual | 1.562500e-2 | 8.945317e-5 | 5.990686e-4 |
| FP32-rotation+inverse | post-MLP/layer output | 1.250000e-1 | 1.711132e-3 | 5.917754e-3 |

The raw attention tensor before inverse is expected to differ because it is in the rotated V
coordinate system. Its layer-3 relative L2 against the normal raw attention was `1.429441` for
BF16-materialized and `1.429464` for both FP32-rotation variants; this is not an invariance
failure. The corresponding recovered tensor is the meaningful comparison.

## Independent reference separation

The normal runtime's BF16 attention kernel differs from the Python double-precision attention
reference by relative L2 `1.749825e-3` at layer 3. This is a control-path numerical difference,
not an OSCAR rotation error.

For `fp32-rotation+inverse`:

- CUDA FP32 reference attention versus Python rotated attention: relative L2 `9.881427e-7`.
- Python FP32 rotated attention plus inverse versus Python normal attention: relative L2
  `1.091371e-6`.
- Runtime FP32 recovered attention versus the normal runtime: relative L2 `1.749785e-3`, which
  tracks the normal runtime's own BF16-versus-Python reference difference.

This proves that FP32 rotation and inverse remove the original rotation-only Q/K score error, but
the current full-model comparison still mixes a diagnostic FP32 attention order with the normal
production BF16 attention order. The custom high-precision path also does not carry an FP32 rotated
cache into the generated-token decode; decode remains on the existing BF16 cache path.

## Full-model 32-token comparison

Exact command:

```powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS = 'D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256 = '6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d1-1-runtime-diagnostics'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe'
```

The harness ran only `T=32`, generated one greedy token, and loaded identical NVFP4 weights for
the normal and candidate runs.

| Variant | Layer-3 rel L2 | Final logit rel L2 | Top1 | Generated token | Verdict |
| --- | ---: | ---: | --- | --- | --- |
| BF16-materialized | 7.739243e-3 | 2.041360e-2 | FAIL | FAIL | FAIL |
| FP32-rotation | 7.705390e-3 | 2.120393e-2 | PASS | PASS | FAIL |
| FP32-inverse | 7.739243e-3 | 2.041360e-2 | FAIL | FAIL | FAIL |
| FP32-rotation+inverse | 5.917754e-3 | 1.693360e-2 | FAIL | FAIL | FAIL |

Final hidden-state results were, respectively, relative L2 `5.981458e-2`, `5.000601e-2`,
`5.981458e-2`, and `4.570910e-2`. Top-10 overlaps were `8/10`, `9/10`, `8/10`, and `9/10`.
All variants first differed at model layer 3. The best variant was still materially different at
the full-model boundary, so it cannot be promoted to a production precision contract yet.

Complete full-model error metrics were:

| Variant | Hidden max / mean / rel L2 | Logit max / mean / rel L2 | Top1 | Top-10 | Generated token |
| --- | --- | --- | --- | ---: | --- |
| BF16-materialized | `1.218750 / .0738913 / .0598146` | `.605469 / .0761244 / .0204136` | FAIL | 8/10 | FAIL |
| FP32-rotation | `.687500 / .0649027 / .0500060` | `.500000 / .0816937 / .0212039` | PASS | 9/10 | PASS |
| FP32-inverse | `1.218750 / .0738913 / .0598146` | `.605469 / .0761244 / .0204136` | FAIL | 8/10 | FAIL |
| FP32-rotation+inverse | `1.062500 / .0578188 / .0457091` | `.453125 / .0636676 / .0169336` | FAIL | 9/10 | FAIL |

Across all 16 full-attention layers, the post-MLP relative L2 for the best combined variant was:

| Layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32-rotation+inverse | .005918 | .036090 | .040406 | .046657 | .077459 | .093267 | .098311 | .107575 | .123346 | .118202 | .117931 | .120446 | .129175 | .118876 | .109305 | .058695 |

## Architecture decision

Option A is the only currently justified production direction:

- Q/K: rotate in FP32 or fused accumulator precision immediately before the future quantizer/cache
  operation; do not store an intermediate rotated-BF16 Q/K.
- V: rotate in FP32 and keep the attention accumulator in FP32 through `R_V.T`; perform only the
  final conversion required by the downstream gate/output interface.
- A matched attention/reference path must be used for prefill and decode. The current diagnostic
  scalar attention is prefill-only and therefore does not close the generated-token gate.

Option B is algebraically possible for V because row-vector attention permits
`(O'@R_V.T)@W_O = O'@(R_V.T@W_O)`, but applying that transformation would require an on-the-fly
or requantized NVFP4 output projection. It was not implemented and is not a safe way to remove
the present precision confound. Q/K cannot be treated as ordinary pre-RoPE projection baking
because the fitted contract is post-normalization and post-RoPE.

## Remaining blocker and stop condition

The exact remaining blocker is a matched real-runtime high-precision attention/cache contract:
the FP32 reference proves the rotation/inverse algebra, while the real model still compares a
custom FP32 attention order and a normal BF16 attention order, and decode still consumes the BF16
rotated cache. Until an implementation preserves the FP32 rotation/inverse contract through the
actual attention and prefill-to-decode cache path and passes the full-model gate, D2 must not start.

**D1.1 result: BLOCKED.**
