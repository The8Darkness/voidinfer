# OSCAR Phase D1.2 — matched high-precision attention/cache control

**Decision: BLOCKED.** The new diagnostic path does use the same FP32 attention and persistent
FP32 cache implementation for unrotated and rotated execution through prefill and ordinary decode,
but the 32-token prefill still diverges after the required final BF16 conversion. The first
divergence is isolated to layer 3's BF16 conversion of the recovered attention output; later
NVFP4/GDN/MLP stages amplify that small difference before the decode token is chosen. Per the stop
condition, the 512-token test was not run. No INT2, recalibration, DFlash/MTP, adaptive-K, weight
baking, or performance work was performed.

## Scope and immutable inputs

The deterministic D1/D1.1 prompt was retained:

```text
prompt[i] = 198 + ((i * 131) mod 4096), i = 0..T-1
```

The C4 assets and model were unchanged:

| Item | Value |
| --- | --- |
| Model | `qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Asset manifest SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| K bank SHA-256 | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` |
| V bank SHA-256 | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |
| Topology | 64 layers; full attention `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; 48 GDN |
| Attention geometry | Q `[256,24,T]`, K/V `[256,4,T]`, GQA `q_head//6` |

## Matched diagnostic architecture

The opt-in mode is enabled with `NINFER_OSCAR_MATCHED_FP32=1`.

Control A (`matched-fp32-unrotated`) and candidate B (`matched-fp32-rotated`) both use:

- post-RMSNorm, post-RoPE source tensors;
- FP32 Q/K/V work tensors;
- the same scalar causal GQA kernel, FP32 FMA accumulation, stable `expf` softmax, and output
  accumulation order;
- the same persistent diagnostic cache object across the prefill and ordinary decode
  `TextContext` instances;
- one final FP32-to-BF16 conversion before the existing BF16 gate and output projection.

For A, FP32 Q/K/V are direct BF16-to-FP32 casts. For B, the same source tensors are transformed as
`Q'=Q@R_K`, `K'=K@R_K`, and `V'=V@R_V` in FP32; the FP32 attention result is recovered with
`R_V.T` before the common BF16 boundary.

The diagnostic cache is separate from the production `PagedKVCache` and is not a Q4/Q2 shadow:

| Property | Contract |
| --- | --- |
| Cache dtype | FP32 K and FP32 V |
| Physical layout | `[head_dim, kv_head, absolute_token]`, contiguous; separate K/V banks per full-attention layer |
| Coordinate A | Original post-RoPE K/V |
| Coordinate B | Rotated post-RoPE `K@R_K` / `V@R_V` |
| Q cache | Not stored; current Q is transformed/cast for each attention call |
| Prefill | Reset at a new base-0 prompt; append each chunk at absolute positions before causal attention |
| Decode | Append the current token at its absolute position and read the same FP32 cache; no production BF16 cache fallback in matched mode |
| Supported diagnostic concurrency | One sequence; the cache owner is the persistent text-cache object |

The cache implementation is declared in
`src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h:70`, allocated and owned in
`oscar_rotated_bf16.cpp:301`, and launched by
`oscar_rotated_bf16.cu:162-246,424-471`. The `TextContext` constructor binds the persistent cache
at `text_context_impl.h:245-251`; base-0 prefill reset is at `text_context_impl.h:1379`.

## Exact runtime insertion points

The existing verified capture boundary remains unchanged:

```text
projection -> Q/K RMSNorm -> RoPE -> candidate Q/K/V boundary
            -> matched FP32 cast/rotation -> diagnostic FP32 cache append
            -> matched FP32 causal GQA attention
            -> FP32 R_V.T recovery for B
            -> common BF16 conversion -> gate -> output projection/residual
```

Relevant source locations:

- `src/targets/qwen3_6/impl/runtime/text_context_impl.h:842-869`: projection, Q/K normalization,
  RoPE, and the unchanged diagnostics/capture boundary.
- `text_context_impl.h:887-931`: matched A/B branch; Q/K/V FP32 preparation, cache append,
  common attention, inverse-V recovery, and the common BF16 conversion.
- `text_context_impl.h:1057-1064`: existing gate and output projection; the
  `post_attention_linear_add` diagnostic tap remains after fused output-projection-plus-residual.
- `text_context_impl.h:1379`: matched cache reset for a new prefill.
- `src/targets/qwen3_6/impl/runtime/layouts_impl.h:419-443`: diagnostic FP32 workspace allowance.
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp:108-166,183-253`: D1.2 opt-in harness, two
  greedy output tokens, deterministic prompt, and A/B gate.

## Build and exact test command

The designated build target succeeded in `D:\AI\build-adaptive-dflash2` using the repository's
CUDA 13.1 and MSVC environment. The resulting executable SHA-256 is:

```text
EC3275ABF6FDA2CD3E724192ACF97405C5CED4696E80DE5513A19193F574B6E7
```

Exact 32-token command:

```powershell
$diag = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d1-2-diagnostics'
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS = 'D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
$env:NINFER_OSCAR_MODEL_SHA256 = '6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e'
$env:NINFER_OSCAR_ROTATION_ASSET_DIR = 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime'
$env:NINFER_OSCAR_RUNTIME_DIAGNOSTIC_DIR = $diag
$env:NINFER_OSCAR_D1_2_MATCHED = '1'
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe'
```

The harness stopped after the required 32-token A/B gate failed. It did not execute the optional
512-token case.

## 32-token comparison

`Layer rel L2` below is the layer-3 post-MLP/layer-output diagnostic. `Attention rel L2` is the
layer-3 recovered attention comparison; for A/B this is FP32 recovered attention, while the
production/control comparison necessarily compares the production BF16 attention result with the
matched control's final BF16 attention result.

| Comparison | Attention rel L2 | Layer rel L2 | Logit rel L2 | Top1 | Decode token | Verdict |
| ---------- | ---------------: | -----------: | -----------: | ---- | ------------ | ------- |
| production BF16 vs matched FP32 control A | `1.563633320e-3` | `5.920210440e-3` | `2.485410400e-2` | agree | `271, 2` | diagnostic control difference |
| matched FP32 control A vs matched FP32 rotated B | `1.368033032e-6` | `3.130950573e-4` | `2.329250450e-2` | **FAIL** | `271,2` vs `198,248044` | **BLOCKED** |

Prefill top-10 overlap was `10/10` for A versus B despite the top-1 mismatch. The final hidden
comparison for A versus B was max absolute `1.375`, mean absolute `8.90450388e-2`, relative L2
`7.18921733e-2`; final logits were max absolute `0.71875`, mean absolute `8.75086815e-2`,
relative L2 `2.32925045e-2`.

## Layer-3 boundary isolation

All triples are `(max absolute, mean absolute, relative L2)`. The Q/K/V rotational errors use the
unchanged C1/D1 FP32 row-right reference on the same deterministic tensor boundary. The score and
softmax values below were recomputed from the matched A/B FP32 Q/K tensors.

| Boundary | Result |
| --- | ---: |
| rotated Q reference error | `3.209392e-7` relative L2 |
| rotated K reference error | `3.082207e-7` relative L2 |
| rotated V reference error | `2.861411e-7` relative L2 |
| QK scores A vs B | `(2.961188560e-6, 5.299079702e-7, 9.274475025e-8)` |
| softmax A vs B | `(4.989892152e-7, 2.155730610e-8, 3.505521042e-7)` |
| recovered attention A vs B, still FP32 | `(1.215934753e-5, 4.705362383e-7, 1.368033032e-6)` |
| post-attention output-projection + residual A vs B, BF16 | `(1.953125000e-3, 3.017438303e-7, 2.898554914e-5)` |
| post-MLP/layer output A vs B, BF16 | `(3.906250000e-3, 1.647995668e-5, 3.130950573e-4)` |

The first material precision boundary is the common call at
`text_context_impl.h:927-930`:

```text
FP32 recovered attention (A: direct attention; B: attention @ R_V.T)
    -> FP32-to-BF16 into `a`
    -> BF16 gate and output projection
```

At layer 3, the FP32 recovered-attention difference is only `1.368033032e-6` relative L2. The
common BF16 conversion changes the downstream fused residual by `2.898554914e-5` relative L2.
The first post-MLP layer output is already `3.130950573e-4` relative L2, and the next GDN/MLP
stage (layer 4) reaches `7.657250444e-3` relative L2. This is the precise observed amplification;
it is not a cache coordinate or layer-selection mismatch.

For reference, production BF16 attention versus matched control A at layer 3 was
`1.563633320e-3` relative L2, and production BF16 versus A at the fused post-attention boundary
was `5.991697936e-4`. This confirms that the matched A/B attention kernel is not the source of the
large control-path difference.

## All full-attention post-attention boundaries

The following are A versus B `post_attention_linear_add` relative L2 values for all 16 full layers:

| Full layer | 3 | 7 | 11 | 15 | 19 | 23 | 27 | 31 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relative L2 | `2.899e-5` | `2.393e-2` | `3.345e-2` | `3.874e-2` | `5.805e-2` | `8.050e-2` | `8.759e-2` | `9.757e-2` |

| Full layer | 35 | 39 | 43 | 47 | 51 | 55 | 59 | 63 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relative L2 | `1.133e-1` | `1.142e-1` | `1.125e-1` | `1.136e-1` | `1.203e-1` | `1.104e-1` | `9.941e-2` | `8.134e-2` |

Layer 3 is therefore the first differing operation; later full-attention values are propagated
consequences, not independent evidence of a wrong per-layer asset mapping.

## Decode interpretation

The diagnostic run did execute the matched FP32 cache on the ordinary decode card. The decode
diagnostic files include the same-token-position cache append/read path (`tokens_1` files) for both
variants. However, A and B already selected different first prefill tokens (`271` versus `198`),
so their layer-3 decode Q tensors are different inputs. Their decode layer-3 recovered-attention
relative L2 was `3.811936202e-1`, which cannot be used as a cache-only comparison. A fair forced
same-token decode test was not started because the required prefill gate had already failed at the
earlier layer-3 BF16 boundary.

## Verdict and next gate

The matched cache architecture is present and the FP32 rotational mathematics remains close at
the attention boundary. The real-model A/B equivalence gate nevertheless fails because the current
common BF16 conversion exposes a small orthogonal-transform round-off difference to the highly
sensitive downstream NVFP4/GDN/MLP computation. This is a precision-contract blocker, not evidence
of a new orientation, RoPE, GQA, layer-map, or calibration problem.

**D1.2: BLOCKED.** Do not begin D2 or INT2. The next diagnostic must either keep the matched
comparison beyond this boundary in a justified higher-precision path or establish a mathematically
controlled BF16 rounding contract before any calibrated cache quantization is introduced.
