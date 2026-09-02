# OSCAR KV fidelity status

Status: **blocked for faithful production OSCAR**. The current benchmarked executable remains an
experimental fixed-Hadamard Q2/VeriCache path. It must not be labeled byte-for-byte upstream
OSCAR until calibrated Qwen3.8 assets and mixed-window runtime tests exist.

## Completed in this pass

- Added the typed `OscarKVLayout` contract in `src/core/oscar_kv_layout.h`.
- Added the selector to paged K/V geometry and single/batched cache views. Full-cache append and
  dense prompt/decode attention now dispatch from the same view value, instead of independently
  reading `NINFER_OSCAR_Q2_TRANSPOSED`.
- Converted StateImage OSCAR re-quantization from two ambiguous boolean layout arguments to the
  same typed selector.
- Added `tools/oscar/validate_dump.py` and its manifest schema, so calibration input provenance,
  hybrid-layer mapping, and Q/K/V hashes are checked before fitting.
- Kept the safe default `Contiguous`; `TransposedQ2` is available only as an explicit layout value
  after its writer/reader exact-byte round trip is verified.

## Verified

The designated Debug/Ninja build source is `D:\AI\voidinfer-adaptive-dflash2`, built with CUDA
13.1 and `sm_120a`. These commands passed after the changes:

```text
cmake --build D:\AI\build-adaptive-dflash2 --config Debug --target ninfer_oscar_kv_test
D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_kv_test.exe
```

Observed result: OSCAR-Q2/Q4 packed and protected tests passed. The target runtime and StateImage
test executables also linked and exited successfully:

```text
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_runtime_mechanisms_test.exe
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_state_image_test.exe
```

## Phase B2 — verified 256-token native smoke capture

The focused capture target is built in the latest designated Debug/Ninja build:

```text
cmake --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_capture_test -j 1
```

The deterministic wrapper is `tools/oscar/capture_qkv_smoke.ps1`. It binds the verified
`qwen3.8-27b` / `nvfp4-dflash2` artifact, arms capture only after Engine construction, submits
256 copies of token id 198 with one greedy output token, and validates the resulting manifest and
all payload hashes. The completed smoke dump is:

```text
results/oscar/captures/phase-b2-qkv-256/manifest.json
```

The native capture passed with `prompt_tokens=256` and `generated_tokens=1`. The fail-closed
validator passed with:

```text
layers=16 chunks=1 useful_tokens=256 dump_bytes=67108864
manifest_sha256=df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89
```

Every expected full-attention layer `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63` has one
aligned Q/K/V capture. Payloads are BF16, serialized as contiguous `[tokens, heads, head_dim]`
with dimension order `[256,24,256]` for Q and `[256,4,256]` for K/V; the raw bytes are token-major,
head-major, dimension-fastest. Q and K are captured after Q/K RMSNorm and in-place RoPE. V is
captured at the corresponding projected BF16 stage without Q/K normalization or RoPE. The hook is
before `causal_softmax_attention`, whose wrapper appends K/V to the runtime cache. No GDN state,
DFlash2 drafter KV, or cache-packed tensor is captured.

Representative first four BF16 values, independently inspected by the validator, are:

| Model layer | Q | K | V |
|---:|---|---|---|
| 3 | `[-0.142578125, -0.333984375, -0.64453125, -0.99609375]` | `[0.271484375, 0.416015625, -0.66796875, -0.1767578125]` | `[-0.515625, 0.01336669921875, 0.435546875, 0.41015625]` |
| 35 | `[0.82421875, 0.01055908203125, -0.021484375, -0.57421875]` | `[0.0120849609375, -0.01202392578125, -0.022705078125, 0.006378173828125]` | `[-0.006103515625, 0.69140625, 0.328125, 0.15234375]` |
| 63 | `[-1.4140625, -0.044677734375, -2.234375, -1.890625]` | `[-0.103515625, -0.06982421875, 0.06396484375, -0.1298828125]` | `[-2.265625, 0.048828125, 1.5859375, 0.0108642578125]` |

The exact implementation boundary is in `src/targets/qwen3_6/impl/runtime/text_context_impl.h:860-868`;
the copy/shape/finite checks and manifest writer are in
`src/targets/qwen3_6/impl/runtime/oscar_qkv_capture.cpp:90-236`. This is capture plumbing only;
no OSCAR fitter, rotation asset, INT2 runtime, or kernel optimization was run.

## Phase B3 — official fitter smoke

The validated Phase B2 raw dump was converted to the official hierarchy under
`results/oscar/dumps/phase-b2-256-official`. Because the official loader skips chunk `0` in
`--chunk-id all` mode, the converter writes the useful smoke data as explicit chunk `1.pt`, and the
fitter is invoked with `--chunk-id 1`.

The unchanged upstream `compute_kv_rotation.py` was pinned from FutureMLS-Lab/OSCAR main commit
`41ebcdba3db5f0ce1339c3727caea80df575d437`; the local script SHA-256 is
`f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b`. The official invocation used
`--method qqt_sst --composition r_h_pbr --head-dim 256 --chunk-id 1` and completed successfully
in 2.015 seconds, producing separate K and V checkpoints for all 16 full-attention layers.

The output checkpoints were independently reloaded and validated as finite FP32 `[256,256]`
rotations with exact layer mapping. Per-layer `max_abs(R @ R.T - I)` was at most
`2.011652722e-08` for K and `1.916613712e-08` for V. A deterministic independent 8x8 fixture
verified the official bit-reversal/Hadamard `R*H*Pbr` composition with max error `2.22e-16`.
This is a fitter smoke result only; the 256-token data is not calibration-quality and the rotations
are not integrated into the runtime.

## Phase C1 — rotated-BF16 invariance

The first mathematical wiring test is complete in
`results/oscar/phase-c1-rotation-invariance.md`. Using the converted BF16 smoke tensors and the
official Phase B3 FP32 K/V checkpoints, the reference path applies `Q'=Q*R_K`, `K'=K*R_K`,
`V'=V*R_V`, causal GQA attention, and `O=O'*R_V.T`. It passes all 16 full-attention layers.

Observed maxima across all layers are `4.971724872e-07` for attention scores,
`7.114301309e-08` for softmax probabilities, and `3.373735723e-06` for recovered post-attention
output. The unrecovered V-rotated output is intentionally non-invariant and reaches `82.3280771`,
confirming that the V inverse is required. The test uses exact BF16 source values with FP64
diagnostic products and does not add runtime wiring or INT2 quantization.

## Phase C2 — slow calibrated INT2 reference

The offline correctness-first INT2 reference is complete in
`results/oscar/phase-c2-int2-reference.md`, with machine-readable evidence in
`results/oscar/phase-c2-int2-reference.json`. It evaluates the existing 256-token smoke data
for the BF16 baseline, rotated BF16, the current fixed-Hadamard Q2 control, and calibrated
OSCAR `qqt_sst` + `r_h_pbr` INT2. Layer 3 was run first, then all 16 full-attention layers.

The calibrated path uses the pinned upstream `simulate_int2_asym` semantics exactly: per-row
FP32 min/max, three affine INT2 levels, and dequantization before causal GQA attention. It
applies `Q'=Q*R_K`, `K'=K*R_K`, `V'=V*R_V`, then `O=O'*R_V.T`. The local implementation was
checked element-for-element against the upstream function, with all codes in `[0,3]` and all
outputs finite. The fixed-Hadamard control remains separate and mirrors the current runtime's
0.93/0.91 K/V clipping and BF16 scale/zero metadata.

The calibrated reference is **PASS for mathematical coherence** across all 16 layers. Maximum
calibrated errors versus original BF16 were `7.058641e+00` for attention score logits,
`8.521844e-01` for softmax probabilities, and `4.403106e+01` for attention output; these are
2-bit smoke quantization measurements, not invariance gates. The calibrated path improved the
mean per-layer maximum attention-output error to `1.715218e+01` versus `4.645777e+01` for the
fixed-Hadamard control. The B2 artifact does not contain output-projection/residual/LM-head
activations, so full layer-output and LM-logit/token agreement remain unavailable.

No CUDA runtime, INT2 packing, 10K/30K calibration, production clipping, or performance work was
performed in C2.

## Phase C3 — 10K calibration pilot

The C3 pilot is recorded in results/oscar/phase-c3-cal10k.md, with machine-readable held-out
evidence in results/oscar/evaluations/phase-c3-cal10k-heldout.json. The validated capture path
was reused without moving the boundary: 40 deterministic independent 256-token requests yielded
10,240 useful tokens for all 16 full-attention layers. A separate four-request, 1,024-token set
with a different formula seed was held out and never passed to the fitter. No GDN recurrent state
or DFlash2 drafter KV was captured.

The raw calibration manifest SHA-256 is
4083f9c7c62e41dacab1ce55322362502d6a33b416eb660e4339413d3a074d6e; its fail-closed
multi-chunk validation checked exact layer/chunk/Q/K/V coverage, topology, GQA mapping, stage,
payload hashes, finite BF16 values, and no unlisted/GDN payloads. The official conversion emitted
1,920 BF16 .pt tensors across official chunks 1..40, shifting raw chunk c to c+1 because the
unchanged upstream --chunk-id all loader skips chunk 0.

The unchanged FutureMLS-Lab OSCAR fitter (main commit
41ebcdba3db5f0ce1339c3727caea80df575d437, local SHA-256
f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b) completed qqt_sst +
r_h_pbr at head dimension 256 for all layers. The K/V checkpoints passed finite FP32 reload,
exact mapping, and orthogonality checks; worst max_abs(R @ R.T - I) was 2.193682191e-08 for K
and 2.131072918e-08 for V. The independent r_h_pbr fixture passed at about 2.2e-16.

On held-out tokens, 10K rotated-BF16 remained invariant: maximum score error 4.062135e-07,
softmax error 8.728284e-08, and recovered attention-output error 3.013387e-06, with 100%
attention-argmax agreement. The upstream per-row INT2 reference using 10K assets had aggregate
score/softmax/output maxima 6.777395/8.739543e-01/37.105069 and output relative L2
0.303087, improving on the previous 256-token assets (41.614878 and 0.317166 for output
max and relative L2). This is KEEP for the next 30K calibration pilot, not a production
runtime quality claim. Group-128 remains the intended first runtime target, but the unchanged
upstream fitter/quantizer operates on D=256 rows; group-128 packing/metadata is not implemented.

## Phase C4 — 30K primary calibration assets

The C4 result is recorded in results/oscar/phase-c4-cal30k.md, with machine-readable held-out
evidence in results/oscar/evaluations/qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1-heldout.json.
The unchanged capture boundary produced 30,720 useful tokens as 120 independent 256-token
requests for all 16 full-attention layers. A separate 1,024-token, seed-40019 set was held out
and never passed to the fitter. No GDN state or DFlash2 drafter KV was captured.

The raw calibration manifest SHA-256 is
8a5f6a6f89f2ab21d178f7644297365012bc6dcebb03388d514ddf12e0b59bd8. The fail-closed validator
passed exact layer/chunk/Q/K/V coverage, topology, GQA mapping, stage uniformity, payload hashes,
finite BF16 values, and no unlisted/GDN payloads. The official conversion emitted 5,760 BF16
.pt tensors across official chunks 1..120; the raw chunk IDs were shifted by one because the
unchanged upstream --chunk-id all loader skips chunk 0.

The unchanged FutureMLS-Lab OSCAR fitter (main commit
41ebcdba3db5f0ce1339c3727caea80df575d437, local SHA-256
f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b) completed qqt_sst + r_h_pbr
at head dimension 256 for all 16 layers. The immutable asset identity is
qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1. K/V checkpoints are finite FP32 [256,256]
assets with worst max_abs(R @ R.T - I) of 2.082505102e-08 for K and 2.296734802e-08 for V;
the independent r_h_pbr fixture passed at 2.2e-16.

On the held-out set, 30K rotated-BF16 remained invariant with maximum score/softmax/recovered
output errors of 4.309937e-07/8.126786e-08/3.746161e-06 and 100% attention-argmax agreement.
Compared with the 10K assets, the 30K INT2 path improved aggregate output mean absolute error
(0.294053 vs 0.295811) and relative L2 (0.302515 vs 0.305705), effectively matched the worst
output maximum (39.595471 vs 39.497786), but regressed score/softmax maxima and argmax agreement.
This is a mixed but favorable calibration result: the versioned 30K assets are ready for the next
runtime-integration phase, while the QKV-only evaluation is not a full model quality claim.
Group-128 remains the intended first runtime target; runtime packing and metadata are not changed.

## Exact remaining gap

The workspace now has validated 30K candidate assets in addition to the 10K pilot and 256-token
smoke dump. The compiled model artifact remains distinct from the raw calibration captures, and
the C4 assets are not yet loaded by the production runtime. The following required work remains:

1. Integrate the layer-specific FP32 K/V assets, query rotation, V inverse/absorption, and INT2
   metadata into an explicit oscar-int2 runtime mode with the intended group-128 contract.
2. Validate runtime bit packing, protected windows, cache behavior, output projection, residual,
   LM-logit, and generated-token fidelity at multiple context lengths.
3. Run BF16-vs-OSCAR exactness at 512/4K/16K/32K/128K plus the quality workloads before any
   performance tuning. The current vericache-nvfp4 and oscar-q2-device labels remain experimental.

The next unblock is runtime integration and end-to-end fidelity validation. Validate every future
asset or dump with the appropriate fail-closed validator before changing the runtime layout default.

The upstream algorithm and serving contract are documented in the
[`FutureMLS-Lab/OSCAR README`](https://github.com/FutureMLS-Lab/OSCAR) and the upstream rotation
implementation linked above.

## Phase D1 — full-runtime rotated-BF16 qualification

The next runtime qualification was attempted with the immutable C4 asset
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`. The implementation is opt-in through
`NINFER_OSCAR_ROTATION_MODE=oscar-rotated-bf16`; it loads raw FP32 K/V banks without LibTorch,
validates the C4 manifest identity/model hash/payload hashes/topology/layer mapping, and emits
asset identity, asset hash, calibrated status, full-attention count, and rotation-mode telemetry.
The on-the-fly path leaves NVFP4 weights and all GDN recurrent state unchanged, applies the C1
`Q*R_K`, `K*R_K`, `V*R_V` convention at full-attention layers only, and applies `R_V.T` before
the existing gate and output projection.

The implementation and deterministic real-model test are in:

```text
src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.h
src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp
src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cu
tests/targets/qwen3_6_27b/test_oscar_runtime.cpp
results/oscar/phase-d1-runtime-rotated-bf16.md
```

The loader and build passed, but D1 is **BLOCKED** by real-runtime numerical drift. Layers 0–2
were bit-identical; the first full-attention layer, model layer 3, showed Q/K score max error
`1.48105e-02`, softmax max `2.17747e-03`, recovered attention max `1.5625e-02`, and post-MLP
relative L2 `7.73924e-03`. The 32-token full-model comparison then showed hidden relative L2
`5.98146e-02`, logits relative L2 `2.04136e-02`, false top-1 agreement, and false generated-token
agreement. Pre-rotation Q/K/V were identical and the stage audit found no orientation, RoPE,
layer-mapping, GQA, or inverse-transpose error; the remaining blocker is BF16 round-off through
the on-the-fly transform/inverse path. The required stop condition prevented 512/4K tests.

Do not label the runtime path equivalent and do not begin INT2 integration until this blocker is
resolved.

## Phase D1.1 — rotation precision contract

D1.1 investigated the D1 layer-3 drift using the immutable C4 asset
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` and the same deterministic 32-token input.
The four diagnostic modes were BF16-materialized, FP32-rotation, FP32-inverse, and
FP32-rotation+inverse. No INT2, recalibration, DFlash2/MTP, adaptive-K, weight baking, or general
kernel optimization was performed.

FP32 row-right rotation reduced layer-3 Q/K/V orientation error to relative L2
`3.209392e-7/3.082207e-7/2.861411e-7`; QK score and softmax errors were
`9.274475e-8/3.505521e-7`. FP32 inverse alone was identical to BF16-materialized, so the
dominant early error is introduced before inverse-V. The combined FP32 diagnostic reduced the
post-attention fused output-projection-plus-residual error to `5.990686e-4` and post-MLP error
to `5.917754e-3` at layer 3.

The independent reference check is important: CUDA FP32 reference attention plus FP32 inverse
versus Python double-precision reference was `1.091371e-6` relative L2, while the normal runtime
BF16 attention order alone differs from the Python reference by `1.749825e-3`. Thus the C1
rotation algebra remains valid; the real-runtime comparison still contains an attention-order
and cache-precision confound. The diagnostic FP32 attention path is prefill-only (`T <= 64`) and
does not provide an FP32 rotated decode cache.

D1.1 remains **BLOCKED** for runtime progression. The evidence-supported contract is FP32 or
fused online Q/K/V rotation with no stored rotated-BF16 intermediate, FP32 attention accumulation,
FP32 `R_V.T` recovery, and only the final downstream conversion. Option B's algebraic V
absorption into `R_V.T @ W_O` was not used because the output projection is NVFP4 and permanent
requantization/baking is prohibited. Q/K remain post-RoPE and cannot be treated as ordinary
pre-RoPE weight baking. A matched prefill/decode implementation must pass the full-model gate
before D2 calibrated INT2 integration.

Full evidence is in `results/oscar/phase-d1-1-rotation-precision.md`.

## Phase D2.1 — official INT2 quantizer / codec parity (2026-09-01)

The serving-side clipped INT2 reference is pinned to FutureMLS-Lab/OSCAR commit
`41ebcdba3db5f0ce1339c3727caea80df575d437`; the exact `oscar_rotation_clip_int2_kv.py` source
SHA-256 is `c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157`. Its contract is
row-wise absolute-value quantile clipping, symmetric clamp, two independent D=256 groups of
128, FP32 affine scale/zero-point metadata, symbols `0..3`, and byte `j` containing dimensions
`j,j+64,j+128,j+192`. K/V clip ratios are `.96/.92`.

The project now has a separate scalar `OscarInt2G128` codec and deterministic golden fixtures;
the existing experimental Q2 path remains unchanged because it uses different clipping,
row-wide parameters, BF16 metadata, zero semantics, and packing. Fifteen synthetic/real cases
cover sign/range extremes, clipping outliers, group boundaries, multiple independent rows, and
rotated K/V samples from layers 3/35/63. The compiled C++ parity test passed exact clipped
values, symbols, and packed bytes, plus FP32 metadata/decode parity, for all 63 rows. Full
details, commands, hashes, and storage accounting are in
`results/oscar/phase-d2-1-int2-codec.md`.

D2.1 is **PASS** for codec mathematics only. The codec is not yet wired into live attention or
mixed BF16 windows; those are D2.2 work.

## Phase D2.2a — typed mixed BF16 / OSCAR INT2 cache representation (2026-09-01)

D2.2a adds a separate physical representation for the intended mixed policy without changing
the validated `OscarInt2G128` contract or the legacy experimental Q2 path. Each full-attention
layer has independent, explicitly tagged page pools for: protected prefix BF16 positions
`[0,64)`, historical middle positions using official group-128 OSCAR INT2, and the most recent
256 positions using BF16. A page contains one storage format only; K and V types are recorded
independently and must agree. Physical token ranges are local to each region pool.

Page metadata records model/full-attention layer, sequence, logical and physical ranges, physical
page index, occupancy/capacity, layout version, group size, K/V storage types, and region role.
Slot metadata repeats the addressing/type identity and page offset. The D2.1 INT2 row contract is
preserved exactly: D=256, two independent 128-wide groups, 64-byte packed K/V payload per row,
and four FP32 metadata values per row and tensor.

The deterministic static construction test passed for 500 tokens and all 16 full-attention layers:
logical positions were resolved exactly once with no holes/overlap; boundaries `63/64`, the
unaligned recent transition `243/244`, and final position `499` passed; K/V types agreed; every
full-attention layer shared the policy; and no GDN state entered the bundle. The validation JSON
is `results/oscar/phase-d2-2a-layout-validation.json`, SHA-256
`fb158c93b4f9e5b6a9c1999392928c4318df3ec13674b572b245d7afb40a2ce7`.

For the 500-token, 16-layer construction, the fully allocated physical bundle is
`23,328,768` bytes: `20,971,520` BF16 data bytes, `1,572,864` INT2 payload bytes,
`393,216` INT2 metadata bytes, `7,168` page-header bytes, and `384,000` slot-table bytes.
This is `1.423875` bytes/value (`11.391` bits/value); historical bulk alone is
`0.3572265625` bytes/value (`2.8578125` bits/value) including its partial page and metadata.
The accounting explicitly excludes implementation-dependent `std::vector` allocator bookkeeping.

D2.2a is **PASS** for typed coexistence and deterministic accounting only. Token aging/promotion,
mixed-window attention, live runtime dispatch, and performance remain unimplemented and must be
qualified separately.

## Phase D2.2b — deterministic BF16-recent to OSCAR-INT2 aging (2026-09-01)

D2.2b adds a separate slow aging fixture on top of the D2.2a typed layout. It consumes rows
already in the validated C4 rotated coordinate system and validates the pinned asset contract:
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`, model SHA-256
`6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`, runtime asset-manifest
SHA-256 `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560`, calibrated=true,
and `qqt_sst+r_h_pbr`. Altered identity and non-full-attention topology are rejected before
rows are accepted. No fitting, rotation application, live attention, or legacy Q2 path is used.

The deterministic bundle appends forced token IDs `0..323`. Token 63 remains protected prefix
BF16; token 64 is initially recent BF16; at context 321, appending token 320 ages token 64;
the next three appends age tokens 65, 66, and 67. Each aging operation decodes the stored BF16
row to FP32, runs the D2.1 official `OscarInt2G128` codec with K/V clips `.96/.92`, writes packed
symbols and FP32 metadata to historical storage, clears the BF16 source, and marks the row once.

Exact parity passed on layers 3, 35, and 63 across all four KV heads for clipped values, FP32
metadata, symbols, and packed bytes. All 16 full-attention layers performed four conversions.
The fixture rejected a second conversion, prefix aging, altered asset identity, GDN layer
inclusion, BF16/historical overlap, and logical holes/overlap. The machine-readable evidence is
`results/oscar/phase-d2-2b-aging-validation.json`, SHA-256
`a702f53e4ea2d0c0120cb6d80312d25a545ba478e7a75f41830450ca00292477`.

At each of the four transitions, physical bytes across all 16 full-attention layers are
`4,194,304` BF16 prefix, `16,777,216` BF16 recent, `524,288` packed INT2 payload, and
`131,072` FP32 INT2 metadata. INT2 totals are page-rounded to one 64-token historical page per
layer; historical logical token counts are 1, 2, 3, and 4. Final context 324 is therefore
prefix 64 / historical 4 / recent 256. The detailed table, headers, slot tables, mixed totals,
and reproduction command are in `results/oscar/phase-d2-2b-aging.md`.

D2.2b is **PASS** for deterministic token aging and exact official codec parity. Live INT2
attention and production decoder integration remain unimplemented.

## Phase D2.2c — fork / rollback / StateImage qualification (2026-09-01)

D2.2c qualifies transitions on top of the unchanged D2.2a typed representation and D2.2b
official `OscarInt2G128` aging. `OscarMixedTransitionCache` adds immutable page-storage blocks,
branch-local sequence/page metadata, copy-on-write replacement for changed blocks, complete
logical/page/slot fingerprints, and parent-lineage commit checks. It has no attention-facing API.

The deterministic base fixture contains logical tokens `0..323`: protected BF16 prefix `0..63`,
historical OSCAR INT2 `64..67`, and BF16 recent `68..323`. It has six pages/layer and 96 page
blocks over the exact 16 full-attention layers. Forking to a distinct sequence shares all 96
immutable blocks at refcount 2 and does not re-encode historical rows. Divergent forced appends
retain shared prefix blocks while changed pages use branch-local blocks; the base remains
unchanged.

Rollback after four speculative appends across the aging boundary restores the complete
pre-speculation fingerprint exactly, including retained INT2 packed payloads and FP32 metadata;
speculative positions are no longer addressable. Commit adopts only a child whose recorded parent
fingerprint matches, producing logical `0..327` once with prefix/historical/recent `64/8/256`.

The project’s existing Qwen `StateImageStore` was reviewed. Its fixed payload owns continuation
hidden, GDN recurrent, and DFlash components, with an optional legacy Q4 shadow; it does not own
the typed OSCAR page pools. To avoid changing that schema or introducing a Q4 shadow, D2.2c uses
a dedicated StateImage-compatible OSCAR snapshot envelope containing validated asset identity and
deterministic original BF16 input lineage. A fresh restore rebuilds the unchanged D2.2b physical
representation and matches every logical/page/slot/payload/metadata fingerprint. It is a
diagnostic state-image qualification, not production StateImage or live-attention integration.

Evidence, exact commands, fingerprints, page/refcount observations, and hashes are in
`results/oscar/phase-d2-2c-state-transitions.md`; machine-readable complete page fingerprints
are in `results/oscar/phase-d2-2c-state-transitions-validation.json` (SHA-256
`df1fa525a1791cf6c6463686d1b3a741901a58825d832cca24c2e96992e16784`). D2.2c is **PASS** for
append → aging → fork → speculative mutation → rollback/commit → restore. Live INT2 attention
remains the next separate phase.

## Phase D2.3a — mixed-cache reference attention parity (2026-09-01)

D2.3a adds a deliberately slow, diagnostic-only `OscarMixedAttentionReader` over the unchanged
D2.2b mixed representation. It resolves causal logical positions through explicit page metadata:
protected prefix BF16, historical official `OscarInt2G128` group-128 INT2, and recent BF16. BF16
rows are converted to FP32 only in temporary per-query buffers. Historical rows use the exact
D2.1 packed bytes and FP32 metadata; no persistent BF16 materialization, legacy Q2 dispatch, or
serving path is involved.

For each query the reader applies the layer-specific calibrated `R_K` to Q, performs scalar FP32
GQA with `q_head / 6`, stable causal softmax, and rotated-coordinate AV, then recovers with
`R_V.T` in FP32. It validates page role/storage pairing and rejects malformed mixed entries.
The input contract is the already-rotated row convention established by D2.2b; this phase does
not re-open projection, RoPE, calibration, or cache-transition semantics.

The test-side independent CPU reference does not call the new reader or page resolver. It reads a
deterministic source-row archive, reconstructs the same prefix/historical/recent classification,
independently invokes the official `OscarInt2G128` encoder/decoder for historical positions, and
runs its own FP32 rotation/attention/recovery sequence. This isolates page addressing and tier
dispatch from mathematical reference parity. It compares the same post-aging INT2 representation;
it does not require lossy post-aging storage to equal the former BF16 row.

Static context 324 has tiers `64/4/256`; context 332 has `64/12/256`. Layer-3 boundary queries
verified `63 -> 64/0/0`, `64 -> 64/1/0`, `68 -> 64/4/1`, and final `323 -> 64/4/256`.
The persistent forced path appended 324–327 and observed historical/recent counts `5/256` through
`8/256`. Every emitted trace contained each logical position exactly once in causal order, with
no holes or duplicates. Layers 35 and 63 passed their complete three-tier query, and the final
static query passed for all 16 full-attention layers.

The rebuilt validation executable passed 31/31 comparisons. Maximum relative-L2 was zero for
rotated Q, QK scores, stable softmax, rotated-coordinate AV, and recovered output; the declared
FP32 parity gate was `1e-6`. No GDN state, production-cache fallback, or legacy Q2 path appeared.
The machine-readable trace evidence is
`results/oscar/phase-d2-3a-reference-attention-validation.json`, SHA-256
`6090dab55f3822b283c39f306e2789d758eb351a137b09f9acdf824735015610`; the detailed report and
reproduction command are in `results/oscar/phase-d2-3a-reference-attention.md`.

D2.3a is **PASS** for mixed-cache reference attention parity. D2.3b may connect this qualified
reader to real Qwen3.8 attention. No performance conclusion is implied, and normal serving,
kernel optimization, DFlash2/MTP, adaptive-K, and INT2 runtime integration remain separate work.

## Phase D2.3b — live Qwen runtime + forced-token reference parity (2026-09-01)

D2.3b adds the opt-in diagnostic mode `oscar-int2-reference-live` to the real Qwen3.8-27B
runtime. The live path consumes actual post-RMSNorm/post-RoPE Q/K and projected V tensors,
applies the immutable C4 layer-specific rotations, appends rotated rows to the typed mixed
cache, ages recent rows with the official `OscarInt2G128` codec, and invokes the slow FP32
mixed reader. The 16 full-attention layers are explicitly enumerated as
`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; GDN state is not an OSCAR KV source.

The 32-token and 324-token runs use the same forced IDs
`997,1001,1003,1005,1007,1009,1011,1013,1015` in the control and live executions. A
test-side independent tap validator reconstructs rows without the live reader or page resolver,
independently decodes BF16/official INT2 rows, and repeats FP32 Q rotation, GQA scores, stable
softmax, AV, and `R_V.T` recovery. It passed 2/2 short taps and 30/30 mixed taps with worst
relative-L2 zero under the `2e-6` relative / `1e-4` max-absolute gate.

The mixed run explicitly exercised prefix/recent boundaries and live aging. Runtime telemetry
reported the complete full-layer bitmap `1111111111111111`, 192 exact K/V codec parity checks,
zero GDN and legacy-Q2 dispatches, `bf16_history_shadow=false`, and no fallback. The C4 runtime
manifest hash is `c1979e86744682733a668739642ad3a945b5a6220e8b3ed983b74d33b82c3afc`; the tap
set contains 32 files and 34,719,936 bytes with aggregate hash
`9f5de2465a55ae983d0c4338657751fe10dfcccc1b35380fb75faffea374b080`.

The BF16-versus-live full-model hidden/logit differences are informational because this phase
qualifies live/reference cache and attention plumbing, not model fidelity: hidden relative-L2
was `0.0710382048` / `0.0451812669` and logit relative-L2 `0.0244162833` / `0.0178153973`
for 32 / 324-token cases, with the forced output sequence preserved. D2.3b is **PASS** and
authorizes D3 model-fidelity qualification. Performance optimization and production StateImage
integration remain out of scope.

Evidence: `results/oscar/phase-d2-3b-live-reference-runtime.md`.

## Phase D3.1 — minimum full-model fidelity gate (2026-09-02)

D3.1 compared the real Qwen3.8-27B NVFP4 runtime with unchanged weights under identical
greedy teacher-forced inputs. The BF16-KV control was compared with both selectable calibrated
runtime banks: `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1` and
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`. Both used the existing diagnostic
`oscar-int2-reference-live` contract: official `OscarInt2G128`, group 128, K/V clips `.96/.92`,
BF16 prefix 64, BF16 recent 256, and no speculation or prefix reuse. The legacy Q2 route and
GDN recurrent state remained excluded.

Fixed-token final-position tests at 32/324/512 tokens completed for both banks. Logit relative
L2 was `0.023341/0.017997/0.025883` for 10K and `0.024416/0.017815/0.023566` for 30K.
Top-10 overlap was 9/10, 10/10, 9/10 for 10K and 8/10, 10/10, 9/10 for 30K. The only
fixed top-1 disagreement was 10K at 32 tokens, where the BF16 margin was `0.03125` and top-5
containment was 5/5. Error did not grow monotonically with INT2 history and remained bounded
at 512 tokens.

A 12-case deterministic natural suite passed objective checks for BF16, 10K, and 30K: 3
arithmetic, 3 coding, 2 structured JSON, 2 retrieval, and 2 copy/exactness cases. The four
retrieval/copy cases used 1,145–1,151-token prompts, crossed the mixed-cache threshold, and
recovered their exact identifiers with both calibrated banks. Objective success was 12/12 for
all three paths. Natural token identity was recorded only as information, not as a gate; valid
JSON whitespace is accepted by the corrected evaluator.

D3.1 is **PASS**. The 30K bank is the provisional baseline based on the deeper live fixed-token
results and its C4 held-out aggregate INT2 relative-L2 advantage (`0.302515` versus `0.305705`);
10K remains selectable and preserved. This authorizes investment in the optimized SM120a OSCAR
attention path. It does not qualify 4K/16K/32K/64K/128K model fidelity; those measurements
remain pending until the optimized path makes them practical. Evidence and exact commands are
in `results/oscar/phase-d3-1-minimum-fidelity.md`.

## Phase D4.1 — correct live-path profile (2026-09-02)

D4.1 profiled the genuine calibrated `oscar-int2-reference-live` path with the immutable
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` runtime bank. Profiling-only counters were
added for runtime Q/K/V rotation and staging, actual recent-to-INT2 conversion, reader-side Q
rotation, official group-128 INT2 K/V decode, QK, stable softmax, AV, `R_V.T`, reader total,
and the complete live full-attention branch. The counters are gated by
`NINFER_OSCAR_D4_1_PROFILE=1`; the OSCAR codec, cache layout, attention arithmetic, model
weights, DFlash2/MTP, and adaptive-K were not changed.

A complete real-runtime 512-token prefill used prefix/history/recent `64/192/256`, 8,192
full-attention reader calls, and 3,072 aging events. The full live attention branch measured
`166.190 s`; the request/verifier wall was `174.529601 s` with model initialization excluded.
Telemetry retained full-layer coverage `1111111111111111`, `oscar_calibrated=true`, and zero
GDN, legacy-Q2, historical-BF16-shadow, and fallback dispatch.

The full 2K/4K causal live sweep was not continued because the 512-token scalar request already
took 174.5 seconds. The permitted genuine mixed-reader microbenchmark used a real typed
`OscarMixedAgingLayerCache`, official `OscarInt2G128` aging/decode, and C4 rotation matrices for
one final causal query. Reader totals were `31.9101/108.7611/213.0691 ms` at 512/2K/4K, with
`192/1,728/3,776` historical tokens. The adjacent history slope was stable at approximately
`50.5 us` per added historical token. At 4K, QK was `56.0506 ms`, AV `50.1997 ms`, and INT2
K/V decode `34.0487/33.9776 ms`; softmax was only `1.6400 ms`.

The dominant long-history cost is scalar causal QK + AV traversal, followed by INT2 K/V decode.
The single recommended D4.2 target is a fused SM120a mixed-cache attention traversal that feeds
INT2 K decode directly into QK and performs V decode/AV in the same logical traversal without a
persistent decoded-K/V buffer. This phase made no optimization claim. D2.3a parity was rerun
after instrumentation and remained 31/31 PASS with zero reported relative-L2.

## Phase D4.2a — fused historical INT2 attention kernel (2026-09-02)

D4.2a adds an isolated SM120a CUDA path for the official historical `OscarInt2G128` region.
The interface takes already-rotated FP32 Q and packed historical K/V rows with FP32 metadata;
prefix/recent BF16 rows, Q rotation, `R_V.T`, and cache policy remain outside the path. A
256-thread score tile is assigned per KV head and 256-token tile. Each thread decodes one K
scalar at a time and consumes it in six GQA dot products, so decoded K is not persisted. A
separate 256-thread AV block per KV head decodes each V scalar directly into six FP32 AV
accumulators. Stable FP32 softmax is retained between them. This is tiled/multi-block SIMT,
not a serial one-warp history scan.

The deterministic independent CPU oracle and GPU path matched at history 128/512/2K/4K with
all finite outputs. Worst relative-L2 was `1.105956699e-06` for softmax and `1.089803277e-06`
for AV at 4K, under the `1e-4` fail-closed gate. RTX 5090 measurements through 32K were
`0.056451/0.161510/0.401674/0.801082/1.444493/2.968627/6.012608 ms` for
128/512/2K/4K/8K/16K/32K, with `119.0–301.7x` speedup over the CPU scalar historical oracle.
These are historical-only correctness/latency results, not mixed-cache or serving throughput.

D4.2a is **PASS**. D4.2b may connect this path to the BF16 protected prefix/recent regions;
end-to-end mixed-cache and model-fidelity qualification remain pending. Evidence:
`results/oscar/phase-d4-2a-fused-history.md`.

## Phase D4.2b — mixed-tier GPU attention (2026-09-02)

D4.2b adds a separate complete GPU reader for the logical `BF16 protected prefix + official
OscarInt2G128 historical bulk + BF16 recent` sequence. The score and AV kernels branch by
logical tier: BF16 rows are converted at point of use, while historical INT2 rows use the
qualified packed bytes and FP32 metadata directly. Historical K/V are never reconstructed into
BF16 buffers. Six GQA query heads share each KV-head decode. A single stable FP32 softmax spans
the entire sequence, which is the mathematically equivalent tier merge and avoids a separate
merge dispatch.

The cache-backed tests used the existing D2.2 aging/page representation and required exact
BF16/INT2 payload and metadata preservation when creating GPU views. No-history, prefix/recent,
first-history, aging, partial-page, 512/2K/4K, forced 512→519, layer 3/35/63, and all 16
full-attention dispatch checks passed. Worst required-case rotated-AV relative-L2 was
`5.148370292e-06` at 4K under the `1e-4` gate; the D2.3 scalar-reader taps also passed. GDN,
legacy Q2, CPU fallback, and BF16 historical shadow dispatches were zero.

Complete mixed attention measured `0.208710/0.667861/1.109395/2.153588/4.373304/9.361144 ms`
at 512/2K/4K/8K/16K/32K, with `213.2–272.3x` speedup over the scalar mixed baseline. These
are diagnostic mixed-reader measurements, not real-model serving throughput. D4.2b is **PASS**;
D4.3 may connect the path to the real Qwen runtime. Evidence:
`results/oscar/phase-d4-2b-mixed-gpu-attention.md`.

Evidence: `results/oscar/phase-d4-1-profile.md`; live profile log SHA-256
`436560e22cd830993d748c51c5b03a77cf7f404f7b051991368dc1eeaca55a25`; microbenchmark log
SHA-256 `9f961107ff0e6118acbb899c1a5ff360bb1c16ff559baffeeb43419419d7d4a7`.

## Phase D4.3 — real Qwen GPU runtime integration (2026-09-02)

D4.3 connects the D4.2b mixed GPU reader to the actual Qwen3.8-27B runtime through the
explicit opt-in `oscar-int2-gpu` mode. The runtime supplies actual post-RoPE Q/K/V tensors,
uses the immutable C4 30K calibrated bank, appends rotated rows through the qualified typed
cache/aging path, dispatches `OscarInt2G128` historical data plus BF16 prefix/recent rows, and
recovers with the existing FP32 `R_V.T` path. The 48 GDN layers and all DFlash2/MTP/adaptive-K
paths remain untouched.

The 321/332/512 and 4K live/reference taps at layers 3/35/63 passed. Maximum relative-L2 over
the GPU-vs-qualified-reader taps was `2.52443e-6` for rotated AV and `2.50990e-6` after V
recovery, below the `1e-4` stage gate. The all-full-attention dispatch bitmap was
`1111111111111111`; GDN, legacy Q2, BF16 historical shadow, CPU serving fallback, and
non-finite output counts were zero/false. The scalar path is used only as an explicitly labelled
validation oracle.

The real-model verifier wall was `2.903730/2.776749/6.889555/486.346291 s` at 321/332/512/4K
fixed-input cases. At 4K, cumulative counters measured QKV rotation `0.590442 s`, actual
append/aging `398.977 s`, explicit typed-page staging `0.132383 s`, fused mixed GPU attention
`46.0793 s`, FP32 recovery `0.339310 s`, and complete full-attention `480.292 s`. The first
integration therefore has a dominant host append/aging cost plus a per-query launch loop; the
isolated D4.2b kernel numbers must not be substituted for this live-model result.

A four-case D3.1-style GPU smoke (arithmetic, coding, JSON, and a 1,148-token retrieval case)
achieved `4/4` objective success. A 16K exploratory run reached logical context 3,072 without
an attention failure, but was stopped because first-integration causal prefill scaling was not
healthy; 32K real-model timing and 4K/16K/32K model-fidelity qualification remain pending.

D4.3 is **PASS** for live GPU integration and authorizes D4.4. The single recommended D4.4
target is persistent GPU-resident incremental typed-page views with device-side recent-to-INT2
aging, eliminating per-invocation host flattening and most repeated aging/staging traffic.
Evidence: `results/oscar/phase-d4-3-live-gpu-runtime.md`.

## Phase D4.4 — GPU-resident incremental cache + device aging (2026-09-02)

D4.4 adds the explicit `oscar-int2-gpu-resident` mode. Each of the 16 full-attention layers
owns persistent device storage for the BF16 protected prefix, packed official `OscarInt2G128`
historical bulk, FP32 metadata, and BF16 recent ring. Actual rotated runtime K/V rows are
published once per append. When a recent row ages out, the device encoder reads it before ring
slot reuse, applies the unchanged `.96/.92` K/V clipping and G128 packing contract, and writes
the packed payload plus FP32 metadata directly to historical storage. The D4.2b mixed reader
consumes these arrays without flattening or reconstructing historical BF16 K/V.

The old `oscar-int2-gpu` staged path remains a comparison control. Performance-mode resident
telemetry reported `gpu_cache_staging_us=0`, `gpu_cache_staging_bytes=0`,
`legacy_q2_dispatched=false`, `bf16_historical_shadow=false`, and `fallback=false`. At the
final 4K-plus-forced-decode point, logical tiers were prefix/history/recent `64/3783/256`;
the persistent resident cache was `63,078,400` bytes with `789,504` bytes workspace. The only
incremental host/device traffic was `262,592` bytes of position metadata; runtime Q/K/V stayed
on device.

Validation compared device output with the qualified host codec for both K and V, all four KV
heads and both groups. It passed exact symbols, packed payload bytes and FP32 metadata for
`60,528` logical token checks, including repeated aging and ring-slot reuse. Live GPU/reference
attention passed at 321, 332, 512, 2K and 4K; the worst recovered relative-L2 was `1.40692e-6`
under the `1e-4` stage gate. The full-attention bitmap stayed
`1111111111111111`, and no GDN state entered OSCAR storage.

Clean real-model wall times were `1.505832/13.525424/46.450017 s` at 512/2K/4K for the
resident path, compared with D4.3 staged `6.889555/486.346291 s` at 512/4K. The 512→4K
request slope improved from approximately `133.8` to `12.54 ms` per added token. A real 16K
attempt reached the mixed regime but was stopped before a final result because the unchanged
per-query full-attention launch loop was still impractical. D4.4 is **PASS** for resident-cache
correctness and the requested scaling objective. D4.5 should batch or fuse that launch loop;
rotation precision, cache policy, codec mathematics, DFlash2/MTP, and adaptive-K remain fixed.

Evidence: `results/oscar/phase-d4-4-gpu-resident-cache.md`.

## Phase D4.V — full validation optimization (2026-09-02)

D4.V optimized the independent CPU full-validation oracle without changing the qualified OSCAR
mathematics, runtime cache, model execution, or any numerical gate. The validator is a separate
post-run reader of immutable live-reference snapshots, so the model run is not synchronously
stopped at every tap and no production cache state is shared with the oracle. It remains
fail-closed at relative-L2 `2e-6` and max-absolute `1e-4`, with finite/layout/topology/tier,
asset-identity, and truncation checks intact. No weaker FAST qualification path was introduced.

The target-local build is `RelWithDebInfo` with `/O2 /Ob3 /Oi /Ot /arch:AVX2 /fp:precise
/UNDEBUG`; assertions remain enabled. The implementation uses fixed-size rows, one INT2 K/V
decode per KV head reused across its six GQA query heads, reusable per-worker workspaces, AVX2
D256 product formation with scalar left-to-right reductions preserved for golden parity, and a
bounded deterministic worker pool of up to eight threads. Packed INT2 extraction remains direct
and scalar because its measured cost is small and the byte-interleaved layout does not justify a
different representation in the independent oracle. Tap snapshots already contain the
authoritative tier rows, so cross-tap state sharing was intentionally not enabled where it could
weaken independence; each snapshot is parsed and checked exactly once.

The unchanged D4.4 real archive contains 30 taps and passed 30/30. External wall time fell from
`869.854 ms` to `179.441 ms` (`4.85x`, `79.37%` reduction). The final profile's worker-summed
times were file parse/I/O inclusive decode `58,766.8 us`, QK `6,201.6 us`, Q rotation
`3,525.3 us`, inverse-V `3,607.0 us`, AV `2,723.8 us`, softmax `263.3 us`, and INT2 K/V
reconstruction `74.4/115.1 us`; these are aggregate work totals, not additive wall time. The
process-level counter reported `15,625 us` CPU over `11,819.2 us` in-process wall due to worker
overlap.

The permanent scalar-golden gate passed 21/21 over contexts `64,320,321,332,512,2048,4096`
and layers `3,35,63`. A deterministic all-16-layer fixture benchmark measured
`12.6115/41.9779/88.2372 ms` at 512/2K/4K. The fixture benchmark is clearly labelled as
synthetic: real tap files currently exist only for the 30-tap context-321 archive, so it is not
a substitute for future real-model 512/2K/4K tap capture. D4.V is **PASS**; FULL validation is
the normal development validator and the next optimization target is CPU QK traversal
batching/fusion around the retained GQA reuse. Evidence and hashes:
`results/oscar/full-validation-optimization.md`.
