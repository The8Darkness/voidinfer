# OSCAR Phase D2.1 — official INT2 quantizer / codec parity

Date: 2026-09-01  
Status: **PASS**  
Scope: deterministic CPU/reference codec and parity only. No live attention integration, mixed
BF16 windows, throughput benchmark, recalibration, DFlash2/MTP change, adaptive-K change, or
CUDA-kernel optimization was performed.

## Authority and pinned source

The reference is the serving-side clipped INT2 implementation in the official
[FutureMLS-Lab/OSCAR repository](https://github.com/FutureMLS-Lab/OSCAR), not the older
`rotation/compute_kv_rotation.py` fitter helper's illustrative `simulate_int2_asym` routine.
The pinned repository commit is:

```text
41ebcdba3db5f0ce1339c3727caea80df575d437
```

The exact source used for the equations is the immutable
[oscar_rotation_clip_int2_kv.py source at that commit](https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/41ebcdba3db5f0ce1339c3727caea80df575d437/sglang-research/python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py),
SHA-256:

```text
c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157
```

The official source uses a row-wise absolute-value threshold, symmetric clipping, grouped
asymmetric affine quantization, and a fused quartered INT2 pack. The generic upstream grouped
KV kernels were also checked for the same byte order: for D=256, byte `j` carries dimensions
`j`, `j+64`, `j+128`, and `j+192`.

## Official contract reproduced

The D2.1 target is D=256, two independent groups of 128 dimensions, three quantization levels
(`0..3`), and FP32 metadata. For one finite row:

1. `clip_index = int(clip_ratio * 256)`, clamped to `0..255`.
2. `threshold = sort(abs(row))[clip_index]`; clamp the row symmetrically to
   `[-threshold, threshold]`.
3. For each independent group `[0..127]` and `[128..255]`, compute
   `scale = max(group_max - group_min, 1e-8) / 3` and
   `zero_point = -group_min / scale` in FP32.
4. Compute the symbol as the truncated non-negative value
   `uint8(value / scale + zero_point + 0.5)`, requiring `0..3`.
5. Decode as `(symbol - zero_point) * scale` in FP32.
6. Store metadata as `[scale_0, zero_point_0, scale_1, zero_point_1]` and pack byte `j` as
   `q[j] | q[j+64]<<2 | q[j+128]<<4 | q[j+192]<<6`.

The configured K/V clips are .96/.92, giving clip indices 245/235 for D=256. The generated
golden manifest records every input row, clipped row, threshold, FP32 metadata vector, symbol
stream, packed bytes, and decoded row in the binary fixture.

## Existing Q2 audit

The existing implementation was intentionally not changed. It remains a separate experimental
control in `src/ops/kv_cache/oscar_codec.cuh`, `src/core/paged_kv_cache.cpp`, and the full-D256
OSCAR append path in `src/ops/kv_cache/append/kernel.cuh`.

| Property | Existing experimental Q2 | Official OSCAR D2.1 | Compatible? |
| --- | --- | --- | --- |
| Clipping | Row-wide min/max; center/half-span shrink with K=.93 and V=.91 | Sorted absolute-value threshold; symmetric clamp; K=.96 and V=.92 | No |
| Scale | One row-wide span divided by 3 | Independent span divided by 3 for each 128-wide group | No |
| Asymmetric range | Stores lower-endpoint `zero`; decodes `q*scale + zero` | Stores affine zero-point; decodes `(q-zero_point)*scale` | No |
| Symbol mapping | 2-bit `0..3`, rounded/clamped from normalized lower-endpoint coordinates | 2-bit `0..3`, truncation after `+0.5` zero-point expression | No |
| D=256 traversal | Full-D256 reduction produces one row-wide parameter pair; transposed Q2 bytes use a `+32` lane arrangement, while the alternate route is contiguous | Two independent groups; quarter-interleaved byte dimensions `j,j+64,j+128,j+192` | No |
| Metadata | BF16 `[scale, zero]`, two values per row | FP32 `[scale_0, zero_0, scale_1, zero_1]`, four values per row | No |
| Public grouping | The public D256 profile labels `quant_group=128`, but its old `scale_extent=2` is metadata-item extent, not two official group parameter pairs | Exactly two independently quantized groups of 128 | No |
| Payload size | 64 bytes for D=256 Q2 | 64 bytes for D=256 Q2 | Payload-only match |

The decisive source locations for this audit are `oscar_codec.cuh:117-180,219-238`,
`paged_kv_cache.cpp:274-310`, `append/kernel.cuh:346-438`, and `d256_profile.h:9-50`.
No existing benchmark/profile was silently retargeted.

## Implementation

Added a distinct host/reference-direction codec:

```text
src/ops/kv_cache/oscar_int2_g128.h:15-39
src/ops/kv_cache/oscar_int2_g128.cpp:12-101
```

`OscarInt2G128EncodedRow` exposes clipped values, four FP32 metadata values, 256 unpacked
symbols, 64 packed bytes, and a deterministic FP32 decoder. The implementation rejects wrong
widths, null pointers, non-finite input, invalid clip ratios, and out-of-range symbols. It is
not wired into the live cache or attention path.

Build registration and test:

```text
src/CMakeLists.txt:112-113
tests/CMakeLists.txt:267-269
tests/ops/test_oscar_int2_g128.cpp:19-193
```

The parity test reads a versioned binary fixture, compares clipping exactly, requires exact
symbol and packed-byte equality, checks FP32 metadata/decode within `2e-6`, rejects non-finite
decode values, and emits a JSON report.

## Golden fixtures and provenance

The reproducible generator is:

```text
tools/oscar/generate_d2_1_int2_golden.py:1-338
```

It first validates the B2 manifest sidecar and selected source payload hashes, then reads two
tokens across all four KV heads from the real B2 post-RoPE BF16 capture at layers 3, 35, and 63,
and applies the validated C4 K/V rotations in FP32. It also produces synthetic edge cases:

| Fixture | Rows | Clip | Coverage |
| --- | ---: | ---: | --- |
| `random_finite` | 3 | K .96 | deterministic seeded finite values |
| `positive_only` | 1 | K .96 | positive-only range |
| `negative_only` | 1 | K .96 | negative-only range |
| `mixed_sign` | 1 | K .96 | mixed-sign range |
| `near_zero_range` | 1 | K .96 | range below the `1e-8` scale floor |
| `clipping_sensitive_outliers` | 1 | K .96 | explicit large outliers |
| `exact_group_boundary` | 1 | K .96 | distinct dimensions 0..127 and 128..255 |
| `multiple_groups` | 3 | V .92 | multiple independent groups/rows |
| `page_fragment_like_rows` | 3 | V .92 | independently ranged rows with no state carry |
| `real_layer_3_k_rotated` / `_v_rotated` | 8 each | K/V | two tokens × four KV heads |
| `real_layer_35_k_rotated` / `_v_rotated` | 8 each | K/V | two tokens × four KV heads |
| `real_layer_63_k_rotated` / `_v_rotated` | 8 each | K/V | two tokens × four KV heads |

Fixture output:

```text
results/oscar/d2-1-int2-fixtures/golden.bin
  cases: 15
  rows: 63
  bytes: 215,506
  SHA-256: a0d2bba734fcefde2999542bba559dd338487270235e1c462dcb7392eac98bfe
results/oscar/d2-1-int2-fixtures/manifest.json
  SHA-256: 590640719a51723c341dea3de194ba90fc85815866ef1099371f8173759e5a15
```

The manifest also records the B2 capture manifest SHA-256
`df0319563e91cc71a6dcc16a5f34f167da47931566f4ae94fc25f246a8425a89`, the selected K/V
payload hashes, C4 rotation `.pt` hashes, model SHA-256
`6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e`, and all synthetic/real
case input and packed-stream hashes.

## Exact parity results

The Python CPU mirror of the pinned official equations generated the golden values. The new
VoidInfer scalar C++ codec was built in `D:\AI\build-adaptive-dflash2` and compared against
those values.

Exact command sequence (PowerShell):

```powershell
$py='D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\python\python-3.12.7-x64-1\python.exe'
$env:PYTHONPATH='D:\AI\tools\oscar-calibration\.venv\Lib\site-packages'
& $py tools\oscar\generate_d2_1_int2_golden.py `
  --output-dir results\oscar\d2-1-int2-fixtures `
  --capture-manifest results\oscar\captures\phase-b2-qkv-256\manifest.json `
  --rotation-dir results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_int2_g128_test.exe' `
  --fixture 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d2-1-int2-fixtures\golden.bin' `
  --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\d2-1-int2-fixtures\cpp-parity.json'
& $py tools\oscar\validate_d2_1_int2_parity.py `
  --manifest results\oscar\d2-1-int2-fixtures\manifest.json `
  --cpp-report results\oscar\d2-1-int2-fixtures\cpp-parity.json `
  --fixture results\oscar\d2-1-int2-fixtures\golden.bin `
  --generator tools\oscar\generate_d2_1_int2_golden.py `
  --capture-manifest results\oscar\captures\phase-b2-qkv-256\manifest.json `
  --rotation-dir results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1
```

The calibration venv's launcher currently points to a removed base Python 3.12.10 executable.
The command above uses the bundled Python 3.12.7 interpreter with the unchanged calibration
venv `site-packages` directory, which contains `torch 2.13.0+cpu`. The only warning was the
known missing NumPy warning from this minimal CPU environment; the generator and parity tests
do not use NumPy. This execution detail is recorded so the run is reproducible rather than
silently claiming a working venv launcher.

Result:

```text
OscarInt2G128 parity: PASS cases=15 rows=63
max_scale_abs=0.000000e+00
max_decoded_abs=0.000000e+00
max_decoded_rel_l2=0.000000e+00
```

Fail-closed validation is in `tools/oscar/validate_d2_1_int2_parity.py:1-100` and passed source
manifest, selected payload, fixture, C4 rotation, C++ report, case-count, row-count, and
threshold checks. The C++ report is
`results/oscar/d2-1-int2-fixtures/cpp-parity.json`, SHA-256
`2def4a15c27f400a20942a900bcfaff700dc43e5854cccbf27638db489d017e9`.

### Real rotated Qwen3.8 K fixtures

| Layer | Rows | Clip | Symbols | Packed bytes | Max metadata abs | Max decoded abs | Rel. L2 |
| ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| 3 K | 8 | .96 / index245 | exact | exact | 0 | 0 | 0 |
| 3 V | 8 | .92 / index235 | exact | exact | 0 | 0 | 0 |
| 35 K | 8 | .96 / index245 | exact | exact | 0 | 0 | 0 |
| 35 V | 8 | .92 / index235 | exact | exact | 0 | 0 | 0 |
| 63 K | 8 | .96 / index245 | exact | exact | 0 | 0 | 0 |
| 63 V | 8 | .92 / index235 | exact | exact | 0 | 0 | 0 |

The complete per-row thresholds and packed-stream hashes are in the golden manifest. The
fixture uses independent encode calls per token/head row, so no scale, clipping, or pack state
can carry across group, token, KV-head, or page-fragment-like boundaries. D=256 is explicitly
split at dimensions 0..127 and 128..255.

## Storage accounting

For one token and one KV head at D=256:

| Component | Size |
| --- | ---: |
| 2-bit payload | 64 bytes = 512 bits |
| FP32 metadata: 4 values | 16 bytes |
| Standalone codec record | 80 bytes |
| Alignment/padding in this record | 0 bytes; payload and metadata naturally align to 16-byte boundaries |

This is 2.0 payload bits/value plus 0.5 metadata bits/value = **2.5 actual standalone
bits/value**, or `80/256 = 0.3125` bytes/value. A K+V pair is 160 bytes per token per KV head;
four KV heads are 640 bytes per token per full-attention layer before any page allocator,
fragmentation, BF16 windows, or slot-table overhead. Those omitted costs belong to D2.2.

## Verdict

**PASS.** The distinct `OscarInt2G128` codec matches the pinned official reference for clipping,
two-group traversal, FP32 metadata, affine zero-point semantics, symbols, quarter-interleaved
packed bytes, and FP32 dequantization across synthetic edge cases and real rotated Qwen3.8
layer 3/35/63 K/V samples. The legacy experimental Q2 path remains a separate control.

The project now has a codec-qualified basis for D2.2 mixed-window/live-cache work. D2.2 may
begin; live OSCAR attention and throughput remain intentionally out of scope here.
