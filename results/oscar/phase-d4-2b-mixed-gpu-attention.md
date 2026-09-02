# OSCAR D4.2b — Mixed-Tier GPU Attention

Date: 2026-09-02  
Status: **PASS — reference-correct and materially faster**

## Scope

This phase connects the qualified D4.2a historical INT2 traversal with the BF16 protected
prefix and BF16 recent regions. It remains a diagnostic/reference GPU path; it is not connected
to the real Qwen serving dispatcher yet. The C4 30K direction, official `OscarInt2G128` codec,
group size 128, prefix 64, recent 256, FP32 accumulation, and existing aging/cache policy are
unchanged.

The path ends at rotated-coordinate AV. Q rotation and `R_V.T` are intentionally outside this
kernel phase and remain on their already-qualified path; recovered attention is therefore not a
GPU-kernel output in this isolated gate. No DFlash2, MTP, adaptive-K, legacy Q2, or GDN path was
modified.

## Implementation

Added a separate mixed-tier API and CUDA implementation:

- `src/ops/softmax_attention/oscar_mixed/launch.h`
- `src/ops/softmax_attention/oscar_mixed/launch.cu`
- `tests/test_oscar_mixed_gpu_attention.cpp`
- `src/CMakeLists.txt` registration in `ninfer_ops`
- `tests/CMakeLists.txt` registration as `ninfer_oscar_mixed_gpu_attention_test`

The API consumes three logical regions:

```text
prefix K/V       [prefix, 4, 256]       raw BF16 bits
historical K/V   [history, 4, 64]       packed INT2 bytes
historical meta  [history, 4, 4]        FP32 [scale, zero] × 2
recent K/V       [recent, 4, 256]        raw BF16 bits
q_rotated        [24, 256]               FP32
```

The logical attention sequence is exactly `prefix → historical → recent`.

The GPU path uses one score kernel, one stable FP32 softmax kernel, and one AV kernel:

1. The score kernel assigns a 256-token logical tile to each 256-thread block per KV head.
   It branches by logical tier, converts BF16 only at use, decodes historical INT2 K directly,
   and feeds six Q-head accumulators for the owning KV head. No historical BF16 reconstruction
   or decoded-K buffer exists.
2. The softmax kernel reduces over the complete logical sequence, not independently per tier.
   This is the numerical merge: one global FP32 max/sum produces probabilities valid across all
   three regions. It avoids a separate partial-state merge kernel while remaining exactly
   equivalent to a correctly merged max/sum/AV formulation.
3. The AV kernel reads the same complete probability rows, converts BF16 prefix/recent values at
   use, decodes historical INT2 V directly, and accumulates six Q heads per KV head in FP32.
   No decoded-V buffer exists.

This design is intentionally simple for D4.2b and preserves the D4.2a historical kernel as a
separate component baseline. It uses 4×`ceil(total_tokens/256)` score blocks and four 256-thread
AV blocks, so it does not use a serial one-warp history scan.

## Reference and cache-backed validation

The test uses an independent scalar mixed reader that reconstructs only the same post-aging
representation: BF16 prefix, official INT2 historical rows, and BF16 recent rows. It compares
scores, stable softmax, and rotated AV. The existing D2.3 scalar
`OscarMixedAttentionReader` remains unmodified and was directly compared at cache-backed
contexts 321, 332, and 512 with identity rotations to isolate tier/cache addressing.

For cache-backed checks, `OscarMixedAgingLayerCache` is populated through its normal append and
aging path. The page/slot view is flattened into the GPU API's contiguous tier views only for
the diagnostic transfer. Exact vector equality was required for all BF16 payloads, INT2 packed
bytes, and FP32 metadata after flattening. This exercises the existing partial-page layout and
does not create a persistent decoded historical buffer.

The policy boundary fixtures were:

| Context | Prefix | Historical | Recent | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 64 | 64 | 0 | 0 | no-history / prefix-only |
| 65 | 64 | 0 | 1 | prefix-to-recent boundary |
| 320 | 64 | 0 | 256 | full recent window, no historical tier |
| 321 | 64 | 1 | 256 | first historical token |
| 322 | 64 | 2 | 256 | first aging/continuation boundary |
| 332 | 64 | 12 | 256 | partial historical page and all three tiers |
| 512 | 64 | 192 | 256 | required mixed smoke |
| 2,048 | 64 | 1,728 | 256 | required long history |
| 4,096 | 64 | 3,776 | 256 | required long history |

All logical positions were addressed once in order. At context 320→321 the recent boundary is
preserved while one row enters historical storage; at context 332 the historical region is a
partial page. No production-cache fallback, legacy Q2 dispatch, CPU fallback, GDN state, or
historical BF16 shadow was used.

## Correctness results

The fail-closed gate is `1e-4` relative L2 for scores, softmax, and rotated AV. All GPU values
were finite. The largest required-case relative L2 was `5.148370292e-06` for rotated AV at 4K.

| Context | Tier counts P/H/R | Score max abs | Score rel L2 | Softmax max abs | Softmax rel L2 | AV max abs | AV rel L2 | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 64/0/0 | `2.861022949e-06` | `1.448742211e-07` | `4.768371582e-07` | `3.529604555e-07` | `8.583068848e-06` | `3.696100066e-07` | PASS |
| 65 | 64/0/1 | `2.861022949e-06` | `1.450810601e-07` | `4.172325134e-07` | `3.475511221e-07` | `9.536743164e-06` | `3.637753991e-07` | PASS |
| 320 | 64/0/256 | `3.814697266e-06` | `1.501081783e-07` | `5.066394806e-07` | `4.879787525e-07` | `1.144409180e-05` | `5.025018481e-07` | PASS |
| 321 | 64/1/256 | `3.814697266e-06` | `1.501709903e-07` | `5.364418030e-07` | `5.051334142e-07` | `1.192092896e-05` | `5.177446383e-07` | PASS |
| 322 | 64/2/256 | `3.814697266e-06` | `1.500825988e-07` | `4.768371582e-07` | `4.743498891e-07` | `1.120567322e-05` | `4.937224958e-07` | PASS |
| 332 | 64/12/256 | `2.861022949e-06` | `1.496781010e-07` | `5.066394806e-07` | `4.728816236e-07` | `1.168251038e-05` | `4.901195894e-07` | PASS |
| 512 | 64/192/256 | `3.814697266e-06` | `1.551467506e-07` | `7.152557373e-07` | `7.163698115e-07` | `1.826882362e-05` | `7.329788900e-07` | PASS |
| 2,048 | 64/1,728/256 | `4.768371582e-06` | `1.578284241e-07` | `1.609325409e-06` | `2.389395604e-06` | `3.099441528e-05` | `2.360437747e-06` | PASS |
| 4,096 | 64/3,776/256 | `4.768371582e-06` | `1.586091400e-07` | `3.159046173e-06` | `5.091746061e-06` | `7.057189941e-05` | `5.148370292e-06` | PASS |

The D2.3 scalar reader comparisons passed at contexts 321, 332, and 512. The relative-L2
values were respectively `1.50e-7/5.05e-7/5.18e-7`, `1.50e-7/4.73e-7/4.90e-7`, and
`1.55e-7/7.16e-7/7.33e-7` for score/softmax/AV. Layers 35 and 63 also passed complete
three-tier contexts with sub-`6e-7` rotated-AV relative L2.

## Forced-decode and layer coverage

Eight deterministic forced continuation taps at contexts 512 through 519 used the same fixed
logical continuation and did not select new tokens from logits. Historical counts advanced from
192 through 199 while recent remained 256. Every tap passed; at context 519 the score/softmax/AV
relative-L2 values were `1.552508024e-07/7.238386956e-07/7.368444699e-07`, with no growth
pattern or addressing defect.

The all-layer fixture invoked the same mixed API for exactly the verified full-attention set:

```text
3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63
dispatch=1111111111111111
GDN dispatch=0
legacy Q2 dispatch=0
CPU fallback=0
```

The selected C4 identity is `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`; the GPU test
does not load or alter rotation files because Q is supplied at the already-rotated kernel
boundary. `R_V.T` recovery remains the existing downstream boundary and was not optimized here.

## Benchmark

CUDA-event timings below are for the complete mixed GPU path. The historical column reruns the
unchanged D4.2a historical-only kernel with the same history rows. The BF16-window column is a
separate mixed-kernel run with history count zero and the same prefix/recent data. `merge_ms` is
zero because this design performs one global softmax over all tiers and has no separate merge
launch; it is not an omitted timing.

The scalar comparison is the CPU implementation of the D2.3 mixed reader arithmetic over the
same post-aging rows. It is a diagnostic baseline, not a serving throughput claim.

Effective logical traffic is:

```text
49,152 + 4,672 × (prefix + recent) + 1,216 × historical bytes
```

The 4,672-byte BF16 coefficient includes K/V BF16 reads plus score/softmax/AV probability
traffic; 1,216 is the corresponding official INT2 K/V traffic. The metric is an effective
logical bandwidth estimate, not the RTX 5090 hardware roofline.

| Context | Mixed ms | Historical ms | BF16-window ms | Merge ms | Scalar mixed ms | Speedup | µs/history token | Effective GB/s | Mixed workspace |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 0.208710 | 0.076944 | 0.144109 | 0 | 44.491900 | 213.175281× | 1.087033 | 8.517371 | 1,581,056 B |
| 2,048 | 0.667861 | 0.351699 | 0.158765 | 0 | 151.762950 | 227.237399× | 0.386494 | 5.458383 | 2,859,008 B |
| 4,096 | 1.109395 | 0.733787 | 0.144075 | 0 | 296.992550 | 267.706716× | 0.293802 | 5.530768 | 4,562,944 B |
| 8,192 | 2.153588 | 1.491880 | 0.144624 | 0 | 586.493700 | 272.333280× | 0.273576 | 5.161871 | 7,970,816 B |
| 16,384 | 4.373304 | 2.943340 | 0.144784 | 0 | 1,185.353100 | 271.042930× | 0.272243 | 4.819701 | 14,786,560 B |
| 32,768 | 9.361144 | 5.815688 | 0.144528 | 0 | 2,328.265000 | 248.715860× | 0.288497 | 4.379909 | 28,418,048 B |

The mixed path is materially faster than the scalar mixed baseline at every measured context.
The BF16-only window cost remains approximately 0.144–0.159 ms across the long-context cases;
history scaling is dominated by the direct decode/QK/AV traversal. The mixed µs/history-token
value is not the D4.2a value: it is recomputed here using complete mixed attention.

## CUDA resources and build identity

The RTX 5090 reported compute capability 12.0. Final `cudaFuncGetAttributes` values were:

| Kernel | Registers/thread | Static shared/block | Max threads/block |
| --- | ---: | ---: | ---: |
| Mixed K decode + six-way QK | 38 | 6,144 B | 1,024 |
| Stable full-sequence softmax | 38 | 2,048 B | 1,024 |
| Mixed V decode + six-way AV | 40 | 0 B | 1,024 |

All launches use 256 threads/block. No tensor-core, TMA, warp-specialized, split-KV, CUDA-graph,
or decoder optimization was introduced.

Build command:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_oscar_mixed_gpu_attention_test -j 4"
```

The focused registered test passed:

```text
1/1 Test #18: ninfer_oscar_mixed_gpu_attention_test ... Passed 25.89 sec
100% tests passed, 0 tests failed out of 1
```

Final identities:

| File | SHA-256 |
| --- | --- |
| `src/ops/softmax_attention/oscar_mixed/launch.cu` | `A0078A86CDD0B219E38870305317E8B724BE79CC66D2E1396FE7F572550665C1` |
| `src/ops/softmax_attention/oscar_mixed/launch.h` | `D3381CDF9B55F60E6C6FFDA262B04331C141A3C1766E353505862F39570C4D91` |
| `tests/test_oscar_mixed_gpu_attention.cpp` | `949F7FADC435DE61929EA9CBD916AE09A301C607E4DE4F1DF21B3920C4A0C6A5` |
| `D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_mixed_gpu_attention_test.exe` | `B9049CBDBA8D8B6A7829F045D1B0C2BD66CDB9119F555334997D6975120B8A71` |

## Decision

**PASS.** The complete logical `BF16 prefix + official OscarInt2G128 historical bulk + BF16
recent` attention path is reference-correct, all three tiers execute on the SM120a GPU path,
and it is materially faster than the scalar mixed reader through 32K. The historical D4.2a
kernel remains unchanged and is measured separately.

D4.3 real Qwen runtime integration is authorized. That next phase must connect the mixed GPU
views to actual runtime page/slot storage, preserve Q rotation and `R_V.T`, and requalify full
model behavior; those operations were intentionally not performed here.
