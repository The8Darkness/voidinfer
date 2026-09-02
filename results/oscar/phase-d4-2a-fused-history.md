# OSCAR D4.2a — Fused Historical INT2 Attention Kernel

Date: 2026-09-02  
Status: **PASS — correctness-qualified and materially faster**

## Scope and gate

This phase implements and qualifies the first SM120a CUDA path for the historical OSCAR
INT2 region only. It does not integrate the BF16 protected prefix or recent window, and it
does not alter rotations, calibration, cache policy, DFlash2, MTP, adaptive-K, or the legacy
Q2 path. The input Q is already in the calibrated rotated coordinate system and historical K/V
are already encoded as official `OscarInt2G128` rows.

The pinned runtime direction is the immutable C4 asset
`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`. The kernel itself is deliberately independent
of the asset files because rotation is outside this historical-region contract.

Target topology:

| Property | Value |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 (`sm_120a`) |
| Q heads | 24 |
| KV heads | 4 |
| GQA ratio | 6 |
| Head dimension | 256 |
| Group size | 128 |
| INT2 groups/head | 2 |
| K/V metadata | 4 FP32 values per row (`[scale, zero]` × 2) |
| Packed payload | 64 bytes per D=256 row |
| Attention accumulation | FP32 |

## Implementation

Added the separate historical-only API and CUDA implementation:

- `src/ops/softmax_attention/oscar_history/launch.h`
- `src/ops/softmax_attention/oscar_history/launch.cu`
- `tests/test_oscar_history_attention.cpp`
- `src/CMakeLists.txt` registration in `ninfer_ops`
- `tests/CMakeLists.txt` registration as `ninfer_oscar_history_attention_test`

The API accepts contiguous arrays with these layouts:

```text
q_rotated  [24, 256]       FP32
k_packed   [history, 4, 64] uint8
k_metadata [history, 4, 4]  FP32
v_packed   [history, 4, 64] uint8
v_metadata [history, 4, 4]  FP32
scores     [24, history]    FP32 scratch
softmax    [24, history]    FP32 scratch
output     [24, 256]        FP32
```

The score kernel launches `4 × ceil(history / 256)` blocks of 256 threads. Each block owns
one KV head and one token tile; each live thread decodes its K row dimension-by-dimension and
feeds six Q-head dot products. The six Q heads sharing a KV head therefore reuse the decoded
K scalar without a decoded-K allocation. Q is staged in 6×256 FP32 shared values (6,144 bytes).

The softmax kernel launches one 256-thread block per Q head and performs stable FP32 max,
exponentiation, sum, and normalization. The AV kernel launches four 256-thread blocks; each
thread owns one output dimension, decodes each V scalar directly from packed bytes and FP32
metadata, and accumulates six GQA outputs in FP32. There is no persistent decoded K or V
buffer. The scores and softmax arrays are intentional attention scratch, not decoded cache
representations. The launch is tiled and multi-block rather than a serial one-warp full-history
scan.

The device decode uses the already-qualified D2.1 equations exactly: byte `j` contains symbols
for dimensions `j`, `j+64`, `j+128`, and `j+192`; group metadata is selected from dimension
`0..127` or `128..255`; reconstruction is `(symbol - zero) * scale`.

## Build and identity

Build command:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_oscar_history_attention_test -j 4"
```

The final build succeeded with the configured `compute_120a,sm_120a` target. The focused CTest
also passed:

```text
1/1 Test #17: ninfer_oscar_history_attention_test ... Passed 20.64 sec
100% tests passed, 0 tests failed out of 1
```

Final source and executable identities:

| File | SHA-256 |
| --- | --- |
| `src/ops/softmax_attention/oscar_history/launch.cu` | `7E08CCEAD62D7808463312D8880F56F56B8070B3125D97113FC226D416F96F56` |
| `src/ops/softmax_attention/oscar_history/launch.h` | `30AC1CDF07C6825F942B1D48ED9242B1AFC13EC255A4EA3E5BCDB1B81EF9B812` |
| `tests/test_oscar_history_attention.cpp` | `A7FCDAF9B1C93C57764A734BDEF23ACA1AACCCAC96694DA1646CA7363DC4A32C` |
| `D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_history_attention_test.exe` | `96A62B91DC7B31338FD2CA46C732287C03ADCE1883A96B13A65E9A7BEF29EF65` |

The executable was run directly with:

```powershell
D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_history_attention_test.exe
```

The test constructs deterministic finite rows through the official host `OscarInt2G128`
encoder, transfers only packed bytes and FP32 metadata to the GPU, and compares against an
independent host decoder/attention oracle. It covers the requested D=256, two-group layout.
The existing scalar D2.3a `OscarMixedAttentionReader` was not modified and remains the oracle
for the later mixed-cache integration; this direct historical oracle isolates kernel arithmetic
from page/tier dispatch during D4.2a.

## Correctness results

The scalar oracle performs the same historical-only causal computation in FP32: packed K decode,
QK, stable softmax, packed V decode, and AV. The fail-closed parity threshold is `1e-4` relative
L2 for each output family. Every required size passed; the largest observed relative-L2 was
`1.089803277e-06` for AV at 4K.

| History tokens | Score max abs | Score rel L2 | Softmax max abs | Softmax rel L2 | AV max abs | AV rel L2 | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | `9.536743164e-07` | `6.722039103e-08` | `1.117587090e-08` | `2.534168573e-07` | `4.768371582e-07` | `1.897782482e-07` | PASS |
| 512 | `9.536743164e-07` | `6.472848924e-08` | `1.676380634e-08` | `4.384841077e-07` | `1.490116119e-06` | `4.231813762e-07` | PASS |
| 2,048 | `1.430511475e-06` | `6.449688073e-08` | `1.722946763e-08` | `7.032439271e-07` | `2.026557922e-06` | `6.533650208e-07` | PASS |
| 4,096 | `1.430511475e-06` | `6.478672532e-08` | `7.683411241e-09` | `1.105956699e-06` | `2.145767212e-06` | `1.089803277e-06` | PASS |

No NaN/Inf was observed. The GPU score/softmax/AV outputs matched the independent decoded
representation within ordinary FP32 operation-order noise; no codec, group, GQA, or boundary
mismatch was found.

## Benchmark results

The scalar comparison is the CPU historical-only oracle, not the D4.1 full mixed-cache model
path. GPU timings are CUDA-event averages after a warm-up. CPU repetitions were 2 through 4K
and 1 above 4K; GPU repetitions were 30/30/10/10/5/5/3 for 128/512/2K/4K/8K/16K/32K.

`effective_GBps` uses the explicit traffic estimate
`1216 * history + 49,152` bytes:

- score stage: K payload/metadata plus score writes and one staged Q load;
- softmax stage: score read, unnormalized probability write, and normalized probability read/write;
- AV stage: V payload/metadata, probability reads, and output writes.

This is an effective logical traffic metric, not a hardware bandwidth ceiling.

| History | GPU ms | CPU scalar ms | Speedup | GPU µs/history token | Effective GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.056451 | 6.718250 | 119.009867× | 0.441025 | 3.627912 |
| 512 | 0.161510 | 26.881900 | 166.440671× | 0.315450 | 4.159138 |
| 2,048 | 0.401674 | 108.529700 | 270.193757× | 0.196130 | 6.322347 |
| 4,096 | 0.801082 | 218.353200 | 272.572994× | 0.195577 | 6.278871 |
| 8,192 | 1.444493 | 435.858200 | 301.737880× | 0.176330 | 6.930200 |
| 16,384 | 2.968627 | 890.038600 | 299.814881× | 0.181191 | 6.727721 |
| 32,768 | 6.012608 | 1,785.529900 | 296.964285× | 0.183490 | 6.635230 |

The path is materially faster than the scalar historical oracle at every measured size. The
per-history-token cost falls from `0.441025 µs` at 128 tokens to `0.183490 µs` at 32K as fixed
launch overhead is amortized. The benchmark remains a correctness-first SIMT implementation;
the effective bandwidth values are intentionally far below a tuned SM120a memory roofline.

## Kernel resources

`cudaFuncGetAttributes` on the final executable reported:

| Kernel | Registers/thread | Static shared memory/block | Max threads/block |
| --- | ---: | ---: | ---: |
| Fused K decode + six-way QK | 32 | 6,144 bytes | 1,024 |
| Stable FP32 softmax | 38 | 2,048 bytes | 1,024 |
| Fused V decode + six-way AV | 40 | 0 bytes | 1,024 |

The launches use 256 threads/block. The score and AV kernels exploit GQA sharing, but no tensor
core, warp-specialized, TMA, split-history reduction, or persistent scheduling optimization was
introduced in this phase.

## Decision and next phase

**PASS.** The historical-only `OscarInt2G128` SM120a path is reference-correct at 128, 512, 2K,
and 4K, benchmarks through 32K, and is materially faster than the retained scalar historical
oracle. It satisfies the no-persistent-decoded-K/V requirement and preserves the official
packed layout and FP32 metadata semantics.

D4.2b is authorized to connect this qualified historical kernel to the BF16 prefix/recent
regions and the mixed-cache attention contract. That integration, end-to-end model fidelity,
and further kernel optimization are intentionally not part of D4.2a.
