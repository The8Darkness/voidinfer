# OSCAR Phase D2.2b — token aging: BF16 recent to OSCAR INT2

Date: 2026-09-01  
Status: **PASS**  
Scope: deterministic recent-to-historical conversion and typed-page validation only. No live
INT2 attention, token promotion in the production decoder, CUDA optimization, DFlash2/MTP,
adaptive-K, recalibration, or legacy Q2 change was performed.

## Contract and implementation

The D2.2a physical layout is retained. The new diagnostic aging bundle is implemented in:

```text
src/core/oscar_mixed_cache_layout.h:1-273
src/core/oscar_mixed_cache_layout.cpp:1-931
tests/test_oscar_mixed_cache_aging.cpp:1-341
tests/CMakeLists.txt:78-80
```

`OscarMixedAgingCacheBundle` contains the verified sixteen full-attention layers only:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

The constructor validates the selected C4 asset contract before accepting rows:

| Contract field | Verified value |
| --- | --- |
| Asset identity | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| C4 runtime asset-manifest SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| Rotation mode | `qqt_sst+r_h_pbr` |
| Calibrated | `true` |
| Topology | 64 layers, 24 Q heads, 4 KV heads, GQA 6, D=256, rotary D=64 |
| INT2 contract | group size 128, two groups, official D2.1 codec |

The input rows are explicitly the already-rotated BF16 K/V coordinate system selected by this
C4 contract. No rotation fitting or attention is performed in D2.2b. The legacy fixed-Hadamard
and experimental Q2 paths are not referenced.

For each append, the slow reference path requires the next logical token and four BF16 K/V heads
of D=256. Once the new context makes a token leave the recent window, `age_token()`:

1. selects only the current oldest non-prefix recent token;
2. decodes its stored BF16 K/V rows to FP32;
3. applies the validated official D2.1 codec independently to K with clip `.96` and V with
   clip `.92`;
4. stores the resulting packed symbols and four FP32 metadata values per K/V row in the typed
   INT2 page; and
5. clears the source BF16 record and marks the token as aged exactly once.

Pages are rebuilt slowly after each append. Each page contains one representation and retains the
D2.2a page/slot metadata, so the historical record cannot be mistaken for a BF16 recent record.
No attention-facing API or production cache dispatch was added.

## Deterministic boundary sequence

The test appends forced deterministic rows for logical token IDs `0..323`, checking tier roles and
addressing after every append:

| Event | Context after append | Result |
| --- | ---: | --- |
| append token 63 | 64 | token 63 remains protected-prefix BF16 |
| append token 64 | 65 | token 64 is BF16 recent |
| append through token 319 | 320 | 64 prefix + 256 recent; no historical token |
| append token 320 | 321 | token 64 ages into INT2 historical |
| append token 321 | 322 | token 65 ages |
| append token 322 | 323 | token 66 ages |
| append token 323 | 324 | token 67 ages |

At every append, every logical position resolves to exactly one slot. The expected policy is:

```text
prefix     = [0,64)                 -> BF16, never aged
historical = [64, context-256)      -> OSCAR INT2 G128
recent     = [max(64,context-256), context) -> BF16
```

The final context is therefore `prefix=64`, `historical=4`, `recent=256`.

## Exact codec parity

For each aged row, the test independently reconstructs the original BF16-decoded FP32 row and
calls the D2.1 `OscarInt2G128` reference-direction codec with the required K/V clip. It compares
the result against the aging record and the rebuilt physical page.

Coverage:

| Layer | KV heads | K clip | V clip | Clipped values | FP32 metadata | Symbols | Packed bytes |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 3 | 4 | .96 | .92 | exact | exact | exact | exact |
| 35 | 4 | .96 | .92 | exact | exact | exact | exact |
| 63 | 4 | .96 | .92 | exact | exact | exact | exact |

The all-layer coverage check confirms all 16 full-attention layers execute the same four aging
conversions. The machine-readable result records `symbols=exact`, `packed_bytes=exact`, and
`fp32_metadata=exact`; no mismatch was observed.

## Fail-closed and no-double-conversion checks

The test passed all of these guards:

- altered/legacy asset identity rejected before cache construction;
- a GDN layer rejected from the aging bundle;
- token 63 rejected as an aging target and retained as prefix BF16;
- a second conversion attempt for already-historical token 67 rejected;
- every historical token has no BF16 payload;
- every non-historical token has BF16 payload and no historical marker;
- historical conversion count equals the number of historical logical positions;
- no logical hole, overlap, disappeared row, or simultaneous INT2/recent residency;
- K/V storage types agree for every resolved slot;
- no GDN state is represented in the bundle.

## Per-transition physical accounting

The following are physical bytes across all 16 full-attention layers. INT2 values are allocated at
the full 64-token page capacity, so the INT2 byte totals remain constant through these first four
aging events even though the logical historical count increases.

| Append token | Aged token | Context | Prefix tokens | Historical tokens | Recent tokens | BF16 prefix bytes | BF16 recent bytes | INT2 payload bytes | INT2 metadata bytes | Page headers | Slot table | Mixed total |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 320 | 64 | 321 | 64 | 1 | 256 | 4,194,304 | 16,777,216 | 524,288 | 131,072 | 5,376 | 246,528 | 21,878,784 |
| 321 | 65 | 322 | 64 | 2 | 256 | 4,194,304 | 16,777,216 | 524,288 | 131,072 | 5,376 | 247,296 | 21,879,552 |
| 322 | 66 | 323 | 64 | 3 | 256 | 4,194,304 | 16,777,216 | 524,288 | 131,072 | 5,376 | 248,064 | 21,880,320 |
| 323 | 67 | 324 | 64 | 4 | 256 | 4,194,304 | 16,777,216 | 524,288 | 131,072 | 5,376 | 248,832 | 21,881,088 |

The final accounting is:

```text
prefix BF16 physical bytes:       4,194,304
recent BF16 physical bytes:      16,777,216
INT2 packed payload bytes:          524,288
INT2 FP32 metadata bytes:            131,072
page-header bytes:                     5,376
slot-table bytes:                    248,832
mixed total bytes:                21,881,088
```

The standalone D2.1 record remains 160 bytes per token/KV head (128 payload + 32 FP32 metadata),
or 2.5 bits/value including metadata. The totals above expose page rounding and slot/header
overhead rather than reporting only the theoretical 2-bit payload. Container allocator overhead
is not included because it is implementation-dependent and not measurable from this typed
representation.

## Reproduction

Build the isolated test in the latest build tree:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_oscar_mixed_cache_aging_test -j 4"
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_mixed_cache_aging_test.exe' `
  --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-2b-aging-validation.json'
```

Observed result:

```text
OscarMixedCache aging: PASS
asset=qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1 calibrated=true
boundary: token 63 prefix, token 64 recent, append token 320 ages 64
aged tokens: 64,65,66,67; all 16 full-attention layers
exact K/V codec parity: layers 3,35,63; all four KV heads
final tiers: prefix=64 historical=4 recent=256
final bytes: bf16_prefix=4194304 bf16_recent=16777216 int2_payload=524288 int2_metadata=131072
```

Machine-readable validation: `results/oscar/phase-d2-2b-aging-validation.json`, SHA-256
`a702f53e4ea2d0c0120cb6d80312d25a545ba478e7a75f41830450ca00292477`.

## Decision

**PASS.** Deterministic BF16 recent tokens aged into the official calibrated OSCAR INT2 G128
representation with exact K/V codec parity, correct page and slot addressing, protected-prefix
preservation, all-16-layer coverage, and no double conversion. The implementation stops before
live attention integration; the next authorized phase may qualify mixed INT2/BF16 attention.
