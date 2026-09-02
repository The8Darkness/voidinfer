# OSCAR Phase D2.2a — mixed BF16 / INT2 cache representation

Date: 2026-09-01  
Status: **PASS**  
Scope: typed physical representation, static construction, boundary validation, and storage
accounting. No token aging, promotion/demotion, live attention, CUDA optimization, DFlash2/MTP,
adaptive-K, or change to the validated `OscarInt2G128` mathematics was performed.

## Policy and design

The logical policy is explicit and non-overlapping:

```text
prefix       = [0, min(context_tokens, 64))                         -> BF16
recent_begin = max(64, context_tokens - 256) for context_tokens >64 -> recent BF16
historical   = [prefix_end, recent_begin)                            -> OSCAR INT2 G128
recent       = [recent_begin, context_tokens)                        -> BF16
```

The protected prefix takes precedence when the context is shorter than 320 tokens. Thus a
320-token context has 64 prefix + 256 recent tokens and no historical bulk; token 321 is the
first context with one historical token. Regions have independent physical page pools, so a
recent-window transition may occur inside a logical page without putting two physical formats
in the same page. Each physical page itself has exactly one representation.

The exact Qwen3.8 full-attention layer set is enforced:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

There is no GDN state type or GDN layer in this KV bundle. GDN recurrent state remains in its
existing state store.

## Typed storage and metadata

Implementation:

```text
src/core/oscar_mixed_cache_layout.h:1-163
src/core/oscar_mixed_cache_layout.cpp:1-401
```

The new types are independent of the legacy `OscarKVLayout`/experimental Q2 path:

| Type | Physical contents |
| --- | --- |
| `OscarMixedBFloat16PageStorage` | Separate K/V BF16 bit-unit vectors, full 64-token page capacity, 4 KV heads × D256 |
| `OscarMixedInt2G128PageStorage` | Separate K/V 64-byte packed payload vectors and K/V four-FP32-metadata vectors, full 64-token page capacity |
| `OscarMixedPageStorage` | Explicit `std::variant` of the two physical formats |
| `OscarMixedPage` | Page metadata, occupied slot metadata, and exactly one typed storage variant |
| `OscarMixedLayerCache` | Region-pool pages plus logical-token resolver for one verified full-attention layer |
| `OscarMixedCacheBundle` | Sixteen layer caches; rejects incomplete/noncanonical layer lists and enforces one policy across layers |

`OscarMixedPageMetadata` and `OscarMixedSlotMetadata` both carry:

- model/full-attention layer;
- sequence ID;
- logical token begin/end;
- region-local physical token begin/end;
- physical page index and slot offset;
- independent K and V storage type fields;
- layout version `1`;
- group size `128`;
- protected-prefix, historical-bulk, or recent-window role.

Validation requires K/V types to agree, requires the role/type pair to be valid, checks the
variant against the metadata, and rejects logical holes, overlaps, wrong page ranges, wrong
layer identity, wrong sequence, wrong layout version, and wrong group size. BF16 pages carry the
same policy group-size field for uniform metadata; no INT2 quantization is applied to BF16 rows.

For every region, physical token numbering is local to that region's pool. The explicit role,
storage type, and physical page index make this unambiguous and avoid deriving representation
from an environment variable.

## Static construction test

The deterministic test is:

```text
tests/test_oscar_mixed_cache_layout.cpp:1-170
```

Build and run:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_oscar_mixed_cache_layout_test -j 4"
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_mixed_cache_layout_test.exe' `
  --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-2a-layout-validation.json'
```

The 500-token construction contains:

```text
prefix:     logical 0..63    (64 BF16 tokens)
historical: logical 64..243  (180 OSCAR INT2 G128 tokens)
recent:     logical 244..499 (256 BF16 tokens)
```

The test passed all of the following:

- position 63 resolves to protected BF16 and position 64 to historical OSCAR INT2;
- positions 243 and 244 resolve to historical INT2 and recent BF16 respectively;
- position 499 resolves to recent BF16;
- every logical position `0..499` resolves exactly once with `[token, token+1)` metadata;
- no logical hole or overlap;
- K and V representation types agree at every slot;
- all 16 full-attention layers use the same region policy and page count;
- no GDN layer/state is present;
- page metadata is 56 bytes and slot metadata is 48 bytes;
- the static JSON report parses successfully.

Result:

```text
OscarMixedCache static construction: PASS
logical positions: 0..499 exactly once; boundaries 63/64 and 243/244 pass
K/V type agreement: pass; all 16 full-attention layers share policy
GDN state: absent from typed KV bundle
metadata sizes: page=56 slot=48 bytes
```

The archived machine-readable result is
`results/oscar/phase-d2-2a-layout-validation.json`, SHA-256
`fb158c93b4f9e5b6a9c1999392928c4318df3ec13674b572b245d7afb40a2ce7`.

## Physical storage accounting

All page storage is allocated at the 64-token physical-page capacity, including a partial final
page of a region. This exposes page rounding rather than hiding it in a theoretical 2-bit number.

Per full-attention layer and page:

| Page type | K/V physical storage | Bytes/page |
| --- | --- | ---: |
| BF16 prefix or recent | `64 × 4 × 256 × 2 bytes` for K plus the same for V | 262,144 (256 KiB) |
| OSCAR INT2 historical | K/V packed payload `64 × 4 × 64` plus K/V metadata `64 × 4 × 4 × FP32` | 40,960 (40 KiB) |

The D2.1 INT2 row remains exactly 64 payload bytes + 16 FP32 metadata bytes = 80 bytes for
one K or V row, or 160 bytes for one K/V pair per KV head. Its standalone accounting is
2.5 bits/value including metadata; payload-only is 2.0 bits/value.

The exact data cost per token and KV head is 1,024 bytes for a BF16 K/V pair
(`2 × 256 × 2`), and 160 bytes for an INT2 K/V pair (`2 × (64 payload + 16 FP32 metadata)`).
No extra alignment bytes are inserted between vector elements or rows; the 56-byte page and
48-byte slot sizes already include their C++ struct padding, and page/slot overhead is counted
separately below.

For the 500-token, 16-layer construction:

| Component | Bytes |
| --- | ---: |
| Logical BF16 token data | 20,971,520 |
| Logical INT2 packed payload | 1,474,560 |
| Logical INT2 FP32 metadata | 368,640 |
| Physical BF16 page storage | 20,971,520 |
| Physical INT2 packed payload | 1,572,864 |
| Physical INT2 metadata | 393,216 |
| Explicit alignment/padding beyond element widths and `sizeof` metadata | 0 |
| Page headers (`128 × 56`) | 7,168 |
| Slot table (`8,000 × 48`) | 384,000 |
| Historical bulk total, including INT2 pages/headers/slots | 2,107,008 |
| Total mixed cache | **23,328,768** |

The 500-token context therefore has 8 pages/layer: 1 BF16 prefix, 3 INT2 historical, and 4
BF16 recent; across 16 layers this is 80 BF16 pages and 48 INT2 pages. The total mixed-cache
effective size is 1.423875 bytes/value, or **11.391 bits/value**, after page headers and slot
metadata. The historical-bulk total is 0.3572265625 bytes/value, or **2.8578125 bits/value**,
including its page/slot overhead and the partially occupied final region page.

The test also computes these representative context sizes across all 16 full-attention layers:

| Context | Prefix | Historical | Recent | Pages | BF16/INT2 pages | Total bytes | Mixed bytes/value | Mixed bits/value |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 64 | 0 | 0 | 16 | 16 / 0 | 4,244,352 | 2.0238647 | 16.190918 |
| 65 | 64 | 0 | 1 | 32 | 32 / 0 | 8,440,320 | 3.9627404 | 31.701923 |
| 320 | 64 | 0 | 256 | 80 | 80 / 0 | 21,221,760 | 2.0238647 | 16.190918 |
| 321 | 64 | 1 | 256 | 96 | 80 / 16 | 21,878,784 | 2.0800234 | 16.640187 |
| 384 | 64 | 64 | 256 | 96 | 80 / 16 | 21,927,168 | 1.7426147 | 13.940918 |
| 500 | 64 | 180 | 256 | 128 | 80 / 48 | 23,328,768 | 1.4238750 | 11.391000 |
| 512 | 64 | 192 | 256 | 128 | 80 / 48 | 23,337,984 | 1.3910522 | 11.128418 |
| 1,024 | 64 | 704 | 256 | 256 | 80 / 176 | 28,981,248 | 0.8637085 | 6.909668 |
| 4,096 | 64 | 3,776 | 256 | 1,024 | 80 / 944 | 62,840,832 | 0.4682007 | 3.745606 |

The table includes physical full-page allocation, 56-byte page headers, and 48-byte slot-table
metadata. It excludes `std::vector` allocator bookkeeping because that overhead is allocator-
implementation-dependent and was not directly measurable from the typed layout; this exclusion
is explicit. No page allocator, fragmentation, slot indirection beyond the recorded metadata,
or BF16-window transition cost is claimed here; those belong to the later runtime phase.

## Decision

**PASS.** The runtime now has a deterministic, explicitly typed representation capable of
coexisting BF16 protected-prefix pages, official calibrated OSCAR INT2 G128 historical pages,
and BF16 recent-window pages. Logical addressing is contiguous and one-to-one, boundaries are
validated including an unaligned recent transition, K/V formats cannot silently disagree, all
16 full-attention layers share the policy, and GDN state is excluded.

Token aging/promotion, live OSCAR attention, mixed-window dispatch, and performance work remain
out of scope. D2.3 or the next explicitly authorized runtime phase may build on this layout.
