# OSCAR Phase D2.3a — Mixed-Cache Reference Attention Parity

Date: 2026-09-01  
Status: **PASS**

## Scope and qualification gate

This phase qualifies a deliberately slow, correctness-first reader over the already qualified
D2.2 mixed representation. It does not connect the reader to normal model serving, change
calibration, change cache-transition semantics, optimize CUDA, or touch DFlash2/MTP/adaptive-K.
The qualified C4 asset is unchanged:

`qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1`

The reader uses the verified Qwen3.8 topology: full-attention layers
`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`, 24 Q heads, 4 KV heads, GQA ratio 6,
head dimension 256, and group size 128. GDN state is not an input to this reader.

## Reader architecture

The new diagnostic API is `ninfer::OscarMixedAttentionReader` in
`src/core/oscar_mixed_attention_reference.h` and
`src/core/oscar_mixed_attention_reference.cpp`.

For one query and one full-attention layer it:

1. consumes the deterministic post-RoPE query fixture in the existing unrotated coordinate
   convention;
2. applies the layer-specific C4 `R_K` in FP32 to produce `Q'`;
3. resolves every logical position through the typed D2.2b cache page map, in causal order;
4. reads protected-prefix and recent BF16 rows directly, converting only the temporary row to
   FP32;
5. reconstructs historical rows from the exact 64-byte packed `OscarInt2G128` payload and four
   FP32 metadata values per row, using the qualified D2.1 decoder;
6. computes GQA scores (`q_head / 6`), stable softmax, and rotated-coordinate AV in FP32;
7. applies `R_V.T` in FP32 and returns recovered FP32 attention.

The reader performs temporary per-query row decoding only. It does not create a persistent or
global BF16 shadow, does not dispatch the legacy Q2 codec, and has no serving or CUDA dispatch.
Page role and storage type are checked together; an invalid prefix/historical/recent pairing is
rejected rather than inferred.

Implementation locations:

- `src/core/oscar_mixed_attention_reference.h:9-49` — tier enum, trace, and reader API.
- `src/core/oscar_mixed_attention_reference.cpp:50-111` — strict page-tier dispatch and BF16/
  INT2 row decoding.
- `src/core/oscar_mixed_attention_reference.cpp:126-238` — input validation, logical traversal,
  Q rotation, FP32 GQA/softmax/AV, and FP32 inverse-V recovery.
- `src/CMakeLists.txt:31` — core source registration.

## Independent reference

`tests/test_oscar_mixed_attention_reference.cpp:199-302` implements a separate CPU comparator.
It does not call the new reader or any page/slot resolution API. It reconstructs each logical row
from the deterministic source archive, classifies the position using the same D2.2 policy, and
independently invokes the official `OscarInt2G128` encode/decode semantics for historical rows.
It then performs its own FP32 rotation, GQA score, stable softmax, AV, and `R_V.T` sequence.

Thus parity compares page traversal and typed dispatch against a source-archive reconstruction,
not the reader against a second view of the same cache object. Historical rows in both sides use
the same post-aging INT2 representation; no pre-aging BF16-to-post-aging INT2 equality is claimed.

## Deterministic fixture and command

The fixture uses deterministic FP32-to-BF16 K/V rows for all four KV heads and all 16 verified
full-attention layers. Rows are supplied in the already-rotated cache coordinate convention used
by D2.2b. Deterministic Q is generated per layer and query position. Static contexts contain:

- context 324: prefix 64, historical INT2 4, recent BF16 256;
- context 332: prefix 64, historical INT2 12, recent BF16 256.

The forced fixture appends tokens 324–327 to a persistent layer-3 cache after the initial
0–323 construction; each append uses the normal D2.2b aging transition before the query.

Exact command:

```powershell
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_mixed_attention_reference_test.exe' `
  --rotation-dir 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1\runtime' `
  --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-3a-reference-attention-validation.json'
```

The executable was rebuilt from the current source in `D:\AI\build-adaptive-dflash2` and exited
with code 0.

## Logical tier traces and coverage

The validation JSON records the complete `logical_positions` and `tier_sequence` arrays for
every comparison. An independent JSON check confirmed that every array is exactly `0..query`,
has no duplicate or missing position, and has a tier entry for every position. Tier codes are
`0=protected-prefix BF16`, `1=historical OSCAR INT2 G128`, and `2=recent BF16`.

Representative layer-3 traces are:

| Suite | Query/context | Prefix / historical / recent | Boundary interpretation |
| --- | ---: | ---: | --- |
| static-context-324 | 63 / 324 | 64 / 0 / 0 | last protected-prefix position |
| static-context-324 | 64 / 324 | 64 / 1 / 0 | first historical position |
| static-context-324 | 68 / 324 | 64 / 4 / 1 | historical INT2 → recent BF16 |
| static-context-324 | 323 / 324 | 64 / 4 / 256 | complete three-tier sequence |
| static-context-332 | 320 / 332 | 64 / 12 / 245 | recent boundary before its final position |
| static-context-332 | 331 / 332 | 64 / 12 / 256 | complete three-tier sequence |
| forced-decode-layer-3 | 324 / 325 | 64 / 5 / 256 | first persistent post-aging append |
| forced-decode-layer-3 | 327 / 328 | 64 / 8 / 256 | fourth persistent post-aging append |

All four forced queries used the same logical positions and source archive as the independent
reference. No hole, duplicate, tier disagreement, GDN state, or legacy Q2 dispatch occurred.

## First divergence

No divergence was observed. Every reader/reference logical trace, tier dispatch, and compared
FP32 stage matched within the declared gate; therefore there is no first differing operation to
isolate.

## Parity metrics

There were 31 comparisons: layer-3 static boundary queries (4 at context 324 and 5 at context
332), layer-35 and layer-63 static checks, four persistent forced-decode checks at layer 3, and
one final static check for each of all 16 full-attention layers.

Every compared stage passed the declared relative-L2 gate of `1e-6`:

| Stage | Maximum absolute error | Maximum mean absolute error | Maximum relative L2 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Rotated Q | 0 | 0 | 0 | PASS |
| QK scores | 0 | 0 | 0 | PASS |
| Stable softmax | 0 | 0 | 0 | PASS |
| Rotated-coordinate AV | 0 | 0 | 0 | PASS |
| Recovered output after `R_V.T` | 0 | 0 | 0 | PASS |

The zero metrics are exact equality in the deterministic scalar FP32 implementation: both paths
use the same float operations once the independently reconstructed historical row has been
decoded. This is an implementation-parity result for the mixed reader, not a claim that INT2
matches the original BF16 model.

### Layer and forced-decode results

Layer 3 passed every static boundary and forced-decode comparison. Layers 35 and 63 passed the
complete context-324 three-tier query. The all-layer coverage pass exercised the layer-specific
C4 rotation bank for all 16 full-attention layers:

`3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`

The forced sequence was deterministic and did not use model logits. After each append the reader
observed historical counts `5,6,7,8` and a 256-token recent window. The reader never fell back
to a production cache.

## Boundary and aging interpretation

The reader addresses token 63 as protected BF16 and token 64 as historical INT2 once the fixture
has aged it. It addresses the exact historical-to-recent boundary once both regions coexist. For
the persistent forced path, the D2.2b cache performs recent-to-historical conversion before each
query; the independent comparator encodes and decodes those same post-aging source rows with the
official codec. Consequently the gate tests the mixed-cache representation actually read after
aging, without incorrectly requiring lossy INT2 storage to equal its former BF16 source.

## Evidence hashes

- C4 runtime manifest: `c1979e86744682733a668739642ad3a945b5a6220e8b3ed983b74d33b82c3afc`
- Validation JSON: `6090dab55f3822b283c39f306e2789d758eb351a137b09f9acdf824735015610`
- Validation executable: `4519869eba85f900d89e4db143c5150959cfa7d7e63d0da17663064fcd8c439e`
- Reader header: `1a32619eb67158c7adab540cfa7c4a3c81a6668cada8ffc5eb4c01fdceb370fd`
- Reader implementation: `539ad5177c3137b8d3538a13250a499a0674c3874a6990f57019079527e356be`
- Parity test: `d6bb68368c369b254c441bced0ce2fb6be448cf8de07f73937211e533d5e286e`

## Verdict and next phase

**PASS.** The mixed BF16-prefix + official OSCAR INT2 historical + BF16-recent reader matches an
independent CPU reference through rotated Q, scores, stable softmax, rotated AV, and FP32
`R_V.T` recovery for static and persistent forced-decode fixtures. All expected full-attention
layers are covered, and no GDN state or legacy Q2 path is dispatched.

D2.3b may connect this qualified attention path to the real Qwen3.8 runtime. This report does not
authorize that integration and makes no performance claim.
