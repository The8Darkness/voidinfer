# OSCAR Phase D2.2c — Fork / Rollback / StateImage Qualification

Date: 2026-09-01  
Status: **PASS** for the diagnostic mixed-cache transition contract.

## Scope and prerequisites

D2.2a typed mixed BF16/INT2 layout and D2.2b deterministic aging were already PASS. This
phase does not add live OSCAR attention, INT2 attention dispatch, CUDA optimization, DFlash2,
MTP, adaptive-K, or a Q4 shadow.

The fixture consumes the validated C4 identity:

| Contract | Verified value |
| --- | --- |
| Asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| Runtime asset-manifest SHA-256 | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| Rotation mode | `qqt_sst+r_h_pbr`, calibrated |
| Full-attention layers | `3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63` |
| KV topology | 4 KV heads × D256; GQA ratio 6; group size 128 |

## Existing StateImage review and diagnostic boundary

The project StateImage implementation was reviewed in
`src/targets/qwen3_6/impl/runtime/state_image_store.h` and
`src/targets/qwen3_6/impl/state/state_image.cpp`. Its existing ownership contract has
`ActiveMutable`, `CheckpointImmutable`, and `ReservedDestination` roles, explicit fork/commit
transfers, and device/host replicas. Its fixed payload layout owns continuation hidden, GDN
recurrent state, and DFlash state; its optional Q4 shadow is explicitly legacy-only.

That payload schema does not own the D2.2 typed OSCAR page pools. Attaching the new pages to it
would either change the production StateImage schema or require the prohibited Q4 shadow. The
phase therefore adds `OscarMixedTransitionCache`, a diagnostic StateImage-compatible adapter:
it uses immutable serialized checkpoint bytes, branch-local identity, copy-on-write immutable
page blocks, exact restore, and commit-lineage checks. The adapter is isolated in
`src/core/oscar_mixed_state_transitions.{h,cpp}` and has no attention-facing API. Its state image
contains the original BF16 input lineage plus the validated asset identity, not a Q4 fallback;
restore deterministically rebuilds the already-proven D2.2b typed representation.

## Deterministic fixture and fingerprint

The test appends forced rows for logical tokens `0..323` to sequence `1001`. The base contains
all three physical regions simultaneously:

| Region | Logical range | Representation |
| --- | ---: | --- |
| Protected prefix | `0..63` | BF16 |
| Historical bulk | `64..67` | OSCAR INT2 G128 |
| Recent window | `68..323` | BF16 |

There are six pages per layer and 96 typed page blocks across the 16 full-attention layers:
prefix page 0, historical page 1, and recent pages 2–5. Every page has explicit layer,
sequence, logical/physical range, role, K/V storage type, layout version, and group size.

The complete fingerprint is emitted in
`results/oscar/phase-d2-2c-state-transitions-validation.json`. Each page entry includes:

- logical and physical range and page/slot identity;
- representation role and K/V storage type;
- K payload hash and V payload hash;
- OSCAR FP32 metadata hash for INT2 pages;
- slot-metadata hash;
- an overall ordered cache digest.

Base fingerprint overall hash: `0x6e621205cf0c784f`. The JSON has 96 base page entries; it is
the complete machine-readable fingerprint rather than a summary-only hash.

## Fork and copy-on-write

`OscarMixedAgingCacheBundle::clone_for_sequence()` rewrites only branch-local sequence metadata;
it preserves encoded historical rows, page payloads, metadata, and conversion counts without
re-encoding. The transition wrapper shares immutable `std::shared_ptr<const PageStorage>` blocks.
Appending or aging rebuilds only changed physical block values; unchanged prefix blocks remain
shared and old blocks are never mutated.

The immediate fork from sequence `1001` to `1002` passed:

| Check | Result |
| --- | ---: |
| Logical/content fingerprint equal | PASS |
| Child sequence identity distinct | PASS (`1002`) |
| Shared page blocks | `96/96` |
| Initial shared page refcount | `2` |
| Historical re-encoding during fork | none |

Separate parent/child forced appends used different row streams for tokens `324..327`. Both
reached context 328 with historical `8` and recent `256`; their content fingerprints differed,
while each retained 16 shared prefix page blocks with the unchanged base. The base remained at
context 324 and could not address speculative token 327. No cross-sequence corruption occurred.

## Rollback

A child at context 324 was snapshotted before four speculative appends. The child then crossed
four BF16-recent → INT2 transitions, reaching historical `8` and recent `256`. Restoring the
pre-speculation image passed all of the following:

| Check | Result |
| --- | ---: |
| Pre-spec image bytes | `21,236,493` |
| Pre-spec image FNV-64 | `0x685bb08d27b7ca39` |
| Complete fingerprint after restore | exact |
| Restored context / tiers | `324` / `64 prefix, 4 historical, 256 recent` |
| Retained historical payload/metadata | unchanged |
| Speculative logical positions `324..327` addressable | no |
| Conversion count restored | exact base count |

No speculative page or slot remained addressable after restore. The restored branch contained no
BF16/historical overlap and no tier hole.

## Commit

A child fork was appended with a fourth forced stream and committed into its unchanged parent
lineage. The commit guard rejected any source that did not match the parent’s recorded lineage
fingerprint; the valid commit passed:

- committed branch sequence identity: `1007`;
- logical context: `328`;
- tiers: prefix `64`, historical `8`, recent `256`;
- logical positions: `0..327` exactly once;
- no duplicate or omitted committed tokens.

## StateImage / restore

A separate branch (sequence `1008`) was extended to context `332`, giving prefix `64`, historical
`12`, and recent `256`, then serialized and restored into a fresh `OscarMixedTransitionCache`.
The deterministic state image was `21,760,845` bytes with FNV-64
`0x75206833adeafe8d`. The restored state image hash and complete page fingerprint matched the
source exactly:

| Check | Result |
| --- | ---: |
| Logical token sequence | exact |
| Tier classifications | exact |
| INT2 packed K/V payloads | exact |
| FP32 INT2 metadata | exact |
| Protected/recent BF16 payloads | exact |
| Page/slot ranges and roles | exact |
| Restored fingerprint overall | `0xca164656a3cbb16a` |
| State image hash after reload | exact |

The restored context’s physical accounting across all 16 layers was:

| Component | Bytes |
| --- | ---: |
| BF16 physical pages | `20,971,520` |
| INT2 packed payload | `524,288` |
| INT2 FP32 metadata | `131,072` |
| Page headers | `5,376` |
| Slot tables | `254,976` |
| Mixed total | `21,887,232` |

The input lineage in the diagnostic image is solely a deterministic restore source. It is not a
mandatory Q4 shadow and is not connected to live attention.

## Layer and GDN coverage

The transition bundle requires the exact 16-element full-attention list and rejects any other
layer. All fork, append/aging, rollback, commit, fingerprint, and restore checks operated over
all 16 layers. No GDN recurrent state is represented as OSCAR KV; the JSON records
`gdn_state_in_oscar_cache=false`.

## Implementation and reproduction

Implementation:

- `src/core/oscar_mixed_cache_layout.h/.cpp`: sequence-preserving aging-cache clone and all-layer
  conversion-count accessors;
- `src/core/oscar_mixed_state_transitions.h/.cpp`: immutable page blocks, fingerprints, fork,
  COW refresh, commit lineage, and deterministic StateImage envelope;
- `tests/test_oscar_mixed_state_transitions.cpp`: forced fixture and fail-closed transition tests;
- `src/CMakeLists.txt` and `tests/CMakeLists.txt`: core/test target registration. The existing
  layout test was also linked to `ninfer_ops` because the D2.2b aging code in `ninfer_core`
  references the official codec.

Build:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_oscar_mixed_state_transitions_test -j 4"
```

Run:

```powershell
& 'D:\AI\build-adaptive-dflash2\tests\ninfer_oscar_mixed_state_transitions_test.exe' --report 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\phase-d2-2c-state-transitions-validation.json'
```

Regression targets `ninfer_oscar_mixed_cache_layout_test`,
`ninfer_oscar_mixed_cache_aging_test`, and `ninfer_oscar_mixed_state_transitions_test` all
completed with exit code 0.

Evidence hashes:

| Artifact | SHA-256 |
| --- | --- |
| Transition executable | `bd039908f526c8d23ecca57a27e2363b130ff7202cc938bb6a8a1c71fd285a20` |
| Validation JSON | `df1fa525a1791cf6c6463686d1b3a741901a58825d832cca24c2e96992e16784` |
| Transition implementation | `4960fc916a047a5416cccbb0dfda62cd7eb88c1cfb8538ef23fe9e7130f312ec` |
| Transition test | `1313f33eb20a876b038c1c7fb9e19b4258868787a0f41fcf12654c101ef9dceb` |

## Verdict

**PASS — representation-stable and logically exact under append → aging → fork → speculative
mutation → rollback/commit → StateImage restore.** The mixed cache transition contract is
qualified for the next separately authorized phase. Live INT2 attention remains intentionally
unimplemented.
