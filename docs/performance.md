# Qwen3.8-27B OSCAR-Q2/Q4 performance

This is the measured status of the Qwen3.8-27B NVFP4 artifact on the physical GeForce RTX 5090.
The active research branch is
[`exp/hierarchical-vericache-20260830`](https://github.com/The8Darkness/voidinfer/tree/exp/hierarchical-vericache-20260830).
The hierarchy is experimental and remains separate from the stable fallback.

## Current hierarchy

| Tier | Representation | Status |
| --- | --- | --- |
| L0 target KV in VRAM | OSCAR-Q2 packed U8 with affine BF16 scale/zero metadata; the recent 128-token window and sink/pivot anchors stay in BF16 sidecars. | Authoritative target attention cache and direct OSCAR attention path. |
| DFlash2 local drafter KV | BF16 fixed cyclic K/V state, separate from target L0. | Active speculative proposal path; deliberately not inherited from the compressed target cache. |
| L1 pinned RAM | Q2-derived OSCAR-Q4 packed U8 with affine metadata and protected sidecars. No persistent device Q4 shadow is allocated. | Host intermediate/persistence tier, not a per-token live logit verifier. |
| L2 RAM | Full authoritative target KV stored as FP16 host records plus protected GDN state. | Authoritative restore/checkpoint tier. |
| L3 NVMe | No active writer; current measured bytes are zero. | Reserved for cold persistence. |

The current host Q4 mirror is derived during host promotion from the resident Q2 state. Q3 is not a
serving or benchmark target. Exact target verification remains authoritative.

The matched measurements below were collected before the DFlash2 local drafter cache was restored
to BF16. They remain useful historical controls, but must be rerun before treating their throughput
or residency values as current.

The default serving decision uses Q2 target L0 plus a separate BF16 DFlash2 local drafter cache and
exact target settlement; it does not run CPU attention over pinned RAM. Nested KV/GDN rollback is
active. `NINFER_DFLASH_FULL_BF16=1` selects a real BF16
DFlash full-attention cache for an A/B experiment, but it is not the default because it was neutral
to slightly slower in the matched run.

## Matched context-512 serving controls

All rows used the same local Qwen3.8-27B DFlash2 artifact, DFlash2 `k=7`, optimized proposal head,
1,024-token prefill chunks, two warmups, five measured repetitions, CUDA Graphs, batch 1, sinks=4,
pivots=4, and host snapshots enabled. The current default protects 128 recent tokens.

| Mode | Prefill tok/s | Round wall ms | Published tok/s | Accepted draft | Exact target accepted | Target disagreements | L0 bytes | Q4 shadow bytes | L2 FP16 bytes | Snapshot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Current Q2 default | 7,061.80 | 22.4202 | 205.172 | 18/35 (51.43%) | 26/49 | 5 | 23,633,920 | 0 | 187,508,736 | passed |
| Q2 + BF16 DFlash full cache (`NINFER_DFLASH_FULL_BF16=1`) | 7,052.78 | 22.4690 | 204.727 | 18/35 (51.43%) | 26/49 | 5 | 25,886,720 | 0 | 187,508,736 | passed |
| Q2 default with 64 protected recent (control) | 7,146.33 | 22.3805 | 160.854 | 13/35 (37.14%) | 21/49 | 6 | 21,012,480 | 0 | 187,508,736 | passed |
| Standalone Q4 L0 diagnostic | 6,983.92 | 22.7720 | 219.568 | 20/35 (57.14%) | 31/49 | 5 | 31,498,240 | 0 | 187,508,736 | passed |

The rows are short direct probes, not broad task-quality claims. The Q4 L0 row is diagnostic only and
uses more device KV; it shows that the shadow-free Q2 gap is chiefly local acceptance. The BF16 full
cache row shows no acceptance change and a small throughput loss, so compressed DFlash full attention
remains the default.

## OSCAR codec comparison

This direct attention fixture compares the packed representation against the same BF16 source
reconstructed in host code. It is a codec/attention result, not a model-task quality score.

| Representation | Protected rows relative to source | Protected rows relative to FP16/BF16 attention oracle |
| --- | ---: | ---: |
| OSCAR-Q2 | 0.001665 | 0.332384 |
| OSCAR-Q4 | 0.001666 | 0.075313 |

The Q4 path is materially closer to the full-precision attention oracle in this fixture. Q2/Q4
append, bit packing, independent K/V affine calibration, rotation, direct low-bit attention, and
protected sidecars pass `ninfer_oscar_kv_test`.

## Stock Windows 150K reference

This is the retained baseline comparison from the Windows phase7 Release binary using the registered
NVFP4 base artifact, BF16 target KV, no speculation, and a deterministic 65,536-token corpus tiled
to 150,000 tokens. The DFlash2 values in this table are an earlier resident probe, not the current
OSCAR host-snapshot run above.

| Context | Stock prefill tok/s | Earlier DFlash2 prefill tok/s | Stock eager decode tok/s | Earlier DFlash2 published tok/s | Earlier accepted draft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 37,500 (25%) | 6,496.34 | 6,520.24 (+0.37%) | 63.54 | 85.87 | 4/7 |
| 75,000 (50%) | 4,874.49 | 4,951.67 (+1.58%) | 58.72 | 42.06 | 3/7 |
| 112,500 (75%) | 3,914.12 | 3,992.00 (+1.99%) | 54.16 | 30.33 | 3/7 |
| 150,000 (100%) | 3,253.01 | not run | 50.38 | not run | DFlash2 preflight required 34.46 GB; 32.50 GB was free |

These decode columns use different measurement units: stock is an eight-token eager product-bench
run, while DFlash2 is licensed/published tokens from one speculative round. They are engineering
references, not multiplicative speedup claims.

## Correctness and open gates

- A matched 46-token greedy trace generated 64/64 identical token IDs between the BF16 no-spec
  reference and exact-target DFlash2 execution.
- Focused DFlash2, OSCAR codec, state-image, hierarchy, and rollback tests pass.
- Qwen3.8 GDN state participates in the state-image fork/restore/copy paths and nested transaction
  accounting.
- Seeded sampling distribution tests, long-context retrieval, coding/reasoning/tool JSON, multi-agent
  reuse, host-side logit verification, and vision/OCR remain open. FP16 host-format conversion is
  implemented and exercised by the snapshot probe; it is not yet a broad quality qualification.
- The 800K aggregate residency plan and 262K single-stream confidence target are research targets;
  they are not current supported-capacity claims.

## Retained optimizations

- DFlash2 split Q/K/V Tensor Core projection removes packed-QKV materialization and copies.
- Fused W8G32 SwiGLU removes the large gate/up temporary and standalone activation launch.
- The selector uses a warp-shuffle top-16 reduction and a compact proposal lattice.
- DFlash2 proposal/selector/verify execution is CUDA-Graph safe.
- OSCAR attention consumes packed Q2/Q4 bytes directly with rotated K/V and no full dequantized KV
  tensor; the current fallback rotation is normalized Hadamard because model-specific OSCAR matrices
  are not yet included in the artifact.
- Direct OSCAR-Q2 append/attention avoids a full device dequantization; host promotion emits the Q4
  mirror without allocating a persistent device Q4 shadow.
- Protected sidecars, nested KV/GDN transactions, immutable COW manifests, ordered host transfers,
  and independent host-snapshot cadence are implemented.

## Reproduce the controls

```powershell
build-windows-hierarchical\bench\ninfer_qwen3_6_27b_dflash_round_bench.exe `
  --artifact C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer `
  --context 32768 --warmup 1 --reps 1 --prefill-chunk 1024 `
  --draft-tokens 7 --proposal-head full --kv-dtype vericache-nvfp4 `
  --hierarchical-vericache --vericache-host-snapshots `
  --vericache-l1-horizon 512 --vericache-host-snapshot-horizon 2048 `
  --vericache-l0-bits 2 --no-cuda-graph
```

Use `--vericache-l0-bits 4` for the single-Q4 L0 diagnostic. Set
`NINFER_DFLASH_FULL_BF16=1` to compare a BF16 DFlash full-attention cache. There is no Q3 control in
the active hierarchy.
