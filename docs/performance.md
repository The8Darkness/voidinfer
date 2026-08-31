# Qwen3.8-27B OSCAR-Q2/Q4 performance

This is the measured status of the Qwen3.8-27B NVFP4 artifact on the physical GeForce RTX 5090.
The active research branch is
[`exp/hierarchical-vericache-20260830`](https://github.com/The8Darkness/voidinfer/tree/exp/hierarchical-vericache-20260830).
The hierarchy is experimental and remains separate from the stable fallback.

## Current hierarchy

| Tier | Representation | Status |
| --- | --- | --- |
| L0 VRAM | OSCAR-Q2 packed U8 with affine BF16 scale/zero metadata; protected recent/sink/pivot rows stay in BF16 sidecars. | Active DFlash2 speculative KV and direct OSCAR attention path. |
| L1 pinned RAM | Independently written OSCAR-Q4 packed U8 with independently fitted metadata and protected sidecars. | Active host snapshot/attention mirror; not yet a live host-logit verifier. |
| L2 RAM | Full 16-bit authoritative target KV plus GDN state. The current Qwen target path stores this as BF16; FP16 byte-format conversion is still open. | Authoritative restore/checkpoint tier. |
| L3 NVMe | No active writer; current measured bytes are zero. | Reserved for cold persistence. |

The Q4 mirror is dual-written from the BF16 append rows in one fused rotated kernel. It is not
reconstructed from lossy Q2 codes. Q3 is not a serving or benchmark target.

The current implementation has not connected the host mirror to a second target-model logit pass.
Consequently, `L0→L1` and `L1→L2` live acceptance counters are zero in the results below; exact
target DFlash2 verification and nested KV/GDN rollback remain active. This distinction is important:
the current measurements validate the storage and transaction path, not a completed hierarchical
quality claim.

## Matched 32K OSCAR controls

Both rows used the same local Qwen3.8-27B DFlash2 artifact, DFlash2 `k=7`, full proposal head,
1,024-token prefill chunks, one warmup, one measured repetition, no CUDA Graph, batch 1, protected
recent=64/sinks=4/pivots=4, and host snapshots enabled. The Q2 row is the default hierarchy; Q4 is a
storage/throughput control.

| L0 | Prefill tok/s | Round wall ms | Published tok/s | Accepted draft | L0 bytes | Extra Q4 shadow bytes | L1 Q4 bytes | L2 bytes | Snapshot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OSCAR-Q2 | 5,702.34 | 57.9637 | 34.5044 | 1/7 (14.29%) | 15,073,280 | 25,559,040 | 146,997,248 | 2,301,437,952 | passed |
| OSCAR-Q4 control | 5,847.05 | 57.9637 | 34.5044 | 1/7 (14.29%) | 25,559,040 | 0 | 146,997,248 | 2,301,437,952 | passed |

The one-round acceptance sample is not a stable throughput estimate. Both rows recorded two exact
target checks, one accepted proposed token, two disagreements, and four nested commits/rollbacks.
They recorded zero live Q2→Q4 and Q4→L2 logit checks because that verifier consumer is not wired
yet.

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
  reuse, vision/OCR, live Q2→Q4 verification, and FP16 host-format validation remain open.
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
- The fused OSCAR dual writer emits Q2 L0 and independent Q4 shadow rows from one BF16 load/rotation.
- Protected sidecars, nested KV/GDN transactions, immutable COW manifests, ordered host transfers,
  and independent host-snapshot cadence are implemented.

## Reproduce the controls

```powershell
build-windows-hierarchical\bench\ninfer_qwen3_6_27b_dflash_round_bench.exe `
  --artifact D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer `
  --context 32768 --warmup 1 --reps 1 --prefill-chunk 1024 `
  --draft-tokens 7 --proposal-head full --kv-dtype vericache-nvfp4 `
  --hierarchical-vericache --vericache-host-snapshots `
  --vericache-l1-horizon 512 --vericache-host-snapshot-horizon 2048 `
  --vericache-l0-bits 2 --no-cuda-graph
```

Use `--vericache-l0-bits 4` for the Q4 L0 control. There is no Q3 control in the active hierarchy.
