# Qwen3.8-27B NVFP4 performance

This report covers the active NVFP4 optimization target on the physical NVIDIA GeForce RTX 5090.
The optimized branch is
[`exp/hierarchical-vericache-20260830`](https://github.com/The8Darkness/voidinfer/tree/exp/hierarchical-vericache-20260830).
It uses DFlash2 `k=7`, the optimized proposal head, VeriCache-NVFP4, and the hierarchical KV/GDN
transaction path. Results are local engineering measurements, not claims that the same speedups
will multiply on another GPU or workload.

## Configurations

| Configuration | Artifact | KV | Speculation | Hierarchy |
| --- | --- | --- | --- | --- |
| Windows reference baseline | registered Qwen3.8-27B NVFP4 | BF16 target KV | off | off |
| Optimized resident comparison | local Qwen3.8-27B NVFP4-DFlash2 | VeriCache-NVFP4: BF16 exact target + U8 NVFP4 DFlash local KV | DFlash2 `k=7` | on, host snapshots off for the long-context ceiling test |
| Current default server | local Qwen3.8-27B NVFP4-DFlash2 | VeriCache-NVFP4 | DFlash2 `k=7` | on, host snapshots on |

The baseline is the Windows phase7 Release binary and the optimized artifact is a larger local
experimental artifact. This is an end-to-end product comparison, not a code-only A/B on identical
weight files.

## 150K context comparison

The corpus contains 65,536 committed token IDs tiled deterministically to 150,000 tokens. The
baseline prefill campaign used `ninfer_bench`, BF16 KV, a 1,024-token prefill chunk, no CUDA Graph,
one warmup, and one measured repetition. The baseline decode column is the same product benchmark
with eight generated tokens after each prompt. The optimized column used the direct DFlash2 round
harness with CUDA Graphs, one warmup, one measured round, and a synthetic repeated-token prompt
with the same context lengths. It reports licensed/published tokens per speculative round.

| Context fraction | Tokens | Baseline prefill tok/s | DFlash2 prefill tok/s | Prefill delta | Baseline eager decode tok/s | DFlash2 published tok/s | DFlash2 acceptance | Result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 25% | 37,500 | 6,496.34 | 6,520.24 | +0.37% | 63.54 | 85.87 | 4/7 (57.14%) | passed resident path |
| 50% | 75,000 | 4,874.49 | 4,951.67 | +1.58% | 58.72 | 42.06 | 3/7 (42.86%) | passed resident path |
| 75% | 112,500 | 3,914.12 | 3,992.00 | +1.99% | 54.16 | 30.33 | 3/7 (42.86%) | passed resident path |
| 100% | 150,000 | 3,253.01 | not run | — | 50.38 | not run | — | DFlash2 preflight required 34.46 GB; 32.50 GB was free |

The prefill change is the cleanest comparison in this table. The decode values are not a direct
multiplier: baseline decode is an eight-token eager product measurement, while DFlash2 publishes
multiple licensed tokens from one verified round and the acceptance-sensitive samples use one
measured repetition.

The native model context is 262,144 tokens. The 150K baseline succeeds on this machine. DFlash2
resident execution succeeds through 112,500 and fails the 150K memory preflight. The server’s
compiled default remains 8,192 tokens because capacity is intentionally startup-fixed.

## Current tier accounting

| Tier | Current bytes/format | Implementation status |
| --- | --- | --- |
| L0 VRAM | 26,869,760 bytes in the tested `k=7` setup; DFlash local KV is U8 packed NVFP4, with protected recent/sink/pivot sidecar data. | Active fast speculative tier. |
| L1 pinned RAM | The host StateImage’s NVFP4/U8 DFlash-local payload and any host-resident DFlash pages; host KV page copies currently use BF16 layout. | Checkpoint/residency mirror, not an independent host-logit verifier. |
| L2 system RAM | BF16 target/main attention KV and BF16 GDN/recurrent state. | Authoritative host restore tier. |
| L3 NVMe | 0 bytes in current runs. | Cold persistence writer is not attached. |

`VeriCache-NVFP4` deliberately keeps the exact target/main KV BF16. Only the DFlash local/speculative
KV is U8 NVFP4 today. The desired direct low-bit exact-target attention and host logits verifier are
future work, so host-tier bytes must not be interpreted as a completed three-level verifier.

With host snapshots enabled, the 32,768-token run reported `L0/L1/L2/L3 =
26,869,760 / 147,652,608 / 2,301,437,952 / 0` bytes. It transferred 167,389,184 StateImage bytes
and 2,281,703,376 KV bytes to pinned host memory. A 36,000-token host-snapshot run passed; the
37,000-token attempt reached the partial-tail COW path and failed on the additional device-page
allocation. The resident no-host configuration is therefore the correct long-context ceiling probe.

## Quality and correctness

| Test | Result | Claim supported |
| --- | --- | --- |
| Matched greedy trace, 46 prompt tokens and 64 generated tokens | 64/64 generated token IDs identical between baseline BF16 no-spec and DFlash2 exact-target execution | This direct greedy equality test passed. |
| DFlash2 greedy settlement | 19 rounds, 130 drafted, 43 accepted, 1 fallback step; focused DFlash2 and hierarchy suites passed | Exact target verification and rollback are active. |
| Matched seeded sampling, 64 generated tokens | Token IDs differed | Speculative sampling is judged by distribution/quality, not seeded sequence identity; no distribution score has been published yet. |
| Context probes at 37.5K, 75K, and 112.5K | Exact target checks and nested KV/GDN transaction accounting completed without an engine error | Structural correctness only; no long-context quality score inferred. |
| Vision/OCR under DFlash2 | Not run | DFlash2 is text-only; vision stays on the BF16-protected route. |

The NVFP4 artifact’s existing scored evaluations are reference-model results, not DFlash2-specific
quality claims. The missing gates are seeded-sampling distribution checks, greedy equality across
long-context retrieval/coding/reasoning/tool-JSON suites, multi-agent prefix reuse, and vision/OCR.

## Retained optimizations

- Direct DFlash2 QKV Tensor Core split output removes packed-QKV materialization and copies; focused
  `T=1,8,16,48,49,64` comparisons pass.
- Fused Qwen3.8 W8G32 gate/up projection removes the large BF16 gate/up temporary and standalone
  activation launch; focused W8/A16 correctness cases pass.
- The DFlash2 selector uses a 512-thread warp-shuffle top-16 reduction with packed survivor keys;
  the synthetic selector measured about 327.7 µs versus 565–570 µs for the previous route.
- The production selector lattice uses 272 semantic floats per token instead of a 5,120-float
  hidden-width temporary; exact-width guard tests pass.
- DFlash2 proposal/selector/verify execution is CUDA-Graph safe. Matched context-512 samples
  reduced steady GPU round latency by 7.98% at C=1 and 7.65% at C=2.
- NVFP4 attention consumes packed bytes directly and uses pair decoding by default; set
  `NINFER_NVFP4_PAIR=0` for the scalar-order fallback.
- Hierarchical VeriCache provides protected local anchors, nested KV/GDN transactions, immutable
  COW manifests, event-ordered host transfers, and independent host-snapshot cadence.
- Non-page-aligned host-prefix forks now use the ceiling page count so a valid partial tail is not
  rejected as an invalid entitlement.

Candidates that are neutral in a broad benchmark remain available for further tuning when they
reduce memory, launches, or working-set pressure; no neutral sample is presented as a throughput
win.

## Reproduce the main tests

Baseline prefill/decode:

```powershell
build-windows-phase7\bench\Release\ninfer_bench.exe `
  --weights models\Qwen3.8-27B-NVFP4-Ninfer\qwen3_8_27b_nvfp4.ninfer `
  --corpus <tiled-150k-token-id-corpus> `
  --prompt-gen "37500,8;75000,8;112500,8;150000,8" `
  --max-ctx 150016 --kv-dtype bf16 --no-cuda-graph `
  --warmup 1 --repetitions 1 --output json
```

Optimized resident DFlash2:

```powershell
build-windows-hierarchical\bench\ninfer_qwen3_6_27b_dflash_round_bench.exe `
  --artifact models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer `
  --context 37500 --warmup 1 --reps 1 --prefill-chunk 1024 `
  --draft-tokens 7 --proposal-head optimized --kv-dtype vericache-nvfp4 `
  --hierarchical-vericache --vericache-l1-horizon 512 --cuda-graph
```

The optimized artifact path and the exact benchmark command are local to the experimental branch;
the raw reports are kept outside the source tree because the 150K corpus is generated test data.
