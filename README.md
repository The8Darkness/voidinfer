# VoidInfer — Qwen3.8-27B OSCAR-Q2/Q4 on an RTX 5090

VoidInfer is a native C++/CUDA inference engine for Qwen3.8-27B on Blackwell `sm_120a`. This
branch focuses on the Qwen3.8-27B NVFP4 model artifact and the DFlash2 + hierarchical VeriCache
serving path. Older target and conversion code may remain in the source tree for compatibility,
but it is not part of the public result set below.

## Current serving default

The isolated branch
[`exp/hierarchical-vericache-20260830`](https://github.com/The8Darkness/voidinfer/tree/exp/hierarchical-vericache-20260830)
serves the local Qwen3.8-27B NVFP4 DFlash2 artifact with:

- DFlash2 `k=7` (`--spec dflash --draft-tokens 7`);
- the optimized private proposal head (`--lm-head-draft`);
- hierarchical VeriCache with OSCAR-Q2 L0, an independently written OSCAR-Q4 host mirror, and
  an authoritative 16-bit host state;
- exact target verification and nested KV/GDN rollback on every speculative round;
- BF16-protected vision/non-DFlash execution when multimodal input is requested.

The server’s compiled default context is 8,192 tokens. The native Qwen3.8 context limit is 262,144
tokens, but usable capacity is fixed by the RTX 5090’s memory budget and the selected runtime
features. `--no-spec --no-hierarchical-vericache` restores the stable non-speculative fallback.

The public baseline below is the default Windows reference: the phase7 Release binary, the same
registered NVFP4 base artifact, BF16 target KV, no speculative backend, and a 150,000-token test
capacity. It is not relabeled as DFlash2; DFlash2 is the optimized comparison path.

## Current KV hierarchy

The names below describe the implementation that exists today, not the eventual research target.

| Tier | Current representation | Current role |
| --- | --- | --- |
| L0 VRAM | DFlash2 local KV is packed `U8` OSCAR-Q2 with independently fitted affine BF16 scale/zero metadata. Recent/sink/pivot anchors use the protected BF16 sidecar. | Fast speculative proposal/attention path. Compression disagreement becomes rejection/rollback; the exact target remains authoritative. |
| L1 pinned system RAM | Independently written packed `U8` OSCAR-Q4 DFlash mirror with its own affine BF16 metadata and protected sidecars. It is produced from the BF16 append rows, not by re-quantizing Q2. | Intermediate storage/attention mirror. It is not yet a live host-logit verifier. |
| L2 system RAM | Full authoritative 16-bit target KV and protected Qwen3.8 GDN state. The current target implementation uses BF16 for this device/host representation; an FP16 byte-format conversion is still a separate task. | Correctness and restore tier for host checkpoints. |
| L3 NVMe | No active cold writer in the current serving path; measured L3 bytes are zero. | Reserved for future cold persistence and never used in frequent verification. |

Important: selecting `VeriCache-NVFP4` does **not** make the exact target/main attention KV low-bit.
The active hierarchy is OSCAR-Q2 for DFlash2 local KV, OSCAR-Q4 for the pinned host mirror, and the
full 16-bit target/GDN state for L2. The host mirror is currently a checkpoint/attention-storage
path; live Q2→Q4 and Q4→L2 logit verification are not attached yet.

The latest 32,768-token Q2 run with host snapshots reported `L0/L1/L2/L3` payload bytes of
`15,073,280 / 146,997,248 / 2,301,437,952 / 0`, plus `25,559,040` bytes for the temporary
independent Q4 source shadow in VRAM. It completed with 166,733,824 StateImage D2H bytes and
2,281,701,376 KV D2H bytes.

## RTX 5090 context comparison

The historical 150K comparison below ran on the physical NVIDIA GeForce RTX 5090 with the Windows
phase7 reference binary. The corpus is the committed 65,536-token corpus tiled deterministically to
150,000 tokens. Baseline prefill uses BF16 KV, 1,024-token chunks, no CUDA Graph, one warmup, and
one measured repetition. Its decode column is an eight-token eager product-bench measurement.
These values are retained as the stock Windows reference; they are not claims about the current
OSCAR hierarchy.

| Requested point | Context | Baseline prefill tok/s | DFlash2 prefill tok/s | Prefill change | Baseline eager decode tok/s* | DFlash2 published tok/s** | DFlash2 accepted draft | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 25% | 37,500 | 6,496.34 | 6,520.24 | +0.37% | 63.54 | 85.87 | 4/7 (57.14%) | historical resident probe |
| 50% | 75,000 | 4,874.49 | 4,951.67 | +1.58% | 58.72 | 42.06 | 3/7 (42.86%) | historical resident probe |
| 75% | 112,500 | 3,914.12 | 3,992.00 | +1.99% | 54.16 | 30.33 | 3/7 (42.86%) | historical resident probe |
| 100% | 150,000 | 3,253.01 | not run | — | 50.38 | not run | — | DFlash2 preflight required 34.46 GB; 32.50 GB free |

\* Baseline decode is an eight-token eager product-bench measurement.<br>
\*\* DFlash2 published throughput is licensed tokens per direct speculative round. The decode
columns are therefore an engineering comparison, not a multiplicative speedup claim: they use
different execution units, and the DFlash2 values are one-round acceptance-sensitive samples.

The latest matched OSCAR-Q2/Q4 host-snapshot probe at 32K measured 5,702.34 prefill tok/s, 34.504
published tok/s, and 1/7 accepted DFlash2 draft tokens. The Q4 control measured 5,847.05 prefill
tok/s and the same 34.504 published tok/s. Both completed without OOM or rollback failure. Both
recorded zero live L0→L1 and L1→L2 logit checks: these are storage/transaction plumbing results,
not end-to-end hierarchical verifier quality. The 800K aggregate plan and 262K single-stream
confidence target remain benchmark work, not current capacity claims.

## Quality and correctness results

The current quality evidence is intentionally separated from throughput:

| Test | Result | Interpretation |
| --- | --- | --- |
| Matched greedy output, 46 prompt tokens, 64 generated tokens | 64/64 generated token IDs identical between the BF16 no-spec baseline and DFlash2 exact-target execution | Greedy equality passed for this direct trace. |
| DFlash2 greedy round path | 19 rounds, 130 drafted, 43 accepted (33.08%), 1 fallback step; focused DFlash2 and hierarchy tests passed | Exact target verification and settlement are active. |
| Matched seeded sampling, 64 generated tokens | Token IDs were not identical | This is not a quality failure criterion for speculative sampling; distribution preservation has not yet been scored. |
| DFlash2 25/50/75% context probes | Exact-target checks completed with nested rollback enabled; no engine error on the resident path | Structural correctness evidence only; no long-context quality score was inferred. |
| OSCAR codec oracle | Q2 protected relative L2 error 0.332; Q4 protected relative L2 error 0.075 on the direct attention fixture | Q4 is materially closer to the FP16/BF16 oracle in the codec test; this is not a model-task quality score. |
| DFlash2 vision/OCR | Not qualified | DFlash2 is text-only; vision remains on the BF16-protected path. |

Conclusion: greedy parity passed in the matched trace, while the requested broad sampling,
long-context retrieval, coding/reasoning/tool-JSON, multi-agent, and vision quality matrix remains
open. The repository makes no unsupported claim that host-tier verification or 150K DFlash2
serving is already quality-qualified.

## Retained optimizations

The active NVFP4/DFlash2 branch retains changes only when a focused correctness/resource gate and a
measurement support them. Neutral end-to-end candidates remain available for further tuning rather
than being misreported as wins.

| Area | Optimization and evidence |
| --- | --- |
| DFlash2 QKV projection | Direct split Q/K/V Tensor Core output removes packed-QKV materialization and copies; focused `T=1,8,16,48,49,64` comparisons pass. |
| DFlash2 fused SwiGLU | Fused W8G32 gate/up output removes the large BF16 gate/up temporary and standalone activation launch; Qwen3.8 W8/A16 correctness cases pass. |
| DFlash2 selector | A 512-thread warp-shuffle top-16 reduction and packed survivor keys reduced the synthetic selector from about 565–570 µs to 327.7 µs. |
| DFlash2 proposal working set | The production selector lattice uses 272 semantic floats per token instead of a 5,120-float hidden-width temporary; focused guard tests pass. |
| DFlash2 CUDA Graphs | The proposal/selector/verify transaction is graph-safe. Matched context-512 samples reduced steady GPU round latency by 7.98% at C=1 and 7.65% at C=2. |
| NVFP4 attention | Direct packed-byte consumption and pair decoding avoid a dequantized KV tensor; scalar-order fallback remains available through `NINFER_NVFP4_PAIR=0`. |
| OSCAR-Q2/Q4 hierarchy | Independent BF16-source dual-write, fused rotated Q2/Q4 append, protected sidecars, nested KV/GDN transactions, immutable COW manifests, asynchronous host transfer ordering, and independent host-snapshot cadence are implemented and tested. |
| Boundary fix | Non-page-aligned host-prefix forks now use the ceiling page count, allowing the partial tail to be copied instead of being rejected as an invalid entitlement. |

The local DFlash2 artifact is intentionally an experimental artifact until its full artifact,
quality, and long-context gates are complete. No separate model profile is included in the current
NVFP4 public results.

## Run the current default

Download the registered NVFP4 base artifact for the stable reference:

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models
```

Run the stable baseline explicitly:

```powershell
& .\build-windows\apps\Release\ninfer-serve.exe `
  .\models\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 150000 --kv-capacity 150000 `
  --no-spec --kv-dtype bf16
```

Run the optimized local DFlash2 artifact:

```powershell
& .\build-windows\apps\ninfer-serve.exe `
  .\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --spec dflash --draft-tokens 7 --lm-head-draft `
  --kv-dtype vericache-nvfp4 --hierarchical-vericache `
  --vericache-host-snapshots
```

The optimized artifact is local to the experimental branch and is not yet a public download.

## Requirements

- Windows 11 x64 or 64-bit Linux;
- NVIDIA GeForce RTX 5090 or another compatible `sm_120a` device;
- NVIDIA driver and CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer;
- C++20-capable compiler, FFmpeg development libraries, libcurl, and Ninja.

The build targets `sm_120a`. Build and benchmark methodology is documented in
[docs/performance.md](docs/performance.md); serving options are in [docs/serving.md](docs/serving.md).

## License

VoidInfer is distributed under the Apache License 2.0. The Qwen3.8-27B source and quantized weight
repositories retain their own licenses; users are responsible for complying with them.
