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
- hierarchical VeriCache with OSCAR-Q2 target L0, a separate BF16 DFlash2 local drafter cache, a
  pinned-host OSCAR-Q4 mirror, and an authoritative FP16 host KV state; the current path has no
  persistent device Q4 shadow and no CPU logit verifier;
- exact target verification and nested KV/GDN rollback on every speculative round;
- 128 recent tokens plus sink/pivot anchors protected in BF16 for DFlash acceptance, and BF16-protected
  vision/non-DFlash execution when multimodal input is requested.

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
| L0 target KV in VRAM | Target KV is packed `U8` OSCAR-Q2 with independently fitted affine BF16 scale/zero metadata. Recent/sink/pivot anchors use the protected BF16 sidecar. | Authoritative target attention path. |
| DFlash2 local drafter KV | BF16 fixed cyclic K/V state, separate from target L0. | Fast speculative proposal/attention path; exact target verification remains authoritative. |
| L1 pinned system RAM | Packed `U8` OSCAR-Q4 mirror with affine BF16 metadata and protected sidecars, derived during host promotion from the resident Q2 state. No device Q4 shadow is allocated. | Host intermediate/persistence tier; it is not a per-token live logit verifier. |
| L2 system RAM | Full authoritative target KV in FP16 host records plus protected high-precision Qwen3.8 GDN state. | Correctness and restore tier for host checkpoints. |
| L3 NVMe | No active cold writer in the current serving path; measured L3 bytes are zero. | Reserved for future cold persistence and never used in frequent verification. |

Important: selecting `VeriCache-NVFP4` enables the experimental hierarchy; it does not make an
approximate cache authoritative. The default DFlash2 path executes with a Q2 target L0 and a
separate BF16 local drafter cache, promotes Q2-derived Q4 to pinned RAM for target persistence, and
settles output with the exact target. `NINFER_DFLASH_FULL_BF16=1` is an opt-in diagnostic that keeps
DFlash2's full-attention cache in BF16; it did not improve the measured result and is not the
default.

Before the BF16 local-drafter change, the context-512 probe reported `L0/L1/L2/L3` live payload
bytes of `23,633,920 / 22,740,992 / 187,508,736 / 0`, with `0` Q4-shadow bytes. Those measurements
are retained as pre-change history; the BF16 local-drafter configuration requires a fresh benchmark
before new current throughput or residency claims are made.

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

The matched context-512 CUDA-Graph probe below predates the BF16 local-drafter change: it used Q2
target L0, a compressed DFlash full cache, 128 protected recent tokens, and five measured rounds.
Its results remain historical until the BF16 local-drafter path is rebenchmarked.

For diagnosis, standalone OSCAR-Q4 L0 reached `219.568` tok/s with `20/35` accepted and a similar
round time; this shows that the remaining shadow-free Q2 gap is primarily local DFlash acceptance,
not full-attention BF16 storage. The prior device-Q4-shadow commit remains a historical reference
(`176.153` tok/s in a fresh five-round build); its extra representation is not used by the current
default. The 800K aggregate plan and 262K single-stream confidence target remain benchmark work,
not current capacity claims.

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
| OSCAR-Q2/Q4 hierarchy | Fused rotated Q2 append, direct packed-byte OSCAR attention, Q2-derived host Q4 promotion, FP16 host L2 conversion, 128-token BF16 recent protection, protected sidecars, nested KV/GDN transactions, immutable COW manifests, asynchronous host transfer ordering, and independent host-snapshot cadence are implemented and tested. |
| DFlash full-cache A/B | A real BF16 full-attention cache is selectable with `NINFER_DFLASH_FULL_BF16=1`; it measured 204.727 versus 205.172 tok/s in the matched protected-Q2 probe, so it remains a diagnostic mode. |
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
  C:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 8192 --kv-capacity 8192 `
  --spec dflash --draft-tokens 7 --lm-head-draft `
  --kv-dtype vericache-nvfp4 --hierarchical-vericache `
  --vericache-host-snapshots
```

The optimized artifact is local to the experimental branch and is not yet a public download.

The benchmark records `dflash_full_attention_cache` as `oscar-q2-device` by default or
`bf16-device` with `NINFER_DFLASH_FULL_BF16=1`. The host Q4 mirror is persistence/intermediate data;
the current branch does not claim that pinned RAM runs a live per-token logit verifier.

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
