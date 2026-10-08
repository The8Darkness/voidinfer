# VoidInfer

[![Windows integrity](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml/badge.svg)](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml)

VoidInfer is a native Windows C++20/CUDA inference engine for **Qwen3.8-27B on a single RTX 5090**
(Blackwell, `sm_120a`). It runs 6-bit EXL3 weights through hand-written CUDA kernels and does
DFlash2 speculative decoding with exact, transactional state rollback. Its default KV tier
(L0 OSCAR: INT2 history on the GPU with an exact FP16 hot-row cache, FP16 planes in host RAM)
stretches one request to **262,144 tokens** on a 32 GiB GPU. It began as a Windows adaptation of
the NInfer codebase and has since grown into its own engine.

## Results at a glance (RTX 5090, 2026-10-08)

Measured head-to-head against the latest NInfer (upstream `81c8ce0` via its native Windows port).
Both engines were built with the same VS 2026 + CUDA 13.3 toolchain, run on the same GPU, and given
token-identical prompts with greedy decoding. Full method and tables:
[docs/ninfer-comparison-2026-10-08.md](docs/ninfer-comparison-2026-10-08.md).

| | VoidInfer | NInfer (best config) | |
|---|---:|---:|---|
| DFlash2 decode, 4K code workload | **211 tok/s** | 191 tok/s | **+11%** |
| DFlash2 decode, 128K context | **180 tok/s** | 172 tok/s | **+5%** |
| DFlash2 committed tokens / round, 4K | **3.74** | 3.05 | **+23%** |
| DFlash2, NInfer's 12 scenario prompts (geomean) | 241 tok/s | 252 tok/s | −5% |
| Model quality, paired NLL, 4K–16K | **best of all configs** | +0.017–0.020 nats/token | 95% CI excludes 0 |
| Base decode, 4K | 79 tok/s | 94 tok/s (16 GiB weights) | −16% |
| Prefill, 4K / 16K | 10.1K / 9.6K tok/s | 13.6K / 13.0K tok/s (NVFP4) | −26% |

What the numbers say:

* VoidInfer gives the highest-quality output of every configuration tested, and its speculative
  decoding commits up to 23% more tokens per verification round.
* NInfer wins on raw bandwidth-bound work (base decode, prefill) because it runs smaller 4-bit
  weight formats. Per byte of weights read, VoidInfer's decode is faster.
* At 128K context VoidInfer is faster than NInfer while keeping only INT2 codes plus an exact window
  on the GPU. Its quality there is statistically level with NInfer's FP8 KV.

These rows were measured before L0 OSCAR with exact hot rows became the default KV tier: the 4K and
16K VoidInfer rows used FP16 device KV, which stays available with `NINFER_EXL3_L0_OSCAR=0`.
The hot rows have since cut L0's quality loss against FP16 KV by about three quarters (below).

## What is inside

**Decode path (single row, ~12.6 ms/token at 4K).** Packed EXL3 GEMV producers fuse the Hadamard
input transform into their prologue. Split-K reductions carry the residual add and the next RMSNorm.
Q/K split, norms and RoPE run as one launch. Greedy argmax is split across CTAs. Programmatic
dependent launch overlaps kernel prologues. A side stream prefetches the next layer's weights into
L2. The whole 64-layer decode stack is captured as one CUDA graph.

**DFlash2 speculative decoding.** A 5-layer block-diffusion draft proposes 7 tokens per round, and
the target verifies a per-round **verification tree** (a main chain plus sibling leaves at any depth,
8 rows) built from calibrated hit tables. Rejected rows roll back exactly: device KV transactions
with checkpoint graphs, and retained-prefix GDN repair overlapped with the next draft. On the
campaign fixtures the tree produces the same tokens as chain-only verification. Draft-side work runs on split-K tensor-core ring attention, a fused
warp-per-candidate selector and a vocabulary-prefix draft head.

**Prefill (~10K tok/s).** Layer-major prefill with MXFP8/NVFP4 block-quantized projection routes and
fused weight decode, FA2-style GQA attention with FP16-accumulated tensor-core steps, and a
chunk-parallel Gated-DeltaNet recurrence.

**L0 OSCAR with exact hot rows (default KV tier).** Attention history beyond an exact 64-token sink
and the exact 960-token recent window is stored as calibrated-rotation INT2 codes on the device, with FP16 planes
in pinned host memory (L2). History attention runs on int8 MMAs for QK and u8 MMAs for PV.
Attention mass is concentrated: a small, slowly changing set of history rows carries most of it.
So each layer keeps 512 rows per KV head as exact FP16 copies on the device (the hot rows, 32 MiB
in total; `NINFER_EXL3_L0_HOT` sets 0..4096). The INT2 kernel skips them, and every history CTA first attends exactly to its share of
them. The INT2 pass also nominates rows whose probability crosses 0.2% of the history mass as a
side effect, and the stalest hot rows are swapped for those candidates each round (copied from L2
beside the next round's encode). On the DFlash2 verify-quality gate (4K/16K × code/prose, 2,016
teacher-forced rows) this cuts the paired NLL loss against FP16 KV from +0.024 to +0.006
nats/token and raises top-1 agreement from 95.1% to 96.8%. It costs 0.2–0.4 ms per speculative
round (1–2%). Rounds take 18.3 ms at 4K and 18.5–20.3 ms from 16K to 128K; parked agent contexts
resume in 0.3–1.9 s.

**VeriCache (opt-in, `NINFER_EXL3_VERICACHE=1`).** Block-parallel verification of L0 output
against the exact FP16 history, after VeriCache (arXiv 2605.17613). The prompt is ingested with exact
FP16 history attention. DFlash2 then drafts blocks of up to 1,024 tokens on L0 without publishing
them. One wide exact pass over the block rescores every row against FP16 history streamed from host
RAM. In tolerance mode (default, `NINFER_EXL3_VERICACHE_DELTA=1.0`), a token is accepted when its
exact logit is within δ of the exact top logit. The first token outside δ is replaced by the exact
argmax and the rest of the block is redrafted. `NINFER_EXL3_VERICACHE=exact` requires exact-greedy
agreement on 64-token blocks. Tokens are published only after verification, and the verified prefix
becomes the next exact root (device GDN checkpoint plus L0 rewind).

Measured with DFlash2 on the code workload, 1,024 output tokens:

* Tolerance mode costs +8% per round at 16K (20.06 vs 18.56 ms) and +9% at 64K (21.21 vs 19.45 ms).
* At 128K each verifier pass takes about 0.6 s. A run with one correction (a 0.53 s fix pass)
  cost +19% (24.3 vs 20.3 ms). Exact prompt ingestion adds about 30% to prefill (33.3 vs 25.6 s).
* A typical 1,024-token block needs no correction, and 13–23 accepted tokens are not the exact
  argmax (all within δ).
* The verifier pass is compute-bound: weight reconstruction plus FP16 GEMMs, and history attention
  at about 1 ms per 1K context per pass.

**Engineering discipline.** Every speed-up is gated by an exact oracle or a paired quality check.
Rejected ideas stay documented with their measurements (see
[docs/engineering-case-study.md](docs/engineering-case-study.md)).

## Progress since the last published snapshot

The `main` snapshot of 2026-09-22 published the R608 harness at 45–48 useful tok/s (grouped wall,
including context creation). The Sept 23 – Oct 8 optimization campaign set three goals:

| Goal | Target | Measured 2026-10-08 |
|---|---:|---|
| Cold prefill | 9,000 tok/s | **met**: 10.1K tok/s at 4K, 9.6K at 16K |
| Target-only (base) decode | 80 tok/s | **met within noise**: 79–80 tok/s at 4K (79.7 on the campaign fixture) |
| DFlash2 decode | 250 tok/s | **met on NInfer's code/structured/translation prompts** (244–445 tok/s); 211 tok/s on the harder 4K code-review workload |

It also added L0 OSCAR (256K contexts), multi-agent context parking, KV-tier fidelity instruments,
verification trees, and the CUDA 13.3 / VS 2026 toolchain move. The full commit history is on `main`.

## Current state and limitations

* Greedy text generation, one request at a time (physical C1); a bounded C2 configuration exists.
  Positive-temperature sampling and EXL3 vision input are not broadly qualified.
* L0 OSCAR with exact hot rows is the default at every context length; it costs +0.006 nats/token
  in paired NLL against FP16 device KV, and at 4K about 3% round time and 5% prefill.
  `NINFER_EXL3_L0_OSCAR=0` selects FP16 device KV (DFlash2 contexts to ~16K on 32 GiB) and
  `NINFER_EXL3_L0_HOT=0` turns the hot rows off. The OSCAR rotations are read from
  `<model directory>/l0_oscar` (or `NINFER_EXL3_L0_OSCAR_ROT`).
* VeriCache is opt-in on the coherent-device Engine route and costs +8–9% round time.
  Plain concurrent streams do not overlap the verifier pass with drafting. SM partitions (green
  contexts) would first need the cooperative decode GEMVs re-gridded for a smaller SM budget.
* The public Engine route is DFlash2-only; base decode is measured on the target-only route.
* Research routes and default-off kernels keep explicit dispositions in
  [docs/current-status.md](docs/current-status.md) and are not presented as supported defaults.

## Build and try it

Requirements: Windows 11 x64, an RTX 5090, **Visual Studio 2026 Build Tools** (MSVC 14.51) with the
C++ workload, **CUDA 13.3**, CMake 3.28+, Ninja, and the pinned vcpkg dependencies described in
[build and run](docs/build-and-run.md). The build accepts only `CMAKE_CUDA_ARCHITECTURES=120a`.

From a `cmd` shell:

```bat
call C:\BuildTools2026\Common7\Tools\VsDevCmd.bat -arch=amd64
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DCMAKE_CUDA_COMPILER="%CUDA_PATH_V13_3%\bin\nvcc.exe" -DBUILD_TESTING=ON
cmake --build build --target ninfer-serve
```

Model weights are not included. The EXL3 serving route and its options are described in
[docs/exl3-serving.md](docs/exl3-serving.md).

## Reading paths

* Results: [NInfer comparison](docs/ninfer-comparison-2026-10-08.md),
  [benchmark harness](bench/engine-compare/README.md), [KV-tier fidelity](docs/OSCAR_KV_FIDELITY.md).
* Overview: [current status](docs/current-status.md),
  [engineering case study](docs/engineering-case-study.md), [interview/demo guide](docs/interview-guide.md).
* Implementation: [architecture](docs/architecture.md), [build and run](docs/build-and-run.md),
  [benchmarking](docs/benchmarking.md), [limitations](docs/limitations.md),
  [maintenance](docs/maintenance.md).

## Provenance and license

VoidInfer descends from the NInfer codebase and a Windows adaptation. It contains adapted EXL3
utility code under `src/exl3/mia_exllamav3`, which keeps its own license notice. Direct dependencies,
adapted code, research references (DFlash2, OSCAR) and original integration work are separated in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). GitHub fork metadata is not used as an authorship claim.

The repository license is Apache-2.0. Model weights and third-party components keep their own terms.
