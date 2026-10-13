# VoidInfer

[![Windows integrity](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml/badge.svg)](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml)

VoidInfer is a native Windows C++20/CUDA inference engine for **Qwen3.8-27B on a single RTX 5090**
(Blackwell, `sm_120a`). It runs 6-bit EXL3 weights through hand-written CUDA kernels and does
DFlash2 speculative decoding with exact, transactional state rollback. Attention history lives on
the GPU as INT4 OSCAR codes (L0) with full-precision history in host RAM, and every drafted token is
verified against that full-precision history before it is published (exact VeriCache). One request
reaches **262,144 tokens** and eight ~108K-token agents (865K tokens) run on one 32 GiB GPU. It began as
a Windows adaptation of the NInfer codebase; the inherited artifact engine has since been removed and
VoidInfer is its own engine.

## Results at a glance (RTX 5090, 2026-10-12)

Measured head-to-head against the latest NInfer (upstream `81c8ce0` via its native Windows port
`a0cf1c17`) on the same GPU and day, with the same VS 2026 + CUDA 13.3 toolchain, token-identical
prompts and greedy decoding. VoidInfer ran its defaults (6-bit EXL3, L0 OSCAR INT4 KV tier, exact
VeriCache); NInfer ran its official NVFP4 and groupwise artifacts with FP8 KV. Full protocol, every
table and the raw data: [docs/ninfer-comparison-2026-10-12.md](docs/ninfer-comparison-2026-10-12.md).

| | VoidInfer | NInfer (best config that runs) | |
|---|---:|---:|---|
| Quality: KL divergence from the BF16 model (4K) | **0.014** | 0.080 | **~6× closer to BF16** |
| Quality: paired NLL, 4K–32K (12,288 identical tokens) | **reference** | +0.016 nats/token | 95% CI excludes 0 |
| Top-1 agreement with the BF16 model (4K) | **97.2%** | 92.7% | |
| DFlash2 decode, code workload, 4K / 16K / 32K / 64K / 128K | 180 / 169 / 145 / 160 / 158 tok/s | 187 / 187 / 190 / 195 / 172 | NInfer +3% to +32% |
| Largest single request (259,483-token prompt), decode | **137 tok/s** with every token verified | 141 tok/s, only with k8v4 KV (FP8 KV does not fit) | NInfer +3% |
| DFlash2 committed tokens / round, 4K / 128K | **3.56 / 3.67** | 2.84 / 3.04 | **+21–25%** |
| DFlash2, NInfer's 12 scenario prompts (geomean) | 212 tok/s | 247 tok/s | NInfer +16% |
| Base decode, 4K | 76 tok/s | 94 tok/s (groupwise, 15.9 GiB weights) | NInfer +23% |
| Prefill, 4K / 128K | 7.8K / 4.9K tok/s | 14.1K / 6.6K tok/s (NVFP4) | NInfer +34–81% |
| Concurrent requests, 4K, C1 / C2 / C4 aggregate | 161 / 258 / 365 tok/s | 189 / 340 / 526 tok/s | NInfer +18–44% |
| C8 at 4K, C4 / C8 at 31K | **396 / 214 / 180 tok/s** | does not start (per-slot FP8 KV exceeds VRAM) | VoidInfer only |
| 8 agents × 99K–113K tokens (865K live context), 4 lanes | all served: **223 s** first turn, 24–25 s resumed turn (52 s in the comparison run)¹ | all served (groupwise; NVFP4 does not start): 310 s, **11 s** resumed | first turn −28%, resume 2.2× slower |
| Vision: image + question, 404 prompt + 256 output tokens | 1.6–1.8 s (BF16 vision tower, every token verified)¹ | 1.0–1.3 s | NInfer |

¹ Measured after the comparison run, same day and machine: agents resume by rebinding their plane
sets instead of copying their history, and the EXL3 route gained image/video input
([update](docs/ninfer-comparison-2026-10-12.md#update-vision-and-agent-resume)).

What the numbers say:

* **Quality is VoidInfer's clear lead.** A 6-bit model whose published tokens are verified against
  full-precision attention history: about six times closer to the BF16 reference than NInfer's 4-bit
  artifacts, and measurably better on real text at every depth.
* **Capacity is the second.** One request reaches 259K tokens with every token verified, and eight
  agents with 865K tokens of live context run on one 32 GiB card, with a 28% faster first turn.
  NInfer's fastest artifact cannot reserve FP8 KV for those sizes.
* **Raw speed is NInfer's.** Its smaller weights read fewer bytes per token, so base decode, prefill
  and batched concurrency are faster, and it resumes parked agents 2.2× faster (host checkpoints).
  VoidInfer's DFlash2 commits 21–25% more tokens per round, which keeps single-request decode within
  4% at 4K and at the largest context, but verifying every token costs 10–30% at 16K–64K and on
  short, high-acceptance prompts.

## Engine direction

VoidInfer is a **quality-first long-context engine**: exact output, maximum context per GPU, then
speed.

1. **Exact output from compressed context.** Attention history lives on the GPU as INT4 OSCAR codes
   (17 KB per token instead of 64 KB), so one card holds a quarter-million-token request or several
   long agents. Drafting runs on those codes; VeriCache re-checks every drafted token against
   attention over the full-precision history streamed from host RAM, and only verified tokens are
   published, so the INT4 codes change speed, not output. (A single request verifies on an
   asynchronous line that reads history beyond the recent window through an FP8 e4m3 mirror,
   measured at KL 0.005 against FP16 history; several lanes verify against FP16.)
2. **6-bit weights.** EXL3 6 bpw keeps the model close to BF16 (KL 0.014). The bandwidth cost
   (21 GiB read per token vs 16–19 GiB) is partly paid back by higher DFlash2 acceptance.
3. **Many long agents per GPU.** Coherent-device lanes batch up to eight requests into one round,
   admit a lane only when its INT4 codes fit in free VRAM, and keep every agent's exact history in
   its own plane set in host RAM: a lane maps the set of the agent it serves behind a fixed address
   range, so resuming an agent is a remap (0.1–0.7 s at 110K tokens), not a copy. Only under host
   memory pressure are idle sets losslessly packed and spilled to disk, so total live context can
   exceed 1M tokens (8 × 125K measured on 2026-10-10).
4. **Images and video, at the same quality bar.** The vision tower runs in BF16 from the checkpoint
   (streamed from host RAM per request, no resident VRAM); its embeddings and M-RoPE positions feed
   the same verified decode.
5. **Next.** The measured gaps are prefill (layer-major GEMMs on 6-bit weights, exact-history
   ingestion), batched concurrency rounds, the exact line's GPU share at 16K–64K, and lanes admitted
   per VRAM for long agents (two of four lanes fit 108K-token agents' INT4 codes).

The 2026-10-08 comparison (older configuration: FP16 device KV at short context, no VeriCache) is kept
at [docs/ninfer-comparison-2026-10-08.md](docs/ninfer-comparison-2026-10-08.md).

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

**Prefill (~4-6K tok/s served).** Layer-major prefill with FP16 projections (MXFP8/NVFP4 block-quantized routes are opt-in) and
fused weight decode, FA2-style GQA attention with FP16-accumulated tensor-core steps, and a
chunk-parallel Gated-DeltaNet recurrence.

**L0 OSCAR with exact hot rows (default KV tier).** Attention history beyond an exact 64-token sink
and the exact 960-token recent window is stored as calibrated-rotation INT4 codes on the device (INT2 until
2026-10-11; the hot-row gate numbers below were measured with INT2), with FP16 planes in pinned host memory
(L2). History attention runs on int8 MMAs for QK and u8 MMAs for PV.
Attention mass is concentrated: a small, slowly changing set of history rows carries most of it.
So each layer keeps 512 rows per KV head as exact FP16 copies on the device (the hot rows, 32 MiB
in total; `VOIDINFER_EXL3_L0_HOT` sets 0..4096). The code-history kernel skips them, and every history CTA first attends exactly to its share of
them. The code-history pass also nominates rows whose probability crosses 0.2% of the history mass as a
side effect, and the stalest hot rows are swapped for those candidates each round (copied from L2
beside the next round's encode). On the DFlash2 verify-quality gate (4K/16K × code/prose, 2,016
teacher-forced rows) this cuts the paired NLL loss against FP16 KV from +0.024 to +0.006
nats/token and raises top-1 agreement from 95.1% to 96.8%. It costs 0.2–0.4 ms per speculative
round (1–2%). Rounds take 18.3 ms at 4K and 18.5–20.3 ms from 16K to 128K; parked agent contexts
resume in 0.3–1.9 s.

**Vision.** `--vision DIR` (a directory with the BF16 `model.visual.*` tensors, e.g. the BF16
checkpoint) enables image and video input. The 27-block tower runs in BF16 with FP32-accumulated
cuBLAS GEMMs; its 0.88 GB of weights stay in pinned host memory and stream through a two-slot device
buffer (19 ms per 1,536-patch image). Merged embeddings replace the placeholder tokens in prefill, and
a per-lane device rotary map gives prompt rows their M-RoPE positions and later rows `row + delta`,
read by every Q/K RoPE launch (captured graphs included). Against the FP32 reference tower, merged
embeddings have cosine 0.9998–0.9999 (Hugging Face BF16: 0.9999). Media prompts do not reuse or
publish prefix roots.

**VeriCache (default with L0, exact).** Block-parallel verification of L0 output against the exact
FP16 history, after VeriCache (arXiv 2605.17613). The prompt is ingested with exact FP16 history
attention. DFlash2 then drafts blocks on L0 without publishing them. One wide exact pass over the
block rescores every row against FP16 history streamed from host RAM. In exact mode (default,
`VOIDINFER_EXL3_VERICACHE=1` or `exact`), a drafted token is kept when its exact logit is within 0.0625
of the exact top logit (twice the measured decode/verifier arithmetic floor); the first token outside
is replaced by the exact argmax and the rest of the block is redrafted. Block lengths follow the
measured pass cost and mismatch rate (learned across requests); rounds may run past a block into the
same pass. `VOIDINFER_EXL3_VERICACHE=tolerance` accepts within 1.0 on fixed 1,024-token blocks;
`=0` disables. Tokens are published only after verification, and the verified prefix becomes the next
exact root (device GDN checkpoint plus L0 rewind).

With one lane (`--max-concurrency 1`) exact verification is asynchronous: an exact line (a second
context on the same FP16 planes with its own GDN state and window) runs each 64-token block on its
own stream from a worker thread while DFlash2 keeps drafting past it, so the PCIe history stream
overlaps the HBM-bound decode rounds. A correction rewinds drafting to the verified root (L0 re-run,
no history stream); the kept tokens and the exact replacement lead the next block on the line. Waits
on that path poll instead of blocking in the driver (a blocked call holds the WDDM context lock and
stalls the other thread). `VOIDINFER_EXL3_VERICACHE_ASYNC=0` restores the synchronous pass. DFlash2
code workload, 512-1,024 tokens, exact: 4K 169-173 tok/s (synchronous 151-162, no VeriCache 197),
32K 150, 128K 130-137 (synchronous 76, no VeriCache 172), 254K 107-112 (synchronous 42, no
VeriCache 158). The remaining cost is the line's own GPU work (~1.2 ms/token at 128K: projections and
exact history attention) and the drafting a correction discards (~0.1% of tokens at 128K).

Synchronous exact mode (several lanes) costs more than tolerance because L0 drafts disagree with
exact history beyond the floor on ~0.1-0.4% of tokens at 4K and ~1% at 32K, which keeps blocks at
16-32 tokens: verify passes take ~20-25% of decode time (tolerance ~8%).

Tolerance mode, measured with DFlash2 on the code workload, 1,024 output tokens:

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

* Greedy generation through the coherent-device Engine, one to eight concurrent requests
  (`--max-concurrency 1..8`), text of any length and, with `--vision`, images and video.
  Sampling uses the model-card presets (non-thinking: temperature 0.7, top-p 0.8, presence penalty
  1.5); thinking budgets, stop strings and frequency penalties are rejected. Media prompts are not
  reused as prefix roots.
* L0 OSCAR (INT4 history codes, exact sink, recent window and 512 hot rows per KV head) with exact
  VeriCache is the default at every context length; L2 history stays in host RAM (FP16 planes, losslessly
  packed when parked, spilled to disk under memory pressure). `VOIDINFER_EXL3_VERICACHE=tolerance`
  trades exactness for speed; `VOIDINFER_EXL3_L0_OSCAR=0` selects FP16 device KV (about 16K with DFlash2).
  The OSCAR rotations are read from `<model directory>/l0_oscar` (or `VOIDINFER_EXL3_L0_OSCAR_ROT`).
  `VOIDINFER_EXL3_L0_CODE_BITS=2` stores the history as INT2 codes instead (9.2 instead of 17.4 KiB per
  token on the device): published tokens stay exact, but about 4x more VeriCache corrections cost
  40-50% of decode speed (32K, 1 / 4 / 8 lanes: 93 / 33 / 23 vs 170 / 54 / 45 tok/s per lane).
* With one lane, verification runs on an asynchronous exact line over the FP16 history;
  `VOIDINFER_EXL3_VERICACHE_LINE_FP8=1` reads history beyond the recent window through an FP8 e4m3
  mirror instead (6-10% faster at 128K-254K, KL 0.005 against FP16 history). With several lanes,
  verification passes are exclusive and use FP16 history.
* Speed gaps to NInfer (2026-10-12): prefill 1.3–1.8×, base decode 1.2×, batched concurrency 1.3–1.4×,
  single-request DFlash2 at 16K–64K 1.1–1.3×, agent resume 2.2×, image requests 1.4–1.6×.
* The public Engine route is DFlash2-only; base decode is measured on the target-only route.
* The Engine applies its benchmarked kernel and route profile at startup
  (`src/exl3/engine_profile.cpp`); `VOIDINFER_*` variables only override it.
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
cmake --build build -j --target voidinfer voidinfer-serve
```

Model weights are not included. Point the server at the EXL3 target and its DFlash2 draft (the target
directory carries the OSCAR rotations in `l0_oscar/`):

```bat
build\apps\voidinfer-serve.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 ^
      --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw --model-id qwen3.8-27b --max-context 65536
```

Add `--vision <BF16 checkpoint directory>` for image and video input and `--max-concurrency N` for up
to eight lanes. The endpoints and options are in [serving](docs/serving.md); the one-request CLI
(`voidinfer`) is in [cli](docs/cli.md).

## Reading paths

* Results: [NInfer comparison 2026-10-12](docs/ninfer-comparison-2026-10-12.md) (earlier: [2026-10-08](docs/ninfer-comparison-2026-10-08.md)),
  [benchmark harness](bench/engine-compare/README.md), [KV-tier fidelity](docs/OSCAR_KV_FIDELITY.md).
* Overview: [current status](docs/current-status.md),
  [engineering case study](docs/engineering-case-study.md), [interview/demo guide](docs/interview-guide.md).
* Implementation: [architecture](docs/architecture.md), [build and run](docs/build-and-run.md),
  [serving](docs/serving.md), [CLI](docs/cli.md), [performance](docs/performance.md),
  [benchmarking](docs/benchmarking.md), [limitations](docs/limitations.md),
  [maintenance](docs/maintenance.md).

## Provenance and license

VoidInfer descends from the NInfer codebase and a Windows adaptation. It contains adapted EXL3
utility code under `src/exl3/mia_exllamav3`, which keeps its own license notice. Direct dependencies,
adapted code, research references (DFlash2, OSCAR) and original integration work are separated in
[UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). GitHub fork metadata is not used as an authorship claim.

The repository license is Apache-2.0. Model weights and third-party components keep their own terms.
