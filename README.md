# VoidInfer — Qwen3.8-27B on an RTX 5090

VoidInfer is a native C++/CUDA inference engine focused on running **Qwen3.8-27B** on a single
NVIDIA GeForce RTX 5090. It supports text, image, and video prompts through the CLI and
OpenAI-/Anthropic-compatible HTTP APIs. The supported execution target is Blackwell `sm_120a`
with CUDA 13.1 or newer.

This repository's public benchmark and optimization scope is Qwen3.8-27B. Older model-specific
implementation and conversion code may remain in the source tree for compatibility, but it is not
part of the current results below.

## Current status

The native Windows Release path is buildable and the Qwen3.8-27B NVFP4 artifact passes the public
Engine/resource, text-serving, multimodal smoke, pressure-resume, and MTP settlement checks.
Optimization results are classified as local measurements until independently reproduced.

The current Strong-v1 throughput targets are:

| Concurrency | Target aggregate decode throughput |
| ---: | ---: |
| C=1 | 220–250 tok/s |
| C=4 | 600–750 tok/s |
| C=8 | 800–1,000 tok/s |

The latest exact local MTP3 saturation recheck is below those targets, so the target is not claimed
as reached:

| Concurrency | Steady decode tok/s | MTP acceptance | Speedup vs C=1 |
| ---: | ---: | ---: | ---: |
| C=1 | 168.7 | 57.2% | 1.00× |
| C=4 | 544.2 | 54.5% | 3.23× |

These points use the Qwen3.8-27B NVFP4 artifact, MTP3, CUDA Graphs, and one RTX 5090. They are
local measurements, not published external results.

## Registered Qwen3.8-27B artifacts

Both artifacts use the `qwen3_8_27b` target. The artifact identity selects the weight profile;
there is no separate runtime model flag.

| Profile | Download | Filename | Size | SHA-256 |
| --- | --- | --- | ---: | --- |
| `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |

The local DFlash2 benchmark artifact is `models/Qwen3.8-27B-NVFP4-DFlash2-NInfer/` with
`qwen3_8_27b_nvfp4.ninfer` (23,719,496,192 bytes). It resolves the `nvfp4-dflash2` profile and
is used for isolated research-serving measurements; it is not presented as a new public download
until its artifact and quality gates are complete.

The NVFP4 profile is mixed by design: Text MLP layers 0–55 use NVFP4, while the embedding,
attention and GDN projections, output head, and Text MLP layers 56–63 use row-scaled FP8. The
registered MTP and Vision objects remain in the artifact.

## Retained optimizations

Every retained change has a focused correctness check and an end-to-end measurement or a directly
measurable resource/runtime gate. Negative candidates remain documented as rejected experiments.

| Area | Retained change | Evidence |
| --- | --- | --- |
| GDN prefill | Direct fused causal-convolution/SwiGLU split into Q/K/V/Z consumers, avoiding packed-QKV extraction copies | The focused causal-convolution test passes; the prefill microbench is about 33.6 µs versus 60.22 µs for the legacy copy path |
| FP8 output projection | Dedicated large-token A16 GEMM route with a shared FP8 codec and `T >= 42` dispatch; small-token decode keeps its tuned route | Generic large-token measurements are about 1.13 ms for T=42–60; the crossover was retained only after end-to-end comparison |
| DFlash2 selector | 512-thread warp-shuffle reduction with packed survivor keys for the deterministic top-16 selector | Focused correctness passes, including same-thread lower-ID ties and guard regions; `248,320 x 8` synthetic selector work is about 327.7 µs versus 565–570 µs for the prior 128-thread path (about 42% lower) |
| DFlash2 selector working set | Semantic 272-float lattice allocation and single-lane deterministic path trace in the production DFlash2 round | Exact-width selector/path tests pass with guards; the lattice temporary is 94.7% smaller per token than the former 5,120-float hidden-width allocation. A matched RTX 5090 eager pair was round-latency neutral within run variance, so no standalone tok/s gain is claimed |
| DFlash2 CUDA Graphs | Capture the DFlash2 proposal/selector/verify transaction now that selector tracing is deterministic on-device | Focused DFlash2 and hierarchy tests pass. Matched RTX 5090 A/B runs at C=1 and C=2 reduced steady GPU round latency by 7.98% and 7.65%; published throughput improved 8.68% and 9.04% with near-matched acceptance |
| Hierarchical checkpoint manifest | Retain immutable COW KV address spaces and align them with a frozen GDN StateImage epoch before host-tier publication/replacement | Focused tests pass; a guarded context-1,024 RTX 5090 run crossed two host boundaries and completed two manifest-backed snapshots. This is an ownership/correctness milestone; host-tier logits are not independently verified |
| DFlash2 QKV | Direct three-output W8G32 row-split GEMM writes Q/K/V without materializing and copying a packed QKV tensor | Exact focused comparison passes; repeated `T=8` microbench samples show about 19–21% lower latency than packed-QKV-plus-copy |
| DFlash2 Qwen3.8 fused SwiGLU | Route the Qwen3.8 W8G32 gate/up projection directly into the fused SiLU-multiply output for DFlash2 v2, removing the 34,816-row BF16 gate/up temporary and standalone activation launch | Qwen3.8 W8_A16 correctness passes for T=1,2,8,16,32,40,41,64; a guarded RTX 5090 C8 run measured 34.2246 ms GPU round, 511.06 published tok/s, and 475/2,800 accepted draft tokens. This is retained as the default DFlash2 proposal route; acceptance variance prevents an end-to-end speedup claim |
| Resource pressure | Bounded continuation search when an incumbent plan exists, reducing pressure-resume planning work without changing the selected contract | The real Qwen3.8 pressure/resume route passes |
| State and KV ownership | Canonical State/KV resource contracts, device binding, sparse-MoE scan reuse, and successful-warmup readiness gating | Public Engine/resource and readiness tests pass; these are correctness/admission improvements, not unsubstantiated tok/s claims |
| Measurement discipline | Targeted Nsight Compute kernel inspection and Nsight Systems end-to-end traces on the RTX 5090 | Nsight Systems 2026.4.1 traces are available; Nsight Compute 2026.2.1 is installed, but hardware-counter access is blocked in the current Windows session by `ERR_NVGPUCTRPERM` |
| Hierarchical VeriCache (research default on the isolated branch) | NVFP4 DFlash2 L0 with protected recent/anchor BF16 sidecar, nested KV/GDN transactions, adaptive fallback horizon, and event-ordered asynchronous host-tier snapshots | Focused append/attention/state tests pass; a matched two-lane 5090 run measured 23.6491 ms GPU round with async host snapshots versus 23.6857 ms control. Acceptance reduced published throughput to 182.992 versus 186.502 tok/s, so no end-to-end speedup or independent host L1/L2 output verification is claimed |

The main implementation points are [src/ops/linear/fp8/](src/ops/linear/fp8/),
[include/ninfer/ops/causal_conv1d_silu.h](include/ninfer/ops/causal_conv1d_silu.h),
[src/runtime/engine/materialization_planner.h](src/runtime/engine/materialization_planner.h),
and the Qwen3.8 runtime under [src/targets/qwen3_6/](src/targets/qwen3_6/). The shared target
directory name is retained by the current source layout; the registered artifact and benchmark
identity are Qwen3.8-27B.

### DFlash2 and DSpark checkpoint

The current source tree contains a Qwen3.8-shaped DFlash2 path with five draft layers, target feature
layers `[5, 19, 33, 47, 61]`, block size 8 (`k=7`), rank 256, top-16 selection, and grouped two-tap
causal convolutions. The direct-QKV and selector changes above are source-level, focused-op validated
optimizations. The local DFlash2 `.ninfer` artifact now loads and serves through the experimental
branch, but its complete C1/C2/C4/C8, quality, tool, and vision qualification is still pending.

The locally present artifact with a DSpark-oriented filename identifies itself as the regular
Qwen3.8 `groupwise-int` profile, so it is not treated as a DSpark or DFlash2 validation artifact.
An E2E DFlash2/DSpark claim remains gated on a correctly converted artifact, quality/parity checks,
and a real Qwen3.8 serving measurement.

### Hierarchical VeriCache research track

The isolated experimental branch
[`exp/hierarchical-vericache-20260830`](https://github.com/The8Darkness/voidinfer/tree/exp/hierarchical-vericache-20260830)
uses hierarchical VeriCache and DFlash2 as the default `ninfer-serve` profile. The underlying
`EngineOptions::hierarchical_vericache` API remains opt-in, and `--no-spec` plus
`--no-hierarchical-vericache` restore the stable fallback. Its current control plane models L0 VRAM compressed speculation, L1 pinned NVFP4/FP8 mirrors,
L2 host FP16 KV plus protected GDN state, and an L3 cold manifest that is never allowed into the
frequent verifier path. DFlash2's exact target remains the correctness fallback while host-tier
consumers are developed; compression therefore affects proposal acceptance, not final greedy output.

Implemented in this track:

- direct NVFP4 local attention consumption without a dequantized KV tensor;
- an opt-in BF16 sidecar for protected recent local-KV tokens, with append, copy, and attention
  coverage and compile-time removal of its branch when disabled;
- nested speculative transactions covering DFlash/MTP KV frontiers and Qwen3.8 GDN state;
- adaptive 24–64 L0→L1 and 256–2,048 L1→L2 horizon controls, with exact-target fallback metrics
  kept separate from real host-verifier counters;
- reuse of the existing pinned StateImage, host FP16 KV extent store, prefix/state DAG, and COW
  ownership machinery for tier accounting and event-ordered asynchronous promotion.

The default server profile is VeriCache-NVFP4, DFlash2 `k=7`, and the optimized proposal head;
implicit vision requests route to MTP so BF16 vision remains protected, and DFlash2 CUDA Graphs
are enabled for the graph-safe selector path. A real default-profile startup/API smoke on the RTX
5090 loaded the DFlash2 artifact in 51.19 s, reported the
`qwen3.8-27b/nvfp4-dflash2` cost profile, returned `/health`=`ok`, and completed the full text
serving contract. This is a serving-path smoke, not a quality or throughput qualification.

The latest repeated-boundary physical RTX 5090 stress comparison uses context 512, DFlash2 `k=7`,
batch 1, optimized proposal head, no CUDA Graph, one warmup round, and 400 measured rounds:

| L0/L1 configuration | GPU round | Wall round | Published tok/s | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| VeriCache-NVFP4 control, hierarchy disabled | 22.9920 ms | 23.0041 ms | 78.2470 | 320/2,800 (11.43%) |
| Hierarchy enabled, no host snapshot | 22.7066 ms | 22.7177 ms | 73.6212 | 269/2,800 (9.61%) |
| Hierarchy + host snapshot, L1→L2 fixed at 256 | 22.6889 ms | 22.6995 ms | 70.1556 | 237/2,800 (8.46%) |

The host row completed three checkpoints with `L0/L1/L2/L3 =
26,869,760/17,367,040/221,063,168/0` bytes, 502,167,552 StateImage D2H bytes, and 71,565,312
KV D2H bytes across 33 pages. The slightly faster GPU round did not translate into a win because
acceptance was lower; this remains an experimental checkpoint/residency result, not a speed claim.

An earlier short comparison at context 2,048, DFlash2 `k=7`, batch 1, no CUDA Graph, and eight
measured rounds was:

| L0 configuration | GPU round | Wall round | Published tok/s | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 control, hierarchy disabled | 28.3265 ms | 28.3390 ms | 83.8067 | 11/56 (19.64%) |
| NVFP4 + 64 recent BF16 + 8 anchors | 27.6280 ms | 27.6389 ms | 81.4069 | 10/56 (17.86%) |
| Same hierarchy + async host snapshot | 27.5497 ms | 27.6001 ms | 81.5213 | 10/56 (17.86%) |
| NVFP4 + 64 recent BF16, no anchors | 26.0720 ms | 26.0851 ms | 91.0481 | 11/56 (19.64%) |

These are research measurements, not multiplicative claims. The host-snapshot row reports
`L0/L1/L2/L3 = 26,869,760/13,434,880/153,954,304/0` bytes and one asynchronous promotion. The
host image is currently a residency/checkpoint mechanism, not an independent verifier; L0→L1 and
L1→L2 verifier counters remain zero. NVFP4/FP8 host mirrors, NVMe persistence, greedy equality,
sampling quality, vision, and 262K/multi-agent workload matrices remain gated on their dedicated
benchmark paths. The native MTP probe now exposes the same hierarchy and transfer telemetry, but
its direct-package Qwen3.8 startup stalled after model allocation and produced no accepted result.

Rejected experiments include FP8 KV, fixed MTP5, proposal-head and graph-boundary variants,
several NVFP4/FP8 schedule changes, and alternate vocabulary-GEMM crossovers. They either lost
acceptance, regressed end-to-end throughput, failed a multimodal stop condition, or did not beat
the retained route. See [EXPERIMENTS.md](EXPERIMENTS.md) for the classification and rollback
record.

## Validation and capability results

The current native validation set includes:

- Release build with MSVC 19.44, CUDA 13.1.80, and `sm_120a`;
- 95 configured CTest entries: 90 passed and 5 expected skips for absent non-Qwen3.8 artifacts;
- DFlash2 focused operator coverage for dynamic convolution, direct QKV, predecessor IDs, lattice
  construction, top-16 selection, and selector-path tracing;
- Qwen3.8 NVFP4 text and Vision serving smoke checks;
- real pressure-resume and concurrent MTP settlement checks;
- synthetic FP8-KV prefill at 252,928 and 262,144 logical context tokens without OOM;
- long-context, reasoning, code, translation, story, and structured-output serving corpus runs.

Existing Qwen3.8-27B NVFP4 capability measurements are:

| Benchmark | Result | Correct / total |
| --- | ---: | ---: |
| IFBench, prompt-level strict | 77.00% | 231 / 300 |
| AIME 2025 | 96.67% | 29 / 30 |
| AIME 2026 | 96.67% | 29 / 30 |
| GPQA-Diamond | 90.40% | 179 / 198 |
| ERQA | 66.25% | 265 / 400 |
| RealWorldQA | 83.53% | 639 / 765 |

These are single-sample evaluation measurements, not pass@k and not a substitute for the
optimization correctness gate. Full methodology and current handoff evidence live in
[PROJECT_STATE.md](PROJECT_STATE.md), [EXPERIMENTS.md](EXPERIMENTS.md),
[docs/performance.md](docs/performance.md), and the
[Qwen3.8 NVFP4 model card](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md).

## Requirements

- Windows 11 x64 or 64-bit Linux;
- NVIDIA GeForce RTX 5090 or another compatible `sm_120a` device;
- NVIDIA driver and CUDA Toolkit 13.1 or newer;
- CMake 3.28 or newer;
- C++20-capable compiler (Visual Studio 2022/MSVC on Windows, GCC or Clang on Linux);
- FFmpeg development libraries, libcurl, and Ninja.

The build is intentionally configured for `sm_120a`.

## Build on Windows

~~~powershell
cmake -S . -B build-windows -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/BuildTools/VC/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DCMAKE_CUDA_ARCHITECTURES=120a `
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON

cmake --build build-windows --config Release --parallel
~~~

## Build on Linux

~~~bash
cmake -S . -B build -G Ninja \\
  -DCMAKE_BUILD_TYPE=Release \\
  -DCMAKE_CUDA_ARCHITECTURES=120a \\
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel
~~~

## Download and run Qwen3.8-27B

~~~bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \\
  qwen3_8_27b_nvfp4.ninfer \\
  --local-dir models
~~~

Run a text request:

~~~powershell
& .\\build-windows\\apps\\Release\\ninfer.exe `
  .\\models\\qwen3_8_27b_nvfp4.ninfer `
  --prompt "Explain prefill and decode in three sentences." `
  --max-context 16384 `
  --max-new 256 `
  --spec mtp --draft-tokens 3 --lm-head-draft
~~~

Run the HTTP server:

~~~powershell
& .\\build-windows\\apps\\Release\\ninfer-serve.exe `
  .\\models\\qwen3_8_27b_nvfp4.ninfer `
  --host 127.0.0.1 --port 8080 `
  --max-context 16384
~~~

The server defaults to hierarchical VeriCache-NVFP4 with DFlash2 `k=7`, optimized proposal
selection, host snapshots, and CUDA Graphs. Add `--vision` at startup for image or video requests;
implicit vision requests use the BF16-protected MTP route. GPU residency and enabled capabilities
are fixed when the process starts.

## Documentation

- [Documentation index](docs/README.md)
- [Performance methodology](docs/performance.md)
- [Qwen3.8-27B artifact contract](docs/maintainer/qwen3.8-27b-artifact.md)
- [Experiment registry](EXPERIMENTS.md)
- [Current project state](PROJECT_STATE.md)
- [Qwen3.8-27B groupwise-int model card](model-cards/Qwen3.8-27B-NInfer/README.md)
- [Qwen3.8-27B NVFP4 model card](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md)

## License

VoidInfer is distributed under the Apache License 2.0. The Qwen3.8-27B source and quantized
weight repositories retain their own licenses; users are responsible for complying with them.
