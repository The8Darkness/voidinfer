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
| Resource pressure | Bounded continuation search when an incumbent plan exists, reducing pressure-resume planning work without changing the selected contract | The real Qwen3.8 pressure/resume route passes |
| State and KV ownership | Canonical State/KV resource contracts, device binding, sparse-MoE scan reuse, and successful-warmup readiness gating | Public Engine/resource and readiness tests pass; these are correctness/admission improvements, not unsubstantiated tok/s claims |
| Measurement discipline | Targeted Nsight Compute kernel inspection and Nsight Systems end-to-end traces on the RTX 5090 | Nsight Systems 2026.4.1 traces are available; Nsight Compute 2026.2.1 is installed, but hardware-counter access is blocked in the current Windows session by `ERR_NVGPUCTRPERM` |

The main implementation points are [src/ops/linear/fp8/](src/ops/linear/fp8/),
[include/ninfer/ops/causal_conv1d_silu.h](include/ninfer/ops/causal_conv1d_silu.h),
[src/runtime/engine/materialization_planner.h](src/runtime/engine/materialization_planner.h),
and the Qwen3.8 runtime under [src/targets/qwen3_6/](src/targets/qwen3_6/). The shared target
directory name is retained by the current source layout; the registered artifact and benchmark
identity are Qwen3.8-27B.

Rejected experiments include FP8 KV, fixed MTP5, proposal-head and graph-boundary variants,
several NVFP4/FP8 schedule changes, and alternate vocabulary-GEMM crossovers. They either lost
acceptance, regressed end-to-end throughput, failed a multimodal stop condition, or did not beat
the retained route. See [EXPERIMENTS.md](EXPERIMENTS.md) for the classification and rollback
record.

## Validation and capability results

The current native validation set includes:

- Release build with MSVC 19.44, CUDA 13.1.80, and `sm_120a`;
- 94 configured CTest entries: 89 passed and 5 expected skips for absent non-Qwen3.8 artifacts;
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
  --max-context 16384 `
  --spec mtp --draft-tokens 3 --lm-head-draft
~~~

Add `--vision` at startup for image or video requests. GPU residency and enabled capabilities are
fixed when the process starts.

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
