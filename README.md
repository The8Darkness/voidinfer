# VoidInfer

[![Windows integrity](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml/badge.svg)](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml)

VoidInfer is a native Windows C++20/CUDA inference engine for **Qwen3.8-27B on a single RTX 5090**
(Blackwell, `sm_120a`): 6-bit EXL3 weights, DFlash2 speculative decoding, up to eight concurrent
lanes, and image/video input. It began as a Windows adaptation of the NInfer codebase and is now its
own engine.

## Why VoidInfer

* **Every published token is verified against full-precision history.** Attention history lives on
  the GPU as compressed INT4 OSCAR codes; drafting runs on them, and exact VeriCache re-checks each
  drafted token against attention over the exact FP16 history in host RAM before it is published.
  The compression changes speed, not output.
* **Closest to the BF16 model.** 6 bpw EXL3 weights plus verified decoding: about 6× lower KL to
  BF16 than NInfer's 4-bit artifacts, and ~15× lower on the weights alone.
* **The most context per GPU.** The full 262,144-token window on one request, and 2.2–3.7× more
  context per lane than NInfer at two to eight concurrent requests.
* **Long agents stay resident.** Each agent's exact history lives in its own plane set in host RAM;
  resuming is a remap, not a copy, and eight ~110K-token agents run on one 32 GiB card.

## Measured against NInfer

Same RTX 5090, same day per table, token-identical prompts, greedy decoding. VoidInfer ran its
defaults (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache); NInfer `a0cf1c17` (its native Windows port
of upstream `81c8ce0`) ran its official NVFP4 and groupwise artifacts with FP8 KV.

**Quality**

| | VoidInfer | NInfer |
|---|---:|---:|
| KL to the BF16 model, served path, 4K ([10-12](docs/ninfer-comparison-2026-10-12.md)) | **0.014** | 0.080 |
| Top-1 agreement with the BF16 model, 4K | **97.2%** | 92.7% |
| Weights only: KL to the FP32 reference, 138,880 tokens in 7 domains ([10-16](docs/quality-comparison-2026-10-16.md)) | **0.0025** | 0.036 |
| Weights only: top-1 agreement / on hard (high-entropy) tokens | **98.7% / 95.1%** | 94.0% / 80.2% |
| Weights only: KL on German / Chinese / C++ and CUDA | **0.0021 / 0.0015 / 0.0007** | 0.063 / 0.038 / 0.016 |
| BF16 argmax agrees with each engine's served greedy answers, 382 GSM8K / MATH-500 / IFEval / HumanEval items ([on-policy](bench/engine-compare/data/quality/onpolicy_20261013.txt)) | **99.3%** | 96.0–96.1% |
| Paired NLL on real text, 4K–32K, 12,288 tokens | **reference** | +0.016 nats/token (95% CI excludes 0) |
| SuperGPQA, answer letter only, all 26,529 questions ([10-20](bench/quality-20261020/README.md)) | **40.21%** | 38.17% (McNemar p = 3e-23) |
| MMLU-Pro, answer letter only, all 12,032 questions | **60.14%** | 58.93% (p = 7e-6) |

On the two full-size multiple-choice sets the fidelity lead shows up as accuracy, growing with
difficulty (SuperGPQA calculation questions +3.1 points, hard +2.8). The smaller task suites (GSM8K,
MATH-500, a 1,000-question step-by-step MMLU-Pro sample, IFEval, HumanEval, competition math, RULER,
NoLiMa, MRCR to 196K, IFBench) are statistically tied.

**Long-running coding agents and AIME 2026** (Oct 23–26 2026)

| Test | Runs per engine | VoidInfer | NInfer | p |
|---|---:|---:|---:|---:|
| Coding agent recalls team decisions across 11 context compactions, ledger library | 6 | **83.3 / 86** (96.9%) | 80.2 / 86 (93.2%) | 0.015 |
| └ team decisions honored in the final code | | **19.3 / 20** | 17.3 / 20 | 0.009 |
| Same test, new project (room-booking library) | 6 | **92.0 / 93** (98.9%) | 88.5 / 93 (95.2%) | 0.037 |
| └ hidden functional tests passed | | **60.5 / 61** | 57.7 / 61 | 0.015 |
| AIME 2026, 82K-token reasoning budget, avg@4 | 4 | **95.0%** (114 / 120) | 92.5% (111 / 120) | 0.50 |
| AIME 2026, 205K-token reasoning budget, avg@8 | 8 | **98.3%** (236 / 240) | 96.7% (232 / 240) | 0.22 |
| AIME 2026, both rounds pooled | 12 | **97.2%** (350 / 360) | 95.3% (343 / 360) | 0.13 |

How these were tested:

* **Engines.** Both on the same RTX 5090. VoidInfer: release v2026.10.22 with its defaults (EXL3
  6 bpw, L0 OSCAR INT4 KV, exact VeriCache, DFlash2). NInfer: `a0cf1c17` with its NVFP4 artifact,
  FP8 KV and DFlash2. Sampling parameters were sent explicitly with every request, so both engines
  received the same settings.
* **Coding agents.**
  * **Agent loop.** The loop is mini-swe-agent style: one bash command per reply, run in a WSL Ubuntu
    sandbox. Thinking is off, sampling is T 0.7 / top_p 0.8 / top_k 20, and each reply is capped
    at 4,096 tokens. Each stage gets at most 30 agent steps.
  * **Context.** One agent per engine, both at a 208,896-token window (NInfer's single-request
    maximum).
  * **Task.** The agent builds a Python package in 12 stages. 12 coding conventions and 20 team
    decisions (4 of them later revised) appear only once, inside "team chat" messages.
  * **Compaction.** Each stage adds ~100K tokens of unrelated file reads, so the context fills up
    every stage. When a prompt would exceed 80% of the window, the agent writes a handoff summary
    and the history is replaced by the brief, the summary and the most recent turns. That makes
    about 11 compactions per run, the same count on both engines.
  * **Scoring.** The final code is scored by hidden tests: functional tests plus one test per
    decision and per convention. The two projects use different code, decisions and tests, and a
    reference solution scores full marks on each. p is a two-sided Mann-Whitney U test over runs.
* **AIME 2026.**
  * **Problems.** All 30 problems (AIME I and II), each sent as the problem text plus an instruction
    to put the final answer in `\boxed{}`. Answers are scored with `math_verify`.
  * **Settings.** Thinking is on, with Qwen's recommended sampling (T 1.0 / top_p 0.95 / top_k 20).
  * **Rounds.** Round 1 ran 2 requests at a time with an 86,016-token window and an 81,920-token
    budget, 4 samples per problem. Round 2 ran one request at a time with a 208,896-token window
    and a 204,800-token budget, 8 samples per problem; the engines alternated pass by pass, and no
    answer hit the budget. p is a paired sign-flip permutation test over the 30 problems.
* **Speed and limits.** NInfer finished every one of these tests faster: 1.5–1.7× in agent
  wall-clock time and 1.35–1.6× per AIME answer. In four further variants of the agent test, neither
  engine was significantly better: frequent revisions, facts delivered in tool output,
  22 compactions, and thinking on.

**Maximum context per lane** ([2026-10-20/21](bench/engine-compare/data/2026-10-20/README.md),
the prompt actually prefilled and answered)

| Concurrent requests | VoidInfer | NInfer | |
|---|---:|---:|---:|
| 1 | **262,144** (full window: 258,089 prompt + 4,055 generated) | 208,896 | 1.25× |
| 2 | **192,512** each | 86,016 each | 2.2× |
| 2, asymmetric | **262,144 + 135,168** | 151,552 + 38,912 | 2.1× total |
| 4 | **69,632** each | 26,624 each | 2.6× |
| 8 | **11,264** each | 3,072 each | 3.7× |

**Also**

* DFlash2 commits **21–25% more tokens per round** (3.56 / 3.67 vs 2.84 / 3.04 at 4K / 128K).
* Eight agents × ~110K tokens (865K live context) on 4 lanes: all served, first turn **28% faster**
  than NInfer (223 vs 310 s); every turn of a 3-turn run served (24/24).
* Vision: both engines answer all 8 image/video requests the same way; startup with vision takes
  **18.6 vs 33.6 s**, and the BF16 tower matches the Hugging Face FP32 tower at cosine
  0.9998–0.9999.

**Speed** ([2026-10-20/22](bench/engine-compare/data/2026-10-20/README.md)). NInfer is faster on raw
speed; the fidelity above costs decode and prefill throughput.

| | VoidInfer | NInfer |
|---|---:|---:|
| Decode, 1 request (4K prompt, 1,024 tokens) | 147 tok/s | **190 tok/s** |
| Decode, 2 requests at once (total) | 228 tok/s | **329 tok/s** |
| Prefill, 3K / 32K / 128K prompt | 3.5K / 3.9K / 3.2K tok/s | **9.3K / 10.3K / 6.2K tok/s** |
| Time to first token, 32K / 128K prompt | 8.1 / 40.8 s | **3.1 / 20.9 s** |
| Agent resume (100K-token agent) | 0.75–0.89 s | **0.13–0.26 s** |

## Model weights

| Role | Hugging Face |
|---|---|
| Target, EXL3 6 bpw | [Mia-AiLab/Qwen3.8-27B-EXL3](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3/tree/SC_6.00bpw_H6_V6), branch `SC_6.00bpw_H6_V6` |
| DFlash2 draft, EXL3 5 bpw | [Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw) (from [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)) |
| Vision tower (optional), BF16 | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |

The target directory also needs the OSCAR rotations in `l0_oscar/` (33 MB, not part of the Hugging
Face repository). The prebuilt package ships them; they are fitted on captured K/V as described in
[tools/oscar](tools/oscar/README.md) and [KV-tier fidelity](docs/OSCAR_KV_FIDELITY.md).

## Host RAM

The exact FP16 attention history is kept in host RAM: **64 KiB per token of context**, so 2 GiB per
32K tokens and 16 GiB for one full 262K window. Eight agents with ~110K tokens each (880K tokens of
context) were served on 4 lanes for 3 turns on a PC with **96 GB RAM**; in that run about four
agents' histories stayed cached and the others were prefilled again on their next turn. Details:
[host memory](docs/build-and-run.md#host-memory).

## Run the prebuilt package

Each [release](https://github.com/The8Darkness/voidinfer/releases) has a self-contained Windows zip
(executables, every DLL they need, the OSCAR rotations): extract it, download the two models above,
copy its `l0_oscar` folder into the target directory and start `voidinfer-serve.exe`. It needs an RTX
5090 and a CUDA 13.3-capable driver, nothing else. Step by step:
[prebuilt Windows package](docs/build-and-run.md#prebuilt-windows-package).

## Build from source

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

Point the server at the target and the draft:

```bat
build\apps\voidinfer-serve.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 ^
      --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw --model-id qwen3.8-27b --max-context 65536
```

Add `--vision <BF16 checkpoint directory>` for image and video input and `--max-concurrency N` for up
to eight lanes. The endpoints and options are in [serving](docs/serving.md); the one-request CLI
(`voidinfer`) is in [cli](docs/cli.md).

## Documentation

* Results: [capacity/speed/agents 2026-10-20](bench/engine-compare/data/2026-10-20/README.md),
  [quality summary and MMLU-Pro/SuperGPQA 2026-10-20](bench/quality-20261020/README.md),
  [model quality 2026-10-16](docs/quality-comparison-2026-10-16.md),
  [NInfer comparison 2026-10-12](docs/ninfer-comparison-2026-10-12.md), [KV-tier fidelity](docs/OSCAR_KV_FIDELITY.md),
  [benchmark harness](bench/engine-compare/README.md).
* How it works: [architecture](docs/architecture.md), [current status](docs/current-status.md),
  [engineering case study](docs/engineering-case-study.md).
* Using it: [build and run](docs/build-and-run.md), [serving](docs/serving.md), [CLI](docs/cli.md),
  [limitations](docs/limitations.md). Full map: [docs/README.md](docs/README.md).

## Provenance and license

VoidInfer descends from the NInfer codebase and a Windows adaptation. It contains adapted EXL3
utility code under `src/exl3/mia_exllamav3`, which keeps its own license notice. Direct dependencies,
adapted code, research references (DFlash2, OSCAR, VeriCache) and original integration work are
separated in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). GitHub fork metadata is not used as an
authorship claim.

The repository license is Apache-2.0. Model weights and third-party components keep their own terms.
