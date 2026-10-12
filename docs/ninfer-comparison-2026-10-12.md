# VoidInfer vs NInfer: full comparison (2026-10-12)

Head-to-head measurement of VoidInfer's current default engine against the latest NInfer on one RTX 5090,
run back-to-back on the same machine and day with token-identical inputs. It covers single-request speed
from 4K to 254K context, NInfer's 12 scenario prompts, base decode and prefill, model quality against a
BF16 reference, concurrent requests (C1–C8), multi-agent long context, the largest single context and
vision input. The earlier [2026-10-08 comparison](ninfer-comparison-2026-10-08.md) measured an older
VoidInfer configuration (FP16 device KV at ≤16K, no VeriCache) and is kept for history.

Raw data and every script are in [`bench/engine-compare/`](../bench/engine-compare/README.md)
(`data/2026-10-12/`).

## Summary

| Metric | VoidInfer | NInfer (best configuration that runs) | Result |
|---|---:|---:|---|
| KL divergence from the BF16 model, 4K (8 × 512 positions) | **0.0137 ± 0.0057** | 0.0801–0.0871 | VoidInfer **~6× closer** |
| Top-1 agreement with the BF16 model, 4K | **97.2%** | 92.4–92.7% | VoidInfer |
| Paired NLL on identical tokens, 4K–32K (12,288 tokens) | reference | +0.0157 to +0.0168 nats/token | VoidInfer **better**, 95% CI excludes 0 |
| DFlash2 decode, code workload, 4K | 180.3 tok/s | 186.6 (NVFP4) | NInfer +3% |
| DFlash2 decode, 16K / 32K / 64K | 169.3 / 144.7 / 159.8 | 187.4 / 190.3 / 195.3 | NInfer +11% / +32% / +22% |
| DFlash2 decode, 128K | 157.9 | 171.9 | NInfer +9% |
| Largest single request (259,483-token prompt) | 136.6 tok/s, TTFT 94.8 s | 140.5 tok/s, TTFT 65.2 s (NVFP4 with k8v4 KV; FP8 KV does not fit) | NInfer +3% |
| DFlash2 committed tokens per round, 4K / 128K / 254K | **3.56 / 3.67 / 3.63** | 2.84 / 3.04 / 2.76 | VoidInfer **+21–32%** |
| DFlash2 on NInfer's 12 scenario prompts (geomean) | 212.1 | 246.8 | NInfer +16% |
| Base decode, 4K / 32K / 128K | 76.4 / 74.0 / 65.7 | 94.3 / 89.8 / 77.6 (groupwise) | NInfer +18–23% |
| Prefill (target only), 4K / 32K / 128K | 7,785 / 6,890 / 4,882 tok/s | 14,127 / 11,729 / 6,559 (NVFP4) | NInfer +34–81% |
| Concurrent requests, 4K: C1 / C2 / C4 aggregate | 161 / 258 / 365 tok/s | 189 / 340 / 526 | NInfer +18–44% |
| Concurrent requests: C8 at 4K, C4 / C8 at 31K | 396 / 214 / 180 tok/s | does not start with per-slot FP8 KV | VoidInfer runs them |
| 8 agents × 99K–113K tokens (865K), 4 lanes, first turn | **223 s**, 8/8 | 310 s, 8/8 (groupwise; NVFP4 does not start) | VoidInfer **−28% wall** |
| Same agents, resumed second turn | 52 s | **11 s** | NInfer **4.6× faster** |
| Vision input (image + question) | not supported (EXL3 route is text-only) | supported, 1.0–1.3 s per request | NInfer |

**Where VoidInfer leads:** model quality (6-bit weights, and every published token verified against
full-precision history), DFlash2 acceptance per round, and capacity. It runs a 259K-token request
without dropping below FP8 history, C8 at 4K and C4/C8 at 31K where NInfer's per-slot reservation does
not start, and its first turn over 865K tokens of agent context is 28% faster.

**Where NInfer leads:** everything bound by weight bandwidth or batch throughput. Its 4-bit artifacts
read 16–19 GiB per token against VoidInfer's 21 GiB, so base decode, prefill and batched concurrency
are faster. Its DFlash2 decode is 3–32% faster on a single request despite lower acceptance, because
VoidInfer's exact verification line costs GPU time beside drafting (most at 16K–64K). NInfer resumes
parked agents 4.6× faster (host checkpoint restore, 1–11 s per agent, against VoidInfer's exchange of
packed FP16 history through host memory). It also serves image input.

**Bug found and fixed during this run.** The first pass of VoidInfer's scenario prompts failed on every
request that ended at EOS before its token budget (5 of 12 prompts): the asynchronous VeriCache line
refused to settle a block on a round closed by the terminal token. Fixed in `98f10860` (the round
accepts a closed phase, as the synchronous path already did); scenarios, base decode, concurrency,
multi-agent and vision were then run on the fixed binary. The first-pass rows are kept in
`data/2026-10-12/ours/results.first-pass.jsonl`.


## Protocol

### Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32,607 MiB, driver 617.14, VBIOS 98.02.2e.40.af, 575 W power limit, PCIe 5.0 x16 (measured 56–57 GB/s host↔device), memory overclocked by the owner (max memory clock reported 14,001 MHz) |
| CPU / RAM | AMD Ryzen 9 9950X3D (16 cores), 95.7 GiB |
| OS | Windows 11 Pro build 26300, WDDM driver model |
| Toolchain (both engines) | Visual Studio 2026 Build Tools (MSVC 14.51), CUDA 13.3.1, `sm_120a`, Release |

### Engines and artifacts

| | VoidInfer | NInfer |
|---|---|---|
| Source | this repository, branch `feat/concurrency`: code-workload rows @ `6d03542b`, all other VoidInfer rows @ `98f10860` (adds only the terminal-round VeriCache fix); the commit that adds this report changes only docs and the harness | [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) @ `a0cf1c17`, the native Windows port, fully merged with upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) `81c8ce0`; unmodified |
| Target weights | Qwen3.8-27B EXL3 6 bpw (H6 head), 21.0 GiB | official `qwen3_8_27b_nvfp4.ninfer` (NVFP4, 19.0 GiB resident) and `qwen3_8_27b.ninfer` (groupwise int, 15.9 GiB) |
| Draft | DFlash2 companion, EXL3 5 bpw, K=7, per-round verification tree | DFlash2 companion embedded in the artifact, K=7, `--lm-head-draft` |
| KV cache | default engine: L0 OSCAR (INT4 history codes on device, exact sink + recent window + 512 exact hot rows per KV head, FP16 history in host RAM) with **exact VeriCache** (every published token is checked against attention over the exact FP16 history) | FP8 E4M3 (NInfer's recommended and published setting); at 254K also NInfer's smaller `k8v4`/`nvfp4` KV where FP8 does not fit |
| Prefill chunk | engine default | `--prefill-chunk 8192` (the port's launcher setting) |
| Server flags | `ninfer-serve --exl3-round coherent-device --max-concurrency N --greedy --no-thinking` | `ninfer-serve --spec dflash2 --draft-tokens 7 --lm-head-draft --greedy --no-thinking --kv-dtype fp8` |

VoidInfer ran with its defaults: the L0 OSCAR tier and exact VeriCache are on at every context length,
and no environment override changes numerics. The FP16-KV device path of the 2026-10-08 report is no
longer used for serving.

### Method

* **Order and isolation.** One GPU job at a time on an otherwise idle machine; NInfer's single-request
  matrix first, then VoidInfer's, then concurrency (engines alternating per configuration and repeat),
  multi-agent, and vision. Each server is started fresh per configuration.
* **Prompts.** Identical system + user chat messages for both engines (`.pmsg` for VoidInfer's Engine
  harness, OpenAI JSON for `ninfer-serve`), Qwen3.8 chat template, thinking disabled. Both engines report
  the same prompt token counts (4,049 / 16,321 / 32,638 / 65,338 / 130,734; 254K: 259,483).
  Workloads: (1) a context-scaling code-review prompt (C++ source, then a request for a 40-item review);
  (2) NInfer's own 12 scenario prompts (code, story, structured, translation, ~2.1K tokens).
* **Decoding.** Greedy (argmax). DFlash2: 1,024 output tokens (or EOS). Base decode: 256 tokens.
* **Repeats.** NInfer: a warm request, then 2 measured requests per prompt (254K: 1). VoidInfer: each
  measured request follows a same-length warm request in-process; 4K–64K and the scenarios were run
  twice, 128K/254K and base once. Tables report means.
* **Rates.** NInfer: the server's `timings` (prefill = prompt tokens / time to first token; decode =
  generated tokens / generation time). VoidInfer: the Engine's prefill and decode timers (decode =
  (n−1) / decode time). Tokens per round = committed tokens / verification rounds (NInfer: drafted
  tokens / 7 rounds).
* **Base decode and target-only prefill.** NInfer: `--spec none`. VoidInfer: the target-only route
  replaying the exact prompt ids exported by the Engine, with the L0 tier on.
* **Concurrency.** One server with `--max-concurrency N`; one warm request with a distinct prompt, then N
  simultaneous greedy requests (`[agent i]`-prefixed variants of one fixture, so no request shares a
  prefix with the warm one). Aggregate = all completion tokens / wall time of the batch, prefill included.
  NInfer: NVFP4, FP8 KV, DFlash2, KV capacity = N × max context, run with and without prefix reuse
  (VoidInfer's server reuses shared prefixes by default, and the fixtures put the code in a system
  message that all requests share; NInfer's prefix reuse did not change its result).
* **Multi-agent.** 8 agents, each a disjoint 400,000-character (99K–113K-token) slice of one C++ source with a review request,
  sent simultaneously; turn 2 appends each agent's reply and a follow-up question (same conversation, so
  prefix reuse applies on both engines). Both servers run 4 lanes; NInfer uses `--kv-capacity auto` and
  `--host-context-mib 48000` for paused requests, VoidInfer its default admission, host KV planes and
  disk spill.
* **Quality.** Paired causal NLL on NInfer's own `ninfer-ppl-1m-v1` corpus: 2 streams × 4 domains
  (English reference, English long-form, Chinese reference, C++/CUDA code) at depths 4K/16K/32K; every
  engine scores the identical 512 target tokens given the identical history (NInfer
  `ninfer-perplexity --context W --stride 512`, second window; NInfer's token digests are asserted equal to
  the ids VoidInfer scores). KL divergence against a BF16 reference model (top-k records, 8 fixtures ×
  512 positions at 4K). VoidInfer is measured on its verification path: teacher-forced logits with
  attention over the full FP16 history, i.e. what VeriCache checks drafted tokens against (L0 drafting
  never publishes a token whose verified logit is more than 0.0625 below the verified maximum). With one
  lane, the asynchronous verification line reads history beyond the recent window through an FP8 e4m3
  mirror (per row and head scale); that mirror was measured separately at KL 0.005 / top-1 99.0–99.2%
  against FP16 history (16K/32K, 2026-10-11). NInfer's quality records come from the same binary
  (`a0cf1c17`), recorded 2026-10-08/10.

## Results

Means of the repeats. VoidInfer's code-workload rows come from binaries at `6d03542b`, everything else
from `98f10860`; the only difference is the terminal-round fix, which the code-workload requests (all
1,024 tokens, no EOS) never reach.

### DFlash2 greedy decode, code-review workload (tok/s, 1,024 outputs)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 180.3 | 186.6 | 177.2 |
| 16K (16,321) | 169.3 | 187.4 | 171.0 |
| 32K (32,638) | 144.7 | 190.3 | 152.6 |
| 64K (65,338) | 159.8 | 195.3 | 157.1 |
| 128K (130,734) | 157.9 | 171.9 | 134.8 |
| 254K (259,483) | 136.6 | — | 118.1 |

### DFlash2 committed tokens per verification round

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 3.56 | 2.84 | 3.05 |
| 16K (16,321) | 3.37 | 2.90 | 3.00 |
| 32K (32,638) | 2.95 | 3.00 | 2.71 |
| 64K (65,338) | 3.40 | 3.20 | 2.87 |
| 128K (130,734) | 3.67 | 3.04 | 2.63 |
| 254K (259,483) | 3.63 | — | 2.59 |

### Time to first token with the DFlash2 draft resident (s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 0.56 | 0.29 | 1.17 |
| 16K (16,321) | 2.33 | 1.25 | 4.68 |
| 32K (32,638) | 5.52 | 2.81 | 9.55 |
| 64K (65,338) | 13.34 | 7.05 | 20.29 |
| 128K (130,734) | 33.18 | 20.10 | 46.07 |
| 254K (259,483) | 94.80 | — | 114.61 |

### Prefill throughput with the DFlash2 draft resident (tok/s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 7,290 | 13,878 | 3,465 |
| 16K (16,321) | 7,007 | 13,087 | 3,490 |
| 32K (32,638) | 5,908 | 11,605 | 3,419 |
| 64K (65,338) | 4,898 | 9,272 | 3,219 |
| 128K (130,734) | 3,941 | 6,504 | 2,838 |
| 254K (259,483) | 2,737 | — | 2,264 |

### Base (non-speculative) greedy decode (tok/s, 256 outputs)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 76.4 | 82.4 | 94.3 |
| 32K (32,638) | 74.0 | 78.8 | 89.8 |
| 128K (130,734) | 65.7 | 69.1 | 77.6 |

### Prefill throughput, target only (tok/s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 7,785 | 14,127 | 3,473 |
| 32K (32,638) | 6,890 | 11,729 | 3,412 |
| 128K (130,734) | 4,882 | 6,559 | 2,832 |

### Largest single request: 254K-token prompt (259,483 tokens), DFlash2, 1,024 outputs

| Engine / KV | Decode (tok/s) | Tokens / round | Prefill (tok/s) | Time to first token (s) |
|---|---:|---:|---:|---:|---:|
| VoidInfer, L0 OSCAR INT4 + exact VeriCache | 136.6 | 3.63 | 2,737 | 94.8 |
| NInfer NVFP4 / FP8 KV | does not start: KV reservation exceeds free VRAM | | | |
| NInfer NVFP4 / k8v4 KV | 140.5 | 2.76 | 3,977 | 65.2 |
| NInfer NVFP4 / NVFP4 KV | 133.6 | 2.92 | 3,017 | 86.0 |
| NInfer groupwise / FP8 KV | 118.1 | 2.59 | 2,264 | 114.6 |

### DFlash2 on NInfer's 12 scenario prompts (~2.1K-token prompts, ≤1,024 outputs, tok/s)

| Scenario | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| code_cuda | 279.2 (495) | 252.7 (1,024) | 255.8 (1,024) |
| code_python | 207.7 (1,024) | 259.5 (1,024) | 222.5 (1,024) |
| code_typescript | 249.5 (149) | 342.7 (94) | 264.4 (42) |
| story_en_mystery | 159.6 (1,024) | 174.6 (1,024) | 155.1 (1,024) |
| story_zh_dialogue | 97.9 (1,024) | 104.6 (1,024) | 91.6 (1,024) |
| story_zh_scifi | 113.1 (1,024) | 115.9 (1,024) | 102.2 (1,024) |
| structured_csv | 250.5 (1,024) | 382.1 (1,024) | 335.2 (1,024) |
| structured_jsonl | 359.0 (1,024) | 441.2 (336) | 412.2 (1,024) |
| structured_sql | 280.7 (1,024) | 374.3 (1,024) | 310.2 (1,024) |
| translation_en_zh | 231.1 (874) | 199.8 (1,024) | 243.2 (754) |
| translation_markdown | 191.6 (815) | 223.3 (752) | 227.7 (861) |
| translation_zh_en | 289.5 (943) | 382.3 (725) | 335.9 (719) |
| **geometric mean** | **212.1** | **246.8** | **225.6** |

Parenthesized: generated tokens (both engines stop at EOS; greedy text differs between weight formats).

### Concurrent requests: aggregate greedy DFlash2 decode (tok/s, N × 1,024 outputs)

Aggregate = all completion tokens / wall time of the batch (prefill included). Mean of 2 runs; per-request range in brackets. NInfer: NVFP4, FP8 KV, `--kv-capacity` = N × max context (KV reserved per slot), prefix reuse off and on (VoidInfer reuses shared prefixes by default; the fixtures put the code in a system message all requests share). Where the reservation exceeds free VRAM the server does not start; a shared `--kv-capacity auto` pool was not measured (the follow-up run was stopped). Ratio = VoidInfer / faster NInfer.

| Prompt | Concurrency | VoidInfer | NInfer NVFP4 / FP8 KV, no prefix reuse | NInfer, prefix reuse | Ratio |
|---|---:|---:|---:|---:|---:|
| code_31k | C2 | 125.1 [62–65] | 176.2 [88–90] | 171.3 [84–88] | 0.71× |
| code_31k | C4 | 213.9 [52–63] | does not start | — | — |
| code_31k | C8 | 180.0 [22–52] | does not start | — | — |
| code_4k | C1 | 160.7 [161–161] | 189.1 [189–190] | 184.2 [184–184] | 0.85× |
| code_4k | C2 | 258.2 [127–137] | 340.2 [170–197] | 318.5 [159–160] | 0.76× |
| code_4k | C4 | 365.4 [86–118] | 526.0 [130–148] | 525.6 [128–143] | 0.69× |
| code_4k | C8 | 396.2 [50–69] | does not start | — | — |

### Multi-agent long context: 8 agents × 99K–113K-token code prompts (865K tokens), 4 lanes, 2 turns

VoidInfer: EXL3 6 bpw, L0 OSCAR INT4 + exact VeriCache, `--max-context 262144`. NInfer: groupwise artifact, FP8 KV, `--kv-capacity auto` (166,720 tokens on device), `--host-context-mib 48000`, `--max-context 131072` (the NVFP4 artifact cannot reserve this configuration). 256 output tokens per agent and turn.

| Engine | Agents | Turn | Requests OK | Prompt tokens (sum) | Completion tokens | Wall (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 8 | 1 | 8/8 | 865,256 | 2,048 | 223 | 30,020 |
| ours | 8 | 2 | 8/8 | 867,504 | 2,048 | 52 | 30,002 |
| ninfer | 8 | 1 | 8/8 | 865,256 | 2,048 | 310 | 27,533 |
| ninfer | 8 | 2 | 8/8 | 867,536 | 2,048 | 11 | 27,533 |

### Vision (image + question, greedy, 256 tokens max)

| Server | Image | Result | Prompt tokens | Latency, warm (s) | Answer (first 100 chars) |
|---|---|---|---:|---:|---|
| ninfer.nv | natural_scene | ok | 404 | 1.21 | This is a cheerful, flat-style digital illustration titled **“A SUNNY HOUSE BY THE MOUNTAINS”**, dis |
| ninfer.nv | visual_chart | ok | 411 | 1.01 | This chart, titled **“NIFER VISION 731”**, appears to be a simple visual representation or test of b |
| ours.exl3-novision | natural_scene | rejected (400) | — | — | {"error":{"code":"vision_disabled","message":"Vision is disabled for this server","param":"messages" |
| ours.exl3-novision | visual_chart | rejected (400) | — | — | {"error":{"code":"vision_disabled","message":"Vision is disabled for this server","param":"messages" |

VoidInfer's EXL3 engine is text-only: started with `--vision`, `ninfer-serve` refuses the configuration
("coherent-device EXL3 requires C1..C8 text/eager/DFlash2-K7 full proposals"); without it, image
requests return HTTP 400 `vision_disabled`. The upstream `.ninfer` route that VoidInfer inherited
cannot load the current NInfer artifacts ("artifact magic is not NInfer v2"), so VoidInfer has no
vision path today.

### Quality

KL divergence against the BF16 reference model (top-k records, 8 fixtures × 512 positions at 4K, 95%
confidence intervals over fixtures):

| Configuration | KL(BF16 ‖ x) | Top-1 agreement with BF16 |
|---|---:|---:|
| **VoidInfer (verification path)** | **0.0137 ± 0.0057** | **97.24%** |
| NInfer groupwise / BF16 KV | 0.0801 ± 0.0278 | 92.38% |
| NInfer groupwise / FP8 KV | 0.0815 ± 0.0289 | 92.43% |
| NInfer NVFP4 / BF16 KV | 0.0847 ± 0.0381 | 92.65% |
| NInfer NVFP4 / FP8 KV | 0.0871 ± 0.0393 | 92.55% |

Paired NLL on identical tokens (mean NLL in nats/token and top-1 accuracy against the real next token;
2 streams × 512 targets per domain and depth):

| Slice | VoidInfer | NInfer gw / BF16 | NInfer gw / FP8 | NInfer NVFP4 / BF16 | NInfer NVFP4 / FP8 |
|---|---:|---:|---:|---:|---:|
| 4K all domains | **1.4044** / 67.4% | 1.4370 / 66.6% | 1.4385 / 66.6% | 1.4353 / 66.7% | 1.4400 / 66.3% |
| 16K all domains | **1.2769** / 68.6% | 1.2888 / 68.6% | 1.2864 / 68.6% | 1.2896 / 68.8% | 1.2809 / 68.8% |
| 32K all domains | **1.3205** / 69.2% | 1.3264 / 69.1% | 1.3268 / 69.1% | 1.3261 / 69.1% | 1.3280 / 69.2% |

| NInfer configuration vs VoidInfer (12,288 paired tokens) | Mean ΔNLL [95% CI] | 4K / 16K / 32K | Top-1 agreement |
|---|---|---|---:|
| groupwise / BF16 KV | +0.0168 [+0.0118, +0.0218] | +0.033 / +0.012 / +0.006 | 93.5% |
| groupwise / FP8 KV | +0.0166 [+0.0115, +0.0218] | +0.034 / +0.010 / +0.006 | 93.6% |
| NVFP4 / BF16 KV | +0.0164 [+0.0115, +0.0213] | +0.031 / +0.013 / +0.006 | 93.6% |
| NVFP4 / FP8 KV | +0.0157 [+0.0109, +0.0205] | +0.036 / +0.004 / +0.007 | 93.6% |

Per-domain tables: [`data/2026-10-12/quality/report.txt`](../bench/engine-compare/data/2026-10-12/quality/report.txt).
Since the 2026-10-08 report, VoidInfer's long-context quality moved from statistically tied with NInfer
(INT2 L0 history without verification) to better at every depth (verification against full-precision
history, MXFP8 prompt projections).

**KV value range (FP16 headroom).** During the quality panel, all K (after RoPE) and V rows of the 16
attention layers were scanned: 6.98 × 10⁹ values, max |K| = 23.1, max |V| = 129.4, none at or above
2¹⁵ (FP16's largest finite value is 65,504), 0.005–0.006% in FP16's subnormal range. FP16 KV storage
has more than 500× range headroom on this model and 3 more mantissa bits than BF16.

## Update: vision and agent resume

Measured later on 2026-10-12, same machine and harness, after two changes to VoidInfer (commits
`7c0b09fa`, `a45ce283`). NInfer was not re-run; its columns are the numbers above.

**Vision.** The EXL3 route now takes images and video (`--exl3-vision <BF16 checkpoint dir>`): the
vision tower runs in BF16 from the checkpoint's own weights (streamed from pinned host memory, no
resident VRAM), and image embeddings plus M-RoPE positions feed the usual verified DFlash2 decode.
Prompts shorter than 2,048 tokens, which the route refused before, are served too.

| Server | Image | Prompt tokens | Latency, warm (s) | Decode (tok/s) | Answer (first 100 chars) |
|---|---|---:|---:|---:|---|
| ours.exl3-vision | natural_scene | 404 | 1.71–1.81 | 178–182 | This is a cheerful, stylized digital illustration titled **“A SUNNY HOUSE BY THE MOUNTAINS”**, disp |
| ours.exl3-vision | visual_chart | 411 | 1.61–1.78 | 185–188 | This chart, titled **"NIFER VISION 731"**, appears to be a simple visual test or demonstration of ba |
| ninfer.nv (above) | natural_scene / visual_chart | 404 / 411 | 1.21 / 1.01 | — | (see the vision table above) |

The encoder takes 19 ms per 1,536-patch image (first request after startup 2.1 s; the tower is warmed
at startup). Against an FP32 Hugging Face reference of the same tower on the same images, the merged
embeddings have mean cosine 0.99981 / 0.99992 and relative L2 error 0.042 / 0.020 (Hugging Face's own
BF16 run: 0.99990 / 0.99994, 0.025 / 0.026). Two images in one turn, an image inside a 3,506-token
prompt, a 1,164-token video, text after media, and four concurrent image requests on four lanes
(275 tok/s aggregate) were all served (`vision_suite.py`).

**Agent resume.** A lane used to own one fixed set of FP16 history planes, so resuming an agent
packed the previous agent's ~6.6 GB history and unpacked the incoming one through the CPU (3.4–6.0 s
per agent). Each agent's history now stays in its own plane set; a lane maps the set of the agent it
serves behind a fixed address range (CUDA VMM alias, 44 ms map / 44–138 ms unmap per 4 GiB), so a
resume is a remap. Sets are packed and spilled only under host memory pressure, by a background pass.

| 8 agents × 99K–113K tokens, 4 lanes, 256 outputs | First turn (s) | Resumed turn (s) | Restore per agent |
|---|---:|---:|---|
| VoidInfer, comparison run | 223 | 52 | — |
| VoidInfer, same build re-measured before the change | 222 | 48.3 | 3.4–6.0 s |
| VoidInfer, plane sets (3 runs) | 223–229 | **24.2 / 24.6 / 25.1** | 0.11–0.66 s; 2.0–5.2 s for the 1–2 agents packed under memory pressure |
| NInfer groupwise (above) | 310 | 11 | — |

Each resumed answer cites only identifiers from its own agent's code slice
(`agents_identity_check.py`). The resumed turn is now dominated by decode: only two of the four
lanes fit 108K-token agents' INT4 codes in VRAM, so the eight agents decode in four waves.

## Limits of this comparison

* Weight formats differ by design (EXL3 6 bpw vs NInfer's official NVFP4 and groupwise artifacts).
  The quality section measures the consequence.
* VoidInfer's KV is a different fidelity design (INT4 drafting codes plus verification against
  full-precision history) than NInfer's FP8 KV; the quality numbers are for VoidInfer's verification
  path. Accepted drafted tokens may sit up to 0.0625 logits below the verified maximum.
* NInfer ran with its recommended settings. Its concurrency runs reserved KV per slot
  (`--kv-capacity` = N × max context); where that does not fit (C8 at 4K, C4/C8 at 31K) a shared
  `--kv-capacity auto` pool might run and was not measured. The multi-agent run used NInfer's
  groupwise artifact because the NVFP4 artifact cannot reserve 4 lanes at 131K (with FP8 or k8v4 KV).
* A first NInfer multi-agent attempt with `--max-context 110592` rejected the longest agents
  ("context length exceeded"); the reported run uses 131,072. Both are kept in the data.
* VoidInfer's base decode and target-only prefill use its target-only route; its public Engine is
  DFlash2-only.
* Single machine, single day; greedy only; thinking disabled.

