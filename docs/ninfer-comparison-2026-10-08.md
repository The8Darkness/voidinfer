# VoidInfer vs NInfer: 1:1 comparison (2026-10-08)

This is a head-to-head measurement of VoidInfer against the latest NInfer engine on the same RTX 5090,
same Windows host, same compiler toolchain, and token-identical inputs. It covers prefill, base
(non-speculative) decode, DFlash2 speculative decode across 4K–128K contexts, NInfer's own 12
scenario prompts, and paired model quality.

Raw results, scripts and the reproduction procedure are in
[`bench/engine-compare/`](../bench/engine-compare/README.md).

## Summary

| Metric | VoidInfer | Best NInfer configuration | Result |
|---|---:|---:|---|
| DFlash2 decode, code workload, 4K | **211.3 tok/s** | 190.9 (NVFP4/BF16 KV) | VoidInfer **+11%** |
| DFlash2 decode, code workload, 16K | 186.2 | 187.1 (NVFP4/FP8 KV) | tie |
| DFlash2 decode, code workload, 32K / 64K | 178.2 / 164.3 (L0 OSCAR) | 189.9 / 194.8 (NVFP4/FP8 KV) | NInfer +7% / +19% |
| DFlash2 decode, code workload, 128K | **179.8** (L0 OSCAR) | 171.6 (NVFP4/FP8 KV) | VoidInfer **+5%** |
| DFlash2 committed tokens per round, 4K | **3.74** | 3.05 | VoidInfer **+23%** |
| DFlash2, NInfer's 12 scenarios (geomean) | 240.6 | 252.3 (NVFP4/BF16 KV) | NInfer +5% |
| Base decode, 4K / 16K | 79.2 / 75.7 | 93.8 / 91.6 (groupwise, ~16 GiB weights) | NInfer +18–21% |
| Prefill, target-only, 4K / 16K | 10,068 / 9,628 tok/s | 13,635 / 12,969 (NVFP4) | NInfer +34–35% |
| Quality: mean NLL vs NInfer, 4K+16K (8,192 paired tokens) | reference | +0.017 to +0.020 nats/token worse | VoidInfer **better** (95% CI excludes 0) |
| Quality: long-context L0 OSCAR vs NInfer, 4K–32K (12,288 tokens) | reference | −0.0003 to −0.0014 | statistically tied |

VoidInfer runs 6-bit EXL3 weights (21.0 GiB) and gives the best quality of every configuration
tested. NInfer's NVFP4 and groupwise-int artifacts are 19.0 and 15.9 GiB resident. They read fewer
bytes per token, which accounts for most of NInfer's base-decode advantage. VoidInfer turns more
of each verification round into committed tokens: per-round verification trees, and plausibly a
6-bit target that agrees more often with the BF16-trained draft. That is why its DFlash2 decode wins
at short context despite the heavier weights. Above 16K, VoidInfer uses its L0 OSCAR INT2 KV-history tier,
which keeps a single request up to 262,144 tokens on one GPU at a small quality cost.

## Engines and artifacts

| | VoidInfer | NInfer |
|---|---|---|
| Source | this repository, `perf/sol6-high-4000-70-200-20260923` @ `57ff07fc` | [headpiece747/ninfer-5090-windows](https://github.com/headpiece747/ninfer-5090-windows) @ `a0cf1c17`, the native Windows port, fully merged with upstream [Neroued/ninfer](https://github.com/Neroued/ninfer) `81c8ce0` (latest, 2026-10-07) |
| Build | MSVC 14.51 (VS 2026 Build Tools) + CUDA 13.3.1, `sm_120a` | identical toolchain, unmodified source |
| Target weights | Qwen3.8-27B EXL3 6 bpw (H6 head), 21.04 GiB | official `qwen3_8_27b_nvfp4.ninfer` (sha256 `74d2c571…`) and `qwen3_8_27b.ninfer` groupwise-int (`81f924d4…`) |
| Draft | z-lab Qwen3.8-27B-DFlash2, EXL3 5 bpw | same DFlash2 companion, embedded in the artifact |
| DFlash2 setup | K=7, per-round verification tree (≤6 sibling leaves, ≤8 verify rows), draft head prefix | K=7, `--lm-head-draft` optimized proposal head (NInfer's published DFlash2 setting) |
| KV cache | FP16 device KV up to 16K; above that L0 OSCAR (INT2 history + exact 64-token sink and 2,048-row window on device, FP16 planes in host RAM) | FP8 E4M3 row-256 (NInfer's recommended/published setting) and BF16 (like-for-like 16-bit) |
| Prefill chunk | 1,024 rows (engine default) | 8,192 (the port's launcher setting; 2% faster than 1,024) |

Hardware: one RTX 5090 (32 GiB), driver 617.14, Windows 11, both engines run back-to-back on the
same machine.

## Method

* **Token-identical prompts.** Each prompt is a system and user chat message pair given to both
  engines (`.pmsg` for VoidInfer's Engine harness, OpenAI JSON for `ninfer-serve`). Both engines
  apply the Qwen3.8 chat template with thinking disabled and report identical prompt token counts
  (4,049 / 16,321 / 32,638 / 65,338 / 130,734).
* **Workloads.** (1) A context-scaling code-review prompt: C++ source as background, then a request
  for a 40-item numbered review. (2) NInfer's own 12 scenario prompts (code, story, structured,
  translation), padded with a ~2K-token unrelated system message.
* **Decoding.** Greedy (exact argmax) on both. Base decode: 256 outputs. DFlash2: 1,024 outputs.
  Both engines stop at EOS.
* **Warm measurements.** NInfer: one server per configuration, then a warm request and two measured
  requests per prompt, with prefix reuse disabled. Rates use the server's `timings` object: prefill =
  prompt tokens / time to first token; decode = (n−1) / generation time. VoidInfer: a same-length
  warm request precedes each measured request in-process. Code-scaling and base points were run
  twice; repeats agree within 1.5%.
* **VoidInfer routes.** DFlash2 goes through the public `Engine` (coherent-device round). Base decode
  and target-only prefill use the target-only `TargetRequest` route, replaying the exact prompt ids
  the Engine exported. VoidInfer's Engine is DFlash2-only, and its target-only route has no L0 tier,
  so it has no base-decode points above 16K.
* **Quality.** Paired causal NLL on NInfer's own `ninfer-ppl-1m-v1` corpus: 2 streams × 4 domains
  (English reference, English long-form, Chinese reference, C++/CUDA code), at depths 4K/16K/32K.
  Each text is cut to exactly W+512 tokens. NInfer runs `ninfer-perplexity --context W --stride 512`,
  whose second window scores tokens [W, W+512) given [512, W). VoidInfer teacher-forces the same ids
  with the same history. NInfer's own stream token digests are asserted equal to the ids VoidInfer
  scored, so every engine scores identical tokens. Top-1 predictions come from NInfer's
  `--topk-record` and VoidInfer's argmax.

## Results

### Prefill throughput, target only (tok/s)

| Context (prompt tokens) | VoidInfer FP16 KV | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|
| 4K (4,049) | 10,068 | **13,526** | 3,462 | 13,635 | 3,518 |
| 16K (16,321) | 9,628 | **12,969** | 3,464 | 12,250 | 3,458 |
| 32K (32,638) | — | 11,539 | 3,398 | 10,415 | 3,321 |
| 64K (65,338) | — | 9,233 | 3,201 | 7,823 | 3,045 |
| 128K (130,734) | — | 6,485 | 2,832 | — | — |

### Prefill throughput with the DFlash2 draft resident (tok/s)

VoidInfer's Engine prefill also fills the draft's context taps and ring.

| Context | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|---:|
| 4K | 7,308 | — | **13,907** | 3,429 |
| 16K | 8,258 | — | **12,830** | 3,456 |
| 32K | — | 7,309 | **11,478** | 3,384 |
| 64K | — | 6,408 | **9,185** | 3,190 |
| 128K | — | 5,083 | **6,463** | 2,824 |

### Base greedy decode (tok/s)

| Context | VoidInfer FP16 KV | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|
| 4K | 79.2 | 81.7 | **93.8** | 81.0 | 93.3 |
| 16K | 75.7 | 79.9 | **91.6** | 78.0 | 89.2 |
| 32K | — | 78.1 | **89.2** | 74.5 | 84.6 |
| 64K | — | 74.6 | **84.9** | 68.5 | 76.9 |
| 128K | — | 68.5 | **77.0** | — | — |

Per resident weight byte, VoidInfer's decode reads at a higher effective rate (approximately
21.0 GiB at 79.2 tok/s ≈ 1.79 TB/s, against 15.9 GiB at 93.8 tok/s ≈ 1.60 TB/s, counting all
resident target weights), so the gap comes from the weight format, not kernel efficiency.

### DFlash2 K=7 greedy decode, code-review workload (tok/s)

| Context | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K | **211.3** | — | 186.0 | 175.9 | 190.9 | 158.0 |
| 16K | 186.2 | — | **187.1** | 169.2 | 175.0 | 175.3 |
| 32K | — | 178.2 | **189.9** | 151.1 | 178.0 | 164.9 |
| 64K | — | 164.3 | **194.8** | 155.5 | 151.8 | 135.3 |
| 128K | — | **179.8** | 171.6 | 133.5 | — | — |

Committed tokens per verification round:

| Context | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|---:|
| 4K | **3.74** | — | 2.84 | 3.05 |
| 16K | **3.48** | — | 2.90 | 3.00 |
| 32K | — | **3.30** | 3.00 | 2.71 |
| 64K | — | 3.16 | **3.20** | 2.87 |
| 128K | — | **3.59** | 3.04 | 2.63 |

VoidInfer's round costs 17.7 ms at 4K (FP16) and 18.5–20.0 ms with L0 OSCAR at 32K–128K.
NInfer's NVFP4 round is about 15 ms, since its target weights are smaller. VoidInfer wins wherever its
higher per-round acceptance outweighs the slower round.

### DFlash2 on NInfer's 12 scenario prompts (tok/s, generated tokens in parentheses)

| Scenario | VoidInfer FP16 KV | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|
| code_cuda | 243.8 (201) | 251.8 (1,024) | 253.2 (1,024) | 273.5 (1,024) | 263.7 (1,024) |
| code_python | 250.8 (127) | 258.5 (1,024) | 220.3 (1,024) | 233.0 (1,024) | 245.2 (1,024) |
| code_typescript | 273.8 (94) | 340.5 (94) | 261.5 (42) | 340.9 (94) | 262.4 (42) |
| story_en_mystery | 169.0 | 174.4 | 153.6 | 170.3 | 148.4 |
| story_zh_dialogue | 103.5 | 104.3 | 90.7 | 105.5 | 95.9 |
| story_zh_scifi | 117.5 | 115.6 | 102.1 | 114.5 | 103.4 |
| structured_csv | 370.4 | 381.0 | 334.5 | 410.4 | 333.3 |
| structured_jsonl | 445.1 (1,024) | 441.6 (336) | 411.4 | 441.2 (336) | 411.8 |
| structured_sql | 346.0 | 372.8 | 308.5 | 339.9 | 322.4 |
| translation_en_zh | 271.7 (772) | 198.9 (1,024) | 241.1 (754) | 276.9 (764) | 236.7 (773) |
| translation_markdown | 217.5 (816) | 223.1 (752) | 225.4 (861) | 226.9 (771) | 211.4 (753) |
| translation_zh_en | 324.4 (789) | 378.7 (725) | 332.4 (719) | 385.5 (875) | 344.7 (834) |
| **geometric mean** | **240.6** | **246.0** | **223.9** | **252.3** | **227.3** |

Unmarked cells generated 1,024 tokens. Greedy text differs between weight formats, so a few short
outputs (VoidInfer's code scenarios ended at EOS after 94–201 tokens) are noisy single points.

### Quality (paired NLL on identical tokens)

Mean NLL (nats/token) and top-1 accuracy against the real next token, over 2 streams × 512 targets
per cell:

| Slice | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer groupwise / BF16 | NInfer groupwise / FP8 | NInfer NVFP4 / BF16 | NInfer NVFP4 / FP8 |
|---|---:|---:|---:|---:|---:|---:|
| 4K all domains | **1.4044** / 67.4% | 1.4140 / 67.5% | 1.4370 / 66.6% | 1.4385 / 66.6% | 1.4353 / 66.7% | 1.4400 / 66.3% |
| 16K all domains | 1.2824 / 68.7% | 1.3055 / 68.4% | 1.2888 / 68.6% | 1.2864 / 68.6% | 1.2896 / 68.8% | **1.2809** / 68.8% |
| 32K all domains | — | 1.3336 / 69.3% | 1.3264 / 69.1% | 1.3268 / 69.1% | **1.3261** / 69.1% | 1.3280 / 69.2% |

Paired differences (positive = worse than the reference), with 95% confidence intervals:

| Configuration | vs VoidInfer FP16 KV (4K+16K, 8,192 tokens) | vs VoidInfer L0 OSCAR (4K–32K, 12,288 tokens) |
|---|---|---|
| VoidInfer L0 OSCAR | +0.0163 ± 0.0047, top-1 agree 96.6% | — |
| NInfer groupwise / BF16 KV | +0.0195 ± 0.0064, agree 93.2% | −0.0003 ± 0.0060 |
| NInfer groupwise / FP8 KV | +0.0191 ± 0.0066, agree 93.4% | −0.0004 ± 0.0061 |
| NInfer NVFP4 / BF16 KV | +0.0190 ± 0.0065, agree 93.3% | −0.0007 ± 0.0060 |
| NInfer NVFP4 / FP8 KV | +0.0170 ± 0.0065, agree 93.2% | −0.0014 ± 0.0062 |

Per-domain tables are in [`quality_report.txt`](../bench/engine-compare/data/quality/quality_report.txt).
VoidInfer's 6-bit weights with FP16 KV are measurably closer to the text than any NInfer
configuration, mostly at 4K. L0 OSCAR spends that margin: it is statistically level with NInfer.
NInfer's FP8 KV costs nothing measurable compared with its BF16 KV.

## Toolchain update made for this comparison

Both engines were built with **VS 2026 Build Tools (MSVC 14.51) and CUDA 13.3.1**. NInfer's Windows
port needs MSVC 14.5x, and VoidInfer moved to the same toolchain so the compiler is held constant.
CUDA 13.3's front end rejects class-template argument deduction for standard lock guards in templated
code, so VoidInfer now names the mutex type explicitly at 198 lock sites. CUDA 13.1 and 13.3 builds of
VoidInfer give identical greedy state hashes and tokens at 4K/16K, with timing inside run-to-run noise
(4K base 79.9 → 79.7 tok/s, DFlash2 175.7 → 174.7 tok/s on the campaign fixture).

## Limits of this comparison

* Weight formats differ by design: each engine runs its own native format (EXL3 6 bpw vs NInfer's
  official NVFP4 and groupwise-int artifacts). The quality section measures the consequence.
* VoidInfer's FP16-KV path is VRAM-limited with DFlash2 and the layer-major prefill cache resident
  (≈150 KiB/token of context-scaled device state): it runs at 16K and fails its device-headroom
  guard at 31K. Longer contexts use L0 OSCAR,
  so the 32K–128K VoidInfer points are a different KV fidelity tier than NInfer's FP8/BF16 KV.
* VoidInfer's base-decode route has no L0 tier, so there are no VoidInfer base-decode points above 16K.
* Single machine, single session; greedy only; one request at a time (C=1).
