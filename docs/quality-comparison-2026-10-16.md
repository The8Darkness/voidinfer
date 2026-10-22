# VoidInfer EXL3 6 bpw vs NInfer NVFP4: model-quality benchmark (2026-10-16/17)

RTX 5090 (driver 617.14), Qwen3.8-27B. VoidInfer `refactor/voidinfer` @ 07bf1c9e (clean), NInfer @ a0cf1c17 (clean).
Everything here was measured on this machine in this run; nothing is carried over from earlier sessions except
where marked "prior measurement". Frozen inputs and settings: `results/MANIFEST.json`. Full tables with every CI:
`results/REPORT_TABLES.md`. Scripts: `scripts/` (copied to `bench/quality-20261016/` in the VoidInfer repo).

## Summary

| question | answer |
|---|---|
| Weight quantization (FP16/BF16 KV both) | EXL3 6 bpw is ~15x closer to the model than NVFP4: KL to the FP32 reference 0.0025 vs 0.0364, top-1 agreement 98.7% vs 94.0%, on 70/70 streams. Significant in every domain and entropy bucket. |
| Where it matters most | High-entropy / difficult tokens (top-1 95.1% vs 80.2%), German (KL 0.0021 vs 0.063), Chinese, English prose. Code is closest for both (but still 23x). |
| KV precision, NInfer | FP8 vs BF16 KV changes NInfer's distribution (pairwise KL 0.034-0.048) but **not its distance to the reference** (0.047 vs 0.046 at 16K/32K, 0.052 vs 0.047 at 64K, 0.064 vs 0.067 at 128K): no measurable fidelity cost of FP8 KV. |
| KV precision, VoidInfer | OSCAR INT4 + hot rows: +0.0000 to +0.0003 KL over FP16 KV at 4K-32K; exact VeriCache distribution = FP16 KV. INT2 + hot rows adds +0.001 to +0.009 over exact KV and grows with context. |
| Practical tasks (RULER multi-key/VT/multi-hop 16K-128K, NoLiMa 4K-32K, IFBench) | All statistically tied. Only IFBench loose prompt-level leans to ours (+4.3 pt, bootstrap CI [+0.3, +8.7], McNemar p = 0.06), which does not survive multiple-comparison correction. |
| Cost | NInfer prefill 1.2-2.5x faster on these runs, decode ~1.2x faster at C1 (IFBench). Prior measurements: NInfer decode is faster at C1 and much faster at C8. |
| Recommendation | See the end: the fidelity gain is real and large, but at these settings it doesn't show up as task accuracy. It justifies the speed cost only where exact distribution fidelity is the goal, not as a general accuracy upgrade. |

## 1. Weight quantization (highest priority)

**Setup.** 70 streams of exactly 2,048 tokens: 10 each of C++/CUDA (VoidInfer + NInfer sources, written in 2026),
Python (exllamav3, NInfer tools, VoidInfer tools), math (MATH-500 problem+solution), English (wikitext, PG-19),
German (5 Gutenberg books, 5 de.wikipedia articles, revision ids pinned), Chinese (zhwiki), reasoning
(s1K-1.1 DeepSeek-R1 traces; off-policy for both engines). The text is cut so that decode-to-encode round-trips exactly,
and NInfer's record token digest proves that both engines scored identical ids. Scored targets are positions
64..2047: **138,880 teacher-forced tokens**.

- **VoidInfer:** EXL3 6 bpw, **FP16 device KV** (`VOIDINFER_EXL3_L0_OSCAR=0`, a reference measurement only).
  64-token prefill, then one decode row per token (the serving decode route).
- **NInfer:** NVFP4 artifact, **BF16 KV**, `ninfer-perplexity` as one 2,048-token window. This is its prefill route,
  which runs NVFP4 activations (W4A4) for T >= 4, as its serving prefill and its DFlash verify passes do.
- **Reference:** the BF16 checkpoint, run with transformers' Qwen3.5 modules **in FP32 compute** (weights upcast
  exactly, TF32 off, FP32 lm_head). The BF16-compute transformers run is reported as an extra candidate. BF16 compute has its own
  noise floor: re-chunking a mathematically identical GEMM moved KL by 4e-4 to 2e-3 on single streams. The
  FP32 reference removes that ambiguity. The FP32 forward was validated against the stock module to ~1e-8 KL.
- **Metrics:** KL(ref ‖ candidate) over the reference top-100. A reference token missing from the candidate's top-100 gets the
  candidate's 100th log-prob, a bound on the true term. This estimator agrees with a top-32-restricted KL to 1%;
  the floor-based variant overestimates ~2x. Also: top-1 agreement, target NLL (ΔNLL vs reference), and
  |p_cand(target) − p_ref(target)|. **CIs:** 95% cluster bootstrap over streams (10,000 resamples). Paired
  differences resample the same streams for both engines.

**Overall (vs FP32 reference, 138,880 tokens; reference NLL 1.2450):**

| candidate | KL | top-1 agree | ΔNLL (nats/token) | perplexity vs ref | mean abs Δp(target) |
|---|---|---|---|---|---|
| BF16 compute (transformers) | 0.00122 [0.00050, 0.00210] | 99.46% [99.39, 99.52] | +0.00017 [−0.00027, +0.00069] | +0.02% | 0.0020 |
| **VoidInfer EXL3 6 bpw** | **0.00245** [0.00132, 0.00397] | **98.68%** [98.49, 98.86] | **+0.00121** [+0.00041, +0.00212] | **+0.12%** | 0.0044 |
| NInfer NVFP4 | 0.03641 [0.02681, 0.04801] | 94.04% [93.28, 94.76] | +0.01405 [+0.00887, +0.02003] | +1.42% | 0.0230 |

**Paired NInfer − VoidInfer:**
- ΔKL **+0.0340** [+0.0253, +0.0444], a **14.8x** ratio.
- Top-1 **−4.64 pt** [−5.24, −4.08].
- ΔNLL **+0.0129** [+0.0075, +0.0187].

NVFP4 has higher KL on **70 of 70** streams. Measured as excess over the BF16-compute floor, EXL3 adds 0.0012 KL and NVFP4 adds 0.0352 (29x).

**Per domain (KL vs FP32 / top-1):**

| domain | BF16 compute | VoidInfer | NInfer | NInfer/VoidInfer KL | ΔTop-1 NInfer−ours |
|---|---|---|---|---|---|
| C++/CUDA | 0.00020 / 99.62% | 0.00067 / 99.44% | 0.0155 / 96.61% | 23x | −2.83 [−3.42, −2.30] |
| Python | 0.00021 / 99.58% | 0.00069 / 99.16% | 0.0168 / 96.05% | 24x | −3.11 [−3.59, −2.68] |
| math | 0.00506 / 99.46% | 0.00620 / 98.97% | 0.0490 / 96.10% | 7.9x | −2.87 [−3.56, −2.24] |
| English | 0.00036 / 99.26% | 0.00107 / 98.22% | 0.0260 / 91.85% | 24x | −6.37 [−6.97, −5.78] |
| German | 0.00025 / 99.28% | 0.00207 / 97.99% | 0.0633 / 90.33% | 31x | −7.66 [−9.73, −5.81] |
| Chinese | 0.00020 / 99.41% | 0.00151 / 97.89% | 0.0380 / 91.80% | 25x | −6.08 [−7.16, −4.59] |
| reasoning | 0.00225 / 99.59% | 0.00496 / 99.11% | 0.0463 / 95.56% | 9.3x | −3.54 [−4.26, −2.96] |

All per-domain KL and top-1 differences exclude zero.
- **Math and reasoning:** their KL is heavy-tailed. A few memorization-sensitive positions carry 60-77% of a stream's KL, and even BF16 compute disagrees with FP32 there. The KL ratio is therefore lower in these domains, and NInfer's math ΔNLL is not significant (+0.0004 [−0.021, +0.023]): at some of those positions NVFP4 happens to land on the true text.
- **German:** the largest gap, KL 31x, top-1 −7.7 pt, ΔNLL +0.031 vs +0.0014.

**By reference entropy (difficulty):**

| BF16 entropy (nats) | tokens | VoidInfer KL / top-1 | NInfer KL / top-1 |
|---|---|---|---|
| < 0.1 | 47,975 | 0.00021 / 99.99% | 0.0040 / 99.92% |
| 0.1-0.5 | 21,548 | 0.00136 / 99.99% | 0.0249 / 99.47% |
| 0.5-1.5 | 30,483 | 0.0046 / 98.67% | 0.0593 / 93.65% |
| 1.5-3.0 | 22,979 | 0.0051 / 96.83% | 0.0703 / 86.70% |
| ≥ 3.0 | 15,895 | 0.0028 / 95.65% | 0.0570 / 80.33% |
| difficult (ref top-1 p < 0.5) | 34,437 | 0.0046 / 95.09% | 0.0682 / 80.22% |

On difficult tokens, NVFP4 picks a different argmax than the reference on 1 in 5 tokens, versus 1 in 20 for EXL3:
- top-1 gap −14.9 pt [−15.9, −14.0];
- KL ratio 14.8x;
- ΔNLL +0.030 vs +0.0023.

Against the BF16-compute reference instead of FP32, the conclusions are identical: KL 0.0032 vs 0.0366, top-1 98.58% vs 94.05%.

## 2. KV precision

**Setup.** There are 12 long streams, each at least 131,584 tokens:
- English: Middlemarch, Karamazov, wikitext;
- German: Buddenbrooks, Effi Briest, Grimm;
- Chinese: zhwiki, Hong Lou Meng;
- code: VoidInfer `src/exl3`, NInfer `qwen3_5`;
- Python: exllamav3;
- reasoning: s1K traces.

For each window W in {4K, 16K, 32K, 64K, 128K}, the prefix of exactly W+512 tokens is scored on its last 512
targets with full causal context: 6,144 tokens per window. Absolute reference: FP32 compute for W ≤ 32K, and
BF16 compute (block-wise masked SDPA) for 64K and 128K. BF16 vs FP32 measures only 0.0003-0.0006 KL at
4K-32K, far below the effects here. NInfer is bit-deterministic, verified with identical reruns.

**Absolute (KL vs reference / top-1):**

| window | ref | BF16 compute | ours FP16 KV | ours INT4 + hot (raw) | ours INT2 + hot (raw) | ours INT4 + exact VeriCache | NInfer BF16 KV | NInfer FP8 KV |
|---|---|---|---|---|---|---|---|---|
| 4K | FP32 | 0.00032 / 99.32% | 0.00210 / 98.50% | 0.00212 / 98.47% | 0.00298 / 98.05% | 0.00211 / 98.60% | 0.0559 / 91.81% | 0.0570 / 91.65% |
| 16K | FP32 | 0.00028 / 99.45% | 0.00176 / 98.36% | 0.00191 / 98.19% | 0.00419 / 97.53% | 0.00179 / 98.37% | 0.0472 / 91.68% | 0.0468 / 91.75% |
| 32K | FP32 | 0.00062 / 99.43% | unsupported* | 0.00207 / 98.26% | 0.00444 / 97.40% | 0.00192 / 98.36% | 0.0472 / 92.25% | 0.0458 / 92.04% |
| 64K | BF16 | — | unsupported* | 0.00357 / 98.31% | 0.01137 / 97.41% | 0.00282 / 98.58% | 0.0516 / 92.15% | 0.0474 / 92.17% |
| 128K | BF16 | — | unsupported* | 0.00275 / 98.27% | 0.00744 / 97.51% | 0.00308 / 98.37% | 0.0640 / 92.55% | 0.0674 / 92.59% |

\* Our FP16 device-KV path is capped at 32,768 tokens (`native_context_extent` ordinary limit), so FP16 KV runs
only at 4K and 16K. Beyond that, the exact-history arm stands in as our exact-KV reference.

**Within-engine pairwise (same weights, prompts, settings; only the KV changes):**

| window | NInfer FP8 vs BF16 KV | ours INT4 raw vs exact | ours INT2 raw vs exact | ours exact VeriCache vs FP16 KV |
|---|---|---|---|---|
| 4K | 0.0435 [0.013, 0.086] / 92.2% | 0.00008 / 99.54% | 0.00095 / 99.01% | 0.00004 / 99.61% |
| 16K | 0.0341 [0.015, 0.059] / 93.2% | 0.00019 / 99.53% | 0.00248 / 98.32% | 0.00006 / 99.76% |
| 32K | 0.0343 [0.013, 0.063] / 93.3% | 0.00021 / 99.46% | 0.00252 / 98.10% | — |
| 64K | 0.0394 [0.016, 0.068] / 93.0% | 0.00129 / 99.53% | 0.0101 / 97.80% | — |
| 128K | 0.0478 [0.017, 0.089] / 93.7% | 0.00128 / 99.56% | 0.0041 / 98.00% | — |

(Exact = FP16 KV at 4K/16K and the exact-history VeriCache arm at 32K+.)

**Reading.**
1. **NInfer:** FP8 and BF16 KV disagree with each other on ~7% of argmaxes, yet both sit at the same distance from the reference at every window. That is lateral jitter from the W4A4 prefill route, amplified by any tiny perturbation, not KV-induced loss. I can't attribute any of NInfer's deficit to FP8 KV. Its deficit is the NVFP4 weight/activation path (the weight panel above, with BF16 KV).
2. **Ours:** INT4 + hot rows adds +0.0000 to +0.0003 KL over exact KV through 32K, ~+0.0008 at 64K, and ~0 at 128K, where the CI overlaps.
3. **Exact VeriCache:** its verifier distribution equals FP16 KV (KL 4-6e-5). In serving, the published tokens come from this exact distribution. The only exception: a draft token within the 0.0625-logit tolerance of the argmax may be accepted.
4. **INT2:** costs 2-4x INT4's excess and grows with context (0.011 at 64K). It is the only KV setting here with a significant ΔNLL: +0.006 [+0.002, +0.012] at 64K and +0.006 [+0.003, +0.009] at 128K.
5. **No cross-engine KV attribution:** none of the cross-engine differences are attributed to KV precision. The weight formats differ, and the weight effect (0.034) is 10-100x larger than any KV effect on our side.

## 3. Practical tasks

Identical prompts, greedy (temperature 0), thinking off (`enable_thinking=false`), C1, each engine in its default serving
configuration: ours L0 INT4 + exact VeriCache + DFlash2 K7; NInfer NVFP4 + FP8 KV + DFlash2 K7 + LM-head draft.

- **RULER:** the NVIDIA/RULER generators at commit c3f5e3b4, seed 42, Qwen tokenizer. Tasks: multi-key NIAH ×3, variable tracking (4-hop), and HotpotQA multi-hop QA. RULERv2's code link (anonymous.4open.science/r/RULERv2) returns `repository_expired`, so these are the RULER v1 versions of the tasks the brief listed. 10 of 750 items share a RULER `index` and are dropped from pairing.
- **NoLiMa:** the official placement code and needle set (58 tests), haystack `rand_book_1`, 6 of the 26 official depth points (0/20/40/60/80/100%), 4K/16K/32K.
- **IFBench:** IFBench_test, 300 prompts, official `run_eval`, max_tokens 4096.

| benchmark | n | NInfer | ours | ours − NInfer [95% CI] | McNemar p |
|---|---|---|---|---|---|
| RULER 16K (all) | 243 | 95.06 | 94.24 | −0.82 [−2.06, 0.00] | 0.50 |
| RULER 32K | 248 | 94.35 | 94.76 | +0.40 [0.00, +1.21] | 1.00 |
| RULER 64K | 149 | 93.29 | 93.29 | 0.00 | 1.00 |
| RULER 128K | 100 | 92.00 | 92.00 | 0.00 | 1.00 |
| NoLiMa 4K | 348 | 71.84 | 73.85 | +2.01 [−1.72, +5.75] | 0.35 |
| NoLiMa 16K | 348 | 62.36 | 62.07 | −0.29 [−4.02, +3.16] | 1.00 |
| NoLiMa 32K | 348 | 43.39 | 44.83 | +1.44 [−2.30, +5.17] | 0.53 |
| IFBench prompt strict | 300 | 35.33 | 38.67 | +3.33 [−0.67, +7.33] | 0.15 |
| IFBench prompt loose | 300 | 37.67 | 42.00 | +4.33 [+0.33, +8.67] | 0.06 |

- **RULER:** multi-key NIAH and variable tracking are 100% for both engines at every length; only HotpotQA (60-76%) varies, and identically. RULER v1 at ≤128K does not separate these engines.
- **NoLiMa:** tied at every length and question type.
- **IFBench (non-thinking greedy):** leans toward ours on 27 vs 14 discordant prompts, but p = 0.06, and it is one of 9 headline comparisons.
- A prior sampled, thinking-on IFBench run (Oct 14) had NInfer ahead (82.7 vs 81.3, tie).

**Runtime (C1, per-request wall time of successful requests):**

| benchmark | NInfer median s / prompt tok/s | ours median s / prompt tok/s |
|---|---|---|
| RULER 16K | 1.57 / 10,043 | 4.00 / 4,023 |
| RULER 32K | 3.63 / 8,727 | 8.34 / 3,957 |
| RULER 64K | 9.82 / 6,610 | 18.46 / 3,543 |
| RULER 128K | 29.0 / 4,486 | 42.9 / 2,639 |
| NoLiMa 16K | 1.48 / 9,318 | 3.86 / 3,927 |
| NoLiMa 32K | 3.80 / 8,296 | 4.49 / 6,995 |
| IFBench (decode-bound) | 674 s total, 213 output tok/s | 784 s total, 172 output tok/s |

Teacher-forced scoring cost (not a serving number): NInfer scores the 143K-token panel in 18 s via prefill; our scorer decodes row by row in 37 min.

**Engine failures found (ours, default configuration; NInfer had none) - fixed after the run (49f35f7d, bae521cd):**
1. RULER 128K: after ~10 consecutive 130K-token requests `KV relief failed: KV spill write`, then `EXL3 engine
   unavailable`. Cause: the prefix cache's byte budget skipped lane-backed KV pages, so retention was bounded only by count
   and ~6.8 GB per request spilled to the 98%-full C: drive. Fixed: those pages count toward the host-KV budget, spills keep
   a free-disk reserve and a failed write is no longer fatal (20 x 128K: spill peak 12 GiB, 0 errors).
2. NoLiMa 16K, 13th request: `tap rows left the shared pool (another context claimed it)`. Cause: the exact line's pass
   rows below the 32-row private tier stayed in the shared tap pool under the line's claim when the drafting context took
   them over. Fixed: the claim is handed over (348/348, draft acceptance unchanged).
3. Found while validating at C4: prompts just over 4,096 tokens crashed (`layer-major retained tap tail extent`; the
   4,096-row L0 blocks of multi-lane servers could be shorter than the retained tap tail). Fixed.
4. Also found at C4: with prefix reuse, stale L0 hot rows from another history changed published answers on multi-lane
   servers (NoLiMa 4K on a C4 server 66.1% vs 73.9% at C1, 27 worse / 0 better). **All benchmark numbers in this report
   ran at C1 and are unaffected** (C1 with the fix: 0 of 60 NoLiMa answers differ). Fixed: hot rows are dropped on every
   rewind/reset (C4: 72.7-73.6% over six runs, speed unchanged).
5. Still open (rare, pre-existing): `batched draft head logits non-finite` with the shared tap pool at C4 (1 of 8 runs);
   `VOIDINFER_EXL3_SHARE_TAPS=0` avoids it.

The affected items of this run were rerun with server restarts before the fixes (spill directory D:,
`--max-private-continuations 4 --max-shared-prefixes 4`, `VOIDINFER_EXL3_SHARE_TAPS=0`: storage/caching only, not numerics).

## Limitations

- **FP32 reference coverage:** the FP32 reference covers the weight panel and the KV windows up to 32K. The 64K and 128K windows use BF16 compute, whose noise floor (≤0.0006 KL here) is far below the effects measured.
- **What the teacher-forced numbers measure:**
  - Ours: the decode route.
  - NInfer: its prefill route, which is also its DFlash verify route, where served tokens are decided.
- **Practical-task configuration:** practical tasks ran at C1. Our batched rounds at C≥2 use FP16-accumulate GEMMs, not measured here. The tasks also ran without thinking and greedy, whereas the model card recommends thinking with sampling.
- **NoLiMa coverage:** NoLiMa used 1 haystack and 6 depth points; the official grid is 5 haystacks × 26.
- **RULER task set:** RULER is v1, because RULERv2 is unavailable.

## Recommendation

- **The quality gain is real.** On distribution fidelity, EXL3 6 bpw is unambiguously better, and the gap is entirely a weight/activation-format effect:
  - ~15x lower KL than NVFP4;
  - 4.6 pt more top-1 agreement overall, 15 pt on difficult tokens;
  - 10x lower ΔNLL;
  - every domain significant;
  - largest gaps in German, Chinese and English prose.
- **KV precision is not a differentiator.** Our INT4 + exact VeriCache serving tier keeps that fidelity at every context up to 128K. NInfer's FP8 KV does not cost it measurable fidelity either.
- **Practical tasks don't move.** That fidelity did **not** produce a statistically significant accuracy advantage on any practical benchmark run here (RULER, NoLiMa, IFBench), nor (prior measurement, Oct 13) in the GSM8K/MATH/MMLU-Pro/IFEval/HumanEval/AIME suites.
- **The cost is concrete.** NInfer prefills 1.2-2.5x faster and decodes faster, and our server had several availability bugs under long multi-request workloads (fixed after the run, see section 3).

**Verdict:** the fidelity advantage justifies the speed/robustness penalty only where matching the BF16 model's distribution is itself the goal:
- sampling-faithful generation;
- long free-form multilingual or creative output, where 1-in-5 argmax flips on uncertain tokens compound;
- evaluation and distillation use.

For retrieval-style long-context and instruction-following workloads as measured, NVFP4 delivers the same task accuracy faster. Priority for VoidInfer: close the prefill/decode gap.
