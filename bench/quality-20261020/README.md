# VoidInfer EXL3 6 bpw vs NInfer NVFP4: quality summary (2026-10-20)

RTX 5090, Qwen3.8-27B. VoidInfer `refactor/voidinfer` (B2 build, EXL3 6 bpw, L0 OSCAR INT4 KV + exact VeriCache,
DFlash2); NInfer config `nv.fp8` (NVFP4 weights, FP8 KV). This page collects the measurements where VoidInfer is ahead of NInfer in
quality. New in this directory: the two letter-only multiple-choice suites. Everything else is taken from
earlier runs; the source of each row is given.

## Headline numbers

| Measure | VoidInfer | NInfer | Gap | Source |
|---|---|---|---|---|
| KL to FP32 reference, 138,880 teacher-forced tokens | **0.0025** | 0.0364 | 14.8x lower | Oct 16 weight panel |
| Next-token top-1 agreement with FP32 reference | **98.68%** | 94.04% | +4.64 pts [+4.08, +5.24] | Oct 16 weight panel |
| Top-1 agreement on difficult tokens (ref top-1 p < 0.5, 34,437 tokens) | **95.09%** | 80.22% | +14.9 pts | Oct 16 weight panel |
| Perplexity above reference | **+0.12%** | +1.42% | | Oct 16 weight panel |
| Streams where the engine is closer to the reference | **70 / 70** | 0 / 70 | | Oct 16 weight panel |
| Served greedy answers: tokens equal to the BF16 argmax (382 items, GSM8K/MATH/IFEval/HumanEval) | **99.31%** | 96.0% | +3.3 pts | Oct 13 on-policy check |
| Long-context top-1 vs reference, 128K window (served KV path) | **98.37%** | 92.59% | +5.8 pts | Oct 16 KV panel |
| **SuperGPQA direct** (26,529 questions) | **40.21%** | 38.17% | **+2.04 pts, p = 3e-23** | this directory |
| SuperGPQA, calculation questions (11,229) | **27.9%** | 24.8% | **+3.14 pts, p = 9e-19** | this directory |
| **MMLU-Pro direct** (12,032 questions) | **60.14%** | 58.93% | **+1.21 pts, p = 7e-6** | this directory |

For reference, BF16-compute transformers scores 0.0012 KL and 99.46% top-1 against the same FP32 reference, so
6 bpw EXL3 sits close to the BF16 noise floor while NVFP4 adds about 30x more error.

## How each measurement was made

**Builds.** Oct 16 panels: VoidInfer `refactor/voidinfer` @ 07bf1c9e, NInfer @ a0cf1c17. Oct 20 suites: VoidInfer
B2 build of `refactor/voidinfer` (host-memory fix 29e01156), NInfer @ a0cf1c17 (`ninfer-serve`, NVFP4 artifact
`qwen3_8_27b_nvfp4.ninfer`, FP8 KV). Same RTX 5090 throughout.

**Weight panel (Oct 16; rows 1–5).** Teacher-forced next-token distributions on fixed text, compared with a
reference model.
- Text: 70 streams of exactly 2,048 tokens, 10 each of C++/CUDA, Python, math (MATH-500 solutions), English,
  German, Chinese and reasoning traces. Positions 64–2047 are scored: 138,880 tokens. NInfer's record digest
  confirms both engines scored identical token ids.
- Reference: the BF16 checkpoint run in FP32 compute (transformers Qwen3.5 modules, weights upcast exactly, TF32
  off, FP32 lm_head).
- VoidInfer: EXL3 6 bpw through the serving decode route, FP16 KV, to isolate the weights. NInfer: `ninfer-perplexity`
  as one 2,048-token window, BF16 KV; this is its serving prefill and DFlash-verify route (W4A4 NVFP4).
- Metrics: KL(reference ‖ engine) over the reference top-100 (a missing token gets the engine's 100th log-prob,
  an upper bound); top-1 agreement = the engine's argmax equals the reference argmax; "difficult" tokens are those
  where the reference's top-1 probability is below 0.5; perplexity from target NLL vs the reference; per-stream
  wins by KL. CIs: 95% cluster bootstrap over streams, 10,000 resamples.
- Scripts: `bench/quality-20261016/` (`build_panel.py`, `bf16_ref_panel.py`, `panel_ours.ps1`, `panel_ninfer.sh`,
  `analyze_panel.py`); full report [docs/quality-comparison-2026-10-16.md](../../docs/quality-comparison-2026-10-16.md).

**Long-context KV panel (Oct 16; 128K row).** 12 streams of at least 131,584 tokens (English, German, Chinese,
code, Python, reasoning). For the 128K window, the first 128K + 512 tokens are fed and the last 512 targets scored
with full causal context (6,144 tokens). Reference: BF16 compute with block-masked attention. VoidInfer runs its
serving KV path (OSCAR INT4 + hot rows + exact VeriCache); NInfer runs FP8 KV. Metric: top-1 agreement as above.
Same report.

**On-policy check (Oct 13; served-answers row).** Each engine's served greedy answers (thinking off) on 382 items
(GSM8K 100, MATH-500 100, IFEval 100, HumanEval 82) are appended to the chat-templated prompt and scored by the
BF16 model in one teacher-forced pass. Metric: the share of answer tokens (plus the stop token) that equal the BF16
argmax. Unlike the panels, this covers the whole serving path: prompt ingestion, KV tier, speculative decoding and
verification. Answers are re-tokenized identically for both engines. Scripts:
`bench/engine-compare/quality/onpolicy_bf16.py`, `onpolicy_report.py`.

## Letter-only multiple choice (new)

Both suites ask for the answer letter only: thinking off, greedy, 16-token budget, C8 / 8,192 context, first
standalone A–J scored (`supergpqa_direct`, `mmlu_pro_direct` in `bench/engine-compare/tasks/suites.py`). Paired
comparison: exact two-sided McNemar on the questions exactly one engine got right (`mc_pair.py`). Raw per-question
records stay in the run directories, not in git.

### SuperGPQA (m-a-p/SuperGPQA rev 4430d445, ODC-BY)

| Engine | Score | No letter in answer | Wall |
|---|---|---|---|
| VoidInfer | **10,666 / 26,529 = 40.21%** | 505 | 7,021 s |
| NInfer | 10,126 / 26,529 = 38.17% | 893 | 2,174 s |

- 72.7% identical answers; one-engine-correct 1,749 vs 1,209: **+2.04 pts, p = 3e-23**.
- Not a format artefact: on the 25,514 questions where both engines answered with a letter the gap is +1.78 pts
  (p = 1e-17). The no-letter answers are mostly a bare option value or the start of an explanation.
- The gap grows with calculation and difficulty: calculation +3.14 pts vs no calculation +1.22; hard +2.79,
  middle +1.90, easy +1.55. Science +2.52, Engineering +2.38. One run per engine.

Full table: [`pairs/supergpqa_direct_pair.txt`](pairs/supergpqa_direct_pair.txt).

### MMLU-Pro (12,032 questions)

| Engine | Score | Wall |
|---|---|---|
| VoidInfer | **7,236 = 60.14%** | 2,814 s |
| NInfer run 1 | 7,090 = 58.93% | 883 s |
| NInfer run 2 | 7,087 = 58.90% | 873 s |

- VoidInfer vs NInfer r1: 596 vs 450 one-engine-correct, **+1.21 pts, p = 7e-6**; vs r2 +1.24 pts, p = 5e-6.
- NInfer r1 vs r2: 99.2% identical answers, +0.02 pts (p = 0.74): run-to-run noise is far below the engine gap.
- Largest category nets: engineering +25, chemistry +24, math +22, physics +21 questions; ours leads in 13 of 14
  categories.

Full tables: [`pairs/mmlu_pro_direct_pair.txt`](pairs/mmlu_pro_direct_pair.txt), [`pairs/mmlu_pro_direct_ninfer_r1_vs_r2.txt`](pairs/mmlu_pro_direct_ninfer_r1_vs_r2.txt).

## Reproduce

```
QB_OUT=<dir> python -I bench/engine-compare/tasks/run_engine.py --config ours --suites supergpqa_direct --conc 8 --max-context 8192
QB_OUT=<dir> python -I bench/engine-compare/tasks/run_engine.py --config nv.fp8 --suites supergpqa_direct --conc 8 --max-context 8192
python -I bench/quality-20261020/mc_pair.py supergpqa_direct <dir>/ours/supergpqa_direct.jsonl <dir>/nv.fp8/supergpqa_direct.jsonl
```

Data: `$QB_DATA/supergpqa.jsonl` is `SuperGPQA-all.jsonl` from the dataset repo; `$QB_DATA/mmlu_pro.jsonl` is the
MMLU-Pro test split.
