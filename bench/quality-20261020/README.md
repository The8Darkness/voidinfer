# VoidInfer EXL3 6 bpw vs NInfer NVFP4: quality summary (2026-10-20)

RTX 5090, Qwen3.8-27B. VoidInfer `refactor/voidinfer` (B2 build, EXL3 6 bpw, L0 OSCAR INT4 KV + exact VeriCache,
DFlash2); NInfer config `nv.fp8` (NVFP4 weights, FP8 KV). This page collects the measurements that separate the
two engines. New in this directory: the two letter-only multiple-choice suites. Everything else is taken from
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

Weight panel and KV panel: `bench/quality-20261016/results/REPORT.md`. On-policy check:
`bench/engine-compare/quality/onpolicy_bf16.py` / `onpolicy_report.py` (run Oct 13; per-token output not committed).

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

Full table: `results/supergpqa_direct_pair.txt`.

### MMLU-Pro (12,032 questions)

| Engine | Score | Wall |
|---|---|---|
| VoidInfer | **7,236 = 60.14%** | 2,814 s |
| NInfer run 1 | 7,090 = 58.93% | 883 s |
| NInfer run 2 | 7,087 = 58.90% | 873 s |

- VoidInfer vs NInfer r1: 596 vs 450 one-engine-correct, **+1.21 pts, p = 7e-6**; vs r2 +1.24 pts, p = 5e-6.
- NInfer r1 vs r2: 99.2% identical answers, +0.02 pts (p = 0.74): run-to-run noise is far below the engine gap.
- Largest category nets: engineering +25, chemistry +24, math +22, physics +21 questions; ours leads in 13 of 14
  categories (philosophy tied).

Full tables: `results/mmlu_pro_direct_pair.txt`, `results/mmlu_pro_direct_ninfer_r1_vs_r2.txt`.

## Where the engines tie

With reasoning on, task accuracy is tied within noise. Long reasoning absorbs the small per-token differences.

| Task | VoidInfer | NInfer | Source |
|---|---|---|---|
| MathArena 228, C1, 200K budget, recommended sampling | 213 | 214 | Oct 19–20, `runs_c1_200k_A` |
| MathArena 228, C4, 82K budget | 202 (204 on the fixed build) | 199 | Oct 18–19, `report_sampling_r1` |
| RULER 16K–128K, NoLiMa 4K–32K | tied | tied | Oct 16 report |
| IFBench (loose prompt-level) | 42.0 | 37.7 | Oct 16 report, p = 0.06 |
| GSM8K, MATH-500, HumanEval, IFEval, AIME and others | tied | tied | Oct 12–13 suites |

NInfer is faster: decode ~1.2–1.4x at C1 and prefill 1.2–2.5x; the letter-only suites ran 3.2x faster on NInfer.

## Reproduce

```
QB_OUT=<dir> python -I bench/engine-compare/tasks/run_engine.py --config ours --suites supergpqa_direct --conc 8 --max-context 8192
QB_OUT=<dir> python -I bench/engine-compare/tasks/run_engine.py --config nv.fp8 --suites supergpqa_direct --conc 8 --max-context 8192
python -I bench/quality-20261020/mc_pair.py supergpqa_direct <dir>/ours/supergpqa_direct.jsonl <dir>/nv.fp8/supergpqa_direct.jsonl
```

Data: `$QB_DATA/supergpqa.jsonl` is `SuperGPQA-all.jsonl` from the dataset repo; `$QB_DATA/mmlu_pro.jsonl` is the
MMLU-Pro test split.
