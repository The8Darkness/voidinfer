# Engine comparison harness (VoidInfer vs NInfer)

Tooling and raw results behind [docs/ninfer-comparison-2026-10-08.md](../../docs/ninfer-comparison-2026-10-08.md).
The scripts record the exact procedure that was run. They use the local layout of the measurement
machine (`D:\AI\...`); change the paths at the top of each script to reproduce elsewhere.

## Layout

| Path | Purpose |
|---|---|
| `build_ninfer.ps1` | Builds the NInfer Windows port (`ninfer`, `ninfer-serve`, `ninfer-perplexity`) with VS 2026 + CUDA 13.3 |
| `prompts/gen.py` | Writes each prompt as both `.pmsg` (VoidInfer Engine harness) and OpenAI JSON (`ninfer-serve`) from the same messages |
| `ninfer_bench.py` | NInfer speed: one `ninfer-serve` per configuration; warm request + N measured requests; greedy, no thinking, no prefix reuse; reads the server `timings` object |
| `run_ninfer_matrix.sh` | Full NInfer matrix: {groupwise, NVFP4} × {BF16 KV ≤64K, FP8 KV ≤128K} × {base, DFlash2 K=7} |
| `ours_bench.ps1` | VoidInfer speed: DFlash2 through the public Engine (device harness, warm-only) and base decode through the target-only route, replaying the Engine's exported prompt ids; `-L0` selects L0 OSCAR |
| `quality/prep.py` | Cuts NInfer's `ninfer-ppl-1m-v1` streams into texts of exactly W+512 tokens (round-trip verified) |
| `run_ninfer_quality.sh` | `ninfer-perplexity --context W --stride 512` per-token log-probs and top-1 records |
| `quality/ours_nll.ps1` | VoidInfer teacher-forced NLL on the same ids and history (`research_exl3_long_nll`), FP16 KV or `-L0` |
| `quality_report.py` | Paired NLL / top-1 tables; asserts NInfer's token digests equal the scored ids |
| `make_tables.py` | Speed tables (averaging repeats) |
| `data/` | Raw NInfer JSON per configuration, VoidInfer `results.jsonl`, quality report and summary, generated tables |

## Reproduce

1. Download the official artifacts (`neroued/Qwen3.8-27B-nvfp4-NInfer`, `neroued/Qwen3.8-27B-NInfer`)
   and verify their SHA-256 against each `artifact-manifest.json`.
2. `build_ninfer.ps1`; build VoidInfer B2 plus `research_exl3_long_nll`.
3. `python prompts/gen.py`, `python quality/prep.py`.
4. Speed: `ours_bench.ps1 -Prompts code_4k:4096,code_16k:16384 -Mode dflash2` (then `-Mode base`;
   `-L0` with prefixes 32768/65536/131072), and `run_ninfer_matrix.sh`.
5. Quality: `quality/ours_nll.ps1` (and `-L0`), `run_ninfer_quality.sh`, then `python quality_report.py`.
6. `python make_tables.py`.

Run one GPU job at a time; the speed numbers assume an otherwise idle GPU.
