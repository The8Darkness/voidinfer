# Quality benchmark 2026-10-16: VoidInfer EXL3 6 bpw vs NInfer NVFP4

Report: `results/REPORT.md` (tables with every CI: `results/REPORT_TABLES.md`; frozen inputs: `results/MANIFEST.json`).
Working directory of the run: `D:\AI\quality-bench-20261016` (panels, references, raw records, task runs, logs).

Pipeline (each step resumable):
1. `fetch_sources.py`, `build_panel.py` (70 x 2048-token weight panel), `build_long_panel.py` (12 long streams, W+512 prefixes).
2. References: `bf16_ref_panel.py` (`--fp32` for the FP32-compute reference), `bf16_ref_long.py` (`--fp32 --max-len`).
3. Candidates: `panel_ours.ps1` (research_exl3_long_nll), `panel_ninfer.sh` (ninfer-perplexity), `queue_kv.sh` (KV arms).
4. Tasks: `gen_ruler.py`, `gen_nolima.py`, `make_items.py`, `run_tasks.py`, `queue_all2.sh` / `queue_post3.sh` / `queue_refs.sh`.
5. Scoring/report: `analyze_panel.py`, `score_tasks.py`, `eval_ifbench.sh` (WSL Ubuntu-24.04), `ifbench_pair.py`, `make_report.py`, `make_manifest.py`.
