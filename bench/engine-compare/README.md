# Engine comparison harness (VoidInfer vs NInfer)

Tooling and raw results behind [docs/ninfer-comparison-2026-10-08.md](../../docs/ninfer-comparison-2026-10-08.md).
The scripts record the exact procedure that was run. They use the local layout of the measurement
machine (`D:\AI\...`); change the paths at the top of each script to reproduce elsewhere.

Our side (`ours_*.ps1`, `quality/ours_nll.ps1`) dot-sources [`ours_env.ps1`](ours_env.ps1), which names
the build, the model package (`--target`, `--draft`, `--vision`) and clears stale `VOIDINFER_*` /
`NINFER_*` variables. The engine applies its own benchmarked profile, so no receipt environment is
loaded; `-Env 'VOIDINFER_NAME=VALUE;...'` adds per-run overrides. The scripts' run records from
before the 2026-10-12 rename used `ninfer-serve` and `NINFER_*` names for our engine.

## Layout

| Path | Purpose |
|---|---|
| `build_ninfer.ps1` | Builds the NInfer Windows port (`ninfer`, `ninfer-serve`, `ninfer-perplexity`) with VS 2026 + CUDA 13.3 |
| `prompts/gen.py` | Writes each prompt as both `.pmsg` (VoidInfer Engine harness) and OpenAI JSON (`ninfer-serve`) from the same messages |
| `ninfer_bench.py` | NInfer speed: one `ninfer-serve` per configuration; warm request + N measured requests; greedy, no thinking, no prefix reuse; reads the server `timings` object |
| `run_ninfer_matrix.sh` | Full NInfer matrix: {groupwise, NVFP4} × {BF16 KV ≤64K, FP8 KV ≤128K} × {base, DFlash2 K=7} |
| `ours_bench.ps1` | VoidInfer speed: DFlash2 through the public Engine (device harness, warm-only) and base decode through the target-only route, replaying the Engine's exported prompt ids |
| `quality/prep.py` | Cuts NInfer's `ninfer-ppl-1m-v1` streams into texts of exactly W+512 tokens (round-trip verified) |
| `run_ninfer_quality.sh` | `ninfer-perplexity --context W --stride 512` per-token log-probs and top-1 records |
| `quality/ours_nll.ps1` | VoidInfer teacher-forced NLL on the same ids and history (`research_exl3_long_nll`, engine default L0 OSCAR; `-Env 'VOIDINFER_EXL3_L0_OSCAR=0'` for the FP16 reference) |
| `quality_report.py` | Paired NLL / top-1 tables; asserts NInfer's token digests equal the scored ids |
| `make_tables.py` | Speed tables (averaging repeats) |
| `data/` | Raw NInfer JSON per configuration, VoidInfer `results.jsonl`, quality report and summary, generated tables |

## Reproduce

1. Download the official artifacts (`neroued/Qwen3.8-27B-nvfp4-NInfer`, `neroued/Qwen3.8-27B-NInfer`)
   and verify their SHA-256 against each `artifact-manifest.json`.
2. `build_ninfer.ps1`; build VoidInfer (`voidinfer-serve`, tests, `research_exl3_long_nll`).
3. `python prompts/gen.py`, `python quality/prep.py`.
4. Speed: `ours_bench.ps1 -Prompts code_4k:4096,code_16k:16384 -Mode dflash2` (then `-Mode base`),
   and `run_ninfer_matrix.sh`.
5. Quality: `quality/ours_nll.ps1 -L0`, `run_ninfer_quality.sh`, then `python quality_report.py`.
6. `python make_tables.py`.

Run one GPU job at a time; the speed numbers assume an otherwise idle GPU.

## 2026-10-12 full comparison

Tooling and raw data behind [docs/ninfer-comparison-2026-10-12.md](../../docs/ninfer-comparison-2026-10-12.md)
are in the same directory (scripts suffixed `_20261012`) and `data/2026-10-12/`.

| Path | Purpose |
|---|---|
| `run_ninfer_20261012.sh`, `run_ninfer_254k_20261012.sh` | NInfer single-request matrix: {NVFP4, groupwise} × DFlash2 4K–128K + 12 scenarios, base decode, 254K (FP8, then `k8v4`/`nvfp4` KV where FP8 does not fit) |
| `run_ours_20261012.ps1` | VoidInfer single-request matrix with its defaults (`ours_bench.ps1 -L0`) |
| `run_all_20261012.ps1`, `run_resume_20261012.ps1` | The sequential GPU queue; the resume re-ran scenarios, base decode and the rest after the terminal-round VeriCache fix (`98f10860`) |
| `conc_client.py`, `ours_conc.ps1`, `ninfer_conc.py`, `run_conc_20261012.ps1`, `run_conc_prefix_20261012.ps1` | Concurrency C1–C8 (one warm request with a distinct prompt, then N simultaneous requests); NInfer with and without prefix reuse |
| `run_conc_retry_20261012.ps1` | NInfer `--kv-capacity auto` retry for the configurations whose per-slot reservation does not start (prepared, **not run**) |
| `agents_client.py`, `ours_agents.ps1`, `ninfer_agents.py`, `run_ninfer_agents_fallback_20261012.ps1`, `rerun_agents_20261012.ps1` | 8 agents × 400,000-character code slices, 2 turns, 4 lanes; NInfer tried in order NVFP4/FP8, NVFP4/k8v4, groupwise/FP8 |
| `vision_client.py`, `run_vision_20261012.ps1`, `run_vision_inherited_20261012.ps1` | Two NInfer example images, 2 greedy requests each |
| `quality/ours_nll.ps1 -L0 -TopK -Tag _d12x -Env 'VOIDINFER_EXL3_L0_EXACT=1;LONG_NLL_LAYER_MAJOR=1'` | VoidInfer verification-path NLL and top-k records |
| `quality/kl_compare.py _l0_d12x`, `quality_report.py "ours_exact=.ours_l0_d12x.tok:ours exact (served)" --out=...` | KL against BF16 and paired NLL against NInfer's recorded quality data |
| `make_tables_20261012.py` | All tables of the report (`data/2026-10-12/tables.md`) |
| `data/2026-10-12/logs/` | Queue log and excerpts of server logs (startup failures, capacity lines, request summaries) |

### 2026-10-12 update: vision and agent resume

Tooling and data behind the [update section](../../docs/ninfer-comparison-2026-10-12.md#update-vision-and-agent-resume)
(`data/2026-10-12/update/`).

| Path | Purpose |
|---|---|
| `ours_vision.ps1` | VoidInfer server with `--vision` (BF16 checkpoint): short text prompts, then `vision_client.py` or (`-Suite`) `vision_suite.py`; `-Env 'VOIDINFER_EXL3_VISION_DUMP=DIR'` dumps the merged image rows |
| `vision_suite.py` | Images, two images in one turn, an image inside a 3.5K-token prompt, a video, text after media, `--concurrent N` simultaneous image requests |
| `vision_ref.py <dump dir> <images...>` | Hugging Face Qwen3.5 vision tower (BF16 and FP32) on the same images against the dumped rows (cosine, relative L2) |
| `ours_agents.ps1 -Env 'VOIDINFER_EXL3_VERICACHE_STATS=1'` | 8-agent run with one `KV_RESTORE` phase line per restore (`*.restore.log`) |
| `agents_identity_check.py <result.json>` | Every answer's backticked identifiers checked against the agent's own code slice |
