# Coding-agent, AIME 2026 and speed benchmarks, 2026-10-23/26

This folder holds the test code and raw results for these sections of the top-level README:
- the *Long-running coding agents* table;
- the *AIME 2026* rows of the *Quality* table;
- the *Decode speed by context length* table.

Both engines ran on the same RTX 5090 (Windows 11, WDDM, 96 GB RAM):

| | VoidInfer | NInfer |
|---|---|---|
| build | release v2026.10.22 package, defaults | `a0cf1c17` (`ninfer-serve`, native Windows port) |
| weights | Qwen3.8-27B EXL3 6 bpw (`SC_6.00bpw_H6_V6`) | Qwen3.8-27B NVFP4 artifact |
| KV cache | L0 OSCAR INT4 + hot rows, exact VeriCache | FP8 |
| speculative decoding | DFlash2 EXL3 5 bpw draft | DFlash2, 7 draft tokens, lm-head draft |

Every request sends its sampling parameters explicitly, so both engines get the same settings. The
exact server command lines are stored with the agent results (`results/<tag>/<engine>.meta.json`).

## Contents

| Path | What |
|---|---|
| `harness/bench.py` | Agent loop, compaction, sandbox execution, scoring |
| `harness/lh2_spec.py`, `lh3_spec.py` | Stage texts, decisions, conventions and filler plan for the two projects |
| `hidden/<suite>/` | Hidden tests (`test_functional.py`, `test_rules.py`), convention checker, reference solution (`ref/`) |
| `harness/report.py` | Aggregates `results/<tag>/<engine>.jsonl` into tables (Mann-Whitney U per row) |
| `harness/aime.py` | AIME 2026 runner and report (avg@k, majority vote, pass@k, paired sign-flip permutation test) |
| `harness/decode_sweep.py` | Decode tok/s and time to first token from 3K to 200K tokens of context |
| `harness/speed_rows.py` | Two requests at once (aggregate decode) and 100K-token agent resume, re-measured on the release build (not in the top-level README) |
| `harness/setup_wsl.sh` | Creates the WSL Python environment (pytest) that runs the agent's commands |
| `results/lh2_c1_w208896`, `results/lh3_c1_w208896` | Per-run agent results for the two README agent tests (6 runs per engine) |
| `results/aime2026`, `results/aime2026_c1_200k` | Every AIME generation: answer, correctness, finish reason, tokens, seconds |
| `results/decode_sweep` | Decode sweep, two-request and resume rows |

The decode and resume scripts reuse `../engine-compare/capacity_20261020.py`, which handles server
start-up, streamed requests and timing.

## Coding agents (`lh2` = ledger library, `lh3` = room-booking library)

- **Loop.** The loop is mini-swe-agent style: each reply holds one bash block, which runs in a WSL Ubuntu sandbox.
  - Commands time out after 120 s, and observations are capped at 24K characters.
  - Thinking is off. Sampling is T 0.7 / top_p 0.8 / top_k 20.
  - Each reply is capped at 4,096 tokens, and each stage gets at most 30 steps.
- **Task.** The agent builds the project in 12 stages. 12 conventions and 20 team decisions appear once, in "team chat" blocks; 4 decisions are revised later.
- **Filler and compaction.**
  - Every stage adds about 100K tokens of unrelated file reads.
  - When the next prompt would exceed 80% of the 208,896-token window, the agent writes a handoff summary. The history is then replaced by the brief, the summary and the recent turns.
  - That gives about 11 compactions per run on both engines.
- **Scoring.** The final code is scored by the hidden functional tests, one test per decision and the convention checker:
  - ledger: 54 + 20 + 12 = 86;
  - room-booking: 61 + 20 + 12 = 93.
  - The reference solutions in `hidden/<suite>/ref` score full marks.

```bat
python -I harness/bench.py --engine ours --suite lh2 --conc 1 --window 208896 --filler 100000 --runs 6 --tag lh2_c1_w208896
python -I harness/bench.py --engine nv   --suite lh2 --conc 1 --window 208896 --filler 100000 --runs 6 --tag lh2_c1_w208896
python -I harness/report.py
```

Use `--suite lh3 --tag lh3_c1_w208896` for the room-booking project. NInfer needs `NV_KV=208896`,
because `--kv-capacity auto` misses this window by a few MB. Before the first run, `wsl -d Ubuntu-24.04 bash harness/setup_wsl.sh`
creates the sandbox Python. The filler is a large C++ source file, pre-tokenized into a JSON list of token IDs
(`AB_SRC_IDS`). The source file itself is not included; any large source file works, though scores may differ slightly.

## AIME 2026

- **Problems:** all 30 (AIME I and II), each sent as the problem plus `Please reason step by step, and put your final answer within \boxed{}.`
- **Settings:** thinking on, T 1.0 / top_p 0.95 / top_k 20.
- **Scoring:** the last `\boxed{}` answer, checked with `math_verify` (the `compmath` scorer in `../engine-compare/tasks/suites.py`).

```bat
rem round 1: 2 requests at a time, 86,016 window, 81,920-token budget, 4 samples per problem
python -I harness/aime.py --engine ours --samples 4
rem round 2: 1 request at a time, 208,896 window, 204,800-token budget, 8 samples; alternate engines per pass with --first k --samples 1
python -I harness/aime.py --engine ours --conc 1 --window 208896 --max-tokens 204800 --tag aime2026_c1_200k --first 0 --samples 1
python -I harness/aime.py --report --conc 1 --window 208896 --max-tokens 204800 --tag aime2026_c1_200k
```

The problem file is not included. Point `AB_AIME_DATA` at a jsonl file with `problem_idx`,
`problem` and `answer` fields (MathArena's AIME 2026 layout). `--report` also works from the stored
results alone.

## Speed

```bat
python -I harness/decode_sweep.py --engine ours
python -I harness/decode_sweep.py --engine nv
python -I harness/decode_sweep.py --report
python -I harness/speed_rows.py --engine ours --exp conc
python -I harness/speed_rows.py --engine ours --exp resume
```

- **Decode sweep.** One request at a time, both engines at a 208,896-token window, greedy decoding.
  - Each request carries a different slice of C++ source (no prefix reuse) and asks for a code review of exactly 1,024 tokens.
  - Each length is run 3 times. Decode is timed from the first to the last streamed token.
- **Two requests at once.** Two simultaneous ~4K-token prompts with 1,024 output tokens each, 3 greedy and 3 sampled rounds.
  - Aggregate tok/s is total output divided by the wall time from the first send to the last token, so it includes time to first token.
- **Resume.** Two ~100K-token agents take turns on one lane: A, B, A (switch), A (same agent), B (switch).
  - Each turn appends a short follow-up question to the conversation, and the time to first token is recorded.

## Paths

The scripts default to the paths of the machine that produced these results. Override them with
environment variables:

| Variable | What it points to |
|---|---|
| `AB_VOID_DIR` | the extracted release package |
| `AB_VOID_TARGET`, `AB_VOID_DRAFT` | the target and draft model directories |
| `AB_NINFER_EXE`, `AB_NINFER_NV` | NInfer's server and model artifact |
| `AB_TOKENIZER` | a Qwen3.8 tokenizer directory |
| `AB_SRC_IDS` | the tokenized filler source |
| `AB_AIME_DATA` | the AIME problems |
| `AB_ROOT`, `AB_WSL_ROOT` | where results go, as a Windows path and as the same folder seen from WSL |

`../engine-compare/capacity_20261020.py` has its own path constants at the top (`SRC`, `TOKENIZER`,
`NINFER_*`).
