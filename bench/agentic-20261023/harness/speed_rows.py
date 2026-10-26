"""Refresh of the README speed rows on the release build (2026-10-26): C2 aggregate decode and 100K agent resume.
Wraps bench/engine-compare/capacity_20261020.py through decode_sweep.py (release package for VoidInfer).

Usage: python -I speed_rows.py --engine ours|nv --exp conc|resume   -> results/decode_sweep/<engine>.jsonl
"""
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_sweep  # noqa: E402  (patches cap.server_cmd, sets CAP_OUT)
cap = decode_sweep.cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["ours", "nv"], required=True)
    ap.add_argument("--exp", choices=["conc", "resume"], required=True)
    a = ap.parse_args()
    if a.exp == "conc":   # 2 simultaneous 4K prompts, 1,024 tokens each, greedy and sampled, 3 reps
        args = argparse.Namespace(engine=a.engine, conc=2, max_context=32768, modes="greedy,sampled", reps=3,
                                  prompt_tokens=4000, out_tokens=1024, extra="", env="")
        cap.cmd_conc(args)
    else:                 # two 100K-token agents alternating turns on one lane
        w = decode_sweep.WINDOW
        extra = "--host-kv-mib 20000" if a.engine == "ours" else f"--kv-capacity {w}"
        args = argparse.Namespace(engine=a.engine, max_context=w, tokens=100000, out_tokens=256, extra=extra, env="")
        cap.cmd_resume(args)


if __name__ == "__main__":
    main()
