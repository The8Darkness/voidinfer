"""Decode tok/s vs context, 4K-200K, both engines (2026-10-26). Wraps bench/engine-compare/capacity_20261020.py
`single`: one server per engine at a 208,896-token window (NInfer's C1 maximum), C1, thinking off, greedy,
1,024 output tokens, a distinct code slice per request (no prefix reuse), streamed so decode excludes prefill.
VoidInfer = release v2026.10.22 package, defaults; NInfer = a0cf1c17 NVFP4, FP8 KV, DFlash2 K7 lm-head draft.

Usage: python -I decode_sweep.py --engine ours|nv [--reps 3]   -> results/decode_sweep/<engine>.jsonl
       python -I decode_sweep.py --report
"""
import argparse, json, os, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "decode_sweep")
os.environ["CAP_OUT"] = OUT
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine-compare"))
import capacity_20261020 as cap  # noqa: E402

VOID_DIR = os.environ.get("AB_VOID_DIR", r"D:\AI\voidinfer-worktrees\quality-neutral-speed-20261020\dist\voidinfer-v2026.10.22-win64-sm120a")
WINDOW = 208896
LENGTHS = [4096, 8192, 16384, 32768, 65536, 98304, 131072, 163840, 200704]
_orig = cap.server_cmd


def server_cmd(engine, conc, max_context, extra):
    cmd = _orig(engine, conc, max_context, extra)
    if engine == "ours":
        cmd[0] = os.path.join(VOID_DIR, "voidinfer-serve.exe")
    return cmd


cap.server_cmd = server_cmd


def report():
    rows = {}
    for e in ("ours", "nv"):
        p = os.path.join(OUT, f"{e}.jsonl")
        if os.path.exists(p):
            for r in map(json.loads, open(p, encoding="utf-8")):
                if r.get("exp") == "single" and r.get("decode_tok_s"):
                    rows.setdefault((e, r["target_tokens"]), []).append(r)
    print("| context | VoidInfer decode tok/s | NInfer decode tok/s | NInfer / VoidInfer | TTFT VoidInfer / NInfer |")
    print("|---|---:|---:|---:|---:|")
    for L in LENGTHS:
        o, n = rows.get(("ours", L), []), rows.get(("nv", L), [])
        f = lambda rs: f"{statistics.mean(r['decode_tok_s'] for r in rs):.0f} ({min(r['decode_tok_s'] for r in rs):.0f}–{max(r['decode_tok_s'] for r in rs):.0f})" if rs else "-"
        ratio = (statistics.mean(r['decode_tok_s'] for r in n) / statistics.mean(r['decode_tok_s'] for r in o)) if o and n else None
        t = lambda rs: f"{statistics.mean(r['ttft_s'] for r in rs):.1f} s" if rs else "-"
        print(f"| {L // 1024}K | {f(o)} | {f(n)} | {f'{ratio:.2f}x' if ratio else '-'} | {t(o)} / {t(n)} |")
    for e in ("ours", "nv"):
        rs = [r for k, v in rows.items() if k[0] == e for r in v]
        if rs:
            print(e, "runs", len(rs), "finish", {x: sum(r["finish"] == x for r in rs) for x in {r["finish"] for r in rs}},
                  "completion tokens min", min(r["completion_tokens"] for r in rs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["ours", "nv"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report()
    extra = "--host-kv-mib 20000" if a.engine == "ours" else f"--kv-capacity {WINDOW}"
    args = argparse.Namespace(engine=a.engine, max_context=WINDOW, lengths=",".join(map(str, LENGTHS)), reps=a.reps,
                              out_tokens=1024, extra=extra, env="")
    cap.cmd_single(args)


if __name__ == "__main__":
    main()
