"""Runs one item file against one engine (started here, stopped at the end); greedy, thinking off.

  run_tasks.py --engine ours|ninfer --items nolima/L16000.jsonl [--conc 1] [--max-context N] [--tag T] [--kv fp8|bf16]

Request (identical for both engines): item messages, max_tokens, temperature 0 (greedy), top_p 1,
chat_template_kwargs.enable_thinking=false. Engines (bench3 harness command lines): ours default
(EXL3 6bpw, L0 OSCAR INT4 KV + exact VeriCache, DFlash2 K7); NInfer NVFP4 recommended config (FP8 KV
auto capacity, DFlash2 K7 + LM-head draft); --kv bf16 switches NInfer to BF16 KV.
Output (resumable): runs_tasks/<engine><tag>/<items-name>.jsonl with content, usage, seconds, and a
<items-name>.run.json summary (wall time, tokens, server log path).
"""
import argparse, json, os, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, r"D:\AI\bench3-20261014\harness")
import run_bench as rb  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True, choices=["ours", "ninfer"])
ap.add_argument("--items", required=True)
ap.add_argument("--conc", type=int, default=1)
ap.add_argument("--max-context", type=int, default=0)
ap.add_argument("--tag", default="")
ap.add_argument("--kv", default="fp8")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--env", nargs="*", default=[], help="K=V server environment overrides (non-numeric, e.g. spill dir)")
a = ap.parse_args()

path_items = a.items if os.path.isabs(a.items) else os.path.join(ROOT, a.items)
items = [json.loads(l) for l in open(path_items, encoding="utf-8")]
if a.limit:
    items = items[: a.limit]
name = os.path.splitext(os.path.basename(path_items))[0]
out_dir = os.path.join(ROOT, "runs_tasks", a.engine + a.tag)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, name + ".jsonl")
done = set()
if os.path.exists(out_path):
    for l in open(out_path, encoding="utf-8"):
        r = json.loads(l)
        if "error" not in r:
            done.add(r["id"])
todo = [it for it in items if it["id"] not in done]
if not todo:
    print(a.engine, name, "nothing to do"); sys.exit(0)

max_context = a.max_context or (max(it.get("length", 4096) for it in items) + 2048)
cmd = rb.command(a.engine, a.conc, max_context)
extra_path = os.path.join(ROOT, "runs_tasks", f"{a.engine}_server_args.txt")
if os.path.exists(extra_path):  # retry passes: caching limits only (see queue_post3.sh)
    cmd += open(extra_path).read().split()
if a.engine == "ninfer" and a.kv != "fp8":
    cmd[cmd.index("--kv-dtype") + 1] = a.kv
log_path = os.path.join(out_dir, f"server.{name}.{int(time.time())}.log")
log = open(log_path, "w", encoding="utf-8")
env = rb.environment()
for kv in a.env:
    k, v = kv.split("=", 1); env[k] = v
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(cmd[0]))
lock = threading.Lock()
out = open(out_path, "a", encoding="utf-8")
stats = {"completion": 0, "prompt": 0, "errors": 0}


def one(it):
    body = {"model": "qwen3.8-27b", "messages": it["messages"], "max_tokens": it.get("max_tokens", 256),
            "temperature": 0.0, "top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}}
    rec = {"id": it["id"]}
    t0 = time.time()
    try:
        r = rb.post(rb.PORT[a.engine], body)
        msg = r["choices"][0]["message"]
        rec.update({"content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                    "finish_reason": r["choices"][0].get("finish_reason"), "usage": r.get("usage"),
                    "seconds": time.time() - t0})
        u = r.get("usage") or {}
        with lock:
            stats["completion"] += u.get("completion_tokens", 0); stats["prompt"] += u.get("prompt_tokens", 0)
    except Exception as e:  # recorded, retried on the next run
        rec.update({"error": repr(e)[:500], "seconds": time.time() - t0})
        with lock:
            stats["errors"] += 1
    with lock:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()


try:
    rb.wait_health(rb.PORT[a.engine], proc)
    t_start = time.time()
    with ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(one, todo))
    wall = time.time() - t_start
    summary = {"engine": a.engine, "tag": a.tag, "items": name, "sent": len(todo), "conc": a.conc,
               "max_context": max_context, "kv": a.kv if a.engine == "ninfer" else "l0-int4+exact-vericache",
               "wall_s": wall, **stats, "env": a.env, "server_log": log_path, "cmd": cmd}
    sp = os.path.join(out_dir, name + ".run.json")
    if os.path.exists(sp):  # a retry pass keeps the first pass's summary; retries get their own file
        sp = os.path.join(out_dir, f"{name}.run.{int(time.time())}.json")
    json.dump(summary, open(sp, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "cmd"}), flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    log.close()
    time.sleep(5)
