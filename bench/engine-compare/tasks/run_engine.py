"""Starts one engine configuration, runs the selected suites against its OpenAI endpoint and stops it.

Outputs (resumable): OUT/<config>/<suite>.jsonl, one record per item with the raw answer content,
reasoning length, finish reason, usage and wall time. Items already recorded are skipped.

  run_engine.py --config ours     --suites gsm8k,math500 --conc 8 --max-context 16384
  run_engine.py --config nv.bf16  --suites gsm8k --conc 8 --max-context 16384
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suites import SUITES

OUT = os.environ.get("QB_OUT", r"D:\AI\quality-bench-20261012\runs")
VOID_BUILD = r"D:\AI\voidinfer-sol6-campaign-20260923\builds\B2"
VOID_TARGET = r"C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6"
VOID_DRAFT = r"C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw"
VCPKG_BIN = r"D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin"
NINFER_EXE = r"D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe"
NINFER_ARTIFACT = {"nv": r"D:\AI\models\ninfer\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer",
                   "gw": r"D:\AI\models\ninfer\Qwen3.8-27B-NInfer\qwen3_8_27b.ninfer"}


def command(config, port, conc, max_context):
    if config == "ours":
        return [os.path.join(VOID_BUILD, "apps", "voidinfer-serve.exe"), "--target", VOID_TARGET, "--draft", VOID_DRAFT,
                "--port", str(port), "--max-context", str(max_context), "--max-concurrency", str(conc),
                "--max-pending-requests", "64", "--pending-timeout-ms", "3600000", "--no-thinking",
                "--model-id", "qwen3.8-27b"]
    artifact, kv = config.split(".")  # nv.bf16, gw.fp8, ...
    return [NINFER_EXE, NINFER_ARTIFACT[artifact], "--port", str(port), "--max-context", str(max_context),
            "--kv-capacity", "auto", "--max-concurrency", str(conc), "--kv-dtype", kv,
            "--max-pending-requests", "64", "--pending-timeout-ms", "3600000", "--no-thinking",
            "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft", "--model-id", "qwen3.8-27b"]


def environment(overrides=""):
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("VOIDINFER", "NINFER"))}
    for kv in filter(None, overrides.split(";")):
        k, v = kv.split("=", 1)
        env[k] = v
    env["PATH"] = VCPKG_BIN + ";" + env.get("PATH", "")
    return env


def post(port, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def wait_health(port, proc, limit=1200):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server never became healthy")


def run_suite(port, config, name, conc, limit, ids=None):
    items_fn, _, max_tokens, thinking = SUITES[name]
    items = list(items_fn())[:limit] if limit else list(items_fn())
    os.makedirs(os.path.join(OUT, config), exist_ok=True)
    path = os.path.join(OUT, config, name + ".jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = [it for it in items if it["id"] not in done and (not ids or it["id"] in ids)]
    lock = threading.Lock()
    out = open(path, "a", encoding="utf-8")
    t_start, tokens = time.time(), [0]

    def one(it):
        messages = it.get("messages") or [{"role": "user", "content": it["prompt"]}]
        body = {"model": "qwen3.8-27b", "messages": messages,
                "max_tokens": max_tokens, "temperature": 0, "presence_penalty": 0, "enable_thinking": thinking}
        t0 = time.time()
        try:
            r = post(port, body)
            msg = r["choices"][0]["message"]
            rec = {"id": it["id"], "content": msg.get("content") or "", "reasoning_chars": len(msg.get("reasoning_content") or ""),
                   "finish_reason": r["choices"][0].get("finish_reason"), "usage": r.get("usage"), "seconds": time.time() - t0}
        except Exception as e:
            detail = e.read().decode(errors="ignore")[:500] if hasattr(e, "read") else ""
            rec = {"id": it["id"], "error": f"{e} {detail}", "seconds": time.time() - t0}
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
            tokens[0] += (rec.get("usage") or {}).get("completion_tokens", 0)

    with ThreadPoolExecutor(conc) as pool:
        list(pool.map(one, todo))
    out.close()
    wall = time.time() - t_start
    print(f"{config} {name}: {len(todo)} items in {wall:.0f} s, {tokens[0] / max(wall, 1e-9):.0f} tok/s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--suites", required=True)
    ap.add_argument("--conc", type=int, default=8)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--port", type=int, default=18500)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--env", default="", help="NAME=VALUE;... added to the server environment")
    ap.add_argument("--ids", default="", help="comma-separated item ids to run (default: all)")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, a.config), exist_ok=True)
    log = open(os.path.join(OUT, a.config, f"server.{int(time.time())}.log"), "w", encoding="utf-8")
    cmd = command(a.config, a.port, a.conc, a.max_context)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environment(a.env), cwd=os.path.dirname(cmd[0]))
    try:
        wait_health(a.port, proc)
        for name in a.suites.split(","):
            run_suite(a.port, a.config, name, a.conc, a.limit, set(filter(None, a.ids.split(","))))
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
        time.sleep(3)


if __name__ == "__main__":
    main()
