# NInfer multi-agent long-context run: ninfer-serve nvfp4 / FP8 KV / DFlash2, --max-concurrency N lanes,
# --kv-capacity auto, Host context for paused requests, prefix reuse on; agents_client.py drives it.
import argparse, os, subprocess, sys, time, urllib.request
EXE = r"D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe"
ap = argparse.ArgumentParser()
ap.add_argument("--artifact", required=True); ap.add_argument("--conc", type=int, default=4)
ap.add_argument("--agents", type=int, default=8); ap.add_argument("--chars", type=int, default=400000)
ap.add_argument("--tokens", type=int, default=256); ap.add_argument("--turns", type=int, default=2)
ap.add_argument("--max-context", type=int, default=131072); ap.add_argument("--host-context-mib", type=int, default=48000)
ap.add_argument("--port", type=int, default=18095); ap.add_argument("--kv", default="fp8"); ap.add_argument("--out", required=True)
a = ap.parse_args()
cmd = [EXE, a.artifact, "--port", str(a.port), "--max-context", str(a.max_context), "--kv-capacity", "auto",
       "--max-concurrency", str(a.conc), "--kv-dtype", a.kv, "--greedy", "--no-thinking", "--prefill-chunk", "8192",
       "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft", "--pending-timeout-ms", "3600000",
       "--max-pending-requests", "16", "--host-context-mib", str(a.host_context_mib), "--model-id", "conc"]
log = open(a.out + ".server.log", "w", encoding="utf-8")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(EXE))
try:
    t0 = time.time(); ok = False
    while time.time() - t0 < 900:
        if proc.poll() is not None: break
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=2); ok = True; break
        except Exception: time.sleep(1)
    if not ok:
        print("server failed to start / exited", proc.returncode); sys.exit(1)
    subprocess.run([sys.executable, r"D:\AI\engine-compare-20261008\agents_client.py", "--port", str(a.port),
                    "--agents", str(a.agents), "--chars", str(a.chars), "--tokens", str(a.tokens),
                    "--turns", str(a.turns), "--out", a.out])
finally:
    proc.terminate()
    try: proc.wait(30)
    except Exception: proc.kill()
