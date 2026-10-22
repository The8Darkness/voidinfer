"""One VoidInfer 8-agent run on 4 lanes (3 turns, ~110K-token agents) with optional server env.
The build comes from CAP_VOID_BUILD (capacity_20261020.py). usage: capacity_agents_20261020.py TAG [ENV]"""
import json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capacity_20261020 as c
tag, env = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
client = os.path.join(HERE, "agents_client.py")
s = c.Server("ours", 4, 262144, "--greedy --default-max-tokens 256 --max-private-continuations 16 --host-kv-mib 20000".split(), env=env)
info = c.server_info(s)
out = os.path.join(c.OUT, f"agents_ours_{tag}_{int(time.time())}.json")
if s.ok:
    subprocess.run([sys.executable, client, "--port", str(c.PORT), "--model", "m", "--agents", "8",
                    "--chars", "400000", "--tokens", "256", "--turns", "3", "--out", out])
s.close()
try:
    turns = json.load(open(out))["turns"]
    c.record("ours", {"exp": "agents", "tag": tag, "env": env,
                      "turns": [{k: t[k] for k in t if k != "texts"} for t in turns], **info})
    for t in turns: print("TURN", tag, t["turn"], round(t["wall_s"], 1), "ok", t["ok"], "err", len(t["errors"]), flush=True)
except Exception as e:
    c.record("ours", {"exp": "agents", "tag": tag, "error": str(e), **info}); print("FAIL", tag, e, flush=True)
