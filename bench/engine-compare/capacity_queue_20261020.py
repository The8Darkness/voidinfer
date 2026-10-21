"""Sequential GPU queue for the 2026-10-20 capacity/speed comparison (capacity_20261020.py).

  python capacity_queue_20261020.py --phases nv_start,nv_fill,ours_fill,asym,single,conc,resume,agents

Found limits are kept in OUT/limits.json so later phases (and reruns) reuse them.
"""
import argparse, json, os, subprocess, sys, time
from types import SimpleNamespace
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capacity_20261020 as c

LIMITS = os.path.join(c.OUT, "limits.json")
FILL_OUT = 4096
NV_SLOTS0 = "--device-state-slots 0"
ONLY = {"ours", "nv"}  # --engines


def load():
    return json.load(open(LIMITS)) if os.path.exists(LIMITS) else {}


def save(key, value):
    d = load(); d[key] = value
    json.dump(d, open(LIMITS, "w"), indent=1)
    print("LIMIT", key, value, flush=True)


def ns(**kw):
    base = dict(engine="ours", conc=1, max_context=0, lengths="", reps=1, modes="greedy,sampled", prompt_tokens=4000,
                out_tokens=1024, first_tokens=262144, first_out=8192, tokens=100000, window=10.0, extra="", env="")
    base.update(kw); return SimpleNamespace(**base)


def round_up(x, m=1024):
    return (x + m - 1) // m * m


def nv_starts(conc, mc, extra):
    s = c.Server("nv", conc, mc, extra.split())
    ok, line, err = s.ok, (s.grep("capacity |") or [""])[0][24:], s.error
    s.close()
    c.record("nv", {"exp": "start", "conc": conc, "max_context": mc, "extra": extra, "ok": ok, "capacity_line": line,
                    "error": err[-400:]})
    return ok


def search(ok, lo, hi, step):
    """Largest x in [lo, hi] (multiples of step) with ok(x), assuming monotone; None when ok(lo) fails."""
    if ok(hi):
        return hi
    if not ok(lo):
        return None
    while hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if mid <= lo: break
        if ok(mid): lo = mid
        else: hi = mid
    return lo


def phase_nv_start():
    for extra, tag in (("", "default"), (NV_SLOTS0, "slots0")):
        for conc in (1, 2, 4, 8):
            m = search(lambda x: nv_starts(conc, x, extra), 8192, 262144, 1024)
            save(f"nv_start_c{conc}_{tag}", m)


def fill_ok(engine, conc, P, extra, mc=0):
    r = c.cmd_fill(ns(engine=engine, conc=conc, prompt_tokens=P, out_tokens=FILL_OUT, extra=extra,
                      max_context=mc or (262144 if engine == "ours" and conc == 1 else
                                         min(262144, round_up(P + FILL_OUT + 1024)))))
    return bool(r.get("all_decoding")) and not r.get("errors")


def nv_fill_ok(conc, P):
    """NInfer with the smallest max-context for P and an explicit KV pool of conc full requests."""
    mc = min(262144, round_up(P + FILL_OUT + 1024))
    return fill_ok("nv", conc, P, f"{NV_SLOTS0} --kv-capacity {conc * mc}", mc=mc)


def phase_nv_fill():
    L = load()
    for conc in (1, 2, 4, 8):
        hi = L.get(f"nv_start_c{conc}_slots0") or 16384
        hi = min(262144 - FILL_OUT - 1024, hi if conc == 1 else 262144 // conc) // 1024 * 1024
        best = search(lambda P: nv_fill_ok(conc, P), 2048, hi, 1024 if conc == 8 else 2048)
        save(f"nv_fill_c{conc}", best)


def phase_ours_fill():
    save("ours_fill_c1", 262144 - FILL_OUT - 1024 if fill_ok("ours", 1, 262144 - FILL_OUT - 1024, "") else None)
    for conc, lo, hi in ((2, 32768, 262144 - FILL_OUT - 1024), (4, 16384, 160000), (8, 8192, 90000)):
        best = search(lambda P: fill_ok("ours", conc, P, ""), lo, hi // 1024 * 1024, 4096)
        save(f"ours_fill_c{conc}", best)


def asym_ok(engine, first, second, mc, extra):
    r = c.cmd_asym(ns(engine=engine, first_tokens=first, first_out=8192, prompt_tokens=second, out_tokens=FILL_OUT,
                      max_context=mc, extra=extra))
    return bool(r.get("both_decoding")) and not r.get("errors")


def phase_asym():
  if "ours" in ONLY:
    best = search(lambda P: asym_ok("ours", 262144, P, 262144, ""), 4096, 262144 - FILL_OUT - 1024, 4096)
    save("ours_asym_second", best)
  if "nv" in ONLY:
    # NInfer: agent 1 at its C2 per-request maximum, agent 2 searched; pool = the largest that starts.
    L = load()
    first = L.get("nv_start_c2_slots0")
    if first:
        pool = search(lambda k: nv_starts(2, first, f"{NV_SLOTS0} --kv-capacity {k}"), first, 2 * first, 2048)
        save("nv_asym_pool", pool)
        if pool:
            best = search(lambda P: asym_ok("nv", first, P, first, f"{NV_SLOTS0} --kv-capacity {pool}"),
                          2048, first - FILL_OUT - 1024, 2048)
            save("nv_asym_first", first); save("nv_asym_second", best)


def phase_nv_c1_high():
    """NInfer C1 above the auto-pool start limit (explicit pool = max context)."""
    def ok(P):
        mc = min(262144, round_up(P + FILL_OUT + 1024))
        return fill_ok("nv", 1, P, f"{NV_SLOTS0} --kv-capacity {mc}", mc=mc)
    save("nv_fill_c1", search(ok, 182272, 262144 - FILL_OUT - 1024, 2048))


def phase_ours_c8_fine():
    save("ours_fill_c8", search(lambda P: fill_ok("ours", 8, P, ""), 8192, 12288, 1024))


def phase_single():
    L = load()
    if "ours" in ONLY:
        c.cmd_single(ns(engine="ours", max_context=262144, lengths="4096,32768", reps=2))
        c.cmd_single(ns(engine="ours", max_context=262144, lengths="131072,262144", reps=1))
    if "nv" not in ONLY:
        return
    m = L.get("nv_start_c1_default") or L.get("nv_start_c1_slots0")
    extra = "" if L.get("nv_start_c1_default") else NV_SLOTS0
    c.cmd_single(ns(engine="nv", max_context=m, lengths="4096,32768", reps=2, extra=extra))
    c.cmd_single(ns(engine="nv", max_context=m, lengths=f"131072,{m}", reps=1, extra=extra))


def phase_conc():
    """4K prompts + 1,024 output at 8,192 max context; NInfer gets an explicit pool of conc x 8,192 tokens."""
    for conc in (1, 2, 4, 8):
        if "ours" in ONLY:
            c.cmd_conc(ns(engine="ours", conc=conc, max_context=8192, reps=3))
        if "nv" not in ONLY:
            continue
        extra = f"--kv-capacity {conc * 8192}"
        if not nv_starts(conc, 8192, extra):
            extra = f"{NV_SLOTS0} {extra}"
        c.cmd_conc(ns(engine="nv", conc=conc, max_context=8192, reps=3, extra=extra))


def phase_resume():
    L = load()
    if "ours" in ONLY:
        c.cmd_resume(ns(engine="ours", max_context=262144, tokens=100000, out_tokens=256))
    if "nv" not in ONLY:
        return
    m = L.get("nv_start_c1_default") or L.get("nv_start_c1_slots0")
    c.cmd_resume(ns(engine="nv", max_context=m, tokens=100000, out_tokens=256,
                    extra="--host-context-mib 48000" + ("" if L.get("nv_start_c1_default") else " " + NV_SLOTS0)))


def phase_agents():
    """8 agents x 400,000-character code slices (~100-113K tokens), 3 turns, 4 lanes (as on 2026-10-12)."""
    client = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents_client.py")
    runs = [("ours", 262144, "--greedy --default-max-tokens 256 --max-private-continuations 16 --host-kv-mib 20000"),
            ("nv", 131072, "--greedy --host-context-mib 48000"),
            ("nv", 131072, "--greedy --host-context-mib 48000 " + NV_SLOTS0)]
    done_nv = False
    for engine, mc, extra in runs:
        if engine not in ONLY or (engine == "nv" and done_nv):
            continue
        s = c.Server(engine, 4, mc, extra.split())
        info = c.server_info(s)
        if not s.ok:
            s.close(); c.record(engine, {"exp": "agents", **info}); continue
        out = os.path.join(c.OUT, f"agents_{engine}_{int(time.time())}.json")
        subprocess.run([sys.executable, client, "--port", str(c.PORT), "--model", "m", "--agents", "8",
                        "--chars", "400000", "--tokens", "256", "--turns", "3", "--out", out])
        s.close()
        try:
            turns = json.load(open(out))["turns"]
            c.record(engine, {"exp": "agents", "turns": [{k: t[k] for k in t if k != "texts"} for t in turns], **info})
            done_nv = done_nv or engine == "nv"
        except Exception as e:
            c.record(engine, {"exp": "agents", "error": str(e), **info})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="ours,nv")
    ap.add_argument("--phases", default="nv_start,nv_fill,ours_fill,asym,single,conc,resume,agents")
    a = ap.parse_args()
    ONLY.clear(); ONLY.update(a.engines.split(","))
    for p in a.phases.split(","):
        t0 = time.time(); print(f"=== phase {p} {time.strftime('%H:%M:%S')}", flush=True)
        globals()["phase_" + p]()
        print(f"=== phase {p} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
