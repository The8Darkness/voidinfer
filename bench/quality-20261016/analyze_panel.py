"""Paired analysis of teacher-forced top-k records against a reference.

  analyze_panel.py --ref bf16 --cand ours=ours:runs/ours_fp16kv --cand nvfp4=ninfer:runs/ninfer_nv_bf16 [--json out.json]

Sources (all aligned to target token index t, scored range [--first, N)):
  bf16           refs/{id}.bf16.topk (+ .npz: target nll, full-vocab entropy) - index i predicts token i+1
  ours:DIR       DIR/{id}.topk + DIR/{id}.tok (research_exl3_long_nll; row j predicts token CONTEXT+j)
  ninfer:DIR     DIR/panel.top100 (ninfer-perplexity record) + DIR/panel.csv (per-token target logprob)
The reference may also be a candidate spec (pairwise mode, e.g. FP16 KV vs FP8 KV of one engine).

Per position: KL(ref || cand) summed over the reference top-K. A reference token missing from the
candidate's top-K gets the candidate's K-th log-prob (an upper bound on its true log-prob), so the
reported KL is a lower bound of the top-K-restricted divergence; `kl_floor` instead uses -27.63 (an
upper bound) and `miss` counts such terms. Also: target NLL, delta NLL vs reference, top-1
agreement, |p_cand(target) - p_ref(target)|.
CIs: 95% cluster bootstrap over streams (10,000 resamples, seed 1); paired differences between
candidates resample the same streams for both.
"""
import argparse, glob, hashlib, json, os, struct, sys
import numpy as np



def read_record_np(path):
    """NInfer top-k record (per_domain_kl.py format NINFKL01), memory-mapped: {sid: (digest, rows, k)}."""
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    head = struct.Struct("<8sIIQ32sII32s64s")
    magic, k, nstreams, positions = head.unpack(bytes(raw[:head.size]))[:4]
    assert magic == b"NINFKL01", path
    off, out = head.size, {}
    for _ in range(nstreams):
        strs = []
        for _ in range(2):
            n = int(raw[off]) | int(raw[off + 1]) << 8; off += 2
            strs.append(bytes(raw[off:off + n]).decode("utf-8")); off += n
        dl = int(raw[off]) | int(raw[off + 1]) << 8; off += 2
        digest = bytes(raw[off:off + dl]).hex(); off += dl
        count = struct.unpack("<I", bytes(raw[off:off + 4]))[0]; off += 4
        out[strs[0]] = (digest, raw[off:off + count * 8 * k].reshape(count, 8 * k), k)
        off += count * 8 * k
    return out


ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FLOOR = -27.631021115928547
ap = argparse.ArgumentParser()
ap.add_argument("--panel", default=os.path.join(ROOT, "panel"))
ap.add_argument("--ref", default="bf16")
ap.add_argument("--cand", action="append", default=[])
ap.add_argument("--first", type=int, default=64)
ap.add_argument("--json")
ap.add_argument("--record", default="panel.top100")
ap.add_argument("--csv", default="panel.csv")
ap.add_argument("--ref-dir", default=os.path.join(ROOT, "refs"))
a = ap.parse_args()

man = json.load(open(os.path.join(a.panel, "manifest.json"), encoding="utf-8"))
streams = man["streams"]
ids = {s["id"]: np.array(list(map(int, open(os.path.join(a.panel, s["id"] + ".ids")).read().split()))) for s in streams}
N = len(next(iter(ids.values())))


def load_bin(path, k):
    raw = np.fromfile(path, dtype=np.uint8)
    rows = raw.size // (8 * k)
    raw = raw[: rows * 8 * k].reshape(rows, 8 * k)
    return raw[:, : 4 * k].copy().view(np.int32), raw[:, 4 * k:].copy().view(np.float32)


def load(spec):
    """Returns {sid: dict(ids [n,K], lp [n,K], tlp [n], top1 [n])} for targets t in [first, N)."""
    out = {}
    if spec == "bf16" or spec.startswith("ref:"):
        rd = spec[4:] if spec.startswith("ref:") else a.ref_dir
        for sid in ids:
            p = os.path.join(rd if os.path.isabs(rd) else os.path.join(ROOT, rd), f"{sid}.bf16.topk")
            if not os.path.exists(p):
                continue
            ti, tl = load_bin(p, 100)
            z = np.load(p[:-5] + ".npz")
            # short panel refs cover predictions of tokens 1..N-1; long refs only the last N-first
            sl = slice(a.first - 1, N - 1) if ti.shape[0] == N - 1 else slice(ti.shape[0] - (N - a.first), None)
            out[sid] = dict(ids=ti[sl], lp=tl[sl], tlp=-z["nll"][sl], top1=ti[sl, 0], ent=z["entropy"][sl])
        return out
    kind, d = spec.split(":", 1)
    d = d if os.path.isabs(d) else os.path.join(ROOT, d)
    if kind == "ours":
        for sid in ids:
            p = os.path.join(d, f"{sid}.topk")
            if not os.path.exists(p):
                continue
            tok = np.loadtxt(os.path.join(d, f"{sid}.tok"))
            ctx = N - tok.shape[0]
            k = os.path.getsize(p) // (8 * tok.shape[0])
            ti, tl = load_bin(p, k)
            assert (tok[:, 2].astype(int) == ids[sid][ctx:]).all(), (sid, "target mismatch")
            sl = slice(a.first - ctx, None)
            out[sid] = dict(ids=ti[sl], lp=tl[sl], tlp=-tok[sl, 0], top1=tok[sl, 1].astype(np.int64))
        return out
    if kind == "ninfer":
        rec = read_record_np(os.path.join(d, a.record))
        csv = {}
        cp = os.path.join(d, a.csv)
        if os.path.exists(cp):
            for line in open(cp, encoding="utf-8"):
                sid_, t_, lp_ = line.rstrip("\n").split(",")
                if sid_ in ids and t_ != "token_index" and int(t_) >= a.first:
                    csv.setdefault(sid_, {})[int(t_)] = float(lp_)
        for sid, (digest, block, k) in rec.items():
            if sid not in ids:
                continue
            want = hashlib.sha256(ids[sid].astype("<u4").tobytes()).hexdigest()
            assert digest == want, (sid, "token digest differs: NInfer tokenized different ids")
            blk = np.array(block[a.first - 1:N - 1])
            ti = blk[:, :4 * k].copy().view(np.int32); tl = blk[:, 4 * k:].copy().view(np.float32)
            c = csv.get(sid, {})
            tlp = np.array([c.get(t, np.nan) for t in range(a.first, N)])
            out[sid] = dict(ids=ti, lp=tl, tlp=tlp, top1=ti[:, 0])
        return out
    raise ValueError(spec)


def per_position(ref, cand):
    rid, rlp, cid, clp = ref["ids"], ref["lp"].astype(np.float64), cand["ids"], cand["lp"].astype(np.float64)
    n, k = rid.shape
    kth = clp[:, -1:]
    # candidate log-prob of each reference top-k token (exact match within the row)
    match = rid[:, :, None] == cid[:, None, :]
    found = match.any(-1)
    lc = np.where(found, (np.where(match, clp[:, None, :], 0.0)).sum(-1), np.nan)
    p = np.exp(rlp)
    kl_lb = (p * (rlp - np.where(found, lc, kth))).sum(-1)
    kl_floor = (p * (rlp - np.where(found, lc, FLOOR))).sum(-1)
    miss = (~found).sum(-1)
    k32 = min(32, k)
    kl32 = (p[:, :k32] * (rlp[:, :k32] - np.where(found[:, :k32], lc[:, :k32], kth))).sum(-1)
    miss32 = (~found[:, :k32]).sum(-1)
    agree = (ref["top1"] == cand["top1"]).astype(np.float64)
    dnll = ref["tlp"] - cand["tlp"]              # cand nll - ref nll
    dp = np.abs(np.exp(cand["tlp"]) - np.exp(ref["tlp"]))
    return dict(kl=kl_lb, kl_floor=kl_floor, miss=miss, kl32=kl32, miss32=miss32, agree=agree, nll=-cand["tlp"], dnll=dnll, dp=dp)


rng = np.random.default_rng(1)
B = 10000


def boot(values_by_stream, weights_by_stream, idx):
    """Ratio-of-sums mean (token weighted) and its cluster-bootstrap CI over streams."""
    v = np.array(values_by_stream)[idx]
    w = np.array(weights_by_stream)[idx]
    est = v.sum() / w.sum()
    draws = rng.integers(0, len(idx), size=(B, len(idx)))
    bs = v[draws].sum(1) / w[draws].sum(1)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), draws


refs = load(a.ref)
cands = {}
for c in a.cand:
    name, spec = c.split("=", 1)
    cands[name] = load(spec)
sids = [s["id"] for s in streams if s["id"] in refs and all(s["id"] in c for c in cands.values())]
dom = {s["id"]: s["domain"] for s in streams}
print(f"ref={a.ref} streams={len(sids)} scored tokens={sum(len(refs[s]['tlp']) for s in sids)} (targets [{a.first},{N}))")
pp = {name: {sid: per_position(refs[sid], c[sid]) for sid in sids} for name, c in cands.items()}
ent = {sid: refs[sid].get("ent") for sid in sids}
report = {"ref": a.ref, "streams": len(sids), "tokens": int(sum(len(refs[s]["tlp"]) for s in sids)), "cands": {}}
METRICS = ["kl", "kl_floor", "agree", "dnll", "dp", "nll", "miss", "kl32", "miss32"]


def summarize(sel_sids, mask_fn=None):
    res = {}
    draws_cache = None
    sums = {}
    for name in cands:
        sums[name] = {}
        for m in METRICS:
            vals, wts = [], []
            for sid in sel_sids:
                x = pp[name][sid][m]
                msk = mask_fn(sid) if mask_fn else np.ones(len(x), bool)
                x = x[msk]
                x = x[~np.isnan(x)] if m in ("dnll", "dp", "nll") else x
                vals.append(float(np.sum(x))); wts.append(len(x))
            sums[name][m] = (np.array(vals), np.array(wts))
    idx = np.arange(len(sel_sids))
    draws = rng.integers(0, len(idx), size=(B, len(idx)))
    for name in cands:
        res[name] = {}
        for m in METRICS:
            v, w = sums[name][m]
            if w.sum() == 0:
                continue
            est = v.sum() / w.sum()
            bs = v[draws].sum(1) / np.maximum(w[draws].sum(1), 1)
            res[name][m] = (est, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        res[name]["tokens"] = int(sums[name]["kl"][1].sum())
    names = list(cands)
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x, y = names[i], names[j]
                key = f"{y}-{x}"
                res[key] = {}
                for m in ("kl", "kl32", "agree", "dnll", "dp"):
                    vx, wx = sums[x][m]; vy, wy = sums[y][m]
                    est = vy.sum() / wy.sum() - vx.sum() / wx.sum()
                    bs = vy[draws].sum(1) / wy[draws].sum(1) - vx[draws].sum(1) / wx[draws].sum(1)
                    res[key][m] = (est, np.percentile(bs, 2.5), np.percentile(bs, 97.5))
                # paired sign of per-stream KL difference
                kx = sums[x]["kl"][0] / sums[x]["kl"][1]; ky = sums[y]["kl"][0] / sums[y]["kl"][1]
                res[key]["streams_y_worse_kl"] = int((ky > kx).sum())
    return res


def fmt(t, scale=1.0, d=5):
    return f"{t[0]*scale:.{d}f} [{t[1]*scale:.{d}f},{t[2]*scale:.{d}f}]"


def show(title, res):
    print(f"\n== {title}")
    for name in cands:
        r = res[name]
        print(f"  {name:10s} n={r['tokens']:6d}  KL {fmt(r['kl'])}  (floor {r['kl_floor'][0]:.5f}, miss/pos {r['miss'][0]:.2f}; top32 {r['kl32'][0]:.5f}, miss {r['miss32'][0]:.3f})"
              f"  top1 {fmt(r['agree'], 100, 2)}%  dNLL {fmt(r['dnll'])}  |dp| {fmt(r['dp'])}  NLL {r['nll'][0]:.4f}")
    for key, r in res.items():
        if "-" in key and isinstance(r, dict) and "kl" in r:
            print(f"  {key:22s} dKL {fmt(r['kl'])}  dTop1 {fmt(r['agree'], 100, 2)}pt  ddNLL {fmt(r['dnll'])}"
                  f"  d|dp| {fmt(r['dp'])}  streams where {key.split('-')[0]} has higher KL: {r['streams_y_worse_kl']}/{len(sids_cur)}")


def tojson(res):
    return {k: ({m: (list(v) if isinstance(v, tuple) else v) for m, v in r.items()}) for k, r in res.items()}


sids_cur = sids
overall = summarize(sids)
show("ALL", overall)
report["all"] = tojson(overall)
report["domains"] = {}
for d in dict.fromkeys(dom[s] for s in sids):
    sids_cur = [s for s in sids if dom[s] == d]
    r = summarize(sids_cur)
    show(f"domain {d}", r)
    report["domains"][d] = tojson(r)
sids_cur = sids
if all(ent[s] is not None for s in sids):
    report["entropy"] = {}
    for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 99)):
        r = summarize(sids, lambda sid, lo=lo, hi=hi: (ent[sid] >= lo) & (ent[sid] < hi))
        show(f"BF16 entropy [{lo},{hi}) nats", r)
        report["entropy"][f"{lo}-{hi}"] = tojson(r)
    # difficult = reference top-1 probability < 0.5
    r = summarize(sids, lambda sid: np.exp(refs[sid]["lp"][:, 0]) < 0.5)
    show("difficult (BF16 top-1 p < 0.5)", r)
    report["difficult"] = tojson(r)
if a.json:
    json.dump(report, open(a.json, "w"), indent=1)
