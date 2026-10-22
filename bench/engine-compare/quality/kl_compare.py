"""KL(BF16 || candidate) on the paired panel, from top-k records.

Reference: w{W}/{id}.bf16.topk (bf16_ref.py). Candidates: ours tags
(w{W}/{id}.ours{tag}.topk, research_exl3_long_nll LONG_NLL_TOPK_OUT) and NInfer
configs (w{W}/ninfer.{cfg}.top60, ninfer-perplexity --topk-record --topk-k 60).
Per position: sum over the reference top-k of p_ref * (lp_ref - lp_cand), with
lp_cand from the candidate's own top-k or a floor (NInfer per_domain_kl Eq. 5
floor, -27.63) when the candidate dropped the token. Also reports top-1
agreement with the reference. Usage: python kl_compare.py [ours tags...]
"""
import glob, json, math, os, struct, sys, statistics as st
import numpy as np
sys.path.insert(0, r"D:\AI\ninfer-5090-windows\tools\release")
from per_domain_kl import read_record  # noqa: E402
import pathlib

Q = os.path.dirname(os.path.abspath(__file__))
K = 512
FLOOR = -27.631021115928547
tags = sys.argv[1:] or ['_l0_kl5']
NIN = ['nv.fp8', 'nv.bf16', 'gw.fp8', 'gw.bf16']


def load_topk(path, k=60):
    a = np.fromfile(path, dtype=np.uint8)
    rows = a.size // (8 * k)
    a = a[:rows * 8 * k].reshape(rows, 8 * k)
    ids = a[:, :4 * k].copy().view(np.int32)
    lp = a[:, 4 * k:].copy().view(np.float32)
    return ids, lp


def kl_rows(rid, rlp, cid, clp):
    out, agree = [], []
    for r in range(rid.shape[0]):
        m = dict(zip(cid[r].tolist(), clp[r].tolist()))
        p = np.exp(rlp[r].astype(np.float64))
        lc = np.array([m.get(t, FLOOR) for t in rid[r].tolist()])
        out.append(float((p * (rlp[r] - lc)).sum()))
        agree.append(1.0 if rid[r, 0] == cid[r, 0] else 0.0)
    return out, agree


def main():
    results = {}
    for W in (4096, 16384, 32768):
        d = os.path.join(Q, f'w{W}')
        refs = {os.path.basename(p)[:-len('.bf16.topk')]: load_topk(p)
                for p in glob.glob(os.path.join(d, '*.bf16.topk'))}
        if not refs:
            continue
        cands = {}
        for tag in tags:
            c = {}
            for sid in refs:
                p = os.path.join(d, f'{sid}.ours{tag}.topk')
                if os.path.exists(p):
                    c[sid] = load_topk(p)
            if c:
                cands['ours' + tag] = c
        for e in NIN:
            p = os.path.join(d, f'ninfer.{e}.top60')
            if os.path.exists(p):
                rec = read_record(pathlib.Path(p))
                c = {}
                for s in rec.streams:
                    if s.id in refs:
                        ids = np.array(s.indices[W - 1:W - 1 + K], dtype=np.int32)
                        lp = np.array(s.logprobs[W - 1:W - 1 + K], dtype=np.float32)
                        c[s.id] = (ids, lp)
                cands['ninfer.' + e] = c
        for name, c in cands.items():
            per, agr = [], []
            for sid, (rid, rlp) in refs.items():
                if sid not in c:
                    continue
                k, a = kl_rows(rid, rlp, *c[sid])
                per.append(st.mean(k)); agr.append(st.mean(a))
            if len(per) > 1:
                m = st.mean(per); ci = 1.96 * st.stdev(per) / len(per) ** 0.5
                results.setdefault(name, []).extend(per)
                print(f'w{W:5d} {name:18s} KL {m:.5f} +-{ci:.5f}  top1-agree {100*st.mean(agr):.2f}%  (n={len(per)})')
    print('-- all windows')
    for name, per in results.items():
        m = st.mean(per); ci = 1.96 * st.stdev(per) / len(per) ** 0.5
        print(f'ALL   {name:18s} KL {m:.5f} +-{ci:.5f} (n={len(per)})')


if __name__ == '__main__':
    main()
