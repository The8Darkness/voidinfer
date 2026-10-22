"""KL(reference run || candidate run) between two of our own top-k records.

Isolates one factor (e.g. the KV tier: FP16 KV vs L0 OSCAR with the same
prefill numerics) without a BF16 reference, so it works at every window.
Usage: python kl_pair.py REF_TAG CAND_TAG [CAND_TAG...] [--windows 4096,16384]
Tags name w{W}/{id}.ours{tag}.topk files (e.g. _klfp, _l0_kl5).
"""
import argparse, glob, os, statistics as st
from kl_compare import load_topk, kl_rows, Q

ap = argparse.ArgumentParser()
ap.add_argument('ref')
ap.add_argument('cands', nargs='+')
ap.add_argument('--windows', default='4096,16384,32768')
a = ap.parse_args()
for W in map(int, a.windows.split(',')):
    d = os.path.join(Q, f'w{W}')
    refs = {os.path.basename(p)[:-len(f'.ours{a.ref}.topk')]: load_topk(p)
            for p in glob.glob(os.path.join(d, f'*.ours{a.ref}.topk'))}
    for tag in a.cands:
        per, agr = [], []
        for sid, (rid, rlp) in refs.items():
            p = os.path.join(d, f'{sid}.ours{tag}.topk')
            if not os.path.exists(p):
                continue
            k, g = kl_rows(rid, rlp, *load_topk(p))
            per.append(st.mean(k)); agr.append(st.mean(g))
        if len(per) > 1:
            m = st.mean(per); ci = 1.96 * st.stdev(per) / len(per) ** 0.5
            print(f'w{W:5d} {a.ref}->{tag:14s} KL {m:.5f} +-{ci:.5f}  top1-agree {100*st.mean(agr):.2f}%  (n={len(per)})')
