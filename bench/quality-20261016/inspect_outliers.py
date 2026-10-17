"""Shows the highest-KL positions of a candidate vs BF16 for given streams (diagnostic)."""
import sys, os, numpy as np
sys.argv = ['analyze_panel.py', '--cand', 'x=' + sys.argv[1]] + ['--streams'] + sys.argv[2:]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'analyze_panel.py'), encoding='utf-8').read().split('refs = load(a.ref)')[0]
src = src.replace('a = ap.parse_args()', 'ap.add_argument("--streams", nargs="*")\na = ap.parse_args()')
exec(compile(src, os.path.join(here, 'analyze_panel.py'), 'exec'))
from tokenizers import Tokenizer
tok = Tokenizer.from_file(r"D:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6\tokenizer.json")
refs = load('bf16'); c = load(a.cand[0].split('=', 1)[1])
for sid in a.streams:
    p = per_position(refs[sid], c[sid]); k = p['kl']
    print(sid, 'mean %.4f median %.5f p99 %.3f max %.2f share-of-top5 %.2f' % (k.mean(), np.median(k), np.percentile(k, 99), k.max(), np.sort(k)[-5:].sum() / k.sum()))
    for i in np.argsort(-k)[:2]:
        print('   pos', 64 + i, 'KL %.2f' % k[i], repr(tok.decode(list(ids[sid][64 + i - 20:64 + i]))), '->', repr(tok.decode([int(ids[sid][64 + i])])))
        print('     ref', [(tok.decode([int(t)]), round(float(np.exp(l)), 3)) for t, l in zip(refs[sid]['ids'][i][:4], refs[sid]['lp'][i][:4])])
        print('     cand', [(tok.decode([int(t)]), round(float(np.exp(l)), 3)) for t, l in zip(c[sid]['ids'][i][:4], c[sid]['lp'][i][:4])])
