"""BF16 on-policy check of served generations (layer-streamed, like bf16_ref.py).

Each engine's greedy answer from the task runs (tasks/run_engine.py) is appended to the
prompt rendered with the model's own chat template, and the BF16 model scores every answer
token (plus the stop token when the engine stopped): whether it is the BF16 argmax, the BF16
regret max_logp - logp(token), and the BF16 NLL. This covers the whole serving path (prompt
ingestion, KV tier, speculative decoding, verification): an engine that loses quality
anywhere picks tokens the BF16 model would not. Answers are re-tokenized from their text,
identically for every engine.

Usage: python onpolicy_bf16.py --configs ours,nv.fp8,gw.fp8 --suites gsm8k:100,math500:100 --out X.json
"""
import argparse, json, os, sys, time, zlib
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=r'D:\AI\models\Qwen3.8-27B-bf16')
ap.add_argument('--runs', default=r'D:\AI\quality-bench-20261012\runs')
ap.add_argument('--configs', default='ours,nv.fp8,gw.fp8')
ap.add_argument('--suites', default='gsm8k:100,math500:100,ifeval:100,humaneval:82')
ap.add_argument('--max-answer', type=int, default=1536)
ap.add_argument('--out', required=True)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tasks'))
from suites import SUITES  # noqa: E402

dev = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(a.model)
im_end = tok.convert_tokens_to_ids('<|im_end|>')
configs = a.configs.split(',')


def load_run(cfg, suite):
    p = os.path.join(a.runs, cfg, suite + '.jsonl')
    recs = {}
    for line in open(p, encoding='utf-8'):
        if line.strip():
            r = json.loads(line)
            if 'error' not in r:
                recs[r['id']] = r
    return recs


# Sequences: the same items for every engine (deterministic subset of the common ids).
seqs = []  # (cfg, suite, id, ids, first_scored, stopped)
for spec in a.suites.split(','):
    suite, n = spec.split(':')
    items_fn, _, _, thinking = SUITES[suite]
    items = {it['id']: it for it in items_fn()}
    runs = {c: load_run(c, suite) for c in configs}
    common = sorted(set(items).intersection(*[set(r) for r in runs.values()]),
                    key=lambda i: zlib.crc32(i.encode()))[:int(n)]
    for iid in common:
        it = items[iid]
        messages = it.get('messages') or [{'role': 'user', 'content': it['prompt']}]
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                         enable_thinking=thinking)
        p_ids = tok(prompt, add_special_tokens=False)['input_ids']
        for c in configs:
            r = runs[c][iid]
            ans = tok(r['content'], add_special_tokens=False)['input_ids'][:a.max_answer]
            stopped = r.get('finish_reason') == 'stop' and len(ans) < a.max_answer
            ids = p_ids + ans + ([im_end] if stopped else [])
            seqs.append((c, suite, iid, ids, len(p_ids), stopped))
total = sum(len(s[3]) for s in seqs)
print(f'{len(seqs)} sequences, {total} tokens', flush=True)

cfg_all = json.load(open(os.path.join(a.model, 'config.json')))
cfg = Qwen3_5TextConfig(**cfg_all['text_config'])
cfg._attn_implementation = 'sdpa'
index = json.load(open(os.path.join(a.model, 'model.safetensors.index.json')))['weight_map']
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.model, f), framework='pt', device='cpu')
    return handles[f].get_tensor(name)


prefix = 'model.language_model.'
torch.set_grad_enabled(False)
emb = tensor(prefix + 'embed_tokens.weight')
hidden = [emb[torch.tensor(s[3])].to(dev).unsqueeze(0) for s in seqs]
del emb
rot = mq.Qwen3_5TextRotaryEmbedding(cfg).to(dev)
t0 = time.time()
for i in range(cfg.num_hidden_layers):
    layer = mq.Qwen3_5DecoderLayer(cfg, i)
    lp = f'{prefix}layers.{i}.'
    sd = {k[len(lp):]: tensor(k) for k in index if k.startswith(lp)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (i, missing, unexpected)
    layer = layer.to(device=dev, dtype=torch.bfloat16)
    for name, p in layer.named_parameters():
        if sd[name].dtype == torch.float32:
            p.data = sd[name].to(dev)
    layer.eval()
    for j, h in enumerate(hidden):
        L = h.shape[1]
        pid = torch.arange(L, device=dev).view(1, -1)
        pe = rot(h, pid.view(1, 1, L).expand(3, 1, L))
        hidden[j] = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pid)
    del layer, sd
    torch.cuda.empty_cache()
    print(f'layer {i} {cfg.layer_types[i]} {time.time()-t0:.0f}s', flush=True)

norm = mq.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, torch.bfloat16)
norm.weight.data = tensor(prefix + 'norm.weight').to(dev, torch.bfloat16)
head_name = 'lm_head.weight' if 'lm_head.weight' in index else prefix + 'embed_tokens.weight'
head = tensor(head_name).to(dev)
records = []
for (c, suite, iid, ids, first, stopped), h in zip(seqs, hidden):
    L = len(ids)
    x = norm(h[0, first - 1:L - 1])                      # predicts ids[first .. L-1]
    logp = torch.log_softmax((x @ head.T).float(), dim=-1)
    tgt = torch.tensor(ids[first:L], device=dev)
    chosen = logp.gather(1, tgt.view(-1, 1)).squeeze(1)
    top2 = torch.topk(logp, 2, dim=-1)
    records.append({'config': c, 'suite': suite, 'id': iid, 'stopped': stopped,
                    'chosen': [round(v, 5) for v in chosen.cpu().tolist()],
                    'best': [round(v, 5) for v in top2.values[:, 0].cpu().tolist()],
                    'second': [round(v, 5) for v in top2.values[:, 1].cpu().tolist()],
                    'argmax_ok': (top2.indices[:, 0] == tgt).cpu().tolist()})
json.dump(records, open(a.out, 'w'))
print('ONPOLICY_DONE', a.out, flush=True)
