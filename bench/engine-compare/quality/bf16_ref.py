"""BF16 reference top-k records for the paired quality panel (layer-streamed).

The 51.75 GiB BF16 Qwen3.8-27B does not fit the 32 GB GPU, so this runs the
transformers Qwen3.5 decoder one layer at a time: every fixture's hidden
states stay on the GPU while each layer's BF16 weights are loaded once and
applied to all fixtures (no per-pass weight streaming). Same inputs as
ours_nll.ps1: each fixture's tail ids (tokens [512, W+512) of the stream);
the scored positions predict tail[W-512 .. W-1] = tokens [W, W+512).

Output per fixture: w{W}/{id}.bf16.topk = for each of the 512 scored
positions, k int32 token ids then k float32 log-probabilities (fp32
log-softmax of BF16 logits), plus {id}.bf16.nll (target NLL per position).
"""
import argparse, glob, json, os, sys, time
import torch
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

ap = argparse.ArgumentParser()
ap.add_argument('--model', default=r'D:\AI\models\Qwen3.8-27B-bf16')
ap.add_argument('--quality', default=r'D:\AI\engine-compare-20261008\quality')
ap.add_argument('--windows', default='4096,16384,32768')
ap.add_argument('--k', type=int, default=60)
ap.add_argument('--layers', type=int, default=0, help='debug: stop after N layers')
a = ap.parse_args()
dev = torch.device('cuda')
K = 512

cfg_all = json.load(open(os.path.join(a.model, 'config.json')))
cfg = Qwen3_5TextConfig(**cfg_all['text_config'])
cfg._attn_implementation = 'sdpa'
n_layers = cfg.num_hidden_layers

index = json.load(open(os.path.join(a.model, 'model.safetensors.index.json')))['weight_map']
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.model, f), framework='pt', device='cpu')
    return handles[f].get_tensor(name)


prefix = 'model.language_model.'
# Fixtures: tail ids per (W, id).
fixtures = []
for W in map(int, a.windows.split(',')):
    for p in sorted(glob.glob(os.path.join(a.quality, f'w{W}', '*.tail.ids'))):
        ids = list(map(int, open(p).read().split()))
        assert len(ids) >= W, (p, len(ids))
        fixtures.append((W, os.path.basename(p)[:-len('.tail.ids')], ids[:W]))
print(f'{len(fixtures)} fixtures, {sum(len(f[2]) for f in fixtures)} tokens', flush=True)

torch.set_grad_enabled(False)
emb = tensor(prefix + 'embed_tokens.weight')
hidden = [emb[torch.tensor(f[2])].to(dev) .unsqueeze(0) for f in fixtures]  # [1, L, 5120] bf16
del emb
rot = mq.Qwen3_5TextRotaryEmbedding(cfg).to(dev)
pos_emb = []
for h in hidden:
    L = h.shape[1]
    pid = torch.arange(L, device=dev).view(1, 1, L).expand(3, 1, L)
    pos_emb.append(rot(h, pid))

t0 = time.time()
for i in range(a.layers or n_layers):
    layer = mq.Qwen3_5DecoderLayer(cfg, i)
    lp = f'{prefix}layers.{i}.'
    sd = {k[len(lp):]: tensor(k) for k in index if k.startswith(lp)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (i, missing, unexpected)
    layer = layer.to(device=dev, dtype=torch.bfloat16)
    # Parameters stored in FP32 by the checkpoint (e.g. GDN A_log) stay FP32.
    for name, p in layer.named_parameters():
        if sd[name].dtype == torch.float32:
            p.data = sd[name].to(dev)
    layer.eval()
    for j, h in enumerate(hidden):
        pid = torch.arange(h.shape[1], device=dev).view(1, -1)
        hidden[j] = layer(h, position_embeddings=pos_emb[j], attention_mask=None, position_ids=pid)
    del layer, sd
    torch.cuda.empty_cache()
    print(f'layer {i} {cfg.layer_types[i]} done {time.time()-t0:.0f}s', flush=True)

norm_w = tensor(prefix + 'norm.weight').to(dev)
head_name = 'lm_head.weight' if 'lm_head.weight' in index else prefix + 'embed_tokens.weight'
head = tensor(head_name).to(dev)
norm = mq.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, torch.bfloat16)
norm.weight.data = norm_w.to(torch.bfloat16)
for (W, sid, ids), h in zip(fixtures, hidden):
    L = len(ids)
    x = norm(h[0, L - K - 1:L - 1])                     # predicts ids[L-K .. L-1]
    logits = (x @ head.T).float()                        # BF16 matmul, fp32 log-softmax
    logp = torch.log_softmax(logits, dim=-1)
    top = torch.topk(logp, a.k, dim=-1)
    tgt = torch.tensor(ids[L - K:L], device=dev)
    nll = -logp.gather(1, tgt.view(-1, 1)).squeeze(1)
    d = os.path.join(a.quality, f'w{W}')
    with open(os.path.join(d, f'{sid}.bf16.topk'), 'wb') as f:
        for r in range(K):
            f.write(top.indices[r].to(torch.int32).cpu().numpy().tobytes())
            f.write(top.values[r].cpu().numpy().astype('float32').tobytes())
    with open(os.path.join(d, f'{sid}.bf16.nll'), 'w') as f:
        f.write('\n'.join(f'{v:.6f}' for v in nll.cpu().tolist()) + '\n')
    print(f'w{W} {sid} mean_nll={nll.mean().item():.5f}', flush=True)
print('BF16_REF_DONE', flush=True)
