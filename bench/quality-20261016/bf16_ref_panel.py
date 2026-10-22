"""Genuine BF16 reference for the weight panel (transformers Qwen3.5 decoder, layer-streamed).

The 51.75 GiB BF16 checkpoint does not fit the 32 GB GPU, so each decoder layer's BF16 weights are
loaded once and applied to every stream's hidden states (kept on the GPU, BF16, batches of B
streams). Activations and weights are BF16 exactly as transformers runs the model; only the final
lm_head product is taken in FP32 (BF16 logits would round values near 20 to 0.125 steps, which
alone adds ~5e-4 KL of reference noise). GDN layers use the transformers torch fallback (no fla).

Per stream (N tokens) writes refs/{id}.bf16.topk: for the N-1 positions predicting tokens 1..N-1,
K int32 ids then K float32 log-probs; and refs/{id}.bf16.npz: target nll, full-vocab entropy,
top-K probability mass per position.
"""
import argparse, json, os, time
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"D:\AI\models\Qwen3.8-27B-bf16")
ap.add_argument("--panel", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "panel"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "refs"))
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--limit", type=int, default=0, help="debug: first N streams")
ap.add_argument("--layers", type=int, default=0, help="debug: stop after N layers")
ap.add_argument("--fp32", action="store_true", help="FP32 compute (BF16 weights upcast exactly, TF32 off): high-precision reference")
ap.add_argument("--rows", type=int, default=0, help="noise-floor variant: MLP in row chunks of this size (mathematically identical)")
a = ap.parse_args()
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)
os.makedirs(a.out, exist_ok=True)

cfg_all = json.load(open(os.path.join(a.model, "config.json")))
cfg = Qwen3_5TextConfig(**cfg_all["text_config"])
cfg._attn_implementation = "sdpa"
index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.model, f), framework="pt", device="cpu")
    return handles[f].get_tensor(name)


prefix = "model.language_model."
man = json.load(open(os.path.join(a.panel, "manifest.json"), encoding="utf-8"))
streams = man["streams"][: a.limit or None]
ids = [list(map(int, open(os.path.join(a.panel, s["id"] + ".ids")).read().split())) for s in streams]
L = len(ids[0])
assert all(len(x) == L for x in ids)
print(f"{len(streams)} streams x {L} tokens", flush=True)

emb = tensor(prefix + "embed_tokens.weight")
batches = [list(range(i, min(i + a.batch, len(ids)))) for i in range(0, len(ids), a.batch)]
DT = torch.float32 if a.fp32 else torch.bfloat16
hidden = [emb[torch.tensor([ids[j] for j in b])].to(dev, DT) for b in batches]  # [B, L, H]
del emb
rot = mq.Qwen3_5TextRotaryEmbedding(cfg).to(dev)
pid3 = torch.arange(L, device=dev).view(1, 1, L).expand(3, 1, L)
cos, sin = rot(hidden[0][:1], pid3)
pid = torch.arange(L, device=dev).view(1, -1)

t0 = time.time()
for i in range(a.layers or cfg.num_hidden_layers):
    layer = mq.Qwen3_5DecoderLayer(cfg, i)
    lp = f"{prefix}layers.{i}."
    sd = {k[len(lp):]: tensor(k) for k in index if k.startswith(lp)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (i, missing, unexpected)
    layer = layer.to(device=dev, dtype=DT)
    if a.rows:
        mlp = layer.mlp
        def chunked_mlp(x, mlp=mlp):
            return torch.cat([mlp.__class__.forward(mlp, x[:, r:r + a.rows]) for r in range(0, x.shape[1], a.rows)], dim=1)
        layer.mlp.forward = chunked_mlp
    for name, p in layer.named_parameters():  # FP32 checkpoint parameters (e.g. GDN A_log) stay FP32
        if sd[name].dtype == torch.float32:
            p.data = sd[name].to(dev)
    layer.eval()
    for j, h in enumerate(hidden):
        B = h.shape[0]
        pe = (cos.expand(B, -1, -1), sin.expand(B, -1, -1))
        hidden[j] = layer(h, position_embeddings=pe, attention_mask=None, position_ids=pid.expand(B, -1))
    del layer, sd
    torch.cuda.empty_cache()
    print(f"layer {i} {cfg.layer_types[i]} {time.time() - t0:.0f}s", flush=True)

norm = mq.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, DT)
norm.weight.data = tensor(prefix + "norm.weight").to(dev, DT)
head_name = "lm_head.weight" if "lm_head.weight" in index else prefix + "embed_tokens.weight"
head = tensor(head_name).to(dev).float()
for b, h in zip(batches, hidden):
    for r, j in enumerate(b):
        x = norm(h[r, : L - 1]).float()                 # predicts ids[1 .. L-1]
        logp = torch.log_softmax(x @ head.T, dim=-1)
        top = torch.topk(logp, a.k, dim=-1)
        tgt = torch.tensor(ids[j][1:], device=dev)
        nll = -logp.gather(1, tgt.view(-1, 1)).squeeze(1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        mass = top.values.exp().sum(-1)
        rec = np.concatenate([top.indices.to(torch.int32).cpu().numpy().view(np.uint8),
                              top.values.cpu().numpy().astype(np.float32).view(np.uint8)], axis=1)
        sid = streams[j]["id"]
        rec.tofile(os.path.join(a.out, f"{sid}.bf16.topk"))
        np.savez(os.path.join(a.out, f"{sid}.bf16.npz"), nll=nll.cpu().numpy(), entropy=ent.cpu().numpy(),
                 mass=mass.cpu().numpy())
        print(f"{sid} nll={nll[63:].mean().item():.4f} ent={ent[63:].mean().item():.3f} mass={mass.mean().item():.4f}", flush=True)
        del logp, p
print("BF16_REF_DONE", time.time() - t0, flush=True)
