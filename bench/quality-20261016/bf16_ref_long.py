"""Genuine BF16 reference for the long-context KV panel (transformers Qwen3.5 modules, layer-streamed).

One causal forward per stream over its longest prefix (131,584 tokens) gives the reference for every
window: the W-window prefixes are token prefixes of it (asserted), so targets [W, W+512) see the
same context [0, t). Hidden states live in pinned host memory between layers; each layer's BF16
weights are loaded once. Memory-bounded pieces, mathematically identical to the module forward:
  - GDN layers run in segments of --seg tokens carrying the causal-conv tail and the recurrent
    state (transformers' own torch_chunk_gated_delta_rule with initial_state), chunk size 64;
  - the MLP and the norms run in row chunks;
  - full attention uses the module itself (SDPA, causal) over the whole sequence.
The final lm_head product is FP32 (see bf16_ref_panel.py). --validate compares this forward
against refs/ from bf16_ref_panel.py on one short-panel stream.
Output: refs_long/w{W}/{id}.bf16.topk (512 x K ids + log-probs) and .npz (nll, entropy, mass).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=r"D:\AI\models\Qwen3.8-27B-bf16")
ap.add_argument("--windows", default="4096,16384,32768,65536,131072")
ap.add_argument("--k", type=int, default=100)
ap.add_argument("--seg", type=int, default=8192)
ap.add_argument("--rows", type=int, default=8192)
ap.add_argument("--only", default="")
ap.add_argument("--validate", action="store_true")
ap.add_argument("--fp32", action="store_true", help="FP32 compute (BF16 weights upcast, TF32 off)")
ap.add_argument("--max-len", type=int, default=0, help="truncate streams (windows above it are skipped)")
ap.add_argument("--out", default="refs_long")
ap.add_argument("--val-ref", default="refs")
a = ap.parse_args()
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_grad_enabled(False)
S = 512
windows = list(map(int, a.windows.split(",")))
DT = torch.float32 if a.fp32 else torch.bfloat16
if a.max_len:
    windows = [w for w in windows if w + 512 <= a.max_len]

cfg = Qwen3_5TextConfig(**json.load(open(os.path.join(a.model, "config.json")))["text_config"])
cfg._attn_implementation = "sdpa"


def chunked_causal_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Exact causal attention in query blocks (FP32: SDPA has no memory-efficient FP32 kernel for head
    dim 256 here and falls back to the O(L^2) math kernel). Same math as sdpa_attention_forward."""
    rep = getattr(module, "num_key_value_groups", 1)
    B, H, L, D = query.shape
    scale = scaling if scaling is not None else D ** -0.5
    out = torch.empty_like(query)
    if query.dtype != torch.float32:
        # BF16: SDPA per query block (tensor cores; FP32 accumulation inside the kernel) with an explicit
        # causal mask for the block; the KV heads are expanded once per layer.
        k_all = key.repeat_interleave(rep, dim=1)
        v_all = value.repeat_interleave(rep, dim=1)
        blk = 4096
        for q0 in range(0, L, blk):
            q1 = min(L, q0 + blk)
            mask = torch.arange(q0, q1, device=query.device)[:, None] >= torch.arange(q1, device=query.device)[None, :]
            out[:, :, q0:q1] = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, q0:q1], k_all[:, :, :q1], v_all[:, :, :q1], attn_mask=mask, scale=scale)
        del k_all, v_all
        return out.transpose(1, 2).contiguous(), None
    blk = 1024
    for q0 in range(0, L, blk):
        q1 = min(L, q0 + blk)
        for h0 in range(0, H, 8):
            h1 = min(H, h0 + 8)
            k = key[:, h0 // rep:(h1 - 1) // rep + 1, :q1].repeat_interleave(rep, dim=1)[:, h0 % rep: h0 % rep + (h1 - h0)]
            v = value[:, h0 // rep:(h1 - 1) // rep + 1, :q1].repeat_interleave(rep, dim=1)[:, h0 % rep: h0 % rep + (h1 - h0)]
            s = (query[:, h0:h1, q0:q1].float() @ k.float().transpose(-1, -2)) * scale
            mask = torch.arange(q0, q1, device=s.device)[:, None] < torch.arange(q1, device=s.device)[None, :]
            s.masked_fill_(mask, float("-inf"))
            out[:, h0:h1, q0:q1] = (torch.softmax(s, dim=-1) @ v.float()).to(out.dtype)
    return out.transpose(1, 2).contiguous(), None


if True:  # bounded memory for every dtype (SDPA oversubscribed VRAM at 131K in BF16, math fallback in FP32)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS["chunked_fp32"] = chunked_causal_attention
    cfg._attn_implementation = "chunked_fp32"  # name kept; scores/softmax in FP32, output in the run dtype
index = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
handles = {}


def tensor(name):
    f = index[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(a.model, f), framework="pt", device="cpu")
    return handles[f].get_tensor(name)


prefix = "model.language_model."
if a.validate:
    sid = a.only or "cpp-00"
    streams = [(sid, list(map(int, open(os.path.join(ROOT, "panel", sid + ".ids")).read().split())))]
    windows = []
else:
    top = os.path.join(ROOT, "long", f"w{max(windows)}")
    man = json.load(open(os.path.join(top, "manifest.json"), encoding="utf-8"))
    streams = []
    for s in man["streams"]:
        if a.only and s["id"] not in a.only.split(","):
            continue
        ids = list(map(int, open(os.path.join(top, s["id"] + ".ids")).read().split()))
        for W in windows:
            short = list(map(int, open(os.path.join(ROOT, "long", f"w{W}", s["id"] + ".ids")).read().split()))
            assert short == ids[: len(short)], (s["id"], W, "window ids are not a prefix")
        streams.append((s["id"], ids[: a.max_len or None]))
print(f"{len(streams)} streams, lengths {sorted(set(len(x) for _, x in streams))}", flush=True)

emb = tensor(prefix + "embed_tokens.weight")
hidden = [emb[torch.tensor(x)].to(DT).pin_memory() for _, x in streams]
del emb
rot = mq.Qwen3_5TextRotaryEmbedding(cfg).to(dev)


def chunked(fn, x):
    out = torch.empty_like(x)
    for r in range(0, x.shape[1], a.rows):
        out[:, r:r + a.rows] = fn(x[:, r:r + a.rows])
    return out


def gdn_forward(m, h):
    """Qwen3_5GatedDeltaNet.forward (no cache) in segments with carried conv tail + recurrent state."""
    B, L, _ = h.shape
    K = m.conv_kernel_size
    out = torch.empty_like(h)
    tail = None
    state = None
    w = m.conv1d.weight.squeeze(1)
    for s0 in range(0, L, a.seg):
        x = h[:, s0:s0 + a.seg]
        n = x.shape[1]
        qkv = m.in_proj_qkv(x).transpose(1, 2)                        # [B, C, n]
        z = m.in_proj_z(x).reshape(B, n, -1, m.head_v_dim)
        b = m.in_proj_b(x)
        aa = m.in_proj_a(x)
        ext = torch.cat([tail, qkv], dim=2) if tail is not None else F.pad(qkv, (K - 1, 0))
        tail = ext[:, :, -(K - 1):]
        conv = F.conv1d(ext, w.unsqueeze(1), m.conv1d.bias, groups=qkv.shape[1])[:, :, -n:]
        conv = F.silu(conv).to(qkv.dtype).transpose(1, 2)
        q, k, v = torch.split(conv, [m.key_dim, m.key_dim, m.value_dim], dim=-1)
        q = q.reshape(B, n, -1, m.head_k_dim)
        k = k.reshape(B, n, -1, m.head_k_dim)
        v = v.reshape(B, n, -1, m.head_v_dim)
        beta = b.sigmoid()
        g = -m.A_log.float().exp() * F.softplus(aa.float() + m.dt_bias)
        rep = m.num_v_heads // m.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        core, state = mq.torch_chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=state,
                                                      output_final_state=True, use_qk_l2norm_in_kernel=True)
        core = m.norm(core.reshape(-1, m.head_v_dim), z.reshape(-1, m.head_v_dim)).reshape(B, n, -1)
        out[:, s0:s0 + n] = m.out_proj(core)
    return out


def layer_forward(layer, h, pe, pid):
    r = h
    x = chunked(layer.input_layernorm, h)
    if layer.block_type == "linear_attention":
        x = gdn_forward(layer.linear_attn, x)
    else:
        x, _ = layer.self_attn(hidden_states=x, attention_mask=None, position_ids=pid, position_embeddings=pe)
    h = r + x
    del x
    return h + chunked(lambda y: layer.mlp(layer.post_attention_layernorm(y)), h)


t0 = time.time()
for i in range(cfg.num_hidden_layers):
    layer = mq.Qwen3_5DecoderLayer(cfg, i)
    lp = f"{prefix}layers.{i}."
    sd = {k[len(lp):]: tensor(k) for k in index if k.startswith(lp)}
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (i, missing, unexpected)
    layer = layer.to(device=dev, dtype=DT)
    for name, p in layer.named_parameters():
        if sd[name].dtype == torch.float32:
            p.data = sd[name].to(dev)
    layer.eval()
    for j, (sid, x) in enumerate(streams):
        L = len(x)
        h = hidden[j].to(dev, non_blocking=True).unsqueeze(0)
        pid = torch.arange(L, device=dev).view(1, -1)
        pe = rot(h[:, :1], pid.view(1, 1, L).expand(3, 1, L)) if layer.block_type == "full_attention" else None
        h = layer_forward(layer, h, pe, pid)
        hidden[j].copy_(h[0])
        del h, pe
    del layer, sd
    torch.cuda.empty_cache()
    print(f"layer {i} {cfg.layer_types[i]} {time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

norm = mq.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev, DT)
norm.weight.data = tensor(prefix + "norm.weight").to(dev, DT)
head_name = "lm_head.weight" if "lm_head.weight" in index else prefix + "embed_tokens.weight"
head = tensor(head_name).to(dev).float()


def score(h, tgt):
    logp = torch.log_softmax(norm(h).float() @ head.T, dim=-1)
    top = torch.topk(logp, a.k, dim=-1)
    nll = -logp.gather(1, torch.tensor(tgt, device=dev).view(-1, 1)).squeeze(1)
    ent = -(logp.exp() * logp).sum(-1)
    rec = np.concatenate([top.indices.to(torch.int32).cpu().numpy().view(np.uint8),
                          top.values.cpu().numpy().astype(np.float32).view(np.uint8)], axis=1)
    return rec, nll.cpu().numpy(), ent.cpu().numpy(), top.values.exp().sum(-1).cpu().numpy()


if a.validate:
    sid, x = streams[0]
    rec, nll, ent, mass = score(hidden[0][: len(x) - 1].to(dev), x[1:])
    raw = np.fromfile(os.path.join(ROOT, a.val_ref, f"{sid}.bf16.topk"), dtype=np.uint8).reshape(len(x) - 1, -1)
    k = a.k
    ri, rl = raw[:, :4 * k].copy().view(np.int32), raw[:, 4 * k:].copy().view(np.float32)
    ni, nl = rec[:, :4 * k].copy().view(np.int32), rec[:, 4 * k:].copy().view(np.float32)
    old = np.load(os.path.join(ROOT, a.val_ref, f"{sid}.bf16.npz"))["nll"]
    p = np.exp(rl[:, :20].astype(np.float64))
    kl = []
    for r in range(len(ri)):
        mp = dict(zip(ni[r].tolist(), nl[r].tolist()))
        kl.append(float((p[r] * (rl[r, :20] - np.array([mp.get(t, nl[r, -1]) for t in ri[r, :20]]))).sum()))
    kl = np.array(kl)
    seg_means = [float(kl[i:i + a.seg].mean()) for i in range(0, len(kl), a.seg)]
    print("VALIDATE per-segment KL means", [f"{v:.1e}" for v in seg_means])
    print(f"VALIDATE {sid}: top1 agree {(ri[:, 0] == ni[:, 0]).mean() * 100:.2f}%  KL(top20) mean {np.mean(kl):.2e} max {np.max(kl):.2e}"
          f"  |dNLL| mean {np.abs(old - nll).mean():.2e}")
else:
    for j, (sid, x) in enumerate(streams):
        for W in windows:
            d = os.path.join(ROOT, a.out, f"w{W}")
            os.makedirs(d, exist_ok=True)
            rec, nll, ent, mass = score(hidden[j][W - 1: W - 1 + S].to(dev), x[W: W + S])
            rec.tofile(os.path.join(d, f"{sid}.bf16.topk"))
            np.savez(os.path.join(d, f"{sid}.bf16.npz"), nll=nll, entropy=ent, mass=mass)
            print(f"w{W} {sid} nll={nll.mean():.4f} ent={ent.mean():.3f}", flush=True)
print("BF16_LONG_DONE", time.time() - t0, flush=True)
