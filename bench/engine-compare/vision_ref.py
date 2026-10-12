# Reference check of VoidInfer's BF16 Vision tower: Hugging Face Qwen3.5 vision model (BF16 and FP32)
# on the same images; compares our dumped FP16 merged rows (NINFER_EXL3_VISION_DUMP) per token.
import sys, glob, os, json, numpy as np, torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

BF16 = r"D:\AI\models\Qwen3.8-27B-bf16"
dump_dir, images = sys.argv[1], sys.argv[2:]
cfg = AutoConfig.from_pretrained(BF16)
proc = AutoImageProcessor.from_pretrained(BF16)
idx = json.load(open(os.path.join(BF16, "model.safetensors.index.json")))["weight_map"]
state = {}
for shard in sorted({f for k, f in idx.items() if k.startswith("model.visual.")}):
    with safe_open(os.path.join(BF16, shard), "pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."): state[k[len("model.visual."):]] = f.get_tensor(k)
models = {}
for name, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    m = M.Qwen3_5VisionModel._from_config(cfg.vision_config, torch_dtype=dtype)
    m.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=True)
    models[name] = m.to("cuda").eval()
dumps = sorted(glob.glob(os.path.join(dump_dir, "vision_*.f16")), key=os.path.getmtime)
def cos(a, b): return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))
def rel(a, b): return np.linalg.norm(a - b) / np.linalg.norm(b)
for image, dump in zip(images, dumps):
    px = proc(images=[Image.open(image).convert("RGB")], return_tensors="pt")
    grid = px["image_grid_thw"]
    out = {}
    with torch.no_grad():
        for name, m in models.items():
            pv = px["pixel_values"].to("cuda", m.dtype)
            out[name] = m(pv, grid_thw=grid.to("cuda")).pooler_output.float().cpu().numpy()
    ours = np.fromfile(dump, dtype=np.float16).astype(np.float32).reshape(-1, 5120)
    print(f"{os.path.basename(image)} grid={grid.tolist()} hf_tokens={out['fp32'].shape[0]} ours={ours.shape[0]} ({os.path.basename(dump)})")
    if ours.shape != out["fp32"].shape: print("  SHAPE MISMATCH"); continue
    for name, x in (("ours", ours), ("hf_bf16", out["bf16"])):
        c = cos(x, out["fp32"])
        print(f"  {name:8s} vs fp32: cos mean {c.mean():.6f} min {c.min():.6f}  rel_l2 {rel(x, out['fp32']):.5f}")
    c = cos(ours, out["bf16"]); print(f"  ours     vs hf_bf16: cos mean {c.mean():.6f} min {c.min():.6f}  rel_l2 {rel(ours, out['bf16']):.5f}")
