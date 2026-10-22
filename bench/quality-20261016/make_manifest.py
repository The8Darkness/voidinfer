"""Writes results/MANIFEST.json: frozen engine commits/builds, model artifacts, dataset revisions, seeds,
item-file hashes and the exact run settings, so every number in the report can be reproduced."""
import glob, hashlib, json, os, subprocess

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def sha(p, limit=None):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True).stdout.strip()


m = {"date": "2026-10-16/17", "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                                     capture_output=True, text=True).stdout.strip()}
void = r"D:\AI\voidinfer-sol6-high-4000-70-200-20260923"
b2 = r"D:\AI\voidinfer-sol6-campaign-20260923\builds\B2"
m["voidinfer"] = {"commit": git(void, "rev-parse", "HEAD"), "branch": git(void, "rev-parse", "--abbrev-ref", "HEAD"),
                  "dirty": bool(git(void, "status", "--porcelain", "--untracked-files=no")),
                  "research_tool_sha256": sha(os.path.join(b2, "tests", "research_exl3_long_nll.exe")),
                  "serve_sha256": sha(os.path.join(b2, "apps", "voidinfer-serve.exe"))}
nin = r"D:\AI\ninfer-5090-windows"
m["ninfer"] = {"commit": git(nin, "rev-parse", "HEAD"), "dirty": bool(git(nin, "status", "--porcelain", "--untracked-files=no")),
               "perplexity_sha256": sha(os.path.join(nin, "build", "apps", "ninfer-perplexity.exe")),
               "serve_sha256": sha(os.path.join(nin, "build", "apps", "ninfer-serve.exe"))}
models = {}
for name, d in {"exl3_6bpw": r"C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6", "dflash2_draft": r"C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw",
                "bf16": r"D:\AI\models\Qwen3.8-27B-bf16"}.items():
    files = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json")) + glob.glob(os.path.join(d, "*.txt"))):
        files[os.path.basename(f)] = sha(f)
    files["_weights"] = {os.path.basename(f): os.path.getsize(f) for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))}
    models[name] = {"path": d, "files": files}
for name, d in {"ninfer_nvfp4": r"D:\AI\models\ninfer\Qwen3.8-27B-nvfp4-NInfer"}.items():
    models[name] = {"path": d, "SHA256SUMS": open(os.path.join(d, "SHA256SUMS")).read().strip(),
                    "artifact_manifest_sha256": sha(os.path.join(d, "artifact-manifest.json"))}
m["models"] = models
repos = os.path.join(ROOT, "repos")
m["datasets"] = {
    "RULER": {"repo": "https://github.com/NVIDIA/RULER", "commit": git(os.path.join(repos, "RULER"), "rev-parse", "HEAD"),
              "note": "RULERv2 code (anonymous.4open.science/r/RULERv2) returned repository_expired -> RULER v1 tasks",
              "essays_sha256": sha(os.path.join(repos, "RULER", "scripts", "data", "synthetic", "json", "PaulGrahamEssays.json")),
              "hotpotqa_sha256": sha(os.path.join(repos, "RULER", "scripts", "data", "synthetic", "json", "hotpotqa.json")),
              "tasks": ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "vt", "qa_2"], "seed": 42,
              "samples": {"16384": 50, "32768": 50, "65536": 30, "131072": 20}},
    "NoLiMa": {"repo": "https://github.com/adobe-research/NoLiMa", "commit": git(os.path.join(repos, "NoLiMa"), "rev-parse", "HEAD"),
               "hf_dataset": "amodaresi/NoLiMa", "hf_revision_at_fetch": "378115b1f136b6ba78f90f78682bc55f70ec3ddd",
               "needle_set_sha256": sha(os.path.join(repos, "NoLiMa", "data", "needlesets", "needle_set.json")),
               "haystack": "rand_shuffle/rand_book_1.txt", "haystack_sha256": sha(os.path.join(repos, "NoLiMa", "data", "haystack", "rand_shuffle", "rand_book_1.txt")),
               "depth_grid_idx": [0, 5, 10, 15, 20, 25], "lengths": [4000, 16000, 32000], "base_seed": 42},
    "IFBench": {"items": r"D:\AI\bench3-20261014\items\ifbench.jsonl", "repo_pin": "allenai/IFBench 1c40f0c1", "hf_pin": "IFBench_test 2e8a48de"},
    "panel_sources": json.load(open(os.path.join(ROOT, "sources", "PROVENANCE.json"), encoding="utf-8")),
}
m["items_sha256"] = {os.path.basename(p): sha(p) for p in sorted(glob.glob(os.path.join(ROOT, "items", "*.jsonl")))}
m["panel"] = {"manifest_sha256": sha(os.path.join(ROOT, "panel", "manifest.json")), "streams": 70, "tokens_per_stream": 2048,
              "scored_targets": "[64, 2048)", "seed": 20261016}
m["long_panel"] = {"streams_json_sha256": sha(os.path.join(ROOT, "long", "streams.json")),
                   "manifests": {w: sha(os.path.join(ROOT, "long", w, "manifest.json")) for w in sorted(os.listdir(os.path.join(ROOT, "long"))) if w.startswith("w")}}
m["settings"] = {
    "teacher_forced_ours": "research_exl3_long_nll: prefill context (16-row + 1024-row wide chunks, full precision), then one decode row per token; top-100 + nll per position",
    "teacher_forced_ninfer": "ninfer-perplexity --context N --stride N-1 (single window), --topk-record --topk-k 100, --per-token-logprobs",
    "ours_kv_arms": {"fp16": "VOIDINFER_EXL3_L0_OSCAR=0", "int4raw": "VOIDINFER_EXL3_L0_EXACT=prefill (default L0 INT4, hot 512)",
                     "int2raw": "VOIDINFER_EXL3_L0_CODE_BITS=2;VOIDINFER_EXL3_L0_EXACT=prefill", "int4vc": "VOIDINFER_EXL3_L0_EXACT=1"},
    "references": {"refs_fp32": "transformers Qwen3.5 modules, BF16 weights upcast to FP32, FP32 compute, TF32 off",
                   "refs": "transformers BF16 compute (SDPA), FP32 lm_head", "long": "bf16_ref_long.py chunked exact attention (FP32 scores), segmented GDN"},
    "tasks_request": {"temperature": 0.0, "top_p": 1.0, "chat_template_kwargs": {"enable_thinking": False}, "concurrency": 1},
    "ours_serve": "default profile: L0 OSCAR INT4 KV + exact VeriCache, DFlash2 K7",
    "ninfer_serve": "NVFP4, --kv-dtype fp8 --kv-capacity auto, --spec dflash2 --draft-tokens 7 --lm-head-draft",
}
json.dump(m, open(os.path.join(ROOT, "results", "MANIFEST.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("ok", len(json.dumps(m)))
