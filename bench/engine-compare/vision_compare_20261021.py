"""Vision function test, VoidInfer (BF16 Vision tower) vs NInfer (--vision), same requests (vision_suite.py:
single images, two images, image inside a long prompt, video, text after media, 2 concurrent image requests),
plus VoidInfer's tower rows against the Hugging Face FP32/BF16 tower (vision_ref.py).
Build from CAP_VOID_BUILD. usage: vision_compare_20261021.py [ours] [nv] [ref]"""
import json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import capacity_20261020 as c

VISION = r"D:\AI\models\Qwen3.8-27B-bf16"
MEDIA = r"D:\AI\ninfer-5090-windows\examples\cli\media"
CASES = "scene,chart,compare,long_prompt,video,text_after"
OUT = os.path.join(c.OUT, "vision")


def suite(engine, extra, env=""):
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"{engine}.json")
    with c.Server(engine, 2, 32768, extra, env) as s:
        info = c.server_info(s)
        if s.ok:
            subprocess.run([sys.executable, os.path.join(HERE, "vision_suite.py"), "--port", str(c.PORT), "--out", out,
                            "--cases", CASES, "--concurrent", "2"])
    rows = json.load(open(out, encoding="utf-8")) if s.ok and os.path.exists(out) else []
    c.record(engine, {"exp": "vision", "ok": sum(r["ok"] for r in rows), "requests": len(rows),
                      "cases": [{k: r.get(k) for k in ("case", "ok", "s", "usage", "concurrent", "status")} for r in rows],
                      **info})


def ref():
    dump = os.path.join(OUT, "dump")
    os.makedirs(dump, exist_ok=True)
    for f in os.listdir(dump): os.remove(os.path.join(dump, f))
    with c.Server("ours", 1, 32768, ["--vision", VISION, "--greedy"], f"VOIDINFER_EXL3_VISION_DUMP={dump}") as s:
        if s.ok:
            subprocess.run([sys.executable, os.path.join(HERE, "vision_suite.py"), "--port", str(c.PORT),
                            "--out", os.path.join(OUT, "ours.dump.json"), "--cases", "scene,chart"])
    r = subprocess.run([sys.executable, os.path.join(HERE, "vision_ref.py"), dump,
                        os.path.join(MEDIA, "natural_scene.png"), os.path.join(MEDIA, "visual_chart.png")],
                       capture_output=True, text=True)
    open(os.path.join(OUT, "ref.txt"), "w", encoding="utf-8").write(r.stdout + r.stderr[-2000:])
    print(r.stdout, flush=True)
    for f in os.listdir(dump): os.remove(os.path.join(dump, f))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["ours", "nv", "ref"]
    if "ours" in steps: suite("ours", ["--vision", VISION, "--greedy"])
    if "nv" in steps: suite("nv", ["--vision", "--greedy"])
    if "ref" in steps: ref()
