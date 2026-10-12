# Vision probe: image + question requests (OpenAI chat content parts, PNG data URLs), greedy.
import argparse, base64, json, time, urllib.request, urllib.error
IMGS = {"natural_scene": r"D:\AI\ninfer-5090-windows\examples\cli\media\natural_scene.png",
        "visual_chart": r"D:\AI\ninfer-5090-windows\examples\cli\media\visual_chart.png"}
Q = {"natural_scene": "Describe this image in detail.",
     "visual_chart": "What does this chart show? List the values you can read."}
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--label", required=True); a = ap.parse_args()
with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r:
    model = json.load(r)["data"][0]["id"]
rows = []
for name, path in IMGS.items():
    url = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
    body = {"model": model, "max_tokens": 256, "temperature": 0, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": Q[name]}]}]}
    for rep in range(2):
        t0 = time.time()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                res = json.load(r)
            row = {"label": a.label, "image": name, "rep": rep, "ok": True, "s": time.time() - t0,
                   "usage": res.get("usage"), "timings": res.get("timings"),
                   "text": res["choices"][0]["message"].get("content", "")}
        except urllib.error.HTTPError as e:
            row = {"label": a.label, "image": name, "rep": rep, "ok": False, "s": time.time() - t0,
                   "status": e.code, "error": e.read().decode(errors="ignore")[:500]}
        except Exception as e:
            row = {"label": a.label, "image": name, "rep": rep, "ok": False, "s": time.time() - t0, "error": str(e)[:500]}
        print(json.dumps({k: (v[:120] if isinstance(v, str) else v) for k, v in row.items()}), flush=True)
        rows.append(row)
json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
