# Vision suite (greedy, OpenAI chat content parts): single images, two images in one turn, an image
# inside a long (>2K-token) prompt, a video, text after media, and N concurrent image requests.
import argparse, base64, json, time, urllib.request, urllib.error, concurrent.futures as cf
M = r"D:\AI\ninfer-5090-windows\examples\cli\media"
def data_url(name, mime):
    return f"data:{mime};base64," + base64.b64encode(open(f"{M}\\{name}", "rb").read()).decode()
IMG = lambda n: {"type": "image_url", "image_url": {"url": data_url(n, "image/png")}}
VID = lambda n: {"type": "video_url", "video_url": {"url": data_url(n, "video/mp4")}}
TXT = lambda t: {"type": "text", "text": t}
LONG = open(r"D:\AI\kvtier\long_code_source.cpp", encoding="utf-8", errors="ignore").read()[:12000]
CASES = {
    "scene": [IMG("natural_scene.png"), TXT("Describe this image in detail.")],
    "chart": [IMG("visual_chart.png"), TXT("What does this chart show? List the values you can read.")],
    "compare": [IMG("compare_left.png"), IMG("compare_right.png"), TXT("What are the differences between these two images?")],
    "long_prompt": [TXT("Here is some source code:\n" + LONG + "\n\nNow look at this image."), IMG("visual_chart.png"),
                    TXT("What is the title of the chart, and which function in the code above is defined first?")],
    "video": [VID("temporal_events.mp4"), TXT("Describe what happens in this video, in order.")],
    "text_after": "What is the capital of France? Answer in one word.",
    "text400": "Summarize this code in three sentences:\n" + LONG[:1500],
}
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cases", default="scene,chart,compare,long_prompt,video,text_after"); ap.add_argument("--concurrent", type=int, default=0)
ap.add_argument("--tokens", type=int, default=256); a = ap.parse_args()
with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=10) as r: model = json.load(r)["data"][0]["id"]
def run(name):
    content = CASES[name]
    body = {"model": model, "max_tokens": a.tokens, "temperature": 0, "messages": [{"role": "user", "content": content}]}
    t0 = time.time()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r: res = json.load(r)
        return {"case": name, "ok": True, "s": round(time.time() - t0, 3), "usage": res.get("usage"),
                "text": res["choices"][0]["message"].get("content", "")}
    except urllib.error.HTTPError as e:
        return {"case": name, "ok": False, "s": round(time.time() - t0, 3), "status": e.code, "error": e.read().decode(errors="ignore")[:400]}
    except Exception as e:
        return {"case": name, "ok": False, "s": round(time.time() - t0, 3), "error": str(e)[:400]}
rows = []
for name in a.cases.split(","):
    row = run(name); rows.append(row)
    print(json.dumps({k: (v[:160].replace("\n", " ") if isinstance(v, str) else v) for k, v in row.items()}, ensure_ascii=True), flush=True)
if a.concurrent:
    names = ["scene", "chart"] * ((a.concurrent + 1) // 2)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrent) as ex: res = list(ex.map(run, names[:a.concurrent]))
    wall = time.time() - t0
    toks = sum(r["usage"]["completion_tokens"] for r in res if r["ok"])
    print(json.dumps({"concurrent": a.concurrent, "ok": sum(r["ok"] for r in res), "wall_s": round(wall, 2),
                      "agg_tok_s": round(toks / wall, 1), "same_as_serial": [r["text"] == next(x["text"] for x in rows if x["case"] == r["case"]) for r in res if r["ok"]]}), flush=True)
    rows += [dict(r, concurrent=a.concurrent) for r in res]
json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
