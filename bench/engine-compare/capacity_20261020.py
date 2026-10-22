"""Context capacity and speed, VoidInfer vs NInfer (2026-10-20). One server per measurement, streamed
OpenAI chat requests, distinct code slices per request (no shared prefix), thinking off.

  single  --engine E --lengths 4096,32768,...      C1 TTFT / prefill tok/s / decode tok/s per prompt length
  conc    --engine E --conc N                      N simultaneous 4K prompts, greedy then sampled, aggregate tok/s
  fill    --engine E --conc N --prompt-tokens P    can N requests of P tokens decode at the same time?
  asym    --engine E --first-tokens A --prompt-tokens B   C2: agent 1 at A tokens, then agent 2 at B tokens
  resume  --engine E --tokens P                    C1: agents A, B alternate turns (switch) and repeat (same agent)

Results append to OUT/<engine>.jsonl. Engines: ours (B2 build, its defaults), nv (NInfer NVFP4, FP8 KV,
DFlash2 K7 lm-head draft, kv-capacity auto). Each request reports usage.prompt_tokens from the server.
"""
import argparse, http.client, json, os, subprocess, sys, threading, time, urllib.request

OUT = os.environ.get("CAP_OUT", r"D:\AI\capacity-20261020")
SRC = r"D:\AI\kvtier\long_code_source.cpp"
TOKENIZER = r"D:\AI\models\Qwen3.8-27B-bf16"
VOID_BUILD = os.environ.get("CAP_VOID_BUILD", r"D:\AI\voidinfer-sol6-campaign-20260923\builds\B2")
VOID_TARGET = r"C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6"
VOID_DRAFT = r"C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw"
VCPKG_BIN = r"D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin"
NINFER_EXE = r"D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe"
NINFER_NV = r"D:\AI\models\ninfer\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer"
PORT = 18600
SAMPLED = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # model card, thinking off (no presence penalty)
REVIEW = "Write a detailed review of this code, function by function."
COPY = "Repeat the code above verbatim from the beginning, adding a one-line comment above every function."


# ---------------------------------------------------------------- prompts
_ids = None


def ids():
    global _ids
    cache = os.path.join(OUT, "source_ids.json")
    if _ids is None:
        if os.path.exists(cache):
            _ids = json.load(open(cache))
        else:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(TOKENIZER)
            _ids = tok(open(SRC, encoding="utf-8", errors="ignore").read())["input_ids"]
            json.dump(_ids, open(cache, "w"))
    return _ids


_tok = None


def text(offset, n):
    """n tokens of code starting at token `offset` (wraps around the source)."""
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    src = ids()
    offset %= len(src)
    piece = (src[offset:] + src)[:n] if offset + n > len(src) else src[offset:offset + n]
    while len(piece) < n:
        piece += src[:n - len(piece)]
    return _tok.decode(piece)


def prompt(n, task=REVIEW, tag=""):
    """The code slice depends only on the tag, so both engines get identical requests."""
    import zlib
    off = zlib.crc32(tag.encode()) * 7919 % len(ids())
    return [{"role": "user", "content": f"[{tag}] Here is source code:\n{text(off, n)}\n\n{task}"}]


# ---------------------------------------------------------------- server
def server_cmd(engine, conc, max_context, extra):
    if engine == "ours":
        return [os.path.join(VOID_BUILD, "apps", "voidinfer-serve.exe"), "--target", VOID_TARGET, "--draft", VOID_DRAFT,
                "--host", "127.0.0.1", "--port", str(PORT), "--max-context", str(max_context),
                "--max-concurrency", str(conc), "--max-pending-requests", "64", "--pending-timeout-ms", "3600000",
                "--no-thinking", "--model-id", "m"] + extra
    kv = [] if "--kv-capacity" in extra else ["--kv-capacity", "auto"]
    return [NINFER_EXE, NINFER_NV, "--host", "127.0.0.1", "--port", str(PORT), "--max-context", str(max_context)] + kv + [
            "--max-concurrency", str(conc), "--kv-dtype", "fp8", "--prefill-chunk", "8192",
            "--max-pending-requests", "64", "--pending-timeout-ms", "3600000", "--no-thinking",
            "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft", "--model-id", "m"] + extra


def gpu_used():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])
    except Exception:
        return -1


class Server:
    def __init__(self, engine, conc, max_context, extra=(), env=""):
        self.engine, self.conc, self.max_context = engine, conc, max_context
        os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
        self.log_path = os.path.join(OUT, "logs", f"{engine}.c{conc}.{max_context}.{int(time.time())}.log")
        e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("VOIDINFER", "NINFER"))}
        e["PATH"] = VCPKG_BIN + ";" + e.get("PATH", "")
        for kv in filter(None, env.split(";")):
            k, v = kv.split("=", 1); e[k] = v
        self.extra = " ".join(extra)
        self.cmd = server_cmd(engine, conc, max_context, list(extra))
        self.log = open(self.log_path, "w", encoding="utf-8")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, env=e,
                                     cwd=os.path.dirname(self.cmd[0]))
        self.ok, self.error = False, ""
        while time.time() - t0 < 1200:
            if self.proc.poll() is not None:
                self.error = f"exited {self.proc.returncode}"; break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2):
                    self.ok = True; break
            except Exception:
                time.sleep(1)
        self.startup_s = time.time() - t0
        self.gpu_mib = gpu_used()
        if not self.ok:
            self.log.flush()
            self.error += " | " + " / ".join(self.tail(6))

    def tail(self, n):
        try:
            return [l.strip() for l in open(self.log_path, encoding="utf-8", errors="ignore").read().splitlines()[-n:]]
        except Exception:
            return []

    def grep(self, *words):
        try:
            return [l.strip() for l in open(self.log_path, encoding="utf-8", errors="ignore") if any(w in l for w in words)]
        except Exception:
            return []

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(30)
            except Exception: self.proc.kill()
        self.log.close()
        time.sleep(3)

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ---------------------------------------------------------------- requests
class Stream:
    """One streamed chat completion; token arrival times are kept. abort() closes the connection."""

    def __init__(self, messages, max_tokens, sampling=None):
        self.body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "stream": True,
                     "stream_options": {"include_usage": True}}
        self.body.update(sampling or {"temperature": 0})
        self.times, self.usage, self.finish, self.error, self.content = [], None, None, None, ""
        self.t_send = self.t_end = None
        self.conn = None
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self):
        self.thread.start(); return self

    def run(self):
        try:
            self.conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=7200)
            self.t_send = time.time()
            self.conn.request("POST", "/v1/chat/completions", json.dumps(self.body), {"Content-Type": "application/json"})
            r = self.conn.getresponse()
            if r.status != 200:
                self.error = f"HTTP {r.status}: {r.read()[:300]!r}"; return
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    self.usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        self.times.append(time.time())
                        self.content += d.get("content") or ""
                    if ch.get("finish_reason"):
                        self.finish = ch["finish_reason"]
        except Exception as e:
            if self.error is None and not getattr(self, "aborted", False):
                self.error = str(e)[:300]
        finally:
            self.t_end = time.time()

    def abort(self):
        self.aborted = True
        try:
            self.conn.sock.shutdown(2); self.conn.close()
        except Exception:
            pass

    def join(self, timeout=None):
        self.thread.join(timeout); return self

    @property
    def ttft(self):
        return self.times[0] - self.t_send if self.times else None

    def result(self):
        ct = (self.usage or {}).get("completion_tokens", len(self.times))
        pt = (self.usage or {}).get("prompt_tokens")
        dec = (ct - 1) / (self.times[-1] - self.times[0]) if len(self.times) > 1 and self.times[-1] > self.times[0] else None
        return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": self.ttft, "decode_tok_s": dec,
                "prefill_tok_s": (pt / self.ttft) if pt and self.ttft else None, "finish": self.finish,
                "error": self.error, "wall_s": (self.t_end or time.time()) - self.t_send if self.t_send else None}


def record(engine, rec):
    rec = {"engine": engine, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{engine}.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    return rec


def server_info(s):
    return {"startup_s": round(s.startup_s, 1), "gpu_mib_idle": s.gpu_mib, "server_ok": s.ok, "server_error": s.error,
            "log": s.log_path, "capacity_line": (s.grep("capacity |") or [""])[0][24:], "extra": s.extra,
            "build": VOID_BUILD if s.engine == "ours" else NINFER_EXE}


def warm(n=512):
    Stream(prompt(n, tag="warm"), 64).start().join()


# ---------------------------------------------------------------- experiments
def cmd_single(a):
    lengths = [int(x) for x in a.lengths.split(",")]
    with Server(a.engine, 1, a.max_context, a.extra.split(), a.env) as s:
        if not s.ok:
            return record(a.engine, {"exp": "single", **server_info(s)})
        warm()
        for L in lengths:
            n = L - a.out_tokens - 64
            for rep in range(a.reps):
                st = Stream(prompt(n, tag=f"single {L} {rep}"), a.out_tokens).start().join()
                record(a.engine, {"exp": "single", "target_tokens": L, "rep": rep, "max_context": a.max_context,
                                  **st.result(), **server_info(s)})


def cmd_conc(a):
    with Server(a.engine, a.conc, a.max_context, a.extra.split(), a.env) as s:
        if not s.ok:
            return record(a.engine, {"exp": "conc", "conc": a.conc, **server_info(s)})
        warm()
        for mode in a.modes.split(","):
            for rep in range(a.reps):
                ss = [Stream(prompt(a.prompt_tokens, tag=f"c{a.conc} {mode} {rep} {i}"), a.out_tokens,
                             SAMPLED if mode == "sampled" else None).start() for i in range(a.conc)]
                for x in ss: x.join()
                rs = [x.result() for x in ss]
                t0 = min(x.t_send for x in ss); t1 = max(x.t_end for x in ss)
                tot = sum(r["completion_tokens"] for r in rs if not r["error"])
                record(a.engine, {"exp": "conc", "conc": a.conc, "mode": mode, "rep": rep, "prompt_tokens": a.prompt_tokens,
                                  "out_tokens": a.out_tokens, "agg_tok_s": tot / (t1 - t0), "wall_s": t1 - t0,
                                  "completion_tokens": tot, "ttft_mean_s": sum(r["ttft_s"] or 0 for r in rs) / len(rs),
                                  "ttft_max_s": max(r["ttft_s"] or 0 for r in rs),
                                  "per_request_decode": [round(r["decode_tok_s"] or 0, 1) for r in rs],
                                  "errors": [r["error"] for r in rs if r["error"]], **server_info(s)})


def concurrent_window(streams, window):
    """After every stream has produced a token: did each one produce tokens within `window` s? Tokens/s."""
    start = max(x.times[0] for x in streams)
    end = start + window
    counts = [sum(1 for t in x.times if start <= t < end) for x in streams]
    return all(c > 0 for c in counts), sum(counts) / window, counts


def run_together(streams, deadline, window, sequential_first=None):
    """Waits until every stream has started (or one finished/failed first, or the deadline), then `window` s."""
    while time.time() < deadline:
        if any(x.error for x in streams):
            break
        if any(x.t_end and x.times and not x.thread.is_alive() for x in streams) and not all(x.times for x in streams):
            break  # one finished before another started: not concurrent
        if all(x.times for x in streams):
            time.sleep(window + 1); break
        time.sleep(0.5)
    started = all(x.times for x in streams)
    ok, rate, counts = concurrent_window(streams, window) if started else (False, None, None)
    for x in streams: x.abort()
    for x in streams: x.join(60)
    return started, ok, rate, counts


def cmd_fill(a):
    mc = a.max_context or max(32768, a.prompt_tokens + a.out_tokens + 512)
    env = a.env + (";VOIDINFER_EXL3_L0_SUSPEND_TRACE=1" if a.engine == "ours" else "")
    with Server(a.engine, a.conc, mc, a.extra.split(), env) as s:
        if not s.ok:
            return record(a.engine, {"exp": "fill", "conc": a.conc, "prompt_tokens_req": a.prompt_tokens,
                                     "max_context": mc, **server_info(s)})
        warm()
        ss = [Stream(prompt(a.prompt_tokens, COPY, tag=f"fill {i}"), a.out_tokens) for i in range(a.conc)]
        for x in ss: x.start(); time.sleep(0.2)
        deadline = time.time() + a.conc * a.prompt_tokens / 1500 + 300
        started, ok, rate, counts = run_together(ss, deadline, a.window)
        rs = [x.result() for x in ss]
        return record(a.engine, {"exp": "fill", "conc": a.conc, "prompt_tokens_req": a.prompt_tokens, "max_context": mc,
                          "all_started": started, "all_decoding": ok, "window_tok_s": rate, "window_counts": counts,
                          "prompt_tokens": [r["prompt_tokens"] for r in rs], "ttft_s": [r["ttft_s"] for r in rs],
                          "errors": [r["error"] for r in rs if r["error"]],
                          "suspend": s.grep("L0_SUSPEND")[:6], **server_info(s)})


def cmd_asym(a):
    mc = a.max_context or 262144
    env = a.env + (";VOIDINFER_EXL3_L0_SUSPEND_TRACE=1" if a.engine == "ours" else "")
    with Server(a.engine, 2, mc, a.extra.split(), env) as s:
        if not s.ok:
            return record(a.engine, {"exp": "asym", **server_info(s)})
        warm()
        first_out = a.first_out
        A = Stream(prompt(a.first_tokens - first_out - 64, COPY, tag="asym A"), first_out).start()
        while not A.times and A.thread.is_alive():
            time.sleep(0.5)
        B = Stream(prompt(a.prompt_tokens, COPY, tag="asym B"), a.out_tokens).start()
        deadline = time.time() + a.prompt_tokens / 1500 + 300
        started, ok, rate, counts = run_together([A, B], deadline, a.window)
        ra, rb = A.result(), B.result()
        return record(a.engine, {"exp": "asym", "max_context": mc, "first_req": a.first_tokens, "second_req": a.prompt_tokens,
                          "first_prompt_tokens": ra["prompt_tokens"], "second_prompt_tokens": rb["prompt_tokens"],
                          "first_ttft_s": ra["ttft_s"], "second_ttft_s": rb["ttft_s"],
                          "both_started": started, "both_decoding": ok, "window_tok_s": rate, "window_counts": counts,
                          "errors": [r["error"] for r in (ra, rb) if r["error"]],
                          "suspend": s.grep("L0_SUSPEND")[:6], **server_info(s)})


def cmd_resume(a):
    """C1: agent A turn 1, agent B turn 1, A turn 2 (switch back), A turn 3 (same agent), B turn 2 (switch)."""
    with Server(a.engine, 1, a.max_context, a.extra.split(), a.env) as s:
        if not s.ok:
            return record(a.engine, {"exp": "resume", **server_info(s)})
        warm()
        conv = {"A": prompt(a.tokens, tag="agent A"), "B": prompt(a.tokens, tag="agent B")}
        n = {"A": 0, "B": 0}
        for step, who in enumerate(["A", "B", "A", "A", "B"]):
            st = Stream(conv[who], a.out_tokens).start().join()
            r = st.result()
            kind = "cold" if n[who] == 0 else ("same agent" if step > 0 and ["A", "B", "A", "A", "B"][step - 1] == who else "switch")
            record(a.engine, {"exp": "resume", "step": step, "agent": who, "kind": kind, "turn": n[who], **r, **server_info(s)})
            conv[who] = conv[who] + [{"role": "assistant", "content": st.content}, {"role": "user", "content": f"Thanks. Now explain the next function in more detail (turn {n[who]})."}]
            n[who] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["single", "conc", "fill", "asym", "resume"])
    ap.add_argument("--engine", choices=["ours", "nv"], required=True)
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--max-context", type=int, default=0)
    ap.add_argument("--lengths", default="4096,32768,131072")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--modes", default="greedy,sampled")
    ap.add_argument("--prompt-tokens", type=int, default=4000)
    ap.add_argument("--out-tokens", type=int, default=1024)
    ap.add_argument("--first-tokens", type=int, default=262144)
    ap.add_argument("--first-out", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=100000)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--env", default="")
    a = ap.parse_args()
    if a.exp in ("single", "resume") and not a.max_context:
        a.max_context = 262144
    if a.exp == "conc" and not a.max_context:
        a.max_context = 32768
    {"single": cmd_single, "conc": cmd_conc, "fill": cmd_fill, "asym": cmd_asym, "resume": cmd_resume}[a.exp](a)


if __name__ == "__main__":
    main()
