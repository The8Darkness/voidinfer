"""Agentic coding benchmark, VoidInfer vs NInfer (2026-10-23).

One bash-only agent loop (mini-swe-agent style text protocol: one ```bash block per reply), identical for both
engines, with harness-side context compaction: when the next prompt would pass COMPACT_AT x window, the model
writes a handoff summary that replaces the conversation. Commands run in WSL Ubuntu.

Suites
  polyglot  Aider polyglot benchmark, Python + C++ exercises; tests hidden during attempt 1, failing test output
            shown for attempt 2 (aider's protocol, run as an agent).
  lh        Long-horizon compaction survival (lh_spec.py): 12 stages building `vxledger`; project conventions only in
            the first message; filler = reference source pasted with every stage to fill the context.

Usage: python -I bench.py --engine ours|nv --suite lh|polyglot --conc N --window W [--filler F] [--runs R] --tag T
"""
import argparse, concurrent.futures as cf, importlib, http.client, json, os, re, subprocess, sys, threading, time, traceback, urllib.request, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lh_spec  # noqa: E402
import lh2_spec  # noqa: E402
import lh3_spec  # noqa: E402

# Paths: override with AB_<NAME> environment variables (defaults are the machine the published runs used).
ROOT = os.environ.get("AB_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WSL_ROOT = os.environ.get("AB_WSL_ROOT", "/mnt/" + ROOT[0].lower() + ROOT[2:].replace("\\", "/"))
DISTRO = "Ubuntu-24.04"
TOKENIZER = os.environ.get("AB_TOKENIZER", r"D:\AI\models\Qwen3.8-27B-bf16")
SRC_IDS = os.environ.get("AB_SRC_IDS", r"D:\AI\capacity-20261020\source_ids.json")
VOID_DIR = os.environ.get("AB_VOID_DIR", r"D:\AI\voidinfer-worktrees\quality-neutral-speed-20261020\dist\voidinfer-v2026.10.22-win64-sm120a")
VOID_TARGET = os.environ.get("AB_VOID_TARGET", r"C:\AI\models\Qwen3.8-27B-exl3-6bpw-H6-V6")
VOID_DRAFT = os.environ.get("AB_VOID_DRAFT", r"C:\AI\models\Qwen3.8-27B-DFlash2-EXL3-5bpw")
NINFER_EXE = os.environ.get("AB_NINFER_EXE", r"D:\AI\ninfer-5090-windows\build\apps\ninfer-serve.exe")
NINFER_NV = os.environ.get("AB_NINFER_NV", r"D:\AI\models\ninfer\Qwen3.8-27B-nvfp4-NInfer\qwen3_8_27b_nvfp4.ninfer")
VCPKG_BIN = os.environ.get("AB_VCPKG_BIN", r"D:\AI\build-voidinfer-mia-parity-fp16kv-fast45-20260919\vcpkg_installed\x64-windows\bin")
PORT = 18700
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 0.0}  # model card T/top_p/top_k, thinking off; no penalty (code)
MAX_TURN_TOKENS = 4096
THINKING = os.environ.get("BENCH_THINKING") == "1"
if THINKING:   # model card thinking sampling; room for the reasoning (not sent back in the history)
    SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
    MAX_TURN_TOKENS = 16384
COMPACT_AT = 0.80
OBS_CHARS = 24000
CMD_TIMEOUT = 120
STAGES = int(os.environ.get("LH_STAGES", "12"))

SYSTEM = """You are an autonomous software engineer working in a Linux shell.
Every reply: a short explanation of your next step, then exactly ONE bash code block (```bash ... ```) with the command(s) to run. The command runs in a fresh non-interactive bash in the working directory {wd}; `cd` and environment variables do not persist between replies. There are no interactive editors: create and edit files with heredocs (cat <<'EOF' > file), sed, or small python scripts. Long command output is truncated in the middle. `python` (3.12, with pytest), g++, make and cmake are available. Do not install packages.
{done}"""

COMPACT_PROMPT = """Your context window is almost full. Write a handoff summary that will REPLACE this entire conversation: after this message you will see only the summary, so it must contain everything you need to continue the work on your own. Include: the overall goal; every rule, convention and constraint the user gave (verbatim where possible); the current stage and its exact requirements; which files exist and what they contain; what is done and what remains; known problems. Do not repeat the reference source code dumps. Keep it under about 1500 words. Reply with the summary only, no bash block."""

SUMMARY_PREFIX = "Summary of the work so far (written by you before your context was compacted; the earlier conversation is gone):\n\n"


# ---------------------------------------------------------------- tokens
_tok = None
_tok_lock = threading.Lock()


def tok():
    global _tok
    with _tok_lock:
        if _tok is None:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def ntok(text):
    return len(tok()(text, add_special_tokens=False)["input_ids"])


_src = None


def filler(tag, n):
    """n tokens of C++ source; the slice depends only on tag, so both engines get identical text."""
    global _src
    if n <= 0:
        return ""
    if _src is None:
        _src = json.load(open(SRC_IDS))
    off = zlib.crc32(tag.encode()) * 7919 % len(_src)
    piece = (_src[off:] + _src)[:n]
    return tok().decode(piece)


# ---------------------------------------------------------------- server
def server_cmd(engine, conc, window):
    cmd = _server_cmd(engine, conc, window)
    return [c for c in cmd if c != "--no-thinking"] if THINKING else cmd


def _server_cmd(engine, conc, window):
    if engine == "ours":
        return [os.path.join(VOID_DIR, "voidinfer-serve.exe"), "--target", VOID_TARGET, "--draft", VOID_DRAFT,
                "--host", "127.0.0.1", "--port", str(PORT), "--max-context", str(window),
                "--max-concurrency", str(conc), "--max-pending-requests", "64", "--pending-timeout-ms", "3600000",
                "--host-kv-mib", str(max(20000, conc * window * 64 // 1024 + 4096)),
                "--no-thinking", "--model-id", "m", "--log-stats-interval-ms", "0"]
    return [NINFER_EXE, NINFER_NV, "--host", "127.0.0.1", "--port", str(PORT), "--max-context", str(window),
            "--kv-capacity", os.environ.get("NV_KV", "auto"), "--max-concurrency", str(conc), "--kv-dtype", "fp8", "--prefill-chunk", "8192",
            "--max-pending-requests", "64", "--pending-timeout-ms", "3600000", "--no-thinking",
            *([ "--host-context-mib", os.environ["NV_HOST_MIB"]] if os.environ.get("NV_HOST_MIB") else []),
            *([ "--device-state-slots", os.environ["NV_STATE_SLOTS"]] if os.environ.get("NV_STATE_SLOTS") else []),
            "--spec", "dflash2", "--draft-tokens", "7", "--lm-head-draft", "--model-id", "m", "--log-stats-interval-ms", "0"]


class Server:
    def __init__(self, engine, conc, window, logdir):
        os.makedirs(logdir, exist_ok=True)
        self.log_path = os.path.join(logdir, f"server.{engine}.c{conc}.{window}.{int(time.time())}.log")
        e = {k: v for k, v in os.environ.items() if not k.upper().startswith(("VOIDINFER", "NINFER"))}
        e["PATH"] = VCPKG_BIN + ";" + e.get("PATH", "")
        e["VOIDINFER_EXL3_KV_SPILL_DIR"] = os.environ.get("AB_KV_SPILL_DIR", r"D:\AI\kv-spill")
        self.cmd = server_cmd(engine, conc, window)
        self.log = open(self.log_path, "w", encoding="utf-8")
        t0 = time.time()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT, env=e, cwd=os.path.dirname(self.cmd[0]))
        self.ok = False
        while time.time() - t0 < 1200:
            if self.proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2):
                    self.ok = True
                    break
            except Exception:
                time.sleep(1)
        self.startup_s = time.time() - t0

    def alive(self):
        return self.proc.poll() is None

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(60)
            except Exception:
                self.proc.kill()
        self.log.close()
        time.sleep(5)


# ---------------------------------------------------------------- LLM call
class CallError(Exception):
    pass


def chat(messages, max_tokens):
    """Streamed chat completion. Returns (text, record)."""
    body = {"model": "m", "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **SAMPLING}
    t0 = time.time()
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=3600)
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = conn.getresponse()
    if r.status != 200:
        raise CallError(f"HTTP {r.status}: {r.read()[:400].decode('utf-8', 'ignore')}")
    text, usage, finish, t_first, reasoning = "", None, None, None, 0
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("error"):
            raise CallError(f"stream error: {str(ev['error'])[:400]}")
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            reasoning += len(d.get("reasoning_content") or "")
            if piece and t_first is None:
                t_first = time.time()
            text += piece
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    conn.close()
    t1 = time.time()
    if usage is None:
        raise CallError("no usage in stream")
    rec = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
           "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
           "wall_s": round(t1 - t0, 3), "ttft_s": round((t_first or t1) - t0, 3), "finish": finish,
           "reasoning_chars": reasoning}
    return text, rec


# ---------------------------------------------------------------- shell (WSL)
def sh(wd, command, timeout=CMD_TIMEOUT):
    script = (f"cd {wd} || exit 99\nsource /root/agentbench/.venv/bin/activate\n"
              f"timeout {timeout} bash -c {shquote(command)} < /dev/null 2>&1\n")
    try:
        p = subprocess.run(["wsl", "-d", DISTRO, "--", "bash", "-s"], input=script.encode(), capture_output=True,
                           timeout=timeout + 60)
        out = p.stdout.decode("utf-8", "replace")
        rc = p.returncode
    except subprocess.TimeoutExpired:
        out, rc = "(command timed out)", 124
    if rc == 124 and "timed out" not in out:
        out += f"\n(command timed out after {timeout} s)"
    return rc, out


def shquote(s):
    return "'" + s.replace("'", "'\"'\"'") + "'"


def truncate(out, limit=OBS_CHARS):
    if len(out) <= limit:
        return out
    h = limit // 2
    return out[:h] + f"\n... [{len(out) - limit} characters truncated] ...\n" + out[-h:]


BASH_RE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.S)


# ---------------------------------------------------------------- agent
class Agent:
    """Conversation with harness-side compaction. All counts and timings go into self.stats/self.calls."""

    def __init__(self, wd, window, done_word, log):
        self.wd, self.window, self.log = wd, window, log
        self.done_word = done_word
        self.messages = [{"role": "system", "content": SYSTEM.format(wd=wd, done=f"When told to signal completion, reply with a bash block containing only `echo {done_word}`.")}]
        self.pending = []            # user messages not yet sent
        self.last_prompt = 0         # prompt tokens of the last call
        self.last_completion = 0
        self.calls, self.summaries = [], []
        self.summary_delivered, self.delivered, self.stage_pending = [], 0, False
        self.compactions = 0
        self.t_llm = self.t_exec = 0.0

    def est_next(self):
        if self.last_prompt == 0:
            base = sum(ntok(m["content"]) + 8 for m in self.messages)
        else:
            base = self.last_prompt + self.last_completion + 8
        return base + sum(ntok(m["content"]) + 8 for m in self.pending)

    def compact(self, reason):
        msgs = [dict(m) for m in self.messages] + [{"role": "user", "content": COMPACT_PROMPT}]
        # the summary request must fit: shrink the longest messages (middle cut) until history + 2K output fits
        for _ in range(12):
            sizes = [ntok(m["content"]) + 8 for m in msgs]
            est = sum(sizes)
            if est + 2048 + 64 <= self.window:
                break
            i = max(range(1, len(msgs) - 1), key=lambda j: sizes[j])
            c = msgs[i]["content"]
            keep = max(400, len(c) // 3)
            msgs[i]["content"] = c[:keep] + "\n... [cut for the summary request] ...\n" + c[-keep:]
        max_t = max(512, min(MAX_TURN_TOKENS, self.window - est - 64))
        t0 = time.time()
        text, rec = chat(msgs, max_t)
        self.t_llm += time.time() - t0
        rec.update(kind="compact", reason=reason)
        self.calls.append(rec)
        summary = BASH_RE.sub("", text).strip()
        self.summaries.append(summary)
        self.summary_delivered.append(self.delivered)
        self.compactions += 1
        self.messages = self.messages[:1] + [{"role": "user", "content": SUMMARY_PREFIX + summary}]
        self.last_prompt = 0
        self.log({"event": "compact", "n": self.compactions, "reason": reason, **rec, "summary_chars": len(summary)})

    def step(self):
        """Sends pending messages, returns the assistant text."""
        for attempt in range(4):
            est = self.est_next()
            if est > COMPACT_AT * self.window:
                self.compact(f"est {est}")
                est = self.est_next()
            if self.pending:
                if self.messages[-1]["role"] == "user":
                    self.messages[-1]["content"] += "\n\n" + "\n\n".join(m["content"] for m in self.pending)
                else:
                    self.messages.append({"role": "user", "content": "\n\n".join(m["content"] for m in self.pending)})
                self.pending = []
                if self.stage_pending:
                    self.delivered += 1
                    self.stage_pending = False
            max_t = max(256, min(MAX_TURN_TOKENS, self.window - est - 64))
            try:
                t0 = time.time()
                text, rec = chat(self.messages, max_t)
                self.t_llm += time.time() - t0
            except (CallError, OSError, http.client.HTTPException) as e:
                msg = str(e)
                self.log({"event": "call_error", "attempt": attempt, "error": msg[:400], "est": est})
                if any(w in msg.lower() for w in ("context", "too long", "exceed", "length")) and len(self.messages) > 2:
                    # undo the merge into the last user message is not needed: compaction summarizes it too
                    self.compact("error: " + msg[:80])
                    continue
                time.sleep(15)
                continue
            rec.update(kind="step", est=est)
            self.calls.append(rec)
            # with thinking, the reasoning is not kept in the history: count only the visible reply
            self.last_prompt = rec["prompt_tokens"]
            self.last_completion = ntok(text) if THINKING else rec["completion_tokens"]
            self.messages.append({"role": "assistant", "content": text})
            self.log({"event": "reply", "n": len(self.calls), "prompt": rec["prompt_tokens"], "text": text[-700:]})
            return text
        raise CallError("4 failed attempts")

    def act(self, text):
        """Runs the bash block of an assistant reply. Returns (done, observation)."""
        blocks = BASH_RE.findall(text)
        if not blocks:
            return False, "Your reply had no ```bash block. Reply with exactly one bash code block."
        cmd = blocks[0].strip()
        if re.fullmatch(rf"echo\s+['\"]?{self.done_word}['\"]?", cmd):
            return True, None
        t0 = time.time()
        rc, out = sh(self.wd, cmd)
        self.t_exec += time.time() - t0
        note = "" if len(blocks) == 1 else "\n(Note: only the first bash block was run.)"
        return False, f"<returncode>{rc}</returncode>\n<output>\n{truncate(out)}\n</output>{note}"

    def run_until_done(self, max_steps):
        steps = 0
        while steps < max_steps:
            text = self.step()
            steps += 1
            done, obs = self.act(text)
            if done:
                return True, steps
            self.pending.append({"role": "user", "content": obs})
            self.log({"event": "obs", "text": obs[:300]})
        return False, steps


# ---------------------------------------------------------------- suites
def lh_job(job, window, filler_tokens, log):
    run_wd = f"/root/agentbench/work/{job['id']}"
    sh("/root", f"rm -rf {run_wd} && mkdir -p {run_wd} && cd {run_wd} && git init -q")
    ag = Agent(run_wd, window, "STAGE_COMPLETE", log)
    stage_info = []
    t_start = time.time()
    status = "ok"
    try:
        for k, spec in enumerate(lh_spec.STAGES[:STAGES]):
            fill = filler(f"lh-stage-{k}", filler_tokens)
            parts = []
            if k == 0:
                parts.append(lh_spec.BRIEF)
            if fill:
                parts.append(f"For context, here is part {k + 1} of the legacy C++ ledger engine we are replacing. It is "
                             f"reference material only; do not port it.\n```cpp\n{fill}\n```")
            parts.append(spec)
            ag.pending.append({"role": "user", "content": "\n\n".join(parts)})
            c0 = ag.compactions
            done, steps = ag.run_until_done(30)
            stage_info.append({"stage": k + 1, "done": done, "steps": steps, "compactions": ag.compactions - c0})
            log({"event": "stage", **stage_info[-1], "elapsed_s": round(time.time() - t_start, 1)})
            if not done:
                ag.pending.append({"role": "user", "content": "Step limit for this stage reached; we move on."})
    except Exception as e:
        status = f"error: {e}"
        log({"event": "job_error", "error": traceback.format_exc()[-1500:]})
    wall = time.time() - t_start
    # score
    _, out = sh(run_wd, f"cp {WSL_ROOT}/hidden/lh/test_functional.py ./_hidden_test_functional.py && "
                        f"python -m pytest -q --tb=no -rA -p no:cacheprovider _hidden_test_functional.py; "
                        f"rm -f _hidden_test_functional.py", timeout=600)
    passed = re.findall(r"^PASSED \S+::(\S+)", out, re.M)
    failed = re.findall(r"^(?:FAILED|ERROR) \S+::(\S+)", out, re.M)
    _, conv = sh(run_wd, f"python {WSL_ROOT}/hidden/lh/check_conventions.py 2>/dev/null | tail -1", timeout=600)
    try:
        conv = json.loads(conv.strip().splitlines()[-1])
    except Exception:
        conv = {"ok": {}, "score": 0, "raw": conv[-500:]}
    recall = []
    for s in ag.summaries:
        hit = {k: bool(re.search(rx, s, re.I)) for k, rx in lh_spec.RECALL.items()}
        recall.append(hit)
    return {"status": status, "wall_s": round(wall, 1), "stages": stage_info, "compactions": ag.compactions,
            "tests_passed": len(passed), "tests_failed": len(failed), "passed": passed, "failed": failed,
            "conventions": conv.get("ok", {}), "conv_score": conv.get("score", 0), "conv_detail": conv.get("detail"),
            "summary_recall": recall, "summaries": ag.summaries, "calls": ag.calls,
            "t_llm_s": round(ag.t_llm, 1), "t_exec_s": round(ag.t_exec, 1)}


def lh2_job(job, window, filler_tokens, log, S=lh2_spec, suite="lh2"):
    """Long-horizon v2/v3: rules only in per-stage team notes (lh2_spec.py / lh3_spec.py); functional + rule tests +
    conventions from hidden/<suite>."""
    run_wd = f"/root/agentbench/work/{job['id']}"
    sh("/root", f"rm -rf {run_wd} && mkdir -p {run_wd} && cd {run_wd} && git init -q")
    ag = Agent(run_wd, window, "STAGE_COMPLETE", log)
    stage_info, side_info = [], []
    t_start = time.time()
    status = "ok"
    try:
        for k, spec in enumerate(S.STAGES[:STAGES]):
            fill = filler(f"{suite}-stage-{k}", filler_tokens)
            parts = []
            if k == 0:
                parts.append(S.BRIEF)
            parts.append(f"Team chat since the last stage:\n```text\n{S.NOTES[k]}\n```")
            if getattr(S, "DISTRACT", None) and S.DISTRACT[k]:
                parts.append(f"{S.DISTRACT_INTRO}\n```text\n{S.DISTRACT[k]}\n```")
            if fill:
                intro = getattr(S, "FILLER_INTRO", "For context, here is part {k} of the legacy C++ ledger engine we are "
                                                   "replacing. It is reference material only; do not port it.")
                parts.append(intro.format(k=k + 1) + f"\n```cpp\n{fill}\n```")
            parts.append(spec)
            ops = S.SETUP(k) if getattr(S, "SETUP", None) else {}
            if ops:   # lh5: exports that exist only during this stage
                src = os.path.join(ROOT, "tmp_ops", job["id"], str(k))
                for rel, text in ops.items():
                    os.makedirs(os.path.dirname(os.path.join(src, rel)), exist_ok=True)
                    with open(os.path.join(src, rel), "w", encoding="utf-8", newline="\n") as f:
                        f.write(text)
                sh(run_wd, f"rm -rf ops && cp -r {WSL_ROOT}/tmp_ops/{job['id']}/{k}/ops ./ops")
            ag.pending.append({"role": "user", "content": "\n\n".join(parts)})
            ag.stage_pending = True
            c0 = ag.compactions
            done, steps = ag.run_until_done(30)
            stage_info.append({"stage": k + 1, "done": done, "steps": steps, "compactions": ag.compactions - c0})
            log({"event": "stage", **stage_info[-1], "elapsed_s": round(time.time() - t_start, 1)})
            if ops:
                sh(run_wd, "rm -rf ops")
            if not done:
                ag.pending.append({"role": "user", "content": "Step limit for this stage reached; we move on."})
            if getattr(S, "SIDE", False) and k < min(STAGES, len(S.STAGES)) - 1:
                side_info.append(side_task(ag, run_wd, suite, k + 1, filler_tokens, log))
    except Exception as e:
        status = f"error: {e}"
        log({"event": "job_error", "error": traceback.format_exc()[-1500:]})
    wall = time.time() - t_start
    H = f"{WSL_ROOT}/hidden/{getattr(S, 'HIDDEN', suite)}"
    _, out = sh(run_wd, f"cp {H}/test_functional.py ./_hidden_test_functional.py && cp {H}/test_rules.py ./_hidden_test_rules.py && "
                        f"python -m pytest -q --tb=no -rA -p no:cacheprovider _hidden_test_functional.py _hidden_test_rules.py; "
                        f"rm -f _hidden_test_functional.py _hidden_test_rules.py", timeout=600)
    passed = re.findall(r"^PASSED (\S+)::(\S+)", out, re.M)
    failed = re.findall(r"^(?:FAILED|ERROR) (\S+)::(\S+)", out, re.M)
    _, conv = sh(run_wd, f"python {H}/check_conventions.py 2>/dev/null | tail -1", timeout=600)
    try:
        conv = json.loads(conv.strip().splitlines()[-1])
    except Exception:
        conv = {"ok": {}, "score": 0, "raw": conv[-500:]}
    recall = []
    for summ, dlv in zip(ag.summaries, ag.summary_delivered):
        hit, stale = {}, {}
        for item, rx in S.RECALL.items():
            if dlv < S.ITEM_STAGE[item]:
                continue
            hit[item] = bool(re.search(rx, summ, re.I))
            if item in S.STALE:
                stale[item] = not hit[item] and bool(re.search(S.STALE[item], summ, re.I))
        recall.append({"delivered": dlv, "hit": hit, "stale": stale})
    return {"status": status, "wall_s": round(wall, 1), "stages": stage_info, "compactions": ag.compactions,
            "tests_passed": sum(1 for f, _ in passed if "functional" in f),
            "rules_passed": sum(1 for f, _ in passed if "rules" in f),
            "passed": [f"{f}::{t}" for f, t in passed], "failed": [f"{f}::{t}" for f, t in failed],
            "conventions": conv.get("ok", {}), "conv_score": conv.get("score", 0), "conv_detail": conv.get("detail"),
            "recall2": recall, "summaries": ag.summaries, "calls": ag.calls, "side": side_info,
            "t_llm_s": round(ag.t_llm, 1), "t_exec_s": round(ag.t_exec, 1)}


def side_task(ag, run_wd, suite, n, filler_tokens, log):
    """lh6: an unrelated look-up task between two stages, with its own ~filler_tokens of legacy source, so every
    stage boundary costs one extra compaction. The answer is a constant planted in the middle of the source."""
    value = zlib.crc32(f"{suite}-side-{n}".encode()) % 90000 + 10000
    fill = filler(f"{suite}-side-{n}", filler_tokens)
    lines = fill.split("\n")
    mid = len(lines) // 2
    lines.insert(mid, f"static constexpr int kSideCheck{n} = {value};  // calibration constant")
    msg = (f"Side task (not part of roomly; it changes nothing about the roomly project, its stages or its decisions). "
           f"Below is another part of the legacy C++ facility service. Find the integer value of the constant "
           f"`kSideCheck{n}` and write only that number into side/answer_{n}.txt (create the directory if needed). "
           f"Then reply with a bash block containing only `echo STAGE_COMPLETE`; the next roomly stage follows after that."
           f"\n```cpp\n" + "\n".join(lines) + "\n```")
    ag.pending.append({"role": "user", "content": msg})
    c0, t0 = ag.compactions, time.time()
    done, steps = ag.run_until_done(8)
    _, out = sh(run_wd, f"cat side/answer_{n}.txt 2>/dev/null")
    ok = out.strip() == str(value)
    info = {"side": n, "done": done, "steps": steps, "compactions": ag.compactions - c0, "ok": ok,
            "wall_s": round(time.time() - t0, 1)}
    log({"event": "side", **info})
    if not done:
        ag.pending.append({"role": "user", "content": "Step limit for the side task reached; we move on."})
    return info


POLY = os.path.join(ROOT, "repos", "polyglot")


def poly_tasks():
    out = []
    for lang in ("python", "cpp"):
        base = os.path.join(POLY, lang, "exercises", "practice")
        for ex in sorted(os.listdir(base)):
            cfg = json.load(open(os.path.join(base, ex, ".meta", "config.json")))
            out.append({"lang": lang, "ex": ex, "files": cfg["files"]})
    return out


def poly_job(job, window, filler_tokens, log):
    t = job["task"]
    lang, ex = t["lang"], t["ex"]
    src = f"{WSL_ROOT}/repos/polyglot/{lang}/exercises/practice/{ex}"
    wd = f"/root/agentbench/work/{job['id']}/{ex}"  # exercism CMakeLists names the target after the directory
    tests = t["files"]["test"]
    sh("/root", f"rm -rf {wd} && mkdir -p {wd} && cp -r {src}/. {wd}/ && cd {wd} && rm -rf .meta .approaches "
                + " ".join(f"'{x}'" for x in tests))
    docs = os.path.join(POLY, lang, "exercises", "practice", ex, ".docs")
    instr = ""
    for name in ("introduction.md", "instructions.md", "instructions.append.md"):
        p = os.path.join(docs, name)
        if os.path.exists(p):
            instr += open(p, encoding="utf-8").read().strip() + "\n\n"
    files = " ".join(t["files"]["solution"])
    task = (f"# {ex}\n\n{instr}Use the above instructions to modify the supplied files: {files}\n"
            "Don't change the names of existing functions or classes, as they may be referenced from other code like "
            "unit tests, etc. Only use standard libraries, don't suggest installing any packages.\n"
            "When you are done, reply with a bash block containing only `echo TASK_COMPLETE`.")
    ag = Agent(wd, window, "TASK_COMPLETE", log)
    ag.pending.append({"role": "user", "content": task})
    t0 = time.time()
    if lang == "python":
        test_cmd = f"python -m pytest -q --tb=short -p no:cacheprovider {' '.join(tests)}"
    else:
        test_cmd = "rm -rf build && mkdir -p build && cd build && cmake -DEXERCISM_RUN_ALL_TESTS=1 .. >/dev/null && make 2>&1"
    restore = f"cp {' '.join(src + '/' + x for x in tests)} . 2>/dev/null; " if lang == "python" else \
        f"for f in {' '.join(tests)}; do cp {src}/$f ./$f; done; "
    res = {"status": "ok"}
    try:
        _, steps1 = ag.run_until_done(20)
        rc, out = sh(wd, restore + test_cmd, timeout=600)
        p1 = rc == 0
        res.update(pass1=p1, steps1=steps1)
        p2, steps2 = p1, 0
        if not p1:
            ag.pending.append({"role": "user", "content": "The hidden tests failed:\n```\n" + out[-4000:] + "\n```\n"
                                                         "Fix the code so the tests pass. The test files are now in the "
                                                         "working directory; do not modify them. Reply with `echo TASK_COMPLETE` when done."})
            _, steps2 = ag.run_until_done(20)
            rc, out = sh(wd, restore + test_cmd, timeout=600)
            p2 = rc == 0
        res.update(pass2=p2, steps2=steps2, test_tail=out[-1500:])
    except Exception as e:
        res.update(status=f"error: {e}", pass1=res.get("pass1", False), pass2=False)
        log({"event": "job_error", "error": traceback.format_exc()[-1500:]})
    res.update(wall_s=round(time.time() - t0, 1), compactions=ag.compactions, calls=ag.calls,
               t_llm_s=round(ag.t_llm, 1), t_exec_s=round(ag.t_exec, 1), lang=lang, ex=ex)
    return res


# ---------------------------------------------------------------- runner
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, choices=["ours", "nv"])
    ap.add_argument("--suite", required=True, choices=["lh", "lh2", "lh3", "lh4", "lh5", "lh6", "lh7", "polyglot"])
    ap.add_argument("--conc", type=int, default=1)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--filler", type=int, default=0)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="polyglot: first N tasks only")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-server", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "results", a.tag)
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{a.engine}.jsonl")
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = {json.loads(l)["id"] for l in open(out_path, encoding="utf-8") if l.strip()}
    if a.suite.startswith("lh"):
        jobs = [{"id": f"{a.tag}.{a.engine}.lh{r}" if a.suite == "lh" else f"{a.tag}.{a.engine}.{a.suite}_{r}"} for r in range(a.runs)]
        if a.suite == "lh":
            fn = lh_job
        else:
            S = importlib.import_module(f"{a.suite}_spec")
            fn = lambda job, w, f, log: lh2_job(job, w, f, log, S=S, suite=a.suite)
    else:
        tasks = poly_tasks()
        if os.environ.get("POLY_LANG"):
            tasks = [t for t in tasks if t["lang"] == os.environ["POLY_LANG"]]
        if a.limit:
            tasks = tasks[:a.limit]
        jobs = [{"id": f"{a.tag}.{a.engine}.{t['lang']}.{t['ex']}.r{r}", "task": t} for r in range(a.runs) for t in tasks]
        fn = poly_job
    jobs = [j for j in jobs if j["id"] not in done_ids]
    print(f"{len(jobs)} jobs to run -> {out_path}", flush=True)
    if not jobs:
        return
    lock = threading.Lock()
    evlog = open(os.path.join(outdir, f"{a.engine}.events.jsonl"), "a", encoding="utf-8")

    def logger(job_id):
        def log(ev):
            with lock:
                evlog.write(json.dumps({"t": round(time.time(), 1), "id": job_id, **ev}) + "\n")
                evlog.flush()
        return log

    tok()  # load once before threads
    srv = None if a.no_server else Server(a.engine, a.conc, a.window, os.path.join(outdir, "logs"))
    if srv is not None and not srv.ok:
        print("server failed to start; see", srv.log_path, flush=True)
        srv.close()
        sys.exit(2)
    meta = {"engine": a.engine, "suite": a.suite, "conc": a.conc, "window": a.window, "filler": a.filler, "thinking": THINKING,
            "startup_s": round(srv.startup_s, 1) if srv else None, "cmd": srv.cmd if srv else None}
    json.dump(meta, open(os.path.join(outdir, f"{a.engine}.meta.json"), "w"), indent=1)
    t0 = time.time()

    def work(job):
        if srv is not None and not srv.alive():
            return {"id": job["id"], "status": "server dead"}
        try:
            r = fn(job, a.window, a.filler, logger(job["id"]))
        except Exception:
            r = {"status": "harness error", "error": traceback.format_exc()[-2000:]}
        r["id"] = job["id"]
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r) + "\n")
        brief = {k: r.get(k) for k in ("status", "wall_s", "compactions", "tests_passed", "rules_passed", "conv_score", "pass1", "pass2") if k in r}
        print(f"[{time.time() - t0:7.0f}s] {job['id']} {brief}", flush=True)
        return r

    with cf.ThreadPoolExecutor(a.conc) as ex:
        list(ex.map(work, jobs))
    if srv is not None:
        alive = srv.alive()
        srv.close()
        if not alive:
            print("SERVER DIED during run; see", srv.log_path, flush=True)
    print(f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
