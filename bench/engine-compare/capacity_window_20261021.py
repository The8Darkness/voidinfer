"""VoidInfer C1 full 262,144-token window and the C8 per-lane fill limit (build from CAP_VOID_BUILD).
usage: capacity_window_20261021.py [c1] [c8]"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import capacity_20261020 as c
import capacity_queue_20261020 as q

WINDOW = 262144


def c1_full():
    """One request whose prompt plus generated tokens reach the whole window: the reply must stop at
    the context limit (finish 'length') with prompt + completion == 262,144."""
    prompt_req = WINDOW - q.FILL_OUT
    with c.Server("ours", 1, WINDOW, ["--greedy"]) as s:
        if not s.ok:
            return c.record("ours", {"exp": "window", "max_context": WINDOW, **c.server_info(s)})
        x = c.Stream(c.prompt(prompt_req, c.COPY, tag="window"), 2 * q.FILL_OUT).start().join(7200)
        r = x.result()
        total = (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
        rec = c.record("ours", {"exp": "window", "max_context": WINDOW, "prompt_tokens_req": prompt_req, **r,
                                "total_tokens": total, "full_window": total == WINDOW and not r["error"],
                                **c.server_info(s)})
    q.save("ours_window_c1", rec["total_tokens"] if rec["full_window"] else None)


def c8_fill():
    q.save("ours_fill_c8", q.search(lambda P: q.fill_ok("ours", 8, P, ""), 12288, 40960, 1024))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["c1", "c8"]
    if "c1" in steps: c1_full()
    if "c8" in steps: c8_fill()
