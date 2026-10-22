"""Import helper: exposes analyze_panel's loaders (load, per_position, ids, streams) without running the report."""
import os, sys
_here = os.path.dirname(os.path.abspath(__file__))


def setup(argv=()):
    sys.argv = ["analyze_panel.py", *argv]
    g = {"__file__": os.path.join(_here, "analyze_panel.py"), "__name__": "panel_lib_exec"}
    src = open(g["__file__"], encoding="utf-8").read().split("refs = load(a.ref)")[0]
    exec(compile(src, g["__file__"], "exec"), g)
    return g
