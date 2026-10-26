"""Checks the 12 roomly project conventions (long-horizon v4) in the current directory; prints one JSON object.
Static checks read the source with ast; runtime checks import the package. Each check is independent:
a crash in one counts as a failure of that convention only. c11 = latest default duration (45 minutes)."""
import ast, csv, importlib, inspect, json, logging, os, pkgutil, re, subprocess, sys, tempfile, traceback
from datetime import datetime, timezone

sys.path.insert(0, os.getcwd())
PKG = "roomly"
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
res, why = {}, {}


def files():
    out = []
    for root, _, fs in os.walk(PKG):
        if "__pycache__" in root:
            continue
        out += [os.path.join(root, f) for f in fs if f.endswith(".py")]
    return sorted(out)


def trees():
    return {f: ast.parse(open(f, encoding="utf-8").read(), f) for f in files()}


def check(name):
    def deco(fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, "crash: " + "".join(traceback.format_exception_only(type(e), e)).strip()[:300]
        res[name], why[name] = bool(ok), detail
        return fn
    return deco


def sched():
    return importlib.import_module(PKG + ".scheduler").Scheduler(clock=lambda: T0)


def all_modules():
    mods = [importlib.import_module(PKG)]
    for m in pkgutil.walk_packages([PKG], PKG + "."):
        if m.name.endswith("__main__"):
            continue
        mods.append(importlib.import_module(m.name))
    return mods


@check("c01_package")
def _():
    fs = files()
    return (len(fs) >= 3 and os.path.exists(os.path.join(PKG, "__init__.py"))), f"{len(fs)} files"


@check("c02_no_print")
def _():
    bad = []
    for f, t in trees().items():
        if os.path.basename(f) == "__main__.py":
            continue
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print":
                bad.append(f"{f}:{n.lineno}")
    return not bad, bad[:5]


@check("c03_roomlyerror_slug")
def _():
    errors = importlib.import_module(PKG + ".errors")
    base = getattr(errors, "RoomlyError")
    seen, bad = {}, []
    for mod in all_modules():
        for name, cls in vars(mod).items():
            if not (inspect.isclass(cls) and issubclass(cls, BaseException)) or cls.__module__ != mod.__name__:
                continue
            if not issubclass(cls, base):
                bad.append(f"{name} not RoomlyError")
                continue
            if cls is base:
                continue
            slug = getattr(cls, "slug", None)
            if not (isinstance(slug, str) and re.fullmatch(r"[a-z]+(-[a-z]+)*", slug)):
                bad.append(f"{name} slug={slug!r}")
            elif slug in seen and seen[slug] is not cls:
                bad.append(f"{name} duplicate {slug}")
            seen[slug] = cls
    for n in ("ValidationError", "UnknownRoomError", "UnknownBookingError", "ConflictError", "CapacityError", "LoadError"):
        c = getattr(errors, n, None)
        if c is None or not issubclass(c, base):
            bad.append(f"{n} missing/not RoomlyError")
    return not bad, bad[:6]


@check("c04_logging")
def _():
    bad = []
    for f, t in trees().items():
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "getLogger":
                a = n.args[0] if n.args else None
                if not (isinstance(a, ast.Constant) and a.value == "roomly"):
                    bad.append(f"{f}:{n.lineno} getLogger({ast.unparse(a) if a else ''})")
    recs = []

    class H(logging.Handler):
        def emit(self, r):
            recs.append(r)
    root = logging.getLogger(); h = H(logging.DEBUG); root.addHandler(h); root.setLevel(logging.DEBUG)
    try:
        s = sched()
        a = s.add_room("Log A", 5)
        s.book(a, "2026-03-02T09:00", 60, "x")
        try:
            s.book(a, "2026-03-02T09:30", 30, "clash")
        except Exception:
            pass
    finally:
        root.removeHandler(h)
    if not recs:
        bad.append("no log records")
    for r in recs:
        if r.name != "roomly":
            bad.append(f"logger {r.name}")
        if not r.getMessage().startswith("roomly: "):
            bad.append(f"message {r.getMessage()[:40]!r}")
    return not bad, bad[:5]


@check("c05_stdlib")
def _():
    bad = []
    for f, t in trees().items():
        for n in ast.walk(t):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                names = [n.module]
            for m in names:
                top = m.split(".")[0]
                if top != PKG and top not in sys.stdlib_module_names:
                    bad.append(f"{f}: {m}")
    return not bad, bad[:5]


@check("c06_docstrings")
def _():
    bad = []
    for f, t in trees().items():
        scopes = [t] + [n for n in t.body if isinstance(n, ast.ClassDef)]
        for sc in scopes:
            for n in sc.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                    if not ast.get_docstring(n):
                        bad.append(f"{f}:{n.lineno} {n.name}")
    return not bad, bad[:6]


@check("c07_minute_times")
def _():
    bad = []
    for f, t in trees().items():
        src = open(f, encoding="utf-8").read()
        if re.search(r"utcnow\s*\(", src):
            bad.append(f"{f}: utcnow")
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", "") in ("now", "today") and not n.args and not n.keywords:
                if "date" in ast.unparse(n.func):
                    bad.append(f"{f}:{n.lineno} naive {ast.unparse(n)}")
    s = sched()
    a = s.add_room("Ts A", 5)
    x = s.book(a, "2026-03-02T09:00", 60, "x")
    b = s.booking(x)
    for k in ("start", "end"):
        if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d", str(b[k])):
            bad.append(f"booking {k} {b[k]!r}")
    d = s.day_bookings(a, "2026-03-02")[0]
    if d["start"] != "2026-03-02T09:00":
        bad.append(f"day_bookings start {d['start']!r}")
    p = os.path.join(tempfile.mkdtemp(), "db.json")
    s.save(p)
    text = open(p, encoding="utf-8").read()
    if re.search(r"T\d\d:\d\d:\d\d|[+-]\d\d:\d\d\"|\dZ\"", text):
        bad.append("saved file has seconds or offsets")
    if "2026-03-02T09:00" not in text:
        bad.append("saved file lacks minute string")
    return not bad, bad[:5]


@check("c08_safe_write")
def _():
    bad = []
    fp = os.path.join(PKG, "fileio.py")
    st = open(fp, encoding="utf-8").read()
    if "os.replace" not in st and "replace(" not in st:
        bad.append("fileio.py has no os.replace")
    for f, t in trees().items():
        if os.path.normpath(f) == os.path.normpath(fp):
            continue
        for n in ast.walk(t):
            if not isinstance(n, ast.Call):
                continue
            fn = ast.unparse(n.func)
            if fn.endswith(("write_text", "write_bytes")) or fn in ("os.rename", "shutil.copy", "shutil.move"):
                bad.append(f"{f}:{n.lineno} {fn}")
            if fn == "open" or fn.endswith(".open"):
                mode = n.args[1] if len(n.args) > 1 else next((k.value for k in n.keywords if k.arg == "mode"), None)
                mv = mode.value if isinstance(mode, ast.Constant) else ("?" if mode is not None else "r")
                if any(c in str(mv) for c in "wax+?"):
                    bad.append(f"{f}:{n.lineno} open mode {mv}")
    fileio = importlib.import_module(PKG + ".fileio")
    calls = []
    orig = fileio.safe_write

    def spy(path, text):
        calls.append(path); return orig(path, text)
    patched = [m for m in all_modules() if getattr(m, "safe_write", None) is orig]
    for m in patched:
        m.safe_write = spy
    try:
        s = sched(); s.add_room("Save A", 5); s.save(os.path.join(tempfile.mkdtemp(), "db.json"))
    finally:
        for m in patched:
            m.safe_write = orig
    if not calls:
        bad.append("save() did not call safe_write")
    return not bad, bad[:5]


@check("c09_exit5")
def _():
    d = tempfile.mkdtemp(); db = os.path.join(d, "db.json")
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    run = lambda *a: subprocess.run([sys.executable, "-m", PKG, db, *a], capture_output=True, text=True, env=env, timeout=60)
    bad = []
    r = run("add-room", "Exit A", "5")
    if r.returncode != 0:
        bad.append(f"add-room rc {r.returncode}")
    rid = r.stdout.strip()
    for args in (("book", "R999", "2030-03-04T09:00", "--duration", "60"), ("book", rid, "2030-03-04T09:00", "--duration", "abc"),
                 ("book", rid, "2030-03-04T09:00", "--duration", "60", "--attendees", "50"), ("frobnicate",),
                 ("book",), ("cancel", "BK-99999")):
        r = run(*args)
        if r.returncode != 5:
            bad.append(f"{' '.join(args[:1])} rc {r.returncode}")
    return not bad, bad


@check("c10_header")
def _():
    bad = [f for f in files() if open(f, encoding="utf-8").readline().rstrip("\r\n") != "# roomly - internal use only"]
    return not bad, bad[:6]


@check("c11_default_45")
def _():
    s = sched()
    a = s.add_room("Dur A", 5)
    b = s.booking(s.book(a, "2026-03-02T09:00", None, "default"))
    bad = []
    if b["end"][-5:] != "09:45":
        bad.append(f"book default end {b['end']!r}")
    p = os.path.join(tempfile.mkdtemp(), "in.csv")
    with open(p, "w", newline="") as f:
        csv.writer(f).writerows([["room", "start", "duration", "title", "attendees"], [a, "2026-03-03T09:00", "", "csv", "1"]])
    s.import_csv(p)
    c = s.day_bookings(a, "2026-03-03")
    if not c or c[0]["end"][-5:] != "09:45":
        bad.append(f"csv default {c[0]['end'] if c else None!r}")
    return not bad, bad


@check("c12_ids")
def _():
    s = sched()
    rs = [s.add_room(f"Id {i}", 5) for i in range(3)]
    bs = [s.book(rs[0], f"2026-03-0{d}T09:00", 30, "x") for d in (2, 3)]
    return rs == ["R001", "R002", "R003"] and bs == ["BK-00001", "BK-00002"], rs + bs


print(json.dumps({"ok": res, "detail": why, "score": sum(res.values())}, default=str))
