"""Checks the 12 vxledger project conventions in the current directory; prints one JSON object.
Static checks read the source with ast; runtime checks import the package. Each check is independent:
a crash in one counts as a failure of that convention only."""
import ast, importlib, inspect, io, json, logging, os, pkgutil, re, subprocess, sys, tempfile, traceback

sys.path.insert(0, os.getcwd())
PKG = "vxledger"
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


@check("c01_package")
def _():
    fs = files()
    return (len(fs) >= 3 and os.path.exists(os.path.join(PKG, "__init__.py"))), f"{len(fs)} files"


@check("c02_no_float")
def _():
    bad = []
    for f, t in trees().items():
        for n in ast.walk(t):
            if isinstance(n, ast.Constant) and isinstance(n.value, float):
                bad.append(f"{f}:{n.lineno} literal")
            if isinstance(n, ast.Name) and n.id == "float":
                bad.append(f"{f}:{n.lineno} float name")
    return not bad, bad[:5]


def all_modules():
    mods = [importlib.import_module(PKG)]
    for m in pkgutil.walk_packages([PKG], PKG + "."):
        if m.name.endswith("__main__"):
            continue
        mods.append(importlib.import_module(m.name))
    return mods


@check("c03_vxerror_codes")
def _():
    errors = importlib.import_module(PKG + ".errors")
    base = getattr(errors, "VxError")
    seen, bad = {}, []
    for mod in all_modules():
        for name, cls in vars(mod).items():
            if not (inspect.isclass(cls) and issubclass(cls, BaseException)) or cls.__module__ != mod.__name__:
                continue
            if not issubclass(cls, base):
                bad.append(f"{name} not VxError")
                continue
            if cls is base:
                continue
            code = cls.__dict__.get("code", getattr(cls, "code", None))
            if not (type(code) is int and 4700 <= code <= 4799):
                bad.append(f"{name} code={code!r}")
            elif code in seen and seen[code] is not cls:
                bad.append(f"{name} duplicate {code}")
            seen[code] = cls
    need = ["AmountError", "ValidationError", "UnknownAccountError", "InsufficientFundsError"]
    for n in need:
        c = getattr(errors, n, None)
        if c is None or not issubclass(c, base):
            bad.append(f"{n} missing/not VxError")
    return not bad, bad[:6]


@check("c04_logging")
def _():
    bad = []
    for f, t in trees().items():
        for n in ast.walk(t):
            if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == "getLogger":
                a = n.args[0] if n.args else None
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str) and
                        (a.value == "vx.ledger" or a.value.startswith("vx.ledger."))):
                    bad.append(f"{f}:{n.lineno} getLogger({ast.unparse(a) if a else ''})")
    # runtime: messages emitted by a typical session all start with "[VX] " and come from vx.ledger
    recs = []

    class H(logging.Handler):
        def emit(self, r):
            recs.append(r)
    root = logging.getLogger(); h = H(logging.DEBUG); root.addHandler(h); root.setLevel(logging.DEBUG)
    try:
        led = importlib.import_module(PKG + ".ledger").Ledger()
        a = led.open_account("Log A", "EUR"); b = led.open_account("Log B", "EUR")
        led.deposit(a, 100); led.transfer(a, b, 10)
        try:
            led.withdraw(b, 1000)
        except Exception:
            pass
    finally:
        root.removeHandler(h)
    if not recs:
        bad.append("no log records")
    for r in recs:
        if not (r.name == "vx.ledger" or r.name.startswith("vx.ledger.")):
            bad.append(f"logger {r.name}")
        if not r.getMessage().startswith("[VX] "):
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


@check("c06_type_hints")
def _():
    bad = []
    for f, t in trees().items():
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_"):
                args = n.args.posonlyargs + n.args.args + n.args.kwonlyargs
                args = [a for a in args if a.arg not in ("self", "cls")]
                if n.args.vararg: args.append(n.args.vararg)
                if n.args.kwarg: args.append(n.args.kwarg)
                if n.returns is None or any(a.annotation is None for a in args):
                    bad.append(f"{f}:{n.lineno} {n.name}")
    return not bad, bad[:6]


@check("c07_utc_z")
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
    led = importlib.import_module(PKG + ".ledger").Ledger()
    a = led.open_account("Ts A", "EUR"); led.deposit(a, 5)
    ts = led.history(a)[-1]["timestamp"]
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", str(ts)):
        bad.append(f"timestamp {ts!r}")
    return not bad, bad[:5]


@check("c08_atomic")
def _():
    bad = []
    sp = os.path.join(PKG, "storage.py")
    st = open(sp, encoding="utf-8").read()
    if "os.replace" not in st and "replace(" not in st:
        bad.append("storage.py has no os.replace")
    for f, t in trees().items():
        if os.path.normpath(f) == os.path.normpath(sp):
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
    # runtime: atomic_write works and save goes through it
    storage = importlib.import_module(PKG + ".storage")
    calls = []
    orig = storage.atomic_write

    def spy(path, data):
        calls.append(path); return orig(path, data)
    d = tempfile.mkdtemp()
    ledger_mod = importlib.import_module(PKG + ".ledger")
    patched = [m for m in all_modules() if getattr(m, "atomic_write", None) is orig]
    for m in patched:
        m.atomic_write = spy
    try:
        led = ledger_mod.Ledger(); led.open_account("Save A", "EUR"); led.save(os.path.join(d, "db.json"))
    finally:
        for m in patched:
            m.atomic_write = orig
    if not calls:
        bad.append("save() did not call atomic_write")
    return not bad, bad[:5]


@check("c09_exit3")
def _():
    d = tempfile.mkdtemp(); db = os.path.join(d, "db.json")
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    run = lambda *a: subprocess.run([sys.executable, "-m", PKG, db, *a], capture_output=True, text=True, env=env, timeout=60)
    bad = []
    r = run("open", "Exit A", "--currency", "EUR")
    if r.returncode != 0:
        bad.append(f"open rc {r.returncode}")
    acc = r.stdout.strip()
    for args in (("withdraw", acc, "5"), ("deposit", acc, "abc"), ("balance", "ACC-999999"), ("frobnicate",),
                 ("deposit", acc)):
        r = run(*args)
        if r.returncode != 3:
            bad.append(f"{' '.join(args[:1])} rc {r.returncode}")
    return not bad, bad


@check("c10_spdx")
def _():
    bad = [f for f in files() if open(f, encoding="utf-8").readline().rstrip("\r\n") != "# SPDX-License-Identifier: MPL-2.0"]
    return not bad, bad[:6]


@check("c11_chf")
def _():
    money = importlib.import_module(PKG + ".money")
    led = importlib.import_module(PKG + ".ledger").Ledger()
    a = led.open_account("Cur A")
    bad = []
    if money.format_amount(123456) != "CHF 1,234.56":
        bad.append(f"format_amount default {money.format_amount(123456)!r}")
    if led.account(a)["currency"] != "CHF":
        bad.append(f"open_account default {led.account(a)['currency']!r}")
    return not bad, bad


@check("c12_acc_ids")
def _():
    led = importlib.import_module(PKG + ".ledger").Ledger()
    ids = [led.open_account(f"Id {i}", "EUR") for i in range(3)]
    return ids == ["ACC-000001", "ACC-000002", "ACC-000003"], ids


print(json.dumps({"ok": res, "detail": why, "score": sum(res.values())}, default=str))
