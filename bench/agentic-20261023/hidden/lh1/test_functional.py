"""Hidden functional tests for the 12 vxledger stages (statement names upper case, hold ids HOLD-0001, default transfer memo). They only check what the stage messages state;
the project conventions (default currency, id format, logging prefix, exit status, ...) are checked by
check_conventions.py. Tests use explicit currencies and returned ids so they never depend on a convention."""
import csv, json, logging, os, subprocess, sys
from datetime import datetime, timezone, timedelta
import pytest


def M():
    from vxledger import money
    return money


def E():
    from vxledger import errors
    return errors


def L(**kw):
    from vxledger.ledger import Ledger
    return Ledger(**kw)


class Clock:
    """Returns `now` however often the ledger calls it; tests move it explicitly."""
    def __init__(self, start=datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)):
        self.now = start

    def __call__(self):
        return self.now


# ---------------------------------------------------------------- stage 1
@pytest.mark.parametrize("s,v", [("12", 1200), ("12.5", 1250), ("-0.05", -5), ("1,234.56", 123456), (" 7.00 ", 700),
                                 ("0", 0), ("1,000,000", 100000000), ("-1,234", -123400), ("3.1", 310)])
def test_s1_parse_ok(s, v):
    assert M().parse_amount(s) == v


@pytest.mark.parametrize("s", ["", "1.234", "12,34", "abc", "1e3", "--1", "1,2345", "1.", ".5x", "12 34"])
def test_s1_parse_bad(s):
    with pytest.raises(E().AmountError):
        M().parse_amount(s)


def test_s1_format():
    f = M().format_amount
    assert f(123456, "EUR") == "EUR 1,234.56"
    assert f(-1200, "USD") == "USD -12.00"
    assert f(5, "EUR") == "EUR 0.05"
    assert f(0, "EUR") == "EUR 0.00"
    assert f(100000000, "GBP") == "GBP 1,000,000.00"


def test_s1_atomic_write(tmp_path):
    from vxledger.storage import atomic_write
    p = tmp_path / "x.txt"
    atomic_write(str(p), "hello")
    atomic_write(str(p), "world")
    assert p.read_text() == "world"
    assert sorted(os.listdir(tmp_path)) == ["x.txt"]


# ---------------------------------------------------------------- stage 2
def test_s2_open_and_account():
    led = L()
    a = led.open_account("Alice", "EUR")
    b = led.open_account("Bob", "EUR")
    assert a != b
    info = led.account(a)
    assert info == {"id": a, "name": "Alice", "currency": "EUR", "balance": 0}
    from vxledger import Ledger  # re-export
    assert Ledger is type(led)


@pytest.mark.parametrize("name,cur", [("", "EUR"), ("   ", "EUR"), ("alice", "EUR"), ("Zed", "eur"), ("Zed", "EURO"), ("Zed", "E1R")])
def test_s2_open_invalid(name, cur):
    led = L()
    led.open_account("Alice", "EUR")
    with pytest.raises(E().ValidationError):
        led.open_account(name, cur)


def test_s2_deposit_withdraw_ids():
    led = L()
    a = led.open_account("A", "EUR")
    b = led.open_account("B", "EUR")
    assert led.deposit(a, 1000) == "TX00000001"
    assert led.deposit(b, 500, "x") == "TX00000002"
    assert led.withdraw(a, 300) == "TX00000003"
    assert led.balance(a) == 700 and led.balance(b) == 500
    assert led.account(a)["balance"] == 700


def test_s2_errors():
    led = L()
    a = led.open_account("A", "EUR")
    for bad in (0, -5):
        with pytest.raises(E().ValidationError):
            led.deposit(a, bad)
        with pytest.raises(E().ValidationError):
            led.withdraw(a, bad)
    led.deposit(a, 100)
    with pytest.raises(E().InsufficientFundsError):
        led.withdraw(a, 101)
    assert led.balance(a) == 100
    for f in (lambda: led.account("nope"), lambda: led.balance("nope"), lambda: led.deposit("nope", 5)):
        with pytest.raises(E().UnknownAccountError):
            f()


# ---------------------------------------------------------------- stage 3
def test_s3_transfer():
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 1000)
    tx = led.transfer(a, b, 400, "rent")
    assert tx == "TX00000002"
    assert led.balance(a) == 600 and led.balance(b) == 400
    assert led.deposit(b, 1) == "TX00000003"


def test_s3_transfer_errors():
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR"); c = led.open_account("C", "USD")
    led.deposit(a, 100)
    with pytest.raises(E().ValidationError):
        led.transfer(a, a, 10)
    with pytest.raises(E().ValidationError):
        led.transfer(a, b, 0)
    with pytest.raises(E().CurrencyMismatchError):
        led.transfer(a, c, 10)
    with pytest.raises(E().InsufficientFundsError):
        led.transfer(a, b, 101)
    with pytest.raises(E().UnknownAccountError):
        led.transfer(a, "nope", 1)
    assert led.balance(a) == 100 and led.balance(b) == 0 and led.balance(c) == 0
    assert led.deposit(a, 1) == "TX00000002"  # failed operations consume no ids


def test_s3_overdraft():
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.set_overdraft(a, 500)
    led.withdraw(a, 300)
    assert led.balance(a) == -300
    led.transfer(a, b, 200)
    assert led.balance(a) == -500
    with pytest.raises(E().InsufficientFundsError):
        led.withdraw(a, 1)
    with pytest.raises(E().InsufficientFundsError):
        led.transfer(a, b, 1)
    with pytest.raises(E().ValidationError):
        led.set_overdraft(a, -1)


# ---------------------------------------------------------------- stage 4
def test_s4_history():
    led = L(clock=Clock())
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    t1 = led.deposit(a, 1000, "salary")
    t2 = led.transfer(a, b, 250, "rent")
    t3 = led.withdraw(a, 50)
    h = led.history(a)
    assert [e["tx"] for e in h] == [t1, t2, t3]
    assert [e["kind"] for e in h] == ["deposit", "transfer_out", "withdrawal"]
    assert [e["amount"] for e in h] == [1000, -250, -50]
    assert [e["balance_after"] for e in h] == [1000, 750, 700]
    assert [e["memo"] for e in h] == ["salary", "rent", ""]
    assert all(set(e) == {"tx", "kind", "amount", "balance_after", "memo", "timestamp"} for e in h)
    hb = led.history(b)
    assert len(hb) == 1 and hb[0]["tx"] == t2 and hb[0]["kind"] == "transfer_in" and hb[0]["amount"] == 250
    assert hb[0]["balance_after"] == 250


def test_s4_clock_used():
    led = L(clock=Clock(start=datetime(2026, 5, 4, 3, 2, 1, tzinfo=timezone.utc)))
    a = led.open_account("A", "EUR")
    led.deposit(a, 5)
    ts = led.history(a)[0]["timestamp"]
    assert ts.startswith("2026-05-04T03:02:01")


def test_s4_clock_other_zone():
    tz = timezone(timedelta(hours=2))
    led = L(clock=lambda: datetime(2026, 5, 4, 3, 0, 0, tzinfo=tz))
    a = led.open_account("A", "EUR")
    led.deposit(a, 5)
    assert led.history(a)[0]["timestamp"].startswith("2026-05-04T01:00:00")


# ---------------------------------------------------------------- stage 5
def _populated():
    led = L(clock=Clock())
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 1000, "m1"); led.set_overdraft(b, 300); led.transfer(b, a, 200, "m2")
    return led, a, b


def test_s5_roundtrip(tmp_path):
    led, a, b = _populated()
    p = str(tmp_path / "db.json")
    led.save(p)
    doc = json.load(open(p))
    assert doc["version"] == 2
    from vxledger.ledger import Ledger
    led2 = Ledger.load(p)
    assert led2.balance(a) == 1200 and led2.balance(b) == -200
    assert led2.history(a) == led.history(a) and led2.history(b) == led.history(b)
    assert led2.account(b) == led.account(b)
    led2.withdraw(b, 100)  # overdraft restored
    with pytest.raises(E().InsufficientFundsError):
        led2.withdraw(b, 1)
    c = led2.open_account("C", "EUR")
    assert c not in (a, b)
    assert led2.deposit(c, 1) == "TX00000004"


def test_s5_load_clock(tmp_path):
    led, a, b = _populated()
    p = str(tmp_path / "db.json"); led.save(p)
    from vxledger.ledger import Ledger
    led2 = Ledger.load(p, clock=lambda: datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    led2.deposit(a, 1)
    assert led2.history(a)[-1]["timestamp"].startswith("2030-01-02T03:04:05")


def test_s5_load_errors(tmp_path):
    from vxledger.ledger import Ledger
    with pytest.raises(E().LoadError):
        Ledger.load(str(tmp_path / "missing.json"))
    bad = tmp_path / "bad.json"; bad.write_text("{not json")
    with pytest.raises(E().LoadError):
        Ledger.load(str(bad))
    led, a, b = _populated()
    p = tmp_path / "v.json"; led.save(str(p))
    doc = json.load(open(p)); doc["version"] = 3; p.write_text(json.dumps(doc))
    with pytest.raises(E().LoadError):
        Ledger.load(str(p))


# ---------------------------------------------------------------- stage 6
def _csv(tmp_path, rows):
    p = tmp_path / "in.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "account", "amount", "memo"])
        w.writerows(rows)
    return str(p)


def test_s6_import(tmp_path):
    led = L(clock=Clock())
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    p = _csv(tmp_path, [
        ["2026-01-05", a, "100.00", "first"],      # line 2 ok
        ["2026-01-06", a, "-30", "atm"],           # line 3 ok
        ["2026-13-01", a, "5", "bad date"],         # line 4 ValidationError
        ["2026-01-07", "ACC-999999", "5", "x"],     # line 5 UnknownAccount
        ["2026-01-08", b, "1.234", "bad amt"],      # line 6 AmountError
        ["2026-01-09", b, "-1", "nsf"],             # line 7 InsufficientFunds
        ["2026-01-10", b, "1,500.25", "big"],       # line 8 ok
        ["2026-01-11", b, "0", "zero"],             # line 9 skipped (ValidationError)
    ])
    r = led.import_csv(p)
    er = E()
    assert r["imported"] == 3 and r["skipped"] == 5
    assert [list(x) for x in r["errors"]] == [[4, er.ValidationError.code], [5, er.UnknownAccountError.code],
                                             [6, er.AmountError.code], [7, er.InsufficientFundsError.code],
                                             [9, er.ValidationError.code]]
    assert led.balance(a) == 7000 and led.balance(b) == 150025
    h = led.history(a)
    assert [e["kind"] for e in h] == ["deposit", "withdrawal"]
    assert h[0]["timestamp"].startswith("2026-01-05T00:00:00") and h[0]["memo"] == "first"


# ---------------------------------------------------------------- stage 7
def test_s7_interest():
    led = L(clock=Clock())
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR"); c = led.open_account("C", "EUR")
    d = led.open_account("D", "EUR"); e = led.open_account("E", "EUR")
    led.deposit(a, 12345); led.deposit(b, 2); led.deposit(c, 6); led.set_overdraft(d, 100); led.withdraw(d, 50)
    led.deposit(e, 10)
    txs = led.apply_interest(2500)
    # a: 3086.25 -> 3086 ; b: 0.5 -> 0 (none) ; c: 1.5 -> 2 ; d negative -> none ; e: 2.5 -> 2
    assert led.balance(a) == 12345 + 3086 and led.balance(b) == 2 and led.balance(c) == 8 and led.balance(d) == -50
    assert led.balance(e) == 12
    assert len(txs) == 3 and txs == sorted(txs)
    assert led.history(a)[-1]["tx"] == txs[0] and led.history(c)[-1]["tx"] == txs[1] and led.history(e)[-1]["tx"] == txs[2]
    last = led.history(a)[-1]
    assert last["kind"] == "interest" and last["amount"] == 3086 and last["memo"] == "interest"
    with pytest.raises(E().ValidationError):
        led.apply_interest(-1)


def test_s7_interest_small_rate():
    led = L()
    a = led.open_account("A", "EUR")
    led.deposit(a, 12345)
    led.apply_interest(25, "q1")  # 30.8625 -> 31
    assert led.balance(a) == 12376
    assert led.history(a)[-1]["memo"] == "q1"


# ---------------------------------------------------------------- stage 8
def test_s8_statement():
    ck = Clock(datetime(2026, 2, 27, 9, tzinfo=timezone.utc))
    led = L(clock=ck)
    a = led.open_account("Alice", "EUR"); b = led.open_account("Bob", "EUR")
    led.deposit(a, 1000)
    ck.now = datetime(2026, 3, 2, 10, tzinfo=timezone.utc)
    led.deposit(a, 500, "refund")
    ck.now = datetime(2026, 3, 15, 11, tzinfo=timezone.utc)
    led.transfer(a, b, 1234)
    ck.now = datetime(2026, 4, 1, 0, tzinfo=timezone.utc)
    led.deposit(a, 1)
    s = led.monthly_statement(a, 2026, 3)
    assert s == "\n".join([
        f"Statement {a} (ALICE) 2026-03",
        "Opening balance: EUR 10.00",
        "2026-03-02 TX00000002 deposit +EUR 5.00 -> EUR 15.00 refund",
        f"2026-03-15 TX00000003 transfer_out -EUR 12.34 -> EUR 2.66 xfer {a}>{b}",
        "Closing balance: EUR 2.66",
    ])
    empty = led.monthly_statement(b, 2026, 1)
    assert empty == f"Statement {b} (BOB) 2026-01\nOpening balance: EUR 0.00\nClosing balance: EUR 0.00"
    with pytest.raises(E().ValidationError):
        led.monthly_statement(a, 2026, 13)


# ---------------------------------------------------------------- stage 9
def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    r = subprocess.run([sys.executable, "-m", "vxledger", str(tmp_path / "db.json"), *args], capture_output=True,
                       text=True, env=env, timeout=60)
    return r


def test_s9_cli_flow(tmp_path):
    r = cli(tmp_path, "open", "Alice", "--currency", "EUR"); assert r.returncode == 0, r.stderr
    a = r.stdout.strip()
    r = cli(tmp_path, "open", "Bob", "--currency", "EUR"); b = r.stdout.strip()
    r = cli(tmp_path, "deposit", a, "1,000.50", "--memo", "pay"); assert r.returncode == 0 and r.stdout.strip() == "TX00000001"
    r = cli(tmp_path, "transfer", a, b, "0.50"); assert r.stdout.strip() == "TX00000002"
    r = cli(tmp_path, "withdraw", b, "0.25"); assert r.stdout.strip() == "TX00000003"
    r = cli(tmp_path, "balance", a); assert r.returncode == 0 and r.stdout.strip() == "EUR 1,000.00"
    r = cli(tmp_path, "balance", b); assert r.stdout.strip() == "EUR 0.25"
    from vxledger.ledger import Ledger
    assert Ledger.load(str(tmp_path / "db.json")).balance(a) == 100000


def test_s9_cli_statement(tmp_path):
    a = cli(tmp_path, "open", "Alice", "--currency", "EUR").stdout.strip()
    cli(tmp_path, "deposit", a, "5")
    ym = datetime.now(timezone.utc).strftime("%Y-%m")
    r = cli(tmp_path, "statement", a, ym)
    assert r.returncode == 0
    lines = r.stdout.strip().splitlines()
    assert lines[0] == f"Statement {a} (ALICE) {ym}" and lines[-1] == "Closing balance: EUR 5.00"


def test_s9_cli_error_message(tmp_path):
    a = cli(tmp_path, "open", "Alice", "--currency", "EUR").stdout.strip()
    r = cli(tmp_path, "withdraw", a, "5")
    assert r.returncode != 0
    from vxledger.errors import InsufficientFundsError
    # log records may also reach stderr (logging's last-resort handler); one line must be the error line
    assert any(l.startswith(f"error {InsufficientFundsError.code}:") for l in r.stderr.splitlines())


# ---------------------------------------------------------------- stage 10
def test_s10_holds():
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 1000); led.set_overdraft(a, 100)
    h1 = led.place_hold(a, 600, "card")
    h2 = led.place_hold(a, 400, "hotel")
    assert (h1, h2) == ("HOLD-0001", "HOLD-0002")
    assert led.available(a) == 100 and led.balance(a) == 1000
    with pytest.raises(E().InsufficientFundsError):
        led.place_hold(a, 101, "x")
    with pytest.raises(E().InsufficientFundsError):
        led.withdraw(a, 101)
    with pytest.raises(E().InsufficientFundsError):
        led.transfer(a, b, 101)
    led.withdraw(a, 100)
    led.release_hold(h1)
    assert led.available(a) == 600
    with pytest.raises(E().UnknownHoldError):
        led.release_hold(h1)
    with pytest.raises(E().UnknownHoldError):
        led.release_hold("HOLD-0099")
    with pytest.raises(E().ValidationError):
        led.place_hold(a, 0, "x")
    assert len(led.history(a)) == 2  # holds are not transactions
    assert led.deposit(b, 1) == "TX00000003"


def test_s10_holds_persist(tmp_path):
    led = L()
    a = led.open_account("A", "EUR")
    led.deposit(a, 1000)
    h = led.place_hold(a, 300, "x")
    p = str(tmp_path / "db.json"); led.save(p)
    from vxledger.ledger import Ledger
    led2 = Ledger.load(p)
    assert led2.available(a) == 700
    led2.release_hold(h)
    assert led2.available(a) == 1000
    assert led2.place_hold(a, 1, "y") == "HOLD-0002"


# ---------------------------------------------------------------- stage 11
class Cap(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG); self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def cap():
    h = Cap()
    root = logging.getLogger()
    old = root.level
    root.addHandler(h); root.setLevel(logging.DEBUG)
    yield h
    root.removeHandler(h); root.setLevel(old)


def test_s11_info_per_mutation(cap):
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    cap.records.clear()
    tx = led.deposit(a, 1000)
    tx2 = led.transfer(a, b, 10)
    h = led.place_hold(a, 5, "r")
    led.release_hold(h)
    led.set_overdraft(b, 10)
    infos = [r for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 5
    msgs = [r.getMessage() for r in infos]
    assert tx in msgs[0] and tx2 in msgs[1] and h in msgs[2] and h in msgs[3] and b in msgs[4]


def test_s11_open_logged(cap):
    led = L()
    a = led.open_account("A", "EUR")
    infos = [r for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 1 and a in infos[0].getMessage()


def test_s11_warning_on_error(cap):
    led = L()
    a = led.open_account("A", "EUR")
    cap.records.clear()
    with pytest.raises(E().InsufficientFundsError):
        led.withdraw(a, 1)
    warns = [r for r in cap.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and str(E().InsufficientFundsError.code) in warns[0].getMessage()
    assert not [r for r in cap.records if r.levelno == logging.INFO]


def test_s11_interest_and_import_logged(cap, tmp_path):
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 10000); led.deposit(b, 10000)
    cap.records.clear()
    txs = led.apply_interest(100)
    infos = [r.getMessage() for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 2 and txs[0] in infos[0] and txs[1] in infos[1]
    cap.records.clear()
    p = _csv(tmp_path, [["2026-01-05", a, "1", "x"], ["2026-01-05", a, "zz", "y"], ["2026-01-05", b, "2", "z"]])
    led.import_csv(p)
    infos = [r for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 2


# ---------------------------------------------------------------- stage 12
def test_s12_search():
    led = L(clock=Clock())
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    t1 = led.deposit(a, 100, "Coffee beans")
    led.deposit(b, 100, "rent")
    t3 = led.transfer(a, b, 10, "coffee split")
    res = led.search("COFFEE")
    assert [(r["tx"], r["account"]) for r in res] == sorted([(t1, a), (t3, a), (t3, b)])
    assert res[0]["memo"] == "Coffee beans" and res[0]["kind"] == "deposit" and res[0]["balance_after"] == 100
    assert led.search("nothing") == []
    with pytest.raises(E().ValidationError):
        led.search("")


def test_s12_error_classes():
    er = E()
    names = ["AmountError", "ValidationError", "UnknownAccountError", "InsufficientFundsError",
             "CurrencyMismatchError", "LoadError", "UnknownHoldError"]
    for n in names:
        assert issubclass(getattr(er, n), Exception)
