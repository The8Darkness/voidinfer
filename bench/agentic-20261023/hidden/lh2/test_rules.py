"""Hidden tests for the 20 team decisions of long-horizon v2 (one test per decision, latest value where a
decision was superseded). Tests use explicit currencies and returned ids where the decision is not about them."""
import csv, json, logging, os, subprocess, sys
from datetime import datetime, timezone
import pytest


def E():
    from vxledger import errors
    return errors


def L(**kw):
    from vxledger.ledger import Ledger
    return Ledger(**kw)


def cli(tmp_path, *args, db="db.json"):
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    return subprocess.run([sys.executable, "-m", "vxledger", str(tmp_path / db), *args], capture_output=True,
                          text=True, env=env, timeout=60)


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


def _csv(tmp_path, rows):
    p = tmp_path / "in.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "account", "amount", "memo"])
        w.writerows(rows)
    return str(p)


def test_d01_name_max_40():
    led = L()
    led.open_account("N" * 40, "EUR")
    led.open_account("  " + "M" * 40 + "  ", "EUR")
    with pytest.raises(E().ValidationError):
        led.open_account("X" * 41, "EUR")


def test_d02_memo_stripped_truncated():
    led = L()
    a = led.open_account("A", "EUR")
    led.deposit(a, 100, "  hello  ")
    led.deposit(a, 100, "  " + "abcdefghij" * 10)
    h = led.history(a)
    assert h[0]["memo"] == "hello"
    assert h[1]["memo"] == ("abcdefghij" * 10)[:60]


def test_d03_error_codes_latest():
    er = E()
    got = {n: getattr(er, n).code for n in ("AmountError", "ValidationError", "UnknownAccountError",
                                             "InsufficientFundsError", "CurrencyMismatchError", "LoadError",
                                             "UnknownHoldError")}
    assert got == {"AmountError": 4701, "ValidationError": 4702, "UnknownAccountError": 4703,
                   "InsufficientFundsError": 4740, "CurrencyMismatchError": 4705, "LoadError": 4706,
                   "UnknownHoldError": 4770}


def test_d04_save_format(tmp_path):
    led = L()
    a = led.open_account("A", "EUR"); led.deposit(a, 5, "x")
    p = tmp_path / "db.json"; led.save(str(p))
    text = p.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"


def test_d05_no_interest_for_jpy():
    led = L()
    y = led.open_account("Yen", "JPY"); c = led.open_account("Franc", "CHF")
    led.deposit(y, 100000); led.deposit(c, 100000)
    txs = led.apply_interest(100)
    assert led.balance(y) == 100000 and led.balance(c) == 101000 and len(txs) == 1


def test_d06_movement_limit_latest(tmp_path):
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 300000000)          # above the superseded 1,000,000.00 cap
    led.deposit(a, 500000000)          # exactly the new limit
    with pytest.raises(E().ValidationError):
        led.deposit(a, 500000001)
    led.set_overdraft(a, 10 ** 12)
    with pytest.raises(E().ValidationError):
        led.withdraw(a, 500000001)
    with pytest.raises(E().ValidationError):
        led.transfer(a, b, 500000001)
    r = led.import_csv(_csv(tmp_path, [["2026-01-05", b, "5,000,000.01", "big"], ["2026-01-05", b, "5,000,000.00", "ok"]]))
    assert r["imported"] == 1 and [list(x) for x in r["errors"]] == [[2, E().ValidationError.code]]


def test_d07_csv_swiss_dates(tmp_path):
    led = L()
    a = led.open_account("A", "EUR")
    r = led.import_csv(_csv(tmp_path, [["05.01.2026", a, "1", "swiss"], ["01/05/2026", a, "1", "slash"]]))
    assert r["imported"] == 1 and r["errors"][0][0] == 3
    assert led.history(a)[0]["timestamp"].startswith("2026-01-05T00:00:00")


def test_d08_default_transfer_memo_latest():
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    led.deposit(a, 100)
    led.transfer(a, b, 10)
    led.transfer(a, b, 10, "   ")
    assert led.history(a)[1]["memo"] == f"xfer {a}>{b}" and led.history(b)[0]["memo"] == f"xfer {a}>{b}"
    assert led.history(a)[2]["memo"] == f"xfer {a}>{b}"


def test_d09_statement_upper():
    led = L(clock=lambda: datetime(2026, 3, 2, tzinfo=timezone.utc))
    a = led.open_account("Alice Smith", "EUR")
    assert led.monthly_statement(a, 2026, 3).splitlines()[0] == f"Statement {a} (ALICE SMITH) 2026-03"


def test_d10_max_3_holds():
    led = L()
    a = led.open_account("A", "EUR"); led.deposit(a, 1000)
    hs = [led.place_hold(a, 10, f"r{i}") for i in range(3)]
    with pytest.raises(E().ValidationError):
        led.place_hold(a, 10, "r4")
    led.release_hold(hs[0])
    led.place_hold(a, 10, "r4")


def test_d11_cli_bad_db_exit4(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    r = cli(tmp_path, "balance", "ACC-000001", db="bad.json")
    assert r.returncode == 4
    (tmp_path / "v.json").write_text(json.dumps({"version": 3}))
    assert cli(tmp_path, "open", "X", db="v.json").returncode == 4


def test_d12_warning_op(cap):
    led = L()
    a = led.open_account("A", "EUR")
    cap.records.clear()
    with pytest.raises(E().InsufficientFundsError):
        led.withdraw(a, 1)
    with pytest.raises(E().UnknownAccountError):
        led.balance("ACC-999999")
    w = [r.getMessage() for r in cap.records if r.levelno == logging.WARNING]
    assert len(w) == 2 and "op=withdraw" in w[0] and "op=balance" in w[1]


def test_d13_hold_reason_required():
    led = L()
    a = led.open_account("A", "EUR"); led.deposit(a, 100)
    for reason in ("", "   "):
        with pytest.raises(E().ValidationError):
            led.place_hold(a, 10, reason)
    assert led.available(a) == 100


def test_d14_cli_open_overdraft(tmp_path):
    r = cli(tmp_path, "open", "Alice", "--currency", "EUR", "--overdraft", "100")
    assert r.returncode == 0, r.stderr
    a = r.stdout.strip()
    assert cli(tmp_path, "withdraw", a, "50").returncode == 0
    assert cli(tmp_path, "balance", a).stdout.strip() == "EUR -50.00"
    assert cli(tmp_path, "withdraw", a, "50.01").returncode == 3


def test_d15_search_min_2():
    led = L()
    a = led.open_account("A", "EUR"); led.deposit(a, 5, "coffee")
    for q in ("", " ", " c ", "x"):
        with pytest.raises(E().ValidationError):
            led.search(q)
    assert [r["memo"] for r in led.search("  co  ")] == ["coffee"]


def test_d16_cli_history(tmp_path):
    a = cli(tmp_path, "open", "Alice", "--currency", "EUR").stdout.strip()
    cli(tmp_path, "deposit", a, "5")
    cli(tmp_path, "withdraw", a, "1.50")
    r = cli(tmp_path, "history", a)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines() == ["TX00000001 deposit 500 500", "TX00000002 withdrawal -150 350"]


def test_d17_info_amounts(cap):
    led = L()
    a = led.open_account("A", "EUR"); b = led.open_account("B", "EUR")
    cap.records.clear()
    led.deposit(a, 123456); led.withdraw(a, 1234); led.transfer(a, b, 5)
    infos = [r.getMessage() for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 3
    assert "EUR 1,234.56" in infos[0] and "EUR 12.34" in infos[1] and "EUR 0.05" in infos[2]


def test_d18_search_account_name():
    led = L()
    a = led.open_account("Alice", "EUR"); led.deposit(a, 5, "coffee")
    r = led.search("coffee")
    assert r[0]["account_name"] == "Alice" and r[0]["account"] == a


def test_d19_hold_id_format():
    led = L()
    a = led.open_account("A", "EUR"); led.deposit(a, 100)
    assert led.place_hold(a, 1, "x") == "HOLD-0001"
    assert led.place_hold(a, 1, "y") == "HOLD-0002"


def test_d20_search_max_25():
    led = L()
    a = led.open_account("A", "EUR")
    txs = [led.deposit(a, 1, f"coffee {i}") for i in range(30)]
    r = led.search("coffee")
    assert len(r) == 25 and [x["tx"] for x in r] == txs[:25]
