"""Hidden tests for the 20 team decisions of long-horizon lh2 (roomly), one test per decision, latest value where a
decision was superseded (D03 grid 5, D06 gap 15, D08 closing 22:00)."""
import csv, json, logging, os, subprocess, sys
from datetime import datetime, timezone
import pytest

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def E():
    from roomly import errors
    return errors


def S(clock=None):
    from roomly.scheduler import Scheduler
    return Scheduler(clock=clock or (lambda: T0))


def room(s, name="Alpha", cap=10):
    return s.add_room(name, cap)


def cli(tmp_path, *args, db="db.json"):
    env = dict(os.environ, PYTHONPATH=os.getcwd())
    return subprocess.run([sys.executable, "-m", "roomly", str(tmp_path / db), *args], capture_output=True,
                          text=True, env=env, timeout=60)


def _csv(tmp_path, rows, name="in.csv"):
    p = tmp_path / name
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room", "start", "duration", "title", "attendees"])
        w.writerows(rows)
    return str(p)


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


def test_d01_name_max_30():
    s = S()
    s.add_room("N" * 30, 5)
    s.add_room("  " + "M" * 30 + "  ", 5)
    with pytest.raises(E().ValidationError):
        s.add_room("X" * 31, 5)


def test_d02_capacity_max_200():
    s = S()
    s.add_room("Hall", 200)
    with pytest.raises(E().ValidationError):
        s.add_room("Atrium", 201)


def test_d03_grid_latest_5():
    s = S(); a = room(s)
    x = s.book(a, "2026-03-02T09:05", 30, "standup")
    s.book(a, "2026-03-02T10:20", 30, "sync")
    for st in ("2026-03-02T12:07", "2026-03-02T12:01", "2026-03-02T12:58"):
        with pytest.raises(E().ValidationError):
            s.book(a, st, 30, "off grid")
    s.move(x, "2026-03-02T14:35")


def test_d04_max_240():
    s = S(); a = room(s)
    s.book(a, "2026-03-02T08:00", 240, "long")
    with pytest.raises(E().ValidationError):
        s.book(a, "2026-03-03T08:00", 241, "too long")


def test_d05_title():
    s = S(); a = room(s)
    ids = [s.book(a, f"2026-03-0{d}T09:00", 30, t) for d, t in
           ((2, "  hello  "), (3, "  " + "abcdefghij" * 10), (4, ""), (5, "   "))]
    assert [s.booking(i)["title"] for i in ids] == ["hello", ("abcdefghij" * 10)[:50], "(no title)", "(no title)"]


def test_d06_gap_latest_15(tmp_path):
    # all starts on the 15-minute grid; the gaps come from durations (50 min -> gap 10, 45 min -> gap 15)
    s = S(); a = room(s, "Alpha"); b = room(s, "Beta"); c = room(s, "Gamma"); d = room(s, "Delta")
    s.book(a, "2026-03-02T09:00", 50, "base")
    with pytest.raises(E().ConflictError):
        s.book(a, "2026-03-02T10:00", 30, "10 min gap after")
    with pytest.raises(E().ConflictError):
        s.book(a, "2026-03-02T08:00", 50, "10 min gap before")
    s.book(a, "2026-03-02T08:00", 45, "15 min gap before")
    s.book(d, "2026-03-02T09:00", 45, "base 2")
    s.book(d, "2026-03-02T10:00", 30, "15 min gap after")
    s.book(b, "2026-03-02T13:00", 50, "m1"); y = s.book(b, "2026-03-02T16:00", 30, "m2")
    with pytest.raises(E().ConflictError):
        s.move(y, "2026-03-02T14:00")
    s.move(y, "2026-03-02T14:15")
    r = s.import_csv(_csv(tmp_path, [[c, "2026-03-03T09:00", "50", "i1", "1"], [c, "2026-03-03T10:00", "30", "i2", "1"]]))
    assert [list(e) for e in r["errors"]] == [[3, E().ConflictError.slug]]
    s.book(c, "2026-03-16T10:00", 30, "blocker")
    with pytest.raises(E().ConflictError):
        s.book_series(c, "2026-03-02T09:00", 3, 7, 50, "weekly")


def test_d07_no_cancel_after_start():
    s = S(lambda: datetime(2026, 3, 2, 9, 0, tzinfo=timezone.utc)); a = room(s)
    past = s.book(a, "2026-03-02T07:00", 30, "past")
    now = s.book(a, "2026-03-02T09:00", 30, "now")
    later = s.book(a, "2026-03-02T11:00", 30, "later")
    for i in (past, now):
        with pytest.raises(E().ValidationError):
            s.cancel(i)
    s.cancel(later)
    assert s.booking(now)["title"] == "now"


def test_d08_hours_latest_22():
    s = S(); a = room(s)
    s.book(a, "2026-03-02T21:00", 60, "evening class")
    s.book(a, "2026-03-03T07:00", 30, "early")
    for st, d in (("2026-03-04T21:30", 60), ("2026-03-04T06:45", 30), ("2026-03-04T22:00", 15)):
        with pytest.raises(E().ValidationError):
            s.book(a, st, d, "outside")


def test_d09_compact_json(tmp_path):
    s = S(); a = room(s); s.book(a, "2026-03-02T09:00", 30, "x")
    p = tmp_path / "db.json"; s.save(str(p))
    text = p.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), sort_keys=True, separators=(",", ":")) + "\n"


def test_d10_load_missing_is_empty(tmp_path):
    from roomly.scheduler import Scheduler
    s = Scheduler.load(str(tmp_path / "nope.json"))
    assert s.rooms() == []
    s.add_room("Alpha", 3)


def test_d11_csv_all_or_nothing(tmp_path):
    s = S(); a = room(s)
    r = s.import_csv(_csv(tmp_path, [[a, "2026-03-02T09:00", "60", "good", "1"], [a, "2026-03-02T12:00", "60", "bad", "99"],
                                     [a, "2026-03-02T15:00", "60", "good too", "1"]]))
    assert r["imported"] == 0 and [list(e) for e in r["errors"]] == [[3, E().CapacityError.slug]]
    assert s.day_bookings(a, "2026-03-02") == []
    s.book(a, "2026-03-02T09:00", 60, "slot is free")


def test_d12_csv_swiss_start(tmp_path):
    s = S(); a = room(s)
    r = s.import_csv(_csv(tmp_path, [[a, "02.03.2026 09:00", "60", "swiss", "1"]], "a.csv"))
    assert r["imported"] == 1 and s.day_bookings(a, "2026-03-02")[0]["start"] == "2026-03-02T09:00"
    r = s.import_csv(_csv(tmp_path, [[a, "03/02/2026 11:00", "60", "slashes", "1"]], "b.csv"))
    assert [list(e) for e in r["errors"]] == [[2, E().ValidationError.slug]]


def test_d13_series_max_12():
    s = S(); a = room(s)
    assert len(s.book_series(a, "2026-03-02T09:00", 12, 7, 30, "ok")) == 12
    with pytest.raises(E().ValidationError):
        s.book_series(a, "2026-03-02T12:00", 13, 7, 30, "too many")


def test_d14_series_skip_weekends():
    s = S(); a = room(s)
    ids = s.book_series(a, "2026-03-05T09:00", 5, 1, 30, "daily")   # Thu Fri [Sat Sun] Mon
    assert [s.booking(i)["start"][:10] for i in ids] == ["2026-03-05", "2026-03-06", "2026-03-09"]


def test_d15_report_sorted_by_name():
    s = S()
    ids = [s.add_room(n, 5) for n in ("beta", "Alpha", "Gamma")]
    for i in ids:
        s.book(i, "2026-03-02T09:00", 30, "x")
    heads = [l.split(":")[0] for l in s.day_report("2026-03-02").splitlines()[1:] if not l.startswith((" ", "Total"))]
    assert heads == [f"{ids[1]} Alpha", f"{ids[0]} beta", f"{ids[2]} Gamma"]


def test_d16_report_omits_empty_rooms():
    s = S(); a = room(s, "Alpha"); b = room(s, "Beta")
    s.book(b, "2026-03-02T09:00", 30, "x")
    s.book(a, "2026-03-03T09:00", 30, "other day")
    lines = s.day_report("2026-03-02").splitlines()
    assert not any(l.startswith(a) for l in lines) and any(l.startswith(b) for l in lines)
    assert s.day_report("2026-03-09").splitlines() == ["Rooms on 2026-03-09", "Total: 0m"]


def test_d17_cli_bad_db_exit6(tmp_path):
    (tmp_path / "bad.json").write_text("{not json")
    assert cli(tmp_path, "report", "2030-01-07", db="bad.json").returncode == 6
    (tmp_path / "v.json").write_text(json.dumps({"version": 1}))
    assert cli(tmp_path, "add-room", "X", "3", db="v.json").returncode == 6


def test_d18_cli_rooms(tmp_path):
    a = cli(tmp_path, "add-room", "Alpha", "10").stdout.strip()
    b = cli(tmp_path, "add-room", "Beta", "4").stdout.strip()
    r = cli(tmp_path, "rooms")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().splitlines() == [f"{a} Alpha 10", f"{b} Beta 4"]


def test_d19_rev(tmp_path):
    from roomly.scheduler import Scheduler
    s = S(); a = room(s)
    x = s.book(a, "2026-03-02T09:00", 30, "x")
    assert s.booking(x)["rev"] == 1
    s.move(x, "2026-03-02T11:00"); s.move(x, "2026-03-02T13:00")
    assert s.booking(x)["rev"] == 3
    p = str(tmp_path / "db.json"); s.save(p)
    assert Scheduler.load(p, clock=lambda: T0).booking(x)["rev"] == 3


def test_d20_info_room_name(cap):
    s = S(); a = s.add_room("Blue Lagoon", 8)
    cap.records.clear()
    x = s.book(a, "2026-03-02T09:00", 30, "x")
    s.move(x, "2026-03-02T11:00")
    s.cancel(x)
    infos = [r.getMessage() for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 3 and all("Blue Lagoon" in m for m in infos)
