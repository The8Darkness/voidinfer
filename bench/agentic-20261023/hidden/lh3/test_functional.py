"""Hidden functional tests for roomly (long-horizon v3), one group per stage.
Inputs are chosen so that no team decision changes the expected outcome: explicit durations, starts on the
15-minute grid between 08:00 and 19:00, gaps of at least 30 minutes, titles without padding, a clock before all
bookings, rooms whose name order equals their id order. Decisions are scored separately in test_rules.py."""
import csv, json, logging, os, re, subprocess, sys
from datetime import datetime, timezone
import pytest

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def E():
    from roomly import errors
    return errors


def TU():
    from roomly import timeutil
    return timeutil


def S(**kw):
    from roomly.scheduler import Scheduler
    kw.setdefault("clock", lambda: T0)
    try:
        return Scheduler(**kw)
    except TypeError:
        return Scheduler()


def S0():
    from roomly.scheduler import Scheduler
    return Scheduler()


def two(s=None):
    s = s or S()
    a = s.add_room("Alpha", 10)
    b = s.add_room("Beta", 4)
    return s, a, b


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


# ---------------------------------------------------------------- stage 1
def test_s1_errors_exist():
    er = E()
    for n in ("ValidationError", "UnknownRoomError", "UnknownBookingError", "ConflictError", "CapacityError", "LoadError"):
        assert issubclass(getattr(er, n), Exception)
    assert len({getattr(er, n) for n in ("ValidationError", "UnknownRoomError", "UnknownBookingError",
                                         "ConflictError", "CapacityError", "LoadError")}) == 6


def test_s1_parse_duration_ok():
    p = TU().parse_duration
    assert [p(x) for x in ("45", "45m", "2h", "1h30m", "1h5m", "1h05m", " 90 ", "10h", "0h15m")] == \
        [45, 45, 120, 90, 65, 65, 90, 600, 15]


def test_s1_parse_duration_bad():
    for x in ("", "0", "0m", "1h60m", "1.5h", "abc", "-5", "1h 30m", "h", "30mm", "2H", "1h300m"):
        with pytest.raises(E().ValidationError):
            TU().parse_duration(x)


def test_s1_format_duration():
    f = TU().format_duration
    assert [f(x) for x in (0, 45, 60, 90, 125, 600)] == ["0m", "45m", "1h", "1h30m", "2h05m", "10h"]
    with pytest.raises(E().ValidationError):
        f(-1)


def test_s1_parse_start():
    d = TU().parse_start(" 2026-03-02T09:15 ")
    assert d == datetime(2026, 3, 2, 9, 15, tzinfo=timezone.utc) and d.utcoffset().total_seconds() == 0


def test_s1_parse_start_bad():
    for x in ("2026-03-02 09:00", "2026-03-02T9:00", "2026-02-30T09:00", "2026-03-02T09:00:00", "", "2026-03-02T24:00"):
        with pytest.raises(E().ValidationError):
            TU().parse_start(x)


def test_s1_format_start_roundtrip():
    t = TU()
    assert t.format_start(t.parse_start("2026-12-31T23:45")) == "2026-12-31T23:45"
    assert t.format_start(datetime(2026, 3, 2, 9, 0, 59, tzinfo=timezone.utc)) == "2026-03-02T09:00"


def test_s1_file_helper_writes(tmp_path):
    from roomly import fileio
    p = tmp_path / "x.txt"
    fileio.safe_write(str(p), "hello\n")
    fileio.safe_write(str(p), "again\n")
    assert p.read_text() == "again\n" and os.listdir(tmp_path) == ["x.txt"]


# ---------------------------------------------------------------- stage 2
def test_s2_import_and_empty():
    from roomly import Scheduler
    assert Scheduler().rooms() == []


def test_s2_add_room_and_get():
    s = S()
    r = s.add_room("  Alpha  ", 10)
    assert s.room(r) == {"id": r, "name": "Alpha", "capacity": 10}


def test_s2_rooms_in_order():
    s, a, b = two()
    c = s.add_room("Gamma", 6)
    assert [x["id"] for x in s.rooms()] == [a, b, c] and [x["name"] for x in s.rooms()] == ["Alpha", "Beta", "Gamma"]


def test_s2_name_validation():
    s = S()
    s.add_room("Alpha", 10)
    for n in ("", "   ", "alpha", " ALPHA "):
        with pytest.raises(E().ValidationError):
            s.add_room(n, 5)


def test_s2_capacity_validation():
    s = S()
    for c in (0, -1, True, "5", 2.0):
        with pytest.raises(E().ValidationError):
            s.add_room(f"Room {c!r}", c)
    assert s.rooms() == []


def test_s2_unknown_room():
    with pytest.raises(E().UnknownRoomError):
        S().room("R999")


# ---------------------------------------------------------------- stage 3
def test_s3_book_and_get():
    s, a, _ = two()
    bid = s.book(a, "2026-03-02T09:00", 90, "Sprint review", 6)
    b = s.booking(bid)
    assert {k: b[k] for k in ("id", "room", "start", "end", "title", "attendees")} == \
        {"id": bid, "room": a, "start": "2026-03-02T09:00", "end": "2026-03-02T10:30", "title": "Sprint review",
         "attendees": 6}


def test_s3_booking_ids_shared_sequence():
    s, a, b = two()
    x = s.book(a, "2026-03-02T09:00", 30, "x")
    y = s.book(b, "2026-03-02T09:00", 30, "y")
    z = s.book(a, "2026-03-02T11:00", 30, "z")
    assert len({x, y, z}) == 3 and x < y < z


def test_s3_overlap_conflict():
    s, a, b = two()
    s.book(a, "2026-03-02T10:00", 60, "first")
    for st, d in (("2026-03-02T10:00", 60), ("2026-03-02T09:30", 60), ("2026-03-02T10:30", 15), ("2026-03-02T09:00", 180)):
        with pytest.raises(E().ConflictError):
            s.book(a, st, d, "clash")
    s.book(b, "2026-03-02T10:00", 60, "other room")


def test_s3_non_overlapping_ok():
    s, a, _ = two()
    s.book(a, "2026-03-02T10:00", 60, "first")
    s.book(a, "2026-03-02T11:30", 60, "later")
    s.book(a, "2026-03-02T08:00", 60, "earlier")
    s.book(a, "2026-03-03T10:00", 60, "next day")


def test_s3_duration_validation():
    s, a, _ = two()
    for d in (0, -30, True, "60"):
        with pytest.raises(E().ValidationError):
            s.book(a, "2026-03-02T09:00", d, "bad")


def test_s3_bad_start():
    s, a, _ = two()
    with pytest.raises(E().ValidationError):
        s.book(a, "2026-03-02 09:00", 30, "bad")


def test_s3_attendees():
    s, a, b = two()
    with pytest.raises(E().CapacityError):
        s.book(b, "2026-03-02T09:00", 30, "big", 5)
    for n in (0, -1):
        with pytest.raises(E().ValidationError):
            s.book(b, "2026-03-02T09:00", 30, "none", n)
    s.book(b, "2026-03-02T09:00", 30, "full", 4)


def test_s3_unknowns():
    s, a, _ = two()
    with pytest.raises(E().UnknownRoomError):
        s.book("R999", "2026-03-02T09:00", 30, "x")
    with pytest.raises(E().UnknownBookingError):
        s.booking("BK-99999")


def test_s3_failed_book_changes_nothing():
    s, a, _ = two()
    first = s.book(a, "2026-03-02T10:00", 60, "first")
    with pytest.raises(E().ConflictError):
        s.book(a, "2026-03-02T10:30", 30, "clash")
    nxt = s.book(a, "2026-03-02T14:00", 30, "next")
    num = lambda i: int(re.sub(r"\D", "", i))
    assert num(nxt) == num(first) + 1


# ---------------------------------------------------------------- stage 4
def test_s4_cancel_frees_slot():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T10:00", 60, "first")
    s.cancel(x)
    s.book(a, "2026-03-02T10:00", 60, "again")
    with pytest.raises(E().UnknownBookingError):
        s.booking(x)


def test_s4_cancel_twice_or_unknown():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T10:00", 60, "first")
    s.cancel(x)
    with pytest.raises(E().UnknownBookingError):
        s.cancel(x)
    with pytest.raises(E().UnknownBookingError):
        s.cancel("BK-99999")


def test_s4_day_bookings_sorted():
    s, a, b = two()
    y = s.book(a, "2026-03-02T14:00", 30, "late")
    x = s.book(a, "2026-03-02T09:00", 30, "early")
    s.book(a, "2026-03-03T09:00", 30, "tomorrow")
    s.book(b, "2026-03-02T09:00", 30, "other room")
    assert [d["id"] for d in s.day_bookings(a, "2026-03-02")] == [x, y]
    assert s.day_bookings(a, "2026-03-04") == []


def test_s4_day_bookings_excludes_cancelled():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 30, "x")
    y = s.book(a, "2026-03-02T11:00", 30, "y")
    s.cancel(x)
    assert [d["id"] for d in s.day_bookings(a, "2026-03-02")] == [y]


def test_s4_day_validation():
    s, a, _ = two()
    for d in ("2026-3-2", "02.03.2026", "2026-02-30", ""):
        with pytest.raises(E().ValidationError):
            s.day_bookings(a, d)
    with pytest.raises(E().UnknownRoomError):
        s.day_bookings("R999", "2026-03-02")


def test_s4_clock_used_for_cancel():
    s, a, _ = two(S(clock=lambda: datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)))
    x = s.book(a, "2026-03-02T09:00", 30, "x")
    s.cancel(x)


# ---------------------------------------------------------------- stage 5
def test_s5_roundtrip(tmp_path):
    from roomly.scheduler import Scheduler
    s, a, b = two()
    x = s.book(a, "2026-03-02T09:00", 90, "Planning", 3)
    s.book(b, "2026-03-02T10:00", 30, "Sync", 2)
    p = str(tmp_path / "db.json")
    s.save(p)
    t = Scheduler.load(p, clock=lambda: T0)
    assert t.rooms() == s.rooms()
    assert t.booking(x)["start"] == "2026-03-02T09:00" and t.booking(x)["end"] == "2026-03-02T10:30"
    assert t.booking(x)["title"] == "Planning" and t.booking(x)["attendees"] == 3
    with pytest.raises(E().ConflictError):
        t.book(a, "2026-03-02T09:30", 30, "clash")


def test_s5_version_field(tmp_path):
    s, a, _ = two()
    p = tmp_path / "db.json"
    s.save(str(p))
    assert json.loads(p.read_text())["version"] == 3


def test_s5_sequences_continue(tmp_path):
    from roomly.scheduler import Scheduler
    s, a, b = two()
    s.book(a, "2026-03-02T09:00", 30, "x")
    last = s.book(a, "2026-03-02T11:00", 30, "y")
    s.cancel(last)
    p = str(tmp_path / "db.json")
    s.save(p)
    t = Scheduler.load(p, clock=lambda: T0)
    nb = t.book(a, "2026-03-02T15:00", 30, "z")
    assert nb != last and nb > last
    assert t.add_room("Gamma", 3) > b


def test_s5_load_errors(tmp_path):
    from roomly.scheduler import Scheduler
    (tmp_path / "bad.json").write_text("{not json")
    (tmp_path / "v.json").write_text(json.dumps({"version": 2}))
    (tmp_path / "nov.json").write_text(json.dumps({"rooms": []}))
    for n in ("bad.json", "v.json", "nov.json"):
        with pytest.raises(E().LoadError):
            Scheduler.load(str(tmp_path / n))


# ---------------------------------------------------------------- stage 6
def test_s6_import_ok(tmp_path):
    s, a, b = two()
    r = s.import_csv(_csv(tmp_path, [[a, "2026-03-02T09:00", "1h", "Standup", "3"],
                                     [b, "2026-03-02T09:00", "30m", "Call", ""],
                                     [a, "2026-03-02T11:00", "90", "Review", "5"]]))
    assert r["imported"] == 3 and r["errors"] == []
    bs = s.day_bookings(a, "2026-03-02")
    assert [(x["start"], x["end"], x["title"], x["attendees"]) for x in bs] == \
        [("2026-03-02T09:00", "2026-03-02T10:00", "Standup", 3), ("2026-03-02T11:00", "2026-03-02T12:30", "Review", 5)]
    assert s.day_bookings(b, "2026-03-02")[0]["attendees"] == 1


def test_s6_import_errors_listed(tmp_path):
    s, a, b = two()
    er = E()
    r = s.import_csv(_csv(tmp_path, [[a, "2026-03-02T09:00", "1h", "ok", "2"],
                                     ["R999", "2026-03-02T09:00", "1h", "unknown room", "1"],
                                     [a, "2026-03-02T09:30", "1h", "clash with row 2", "1"],
                                     [b, "2026-03-02T09:00", "1h", "too many", "9"],
                                     [b, "yesterday", "1h", "bad start", "1"],
                                     [b, "2026-03-02T12:00", "1.5h", "bad duration", "1"],
                                     [b, "2026-03-02T14:00", "1h", "bad attendees", "x"]]))
    assert [list(e) for e in r["errors"]] == [[3, er.UnknownRoomError.slug], [4, er.ConflictError.slug],
                                              [5, er.CapacityError.slug], [6, er.ValidationError.slug],
                                              [7, er.ValidationError.slug], [8, er.ValidationError.slug]]


def test_s6_rows_see_earlier_rows(tmp_path):
    s, a, _ = two()
    r = s.import_csv(_csv(tmp_path, [[a, "2026-03-02T09:00", "1h", "one", "1"],
                                     [a, "2026-03-02T10:30", "1h", "two", "1"]]))
    assert r["imported"] == 2
    with pytest.raises(E().ConflictError):
        s.book(a, "2026-03-02T10:30", 30, "clash")


# ---------------------------------------------------------------- stage 7
def test_s7_weekly_series():
    s, a, _ = two()
    ids = s.book_series(a, "2026-03-02T09:00", 4, 7, 60, "Weekly", 3)
    assert len(ids) == 4
    assert [s.booking(i)["start"] for i in ids] == ["2026-03-02T09:00", "2026-03-09T09:00", "2026-03-16T09:00",
                                                    "2026-03-23T09:00"]
    assert all(s.booking(i)["title"] == "Weekly" and s.booking(i)["attendees"] == 3 for i in ids)


def test_s7_every_days():
    s, a, _ = two()
    ids = s.book_series(a, "2026-03-02T09:00", 3, 2, 30, "Alt")
    assert [s.booking(i)["start"][:10] for i in ids] == ["2026-03-02", "2026-03-04", "2026-03-06"]


def test_s7_series_all_or_nothing():
    s, a, _ = two()
    s.book(a, "2026-03-16T09:30", 30, "blocker")
    with pytest.raises(E().ConflictError):
        s.book_series(a, "2026-03-02T09:00", 4, 7, 60, "Weekly")
    assert s.day_bookings(a, "2026-03-02") == [] and s.day_bookings(a, "2026-03-09") == []


def test_s7_series_validation():
    s, a, _ = two()
    with pytest.raises(E().ValidationError):
        s.book_series(a, "2026-03-02T09:00", 3, 0, 30, "x")
    with pytest.raises(E().ValidationError):
        s.book_series(a, "2026-03-02T09:00", 0, 7, 30, "x")
    with pytest.raises(E().UnknownRoomError):
        s.book_series("R999", "2026-03-02T09:00", 3, 7, 30, "x")


# ---------------------------------------------------------------- stage 8
def test_s8_day_report():
    s, a, b = two()
    s.book(a, "2026-03-02T11:00", 30, "Sync", 2)
    s.book(a, "2026-03-02T09:00", 90, "Sprint review", 6)
    s.book(b, "2026-03-02T10:00", 60, "Interview", 3)
    s.book(b, "2026-03-03T10:00", 60, "Tomorrow", 3)
    assert s.day_report("2026-03-02") == "\n".join([
        "Rooms on 2026-03-02",
        f"{a} Alpha: 2 bookings, 2h",
        "  09:00-10:30 Sprint review (6)",
        "  11:00-11:30 Sync (2)",
        f"{b} Beta: 1 bookings, 1h",
        "  10:00-11:00 Interview (3)",
        "Total: 3h"])


def test_s8_report_bad_day():
    s, a, _ = two()
    with pytest.raises(E().ValidationError):
        s.day_report("March 2")


def test_s8_report_minutes():
    s, a, b = two()
    s.book(a, "2026-03-02T09:00", 45, "A", 1)
    s.book(b, "2026-03-02T09:00", 20, "B", 1)
    rep = s.day_report("2026-03-02").splitlines()
    assert f"{a} Alpha: 1 bookings, 45m" in rep and f"{b} Beta: 1 bookings, 20m" in rep and rep[-1] == "Total: 1h05m"


# ---------------------------------------------------------------- stage 9
def test_s9_cli_flow(tmp_path):
    r = cli(tmp_path, "add-room", "Alpha", "10")
    assert r.returncode == 0, r.stderr
    room = r.stdout.strip()
    r = cli(tmp_path, "book", room, "2030-03-04T09:00", "--duration", "1h30m", "--title", "Planning", "--attendees", "4")
    assert r.returncode == 0, r.stderr
    bid = r.stdout.strip()
    r = cli(tmp_path, "list", room, "2030-03-04")
    assert r.returncode == 0 and r.stdout.strip() == f"{bid} 09:00-10:30 Planning"


def test_s9_cli_report(tmp_path):
    room = cli(tmp_path, "add-room", "Alpha", "10").stdout.strip()
    cli(tmp_path, "book", room, "2030-03-04T09:00", "--duration", "60", "--title", "Sync", "--attendees", "2")
    r = cli(tmp_path, "report", "2030-03-04")
    assert r.returncode == 0 and r.stdout.strip().splitlines() == \
        ["Rooms on 2030-03-04", f"{room} Alpha: 1 bookings, 1h", "  09:00-10:00 Sync (2)", "Total: 1h"]


def test_s9_cli_cancel_persists(tmp_path):
    room = cli(tmp_path, "add-room", "Alpha", "10").stdout.strip()
    bid = cli(tmp_path, "book", room, "2030-03-04T09:00", "--duration", "60", "--title", "X").stdout.strip()
    r = cli(tmp_path, "cancel", bid)
    assert r.returncode == 0 and r.stdout.strip() == ""
    assert cli(tmp_path, "list", room, "2030-03-04").stdout.strip() == ""


def test_s9_cli_error_line(tmp_path):
    room = cli(tmp_path, "add-room", "Alpha", "2").stdout.strip()
    r = cli(tmp_path, "book", room, "2030-03-04T09:00", "--duration", "60", "--attendees", "3")
    assert r.returncode != 0 and any(l.startswith(f"error {E().CapacityError.slug}:") for l in r.stderr.splitlines())
    r = cli(tmp_path, "book", "R999", "2030-03-04T09:00", "--duration", "60")
    assert r.returncode != 0 and any(l.startswith(f"error {E().UnknownRoomError.slug}:") for l in r.stderr.splitlines())


def test_s9_cli_saves(tmp_path):
    cli(tmp_path, "add-room", "Alpha", "10")
    cli(tmp_path, "add-room", "Beta", "5")
    from roomly.scheduler import Scheduler
    assert [r["name"] for r in Scheduler.load(str(tmp_path / "db.json")).rooms()] == ["Alpha", "Beta"]


# ---------------------------------------------------------------- stage 10
def test_s10_move():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 90, "Plan", 3)
    s.move(x, "2026-03-02T14:00")
    b = s.booking(x)
    assert (b["start"], b["end"], b["title"], b["attendees"], b["id"]) == \
        ("2026-03-02T14:00", "2026-03-02T15:30", "Plan", 3, x)


def test_s10_move_overlapping_itself():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 90, "Plan")
    s.move(x, "2026-03-02T09:30")
    assert s.booking(x)["end"] == "2026-03-02T11:00"


def test_s10_move_conflict_changes_nothing():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 60, "x")
    s.book(a, "2026-03-02T13:00", 60, "y")
    with pytest.raises(E().ConflictError):
        s.move(x, "2026-03-02T12:30")
    assert s.booking(x)["start"] == "2026-03-02T09:00"
    s.book(a, "2026-03-02T11:00", 60, "slot still free")


def test_s10_move_unknown_and_cancelled():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 60, "x")
    s.cancel(x)
    for i in (x, "BK-99999"):
        with pytest.raises(E().UnknownBookingError):
            s.move(i, "2026-03-02T12:00")


def test_s10_move_persists(tmp_path):
    from roomly.scheduler import Scheduler
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 60, "x")
    s.move(x, "2026-03-03T10:00")
    p = str(tmp_path / "db.json"); s.save(p)
    assert Scheduler.load(p, clock=lambda: T0).booking(x)["start"] == "2026-03-03T10:00"


# ---------------------------------------------------------------- stage 11
def test_s11_info_per_mutation(cap):
    s = S()
    cap.records.clear()
    a = s.add_room("Alpha", 10)
    x = s.book(a, "2026-03-02T09:00", 60, "x")
    s.move(x, "2026-03-02T11:00")
    s.cancel(x)
    infos = [r.getMessage() for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 4 and a in infos[0] and all(x in m for m in infos[1:])


def test_s11_info_series_and_import(cap, tmp_path):
    s, a, b = two()
    cap.records.clear()
    ids = s.book_series(a, "2026-03-02T09:00", 3, 7, 30, "w")
    s.import_csv(_csv(tmp_path, [[b, "2026-03-02T09:00", "1h", "i1", "1"], [b, "2026-03-02T11:00", "1h", "i2", "1"]]))
    infos = [r.getMessage() for r in cap.records if r.levelno == logging.INFO]
    assert len(infos) == 5 and all(any(i in m for m in infos) for i in ids)


def test_s11_warning_with_slug(cap):
    s, a, _ = two()
    s.book(a, "2026-03-02T09:00", 60, "x")
    cap.records.clear()
    with pytest.raises(E().ConflictError):
        s.book(a, "2026-03-02T09:30", 30, "y")
    with pytest.raises(E().UnknownRoomError):
        s.room("R999")
    w = [r.getMessage() for r in cap.records if r.levelno == logging.WARNING]
    assert len(w) == 2 and E().ConflictError.slug in w[0] and E().UnknownRoomError.slug in w[1]


def test_s11_no_info_on_failure(cap):
    s, a, _ = two()
    cap.records.clear()
    with pytest.raises(E().ValidationError):
        s.add_room("", 3)
    assert not [r for r in cap.records if r.levelno == logging.INFO]


# ---------------------------------------------------------------- stage 12
def test_s12_find():
    s, a, b = two()
    y = s.book(a, "2026-03-03T09:00", 30, "Team Sync")
    x = s.book(b, "2026-03-02T09:00", 30, "sync with HR")
    s.book(a, "2026-03-02T09:00", 30, "Lunch")
    assert [r["id"] for r in s.find("  SYNC ")] == [x, y]
    assert s.find("nothing") == []


def test_s12_find_excludes_cancelled():
    s, a, _ = two()
    x = s.book(a, "2026-03-02T09:00", 30, "Sync")
    s.cancel(x)
    assert s.find("sync") == []


def test_s12_find_empty_query():
    s, _, _ = two()
    for q in ("", "   "):
        with pytest.raises(E().ValidationError):
            s.find(q)


def test_s12_find_order_same_start():
    s, a, b = two()
    x = s.book(a, "2026-03-02T09:00", 30, "Sync A")
    y = s.book(b, "2026-03-02T09:00", 30, "Sync B")
    assert [r["id"] for r in s.find("sync")] == [x, y]
    assert s.find("sync")[0]["room"] == a
