# roomly - internal use only
"""The Scheduler: rooms and bookings."""
import csv
import functools
import json
import logging
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from roomly.errors import (CapacityError, ConflictError, LoadError, RoomlyError, UnknownBookingError,
                           UnknownRoomError, ValidationError)
from roomly.fileio import safe_write
from roomly.timeutil import format_duration, format_start, parse_duration, parse_start

log = logging.getLogger("roomly")

NAME_MAX = 30          # D01
CAPACITY_MAX = 200     # D02
GRID = 5               # D03 (latest)
DURATION_MAX = 240     # D04
TITLE_MAX = 50         # D05
GAP = 15               # D06 (latest)
OPEN, CLOSE = 7 * 60, 22 * 60   # D08 (latest)
DEFAULT_DURATION = 50  # c11 (latest)
SERIES_MAX = 12        # D13


def _logged(fn):
    @functools.wraps(fn)
    def wrap(*a, **kw):
        try:
            return fn(*a, **kw)
        except RoomlyError as e:
            log.warning("roomly: %s failed slug=%s", fn.__name__, e.slug)
            raise
    return wrap


def _parse_day(day: str) -> date:
    if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d\d-\d\d", day):
        raise ValidationError(f"bad day {day!r}")
    try:
        return date.fromisoformat(day)
    except ValueError:
        raise ValidationError(f"bad day {day!r}") from None


def _parse_csv_start(text: str) -> datetime:
    s = text.strip()
    m = re.fullmatch(r"(\d\d)\.(\d\d)\.(\d{4}) (\d\d):(\d\d)", s)   # D12
    if m:
        return parse_start(f"{m[3]}-{m[2]}-{m[1]}T{m[4]}:{m[5]}")
    return parse_start(s)


class Scheduler:
    """Rooms and their bookings."""

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        """Creates an empty scheduler; clock returns the current aware UTC time."""
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._rooms: dict[str, dict] = {}
        self._bookings: dict[str, dict] = {}
        self._room_seq = 0
        self._booking_seq = 0

    # ------------------------------------------------------------ rooms
    @_logged
    def add_room(self, name: str, capacity: int) -> str:
        """Adds a room and returns its id."""
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > NAME_MAX:
            raise ValidationError("bad room name")
        name = name.strip()
        if any(r["name"].lower() == name.lower() for r in self._rooms.values()):
            raise ValidationError(f"duplicate room name {name!r}")
        if type(capacity) is not int or not 1 <= capacity <= CAPACITY_MAX:
            raise ValidationError(f"bad capacity {capacity!r}")
        self._room_seq += 1
        rid = f"R{self._room_seq:03d}"
        self._rooms[rid] = {"id": rid, "name": name, "capacity": capacity}
        log.info("roomly: added room %s %s", rid, name)
        return rid

    def _room(self, room_id: str) -> dict:
        r = self._rooms.get(room_id)
        if r is None:
            raise UnknownRoomError(f"unknown room {room_id!r}")
        return r

    @_logged
    def room(self, room_id: str) -> dict:
        """Returns the room dict."""
        return dict(self._room(room_id))

    def rooms(self) -> list[dict]:
        """Returns all rooms in id order."""
        return [dict(self._rooms[k]) for k in sorted(self._rooms)]

    # ------------------------------------------------------------ bookings
    def _check(self, room_id: str, start: datetime, duration: int, attendees: int, exclude: str | None = None,
               extra: list[tuple[datetime, datetime]] = ()) -> datetime:
        room = self._room(room_id)
        if type(duration) is not int or not 0 < duration <= DURATION_MAX:
            raise ValidationError(f"bad duration {duration!r}")
        if start.minute % GRID:
            raise ValidationError("start not on the minute grid")
        end = start + timedelta(minutes=duration)
        if start.hour * 60 + start.minute < OPEN or end.date() != start.date() or end.hour * 60 + end.minute > CLOSE:
            raise ValidationError("outside opening hours")
        if type(attendees) is not int or attendees < 1:
            raise ValidationError(f"bad attendees {attendees!r}")
        if attendees > room["capacity"]:
            raise CapacityError(f"{attendees} attendees exceed capacity {room['capacity']}")
        gap = timedelta(minutes=GAP)
        others = [(b["start"], b["end"]) for b in self._bookings.values()
                  if b["room"] == room_id and b["id"] != exclude] + [(s, e) for s, e in extra]
        for s, e in others:
            if start < e + gap and s < end + gap:
                raise ConflictError("slot taken")
        return end

    def _new_booking(self, room_id: str, start: datetime, end: datetime, title: str, attendees: int) -> str:
        self._booking_seq += 1
        bid = f"BK-{self._booking_seq:05d}"
        self._bookings[bid] = {"id": bid, "room": room_id, "start": start, "end": end, "title": title,
                               "attendees": attendees, "rev": 1}
        log.info("roomly: booked %s in %s (%s) at %s", bid, room_id, self._rooms[room_id]["name"], format_start(start))
        return bid

    @staticmethod
    def _title(title: str) -> str:
        t = (title or "").strip()[:TITLE_MAX]
        return t or "(no title)"

    @_logged
    def book(self, room_id: str, start: str, duration: int | None = None, title: str = "", attendees: int = 1) -> str:
        """Books a room and returns the booking id."""
        self._room(room_id)
        s = parse_start(start)
        d = DEFAULT_DURATION if duration is None else duration
        end = self._check(room_id, s, d, attendees)
        return self._new_booking(room_id, s, end, self._title(title), attendees)

    def _public(self, b: dict) -> dict:
        return {"id": b["id"], "room": b["room"], "start": format_start(b["start"]), "end": format_start(b["end"]),
                "title": b["title"], "attendees": b["attendees"], "rev": b["rev"]}

    def _booking(self, booking_id: str) -> dict:
        b = self._bookings.get(booking_id)
        if b is None:
            raise UnknownBookingError(f"unknown booking {booking_id!r}")
        return b

    @_logged
    def booking(self, booking_id: str) -> dict:
        """Returns the booking dict of an active booking."""
        return self._public(self._booking(booking_id))

    @_logged
    def cancel(self, booking_id: str) -> None:
        """Cancels a booking that has not started yet."""
        b = self._booking(booking_id)
        if b["start"] <= self._clock():   # D07
            raise ValidationError("booking already started")
        del self._bookings[booking_id]
        log.info("roomly: cancelled %s in %s (%s)", booking_id, b["room"], self._rooms[b["room"]]["name"])

    @_logged
    def day_bookings(self, room_id: str, day: str) -> list[dict]:
        """Active bookings of a room that start on day, sorted by start."""
        self._room(room_id)
        d = _parse_day(day)
        bs = [b for b in self._bookings.values() if b["room"] == room_id and b["start"].date() == d]
        return [self._public(b) for b in sorted(bs, key=lambda b: (b["start"], b["id"]))]

    # ------------------------------------------------------------ persistence
    @_logged
    def save(self, path: str) -> None:
        """Saves the schedule as JSON."""
        doc = {"version": 3, "room_seq": self._room_seq, "booking_seq": self._booking_seq,
               "rooms": [self._rooms[k] for k in sorted(self._rooms)],
               "bookings": [self._public(self._bookings[k]) for k in sorted(self._bookings)]}
        safe_write(path, json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n")   # D09

    @classmethod
    def load(cls, path: str, clock: Callable[[], datetime] | None = None) -> "Scheduler":
        """Loads a schedule saved with save(); a missing file gives an empty scheduler."""
        try:
            return cls._load(path, clock)
        except RoomlyError as e:
            log.warning("roomly: load failed slug=%s", e.slug)
            raise

    @classmethod
    def _load(cls, path: str, clock: Callable[[], datetime] | None) -> "Scheduler":
        s = cls(clock=clock)
        if not os.path.exists(path):   # D10
            return s
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            if not isinstance(doc, dict) or doc.get("version") != 3:
                raise LoadError("unsupported file version")
            s._room_seq, s._booking_seq = int(doc["room_seq"]), int(doc["booking_seq"])
            for r in doc["rooms"]:
                s._rooms[r["id"]] = {"id": r["id"], "name": r["name"], "capacity": int(r["capacity"])}
            for b in doc["bookings"]:
                s._bookings[b["id"]] = {"id": b["id"], "room": b["room"], "start": parse_start(b["start"]),
                                        "end": parse_start(b["end"]), "title": b["title"],
                                        "attendees": int(b["attendees"]), "rev": int(b["rev"])}
        except LoadError:
            raise
        except (OSError, ValueError, KeyError, TypeError, RoomlyError) as e:
            raise LoadError(f"cannot load {path}: {e}") from None
        return s

    # ------------------------------------------------------------ CSV
    @_logged
    def import_csv(self, path: str) -> dict:
        """Books the rows of a CSV file; all or nothing."""
        tentative: list[tuple[str, datetime, datetime, str, int]] = []
        errors = []
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if not rows or [c.strip() for c in rows[0]] != ["room", "start", "duration", "title", "attendees"]:
            raise ValidationError("bad CSV header")
        for line, row in enumerate(rows[1:], start=2):
            try:
                if len(row) != 5:
                    raise ValidationError("bad row")
                room_id, start, dur, title, att = row
                room_id = room_id.strip()
                self._room(room_id)
                s = _parse_csv_start(start)
                d = parse_duration(dur) if dur.strip() else DEFAULT_DURATION
                if att.strip() and not re.fullmatch(r"\d+", att.strip()):
                    raise ValidationError("bad attendees")
                a = int(att) if att.strip() else 1
                extra = [(ts, te) for r, ts, te, _, _ in tentative if r == room_id]
                end = self._check(room_id, s, d, a, extra=extra)
                tentative.append((room_id, s, end, self._title(title), a))
            except RoomlyError as e:
                errors.append([line, e.slug])
        if errors:   # D11
            return {"imported": 0, "errors": errors}
        for room_id, s, end, title, a in tentative:
            self._new_booking(room_id, s, end, title, a)
        return {"imported": len(tentative), "errors": []}

    # ------------------------------------------------------------ series
    @_logged
    def book_series(self, room_id: str, start: str, count: int, every_days: int = 7, duration: int | None = None,
                    title: str = "", attendees: int = 1) -> list[str]:
        """Books a recurring series; all or nothing."""
        self._room(room_id)
        s0 = parse_start(start)
        if type(count) is not int or not 1 <= count <= SERIES_MAX:   # D13
            raise ValidationError(f"bad count {count!r}")
        if type(every_days) is not int or every_days < 1:
            raise ValidationError(f"bad every_days {every_days!r}")
        d = DEFAULT_DURATION if duration is None else duration
        planned: list[tuple[datetime, datetime]] = []
        for k in range(count):
            s = s0 + timedelta(days=k * every_days)
            if s.weekday() >= 5:   # D14
                continue
            planned.append((s, self._check(room_id, s, d, attendees, extra=planned)))
        t = self._title(title)
        return [self._new_booking(room_id, s, e, t, attendees) for s, e in planned]

    # ------------------------------------------------------------ report
    @_logged
    def day_report(self, day: str) -> str:
        """Text report of all bookings on a day."""
        d = _parse_day(day)
        lines = [f"Rooms on {d.isoformat()}"]
        total = 0
        for r in sorted(self._rooms.values(), key=lambda r: (r["name"].lower(), r["id"])):   # D15
            bs = sorted((b for b in self._bookings.values() if b["room"] == r["id"] and b["start"].date() == d),
                        key=lambda b: (b["start"], b["id"]))
            if not bs:   # D16
                continue
            mins = sum(int((b["end"] - b["start"]).total_seconds()) // 60 for b in bs)
            total += mins
            lines.append(f"{r['id']} {r['name']}: {len(bs)} bookings, {format_duration(mins)}")
            for b in bs:
                lines.append(f"  {b['start']:%H:%M}-{b['end']:%H:%M} {b['title']} ({b['attendees']})")
        lines.append(f"Total: {format_duration(total)}")
        return "\n".join(lines)

    # ------------------------------------------------------------ move
    @_logged
    def move(self, booking_id: str, new_start: str) -> None:
        """Moves a booking to a new start, keeping its duration."""
        b = self._booking(booking_id)
        s = parse_start(new_start)
        dur = int((b["end"] - b["start"]).total_seconds()) // 60
        end = self._check(b["room"], s, dur, b["attendees"], exclude=booking_id)
        b["start"], b["end"] = s, end
        b["rev"] += 1   # D19
        log.info("roomly: moved %s in %s (%s) to %s", booking_id, b["room"], self._rooms[b["room"]]["name"],
                 format_start(s))

    # ------------------------------------------------------------ search
    @_logged
    def find(self, text: str) -> list[dict]:
        """Active bookings whose title contains text (case-insensitive)."""
        q = (text or "").strip().lower()
        if not q:
            raise ValidationError("empty search text")
        bs = [b for b in self._bookings.values() if q in b["title"].lower()]
        return [self._public(b) for b in sorted(bs, key=lambda b: (b["start"], b["id"]))]
