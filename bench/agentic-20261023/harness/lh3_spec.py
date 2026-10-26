"""Long-horizon v3: `roomly` (meeting-room booking) over 12 stages, rules buried mid-session.

Same protocol as v2 (lh2_spec.py) with a new project and new decisions, to check that the v2 result is not an
artifact of the vxledger task: 12 conventions and 20 team decisions arrive only inside "team chat" blocks of the
stage messages, 4 are superseded later (D03 grid 15->5, D06 gap 10->15, D08 closing 20:00->22:00, c11 default
duration 30->50). Stage texts only say "as agreed".
Scoring: hidden/lh3/test_functional.py (stage behaviour, inputs chosen so that no decision changes the outcome),
hidden/lh3/test_rules.py (one test per decision, latest values), hidden/lh3/check_conventions.py.
"""

FILLER_INTRO = ("For context, here is part {k} of the source of the legacy C++ facility service we are replacing. "
                "It is reference material only; do not port it.")

BRIEF = """We are building `roomly`, a small meeting-room booking library with a command-line interface, in Python 3.12, in the current directory. The work comes in 12 stages; I will send each stage after you finish the previous one. When a stage is done (implemented and checked by you), reply with a bash block containing only `echo STAGE_COMPLETE`.

Each stage message starts with an excerpt of our team chat since the previous stage. Conventions and decisions agreed there are binding for the whole project from then on (a later decision can change an earlier one). I will NOT repeat them later; the stage descriptions just say "as agreed". Keep track of them yourself."""

NOTES = [
# 1: c01 c02 c03 c05 c08 c11(30) c12 D01 D02
"""[08:55] Lena: good morning, roomly kickoff 🎉
[08:57] Sam (lead): Convention: all code lives in the package `roomly/` (import name `roomly`).
[09:00] Omar: debugging is easier with print statements everywhere
[09:01] Sam (lead): Convention: no print() calls anywhere in roomly except in roomly/__main__.py (the CLI). Libraries don't print.
[09:04] Lena: error codes like the old service, 9001, 9002...?
[09:05] Sam (lead): Convention: no numeric codes. Every exception class derives from `RoomlyError` (in roomly/errors.py) and has a class attribute `slug`: a unique string of lowercase words joined by hyphens, e.g. "unknown-room".
[09:08] Omar: can we use dateutil? parsing is painful
[09:09] Sam (lead): Convention: standard library only, no third-party packages in roomly.
[09:12] Lena: saving files - plain open(path, "w")?
[09:13] Sam (lead): Convention: every file write goes through `safe_write(path: str, text: str) -> None` in roomly/fileio.py, which writes a temporary file in the same directory and then calls os.replace.
[09:16] Omar: default meeting length when nobody says? 60 like Outlook?
[09:17] Lena: most of ours are short, 30 maybe
[09:18] Sam (lead): Convention: wherever a duration is optional, the default is 30 minutes. (we'll see how it goes)
[09:20] Omar: ids - uuids?
[09:21] Sam (lead): Convention: room ids are "R" + a 3-digit zero-padded sequence number starting at 1 (R001, R002, ...); booking ids are "BK-" + a 5-digit zero-padded sequence number starting at 1 (BK-00001, BK-00002, ...).
[09:24] Lena: room names on the door signs are tiny, how long may they be?
[09:25] Omar: 64?
[09:26] Sam (lead): Decision: room names may have at most 30 characters after stripping; longer raises ValidationError.
[09:28] Lena: and capacity - the atrium holds 500 for town halls
[09:29] Sam (lead): Decision: capacity is an integer from 1 to 200. The atrium is not a meeting room.""",
# 2: c06 c07 D03(15) D04 D05
"""[10:31] Omar: type hints or docstrings? not both please
[10:32] Sam (lead): Convention: every public function and method (name not starting with _) has a docstring. Type hints are optional.
[10:35] Lena: time format for what we return - with seconds? with +00:00?
[10:36] Sam (lead): Convention: every date-time that roomly returns or stores is a string "YYYY-MM-DDTHH:MM" (minute precision, no seconds, no offset), always UTC. Never use naive datetimes internally.
[10:40] Omar: should starts be free-form? someone will book 09:07
[10:41] Sam (lead): Decision: a booking must start on a 15-minute grid (minute 00, 15, 30 or 45), otherwise ValidationError.
[10:43] Lena: max length? people block rooms for whole days
[10:44] Omar: 8 hours?
[10:45] Sam (lead): Decision: a single booking may last at most 240 minutes (4h); longer raises ValidationError.
[10:48] Lena: titles get pasted from emails with whitespace and novels
[10:49] Sam (lead): Decision: titles are stored stripped and truncated to their first 50 characters; an empty title is stored as "(no title)".""",
# 3: c04 c10 D06(10) D07 D08(20:00)
"""[13:05] Omar: logging: just logging.info at module level?
[13:06] Sam (lead): Convention: log only through logging.getLogger("roomly") (exactly that name, no child loggers) and every log message starts with "roomly: ".
[13:09] Lena: license header?
[13:10] Sam (lead): Convention: every .py file in roomly starts with the line `# roomly - internal use only`.
[13:14] Omar: facilities wants time to clean between meetings
[13:15] Lena: 5 minutes is enough
[13:16] Sam (lead): Decision: two bookings of the same room need a gap of at least 10 minutes between the end of one and the start of the next; a smaller gap is a ConflictError.
[13:20] Lena: people cancel meetings that are already running to hide no-shows
[13:21] Sam (lead): Decision: a booking whose start is at or before the current time (clock) cannot be cancelled; that raises ValidationError.
[13:24] Omar: building hours?
[13:25] Sam (lead): Decision: a booking must lie completely within 07:00-20:00 (UTC) of its day: start at or after 07:00, end at or before 20:00. Otherwise ValidationError.""",
# 4: D09
"""[11:40] Lena: the save file - pretty print?
[11:41] Omar: it's synced to the kiosks over a slow line, smaller is better
[11:42] Sam (lead): Decision: the saved file is compact JSON: json.dumps(data, sort_keys=True, separators=(",", ":")) followed by one trailing newline.
[11:45] Omar: should we also gzip it?
[11:46] Sam (lead): No.""",
# 5: c09 D10 D17
"""[15:30] Omar: CLI exit codes - 1 for errors?
[15:31] Sam (lead): Convention: the command-line interface exits with status 5 on any user or validation error (including bad arguments and unknown commands) and 0 on success.
[15:34] Lena: first start on a kiosk has no file yet
[15:35] Sam (lead): Decision: Scheduler.load of a path that does not exist returns an empty scheduler (no error). Corrupt files still raise LoadError.
[15:38] Omar: and ops wants to page on a corrupt file
[15:39] Sam (lead): Decision: when the CLI cannot load the DB file it prints the error line and exits with status 6 instead of 5.""",
# 6: D11 D12 D13
"""[09:12] Lena: CSV import from the old system - if row 40 fails, rows 1-39 are booked and people get confused
[09:13] Sam (lead): Decision: the CSV import is all-or-nothing: if any row fails, nothing at all is booked; the result then has "imported": 0 and still lists the errors of all failing rows.
[09:16] Omar: the old export writes dates like 02.03.2026 09:00
[09:17] Sam (lead): Decision: the CSV import also accepts the start as "DD.MM.YYYY HH:MM" besides the normal format.
[09:20] Lena: recurring meetings next stage - limit?
[09:21] Omar: 52, a year of weeklies
[09:22] Sam (lead): Decision: a series has at most 12 occurrences (count 1-12), otherwise ValidationError.""",
# 7: D14 D15 D16
"""[14:00] Omar: daily series hit Saturdays and the building is closed
[14:01] Sam (lead): Decision: series occurrences that fall on a Saturday or Sunday are skipped (not booked, no error); they still count towards `count`.
[14:05] Lena: the day report - rooms by id is useless for reception
[14:06] Sam (lead): Decision: the day report lists rooms sorted by room name, case-insensitively.
[14:08] Omar: and empty rooms make it two pages long
[14:09] Sam (lead): Decision: rooms without bookings that day are left out of the day report.""",
# 8: D03 superseded (5) c11 superseded (50) D18
"""[10:20] Lena: the 15-minute grid annoys everyone, lots of 09:05 standups
[10:21] Sam (lead): Decision (changes the earlier grid): a booking must now start on a 5-minute grid (minute divisible by 5). The 15-minute rule is gone.
[10:24] Omar: and the 30 minute default is always too short
[10:25] Sam (lead): Convention change: wherever a duration is optional, the default is now 50 minutes (no longer 30).
[10:29] Lena: CLI needs to show the rooms
[10:30] Sam (lead): Decision: the CLI gets a command `rooms` that prints one line per room in id order: `<id> <name> <capacity>`.""",
# 9: D06 superseded (15) D19
"""[16:10] Omar: facilities says 10 minutes isn't enough for cleaning
[16:11] Sam (lead): Decision (changes the earlier gap): the gap between two bookings of the same room is now at least 15 minutes. It applies everywhere a booking is created or changed.
[16:15] Lena: kiosks cache bookings, they need to know when one changed
[16:16] Sam (lead): Decision: every booking dict has a key "rev": 1 when created, +1 every time the booking is moved. It survives save/load.""",
# 10: D08 superseded (22:00) D20
"""[12:00] Lena: evening classes moved in, the building is open later now
[12:01] Sam (lead): Decision (changes the earlier opening hours): bookings must lie within 07:00-22:00 now (end at or before 22:00).
[12:05] Omar: audit logs only have room ids, nobody knows R007
[12:06] Sam (lead): Decision: the INFO log record of every book, move and cancel also contains the room's name.""",
# 11: chatter only
"""[14:30] Omar: should we add room features (projector, whiteboard)?
[14:31] Sam (lead): Not in this project.
[14:33] Lena: maybe allow 30-minute gaps for the boardroom?
[14:34] Sam (lead): No, keep what we decided.""",
# 12: chatter only
"""[17:00] Lena: last stage, demo on Thursday!
[17:01] Sam (lead): Please make sure everything we agreed on in this chat is implemented before you hand it in.""",
]

STAGES = [
# 1
"""Stage 1 - errors, time helpers and file helper.
Create roomly/__init__.py, roomly/errors.py, roomly/timeutil.py and roomly/fileio.py.
- errors.py defines ValidationError, UnknownRoomError, UnknownBookingError, ConflictError, CapacityError and LoadError (structure as agreed).
- `timeutil.parse_duration(text: str) -> int` returns minutes. Accepted forms (surrounding whitespace ignored): '45' (plain minutes), '45m', '2h', '1h30m', '1h5m', '1h05m'. In 'XhYm' Y has 1 or 2 digits and must be 0-59. The result must be > 0. Anything else ('', '0', '0m', '1h60m', '1.5h', 'abc', '-5', '1h 30m', 'h', '30mm', '2H') raises ValidationError.
- `timeutil.format_duration(minutes: int) -> str`: 0 -> '0m', 45 -> '45m', 60 -> '1h', 90 -> '1h30m', 125 -> '2h05m'. A negative value raises ValidationError.
- `timeutil.parse_start(text: str) -> datetime` parses exactly 'YYYY-MM-DDTHH:MM' (surrounding whitespace ignored; it must be a real date and time) into a timezone-aware UTC datetime; anything else ('2026-03-02 09:00', '2026-03-02T9:00', '2026-02-30T09:00', '2026-03-02T09:00:00') raises ValidationError.
- `timeutil.format_start(dt: datetime) -> str` returns 'YYYY-MM-DDTHH:MM' (the inverse of parse_start).
- the file helper as agreed.""",
# 2
"""Stage 2 - rooms.
Create roomly/scheduler.py with a class `Scheduler` (also importable as `from roomly import Scheduler`).
- `Scheduler()` creates an empty scheduler.
- `add_room(self, name: str, capacity: int) -> str` adds a room and returns its id (format as agreed). The name is stored stripped; it must be non-empty, within the agreed length and unique case-insensitively among rooms, otherwise ValidationError. capacity must be an int (not a bool) within the agreed range, otherwise ValidationError.
- `room(self, room_id: str) -> dict` returns {"id", "name", "capacity"}; an unknown id raises UnknownRoomError (this applies to every method that takes a room id).
- `rooms(self) -> list[dict]` returns all rooms (same dicts) in id order.""",
# 3
"""Stage 3 - bookings.
- `book(self, room_id: str, start: str, duration: int | None = None, title: str = "", attendees: int = 1) -> str` books a room and returns a booking id (format as agreed; one sequence for the whole scheduler). start uses the parse_start format. duration is in minutes (default as agreed) and must be > 0 and within the agreed maximum. The start must be on the agreed minute grid and the booking within the agreed hours. These violations raise ValidationError. attendees must be an int >= 1 (ValidationError) and at most the room's capacity (CapacityError). A booking that overlaps another booking of the same room raises ConflictError: two bookings overlap if each starts before the other ends (one ending at 10:00 and another starting at 10:00 do not overlap); spacing rules as agreed. On any error nothing changes. Title handling as agreed.
- `booking(self, booking_id: str) -> dict` returns {"id", "room", "start", "end", "title", "attendees"} with start and end as agreed; an unknown id raises UnknownBookingError.""",
# 4
"""Stage 4 - clock, cancellations and day lists.
- The constructor becomes `Scheduler(clock: Callable[[], datetime] | None = None)`. The clock returns the current time as a timezone-aware UTC datetime; without it use the real current UTC time.
- `cancel(self, booking_id: str) -> None` cancels a booking (other rules as agreed). A cancelled booking frees its slot and no longer exists for booking(), day lists or anything else: booking() and cancel() of a cancelled or unknown id raise UnknownBookingError.
- `day_bookings(self, room_id: str, day: str) -> list[dict]` returns the room's bookings that start on day ('YYYY-MM-DD'; anything else raises ValidationError) as booking() dicts sorted by start.""",
# 5
"""Stage 5 - persistence.
- `save(self, path: str) -> None` writes the whole schedule as one JSON document (file format as agreed) whose top-level object has "version": 3. The rest of the structure is up to you, but loading must restore everything: rooms, bookings and both id sequences (a new room or booking after loading continues the numbering, also when the highest booking was cancelled before saving).
- `Scheduler.load(path: str, clock=None) -> "Scheduler"` (a classmethod; clock as in the constructor) reads such a file. Invalid JSON, a missing or wrong "version" or an unreadable structure raises LoadError. A path that does not exist: as agreed.""",
# 6
"""Stage 6 - CSV import.
- `import_csv(self, path: str) -> dict` books the rows of a CSV file with the header `room,start,duration,title,attendees`. start uses the parse_start format (other formats as agreed). duration uses the parse_duration format and may be empty (= default duration). attendees may be empty (= 1). Each row is checked exactly like book(); rows are processed in file order and a row sees the bookings of the earlier valid rows.
- It returns {"imported": <number of bookings created>, "errors": [[line_number, slug], ...]}: line_number counts the header as line 1, slug is the slug of the error class the row caused (a malformed field counts as ValidationError). What happens to the valid rows when other rows fail is as agreed.""",
# 7
"""Stage 7 - recurring series.
- `book_series(self, room_id: str, start: str, count: int, every_days: int = 7, duration: int | None = None, title: str = "", attendees: int = 1) -> list[str]` books a series: occurrence k (k = 0 .. count-1) starts at start + k * every_days days, same time of day, duration, title and attendees, subject to the agreed limits and rules for series. every_days must be >= 1 (ValidationError). Each booked occurrence is checked like book(); if any of them fails, nothing is booked and that error is raised. It returns the booking ids in date order.""",
# 8
"""Stage 8 - day report.
- `day_report(self, day: str) -> str` returns this text (lines joined with "\\n", no trailing newline; a bad day raises ValidationError):
    Rooms on <YYYY-MM-DD>
    then for each room (selection and order as agreed) a line
    <room id> <room name>: <n> bookings, <format_duration(total booked minutes of the room that day)>
    followed by one line per booking of that room starting that day, sorted by start:
      <HH:MM>-<HH:MM> <title> (<attendees>)        (indented by two spaces)
    and finally
    Total: <format_duration(total booked minutes of all rooms that day)>
  Example booking line: `  09:00-10:30 Sprint review (6)`.""",
# 9
"""Stage 9 - command-line interface.
Create roomly/__main__.py so that `python -m roomly DB COMMAND ...` works. DB is the path of a schedule JSON file (stage 5 format; a path that does not exist behaves as agreed). Every command that changes the schedule saves it back to DB.
Commands:
  add-room NAME CAPACITY                                     prints the new room id
  book ROOM START [--duration D] [--title T] [--attendees N] prints the booking id (D in parse_duration format)
  cancel BOOKING                                             prints nothing
  list ROOM DAY                                              one line per booking: `<id> <HH:MM>-<HH:MM> <title>`
  report DAY                                                 prints the day report
plus any commands and options we agreed on in the team chat. On an error print `error <slug>: <message>` to stderr and exit with the agreed status.""",
# 10
"""Stage 10 - moving bookings.
- `move(self, booking_id: str, new_start: str) -> None` moves an existing booking to a new start in the same room, keeping its duration, title and attendees. The new slot is checked like book() (a booking never conflicts with itself); on any error nothing changes. Unknown or cancelled ids raise UnknownBookingError. Booking fields as agreed. A moved booking keeps its id and moves must survive save/load.""",
# 11
"""Stage 11 - audit logging.
- Every successful mutating operation (add_room, every created booking including series occurrences and imported rows, cancel, move) emits exactly one INFO log record that contains the id it created or touched (room id or booking id), following the logging conventions and the agreed message contents.
- Every public method call that fails with one of our errors emits exactly one WARNING record that contains the error's slug, then raises as before.""",
# 12
"""Stage 12 - search and final check.
- `find(self, text: str) -> list[dict]` returns the bookings whose title contains text (case-insensitive; text is stripped and an empty text raises ValidationError) as booking() dicts sorted by start, then id.
- Finally, review the whole package once more: make sure every stage still works and that the code follows every convention and decision we agreed on, then finish with STAGE_COMPLETE.""",
]

# Recall of the LATEST value of each item in a handoff summary (case-insensitive), scored only in summaries
# written after the stage that introduced the item (ITEM_STAGE, 1-based).
RECALL = {
    "c01_package": r"roomly/|package\W+.?roomly",
    "c02_no_print": r"print",
    "c03_roomlyerror_slug": r"RoomlyError(?=[\s\S]*slug)|slug[\s\S]*RoomlyError",
    "c04_logging": r"roomly: ",
    "c05_stdlib": r"standard[ -]library|stdlib|third[- ]party|no (external|extra) (packages|dependencies)",
    "c06_docstrings": r"docstring",
    "c07_minute_times": r"YYYY-MM-DDTHH:MM(?!:)|minute precision|no seconds",
    "c08_safe_write": r"safe_write|os\.replace",
    "c09_exit5": r"(status|code|exit)\D{0,12}5\b",
    "c10_header": r"internal use only",
    "c11_default_50": r"default[^\n]{0,40}\b50\b|\b50\b[^\n]{0,30}default",
    "c12_ids": r"R001|BK-0|BK-\\d|BK-\{|BK-<|5-digit|five-digit",
    "d01_name30": r"\b30\b[^\n]{0,30}char|char[^\n]{0,30}\b30\b",
    "d02_cap200": r"\b200\b",
    "d03_grid5": r"5-minute|5 minute|multiple of 5\b|divisible by 5\b|% ?5\b",
    "d04_max240": r"\b240\b|4 ?h(our)?s?\b",
    "d05_no_title": r"\(no title\)",
    "d06_gap15": r"15[- ]?min[^\n]{0,40}(gap|buffer|between|clean)|(gap|buffer)[^\n]{0,40}\b15\b",
    "d07_no_cancel_started": r"cancel[^\n]{0,80}(start|past|already|clock|now)|(start|past|already)[^\n]{0,80}cancel",
    "d08_close22": r"22:00",
    "d09_compact_json": r"separators|compact",
    "d10_load_missing": r"(missing|non-?existent|not exist|doesn't exist)[^\n]{0,80}empty|empty[^\n]{0,80}(missing|non-?existent|not exist)",
    "d11_csv_atomic": r"all-or-nothing|all or nothing|nothing (at all )?is (booked|imported)|roll ?back",
    "d12_csv_dates": r"DD\.MM\.YYYY",
    "d13_series12": r"\b12\b[^\n]{0,40}occurr|occurr[^\n]{0,40}\b12\b|count[^\n]{0,20}1\s*(-|to|\.\.)\s*12",
    "d14_weekend": r"weekend|saturday",
    "d15_report_by_name": r"(sort|order)[^\n]{0,40}name|name[^\n]{0,40}(sort|order)",
    "d16_report_omit_empty": r"omit|left out|without bookings|no bookings|skip[^\n]{0,20}(empty|rooms)",
    "d17_exit6": r"(status|exit|code)[^\n]{0,20}\b6\b",
    "d18_rooms_cmd": r"`rooms`|rooms command|command `?rooms",
    "d19_rev": r"\brev\b",
    "d20_info_room_name": r"room'?s? name[^\n]{0,80}(info|log)|(info|log)[^\n]{0,80}room'?s? name",
}
ITEM_STAGE = {"c01_package": 1, "c02_no_print": 1, "c03_roomlyerror_slug": 1, "c05_stdlib": 1, "c08_safe_write": 1,
              "c11_default_50": 8, "c12_ids": 1, "d01_name30": 1, "d02_cap200": 1,
              "c06_docstrings": 2, "c07_minute_times": 2, "d03_grid5": 8, "d04_max240": 2, "d05_no_title": 2,
              "c04_logging": 3, "c10_header": 3, "d06_gap15": 9, "d07_no_cancel_started": 3, "d08_close22": 10,
              "d09_compact_json": 4, "c09_exit5": 5, "d10_load_missing": 5, "d17_exit6": 5,
              "d11_csv_atomic": 6, "d12_csv_dates": 6, "d13_series12": 6,
              "d14_weekend": 7, "d15_report_by_name": 7, "d16_report_omit_empty": 7,
              "d18_rooms_cmd": 8, "d19_rev": 9, "d20_info_room_name": 10}
# Stale values of superseded items (present without the new value = the summary carries the old rule).
STALE = {"d03_grid5": r"15-minute grid|multiple of 15|00, 15, 30|quarter[- ]hour|% ?15\b",
         "d06_gap15": r"10[- ]?min[^\n]{0,40}(gap|buffer|between|clean)|(gap|buffer)[^\n]{0,40}\b10\b",
         "d08_close22": r"20:00",
         "c11_default_50": r"default[^\n]{0,40}\b30\b|\b30\b[^\n]{0,30}default"}
