"""Long-horizon v7: `roomly` with DEFERRED decisions - the compaction-memory test.

Same project, stage texts and functional tests as v3 (lh3_spec.py), but all 12 conventions and 28 team decisions
arrive in the team chat of stages 1-3 (a kickoff planning that covers the whole project), most of them for features
that are built only in stages 5-12. Until a feature is built, its decision exists only in the conversation, i.e. in
the chain of handoff summaries: a decision for stage 10 has to survive ~7 compactions before it can be applied.
Values are deliberately unusual (27 characters, 225 minutes, 12-minute gap, 07:30-20:45, ...) so that model priors
cannot recover them. Nothing is ever revised. Stage chats 4-12 are chatter only.
Scoring: hidden/lh7/test_functional.py (= v3), hidden/lh7/test_rules.py (28 decisions),
hidden/lh7/check_conventions.py (c11 default 35).
"""
import lh3_spec as _v3

FILLER_INTRO = _v3.FILLER_INTRO
BRIEF = _v3.BRIEF

NOTES = [
# 1: c01 c02 c03 c05 c08 c11(35) c12 D01 D02 D13 D17 D21 D28
"""[08:55] Lena: good morning, roomly kickoff 🎉 Sam wants to settle the whole plan today so we never have to discuss it again
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
[09:17] Lena: facilities measured it, the median meeting here is 35 minutes
[09:18] Sam (lead): Convention: wherever a duration is optional, the default is 35 minutes.
[09:20] Omar: ids - uuids?
[09:21] Sam (lead): Convention: room ids are "R" + a 3-digit zero-padded sequence number starting at 1 (R001, R002, ...); booking ids are "BK-" + a 5-digit zero-padded sequence number starting at 1 (BK-00001, BK-00002, ...).
[09:24] Lena: room names on the door signs are tiny, how long may they be?
[09:25] Omar: 64?
[09:26] Sam (lead): Decision: room names may have at most 27 characters after stripping (that is what fits on the e-ink sign); longer raises ValidationError.
[09:28] Lena: and capacity - the atrium holds 500 for town halls
[09:29] Sam (lead): Decision: capacity is an integer from 1 to 120. The atrium is not a meeting room.
[09:33] Omar: jumping ahead - recurring meetings (that's stage 7 or so), limit?
[09:34] Lena: 52, a year of weeklies
[09:35] Sam (lead): Decision: a series has at most 9 occurrences (count 1-9), otherwise ValidationError. People must renew them every quarter.
[09:38] Omar: and when the CLI (later stage) can't read the DB file, ops wants to page on it
[09:39] Sam (lead): Decision: when the CLI cannot load the DB file it prints the error line and exits with status 6 instead of the usual error status.
[09:42] Lena: the CSV import from the old system will come later too - the old export dumps whole years
[09:43] Sam (lead): Decision: a CSV import file may have at most 37 data rows (header not counted). A bigger file raises ValidationError from import_csv before anything is booked.
[09:46] Omar: saving over the DB file scares me
[09:47] Sam (lead): Decision: when save() writes to a path that already exists, it first writes the previous file content unchanged to `<path>.bak` (also through safe_write), then the new content to the path.""",
# 2: c06 c07 D03 D04 D05 D14 D15 D19 D23 D24
"""[10:31] Omar: type hints or docstrings? not both please
[10:32] Sam (lead): Convention: every public function and method (name not starting with _) has a docstring. Type hints are optional.
[10:35] Lena: time format for what we return - with seconds? with +00:00?
[10:36] Sam (lead): Convention: every date-time that roomly returns or stores is a string "YYYY-MM-DDTHH:MM" (minute precision, no seconds, no offset), always UTC. Never use naive datetimes internally.
[10:40] Omar: should starts be free-form? someone will book 09:07
[10:41] Sam (lead): Decision: a booking must start on a 5-minute grid (minute divisible by 5), otherwise ValidationError.
[10:43] Lena: max length? people block rooms for whole days
[10:44] Omar: 8 hours?
[10:45] Sam (lead): Decision: a single booking may last at most 225 minutes; longer raises ValidationError.
[10:48] Lena: titles get pasted from emails with whitespace and novels
[10:49] Sam (lead): Decision: titles are stored stripped and truncated to their first 42 characters; an empty title is stored as "(no title)".
[10:52] Omar: continuing the plan for the later stages - daily series will hit weekends and the building is closed
[10:53] Sam (lead): Decision: series occurrences that fall on a Saturday or Sunday are skipped (not booked, no error); they still count towards `count`.
[10:56] Lena: the day report (later stage) - rooms by id is useless for reception
[10:57] Sam (lead): Decision: the day report lists rooms sorted by room name, case-insensitively.
[10:59] Omar: and the report printer at reception is narrow
[11:00] Sam (lead): Decision: in the day report, a booking title longer than 20 characters is shown as its first 17 characters followed by "..." (the stored title stays unchanged).
[11:03] Lena: when we get to moving bookings: kiosks cache bookings, they need to know when one changed
[11:04] Sam (lead): Decision: every booking dict has a key "rev": 1 when created, +1 every time the booking is moved. It survives save/load.
[11:06] Omar: and some people move the same meeting all week long
[11:07] Sam (lead): Decision: a booking can be moved at most 3 times; a 4th move raises ValidationError and changes nothing.""",
# 3: c04 c09 c10 D06 D07 D08 D09 D10 D11 D12 D16 D18 D20 D22 D25 D26 D27
"""[13:05] Omar: logging: just logging.info at module level?
[13:06] Sam (lead): Convention: log only through logging.getLogger("roomly") (exactly that name, no child loggers) and every log message starts with "roomly: ".
[13:09] Lena: license header?
[13:10] Sam (lead): Convention: every .py file in roomly starts with the line `# roomly - internal use only`.
[13:12] Omar: CLI exit codes for later - 1 for errors?
[13:13] Sam (lead): Convention: the command-line interface exits with status 5 on any user or validation error (including bad arguments and unknown commands) and 0 on success.
[13:14] Omar: facilities wants time to clean between meetings
[13:15] Lena: 5 minutes is enough
[13:16] Sam (lead): Decision: two bookings of the same room need a gap of at least 12 minutes between the end of one and the start of the next; a smaller gap is a ConflictError. It applies everywhere a booking is created or changed.
[13:20] Lena: people cancel meetings that are already running to hide no-shows
[13:21] Sam (lead): Decision: a booking whose start is at or before the current time (clock) cannot be cancelled; that raises ValidationError.
[13:24] Omar: building hours?
[13:25] Sam (lead): Decision: a booking must lie completely within 07:30-20:45 (UTC) of its day: start at or after 07:30, end at or before 20:45. Otherwise ValidationError.
[13:28] Lena: rest of the plan, persistence: the save file - pretty print?
[13:29] Omar: it's synced to the kiosks over a slow line, smaller is better
[13:30] Sam (lead): Decision: the saved file is compact JSON: json.dumps(data, sort_keys=True, separators=(",", ":")) followed by one trailing newline.
[13:33] Lena: first start on a kiosk has no file yet
[13:34] Sam (lead): Decision: Scheduler.load of a path that does not exist returns an empty scheduler (no error). Corrupt files still raise LoadError.
[13:37] Omar: CSV import - if row 40 fails, rows 1-39 are booked and people get confused
[13:38] Sam (lead): Decision: the CSV import is all-or-nothing: if any row fails, nothing at all is booked; the result then has "imported": 0 and still lists the errors of all failing rows.
[13:40] Lena: the old export writes dates like 02.03.2026 09:00
[13:41] Sam (lead): Decision: the CSV import also accepts the start as "DD.MM.YYYY HH:MM" besides the normal format.
[13:44] Omar: series every_days - someone will enter 365
[13:45] Sam (lead): Decision: every_days of a series must be from 1 to 14, otherwise ValidationError.
[13:47] Lena: the day report with empty rooms is two pages long
[13:48] Sam (lead): Decision: rooms without bookings that day are left out of the day report.
[13:50] Omar: the CLI needs to show the rooms
[13:51] Sam (lead): Decision: the CLI gets a command `rooms` that prints one line per room in id order: `<id> <name> <capacity>`.
[13:53] Lena: audit logs only have room ids, nobody knows R007
[13:54] Sam (lead): Decision: the INFO log record of every book, move and cancel also contains the room's name.
[13:57] Omar: search (last stage): someone will search for "a" and get everything
[13:58] Sam (lead): Decision: find() needs at least 3 characters of search text after stripping, otherwise ValidationError.
[13:59] Sam (lead): Decision: find() returns at most 25 matches (the first 25 in its sort order).
[14:01] Lena: and reception wants search on the command line once find exists
[14:02] Sam (lead): Decision: the CLI gets a command `find TEXT` that prints one line per match: `<id> <room id> <start> <title>`.
[14:04] Sam (lead): That's the whole plan. I won't repeat any of it, please keep track.""",
# 4-12: chatter only
"""[11:40] Lena: anyone seen the new coffee machine manual?
[11:41] Omar: it's in the kitchen drawer
[11:45] Lena: thanks""",
"""[15:30] Omar: the kiosk vendor called, hardware ships next month
[15:31] Sam (lead): fine, nothing changes for us""",
"""[09:12] Lena: reminder: retro on Friday
[09:13] Omar: 👍""",
"""[14:00] Omar: should we add room features (projector, whiteboard)?
[14:01] Sam (lead): Not in this project.""",
"""[10:20] Lena: the facility team liked the demo of the report
[10:21] Sam (lead): good, keep going as planned""",
"""[16:10] Omar: maybe allow 30-minute gaps for the boardroom?
[16:11] Sam (lead): No, keep what we decided at kickoff.""",
"""[12:00] Lena: I'm out tomorrow
[12:01] Sam (lead): ok""",
"""[14:30] Omar: are we still on track?
[14:31] Sam (lead): yes, everything as planned at kickoff.""",
"""[17:00] Lena: last stage, demo on Thursday!
[17:01] Sam (lead): Please make sure everything we agreed on in the kickoff chat is implemented before you hand it in.""",
]

STAGES = list(_v3.STAGES)


def _hook(i, old, new):
    assert STAGES[i].count(old) == 1, (i, old)
    STAGES[i] = STAGES[i].replace(old, new)


_hook(4, "(file format as agreed)", "(file format and backups as agreed)")
_hook(5, "(other formats as agreed)", "(other formats and size limits as agreed)")
_hook(7, "<title> (<attendees>)", "<title, displayed as agreed> (<attendees>)")
_hook(9, "Booking fields as agreed.", "Booking fields and limits as agreed.")
_hook(11, "(case-insensitive; text is stripped and an empty text raises ValidationError)",
      "(case-insensitive; text is stripped and an empty text raises ValidationError; other limits as agreed)")
_hook(11, "Finally, review", "Add the agreed command-line commands for search. Finally, review")

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
    "c11_default_35": r"default[^\n]{0,40}\b35\b|\b35\b[^\n]{0,30}default",
    "c12_ids": r"R001|BK-0|BK-\\d|BK-\{|BK-<|5-digit|five-digit",
    "d01_name27": r"\b27\b",
    "d02_cap120": r"\b120\b",
    "d03_grid5": r"5-minute|5 minute|multiple of 5\b|divisible by 5\b|% ?5\b",
    "d04_max225": r"\b225\b",
    "d05_title42": r"\b42\b",
    "d06_gap12": r"\b12[- ]?min|(gap|buffer)[^\n]{0,40}\b12\b|\b12\b[^\n]{0,20}(gap|buffer)",
    "d07_no_cancel_started": r"cancel[^\n]{0,80}(start|past|already|clock|now)|(start|past|already)[^\n]{0,80}cancel",
    "d08_hours": r"07:30[\s\S]{0,40}20:45|7:30[\s\S]{0,40}20:45",
    "d09_compact_json": r"separators|compact",
    "d10_load_missing": r"(missing|non-?existent|not exist|doesn't exist)[^\n]{0,80}empty|empty[^\n]{0,80}(missing|non-?existent|not exist)",
    "d11_csv_atomic": r"all-or-nothing|all or nothing|nothing (at all )?is (booked|imported)|roll ?back",
    "d12_csv_dates": r"DD\.MM\.YYYY",
    "d13_series9": r"\b9\b[^\n]{0,40}occurr|occurr[^\n]{0,40}\b9\b|count[^\n]{0,20}1\s*(-|to|\.\.)\s*9\b",
    "d14_weekend": r"weekend|saturday",
    "d15_report_by_name": r"(sort|order)[^\n]{0,40}name|name[^\n]{0,40}(sort|order)",
    "d16_report_omit_empty": r"omit|left out|without bookings|no bookings|skip[^\n]{0,20}(empty|rooms)",
    "d17_exit6": r"(status|exit|code)[^\n]{0,20}\b6\b",
    "d18_rooms_cmd": r"`rooms`|rooms command|command `?rooms",
    "d19_rev": r"\brev\b",
    "d20_info_room_name": r"room'?s? name[^\n]{0,80}(info|log)|(info|log)[^\n]{0,80}room'?s? name",
    "d21_csv_max37": r"\b37\b",
    "d22_every14": r"every_days[^\n]{0,40}\b14\b|\b14\b[^\n]{0,40}every",
    "d23_report_titles": r"\b17\b[^\n]{0,60}\.\.\.|\.\.\.[^\n]{0,60}\b17\b|\b20\b[^\n]{0,40}\b17\b",
    "d24_move_max3": r"(mov\w*)[^\n]{0,40}\b(3|three)\b[^\n]{0,10}times|at most (3|three) (times|moves)|\b(3|three) moves|4th move|fourth move",
    "d25_find_min3": r"(3|three) char|at least (3|three)",
    "d26_find_max25": r"\b25\b",
    "d27_cli_find": r"find TEXT|`find|find command|command `?find",
    "d28_backup": r"\.bak",
}
ITEM_STAGE = {k: 1 for k in ("c01_package", "c02_no_print", "c03_roomlyerror_slug", "c05_stdlib", "c08_safe_write",
                             "c11_default_35", "c12_ids", "d01_name27", "d02_cap120", "d13_series9", "d17_exit6",
                             "d21_csv_max37", "d28_backup")}
ITEM_STAGE.update({k: 2 for k in ("c06_docstrings", "c07_minute_times", "d03_grid5", "d04_max225", "d05_title42",
                                  "d14_weekend", "d15_report_by_name", "d19_rev", "d23_report_titles", "d24_move_max3")})
ITEM_STAGE.update({k: 3 for k in RECALL if k not in ITEM_STAGE})
# stage (1-based) in which the feature a decision governs is built; before that the decision lives only in the chat
# history and the summaries
USE_STAGE = {"d01_name27": 2, "d02_cap120": 2, "d03_grid5": 3, "d04_max225": 3, "d05_title42": 3, "d06_gap12": 3,
             "d07_no_cancel_started": 4, "d08_hours": 3, "d09_compact_json": 5, "d10_load_missing": 5,
             "d11_csv_atomic": 6, "d12_csv_dates": 6, "d13_series9": 7, "d14_weekend": 7, "d15_report_by_name": 8,
             "d16_report_omit_empty": 8, "d17_exit6": 9, "d18_rooms_cmd": 9, "d19_rev": 10, "d20_info_room_name": 11,
             "d21_csv_max37": 6, "d22_every14": 7, "d23_report_titles": 8, "d24_move_max3": 10, "d25_find_min3": 12,
             "d26_find_max25": 12, "d27_cli_find": 12, "d28_backup": 5}
STALE = {}
HIDDEN = "lh7"
