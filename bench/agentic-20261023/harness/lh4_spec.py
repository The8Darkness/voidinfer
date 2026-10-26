"""Long-horizon v4: `roomly` again (same stages, brief and functional tests as lh3_spec.py), harder memory load.

Differences to v3, all in the team chat:
- 9 of the 20 items change during the project, 3 of them twice (D03 grid 15->5->10, D06 gap 10->15->20,
  c11 default duration 30->50->45), D01 name 30->40, D04 max 180, D05 title 40, D08 closing 22:00, D13 series 8,
  and one decision is withdrawn (D14 weekend skip, stage 10: weekends are booked normally).
- Interference: from stage 2 on, each stage also carries a forwarded excerpt of the chat of a sibling team
  (`deskly`, desk booking) with competing values for the same kinds of rules. It is labelled as another project.
- Changes land in code written several stages earlier (grid/gap/hours/default are used by book, move, CSV, series).
Scoring: hidden/lh4 (functional tests identical to lh3, rule tests and conventions at the latest values).
"""
import lh3_spec

FILLER_INTRO = lh3_spec.FILLER_INTRO
BRIEF = lh3_spec.BRIEF
STAGES = lh3_spec.STAGES

NOTES = [
# 1: c01 c02 c03 c05 c08 c11(30) c12 D01(30) D02
lh3_spec.NOTES[0],
# 2: c06 c07 D03(15) D04(240) D05(50)
lh3_spec.NOTES[1],
# 3: c04 c10 D06(10) D07 D08(20:00)
lh3_spec.NOTES[2],
# 4: D09, c11 -> 50
"""[11:40] Lena: the save file - pretty print?
[11:41] Omar: it's synced to the kiosks over a slow line, smaller is better
[11:42] Sam (lead): Decision: the saved file is compact JSON: json.dumps(data, sort_keys=True, separators=(",", ":")) followed by one trailing newline.
[11:45] Omar: should we also gzip it?
[11:46] Sam (lead): No.
[11:50] Lena: the 30 minute default is always too short, people extend every meeting
[11:52] Sam (lead): Convention change: wherever a duration is optional, the default is now 50 minutes (no longer 30).""",
# 5: c09 D10 D17, D01 -> 40, D03 -> 5
"""[15:30] Omar: CLI exit codes - 1 for errors?
[15:31] Sam (lead): Convention: the command-line interface exits with status 5 on any user or validation error (including bad arguments and unknown commands) and 0 on success.
[15:34] Lena: first start on a kiosk has no file yet
[15:35] Sam (lead): Decision: Scheduler.load of a path that does not exist returns an empty scheduler (no error). Corrupt files still raise LoadError.
[15:38] Omar: and ops wants to page on a corrupt file
[15:39] Sam (lead): Decision: when the CLI cannot load the DB file it prints the error line and exits with status 6 instead of 5.
[15:42] Lena: the new door signs are wider, "Large Conference Room North Wing" doesn't fit in 30
[15:43] Sam (lead): Decision (changes the earlier name limit): room names may now have at most 40 characters after stripping.
[15:46] Omar: and the 15-minute grid annoys everyone, lots of 09:05 standups
[15:47] Sam (lead): Decision (changes the earlier grid): a booking must now start on a 5-minute grid (minute divisible by 5). The 15-minute rule is gone.""",
# 6: D11 D12 D13(12), D04 -> 180, D06 -> 15
"""[09:12] Lena: CSV import from the old system - if row 40 fails, rows 1-39 are booked and people get confused
[09:13] Sam (lead): Decision: the CSV import is all-or-nothing: if any row fails, nothing at all is booked; the result then has "imported": 0 and still lists the errors of all failing rows.
[09:16] Omar: the old export writes dates like 02.03.2026 09:00
[09:17] Sam (lead): Decision: the CSV import also accepts the start as "DD.MM.YYYY HH:MM" besides the normal format.
[09:20] Lena: recurring meetings next stage - limit?
[09:21] Omar: 52, a year of weeklies
[09:22] Sam (lead): Decision: a series has at most 12 occurrences (count 1-12), otherwise ValidationError.
[09:26] Lena: people block rooms for 4 hours and leave after one
[09:27] Sam (lead): Decision (changes the earlier maximum): a single booking may last at most 180 minutes now; longer raises ValidationError.
[09:30] Omar: facilities says 10 minutes isn't enough for cleaning
[09:31] Sam (lead): Decision (changes the earlier gap): the gap between two bookings of the same room is now at least 15 minutes. It applies everywhere a booking is created or changed.""",
# 7: D14 D15 D16, D05 -> 40, D08 -> 22:00
"""[14:00] Omar: daily series hit Saturdays and the building is closed
[14:01] Sam (lead): Decision: series occurrences that fall on a Saturday or Sunday are skipped (not booked, no error); they still count towards `count`.
[14:05] Lena: the day report - rooms by id is useless for reception
[14:06] Sam (lead): Decision: the day report lists rooms sorted by room name, case-insensitively.
[14:08] Omar: and empty rooms make it two pages long
[14:09] Sam (lead): Decision: rooms without bookings that day are left out of the day report.
[14:12] Lena: kiosk screens cut titles at 40 characters anyway
[14:13] Sam (lead): Decision (changes the earlier title rule): titles are now truncated to their first 40 characters (still stripped, empty still "(no title)").
[14:16] Omar: evening classes moved in, the building is open later now
[14:17] Sam (lead): Decision (changes the earlier opening hours): bookings must lie within 07:00-22:00 now (end at or before 22:00).""",
# 8: D18, c11 -> 45
"""[10:29] Lena: CLI needs to show the rooms
[10:30] Sam (lead): Decision: the CLI gets a command `rooms` that prints one line per room in id order: `<id> <name> <capacity>`.
[10:34] Omar: 50 minute default turned out odd, calendars show 45-minute slots
[10:35] Sam (lead): Convention change: wherever a duration is optional, the default is now 45 minutes (no longer 50).""",
# 9: D19, D06 -> 20, D13 -> 8
"""[16:15] Lena: kiosks cache bookings, they need to know when one changed
[16:16] Sam (lead): Decision: every booking dict has a key "rev": 1 when created, +1 every time the booking is moved. It survives save/load.
[16:20] Omar: facilities again: 15 minutes is still too tight for the big rooms
[16:21] Sam (lead): Decision (changes the gap again): the gap between two bookings of the same room is now at least 20 minutes, everywhere a booking is created or changed.
[16:24] Lena: 12-week series block rooms for a whole quarter
[16:25] Sam (lead): Decision (changes the earlier series limit): a series now has at most 8 occurrences (count 1-8).""",
# 10: D20, D03 -> 10, D14 withdrawn
"""[12:05] Omar: audit logs only have room ids, nobody knows R007
[12:06] Sam (lead): Decision: the INFO log record of every book, move and cancel also contains the room's name.
[12:10] Lena: the 5-minute grid makes the kiosk calendar unreadable
[12:11] Sam (lead): Decision (changes the grid again): a booking must now start on a 10-minute grid (minute 00, 10, 20, 30, 40 or 50).
[12:15] Omar: building is open on weekends since the coworking deal
[12:16] Sam (lead): Decision (withdraws the weekend rule): series occurrences on Saturday and Sunday are now booked like any other day. Nothing is skipped any more.""",
# 11: chatter only
"""[14:30] Omar: the deskly team only needs 5 minutes between desk bookings, can we go back to a smaller gap?
[14:31] Sam (lead): No. Desks are not rooms; keep what we decided for roomly.
[14:33] Lena: should we add room features (projector, whiteboard)?
[14:34] Sam (lead): Not in this project.""",
# 12: chatter only
"""[17:00] Lena: last stage, demo on Thursday!
[17:01] Sam (lead): Please make sure everything we agreed on in this chat is implemented (latest version of every decision) before you hand it in.""",
]

DISTRACT_INTRO = ("FYI, forwarded from #deskly (the desk-booking team; a different project with its own rules, "
                  "none of this applies to roomly):")
DISTRACT = [
None,
"""[10:02] Mia (deskly lead): Decision: desk bookings start on a 30-minute grid.
[10:05] Jonas: max length? some people sit at a desk all day
[10:06] Mia (deskly lead): Decision: a desk booking lasts at most 600 minutes.
[10:09] Mia (deskly lead): Decision: notes are truncated to 80 characters; empty notes stay empty.""",
"""[13:30] Jonas: cleaning between desk users?
[13:31] Mia (deskly lead): Decision: 5 minutes gap between two bookings of the same desk.
[13:35] Mia (deskly lead): Decision: desks can be booked from 06:00 to 21:00.""",
"""[11:20] Jonas: default length if the app doesn't send one?
[11:21] Mia (deskly lead): Convention: default duration is 60 minutes.
[11:25] Mia (deskly lead): Decision: the deskly save file is pretty-printed JSON (indent=2), easier to diff.""",
"""[15:50] Jonas: exit codes for the deskly CLI?
[15:51] Mia (deskly lead): Convention: exit 3 on user errors, 4 if the data file is broken.
[15:55] Mia (deskly lead): Decision: desk names up to 64 characters.""",
"""[09:40] Jonas: CSV import of desk plans, what if a row fails?
[09:41] Mia (deskly lead): Decision: import the valid rows anyway and report the failing ones (partial import is fine for desks).
[09:45] Mia (deskly lead): Decision: recurring desk bookings up to 52 occurrences.""",
"""[14:20] Jonas: weekend desk bookings?
[14:21] Mia (deskly lead): Decision: allowed, deskly never skips weekends.
[14:25] Mia (deskly lead): Decision: the deskly day report lists desks by desk id and also shows free desks.""",
"""[10:40] Mia (deskly lead): Convention change: default desk booking is 90 minutes now.
[10:44] Jonas: and the gap?
[10:45] Mia (deskly lead): Decision: no gap needed between desk bookings any more (0 minutes).""",
"""[16:30] Mia (deskly lead): Decision: deskly moves to a 15-minute grid.
[16:33] Mia (deskly lead): Decision: desk bookings may now last up to 720 minutes.""",
"""[12:20] Mia (deskly lead): Decision: deskly titles are truncated to 30 characters now.
[12:24] Mia (deskly lead): Decision: deskly is open 05:00-23:00.""",
"""[14:40] Mia (deskly lead): Decision: deskly series up to 104 occurrences.""",
None,
]

RECALL = dict(lh3_spec.RECALL)
for k in ("d01_name30", "d03_grid5", "d04_max240", "d05_no_title", "d06_gap15", "d13_series12", "d14_weekend",
          "c11_default_50"):
    del RECALL[k]
RECALL.update({
    "c11_default_45": r"default[^\n]{0,40}\b45\b|\b45\b[^\n]{0,30}default",
    "d01_name40": r"\b40\b[^\n]{0,30}char[^\n]{0,30}name|name[^\n]{0,40}\b40\b",
    "d03_grid10": r"10-minute grid|10 minute grid|multiple of 10\b|divisible by 10\b|% ?10\b|grid[^\n]{0,30}\b10\b",
    "d04_max180": r"\b180\b|3 ?h(our)?s?\b",
    "d05_title40": r"title[^\n]{0,60}\b40\b|\b40\b[^\n]{0,40}title",
    "d05_no_title": r"\(no title\)",
    "d06_gap20": r"20[- ]?min[^\n]{0,40}(gap|buffer|between|clean)|(gap|buffer)[^\n]{0,40}\b20\b",
    "d13_series8": r"\b8\b[^\n]{0,40}occurr|occurr[^\n]{0,40}\b8\b|count[^\n]{0,20}1\s*(-|to|\.\.)\s*8\b",
    "d14_weekend_booked": r"weekend[^\n]{0,80}(booked|allowed|no longer|not skipped|any other day)|(no longer|not|don't|do not|never)[^\n]{0,20}skip",
})
ITEM_STAGE = dict(lh3_spec.ITEM_STAGE)
for k in ("d01_name30", "d03_grid5", "d04_max240", "d06_gap15", "d13_series12", "d14_weekend", "c11_default_50"):
    del ITEM_STAGE[k]
ITEM_STAGE.update({"c11_default_45": 8, "d01_name40": 5, "d03_grid10": 10, "d04_max180": 6, "d05_title40": 7,
                   "d06_gap20": 9, "d13_series8": 9, "d14_weekend_booked": 10, "d08_close22": 7})
STALE = {
    "c11_default_45": r"default[^\n]{0,40}\b(30|50)\b|\b(30|50)\b[^\n]{0,30}default",
    "d01_name40": r"\b30\b[^\n]{0,30}char[^\n]{0,30}name|name[^\n]{0,40}\b30\b",
    "d03_grid10": r"5-minute grid|5 minute grid|multiple of 5\b|divisible by 5\b|% ?5\b|15-minute grid|multiple of 15|% ?15\b",
    "d04_max180": r"\b240\b",
    "d05_title40": r"title[^\n]{0,60}\b50\b|\b50\b[^\n]{0,40}title",
    "d06_gap20": r"(10|15)[- ]?min[^\n]{0,40}(gap|buffer|between|clean)|(gap|buffer)[^\n]{0,40}\b(10|15)\b",
    "d08_close22": r"20:00",
    "d13_series8": r"\b12\b[^\n]{0,40}occurr|occurr[^\n]{0,40}\b12\b",
    "d14_weekend_booked": r"(weekend|saturday)[^\n]{0,40}skipped|skip[^\n]{0,30}(weekend|saturday)",
}
