"""Long-horizon v5: `roomly` with the lh3 decisions and values, but 8 of the 20 decisions are not stated in the chat.
The chat only says where to look: a file under ops/ that arrives with that stage (a printer table, fire-code text,
policy export, facilities API dumps, pager rules, calendar-sync limits, a legacy CSV sample). The agent has to read
or grep it to learn the value; the ops/ files are removed after their stage, so the value survives only in the
agent's context, its summaries or whatever it writes down itself.
Tool-delivered: D01 (name 30), D02 (capacity 200), D04 (max 240), D06 (gap 10, then 15 from a new dump at stage 9),
D08 (07:00-20:00, then 22:00 from a new dump at stage 10), D12 (legacy date format), D13 (series 12), D17 (exit 6).
Everything else (conventions, D03, D05, D07, D09-D11, D14-D16, D18-D20) is in the chat exactly as in lh3.
Scoring: hidden/lh5 = copy of hidden/lh3 (same reference and tests).
"""
import json
import random

import lh3_spec

FILLER_INTRO = lh3_spec.FILLER_INTRO
STAGES = lh3_spec.STAGES
RECALL, ITEM_STAGE, STALE = lh3_spec.RECALL, lh3_spec.ITEM_STAGE, lh3_spec.STALE

BRIEF = lh3_spec.BRIEF + """

Our office is building HQ-3. Some decisions refer to exports from other teams that appear under ops/ in the working directory. These exports contain internal data and are removed again after the stage in which they arrive, so take what you need from them while they are there. Values we tell you to take from them are binding like any other decision."""

_n = lh3_spec.NOTES
NOTES = [
# 1: D01 and D02 from files
_n[0].replace("""[09:26] Sam (lead): Decision: room names may have at most 30 characters after stripping; longer raises ValidationError.""",
              """[09:26] Sam (lead): Decision: a room name may be at most as long as one line of our door-sign printers (the DS-410 model at HQ-3; `max_chars` in ops/door_sign_printers.csv), counted after stripping; longer raises ValidationError.""")
     .replace("""[09:29] Sam (lead): Decision: capacity is an integer from 1 to 200. The atrium is not a meeting room.""",
              """[09:29] Sam (lead): Decision: capacity is an integer from 1 up to the fire-code maximum for a single meeting room in ops/fire_code_hq3.txt. The atrium is not a meeting room."""),
# 2: D04 from file
_n[1].replace("""[10:44] Omar: 8 hours?
[10:45] Sam (lead): Decision: a single booking may last at most 240 minutes (4h); longer raises ValidationError.""",
              """[10:44] Omar: 8 hours?
[10:45] Sam (lead): Decision: a single booking may last at most what HR's booking policy allows for our site (`max_booking_minutes` of HQ-3 in ops/booking_policy_export.ini); longer raises ValidationError."""),
# 3: D06 and D08 from the facilities dump
_n[2].replace("""[13:15] Lena: 5 minutes is enough
[13:16] Sam (lead): Decision: two bookings of the same room need a gap of at least 10 minutes between the end of one and the start of the next; a smaller gap is a ConflictError.""",
              """[13:15] Lena: 5 minutes is enough
[13:16] Sam (lead): Decision: two bookings of the same room need a gap of at least HQ-3's `cleaning_gap_minutes` (facilities API dump in ops/facilities_api_dump.json) between the end of one and the start of the next; a smaller gap is a ConflictError.""")
     .replace("""[13:25] Sam (lead): Decision: a booking must lie completely within 07:00-20:00 (UTC) of its day: start at or after 07:00, end at or before 20:00. Otherwise ValidationError.""",
              """[13:25] Sam (lead): Decision: a booking must lie completely within HQ-3's opening hours from the same facilities dump (UTC; start at or after opening, end at or before closing). Otherwise ValidationError."""),
_n[3],
# 5: D17 from the pager rules
_n[4].replace("""[15:39] Sam (lead): Decision: when the CLI cannot load the DB file it prints the error line and exits with status 6 instead of 5.""",
              """[15:39] Sam (lead): Decision: when the CLI cannot load the DB file it prints the error line and exits with the status that ops' pager rule for that case expects (service roomly-cli in ops/pager_rules.yaml) instead of 5."""),
# 6: D12 and D13 from files
_n[5].replace("""[09:16] Omar: the old export writes dates like 02.03.2026 09:00
[09:17] Sam (lead): Decision: the CSV import also accepts the start as "DD.MM.YYYY HH:MM" besides the normal format.""",
              """[09:16] Omar: the old export writes dates differently, I put a sample in ops/legacy_export_sample.csv
[09:17] Sam (lead): Decision: the CSV import also accepts the start in exactly the format of the old system's export (see that sample) besides the normal format. Other formats stay errors.""")
     .replace("""[09:21] Omar: 52, a year of weeklies
[09:22] Sam (lead): Decision: a series has at most 12 occurrences (count 1-12), otherwise ValidationError.""",
              """[09:21] Omar: 52, a year of weeklies
[09:22] Sam (lead): Decision: a series has at most as many occurrences as the calendar sync allows for room calendars (ops/calsync_limits.md); count must be from 1 to that limit, otherwise ValidationError."""),
_n[6],
_n[7],
# 9: D06 update from a new dump
_n[8].replace("""[16:10] Omar: facilities says 10 minutes isn't enough for cleaning
[16:11] Sam (lead): Decision (changes the earlier gap): the gap between two bookings of the same room is now at least 15 minutes. It applies everywhere a booking is created or changed.""",
              """[16:10] Omar: facilities says the cleaning gap wasn't enough, they changed it; new API dump in ops/facilities_api_dump.json
[16:11] Sam (lead): Decision (changes the earlier gap): the gap between two bookings of the same room is now HQ-3's `cleaning_gap_minutes` from this new dump. It applies everywhere a booking is created or changed."""),
# 10: D08 update from a new dump
_n[9].replace("""[12:00] Lena: evening classes moved in, the building is open later now
[12:01] Sam (lead): Decision (changes the earlier opening hours): bookings must lie within 07:00-22:00 now (end at or before 22:00).""",
              """[12:00] Lena: evening classes moved in, the building hours changed; facilities sent another dump (ops/facilities_api_dump.json)
[12:01] Sam (lead): Decision (changes the earlier opening hours): bookings must lie within HQ-3's opening hours from this new dump."""),
_n[10],
_n[11],
]
for k, (a, b) in enumerate(zip(NOTES, _n)):
    assert k in (3, 6, 7, 10, 11) or a != b, f"note {k + 1} unchanged"


# ------------------------------------------------------------------ ops/ files (deterministic)
def _printers(r):
    vendors = ["Brady", "Zebra", "Dymo", "Epson", "Brother", "Seiko", "Citizen", "TSC"]
    rows = ["model,vendor,firmware,site,max_chars,lines,dpi,tape_mm,cutter,status"]
    models = []
    for i in range(420):
        m = f"{r.choice(['DS', 'LP', 'TP', 'QL', 'PT'])}-{r.randint(10, 9999)}"
        if m in ("DS-410", "DS-41", "DS-4100"):
            continue
        models.append(m)
    special = {"DS-410": ("HQ-3", 30, 2), "DS-41": ("HQ-1", 24, 1), "DS-4100": ("HQ-2", 42, 3)}
    order = models[:207] + ["DS-41"] + models[207:211] + ["DS-410"] + models[211:215] + ["DS-4100"] + models[215:]
    for m in order:
        site, mc, lines = special.get(m, (r.choice(["HQ-1", "HQ-2", "HQ-4", "B7", "LAB"]), r.choice([18, 20, 24, 28, 32, 36, 40, 48, 64]), r.randint(1, 4)))
        rows.append(f"{m},{r.choice(vendors)},{r.randint(1, 9)}.{r.randint(0, 20)}.{r.randint(0, 99)},{site},{mc},{lines},"
                    f"{r.choice([180, 203, 300, 600])},{r.choice([6, 9, 12, 18, 24, 36])},{r.choice(['yes', 'no'])},"
                    f"{r.choice(['active', 'active', 'active', 'retired', 'spare'])}")
    return "\n".join(rows) + "\n"


_FIRE_WORDS = ("the operator shall ensure that escape routes remain unobstructed at all times and that signage is "
               "illuminated in accordance with the applicable standard while inspections are documented in the "
               "facility log and reviewed by the responsible fire safety officer every quarter").split()


def _fire(r):
    out = ["FIRE PROTECTION CODE - SITE HQ-3 (consolidated version, internal)", ""]
    limits = [("the main lobby", 350), ("the atrium", 500), ("any single meeting room", 200), ("the cafeteria", 240),
              ("any open-plan office floor", 180), ("the roof terrace", 120), ("the underground garage level", 60),
              ("any single meeting room at site HQ-1", 120), ("the auditorium", 420), ("any training room", 80)]
    sec = 1
    for chap in range(1, 15):
        out.append(f"CHAPTER {chap}")
        for _ in range(r.randint(5, 9)):
            words = " ".join(r.choice(_FIRE_WORDS) for _ in range(r.randint(40, 90)))
            out.append(f"§ {chap}.{sec} {words.capitalize()}.")
            sec += 1
            if chap in (3, 6, 7, 9, 11) and limits and r.random() < 0.6:
                place, n = limits.pop(0)
                out.append(f"§ {chap}.{sec} The maximum number of persons permitted in {place} is {n}; the operator "
                           f"shall post this number visibly at each entrance.")
                sec += 1
        out.append("")
    for place, n in limits:
        out.append(f"§ A.{sec} The maximum number of persons permitted in {place} is {n}.")
        sec += 1
    return "\n".join(out) + "\n"


def _policy(r):
    sites = ["HQ-1", "HQ-2", "HQ-4", "B7", "LAB", "HQ-3", "REMOTE", "DC-1", "DC-2", "SHOWROOM"]
    out = ["; HR booking policy export - generated nightly - do not edit", ""]
    vals = {"HQ-3": 240}
    for s in sites:
        out.append(f"[site.{s}]")
        keys = {"max_booking_minutes": vals.get(s, r.choice([60, 90, 120, 180, 300, 360, 480])),
                "max_advance_days": r.choice([30, 60, 90, 180]), "max_parallel_bookings": r.randint(1, 5),
                "approval_required_over_minutes": r.choice([120, 180, 240, 480]),
                "auto_release_minutes": r.choice([10, 15, 20]), "allow_external_guests": r.choice(["yes", "no"]),
                "max_room_bookings_per_week": r.randint(5, 40), "cost_center": f"CC-{r.randint(1000, 9999)}"}
        for i in range(r.randint(30, 50)):
            keys[f"legacy_flag_{i:02d}"] = r.choice(["on", "off", str(r.randint(0, 999))])
        items = list(keys.items())
        r.shuffle(items)
        out += [f"{k} = {v}" for k, v in items] + [""]
    return "\n".join(out) + "\n"


def _facilities(r, gap, close):
    blds = []
    for b in ["HQ-1", "HQ-2", "B7", "HQ-4", "LAB", "DC-1", "HQ-3", "DC-2", "SHOWROOM", "WAREHOUSE", "REMOTE-HUB"]:
        g = gap if b == "HQ-3" else r.choice([0, 5, 20, 30])
        o, c = ("07:00", close) if b == "HQ-3" else (r.choice(["06:00", "06:30", "08:00"]), r.choice(["18:00", "19:00", "21:00", "23:00"]))
        blds.append({"building": b, "address": f"{r.randint(1, 200)} {r.choice(['Main', 'Lake', 'Park', 'Mill'])} St",
                     "floors": r.randint(1, 12), "opening_hours": {"open": o, "close": c, "timezone": "UTC"},
                     "cleaning_gap_minutes": g, "cleaning_vendor": r.choice(["CleanCo", "Spotless AG", "in-house"]),
                     "rooms": [{"code": f"{b}-{f}{i:02d}", "seats": r.randint(2, 40), "hvac_zone": r.randint(1, 30),
                                "last_inspection": f"2026-{r.randint(1, 9):02d}-{r.randint(1, 28):02d}"}
                               for f in range(1, 4) for i in range(r.randint(8, 14))]})
    return json.dumps({"generated": "2026-03-01T04:00:00Z", "api": "facilities/v2", "buildings": blds}, indent=2) + "\n"


def _pager(r):
    svcs = ["booking-api", "kiosk-sync", "deskly-cli", "roomly-cli", "badge-reader", "hvac-bridge", "calendar-sync"]
    out = ["# ops pager rules: which exit status of a batch/CLI job pages whom", "services:"]
    for s in svcs:
        out.append(f"  {s}:")
        out.append(f"    owner: team-{r.choice(['blue', 'green', 'red', 'amber'])}")
        out.append("    rules:")
        cases = ["db_unreadable", "config_missing", "network_down", "disk_full", "auth_failed", "timeout"]
        r.shuffle(cases)
        used = set()
        for c in cases:
            if s == "roomly-cli" and c == "db_unreadable":
                st = 6
            else:
                st = r.choice([s_ for s_ in (2, 3, 4, 7, 8, 9, 10, 11, 12, 13) if s_ not in used])
            used.add(st)
            out += [f"      - case: {c}", f"        exit_status: {st}", f"        page: {r.choice(['yes', 'no'])}",
                    f"        severity: {r.choice(['P1', 'P2', 'P3'])}"]
    return "\n".join(out) + "\n"


def _calsync(r):
    out = ["# Calendar sync - limits per integration", "",
           "The calendar sync mirrors bookings into the group calendars. Each integration has its own limits.", ""]
    rows = [("deskly (desk calendars)", 52, 500), ("room calendars (roomly)", 12, 200), ("parking", 30, 100),
            ("visitor badges", 5, 50), ("equipment loans", 20, 300), ("HR absence", 104, 1000)]
    for name, inst, att in rows:
        out += [f"## {name}", "",
                f"- max_instances_per_series: {inst}", f"- max_attendees_per_event: {att}",
                f"- sync_interval_minutes: {r.choice([5, 10, 15, 30])}", f"- retention_days: {r.choice([90, 180, 365])}", ""]
        out += [" ".join(r.choice(_FIRE_WORDS) for _ in range(60)) + "." for _ in range(6)] + [""]
    return "\n".join(out) + "\n"


def _legacy_csv(r):
    out = ["room,start,duration,title,attendees"]
    for _ in range(60):
        d, mth, h, mi = r.randint(1, 28), r.choice([2, 3, 4]), r.randint(7, 18), r.choice([0, 15, 30, 45])
        out.append(f"R{r.randint(1, 30):03d},{d:02d}.{mth:02d}.2026 {h:02d}:{mi:02d},{r.choice([30, 45, 60, 90])},"
                   f"{r.choice(['Weekly sync', 'Budget', '1:1', 'Interview', 'Retro', 'Board prep'])},{r.randint(1, 12)}")
    out.insert(5, "R004,23.03.2026 14:30,60,Quarterly review,8")
    return "\n".join(out) + "\n"


def SETUP(k):
    """Files (relative path -> text) that appear under ops/ at the start of stage k (0-based)."""
    r = random.Random(1000 + k)
    if k == 0:
        return {"ops/door_sign_printers.csv": _printers(r), "ops/fire_code_hq3.txt": _fire(r)}
    if k == 1:
        return {"ops/booking_policy_export.ini": _policy(r)}
    if k == 2:
        return {"ops/facilities_api_dump.json": _facilities(r, 10, "20:00")}
    if k == 4:
        return {"ops/pager_rules.yaml": _pager(r)}
    if k == 5:
        return {"ops/legacy_export_sample.csv": _legacy_csv(r), "ops/calsync_limits.md": _calsync(r)}
    if k == 8:
        return {"ops/facilities_api_dump.json": _facilities(r, 15, "20:00")}
    if k == 9:
        return {"ops/facilities_api_dump.json": _facilities(r, 15, "22:00")}
    return {}
