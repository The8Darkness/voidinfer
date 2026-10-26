# roomly - internal use only
"""Command-line interface: python -m roomly DB COMMAND ..."""
import argparse
import sys

from roomly.errors import LoadError, RoomlyError
from roomly.scheduler import Scheduler
from roomly.timeutil import parse_duration


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        """Reports a usage error with the agreed exit status."""
        print(f"error validation: {message}", file=sys.stderr)
        sys.exit(5)


def main(argv: list[str] | None = None) -> int:
    """Runs one CLI command and returns the exit status."""
    p = _Parser(prog="roomly")
    p.add_argument("db")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    a = sub.add_parser("add-room"); a.add_argument("name"); a.add_argument("capacity")
    b = sub.add_parser("book"); b.add_argument("room"); b.add_argument("start")
    b.add_argument("--duration"); b.add_argument("--title", default=""); b.add_argument("--attendees", default="1")
    c = sub.add_parser("cancel"); c.add_argument("booking")
    ls = sub.add_parser("list"); ls.add_argument("room"); ls.add_argument("day")
    r = sub.add_parser("report"); r.add_argument("day")
    sub.add_parser("rooms")   # D18
    fd = sub.add_parser("find"); fd.add_argument("text")   # D27
    args = p.parse_args(argv)
    try:
        s = Scheduler.load(args.db)
    except LoadError as e:
        print(f"error {e.slug}: {e}", file=sys.stderr)
        return 6   # D17
    try:
        if args.cmd == "add-room":
            if not args.capacity.isdigit():
                print("error validation: bad capacity", file=sys.stderr)
                return 5
            print(s.add_room(args.name, int(args.capacity)))
        elif args.cmd == "book":
            if not args.attendees.isdigit():
                print("error validation: bad attendees", file=sys.stderr)
                return 5
            dur = parse_duration(args.duration) if args.duration is not None else None
            print(s.book(args.room, args.start, dur, args.title, int(args.attendees)))
        elif args.cmd == "cancel":
            s.cancel(args.booking)
        elif args.cmd == "list":
            for x in s.day_bookings(args.room, args.day):
                print(f"{x['id']} {x['start'][11:]}-{x['end'][11:]} {x['title']}")
            return 0
        elif args.cmd == "report":
            print(s.day_report(args.day))
            return 0
        elif args.cmd == "find":
            for x in s.find(args.text):
                print(f"{x['id']} {x['room']} {x['start']} {x['title']}")
            return 0
        elif args.cmd == "rooms":
            for x in s.rooms():
                print(f"{x['id']} {x['name']} {x['capacity']}")
            return 0
        s.save(args.db)
        return 0
    except RoomlyError as e:
        print(f"error {e.slug}: {e}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    sys.exit(main())
