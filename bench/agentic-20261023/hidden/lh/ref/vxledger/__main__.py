# SPDX-License-Identifier: MPL-2.0
import argparse
import os
import sys

from vxledger.errors import VxError
from vxledger.ledger import Ledger
from vxledger.money import format_amount, parse_amount


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(f"error 4702: {message}", file=sys.stderr)
        sys.exit(3)


def main(argv: list[str] | None = None) -> int:
    p = _Parser(prog="vxledger")
    p.add_argument("db")
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    o = sub.add_parser("open"); o.add_argument("name"); o.add_argument("--currency")
    for c in ("deposit", "withdraw"):
        x = sub.add_parser(c); x.add_argument("acc"); x.add_argument("amount"); x.add_argument("--memo", default="")
    t = sub.add_parser("transfer"); t.add_argument("src"); t.add_argument("dst"); t.add_argument("amount")
    t.add_argument("--memo", default="")
    b = sub.add_parser("balance"); b.add_argument("acc")
    s = sub.add_parser("statement"); s.add_argument("acc"); s.add_argument("ym")
    a = p.parse_args(argv)
    try:
        led = Ledger.load(a.db) if os.path.exists(a.db) else Ledger()
        if a.cmd == "open":
            print(led.open_account(a.name, a.currency))
        elif a.cmd in ("deposit", "withdraw"):
            print(getattr(led, a.cmd)(a.acc, parse_amount(a.amount), a.memo))
        elif a.cmd == "transfer":
            print(led.transfer(a.src, a.dst, parse_amount(a.amount), a.memo))
        elif a.cmd == "balance":
            print(format_amount(led.balance(a.acc), led.account(a.acc)["currency"]))
            return 0
        elif a.cmd == "statement":
            try:
                y, m = (int(v) for v in a.ym.split("-"))
            except ValueError:
                print("error 4702: bad month", file=sys.stderr)
                return 3
            print(led.monthly_statement(a.acc, y, m))
            return 0
        led.save(a.db)
        return 0
    except VxError as e:
        print(f"error {e.code}: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
