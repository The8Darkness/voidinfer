# SPDX-License-Identifier: MPL-2.0
import re

from vxledger.errors import AmountError

DEFAULT_CURRENCY = "EUR"
_RE = re.compile(r"(-?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{1,2}))?")


def parse_amount(text: str) -> int:
    m = _RE.fullmatch(text.strip())
    if not m:
        raise AmountError(f"bad amount {text!r}")
    whole = int(m.group(2).replace(",", ""))
    frac = int((m.group(3) or "").ljust(2, "0"))
    v = whole * 100 + frac
    return -v if m.group(1) else v


def format_amount(cents: int, currency: str | None = None) -> str:
    cur = currency or DEFAULT_CURRENCY
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{cur} {sign}{c // 100:,}.{c % 100:02d}"
