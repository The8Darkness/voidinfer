# SPDX-License-Identifier: MPL-2.0
import csv
import json
import logging
import re
from datetime import datetime, timezone
from typing import Callable

from vxledger.errors import (CurrencyMismatchError, InsufficientFundsError, LoadError, UnknownAccountError,
                             UnknownHoldError, ValidationError, VxError)
from vxledger.money import DEFAULT_CURRENCY, format_amount, parse_amount
from vxledger.storage import atomic_write

log = logging.getLogger("vx.ledger")

MAX_NAME = 40          # D01
MAX_MEMO = 60          # D02
MAX_MOVE = 500000000   # D06 (superseded 100000000 -> 500000000 at stage 8)
MAX_HOLDS = 3          # D10
MAX_RESULTS = 25       # D20


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _memo(memo: str) -> str:
    return memo.strip()[:MAX_MEMO]


def _logged(fn):
    def _wrap(self, *a, **k):
        if getattr(self, "_depth", 0):
            return fn(self, *a, **k)
        self._depth = 1
        try:
            return fn(self, *a, **k)
        except VxError as e:
            log.warning("[VX] op=%s failed: error %d: %s", fn.__name__, e.code, e)
            raise
        finally:
            self._depth = 0
    _wrap.__name__ = fn.__name__
    return _wrap


class Ledger:
    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._acc: dict[str, dict] = {}
        self._hist: dict[str, list[dict]] = {}
        self._holds: dict[str, dict] = {}
        self._next_acc = 1
        self._next_tx = 1
        self._next_hold = 1
        self._depth = 0

    # ------------------------------------------------------------ helpers
    def _get(self, account_id: str) -> dict:
        if account_id not in self._acc:
            raise UnknownAccountError(f"unknown account {account_id}")
        return self._acc[account_id]

    def _tx(self) -> str:
        t = f"TX{self._next_tx:08d}"
        self._next_tx += 1
        return t

    def _entry(self, acc: dict, tx: str, kind: str, amount: int, memo: str, ts: str) -> None:
        acc["balance"] += amount
        self._hist[acc["id"]].append({"tx": tx, "kind": kind, "amount": amount, "balance_after": acc["balance"],
                                      "memo": _memo(memo), "timestamp": ts})

    def _now(self) -> str:
        return _ts(self._clock())

    def _check_debit(self, account_id: str, cents: int) -> None:
        if cents > self.available(account_id):
            raise InsufficientFundsError(f"insufficient funds in {account_id}")

    @staticmethod
    def _positive(cents: int) -> None:
        if not isinstance(cents, int) or cents <= 0:
            raise ValidationError("amount must be a positive number of cents")

    @staticmethod
    def _limit(cents: int) -> None:
        if cents > MAX_MOVE:
            raise ValidationError("amount exceeds the per-transaction limit")

    def _credit(self, account_id: str, cents: int, memo: str, ts: str, kind: str, limit: bool = True) -> str:
        a = self._get(account_id)
        self._positive(cents)
        if limit:
            self._limit(cents)
        tx = self._tx()
        self._entry(a, tx, kind, cents, memo, ts)
        log.info("[VX] %s %s %s into %s", kind, tx, format_amount(cents, a["currency"]), account_id)
        return tx

    def _debit(self, account_id: str, cents: int, memo: str, ts: str) -> str:
        a = self._get(account_id)
        self._positive(cents)
        self._limit(cents)
        self._check_debit(account_id, cents)
        tx = self._tx()
        self._entry(a, tx, "withdrawal", -cents, memo, ts)
        log.info("[VX] withdrawal %s %s from %s", tx, format_amount(cents, a["currency"]), account_id)
        return tx

    # ------------------------------------------------------------ accounts
    @_logged
    def open_account(self, name: str, currency: str | None = None) -> str:
        name = name.strip() if isinstance(name, str) else ""
        if not name or len(name) > MAX_NAME or any(a["name"].lower() == name.lower() for a in self._acc.values()):
            raise ValidationError("account name must be 1-40 characters and unique")
        cur = DEFAULT_CURRENCY if currency is None else currency
        if not re.fullmatch(r"[A-Z]{3}", cur):
            raise ValidationError(f"bad currency {cur!r}")
        aid = f"ACC-{self._next_acc:06d}"
        self._next_acc += 1
        self._acc[aid] = {"id": aid, "name": name, "currency": cur, "balance": 0, "overdraft": 0}
        self._hist[aid] = []
        log.info("[VX] opened %s", aid)
        return aid

    @_logged
    def account(self, account_id: str) -> dict:
        a = self._get(account_id)
        return {"id": a["id"], "name": a["name"], "currency": a["currency"], "balance": a["balance"]}

    @_logged
    def balance(self, account_id: str) -> int:
        return self._get(account_id)["balance"]

    @_logged
    def available(self, account_id: str) -> int:
        a = self._get(account_id)
        held = sum(h["cents"] for h in self._holds.values() if h["account"] == account_id)
        return a["balance"] + a["overdraft"] - held

    @_logged
    def deposit(self, account_id: str, cents: int, memo: str = "") -> str:
        return self._credit(account_id, cents, memo, self._now(), "deposit")

    @_logged
    def withdraw(self, account_id: str, cents: int, memo: str = "") -> str:
        return self._debit(account_id, cents, memo, self._now())

    @_logged
    def transfer(self, src: str, dst: str, cents: int, memo: str = "") -> str:
        s, d = self._get(src), self._get(dst)
        if src == dst:
            raise ValidationError("source and destination are the same")
        self._positive(cents)
        self._limit(cents)
        if s["currency"] != d["currency"]:
            raise CurrencyMismatchError("currencies differ")
        self._check_debit(src, cents)
        if not memo.strip():
            memo = f"xfer {src}>{dst}"  # D08 (superseded "transfer SRC->DST" at stage 10)
        tx, ts = self._tx(), self._now()
        self._entry(s, tx, "transfer_out", -cents, memo, ts)
        self._entry(d, tx, "transfer_in", cents, memo, ts)
        log.info("[VX] transfer %s %s %s -> %s", tx, format_amount(cents, s["currency"]), src, dst)
        return tx

    @_logged
    def set_overdraft(self, account_id: str, limit_cents: int) -> None:
        a = self._get(account_id)
        if limit_cents < 0:
            raise ValidationError("overdraft must be >= 0")
        a["overdraft"] = limit_cents
        log.info("[VX] overdraft of %s set to %d", account_id, limit_cents)

    @_logged
    def history(self, account_id: str) -> list[dict]:
        self._get(account_id)
        return [dict(e) for e in self._hist[account_id]]

    # ------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        doc = {"version": 2, "accounts": list(self._acc.values()), "history": self._hist, "holds": self._holds,
               "next": [self._next_acc, self._next_tx, self._next_hold]}
        atomic_write(path, json.dumps(doc, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str, clock: Callable[[], datetime] | None = None) -> "Ledger":
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("[VX] op=load failed: error %d", LoadError.code)
            raise LoadError(f"cannot load {path}: {e}") from e
        if not isinstance(doc, dict) or doc.get("version") != 2:
            log.warning("[VX] op=load failed: error %d", LoadError.code)
            raise LoadError("unsupported ledger version")
        led = cls(clock)
        led._acc = {a["id"]: a for a in doc["accounts"]}
        led._hist = doc["history"]
        led._holds = doc["holds"]
        led._next_acc, led._next_tx, led._next_hold = doc["next"]
        return led

    # ------------------------------------------------------------ import / interest / statements
    @staticmethod
    def _date(text: str) -> datetime:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):  # D07
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        raise ValidationError(f"bad date {text!r}")

    def import_csv(self, path: str) -> dict:
        imported, errors = 0, []
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        for i, row in enumerate(rows[1:], start=2):
            try:
                date, acc, amount, memo = (row + ["", "", "", ""])[:4]
                d = self._date(date)
                cents = parse_amount(amount)
                self._get(acc)
                if cents >= 0:
                    self._credit(acc, cents, memo, _ts(d), "deposit")
                else:
                    self._debit(acc, -cents, memo, _ts(d))
                imported += 1
            except VxError as e:
                log.warning("[VX] op=import_csv line %d skipped: error %d", i, e.code)
                errors.append([i, e.code])
        return {"imported": imported, "skipped": len(errors), "errors": errors}

    @_logged
    def apply_interest(self, rate_bp: int, memo: str = "interest") -> list[str]:
        if rate_bp < 0:
            raise ValidationError("rate must be >= 0")
        out = []
        for aid in sorted(self._acc):
            a = self._acc[aid]
            b = a["balance"]
            if b <= 0 or a["currency"] == "JPY":  # D05
                continue
            q, r = divmod(b * rate_bp, 10000)
            if 2 * r > 10000 or (2 * r == 10000 and q % 2 == 1):
                q += 1
            if q > 0:
                out.append(self._credit(aid, q, memo, self._now(), "interest", limit=False))
        return out

    @_logged
    def monthly_statement(self, account_id: str, year: int, month: int) -> str:
        a = self._get(account_id)
        if not 1 <= month <= 12:
            raise ValidationError("month must be 1..12")
        ym = f"{year:04d}-{month:02d}"
        cur = a["currency"]
        opening, lines, closing = 0, [], None
        for e in self._hist[account_id]:
            if e["timestamp"][:7] < ym:
                opening = e["balance_after"]
            elif e["timestamp"][:7] == ym:
                sign = "+" if e["amount"] >= 0 else "-"
                ln = (f"{e['timestamp'][:10]} {e['tx']} {e['kind']} {sign}{format_amount(abs(e['amount']), cur)}"
                      f" -> {format_amount(e['balance_after'], cur)}")
                lines.append(ln + (f" {e['memo']}" if e["memo"] else ""))
                closing = e["balance_after"]
        closing = opening if closing is None else closing
        return "\n".join([f"Statement {account_id} ({a['name'].upper()}) {ym}",
                          f"Opening balance: {format_amount(opening, cur)}",
                          *lines, f"Closing balance: {format_amount(closing, cur)}"])

    # ------------------------------------------------------------ holds
    @_logged
    def place_hold(self, account_id: str, cents: int, reason: str) -> str:
        self._get(account_id)
        if not reason.strip():  # D13
            raise ValidationError("hold reason must not be empty")
        self._positive(cents)
        if sum(1 for h in self._holds.values() if h["account"] == account_id) >= MAX_HOLDS:  # D10
            raise ValidationError("too many active holds")
        self._check_debit(account_id, cents)
        hid = f"HOLD-{self._next_hold:04d}"  # D19
        self._next_hold += 1
        self._holds[hid] = {"account": account_id, "cents": cents, "reason": reason}
        log.info("[VX] hold %s %d on %s", hid, cents, account_id)
        return hid

    @_logged
    def release_hold(self, hold_id: str) -> None:
        if hold_id not in self._holds:
            raise UnknownHoldError(f"unknown hold {hold_id}")
        del self._holds[hold_id]
        log.info("[VX] released %s", hold_id)

    @_logged
    def search(self, text: str) -> list[dict]:
        q = text.strip().lower()
        if len(q) < 2:  # D15
            raise ValidationError("search text must have at least 2 characters")
        out = [dict(e, account=aid, account_name=self._acc[aid]["name"])  # D18
               for aid, h in self._hist.items() for e in h if q in e["memo"].lower()]
        return sorted(out, key=lambda e: (e["tx"], e["account"]))[:MAX_RESULTS]
