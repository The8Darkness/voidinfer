"""Long-horizon compaction-survival task: build `vxledger` over 12 stages.

The 12 project conventions appear ONLY in the first user message. Later stages refer to "the project
conventions" without repeating them, so after a context compaction the agent only knows them through its own
summary. Functional hidden tests (hidden/lh/test_functional.py) check what each stage message states;
the convention checker (hidden/lh/check_conventions.py) checks the 12 conventions.
"""

BRIEF = """We are building `vxledger`, a small ledger library with a command-line interface, in Python 3.12, in the current directory. The work comes in 12 stages; I will send each stage after you finish the previous one. When a stage is done (implemented and checked by you), reply with a bash block containing only `echo STAGE_COMPLETE`.

Project conventions. They apply to every stage and every file for the whole project. I will NOT repeat them later, so keep them in mind:
1. All code lives in the package `vxledger/` (import name `vxledger`).
2. Money is always integer cents. Never use float anywhere in vxledger (no float literals, no float() calls).
3. Every exception class derives from `VxError` (defined in vxledger/errors.py) and has an integer class attribute `code` in the range 4700-4799, unique per class.
4. Logging only through logging.getLogger("vx.ledger"), and every log message starts with "[VX] ".
5. Standard library only: no third-party packages in vxledger.
6. Every public function and method (name not starting with _) has type hints for all parameters and for the return value.
7. Timestamps are ISO-8601 UTC strings with seconds precision and a trailing "Z", e.g. 2026-03-01T12:00:00Z. Never use naive datetimes.
8. All file writes go through `atomic_write(path: str, data: str) -> None` in vxledger/storage.py, which writes a temporary file in the same directory and then calls os.replace.
9. The command-line interface exits with status 3 on any user or validation error (including bad arguments), and 0 on success.
10. Every .py file in vxledger starts with the line `# SPDX-License-Identifier: MPL-2.0`.
11. The default currency is CHF wherever a currency is optional.
12. Account ids are `ACC-` followed by a 6-digit zero-padded sequence number starting at 1 (ACC-000001, ACC-000002, ...).
"""

STAGES = [
# 1
"""Stage 1 - money helpers and storage.
Create vxledger/__init__.py, vxledger/errors.py, vxledger/money.py and vxledger/storage.py.
- `money.parse_amount(text: str) -> int` parses a decimal amount into cents. It accepts an optional leading '-', optional thousands separators ',' (only in correct groups of three digits) and 0 to 2 decimal digits. Surrounding whitespace is ignored. Examples: '12' -> 1200, '12.5' -> 1250, '-0.05' -> -5, '1,234.56' -> 123456, ' 7.00 ' -> 700. Anything else ('', '1.234', '12,34', 'abc', '1e3', '--1', '1,2345') raises `AmountError` (defined in errors.py).
- `money.format_amount(cents: int, currency: str | None = None) -> str` returns '<currency> <amount>' with thousands separators and exactly two decimals; negative amounts put the minus sign after the currency: format_amount(123456, 'EUR') -> 'EUR 1,234.56', format_amount(-1200, 'USD') -> 'USD -12.00', format_amount(5, 'EUR') -> 'EUR 0.05'. Without a currency it uses the project default.
- `storage.atomic_write` as described in the conventions.""",
# 2
"""Stage 2 - accounts.
Create vxledger/ledger.py with a class `Ledger` (also importable as `from vxledger import Ledger`).
- `Ledger()` creates an empty ledger.
- `open_account(self, name: str, currency: str | None = None) -> str` creates an account and returns its id. The name must be non-empty after stripping and unique case-insensitively among accounts, otherwise raise `ValidationError`. A given currency must be exactly 3 uppercase ASCII letters, otherwise `ValidationError`.
- `account(self, account_id: str) -> dict` returns {"id", "name", "currency", "balance"}; an unknown id raises `UnknownAccountError` (this applies to every method that takes an account id).
- `deposit(self, account_id: str, cents: int, memo: str = "") -> str` adds money and returns a transaction id: 'TX' followed by an 8-digit zero-padded sequence number that starts at 1 and is shared by all transactions of the ledger (TX00000001, TX00000002, ...). cents must be > 0, otherwise `ValidationError`.
- `withdraw(self, account_id: str, cents: int, memo: str = "") -> str` removes money, same id scheme and amount rule; the balance must not go below zero, otherwise `InsufficientFundsError` and nothing changes.
- `balance(self, account_id: str) -> int` returns the balance in cents.""",
# 3
"""Stage 3 - transfers and overdrafts.
- `transfer(self, src: str, dst: str, cents: int, memo: str = "") -> str` moves money and returns ONE transaction id for the whole transfer. src == dst raises `ValidationError`; cents must be > 0; both accounts must have the same currency, otherwise `CurrencyMismatchError`; if src would go below its allowed minimum, raise `InsufficientFundsError`. On any error nothing changes.
- `set_overdraft(self, account_id: str, limit_cents: int) -> None` sets how far below zero the account may go (default 0). limit_cents < 0 raises `ValidationError`. withdraw and transfer both respect the overdraft: the balance may go down to -limit_cents but not lower.""",
# 4
"""Stage 4 - history and a clock.
- The constructor becomes `Ledger(clock: Callable[[], datetime] | None = None)`. The clock returns a timezone-aware datetime and is used to timestamp every transaction; without it use the current UTC time.
- `history(self, account_id: str) -> list[dict]` returns the account's entries in chronological order. Each entry is a dict with exactly these keys: "tx" (transaction id), "kind" (one of 'deposit', 'withdrawal', 'transfer_in', 'transfer_out'), "amount" (signed cents: positive for deposit and transfer_in, negative for withdrawal and transfer_out), "balance_after" (the account balance right after the entry), "memo" and "timestamp" (in the project timestamp format). A transfer creates one entry in each of the two accounts with the same "tx".""",
# 5
"""Stage 5 - persistence.
- `save(self, path: str) -> None` writes the whole ledger as one JSON document whose top-level object has "version": 2. The rest of the structure is up to you, but loading must restore everything: accounts, balances, overdrafts, history and the id sequences (a new account or transaction after loading continues the numbering).
- `Ledger.load(path: str) -> "Ledger"` (a classmethod; it takes an optional `clock` keyword argument like the constructor) reads such a file. A missing file, invalid JSON or a version other than 2 raises `LoadError`.""",
# 6
"""Stage 6 - CSV import.
- `import_csv(self, path: str) -> dict` imports transactions from a CSV file with the header `date,account,amount,memo`. A positive amount is a deposit, a negative amount a withdrawal; amounts use the parse_amount format (quote amounts that contain commas). Rows are applied in file order. A row with a bad date, a bad amount, a zero amount, an unknown account or insufficient funds is skipped. The history timestamp of an imported row is its date (YYYY-MM-DD) at 00:00:00 UTC, in the project timestamp format.
- It returns {"imported": <rows applied>, "skipped": <rows skipped>, "errors": [[line_number, code], ...]} where line_number counts the header as line 1 and code is the integer code of the error that caused the skip (a bad date counts as a ValidationError).""",
# 7
"""Stage 7 - interest.
- `apply_interest(self, rate_bp: int, memo: str = "interest") -> list[str]` credits interest to every account with a positive balance: rate_bp is in basis points (1/100 of a percent); the credited amount is balance * rate_bp / 10000 rounded to whole cents with round-half-to-even, computed exactly. Accounts whose interest rounds to 0 get nothing. It returns the new transaction ids in account-id order. The history kind of these entries is 'interest' (amount positive). rate_bp < 0 raises ValidationError.""",
# 8
"""Stage 8 - monthly statements.
- `monthly_statement(self, account_id: str, year: int, month: int) -> str` returns this exact text (lines joined with "\\n", no trailing newline):
    Statement <id> (<name>) <YYYY-MM>
    Opening balance: <format_amount(opening, currency)>
    <one line per history entry whose timestamp falls in that month>
    Closing balance: <format_amount(closing, currency)>
  Each entry line is: `<YYYY-MM-DD> <tx> <kind> <sign><format_amount(abs(amount), currency)> -> <format_amount(balance_after, currency)>` followed by ` <memo>` only if the memo is non-empty. sign is '+' or '-'. Example: `2026-03-02 TX00000002 deposit +EUR 5.00 -> EUR 15.00 refund`.
  Opening is the balance_after of the last entry before the month (0 if none); closing is the balance_after of the last entry in the month, or the opening if there is none. month outside 1..12 raises ValidationError.""",
# 9
"""Stage 9 - command-line interface.
Create vxledger/__main__.py so that `python -m vxledger DB COMMAND ...` works. DB is the path of a ledger JSON file (stage 5 format); if it does not exist the CLI starts from an empty ledger. Every command that changes the ledger saves it back to DB.
Commands:
  open NAME [--currency CUR]        prints the new account id
  deposit ACC AMOUNT [--memo M]     prints the transaction id
  withdraw ACC AMOUNT [--memo M]    prints the transaction id
  transfer SRC DST AMOUNT [--memo M]  prints the transaction id
  balance ACC                       prints format_amount(balance, account currency)
  statement ACC YYYY-MM             prints the monthly statement
AMOUNT uses the parse_amount format. On an error print `error <code>: <message>` to stderr and exit with the project's error status.""",
# 10
"""Stage 10 - holds.
- `place_hold(self, account_id: str, cents: int, reason: str) -> str` reserves money and returns a hold id 'HOLD-<n>' (n starts at 1, ledger-wide). cents must be > 0 and may not exceed the available amount (InsufficientFundsError).
- `release_hold(self, hold_id: str) -> None` removes a hold; an unknown or already released hold id raises `UnknownHoldError`.
- `available(self, account_id: str) -> int` is balance + overdraft limit - sum of the account's active holds.
- From now on withdraw and transfer check the amount against available() instead of balance + overdraft. Holds are not transactions (no history entries, no TX ids) and must survive save/load.""",
# 11
"""Stage 11 - audit logging.
- Every successful mutating operation (open_account, deposit, withdraw, transfer, set_overdraft, place_hold, release_hold, each interest credit, each imported CSV row) emits exactly one INFO log record that contains the id it created or touched (account id, transaction id or hold id), following the project logging conventions.
- Every operation that fails with one of our errors emits one WARNING record that contains the error's integer code, then raises as before.""",
# 12
"""Stage 12 - search and final check.
- `search(self, text: str) -> list[dict]` returns all history entries of all accounts whose memo contains text (case-insensitive). Each result is the history entry dict plus an "account" key; results are sorted by transaction id and then account id. An empty text raises ValidationError.
- Finally, review the whole package once more: make sure every stage still works and that the code follows every project convention, then finish with STAGE_COMPLETE.""",
]

# regexes for "does this compaction summary still carry convention k" (case-insensitive)
RECALL = {
    "c01_package": r"vxledger/|package\W+.?vxledger",
    "c02_no_float": r"float",
    "c03_vxerror_codes": r"47\d\d",
    "c04_logging": r"\[VX\]",
    "c05_stdlib": r"standard[ -]library|stdlib|third[- ]party|no (external|extra) (packages|dependencies)",
    "c06_type_hints": r"type[- ]hint|annotat",
    "c07_utc_z": r"trailing\W+z|ending in\W+z|with\W+z\b|\dZ\b|'Z'|\"Z\"",
    "c08_atomic": r"atomic_write|os\.replace",
    "c09_exit3": r"(status|code|exit)\D{0,12}3\b",
    "c10_spdx": r"SPDX|MPL-2\.0",
    "c11_chf": r"\bCHF\b",
    "c12_acc_ids": r"ACC-0|ACC-\\d|ACC-#|ACC-\{|ACC-\s*\+?\s*6|ACC-<|ACC-\w*\s*\(6|ACC- followed|ACC-",
}
