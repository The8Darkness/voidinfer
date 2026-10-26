"""Long-horizon v2: `vxledger` over 12 stages, rules buried mid-session.

Unlike v1 there is no tidy convention list. 12 conventions and 20 team decisions arrive inside "team notes"
blocks of the stage messages (chatter, rejected alternatives, near-miss values), many of them apply only to a
LATER stage, and 5 are superseded later (the agent must carry the latest value). Stage texts only say "as
agreed" where a decision applies. With ~100K tokens of reference filler per stage the agent compacts before
almost every stage, so a rule given in stage k must survive up to 12-k handoff summaries.
Scoring: hidden/lh2/test_functional.py (stage behaviour), hidden/lh2/test_rules.py (one test per decision),
hidden/lh2/check_conventions.py (12 conventions, c11 = latest default currency EUR).
"""

BRIEF = """We are building `vxledger`, a small ledger library with a command-line interface, in Python 3.12, in the current directory. The work comes in 12 stages; I will send each stage after you finish the previous one. When a stage is done (implemented and checked by you), reply with a bash block containing only `echo STAGE_COMPLETE`.

Each stage message starts with an excerpt of our team chat since the previous stage. Conventions and decisions agreed there are binding for the whole project from then on (a later decision can change an earlier one). I will NOT repeat them later; the stage descriptions just say "as agreed". Keep track of them yourself."""

NOTES = [
# 1: c01 c02 c05 c08 c11(CHF) c12 D01 D02
"""[09:02] Priya: morning! kicking off vxledger today
[09:03] Maria (lead): Convention: all code lives in the package `vxledger/` (import name `vxledger`).
[09:05] Tom: for display can we just use floats and round at the end?
[09:06] Maria (lead): No. Convention: money is always integer cents. Never use float anywhere in vxledger (no float literals, no float() calls).
[09:08] Jonas: can I pull in `attrs` or `pydantic` for the models?
[09:09] Maria (lead): Convention: standard library only, no third-party packages in vxledger.
[09:11] Priya: storage - do we just open(path, "w")?
[09:12] Maria (lead): Convention: all file writes go through `atomic_write(path: str, data: str) -> None` in vxledger/storage.py, which writes a temporary file in the same directory and then calls os.replace.
[09:15] Tom: default currency EUR? most of our pilot users are in Germany
[09:16] Maria (lead): Convention: the default currency is CHF wherever a currency is optional. (we can revisit later)
[09:18] Jonas: account ids - ACC-1, ACC-2?
[09:19] Maria (lead): Convention: account ids are `ACC-` followed by a 6-digit zero-padded sequence number starting at 1 (ACC-000001, ACC-000002, ...).
[09:21] Priya: name length limit for accounts? 32 like the old system?
[09:22] Tom: 64 would be safer
[09:23] Maria (lead): Decision: account names may have at most 40 characters after stripping; longer names raise ValidationError.
[09:25] Jonas: and memos? the old engine truncated at 80
[09:26] Maria (lead): Decision: memos are stored stripped of surrounding whitespace and truncated to their first 60 characters. History, statements and search all show the stored memo.
[09:27] Priya: 👍""",
# 2: c03 c06 D03 D04 D05 D06(1,000,000.00 deposits)
"""[10:40] Jonas: exceptions - one big LedgerError?
[10:41] Maria (lead): Convention: every exception class derives from `VxError` (in vxledger/errors.py) and has an integer class attribute `code` in the range 4700-4799, unique per class.
[10:43] Maria (lead): Decision: these codes: AmountError 4701, ValidationError 4702, UnknownAccountError 4703, InsufficientFundsError 4704, CurrencyMismatchError 4705, LoadError 4706, UnknownHoldError 4707.
[10:44] Tom: type hints everywhere? even private helpers?
[10:45] Maria (lead): Convention: every public function and method (name not starting with _) has type hints for all parameters and for the return value. Private helpers are up to you.
[10:48] Priya: the JSON save file - ops wants to diff it in git
[10:49] Maria (lead): Decision: the saved ledger file is pretty-printed JSON with indent=2 and sort_keys=True, followed by one trailing newline.
[10:52] Jonas: interest on yen accounts is a mess legally
[10:53] Maria (lead): Decision: accounts in currency JPY never receive interest.
[10:55] Tom: should we cap deposits? fraud team asked for 10,000,000.00
[10:56] Priya: that's a lot, maybe 100,000.00?
[10:57] Maria (lead): Decision: a single deposit may not exceed 1,000,000.00 (100000000 cents); larger deposits raise ValidationError.""",
# 3: c04 c10 D07 D08(transfer SRC->DST)
"""[14:02] Tom: logging - print() is fine for now?
[14:03] Maria (lead): Convention: logging only through logging.getLogger("vx.ledger"), and every log message starts with "[VX] ".
[14:06] Jonas: license headers?
[14:07] Maria (lead): Convention: every .py file in vxledger starts with the line `# SPDX-License-Identifier: MPL-2.0`.
[14:10] Priya: the bank exports for the CSV import come with Swiss dates like 05.01.2026
[14:11] Tom: could we also take 01/05/2026?
[14:12] Maria (lead): Decision: the CSV import accepts dates as YYYY-MM-DD and as DD.MM.YYYY. Nothing else (no slashes).
[14:15] Jonas: empty memo on transfers looks bad in statements
[14:16] Maria (lead): Decision: a transfer without memo (empty or whitespace) gets the memo "transfer SRC->DST" with the two account ids, e.g. "transfer ACC-000001->ACC-000002".""",
# 4: c07 D09 D10
"""[11:20] Priya: timestamps - local time ok?
[11:21] Maria (lead): Convention: timestamps are ISO-8601 UTC strings with seconds precision and a trailing "Z", e.g. 2026-03-01T12:00:00Z. Never use naive datetimes.
[11:24] Tom: statements look odd with lowercase names
[11:25] Maria (lead): Decision: in the statement header the account name is printed in upper case.
[11:30] Jonas: thinking ahead about holds - unlimited per account?
[11:31] Priya: card schemes allow 5
[11:32] Maria (lead): Decision: at most 3 active holds per account; placing a 4th raises ValidationError.""",
# 5: c09 D11 D12
"""[16:01] Jonas: CLI exit codes - 1 for errors like everyone else?
[16:02] Maria (lead): Convention: the command-line interface exits with status 3 on any user or validation error (including bad arguments), and 0 on success.
[16:04] Tom: except when the DB file itself is broken, ops wants to alert on that separately
[16:05] Maria (lead): Decision: when the CLI cannot load the DB file (corrupt JSON, wrong version, unreadable) it prints the error line and exits with status 4 instead of 3.
[16:09] Priya: warnings in the logs don't say which call failed
[16:10] Maria (lead): Decision: every WARNING log record's message contains "op=<name of the public method that failed>", e.g. "op=withdraw".""",
# 6: D13 D14
"""[09:45] Priya: support saw holds with an empty reason
[09:46] Maria (lead): Decision: a hold reason must be non-empty after stripping, otherwise ValidationError.
[09:50] Jonas: CLI: can `open` set an overdraft directly?
[09:51] Maria (lead): Decision: the CLI `open` command accepts `--overdraft AMOUNT` (parse_amount format) and sets the overdraft of the new account.
[09:53] Tom: and a --limit flag for holds?
[09:54] Maria (lead): No, not now.""",
# 7: D03 superseded (4740/4770) D15 D16
"""[13:30] Tom: monitoring maps 4704 to something else already, clash!
[13:31] Maria (lead): Decision (changes the earlier code table): InsufficientFundsError is now 4740 and UnknownHoldError is now 4770. All other codes stay as they are.
[13:35] Priya: search with one letter returns half the ledger
[13:36] Maria (lead): Decision: search strips the query; a query shorter than 2 characters after stripping raises ValidationError.
[13:40] Jonas: CLI needs a way to see transactions
[13:41] Maria (lead): Decision: the CLI gets a command `history ACC` that prints one line per history entry: `<tx> <kind> <amount> <balance_after>`, amounts as plain integer cents (e.g. `TX00000001 deposit 500 500`).""",
# 8: D06 superseded (5,000,000.00, all movements) c11 superseded (EUR) D17
"""[10:10] Tom: the deposit cap is too low for corporate clients
[10:11] Priya: 2,500,000.00?
[10:12] Maria (lead): Decision (changes the earlier cap): the limit is now 5,000,000.00 (500000000 cents) and it applies to every single deposit, withdrawal and transfer, including imported CSV rows (an over-limit row is skipped with ValidationError).
[10:15] Jonas: pilot moved from Zurich to Frankfurt
[10:16] Maria (lead): Convention change: the default currency is now EUR wherever a currency is optional (no longer CHF).
[10:20] Priya: audit wants amounts in the info logs
[10:21] Maria (lead): Decision: the INFO log record of every deposit, withdrawal and transfer contains the moved amount formatted with format_amount(cents, account currency), e.g. "EUR 12.34".""",
# 9: D18 D19
"""[15:00] Jonas: search results only have the account id, UI wants the name
[15:01] Maria (lead): Decision: every search result also has the key "account_name" with the account's name.
[15:05] Tom: hold ids should sort nicely
[15:06] Priya: HOLD-00001?
[15:07] Maria (lead): Decision: hold ids are "HOLD-" followed by a 4-digit zero-padded number: HOLD-0001, HOLD-0002, ...""",
# 10: D08 superseded (xfer SRC>DST) D20
"""[11:11] Tom: "transfer ACC-...->ACC-..." memos are too long for the bank's 35-char field
[11:12] Maria (lead): Decision (changes the earlier default): a transfer without memo now gets "xfer SRC>DST", e.g. "xfer ACC-000001>ACC-000002".
[11:15] Priya: search on big ledgers returns thousands of rows
[11:16] Maria (lead): Decision: search returns at most 25 results (the first 25 in the sorted order).""",
# 11: chatter only
"""[14:40] Jonas: should we add a `close_account`?
[14:41] Maria (lead): Not in this project.
[14:44] Tom: maybe raise the hold limit to 5?
[14:45] Maria (lead): No, keep what we decided.""",
# 12: chatter only
"""[17:00] Priya: last stage! release on Monday
[17:01] Maria (lead): Please make sure everything we agreed on in this chat is implemented before you hand it in.""",
]

STAGES = [
# 1
"""Stage 1 - money helpers and storage.
Create vxledger/__init__.py, vxledger/errors.py, vxledger/money.py and vxledger/storage.py.
- `money.parse_amount(text: str) -> int` parses a decimal amount into cents. It accepts an optional leading '-', optional thousands separators ',' (only in correct groups of three digits) and 0 to 2 decimal digits. Surrounding whitespace is ignored. Examples: '12' -> 1200, '12.5' -> 1250, '-0.05' -> -5, '1,234.56' -> 123456, ' 7.00 ' -> 700. Anything else ('', '1.234', '12,34', 'abc', '1e3', '--1', '1,2345') raises `AmountError` (defined in errors.py).
- `money.format_amount(cents: int, currency: str | None = None) -> str` returns '<currency> <amount>' with thousands separators and exactly two decimals; negative amounts put the minus sign after the currency: format_amount(123456, 'EUR') -> 'EUR 1,234.56', format_amount(-1200, 'USD') -> 'USD -12.00', format_amount(5, 'EUR') -> 'EUR 0.05'. Without a currency it uses the project default currency.
- `storage.atomic_write` as agreed.""",
# 2
"""Stage 2 - accounts.
Create vxledger/ledger.py with a class `Ledger` (also importable as `from vxledger import Ledger`).
- `Ledger()` creates an empty ledger.
- `open_account(self, name: str, currency: str | None = None) -> str` creates an account and returns its id (format as agreed). The name must be non-empty after stripping, within the agreed length and unique case-insensitively among accounts, otherwise raise `ValidationError`. A given currency must be exactly 3 uppercase ASCII letters, otherwise `ValidationError`.
- `account(self, account_id: str) -> dict` returns {"id", "name", "currency", "balance"}; an unknown id raises `UnknownAccountError` (this applies to every method that takes an account id).
- `deposit(self, account_id: str, cents: int, memo: str = "") -> str` adds money and returns a transaction id: 'TX' followed by an 8-digit zero-padded sequence number that starts at 1 and is shared by all transactions of the ledger (TX00000001, TX00000002, ...). cents must be > 0 and within the agreed limit, otherwise `ValidationError`.
- `withdraw(self, account_id: str, cents: int, memo: str = "") -> str` removes money, same id scheme; cents must be > 0 (`ValidationError`); the balance must not go below zero, otherwise `InsufficientFundsError` and nothing changes.
- `balance(self, account_id: str) -> int` returns the balance in cents.
- Error classes and codes as agreed.""",
# 3
"""Stage 3 - transfers and overdrafts.
- `transfer(self, src: str, dst: str, cents: int, memo: str = "") -> str` moves money and returns ONE transaction id for the whole transfer. src == dst raises `ValidationError`; cents must be > 0; both accounts must have the same currency, otherwise `CurrencyMismatchError`; if src would go below its allowed minimum, raise `InsufficientFundsError`. On any error nothing changes. Memo handling as agreed.
- `set_overdraft(self, account_id: str, limit_cents: int) -> None` sets how far below zero the account may go (default 0). limit_cents < 0 raises `ValidationError`. withdraw and transfer both respect the overdraft: the balance may go down to -limit_cents but not lower.""",
# 4
"""Stage 4 - history and a clock.
- The constructor becomes `Ledger(clock: Callable[[], datetime] | None = None)`. The clock returns a timezone-aware datetime and is used to timestamp every transaction; without it use the current UTC time.
- `history(self, account_id: str) -> list[dict]` returns the account's entries in chronological order. Each entry is a dict with exactly these keys: "tx" (transaction id), "kind" (one of 'deposit', 'withdrawal', 'transfer_in', 'transfer_out'), "amount" (signed cents: positive for deposit and transfer_in, negative for withdrawal and transfer_out), "balance_after" (the account balance right after the entry), "memo" (as agreed) and "timestamp" (as agreed). A transfer creates one entry in each of the two accounts with the same "tx".""",
# 5
"""Stage 5 - persistence.
- `save(self, path: str) -> None` writes the whole ledger as one JSON document (file format as agreed) whose top-level object has "version": 2. The rest of the structure is up to you, but loading must restore everything: accounts, balances, overdrafts, history and the id sequences (a new account or transaction after loading continues the numbering).
- `Ledger.load(path: str) -> "Ledger"` (a classmethod; it takes an optional `clock` keyword argument like the constructor) reads such a file. A missing file, invalid JSON or a version other than 2 raises `LoadError`.""",
# 6
"""Stage 6 - CSV import.
- `import_csv(self, path: str) -> dict` imports transactions from a CSV file with the header `date,account,amount,memo`. A positive amount is a deposit, a negative amount a withdrawal; amounts use the parse_amount format (quote amounts that contain commas). Rows are applied in file order. A row with a bad date (date formats as agreed), a bad amount, a zero amount, an unknown account or insufficient funds is skipped. The history timestamp of an imported row is its date at 00:00:00 UTC, in the project timestamp format.
- It returns {"imported": <rows applied>, "skipped": <rows skipped>, "errors": [[line_number, code], ...]} where line_number counts the header as line 1 and code is the integer code of the error that caused the skip (a bad date counts as a ValidationError).""",
# 7
"""Stage 7 - interest.
- `apply_interest(self, rate_bp: int, memo: str = "interest") -> list[str]` credits interest to every account with a positive balance, except as agreed: rate_bp is in basis points (1/100 of a percent); the credited amount is balance * rate_bp / 10000 rounded to whole cents with round-half-to-even, computed exactly. Accounts whose interest rounds to 0 get nothing. It returns the new transaction ids in account-id order. The history kind of these entries is 'interest' (amount positive). rate_bp < 0 raises ValidationError. Interest credits are not subject to the deposit limit.""",
# 8
"""Stage 8 - monthly statements.
- `monthly_statement(self, account_id: str, year: int, month: int) -> str` returns this exact text (lines joined with "\\n", no trailing newline):
    Statement <id> (<name, as agreed>) <YYYY-MM>
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
plus any commands and options we agreed on in the team chat. AMOUNT uses the parse_amount format. On an error print `error <code>: <message>` to stderr and exit with the agreed status.""",
# 10
"""Stage 10 - holds.
- `place_hold(self, account_id: str, cents: int, reason: str) -> str` reserves money and returns a hold id (format as agreed; the number starts at 1, ledger-wide). cents must be > 0 and may not exceed the available amount (InsufficientFundsError). Other limits on holds as agreed.
- `release_hold(self, hold_id: str) -> None` removes a hold; an unknown or already released hold id raises `UnknownHoldError`.
- `available(self, account_id: str) -> int` is balance + overdraft limit - sum of the account's active holds.
- From now on withdraw and transfer check the amount against available() instead of balance + overdraft. Holds are not transactions (no history entries, no TX ids) and must survive save/load.""",
# 11
"""Stage 11 - audit logging.
- Every successful mutating operation (open_account, deposit, withdraw, transfer, set_overdraft, place_hold, release_hold, each interest credit, each imported CSV row) emits exactly one INFO log record that contains the id it created or touched (account id, transaction id or hold id), following the logging conventions and the agreed message contents.
- Every operation that fails with one of our errors emits one WARNING record that contains the error's integer code (and the agreed content), then raises as before.""",
# 12
"""Stage 12 - search and final check.
- `search(self, text: str) -> list[dict]` returns all history entries of all accounts whose memo contains the query (case-insensitive). Each result is the history entry dict plus an "account" key (and the other agreed fields); results are sorted by transaction id and then account id. Query rules and result limit as agreed.
- Finally, review the whole package once more: make sure every stage still works and that the code follows every convention and decision we agreed on, then finish with STAGE_COMPLETE.""",
]

# Recall of the LATEST value of each item in a handoff summary (case-insensitive). Items are only scored in
# summaries written after the stage that introduced them (ITEM_STAGE, 1-based).
RECALL = {
    "c01_package": r"vxledger/|package\W+.?vxledger",
    "c02_no_float": r"float",
    "c03_vxerror_codes": r"VxError",
    "c04_logging": r"\[VX\]",
    "c05_stdlib": r"standard[ -]library|stdlib|third[- ]party|no (external|extra) (packages|dependencies)",
    "c06_type_hints": r"type[- ]hint|annotat",
    "c07_utc_z": r"trailing\W+z|ending in\W+z|with\W+z\b|\dZ\b|'Z'|\"Z\"",
    "c08_atomic": r"atomic_write|os\.replace",
    "c09_exit3": r"(status|code|exit)\D{0,12}3\b",
    "c10_spdx": r"SPDX|MPL-2\.0",
    "c11_default_eur": r"default\W+(currency\W+)?(is\W+)?(now\W+)?EUR|EUR\W+(is\W+)?(now\W+)?(the\W+)?default",
    "c12_acc_ids": r"ACC-0|ACC-\\d|ACC-\{|ACC-<|6-digit|six-digit",
    "d01_name40": r"\b40\b",
    "d02_memo60": r"\b60\b",
    "d03_codes": r"4740(?=[\s\S]*4770)|4770(?=[\s\S]*4740)",
    "d04_json_format": r"sort_keys|sorted keys",
    "d05_jpy": r"JPY|yen",
    "d06_limit": r"5,000,000|500000000|500,000,000|5 000 000|5M\b",
    "d07_csv_dates": r"DD\.MM\.YYYY|%d\.%m\.%Y",
    "d08_xfer": r"xfer",
    "d09_upper": r"upper",
    "d10_max3holds": r"(\b3\b|three)[^\n]{0,40}hold|hold[^\n]{0,40}(\b3\b|three)",
    "d11_exit4": r"(status|exit|code)[^\n]{0,20}\b4\b",
    "d12_op": r"op=",
    "d13_reason": r"reason[^\n]{0,60}(empty|strip)|(empty|strip)[^\n]{0,60}reason",
    "d14_overdraft_flag": r"--overdraft",
    "d15_min2": r"\b2 char|at least 2|two char|shorter than 2|< ?2\b|less than 2",
    "d16_history_cmd": r"history ACC|`history`|history command|command `?history",
    "d17_info_amount": r"format_amount[^\n]{0,80}(info|log)|(info|log)[^\n]{0,80}(format_amount|amount)",
    "d18_account_name": r"account_name",
    "d19_hold_ids": r"HOLD-0001|HOLD-\{?:?04d|4-digit|four-digit",
    "d20_max25": r"\b25\b",
}
ITEM_STAGE = {"c01_package": 1, "c02_no_float": 1, "c05_stdlib": 1, "c08_atomic": 1, "c11_default_eur": 8,
              "c12_acc_ids": 1, "d01_name40": 1, "d02_memo60": 1, "c03_vxerror_codes": 2, "c06_type_hints": 2,
              "d03_codes": 7, "d04_json_format": 2, "d05_jpy": 2, "d06_limit": 8, "c04_logging": 3, "c10_spdx": 3,
              "d07_csv_dates": 3, "d08_xfer": 10, "c07_utc_z": 4, "d09_upper": 4, "d10_max3holds": 4,
              "c09_exit3": 5, "d11_exit4": 5, "d12_op": 5, "d13_reason": 6, "d14_overdraft_flag": 6,
              "d15_min2": 7, "d16_history_cmd": 7, "d17_info_amount": 8, "d18_account_name": 9,
              "d19_hold_ids": 9, "d20_max25": 10}
# Stale values of superseded items (present without the new value = the summary carries the old rule).
STALE = {"d03_codes": r"4704|4707", "d06_limit": r"1,000,000|100000000", "d08_xfer": r"transfer SRC->DST|transfer ACC-\S+->",
         "c11_default_eur": r"default\W+(currency\W+)?(is\W+)?CHF|CHF\W+(is\W+)?(the\W+)?default"}
