#!/usr/bin/env python3
"""Durable SQLite-backed experiment queue with leased state transitions."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as datetime_module
import json
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

ACTIVE_STATES = frozenset(
    {
        "CHECKPOINTED",
        "SERVICE_STOPPING",
        "GPU_FREE",
        "BUILDING",
        "TESTING",
        "BENCHMARKING",
        "PROFILING",
        "COLLECTING",
        "RESTORING",
        "HEALTHY",
        "RESUME_PENDING",
    }
)
TERMINAL_STATES = frozenset({"DONE", "FAILED_BLOCKED"})
RETRYABLE_STATES = frozenset({"IDLE", "FAILED_RECOVERABLE"})
ALL_STATES = frozenset({"IDLE", *ACTIVE_STATES, *TERMINAL_STATES, "FAILED_RECOVERABLE"})

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "IDLE": frozenset({"CHECKPOINTED", "FAILED_BLOCKED"}),
    "CHECKPOINTED": frozenset({"SERVICE_STOPPING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "SERVICE_STOPPING": frozenset({"GPU_FREE", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "GPU_FREE": frozenset({"BUILDING", "BENCHMARKING", "PROFILING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "BUILDING": frozenset({"TESTING", "COLLECTING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "TESTING": frozenset({"BENCHMARKING", "PROFILING", "COLLECTING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "BENCHMARKING": frozenset({"COLLECTING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "PROFILING": frozenset({"COLLECTING", "RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "COLLECTING": frozenset({"RESTORING", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "RESTORING": frozenset({"HEALTHY", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "HEALTHY": frozenset({"RESUME_PENDING", "DONE", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "RESUME_PENDING": frozenset({"DONE", "FAILED_RECOVERABLE", "FAILED_BLOCKED"}),
    "FAILED_RECOVERABLE": frozenset({"CHECKPOINTED", "FAILED_BLOCKED"}),
    "FAILED_BLOCKED": frozenset(),
    "DONE": frozenset(),
}

REQUIRED_MANIFEST_FIELDS = (
    "run_id",
    "revision",
    "hypothesis",
    "baseline_run_ids",
    "hard_gates",
    "success_metric",
    "max_gpu_hours",
    "rollback",
    "owner",
    "artifact",
    "corpus",
    "environment",
    "metric_schema",
    "commands",
    "result_paths",
)


class QueueError(RuntimeError):
    """Raised when a queue operation violates its durable contract."""


@dataclasses.dataclass(frozen=True)
class Job:
    id: int
    status: str
    priority: int
    owner: str | None
    lease_expires_at: str | None
    attempts: int
    manifest: dict[str, Any]
    created_at: str
    updated_at: str
    error: str | None = None


@dataclasses.dataclass(frozen=True)
class JournalEntry:
    job_id: int
    from_status: str | None
    to_status: str
    owner: str | None
    note: str | None
    created_at: str


def utc_now() -> str:
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(timespec="milliseconds")


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise QueueError(f"experiment manifest is missing required fields: {', '.join(missing)}")
    for field in ("run_id", "revision", "hypothesis", "success_metric", "rollback", "owner", "metric_schema"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise QueueError(f"{field} must be a non-empty string")
    if not isinstance(manifest["baseline_run_ids"], list):
        raise QueueError("baseline_run_ids must be a JSON array")
    if not isinstance(manifest["hard_gates"], list) or not manifest["hard_gates"]:
        raise QueueError("hard_gates must be a non-empty JSON array")
    if not isinstance(manifest["artifact"], (str, dict)) or not manifest["artifact"]:
        raise QueueError("artifact must identify the benchmark artifact")
    if not isinstance(manifest["corpus"], (str, dict)) or not manifest["corpus"]:
        raise QueueError("corpus must identify the benchmark corpus")
    if not isinstance(manifest["environment"], dict):
        raise QueueError("environment must be a JSON object")
    if not isinstance(manifest["commands"], list) or not manifest["commands"]:
        raise QueueError("commands must be a non-empty JSON array")
    for command in manifest["commands"]:
        argv = command.get("argv") if isinstance(command, dict) else command
        if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
            raise QueueError("each command must be a non-empty argument array")
    if not isinstance(manifest["result_paths"], list):
        raise QueueError("result_paths must be a JSON array")
    if not isinstance(manifest["max_gpu_hours"], (int, float)) or manifest["max_gpu_hours"] <= 0:
        raise QueueError("max_gpu_hours must be positive")
    try:
        value = json.loads(json.dumps(dict(manifest), sort_keys=True))
    except (TypeError, ValueError) as error:
        raise QueueError(f"experiment manifest is not JSON-serializable: {error}") from error
    if not isinstance(value, dict):
        raise QueueError("experiment manifest must serialize to an object")
    return value


class ExperimentQueue:
    """A single-writer-safe SQLite queue for project-owned GPU experiments."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ExperimentQueue:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL CHECK (status IN (
                    'IDLE', 'CHECKPOINTED', 'SERVICE_STOPPING', 'GPU_FREE',
                    'BUILDING', 'TESTING', 'BENCHMARKING', 'PROFILING',
                    'COLLECTING', 'RESTORING', 'HEALTHY', 'RESUME_PENDING',
                    'DONE', 'FAILED_RECOVERABLE', 'FAILED_BLOCKED'
                )),
                priority INTEGER NOT NULL DEFAULT 100,
                owner TEXT,
                lease_expires_at TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                manifest_json TEXT NOT NULL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS jobs_claim_idx
                ON jobs(status, priority DESC, created_at ASC);
            CREATE TABLE IF NOT EXISTS journal (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL REFERENCES jobs(id),
                from_status TEXT,
                to_status TEXT NOT NULL,
                owner TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS journal_job_idx ON journal(job_id, sequence);
            """
        )

    @staticmethod
    def _row_int(row: sqlite3.Row, column: str) -> int:
        value = row[column]
        try:
            return int(value)
        except (TypeError, ValueError) as error:
            raise QueueError(f"queue row has invalid integer {column}: {value!r}") from error

    @staticmethod
    def _row_manifest(row: sqlite3.Row) -> dict[str, Any]:
        try:
            value = json.loads(row["manifest_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise QueueError(f"queue row has invalid manifest JSON: {error}") from error
        if not isinstance(value, dict):
            raise QueueError("queue row manifest is not an object")
        return value

    @classmethod
    def _row_to_job(cls, row: sqlite3.Row) -> Job:
        return Job(
            id=cls._row_int(row, "id"),
            status=str(row["status"]),
            priority=cls._row_int(row, "priority"),
            owner=row["owner"],
            lease_expires_at=row["lease_expires_at"],
            attempts=cls._row_int(row, "attempts"),
            manifest=cls._row_manifest(row),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error=row["error"],
        )

    def enqueue(self, manifest: Mapping[str, Any], priority: int = 100) -> Job:
        checked = validate_manifest(manifest)
        now = utc_now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._connection.execute(
                """
                INSERT INTO jobs(status, priority, manifest_json, created_at, updated_at)
                VALUES ('IDLE', ?, ?, ?, ?)
                """,
                (priority, json.dumps(checked, separators=(",", ":")), now, now),
            )
            if cursor.lastrowid is None:
                raise QueueError("SQLite did not return an experiment job id")
            job_id = cursor.lastrowid
            self._append_journal(job_id, None, "IDLE", None, "enqueued")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(job_id)

    def get(self, job_id: int) -> Job:
        row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise QueueError(f"unknown experiment job {job_id}")
        return self._row_to_job(row)

    def list(self, statuses: Iterable[str] | None = None) -> list[Job]:
        values = tuple(statuses or ())
        if any(status not in ALL_STATES for status in values):
            raise QueueError("unknown job status")
        rows = self._connection.execute(
            "SELECT * FROM jobs ORDER BY priority DESC, id ASC"
        ).fetchall()
        if values:
            selected = set(values)
            rows = [row for row in rows if row["status"] in selected]
        return [self._row_to_job(row) for row in rows]

    def claim(self, owner: str, lease_seconds: int = 3600) -> Job | None:
        if not owner:
            raise QueueError("owner is required to claim a job")
        if lease_seconds <= 0:
            raise QueueError("lease_seconds must be positive")
        now = utc_now()
        expiry = (
            datetime_module.datetime.now(datetime_module.timezone.utc)
            + datetime_module.timedelta(seconds=lease_seconds)
        ).isoformat(timespec="milliseconds")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT id, status FROM jobs
                WHERE status IN ('IDLE', 'FAILED_RECOVERABLE')
                ORDER BY priority DESC, id ASC
                LIMIT 1
                """
            ).fetchone()
            if row is None:
                self._connection.execute("COMMIT")
                return None
            job_id = self._row_int(row, "id")
            from_status = str(row["status"])
            changed = self._connection.execute(
                """
                UPDATE jobs
                SET status = 'CHECKPOINTED', owner = ?, lease_expires_at = ?,
                    attempts = attempts + 1, error = NULL, updated_at = ?
                WHERE id = ? AND status IN ('IDLE', 'FAILED_RECOVERABLE')
                """,
                (owner, expiry, now, job_id),
            ).rowcount
            if changed != 1:
                self._connection.execute("ROLLBACK")
                return None
            self._append_journal(job_id, from_status, "CHECKPOINTED", owner, "claimed")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(job_id)

    def transition(
        self,
        job_id: int,
        target: str,
        owner: str,
        note: str | None = None,
        lease_seconds: int = 3600,
    ) -> Job:
        if target not in ALL_STATES:
            raise QueueError(f"unknown target status {target}")
        if not owner:
            raise QueueError("owner is required for a transition")
        current = self.get(job_id)
        if target not in ALLOWED_TRANSITIONS[current.status]:
            raise QueueError(f"invalid transition {current.status} -> {target}")
        if current.owner != owner:
            raise QueueError(f"job {job_id} is owned by {current.owner!r}, not {owner!r}")
        now = utc_now()
        expiry = None if target in TERMINAL_STATES else (
            datetime_module.datetime.now(datetime_module.timezone.utc)
            + datetime_module.timedelta(seconds=lease_seconds)
        ).isoformat(timespec="milliseconds")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self._connection.execute(
                """
                UPDATE jobs SET status = ?, lease_expires_at = ?, updated_at = ?, error = ?
                WHERE id = ? AND status = ? AND owner = ?
                """,
                (target, expiry, now, note if target.startswith("FAILED") else None, job_id, current.status, owner),
            ).rowcount
            if changed != 1:
                raise QueueError(f"job {job_id} changed before transition could commit")
            self._append_journal(job_id, current.status, target, owner, note)
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(job_id)

    def renew(self, job_id: int, owner: str, lease_seconds: int = 3600) -> Job:
        current = self.get(job_id)
        if current.owner != owner:
            raise QueueError(f"job {job_id} is owned by {current.owner!r}, not {owner!r}")
        if current.status in TERMINAL_STATES:
            raise QueueError(f"cannot renew terminal job {job_id}")
        expiry = (
            datetime_module.datetime.now(datetime_module.timezone.utc)
            + datetime_module.timedelta(seconds=lease_seconds)
        ).isoformat(timespec="milliseconds")
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self._connection.execute(
                "UPDATE jobs SET lease_expires_at = ?, updated_at = ? WHERE id = ? AND owner = ?",
                (expiry, utc_now(), job_id, owner),
            ).rowcount
            if changed != 1:
                raise QueueError(f"job {job_id} changed before lease renewal could commit")
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        return self.get(job_id)

    def recover_expired(self, now: datetime_module.datetime | None = None) -> list[int]:
        moment = now or datetime_module.datetime.now(datetime_module.timezone.utc)
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            rows = self._connection.execute(
                "SELECT id, status, owner, lease_expires_at FROM jobs WHERE status NOT IN ('DONE', 'FAILED_BLOCKED') AND lease_expires_at IS NOT NULL"
            ).fetchall()
            recovered: list[int] = []
            for row in rows:
                expiry = datetime_module.datetime.fromisoformat(str(row["lease_expires_at"]))
                if expiry >= moment:
                    continue
                job_id = self._row_int(row, "id")
                owner = row["owner"]
                from_status = str(row["status"])
                lease_expires_at = row["lease_expires_at"]
                changed = self._connection.execute(
                    "UPDATE jobs SET status = 'FAILED_RECOVERABLE', lease_expires_at = NULL, updated_at = ?, error = ? WHERE id = ? AND status = ? AND lease_expires_at = ?",
                    (utc_now(), "lease expired; supervisor recovery required", job_id, from_status, lease_expires_at),
                ).rowcount
                if changed == 1:
                    self._append_journal(job_id, from_status, "FAILED_RECOVERABLE", owner, "lease expired")
                    recovered.append(job_id)
            self._connection.execute("COMMIT")
            return recovered
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def journal(self, job_id: int) -> list[JournalEntry]:
        rows = self._connection.execute(
            "SELECT job_id, from_status, to_status, owner, note, created_at FROM journal WHERE job_id = ? ORDER BY sequence",
            (job_id,),
        ).fetchall()
        return [
            JournalEntry(
                job_id=self._row_int(row, "job_id"),
                from_status=row["from_status"],
                to_status=str(row["to_status"]),
                owner=row["owner"],
                note=row["note"],
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

    def _append_journal(
        self,
        job_id: int,
        from_status: str | None,
        to_status: str,
        owner: str | None,
        note: str | None,
    ) -> None:
        self._connection.execute(
            "INSERT INTO journal(job_id, from_status, to_status, owner, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, from_status, to_status, owner, note, utc_now()),
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QueueError(f"could not read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise QueueError("manifest root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("--manifest", type=Path, required=True)
    enqueue.add_argument("--priority", type=int, default=100)
    claim = commands.add_parser("claim")
    claim.add_argument("--owner", required=True)
    claim.add_argument("--lease-seconds", type=int, default=3600)
    transition = commands.add_parser("transition")
    transition.add_argument("job_id", type=int)
    transition.add_argument("status", choices=sorted(ALL_STATES))
    transition.add_argument("--owner", required=True)
    transition.add_argument("--note")
    transition.add_argument("--lease-seconds", type=int, default=3600)
    renew = commands.add_parser("renew")
    renew.add_argument("job_id", type=int)
    renew.add_argument("--owner", required=True)
    renew.add_argument("--lease-seconds", type=int, default=3600)
    recover = commands.add_parser("recover")
    recover.add_argument("--now")
    commands.add_parser("list")
    show = commands.add_parser("show")
    show.add_argument("job_id", type=int)
    return parser


def _print_job(job: Job) -> None:
    print(json.dumps(dataclasses.asdict(job), sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with ExperimentQueue(args.db) as queue:
            if args.command == "enqueue":
                _print_job(queue.enqueue(_load_manifest(args.manifest), args.priority))
            elif args.command == "claim":
                job = queue.claim(args.owner, args.lease_seconds)
                if job is not None:
                    _print_job(job)
            elif args.command == "transition":
                _print_job(queue.transition(args.job_id, args.status, args.owner, args.note, args.lease_seconds))
            elif args.command == "renew":
                _print_job(queue.renew(args.job_id, args.owner, args.lease_seconds))
            elif args.command == "recover":
                moment = datetime_module.datetime.fromisoformat(args.now) if args.now else None
                print(json.dumps(queue.recover_expired(moment)))
            elif args.command == "list":
                print(json.dumps([dataclasses.asdict(job) for job in queue.list()], sort_keys=True))
            elif args.command == "show":
                _print_job(queue.get(args.job_id))
                print(json.dumps([dataclasses.asdict(entry) for entry in queue.journal(args.job_id)], sort_keys=True))
        return 0
    except QueueError as error:
        print(f"queue error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
