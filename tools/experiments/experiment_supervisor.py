#!/usr/bin/env python3
"""Run one queued experiment with project-owned service recovery and safe process checks."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .experiment_queue import ACTIVE_STATES, ExperimentQueue, Job, QueueError

PHASES = frozenset(ACTIVE_STATES)


class SupervisorError(RuntimeError):
    """Raised when an experiment cannot be safely run or recovered."""


@dataclasses.dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    executable: str
    argv: tuple[str, ...]
    started_at_unix: float
    argv_sha256: str

    @classmethod
    def from_json(cls, value: Any) -> ProcessIdentity:
        if not isinstance(value, dict):
            raise SupervisorError("process identity must be an object")
        try:
            pid = value["pid"]
            executable = value["executable"]
            argv = value["argv"]
            started_at = value["started_at_unix"]
            digest = value["argv_sha256"]
        except KeyError as error:
            raise SupervisorError(f"process identity is missing {error.args[0]}") from error
        if (
            not isinstance(pid, int)
            or pid <= 0
            or not isinstance(executable, str)
            or not executable
            or not isinstance(argv, list)
            or not all(isinstance(item, str) for item in argv)
            or not isinstance(started_at, (int, float))
            or not isinstance(digest, str)
        ):
            raise SupervisorError("process identity has invalid fields")
        try:
            started_at_unix = float(started_at)
        except (TypeError, ValueError) as error:
            raise SupervisorError("process identity has an invalid start timestamp") from error
        return cls(pid, executable, tuple(argv), started_at_unix, digest)


def argv_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(argv), separators=(",", ":")).encode("utf-8")).hexdigest()


def read_process_identity(path: Path) -> ProcessIdentity:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SupervisorError(f"could not read process identity {path}: {error}") from error
    return ProcessIdentity.from_json(value)


def write_process_identity(path: Path, process: subprocess.Popen[str], executable: Path, argv: Sequence[str]) -> ProcessIdentity:
    identity = ProcessIdentity(
        pid=process.pid,
        executable=str(executable.resolve()),
        argv=tuple(argv),
        started_at_unix=time.time(),
        argv_sha256=argv_digest(argv),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dataclasses.asdict(identity), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return identity


def _windows_process_info(pid: int) -> tuple[str, str, float | None] | None:
    if os.name != "nt":
        return None
    script = (
        "$p = Get-CimInstance Win32_Process -Filter 'ProcessId = "
        + str(pid)
        + "'; if ($null -eq $p) { exit 1 }; "
        "[pscustomobject]@{Path=$p.ExecutablePath; CommandLine=$p.CommandLine; CreationDate=$p.CreationDate} | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    path = value.get("Path")
    command_line = value.get("CommandLine")
    creation = value.get("CreationDate")
    if not isinstance(path, str) or not isinstance(command_line, str):
        return None
    creation_unix: float | None = None
    if isinstance(creation, str) and len(creation) >= 14:
        try:
            import datetime as datetime_module

            parsed = datetime_module.datetime.strptime(creation[:14], "%Y%m%d%H%M%S")
            creation_unix = parsed.replace(tzinfo=datetime_module.timezone.utc).timestamp()
        except ValueError:
            creation_unix = None
    return path, command_line, creation_unix


def process_matches(identity: ProcessIdentity) -> bool:
    if identity.argv_sha256 != argv_digest(identity.argv):
        return False
    if os.name == "nt":
        info = _windows_process_info(identity.pid)
        if info is None:
            return False
        executable, command_line, creation_unix = info
        if Path(executable).resolve() != Path(identity.executable).resolve():
            return False
        if creation_unix is not None and abs(creation_unix - identity.started_at_unix) > 5.0:
            return False
        return all(argument in command_line for argument in identity.argv)
    proc_path = Path(f"/proc/{identity.pid}")
    if not proc_path.exists():
        return False
    try:
        executable = Path(os.readlink(proc_path / "exe")).resolve()
        command_line = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, UnicodeDecodeError):
        return False
    return executable == Path(identity.executable).resolve() and all(
        argument in command_line for argument in identity.argv
    )


def stop_owned_process(identity_path: Path, timeout_seconds: float = 30.0) -> None:
    identity = read_process_identity(identity_path)
    if not process_matches(identity):
        raise SupervisorError(
            f"refusing to signal PID {identity.pid}: executable, argv, or start identity changed"
        )
    if os.name == "nt":
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", f"Stop-Process -Id {identity.pid}"],
            check=True,
        )
    else:
        os.kill(identity.pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not process_matches(identity):
            identity_path.unlink(missing_ok=True)
            return
        time.sleep(0.1)
    if process_matches(identity):
        raise SupervisorError(f"owned process PID {identity.pid} did not exit after graceful stop")


def health_check(url: str, timeout_seconds: float) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SupervisorError("health URL must be a loopback HTTP(S) endpoint")
    deadline = time.monotonic() + timeout_seconds
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(  # noqa: S310 - scheme and loopback host are validated above
                url, timeout=min(5.0, max(0.1, deadline - time.monotonic()))
            ) as response:
                payload = json.loads(response.read())
            if payload == {"status": "ok"}:
                return
            last_error = f"unexpected health payload: {payload!r}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise SupervisorError(f"service did not become healthy: {last_error}")


def command_entries(manifest: dict[str, Any]) -> list[tuple[str, list[str]]]:
    commands = manifest["commands"]
    entries: list[tuple[str, list[str]]] = []
    for command in commands:
        if isinstance(command, dict):
            phase = command.get("phase")
            argv = command.get("argv")
        else:
            phase = "BENCHMARKING"
            argv = command
        if phase not in PHASES or not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
            raise SupervisorError("commands must contain {phase, argv} with non-empty argument arrays")
        entries.append((phase, argv))
    return entries


def run_command(argv: Sequence[str], cwd: Path, log_dir: Path, index: int) -> None:
    stdout_path = log_dir / f"command-{index:02d}.stdout.log"
    stderr_path = log_dir / f"command-{index:02d}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(list(argv), cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    if result.returncode != 0:
        raise SupervisorError(f"command {index} exited with status {result.returncode}: {list(argv)!r}")


def _transition(queue: ExperimentQueue, job: Job, status: str, owner: str, note: str | None = None) -> Job:
    return queue.transition(job.id, status, owner, note)


def run_job(
    queue: ExperimentQueue,
    job: Job,
    owner: str,
    project_root: Path,
    log_dir: Path,
    service_identity: Path | None,
    health_url: str | None,
    health_timeout: float,
) -> Job:
    if job.owner != owner or job.status != "CHECKPOINTED":
        raise SupervisorError(f"job {job.id} must be CHECKPOINTED and owned by {owner!r}")
    manifest = job.manifest
    entries = command_entries(manifest)
    log_dir.mkdir(parents=True, exist_ok=True)
    current = job
    failure: SupervisorError | None = None
    try:
        current = _transition(queue, current, "SERVICE_STOPPING", owner, "stopping only validated project-owned service")
        if service_identity is not None and service_identity.exists():
            stop_owned_process(service_identity)
        current = _transition(queue, current, "GPU_FREE", owner, "project-owned service stopped")
        for index, (phase, argv) in enumerate(entries):
            if phase == "RESTORING":
                continue
            current = _transition(queue, current, phase, owner, f"running command {index}")
            run_command(argv, project_root, log_dir, index)
        current = _transition(queue, current, "COLLECTING", owner, "experiment commands completed")
    except (OSError, QueueError, SupervisorError) as error:
        failure = error if isinstance(error, SupervisorError) else SupervisorError(str(error))

    try:
        if current.status not in {"DONE", "FAILED_BLOCKED"}:
            current = _transition(queue, current, "RESTORING", owner, "restoring project-owned service")
            for index, (phase, argv) in enumerate(entries):
                if phase == "RESTORING":
                    run_command(argv, project_root, log_dir, index)
            if health_url is not None:
                health_check(health_url, health_timeout)
            current = _transition(queue, current, "HEALTHY", owner, "service health verified" if health_url else "restore command completed")
            current = _transition(queue, current, "RESUME_PENDING", owner, "durable completion marker written")
            if failure is None:
                current = _transition(queue, current, "DONE", owner, "experiment completed")
            else:
                current = _transition(queue, current, "FAILED_RECOVERABLE", owner, str(failure))
    except (OSError, QueueError, SupervisorError) as error:
        recovery_error = error if isinstance(error, SupervisorError) else SupervisorError(str(error))
        if current.status not in {"DONE", "FAILED_BLOCKED"}:
            current = _transition(queue, current, "FAILED_BLOCKED", owner, f"recovery failed: {recovery_error}")
        raise SupervisorError(f"experiment failed and service recovery was not verified: {recovery_error}") from recovery_error
    if failure is not None:
        raise failure
    return current


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--owner", default=f"supervisor-{os.getpid()}")
    parser.add_argument("--job-id", type=int)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--service-identity", type=Path)
    parser.add_argument("--health-url")
    parser.add_argument("--health-timeout", type=float, default=120.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        with ExperimentQueue(args.db) as queue:
            job = queue.get(args.job_id) if args.job_id is not None else queue.claim(args.owner)
            if job is None:
                raise SupervisorError("no queued experiment is available")
            run_id = str(job.manifest.get("run_id", f"job-{job.id}"))
            log_dir = args.log_dir or args.project_root / "logs" / "experiments" / run_id
            result = run_job(
                queue,
                job,
                args.owner,
                args.project_root.resolve(),
                log_dir.resolve(),
                args.service_identity.resolve() if args.service_identity else None,
                args.health_url,
                args.health_timeout,
            )
            print(json.dumps(dataclasses.asdict(result), sort_keys=True))
        return 0
    except (QueueError, SupervisorError) as error:
        print(f"supervisor error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
