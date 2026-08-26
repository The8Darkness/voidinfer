"""Small, deterministic provenance helpers shared by experiment runners."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


class MetadataError(RuntimeError):
    """Raised when required experiment provenance cannot be collected."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise MetadataError(f"could not hash {path}: {error}") from error
    return digest.hexdigest()


def file_fingerprint(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise MetadataError(f"could not stat {path}: {error}") from error
    return {"path": str(path), "bytes": size, "sha256": sha256_file(path)}


def git_revision(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise MetadataError(f"could not invoke git in {repo_root}: {error}") from error
    revision = result.stdout.strip()
    if result.returncode != 0 or not revision:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise MetadataError(f"could not identify repository revision: {detail}")
    return revision


def environment_snapshot(repo_root: Path) -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "cwd": str(repo_root),
        "pid": os.getpid(),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise MetadataError(f"could not atomically write {path}: {error}") from error
