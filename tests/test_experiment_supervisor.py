from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.experiments.experiment_queue import ExperimentQueue
from tools.experiments.experiment_supervisor import run_job


def manifest_for(command: list[str], restore: list[str]) -> dict[str, object]:
    return {
        "run_id": "supervisor-test",
        "revision": "test-revision",
        "hypothesis": "supervisor restores after a bounded command",
        "baseline_run_ids": ["baseline-test"],
        "hard_gates": ["command result is journaled"],
        "success_metric": "durable completion state",
        "max_gpu_hours": 0.1,
        "rollback": "restore marker",
        "owner": "test",
        "artifact": {"path": "model.ninfer", "sha256": "artifact-hash"},
        "corpus": {"path": "corpus.ids", "sha256": "corpus-hash"},
        "environment": {"os": "test"},
        "metric_schema": "ninfer-experiment-v1",
        "commands": [
            {"phase": "BUILDING", "argv": command},
            {"phase": "RESTORING", "argv": restore},
        ],
        "result_paths": ["logs/experiments/supervisor-test"],
    }


def python_marker(path: Path, value: str) -> list[str]:
    return [sys.executable, "-c", f"from pathlib import Path; Path({str(path)!r}).write_text({value!r})"]


def test_supervisor_runs_and_records_recovery_path(tmp_path: Path) -> None:
    completed = tmp_path / "completed"
    restored = tmp_path / "restored"
    with ExperimentQueue(tmp_path / "queue.db") as queue:
        job = queue.enqueue(manifest_for(python_marker(completed, "done"), python_marker(restored, "restored")))
        claimed = queue.claim("supervisor-test")
        assert claimed is not None
        result = run_job(
            queue,
            claimed,
            "supervisor-test",
            tmp_path,
            tmp_path / "logs",
            None,
            None,
            1.0,
        )
        assert result.status == "DONE"
        assert completed.read_text() == "done"
        assert restored.read_text() == "restored"
        assert [entry.to_status for entry in queue.journal(job.id)][-4:] == [
            "RESTORING",
            "HEALTHY",
            "RESUME_PENDING",
            "DONE",
        ]


def test_supervisor_marks_failed_command_recoverable_after_restore(tmp_path: Path) -> None:
    restored = tmp_path / "restored"
    failing = [sys.executable, "-c", "raise SystemExit(7)"]
    with ExperimentQueue(tmp_path / "queue.db") as queue:
        job = queue.enqueue(manifest_for(failing, python_marker(restored, "restored")))
        claimed = queue.claim("supervisor-test")
        assert claimed is not None
        with pytest.raises(RuntimeError, match="exited with status 7"):
            run_job(
                queue,
                claimed,
                "supervisor-test",
                tmp_path,
                tmp_path / "logs",
                None,
                None,
                1.0,
            )
        assert queue.get(job.id).status == "FAILED_RECOVERABLE"
        assert restored.read_text() == "restored"
        assert (tmp_path / "logs" / "command-00.stderr.log").read_text() == ""
