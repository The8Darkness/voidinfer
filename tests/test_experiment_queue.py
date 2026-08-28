from __future__ import annotations

import datetime as datetime_module
import json
from pathlib import Path

import pytest

from tools.experiments.experiment_queue import (
    ExperimentQueue,
    QueueError,
    validate_manifest,
)

MANIFEST = {
    "run_id": "queue-test",
    "revision": "test-revision",
    "hypothesis": "queue transitions survive supervisor restart",
    "baseline_run_ids": ["baseline-1"],
    "hard_gates": ["unit tests pass", "service restored"],
    "success_metric": "recoverable lease and complete journal",
    "max_gpu_hours": 0.25,
    "rollback": "restore last-known-good service",
    "owner": "test",
    "artifact": {"path": "model.ninfer", "sha256": "artifact-hash"},
    "corpus": {"path": "corpus.ids", "sha256": "corpus-hash"},
    "environment": {"os": "test"},
    "metric_schema": "ninfer-experiment-v1",
    "commands": [["python", "-m", "pytest"]],
    "result_paths": ["results/queue-test.json"],
}


def test_manifest_validation_rejects_missing_required_fields() -> None:
    with pytest.raises(QueueError, match="missing required fields"):
        validate_manifest({})


def test_claim_and_state_transitions_are_durable(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite3"
    with ExperimentQueue(database) as queue:
        job = queue.enqueue(MANIFEST, priority=200)
        assert job.status == "IDLE"

    with ExperimentQueue(database) as queue:
        claimed = queue.claim("supervisor-a", lease_seconds=60)
        assert claimed is not None
        assert claimed.id == job.id
        assert claimed.status == "CHECKPOINTED"
        assert queue.claim("supervisor-b") is None
        queue.transition(job.id, "SERVICE_STOPPING", "supervisor-a")
        queue.transition(job.id, "GPU_FREE", "supervisor-a")
        queue.transition(job.id, "BUILDING", "supervisor-a")
        queue.transition(job.id, "TESTING", "supervisor-a")
        queue.transition(job.id, "BENCHMARKING", "supervisor-a")
        queue.transition(job.id, "COLLECTING", "supervisor-a")
        queue.transition(job.id, "RESTORING", "supervisor-a")
        queue.transition(job.id, "HEALTHY", "supervisor-a")
        done = queue.transition(job.id, "DONE", "supervisor-a")
        assert done.lease_expires_at is None
        assert [entry.to_status for entry in queue.journal(job.id)] == [
            "IDLE",
            "CHECKPOINTED",
            "SERVICE_STOPPING",
            "GPU_FREE",
            "BUILDING",
            "TESTING",
            "BENCHMARKING",
            "COLLECTING",
            "RESTORING",
            "HEALTHY",
            "DONE",
        ]


def test_owner_and_transition_guards(tmp_path: Path) -> None:
    with ExperimentQueue(tmp_path / "queue.db") as queue:
        job = queue.enqueue(MANIFEST)
        queue.claim("owner-a")
        with pytest.raises(QueueError, match="owned by"):
            queue.transition(job.id, "SERVICE_STOPPING", "owner-b")
        with pytest.raises(QueueError, match="invalid transition"):
            queue.transition(job.id, "DONE", "owner-a")


def test_expired_lease_is_recoverable_and_can_be_reclaimed(tmp_path: Path) -> None:
    with ExperimentQueue(tmp_path / "queue.db") as queue:
        job = queue.enqueue(MANIFEST)
        claimed = queue.claim("owner-a", lease_seconds=1)
        assert claimed is not None
        expiry = datetime_module.datetime.fromisoformat(claimed.lease_expires_at or "")
        recovered = queue.recover_expired(expiry + datetime_module.timedelta(seconds=1))
        assert recovered == [job.id]
        assert queue.get(job.id).status == "FAILED_RECOVERABLE"
        replacement = queue.claim("owner-b")
        assert replacement is not None
        assert replacement.id == job.id
        assert replacement.attempts == 2


def test_cli_serializes_job_as_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from tools.experiments import experiment_queue as queue_module

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(MANIFEST), encoding="utf-8")
    assert (
        queue_module.main(
            ["--db", str(tmp_path / "queue.db"), "enqueue", "--manifest", str(manifest)]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "IDLE"
    assert payload["manifest"]["hypothesis"] == MANIFEST["hypothesis"]
