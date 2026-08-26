# Experiment control plane

The experiment control plane is project-owned and intentionally offline. It prevents a benchmark
from silently claiming the GPU while a project service is still running, and leaves a durable
journal when the supervisor or workstation disappears.

## Queue

Create a manifest with these fields:

- `run_id`, `revision`, `hypothesis`, `baseline_run_ids`, `hard_gates`, `success_metric`;
- `max_gpu_hours`, `rollback`, `owner`, `artifact`, `corpus`, `environment`, `metric_schema`;
- `commands`: argument arrays or `{ "phase": "BUILDING", "argv": [...] }` records;
- `result_paths`.

Commands are argv arrays, never shell strings. The SQLite queue uses WAL plus `synchronous=FULL`,
leases, an owner check, an allowlisted state machine, and a journal in one transaction per state
transition. An expired lease becomes `FAILED_RECOVERABLE`; it is never silently treated as done.

```powershell
python -m tools.experiments.experiment_queue `
  --db logs/experiments/queue.sqlite3 enqueue `
  --manifest profiles/bench/<run>/manifest.json
python -m tools.experiments.experiment_queue `
  --db logs/experiments/queue.sqlite3 list
```

`run_ninfer_bench_matrix.py` writes a queue-compatible manifest containing repository, artifact,
corpus, and environment provenance. It remains directly runnable for small local checks; use the
queue for a service-offline GPU campaign.

## Supervisor

Claim and execute one queued job with project-owned restoration:

```powershell
python -m tools.experiments.experiment_supervisor `
  --db logs/experiments/queue.sqlite3 `
  --project-root D:\AI\voidinfer `
  --health-url http://127.0.0.1:8080/health
```

Only an explicitly supplied process identity can be stopped. The supervisor rechecks PID,
executable, command arguments, and start identity immediately before signaling. It will refuse a
changed or missing identity rather than risk an unrelated process. Restoration must complete and,
when supplied, the loopback health endpoint must return `{ "status": "ok" }` before a successful
job is journaled as `DONE`.

The queue and supervisor are standard-library Python and require no service dependency. Long GPU
jobs should be pre-written in manifests, use the queue lease, write results under a unique run
folder, and keep raw logs/profiles outside normal source paths when they are too large for git.
