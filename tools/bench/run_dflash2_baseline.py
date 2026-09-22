#!/usr/bin/env python3
"""Run the reproducible Qwen3.8 DFlash2 baseline matrix.

The executable emits one ``key,value`` record per line. This runner preserves those records,
adds the command/return-code provenance, and writes JSON suitable for later ablation comparison.
Use ``--profile-phases`` for opt-in CUDA-event draft/selector/verifier timing; the default keeps
the hot path uninstrumented.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_CONTEXTS = (4096, 16384, 32768, 65536, 131072, 262144)
DEFAULT_BATCHES = (1, 2, 4, 8)
DEFAULT_WORKLOADS = ("normal", "coding", "reasoning", "repetitive", "tool")


def parse_int_list(value: str, label: str) -> list[int]:
    values = []
    for item in value.split(","):
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{label} contains a non-integer: {item}") from exc
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{label} values must be positive")
        values.append(parsed)
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return values


def parse_records(stdout: str) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in stdout.splitlines():
        if not line or "," not in line:
            continue
        key, value = line.split(",", 1)
        records[key] = value
    return records


def run_case(args: argparse.Namespace, context: int, batch: int, workload: str) -> dict:
    command = [
        str(args.executable),
        "--artifact",
        str(args.artifact),
        "--context",
        str(context),
        "--batch",
        str(batch),
        "--warmup",
        str(args.warmup),
        "--reps",
        str(args.reps),
        "--prefill-chunk",
        str(args.prefill_chunk),
        "--draft-tokens",
        "7",
        "--proposal-head",
        args.proposal_head,
        "--kv-dtype",
        "vericache-nvfp4",
        "--hierarchical-vericache",
        "--no-cuda-graph",
        "--workload",
        workload,
    ]
    if args.profile_phases:
        command.append("--profile-phases")
    if args.adaptive_k:
        command.append("--adaptive-k")

    environment = os.environ.copy()
    # The Windows targets copy their FFmpeg runtime DLLs beside the executable. Put that
    # directory first so matrix runs do not depend on an unrelated build or the caller's PATH.
    runtime_paths = [args.executable.parent, *args.path_prepend]
    environment["PATH"] = os.pathsep.join(
        [*(str(path) for path in runtime_paths), environment.get("PATH", "")]
    )

    completed = subprocess.run(
        command,
        cwd=args.repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    records = parse_records(completed.stdout)
    result = {
        "context": context,
        "batch": batch,
        "workload": workload,
        "returncode": completed.returncode,
        "command": command,
        "records": records,
    }
    if completed.stderr:
        result["stderr"] = completed.stderr
    if completed.returncode != 0:
        print(
            f"FAILED context={context} batch={batch} workload={workload}: "
            f"returncode={completed.returncode}",
            file=sys.stderr,
        )
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
    else:
        tok_per_second = records.get("published_tokens_per_second", "n/a")
        print(f"context={context:6d} batch={batch} workload={workload:10s} tok/s={tok_per_second}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_default = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--repo", type=Path, default=repo_default, help="repository root (default: script repo)"
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=repo_default.parent / "build-adaptive-dflash2" / "bench" /
        "ninfer_qwen3_6_27b_dflash_round_bench.exe",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path(
            "C:/AI/voidinfer/models/Qwen3.8-27B-NVFP4-DFlash2-NInfer/"
            "qwen3_8_27b_nvfp4.ninfer"
        ),
    )
    parser.add_argument("--contexts", type=lambda value: parse_int_list(value, "contexts"),
                        default=list(DEFAULT_CONTEXTS))
    parser.add_argument("--batches", type=lambda value: parse_int_list(value, "batches"),
                        default=list(DEFAULT_BATCHES))
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--proposal-head", choices=("full", "optimized"), default="full")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--adaptive-k", action="store_true",
                        help="enable the eager-only Adaptive DFlash2 K heuristic")
    parser.add_argument(
        "--path-prepend",
        action="append",
        type=Path,
        default=[],
        help="directory to prepend to PATH; repeat for CUDA/dependency DLL directories",
    )
    parser.add_argument("--output", type=Path, default=Path("dflash2_baseline.json"))
    args = parser.parse_args()

    args.repo = args.repo.resolve()
    args.executable = args.executable.resolve()
    args.artifact = args.artifact.resolve()
    workloads = [item.strip() for item in args.workloads.split(",") if item.strip()]
    unknown = sorted(set(workloads) - set(DEFAULT_WORKLOADS))
    if unknown:
        parser.error(f"unknown workload(s): {', '.join(unknown)}")
    if not args.executable.is_file():
        parser.error(f"benchmark executable does not exist: {args.executable}")
    if not args.artifact.is_file():
        parser.error(f"artifact does not exist: {args.artifact}")

    results = []
    for workload in workloads:
        for context in args.contexts:
            for batch in args.batches:
                results.append(run_case(args, context, batch, workload))

    output = {
        "format": "ninfer_dflash2_baseline_matrix_v1",
        "repo": str(args.repo),
        "executable": str(args.executable),
        "artifact": str(args.artifact),
        "profile_phases": args.profile_phases,
        "adaptive_k": args.adaptive_k,
        "contexts": args.contexts,
        "batches": args.batches,
        "workloads": workloads,
        "warmup": args.warmup,
        "reps": args.reps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    failed = sum(result["returncode"] != 0 for result in results)
    print(f"wrote {args.output} ({len(results)} cases, {failed} failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
