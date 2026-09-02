#!/usr/bin/env python3
"""Evaluate immutable C4 OSCAR assets against a fresh held-out capture."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_base_module(path: Path) -> Any:
    require(path.is_file(), f"shared evaluator is missing: {path}")
    spec = importlib.util.spec_from_file_location("oscar_c3_evaluator", path)
    require(spec is not None and spec.loader is not None,
            "cannot load shared C3 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_layer(base: Any, dump_root: Path, layer: int, chunks: list[int],
                   rk: torch.Tensor, rv: torch.Tensor, upstream: Any,
                   quantized: bool, metrics: Any) -> None:
    rk = rk.double()
    rv = rv.double()
    future = torch.triu(torch.ones(base.T, base.T, dtype=torch.bool), diagonal=1)
    for chunk in chunks:
        q = base.load_tensor(dump_root / f"layer_{layer}" / "q" / f"{chunk}.pt",
                             (base.T, base.Q_HEADS, base.D),
                             f"layer {layer} chunk {chunk} Q")
        k = base.load_tensor(dump_root / f"layer_{layer}" / "k" / f"{chunk}.pt",
                             (base.T, base.KV_HEADS, base.D),
                             f"layer {layer} chunk {chunk} K")
        v = base.load_tensor(dump_root / f"layer_{layer}" / "v" / f"{chunk}.pt",
                             (base.T, base.KV_HEADS, base.D),
                             f"layer {layer} chunk {chunk} V")
        reference = base.attention(q, k, v, future)
        q_rot = q @ rk
        k_rot = k @ rk
        v_rot = v @ rv
        if quantized:
            k_rot = base.calibrated_int2(k_rot, upstream)
            v_rot = base.calibrated_int2(v_rot, upstream)
        candidate = base.attention(q_rot, k_rot, v_rot, future)
        candidate["output"] = candidate["output"] @ rv.T
        metrics.add(reference, candidate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heldout_dump", type=Path)
    parser.add_argument("cal30k_rotation_dir", type=Path)
    parser.add_argument("cal10k_rotation_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--shared-evaluator", type=Path,
                        default=Path(__file__).with_name("evaluate_cal10k_heldout.py"))
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    base = load_base_module(args.shared_evaluator.resolve())
    heldout_dump = args.heldout_dump.resolve()
    cal30k_dir = args.cal30k_rotation_dir.resolve()
    cal10k_dir = args.cal10k_rotation_dir.resolve()
    upstream_path = args.upstream.resolve()
    conversion = base.validate_conversion(heldout_dump)
    cal30k_k, cal30k_v, cal30k_assets = base.load_rotations(cal30k_dir)
    cal10k_k, cal10k_v, cal10k_assets = base.load_rotations(cal10k_dir)
    upstream = base.load_upstream(upstream_path)
    chunks = conversion["chunks"]

    aggregate = {
        "rotated_bf16_30k_assets": base.PathMetrics(),
        "calibrated_oscar_int2_30k_assets": base.PathMetrics(),
        "calibrated_oscar_int2_10k_assets": base.PathMetrics(),
    }
    per_layer: list[dict[str, Any]] = []
    for layer in base.LAYERS:
        layer_metrics = {
            "rotated_bf16_30k_assets": base.PathMetrics(),
            "calibrated_oscar_int2_30k_assets": base.PathMetrics(),
            "calibrated_oscar_int2_10k_assets": base.PathMetrics(),
        }
        evaluate_layer(base, heldout_dump, layer, chunks, cal30k_k[layer],
                       cal30k_v[layer], upstream, False,
                       layer_metrics["rotated_bf16_30k_assets"])
        evaluate_layer(base, heldout_dump, layer, chunks, cal30k_k[layer],
                       cal30k_v[layer], upstream, True,
                       layer_metrics["calibrated_oscar_int2_30k_assets"])
        evaluate_layer(base, heldout_dump, layer, chunks, cal10k_k[layer],
                       cal10k_v[layer], upstream, True,
                       layer_metrics["calibrated_oscar_int2_10k_assets"])
        for name, metric in layer_metrics.items():
            base.merge_path_metrics(aggregate[name], metric)
        per_layer.append({
            "layer": layer,
            **{name: metric.finish() for name, metric in layer_metrics.items()},
        })

    def output_max(row: dict[str, Any], name: str) -> float:
        return row[name]["attention_output_original_coordinates"]["max_abs"]

    def rel_l2(row: dict[str, Any], name: str) -> float:
        return row[name]["attention_output_original_coordinates"]["relative_l2"]

    report = {
        "schema": "oscar-c4-heldout-evaluation-v1",
        "heldout_conversion": conversion,
        "heldout_tokens": conversion["useful_tokens"],
        "heldout_chunks": chunks,
        "layers": list(base.LAYERS),
        "q_heads": base.Q_HEADS,
        "kv_heads": base.KV_HEADS,
        "gqa_ratio": base.GQA,
        "head_dim": base.D,
        "target_runtime_group_size": 128,
        "reference": "BF16 original-coordinate causal GQA attention; all candidate outputs recovered with R_V.T",
        "reference_quantization": "official upstream simulate_int2_asym per [token, KV-head, 256] row; group-128 is the intended runtime target, not changed here",
        "upstream_path": str(upstream_path),
        "upstream_sha256": base.sha256(upstream_path),
        "cal30k_assets": cal30k_assets,
        "cal10k_assets": cal10k_assets,
        "paths": {
            "rotated_bf16_30k_assets": "Q/K/V rotated by 30K assets; no quantization; V output recovered with R_V.T",
            "calibrated_oscar_int2_30k_assets": "Q/K rotated by 30K R_K; K/V upstream INT2; V output recovered with R_V.T",
            "calibrated_oscar_int2_10k_assets": "same reference semantics using the previous 10K assets",
        },
        "aggregate": {name: metric.finish() for name, metric in aggregate.items()},
        "per_layer": per_layer,
        "outliers": {
            name: [
                {"layer": row["layer"], "max_abs": output_max(row, name),
                 "relative_l2": rel_l2(row, name)}
                for row in sorted(per_layer, key=lambda item: output_max(item, name),
                                  reverse=True)
            ]
            for name in ("calibrated_oscar_int2_30k_assets",
                         "calibrated_oscar_int2_10k_assets")
        },
    }
    output = args.report_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"OSCAR C4 held-out evaluation: PASS heldout_tokens={conversion['useful_tokens']} "
          f"chunks={len(chunks)} layers={len(base.LAYERS)}")
    for name, metric in aggregate.items():
        result = metric.finish()
        output_metrics = result["attention_output_original_coordinates"]
        print(f"{name}: score_max={result['attention_score_logits']['max_abs']:.9e} "
              f"softmax_max={result['softmax']['max_abs']:.9e} "
              f"output_max={output_metrics['max_abs']:.9e} "
              f"output_rel_l2={output_metrics['relative_l2']:.9e}")
    print(f"report_json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
