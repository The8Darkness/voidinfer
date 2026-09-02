#!/usr/bin/env python3
"""Evaluate C3 OSCAR rotations on held-out native-capture chunks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


LAYERS = tuple(range(3, 64, 4))
Q_HEADS = 24
KV_HEADS = 4
GQA = 6
D = 256
T = 256
TYPES = ("q", "k", "v")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"missing/empty JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def expected_shape(kind: str) -> tuple[int, ...]:
    return (T, Q_HEADS, D) if kind == "q" else (T, KV_HEADS, D)


def safe_child(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}: path missing")
    path = (root / Path(relative)).resolve()
    require(root in path.parents and path != root, f"{label}: path escapes root")
    require(path.is_file() and path.stat().st_size > 0, f"{label}: file missing/empty")
    return path


def validate_conversion(dump_root: Path) -> dict[str, Any]:
    manifest_path = dump_root / "conversion_manifest.json"
    data = load_json(manifest_path)
    require(data.get("schema") == "oscar-official-pt-conversion-v1",
            "unexpected conversion manifest schema")
    model = data.get("model", {})
    require(model.get("total_layers") == 64 and model.get("query_heads") == Q_HEADS
            and model.get("kv_heads") == KV_HEADS and model.get("head_dim") == D,
            "held-out conversion topology mismatch")
    require(data.get("tensor_layout") == "[tokens, heads, head_dim]",
            "held-out conversion tensor orientation mismatch")
    chunks = [int(value) for value in data.get("official_chunk_ids", [])]
    require(chunks and chunks == list(range(1, len(chunks) + 1)),
            "official chunk IDs must be consecutive and nonzero")
    records = data.get("tensors")
    require(isinstance(records, list) and len(records) == len(LAYERS) * len(chunks) * 3,
            "held-out tensor record count mismatch")
    expected = {(layer, chunk, kind) for layer in LAYERS for chunk in chunks for kind in TYPES}
    actual: set[tuple[int, int, str]] = set()
    listed: set[str] = set()
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"conversion tensor {index}: invalid record")
        layer = int(record.get("layer", -1))
        chunk = int(record.get("official_chunk_id", -1))
        kind = record.get("qkv_type")
        require((layer, chunk, kind) in expected,
                f"conversion tensor {index}: unknown layer/chunk/type")
        key = (layer, chunk, kind)
        require(key not in actual, f"conversion tensor {index}: duplicate record")
        actual.add(key)
        require(record.get("dtype") == "torch.bfloat16"
                and record.get("shape") == list(expected_shape(kind)),
                f"conversion tensor {index}: dtype/shape mismatch")
        path = safe_child(dump_root, record.get("pt_file"),
                          f"conversion tensor {index}")
        require(sha256(path) == record.get("pt_sha256"),
                f"conversion tensor {index}: .pt hash mismatch")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16
                and tuple(tensor.shape) == expected_shape(kind),
                f"conversion tensor {index}: reload shape/dtype mismatch")
        require(torch.isfinite(tensor.float()).all().item(),
                f"conversion tensor {index}: non-finite values")
        listed.add(path.relative_to(dump_root).as_posix())
    require(actual == expected, "held-out conversion does not cover exact layer/chunk Q/K/V set")
    require({path.relative_to(dump_root).as_posix() for path in dump_root.rglob("*.pt")} == listed,
            "unlisted or missing held-out .pt file")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "useful_tokens": int(data["useful_tokens"]),
        "chunks": chunks,
        "raw_manifest_sha256": data["source_manifest_sha256"],
        "input_kind": data.get("source_validation", {}).get("input_kind"),
        "target_runtime_group_size": data.get("target_runtime_group_size"),
    }


def load_rotations(rotation_dir: Path) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor], dict[str, Any]]:
    k_path = rotation_dir / "k_rotation_qqt_r_h_pbr.pt"
    v_path = rotation_dir / "v_rotation_sst_r_h_pbr.pt"
    result: list[dict[int, torch.Tensor]] = []
    for path, objective in ((k_path, "qqt_r_h_pbr"), (v_path, "sst_r_h_pbr")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(isinstance(state, dict) and state.get("objective") == objective,
                f"{path.name}: objective/checkpoint mismatch")
        layers = state.get("layers")
        require(isinstance(layers, dict), f"{path.name}: layers missing")
        loaded: dict[int, torch.Tensor] = {}
        for key, entry in layers.items():
            layer = int(key)
            require(layer in LAYERS and isinstance(entry, dict),
                    f"{path.name}: unknown layer {layer}")
            rotation = entry.get("rotation")
            require(isinstance(rotation, torch.Tensor)
                    and rotation.dtype == torch.float32
                    and tuple(rotation.shape) == (D, D)
                    and torch.isfinite(rotation).all().item(),
                    f"{path.name}: invalid layer {layer} rotation")
            loaded[layer] = rotation.contiguous()
        require(set(loaded) == set(LAYERS), f"{path.name}: exact layer set mismatch")
        result.append(loaded)
    return result[0], result[1], {
        "k_path": str(k_path),
        "k_sha256": sha256(k_path),
        "v_path": str(v_path),
        "v_sha256": sha256(v_path),
    }


def load_upstream(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("oscar_upstream_c3", path)
    require(spec is not None and spec.loader is not None, "cannot load upstream fitter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(hasattr(module, "simulate_int2_asym") and hasattr(module, "build_hadamard"),
            "upstream quantizer/helpers are missing")
    return module


def load_tensor(path: Path, shape: tuple[int, ...], label: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.bfloat16
            and tuple(tensor.shape) == shape and torch.isfinite(tensor.float()).all().item(),
            f"{label}: invalid tensor")
    return tensor.contiguous().double()


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.max_abs = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.reference_sq = 0.0

    def add(self, reference: torch.Tensor, candidate: torch.Tensor,
            mask: torch.Tensor | None = None) -> None:
        difference = (candidate - reference).abs()
        if mask is not None:
            difference = difference.masked_select(mask)
            reference = reference.masked_select(mask)
        require(difference.numel() > 0, "metric received no elements")
        self.count += difference.numel()
        self.max_abs = max(self.max_abs, float(difference.max().item()))
        self.sum_abs += float(difference.sum().item())
        self.sum_sq += float(difference.square().sum().item())
        self.reference_sq += float(reference.square().sum().item())

    def finish(self) -> dict[str, float]:
        require(self.count > 0, "empty metric accumulator")
        return {
            "max_abs": self.max_abs,
            "mean_abs": self.sum_abs / self.count,
            "rms": math.sqrt(self.sum_sq / self.count),
            "relative_l2": math.sqrt(self.sum_sq) /
                          max(math.sqrt(self.reference_sq), 1.0e-24),
        }


class PathMetrics:
    def __init__(self) -> None:
        self.score = MetricAccumulator()
        self.softmax = MetricAccumulator()
        self.output = MetricAccumulator()
        self.argmax_equal = 0
        self.argmax_total = 0

    def add(self, reference: dict[str, torch.Tensor],
            candidate: dict[str, torch.Tensor]) -> None:
        valid = ~reference["future_mask"].unsqueeze(0)
        self.score.add(reference["scores"], candidate["scores"], valid)
        self.softmax.add(reference["probabilities"], candidate["probabilities"])
        self.output.add(reference["output"], candidate["output"])
        equal = torch.argmax(reference["scores"], dim=-1) == torch.argmax(
            candidate["scores"], dim=-1)
        self.argmax_equal += int(equal.sum().item())
        self.argmax_total += equal.numel()

    def finish(self) -> dict[str, Any]:
        return {
            "attention_score_logits": self.score.finish(),
            "softmax": self.softmax.finish(),
            "attention_output_original_coordinates": self.output.finish(),
            "attention_argmax": {
                "agree": self.argmax_equal,
                "total": self.argmax_total,
                "agreement": self.argmax_equal / self.argmax_total,
            },
            "layer_output": {
                "available": False,
                "reason": "QKV dumps do not contain output-projection, residual, or post-attention activation tensors",
            },
        }


def merge_path_metrics(target: PathMetrics, source: PathMetrics) -> None:
    for name in ("score", "softmax", "output"):
        target_metric = getattr(target, name)
        source_metric = getattr(source, name)
        target_metric.count += source_metric.count
        target_metric.max_abs = max(target_metric.max_abs, source_metric.max_abs)
        target_metric.sum_abs += source_metric.sum_abs
        target_metric.sum_sq += source_metric.sum_sq
        target_metric.reference_sq += source_metric.reference_sq
    target.argmax_equal += source.argmax_equal
    target.argmax_total += source.argmax_total


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              future_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    mapping = torch.arange(Q_HEADS, dtype=torch.long) // GQA
    kg = k[:, mapping, :]
    vg = v[:, mapping, :]
    scores = torch.einsum("thd,shd->hts", q, kg) / math.sqrt(D)
    scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shd->thd", probabilities, vg)
    require(torch.isfinite(scores.masked_select(~future_mask.unsqueeze(0))).all().item()
            and torch.isfinite(probabilities).all().item()
            and torch.isfinite(output).all().item(),
            "attention produced non-finite values")
    return {"scores": scores, "probabilities": probabilities, "output": output,
            "future_mask": future_mask}


def calibrated_int2(x: torch.Tensor, upstream: Any) -> torch.Tensor:
    decoded = upstream.simulate_int2_asym(x)
    require(torch.isfinite(decoded).all().item(), "upstream INT2 output is non-finite")
    return decoded


def layer_path(dump_root: Path, layer: int, chunks: list[int], rk: torch.Tensor,
               rv: torch.Tensor, upstream: Any, metrics: dict[str, PathMetrics]) -> None:
    rk = rk.double()
    rv = rv.double()
    future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    for chunk in chunks:
        q = load_tensor(dump_root / f"layer_{layer}" / "q" / f"{chunk}.pt",
                        (T, Q_HEADS, D), f"layer {layer} chunk {chunk} Q")
        k = load_tensor(dump_root / f"layer_{layer}" / "k" / f"{chunk}.pt",
                        (T, KV_HEADS, D), f"layer {layer} chunk {chunk} K")
        v = load_tensor(dump_root / f"layer_{layer}" / "v" / f"{chunk}.pt",
                        (T, KV_HEADS, D), f"layer {layer} chunk {chunk} V")
        base = attention(q, k, v, future)
        q_rot = q @ rk
        k_rot = k @ rk
        v_rot = v @ rv
        rotated = attention(q_rot, k_rot, v_rot, future)
        rotated["output"] = rotated["output"] @ rv.T
        metrics["rotated_bf16"].add(base, rotated)

        k_cal = calibrated_int2(k_rot, upstream)
        v_cal = calibrated_int2(v_rot, upstream)
        calibrated = attention(q_rot, k_cal, v_cal, future)
        calibrated["output"] = calibrated["output"] @ rv.T
        metrics["calibrated_oscar_int2"].add(base, calibrated)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("heldout_dump", type=Path)
    parser.add_argument("cal10k_rotation_dir", type=Path)
    parser.add_argument("previous_rotation_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    heldout_dump = args.heldout_dump.resolve()
    c3_dir = args.cal10k_rotation_dir.resolve()
    prev_dir = args.previous_rotation_dir.resolve()
    upstream_path = args.upstream.resolve()
    conversion = validate_conversion(heldout_dump)
    c3_k, c3_v, c3_assets = load_rotations(c3_dir)
    prev_k, prev_v, prev_assets = load_rotations(prev_dir)
    upstream = load_upstream(upstream_path)
    chunks = conversion["chunks"]
    per_layer: list[dict[str, Any]] = []
    aggregate = {"rotated_bf16": PathMetrics(),
                 "calibrated_oscar_int2": PathMetrics(),
                 "previous_256_calibrated_int2": PathMetrics()}

    for layer in LAYERS:
        metrics = {"rotated_bf16": PathMetrics(),
                   "calibrated_oscar_int2": PathMetrics(),
                   "previous_256_calibrated_int2": PathMetrics()}
        layer_path(heldout_dump, layer, chunks, c3_k[layer], c3_v[layer], upstream, metrics)
        # Evaluate the previous 256-token assets separately, with no reuse of
        # C3 metrics or calibration tensors.
        prev_metrics = {"rotated_bf16": PathMetrics(),
                        "calibrated_oscar_int2": PathMetrics(),
                        "previous_256_calibrated_int2": PathMetrics()}
        layer_path(heldout_dump, layer, chunks, prev_k[layer], prev_v[layer], upstream, prev_metrics)
        metrics["previous_256_calibrated_int2"] = prev_metrics["calibrated_oscar_int2"]
        for name in ("rotated_bf16", "calibrated_oscar_int2"):
            merge_path_metrics(aggregate[name], metrics[name])
        merge_path_metrics(aggregate["previous_256_calibrated_int2"],
                           metrics["previous_256_calibrated_int2"])
        # For the previous path, retain only the calibrated result as requested.
        per_layer.append({
            "layer": layer,
            "rotated_bf16_10k_assets": metrics["rotated_bf16"].finish(),
            "calibrated_oscar_int2_10k_assets": metrics["calibrated_oscar_int2"].finish(),
            "calibrated_oscar_int2_previous_256_assets": metrics["previous_256_calibrated_int2"].finish(),
        })

    report = {
        "schema": "oscar-c3-heldout-evaluation-v1",
        "heldout_conversion": conversion,
        "heldout_tokens": conversion["useful_tokens"],
        "heldout_chunks": chunks,
        "layers": list(LAYERS),
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "gqa_ratio": GQA,
        "head_dim": D,
        "target_runtime_group_size": 128,
        "reference_quantization": "official upstream simulate_int2_asym per [token, KV-head, 256] row; group-128 is the intended runtime target, not changed here",
        "upstream_path": str(upstream_path),
        "upstream_sha256": sha256(upstream_path),
        "cal10k_assets": c3_assets,
        "previous_256_assets": prev_assets,
        "paths": {
            "rotated_bf16_10k_assets": "Q/K/V rotated by 10K assets; V output recovered with R_V.T",
            "calibrated_oscar_int2_10k_assets": "Q/K rotated by 10K R_K; K/V upstream INT2; V output recovered with R_V.T",
            "calibrated_oscar_int2_previous_256_assets": "same reference semantics using previous 256-token B3 assets",
        },
        "aggregate": {name: aggregate[name].finish()
                      for name in ("rotated_bf16", "calibrated_oscar_int2",
                                   "previous_256_calibrated_int2")},
        "per_layer": per_layer,
        "outliers": {
            "calibrated_10k_by_attention_output_max_abs": [
                {"layer": row["layer"],
                 "max_abs": row["calibrated_oscar_int2_10k_assets"]
                 ["attention_output_original_coordinates"]["max_abs"]}
                for row in sorted(per_layer, key=lambda item: item[
                    "calibrated_oscar_int2_10k_assets"][
                    "attention_output_original_coordinates"]["max_abs"], reverse=True)
            ],
        },
    }
    args.report_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_json.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
    print(f"OSCAR C3 held-out evaluation: PASS heldout_tokens={conversion['useful_tokens']} "
          f"chunks={len(chunks)} layers={len(LAYERS)}")
    for name in ("rotated_bf16", "calibrated_oscar_int2",
                 "previous_256_calibrated_int2"):
        print(f"{name}: score_max={report['aggregate'][name]['attention_score_logits']['max_abs']:.9e} "
              f"softmax_max={report['aggregate'][name]['softmax']['max_abs']:.9e} "
              f"output_max={report['aggregate'][name]['attention_output_original_coordinates']['max_abs']:.9e} "
              f"output_rel_l2={report['aggregate'][name]['attention_output_original_coordinates']['relative_l2']:.9e}")
    print(f"report_json={args.report_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
