#!/usr/bin/env python3
"""Reference rotated-BF16 OSCAR invariance test without INT2 quantization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_LAYERS = tuple(range(3, 64, 4))
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
GQA_RATIO = 6
TOKENS = 256
CHUNK_ID = 1
SCORE_TOL = 1e-5
SOFTMAX_TOL = 1e-6
OUTPUT_TOL = 1e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_tensor(path: Path, shape: tuple[int, ...], label: str) -> torch.Tensor:
    require(path.is_file(), f"{label}: missing {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor), f"{label}: .pt payload is not a tensor")
    require(tensor.dtype == torch.bfloat16, f"{label}: expected torch.bfloat16, got {tensor.dtype}")
    require(tuple(tensor.shape) == shape,
            f"{label}: expected shape {shape}, got {tuple(tensor.shape)}")
    require(torch.isfinite(tensor.float()).all().item(), f"{label}: non-finite input")
    return tensor.contiguous()


def load_rotation(path: Path, objective: str) -> dict[int, torch.Tensor]:
    require(path.is_file() and path.stat().st_size > 0, f"missing rotation asset: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict), f"{path.name}: invalid checkpoint")
    require(state.get("objective") == objective, f"{path.name}: objective mismatch")
    layers = state.get("layers")
    require(isinstance(layers, dict), f"{path.name}: layers map missing")
    normalized: dict[int, torch.Tensor] = {}
    for key, entry in layers.items():
        layer = int(key)
        require(isinstance(entry, dict), f"{path.name}: layer {layer} entry invalid")
        rotation = entry.get("rotation")
        require(isinstance(rotation, torch.Tensor), f"{path.name}: layer {layer} rotation missing")
        require(rotation.dtype == torch.float32 and tuple(rotation.shape) == (HEAD_DIM, HEAD_DIM),
                f"{path.name}: layer {layer} rotation metadata mismatch")
        require(torch.isfinite(rotation).all().item(), f"{path.name}: layer {layer} non-finite")
        require(entry.get("layer_id") == layer, f"{path.name}: layer {layer} mapping mismatch")
        normalized[layer] = rotation.contiguous()
    require(set(normalized) == set(EXPECTED_LAYERS),
            f"{path.name}: expected exact full-attention layer mapping")
    return normalized


def diff_stats(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    difference = (left - right).abs()
    if mask is not None:
        difference = difference.masked_select(mask)
    return {
        "max_abs": float(difference.max().item()),
        "rms": float(torch.sqrt(torch.mean(difference.square())).item()),
    }


def grouped_kv(tensor: torch.Tensor) -> torch.Tensor:
    mapping = torch.arange(Q_HEADS, dtype=torch.long) // GQA_RATIO
    return tensor[:, mapping, :]


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Inputs are [T, Hq/Hkv, D] and are already in the post-RoPE boundary layout.
    k_grouped = grouped_kv(k)
    v_grouped = grouped_kv(v)
    scores = torch.einsum("thd,shd->hts", q, k_grouped) / math.sqrt(HEAD_DIM)
    future = torch.triu(torch.ones(TOKENS, TOKENS, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(future.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shd->thd", probabilities, v_grouped)
    return scores, probabilities, output


def run_layer(
    dump_root: Path,
    k_rotations: dict[int, torch.Tensor],
    v_rotations: dict[int, torch.Tensor],
    layer: int,
) -> dict[str, Any]:
    q = load_tensor(dump_root / f"layer_{layer}" / "q" / f"{CHUNK_ID}.pt",
                    (TOKENS, Q_HEADS, HEAD_DIM), f"layer {layer} Q")
    k = load_tensor(dump_root / f"layer_{layer}" / "k" / f"{CHUNK_ID}.pt",
                    (TOKENS, KV_HEADS, HEAD_DIM), f"layer {layer} K")
    v = load_tensor(dump_root / f"layer_{layer}" / "v" / f"{CHUNK_ID}.pt",
                    (TOKENS, KV_HEADS, HEAD_DIM), f"layer {layer} V")

    # Preserve the exact BF16 source values, but compute the reference products in
    # float64 so the test isolates matrix convention and V inverse semantics from
    # an additional intermediate BF16 rounding step.
    q_ref = q.double()
    k_ref = k.double()
    v_ref = v.double()
    r_k = k_rotations[layer].double()
    r_v = v_rotations[layer].double()

    scores_ref, probabilities_ref, output_ref = causal_attention(q_ref, k_ref, v_ref)
    q_rotated = q_ref @ r_k
    k_rotated = k_ref @ r_k
    v_rotated = v_ref @ r_v
    scores_rotated, probabilities_rotated, raw_output_rotated = causal_attention(
        q_rotated, k_rotated, v_rotated
    )
    recovered_output = raw_output_rotated @ r_v.T

    valid_scores = torch.isfinite(scores_ref) & torch.isfinite(scores_rotated)
    score = diff_stats(scores_ref, scores_rotated, valid_scores)
    softmax = diff_stats(probabilities_ref, probabilities_rotated)
    raw_output = diff_stats(output_ref, raw_output_rotated)
    recovered = diff_stats(output_ref, recovered_output)
    q_roundtrip = diff_stats(q_ref, q_rotated @ r_k.T)
    k_roundtrip = diff_stats(k_ref, k_rotated @ r_k.T)
    v_roundtrip = diff_stats(v_ref, v_rotated @ r_v.T)

    passed = (
        score["max_abs"] <= SCORE_TOL
        and softmax["max_abs"] <= SOFTMAX_TOL
        and recovered["max_abs"] <= OUTPUT_TOL
    )
    result = {
        "layer": layer,
        "source_dtype": "torch.bfloat16",
        "diagnostic_compute_dtype": "torch.float64",
        "q_shape": [TOKENS, Q_HEADS, HEAD_DIM],
        "k_shape": [TOKENS, KV_HEADS, HEAD_DIM],
        "v_shape": [TOKENS, KV_HEADS, HEAD_DIM],
        "gqa_mapping": [head // GQA_RATIO for head in range(Q_HEADS)],
        "score": score,
        "softmax": softmax,
        "raw_rotated_attention_output_before_v_inverse": raw_output,
        "recovered_attention_output_after_v_inverse": recovered,
        "q_roundtrip": q_roundtrip,
        "k_roundtrip": k_roundtrip,
        "v_roundtrip": v_roundtrip,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(
            f"layer {layer}: rotated-BF16 invariance failed; "
            f"score={score['max_abs']:.3e}, softmax={softmax['max_abs']:.3e}, "
            f"recovered_output={recovered['max_abs']:.3e}"
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_root", type=Path)
    parser.add_argument("rotation_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    dump_root = args.dump_root.resolve()
    rotation_dir = args.rotation_dir.resolve()
    report_path = args.report.resolve()
    k_rotations = load_rotation(rotation_dir / "k_rotation_qqt_r_h_pbr.pt", "qqt_r_h_pbr")
    v_rotations = load_rotation(rotation_dir / "v_rotation_sst_r_h_pbr.pt", "sst_r_h_pbr")

    results: list[dict[str, Any]] = []
    for layer in EXPECTED_LAYERS:
        results.append(run_layer(dump_root, k_rotations, v_rotations, layer))

    report = {
        "schema": "oscar-rotation-invariance-v1",
        "layers": list(EXPECTED_LAYERS),
        "tokens": TOKENS,
        "chunk_id": CHUNK_ID,
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "gqa_ratio": GQA_RATIO,
        "head_dim": HEAD_DIM,
        "orientation": "row vectors [tokens, heads, head_dim]; Q'=Q@Rk, K'=K@Rk, V'=V@Rv, O=O'@Rv.T",
        "tolerances": {
            "score_max_abs": SCORE_TOL,
            "softmax_max_abs": SOFTMAX_TOL,
            "recovered_attention_output_max_abs": OUTPUT_TOL,
        },
        "notes": [
            "Source tensors retain exact BF16 values; diagnostic products use float64.",
            "The raw rotated attention output is intentionally not invariant until V inverse is applied.",
            "The captured activation set does not contain the model output projection or residual input, so the layer residual is not tested here.",
        ],
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"OSCAR rotated-BF16 invariance: PASS layers={len(results)} tokens={TOKENS} "
        f"score_tol={SCORE_TOL:.1e} softmax_tol={SOFTMAX_TOL:.1e} output_tol={OUTPUT_TOL:.1e}"
    )
    for result in results:
        print(
            f"layer={result['layer']} score={result['score']['max_abs']:.9e} "
            f"softmax={result['softmax']['max_abs']:.9e} "
            f"raw_v_rot={result['raw_rotated_attention_output_before_v_inverse']['max_abs']:.9e} "
            f"recovered={result['recovered_attention_output_after_v_inverse']['max_abs']:.9e}"
        )
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
