#!/usr/bin/env python3
"""Validate the D1.3 matched-FP32 OSCAR cache fixture without NumPy/PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from pathlib import Path


FULL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
D = 256
H_Q = 24
H_KV = 4
GQA = 6
SCALE = 1.0 / math.sqrt(D)


def read_f32(path: Path, count: int) -> list[float]:
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise RuntimeError(f"{path} has {len(raw)} bytes; expected {count * 4}")
    values = list(struct.unpack(f"<{count}f", raw))
    if not all(math.isfinite(x) for x in values):
        raise RuntimeError(f"{path} contains NaN/Inf")
    return values


def read_i32(path: Path, count: int) -> list[int]:
    raw = path.read_bytes()
    if len(raw) != count * 4:
        raise RuntimeError(f"{path} has {len(raw)} bytes; expected {count * 4}")
    return list(struct.unpack(f"<{count}i", raw))


def metric(ref: list[float], actual: list[float]) -> dict[str, float]:
    if len(ref) != len(actual) or not ref:
        raise RuntimeError("metric shape mismatch")
    diffs = [abs(a - b) for a, b in zip(ref, actual)]
    diff2 = sum(x * x for x in diffs)
    ref2 = sum(x * x for x in ref)
    return {
        "max_abs": max(diffs),
        "mean_abs": sum(diffs) / len(diffs),
        "relative_l2": math.sqrt(diff2 / max(ref2, 1.0e-300)),
    }


def tensor_vectors(data: list[float], heads: int, tokens: int) -> list[list[list[float]]]:
    expected = D * heads * tokens
    if len(data) != expected:
        raise RuntimeError(f"tensor has {len(data)} values; expected {expected}")
    return [
        [data[(token * heads + head) * D : (token * heads + head + 1) * D]
         for head in range(heads)]
        for token in range(tokens)
    ]


def rotate_right(vector: list[float], matrix: list[float]) -> list[float]:
    return [sum(vector[i] * matrix[i * D + j] for i in range(D)) for j in range(D)]


def flatten_vectors(vectors: list[list[list[float]]]) -> list[float]:
    return [x for token in vectors for head in token for x in head]


def scores(q: list[list[float]], keys: list[list[list[float]]]) -> list[float]:
    result: list[float] = []
    for q_head in range(H_Q):
        kv_head = q_head // GQA
        for token in keys:
            result.append(sum(a * b for a, b in zip(q[q_head], token[kv_head])) * SCALE)
    return result


def softmax(values: list[float]) -> list[float]:
    result: list[float] = []
    for begin in range(0, len(values), len(values) // H_Q):
        row = values[begin : begin + len(values) // H_Q]
        maximum = max(row)
        exps = [math.exp(x - maximum) for x in row]
        denom = sum(exps)
        result.extend(x / denom for x in exps)
    return result


def path(root: Path, token_count: int, step: int, layer: int, label: str, ext: str) -> Path:
    return root / (
        f"32.tokens_{token_count}.step_{step:02d}.layer_{layer}.{label}.{ext}"
    )


def cache_path(root: Path, token_count: int, step: int, layer: int, label: str) -> Path:
    return root / (
        f"32.cache_tokens_{token_count}.step_{step:02d}.layer_{layer}.{label}.fp32"
    )


def require_full_layer_files(root: Path, steps: int) -> None:
    seen: set[tuple[int, int, str]] = set()
    pattern = re.compile(r"\.step_(\d+)\.layer_(\d+)\.(q_fp32|k_fp32|v_fp32|attention_fp32)\.fp32$")
    for file in root.glob("*.fp32"):
        match = pattern.search(file.name)
        if match:
            seen.add((int(match.group(1)), int(match.group(2)), match.group(3)))
    expected = {
        (step, layer, label)
        for step in range(steps + 1)
        for layer in FULL_LAYERS
        for label in ("q_fp32", "k_fp32", "v_fp32", "attention_fp32")
    }
    unexpected = sorted(seen - expected)
    missing = sorted(expected - seen)
    if unexpected:
        raise RuntimeError(f"unexpected/GDN capture records: {unexpected[:8]}")
    if missing:
        raise RuntimeError(f"missing full-attention capture records: {missing[:8]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unrotated", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--rotation-k", type=Path, required=True)
    parser.add_argument("--rotation-v", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.layer not in FULL_LAYERS:
        raise RuntimeError("requested layer is not a verified full-attention layer")
    if args.steps <= 0:
        raise RuntimeError("decode step count must be positive")
    for root in (args.unrotated, args.rotated):
        require_full_layer_files(root, args.steps)

    matrix_count = len(FULL_LAYERS) * D * D
    k_all = read_f32(args.rotation_k, matrix_count)
    v_all = read_f32(args.rotation_v, matrix_count)
    layer_index = FULL_LAYERS.index(args.layer)
    offset = layer_index * D * D
    k_rotation = k_all[offset : offset + D * D]
    v_rotation = v_all[offset : offset + D * D]

    positions_a: list[int] = []
    positions_b: list[int] = []
    keys_a: list[list[list[float]]] = []
    keys_b: list[list[list[float]]] = []
    values_a: list[list[list[float]]] = []
    values_b: list[list[list[float]]] = []
    rows: list[dict[str, object]] = []
    max_q = 0.0
    max_cache_copy_a = 0.0

    def load_append(root: Path, token_count: int, step: int, label: str, heads: int) -> list[list[list[float]]]:
        return tensor_vectors(
            read_f32(path(root, token_count, step, args.layer, label, "fp32"), D * heads * token_count),
            heads,
            token_count,
        )

    for step, token_count in [(0, 32)] + [(step, 1) for step in range(1, args.steps + 1)]:
        pos_path_a = path(args.unrotated, token_count, step, args.layer, "positions", "i32")
        pos_path_b = path(args.rotated, token_count, step, args.layer, "positions", "i32")
        current_pos_a = read_i32(pos_path_a, token_count)
        current_pos_b = read_i32(pos_path_b, token_count)
        if current_pos_a != current_pos_b:
            raise RuntimeError(f"position mismatch at step {step}: {current_pos_a} != {current_pos_b}")
        if positions_a and current_pos_a[0] != positions_a[-1] + 1:
            raise RuntimeError(f"non-contiguous cache position at step {step}: {current_pos_a}")
        positions_a.extend(current_pos_a)
        positions_b.extend(current_pos_b)

        q_a = load_append(args.unrotated, token_count, step, "q_fp32", H_Q)
        q_b = load_append(args.rotated, token_count, step, "q_fp32", H_Q)
        k_a = load_append(args.unrotated, token_count, step, "k_fp32", H_KV)
        k_b = load_append(args.rotated, token_count, step, "k_fp32", H_KV)
        v_a = load_append(args.unrotated, token_count, step, "v_fp32", H_KV)
        v_b = load_append(args.rotated, token_count, step, "v_fp32", H_KV)
        for token in range(token_count):
            for head in range(H_Q):
                max_q = max(max_q, metric(q_b[token][head], rotate_right(q_a[token][head], k_rotation))["relative_l2"])
        keys_a.extend(k_a)
        keys_b.extend(k_b)
        values_a.extend(v_a)
        values_b.extend(v_b)

        cache_count = positions_a[-1] + 1
        cache_a_k = tensor_vectors(
            read_f32(cache_path(args.unrotated, cache_count, step, args.layer, "cache_k_fp32"),
                     D * H_KV * cache_count),
            H_KV,
            cache_count,
        )
        cache_b_k = tensor_vectors(
            read_f32(cache_path(args.rotated, cache_count, step, args.layer, "cache_k_fp32"),
                     D * H_KV * cache_count),
            H_KV,
            cache_count,
        )
        cache_a_v = tensor_vectors(
            read_f32(cache_path(args.unrotated, cache_count, step, args.layer, "cache_v_fp32"),
                     D * H_KV * cache_count),
            H_KV,
            cache_count,
        )
        cache_b_v = tensor_vectors(
            read_f32(cache_path(args.rotated, cache_count, step, args.layer, "cache_v_fp32"),
                     D * H_KV * cache_count),
            H_KV,
            cache_count,
        )
        max_cache_copy_a = max(
            max_cache_copy_a,
            metric(flatten_vectors(keys_a), flatten_vectors(cache_a_k))["relative_l2"],
            metric(flatten_vectors(values_a), flatten_vectors(cache_a_v))["relative_l2"],
        )
        k_expected = flatten_vectors(
            [[rotate_right(vector, k_rotation) for vector in token] for token in cache_a_k]
        )
        v_expected = flatten_vectors(
            [[rotate_right(vector, v_rotation) for vector in token] for token in cache_a_v]
        )
        k_actual = flatten_vectors(cache_b_k)
        v_actual = flatten_vectors(cache_b_v)
        k_error = metric(k_expected, k_actual)
        v_error = metric(v_expected, v_actual)

        if step == 0:
            continue
        score_a = scores(q_a[0], keys_a)
        score_b = scores(q_b[0], keys_b)
        softmax_a = softmax(score_a)
        softmax_b = softmax(score_b)
        recovered_a = read_f32(
            path(args.unrotated, 1, step, args.layer, "attention_fp32", "fp32"), D * H_Q
        )
        recovered_b = read_f32(
            path(args.rotated, 1, step, args.layer, "attention_recovered_fp32", "fp32"), D * H_Q
        )
        rows.append(
            {
                "decode_pos": step,
                "absolute_position": current_pos_a[0],
                "cache_k": k_error,
                "cache_v": v_error,
                "score": metric(score_a, score_b),
                "softmax": metric(softmax_a, softmax_b),
                "recovered_attention": metric(recovered_a, recovered_b),
            }
        )

    max_cache_k = max(row["cache_k"]["relative_l2"] for row in rows)  # type: ignore[index]
    max_cache_v = max(row["cache_v"]["relative_l2"] for row in rows)  # type: ignore[index]
    max_score = max(row["score"]["relative_l2"] for row in rows)  # type: ignore[index]
    max_softmax = max(row["softmax"]["relative_l2"] for row in rows)  # type: ignore[index]
    max_attention = max(row["recovered_attention"]["relative_l2"] for row in rows)  # type: ignore[index]
    result = {
        "layer": args.layer,
        "prefill_tokens": 32,
        "ordinary_decode_steps": args.steps,
        "cache_positions": positions_a,
        "positions_identical": positions_a == positions_b,
        "cache_token_count": len(positions_a),
        "max_q_transform_relative_l2": max_q,
        "max_cache_copy_a_relative_l2": max_cache_copy_a,
        "max_cache_k_relative_l2": max_cache_k,
        "max_cache_v_relative_l2": max_cache_v,
        "max_score_relative_l2": max_score,
        "max_softmax_relative_l2": max_softmax,
        "max_recovered_attention_relative_l2": max_attention,
        "rows": rows,
        "pass": positions_a == positions_b and max_cache_copy_a <= 1.0e-6 and
        max_cache_k <= 1.0e-5 and max_cache_v <= 1.0e-5 and max_attention <= 1.0e-5,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise RuntimeError("D1.3 cache/attention qualification failed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"D1.3 validation: FAIL: {error}")
        raise
