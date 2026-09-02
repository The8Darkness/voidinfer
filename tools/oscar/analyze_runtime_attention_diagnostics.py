#!/usr/bin/env python3
"""Analyze the first full-attention stage of a D1 runtime comparison."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


HEAD_DIM = 256
Q_HEADS = 24
KV_HEADS = 4
GQA = 6
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def read_bf16(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 2:
        raise ValueError(f"odd BF16 byte count: {path}")
    values: list[float] = []
    for (bits,) in struct.iter_unpack("<H", raw):
        values.append(struct.unpack("<f", struct.pack("<I", bits << 16))[0])
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"empty or non-finite BF16 tensor: {path}")
    return values


def read_fp32(path: Path) -> list[float]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"non-FP32 byte count: {path}")
    values = list(struct.unpack("<" + "f" * (len(raw) // 4), raw))
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError(f"empty or non-finite FP32 tensor: {path}")
    return values


def read_any(path_without_extension: Path) -> list[float]:
    bf16 = Path(str(path_without_extension) + ".bf16")
    fp32 = Path(str(path_without_extension) + ".fp32")
    if bf16.exists():
        return read_bf16(bf16)
    if fp32.exists():
        return read_fp32(fp32)
    raise FileNotFoundError(path_without_extension)


def metrics(reference: list[float], candidate: list[float]) -> tuple[float, float, float]:
    if len(reference) != len(candidate):
        raise ValueError("tensor size mismatch")
    diffs = [abs(a - b) for a, b in zip(reference, candidate)]
    denom = math.sqrt(max(sum(a * a for a in reference), 1.0e-300))
    return max(diffs), sum(diffs) / len(diffs), math.sqrt(sum(d * d for d in diffs)) / denom


def at(values: list[float], dim: int, heads: int, head: int, token: int, component: int) -> float:
    return values[component + dim * (head + heads * token)]


def read_matrix(path: Path, matrix_index: int) -> list[float]:
    raw = path.read_bytes()
    matrix_bytes = HEAD_DIM * HEAD_DIM * 4
    begin = matrix_index * matrix_bytes
    end = begin + matrix_bytes
    if end > len(raw):
        raise ValueError(f"rotation bank is missing matrix {matrix_index}: {path}")
    return list(struct.unpack("<" + "f" * (HEAD_DIM * HEAD_DIM), raw[begin:end]))


def rotate_rows(values: list[float], heads: int, tokens: int, matrix: list[float], transpose: bool) -> list[float]:
    result = [0.0] * len(values)
    for token in range(tokens):
        for head in range(heads):
            for output_dim in range(HEAD_DIM):
                total = 0.0
                for input_dim in range(HEAD_DIM):
                    row = output_dim if transpose else input_dim
                    col = input_dim if transpose else output_dim
                    total += at(values, HEAD_DIM, heads, head, token, input_dim) * matrix[row * HEAD_DIM + col]
                result[output_dim + HEAD_DIM * (head + heads * token)] = total
    return result


def scores(q: list[float], k: list[float], tokens: int) -> list[float]:
    result: list[float] = []
    for query_token in range(tokens):
        for q_head in range(Q_HEADS):
            for key_token in range(query_token + 1):
                kv_head = q_head // GQA
                dot = sum(
                    at(q, HEAD_DIM, Q_HEADS, q_head, query_token, component)
                    * at(k, HEAD_DIM, KV_HEADS, kv_head, key_token, component)
                    for component in range(HEAD_DIM)
                )
                result.append(dot * ATTN_SCALE)
    return result


def softmax_rows(q: list[float], k: list[float], tokens: int) -> list[float]:
    result: list[float] = []
    for query_token in range(tokens):
        for q_head in range(Q_HEADS):
            row = []
            kv_head = q_head // GQA
            for key_token in range(query_token + 1):
                dot = sum(
                    at(q, HEAD_DIM, Q_HEADS, q_head, query_token, component)
                    * at(k, HEAD_DIM, KV_HEADS, kv_head, key_token, component)
                    for component in range(HEAD_DIM)
                )
                row.append(dot * ATTN_SCALE)
            peak = max(row)
            exp_row = [math.exp(value - peak) for value in row]
            total = sum(exp_row)
            result.extend(value / total for value in exp_row)
    return result


def attention_output(q: list[float], k: list[float], v: list[float], tokens: int) -> list[float]:
    """Reference causal attention in Python double precision, retaining row layout."""
    result = [0.0] * (HEAD_DIM * Q_HEADS * tokens)
    for query_token in range(tokens):
        for q_head in range(Q_HEADS):
            kv_head = q_head // GQA
            row = []
            for key_token in range(query_token + 1):
                dot = sum(
                    at(q, HEAD_DIM, Q_HEADS, q_head, query_token, component)
                    * at(k, HEAD_DIM, KV_HEADS, kv_head, key_token, component)
                    for component in range(HEAD_DIM)
                )
                row.append(dot * ATTN_SCALE)
            peak = max(row)
            weights = [math.exp(value - peak) for value in row]
            total = sum(weights)
            for component in range(HEAD_DIM):
                result[component + HEAD_DIM * (q_head + Q_HEADS * query_token)] = sum(
                    weight / total
                    * at(v, HEAD_DIM, KV_HEADS, kv_head, key_token, component)
                    for key_token, weight in enumerate(weights)
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--k-bank", type=Path)
    parser.add_argument("--v-bank", type=Path)
    parser.add_argument("--precision-mode", default="bf16-materialized")
    args = parser.parse_args()

    def tensor(root: Path, name: str) -> list[float]:
        return read_any(root / f"{args.tokens}.tokens_{args.tokens}.layer_{args.layer}.{name}")

    nq, nk, nv = (tensor(args.normal, name) for name in ("q", "k", "v"))
    rq, rk, rv = (tensor(args.rotated, name) for name in ("q", "k", "v"))
    rotation_labels = ("q_fp32", "k_fp32", "v_fp32") if args.precision_mode in {
        "fp32-rotation", "fp32-rotation+inverse"
    } else ("q_rot", "k_rot", "v_rot")
    rqr, rkr, rvr = (tensor(args.rotated, name) for name in rotation_labels)
    na = tensor(args.normal, "attention_raw")
    raw_label = "attention_fp32" if args.precision_mode in {
        "fp32-rotation", "fp32-rotation+inverse"
    } else "attention_raw"
    ra = tensor(args.rotated, raw_label)
    recovered = tensor(args.rotated, "attention_recovered")

    print(f"pre_q {metrics(nq, rq)}")
    print(f"pre_k {metrics(nk, rk)}")
    print(f"pre_v {metrics(nv, rv)}")
    if args.k_bank is not None:
        k_matrix = read_matrix(args.k_bank, 0)
        print(f"q_rotation_orientation {metrics(rotate_rows(nq, Q_HEADS, args.tokens, k_matrix, False), rqr)}")
        print(f"k_rotation_orientation {metrics(rotate_rows(nk, KV_HEADS, args.tokens, k_matrix, False), rkr)}")
    if args.v_bank is not None:
        v_matrix = read_matrix(args.v_bank, 0)
        expected_v = rotate_rows(nv, KV_HEADS, args.tokens, v_matrix, False)
        print(f"v_rotation_orientation {metrics(expected_v, rvr)}")
    else:
        v_matrix = None
    print(f"rotated_v_vs_original {metrics(nv, rvr)}")
    print(f"attention_raw_vs_recovered {metrics(na, recovered)}")
    print(f"attention_raw_coordinate_difference {metrics(na, ra)}")
    normal_scores = scores(nq, nk, args.tokens)
    rotated_scores = scores(rqr, rkr, args.tokens)
    print(f"qk_scores {metrics(normal_scores, rotated_scores)}")
    normal_softmax = softmax_rows(nq, nk, args.tokens)
    rotated_softmax = softmax_rows(rqr, rkr, args.tokens)
    print(f"softmax {metrics(normal_softmax, rotated_softmax)}")
    python_normal_attention = attention_output(nq, nk, nv, args.tokens)
    python_rotated_attention = attention_output(rqr, rkr, rvr, args.tokens)
    print(f"python_normal_attention_vs_runtime {metrics(python_normal_attention, na)}")
    print(f"python_rotated_attention_vs_runtime {metrics(python_rotated_attention, ra)}")
    if v_matrix is not None:
        expected_recovered = rotate_rows(python_rotated_attention, Q_HEADS, args.tokens,
                                         v_matrix, True)
        print(f"python_recovered_attention_vs_normal {metrics(python_normal_attention, expected_recovered)}")
        print(f"inverse_orientation {metrics(rotate_rows(ra, Q_HEADS, args.tokens, v_matrix, True), recovered)}")
    recovered_fp32_path = (args.rotated /
                           f"{args.tokens}.tokens_{args.tokens}.layer_{args.layer}.attention_recovered_fp32.fp32")
    if recovered_fp32_path.exists():
        recovered_fp32 = read_fp32(recovered_fp32_path)
        print(f"recovered_fp32_vs_normal {metrics(na, recovered_fp32)}")
        print(f"python_recovered_fp32_vs_normal {metrics(python_normal_attention, recovered_fp32)}")


if __name__ == "__main__":
    main()
