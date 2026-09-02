#!/usr/bin/env python3
"""Fail-closed validator for multi-chunk native OSCAR calibration captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

SCHEMA = "oscar-qkv-v1"
EXPECTED_LAYERS = tuple(range(3, 64, 4))
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
STAGE = "post_qk_rmsnorm_post_rope_pre_causal_attention_cache_append"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_payload(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}: file path missing")
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"{label}: absolute file path is forbidden")
    payload = (root / candidate).resolve()
    require(payload != root and root in payload.parents, f"{label}: payload escapes root")
    require(payload.is_file(), f"{label}: payload is missing")
    return payload


def expected_shape(kind: str, tokens: int) -> list[int]:
    if kind == "q":
        return [tokens, Q_HEADS, HEAD_DIM]
    if kind in ("k", "v"):
        return [tokens, KV_HEADS, HEAD_DIM]
    raise ValueError(f"unknown tensor kind {kind!r}")


def expected_bytes(kind: str, tokens: int) -> int:
    return math.prod(expected_shape(kind, tokens)) * 2


def validate_sidecar(manifest: Path, sidecar: Path) -> None:
    fields = sidecar.read_text(encoding="utf-8").split()
    require(fields and SHA256_RE.fullmatch(fields[0]) is not None,
            "manifest SHA-256 sidecar is malformed")
    require(fields[0].lower() == sha256(manifest),
            "manifest SHA-256 sidecar does not match manifest")


def read_bf16(path: Path, expected_size: int, label: str) -> list[float]:
    payload = path.read_bytes()
    require(len(payload) == expected_size,
            f"{label}: expected {expected_size} bytes, got {len(payload)}")
    require(payload, f"{label}: payload is empty")
    bits = torch.frombuffer(bytearray(payload), dtype=torch.uint16).clone()
    values = bits.view(torch.bfloat16)
    require(torch.isfinite(values.float()).all().item(),
            f"{label}: NaN/Inf or non-finite BF16 value")
    return values[:4].float().tolist()


def validate(manifest_path: Path, sidecar: Path | None,
             expected_tokens: int, expected_chunks: int) -> dict[str, Any]:
    require(expected_tokens > 0 and expected_chunks > 0,
            "expected token/chunk counts must be positive")
    root = manifest_path.parent.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(data.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    require(data.get("post_rope_qkv") is True, "capture must be post-RoPE Q/K/V")
    require(data.get("dtype") == "bf16", "capture dtype must be bf16")
    require(data.get("capture_stage") == STAGE, "capture stage mismatch")
    require(data.get("complete") is True, "capture manifest is incomplete")
    require(data.get("expected_useful_tokens") == expected_tokens,
            "manifest expected token count mismatch")
    require(data.get("useful_tokens") == expected_tokens,
            "manifest useful token count mismatch")

    model = data.get("model")
    require(isinstance(model, dict), "model metadata is missing")
    require(model.get("id") == "qwen3.8-27b", "model id mismatch")
    require(model.get("total_layers") == 64, "expected 64 total layers")
    require(model.get("query_heads") == Q_HEADS and model.get("kv_heads") == KV_HEADS,
            "Q/KV head topology mismatch")
    require(model.get("head_dim") == HEAD_DIM and model.get("rotary_dim") == ROTARY_DIM,
            "head dimension or rotary dimension mismatch")
    model_sha = model.get("sha256")
    require(isinstance(model_sha, str) and SHA256_RE.fullmatch(model_sha) is not None,
            "model SHA-256 is missing or malformed")

    executable = data.get("executable")
    require(isinstance(executable, dict), "executable metadata is missing")
    executable_sha = executable.get("sha256")
    require(isinstance(executable_sha, str) and SHA256_RE.fullmatch(executable_sha) is not None,
            "executable SHA-256 is missing or malformed")
    source_identity = data.get("source_identity")
    require(isinstance(source_identity, str) and source_identity,
            "source identity is missing")

    gqa = data.get("gqa")
    require(isinstance(gqa, dict) and gqa.get("ratio") == 6,
            "GQA ratio must be 6")
    require(gqa.get("q_head_to_kv_head") == [head // 6 for head in range(Q_HEADS)],
            "GQA mapping is invalid")

    input_meta = data.get("input")
    require(isinstance(input_meta, dict) and isinstance(input_meta.get("kind"), str)
            and input_meta["kind"], "input provenance is missing")

    captures = data.get("captures")
    require(isinstance(captures, list), "captures must be a list")
    require(len(captures) == len(EXPECTED_LAYERS) * expected_chunks,
            "capture count does not match full-layer/chunk topology")
    expected_pairs = {(layer, chunk) for chunk in range(expected_chunks)
                      for layer in EXPECTED_LAYERS}
    capture_index: dict[tuple[int, int], dict[str, Any]] = {}
    listed_files: set[str] = set()
    total_bytes = 0
    representative: dict[str, dict[str, list[float]]] = {}
    for index, item in enumerate(captures):
        require(isinstance(item, dict), f"capture {index}: record is not an object")
        layer = item.get("model_layer")
        chunk = item.get("chunk_id")
        tokens = item.get("tokens")
        require(layer in EXPECTED_LAYERS,
                f"capture {index}: GDN or unknown layer {layer!r}")
        require(isinstance(chunk, int) and 0 <= chunk < expected_chunks,
                f"capture {index}: chunk id is out of range")
        require(isinstance(tokens, int) and tokens > 0,
                f"capture {index}: invalid token count")
        require(item.get("full_attention_index") == EXPECTED_LAYERS.index(layer),
                f"capture {index}: full-attention index mismatch")
        require(item.get("kind") == "full_attention" and item.get("dtype") == "bf16",
                f"capture {index}: kind/dtype mismatch")
        require(item.get("capture_stage") == STAGE,
                f"capture {index}: capture stage mismatch")
        require(item.get("q_heads") == Q_HEADS and item.get("kv_heads") == KV_HEADS
                and item.get("head_dim") == HEAD_DIM,
                f"capture {index}: geometry mismatch")
        key = (layer, chunk)
        require(key not in capture_index, f"capture {index}: duplicate layer/chunk")
        capture_index[key] = item
        require(tokens == captures[0].get("tokens"),
                f"capture {index}: chunk token count differs")

        dimensions = item.get("dimensions")
        files = item.get("files")
        hashes = item.get("sha256")
        require(isinstance(dimensions, dict) and isinstance(files, dict)
                and isinstance(hashes, dict), f"capture {index}: maps are missing")
        is_rep = layer in (EXPECTED_LAYERS[0], EXPECTED_LAYERS[len(EXPECTED_LAYERS) // 2],
                           EXPECTED_LAYERS[-1]) and chunk in (0, expected_chunks - 1)
        if is_rep:
            representative[f"layer_{layer}/chunk_{chunk:04d}"] = {}
        for kind in ("q", "k", "v"):
            shape = expected_shape(kind, tokens)
            require(dimensions.get(kind) == shape,
                    f"capture {layer}/chunk_{chunk}/{kind}: dimensions mismatch")
            payload = safe_payload(root, files.get(kind),
                                   f"capture {layer}/chunk_{chunk}/{kind}")
            expected_hash = hashes.get(kind)
            require(isinstance(expected_hash, str)
                    and SHA256_RE.fullmatch(expected_hash) is not None,
                    f"capture {layer}/chunk_{chunk}/{kind}: hash missing/malformed")
            preview = read_bf16(payload, expected_bytes(kind, tokens),
                                f"capture {layer}/chunk_{chunk}/{kind}")
            require(sha256(payload) == expected_hash.lower(),
                    f"capture {layer}/chunk_{chunk}/{kind}: content hash mismatch")
            listed_files.add(payload.relative_to(root).as_posix())
            total_bytes += payload.stat().st_size
            if is_rep:
                representative[f"layer_{layer}/chunk_{chunk:04d}"][kind] = preview

    require(set(capture_index) == expected_pairs,
            "capture set is not exactly every full-attention layer/chunk pair")
    require(data.get("capture_count") == len(captures),
            "capture_count does not match captures")
    require(data.get("dump_bytes") == total_bytes,
            "dump_bytes does not match payload sizes")
    chunk_records = [captures[index] for index in range(0, len(captures), len(EXPECTED_LAYERS))]
    require(sum(item["tokens"] for item in chunk_records) == expected_tokens,
            "chunk token total does not match expected useful tokens")

    tensor_records = data.get("tensor_records")
    require(isinstance(tensor_records, list)
            and len(tensor_records) == len(captures) * 3,
            "tensor_records must contain exactly three records per capture")
    tensor_index: set[tuple[int, int, str]] = set()
    for index, record in enumerate(tensor_records):
        require(isinstance(record, dict), f"tensor record {index}: invalid record")
        layer = record.get("layer_index")
        chunk = record.get("chunk_id")
        kind = record.get("qkv_type")
        require(layer in EXPECTED_LAYERS and (layer, chunk) in capture_index,
                f"tensor record {index}: GDN/unknown layer or chunk")
        require(kind in ("q", "k", "v"), f"tensor record {index}: invalid Q/K/V type")
        require(record.get("model_sha256") == model_sha
                and record.get("executable_sha256") == executable_sha
                and record.get("source_identity") == source_identity,
                f"tensor record {index}: provenance mismatch")
        require(record.get("full_attention_index") == EXPECTED_LAYERS.index(layer)
                and record.get("dtype") == "bf16"
                and record.get("q_heads") == Q_HEADS
                and record.get("kv_heads") == KV_HEADS
                and record.get("head_dim") == HEAD_DIM,
                f"tensor record {index}: geometry mismatch")
        capture = capture_index[(layer, chunk)]
        require(record.get("token_count") == capture["tokens"]
                and record.get("capture_stage") == STAGE,
                f"tensor record {index}: token/stage mismatch")
        key = (layer, chunk, kind)
        require(key not in tensor_index, f"tensor record {index}: duplicate record")
        tensor_index.add(key)
        require(record.get("dimensions") == capture["dimensions"][kind]
                and record.get("file") == capture["files"][kind]
                and record.get("data_sha256") == capture["sha256"][kind]
                and record.get("byte_count") == expected_bytes(kind, capture["tokens"]),
                f"tensor record {index}: capture disagreement")
    require(tensor_index == {(layer, chunk, kind) for layer, chunk in expected_pairs
                             for kind in ("q", "k", "v")},
            "tensor records are not exactly every Q/K/V capture")

    actual_bin_files = {path.relative_to(root).as_posix() for path in root.rglob("*.bin")}
    require(actual_bin_files == listed_files,
            "unlisted or missing .bin payload exists")
    if sidecar is not None:
        require(sidecar.is_file(), "manifest SHA-256 sidecar is missing")
        validate_sidecar(manifest_path, sidecar)

    return {
        "layers": len(EXPECTED_LAYERS),
        "chunks": expected_chunks,
        "chunk_tokens": captures[0]["tokens"],
        "useful_tokens": expected_tokens,
        "dump_bytes": total_bytes,
        "manifest_sha256": sha256(manifest_path),
        "input_kind": input_meta["kind"],
        "representative": representative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--manifest-sha256", type=Path, required=True)
    parser.add_argument("--expected-useful-tokens", type=int, required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    args = parser.parse_args()
    try:
        summary = validate(args.manifest.resolve(), args.manifest_sha256.resolve(),
                           args.expected_useful_tokens, args.expected_chunks)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"OSCAR calibration QKV dump: FAIL: {error}")
        return 1
    print("OSCAR calibration QKV dump: PASS "
          f"layers={summary['layers']} chunks={summary['chunks']} "
          f"chunk_tokens={summary['chunk_tokens']} useful_tokens={summary['useful_tokens']} "
          f"dump_bytes={summary['dump_bytes']} "
          f"manifest_sha256={summary['manifest_sha256']}")
    for label, preview in summary["representative"].items():
        print(f"representative {label} "
              f"q0={preview['q']} k0={preview['k']} v0={preview['v']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
