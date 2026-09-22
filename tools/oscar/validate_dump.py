#!/usr/bin/env python3
"""Fail-closed validator for the native OSCAR Q/K/V smoke capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "oscar-qkv-v1"
EXPECTED_FULL_LAYERS = tuple(range(3, 64, 4))
EXPECTED_Q_HEADS = 24
EXPECTED_KV_HEADS = 4
EXPECTED_HEAD_DIM = 256
EXPECTED_ROTARY_DIM = 64
EXPECTED_USEFUL_TOKENS = 256
EXPECTED_STAGE = "post_qk_rmsnorm_post_rope_pre_causal_attention_cache_append"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_payload(root: Path, relative: Any, label: str) -> Path:
    require(isinstance(relative, str) and relative, f"{label}: file path missing")
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"{label}: absolute file path is forbidden")
    payload = (root / candidate).resolve()
    require(payload != root and root in payload.parents, f"{label}: payload escapes manifest directory")
    require(payload.is_file(), f"{label}: payload is missing")
    return payload


def read_bf16(path: Path, expected_bytes: int, label: str) -> tuple[bytes, list[float]]:
    data = path.read_bytes()
    require(len(data) == expected_bytes, f"{label}: expected {expected_bytes} bytes, got {len(data)}")
    require(data, f"{label}: payload is empty")
    values: list[float] = []
    for offset in range(0, len(data), 2):
        bits = int.from_bytes(data[offset:offset + 2], "little")
        require((bits & 0x7F80) != 0x7F80, f"{label}: NaN/Inf at BF16 element {offset // 2}")
        value = struct.unpack("<f", struct.pack("<I", bits << 16))[0]
        require(math.isfinite(value), f"{label}: non-finite value at element {offset // 2}")
        if len(values) < 4:
            values.append(value)
    return data, values


def expected_shape(tensor_type: str, tokens: int) -> list[int]:
    if tensor_type == "q":
        return [tokens, EXPECTED_Q_HEADS, EXPECTED_HEAD_DIM]
    if tensor_type in ("k", "v"):
        return [tokens, EXPECTED_KV_HEADS, EXPECTED_HEAD_DIM]
    raise ValueError(f"unknown Q/K/V type {tensor_type!r}")


def expected_bytes(tensor_type: str, tokens: int) -> int:
    shape = expected_shape(tensor_type, tokens)
    return math.prod(shape) * 2


def validate_manifest_hash(manifest_path: Path, sidecar: Path) -> None:
    fields = sidecar.read_text(encoding="utf-8").split()
    require(fields and SHA256_RE.fullmatch(fields[0]) is not None,
            "manifest SHA-256 sidecar is malformed")
    require(fields[0].lower() == sha256(manifest_path),
            "manifest SHA-256 sidecar does not match manifest")


def validate(manifest_path: Path, manifest_sha256_path: Path | None = None) -> dict[str, Any]:
    root = manifest_path.parent.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(data.get("schema") == SCHEMA, f"schema must be {SCHEMA!r}")
    require(data.get("post_rope_qkv") is True, "captures must be post-RoPE Q/K/V")
    require(data.get("dtype") == "bf16", "capture dtype must be bf16")
    require(data.get("capture_stage") == EXPECTED_STAGE, "capture stage mismatch")
    require(data.get("complete") is True, "capture manifest is incomplete")
    require(data.get("expected_useful_tokens") == EXPECTED_USEFUL_TOKENS,
            "smoke manifest must target 256 useful tokens")
    require(data.get("useful_tokens") == EXPECTED_USEFUL_TOKENS,
            "smoke manifest does not contain exactly 256 useful tokens")

    model = data.get("model")
    require(isinstance(model, dict), "model metadata is missing")
    require(model.get("id") == "qwen3.8-27b", "model id mismatch")
    require(model.get("total_layers") == 64, "expected 64 transformer layers")
    require(model.get("query_heads") == EXPECTED_Q_HEADS, "expected 24 query heads")
    require(model.get("kv_heads") == EXPECTED_KV_HEADS, "expected 4 KV heads")
    require(model.get("head_dim") == EXPECTED_HEAD_DIM, "expected head dimension 256")
    require(model.get("rotary_dim") == EXPECTED_ROTARY_DIM, "expected rotary dimension 64")
    model_sha = model.get("sha256")
    require(isinstance(model_sha, str) and SHA256_RE.fullmatch(model_sha) is not None,
            "model SHA-256 is missing or malformed")

    executable = data.get("executable")
    require(isinstance(executable, dict), "executable metadata is missing")
    executable_sha = executable.get("sha256")
    require(isinstance(executable_sha, str) and SHA256_RE.fullmatch(executable_sha) is not None,
            "executable SHA-256 is missing or malformed")
    source_identity = data.get("source_identity")
    require(isinstance(source_identity, str) and source_identity, "source identity is missing")

    gqa = data.get("gqa")
    require(isinstance(gqa, dict) and gqa.get("ratio") == 6, "GQA ratio must be 6")
    require(gqa.get("q_head_to_kv_head") == [head // 6 for head in range(EXPECTED_Q_HEADS)],
            "GQA q-head to KV-head mapping is invalid")

    captures = data.get("captures")
    require(isinstance(captures, list), "captures must be a list")
    require(len(captures) == len(EXPECTED_FULL_LAYERS),
            "smoke capture must contain exactly one chunk for all 16 full-attention layers")
    actual_layers = tuple(item.get("model_layer") for item in captures)
    require(actual_layers == EXPECTED_FULL_LAYERS,
            "full-attention layer ids must be exactly 3,7,...,63 in order")

    capture_index: dict[tuple[int, int], dict[str, Any]] = {}
    total_bytes = 0
    representative: dict[int, dict[str, list[float]]] = {}
    for capture_index_in_list, item in enumerate(captures):
        require(isinstance(item, dict), f"capture {capture_index_in_list}: record is not an object")
        layer = item.get("model_layer")
        full_index = item.get("full_attention_index")
        chunk_id = item.get("chunk_id")
        tokens = item.get("tokens")
        require(layer in EXPECTED_FULL_LAYERS,
                f"capture {capture_index_in_list}: GDN or unknown layer {layer!r}")
        require(full_index == EXPECTED_FULL_LAYERS.index(layer),
                f"capture {capture_index_in_list}: full-attention index mismatch")
        require(chunk_id == 0, f"capture {capture_index_in_list}: smoke chunk id must be 0")
        require(item.get("kind") == "full_attention",
                f"capture {capture_index_in_list}: kind must be full_attention")
        require(item.get("dtype") == "bf16", f"capture {capture_index_in_list}: dtype mismatch")
        require(item.get("capture_stage") == EXPECTED_STAGE,
                f"capture {capture_index_in_list}: capture stage mismatch")
        require(isinstance(tokens, int) and tokens == EXPECTED_USEFUL_TOKENS,
                f"capture {capture_index_in_list}: token count must be 256")
        require(item.get("q_heads") == EXPECTED_Q_HEADS and
                item.get("kv_heads") == EXPECTED_KV_HEADS and
                item.get("head_dim") == EXPECTED_HEAD_DIM,
                f"capture {capture_index_in_list}: head geometry mismatch")
        key = (layer, chunk_id)
        require(key not in capture_index, f"capture {capture_index_in_list}: duplicate layer/chunk")
        capture_index[key] = item

        dimensions = item.get("dimensions")
        require(isinstance(dimensions, dict), f"capture {capture_index_in_list}: dimensions missing")
        files = item.get("files")
        hashes = item.get("sha256")
        require(isinstance(files, dict) and isinstance(hashes, dict),
                f"capture {capture_index_in_list}: file/hash maps missing")
        representative[layer] = {}
        for tensor_type in ("q", "k", "v"):
            shape = expected_shape(tensor_type, tokens)
            require(dimensions.get(tensor_type) == shape,
                    f"capture {capture_index_in_list}: {tensor_type} dimensions mismatch")
            payload = safe_payload(root, files.get(tensor_type),
                                   f"capture {layer}/chunk_{chunk_id}/{tensor_type}")
            expected_hash = hashes.get(tensor_type)
            require(isinstance(expected_hash, str) and SHA256_RE.fullmatch(expected_hash) is not None,
                    f"capture {layer}/chunk_{chunk_id}/{tensor_type}: hash missing or malformed")
            expected_size = expected_bytes(tensor_type, tokens)
            payload_bytes, preview = read_bf16(
                payload, expected_size, f"capture {layer}/chunk_{chunk_id}/{tensor_type}")
            require(sha256(payload) == expected_hash.lower(),
                    f"capture {layer}/chunk_{chunk_id}/{tensor_type}: content hash mismatch")
            representative[layer][tensor_type] = preview
            total_bytes += len(payload_bytes)

    require(data.get("capture_count") == len(captures), "capture_count does not match captures")
    require(data.get("dump_bytes") == total_bytes, "dump_bytes does not match payload sizes")

    tensor_records = data.get("tensor_records")
    require(isinstance(tensor_records, list) and len(tensor_records) == len(captures) * 3,
            "tensor_records must contain exactly three records per layer/chunk")
    tensor_index: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index, record in enumerate(tensor_records):
        require(isinstance(record, dict), f"tensor record {index}: record is not an object")
        layer = record.get("layer_index")
        chunk_id = record.get("chunk_id")
        tensor_type = record.get("qkv_type")
        require(layer in EXPECTED_FULL_LAYERS,
                f"tensor record {index}: GDN or unknown layer {layer!r}")
        require(tensor_type in ("q", "k", "v"), f"tensor record {index}: Q/K/V type is invalid")
        require(record.get("model_sha256") == model_sha,
                f"tensor record {index}: model SHA-256 mismatch")
        require(record.get("executable_sha256") == executable_sha,
                f"tensor record {index}: executable SHA-256 mismatch")
        require(record.get("source_identity") == source_identity,
                f"tensor record {index}: source identity mismatch")
        require(record.get("full_attention_index") == EXPECTED_FULL_LAYERS.index(layer),
                f"tensor record {index}: full-attention index mismatch")
        require(record.get("dtype") == "bf16", f"tensor record {index}: dtype mismatch")
        require(record.get("q_heads") == EXPECTED_Q_HEADS and
                record.get("kv_heads") == EXPECTED_KV_HEADS and
                record.get("head_dim") == EXPECTED_HEAD_DIM,
                f"tensor record {index}: head geometry mismatch")
        require(record.get("token_count") == EXPECTED_USEFUL_TOKENS,
                f"tensor record {index}: token count mismatch")
        require(record.get("capture_stage") == EXPECTED_STAGE,
                f"tensor record {index}: capture stage mismatch")
        key = (layer, chunk_id, tensor_type)
        require(key not in tensor_index, f"tensor record {index}: duplicate record")
        tensor_index[key] = record
        capture = capture_index.get((layer, chunk_id))
        require(capture is not None, f"tensor record {index}: layer/chunk capture is missing")
        require(record.get("dimensions") == capture["dimensions"][tensor_type],
                f"tensor record {index}: dimensions disagree with capture")
        require(record.get("file") == capture["files"][tensor_type],
                f"tensor record {index}: file disagrees with capture")
        require(record.get("data_sha256") == capture["sha256"][tensor_type],
                f"tensor record {index}: hash disagrees with capture")
        require(record.get("byte_count") == expected_bytes(tensor_type, EXPECTED_USEFUL_TOKENS),
                f"tensor record {index}: byte count mismatch")

    for layer in EXPECTED_FULL_LAYERS:
        for tensor_type in ("q", "k", "v"):
            require((layer, 0, tensor_type) in tensor_index,
                    f"missing {tensor_type} tensor for full-attention layer {layer}")

    if manifest_sha256_path is not None:
        require(manifest_sha256_path.is_file(), "manifest SHA-256 sidecar is missing")
        validate_manifest_hash(manifest_path, manifest_sha256_path)

    return {
        "layers": len(EXPECTED_FULL_LAYERS),
        "chunks": 1,
        "useful_tokens": EXPECTED_USEFUL_TOKENS,
        "dump_bytes": total_bytes,
        "representative": representative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--manifest-sha256", type=Path)
    args = parser.parse_args()
    try:
        manifest_path = args.manifest.resolve()
        summary = validate(manifest_path, args.manifest_sha256.resolve()
                           if args.manifest_sha256 is not None else None)
    except (OSError, ValueError, json.JSONDecodeError, struct.error) as error:
        print(f"OSCAR QKV dump: FAIL: {error}")
        return 1

    manifest_hash = sha256(manifest_path)
    print(f"OSCAR QKV dump: PASS layers={summary['layers']} chunks={summary['chunks']} "
          f"useful_tokens={summary['useful_tokens']} dump_bytes={summary['dump_bytes']} "
          f"manifest_sha256={manifest_hash}")
    for layer in (EXPECTED_FULL_LAYERS[0], EXPECTED_FULL_LAYERS[len(EXPECTED_FULL_LAYERS) // 2],
                  EXPECTED_FULL_LAYERS[-1]):
        preview = summary["representative"][layer]
        print(f"representative layer={layer} "
              f"q0={preview['q']} k0={preview['k']} v0={preview['v']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
