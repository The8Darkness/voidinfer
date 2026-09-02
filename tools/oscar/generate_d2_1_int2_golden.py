#!/usr/bin/env python3
"""Generate deterministic golden vectors for the official OSCAR INT2 codec.

The reference equations mirror the pinned FutureMLS-Lab OSCAR Triton path.  This
script deliberately emits all intermediate values needed to diagnose a codec
discrepancy; it does not call the runtime codec and it does not run attention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any

import torch


HEAD_DIM = 256
GROUP_SIZE = 128
GROUPS = 2
LEVELS = 3
K_CLIP = 0.96
V_CLIP = 0.92
MODEL_SHA256 = "6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e"
FULL_LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
UPSTREAM_COMMIT = "41ebcdba3db5f0ce1339c3727caea80df575d437"
UPSTREAM_SOURCE_SHA256 = "c1d7fd911c688cf29df9b98ce19fb48c6e7147ea6fcc81761e33cbf5f38b4157"
UPSTREAM_SOURCE_URL = (
    "https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/"
    "41ebcdba3db5f0ce1339c3727caea80df575d437/"
    "sglang-research/python/sglang/QuantKernel/oscar_rotation_clip_int2_kv.py"
)
STAGE = "post_qk_rmsnorm_post_rope_pre_causal_attention_cache_append"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def safe_path(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    require(not candidate.is_absolute(), f"{label}: absolute path")
    resolved = (root / candidate).resolve()
    require(root.resolve() in resolved.parents and resolved.is_file(), f"{label}: missing/escape")
    return resolved


def validate_capture_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path, dict[str, str]]:
    root = manifest_path.parent.resolve()
    sidecar = root / "manifest.sha256"
    require(sidecar.is_file(), "capture manifest sidecar is missing")
    sidecar_hash = sidecar.read_text(encoding="utf-8").split()[0]
    require(sidecar_hash == sha256_file(manifest_path), "capture manifest sidecar hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "oscar-qkv-v1", "unexpected capture schema")
    require(manifest.get("complete") is True, "capture manifest is not complete")
    require(manifest.get("useful_tokens") == 256, "D2.1 requires the validated 256-token capture")
    model = manifest.get("model", {})
    for field, expected in (("sha256", MODEL_SHA256), ("total_layers", 64),
                            ("query_heads", 24), ("kv_heads", 4),
                            ("head_dim", HEAD_DIM), ("rotary_dim", 64)):
        require(model.get(field) == expected, f"capture model field {field} mismatch")
    require(manifest.get("post_rope_qkv") is True, "capture is not post-RoPE")
    require(manifest.get("dtype") == "bf16", "capture is not BF16")
    require(manifest.get("capture_stage") == STAGE, "capture stage mismatch")

    records = {int(record["model_layer"]): record for record in manifest.get("captures", [])}
    require(tuple(sorted(records)) == FULL_LAYERS, "capture full-attention layer list mismatch")
    selected: dict[str, str] = {}
    for layer in (3, 35, 63):
        record = records[layer]
        require(record["kind"] == "full_attention" and record["chunk_id"] == 0,
                f"capture record mismatch for layer {layer}")
        require(record["dimensions"]["k"] == [256, 4, 256] and
                record["dimensions"]["v"] == [256, 4, 256],
                f"capture shape mismatch for layer {layer}")
        for kind in ("k", "v"):
            payload = safe_path(root, record["files"][kind], f"layer {layer} {kind}")
            actual_hash = sha256_file(payload)
            require(actual_hash == record["sha256"][kind],
                    f"capture hash mismatch for layer {layer} {kind}")
            require(payload.stat().st_size == 256 * 4 * HEAD_DIM * 2,
                    f"capture byte count mismatch for layer {layer} {kind}")
            selected[f"layer_{layer}_{kind}"] = actual_hash
    return manifest, root, selected


def read_bf16_rows(path: Path, rows: int) -> torch.Tensor:
    payload = path.read_bytes()
    expected = rows * 4 * HEAD_DIM * 2
    require(len(payload) == expected, f"unexpected BF16 payload size: {path}")
    words = torch.tensor(struct.unpack("<" + "H" * (len(payload) // 2), payload),
                         dtype=torch.int32)
    bits = (words << 16).contiguous()
    values = bits.view(torch.float32).reshape(rows, 4, HEAD_DIM)
    require(torch.isfinite(values).all().item(), f"non-finite BF16 payload: {path}")
    return values


def load_rotation(rotation_dir: Path, kind: str, layer: int) -> torch.Tensor:
    filename = "k_rotation_qqt_r_h_pbr.pt" if kind == "k" else "v_rotation_sst_r_h_pbr.pt"
    bank = torch.load(rotation_dir / filename, map_location="cpu", weights_only=True)
    matrix = bank["layers"][layer]["rotation"].to(dtype=torch.float32).contiguous()
    require(tuple(matrix.shape) == (HEAD_DIM, HEAD_DIM), f"rotation shape mismatch {kind} layer {layer}")
    require(torch.isfinite(matrix).all().item(), f"rotation is non-finite {kind} layer {layer}")
    return matrix


def official_encode(values: torch.Tensor, clip_ratio: float) -> dict[str, torch.Tensor]:
    """Mirror oscar_rotation_clip_int2_kv.py for one FP32 row."""
    row = values.to(dtype=torch.float32).contiguous()
    require(tuple(row.shape) == (HEAD_DIM,), "fixture row must have D=256")
    require(torch.isfinite(row).all().item(), "fixture row is non-finite")
    if clip_ratio <= 0.0:
        clipped = row.clone()
        clip_index = -1
        threshold = float("inf")
    else:
        clip_index = min(max(int(clip_ratio * HEAD_DIM), 0), HEAD_DIM - 1)
        threshold = torch.sort(row.abs())[0][clip_index].item()
        clipped = torch.clamp(row, -threshold, threshold)
    grouped = clipped.reshape(GROUPS, GROUP_SIZE)
    minimum = grouped.amin(dim=-1)
    maximum = grouped.amax(dim=-1)
    scale = torch.maximum(maximum - minimum, torch.full_like(minimum, 1.0e-8)) / LEVELS
    zero = -minimum / scale
    quantized_float = grouped / scale[:, None] + zero[:, None] + 0.5
    symbols = quantized_float.to(dtype=torch.uint8).reshape(HEAD_DIM)
    require(bool(((symbols <= LEVELS)).all().item()), "official fixture produced out-of-range symbol")
    packed = torch.empty(64, dtype=torch.uint8)
    for index in range(64):
        packed[index] = int(symbols[index].item()) | (int(symbols[index + 64].item()) << 2) | \
            (int(symbols[index + 128].item()) << 4) | (int(symbols[index + 192].item()) << 6)
    decoded = ((symbols.to(dtype=torch.float32) - zero.repeat_interleave(GROUP_SIZE)) *
               scale.repeat_interleave(GROUP_SIZE))
    return {
        "clipped": clipped,
        "scales_zeros": torch.stack((scale, zero), dim=1).reshape(4),
        "symbols": symbols,
        "packed": packed,
        "decoded": decoded,
        "clip_index": torch.tensor(clip_index, dtype=torch.int32),
        "threshold": torch.tensor(threshold, dtype=torch.float32),
    }


def rows_from_values(rows: torch.Tensor, name: str, source_kind: str,
                     source_layer: int, clip_ratio: float) -> dict[str, Any]:
    encoded = [official_encode(row, clip_ratio) for row in rows]
    return {
        "name": name,
        "source_kind": source_kind,
        "source_layer": source_layer,
        "clip_ratio": clip_ratio,
        "rows": rows.to(dtype=torch.float32).contiguous(),
        "encoded": encoded,
    }


def synthetic_cases() -> list[dict[str, Any]]:
    generator = torch.Generator().manual_seed(0xD201)
    random_rows = torch.randn((3, HEAD_DIM), generator=generator, dtype=torch.float32)
    positive = torch.linspace(0.125, 2.0, HEAD_DIM, dtype=torch.float32).reshape(1, HEAD_DIM)
    negative = -torch.linspace(0.125, 2.0, HEAD_DIM, dtype=torch.float32).reshape(1, HEAD_DIM)
    mixed = torch.linspace(-3.0, 3.0, HEAD_DIM, dtype=torch.float32).reshape(1, HEAD_DIM)
    near_zero = (torch.arange(HEAD_DIM, dtype=torch.float32) - 127.5).reshape(1, HEAD_DIM) * 1.0e-12
    outliers = torch.linspace(-1.0, 1.0, HEAD_DIM, dtype=torch.float32)
    outliers[3], outliers[127], outliers[128], outliers[251] = 80.0, -60.0, 100.0, -90.0
    boundary = torch.cat((
        torch.linspace(-0.75, 0.75, GROUP_SIZE),
        torch.linspace(10.0, 13.0, GROUP_SIZE),
    )).reshape(1, HEAD_DIM)
    multi = torch.stack((
        torch.arange(HEAD_DIM, dtype=torch.float32) / 31.0,
        -torch.arange(HEAD_DIM, dtype=torch.float32) / 17.0,
        torch.sin(torch.arange(HEAD_DIM, dtype=torch.float32) / 9.0),
    ))
    fragmented = torch.stack((random_rows[0], random_rows[1] * 0.03125, random_rows[2] * 17.0))
    return [
        rows_from_values(random_rows, "random_finite", "synthetic", -1, K_CLIP),
        rows_from_values(positive, "positive_only", "synthetic", -1, K_CLIP),
        rows_from_values(negative, "negative_only", "synthetic", -1, K_CLIP),
        rows_from_values(mixed, "mixed_sign", "synthetic", -1, K_CLIP),
        rows_from_values(near_zero, "near_zero_range", "synthetic", -1, K_CLIP),
        rows_from_values(outliers.reshape(1, HEAD_DIM), "clipping_sensitive_outliers", "synthetic", -1, K_CLIP),
        rows_from_values(boundary, "exact_group_boundary", "synthetic", -1, K_CLIP),
        rows_from_values(multi, "multiple_groups", "synthetic", -1, V_CLIP),
        rows_from_values(fragmented, "page_fragment_like_rows", "synthetic", -1, V_CLIP),
    ]


def real_cases(capture_root: Path, rotation_dir: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    manifest = json.loads((capture_root / "manifest.json").read_text(encoding="utf-8"))
    records = {int(record["model_layer"]): record for record in manifest["captures"]}
    for layer in (3, 35, 63):
        record = records[layer]
        for kind, clip_ratio in (("k", K_CLIP), ("v", V_CLIP)):
            path = capture_root / record["files"][kind]
            source = read_bf16_rows(path, 256)[:2].reshape(8, HEAD_DIM)
            rotation = load_rotation(rotation_dir, kind, layer)
            rotated = source @ rotation
            require(torch.isfinite(rotated).all().item(), f"rotated real sample is non-finite {layer} {kind}")
            cases.append(rows_from_values(rotated, f"real_layer_{layer}_{kind}_rotated",
                                          "real_rotated_capture", layer, clip_ratio))
    return cases


def write_string(stream: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    stream.write(struct.pack("<I", len(encoded)))
    stream.write(encoded)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().to("cpu").contiguous().reshape(-1).tolist()
    return struct.pack("<" + "f" * len(flat), *(float(value) for value in flat))


def write_fixture(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("wb") as stream:
        stream.write(b"OSCAR21\0")
        stream.write(struct.pack("<II", 1, len(cases)))
        for case in cases:
            write_string(stream, case["name"])
            write_string(stream, case["source_kind"])
            rows = case["rows"]
            stream.write(struct.pack("<iIf", case["source_layer"], rows.shape[0], case["clip_ratio"]))
            for row, result in zip(rows, case["encoded"]):
                for field in (row, result["clipped"], result["scales_zeros"],
                              result["symbols"], result["packed"], result["decoded"]):
                    if field.dtype == torch.uint8:
                        stream.write(bytes(field.tolist()))
                    else:
                        stream.write(tensor_bytes(field.to(dtype=torch.float32)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, capture_root, selected_hashes = validate_capture_manifest(args.capture_manifest)
    cases = synthetic_cases() + real_cases(capture_root, args.rotation_dir)
    fixture_path = args.output_dir / "golden.bin"
    write_fixture(fixture_path, cases)

    case_manifest = []
    for case in cases:
        all_results = case["encoded"]
        case_manifest.append({
            "name": case["name"],
            "source_kind": case["source_kind"],
            "source_layer": case["source_layer"],
            "clip_ratio": case["clip_ratio"],
            "clip_index": int(all_results[0]["clip_index"].item()),
            "rows": int(case["rows"].shape[0]),
            "input_sha256": sha256_bytes(tensor_bytes(case["rows"])),
            "thresholds": [float(result["threshold"].item()) for result in all_results],
            "packed_sha256": sha256_bytes(b"".join(bytes(result["packed"].tolist()) for result in all_results)),
        })
    source_files = {
        "generator": sha256_file(Path(__file__)),
        "capture_manifest": sha256_file(args.capture_manifest),
        "rotation_k_pt": sha256_file(args.rotation_dir / "k_rotation_qqt_r_h_pbr.pt"),
        "rotation_v_pt": sha256_file(args.rotation_dir / "v_rotation_sst_r_h_pbr.pt"),
    }
    output = {
        "schema": "oscar-d2-1-golden-v1",
        "upstream": {
            "repository": "FutureMLS-Lab/OSCAR",
            "commit": UPSTREAM_COMMIT,
            "source": UPSTREAM_SOURCE_URL,
            "source_sha256": UPSTREAM_SOURCE_SHA256,
        },
        "official_semantics": {
            "head_dim": HEAD_DIM,
            "group_size": GROUP_SIZE,
            "groups": GROUPS,
            "levels": LEVELS,
            "k_clip_ratio": K_CLIP,
            "v_clip_ratio": V_CLIP,
            "clip_rule": "threshold=sort(abs(row))[int(clip_ratio*head_dim)], symmetric clamp",
            "scale_rule": "max(group_max-group_min,1e-8)/3 in FP32",
            "zero_rule": "zero_point=-group_min/scale in FP32",
            "quant_rule": "uint8(value/scale+zero_point+0.5), symbols 0..3",
            "decode_rule": "(symbol-zero_point)*scale in FP32",
            "packing": "byte[j]=q[j]|q[j+64]<<2|q[j+128]<<4|q[j+192]<<6",
            "metadata": "FP32 [scale_group0,zero_group0,scale_group1,zero_group1]",
        },
        "model": {
            "sha256": MODEL_SHA256,
            "full_attention_layers": list(FULL_LAYERS),
            "kv_heads": 4,
            "head_dim": HEAD_DIM,
        },
        "capture": {
            "manifest": str(args.capture_manifest.resolve()),
            "manifest_sha256": sha256_file(args.capture_manifest),
            "schema": manifest["schema"],
            "stage": manifest["capture_stage"],
            "selected_payload_hashes": selected_hashes,
        },
        "source_files": source_files,
        "cases": case_manifest,
        "fixture": {
            "path": str(fixture_path.resolve()),
            "bytes": fixture_path.stat().st_size,
            "sha256": sha256_file(fixture_path),
            "case_count": len(cases),
            "row_count": sum(case["rows"].shape[0] for case in cases),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"fixture": str(fixture_path), "sha256": output["fixture"]["sha256"],
                      "cases": len(cases), "rows": output["fixture"]["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
