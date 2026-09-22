#!/usr/bin/env python3
"""Convert a validated native OSCAR smoke dump to the upstream .pt hierarchy.

The native dump stores exact BF16 bytes in [tokens, heads, head_dim] order. This
converter keeps those BF16 bit patterns, writes torch tensors without a float32
round trip, reloads every file, and records both source and .pt-file hashes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from validate_dump import sha256, validate


EXPECTED_LAYERS = tuple(range(3, 64, 4))
EXPECTED_TYPES = ("q", "k", "v")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def convert_tensor(
    source_root: Path,
    output_root: Path,
    item: dict[str, Any],
    tensor_type: str,
    chunk_id: int,
) -> dict[str, Any]:
    layer = int(item["model_layer"])
    shape = tuple(int(value) for value in item["dimensions"][tensor_type])
    source_path = (source_root / item["files"][tensor_type]).resolve()
    source_bytes = source_path.read_bytes()
    source_hash = sha256(source_path)
    require(source_hash == item["sha256"][tensor_type],
            f"layer {layer} {tensor_type}: source hash changed during conversion")
    require(len(source_bytes) == 2 * torch.tensor(shape).prod().item(),
            f"layer {layer} {tensor_type}: source byte count does not match shape")

    # uint16 -> bfloat16 is a bit-preserving view. The clone owns the bytes so
    # torch.save cannot retain a view into the temporary input buffer.
    source_bits = torch.frombuffer(bytearray(source_bytes), dtype=torch.uint16).clone()
    tensor = source_bits.view(torch.bfloat16).reshape(shape).contiguous()
    require(tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == shape,
            f"layer {layer} {tensor_type}: converted tensor metadata mismatch")

    output_path = output_root / f"layer_{layer}" / tensor_type / f"{chunk_id}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    require(isinstance(reloaded, torch.Tensor),
            f"layer {layer} {tensor_type}: reload did not produce a tensor")
    require(reloaded.dtype == torch.bfloat16 and tuple(reloaded.shape) == shape,
            f"layer {layer} {tensor_type}: reloaded tensor metadata mismatch")
    reloaded_bits = reloaded.contiguous().view(torch.uint16).reshape(-1)
    require(torch.equal(source_bits, reloaded_bits),
            f"layer {layer} {tensor_type}: BF16 values changed after .pt reload")

    return {
        "layer": layer,
        "qkv_type": tensor_type,
        "chunk_id": chunk_id,
        "shape": list(shape),
        "dtype": "torch.bfloat16",
        "source_file": item["files"][tensor_type],
        "source_sha256": source_hash,
        "pt_file": output_path.relative_to(output_root).as_posix(),
        "pt_sha256": sha256(output_path),
        "pt_bytes": output_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="validated native Phase B2 manifest.json")
    parser.add_argument("output", type=Path, help="fresh official OSCAR dump directory")
    parser.add_argument("--chunk-id", type=int, default=1,
                        help="explicit nonzero official fitter chunk id (default: 1)")
    args = parser.parse_args()

    require(args.chunk_id >= 1, "chunk id must be >= 1; upstream --chunk-id all skips chunk 0")
    manifest_path = args.manifest.resolve()
    source_root = manifest_path.parent
    output_root = args.output.resolve()
    require(manifest_path.is_file(), f"source manifest is missing: {manifest_path}")
    require(not output_root.exists(),
            f"output directory already exists; use a fresh directory: {output_root}")

    sidecar = source_root / "manifest.sha256"
    summary = validate(manifest_path, sidecar if sidecar.is_file() else None)
    require(summary["layers"] == len(EXPECTED_LAYERS) and summary["useful_tokens"] == 256,
            "source manifest did not pass the Phase B2 smoke contract")
    source_manifest_hash = sha256(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    captures = data["captures"]
    require(tuple(item["model_layer"] for item in captures) == EXPECTED_LAYERS,
            "source layer order is not the verified full-attention order")

    output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for item in captures:
            for tensor_type in EXPECTED_TYPES:
                records.append(convert_tensor(source_root, output_root, item, tensor_type, args.chunk_id))
    except Exception:
        shutil.rmtree(output_root)
        raise

    conversion_manifest = {
        "schema": "oscar-official-pt-conversion-v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": source_manifest_hash,
        "source_schema": data["schema"],
        "source_capture_stage": data["capture_stage"],
        "model": data["model"],
        "gqa": data["gqa"],
        "useful_tokens": data["useful_tokens"],
        "official_fitter_chunk_id": args.chunk_id,
        "chunk_policy": "explicit nonzero chunk; do not invoke upstream --chunk-id all",
        "tensor_layout": "[tokens, heads, head_dim]",
        "value_preservation": "BF16 uint16 bit patterns compared before and after torch.save/load",
        "tensors": records,
    }
    conversion_manifest_path = output_root / "conversion_manifest.json"
    conversion_manifest_path.write_text(
        json.dumps(conversion_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"OSCAR .pt conversion: PASS layers={len(EXPECTED_LAYERS)} tensors={len(records)} "
        f"tokens={data['useful_tokens']} chunk_id={args.chunk_id} "
        f"source_manifest_sha256={source_manifest_hash}"
    )
    print(f"conversion_manifest_sha256={sha256(conversion_manifest_path)}")
    for layer in (EXPECTED_LAYERS[0], EXPECTED_LAYERS[len(EXPECTED_LAYERS) // 2], EXPECTED_LAYERS[-1]):
        layer_records = [record for record in records if record["layer"] == layer]
        print(f"layer={layer} " + " ".join(
            f"{record['qkv_type']}={record['shape']}:{record['pt_sha256']}"
            for record in layer_records
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
