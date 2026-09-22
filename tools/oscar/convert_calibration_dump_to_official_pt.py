#!/usr/bin/env python3
"""Convert a validated multi-chunk native OSCAR capture to official .pt files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from validate_calibration_dump import EXPECTED_LAYERS, sha256, validate


TYPES = ("q", "k", "v")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def convert_tensor(source_root: Path, output_root: Path, item: dict[str, Any],
                   kind: str, official_chunk: int) -> dict[str, Any]:
    layer = int(item["model_layer"])
    shape = tuple(int(value) for value in item["dimensions"][kind])
    source_path = (source_root / item["files"][kind]).resolve()
    source_bytes = source_path.read_bytes()
    source_hash = sha256(source_path)
    require(source_hash == item["sha256"][kind],
            f"layer {layer} {kind}: source hash changed during conversion")
    require(len(source_bytes) == 2 * torch.tensor(shape).prod().item(),
            f"layer {layer} {kind}: source byte count/shape mismatch")
    source_bits = torch.frombuffer(bytearray(source_bytes), dtype=torch.uint16).clone()
    tensor = source_bits.view(torch.bfloat16).reshape(shape).contiguous()
    require(tensor.dtype == torch.bfloat16 and tuple(tensor.shape) == shape,
            f"layer {layer} {kind}: tensor metadata mismatch")
    output_path = output_root / f"layer_{layer}" / kind / f"{official_chunk}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, output_path)
    reloaded = torch.load(output_path, map_location="cpu", weights_only=True)
    require(isinstance(reloaded, torch.Tensor)
            and reloaded.dtype == torch.bfloat16
            and tuple(reloaded.shape) == shape,
            f"layer {layer} {kind}: .pt reload metadata mismatch")
    require(torch.equal(source_bits, reloaded.contiguous().view(torch.uint16).reshape(-1)),
            f"layer {layer} {kind}: BF16 values changed after .pt reload")
    return {
        "layer": layer,
        "qkv_type": kind,
        "raw_chunk_id": int(item["chunk_id"]),
        "official_chunk_id": official_chunk,
        "shape": list(shape),
        "dtype": "torch.bfloat16",
        "source_file": item["files"][kind],
        "source_sha256": source_hash,
        "pt_file": output_path.relative_to(output_root).as_posix(),
        "pt_sha256": sha256(output_path),
        "pt_bytes": output_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-useful-tokens", type=int, required=True)
    parser.add_argument("--expected-chunks", type=int, required=True)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    source_root = manifest_path.parent
    output_root = args.output.resolve()
    require(manifest_path.is_file(), f"source manifest is missing: {manifest_path}")
    require(not output_root.exists(), f"output directory already exists: {output_root}")
    sidecar = source_root / "manifest.sha256"
    require(sidecar.is_file(), "source manifest sidecar is required")
    summary = validate(manifest_path, sidecar, args.expected_useful_tokens, args.expected_chunks)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(tuple(sorted({int(item["model_layer"]) for item in data["captures"]}))
            == EXPECTED_LAYERS, "source layer set is not exact full-attention topology")
    output_root.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    try:
        for item in data["captures"]:
            # Upstream --chunk-id all deliberately omits chunk 0. The raw
            # capture starts at 0, so every official chunk is shifted by one.
            official_chunk = int(item["chunk_id"]) + 1
            for kind in TYPES:
                records.append(convert_tensor(source_root, output_root, item, kind,
                                              official_chunk))
    except Exception:
        shutil.rmtree(output_root)
        raise
    require(len(records) == args.expected_chunks * len(EXPECTED_LAYERS) * len(TYPES),
            "conversion record count mismatch")
    conversion = {
        "schema": "oscar-official-pt-conversion-v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "source_validation": summary,
        "source_schema": data["schema"],
        "source_capture_stage": data["capture_stage"],
        "model": data["model"],
        "gqa": data["gqa"],
        "useful_tokens": data["useful_tokens"],
        "raw_chunks": args.expected_chunks,
        "official_chunk_ids": list(range(1, args.expected_chunks + 1)),
        "chunk_policy": "raw chunk c mapped to official chunk c+1; official fitter --chunk-id all skips no useful data",
        "tensor_layout": "[tokens, heads, head_dim]",
        "target_runtime_group_size": 128,
        "value_preservation": "BF16 uint16 bit patterns compared before and after torch.save/load",
        "tensors": records,
    }
    path = output_root / "conversion_manifest.json"
    path.write_text(json.dumps(conversion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OSCAR calibration .pt conversion: PASS layers={len(EXPECTED_LAYERS)} "
          f"chunks={args.expected_chunks} tensors={len(records)} "
          f"tokens={data['useful_tokens']} source_manifest_sha256={sha256(manifest_path)}")
    print(f"conversion_manifest_sha256={sha256(path)}")
    for layer in (EXPECTED_LAYERS[0], EXPECTED_LAYERS[len(EXPECTED_LAYERS) // 2],
                  EXPECTED_LAYERS[-1]):
        selected = [record for record in records
                    if record["layer"] == layer and record["official_chunk_id"] in (1, args.expected_chunks)]
        print(f"representative layer={layer} " + " ".join(
            f"{item['qkv_type']}/chunk{item['official_chunk_id']}={item['shape']}:{item['pt_sha256']}"
            for item in selected
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
