"""Export validated OSCAR .pt matrices to a runtime-safe raw FP32 bank."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
IDENTITY = "qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1"
MODEL_SHA256 = "6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e"
ASSET_MANIFEST_SHA256 = "4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560"


def load_bank(path: Path) -> bytes:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("layers"), dict):
        raise ValueError(f"unexpected rotation asset structure: {path}")
    rows: list[bytes] = []
    if tuple(sorted(payload["layers"])) != LAYERS:
        raise ValueError(f"layer mapping mismatch in {path}")
    for layer in LAYERS:
        entry = payload["layers"][layer]
        tensor = entry.get("rotation") if isinstance(entry, dict) else None
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (256, 256):
            raise ValueError(f"invalid layer {layer} shape in {path}")
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError(f"invalid layer {layer} dtype/device in {path}")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"non-finite layer {layer} in {path}")
        # Avoid depending on NumPy in the intentionally CPU-only calibration environment.
        rows.append(bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist()))
    return b"".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=Path, required=True)
    parser.add_argument("--v", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", default=IDENTITY)
    parser.add_argument("--model-sha256", default=MODEL_SHA256)
    parser.add_argument("--asset-manifest-sha256", default=ASSET_MANIFEST_SHA256)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("k_rotation_fp32.bin", "v_rotation_fp32.bin", "runtime_manifest.txt")
    if any((output / name).exists() for name in names):
        raise FileExistsError(f"refusing to overwrite existing runtime asset in {output}")

    k_bytes = load_bank(args.k)
    v_bytes = load_bank(args.v)
    expected = 16 * 256 * 256 * 4
    if len(k_bytes) != expected or len(v_bytes) != expected:
        raise ValueError("runtime rotation bank has an invalid byte count")
    (output / names[0]).write_bytes(k_bytes)
    (output / names[1]).write_bytes(v_bytes)
    manifest = "\n".join(
        [
            "schema=oscar-runtime-rotation-v1",
            f"asset_identity={args.identity}",
            f"model_sha256={args.model_sha256}",
            f"asset_manifest_sha256={args.asset_manifest_sha256}",
            "total_layers=64",
            f"full_attention_layers={','.join(map(str, LAYERS))}",
            "q_heads=24",
            "kv_heads=4",
            "gqa_ratio=6",
            "head_dim=256",
            "rotary_dim=64",
            "layout=runtime[d,h,t];source[tokens,heads,head_dim]",
            "dtype=fp32",
            "calibrated=true",
            "rotation_mode=qqt_sst+r_h_pbr",
            "k_layers=16",
            "v_layers=16",
            f"k_file={names[0]}",
            f"v_file={names[1]}",
            f"k_sha256={hashlib.sha256(k_bytes).hexdigest()}",
            f"v_sha256={hashlib.sha256(v_bytes).hexdigest()}",
            "",
        ]
    )
    (output / names[2]).write_text(manifest, encoding="ascii")
    print(f"output={output}")
    print(f"k_sha256={hashlib.sha256(k_bytes).hexdigest()}")
    print(f"v_sha256={hashlib.sha256(v_bytes).hexdigest()}")
    print(f"bytes_each={len(k_bytes)}")


if __name__ == "__main__":
    main()
