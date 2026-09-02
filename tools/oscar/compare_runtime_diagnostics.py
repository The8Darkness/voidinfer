"""Compare the optional real-runtime BF16 layer diagnostics."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def read_bf16(path: Path) -> list[float]:
    data = path.read_bytes()
    if len(data) % 2:
        raise ValueError(f"odd BF16 byte count: {path}")
    values = []
    for (bits,) in struct.iter_unpack("<H", data):
        (value,) = struct.unpack("<f", struct.pack("<I", bits << 16))
        if not math.isfinite(value):
            raise ValueError(f"non-finite value: {path}")
        values.append(value)
    return values


def metrics(reference: list[float], candidate: list[float]) -> tuple[float, float, float]:
    if len(reference) != len(candidate):
        raise ValueError("diagnostic shapes differ")
    diffs = [abs(a - b) for a, b in zip(reference, candidate)]
    ref_l2 = math.sqrt(sum(a * a for a in reference))
    diff_l2 = math.sqrt(sum(d * d for d in diffs))
    return max(diffs), sum(diffs) / len(diffs), diff_l2 / max(ref_l2, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--rotated", type=Path, required=True)
    parser.add_argument("--stage", default="layer",
                        help="diagnostic stage suffix without .bf16 (default: layer)")
    args = parser.parse_args()
    layers = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)
    first = None
    for layer in range(64):
        if args.stage != "layer" and layer not in layers:
            continue
        suffix = ".bf16" if args.stage == "layer" else f".{args.stage}.bf16"
        name = f".tokens_32.layer_{layer}{suffix}"
        ref = read_bf16(Path(str(args.normal) + name))
        got = read_bf16(Path(str(args.rotated) + name))
        max_abs, mean_abs, relative_l2 = metrics(ref, got)
        print(f"layer={layer} full_attention={layer in layers} max_abs={max_abs:.9e} "
              f"mean_abs={mean_abs:.9e} relative_l2={relative_l2:.9e}")
        if first is None and max_abs != 0.0:
            first = layer
    print(f"first_nonidentical_layer={first}")


if __name__ == "__main__":
    main()
