#!/usr/bin/env python3
"""Fail-closed validation for official OSCAR rotation checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import torch


EXPECTED_LAYERS = tuple(range(3, 64, 4))
HEAD_DIM = 256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"rotation asset is missing: {path}")
    require(path.stat().st_size > 0, f"rotation asset is empty: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict), f"rotation asset is not a checkpoint dict: {path}")
    return state


def validate_asset(path: Path, objective: str) -> dict[str, Any]:
    first = load_state(path)
    second = load_state(path)
    require(first.get("format_version") == 1, f"{path.name}: format version mismatch")
    require(first.get("objective") == objective, f"{path.name}: objective mismatch")
    require(first.get("source_grouping") == "layer", f"{path.name}: source grouping mismatch")
    layers = first.get("layers")
    second_layers = second.get("layers")
    require(isinstance(layers, dict) and isinstance(second_layers, dict),
            f"{path.name}: layers map is missing")
    normalized_layers = {int(key): value for key, value in layers.items()}
    normalized_second = {int(key): value for key, value in second_layers.items()}
    require(set(normalized_layers) == set(EXPECTED_LAYERS),
            f"{path.name}: layer mapping is not exactly {EXPECTED_LAYERS}")
    require(set(normalized_second) == set(EXPECTED_LAYERS),
            f"{path.name}: reload layer mapping changed")

    rows: list[dict[str, Any]] = []
    for layer in EXPECTED_LAYERS:
        entry = normalized_layers[layer]
        reloaded = normalized_second[layer]
        require(isinstance(entry, dict), f"{path.name}: layer {layer} entry is invalid")
        require(entry.get("layer_id") == layer, f"{path.name}: layer {layer} id mismatch")
        rotation = entry.get("rotation")
        eigenvalues = entry.get("eigenvalues")
        reload_rotation = reloaded.get("rotation") if isinstance(reloaded, dict) else None
        reload_eigenvalues = reloaded.get("eigenvalues") if isinstance(reloaded, dict) else None
        require(isinstance(rotation, torch.Tensor) and isinstance(eigenvalues, torch.Tensor),
                f"{path.name}: layer {layer} tensors are missing")
        require(rotation.dtype == torch.float32 and eigenvalues.dtype == torch.float32,
                f"{path.name}: layer {layer} asset is not serialized FP32")
        require(tuple(rotation.shape) == (HEAD_DIM, HEAD_DIM),
                f"{path.name}: layer {layer} rotation shape mismatch")
        require(tuple(eigenvalues.shape) == (HEAD_DIM,),
                f"{path.name}: layer {layer} eigenvalue shape mismatch")
        require(torch.isfinite(rotation).all().item() and torch.isfinite(eigenvalues).all().item(),
                f"{path.name}: layer {layer} contains non-finite values")
        require(isinstance(reload_rotation, torch.Tensor) and
                isinstance(reload_eigenvalues, torch.Tensor) and
                torch.equal(rotation, reload_rotation) and
                torch.equal(eigenvalues, reload_eigenvalues),
                f"{path.name}: layer {layer} load/reload is not deterministic")
        identity = torch.eye(HEAD_DIM, dtype=torch.float64)
        orthogonality_error = float(
            (rotation.double() @ rotation.double().T - identity).abs().max().item()
        )
        require(math.isfinite(orthogonality_error),
                f"{path.name}: layer {layer} orthogonality error is non-finite")
        rows.append({"layer": layer, "max_abs_r_rt_minus_i": orthogonality_error})

    return {
        "path": str(path),
        "sha256": sha256(path),
        "objective": objective,
        "dtype": "torch.float32",
        "rotation_shape": [HEAD_DIM, HEAD_DIM],
        "layers": rows,
        "max_orthogonality_error": max(row["max_abs_r_rt_minus_i"] for row in rows),
    }


def validate_composition(official_script: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("official_oscar_rotation", official_script)
    require(spec is not None and spec.loader is not None, "cannot load official fitter module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dimension = 8
    eigenvalues = torch.tensor([0.25, 7.0, -3.0, 2.0, 5.0, -1.0, 4.0, 1.0], dtype=torch.float64)
    sample = torch.arange(1, dimension * dimension + 1, dtype=torch.float64).reshape(
        dimension, dimension
    )
    rotation, _ = torch.linalg.qr(sample)

    # Independent fixture construction: explicit Sylvester H_8 and explicit
    # descending-eigenvalue bit reversal, rather than reusing fitter helpers.
    hadamard = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1],
            [1, -1, -1, 1, 1, -1, -1, 1],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [1, -1, 1, -1, -1, 1, -1, 1],
            [1, 1, -1, -1, -1, -1, 1, 1],
            [1, -1, -1, 1, -1, 1, 1, -1],
        ],
        dtype=torch.float64,
    ) / math.sqrt(dimension)
    sorted_indices = torch.argsort(eigenvalues, descending=True).tolist()
    permutation = torch.zeros(dimension, dtype=torch.long)
    for index in range(dimension):
        reversed_index = int(f"{index:03b}"[::-1], 2)
        permutation[reversed_index] = sorted_indices[index]
    pbr = torch.eye(dimension, dtype=torch.float64)[:, permutation]
    expected = rotation @ hadamard @ pbr
    actual = module.compose_rotation(
        rotation, eigenvalues, module.build_hadamard(dimension), "r_h_pbr"
    )
    composition_error = float((actual - expected).abs().max().item())
    pbr_orthogonality_error = float(
        (pbr @ pbr.T - torch.eye(dimension, dtype=torch.float64)).abs().max().item()
    )
    require(composition_error <= 1e-12, f"R*H*Pbr fixture mismatch: {composition_error}")
    require(pbr_orthogonality_error <= 1e-12,
            f"bit-reversal permutation is not orthogonal: {pbr_orthogonality_error}")
    return {
        "dimension": dimension,
        "composition": "r_h_pbr",
        "composition_max_abs_error": composition_error,
        "pbr_orthogonality_max_abs_error": pbr_orthogonality_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rotation_dir", type=Path)
    parser.add_argument("official_script", type=Path)
    args = parser.parse_args()
    rotation_dir = args.rotation_dir.resolve()
    official_script = args.official_script.resolve()
    require(official_script.is_file(), f"official fitter script is missing: {official_script}")
    k_result = validate_asset(rotation_dir / "k_rotation_qqt_r_h_pbr.pt", "qqt_r_h_pbr")
    v_result = validate_asset(rotation_dir / "v_rotation_sst_r_h_pbr.pt", "sst_r_h_pbr")
    fixture = validate_composition(official_script)
    report = {
        "schema": "oscar-rotation-validation-v1",
        "expected_layers": list(EXPECTED_LAYERS),
        "head_dim": HEAD_DIM,
        "k": k_result,
        "v": v_result,
        "composition_fixture": fixture,
    }
    report_path = rotation_dir / "rotation_validation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"OSCAR rotations: PASS layers={len(EXPECTED_LAYERS)} head_dim={HEAD_DIM} "
        f"K_sha256={k_result['sha256']} V_sha256={v_result['sha256']}"
    )
    print(
        f"max_abs(R@R.T-I): K={k_result['max_orthogonality_error']:.9e} "
        f"V={v_result['max_orthogonality_error']:.9e}"
    )
    print(
        f"composition_fixture: PASS dimension=8 composition=r_h_pbr "
        f"max_abs_error={fixture['composition_max_abs_error']:.1e}"
    )
    for row_k, row_v in zip(k_result["layers"], v_result["layers"]):
        print(
            f"layer={row_k['layer']} K={row_k['max_abs_r_rt_minus_i']:.9e} "
            f"V={row_v['max_abs_r_rt_minus_i']:.9e}"
        )
    print(f"validation_report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
