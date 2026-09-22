#!/usr/bin/env python3
"""Slow, correctness-first OSCAR INT2 reference comparison.

This is offline tooling only. It evaluates the existing 256-token smoke tensors in
the row-vector convention established by Phase C1. The calibrated path calls the
pinned upstream simulate_int2_asym implementation; the fixed-Hadamard control
mirrors the current experimental runtime affine/clipping contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

import torch


EXPECTED_LAYERS = tuple(range(3, 64, 4))
Q_HEADS = 24
KV_HEADS = 4
GQA_RATIO = 6
HEAD_DIM = 256
TOKENS = 256
CHUNK_ID = 1
BITS = 2
LEVELS = 3

C1_SCORE_TOL = 1.0e-5
C1_SOFTMAX_TOL = 1.0e-6
C1_OUTPUT_TOL = 1.0e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diff_stats(left: torch.Tensor, right: torch.Tensor,
               mask: torch.Tensor | None = None) -> dict[str, float]:
    difference = (left - right).abs()
    if mask is not None:
        difference = difference.masked_select(mask)
        left = left.masked_select(mask)
    require(difference.numel() > 0, "cannot measure an empty difference")
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rms": float(torch.sqrt(torch.mean(difference.square())).item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference) /
            torch.clamp(torch.linalg.vector_norm(left), min=1.0e-24)
        ),
    }


def load_tensor(path: Path, shape: tuple[int, ...], label: str) -> torch.Tensor:
    require(path.is_file() and path.stat().st_size > 0, f"{label}: missing/empty {path}")
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(tensor, torch.Tensor), f"{label}: .pt payload is not a tensor")
    require(tensor.dtype == torch.bfloat16,
            f"{label}: expected torch.bfloat16, got {tensor.dtype}")
    require(tuple(tensor.shape) == shape,
            f"{label}: expected shape {shape}, got {tuple(tensor.shape)}")
    require(torch.isfinite(tensor.float()).all().item(), f"{label}: non-finite input")
    return tensor.contiguous()


def load_rotation(path: Path, objective: str) -> dict[int, torch.Tensor]:
    require(path.is_file() and path.stat().st_size > 0, f"missing rotation asset: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(state, dict), f"{path.name}: invalid checkpoint")
    require(state.get("objective") == objective,
            f"{path.name}: expected objective {objective!r}")
    layers = state.get("layers")
    require(isinstance(layers, dict), f"{path.name}: layers map missing")
    rotations: dict[int, torch.Tensor] = {}
    for key, entry in layers.items():
        layer = int(key)
        require(isinstance(entry, dict), f"{path.name}: layer {layer} entry invalid")
        rotation = entry.get("rotation")
        require(isinstance(rotation, torch.Tensor),
                f"{path.name}: layer {layer} rotation missing")
        require(rotation.dtype == torch.float32 and tuple(rotation.shape) == (HEAD_DIM, HEAD_DIM),
                f"{path.name}: layer {layer} rotation must be FP32 [256,256]")
        require(torch.isfinite(rotation).all().item(),
                f"{path.name}: layer {layer} rotation is non-finite")
        require(entry.get("layer_id") == layer,
                f"{path.name}: layer {layer} mapping mismatch")
        rotations[layer] = rotation.contiguous()
    require(set(rotations) == set(EXPECTED_LAYERS),
            f"{path.name}: expected exact full-attention layer set")
    return rotations


def load_upstream(path: Path) -> Any:
    require(path.is_file(), f"missing upstream fitter: {path}")
    spec = importlib.util.spec_from_file_location("oscar_upstream_c2", path)
    require(spec is not None and spec.loader is not None,
            f"cannot load upstream fitter module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(hasattr(module, "build_hadamard"), "upstream build_hadamard is missing")
    require(hasattr(module, "simulate_int2_asym"),
            "upstream simulate_int2_asym is missing")
    return module


def validate_provenance(dump_root: Path, c1_json: Path) -> dict[str, Any]:
    conversion = dump_root / "conversion_manifest.json"
    require(conversion.is_file() and conversion.stat().st_size > 0,
            f"missing conversion manifest: {conversion}")
    conversion_data = json.loads(conversion.read_text(encoding="utf-8"))
    require(conversion_data.get("schema") == "oscar-official-pt-conversion-v1",
            "unexpected conversion manifest schema")
    require(conversion_data.get("official_fitter_chunk_id") == CHUNK_ID,
            "unexpected converted chunk id")
    model = conversion_data.get("model", {})
    require(model.get("query_heads") == Q_HEADS and model.get("kv_heads") == KV_HEADS,
            "conversion manifest GQA topology mismatch")
    require(model.get("head_dim") == HEAD_DIM and model.get("total_layers") == 64,
            "conversion manifest head/layer topology mismatch")
    require(conversion_data.get("tensor_layout") == "[tokens, heads, head_dim]",
            "conversion manifest tensor orientation mismatch")

    require(c1_json.is_file() and c1_json.stat().st_size > 0,
            f"missing C1 machine report: {c1_json}")
    c1 = json.loads(c1_json.read_text(encoding="utf-8"))
    require(c1.get("layers") == list(EXPECTED_LAYERS), "C1 layer set mismatch")
    require(c1.get("tokens") == TOKENS and c1.get("chunk_id") == CHUNK_ID,
            "C1 input geometry mismatch")
    require(len(c1.get("results", [])) == len(EXPECTED_LAYERS),
            "C1 result count mismatch")
    require(all(item.get("passed") for item in c1.get("results", [])),
            "C1 prerequisite did not pass for every layer")
    for item in c1["results"]:
        require(item["score"]["max_abs"] <= C1_SCORE_TOL,
                f"C1 score gate failed at layer {item['layer']}")
        require(item["softmax"]["max_abs"] <= C1_SOFTMAX_TOL,
                f"C1 softmax gate failed at layer {item['layer']}")
        require(item["recovered_attention_output_after_v_inverse"]["max_abs"] <= C1_OUTPUT_TOL,
                f"C1 output gate failed at layer {item['layer']}")
    return {
        "conversion_manifest_sha256": sha256_file(conversion),
        "conversion_manifest": conversion_data,
        "c1_report_sha256": sha256_file(c1_json),
        "c1_report": c1,
    }


def grouped_kv(tensor: torch.Tensor) -> torch.Tensor:
    mapping = torch.arange(Q_HEADS, dtype=torch.long) // GQA_RATIO
    return tensor[:, mapping, :]


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     future_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require(tuple(q.shape) == (TOKENS, Q_HEADS, HEAD_DIM), "invalid Q attention shape")
    require(tuple(k.shape) == (TOKENS, KV_HEADS, HEAD_DIM), "invalid K attention shape")
    require(tuple(v.shape) == (TOKENS, KV_HEADS, HEAD_DIM), "invalid V attention shape")
    k_grouped = grouped_kv(k)
    v_grouped = grouped_kv(v)
    scores = torch.einsum("thd,shd->hts", q, k_grouped) / math.sqrt(HEAD_DIM)
    scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("hts,shd->thd", probabilities, v_grouped)
    valid_scores = scores.masked_select(~future_mask.unsqueeze(0))
    require(torch.isfinite(valid_scores).all().item(),
            "attention scores contain NaN/Inf in valid causal positions")
    require(torch.isfinite(probabilities).all().item() and torch.isfinite(output).all().item(),
            "attention probabilities/output contain NaN/Inf")
    return scores, probabilities, output


def fixed_affine_params(x: torch.Tensor, is_value: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror current oscar_host_quant_params/cyclic_oscar_quant_params for Q2."""
    x32 = x.float()
    minimum = x32.amin(dim=-1, keepdim=True)
    maximum = x32.amax(dim=-1, keepdim=True)
    ratio = 0.91 if is_value else 0.93
    center = 0.5 * (minimum + maximum)
    half_span = 0.5 * (maximum - minimum) * ratio
    zero = center - half_span
    scale = torch.where(2.0 * half_span > 1.0e-8,
                        (2.0 * half_span) / LEVELS,
                        torch.ones_like(half_span))
    return scale, zero


def fixed_hadamard_int2(x: torch.Tensor, hadamard: torch.Tensor,
                        is_value: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    rotated = x @ hadamard
    scale, zero = fixed_affine_params(rotated, is_value)
    normalized = (rotated.float() - zero) / scale
    # The CUDA control uses __float2int_rn/lrintf; torch.round is its host
    # round-to-nearest reference for these non-tie smoke values.
    codes = torch.round(normalized).to(torch.int32).clamp(0, LEVELS)
    require(torch.all((codes >= 0) & (codes <= LEVELS)).item(),
            "fixed-Hadamard Q2 code outside [0,3]")
    # The runtime stores scale/zero as BF16 metadata and decodes with them.
    stored_scale = scale.to(torch.bfloat16).float()
    stored_zero = zero.to(torch.bfloat16).float()
    decoded = (codes.float() * stored_scale + stored_zero).to(torch.float64)
    require(torch.isfinite(decoded).all().item(),
            "fixed-Hadamard dequantized values non-finite")
    params = {
        "quantizer": "existing_runtime_oscar_affine_q2",
        "bits": BITS,
        "grouping": "one full D=256 row per token and KV head",
        "k_clip_ratio": 0.93,
        "v_clip_ratio": 0.91,
        "metadata_dtype": "BF16 (scale, zero)",
        "code_min": int(codes.min().item()),
        "code_max": int(codes.max().item()),
        "mean_scale": float(stored_scale.double().mean().item()),
    }
    return decoded, codes, params


def calibrated_int2(x: torch.Tensor, upstream: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Apply the pinned upstream simulate_int2_asym exactly."""
    x32 = x.float()
    minimum = x32.amin(dim=-1, keepdim=True)
    maximum = x32.amax(dim=-1, keepdim=True)
    scale = (maximum - minimum).clamp(min=1.0e-8) / LEVELS
    zero = -minimum / scale
    codes = (x32 / scale + zero + 0.5).to(torch.int32).clamp(0, LEVELS)
    expected = upstream.simulate_int2_asym(x)
    decoded = ((codes.float() - zero) * scale).to(x.dtype)
    require(torch.equal(decoded, expected),
            "local calibrated INT2 implementation diverges from upstream simulate_int2_asym")
    require(torch.all((codes >= 0) & (codes <= LEVELS)).item(),
            "calibrated OSCAR INT2 code outside [0,3]")
    require(torch.isfinite(decoded).all().item(),
            "calibrated OSCAR dequantized values non-finite")
    params = {
        "quantizer": "FutureMLS-Lab_OSCAR_simulate_int2_asym",
        "bits": BITS,
        "grouping": "last dimension of each [token, KV-head, head_dim] row",
        "scale": "(row_max-row_min)/3, clamped to 1e-8 before division",
        "zero": "-row_min/scale",
        "rounding": "int32(x_fp32/scale + zero + 0.5), clamped [0,3]",
        "metadata_dtype": "not serialized; upstream simulation dequantizes from FP32 params",
        "code_min": int(codes.min().item()),
        "code_max": int(codes.max().item()),
        "mean_scale": float(scale.double().mean().item()),
    }
    return decoded, codes, params


def quant_error(original: torch.Tensor, decoded: torch.Tensor) -> dict[str, float]:
    return diff_stats(original, decoded)


def attention_metrics(reference: dict[str, torch.Tensor],
                      candidate: dict[str, torch.Tensor]) -> dict[str, Any]:
    score_mask = ~reference["future_mask"].unsqueeze(0)
    score = diff_stats(reference["scores"], candidate["scores"], score_mask)
    softmax = diff_stats(reference["probabilities"], candidate["probabilities"])
    output = diff_stats(reference["output"], candidate["output"])
    reference_argmax = torch.argmax(reference["scores"], dim=-1)
    candidate_argmax = torch.argmax(candidate["scores"], dim=-1)
    argmax_equal = reference_argmax == candidate_argmax
    return {
        "attention_score_logits": score,
        "softmax": softmax,
        "attention_output_original_coordinates": output,
        "attention_argmax": {
            "agree": int(argmax_equal.sum().item()),
            "total": int(argmax_equal.numel()),
            "agreement": float(argmax_equal.float().mean().item()),
        },
        "layer_output": {
            "available": False,
            "reason": "Phase B2 contains no output-projection weights, residual input, or post-projection activation",
        },
    }


def run_layer(dump_root: Path, layer: int, k_rotations: dict[int, torch.Tensor],
              v_rotations: dict[int, torch.Tensor], hadamard: torch.Tensor,
              upstream: Any) -> dict[str, Any]:
    q = load_tensor(dump_root / f"layer_{layer}" / "q" / f"{CHUNK_ID}.pt",
                    (TOKENS, Q_HEADS, HEAD_DIM), f"layer {layer} Q")
    k = load_tensor(dump_root / f"layer_{layer}" / f"k" / f"{CHUNK_ID}.pt",
                    (TOKENS, KV_HEADS, HEAD_DIM), f"layer {layer} K")
    v = load_tensor(dump_root / f"layer_{layer}" / f"v" / f"{CHUNK_ID}.pt",
                    (TOKENS, KV_HEADS, HEAD_DIM), f"layer {layer} V")
    q_ref, k_ref, v_ref = q.double(), k.double(), v.double()
    r_k, r_v = k_rotations[layer].double(), v_rotations[layer].double()
    h = hadamard.double()
    future_mask = torch.triu(torch.ones(TOKENS, TOKENS, dtype=torch.bool), diagonal=1)

    base_scores, base_probabilities, base_output = causal_attention(
        q_ref, k_ref, v_ref, future_mask)
    reference = {
        "scores": base_scores,
        "probabilities": base_probabilities,
        "output": base_output,
        "future_mask": future_mask,
    }

    q_rotated = q_ref @ r_k
    k_rotated = k_ref @ r_k
    v_rotated = v_ref @ r_v
    rot_scores, rot_probabilities, rot_raw = causal_attention(
        q_rotated, k_rotated, v_rotated, future_mask)
    rotated_output = rot_raw @ r_v.T
    rotated = {
        "scores": rot_scores,
        "probabilities": rot_probabilities,
        "output": rotated_output,
        "future_mask": future_mask,
    }
    c1_metrics = attention_metrics(reference, rotated)
    require(c1_metrics["attention_score_logits"]["max_abs"] <= C1_SCORE_TOL,
            f"layer {layer}: rotated-BF16 score prerequisite failed")
    require(c1_metrics["softmax"]["max_abs"] <= C1_SOFTMAX_TOL,
            f"layer {layer}: rotated-BF16 softmax prerequisite failed")
    require(c1_metrics["attention_output_original_coordinates"]["max_abs"] <= C1_OUTPUT_TOL,
            f"layer {layer}: rotated-BF16 output prerequisite failed")

    q_h, k_h, v_h = q_ref @ h, k_ref @ h, v_ref @ h
    k_fixed, k_fixed_codes, fixed_k_params = fixed_hadamard_int2(k_h, h, False)
    v_fixed, v_fixed_codes, fixed_v_params = fixed_hadamard_int2(v_h, h, True)
    fixed_scores, fixed_probabilities, fixed_raw = causal_attention(
        q_h, k_fixed, v_fixed, future_mask)
    fixed = {
        "scores": fixed_scores,
        "probabilities": fixed_probabilities,
        "output": fixed_raw @ h.T,
        "future_mask": future_mask,
    }

    k_cal, k_cal_codes, cal_k_params = calibrated_int2(k_rotated, upstream)
    v_cal, v_cal_codes, cal_v_params = calibrated_int2(v_rotated, upstream)
    cal_scores, cal_probabilities, cal_raw = causal_attention(
        q_rotated, k_cal, v_cal, future_mask)
    calibrated = {
        "scores": cal_scores,
        "probabilities": cal_probabilities,
        "output": cal_raw @ r_v.T,
        "future_mask": future_mask,
    }

    return {
        "layer": layer,
        "source_dtype": "torch.bfloat16",
        "diagnostic_compute_dtype": "torch.float64; upstream quantizer parameters use FP32",
        "shapes": {
            "q": [TOKENS, Q_HEADS, HEAD_DIM],
            "k": [TOKENS, KV_HEADS, HEAD_DIM],
            "v": [TOKENS, KV_HEADS, HEAD_DIM],
        },
        "gqa_mapping": [head // GQA_RATIO for head in range(Q_HEADS)],
        "rotated_bf16": {
            "metrics_vs_bf16": c1_metrics,
            "q_roundtrip": diff_stats(q_ref, q_rotated @ r_k.T),
            "k_roundtrip": diff_stats(k_ref, k_rotated @ r_k.T),
            "v_roundtrip": diff_stats(v_ref, v_rotated @ r_v.T),
        },
        "fixed_hadamard_int2": {
            "metrics_vs_bf16": attention_metrics(reference, fixed),
            "k_quantization_error_in_hadamard_basis": quant_error(k_h, k_fixed),
            "v_quantization_error_in_hadamard_basis": quant_error(v_h, v_fixed),
            "k_params": fixed_k_params,
            "v_params": fixed_v_params,
            "codes": {
                "k_min": int(k_fixed_codes.min().item()),
                "k_max": int(k_fixed_codes.max().item()),
                "v_min": int(v_fixed_codes.min().item()),
                "v_max": int(v_fixed_codes.max().item()),
            },
        },
        "calibrated_oscar_int2": {
            "metrics_vs_bf16": attention_metrics(reference, calibrated),
            "k_quantization_error_in_rotated_basis": quant_error(k_rotated, k_cal),
            "v_quantization_error_in_rotated_basis": quant_error(v_rotated, v_cal),
            "k_params": cal_k_params,
            "v_params": cal_v_params,
            "codes": {
                "k_min": int(k_cal_codes.min().item()),
                "k_max": int(k_cal_codes.max().item()),
                "v_min": int(v_cal_codes.min().item()),
                "v_max": int(v_cal_codes.max().item()),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_root", type=Path)
    parser.add_argument("rotation_dir", type=Path)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--c1-json", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--layers", default="all",
                        help="comma-separated layer ids or all (default: all)")
    parser.add_argument("--representative-command", default="")
    args = parser.parse_args()
    torch.set_num_threads(1)
    dump_root = args.dump_root.resolve()
    rotation_dir = args.rotation_dir.resolve()
    upstream_path = args.upstream.resolve()
    c1_json = args.c1_json.resolve()
    report_json = args.report_json.resolve()

    provenance = validate_provenance(dump_root, c1_json)
    k_path = rotation_dir / "k_rotation_qqt_r_h_pbr.pt"
    v_path = rotation_dir / "v_rotation_sst_r_h_pbr.pt"
    k_rotations = load_rotation(k_path, "qqt_r_h_pbr")
    v_rotations = load_rotation(v_path, "sst_r_h_pbr")
    upstream = load_upstream(upstream_path)
    hadamard = upstream.build_hadamard(HEAD_DIM).double().contiguous()
    identity = torch.eye(HEAD_DIM, dtype=torch.float64)
    require(torch.max(torch.abs(hadamard @ hadamard.T - identity)).item() <= 1.0e-12,
            "upstream Hadamard is not orthogonal")

    if args.layers.strip().lower() == "all":
        layers = list(EXPECTED_LAYERS)
    else:
        layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
        require(layers and set(layers).issubset(set(EXPECTED_LAYERS)),
                "--layers must be full-attention layer ids")

    results = [run_layer(dump_root, layer, k_rotations, v_rotations, hadamard, upstream)
               for layer in layers]
    require(results, "no layers were evaluated")
    cal_outlier = max(
        results,
        key=lambda result: result["calibrated_oscar_int2"]["metrics_vs_bf16"]
        ["attention_output_original_coordinates"]["max_abs"],
    )
    command = " ".join(shlex.quote(item) for item in sys.argv)
    report = {
        "schema": "oscar-int2-reference-v1",
        "recorded": "2026-09-01 (Europe/Berlin)",
        "classification": "PASS",
        "dump_root": str(dump_root),
        "rotation_dir": str(rotation_dir),
        "tokens": TOKENS,
        "chunk_id": CHUNK_ID,
        "layers": layers,
        "expected_full_attention_layers": list(EXPECTED_LAYERS),
        "q_heads": Q_HEADS,
        "kv_heads": KV_HEADS,
        "gqa_ratio": GQA_RATIO,
        "head_dim": HEAD_DIM,
        "bits": BITS,
        "orientation": "row vectors [tokens, heads, head_dim]; Q'=Q@Rk, K'=K@Rk, V'=V@Rv, O=O'@Rv.T",
        "provenance": provenance,
        "rotation_hashes": {"k": sha256_file(k_path), "v": sha256_file(v_path)},
        "upstream_path": str(upstream_path),
        "upstream_sha256": sha256_file(upstream_path),
        "quantization_authority": "FutureMLS-Lab OSCAR simulate_int2_asym",
        "command": command,
        "representative_command": args.representative_command or command,
        "results": results,
        "outliers": {
            "calibrated_attention_output_max_abs": {
                "layer": cal_outlier["layer"],
                "value": cal_outlier["calibrated_oscar_int2"]["metrics_vs_bf16"]
                ["attention_output_original_coordinates"]["max_abs"],
            },
        },
        "notes": [
            "BF16 source values are retained exactly; diagnostic attention products use FP64.",
            "The calibrated quantizer is checked element-for-element against the pinned upstream function.",
            "The fixed-Hadamard control mirrors current runtime clipping and BF16 metadata, not upstream OSCAR simulation.",
            "The smoke capture has no output projection/residual/LM-head tensors, so layer-output and LM-logit metrics are unavailable.",
            "No CUDA runtime, INT2 packing, 10K/30K calibration, or fitter invocation was performed in C2.",
        ],
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"OSCAR INT2 reference: PASS path=calibrated layers={len(results)} tokens={TOKENS}")
    for result in results:
        cal = result["calibrated_oscar_int2"]["metrics_vs_bf16"]
        fixed = result["fixed_hadamard_int2"]["metrics_vs_bf16"]
        print(
            f"layer={result['layer']} "
            f"fixed_out={fixed['attention_output_original_coordinates']['max_abs']:.9e} "
            f"cal_out={cal['attention_output_original_coordinates']['max_abs']:.9e} "
            f"cal_rel={cal['attention_output_original_coordinates']['relative_l2']:.9e} "
            f"cal_argmax={cal['attention_argmax']['agreement'] * 100:.2f}%"
        )
    print(f"json={report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
