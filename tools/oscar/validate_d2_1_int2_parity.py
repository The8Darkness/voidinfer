#!/usr/bin/env python3
"""Fail-closed validator for the D2.1 golden fixture and C++ parity report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_SHA256 = "6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cpp-report", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--rotation-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "oscar-d2-1-golden-v1", "golden schema mismatch")
    require(manifest["upstream"]["commit"] ==
            "41ebcdba3db5f0ce1339c3727caea80df575d437", "upstream commit mismatch")
    require(manifest["model"]["sha256"] == MODEL_SHA256, "model hash mismatch")
    require(manifest["model"]["full_attention_layers"] == list(LAYERS),
            "full-attention layer list mismatch")
    require(manifest["official_semantics"]["group_size"] == 128, "group size mismatch")
    require(manifest["official_semantics"]["k_clip_ratio"] == 0.96 and
            manifest["official_semantics"]["v_clip_ratio"] == 0.92,
            "clip ratio mismatch")
    require(sha256(args.manifest) != "", "manifest hash could not be computed")
    require(sha256(args.fixture) == manifest["fixture"]["sha256"], "fixture hash mismatch")
    require(args.fixture.stat().st_size == manifest["fixture"]["bytes"], "fixture size mismatch")
    require(sha256(args.generator) == manifest["source_files"]["generator"],
            "generator hash mismatch")
    require(sha256(args.capture_manifest) == manifest["source_files"]["capture_manifest"],
            "capture manifest hash mismatch")
    require(sha256(args.rotation_dir / "k_rotation_qqt_r_h_pbr.pt") ==
            manifest["source_files"]["rotation_k_pt"], "K rotation hash mismatch")
    require(sha256(args.rotation_dir / "v_rotation_sst_r_h_pbr.pt") ==
            manifest["source_files"]["rotation_v_pt"], "V rotation hash mismatch")

    capture_root = args.capture_manifest.parent
    capture = json.loads(args.capture_manifest.read_text(encoding="utf-8"))
    records = {int(record["model_layer"]): record for record in capture["captures"]}
    require(capture["model"]["sha256"] == MODEL_SHA256, "source capture model hash mismatch")
    for layer in (3, 35, 63):
        for kind in ("k", "v"):
            payload = (capture_root / records[layer]["files"][kind]).resolve()
            require(payload.is_file(), f"missing source payload {layer} {kind}")
            require(sha256(payload) == records[layer]["sha256"][kind],
                    f"source payload hash mismatch {layer} {kind}")

    cpp = json.loads(args.cpp_report.read_text(encoding="utf-8"))
    require(cpp.get("schema") == "oscar-d2-1-cpp-parity-v1", "C++ report schema mismatch")
    require(cpp.get("passed") is True, "C++ parity report is not PASS")
    require(cpp.get("total_rows") == manifest["fixture"]["row_count"], "row count mismatch")
    require(len(cpp.get("cases", [])) == manifest["fixture"]["case_count"], "case count mismatch")
    require(all(float(case["max_clipped_abs"]) == 0.0 for case in cpp["cases"]),
            "C++ clipped parity is non-exact")
    require(all(float(case["max_scale_abs"]) <= 2.0e-6 for case in cpp["cases"]),
            "C++ scale parity exceeds tolerance")
    require(all(float(case["max_decoded_relative_l2"]) <= 2.0e-6 for case in cpp["cases"]),
            "C++ decode parity exceeds tolerance")
    print(json.dumps({
        "status": "PASS",
        "fixture_sha256": manifest["fixture"]["sha256"],
        "cases": manifest["fixture"]["case_count"],
        "rows": manifest["fixture"]["row_count"],
        "packed_symbol_parity": "exact",
        "max_scale_abs": cpp["max_scale_abs"],
        "max_decoded_abs": cpp["max_decoded_abs"],
        "max_decoded_relative_l2": cpp["max_decoded_relative_l2"],
    }, indent=2))


if __name__ == "__main__":
    main()
