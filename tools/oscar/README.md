# OSCAR QKV capture and calibration staging

This directory is the staging boundary for a faithful OSCAR implementation. `validate_dump.py`
checks the manifest and payload hashes; it intentionally does not pretend that a fixed-Hadamard
Q2 page is an OSCAR calibration result.

The capture manifest is JSON with this shape:

```json
{
  "schema": "oscar-qkv-v1",
  "model_id": "qwen3.8-27b",
  "dtype": "bf16",
  "post_rope_qkv": true,
  "model": {
    "total_layers": 64,
    "query_heads": 24,
    "kv_heads": 4,
    "head_dim": 256,
    "rotary_dim": 64
  },
  "captures": [
    {
      "model_layer": 3,
      "full_attention_index": 0,
      "kind": "full_attention",
      "tokens": 256,
      "q_heads": 24,
      "kv_heads": 4,
      "head_dim": 256,
      "files": {"q": "layer_003/q.bin", "k": "layer_003/k.bin", "v": "layer_003/v.bin"},
      "sha256": {"q": "...", "k": "...", "v": "..."}
    }
  ]
}
```

For this Qwen3.8 hybrid topology the 0-based full-attention layers are `3, 7, ..., 63` (16
layers); the other 48 layers are GDN and must not be sent through OSCAR. Captures must be made at
the post-RoPE Q/K/V boundary, with separate K and V payloads. Run the validator before fitting:

```text
python tools/oscar/validate_dump.py <dump-root>/manifest.json
```

## Phase B2 smoke capture

The native opt-in smoke path captures only the verified full-attention layers at the post-Q/K
RMSNorm + post-RoPE boundary, before causal attention/cache append. It submits 256 deterministic
copies of token id 198 and writes BF16 raw payloads plus provenance to
`results/oscar/captures/phase-b2-qkv-256`.

Build the focused target in the latest build tree:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_capture_test -j 1"
```

Run the reproducible wrapper from an elevated/native Developer PowerShell when the isolated
Python launcher is restricted by the host shell:

```powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\capture_qkv_smoke.ps1' -PythonExe 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe'
```

The validator is fail-closed: it checks the exact 16-layer topology, Q/K/V shape and token
alignment, GQA mapping, stage identity, finite BF16 contents, exact file sizes, payload hashes,
per-tensor provenance, and the manifest SHA-256 sidecar. The completed Phase B2 smoke result is
documented in `results/oscar/phase-b2-qkv-smoke-capture.md`.

## Phase B3 fitter smoke

`convert_b2_to_official_pt.py` first validates the Phase B2 manifest, then writes the upstream
hierarchy using exact BF16 bit patterns:

```text
<dump>/layer_<id>/q/<chunk_id>.pt
<dump>/layer_<id>/k/<chunk_id>.pt
<dump>/layer_<id>/v/<chunk_id>.pt
```

The upstream fitter skips chunk `0` when `--chunk-id all` is used. The 256-token smoke capture is
therefore converted with explicit chunk id `1`, and the fitter is invoked with `--chunk-id 1`.
This keeps the useful smoke data and does not alter upstream mathematics.

The official source is pinned locally as `tools/oscar/upstream_compute_kv_rotation.py` from
FutureMLS-Lab/OSCAR commit `41ebcdba3db5f0ce1339c3727caea80df575d437`; the downloaded script SHA-256
is `f7f9f738be5dea75cc1d7e8d6928e4ee91df1cbf1e1c7b7d9c7d308b55d4da8b`.

Run the official smoke fit with:

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe' 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\upstream_compute_kv_rotation.py' --dump-path 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\dumps\phase-b2-256-official' --output-dir 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-b3-qqt-sst-r-h-pbr' --chunk-id 1 --head-dim 256 --method qqt_sst --composition r_h_pbr
```

Validate the resulting K/V assets with `validate_rotations.py`. It checks FP32 serialization,
`[256,256]` shape, exact layer mapping, finite values, deterministic reload, per-layer
`max_abs(R @ R.T - I)`, and an independent `R*H*Pbr` fixture. The Phase B3 result is documented in
`results/oscar/phase-b3-fitter-smoke.md`.

The smoke fitter is now pinned and runnable offline. Production calibration still requires
10K/30K data, calibrated clipping, and independent K/V assets at group size 128. The official
algorithm reproduces GQA-aware `qqt` K covariance, score-aware `sst` V covariance,
descending-eigenvalue ordering with bit-reversal permutation, and `R * H_d * P_br`. The official reference is
[`compute_kv_rotation.py`](https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/main/rotation/compute_kv_rotation.py).
