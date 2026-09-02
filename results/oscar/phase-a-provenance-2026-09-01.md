# OSCAR Phase-A provenance

Recorded: 2026-09-01, Europe/Berlin

## Source and build identity

| Item | Value |
| --- | --- |
| Canonical checkout | `D:\AI\voidinfer` |
| Canonical branch | `handoff/phase2-context-resource-20260827` |
| Canonical HEAD at check | `dd7192bc3cc2330e73629cc75e35cc5b40aed691` |
| Canonical worktree at check | clean; source snapshot below is separate and uncommitted |
| Designated build source | `D:\AI\voidinfer-adaptive-dflash2` |
| Designated build directory | `D:\AI\build-adaptive-dflash2` |
| Generator/configuration | Ninja / Debug |
| CUDA architecture | `120a` / `sm_120a` |
| CMake home | `D:/AI/voidinfer-adaptive-dflash2` |
| CUDA compiler | CUDA 13.1 `nvcc.exe` |
| Host compiler | MSVC 19.44 via VS 2022 Build Tools |

The snapshot was preserved as the build source because the designated executable is configured
against it. The canonical checkout was not overwritten or rebased.

## Designated artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `D:\AI\build-adaptive-dflash2\bench\ninfer_qwen3_6_27b_dflash_round_bench.exe` | 240,593,920 | `45E9EE1E86AE5BA9E212EEAABFF14FD166499A0E5EFE3C4CCCC5438D8BA829C5` |
| `D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer` | 23,719,496,192 | `6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E` |

The benchmark executable hash remains unchanged by this Phase-A work; focused test and library
outputs were rebuilt in the same Debug build directory.

## Topology audit

The Qwen3.8-shaped hybrid target has 64 transformer layers, 24 query heads, 4 KV heads, head
dimension 256, and rotary dimension 64. The 0-based full-attention layers are:

```text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
```

That is 16 full-attention layers and 48 GDN layers. OSCAR is scoped to the former; the GDN path is
not routed through the paged OSCAR cache.

## Phase-A result

`OscarKVLayout` is now a typed value carried by paged geometry and single/batched views. Full-cache
append, dense prompt/decode, and StateImage conversion use that value. The old
`NINFER_OSCAR_Q2_TRANSPOSED` environment variable no longer independently changes writer or reader
dispatch. The safe default is contiguous until an exact transposed writer/decoder byte round trip
has been proved.

This is a provenance/layout safety milestone, not a calibrated OSCAR result. No post-RoPE Q/K/V
capture dump, fitted layer-specific K/V rotation assets, or native fitter output is present yet.
