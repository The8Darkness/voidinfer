# KV-tier fidelity

How close the L0 OSCAR history tier is to FP16 history, and the instruments that measure it.
Hardware and model as in [performance](performance.md); quality is always measured on identical
token ids.

## The tier

Beyond an exact 64-token sink and the exact 960-token recent window, attention history is stored on
the device as OSCAR INT4 codes (17.4 KiB per token over 16 attention layers × 4 KV heads × 256, versus
64 KiB in FP16). Keys and values are rotated with calibrated per-head rotations and centered keys
before quantization (FutureMLS-Lab OSCAR, arXiv 2605.17757). 512 rows per KV head stay as exact FP16
copies on the device (hot rows), chosen from the rows the code pass nominates. Exact FP16 history
stays in host memory, and every published token is checked against it by exact VeriCache.

Rotation assets live in `<target>/l0_oscar/` (`k_mean.bin`, `k_rot_perhead.bin`, `v_rot_perhead.bin`;
`VOIDINFER_EXL3_L0_OSCAR_ROT` names another directory). They were fitted per head with centering on
K/V captured by `tests/research_exl3_kv_tier_capture` (about 384K calibration tokens); fits on 16K,
128K and 393K tokens landed within fit-to-fit noise of each other. The earlier shared-rotation
calibration campaign (FutureMLS-Lab fitter, 10K/30K tokens, phases B2-D4 on the removed runtime) is in
the git history before 2026-10-12.

## Codec study (attention-level fake quantization, KL vs FP16 KV, no hot rows)

| Codec | Bytes / token | KL 4K | KL 16K |
|---|---:|---:|---:|
| OSCAR INT2 | 9.2 KiB | 0.035 | 0.042 |
| KVarN 2-bit | ~9 KiB | 0.032 | 0.044 |
| OSCAR INT3 | ~13 KiB | 0.013 | 0.018 |
| KVarN K4/V2 | ~13 KiB | 0.017 | 0.020 |
| **OSCAR INT4** | 17.4 KiB | **0.005** | **0.008** |

Value precision matters more than key precision (K4/V2 is clearly worse than INT4). INT2 alone is
lossy (about 6% of tokens disagree with FP16 history), which is why the tier moved to INT4 with hot
rows rather than relying on verification to repair INT2.

## End-to-end results

| Configuration | Measure | Value |
|---|---|---|
| L0 INT4 + 512 hot rows vs FP16 KV (KV tier alone) | KL, 4K / 16K | 0.0021 / 0.0030 (INT2 was 0.0081 / 0.0125) |
| Served default (MXFP8 prefill, L0 INT4, exact VeriCache) vs BF16 model | KL, 4K | 0.014; top-1 agreement 97.2% |
| Single-lane exact line with FP8 e4m3 history mirror vs FP16 history | KL | 0.005 |
| VoidInfer vs NInfer (NVFP4/groupwise, FP8 KV), 4K-32K | paired NLL | 0.016 nats/token better, 95% CI excludes 0 |

NLL against real text cannot rank quantization quality (a 5 bpw target beat 6 bpw on it); KL against
the BF16 model does, consistently across the eight fixtures. Rank changes with KL at 4K and compare
KV-tier variants pairwise at 16K.

## Instruments

In `bench/engine-compare/quality/` (`quality_report.py` one level up):

| Tool | Measures |
|---|---|
| `bf16_ref.py` | BF16 reference top-k (layer-streamed transformers), 4K fixtures |
| `ours_nll.ps1 -TopK` | our teacher-forced NLL and top-60 records (`research_exl3_long_nll`) |
| `kl_compare.py` | KL(BF16 ‖ candidate) and top-1 agreement, ours and NInfer |
| `kl_pair.py` | KL between two of our runs (isolates the KV tier: FP16 KV vs L0) |
| `prep.py`, `quality_report.py` | token-exact texts and paired NLL tables against NInfer |

The serving-equivalent environment for `ours_nll.ps1` is
`-L0 -Env 'VOIDINFER_EXL3_L0_EXACT=prefill;LONG_NLL_LAYER_MAJOR=1'`. `VOIDINFER_EXL3_KV_FAKEQUANT`
selects the attention-level codec study.
