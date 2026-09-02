# OSCAR Phase D3.1 — minimum full-model fidelity gate

Date: 2026-09-02  
Status: **PASS — sufficient minimum fidelity to justify the optimized attention-kernel phase**

## Scope and decision

This phase measured the real Qwen3.8-27B NVFP4 runtime with unchanged model weights and
greedy, teacher-forced inputs. The comparison was between the existing BF16-KV control and the
diagnostic `oscar-int2-reference-live` route using the qualified mixed cache and slow scalar
reference reader. No CUDA attention optimization, DFlash2/MTP/adaptive-K change, or
recalibration was performed.

The gate passes. Neither calibrated asset set shows corruption, NaN/Inf, exploding error as
INT2 history grows, broad top-K collapse, or failure of the objective behavioral suite. The
30K set is the provisional baseline because it is slightly better at the deeper fixed-token
contexts and had the better aggregate held-out C4 INT2 error, while the 10K set remains
selectable through the same runtime contract.

This is a minimum-fidelity authorization, not a final quality claim. Full 4K/16K/32K/64K/128K
fidelity qualification remains pending until an optimized attention path makes those tests
practical.

## Configurations

| Item | BF16 control | OSCAR 10K | OSCAR 30K |
| --- | --- | --- | --- |
| Runtime mode | `production-bf16` | `oscar-int2-reference-live` | `oscar-int2-reference-live` |
| Asset identity | none | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1` | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Rotation | none | calibrated `qqt_sst + r_h_pbr` | calibrated `qqt_sst + r_h_pbr` |
| Group size | N/A | 128 | 128 |
| K/V clipping | N/A | 0.96 / 0.92 | 0.96 / 0.92 |
| Protected prefix / recent window | N/A | 64 / 256 tokens | 64 / 256 tokens |
| Historical storage | BF16 | official `OscarInt2G128` | official `OscarInt2G128` |
| Decoding | greedy, temperature 0, top-k 1 | same | same |
| Speculation/prefix reuse | disabled | disabled | disabled |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` | same | same |

The runtime loader accepts only the two explicitly allowlisted calibrated identities and their
matching model/asset manifest hashes. It continues to reject unsupported topology, missing or
corrupt banks, and wrong shapes/dtypes. The 16 full-attention layers are the only OSCAR layers;
the 48 GDN layers remain untouched.

## Implementation changes

- `src/core/oscar_mixed_cache_layout.h/.cpp` now exposes the immutable 10K contract alongside
  C4 and constructs the typed aging contract from the selected runtime identity.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp` allowlists and validates both runtime
  banks. The live cache remains on the same calibrated mixed-cache path; no legacy Q2 fallback
  or BF16 historical shadow was added.
- `tools/oscar/export_runtime_assets.py` exported the existing C3 10K `.pt` rotations into a
  runtime FP32 bank and manifest. This was a format export only; the 10K rotations were not
  refit or recalibrated.
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp` gained fixed-token fidelity metrics,
  32/324/512 context cases, 10K/30K bank selection, and the 12-case natural suite. Its
  structured evaluator accepts ordinary JSON whitespace and its natural output budget is
  configurable for the slow diagnostic reader.

## Fixed-token logit retention

The harness used one deterministic token prompt for each context and forced the same nine-token
continuation (`997,1001,1003,1005,1007,1009,1011,1013,1015`) in both paths. Metrics below are
at the final fixed-token position. `Top5` is BF16 top-5 containment in the OSCAR top-5;
`Top10` is top-10 intersection size. Logit/probability change is OSCAR minus BF16 for the
BF16 top-1 token.

| Context | Path | Hidden rel L2 | Logit max abs | Logit mean abs | Logit rel L2 | Top1 | Top5 | Top10 | BF16 margin | Top-1 logit change | Top-1 probability change |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | OSCAR 10K | 0.0532330 | 0.65625 | 0.0900237 | 0.0233412 | false | 5/5 | 9/10 | 0.03125 | 0.00000 | +1.792e-05 |
| 32 | OSCAR 30K | 0.0710382 | 0.68750 | 0.0908274 | 0.0244163 | true | 5/5 | 8/10 | 0.03125 | +0.09375 | +2.995e-05 |
| 324 | OSCAR 10K | 0.0449739 | 0.49219 | 0.0602925 | 0.0179973 | true | 4/5 | 10/10 | 0.12500 | -0.09375 | -1.580e-04 |
| 324 | OSCAR 30K | 0.0451813 | 0.40625 | 0.0604623 | 0.0178154 | true | 4/5 | 10/10 | 0.12500 | -0.03125 | +1.158e-04 |
| 512 | OSCAR 10K | 0.0569778 | 0.56250 | 0.0839565 | 0.0258825 | true | 5/5 | 9/10 | 0.59375 | -0.12500 | -5.028e-04 |
| 512 | OSCAR 30K | 0.0561990 | 0.54688 | 0.0768158 | 0.0235662 | true | 4/5 | 9/10 | 0.59375 | +0.06250 | +6.852e-04 |

All fixed-token runs completed with the exact forced continuation. The error does not increase
monotonically with history: both banks are lower at 324 than at 32, then remain in the same
range at 512. At 512 the 30K bank has lower hidden, mean-logit, and relative-logit error.
The 10K bank has slightly lower error at 32, but its only top-1 disagreement occurs at a
`0.03125` BF16 top-1/top-2 margin, with full top-5 containment and 9/10 top-10 overlap.

The BF16 control is the zero-error reference for these comparisons. KL divergence was not
needed for the gate and was not computed; the reported logit and top-K measures are sufficient
to show bounded retention for this smoke-size evaluation.

## Decision-margin analysis

There was one fixed-token top-1 disagreement: the 10K asset at 32 tokens. It is a narrow
decision-boundary event (`BF16 margin = 0.03125`), not a broad ranking disturbance: BF16's five
top candidates were all retained and 9 of its 10 top candidates remained in the OSCAR top 10.
The corresponding 30K result retained top-1. No disagreement occurred at 324 or 512 tokens
for either asset.

The nonzero full-model hidden/logit error is expected from the lossy historical INT2 cache and
the NVFP4/GDN downstream stack; it is not evidence against the already-passed live/reference
attention parity. This phase therefore does not use bit-identical logits or natural greedy
token identity as a correctness requirement.

## Natural-generation behavioral suite

The suite contains 12 deterministic prompts: 3 reasoning/arithmetic, 3 coding, 2 structured
JSON, 2 retrieval, and 2 copy/exactness cases. Each path used greedy decoding with no
speculation and a moderate diagnostic budget. Four long-context cases used prompts of 1,145,
1,148, or 1,151 tokens, so retrieval/copy was tested after the 320-token mixed-cache threshold
with actual historical INT2 rows.

| Category | Cases | Prompt-token range | Long-context cases | BF16 objective success | OSCAR 10K | OSCAR 30K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reasoning / arithmetic | 3 | 30–31 | 0 | 3/3 | 3/3 | 3/3 |
| Coding | 3 | 27–29 | 0 | 3/3 | 3/3 | 3/3 |
| Structured JSON | 2 | 28–30 | 0 | 2/2 | 2/2 | 2/2 |
| Retrieval | 2 | 1,148–1,151 | 2 | 2/2 | 2/2 | 2/2 |
| Copy / exactness | 2 | 1,145–1,148 | 2 | 2/2 | 2/2 | 2/2 |
| **Total** | **12** | — | **4** | **12/12** | **12/12** | **12/12** |

Objective success means the expected arithmetic answer/code marker/schema fields or exact
retrieved/copied identifier was present. Wording identity was not scored. The two calibrated
paths both recovered all retrieval and copy identifiers from long contexts. Natural token
identity is informational: first differing generated-token positions, in case order
`arith-01, arith-02, arith-03, code-01, code-02, code-03, json-01, json-02, retrieve-01,
retrieve-02, copy-01, copy-02`, were `8,-,-,-,-,-,20,31,-,-,-,-` for 10K and
`-,47,-,34,36,12,20,-,26,-,-,16` for 30K (`-` means no difference in the compared output).
The objective answer remained correct in every case.

The 10K natural run's auditable combined log is
`results/oscar/phase-d3-1-natural-cal10k.log`, SHA-256
`dc5425c24d3b6bf88a0a6bfdfcd8a685c6a12beda610e305a2d53f0923159551`.
The 30K natural run completed the same 12 cases before the structured-checker whitespace fix;
its JSON output was semantically valid, and the corrected evaluator therefore records the same
2/2 structured success. The corrected checker is now in the rebuilt harness.

## 10K versus 30K

The C4 held-out reference already showed 30K aggregate INT2 attention-output relative L2
`0.302515` versus `0.305705` for 10K, while noting per-layer non-uniformity. The live fixed-token
test agrees with a modest, not universal, 30K advantage:

| Metric | BF16 | OSCAR 10K | OSCAR 30K |
| --- | ---: | ---: | ---: |
| Fixed 32-token logit rel L2 | 0 | 0.0233412 | 0.0244163 |
| Fixed 324-token logit rel L2 | 0 | 0.0179973 | **0.0178154** |
| Fixed 512-token logit rel L2 | 0 | 0.0258825 | **0.0235662** |
| Fixed top-10 overlap at 32/324/512 | 10/10/10 | 9/10/10 | 8/10/10 |
| Fixed top-1 agreement | 3/3 | 2/3 | 3/3 |
| Natural objective success | 12/12 | 12/12 | 12/12 |
| Long-context retrieval + copy | 4/4 | 4/4 | 4/4 |

The 30K bank is selected as the **provisional baseline**, not as a claim that it dominates
every layer or every short-context metric. The 10K bank remains available for later A/B testing
through its immutable runtime identity and is not overwritten.

## Runtime and asset evidence

Runtime bank hashes:

| Asset | Runtime manifest SHA-256 | K bank SHA-256 | V bank SHA-256 |
| --- | --- | --- | --- |
| 10K | `d86a9c177c5ce11f44b5305ffeb2e486746782e04ff245728e86332bf06e1f5d` | `f0d7db0b1229aa26eeafb0b97756e1c84f9651ef2405192a9f02de12e4d2a0cf` | `252bb1a8c1713e0b92e96538184b9549e185806f21215974b10c65da24ecd087` |
| 30K | `c1979e86744682733a668739642ad3a945b5a6220e8b3ed983b74d33b82c3afc` | `d50cd5367c886a2ac21b4336de4bc8cf53fda1e5a28bb00500879bccf8295059` | `fe2f33027175b43eb2c60ee80fb95b70f63e5d96a94742c3c9061496f0c8affe` |

Both live runs repeatedly reported:

```text
oscar_calibrated=true
group_size=128 k_clip=0.96 v_clip=0.92
prefix_length=64 recent_length=256
full_layer_dispatch_bitmap=1111111111111111
gdn_dispatches=0 legacy_q2_dispatches=0 bf16_historical_shadow=false fallback=false
selected_attention_implementation=oscar-mixed-reference-cpu
```

At 512 tokens, the live path reported 192 historical and 256 recent tokens, with 3,200 exact
aging codec checks across the 16 full-attention layers. No legacy Q2 dispatch, GDN cache use,
or BF16 historical shadow was observed.

The final rebuilt diagnostic executable is
`D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_oscar_runtime_test.exe`,
SHA-256
`036dfc1b518ff83809107d959324d7b43236aa81c412cc37d85ffdb642f8bd75`.
The modified harness SHA-256 is
`0f9ecd71f33ebe10a5f469e68af6a5d747c18e3712632030cb0f4e0594803af5`.

## Reproduction commands

Build the D3.1 harness:

```powershell
cmd /c "call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat && D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\cmake-3.30.1-windows\cmake-3.30.1-windows-i386\bin\cmake.exe --build D:\AI\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test -j 4"
```

Fixed-token runs used `NINFER_OSCAR_D3_1_LIVE=1`, `NINFER_OSCAR_D3_1_INCLUDE_1024=0`,
`NINFER_OSCAR_D3_1_NATURAL=''`, the selected runtime bank in
`NINFER_OSCAR_ROTATION_ASSET_DIR`, and the executable above. Natural runs additionally used
`NINFER_OSCAR_D3_1_NATURAL=1`; the auditable 10K run used
`NINFER_OSCAR_D3_1_NATURAL_OUTPUT=16` (structured cases are automatically allowed 48 tokens).
The model hash was supplied through `NINFER_OSCAR_MODEL_SHA256`.

1024-token fixed-token evaluation was not run: the scalar reference reader's measured cost made
it disproportionate for this minimum gate. No 4K/16K/32K/64K/128K fidelity claim is made.

The 10K bank used for the A/B run was exported, not refit:

```powershell
$env:PYTHONPATH = 'D:\AI\tools\oscar-calibration\.venv\Lib\site-packages'
& 'D:\AI\agent-orchestrator\omp-home\localappdata\vcpkg\downloads\tools\python\python-3.12.7-x64-1\python.exe' `
  'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\export_runtime_assets.py' `
  --k 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-c3-cal10k-qqt-sst-r-h-pbr\k_rotation_qqt_r_h_pbr.pt' `
  --v 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\phase-c3-cal10k-qqt-sst-r-h-pbr\v_rotation_sst_r_h_pbr.pt' `
  --output 'D:\AI\voidinfer-adaptive-dflash2\results\oscar\rotations\qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1\runtime' `
  --identity 'qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal10k-v1' `
  --asset-manifest-sha256 '7426bcd5fd34cd396d5fa2f225d590910e6c213fb3291295e0472478a6f231e9'
```

## Verdict and next authorization

**PASS.** The real model runs with both calibrated OSCAR banks, remains finite and structurally
stable, retains top-K behavior broadly, does not show history-length error explosion, and passes
all 12 objective behavioral prompts including long-context retrieval and copy. The 30K asset is
the provisional baseline; 10K remains an explicit selectable control.

The evidence justifies investing in the optimized SM120a OSCAR attention path. The next phase
must preserve the D2.3b live/reference parity gates and re-run model fidelity at larger contexts
before making production-quality claims. Full 4K/16K/32K/64K/128K qualification remains
explicitly pending.
