# OSCAR D4.3 — real Qwen GPU runtime integration

Date: 2026-09-02  
Status: **PASS — live GPU path is correctness-qualified; large-context model timing remains pending**

## Scope and decision

This phase connected the real Qwen3.8-27B NVFP4 runtime to the qualified D4.2b mixed-tier
GPU reader. The opt-in mode is `oscar-int2-gpu` (the optimized diagnostic successor to
`oscar-int2-reference-live`). It uses the immutable 30K calibrated asset, actual runtime
Q/K/V tensors, the existing typed cache/aging policy, and the D4.2b `OscarInt2G128` mixed
kernel. No INT2 mathematics, calibration, DFlash2/MTP/adaptive-K, Q rotation, or `R_V.T`
mathematics was changed.

The correctness gate passes. Live GPU attention matched the qualified scalar
`oscar-int2-reference-live` oracle at the selected layer taps, all 16 full-attention layers
dispatched, and no GDN, legacy-Q2, BF16-history, CPU fallback, or corruption path appeared.
The first integration is not yet a practical long-context serving implementation: at 4K,
host-side aging/append work dominates the request and the GPU kernel is still launched once
per query column. The 16K exploratory run was stopped after reaching a 3,072-token logical
context because it was not a healthy timing path; therefore 16K/32K real-model throughput
claims remain pending D4.4.

## Qualified configuration

| Item | Value |
| --- | --- |
| Model artifact | `D:\\AI\\voidinfer\\models\\Qwen3.8-27B-NVFP4-DFlash2-NInfer\\qwen3_8_27b_nvfp4.ninfer` |
| Model SHA-256 | `6cc7560ae3427d8fa87b75c17e41328116b71b068c4c4dc06137fb73b656f64e` |
| OSCAR asset | `qwen3.8-27b-oscar-qqt-sst-rhpbr-g128-cal30k-v1` |
| Runtime asset hash | `4d6d7af496238c1c65c95cc9425f18c3d9cb6028d4dda42d4d0de91e9efaf560` |
| Rotation / codec | `qqt_sst + r_h_pbr`; official `OscarInt2G128` |
| Topology | 64 layers; full attention `3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63`; 48 GDN layers untouched |
| Attention shape | Q/KV heads `24/4`, GQA `6`, head dimension `256` |
| Cache policy | BF16 prefix `64` + OSCAR INT2 historical + BF16 recent `256` |
| Clipping / metadata | K/V `0.96/0.92`; FP32 metadata |
| Mode | explicit `NINFER_OSCAR_ROTATION_MODE=oscar-int2-gpu` |

The asset loader continues to fail closed on model/asset hash, topology, layer mapping, shape,
dtype, and manifest failures. Runtime telemetry reports `oscar_calibrated=true`, the asset
identity/hash, `group_size=128`, both clip ratios, and `selected_attention_implementation=
oscar-mixed-gpu-d4-2b`.

## Implementation locations

- `src/targets/qwen3_6/impl/runtime/text_context_impl.h:938-1127` — opt-in live GPU branch in
  `TextContext::attn_mix`; it consumes actual post-RoPE runtime tensors, performs validated
  FP32 rotations, appends real rotated K/V through the existing aging cache, launches the
  mixed reader, applies FP32 `R_V.T`, and performs the existing downstream BF16 conversion.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:592-725` — typed page validation,
  one-time-per-layer-invocation contiguous staging, and device-buffer management in
  `OscarLiveMixedReferenceCache::prepare_gpu`.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:727-773` — causal query dispatch
  in `OscarLiveMixedReferenceCache::attention_gpu` to the D4.2b CUDA API.
- `src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp:775-856` — unchanged qualified
  append/aging path and exact standalone `OscarInt2G128` parity guard.
- `src/ops/softmax_attention/oscar_mixed/launch.h:26` and `launch.cu:192-` — D4.2b mixed
  GPU interface and implementation. Historical K/V are decoded at use; no persistent decoded
  K/V buffer is created.
- `tests/targets/qwen3_6_27b/test_oscar_runtime.cpp:598-699` — D4.3 forced-token/performance
  harness; `:307-390` and `:475-530` add the small natural smoke and performance-only switch.

For this first integration, `prepare_gpu` flattens the typed page data into contiguous host
arrays and copies it to persistent per-cache device views on every layer invocation. This is
intentional staging, not a hidden BF16 historical shadow: historical arrays remain packed INT2
plus FP32 metadata, and the D4.2b kernel decodes directly. The staging bytes and time are
reported below.

## Exact build and test commands

Build:

```powershell
cmd /c "call C:\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat && D:\\AI\\agent-orchestrator\\omp-home\\localappdata\\vcpkg\\downloads\\tools\\cmake-3.30.1-windows\\cmake-3.30.1-windows-i386\\bin\\cmake.exe --build D:\\AI\\build-adaptive-dflash2 --target ninfer_qwen3_6_27b_oscar_runtime_test -j 4"
```

The final binary was rebuilt after the telemetry and harness changes:

```text
D:\\AI\\build-adaptive-dflash2\\tests\\ninfer_qwen3_6_27b_oscar_runtime_test.exe
SHA-256 84B979EE1EDEABB82356B8AFFFBD12DF11F58BBA9ACD41A3D1734C8449C8A4C3
```

Source identities used by that binary:

```text
src/targets/qwen3_6/impl/runtime/oscar_rotated_bf16.cpp
  540DFE4367E5B8B7F131428DCC12F7DAC353F20CB7E345C852E7070E50E75020
src/targets/qwen3_6/impl/runtime/text_context_impl.h
  91FE93F4F02CE4DEAC16917453B5C94BDD7EC573D3011F5DDDFFC2F1C0888364
tests/targets/qwen3_6_27b/test_oscar_runtime.cpp
  00A2B749A80CB46281FEB78C0737B2347BADBF7434FDEF61FBDA8E0E200A9737
```

The D4.2b focused GPU test remained green after the integration build:

```text
1/1 Test #18: ninfer_oscar_mixed_gpu_attention_test ... Passed 27.67 sec
100% tests passed, 0 tests failed out of 1
```

The main correctness command used the verified model SHA, C4 runtime directory, diagnostic
directory, `NINFER_OSCAR_D4_3_LIVE=1`, and the rebuilt executable. It ran fixed input cases
321, 332, and 512 with nine forced continuation IDs:

```text
997,1001,1003,1005,1007,1009,1011,1013,1015
```

The 4K run used the same harness. A performance-only 16K probe used
`NINFER_OSCAR_D4_3_PERF_ONLY=1`; it was deliberately terminated at logical context 3,072
after confirming healthy asset/topology telemetry but impractical first-integration scaling.

## Live/reference attention parity

The optimized path’s inline tap compares the actual GPU output with the qualified scalar
mixed-cache reader for the same runtime cache and query. The gate is relative L2 `<=1e-4`
and max absolute error `<=1e-3`; all taps passed. Values below are the maximum relative L2
over the layer-3/35/63 taps in each requested case.

| Requested case | Tap count | Rotated AV rel L2 | Recovered attention rel L2 | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 321 | 30 | `5.14381e-7` | `5.97630e-7` | PASS |
| 332 | 30 | `5.58012e-7` | `6.55704e-7` | PASS |
| 512 | 30 | `7.59950e-7` | `8.28918e-7` | PASS |
| 4,096 | 72 | `2.52443e-6` | `2.50990e-6` | PASS |

The 4K tap maximum was still more than 39 times below the relative-L2 gate. The GPU branch
uses the scalar reader only as an explicitly labelled validation oracle when
`NINFER_OSCAR_D4_3_VALIDATE_REFERENCE=1`; it is not a serving fallback. The scalar telemetry
label is `oscar-mixed-reference-cpu-oracle-only` in that situation.

## Mixed-cache boundary and dispatch checks

The actual runtime append path produced real rotated rows and passed them through the existing
typed cache. At the 512-token final telemetry point the cache contained prefix/history/recent
`64/192/256`; at the 4K final point it contained `64/3,784/256`. The historical packed payload
and metadata at 4K were `31,457,280` and `7,864,320` bytes respectively in the cumulative
runtime telemetry.

Every GPU request reported:

```text
full_layer_dispatch_bitmap=1111111111111111
gdn_dispatches=0
legacy_q2_dispatched=false
bf16_historical_shadow=false
fallback=false
selected_layout=mixed-bf16-prefix-oscar-int2-g128-bf16-recent
```

The 321/332/512 runs also exercised the first-history and aging boundaries; the 4K run exercised
repeated recent-to-INT2 aging with actual runtime rows. Each aged row is parity-checked against
the standalone official codec before the GPU view is staged. No GDN state enters this cache.

## Full-model fixed-token diagnostics

These are informational model-fidelity measures, not the implementation gate. The BF16 and
GPU runs used identical nine-token forced continuations. `Top10` is the intersection count at
the final captured logit vector.

| Input case | Comparison | Hidden rel L2 | Logit max abs | Logit mean abs | Logit rel L2 | Top1 | Top10 | Forced continuation |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 321 | BF16 vs GPU | `0.0454180` | `0.4453125` | `0.0712038` | `0.0225004` | yes | 10/10 | exact |
| 321 | scalar live vs GPU | `0.0443007` | `0.4531250` | `0.0619992` | `0.0194630` | yes | 9/10 | exact |
| 332 | BF16 vs GPU | `0.0482597` | `0.7109375` | `0.0660142` | `0.0190280` | yes | 10/10 | exact |
| 512 | BF16 vs GPU | `0.0524625` | `0.6015625` | `0.0803308` | `0.0246913` | yes | 10/10 | exact |

The nonzero BF16-vs-OSCAR drift is expected INT2/model-fidelity behavior and is not evidence
of a live attention implementation defect. The optimized-vs-scalar live attention gate is the
primary correctness result.

## Natural fidelity smoke

A small D3.1-style smoke used the unchanged objective evaluators with four cases: one
arithmetic/reasoning prompt, one coding prompt, one structured JSON prompt, and one 1,148-token
retrieval prompt. The GPU mode used a 16-token output budget (structured JSON was allowed its
existing 48-token closeout). All four objective checks passed in both controls; the GPU path
also passed all four.

| Case class | Prompt tokens | BF16 objective | GPU objective | First token difference |
| --- | ---: | --- | --- | ---: |
| Arithmetic | 30 | PASS | PASS | 8 |
| Coding | 27 | PASS | PASS | 16 |
| Structured JSON | 28 | PASS | PASS | 48 |
| Retrieval / historical INT2 | 1,148 | PASS | PASS | 16 |
| **Total** | — | **4/4** | **4/4** | — |

This is a smoke only. The broader D3.1 scalar/reference suite remains the quality evidence for
the 12-case, four-long-context workload; this phase does not claim a new broad quality score.

## Real-model performance

Verifier time is request wall time for the deterministic fixed-token run, including model
execution but excluding process setup as measured by the harness. Prompt tok/s is
`requested_input_tokens / verifier_ms`; it is not a steady-state decode claim.

| Input case | BF16 control ms | scalar `reference-live` ms | optimized GPU ms | GPU prompt tok/s | Forced IDs |
| ---: | ---: | ---: | ---: | ---: | --- |
| 321 | 878.680 | 83,164.112 | 2,903.730 | 110.55 | exact |
| 332 | 188.900 | — | 2,776.749 | 119.57 | exact |
| 512 | 201.538 | — | 6,889.555 | 74.31 | exact |
| 4,096 | — | — | 486,346.291 | 8.42 | exact |

The separate scalar D4.1 real-live 512 run was `174.529601 s` including its scalar reader and
is the relevant historical context; the new GPU path is approximately 25.3× faster than that
wall result at 512, but the runs were not a tightly paired benchmark. The BF16 control is much
faster because it does not perform OSCAR rotation, staging, aging, or mixed INT2 attention and
is not a like-for-like kernel baseline.

The user-supplied historical experimental route reported approximately 90 tok/s at 16K. It
was not rerun here and is retained only as historical context, not as a comparable D4.3 result.

## Runtime stage profile

The profile counters are cumulative over all full-attention layer invocations in one request.
The D4.2b kernel fuses INT2 K decode + QK + global softmax input + INT2 V decode + AV, so those
sub-stages cannot be separated inside this first live launch. `gpu_mixed_kernel_us` is their
combined GPU time. `gpu_cache_staging_*` is the explicit typed-page flatten/copy overhead.

| Stage | 512 final profile | 4K final profile |
| --- | ---: | ---: |
| Q/K/V rotation | `149.634 ms` | `590.442 ms` |
| Recent→INT2 append/aging | `2.371 s` | `398.977 s` |
| Explicit cache staging | `7.936 ms`, 39.715 MB cumulative | `132.383 ms`, 1.114 GB cumulative |
| Fused mixed GPU kernel | `1.178 s` | `46.079 s` |
| `R_V.T` recovery | `42.235 ms` | `339.310 ms` |
| Complete full-attention branch | `6.565 s` | `480.292 s` |
| Request/verifier wall | `6.890 s` | `486.346 s` |

At 4K, append/aging is about 83% of the complete full-attention branch. It includes the
first-integration CPU-side conversion/append work for actual runtime rows and the repeated
qualified aging path; the explicit cache flatten/copy is smaller in wall time but grows in
cumulative traffic. The fused GPU traversal is about 9.6% of the branch. Q rotation and
`R_V.T` remain intentionally unoptimized, as required.

### History slope

Across the completed real-model 512→4K pair:

- request wall slope: `(486.346 - 6.890) s / 3,584` ≈ **133.8 ms per additional prompt token**;
- complete full-attention slope: `(480.292 - 6.565) s / 3,584` ≈ **132.6 ms/token**;
- cumulative fused mixed-kernel slope: `(46.079 - 1.178) s / 3,584` ≈ **12.5 ms/token**;
- cumulative explicit staging slope: about **0.300 MB per added prompt token**.

This is an integration-request slope, not a steady-state one-token decode slope. It reflects the
current causal prefill implementation and is visibly superlinear as context grows. At the
partial 16K exploratory run, telemetry reached historical `2,752` / recent `256` / prefix `64`
at logical context `3,072` without a correctness failure, but the projected completion cost was
not healthy enough to call a 16K benchmark.

For context, the isolated D4.2b mixed kernel—not the full model—measured `0.208710 ms` at 512,
`1.109395 ms` at 4K, `4.373304 ms` at 16K, and `9.361144 ms` at 32K. Those numbers demonstrate
the kernel’s isolated scaling but must not be substituted for the real-runtime result above.

## Workspace and memory observations

The integration did not add a persistent decoded historical buffer or a Q4 shadow. The first
implementation uses scoped FP32 Q/K/V/attention/recovery buffers for each prefill chunk and
persistent typed contiguous GPU views for the staged prefix/history/recent arrays. With the
current 256-token prefill chunk, the scoped FP32 tensor scratch is approximately 20.97 MB per
active layer invocation. At `max_context=4,112`, the persistent staged-view allocation is
approximately 4.51 MiB, excluding allocator/runtime bookkeeping. The telemetry’s 4K
`1,113,563,136` bytes is cumulative staging traffic across repeated layer invocations, not
resident VRAM.

Process-wide VRAM was not used as an acceptance metric because model residency and allocator
state are shared with the unchanged Qwen runtime. The measured workspace and cumulative transfer
traffic are the reproducible D4.3 memory figures.

## Failure/guard review

- Asset identity and model SHA: PASS.
- Actual runtime Q/K/V input path: PASS.
- Real rotated cache append and repeated aging: PASS.
- GPU mixed reader vs scalar oracle: PASS at 321/332/512/4K.
- All full-attention dispatch: PASS, bitmap `1111111111111111`.
- GDN dispatch: PASS, zero.
- Legacy Q2 dispatch: PASS, false/zero.
- Persistent BF16 historical shadow: PASS, false.
- CPU fallback: PASS, false; CPU use was validation-oracle-only.
- NaN/Inf/logit corruption: PASS; no non-finite result or rollback/layout failure.
- 16K/32K real-model timing: **not qualified** because the first integration was not healthy
  enough to complete 16K; no quality conclusion is drawn from this deferral.

## Decision and next authorization

**PASS.** The qualified D4.2b mixed-tier GPU attention path now executes inside the real Qwen3.8
full-attention runtime with actual rotated cache data, correct mixed-tier addressing, and live
reference parity. The 512 and 4K runs are finite and forced-token stable. D4.3 does not claim
16K/32K real-model throughput or long-context model fidelity; those are pending after the first
integration overhead is removed.

The single recommended D4.4 optimization is:

> **Replace per-invocation host flattening/CPU aging with persistent GPU-resident incremental
> typed page views and device-side recent→INT2 aging, publishing only appended/aged ranges to
> the D4.2b reader.**

This directly attacks the measured 398.977 s 4K aging/append cost and also removes most of the
1.114 GB cumulative staging traffic. The per-query launch loop should remain the second
optimization candidate, but it is not the selected D4.4 target. D4.4 is authorized to begin;
no D4.4 code is included in this phase.
