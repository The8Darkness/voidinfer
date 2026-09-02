# OSCAR Phase B1 — Calibration Environment and Qwen3.8 Topology

Recorded: 2026-09-01 (Europe/Berlin)

## Scope and stop condition

This is an environment, topology, and capture-boundary audit only. No QKV capture,
calibration, rotation asset generation, CUDA-kernel optimization, DFlash2/MTP work, or
adaptive-K work was performed. The proposed capture point below is documentation only;
the runtime was not instrumented.

## Evidence basis

The audit used the following sources and checks:

| Evidence | Verified result |
|---|---|
| Latest source tree | D:\AI\voidinfer-adaptive-dflash2 |
| Latest build | D:\AI\build-adaptive-dflash2; Ninja Debug; CUDA architecture 120a; CMake home D:/AI/voidinfer-adaptive-dflash2 |
| Qwen3.8-27B artifact | D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer; model id qwen3.8-27b; weights id nvfp4-dflash2; 23,719,496,192 bytes; SHA-256 6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E |
| Artifact metadata | 1,190 objects, 1,184 tensors, 6 resources; full-attention and GDN object families match the runtime split described below |
| Runtime load-plan check | D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_load_plan_test.exe exited 0 with the DFlash2 artifact bound through NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS |
| Official fitter | [FutureMLS-Lab OSCAR compute_kv_rotation.py](https://raw.githubusercontent.com/FutureMLS-Lab/OSCAR/main/rotation/compute_kv_rotation.py) was inspected; it consumes per-layer q/k/v .pt chunks, reshapes them as (-1, heads, head_dim), and is GQA-aware |

Prior Phase-A references reviewed and preserved: docs/OSCAR_KV_FIDELITY.md,
results/oscar/phase-a-provenance-2026-09-01.md, PROJECT_GOALS.md, PROJECT_STATE.md,
EXPERIMENTS.md, and the existing tools/oscar/README.md plus validate_dump.py staging.

The artifact metadata was inspected with:

~~~powershell
Set-Location D:\AI\voidinfer-adaptive-dflash2
D:\AI\tools\oscar-calibration\.venv\Scripts\python.exe -m tools.artifact.inspect D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer --json
~~~

The load-plan check was run with:

~~~powershell
$env:NINFER_QWEN3_8_27B_NVFP4_DFLASH2_WEIGHTS = 'D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer'
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_load_plan_test.exe
~~~

## A. Reproducible offline calibration environment

The fitter environment is deliberately outside the VoidInfer build and contains CPU
PyTorch only. LibTorch was not added to VoidInfer.

| Item | Verified value |
|---|---|
| Environment | D:\AI\tools\oscar-calibration\.venv |
| Setup script | D:\AI\voidinfer-adaptive-dflash2\tools\oscar\setup_calibration_env.ps1 |
| Python | 3.12.10, 64-bit Windows |
| PyTorch | 2.13.0+cpu |
| CUDA available in fitter env | False |
| Freeze | D:\AI\tools\oscar-calibration\requirements.freeze.txt |
| Verification report | D:\AI\tools\oscar-calibration\verification\verification.json |
| Tensor round-trip fixture | D:\AI\tools\oscar-calibration\verification\roundtrip.pt |

Reproduction command:

~~~powershell
& 'C:\Program Files\PowerShell\7\pwsh.exe' -NoProfile -ExecutionPolicy Bypass -File 'D:\AI\voidinfer-adaptive-dflash2\tools\oscar\setup_calibration_env.ps1'
~~~

The script creates the venv, installs the pinned CPU wheel, writes pip freeze, and
executes the deterministic verification. The resulting freeze is:

~~~text
filelock==3.32.3
fsspec==2026.7.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
setuptools==78.1.0
sympy==1.14.0
torch==2.13.0+cpu
typing_extensions==4.16.0
~~~

Verification results:

1. import torch passed (torch==2.13.0+cpu).
2. A deterministic float64 tensor of shape [4, 6] was saved to and loaded from
   roundtrip.pt; contents and shape matched exactly.
3. With seed 20260901, a deterministic symmetric float64 [128, 128] matrix ran
   through torch.linalg.eigh(). Eigenvalue shape was [128], eigenvector shape was
   [128, 128], all values were finite, and the eigenvalue range was
   -15.809326280254956 .. 15.463968551352416.

PyTorch emitted a non-fatal optional-NumPy warning because NumPy is not installed in
this minimal CPU environment. The required import, .pt round-trip, and eigendecomposition
checks all passed; the inspected OSCAR fitter itself does not require a NumPy import.

## B. Verified Qwen3.8-27B topology

Qwen3.8-27B is packaged as qwen3.8-27b but uses the shared Qwen3.6-family runtime
implementation. This was verified from the package identity, the loaded artifact object
families, and the runtime configuration/topology helpers—not inferred from the model name.

| Property | Exact verified value | Runtime/artifact evidence |
|---|---:|---|
| Text transformer layers | 64 | src/targets/qwen3_6_27b/impl/config.h:14 |
| Full-attention layers | 16 | config.h:54-68; hybrid_topology.h:7-25 |
| GDN layers | 48 | complement of the full-attention set; config.h:58-68 |
| Query heads | 24 | config.h:29 |
| KV heads | 4 | config.h:30 |
| GQA ratio | 6 (24 / 4) | one KV head serves six query heads |
| Head dimension | 256 | config.h:31; artifact q/k norms are [256] |
| Rotary dimension | 64 | config.h:32; ops::rope call in text_context_impl.h:860 |
| Text hidden width | 5,120 | config.h:13; projection input columns in artifact |
| Q projection width | 6,144 (24 * 256) | config.h:41; runtime workspace |
| K/V projection width | 1,024 (4 * 256) each | config.h:42; runtime workspace |

The topology rule is (layer + 1) % 4 == 0 for full attention. Therefore the exact
0-based full-attention list is:

~~~text
3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63
~~~

The exact 0-based GDN list is:

~~~text
0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45,
46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62
~~~

The artifact independently shows the split. Every verified full-attention layer has:

~~~text
text/layers/{l}/attention/query_key_gate_value  FP8_E4M3FN_ROW_BF16S [14336, 5120]
text/layers/{l}/attention/query_norm            BF16                  [256]
text/layers/{l}/attention/key_norm              BF16                  [256]
text/layers/{l}/attention/output                FP8_E4M3FN_ROW_BF16S [5120, 6144]
~~~

The [14336, 5120] parent is the fused row order [query 6144, key 1024,
output_gate 6144, value 1024]. A representative GDN layer instead has:

~~~text
text/layers/{l}/gdn/query_key_value_z  FP8_E4M3FN_ROW_BF16S [16384, 5120]
text/layers/{l}/gdn/norm               BF16                  [128]
text/layers/{l}/gdn/output             FP8_E4M3FN_ROW_BF16S [5120, 6144]
~~~

TextContext::run_layers dispatches the 16 full layers to attn_mix and the 48 other
layers to gdn_mix (src/targets/qwen3_6/impl/runtime/text_context_impl.h:1009-1055).
The GDN path performs convolution and recurrent gated-delta-net updates; its recurrent
state is not a paged K/V cache and is not an OSCAR capture source. OSCAR scope is therefore
restricted to the 16 full-attention layers above.

## Runtime tensor shapes and ordering

The workspace recipe allocates the projection roots as [rows, tokens]:

~~~text
hidden       [5120, T]
query        [6144, T]
output_gate  [6144, T]
key          [1024, T]
value        [1024, T]
~~~

src/targets/qwen3_6/impl/runtime/text_context_impl.h:839-842 views these as:

~~~text
q  = [D=256, Hq=24,  T]
k  = [D=256, Hkv=4,  T]
v  = [D=256, Hkv=4,  T]
~~~

The NInfer tensor convention has ne[0] as the fastest logical dimension and contiguous
strides derived from that (src/core/tensor.h; src/core/tensor.cpp:50-56). Thus the
physical element order for a single-batch contiguous view is:

~~~text
runtime logical [D, H, T]  -> physical [T][H][D]
                              D fastest, then head, then token
~~~

For the batched path, the runtime uses [D, H, width, batch] views at
text_context_impl.h:865-879; the physical order is [batch][token][head][D].
Consequently, the future OSCAR serialization should use [T, H, D] for a single batch,
or [B, T, H, D] for batched capture, preserving batch-major token order. A future
capture implementation must transpose/copy from the runtime view before saving if it
uses a tensor representation whose logical axes are [D,H,T]; passing that raw view to
the official fitter would make its reshape(-1, H, D) interpretation wrong.

For Q, H=24; for K and V, H=4. Q head h maps to KV head floor(h / 6). The
token index is shared across Q, K, and V for each projection column; K/V use the same
KV-head order. The future dump must retain that alignment.

## C. Exact proposed Q/K/V capture boundary

The verified full-attention sequence in TextContext::attn_mix is:

~~~text
x
  -> input RMSNorm -> h                                  (line 837)
  -> fused/split projection -> q, gate, k, v             (lines 847-848)
  -> q RMSNorm -> qn; k RMSNorm -> kn                    (lines 851-854)
  -> in-place RoPE(qn, kn)                                (line 860)
  -> causal_softmax_attention(qn, kn, v, ...)             (lines 876 or 881)
       -> cache append inside the attention wrapper
       -> attention computation
~~~

The exact future candidate boundary is immediately after ops::rope(...) completes at
src/targets/qwen3_6/impl/runtime/text_context_impl.h:860, and immediately before the
attention output allocation/dispatch at lines 862-881:

~~~text
// documentation-only proposed boundary; not implemented in Phase B1
Q_capture = qn;  // after q RMSNorm and after RoPE; [256, 24, T]
K_capture = kn;  // after k RMSNorm and after RoPE; [256,  4, T]
V_capture = v;   // projection output; no RMSNorm and no RoPE; [256, 4, T]
// then call causal_softmax_attention(...)
~~~

This boundary is OSCAR-compatible for the following reasons:

- Q and K are the exact post-normalization, post-RoPE tensors consumed by attention.
- V is the corresponding projected BF16 tensor, still aligned to the same token columns
  and KV heads as K, and is not transformed by Q/K normalization or RoPE in this runtime.
- The subsequent causal_softmax_attention wrapper appends K/V to the paged cache before
  launching attention (src/ops/softmax_attention/dense/causal_cache/prompt.cu:135-145,
  and the OSCAR path at :168-182). Capturing after that point could observe a quantized/
  packed cache representation instead of the source K/V values required for calibration.
- gate is a separate output-gating tensor and is not an OSCAR Q/K/V capture.
- No post-attention or GDN recurrent-state tensor belongs in this capture.

For future capture insertion, both prompt/prefill and verify execution should use the same
semantic boundary. The two attention call forms at text_context_impl.h:876 and :881
consume the same qn, kn, and v; only batching differs. A later implementation must
copy the source tensors before cache append, convert them to the documented host layout,
and preserve Q/K/V token alignment. It must not dump the cache rows produced by
kv_cache_append_batch_launch or kv_cache_append_oscar_batch_launch.

## Exact source locations for a later insertion

| Role | Source location | Finding |
|---|---|---|
| Qwen3.8 package identity | src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h | Package id is qwen3.8-27b; it selects the shared Qwen3.6-family runtime |
| Model constants | src/targets/qwen3_6_27b/impl/config.h:12-68 | 64 layers, 24/4 heads, D=256, rotary dim=64, compile-time 16/48 split |
| Layer rule/indexing | src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h:7-25 | Full layer rule and full/GDN ordinal mapping |
| Projection outputs | src/targets/qwen3_6_27b/impl/variant.cpp:197-209 | Materializes separate BF16 query, gate, key, and value outputs |
| Runtime projection/workspace shapes | src/targets/qwen3_6/impl/runtime/workspace_recipe.h:13-81 | [rows,T] roots and normalized result buffers |
| Candidate boundary | src/targets/qwen3_6/impl/runtime/text_context_impl.h:828-889 | Projection, Q/K normalization, in-place RoPE, and attention call sequence |
| Full/GDN dispatch | src/targets/qwen3_6/impl/runtime/text_context_impl.h:891-1055 | GDN recurrent path is separate from full attention |
| Cache append boundary | src/ops/softmax_attention/dense/causal_cache/prompt.cu:135-182 | K/V append occurs inside the attention wrapper after the proposed boundary |
| Append source layout | src/ops/kv_cache/append/kernel.cuh:440-499 | Append kernels consume the BF16 source K/V in token/head/D order before packing |
| RoPE contract | include/ninfer/ops/rope.h:9-37; src/ops/wrapper/rope.cpp:101-124 | In-place Q/K RoPE; Q/K are [D,H,T], rotary prefix is 64 of D=256 |
| Attention/cache dispatch | src/ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp:417-468 | Selects cache path after receiving the source Q/K/V tensors |

## Unresolved questions before Phase B2 capture

1. The current local validator/schema uses per-layer .bin files and an oscar-qkv-v1
   manifest, while the official fitter expects layer_<id>/{q,k,v}/{chunk_id}.pt and
   skips chunk 0 when --chunk-id all is used. A serialization/manifest bridge must be
   chosen before capture; no bridge was implemented here.
2. The official fitter defaults to --head-dim 128; Qwen3.8 requires an explicit head
   dimension of 256. The fitter invocation and any layer-specific asset naming must be
   validated against the 16 full-attention layer set before calibration.
3. The capture transport, chunk size, warmup-token policy, and host-copy synchronization
   remain unspecified. In particular, official chunk loading has a six-token warmup
   convention that must not be applied accidentally to a future local manifest.
4. Whether calibration is text-only or includes multimodal prompts remains open. The
   runtime can pass 1-D positions or multimodal [T,3] MRoPE positions; the chosen
   calibration corpus must define which position path is represented.
5. Existing fixed-Hadamard OSCAR-Q2/VeriCache code and the current nvfp4-dflash2 artifact
   are runtime experiments, not evidence that calibrated upstream OSCAR assets already
   exist. This audit does not qualify either as faithful OSCAR calibration.

## Phase-B1 result

The offline CPU fitter environment is reproducible and passes all required primitives. The
loaded Qwen3.8-27B artifact/runtime is verified as a 64-layer hybrid with 16 full-attention
layers (3,7,...,63) and 48 GDN layers, using 24 Q heads, 4 KV heads, head dimension 256,
rotary dimension 64, and GQA ratio 6. The exact candidate capture source is qn, kn, and
v immediately after post-normalization Q/K RoPE and before causal_softmax_attention/cache
append. No QKV dump or calibration was run.
