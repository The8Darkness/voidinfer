# SM120 / DFlash2 donor audit

Updated: 2026-08-26 UTC. Donors are references, not production dependencies.

## Re-audited refs

- `seanyourhighness/vllm-sm12x-nvfp4-dflash2` `main`: `5f2fd9d76544`
- `vllm-project/vllm` `main`: `0a5ad6f0d42c`
- vLLM PR #52816 head: `3406ec1dae99` (merged PR commit `b389ac29465b`)
- `flashinfer-ai/flashinfer` `main`: `919a24e5b1d9`
- FlashInfer PR #4346 head: `bce8f44e319`

## Classification

| Feature | Evidence | NInfer disposition |
| --- | --- | --- |
| NVFP4 KV | FlashInfer PR #4346 `include/flashinfer/attention/prefill.cuh` repacks packed FP4 K/V and UE4M3 scales for single/ragged/paged prefill | Real gap: NInfer currently validates/exposes BF16 and I8 KV only. Requires page payload/scales, append, attention, workspace, graph, and quality changes; no blind port |
| SM120 prefill | PR #4346 reports a focused 3.7% prefill improvement on CT102/SM120/CUDA 13.0 | Backend-specific donor; profile NInfer first |
| DFlash2 | vLLM PR #52816 adds Qwen DFlash2, grouped convolution, candidate selector, and V2 runner behavior | Architectural donor only; NInfer has DFlash for 35B but no Qwen3.8 DFlash2 identity |
| ReplaySSM | Overlay adds compact FP32 checkpoints and replay rings | NInfer already has ReplaySSM records and transactional MTP/DFlash state; compare semantics only |
| Runtime-K/graphs | Overlay threads active `1+K` through compilation/graph keys | NInfer uses startup-fixed windows; defer dynamic-K until graph/active-lane measurements justify contract cost |
| fused M-RoPE/vision | Overlay fuses QK norm, three-axis M-RoPE, and gating in Triton/vLLM | NInfer already owns three-axis positions and M-RoPE; only consider a target-shape CUDA fusion after numerical references |

The overlay is Python/vLLM-specific and contains no drop-in NInfer code. Its repository-authored RTX 5090 rates, acceptance, NIAH, tools, and vision smoke results remain `PUBLISHED_EXTERNAL_RESULT` until reproduced natively. Current FlashInfer work is preferred over the older overlay implementation, and open PR #4346 is not an authority for integrated NInfer behavior.

## Safe experiment order

1. CPU/reference-only packed FP4 repack oracle against representative bytes/scales.
2. Current NInfer BF16/I8 attention validation and exact byte accounting.
3. Only after full baseline/quality gates: one fixed SM120 NVFP4 prefill shape with independent oracle.
4. Separately qualify groupwise NInfer Qwen3.8 DFlash2 and ReplaySSM K0-K7 before any all-NVFP4 artifact/export work.

No overlay or FlashInfer code has been copied into VoidInfer.
