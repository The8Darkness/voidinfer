# q27 audit

Updated: 2026-08-26 UTC. Remote: `https://github.com/signalnine/q27`.

## Re-audited refs

- `master`: `9b10e16fc568d970805e52280cce48d01af4c904`
- `adaptive-suffix-width`: `9cf9654510945e`
- `prefill-attn`: `fa028d2709961`
- `prefill-attn-fp8mma`: `a9a30d087ae3`
- `verify-gemv`: `413dab5395dd`

## Findings and disposition

1. **Parser drift is actionable.** q27 master improves bare mode-22 calls, ambiguous chunk boundaries, THINK/TEXT routing, zero-argument calls, mixed dialect batches, and trailing closers in `src/api_common.h`; it reports 159/159 corpus coverage plus sanitizer fuzzing. Current NInfer `src/serve/tool_call_parser.{h,cpp}` recognizes a narrower wrapped syntax. Reuse fixtures and behavioral requirements, not the monolithic API implementation. This is a bounded serving-correctness experiment.
2. **Persistence is semantic guidance, not a patch.** q27 P16 stores token IDs, KV rows, recurrent state, compatibility hashes, owner-only files, and atomic/security controls. NInfer's `.ninfer` artifact ABI and current typed checkpoints differ; implement persistence only after host/device state containers are qualified, with a versioned NInfer schema and corruption rejection.
3. **GDN checkpoint semantics align.** q27 P8/P9 require recurrent matrices, convolution history, hidden/position/speculative state, and KV frontier together. NInfer's ReplaySSM and typed current/rewrite checkpoints already enforce a stronger target-defined contract. Do not add a competing snapshot representation.
4. **Suffix drafting remains deferred.** q27 `src/suffixdraft.h` is a host proposer verified by the target, but W16 widening was a NO-GO (434.5 to 368.5 t/s in its own stack); context length was a poor adaptive proxy. Keep NInfer MTP/DFlash as the production proposer baseline until a common proposer interface and end-to-end cost gate exist.
5. **Negative kernel results matter.** q27's isolated prefill `ldmatrix` gain was 4.1%; verify-GEMV tensor-core work was rejected while safe vectorized loads yielded 5.9%. Neither justifies a blind NInfer port.

## Next bounded q27 experiment

After Phase 1 serving harness work, build a small NInfer parser drift corpus under `tests/` covering bare mode-22, mixed openers, split chunks, THINK routing, zero-argument calls, fenced/prose false positives, and trailing closers. Gate it with existing tool-call security tests and streaming replay. Stop after three implementation attempts or if grammar/template semantics become ambiguous. Suffix drafting and q27 low-bit WHT remain research-only.
