# Known issues

Updated: 2026-08-26 UTC.

- Five real-model CTest entries are skipped because Qwen3.6/35B artifacts and fixtures are not installed locally. The existing opt-in prefix/resource test now covers the Qwen3.8 NVFP4 Engine route, Host State/KV restore, multimodal reuse, cancellation/eviction, and concurrent MTP settlement; the Qwen3.6 thinking-preservation fixture has Qwen3.8-specific reuse-path expectations and is not a Qwen3.8 gate.
- `tools/smoke/serve_contract.py` exercises media and therefore requires a server started with `--vision`; the documented startup path must not be treated as a multimodal smoke by omission.
- Windows-aware defaults are fixed for the serving corpus, concurrency, thinking-preservation smoke, and native benchmark matrix. The local Qwen3.8 NVFP4 text/Vision smokes, 43-case core matrix, and separate C1/C2/C4/C8 decode-saturation points pass; closed-loop corpus/task coverage remains outstanding.
- Existing MSVC builds emit module-dependency-file and signed/unsigned-comparison warnings. They are baseline warnings, not silently classified as clean.
- Canonical `Neroued/ninfer` `dev` is a large coupled runtime/resource cutover. The current integration tree builds and passes the unit/Op suite, but it remains uncommitted and requires independent review before becoming the rollback point. Partial cherry-picks are unsafe.
- Local DSpark-named artifact identifies as `qwen3.8-27b/groupwise-int`; its role as a DSpark drafter/speculator is unverified and must not be assumed from its filename.
- Context-cost transfer calibration is intentionally rejecting the current single roofline fit on this GPU: D2H/D2D/H2D held-out p95 error is unstable across fragmented geometries. The compiled conservative defaults remain active; do not publish measured transfer presets until the fit is improved or a negative-result disposition is recorded.
- Overlay, q27, vLLM, FlashInfer, and published benchmark claims are external evidence until matched local artifacts/configuration and native Windows measurements exist.
