# Known issues

Updated: 2026-08-26 UTC.

- Five real-model CTest entries are skipped because Qwen3.6/35B artifacts and fixtures are not installed locally; no Qwen3.8 real-model C++ integration test is registered.
- `tools/smoke/serve_contract.py` exercises media and therefore requires a server started with `--vision`; the documented startup path must not be treated as a multimodal smoke by omission.
- Windows-aware defaults are fixed for the serving corpus, concurrency, thinking-preservation smoke, and native benchmark matrix; full model campaign coverage remains outstanding.
- Existing MSVC builds emit module-dependency-file and signed/unsigned-comparison warnings. They are baseline warnings, not silently classified as clean.
- Canonical `Neroued/ninfer` `dev` is a large coupled runtime/resource cutover. Partial cherry-picks are unsafe; use an isolated worktree and preserve the current commit as rollback.
- Local DSpark-named artifact identifies as `qwen3.8-27b/groupwise-int`; its role as a DSpark drafter/speculator is unverified and must not be assumed from its filename.
- Overlay, q27, vLLM, FlashInfer, and published benchmark claims are external evidence until matched local artifacts/configuration and native Windows measurements exist.
