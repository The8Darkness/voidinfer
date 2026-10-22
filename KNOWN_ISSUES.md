# Known issues

Updated: 2026-10-12.

- The Engine rejects requests with a thinking budget, stop strings or penalties, while the CLI
  (`--thinking-budget`, `--stop`, `--reasoning-stop`) and server (`--default-thinking-budget`,
  OpenAI `stop`, penalty fields) still parse them.
- Startup prints kernel-dispatch diagnostics (`FA2_PREFILL_DISPATCH`, `EXL3_GENERIC_NARROW`) to stderr
  during warmup.
- The resident (pre-L0) OSCAR runtime (`src/exl3/oscar_runtime.*`, `VOIDINFER_OSCAR_EXL3`) is refused
  by the Engine but still compiled; it shares the exact host-state restore path with the live route.
- `voidinfer_media_client_loopback_test` is disabled in CTest.
- MSVC builds emit module-dependency-file and signed/unsigned-comparison warnings; they are baseline
  warnings, not a clean build.
