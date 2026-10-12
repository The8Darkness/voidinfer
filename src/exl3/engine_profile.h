#pragma once

namespace voidinfer::exl3 {
// The kernel and route selections the engine is benchmarked and quality-gated with (coherent-device
// lanes, L0 OSCAR history, exact VeriCache, DFlash2 K=7). Every VOIDINFER_* switch in the table that
// is not already set in the process environment is set to its profile value, so a plain
// `voidinfer-serve --target DIR --draft DIR` runs the qualified configuration. Explicitly set
// (non-empty) variables win; the coherent-device startup check rejects values that leave the
// qualified route.
// Idempotent and thread-safe; call before the first engine or text model is constructed.
void apply_engine_profile();
} // namespace voidinfer::exl3
