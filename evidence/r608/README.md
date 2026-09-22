# R608: untimed native same-weight DFlash2 baseline

Original receipt: `native-r608-dflash-untimed-4096x128-r1` (2026-09-22).

This is native same-weight ordinary FP16 device-KV DFlash2 at physical C1 with exact target
verification. It is not target-only, HTTP serving, pure decode-kernel timing, model-start latency, or
client-visible TTFT.

- Source paths: 1,283
- Source aggregate: `60d11dabfb0070902ec6a835a54009909c5103884182d977e2481d38d7b2b6e3`
- Binary SHA-256: `304073b3baa8e4344beaebf358f9d2b504566542ebea5133dfc9f80a7bcecf46`
- Status: PASS; exit code 0
- Recorded checks: 2,822/2,822 pass
- Run rows: 12/12 exact
- Timing instrumentation: off (`VOIDINFER_PARITY_DFLASH_TIMING=0`)

Each arm contains three sequential requests and 384 useful outputs. Throughput is derived from the
grouped wall value in [workloads.csv](workloads.csv), not a median or best request:

`useful_tok_s = useful_outputs * 1000 / wall_ms`

The source aggregate predates later diagnostic-only R609–R612 changes. The publication snapshot
aggregate is different and does not retroactively requalify R608.
