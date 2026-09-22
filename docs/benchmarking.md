# Benchmarking and evidence

## Claim contract

Every result should identify source aggregate or commit, binary hash, model identity, effective
configuration, actual dispatch counters, fixtures, output counts, cache state, concurrency,
instrumentation, timing boundary, and exactness gates. Different binaries or workload contracts are
not pooled as one median. Stage timings are reported separately from useful-output throughput.

## R608 definition

R608 uses native same-weight ordinary FP16 device-KV DFlash2 at physical C1, a 4,096-token prefix,
128 useful outputs per request, three requests per arm, and code/prose fixtures. The harness starts
the grouped wall timer after an initial no-context synchronization. Cold creates a context for each
request; reused creates one lane and reuses it. The timer covers acquire, decode, release, retirement
of intermediate roots, and final lane destruction/synchronization. It does not cover process/model
startup, HTTP/SSE transport, or client-first-visible TTFT.

Useful throughput is exactly `384 * 1000 / grouped_wall_ms`. It is neither a per-run median nor a
best sample. Keep `VOIDINFER_PARITY_DFLASH_TIMING=0`; `NINFER_DFLASH2_TIMING` synchronizes proposal
phases and is only for attribution.

## Correctness dimensions

- Token equality: generated IDs for a frozen trajectory.
- Represented-state equality: target state, committed taps, and reconstructed ring.
- Sampling correctness: distribution and transaction semantics; not implied by greedy equality.
- Task quality: held-out behavior; not implied by exact replay.
- Performance: matched wall time with the same contract; not implied by a microbenchmark.

R608 passed 2,822 recorded checks and 12 run rows. R49 is a useful negative example: token subgroups
matched while state/tap/ring equality failed, so it remained off. R612 passed its scoped exactness
screen but was 17.08–21.15% slower, so it was rejected.

Curated data is under [evidence](../evidence/README.md). Original private receipts retain the dated
names listed there; the repository contains compact exports, not weights, binaries, profiler traces,
or private session history.
