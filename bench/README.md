# Benchmarks

[`engine-compare/`](engine-compare/README.md) is the benchmark harness: single-request DFlash2 and
base decode, prefill, concurrency (C1-C8), multi-agent long context, vision, and the KL / paired-NLL
quality instruments, each with a matching NInfer driver for head-to-head runs.

Our side is driven by `ours_*.ps1`, which dot-source [`ours_env.ps1`](engine-compare/ours_env.ps1)
for the model package and build. The engine applies its own benchmarked profile, so a run needs no
receipt environment. Results of the latest comparison:
[docs/ninfer-comparison-2026-10-12.md](../docs/ninfer-comparison-2026-10-12.md); current numbers:
[docs/performance.md](../docs/performance.md).
