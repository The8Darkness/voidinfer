# Performance

Qwen3.8-27B, EXL3 6 bpw target with DFlash2 K=7, L0 OSCAR INT4 history with exact hot rows, exact
VeriCache; RTX 5090, VS 2026 + CUDA 13.3. Full protocol and raw data:
[engine comparison 2026-10-12](ninfer-comparison-2026-10-12.md).

## Single request (DFlash2, code workload, 1,024 outputs)

| Prompt | 4K | 16K | 32K | 64K | 128K | 259K |
|---|---:|---:|---:|---:|---:|---:|
| Decode tok/s | 180 | 169 | 145 | 160 | 158 | 137 |
| Committed tokens / round | 3.56 | | | | 3.67 | |

Prefill is 7.8K tok/s at 4K and 4.9K tok/s at 128K (exact-history ingestion). Target-only base decode
is 76 tok/s at 4K. On the twelve scenario prompts in [`examples/cli`](../examples/cli/) DFlash2 averages
212 tok/s (geomean).

## Concurrency (code prompts, 1,024 outputs per request, aggregate tok/s)

| Prompt | C1 | C2 | C4 | C8 |
|---|---:|---:|---:|---:|
| 4K | 161-163 | 258 | 356-386 | 396 |
| 31K | | 125 | 214 | 180 |

## Long-context agents

Eight agents with 99K-113K-token code prompts (865K tokens of live context) on four lanes: the first
turn completes in 223 s; a resumed turn takes 24-25 s, because each agent's exact history stays in
its own plane set and resuming is a remap (0.1-0.7 s per agent).

## Quality

| Measure (4K unless noted) | Value |
|---|---|
| KL divergence from the BF16 model | 0.014 |
| Top-1 agreement with the BF16 model | 97.2% |
| Paired NLL vs NInfer NVFP4/FP8 KV, 4K-32K | 0.016 nats/token better (95% CI excludes 0) |

Instruments: [KV-tier fidelity](OSCAR_KV_FIDELITY.md) and `bench/engine-compare/quality/`.

## Reproduce

`bench/engine-compare/` holds the harness. `ours_env.ps1` names the model package and the build;
everything else is the engine's own profile.

```powershell
cd bench\engine-compare
.\ours_conc.ps1 -Conc 4 -Label c4                       # concurrency, 4K code
.\ours_bench.ps1 -Prompts code_4k:4096,code_128k:131072  # single request, Engine device harness
.\ours_agents.ps1 -Conc 4 -Out D:\runs\agents.json      # 8 long agents, 2 turns
.\quality\ours_nll.ps1 -Windows 4096 -L0                # paired NLL windows
```

Concurrency numbers vary by about ±8% between runs on this machine; compare rounds by milliseconds per
round or repeat interleaved A/B runs before drawing conclusions.
