# Current status

## Status 2026-10-12 (VoidInfer-only engine)

The EXL3 coherent-device lane engine is the only engine. Removed with the inherited route: the
`.ninfer` artifact container, its converters, loader and per-target runtimes (Qwen3.6-27B,
Qwen3.6-35B-A3B), MTP, the gated 6-bit Vision tower and the research media route, the inherited
Python serve/TTFT campaign tools, and every shared Op kernel the engine no longer launches (214 -> 20
op sources). Everything is renamed to VoidInfer (`voidinfer`, `voidinfer-serve`, `VOIDINFER_*`).

The Engine now applies its benchmarked profile itself (`src/exl3/engine_profile.cpp`, 189 switches
from the 2026-10-08 FastDeviceTransaction receipt), so `voidinfer-serve --target DIR --draft DIR` runs
the qualified configuration without a receipt CSV. Harness check on the same binary, 4K code
workload, 1,024 outputs: C1 163 tok/s, C4 356-386 tok/s (the receipt CSV gave 306-337).

## Status 2026-10-12 (NInfer comparison)

Full head-to-head against NInfer `a0cf1c17` (upstream `81c8ce0`) on 2026-10-12:
[ninfer-comparison-2026-10-12.md](ninfer-comparison-2026-10-12.md).

| Area | VoidInfer default (L0 OSCAR INT4 + exact VeriCache) | NInfer best configuration |
| --- | --- | --- |
| Quality | KL vs BF16 0.0137, top-1 97.2%; paired NLL better by 0.016 nats/token (CI excludes 0) | KL 0.080–0.087, top-1 92.4–92.7% |
| DFlash2 C1, code 4K / 16K / 32K / 64K / 128K / 254K | 180 / 169 / 145 / 160 / 158 / 137 tok/s, 2.95–3.67 tokens/round | 187 / 187 / 190 / 195 / 172 / 141 (254K only with k8v4 KV) |
| Scenarios (12 NInfer prompts, geomean) | 212 tok/s | 247 tok/s |
| Base decode 4K / prefill 4K | 76 tok/s / 7.8K tok/s | 94 tok/s (groupwise) / 14.1K tok/s (NVFP4) |
| Concurrency 4K C2 / C4 / C8 | 258 / 365 / 396 tok/s | 340 / 526 / does not start (per-slot FP8 KV) |
| 8 agents x 99K-113K tokens, 4 lanes | first turn 223 s, resume 52 s (24-25 s after `a45ce283`, plane sets) | first turn 310 s, resume 11 s (groupwise) |
| Vision | none in the comparison run; since `7c0b09fa` images and video (`--vision`), 1.6-1.8 s per image request | image and video, 1.0-1.3 s per image request |

Fixed during the run: `98f10860` asynchronous VeriCache blocks settle on a round closed by the terminal
token (every C1 request that ended at EOS before its budget failed with "asynchronous verification settle").

After the run: `7c0b09fa` adds image/video input (BF16 vision tower streamed from host, M-RoPE rotary map
per lane) and serves prompts under 2,048 tokens (virtually saturated DFlash2 ring); `a45ce283` keeps
each agent's history in its own plane set and resumes agents by remapping it (restore 3.4-6.0 s -> 0.1-0.7 s).
Details: [update section](ninfer-comparison-2026-10-12.md#update-vision-and-agent-resume).

## Status 2026-10-08

Toolchain: VS 2026 Build Tools (MSVC 14.51) + CUDA 13.3.1, `sm_120a`, RTX 5090 (driver 617.14).
CUDA 13.1 and 13.3 builds give identical greedy state hashes and tokens at 4K/16K.

| Capability | State/default | Measured (2026-10-08) |
| --- | --- | --- |
| Cold prefill, target-only | Default (layer-major, FP16 projections, FA2 attention); `VOIDINFER_EXL3_PREFILL_MXFP8=1` opts into MXFP8/NVFP4 block-quantized projections (about 1.5x faster; KL vs FP16 KV + FP16 prefill 0.0086 / 0.0104 instead of 0.0014 / 0.0024 at 4K / 16K) | Served C1, 31K prompts: 3.8K tok/s FP16, 5.9K MXFP8 (2026-10-13); MXFP8 target-only 10.1K tok/s at 4K, 9.6K at 16K (2026-10-08) |
| Base greedy decode, target-only | Default (whole-stack decode graph, fused reductions, L2 prefetch) | 79.2 tok/s at 4K, 75.7 at 16K |
| DFlash2 K=7 decode, public Engine (coherent-device C1) | Default (verification trees, ≤8 verify rows) | 211 tok/s at 4K, 186 at 16K (code-review workload); 241 tok/s geomean on NInfer's 12 scenario prompts |
| L0 OSCAR KV tier with exact hot rows | Default at every context: exact 960-row recent window (`VOIDINFER_EXL3_L0_RECENT`, 0..960) and 512 hot rows per KV head (`VOIDINFER_EXL3_L0_HOT`, 0..4096); `VOIDINFER_EXL3_L0_OSCAR=0` opts out; rotations from `<model directory>/l0_oscar` | DFlash2 code workload 18.3 / 18.5 / 18.8 / 19.4 / 20.3 ms per round at 4K / 16K / 32K / 64K / 128K (hot rows +0.2–0.4 ms); prefill 6.9K / 7.7K / 7.3K / 6.4K / 5.1K tok/s; verify-quality gate vs FP16 KV: +0.006 nats/token, mean abs. delta 0.037, top-1 96.8% (INT2 history alone: +0.024, 0.067, 95.1%) |
| VeriCache block verification of L0 | Default with L0 on the coherent-device Engine: exact (`_DELTA` 0.0625, adaptive blocks up to `_BLOCK` 1024; one lane: asynchronous exact line, 64-token blocks, `VOIDINFER_EXL3_VERICACHE_ASYNC=0` off; the line attends history past the window as an FP8 e4m3 mirror (per row/head scale, half the PCIe bytes), `VOIDINFER_EXL3_VERICACHE_LINE_FP8=0` keeps FP16 — 4K 169-173, 128K 157 (FP16 line 147), 254K 126 (FP16 line 113) tok/s; two lanes: one asynchronous line per lane for requests of 12K+ tokens, line passes ordered with the prompt passes — HTTP C2 16K 157 vs 115, 31K 131 vs 115 tok/s; shorter requests and C3+ keep exclusive verify passes on 256+-row reconstruct-route blocks; batched rounds close their rendezvous when the missing lanes wait at the other gate — HTTP 4K C4 382-387, C8 381 tok/s); `VOIDINFER_EXL3_VERICACHE=tolerance` (1.0, 1,024-token blocks) or `=0`; exact-history prompt ingestion; tokens published only after verification | DFlash2 code workload, 1,024 tokens: 20.06 vs 18.56 ms per round at 16K (+8%), 21.21 vs 19.45 at 64K (+9%), 24.3 vs 20.3 at 128K with one correction (+19%; exact prefill 33.3 vs 25.6 s); typically no corrections per 1,024-token block; verify-quality gate passes with VeriCache on |
| Concurrent long-context agents (coherent-device C2+) | Default: a lane takes a request only when its projected INT4 L0 codes (input + output allowance, 17,408 bytes per token) fit in the free device memory beside the growth the other active lanes still have ahead, keeping `VOIDINFER_EXL3_L0_CODE_RESERVE_MIB` (default 512) free; younger requests wait behind the oldest one that does not fit, a lone lane always runs, and a waiting request runs anyway shortly before its queue deadline; `VOIDINFER_EXL3_L0_CODE_ADMISSION=0` off. Without it WDDM silently demotes code planes to system memory past physical VRAM | 8 agents x ~106K tokens (849K) on C4, 256 tokens per turn: turns of 218 / 49 / 45 s, peak VRAM 30.0 GiB (previously 266 s for the first turn at 32.1 GiB, then the Engine failed with "growable region access"); C8 x 31K prompts (4.5 GB of codes, more than fits): 185 vs 79-120 tok/s, 42 vs 155-245 ms per lane round |
| Root snapshots (export / restore of exact host states and draft rings) | Default: draft-ring export and restore move the ring range through one pinned stage under one stream fence; recurrent planes without a pinned slab (pool exhausted by retained roots), convolution states, logits, embedding and taps download through a 64 MiB pinned stage with one fence per fill and parallel fills; GDN host restores are stream-ordered | Draft ring export 16 -> 5.7 ms and restore 7-62 -> 4.1 ms at 2,047 rows; exact exports per C1 request 100-131 -> 39-70 ms once the slab pool is exhausted (35 ms with pooled slabs); byte-identical states (root export/restore diagnostic hashes unchanged) |
| FP16 device KV with DFlash2 | `VOIDINFER_EXL3_L0_OSCAR=0` | 17.7 ms per round at 4K; fits 16K, the layer-major prefill headroom guard refuses 31K |
| Model quality vs NInfer (identical tokens) | — | FP16 KV better by 0.017–0.020 nats/token (95% CI excludes 0); L0 OSCAR (INT2 only, before hot rows) statistically tied |

Head-to-head results against NInfer are in [ninfer-comparison-2026-10-08.md](ninfer-comparison-2026-10-08.md).
The 2026-09-22 publication table below remains as the historical record of that snapshot.

## Research routes and default-off switches

| Route | State | Where | Boundary |
| --- | --- | --- | --- |
| FP16 device KV (`VOIDINFER_EXL3_L0_OSCAR=0`) | Diagnostic reference only | `src/exl3/text_model.*` | Fits about 16K with DFlash2; never a context tier for serving |
| Tolerance VeriCache (`VOIDINFER_EXL3_VERICACHE=tolerance`) | Default off | `src/exl3/vericache_*` | Trades exactness for 8-9% per round |
| Resident OSCAR runtime (`VOIDINFER_OSCAR_EXL3=1`) | Research, refused by the Engine | `src/exl3/oscar_runtime.*` | Pre-L0 OSCAR route; still entangled with the exact host-state restore path |
| Positive-temperature sampling | Runs, not broadly qualified | `src/exl3/sampling_policy.h`, `sampled_select.*` | Greedy authority does not establish distribution preservation |
| Thinking budgets, stop strings, penalties | Rejected by the Engine | `src/exl3/exl3_engine_core.cpp` | Parsed by CLI/server, refused at submission |
| Segmented resident prefix | Rejected, default off | `src/exl3/text_model.*` | R612 exact screen, 17-21% slower |
| WMMA32 prefill | Quality-pending, default off | EXL3 prefill | Does not match the frozen exact token stream |

## Corrections to historical pointers

R608 is a same-weight DFlash2 workload with target verification and an ordinary FP16 device-KV
state. The heading “target-only” in one immutable R608 handover is inaccurate; the receipt metadata
and current documentation are authoritative. R610–R612 are eight-output diagnostic screens and do
not supersede R608. R604's profiler-perturbed CUDA API proportions are attribution evidence, not
normal production wall-time shares.

Mia parity, broad quality qualification and Engine/client TTFT parity remain unmeasured. `null`
means not measured; it is not inferred from nearby data.
