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
| L0 OSCAR KV tier with exact hot rows | Default at every context: exact 960-row recent window (`VOIDINFER_EXL3_L0_RECENT`, 0..960) and 256 hot rows per KV head (`VOIDINFER_EXL3_L0_HOT`, 0..4096; 512 until Oct 15: 32 MiB less per context, but VeriCache corrects about twice as often - sampled LCB C1 / C2 / C4 / C8 155 / 215 / 309 / 328 vs 171 / 226 / 332 / 354 tok/s with 512); `VOIDINFER_EXL3_L0_OSCAR=0` opts out; rotations from `<model directory>/l0_oscar`; `VOIDINFER_EXL3_L0_CODE_BITS=2` stores INT2 codes (9.2 instead of 17.4 KiB per token; with exact VeriCache 4x the corrections, 32K C1 / C4 / C8 decode 93 / 33 / 23 vs 170 / 54 / 45 tok/s per lane, 2026-10-13) | DFlash2 code workload 18.3 / 18.5 / 18.8 / 19.4 / 20.3 ms per round at 4K / 16K / 32K / 64K / 128K (hot rows +0.2–0.4 ms); prefill 6.9K / 7.7K / 7.3K / 6.4K / 5.1K tok/s; verify-quality gate vs FP16 KV: +0.006 nats/token, mean abs. delta 0.037, top-1 96.8% (INT2 history alone: +0.024, 0.067, 95.1%) |
| VeriCache block verification of L0 | Default with L0 on the coherent-device Engine: exact (`_DELTA` 0.0625, adaptive blocks up to `_BLOCK` 1024; one lane: asynchronous exact line, 48-token blocks (`VOIDINFER_EXL3_VERICACHE_ASYNC_BLOCK`; 64 until Oct 15, C4 sampled 92.9 -> 98.1 kept tok/s per lane), lead `VOIDINFER_EXL3_VERICACHE_ASYNC_LEAD` 200 (128/96 no change), `VOIDINFER_EXL3_VERICACHE_ASYNC=0` off; the line attends the exact FP16 history; `VOIDINFER_EXL3_VERICACHE_LINE_FP8=1` reads history past the window from an FP8 e4m3 mirror instead (per row/head scale, half the PCIe bytes, KL 0.005 against FP16 history) — 4K 169-173, 128K 147 (FP8 line 157), 254K 113 (FP8 line 126) tok/s; two to four lanes: one asynchronous line per lane (~480 MiB device memory each at 40K; four 32K-output requests still fit at C4 x 40K), line passes ordered with the prompt passes, used by every sampled request and at two lanes by greedy requests of 12K+ tokens — HTTP C2 greedy 16K 157 vs 115, 31K 131 vs 115 tok/s; C4 sampled (T 1.0, L0 flips one sampled token per ~150-500 past the exact window) 4K 313 vs 279, 12K 282 vs 222, 24K 244 vs 180 tok/s; other requests and C5+ keep exclusive verify passes on 256+-row reconstruct-route blocks (sampled requests at most `VOIDINFER_EXL3_VERICACHE_SAMPLED_BLOCK`, default 256 rows: sampled LCB C8 ~314 -> 341-343 tok/s; greedy keeps up to 1,024, where smaller caps gave no gain); batched rounds close their rendezvous when the missing lanes wait at the other gate — HTTP 4K C4 382-387, C8 381 tok/s); `VOIDINFER_EXL3_VERICACHE=tolerance` (1.0, 1,024-token blocks) or `=0`; exact-history prompt ingestion; tokens published only after verification (each request's first block is short - `VOIDINFER_EXL3_VERICACHE_FIRST_BLOCK`, 32 rows on exclusive passes, 16 on lines, 0 off: TTFT C1 602 -> 391 ms, C4 greedy median 4.6 -> 2.1 s, C8 sampled 8.1 -> 4.5 s, throughput unchanged or better) | DFlash2 code workload, 1,024 tokens: 20.06 vs 18.56 ms per round at 16K (+8%), 21.21 vs 19.45 at 64K (+9%), 24.3 vs 20.3 at 128K with one correction (+19%; exact prefill 33.3 vs 25.6 s); typically no corrections per 1,024-token block; verify-quality gate passes with VeriCache on |
| Concurrent long-context agents (coherent-device C2+) | Default: a lane takes a request only when its projected L0 codes (input + output allowance, 17,408 bytes per token for INT4, 9,216 for INT2) fit in the free device memory beside the growth the other active lanes still have ahead, keeping `VOIDINFER_EXL3_L0_CODE_RESERVE_MIB` (default 512) free; younger requests wait behind the oldest one that does not fit, a lone lane always runs, and a waiting request runs anyway shortly before its queue deadline; `VOIDINFER_EXL3_L0_CODE_ADMISSION=0` off. The output allowance is `VOIDINFER_EXL3_L0_ADMIT_OUTPUT_TOKENS` (default 4,096), not the whole max_tokens: when free device memory falls below `VOIDINFER_EXL3_L0_SUSPEND_MIB` (default 256) the youngest decoding lane suspends at its next round (leaves batched rounds, device sync, INT4 code/meta rows to pinned host memory, device backing released: `Exl3TextContext::suspend_l0_codes`) and resumes, oldest suspended lane first, once its codes fit with the threshold and the code reserve to spare; the oldest decoding lane never suspends and no request is admitted while a lane is suspended; `VOIDINFER_EXL3_L0_SUSPEND=0` restores whole-max_tokens admission, `VOIDINFER_EXL3_L0_SUSPEND_TRACE=1` logs each suspension. Without it WDDM silently demotes code planes to system memory past physical VRAM | 8 agents x ~106K tokens (849K) on C4, 256 tokens per turn: turns of 218 / 49 / 45 s, peak VRAM 30.0 GiB (previously 266 s for the first turn at 32.1 GiB, then the Engine failed with "growable region access"); C8 x 31K prompts (4.5 GB of codes, more than fits): 185 vs 79-120 tok/s, 42 vs 155-245 ms per lane round |
| Shared tap pool (coherent-device C2+) | Default with the shared prompt workspace: each context (lane or exact line) keeps 32-row tap and embedding-trace planes; prompt, verifier and line passes wider than 32 rows write one model-level 1,024-row pool (claimed per pass, ordered by an event). A pass's last 32 rows stay private; older rows stay readable from the pool until another context claims it (later reads of them fail; root exports and transaction checkpoints keep the rows still held, exports write zeros for the rest — only the draft reads taps, right after the forward that wrote them). `VOIDINFER_EXL3_SHARE_TAPS=0`: private 1,024-row planes; `VOIDINFER_EXL3_TAP_POOL_TRACE=1` logs claims and pooled reads. Layer-major prefill now publishes the final-chunk taps of all five tap layers (previously only GDN bulk-MLP layers did; full-attention tap layers kept an older forward's rows) | C4 with exact lines: 61 -> 3 MiB of taps per context, peak device memory at C4 x 32K outputs 29,842 -> 29,411 MiB; sampled LCB 8 x 6,144 tokens 87.1 / 87.6 / 87.0 vs 86.6 / 88.2 kept tok/s per lane (off), greedy and 30K-prompt C4 runs unchanged |
| Eight-lane batched rounds (coherent-device C5-C8) | Default: one batched-round coordinator for up to eight lanes (`VOIDINFER_EXL3_BATCHED_GROUP_LANES`, 2..8, default 8; 4 = two groups of four): one 64-row forward (`VOIDINFER_EXL3_BATCH_ROWS`=64 set by the Engine: 64-row private small tier, hidden planes, taps and GDN history), M64 trellis producers (four M16 tiles, output in two 32-row passes; `VOIDINFER_EXL3_M64_STAGES` 3 default, FP16 MMA accumulation in batched decode rounds and batched drafts only - C8 sampled 51.8 -> 49.5 ms/round with no more VeriCache corrections, `VOIDINFER_EXL3_M64_FP16_ACC=0` FP32; exact verify, line and prefill passes of 33..64 rows always accumulate in FP32), the head in 32-row passes, batched drafts of up to eight lanes (`VOIDINFER_EXL3_DRAFT_BATCH_LANES`). Graphs are captured for the full group, the groups missing one lane (`VOIDINFER_EXL3_BATCHED_GRAPH_MIN_WIDTH`) and every subset inside lanes 0-3 or 4-7; any other group runs as two half-group replays (eager forwards of 5-6 lanes cost 50-125 ms, a captured eight-lane forward 29 ms; each graph ~14 MiB). `VOIDINFER_EXL3_BATCHED_STATS=1` prints widths/splits, `VOIDINFER_EXL3_BATCHED_FORWARD_TIMING=1` forward ms by width and group closes. Rendezvous wait (`VOIDINFER_EXL3_BATCHED_ROUND_WAIT_US`): 40 ms for groups of more than four lanes, 20 ms otherwise (was 5 ms: late lanes ran narrow split/solo rounds and stayed out of phase) | Sampled LCB, 8 lanes x 32K, 32 requests x 4,096 tokens: 315.2 -> 341.2 tok/s (per lane 43.2 -> 49.6 kept tok/s), 371.5 with the 40 ms wait; C4 8 x 6,144 sampled 87.5 / 86.4 -> 90.7 / 92.8 kept tok/s per lane with 20 ms; greedy 8 x 3,072: 384.6 -> 394.7 tok/s (510.8 with the 40 ms wait), same correction rate; C4 unchanged (87.5 kept tok/s per lane). M64 producer: 15% less GEMM time than two M32 launches (FP32 accumulation, ~53% of the tensor peak at M32); free device memory after startup 3.72 -> 2.85 GiB |
| Concurrency sweep (Oct 15, sampled LCB prompts x 4,096 tokens, 32K context, hot 256) | C1 2 requests, C1 greedy 1, C2 4, C3 6, C4 8, C5-C7 2 per lane, C8 16 | C1 155.6 (greedy 167.9), C2 225.3, C3 273.4, C4 328.2, C5 303.9, C6 338.1, C7 330.8, C8 343.3 tok/s aggregate; C5-C7 one group vs groups of four 277.8 / 299.4 / 290.5. Batched graphs: ~2,000 (2 lanes) to ~5,800 (8 lanes) nodes, ~3 KB device memory per node |
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
| Thinking budgets, stop strings, frequency penalty | Rejected by the Engine (presence penalty is served) | `src/exl3/exl3_engine_core.cpp` | Parsed by CLI/server, refused at submission |
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
