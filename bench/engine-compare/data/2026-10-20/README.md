# Capacity, speed and agent comparison, 2026-10-20/21

VoidInfer `perf/quality-neutral-speed-20261020` (EXL3 6 bpw, L0 OSCAR INT4 KV,
exact VeriCache, DFlash2) vs NInfer a0cf1c17 (NVFP4 weights, FP8 KV), RTX 5090,
96 GB RAM, Windows 11 (WDDM). Harness: `capacity_20261020.py` (phases) and
`capacity_queue_20261020.py` (queue runner); raw rows in `ours.jsonl` / `nv.jsonl`,
limits in `limits.json` (latest build) and `limits_b2.json` (Oct 9 B2 build).

## Max context per lane (prompt actually prefilled and answered)

| Lanes | VoidInfer | NInfer (fill) | NInfer start limit default / slots0 |
|---|---|---|---|
| C1 | 262,144 (258,089 prompt + 4,055 generated, stops at the window) | 208,896 | 177,152 / 182,272 |
| C2 | 192,512 each | 86,016 each | 139,264 / 151,552 |
| C4 | 69,632 each (B2: 65,536) | 26,624 each | 65,536 / 89,088 |
| C8 | 11,264 each (3/3; 12,288 passes 3/5, VRAM-aware lane admission holds the 8th lane) | 3,072 each | 8,192 / 13,312 |

Asymmetric C2 (first lane full, second gets the rest): VoidInfer 262K + 135,168;
NInfer 151,552 + 38,912 (pool 184,320).

## Single request (C1)

| Prompt | VoidInfer TTFT / prefill tok/s / decode tok/s | NInfer |
|---|---|---|
| 3K | 0.87 s / 3.5K / 160-173 | 0.33 s / 9.3K / 182-200 |
| 32K | 8.1 s / 3.9K / 142-146 | 3.1 s / 10.3K / 166-170 |
| 128K | 40.8 s / 3.2K / 161 | 20.9 s / 6.2K / 152 |
| 261K | 106.0 s / 2.5K / 106 | n/a (max 208K) |
| 176K | n/a | 35.0 s / 5.0K / 151 |

## Concurrency (4K prompts, aggregate tok/s greedy / sampled, TTFT)

| | VoidInfer | NInfer |
|---|---|---|
| C1 | 152-180 / 149-152, 0.96 s | 177-214 / 174-189, 0.40 s |
| C2 | 188-230 / 210-230, 1.4-1.9 s | 289-317 / 299-325, 0.6 s |
| C4 | 286-300 / 239-280, 2.7-3.1 s | 497-514 / 467-482, 1.0 s |
| C8 | 328-343 / 262-294, 5.5 s | 689-775 / 673-755, 1.8 s |

Recheck 2026-10-22 (release build v2026.10.22, code_4k, 1,024 tokens, greedy, 2 alternating runs,
wall time including TTFT): C1 143 / 150 vs 190 / 189 tok/s; C2 226 / 230 vs 317 / 342 tok/s
aggregate (VoidInfer vs NInfer).

## Agent switching / resume (100K-token agents, C1, TTFT)

| | Cold A / B | Switch to A | Same agent A | Switch to B |
|---|---|---|---|---|
| VoidInfer | 28.3 / 28.7 s | 0.75 s | 0.51 s | 0.89 s |
| NInfer | 13.3 / 13.9 s | 0.13 s | 0.08 s | 0.26 s |

Decode after resume: VoidInfer 125-136 tok/s, NInfer 170-258 tok/s.
Startup: VoidInfer ~8 s, NInfer 5-7 s (40 s cold file cache).

## 8 agents on 4 lanes (400K chars ~110K tokens each, 3 turns)

| Build | Turn 1 | Turn 2 | Turn 3 | Disk spill |
|---|---|---|---|---|
| before fix (6 runs, incl. B2) | 5-8/8 | 0-8/8 | 0/8 | 0-29 GiB |
| stale-snapshot release only | 8/8 | 8/8 | 0/8 | 9 GiB |
| both fixes (default now) | 8/8 | 8/8 | 8/8 | 4 GiB |

Turn wall times with both fixes: 294 / 266 / 297 s. Physical RAM never fell
below 33.8 GB free; Windows commit fell to 5.0 GB free (limit 109.7 GB = 96 GB
RAM + 14 GB pagefile). The binding limit is commit, not RAM: WDDM charges
~27 GiB of commit for the engine's VRAM allocations (write-combined private
regions, not resident), so ~31 GB is charged at startup before any KV exists.
The spill store (packs in RAM first, spills only when commit is short) wrote
4 GiB; a larger pagefile would raise the commit limit without being written.
`--host-kv-mib 20000` holds ~4 of the 8 histories, so most turns re-prefill
(cache hits 1/16), as in the earlier runs.

Bugs fixed in this round (both default):

1. The coordinator's committed resident union kept every
root it had locked until the next successful commit, including roots the prefix
cache had already evicted. Under host memory pressure every admission failed the
8 GiB reserve check before committing, so ~29 GB of idle plane sets stayed pinned
with an empty cache and every request got 429 ("exceeds available host memory").
Now a refused admission first commits the union without the update (only drops
pages), releases idle plane rows and retries, before evicting cached roots.
2. The resident set refused any page lock while available memory was below the
reserve, even when every page was already resident: an admitted root lives in the
planes its prefill just filled, so locking it commits nothing. Every request then
prefilled for ~36 s and got 429 without freeing anything. The reserve check now
applies only when pages still have to be faulted in.
`VOIDINFER_EXL3_VERICACHE_STATS=1` prints a `HOST_LEDGER` breakdown on refusals.
Disk spill (`%TEMP%\voidinfer-kv-spill-<pid>.bin`, delete-on-close) wrote 0 bytes
in the C1-C8 capacity runs; only the 8-agent run spilled (above).

## Vision (vision_compare_20261021.py, C2 servers, greedy, 256 tokens)

Both engines answer all 8 requests (2 single images, two images in one turn, an image inside a
3.5K-token prompt, a 1,164-token video, text after media, 2 concurrent image requests) with the
same content: chart title, red-circle counts and the star, the first function in the code, the
video's phase sequence. Identical prompt token counts (404 / 411 / 505 / 3,506 / 1,164).

| Request | VoidInfer (s) | NInfer (s) |
|---|---|---|
| scene image | 1.96 | 1.48 |
| chart image | 1.84 | 1.21 |
| two images | 0.96 | 0.81 |
| image in 3.5K prompt | 1.59 | 1.03 |
| video | 1.78 | 1.38 |
| 2 concurrent images | 2.59 s, 198 tok/s, identical to serial | 1.42 s, 360 tok/s, differs from serial |
| startup with vision | 18.6 s | 33.6 s |

VoidInfer's BF16 tower (weights in pinned host memory, streamed to the GPU) against the Hugging
Face FP32 tower: cosine mean 0.99980 / 0.99993, min 0.9909 / 0.9976, rel L2 0.045 / 0.018 (scene /
chart); HF's own BF16 tower: 0.99990 / 0.99994, rel L2 0.025 / 0.026 (`vision/ref.txt`).

Not run: NInfer 8 agents on 2 lanes.

Notes: the numbers are single runs unless a range is given; ranges are repeats.
VoidInfer stays at 6 bpw / L0 OSCAR / exact VeriCache (the higher-quality path:
KL 0.014 vs NInfer 0.08, see 2026-10-12 notes); NInfer is faster on every speed axis.
