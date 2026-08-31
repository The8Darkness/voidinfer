# Single-GPU serving performance

## Hierarchical VeriCache experiment

The isolated branch `exp/hierarchical-vericache-20260830` makes the hierarchical VeriCache path the
default `ninfer-serve` profile for Qwen3.8-27B DFlash2/NVFP4. The underlying Engine option remains
available as an opt-in API, and `--no-spec --no-hierarchical-vericache` restores the stable
fallback. It treats compressed L0 errors as speculative rejection effects and retains the exact
target as the correctness fallback. The current implementation has nested
KV/GDN transactions, adaptive L0→L1 (24–64) and L1→L2 (256–2,048) controls, direct NVFP4 attention,
a protected recent-token sidecar, and a typed host-KV/StateImage checkpoint path shared by MTP and
DFlash. Existing pinned StateImage and host KV stores are used for truthful L1/L2 residency and
transfer accounting; they are not yet an independent host output verifier.

The latest repeated-boundary stress run uses the Qwen3.8-27B NVFP4 DFlash2 artifact on the physical
RTX 5090, batch 1, context 512, `k=7`, optimized proposal head, no CUDA Graph, one warmup round,
and 400 measured rounds. The `l1=256` row intentionally forces three host checkpoint boundaries
within this short run. Published throughput is licensed tokens divided by wall time, so it is
acceptance-sensitive and must not be read as a kernel-only speedup:

The server default is VeriCache-NVFP4 + DFlash2 `k=7` + optimized proposal head with host-tier
snapshots enabled. The direct round harnesses still pass explicit flags so control and experimental
rows remain reproducible.

| Configuration | GPU round | Wall round | Published tok/s | Draft acceptance | L0/L1/L2/L3 bytes |
| --- | ---: | ---: | ---: | ---: | --- |
| VeriCache-NVFP4 control, hierarchy disabled | 22.9920 ms | 23.0041 ms | 78.2470 | 320/2,800 (11.43%) | 0/0/0/0 |
| Hierarchy enabled, no host snapshot | 22.7066 ms | 22.7177 ms | 73.6212 | 269/2,800 (9.61%) | 26,869,760/0/0/0 |
| Hierarchy + host snapshot, L1→L2 fixed at 256 | 22.6889 ms | 22.6995 ms | 70.1556 | 237/2,800 (8.46%) | 26,869,760/17,367,040/221,063,168/0 |

The host row materialized three checkpoints and measured 502,167,552 StateImage bytes plus
71,565,312 KV bytes across 33 KV pages; synchronized KV D2H time was 1.94 ms in aggregate at
that revision. The GPU round was 1.32% below the control, but the measured acceptance was lower,
so the combined published rate was 10.3% below the control. This is not a multiplicative claim:
repeated runs show acceptance sensitivity. The row is retained as a pre-overlap reference; the
follow-up below changes only the host-promotion synchronization/ownership path.

### E018: event-ordered host promotion follow-up

The follow-up uses the same physical RTX 5090, DFlash2 `k=7`, optimized proposal head, no CUDA
Graph, context 512, one warmup round, and 100 measured rounds, but exercises two concurrent lanes.
Host KV extents remain reserved and invisible until the transfer completion event; superseded KV
descriptors are released only after DMA drains. This validates overlap and cleanup, not an
independent host-tier output verifier:

| Configuration | GPU round | Wall round | Published tok/s | Draft acceptance | Host transfer telemetry |
| --- | ---: | ---: | ---: | ---: | --- |
| VeriCache-NVFP4 control, hierarchy disabled | 23.6857 ms | 23.6995 ms | 186.502 | 242/1,400 (17.29%) | none |
| Hierarchy + async host snapshot | 23.6491 ms | 23.6622 ms | 182.992 | 233/1,400 (16.64%) | 334,778,368 StateImage D2H + 71,303,168 KV D2H bytes / 32 pages; KV DMA 0.148862 s |

The async row is 0.15% lower in GPU round time but 1.88% lower in aggregate published tok/s due
to its lower acceptance in this pair. It is therefore retained as infrastructure with no speedup
claim. The host output verifier, low-bit direct verifier, NVMe persistence, and long-context
quality gates remain separate research work.

### E019: DFlash2 selector working-set follow-up

The DFlash2 selector lattice carries only 16 candidate IDs and a 16×16 predecessor-conditioned
score matrix. The production caller now allocates the semantic width of 272 FP32 values per token
instead of the drafter hidden width of 5,120, and the sequential path trace launches one participating
thread instead of a 32-thread CTA. The public op still accepts larger padded widths for compatibility.
The exact-width selector and trace cases pass with guard checks.

The physical RTX 5090 comparison used the same Qwen3.8-27B NVFP4 DFlash2 artifact, context 512,
`k=7`, optimized proposal head, CUDA Graphs, one warmup round, and 200 measured rounds:

| Configuration | GPU round | Wall round | Published tok/s | Draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| VeriCache-NVFP4 control, hierarchy disabled | 21.7685 ms | 21.7784 ms | 79.4367 | 146/1,400 (10.43%) |
| Hierarchy + host snapshot, trimmed selector working set | 21.7017 ms | 21.7124 ms | 96.0281 | 217/1,400 (15.50%) |

The hierarchy row is 0.31% lower in round GPU time, but the published-rate difference is dominated
by acceptance variation. The change is retained for the measured memory/launch reduction, not as an
independent end-to-end speedup claim.

### E020: graph-safe DFlash2 capture

The selector trace now runs as a deterministic single-lane device operation, which removes the
last graph-safety reason for forcing DFlash2 eager. The existing graph profiles capture the full
proposal/selector/verify transaction; focused DFlash2 and hierarchical correctness tests pass.
The following physical RTX 5090 A/B runs use the Qwen3.8-27B NVFP4 DFlash2 artifact, context 512,
`k=7`, optimized proposal head, VeriCache-NVFP4, host snapshots, one warmup round, and 100 measured
rounds. The graph and eager rows were run with identical workload and hierarchy settings:

| Batch | Graph mode | GPU round | Wall round | Published tok/s | Draft acceptance |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | enabled | 19.6386 ms | 19.6470 ms | 102.306 | 101/700 (14.43%) |
| 1 | eager control | 21.3438 ms | 21.3524 ms | 94.1346 | 101/700 (14.43%) |
| 2 | enabled | 21.8301 ms | 21.8393 ms | 201.013 | 239/1,400 (17.07%) |
| 2 | eager control | 23.6388 ms | 23.6489 ms | 184.364 | 236/1,400 (16.86%) |

Graph mode lowers steady GPU round latency by 7.98% at C=1 and 7.65% at C=2. Published
throughput rises 8.68% and 9.04%, respectively; because the C=2 acceptance differs by three
tokens, the round-latency result is the cleaner attribution. The default-serving smoke also
passed the full text HTTP contract after graph capture, reporting DFlash2 `k=7`, hierarchical
VeriCache, host snapshots, and the `nvfp4-dflash2` cost profile. L0→L1/L1→L2 host-verifier counters
remain zero; these results validate graph execution around the existing exact-target verifier,
not authoritative host-tier output equality.

### E021: immutable hierarchical host-checkpoint manifest

The host-promotion path now retains the inactive COW KV address spaces that identify the exact
checkpoint prefix, records the frozen StateImage content epoch, and validates that the two are
published together only when every required page has a current host replica with sufficient
committed coverage. The first manifest is reaped and validated before the next one replaces it;
this makes host-tier ownership explicit without pretending that host storage is already an
independent logits verifier.

The guarded physical RTX 5090 stress run used the Qwen3.8-27B NVFP4 DFlash2 artifact, context
1,024, C=1, graph mode, `k=7`, optimized proposal head, VeriCache-NVFP4, host snapshots, one
warmup round, and 300 measured rounds:

| GPU round | Wall round | Published tok/s | Draft acceptance | Host checkpoints | Host transfer telemetry |
| ---: | ---: | ---: | ---: | ---: | --- |
| 22.1042 ms | 22.1153 ms | 89.9829 | 297/2,100 (14.14%) | 2 | 334,778,368 StateImage D2H bytes + 111,149,056 KV D2H bytes / 49 pages |

The run crossed the replacement boundary without an ownership, stale-replica, or GDN-epoch
failure. It is a structural correctness/residency milestone; L0→L1/L1→L2 host-verifier counters
remain zero and no independent host-tier output-equality or end-to-end speedup claim is made.

For historical comparison, an earlier short RTX 5090 run used batch 1, context 2,048, DFlash2
`k=7`, no CUDA Graph, three warmup rounds, and eight measured rounds. The published rate is
licensed tokens divided by wall time for this fixed round harness; acceptance can therefore
dominate it even when the GPU round is faster:

| L0 configuration | GPU round | Wall round | Published tok/s | Draft acceptance | Tier bytes (L0/L1/L2/L3) |
| --- | ---: | ---: | ---: | ---: | --- |
| NVFP4 control, hierarchy disabled | 28.3265 ms | 28.3390 ms | 83.8067 | 11/56 (19.64%) | not reported |
| NVFP4 + 64 recent BF16 + 8 anchor tokens | 27.6280 ms | 27.6389 ms | 81.4069 | 10/56 (17.86%) | 26,869,760/0/0/0 |
| Same hierarchy + async host snapshot | 27.5497 ms | 27.6001 ms | 81.5213 | 10/56 (17.86%) | 26,869,760/13,434,880/153,954,304/0 |
| NVFP4 + 64 recent BF16, no anchors | 26.0720 ms | 26.0851 ms | 91.0481 | 11/56 (19.64%) | 26,214,400/0/0/0 |

The protected sidecar remains an opt-in quality-protection mechanism, not a claimed speedup. The
anchor configuration reduced GPU round time versus that control, but accepted one fewer draft
token in this run and therefore published fewer tokens per second. The no-anchor row is a useful
isolating measurement, not a replacement for the required sink/pivot quality tests. Exact-target
fallback recorded 11 checks, 77 proposed tokens, 15 accepted draft tokens, and 11 disagreements;
the adaptive L0→L1 horizon contracted to 24. The host-snapshot row materialized one asynchronous
StateImage promotion (167,389,184 bytes total) and preserved the same observed acceptance as its
anchor-only counterpart, but it did not independently verify host-tier logits. Real L0→L1/L1→L2
verification, NVFP4/FP8 output equality, NVMe persistence, greedy/sampling quality, vision, and
262K/multi-agent workload matrices remain pending. The MTP benchmark now has matching
VeriCache/hierarchy flags, but its direct-package Qwen3.8 probe stalled after allocating the
model and was terminated without a performance result; no MTP claim is made from that probe.

Tested Git revisions:

- Qwen3.8-27B NVFP4 MTP0 context-length serving:
  `f08597d6eaafce5b875934aaa85854fcd5426df8`;
- Qwen3.8-27B NVFP4 MTP3 single-request and concurrent fixed-corpus serving:
  `32c9881b6783949df4999422a764b3dcaa111b13`;
- Concurrent MTP3 decode saturation for the three measured Qwen3.6 artifact profiles:
  `26da9df7c1b3d3c04ea7bbd730271aa01d00742a`;
- Refreshed Qwen3.6-35B-A3B and Qwen3.6-27B NVFP4 MTP3:
  `f4f21cc36bd1a83cbc046f668719d591dc9c1e2e`;
- Qwen3.6-35B-A3B stored MTP3 response audit:
  `b1a220f028aa750f75bceb3522ac00bbaab7e42d`;
- Qwen3.6-35B-A3B DFlash block=8 (`k=7`):
  `0dc94097e8ec5c5bcf59b9e13e9d1852f504eb61`;
- Qwen3.6-27B NVFP4 accuracy and MTP0:
  `b3d4d0f50b868711c62432bbd68e746217a2f49a`;
- Qwen3.6-27B groupwise-int MTP3: `5ea3242a206cdb0c4c1beaeb9d8a3048e6248423`;
- Qwen3.6-35B-A3B MTP0 and Qwen3.6-27B groupwise-int MTP0:
  `0795169393cab0f2c16246d4bac20dee735dc2a4`.

The Qwen3.6 measurements characterize its three registered artifact profiles independently on one
NVIDIA GeForce RTX 5090. They cover long-context prefill and baseline decode with speculative
decoding disabled, plus long-reasoning and cross-scenario decode with MTP and DFlash. The Qwen3.6
concurrent decode-saturation campaign measures all three profiles at C=1, 2, 4, and 8. The
Qwen3.8-27B NVFP4 campaign covers the MTP0 long-context profile and the complete MTP3
speculative-decode corpus at C=1, 2, 4, and 8; its C=1 point also supplies the single-request MTP3
results below. The registered Qwen3.8-27B `groupwise-int` profile remains outside the published
benchmark campaign.

The single-request corpus requests were submitted serially to a persistent `ninfer-serve` process
over the loopback OpenAI-compatible HTTP endpoint. Each reported corpus fixture used five fixed
seeds. Values are arithmetic mean ± sample standard deviation, and server warm-up completes before
the measured requests. The concurrent campaign has its own sustained-wave method below.

## Single-request serving performance method

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32 GiB |
| CUDA compile/runtime | 13.1 / 13.1 |
| CUDA driver API | 13.3 for NVFP4 and refreshed 35B MTP3; 13.1 for the remaining single-request campaigns |
| Request mode | One active request, `stream=false` |
| Maximum context | 262,144 tokens; 131,072 for refreshed NVFP4 MTP3 |
| Prefill chunk | 1,024 tokens |
| KV cache | INT8 group-64 |
| CUDA Graph | Enabled |
| Prefix reuse | Disabled |
| Sampling | Temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0 |
| Greedy profile | Exact argmax (`--sampling greedy` in the corpus runner) |
| MTP0 | no `--spec` |
| MTP3 | `--spec mtp --draft-tokens 3 --lm-head-draft` |
| DFlash block=8 | `--spec dflash --draft-tokens 7 --lm-head-draft` |

The MTP0 profile uses four Long NIAH prompts with approximately 8K, 64K, 128K, and 256K tokens.
Thinking is disabled and the output budget is 128 tokens. These runs measure prefill throughput,
server-internal time to first token, and baseline decode throughput at each context length. Content
scenarios are not repeated with MTP disabled because they do not change the baseline decode path.

The speculative-decode corpus contains three long-reasoning fixtures with thinking enabled and a
65,536-token output limit, followed by twelve fixtures covering code, story, translation, and
structured output. The cross-scenario fixtures disable thinking and use a 4,096-token output limit.
The tables report actual completion lengths rather than assuming that every request reaches its
limit.

Metrics are computed from the server's unrounded phase timings and speculative-decode counters:

```text
prefill_tok_s = prompt_tokens / prefill_seconds
server_ttft_ms = 1000 * (prepare_seconds + vision_seconds + prefill_seconds)
decode_tok_s = (completion_tokens - 1) / decode_seconds
spec_acceptance = accepted_tokens / drafted_tokens
spec_tokens_per_round = 1 + accepted_tokens / speculative_rounds
```

Decode throughput is a transport/execution measurement, not a correctness score. The response text,
finish reason, and fixture-level structural requirements are audited separately below. A request
that exhausts its output budget or enters a repetition loop remains useful as a sustained-decode
stress sample, but is not presented as a successfully completed task.

## Qwen3.8-27B NVFP4 concurrent MTP3 corpus makespan

This campaign uses the complete speculative-decode corpus described above: three long-reasoning
fixtures and twelve cross-scenario fixtures, each with five fixed seeds, for 75 requests. The
runner shuffles that fixed request set once with seed `20260811` and preserves the same ordered HTTP
send sequence at every concurrency. Exactly C persistent client workers each submit their next
request only after receiving the current response. C=1 is therefore a serial single-request corpus
on one persistent server and supplies the per-fixture Qwen3.8 results in the final section.

Each point starts a fresh server on an RTX 5090 with CUDA 13.1 compile/runtime, CUDA driver API
13.3, stochastic sampling, INT8 group-64 KV, a 1,024-token prefill chunk, CUDA Graphs, prefix reuse
disabled, a 131,072-token per-request context ceiling, `--kv-capacity auto`, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Makespan begins when all client workers are released
and ends when the final complete HTTP response has been read. Prefill and decode rates divide the
corresponding server token totals by that full makespan; average batch includes the entire run,
including workload transitions and drain.

| C | Requests | Computed prefill tokens | Decode tokens | Makespan (s) | Requests/s | Prefill tok/s | Decode tok/s | Avg batch | MTP acceptance | Speedup vs. C1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75 | 15,460 | 752,160 | 4,670.27 | 0.0161 | 3.3 | 161.1 | 1.00 | 60.8% | 1.00× |
| 2 | 75 | 15,460 | 739,951 | 2,510.78 | 0.0299 | 6.2 | 294.7 | 1.98 | 59.2% | 1.86× |
| 4 | 75 | 15,460 | 713,384 | 1,647.74 | 0.0455 | 9.4 | 432.9 | 3.29 | 58.0% | 2.83× |
| 8 | 75 | 15,460 | 723,602 | 2,164.90 | 0.0346 | 7.1 | 334.2 | 2.36 | 57.6% | 2.16× |

All 300 requests completed without a request, CUDA, or out-of-memory failure. C=4 gives the
shortest complete-corpus makespan. C=8 is limited by memory pressure, which constrains effective
batching and makes the end-to-end result slower than C=4. Sampling is stochastic: prompts, seeds,
and send order are fixed, but concurrency-specific numerical routes can change sampled
continuations and their lengths. The makespan speedup is therefore a fixed-workload serving result
rather than a fixed-token normalization; the exact decode-token totals are retained in the table.

## Concurrent MTP3 decode saturation

The concurrent campaign uses the `long_decode_aime26_15` fixture with thinking enabled. The
rendered prompt is 293 tokens, and every request has an 8,192-token output budget. For each
concurrency C, the runner starts a fresh `ninfer-serve` process with `max_concurrency=C`, releases
C non-stream requests together using distinct fixed seeds, and waits for every HTTP response.
Startup and server warmup occur before the measured wave.

All points use an RTX 5090, CUDA 13.1 compile/runtime, CUDA driver API 13.3, stochastic sampling
(temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.0), INT8 group-64 KV, a 1,024-token
prefill chunk, CUDA Graphs, prefix reuse disabled, and
`--spec mtp --draft-tokens 3 --lm-head-draft`. Each request has a 16,384-token context ceiling.
`--kv-capacity auto` resolved to exactly `C * 16,384` tokens at every point.

Saturated throughput uses only complete one-second server intervals satisfying all of the following:

- computed prefill tokens are zero;
- `running=C`, `prefilling=0`, and `decode_ready=C`;
- at least one decode round completed;
- every decode round had exactly C rows.

Ramp-up, prefill, and drain intervals are excluded. The reported aggregate rate is:

```text
steady_decode_tok_s = sum(committed_decode_tokens) / sum(interval_seconds)
```

Wave makespan starts when the client threads are released and ends after the last complete HTTP
response. MTP acceptance is aggregated over the complete wave. Each row below is one sustained
wave rather than a repeated-sample mean.

| Model profile | C | Steady (s) | Avg batch | Aggregate decode tok/s | MTP acceptance | Speedup vs. C1 | Wave makespan (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 1 | 43.01 | 1.00 | 185.8 | 68.2% | 1.00× | 44.23 |
| Qwen3.6-27B `groupwise-int` | 2 | 65.01 | 2.00 | 247.0 | 69.0% | 1.33× | 66.67 |
| Qwen3.6-27B `groupwise-int` | 4 | 102.02 | 4.00 | 309.5 | 68.4% | 1.67× | 107.49 |
| Qwen3.6-27B `groupwise-int` | 8 | 118.02 | 8.00 | 535.0 | 68.3% | 2.88× | 125.20 |
| Qwen3.6-27B `nvfp4` | 1 | 39.01 | 1.00 | 202.4 | 69.3% | 1.00× | 40.46 |
| Qwen3.6-27B `nvfp4` | 2 | 39.01 | 2.00 | 399.7 | 71.4% | 1.97× | 41.82 |
| Qwen3.6-27B `nvfp4` | 4 | 44.01 | 4.00 | 699.7 | 69.3% | 3.46× | 47.92 |
| Qwen3.6-27B `nvfp4` | 8 | 55.01 | 8.00 | 1,146.9 | 68.6% | 5.67× | 58.57 |
| Qwen3.6-35B-A3B `groupwise-int` | 1 | 12.00 | 1.00 | 593.0 | 67.2% | 1.00× | 13.75 |
| Qwen3.6-35B-A3B `groupwise-int` | 2 | 17.00 | 2.00 | 877.7 | 68.2% | 1.48× | 18.87 |
| Qwen3.6-35B-A3B `groupwise-int` | 4 | 26.01 | 4.00 | 1,166.0 | 69.8% | 1.97× | 28.43 |
| Qwen3.6-35B-A3B `groupwise-int` | 8 | 48.01 | 8.00 | 1,313.8 | 67.3% | 2.22× | 50.20 |

All 45 requests reached their output limit, producing 368,640 completion tokens. The campaign
contained 608 complete full-batch steady intervals and had no request, CUDA, or out-of-memory
failure. At C=8, available device memory after startup was 2.66 GiB for 27B groupwise-int,
2.18 GiB for 27B NVFP4, and 4.38 GiB for 35B-A3B.

## Reproduction

Build `ninfer-serve` and prepare the registered `.ninfer` artifacts. The refreshed per-target
serving tables use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 262144 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_35b_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 \
  --output profiles/bench/serve_corpus_27b_mtp3_20260724

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_27b_nvfp4_w8_20260731

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan --concurrency 1 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp0 --sampling stochastic \
  --output profiles/bench/serve_corpus_qwen3_8_27b_nvfp4_mtp0_20260817

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_8_27b=out/qwen3_8_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --max-context 131072 --kv-capacity auto \
  --output profiles/bench/concurrent_corpus_qwen3_8_27b_nvfp4_mtp3_20260817
```

The concurrent decode-saturation campaigns use:

```bash
python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_27b_nvfp4_mtp3_20260811

python3 tools/bench/run_serve_concurrency.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 --concurrency 8 \
  --decode-tokens 8192 --max-context 16384 --kv-capacity auto \
  --output profiles/bench/concurrent_decode_35b_mtp3_20260811
```

Use `--mode dflash7` for the corresponding DFlash block=8 campaign; add `--sampling greedy` for
the exact-argmax profile.

Omit `--mode` and supply the two measured Qwen3.6 groupwise-int artifacts to run the complete
published Qwen3.6 MTP0/MTP3 campaign:

```bash
python3 tools/bench/run_serve_corpus.py \
  --serve build/apps/ninfer-serve \
  --artifact qwen3_6_35b_a3b=out/qwen3_6_35b_a3b.ninfer \
  --artifact qwen3_6_27b=out/qwen3_6_27b.ninfer \
  --output profiles/bench/serve_corpus_20260720
```

For the 27B NVFP4 accuracy run, start the model service with:

```bash
build/apps/ninfer-serve out/qwen3_6_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18080 \
  --max-context 262144 --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

Then run the repository's full 27B reasoning suite in a separate shell:

```bash
PYTHONPATH=eval eval/.venv/bin/python -m ninfer_eval run \
  --config eval/configs/qwen3_6_27b_reasoning.yaml \
  --suite reasoning_full
```

## `qwen3_6_35b_a3b`

### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 15,544.3 ± 242.4 | 500.2 ± 7.8 | 271.1 ± 3.6 |
| 64,512 | 5 | 10,809.0 ± 95.3 | 6,009.9 ± 52.6 | 242.9 ± 1.3 |
| 130,048 | 5 | 7,828.4 ± 34.1 | 16,693.3 ± 71.2 | 219.4 ± 1.6 |
| 260,096 | 5 | 5,157.1 ± 52.4 | 50,598.8 ± 519.7 | 188.2 ± 2.1 |

### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,223.0 ± 2,224.1 | 726.2 ± 22.9 | 82.8% ± 3.4% | 3.48 ± 0.10 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 620.3 ± 8.1 | 72.7% ± 1.4% | 3.18 ± 0.04 |
| `long_decode_aime26_30` | 5 | 52,977.8 ± 11,849.6 | 671.9 ± 8.8 | 80.1% ± 2.7% | 3.40 ± 0.08 |

### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 657.6 ± 34.3 | 70.3% ± 5.5% | 3.11 ± 0.16 |
| Story | 15 | 456.2 ± 36.6 | 38.0% ± 6.0% | 2.14 ± 0.18 |
| Translation | 15 | 649.7 ± 33.0 | 67.6% ± 5.1% | 3.03 ± 0.15 |
| Structured | 15 | 770.9 ± 29.3 | 89.1% ± 4.9% | 3.67 ± 0.15 |

### DFlash block=8 (`k=7`), stochastic sampling

The fixtures, five seeds, sampling parameters, and output limits are identical to MTP3. Different
speculative backends consume random values differently, so this is a fixed-workload comparison
rather than a token-identical paired-output comparison.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 8,495.4 ± 2,221.2 | 764.1 ± 55.6 | 65.2% ± 5.4% | 5.56 ± 0.38 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 584.0 ± 33.3 | 51.1% ± 3.7% | 4.58 ± 0.26 |
| `long_decode_aime26_30` | 5 | 53,330.4 ± 11,198.5 | 638.3 ± 15.8 | 56.4% ± 2.5% | 4.95 ± 0.17 |

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 562.3 ± 36.2 | 43.0% ± 3.7% | 4.01 ± 0.26 |
| Story | 15 | 261.7 ± 51.1 | 12.1% ± 5.3% | 1.85 ± 0.37 |
| Translation | 15 | 490.8 ± 62.6 | 34.8% ± 6.3% | 3.44 ± 0.44 |
| Structured | 15 | 786.4 ± 124.7 | 66.5% ± 13.5% | 5.66 ± 0.94 |

#### Decode throughput versus MTP3

| Workload | MTP3 tok/s | DFlash tok/s | DFlash change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 726.2 | 764.1 | +5.2% |
| `long_decode_aime26_15` | 620.3 | 584.0 | -5.9% |
| `long_decode_aime26_30` | 671.9 | 638.3 | -5.0% |
| Code | 657.6 | 562.3 | -14.5% |
| Story | 456.2 | 261.7 | -42.6% |
| Translation | 649.7 | 490.8 | -24.5% |
| Structured | 770.9 | 786.4 | +2.0% |

### DFlash block=8 (`k=7`), greedy sampling

Greedy uses exact argmax; all other corpus and server settings remain unchanged. The five seeds
repeat the same deterministic generation path, so within-fixture standard deviation measures
runtime variation rather than output variation.

#### Long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 6,692.0 ± 0.0 | 872.4 ± 3.3 | 74.4% ± 0.0% | 6.21 ± 0.00 |
| `long_decode_aime26_15` | 5 | 65,536.0 ± 0.0 | 651.6 ± 0.6 | 58.6% ± 0.0% | 5.10 ± 0.00 |
| `long_decode_aime26_30` | 5 | 65,536.0 ± 0.0 | 994.9 ± 3.4 † | 98.0% ± 0.0% | 7.86 ± 0.00 |

† The generation is a deterministic repetition loop, not a valid AIME response. The raw rate is
retained to describe what was measured, but is excluded from performance comparisons.

#### Cross-scenario decode

| Category | Samples | Decode tok/s | DFlash acceptance | DFlash tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 599.8 ± 12.3 | 46.4% ± 1.4% | 4.25 ± 0.10 |
| Story | 15 | 291.5 ± 55.6 | 14.9% ± 5.7% | 2.04 ± 0.40 |
| Translation | 15 | 475.5 ± 50.6 | 33.0% ± 5.1% | 3.31 ± 0.36 |
| Structured | 15 | 869.0 ± 120.2 | 74.5% ± 13.1% | 6.21 ± 0.92 |

#### Decode throughput versus stochastic DFlash

| Workload | Stochastic tok/s | Greedy tok/s | Greedy change |
|---|---:|---:|---:|
| `long_decode_aime26_01` | 764.1 | 872.4 | +14.2% |
| `long_decode_aime26_15` | 584.0 | 651.6 | +11.6% |
| `long_decode_aime26_30` | 638.3 | 994.9 † | not comparable † |
| Code | 562.3 | 599.8 | +6.7% |
| Story | 261.7 | 291.5 | +11.4% |
| Translation | 490.8 | 475.5 | -3.1% |
| Structured | 786.4 | 869.0 | +10.5% |

### Speculative-decode output audit

The audit covers all 225 stored July responses from the 35B-A3B MTP3 stochastic-sampler, DFlash
stochastic-sampler, and DFlash greedy campaigns. It checks termination, exact repetition, and
fixture-specific mechanical constraints. AIME 1 was checked algebraically; the AIME 30 answer
(`393`) was checked by independent enumeration. This audit does not attempt to assign a subjective
quality score to prose or translations.

#### Long-reasoning answers

| Fixture | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| `long_decode_aime26_01` | 5/5 correct, natural stop | 5/5 correct, natural stop | 5/5 correct, natural stop |
| `long_decode_aime26_15` | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit | 0/5 answers; all reach 65,536-token limit |
| `long_decode_aime26_30` | 3/5 correct, 1 wrong, 1 no answer | 2/5 correct, 1 wrong, 2 no answer | 0/5 answers; all enter the same repetition loop |

The greedy AIME 30 response has an empty final-content field and fills its 65,536-token reasoning
budget. The exact line `Wait, $x_7 x_1 x_3$ is $x_7 x_1 x_3$.` occurs 2,406 times among 2,538
non-empty reasoning lines. Its 98.0% acceptance and 994.9 tok/s therefore characterize a highly
predictable pathological loop, not normal reasoning performance.

AIME 15 is also not a valid completion in any of the three campaigns: every sample exhausts the
budget without a boxed answer. Its output is long, non-convergent reasoning rather than the short
exact cycle seen in greedy AIME 30. The AIME 15 rates may be read only as sustained long-decode
throughput.

#### Cross-scenario outputs

| Category | MTP3 stochastic sampler | DFlash stochastic sampler | DFlash greedy |
|---|---|---|---|
| Code | 1/15 natural stops; 0/15 prompt-complete | 2/15 natural stops; 0/15 prompt-complete | 0/15 natural stops |
| Story | 9/15 natural stops; the nine Chinese outputs pass requested division and minimum length | 8/15 natural stops; the eight Chinese outputs pass requested division and minimum length | 10/15 natural stops; five Chinese dialogue outputs are under length |
| Translation | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks | 15/15 natural stops; 15/15 pass structural checks |
| Structured | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract | 0/15 satisfy the requested complete record/script contract |

The code prompts require complete runnable multi-file deliverables, but almost all outputs end at the
4,096-token limit. The three natural-stop exceptions also contain decisive contract failures: the
MTP3 CUDA response substitutes CUDA 12.8 and an older architecture list; the DFlash CUDA response
copies FP32 input into a half-sized 16-bit allocation and passes raw `unsigned short` values to BF16
intrinsics; and the DFlash Python response never writes its advertised JSONL event stream to the
configured log file. Code throughput is therefore a truncated-generation stress result, not
successful code-generation throughput.

All English mystery samples reach the output limit with an unfinished ending. The naturally stopped
Chinese stories have the requested chapter/act counts; the MTP3 and stochastic-DFlash samples also
meet their requested Chinese-character minima. Greedy's five dialogue stories contain 3,239 Chinese
characters each, below the requested 3,500. Story results are consequently a mixed normal/truncated
workload.

All translation outputs stop naturally. Each plain-document result preserves six sections and
provides at least twenty glossary entries; each Markdown result preserves heading levels, the
six-line table, all required inline identifiers, and the exact fenced JSON object. Translation is
the cleanest cross-scenario normal-completion comparison in this corpus.

The structured prompts intentionally exceed what these generations fit into 4,096 tokens. MTP3,
stochastic DFlash, and greedy DFlash produce only 49–60, 49–58, and 57 valid JSONL records,
respectively, versus the requested 160. Their complete-width CSV ranges are 122–139, 121–143, and
133 rows versus the requested 220. No SQL output satisfies all four tables, two views, at least 80
rows, and six final analytical queries. These high-acceptance results describe predictable partial
record generation only.

The exact-line and repeated-token scan found no other response with a short-cycle collapse comparable
to greedy AIME 30. Output-limit and prompt-compliance failures above remain material even when no
repetition loop is present.

## `qwen3_6_27b`

### EvalScope reasoning accuracy

Both weight profiles were evaluated through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and a 262,144-token context limit. EvalScope 1.9.0 used 0-shot prompts, rule-based
scoring, and one sample per problem with temperature 0.6, top-p 0.95, top-k 20, presence penalty
1.0, and seed 42. All 258 samples completed and were scored for each profile.

| Weights ID | AIME 2025 | AIME 2026 | GPQA-Diamond |
|---|---:|---:|---:|
| `groupwise-int` | 86.67% (26 / 30) | 93.33% (28 / 30) | 86.87% (172 / 198) |
| `nvfp4` | 93.33% (28 / 30) | 93.33% (28 / 30) | 84.34% (167 / 198) |

These are single-sample results under the stated evaluation profile, not pass@k scores. Each
benchmark remains independently reportable; no combined score is computed.

### `groupwise-int`

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 3,218.1 ± 4.3 | 2,392.4 ± 3.0 | 77.6 ± 0.1 |
| 64,512 | 5 | 2,655.9 ± 2.9 | 24,335.7 ± 25.2 | 70.7 ± 0.1 |
| 130,048 | 5 | 2,185.3 ± 0.3 | 59,590.3 ± 8.9 | 64.5 ± 0.1 |
| 260,096 | 5 | 1,614.8 ± 0.6 | 161,221.8 ± 62.5 | 54.8 ± 0.1 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 10,686.2 ± 553.8 | 175.4 ± 1.0 | 77.9% ± 0.9% | 3.34 ± 0.03 |
| `long_decode_aime26_15` | 5 | 61,604.2 ± 5,677.9 | 161.9 ± 2.8 | 73.4% ± 1.7% | 3.20 ± 0.05 |
| `long_decode_aime26_30` | 5 | 47,339.8 ± 9,162.2 | 172.2 ± 0.9 | 78.8% ± 0.8% | 3.36 ± 0.02 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 167.0 ± 5.4 | 72.3% ± 3.5% | 3.17 ± 0.11 |
| Story | 15 | 112.6 ± 9.4 | 37.8% ± 5.9% | 2.13 ± 0.18 |
| Translation | 15 | 161.5 ± 11.3 | 68.3% ± 7.2% | 3.05 ± 0.22 |
| Structured | 15 | 193.0 ± 18.8 | 88.7% ± 11.7% | 3.66 ± 0.35 |

### `nvfp4`

The fixtures, seeds, sampling parameters, output limits, and runtime options are identical to the
groupwise-int serving campaign. Quantization can change sampled tokens, so the MTP3 results are a
fixed-workload comparison rather than a token-identical output comparison.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 11,191.5 ± 70.2 | 692.5 ± 4.3 | 86.4 ± 0.5 |
| 64,512 | 5 | 6,298.5 ± 97.6 | 10,288.6 ± 159.3 | 78.0 ± 1.2 |
| 130,048 | 5 | 4,204.7 ± 14.1 | 31,012.5 ± 104.6 | 71.2 ± 0.2 |
| 260,096 | 5 | 2,510.6 ± 16.8 | 103,761.1 ± 698.8 | 59.9 ± 0.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 12,053.4 ± 820.9 | 231.0 ± 3.0 | 80.2% ± 1.2% | 3.41 ± 0.04 |
| `long_decode_aime26_15` | 5 | 63,109.0 ± 5,426.9 | 213.1 ± 4.2 | 76.3% ± 2.0% | 3.29 ± 0.06 |
| `long_decode_aime26_30` | 5 | 57,166.4 ± 9,204.9 | 223.3 ± 1.8 | 81.1% ± 1.5% | 3.43 ± 0.04 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 220.3 ± 8.2 | 74.2% ± 4.0% | 3.23 ± 0.12 |
| Story | 15 | 148.8 ± 11.6 | 39.2% ± 5.7% | 2.18 ± 0.17 |
| Translation | 15 | 213.6 ± 12.2 | 70.5% ± 6.0% | 3.12 ± 0.18 |
| Structured | 15 | 252.2 ± 16.3 | 89.8% ± 8.0% | 3.69 ± 0.24 |

The baseline and speculative-decode suites intentionally measure different supported workloads.
No per-scenario baseline/speculative speedup is reported.

## `qwen3_8_27b`

### `nvfp4`

The MTP0 table comes from the serial Long NIAH campaign described by the single-request method. The
MTP3 tables come from the C=1 point of the fixed concurrent-corpus campaign, which serially runs the
same three long-reasoning and twelve cross-scenario fixtures. Each fixture has five fixed seeds. The
tables report arithmetic mean ± sample standard deviation from the server's per-request phase
timings and speculative counters.

#### MTP0 context-length profile

| Prompt tokens | Samples | Prefill tok/s | Server TTFT (ms) | Decode tok/s |
|---:|---:|---:|---:|---:|
| 7,680 | 5 | 8,340.4 ± 13.0 | 931.6 ± 1.6 | 71.2 ± 0.1 |
| 64,512 | 5 | 5,297.9 ± 259.2 | 12,281.1 ± 561.5 | 65.7 ± 0.8 |
| 130,048 | 5 | 3,544.7 ± 25.3 | 36,853.5 ± 259.4 | 59.6 ± 0.9 |
| 260,096 | 5 | 2,203.1 ± 13.4 | 118,354.8 ± 717.2 | 52.9 ± 2.3 |

#### MTP3 long-reasoning decode

| Fixture | Samples | Completion tokens | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|---:|
| `long_decode_aime26_01` | 5 | 1,465.4 ± 417.3 | 195.2 ± 4.6 | 76.0% ± 2.4% | 3.28 ± 0.07 |
| `long_decode_aime26_15` | 5 | 65,414.4 ± 271.9 | 151.4 ± 2.0 | 56.2% ± 1.1% | 2.69 ± 0.03 |
| `long_decode_aime26_30` | 5 | 50,023.4 ± 14,839.1 | 167.5 ± 23.7 | 64.6% ± 14.9% | 2.94 ± 0.45 |

#### MTP3 cross-scenario decode

Each category contains three fixtures and five seeds per fixture, for 15 samples.

| Category | Samples | Decode tok/s | MTP acceptance | MTP tokens/round |
|---|---:|---:|---:|---:|
| Code | 15 | 194.3 ± 6.1 | 76.4% ± 3.9% | 3.29 ± 0.12 |
| Story | 15 | 126.1 ± 10.9 | 37.4% ± 5.8% | 2.12 ± 0.17 |
| Translation | 15 | 192.3 ± 11.9 | 75.0% ± 6.5% | 3.25 ± 0.19 |
| Structured | 15 | 219.8 ± 8.6 | 90.8% ± 5.1% | 3.72 ± 0.15 |

### E026: default pairwise NVFP4 attention

The default packed-NVFP4 attention route now decodes two values per packed byte owner. Dense
causal attention uses the pair route by default, while DFlash2 local attention uses the
scalar-order-preserving pair route by default; `NINFER_NVFP4_PAIR=0` restores the scalar fallback.
Both focused correctness suites pass in default and fallback modes. The dense cached operator was
about 9–10% faster at 8K on the RTX 5090, but the serving result is the deciding gate.

With the Qwen3.8-27B NVFP4 DFlash2 artifact, context 512, `k=7`, optimized proposal head,
VeriCache-NVFP4, hierarchical host snapshots, CUDA Graphs, one warmup round, and five measured
rounds, fresh default/scalar samples were:

| Batch | Default wall round | Scalar wall round | Default tok/s | Scalar tok/s | Draft acceptance |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 18.6615 ms | 18.8937 ms | 192.911 | 190.540 | 13/35 vs 13/35 |
| 8 | 32.3135 ms | 32.7918 ms | 736.535 | 731.891 | 79/280 vs 80/280 |

The C8 acceptance difference is retained as a qualification signal, so these measurements support
the default but do not establish a universal end-to-end multiplier. `NINFER_NVFP4_XQA=1/2`
remains an opt-in research path because its current grouped prototypes are slower on this GPU.
