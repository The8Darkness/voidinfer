#### DFlash2 greedy decode, code-review workload (tok/s, 1,024 outputs)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 180.3 | 186.6 | 177.2 |
| 16K (16,321) | 169.3 | 187.4 | 171.0 |
| 32K (32,638) | 144.7 | 190.3 | 152.6 |
| 64K (65,338) | 159.8 | 195.3 | 157.1 |
| 128K (130,734) | 157.9 | 171.9 | 134.8 |
| 254K (259,483) | 136.6 | — | 118.1 |

#### DFlash2 committed tokens per verification round

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 3.56 | 2.84 | 3.05 |
| 16K (16,321) | 3.37 | 2.90 | 3.00 |
| 32K (32,638) | 2.95 | 3.00 | 2.71 |
| 64K (65,338) | 3.40 | 3.20 | 2.87 |
| 128K (130,734) | 3.67 | 3.04 | 2.63 |
| 254K (259,483) | 3.63 | — | 2.59 |

#### Time to first token with the DFlash2 draft resident (s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 0.56 | 0.29 | 1.17 |
| 16K (16,321) | 2.33 | 1.25 | 4.68 |
| 32K (32,638) | 5.52 | 2.81 | 9.55 |
| 64K (65,338) | 13.34 | 7.05 | 20.29 |
| 128K (130,734) | 33.18 | 20.10 | 46.07 |
| 254K (259,483) | 94.80 | — | 114.61 |

#### Prefill throughput with the DFlash2 draft resident (tok/s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 7,290 | 13,878 | 3,465 |
| 16K (16,321) | 7,007 | 13,087 | 3,490 |
| 32K (32,638) | 5,908 | 11,605 | 3,419 |
| 64K (65,338) | 4,898 | 9,272 | 3,219 |
| 128K (130,734) | 3,941 | 6,504 | 2,838 |
| 254K (259,483) | 2,737 | — | 2,264 |

#### Base (non-speculative) greedy decode (tok/s, 256 outputs)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 76.4 | 82.4 | 94.3 |
| 32K (32,638) | 74.0 | 78.8 | 89.8 |
| 128K (130,734) | 65.7 | 69.1 | 77.6 |

#### Prefill throughput, target only (tok/s)

| Context (prompt tokens) | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| 4K (4,049) | 7,785 | 14,127 | 3,473 |
| 32K (32,638) | 6,890 | 11,729 | 3,412 |
| 128K (130,734) | 4,882 | 6,559 | 2,832 |

#### Largest single request: 254K-token prompt (259,483 tokens), DFlash2, 1,024 outputs

| Engine / KV | Decode (tok/s) | Tokens / round | Prefill (tok/s) | Time to first token (s) |
|---|---:|---:|---:|---:|---:|
| VoidInfer, L0 OSCAR INT4 + exact VeriCache | 136.6 | 3.63 | 2,737 | 94.8 |
| NInfer NVFP4 / FP8 KV | does not start: KV reservation exceeds free VRAM | | | |
| NInfer NVFP4 / k8v4 KV | 140.5 | 2.76 | 3,977 | 65.2 |
| NInfer NVFP4 / NVFP4 KV | 133.6 | 2.92 | 3,017 | 86.0 |
| NInfer groupwise / FP8 KV | 118.1 | 2.59 | 2,264 | 114.6 |

#### DFlash2 on NInfer's 12 scenario prompts (~2.1K-token prompts, ≤1,024 outputs, tok/s)

| Scenario | VoidInfer (EXL3 6 bpw, L0 OSCAR INT4 KV, exact VeriCache) | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV |
|---|---:|---:|---:|
| code_cuda | 279.2 (495) | 252.7 (1,024) | 255.8 (1,024) |
| code_python | 207.7 (1,024) | 259.5 (1,024) | 222.5 (1,024) |
| code_typescript | 249.5 (149) | 342.7 (94) | 264.4 (42) |
| story_en_mystery | 159.6 (1,024) | 174.6 (1,024) | 155.1 (1,024) |
| story_zh_dialogue | 97.9 (1,024) | 104.6 (1,024) | 91.6 (1,024) |
| story_zh_scifi | 113.1 (1,024) | 115.9 (1,024) | 102.2 (1,024) |
| structured_csv | 250.5 (1,024) | 382.1 (1,024) | 335.2 (1,024) |
| structured_jsonl | 359.0 (1,024) | 441.2 (336) | 412.2 (1,024) |
| structured_sql | 280.7 (1,024) | 374.3 (1,024) | 310.2 (1,024) |
| translation_en_zh | 231.1 (874) | 199.8 (1,024) | 243.2 (754) |
| translation_markdown | 191.6 (815) | 223.3 (752) | 227.7 (861) |
| translation_zh_en | 289.5 (943) | 382.3 (725) | 335.9 (719) |
| **geometric mean** | **212.1** | **246.8** | **225.6** |

Parenthesized: generated tokens (both engines stop at EOS; greedy text differs between weight formats).

#### Concurrent requests: aggregate greedy DFlash2 decode (tok/s, N × 1,024 outputs)

Aggregate = all completion tokens / wall time of the batch (prefill included). Mean of 2 runs; per-request range in brackets. NInfer: NVFP4, FP8 KV, `--kv-capacity` = N × max context (KV reserved per slot), prefix reuse off and on (VoidInfer reuses shared prefixes by default; the fixtures put the code in a system message all requests share). Where the reservation exceeds free VRAM the server does not start; a shared `--kv-capacity auto` pool was not measured (the follow-up run was stopped). Ratio = VoidInfer / faster NInfer.

| Prompt | Concurrency | VoidInfer | NInfer NVFP4 / FP8 KV, no prefix reuse | NInfer, prefix reuse | Ratio |
|---|---:|---:|---:|---:|---:|
| code_31k | C2 | 125.1 [62–65] | 176.2 [88–90] | 171.3 [84–88] | 0.71× |
| code_31k | C4 | 213.9 [52–63] | does not start | — | — |
| code_31k | C8 | 180.0 [22–52] | does not start | — | — |
| code_4k | C1 | 160.7 [161–161] | 189.1 [189–190] | 184.2 [184–184] | 0.85× |
| code_4k | C2 | 258.2 [127–137] | 340.2 [170–197] | 318.5 [159–160] | 0.76× |
| code_4k | C4 | 365.4 [86–118] | 526.0 [130–148] | 525.6 [128–143] | 0.69× |
| code_4k | C8 | 396.2 [50–69] | does not start | — | — |

#### Multi-agent long context: 8 agents × 99K–113K-token code prompts (865K tokens), 4 lanes, 2 turns

VoidInfer: EXL3 6 bpw, L0 OSCAR INT4 + exact VeriCache, `--max-context 262144`. NInfer: groupwise artifact, FP8 KV, `--kv-capacity auto` (166,720 tokens on device), `--host-context-mib 48000`, `--max-context 131072` (the NVFP4 artifact cannot reserve this configuration). 256 output tokens per agent and turn.

| Engine | Agents | Turn | Requests OK | Prompt tokens (sum) | Completion tokens | Wall (s) | Peak VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ours | 8 | 1 | 8/8 | 865,256 | 2,048 | 223 | 30,020 |
| ours | 8 | 2 | 8/8 | 867,504 | 2,048 | 52 | 30,002 |
| ninfer | 8 | 1 | 8/8 | 865,256 | 2,048 | 310 | 27,533 |
| ninfer | 8 | 2 | 8/8 | 867,536 | 2,048 | 11 | 27,533 |

#### Vision (image + question, greedy, 256 tokens max)

| Server | Image | Result | Prompt tokens | Latency, warm (s) | Answer (first 100 chars) |
|---|---|---|---:|---:|---|
| ninfer.nv | natural_scene | ok | 404 | 1.21 | This is a cheerful, flat-style digital illustration titled **“A SUNNY HOUSE BY THE MOUNTAINS”**, dis |
| ninfer.nv | visual_chart | ok | 411 | 1.01 | This chart, titled **“NIFER VISION 731”**, appears to be a simple visual representation or test of b |
| ours.exl3-novision | natural_scene | rejected (400) | — | — | {"error":{"code":"vision_disabled","message":"Vision is disabled for this server","param":"messages" |
| ours.exl3-novision | visual_chart | rejected (400) | — | — | {"error":{"code":"vision_disabled","message":"Vision is disabled for this server","param":"messages" |
