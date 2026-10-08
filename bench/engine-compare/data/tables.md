#### Prefill throughput, target only (tok/s)

| Context (prompt tokens) | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K (4,049) | 10,068 | — | 13,526 | 3,462 | 13,635 | 3,518 |
| 16K (16,321) | 9,628 | — | 12,969 | 3,464 | 12,250 | 3,458 |
| 32K (32,638) | — | — | 11,539 | 3,398 | 10,415 | 3,321 |
| 64K (65,338) | — | — | 9,233 | 3,201 | 7,823 | 3,045 |
| 128K (130,734) | — | — | 6,485 | 2,832 | — | — |

#### Base (non-speculative) greedy decode (tok/s)

| Context (prompt tokens) | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K (4,049) | 79.2 | — | 81.7 | 93.8 | 81.0 | 93.3 |
| 16K (16,321) | 75.7 | — | 79.9 | 91.6 | 78.0 | 89.2 |
| 32K (32,638) | — | — | 78.1 | 89.2 | 74.5 | 84.6 |
| 64K (65,338) | — | — | 74.6 | 84.9 | 68.5 | 76.9 |
| 128K (130,734) | — | — | 68.5 | 77.0 | — | — |

#### DFlash2 K=7 greedy decode on the code-review workload (tok/s)

| Context (prompt tokens) | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K (4,049) | 211.3 | — | 186.0 | 175.9 | 190.9 | 158.0 |
| 16K (16,321) | 186.2 | — | 187.1 | 169.2 | 175.0 | 175.3 |
| 32K (32,638) | — | 178.2 | 189.9 | 151.1 | 178.0 | 164.9 |
| 64K (65,338) | — | 164.3 | 194.8 | 155.5 | 151.8 | 135.3 |
| 128K (130,734) | — | 179.8 | 171.6 | 133.5 | — | — |

#### DFlash2 committed tokens per verification round

| Context (prompt tokens) | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K (4,049) | 3.74 | — | 2.84 | 3.05 | 2.94 | 2.76 |
| 16K (16,321) | 3.48 | — | 2.90 | 3.00 | 2.78 | 3.14 |
| 32K (32,638) | — | 3.30 | 3.00 | 2.71 | 2.95 | 3.05 |
| 64K (65,338) | — | 3.16 | 3.20 | 2.87 | 2.70 | 2.67 |
| 128K (130,734) | — | 3.59 | 3.04 | 2.63 | — | — |

#### Prefill throughput with the DFlash2 draft resident (tok/s)

VoidInfer's Engine prefill also fills the draft's context taps/ring; NInfer's prefill with DFlash2 resident.

| Context (prompt tokens) | VoidInfer FP16 KV | VoidInfer L0 OSCAR | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|---:|
| 4K (4,049) | 7,308 | — | 13,907 | 3,429 | 13,449 | 3,470 |
| 16K (16,321) | 8,258 | — | 12,830 | 3,456 | 12,227 | 3,431 |
| 32K (32,638) | — | 7,309 | 11,478 | 3,384 | 10,379 | 3,300 |
| 64K (65,338) | — | 6,408 | 9,185 | 3,190 | 7,789 | 3,030 |
| 128K (130,734) | — | 5,083 | 6,463 | 2,824 | — | — |

#### DFlash2 on NInfer's 12 scenario prompts (≈2.1K-token prompts, ≤1,024 outputs, tok/s)

| Scenario | VoidInfer FP16 KV | NInfer NVFP4 / FP8 KV | NInfer groupwise / FP8 KV | NInfer NVFP4 / BF16 KV | NInfer groupwise / BF16 KV |
|---|---:|---:|---:|---:|---:|
| code_cuda | 243.8 (201) | 251.8 (1,024) | 253.2 (1,024) | 273.5 (1,024) | 263.7 (1,024) |
| code_python | 250.8 (127) | 258.5 (1,024) | 220.3 (1,024) | 233.0 (1,024) | 245.2 (1,024) |
| code_typescript | 273.8 (94) | 340.5 (94) | 261.5 (42) | 340.9 (94) | 262.4 (42) |
| story_en_mystery | 169.0 (1,024) | 174.4 (1,024) | 153.6 (1,024) | 170.3 (1,024) | 148.4 (1,024) |
| story_zh_dialogue | 103.5 (1,024) | 104.3 (1,024) | 90.7 (1,024) | 105.5 (1,024) | 95.9 (1,024) |
| story_zh_scifi | 117.5 (1,024) | 115.6 (1,024) | 102.1 (1,024) | 114.5 (1,024) | 103.4 (1,024) |
| structured_csv | 370.4 (1,024) | 381.0 (1,024) | 334.5 (1,024) | 410.4 (1,024) | 333.3 (1,024) |
| structured_jsonl | 445.1 (1,024) | 441.6 (336) | 411.4 (1,024) | 441.2 (336) | 411.8 (1,024) |
| structured_sql | 346.0 (1,024) | 372.8 (1,024) | 308.5 (1,024) | 339.9 (1,024) | 322.4 (1,024) |
| translation_en_zh | 271.7 (772) | 198.9 (1,024) | 241.1 (754) | 276.9 (764) | 236.7 (773) |
| translation_markdown | 217.5 (816) | 223.1 (752) | 225.4 (861) | 226.9 (771) | 211.4 (753) |
| translation_zh_en | 324.4 (789) | 378.7 (725) | 332.4 (719) | 385.5 (875) | 344.7 (834) |
| **geometric mean** | **240.6** | **246.0** | **223.9** | **252.3** | **227.3** |

Parenthesized: generated tokens (both engines stop at EOS; greedy text diverges between weight formats).
