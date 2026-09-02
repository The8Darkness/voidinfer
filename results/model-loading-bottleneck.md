# VoidInfer model-loading bottleneck probe

Date: 2026-09-02  
Scope: `D:\AI\voidinfer-adaptive-dflash2`  
Artifact: `qwen3_8_27b_nvfp4.ninfer`  
Artifact size: `23,719,496,192` bytes  
Artifact SHA-256: `6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E`

## Executive result

The current D: load is storage-read bound. In the clean instrumented run, the engine spent `39.0539 s` in direct reads for `20,375,654,400` bytes, versus `39.6894 s` total load time. The direct-read rate was `521.7 MB/s` decimal. Process CPU utilization was only `2.00%`.

H2D was not the bottleneck: the transfer stream was active for only `0.3750 s`, with an effective transfer rate of approximately `54.3 GB/s`; its CUDA event span was `38.9338 s` because the stream was waiting for host chunks between copies. Parsing, allocation, resource copying, binding, and synchronization were all sub-second or millisecond-scale relative to disk I/O.

The loader is therefore not literally `read -> H2D -> synchronize -> next read`. It is:

```text
blocking direct ReadFile (one outstanding request)
  -> enqueue cudaMemcpyAsync on the transfer stream
  -> record a slot-completion event
  -> issue the next blocking ReadFile
```

The four pinned staging slots allow H2D to overlap a later read, and slot events are waited on only when a slot is reused or at finalization. However, the disk side has no read-ahead queue: the `OVERLAPPED` Windows request is immediately waited with `GetOverlappedResult(..., TRUE)`. This makes storage effectively depth 1.

The same unchanged loader on the exact hash-matched artifact placed temporarily on C: completed in `3.8885 s` load time, with `3.1975 s` in direct reads (`6,372.4 MB/s`) and `27.6%` process CPU. C: and D: are separate healthy 4 TB WD_BLACK SN850X NVMe devices. The C: result is a measured placement result, not a model-format change; the temporary copy was deleted after testing and the D: original was preserved.

## Instrumentation added

The probe instrumentation is diagnostic-only and does not change the artifact format or model tensors.

| Area | Instrumentation |
| --- | --- |
| File open/map | Timed `MappedFile` construction and full-file mapping. |
| Directory parse/validation | Timed JSON directory parsing and ordered/aligned object validation. |
| Direct file reads | Count, bytes, minimum/maximum request size, elapsed request time, and maximum outstanding depth. |
| Host/device memory | Device-arena allocation, pinned staging allocation, and host resource copies. |
| H2D | CUDA stream span and per-chunk CUDA event active time. |
| Synchronization | Slot-event waits, final transfer synchronization, and startup device synchronizations. |
| Binding/planning | Model construction/view binding, load planning, and instance construction. |
| CPU | Process CPU time via `GetProcessTimes`. |

The current materialization path performs no numeric tensor conversion. Encoded artifact payloads are copied into device allocations; `tensor_binding_seconds` is view/model construction rather than a conversion pass.

## Independent sequential-read measurements

`tools/model_read_probe.exe` uses the same Windows direct-I/O properties as the engine (`FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`) and reads 4 KiB-aligned portions of the artifact. The final `2,560` bytes are skipped because the no-buffering request must be aligned.

### D:

| Buffer | Requests | Elapsed | Decimal rate | MiB/s |
| ---: | ---: | ---: | ---: | ---: |
| 4 MiB | 5,656 | 42.997 s | 551.659 MB/s | 526.103 |
| 64 MiB | 354 | 36.988 s | 641.271 MB/s | 611.564 |
| 64 MiB repeat | 354 | 42.700 s | 555.487 MB/s | 529.754 |
| 256 MiB | 89 | 47.605 s | 498.253 MB/s | 475.171 |

The independent D: result varies from `498–641 MB/s` across these runs; the repeated 64 MiB runs were `555–641 MB/s`. The clean real-engine result (`522 MB/s`) is within this measured single-drive range.

### C:

The exact hash-matched temporary copy was read from C: before it was removed:

| Buffer | Requests | Elapsed | Decimal rate | MiB/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 MiB | 354 | 3.683 s | 6,439.705 MB/s | 6,141.381 |
| 64 MiB repeat | 354 | 3.674 s | 6,455.884 MB/s | 6,156.811 |
| 256 MiB | 89 | 3.689 s | 6,429.852 MB/s | 6,131.984 |

The C: measurements are internally consistent at approximately `6.43 GB/s` and were taken with the same direct-I/O probe. A sustained cold-cache retest is still prudent before treating that number as an operational guarantee.

## Real engine load measurements

Command used:

```powershell
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_model_load_probe.exe `
  D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer 512
```

The probe was built `RelWithDebInfo` with Ninja, CUDA `13.1`, and the existing SM120 configuration. Build target: `ninfer_qwen3_6_27b_model_load_probe`.

| Stage | D: clean run | C: exact-copy run |
| --- | ---: | ---: |
| Wall time | 39.7564 s | 3.9582 s |
| Engine load time | 39.6894 s | 3.8885 s |
| Upload/materialization | 39.0823 s | 3.3193 s |
| Direct-read elapsed | 39.0539 s | 3.1975 s |
| Direct-read bytes | 20,375,654,400 | 20,375,654,400 |
| Direct-read requests | 304 | 304 |
| Request-size range | 41,668,608–67,108,864 B | 41,668,608–67,108,864 B |
| Maximum outstanding reads | 1 | 1 |
| Direct-read rate | 521.7 MB/s | 6,372.4 MB/s |
| H2D stream span | 38.9338 s | 3.1752 s |
| H2D active time | 0.3750 s | 0.3631 s |
| Effective H2D rate | 54.3 GB/s | 56.1 GB/s |
| Device allocation | 0.0209 s | 0.0207 s |
| Pinned staging allocation | 0.0170 s | 0.0182 s |
| Host resource copy | 0.0040 s | 0.0036 s |
| Tensor binding/model construction | 0.5476 s | 0.5090 s |
| Planning | 0.0009 s | 0.0008 s |
| Materialization synchronization | 0.0023 s | 0.1044 s |
| Startup synchronization | 0.000015 s | 0.000009 s |
| Process CPU utilization | 2.00% | 27.63% |
| Peak pinned staging | 268,435,456 B | 268,435,456 B |
| Device tensors/resources | 659 / 6 | 659 / 6 |

The C: run is `10.21x` faster in engine load time. The H2D active duration is nearly unchanged between volumes, which is expected: the same `20.376 GB` of opaque NVFP4 payload is copied to the same GPU. The C: materialization-sync value is an observed one-run variance and is not load-dominant.

The loader reads only the planned materialized ranges, not every byte of the nominal `23.719 GB` file. `artifact_bytes_read` was `20,388,491,577` bytes including resource reads; `host_to_device_bytes` was `20,375,587,264` bytes. The remaining file content is mapped/available to the reader but is not part of this load plan.

## Windows disk/process observations

During an instrumented D: engine run, localized Windows performance counters reported:

| Counter | Observation |
| --- | ---: |
| Logical D: read rate | `476.3 MB/s` average, `539.3 MB/s` maximum |
| Logical D: active-time counter | `105.6%` average; this counter can exceed 100% due to its accounting model and indicates continuous high activity |
| Logical D: maximum queue length | 3 |
| Total-machine CPU | `5.68%` average, `8.69%` maximum |
| Probe-process CPU | `2.00%` |

The logical-disk counter stayed busy while the application process remained mostly blocked in direct reads. This is consistent with a saturated or near-saturated D: read path, not an idle drive waiting for the GPU. The C: engine result provides a separate loader-rate measurement; direct-read timing gives `3.1975 s` of C: read-side active interval for the same `20.376 GB`.

## Stage diagnosis

1. **Primary: direct storage reads on D:.** They account for `39.0539/39.6894 = 98.4%` of load time.
2. **Not parsing/deserialization.** Open/map, directory parse, and validation total about `2.6 ms` on D:.
3. **Not host allocation/copy.** Device allocation, pinned allocation, and resource copy total about `42.9 ms` on D:.
4. **Not tensor conversion.** No numeric conversion stage exists in this path; binding is opaque view/model construction (`0.548 s`).
5. **Not H2D bandwidth.** CUDA transfer active time is `0.375 s`; the transfer stream span is long because it is fed intermittently by the blocking reader.
6. **Not startup synchronization.** Startup sync was approximately `15 us`; materialization sync was `2.3 ms` in the clean D: run.

The observed engine behavior is therefore a depth-1 direct reader with H2D enqueue overlap, rather than a fully serialized H2D/synchronize loop. The four staging slots do not imply four outstanding disk requests: one slot is synchronously filled before the next slot is read.

## Single-drive opportunity

The best independent D: raw run reached `641.3 MB/s`, while the clean real engine reached `521.7 MB/s` on the same planned-byte count. At the best measured raw rate, the `20.376 GB` planned payload has a read-only lower bound of approximately `31.8 s`; the current real D: load is `39.7 s`. Thus a true bounded read-ahead implementation could plausibly recover application gaps, but the measured upside on D: is at most about `1.25–1.3x` for this load, not an order-of-magnitude improvement. The D: disk is already continuously busy, so the user’s conditional “storage not saturated” case was not met and no speculative loader rewrite was made.

The smallest safe D:-side experiment, if the artifact must remain on D:, is a bounded Windows overlapped-read/IOCP queue of four to eight aligned 64 MiB pinned buffers. It should preserve source order, feed the existing H2D transfer stream, and retain slot completion events. It must be benchmarked against the current depth-1 path because the disk, not the GPU, is the limiting resource.

## Dual-drive sharding assessment

The volumes are physically distinct:

| Volume | Disk | Device | Serial suffix |
| --- | ---: | --- | --- |
| C: | 0 | WD_BLACK SN850X 4000GB NVMe | `...DAC6` |
| D: | 1 | WD_BLACK SN850X 4000GB NVMe | `...D7D3` |

Dual-drive sharding is **not justified as the first solution** from these measurements. A full-artifact C: placement already measured `3.89 s` engine load, versus `39.69 s` from D:. A balanced split would still be gated by the slow D: shard and would be much slower than keeping the whole artifact on the currently faster C: volume. Sharding becomes worthwhile only if:

* C: cannot hold or reliably sustain a full artifact;
* a cold-cache C: retest invalidates the observed rate; or
* a later experiment has a genuinely balanced partition and concurrent read path.

This probe did not use RAID0 and did not change the `.ninfer` format.

## Fastest safe loading architecture

The fastest currently demonstrated safe architecture is:

```text
keep the existing artifact and loader
  + place the complete hash-verified artifact on C:
  + retain direct aligned reads and asynchronous H2D copies
```

It requires no format redesign, no model-weight transformation, and no CUDA changes. Before operationalizing it, repeat a sustained/cold C: measurement and verify enough free space for the artifact plus normal working headroom.

If the artifact must remain on D:, the one next implementation experiment should be a bounded depth-4/8 asynchronous read-ahead queue. It is a bounded opportunity to approach the measured D: raw ceiling, not a remedy for a compute or H2D bottleneck.

## Reproduction and source identity

Independent read command:

```powershell
D:\AI\build-adaptive-dflash2\tools\model_read_probe.exe <model.ninfer> 64
```

Engine probe command:

```powershell
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_model_load_probe.exe <model.ninfer> 512
```

Relevant source locations:

* `src/artifact/reader.cpp:241-245` — Windows direct handle flags.
* `src/artifact/reader.cpp:316-361` — direct request and immediate overlapped completion; diagnostic read counters.
* `src/artifact/materializer.cpp:18-19` — four 64 MiB slot limit.
* `src/artifact/materializer.cpp:226-305` — slot wait, blocking read, asynchronous H2D enqueue, slot event, final synchronization.
* `src/targets/registry.cpp:106-183` — planner, materialization, construction, binding, and startup timing boundaries.
* `tools/model_read_probe.cpp` — independent direct sequential reader.
* `tests/targets/qwen3_6_27b/test_model_load_probe.cpp` — real-engine probe and process CPU measurement.

Diagnostic source hashes:

| File | SHA-256 |
| --- | --- |
| `src/artifact/reader.cpp` | `E1B3B489A5154099362C44933AF031842550BE75DF50302A632365AD1F27439E` |
| `src/artifact/reader.h` | `6C7D425A5C5F0AFAC8F195AA3AC6BBF2404BF68EEB7B7A0C5A1935936B2A6C5F` |
| `src/artifact/materializer.cpp` | `D6B76842060FC795E6B1D5B1CD91346FC8B029BE0573F17105C741177B855849` |
| `src/artifact/materializer.h` | `DDFD5BB8F9A859160590F4C357903DEDF80969ED6E98B77B5A8C71F0AD587DBB` |
| `src/targets/registry.cpp` | `9B724F4A407D133829BF5F649C8C5828A57FB123D2E95F8D50867947073C3015` |
| `include/ninfer/types.h` | `E2DFD71819B60DAB5B0EAC199370AA7C4B0A54E4156AE3EBB37923A5C637168F` |
| `tools/model_read_probe.cpp` | `C3002412DABBD601698085C13F8251EF96383F53C19614A2A43149B5CDD94DE6` |
| `tests/targets/qwen3_6_27b/test_model_load_probe.cpp` | `5CBB55D28F62A7EAAC1FFD2D3ECA46CE66890668DF37170F90CB43AC03AF5565` |
| `tools/model_read_probe.exe` | `9040B1A27A093844FFAC75F3D2936A7B6B89160E76D6C9B9FE14DCB2FD799F01` |
| `ninfer_qwen3_6_27b_model_load_probe.exe` | `8BD173A7BCF772B3CB4D42AFE15EA0FAD3E021C9FF70A6387171BF76917A3CCA` |

Raw outputs are retained under `results/model-*` for audit, including the clean D: run, C: engine run, and independent read probes. The temporary C: artifact copy used for the experiment was verified against the D: SHA-256 and removed; the original D: artifact was not altered.

## Final decision

* **Current bottleneck:** D: direct storage reads; approximately `98.4%` of engine load time.
* **Theoretical single-drive opportunity:** D: can approach the best observed raw `641 MB/s`, reducing the planned-read lower bound to about `31.8 s`; read-ahead can recover at most roughly `1.3x` over the current engine path.
* **Measured achievable single-drive loading speed:** D: `498–641 MB/s` independent raw across tested buffers/runs and `521.7 MB/s` in the clean real engine; C: `6.43 GB/s` raw and `6.37 GB/s` in the real engine.
* **Dual-drive sharding justified?:** No, not as the first solution. Full-artifact C: placement is already the faster measured architecture; a split involving slower D: would be gated by D:.
* **ONE recommended next optimization:** place the complete hash-verified artifact on C: and repeat a sustained/cold validation; if D: residency is mandatory, replace the depth-1 reader with a bounded overlapped read-ahead queue as a separate follow-up experiment.

**STOP: model loading was profiled; no model-format redesign, RAID0, CUDA-kernel optimization, or runtime semantic change was performed.**
