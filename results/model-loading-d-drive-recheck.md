# D: model-read recheck

Date: 2026-09-02  
Scope: `D:\AI\voidinfer-adaptive-dflash2`  
Model: `D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer`  
Model size: `23,719,496,192` bytes  
Model SHA-256: `6CC7560AE3427D8FA87B75C17E41328116B71B068C4C4DC06137FB73B656F64E`

## Result

The prior D: slowdown is **not reproducible** in this recheck. The same existing direct-I/O probe now reads D: at `7.63–8.07 GB/s` decimal across all five runs. The real engine load reached `3.614 s`, with `2.873 s` in direct reads (`7.09 GB/s`).

This is a state change from the earlier `~0.50–0.64 GB/s` D: measurements and `~39.7 s` real load. No model, firmware, format, security setting, or drive configuration was changed during this recheck.

The strongest static difference discovered is that the D: model file is NTFS-compressed, while the temporary hash-matched C: comparison copy was uncompressed. That is a plausible follow-up suspect, but it is not a proven explanation of the earlier slowdown because D: is currently fast while still compressed.

## D: sequential-read runs

Command used for every run:

```powershell
D:\AI\build-adaptive-dflash2\tools\model_read_probe.exe `
  D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer <buffer_MiB>
```

The probe uses `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN`. Each run reads `23,719,493,632` aligned bytes in the `23,719,496,192`-byte file and skips the final `2,560` bytes because Windows no-buffering requests must be 4 KiB aligned.

| Run | Buffer | Requests | Elapsed | Decimal MB/s | MiB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 MiB | 354 | 2.939 s | 8,070.595 | 7,696.719 |
| 2 | 64 MiB | 354 | 3.077 s | 7,709.893 | 7,352.726 |
| 3 | 64 MiB | 354 | 3.098 s | 7,656.785 | 7,302.079 |
| 4 | 256 MiB | 89 | 3.108 s | 7,631.994 | 7,278.437 |
| 5 | 256 MiB | 89 | 3.111 s | 7,625.129 | 7,271.890 |

64 MiB results: median `7,709.893 MB/s`, mean `7,812.424 MB/s`. 256 MiB results: mean `7,628.562 MB/s`. The conservative current sustained D: result is therefore `7.63 GB/s` decimal.

Raw-run log: `results/model-read-D-recheck.log`  
Log SHA-256: `86A0E5C942F3C3A234BB5E7E4B2A62F2D02C3639BA1D9AEAEB62B3F1982ADBF5`

## C: comparison

A temporary copy was created only for this comparison, verified against the model SHA-256, and removed after testing. The C: copy was in an uncompressed temporary directory.

| Buffer | Requests | Elapsed | Decimal MB/s | MiB/s |
| ---: | ---: | ---: | ---: | ---: |
| 64 MiB | 354 | 4.872 s | 4,868.166 | 4,642.645 |
| 256 MiB | 89 | 3.756 s | 6,314.499 | 6,021.976 |

C: remains multi-GB/s, although this pass was slower and more variable than the earlier `~6.43 GB/s` C: measurements. D: is faster than C: in the current recheck. The comparison is not a fixed hardware ranking; both drives show state/cache/thermal variability across runs.

C comparison log: `results/model-read-C-recheck.log`  
Log SHA-256: `0E23492C642F99CFEB9A61B0C47F55615780B4AA5A3EBB6BEF99D95826F47AAC`

## Real engine D: load

Command:

```powershell
D:\AI\build-adaptive-dflash2\tests\ninfer_qwen3_6_27b_model_load_probe.exe `
  D:\AI\voidinfer\models\Qwen3.8-27B-NVFP4-DFlash2-NInfer\qwen3_8_27b_nvfp4.ninfer 512
```

| Metric | Recheck result |
| --- | ---: |
| Wall time | 3.68358 s |
| Engine load time | 3.61367 s |
| Upload/materialization | 3.00160 s |
| Direct-read time | 2.87270 s |
| Direct-read bytes | 20,375,654,400 B |
| Direct-read requests | 304 |
| Request-size range | 41,668,608–67,108,864 B |
| Maximum outstanding reads | 1 |
| Direct-read rate | 7,092.858 MB/s |
| H2D active time | 0.395311 s |
| H2D active rate | ~51.55 GB/s |
| H2D stream span | 2.85737 s |
| Device allocation | 0.019419 s |
| Pinned staging allocation | 0.017236 s |
| Host resource copy | 0.003698 s |
| Materialization synchronization | 0.111103 s |
| Tensor binding/model construction | 0.554021 s |
| Planner | 0.000807 s |
| Startup synchronization | 0.000012 s |
| Process CPU utilization | 99.258% |
| Peak pinned staging | 268,435,456 B |

The high process CPU in this fast run is expected to expose the next non-I/O portion of startup; it is not evidence of the earlier D: storage stall. H2D active time is still small relative to the read and total-load stages. The long H2D stream span includes time between enqueued chunks.

Engine log: `results/model-load-D-recheck.out`  
Log SHA-256: `A56A033C4582FBF8A5A847886D103406CAA39F5F8FDDA5E49847E7B338F67D33`

## Hardware and Windows checks

Both volumes are separate physical devices:

| Volume | Disk | Device | Controller bus | PCIe link |
| --- | ---: | --- | ---: | --- |
| C: | 0 | WD_BLACK SN850X 4000GB, serial suffix `DAC6` | 2 | current/max Gen4 x4 |
| D: | 1 | WD_BLACK SN850X 4000GB, serial suffix `D7D3` | 105 | current/max Gen4 x4 |

| Check | Result |
| --- | --- |
| Model firmware | Both report `624361WD`. |
| NVMe driver | Microsoft Standard NVM Express Controller `10.0.26100.9278`. |
| Disk/volume health | Both disks `Healthy`/`OK`; D: is NTFS `Healthy`/`OK`. |
| D: free space | `866,776,526,848` B; 4 KiB allocation units. |
| Active power plan | `HYDRA`. No power setting was changed. |
| Storage/NVMe event log | No matching `stornvme`/`disk`/`Ntfs`/`WHEA-Logger` warning/error events were returned for the last 14 days; this is not a substitute for vendor SMART data. |
| Current post-test D: activity | Read-rate, active-time, and queue counters were idle at the post-test snapshot. |

No PCIe downgrade is present. No firmware mismatch is present. No Windows storage event evidence of a persistent device fault was found.

Several checks were access-limited and are explicitly unresolved:

* `Get-StorageReliabilityCounter` could not access the required CIM resource, so temperature, thermal-throttle history, SMART error counts, wear, and power-on counters were not obtained.
* `manage-bde`/BitLocker CIM queries returned access denied; encryption state is not proven by this run.
* `fltmc filters` returned access denied, so the complete filter-driver stack was not enumerated.
* `fsutil dirty query C:` and `fsutil dirty query D:` returned access denied. Volume health still reported Healthy/OK.
* `smartctl`/`nvme` command-line tools were not installed.
* A point-in-time process I/O counter query was unavailable; the post-test disk snapshot was idle. Background processes including Steam, SearchIndexer, and Microsoft Defender were present, but their presence alone does not prove they caused the previous stall.

## File-layout finding

The D: model file has attributes `Archive, Compressed`. `compact /q` reports that the containing D: directory is configured to compress newly added files; the file is compressed with `23,719,496,192` data bytes stored in `23,545,925,632` bytes, approximately a `1.0:1` compression ratio. A VCN extent query returned `346,751` records, consistent with a compressed/sparse NTFS representation rather than a simple contiguous uncompressed file.

The temporary C: comparison copy had only the `Archive` attribute and `compact /q` reported it as uncompressed. This is the main static confounder between the volumes. It is not yet isolated as causal: the same compressed D: file is presently reading at `7.63–8.07 GB/s`, while the previous slow result was `0.50–0.64 GB/s`.

## Interpretation

The current recheck rules out a persistent D: limitation caused by:

* a negotiated PCIe link downgrade;
* an unhealthy disk state visible to Windows;
* a firmware mismatch between the two drives;
* the basic Microsoft NVMe driver;
* a permanently low sequential-read ceiling.

The prior D: slowdown is therefore a transient condition or an interaction not active during this run. The leading candidates are:

1. transient device power/thermal state or background I/O contention;
2. NTFS compression/filter-path behavior on the D: file, which differs from the C: copy;
3. a workload-specific Windows/storage state that has since cleared.

The previous slow run’s high D: logical-disk activity and low process CPU were consistent with waiting on storage. The current fast run’s near-100% process CPU shows that the bottleneck has moved away from D: media reads; no async reader or model-format redesign is justified by this recheck alone.

## Final decision

* **D: current sustained speed:** `7.63 GB/s` conservative raw sequential rate; `7.09 GB/s` direct-read rate in the real engine.
* **C: comparison:** `4.87 GB/s` at 64 MiB and `6.31 GB/s` at 256 MiB in this pass; earlier C: results were about `6.43 GB/s`.
* **Slowdown reproducible?:** **No.** D: is currently roughly `12–16×` faster than the old `0.50–0.64 GB/s` range.
* **Most likely cause:** a transient D: storage/device/host state; NTFS compression is the main persistent confounder and remains unisolated.
* **ONE recommended next diagnostic/fix:** create a temporary **uncompressed, hash-matched copy on D:** in a non-compressed test directory and repeat the same 3×64 MiB + 2×256 MiB probe after the system is idle. If it is consistently fast, use an uncompressed production placement for the model; if it is slow while the compressed original is fast, investigate NTFS compression/filter behavior with administrator-level SMART/filter telemetry before changing the production file.

The temporary C: copy was verified against the model SHA-256 and removed. The original D: model remains present and unchanged. No firmware, formatting, security, power, RAID, or destructive filesystem operation was performed.

**STOP.**
