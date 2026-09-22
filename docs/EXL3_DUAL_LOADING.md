# Default C/D EXL3 loading

`Exl3TextModel::load` uses parallel C/D loading when it finds a verified manifest
at `<model>/.ninfer/dual-load.json`, either in the primary model directory or its
absolute C/D counterpart. This default applies to callers in the engine, tests
and benchmarks without a wrapper or environment override.

The installed target sidecar is
`<target-model>\.ninfer\dual-load.json`.
Its C/D payload identities were verified by full SHA256 comparison. Native loads
check file sizes, write timestamps and tensor framing while holding read-only
handles that prevent replacement or writes. The manifest is a trusted external
verification record; the loader does not rehash entire shards on each startup.
Reverify the files and republish the manifest after replacing either copy.

Two readers use separate16MiB pinned buffers. Large raw target tensors are copied
to their existing GPU allocations as chunks complete. All CUDA calls remain on
the loading thread. Workers, buffers and file handles are released before load
returns. Small tensors and their existing dtype conversions retain the ordinary
reader. Draft loading is unchanged.

## Opt out

Set `NINFER_EXL3_DUAL_LOAD=0` for single-drive loading. Remove it to restore the
default. The C++ equivalent is:

```cpp
ninfer::exl3::Exl3LoadOptions options;
options.disable_dual_artifact_loading = true;
auto model = ninfer::exl3::Exl3TextModel::load(model_directory, max_context, options);
```

An explicit `options.verified_dual_manifest` or
`NINFER_EXL3_DUAL_READ_MANIFEST` can select another verified record. Opt-out takes
precedence. `NINFER_EXL3_DUAL_LOAD=1` requires a verified manifest rather than
allowing single-drive loading when none is available. Invalid published records
fail explicitly. `NINFER_EXL3_DUAL_READ_AUDIT=1` enables the expensive qualification
comparison against original tensor reads; it is excluded from timing runs.

The separate `.ninfer` artifact engine already defaults to its existing dynamic
dual-drive loader, with `EngineOptions::disable_dual_artifact_loading` and
CLI/server `--no-dual-load` opt-outs.

Performance and qualification status belong to the current autonomous result
and checkpoint records, not to an assumed cold-disk throughput guarantee.
