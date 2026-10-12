# Build and run on Windows

## Toolchain

- Windows 11 x64 and an RTX 5090 (`sm_120a`; any other CUDA architecture is rejected at configure).
- Visual Studio 2026 Build Tools (MSVC 14.51) with the C++ workload.
- CUDA Toolkit 13.3, CMake 3.28 or newer, Ninja, Git, Python 3.
- vcpkg dependencies selected by the repository's CMake configuration (FFmpeg and libcurl for media
  and HTTP).

```powershell
cmd /c "call C:\BuildTools2026\Common7\Tools\VsDevCmd.bat -arch=amd64 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CUDA_COMPILER=%CUDA_PATH_V13_3%\bin\nvcc.exe -DBUILD_TESTING=ON"
cmake --build build -j --target voidinfer voidinfer-serve
ctest --test-dir build --output-on-failure
```

`VOIDINFER_BUILD_APPS=ON` (the CLI and server) is the default; tests are opt-in through
`BUILD_TESTING`. Tests that need the model skip (exit 77) unless `VOIDINFER_EXL3_TARGET_PATH` and
`VOIDINFER_EXL3_DFLASH2_PATH` name it.

## Model package

Weights are not in Git. The Engine needs:

| Directory | Contents |
|---|---|
| target | Qwen3.8-27B EXL3 6 bpw (`Qwen3.8-27B-exl3-6bpw-H6-V6`), with the OSCAR rotations in `l0_oscar/` |
| draft | the DFlash2 draft (`Qwen3.8-27B-DFlash2-EXL3-5bpw`) |
| vision (optional) | the BF16 checkpoint or any directory with the `model.visual.*` tensors |

A `<target>/.voidinfer/dual-load.json` manifest lets the loader read weights from verified C: and D:
mirrors concurrently ([EXL3 dual loading](EXL3_DUAL_LOADING.md)).

## CLI smoke

```powershell
$model = @('--target', 'C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6', '--draft', 'C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw')
& .\build\apps\voidinfer.exe @model --prompt 'Return exactly: ready' --no-thinking --greedy --max-new 8
```

## Server smoke

```powershell
& .\build\apps\voidinfer-serve.exe @model --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType 'application/json' `
  -Body '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0}'
```

No environment variables are needed: the Engine applies its benchmarked profile at startup. Check the
startup line for lanes, context, weights, runtime and free memory, host KV and Vision. Add
`--vision DIR` for image and video input. Interface details are in [serving.md](serving.md) and
[cli.md](cli.md).

Optional Nsight Systems/Compute tools are for profiling and are not runtime dependencies. Profiling
changes timing; keep attribution captures separate from production-style measurements.

## Bounded host-only configuration checks

During a no-GPU maintenance phase, the ordinary configure/build/CTest commands
above are not the host-only entry point. On Windows, use:

```powershell
pwsh -NoProfile -File tools/maintenance/Test-CleanupJob.ps1
pwsh -NoProfile -File tools/maintenance/Run-Exl3HostContracts.ps1
```

The first command checks Job Object membership, effective affinity `0xFF` and
BelowNormal priority in a parent, child and grandchild, including an attempted
affinity widening. The second directly compiles and runs only the environment
option/restoration and existing candidate-contract tests. It uses the same job
limits, one compiler at a time, `/MP1`, `/cgthreads1` and single-threaded library
settings. The Job Object has no breakaway permission and terminates remaining
children when its owning shell exits. Run these scripts in a disposable shell;
their environment and containment apply to that shell's lifetime.

The runner accepts explicit MSVC/Windows SDK roots and versions; its defaults
match the local MSVC 14.44 / Windows SDK 10.0.26100.0 installation. It has no
CUDA includes/linkage, project configure, device probe, dependency installation
or test discovery. Outputs stay in `build-host-contracts/`. To extend the closed
allowlist, audit the added test's complete include/link/launch path first.

These checks do not build or qualify the engine, CUDA translation units or the
full DFlash2 harness. Separate authorization and matched runtime validation are
required before attributing performance or promoting the cleaned source.
