# Build and run on Windows

## Toolchain

- Windows 11 x64 and an NVIDIA GPU supporting `sm_120a` (the current CMake contract).
- Visual Studio 2026 Build Tools (MSVC 14.51) with the C++ workload.
- CUDA Toolkit 13.3 (validated 2026-10-08; CUDA 13.1 with VS 2022 also builds), CMake 3.28 or newer, Ninja, Git, Python 3.
- vcpkg dependencies selected by the repository's CMake configuration; FFmpeg and libcurl are
  runtime/build dependencies for media and HTTP support.

From a Developer PowerShell or an ordinary PowerShell that initializes MSVC:

```powershell
cmd /c "call C:\BuildTools2026\Common7\Tools\VsDevCmd.bat -arch=amd64 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DCMAKE_CUDA_COMPILER=%CUDA_PATH_V13_3%\bin\nvcc.exe -DBUILD_TESTING=ON"
cmake --build build --target ninfer-cli ninfer-serve
ctest --test-dir build --output-on-failure
```

`NINFER_BUILD_APPS=ON` is the default; tests and benchmarks are opt-in through `BUILD_TESTING` and
`NINFER_BUILD_BENCHMARKS`. A non-`120a` CUDA architecture is rejected intentionally. Do not claim a
clean build merely because an older binary runs.

## CLI smoke

Weights are not in Git. Set a path explicitly:

```powershell
$env:VOIDINFER_MODEL = 'C:\models\qwen3_8_27b.ninfer'
& .\build\apps\ninfer-cli.exe $env:VOIDINFER_MODEL --prompt 'Return exactly: ready'
```

## Server smoke

```powershell
& .\build\apps\ninfer-serve.exe $env:VOIDINFER_MODEL --host 127.0.0.1 --port 8080
Invoke-RestMethod http://127.0.0.1:8080/v1/chat/completions -Method Post `
  -ContentType 'application/json' `
  -Body '{"model":"voidinfer","messages":[{"role":"user","content":"Return exactly: ready"}],"temperature":0}'
```

Use only loopback for a local smoke. Check the startup log for the effective model, context,
concurrency, speculative backend, and cache policy. Interface details are in [serving.md](serving.md).

## EXL3 paths

For the explicit target/draft route, use environment variables rather than developer-specific
paths:

```powershell
$env:VOIDINFER_TARGET_MODEL = 'C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6'
$env:VOIDINFER_DRAFT_MODEL = 'C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw'
```

The exact runnable command depends on the EXL3 test/server target built by the selected profile;
see [exl3-serving.md](exl3-serving.md). The publication procedure did not redistribute either model.

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
