# Build and run on Windows

## Toolchain

- Windows 11 x64 and an NVIDIA GPU supporting `sm_120a` (the current CMake contract).
- Visual Studio 2022 Build Tools with MSVC C++.
- CUDA Toolkit 13.1 or newer, CMake 3.28 or newer, Ninja, Git, Python 3.
- vcpkg dependencies selected by the repository's CMake configuration; FFmpeg and libcurl are
  runtime/build dependencies for media and HTTP support.

From a Developer PowerShell or an ordinary PowerShell that initializes MSVC:

```powershell
cmd /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=amd64 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON"
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
