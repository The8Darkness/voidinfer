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

## Prebuilt Windows package

[GitHub Releases](https://github.com/The8Darkness/voidinfer/releases) carry
`voidinfer-<version>-win64-sm120a.zip`: `voidinfer.exe`, `voidinfer-serve.exe`, every DLL they load
(FFmpeg, libcurl, zlib, cuBLAS/cuBLASLt, the MSVC runtime), the OSCAR rotations in `l0_oscar\` and
the license notices. No Visual Studio, CUDA toolkit, vcpkg or VC++ redistributable is needed.

Requirements: Windows 11 x64, an RTX 5090 with an NVIDIA driver that supports CUDA 13.3 (tested with
621.01), and host RAM for the exact attention history: 64 KiB per token of context, see
[host memory](#host-memory). The binaries contain `sm_120a` code only and do not run on other GPUs.

1. Extract the zip and check it against the `.sha256` file published next to it:
   `(Get-FileHash -Algorithm SHA256 voidinfer-<version>-win64-sm120a.zip).Hash`.
2. Download the weights (with `pip install -U huggingface_hub`):

   ```powershell
   hf download Mia-AiLab/Qwen3.8-27B-EXL3 --revision SC_6.00bpw_H6_V6 --local-dir C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6
   hf download Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw --local-dir C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw
   ```

   For image and video input, also download the BF16 checkpoint (`hf download Qwen/Qwen3.8-27B
   --local-dir C:\models\Qwen3.8-27B-bf16`); only its `model.visual.*` tensors are read.
3. Copy the package's `l0_oscar` folder into the target directory
   (`Copy-Item -Recurse .\l0_oscar C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6\`), or point
   `VOIDINFER_EXL3_L0_OSCAR_ROT` at it. The rotations belong to this exact target revision.
4. Run one prompt, then the server:

   ```powershell
   .\voidinfer.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw `
       --prompt 'What is the capital of France? Answer in one sentence.' --no-thinking --greedy --max-new 32
   .\voidinfer-serve.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw `
       --model-id qwen3.8-27b --max-context 65536 --port 8080 --webui
   ```

   The server speaks the OpenAI (`/v1/chat/completions`, `/v1/responses`) and Anthropic
   (`/v1/messages`) protocols at `http://127.0.0.1:8080`; `--webui` also downloads and serves the llama.cpp chat page there. Add
   `--vision C:\models\Qwen3.8-27B-bf16` for media, `--max-concurrency N` (1-8) for concurrent
   requests and `--max-context 262144` for the full window at one lane. All options:
   [serving](serving.md) and [CLI](cli.md).

Maintainers build the package from a Release build in a VS 2026 developer shell with
`tools\release\New-VoidInferPackage.ps1 -BuildDir build -RotationsDir <target>\l0_oscar`, which
walks the executables' DLL dependencies, writes `dist\voidinfer-<yyyy.MM.dd>-<commit>-win64-sm120a.zip`
and its SHA-256, and fails on any dependency it cannot place.

## Model package

Weights are not in Git. The Engine needs:

| Directory | Contents |
|---|---|
| target | Qwen3.8-27B EXL3 6 bpw ([Mia-AiLab/Qwen3.8-27B-EXL3](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3/tree/SC_6.00bpw_H6_V6), branch `SC_6.00bpw_H6_V6`), with the OSCAR rotations in `l0_oscar/` (fitted locally, [tools/oscar](../tools/oscar/README.md)) |
| draft | the DFlash2 draft ([Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw)) |
| vision (optional) | the BF16 checkpoint ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)) or any directory with the `model.visual.*` tensors |

A `<target>/.voidinfer/dual-load.json` manifest lets the loader read weights from verified C: and D:
mirrors concurrently ([EXL3 dual loading](EXL3_DUAL_LOADING.md)).

## Host memory

The GPU keeps only the compressed INT4 history codes (about 17 KiB per token). The exact FP16
attention history that every token is verified against lives in host RAM: **64 KiB per token**
(16 full-attention layers × 4 KV heads × 256 dimensions × K and V × 2 bytes).

| Context held in RAM | Exact FP16 history |
|---|---:|
| 32K tokens | 2 GiB |
| 64K tokens | 4 GiB |
| 128K tokens | 8 GiB |
| 262,144 tokens (one full window) | 16 GiB |
| 8 agents × 110K tokens | 54 GiB |

Every running request holds its whole history. Finished conversations stay in RAM as retained roots,
up to `--host-kv-mib` (default: the largest of 8 GiB, `--max-context` × 64 KiB and 40% of physical
memory), so a returning agent resumes without a new prefill. A root evicted from that budget is
prefilled again on its next turn. Under memory pressure idle histories are packed losslessly and only
then spilled to a delete-on-close file in `%TEMP%`.

Windows also charges commit for the engine's GPU allocations (about 27 GiB at startup, never resident
in RAM). The commit limit is RAM plus the pagefile, so a small pagefile can force a spill while RAM
is still free.

Tested: 8 agents with ~110K tokens each (880K tokens of context) for 3 turns on 4 lanes, on a PC
with **96 GB RAM** and a 14 GB pagefile. All 24 turns were served, at least 33.8 GB of RAM stayed
free, and 4 GiB of idle history spilled when commit ran short. The run used
`--host-kv-mib 20000`, which retains about four of the eight histories; the others were prefilled
again on their next turn. Data: [2026-10-20 agents](../bench/engine-compare/data/2026-10-20/README.md).

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
