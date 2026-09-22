# VoidInfer

[![Windows integrity](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml/badge.svg)](https://github.com/The8Darkness/voidinfer/actions/workflows/windows-integrity.yml)

VoidInfer is an experimental native Windows C++20/CUDA inference engine focused on Qwen3.8-27B,
Blackwell (`sm_120a`), EXL3 weight execution, and exact-state speculative decoding research. It is
an engineering project—not a packaged general-purpose runtime—and deliberately keeps qualified
serving behavior separate from benchmark-only research routes.

## What works today

- Native model loading, CUDA execution, prefill/decode, and OpenAI-compatible HTTP serving.
- A qualified greedy-text EXL3 slice at physical concurrency 1; bounded C2 evidence exists for
  specific configurations. Positive-temperature sampling and EXL3 media are not broadly qualified.
- A same-weight FP16 device-KV DFlash2 research harness with exact token/state/tap/ring gates.
- Host-backed state, rollback, resource ownership, and lifecycle tests for experimental cache and
  speculative paths.
- Default-off kernels and cache experiments retained with explicit dispositions rather than being
  presented as supported defaults.

The public server and the native DFlash2 benchmark are distinct execution paths. A benchmark result
does not imply that the same optimization is wired into HTTP serving.

## Current qualified benchmark

R608 is the dated full-workload authority for the native same-weight ordinary-FP16-device-KV
DFlash2 C1 harness. Each row is one grouped arm: three sequential requests, a 4,096-token prefix,
128 useful outputs per request, and 384 useful outputs total. Wall time includes context creation as
defined by the cold/reused arm, acquire, decode, release, intermediate-root retirement, and final
lane teardown; it excludes model-process startup and is not client-visible TTFT.

| Fixture | Arm | Group wall (ms) | Useful output tok/s |
| --- | --- | ---: | ---: |
| Code | Cold | 8,504.60 | 45.15 |
| Code | Reused | 8,028.82 | 47.83 |
| Prose | Cold | 14,999.70 | 25.60 |
| Prose | Reused | 14,431.40 | 26.61 |

The receipt recorded 2,822 passing checks and 12 exact run rows. This is scoped replay evidence,
not universal numerical equivalence, quality qualification, Mia parity, or a 70/200/3000 target
claim. Performance runs use `VOIDINFER_PARITY_DFLASH_TIMING=0`; the phase-timing mode inserts
synchronization and is attribution-only. See [evidence/r608](evidence/r608/README.md) and
[benchmark methodology](docs/benchmarking.md).

## Build and try it

Requirements are Windows 11 x64, Visual Studio 2022 Build Tools, CUDA 13.1+, CMake 3.28+, Ninja,
and the pinned vcpkg dependencies described in [build and run](docs/build-and-run.md). The build is
intentionally restricted to `CMAKE_CUDA_ARCHITECTURES=120a`.

```powershell
cmd /c "call C:\BuildTools\Common7\Tools\VsDevCmd.bat -arch=amd64 && cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON"
cmake --build build --target ninfer-cli ninfer-serve

$env:VOIDINFER_MODEL = 'C:\models\qwen3_8_27b.ninfer'
& .\build\apps\ninfer-cli.exe $env:VOIDINFER_MODEL --prompt 'Write one sentence about CUDA.'
```

Model weights are not included. Treat commands as the supported clean-checkout procedure; the
September 22 publication validation boundary is recorded in [current status](docs/current-status.md).

## Reading paths

For a quick review: [current status](docs/current-status.md),
[engineering case study](docs/engineering-case-study.md), and
[interview/demo guide](docs/interview-guide.md).

For implementation work: [architecture](docs/architecture.md),
[build and run](docs/build-and-run.md), [benchmarking](docs/benchmarking.md),
[limitations](docs/limitations.md), and [maintenance](docs/maintenance.md).

## Provenance and license

VoidInfer descends from the NInfer codebase and a Windows adaptation. The repository also contains
adapted EXL3 utility code under `src/exl3/mia_exllamav3`, which retains its own license notice.
Direct dependencies, adapted code, research references, and original integration work are separated
in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md). GitHub fork metadata is not used as an authorship claim.

The repository-level license remains Apache-2.0. Model weights and third-party components retain
their own terms.
