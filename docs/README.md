# Qwen3.8-27B documentation

Start with the [project README](../README.md) to build VoidInfer, download a Qwen3.8-27B artifact,
and run the CLI or HTTP server.

## User guides

| Document | Purpose |
| --- | --- |
| [CLI](cli.md) | text, chat history, image/video input, output streams, sampling, MTP, and runtime options |
| [HTTP serving](serving.md) | OpenAI Responses/Chat Completions, Anthropic Messages, state, streaming, token counting, and authentication |
| [Performance](performance.md) | Qwen3.8-27B throughput, context, MTP measurements, and reproduction commands |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, long-decode, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Qwen3.8-27B artifacts

| Profile | Download | Versioned model card |
| --- | --- | --- |
| `groupwise-int` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | [model card](../model-cards/Qwen3.8-27B-NInfer/README.md) |
| `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |

Both profiles resolve the registered `qwen3_8_27b` target. The NVFP4 profile is the current
single-GPU optimization target.

## Optimization and validation records

- [Experiment registry](../EXPERIMENTS.md) — accepted, rejected, and pending optimization claims.
- [Current project state](../PROJECT_STATE.md) — hardware, artifact identity, baseline evidence,
  and milestone history.
- [Qwen3.8-27B artifact contract](maintainer/qwen3.8-27b-artifact.md) — storage formats,
  tensor inventory, and source mapping.
- [Engine architecture](maintainer/engine-architecture.md) — execution ownership and request
  lifecycle.
- [Resource and context-cache contract](maintainer/resource-scheduling-and-context-cache.md) —
  materialization, continuation, checkpoint, and cache ownership.
- [Paged KV cache](maintainer/paged-kv-cache.md) — page layout, dtype contracts, and capacity.
- [Op development](maintainer/op-development.md) — correctness, ownership, and performance gates.
- [ReplaySSM GDN reference](maintainer/replayssm-gdn.md) — Qwen3.8 GDN execution details.
- [Linear benchmark contract](maintainer/linear-benchmark.md) — registered kernel benchmark suites.

The source tree contains compatibility code for older registered targets, but those targets are not
part of the public Qwen3.8-27B result set documented here.
