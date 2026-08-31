# Qwen3.8-27B NVFP4 documentation

Start with the [project README](../README.md) to build VoidInfer and run the current DFlash2 plus
hierarchical VeriCache serving profile on an RTX 5090.

## User guides

| Document | Purpose |
| --- | --- |
| [CLI](cli.md) | text generation, chat history, sampling, and runtime options |
| [HTTP serving](serving.md) | OpenAI/Anthropic-compatible serving and the DFlash2 default |
| [Performance](performance.md) | NVFP4/DFlash2 context, speed, quality, and tier measurements |
| [CLI examples](../examples/cli/) | committed text, multimodal, thinking, and long-context inputs |

The executable `--help` output is the exact source for command-line option spelling and defaults.

## Current artifact

| Profile | Download | Model card |
| --- | --- | --- |
| `nvfp4` | [Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | [NVFP4 model card](../model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) |

The registered artifact is the Qwen3.8-27B NVFP4 base model. The local experimental DFlash2 artifact
adds the DFlash2 proposal weights and is documented in the project README; it is not yet a public
download.

## Optimization and validation records

- [Project README](../README.md) — current default, exact KV hierarchy, and published results.
- [Performance report](performance.md) — reproducible RTX 5090 measurements and open quality gates.
- [Experiment registry](../EXPERIMENTS.md) — accepted, neutral, rejected, and pending optimization claims.
- [Current project state](../PROJECT_STATE.md) — hardware, artifact identity, and milestone history.
- [Qwen3.8-27B artifact contract](maintainer/qwen3.8-27b-artifact.md) — storage format and tensor mapping.
- [HTTP serving](serving.md) — runtime flags and endpoint behavior.

The isolated `exp/hierarchical-vericache-20260830` branch is the active NVFP4 research track for
hierarchical L0/L1/L2/L3 residency. Host-tier verification, NVMe persistence, and the full quality
matrix remain explicitly open until measured.
