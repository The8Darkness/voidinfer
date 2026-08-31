# Qwen3.8-27B OSCAR-Q2/Q4 documentation

Start with the [project README](../README.md) to build VoidInfer and run the current DFlash2 plus
OSCAR-Q2/Q4 hierarchical VeriCache serving profile on an RTX 5090.

## User guides

| Document | Purpose |
| --- | --- |
| [CLI](cli.md) | text generation, chat history, sampling, and runtime options |
| [HTTP serving](serving.md) | OpenAI/Anthropic-compatible serving and the DFlash2 default |
| [Performance](performance.md) | OSCAR-Q2/Q4, DFlash2 context, speed, quality, and tier measurements |
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

The isolated `exp/hierarchical-vericache-20260830` branch is the active Qwen3.8 NVFP4 research track
for OSCAR-Q2 L0, OSCAR-Q4 L1, 16-bit L2/L3 residency. Live host-tier logit verification, NVMe
persistence, and the full quality matrix remain explicitly open until measured.
