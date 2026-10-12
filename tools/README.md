# VoidInfer maintainer tools

`tools/` holds maintainer workflows; normal use starts at the [project README](../README.md). Run
Python tools from the repository root with Python 3.11+. The benchmark harness lives in
[`bench/engine-compare/`](../bench/engine-compare/README.md), not here.

| Task | Location |
|---|---|
| Exercise a running server's OpenAI/Anthropic surfaces | [`smoke/serve_contract.py`](smoke/serve_contract.py) |
| Protocol clients (Chat, Responses, Messages; SSE parsing) | [`voidinfer_serve/`](voidinfer_serve/), [`streaming_http/`](streaming_http/) |
| OSCAR rotation calibration, QKV capture and INT2 golden fixtures | [`oscar/`](oscar/README.md) |
| Repository cleanup and host-only contract checks | [`maintenance/`](maintenance/) |
| Sustained device-memory bandwidth | [`hbm_bandwidth_probe.cu`](hbm_bandwidth_probe.cu) (standalone `nvcc` build) |
| Model-file read throughput | [`model_read_probe.cpp`](model_read_probe.cpp) (standalone) |

## Serving smoke

After starting `voidinfer-serve` (with `--model-id qwen3.8-27b`) in another terminal:

```powershell
python -m tools.smoke.serve_contract --base-url http://127.0.0.1:8080 --model qwen3.8-27b
```

The client exercises OpenAI Chat/Responses, Anthropic, streaming, usage, multimodal and tool-call
responses against the resident process. Add `--skip-media` for a server started without `--vision`.
