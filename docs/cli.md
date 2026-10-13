# VoidInfer CLI

`build/apps/voidinfer` runs one request through the same Engine as the server: the EXL3 6 bpw
target, its DFlash2 draft (K=7), L0 OSCAR history and exact VeriCache. Build it as described in
[build and run](build-and-run.md).

## Text input

```powershell
.\build\apps\voidinfer.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 `
  --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw `
  --prompt "Summarize the difference between prefill and decode." --max-new 256
```

Exactly one of `--prompt` and `--messages` is required. Answer content streams to stdout;
reasoning, model loading, timings, throughput, GPU memory and speculative statistics go to stderr,
so the two can be redirected independently (`> answer.txt 2> run.log`).

Thinking is enabled by default. `--reasoning-effort low|medium|xhigh` selects an effort when the
model's chat template exposes one; `--no-thinking` renders a direct-response prompt (not combinable
with `--reasoning-effort`).

The Engine currently rejects requests that carry a thinking budget (`--thinking-budget`), stop
strings (`--stop`, `--reasoning-stop`) or a frequency penalty; stop token ids (`--stop-token-id`) are
supported. `--max-new` counts every committed token.

## Images and video

`--vision DIR` names a directory with the BF16 Vision tower tensors (`model.visual.*`, for example
the BF16 checkpoint) and enables image and video message parts. The tower streams its weights from
pinned host memory per request; it holds no resident VRAM.

```powershell
.\build\apps\voidinfer.exe --target $target --draft $draft --vision C:\models\Qwen3.8-27B-bf16 `
  --messages examples\cli\messages\image_chart.json --max-new 128
```

## Structured messages

`--messages` accepts a non-empty JSON message array or an object containing `messages` and an
optional `tools` array. Roles: `system`, `developer`, `user`, `assistant`, `tool`. Content is a
string or an ordered array of parts:

| Content type | Source field | Accepted source |
|---|---|---|
| text | `text` | string |
| image / image_url | `image` or `image_url` | local path, HTTP(S) URL, or base64 data URI |
| video / video_url | `video` or `video_url` | local path, HTTP(S) URL, or base64 data URI |

Assistant history may carry `reasoning_content` and `tool_calls`; a tool result uses role `tool`
and `tool_call_id`. Run message files from the repository root when they contain
repository-relative media paths. [`examples/cli/`](../examples/cli/) has text, image, video,
thinking and long-context inputs.

## Options

| Option | Meaning | Default |
|---|---|---:|
| `--target DIR` | EXL3 target model directory (required) | |
| `--draft DIR` | DFlash2 draft directory (required) | |
| `--vision DIR` | BF16 Vision tower directory; enables image/video parts | off |
| `--max-context N` | context ceiling, 2048..262144 | `32768` |
| `--prefill-chunk N` | text prefill chunk | `1024` |
| `--max-new N` | output-token limit | `128` |
| `--device N` | CUDA device index | `0` |
| `--no-thinking` | direct-response prompt rendering | thinking on |
| `--thinking-budget N` | thinking-token cap (currently rejected by the Engine) | unlimited |
| `--reasoning-effort low\|medium\|xhigh` | effort exposed by the chat template | template default |
| `--greedy` | exact argmax decoding (temperature 0) | off |
| `--temperature F`, `--top-p F`, `--top-k N`, `--min-p F` | sampler overrides | model/mode preset |
| `--seed N` | sampling seed | `0` |
| `--stop-token-id N` | repeatable stop token id | none |
| `--stop TEXT`, `--reasoning-stop TEXT` | stop strings (currently rejected by the Engine) | none |
| `--raw-output`, `--print-token-ids` | raw frontend stream; token ids in diagnostics | off |

Sampler defaults come from the model's thinking or non-thinking preset; each flag overrides one
field. The non-thinking preset is the model card's instruct preset (temperature 0.7, top-p 0.8,
top-k 20, presence penalty 1.5); the thinking preset comes from `generation_config.json`. The
presence penalty applies to the generated tokens; frequency penalties are not supported.

The Engine reserves weights (19.3 GiB), the L0 history codes and its runtime at startup; exact FP16
history lives in host memory (at least `--max-context` × 64 KiB). Run `voidinfer --help` for the
exact option contract.
