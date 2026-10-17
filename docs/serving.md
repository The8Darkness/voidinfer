# HTTP serving

`build/apps/voidinfer-serve` loads the EXL3 target and its DFlash2 draft and exposes OpenAI- and
Anthropic-compatible HTTP endpoints over one resident VoidInfer Engine: DFlash2 `K=7`, L0 OSCAR
history on the device, exact FP16 history in host memory and exact VeriCache on every published
token.

## Start the server

```powershell
.\build\apps\voidinfer-serve.exe --target C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6 `
  --draft C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw `
  --host 127.0.0.1 --port 8080 --model-id qwen3.8-27b --max-context 65536 --max-concurrency 2
```

The engine applies its benchmarked kernel and route profile at startup
(`src/exl3/engine_profile.cpp`); no environment variables are needed. The startup log reports
lanes, context, weights, runtime and free memory, host KV, prefix capacity, the speculative profile
and whether Vision is on. When `--model-id` is omitted, the server advertises the target's weights id
(`SC_6.00bpw_H6_V6` for the 6 bpw package).

Vision is off by default and media requests fail with HTTP 400 `vision_disabled`. `--vision DIR`
names a directory with the BF16 Vision tower (`model.visual.*`, for example the BF16 checkpoint).
The tower streams its weights from pinned host memory per request and holds no resident VRAM.
Media prompts neither reuse nor publish prefix roots. A later request cannot enable a capability
omitted at startup.

## Endpoints

| Method and path | Behavior |
|---|---|
| `GET /health` | process health |
| `GET /v1/models` | configured OpenAI model alias |
| `GET /v1/models/{id}` | lookup of the configured alias |
| `POST /v1/chat/completions` | OpenAI-style chat generation |
| `POST /v1/responses` | OpenAI Responses Core generation, state, typed Items, and SSE |
| `POST /v1/responses/input_tokens` | Responses prompt-token count without generation |
| `GET /v1/responses/{id}` | retrieve a locally stored terminal Response |
| `DELETE /v1/responses/{id}` | delete a locally stored Response |
| `GET /v1/responses/{id}/input_items` | list that Response's normalized input Items |
| `POST /v1/messages` | Anthropic-style message generation |
| `POST /v1/messages/count_tokens` | checkpoint-native expanded input-token count |

## OpenAI Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "Answer concisely."},
      {"role": "user", "content": "What is speculative decoding?"}
    ],
    "max_tokens": 128
  }'
```

The endpoint supports:

- `system`, `developer`, `user`, `assistant`, and `tool` history;
- string content and ordered text, `image_url`, and `video_url` parts;
- `max_completion_tokens` and the legacy `max_tokens` spelling;
- `temperature`, `top_p`, `top_k` (at most 20 candidates are kept: `0` or values above 20 mean 20),
  `min_p`, `presence_penalty` (OpenAI semantics over the generated tokens, also with temperature 0)
  and a nonnegative `seed`; `frequency_penalty` is parsed but the Engine rejects non-zero values;
- stop strings are parsed but currently rejected by the Engine;
- non-streaming responses and server-sent event streams;
- `stream_options.include_usage`;
- function tools, tool choices, assistant tool-call history, and tool-result messages;
- the top-level `reasoning_effort` field;
- the `enable_thinking` extension;
- `chat_template_kwargs.preserve_thinking` and the top-level `preserve_thinking` alias.

The request `model` must equal the public model ID: the target's weights id by default, or the
explicit `--model-id` override. Reasoning is returned separately as `reasoning_content`; answer
text remains in `content`.

Message roles retain their input order through schema translation. The Qwen family frontend maps
both `system` and `developer` to system-class ChatML blocks at their original positions; it does not
move later instructions to the beginning of the conversation. A leading instruction keeps the
chat template's existing tool/reasoning-instruction composition.

At startup, the server resolves prompt capabilities from the target directory's chat template. It
does not infer them from the request's `model` field. A recognized effort-capable template exposes `low`,
`medium`, and `xhigh`; omitting effort uses that template's declared default. An explicit effort
not exposed by the loaded template returns HTTP 400 with code
`reasoning_effort_not_supported` before prompt preparation.

`--default-thinking-budget N` sets a process default thinking cap for thinking-enabled requests.
The EXL3 Engine currently rejects requests that carry a thinking budget, so leave it unset.

For Chat Completions, `reasoning_effort: "none"` disables thinking. `low`, `medium`, and `xhigh`
select the corresponding template effort when available. The other OpenAI protocol values
`minimal`, `high`, and `max` are parsed but rejected when the loaded template does not expose them.
`enable_thinking` controls the same new-turn thinking switch; a contradictory combination with
`reasoning_effort` returns `conflicting_template_option`.

`preserve_thinking` controls whether reasoning from closed assistant turns remains in later
prompts (each closed turn keeps its `<think>` block, empty when it had no reasoning). It defaults
to the server setting, which is on, as in the model's chat template, unless `--no-preserve-thinking`
is used. If both OpenAI spellings are present they must carry the same boolean value. Unknown
non-null `chat_template_kwargs` are rejected.

Streaming begins with an assistant-role chunk, sends separate reasoning and content deltas, then a
finish-reason chunk and `[DONE]`. When `stream_options.include_usage` is true, a final empty
`choices` chunk contains completed usage.

### Multimodal request

Start the server with `--vision` before sending media:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
        {"type": "text", "text": "Describe this image."}
      ]
    }],
    "max_tokens": 128
  }'
```

OpenAI image and video sources may be HTTP(S) URLs or base64 data URLs.

Text and media requests use one complete-prompt context contract. After chat-template rendering and
media-token expansion, the result must fit Engine `--max-context`. The current Vision runtime also
has a 32,768 merged-token envelope (131,072 raw patches); the effective Vision limit is therefore
`min(--max-context, 32768)`. There is no fixed image/video item-count limit: item count is admitted
through aggregate source-byte, decoded-pixel, raw-patch, Vision-token, and live-memory budgets.

Media cache misses run as independent decode → resize → BF16-pack tasks on a bounded host worker
pool. Prepared payloads are keyed by SHA-256 of the acquired bytes plus modality, so repeated media
in later requests reuses the exact immutable BF16 patch input; concurrent identical misses use one
single-flight build. `--media-cache-mib` bounds LRU-retained payloads, while
`--media-live-mib` bounds every cache-, request-, or runtime-referenced payload. Cache eviction does
not invalidate a request reference, and live bytes are returned only when the final reference is
released. A request-level preparation gate derived from the live limit prevents concurrent partial
builds from deadlocking the memory account.

An expanded prompt beyond `--max-context` returns HTTP 400 `context_length_exceeded`, including
the prepared token count and configured context ceiling. A media preprocessing resource rejection
returns HTTP 400 `media_budget_exceeded`. HTTP 413 `request_too_large` is reserved for a raw request
body that exceeds `--max-request-mib` before JSON parsing; it is not used for model-context or media
resource errors.

## OpenAI Responses Core

VoidInfer implements the typed-Item and semantic-event core of the OpenAI
[Responses API](https://developers.openai.com/api/reference/resources/responses/overview). It is intentionally not
advertised as full parity with OpenAI-hosted tools, durable cloud storage, background jobs,
Conversations, or compaction.

### Create a Response

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "instructions": "Answer concisely.",
    "input": "What is speculative decoding?",
    "max_output_tokens": 128,
    "store": true
  }'
```

The same endpoint works with OpenAI SDKs by replacing their base URL:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local-secret")
response = client.responses.create(
    model="qwen3.8-27b",
    instructions="Answer concisely.",
    input="What is speculative decoding?",
    max_output_tokens=128,
)
print(response.output_text)  # SDK helper derived from response.output
```

`output_text` is an SDK convenience property. It is not emitted as a top-level wire field; the
wire response contains typed `output` Items.

### Create request fields

| Field | Responses Core contract |
|---|---|
| `model` | required non-empty string; must equal the public model ID (weights id or `--model-id`) |
| `input` | required string or non-empty typed Item array |
| `instructions` | optional string, inserted before the reconstructed conversation for this request only |
| `previous_response_id` | optional ID of a retained local Response |
| `max_output_tokens` | integer at least `16`; default is `--default-max-tokens` |
| `stream` | boolean; `true` selects Responses SSE rather than a JSON body |
| `store` | boolean, default `true`; controls local retrieval and continuation state |
| `temperature` | finite number in `[0,2]` |
| `top_p` | finite number in `[0,1]` |
| `metadata` | at most 16 string pairs; keys at most 64 characters and values at most 512 |
| `reasoning.effort` | `none` disables thinking; `low`, `medium`, or `xhigh` selects an effort exposed by the loaded chat template; `minimal`, `high`, and `max` return `reasoning_effort_not_supported` for the registered templates |
| `chat_template_kwargs.preserve_thinking` | optional boolean controlling whether closed-turn reasoning remains in reconstructed prompts |
| `preserve_thinking` | top-level alias for the same option; conflicting values are rejected |
| `text.format` | omitted or `{"type":"text"}` only |
| `tools` | flat Responses function definitions; see below |
| `tool_choice` | `auto` or `none` |
| `parallel_tool_calls` | omitted or `true` |
| `truncation` | omitted or `disabled`; overlong input fails instead of silently dropping Items |
| `top_logprobs` | omitted or `0` |
| `service_tier` | omitted, `auto`, or `default`; the response reports `default` |
| `background` | omitted or `false` |
| `include` | omitted or an empty array |
| `stream_options` | omitted or `{"include_obfuscation":false}` |

Unknown top-level fields fail with `unknown_parameter`. Recognized but unsupported features fail
with a field-specific 400 error instead of being silently ignored.

### Input Item contract

String `input` is normalized to one user `message` with an `input_text` part. Array input accepts:

| Item | Supported form |
|---|---|
| `message` | roles `user`, `assistant`, `system`, and `developer`; string content or typed content array |
| `input_text` | message content part containing string `text` |
| `output_text` | assistant-message replay part containing string `text` |
| `input_image` | user-message part with HTTP(S) or data-URI `image_url`; detail omitted or `auto`; requires server `--vision` |
| `input_video` | VoidInfer extension with HTTP(S) or data-URI `video_url`; requires server `--vision` |
| `reasoning` | raw replay Item with an empty `summary` and `reasoning_text` content parts |
| `function_call` | completed assistant call with optional `id`, and required `call_id`, `name`, and JSON-object string `arguments` |
| `function_call_output` | completed tool result with required `call_id` and string `output` |

Adjacent function-call Items are grouped into one assistant history turn. A reasoning Item attaches
to the following assistant message or function call. Input Item IDs are preserved when supplied and
generated otherwise; duplicate IDs fail.

System and developer message Items retain their positions in the input array. Top-level
`instructions` is represented as a leading developer turn for the current request; target-specific
role lowering occurs only in the Qwen family frontend.

`input_file`, `input_audio`, image `file_id`, non-`auto` image detail, reasoning summaries or
encrypted reasoning, message `phase`, and other Item/content types are not supported. HTTP media
URLs stored in a response chain are fetched again when that chain is continued; use data URIs when
the historical media bytes must be immutable.

### Function tools

Responses function definitions are flat rather than Chat Completions' nested `function` object:

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "Get current weather",
  "parameters": {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"]
  },
  "strict": false
}
```

The server renders these definitions in the Qwen prompt and parses model output into separate
`function_call` output Items. Each output has a protocol Item `id` (`fc_...`) and a distinct
`call_id` (`call_...`). The client executes the function and sends a `function_call_output` Item in
a later request. VoidInfer does not execute functions or enforce JSON Schema through constrained
decoding, so `strict:true`, `tool_choice:required`, named tool choice, hosted tools, MCP tools, and
custom free-form tools are rejected.

### Response object and usage

A terminal wire response has `object: "response"`, one of `completed`, `incomplete`, or
`cancelled` in `status`, and a typed `output` array. It may contain:

- a `reasoning` Item containing raw `reasoning_text` and an empty summary;
- an assistant `message` containing an `output_text` part;
- one or more `function_call` Items.

Ordinary model/string stops produce `completed`. Output-token or context-capacity exhaustion
produces `incomplete` with `incomplete_details.reason: "max_output_tokens"`. Errors accepted after
an SSE response has started produce `response.failed`; validation and preparation errors remain
normal HTTP error responses.

Usage is checkpoint-native:

```json
{
  "input_tokens": 42,
  "input_tokens_details": {"cached_tokens": 17},
  "output_tokens": 12,
  "output_tokens_details": {"reasoning_tokens": 5},
  "total_tokens": 54
}
```

`input_tokens` includes the chat template and expanded media tokens. `cached_tokens` is the exact
checkpoint-proven prompt prefix reused by Engine. `output_tokens` is the count of accepted generated token
IDs, including a withheld stop token when applicable. `reasoning_tokens` is counted in the Qwen
output decoder while accepted tokens are still in the reasoning channel; it is not estimated by
re-tokenizing decoded text.

### Responses streaming

Set `stream:true` for semantic Server-Sent Events. Every frame uses both the SSE event name and a
matching JSON `type`, and every JSON event has a monotonically increasing `sequence_number`:

```text
event: response.output_text.delta
data: {"type":"response.output_text.delta","sequence_number":7,...}

```

The normal lifecycle is:

1. `response.created`, then `response.in_progress`;
2. `response.output_item.added` and `response.content_part.added`;
3. zero or more `response.reasoning_text.delta` or `response.output_text.delta` events;
4. matching `*.done`, `response.content_part.done`, and `response.output_item.done` events;
5. exactly one `response.completed`, `response.incomplete`, or `response.failed` terminal event.

Function arguments use `response.function_call_arguments.delta` and `.done`. IDs, output indices,
and content indices remain stable, and concatenated deltas equal the terminal Item. Responses SSE
does not emit the Chat Completions `[DONE]` sentinel. With tools enabled, ordinary answer text still
streams immediately; only an ambiguous `<tool_call>` suffix or the structured tool region is held.
Malformed tool markup is flushed back as ordinary text without losing bytes.

### Local response state and resources

`store` defaults to `true`. Stored Responses live only in this server process and are bounded by an
LRU store. They are lost on restart and are not OpenAI's durable cloud retention service.

`previous_response_id` reconstructs the complete stored input/output Item history before the new
input. The current `instructions` value is placed first but is not saved into the continuation
context, matching the Responses rule that previous top-level instructions do not carry forward.
Function definitions are request configuration rather than conversation Items and must be sent
again on tool-result turns. The reconstructed prompt follows the ordinary Engine path, so compatible
checkpoint reuse applies naturally.

A stored Response also retains its resolved `preserve_thinking` value. A child which omits the
field inherits the parent value. An explicit different value creates a new semantic branch; prompt
rendering and identity still determine reuse. Changing the boolean alone never invalidates an exact
checkpoint already proved compatible by the model runtime.

For Engine-local reuse, a stored root Response receives one bounded session key derived from its
response ID, and every `previous_response_id` child inherits that key. `store:false` roots remain
anonymous; a `store:false` child may read its inherited session checkpoint but does not replace the
stored chain's latest endpoint. Response-store eviction or deletion removes the HTTP object, not an
independently retained Engine checkpoint; the latter remains bounded by the Engine's own retention
and pressure policy. No session key or cache marker is added to the HTTP schema.

Resource behavior:

| Endpoint | Contract |
|---|---|
| `GET /v1/responses/{id}` | returns the stored terminal object, or 404 `response_not_found` |
| `DELETE /v1/responses/{id}` | removes public retrieval and returns `response.deleted`; descendant contexts already retained by other Responses remain usable |
| `GET /v1/responses/{id}/input_items` | returns normalized Items supplied to that request; supports `after`, `limit` `1..100` (default `20`), and `order` `asc|desc` (default `desc`) |
| `POST /v1/responses/{id}/cancel` | explicitly fails because background execution is unsupported |
| `POST /v1/responses/compact` | explicitly fails with `compaction_not_supported` |

`store:false` Responses cannot be retrieved or used as `previous_response_id`. LRU eviction and
explicit deletion also make an ID unavailable. A single Response larger than the configured store
capacity fails with `response_store_capacity_exceeded` rather than silently pretending it was
stored.

### Responses input token count

`POST /v1/responses/input_tokens` accepts exactly `model` and `input`, performs the same typed Item,
template, and media expansion, and does not run generation:

```bash
curl http://127.0.0.1:8080/v1/responses/input_tokens \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","input":"Count this prompt."}'
```

```json
{"object":"response.input_tokens","input_tokens":11}
```

Unsupported Create fields include Conversations, prompt templates, context management, hosted
moderation, prompt-cache controls, safety/user identifiers, Structured Outputs/JSON mode,
non-empty `include`, background execution, compaction, files/audio, and OpenAI-hosted/MCP/custom
tools. These are compatibility boundaries, not silently accepted placeholders.

## Anthropic Messages

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Explain prefix reuse in one sentence."}
    ]
  }'
```

The endpoint supports top-level system text, ordered mid-conversation system messages,
user/assistant history, text and image blocks, thinking blocks, tool-use history, tool results,
client-defined tools, non-streaming responses, and Anthropic SSE events.
Mid-conversation system messages remain at their `messages` array position and are not merged into
the top-level system instruction. A system section must follow a user/tool-result message and be
final or immediately precede an assistant message; it cannot interrupt a tool-use/tool-result pair.
Consecutive system messages remain separate ordered turns.

Anthropic ephemeral `cache_control` on top-level system text blocks and client tool definitions
marks a shared stable-prefix boundary. Because the Qwen prompt renders tools before the leading
system instruction, the server retains the last marked tool boundary unless a later marked system
boundary exists; when several system blocks are marked, it retains the last one. The boundary is a
retention hint rather than a forced hit: reuse still requires exact rendered-token compatibility.
Ephemeral `cache_control` on the final content block of a user or assistant message marks the
normalized message boundary as a private long anchor; a non-final message-block breakpoint is
rejected because it cannot be represented as an exact Qwen message frontier.

`thinking.type: "disabled"` disables thinking; other supported values enable it.
The independent top-level `preserve_thinking` boolean controls closed-turn history and otherwise
uses the server default.

Anthropic `output_config.effort` accepts the protocol values `low`, `medium`, `high`, `xhigh`, and
`max`. The value is then checked against the loaded chat template in the same way as the OpenAI
endpoints; the registered effort-capable template exposes `low`, `medium`, and `xhigh`. Combining
an effort with `thinking.type: "disabled"` is rejected as contradictory.

Anthropic's `model` field is treated as a response label and does not select the loaded model.

`POST /v1/messages/count_tokens` uses the model's tokenizer, chat template, and media expansion
without running GPU generation:

```bash
curl http://127.0.0.1:8080/v1/messages/count_tokens \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [{"role": "user", "content": "Count this prompt."}]
  }'
```

## Authentication and CORS

Pass `--api-key VALUE` to require the same value as an OpenAI bearer token or Anthropic
`x-api-key` header. `GET /health` and CORS preflight requests remain unauthenticated.

```bash
curl http://127.0.0.1:8080/v1/models \
  -H 'Authorization: Bearer local-secret'
```

`--cors` adds permissive browser CORS headers. It is disabled by default.

## Server options

| Option | Meaning | Default |
|---|---|---:|
| `--target DIR` | EXL3 target model directory (required) | |
| `--draft DIR` | DFlash2 draft directory (required) | |
| `--vision DIR` | BF16 Vision tower directory; enables image/video input | off |
| `--dual-manifest FILE` | verified C:/D: mirror manifest for dual-drive weight reads | `<target>/.voidinfer/dual-load.json` if present |
| `--host H` | listen address | `127.0.0.1` |
| `--port N` | listen port | `8080` |
| `--api-key KEY` | required bearer or `x-api-key` value | unset |
| `--model-id ID` | public OpenAI model alias | target weights id |
| `--device N` | CUDA device index | `0` |
| `--max-context N` | context ceiling of each request, `2048..262144` | `32768` |
| `--max-concurrency N` | lanes (concurrent requests), `1..8` | `1` |
| `--max-pending-requests N` | additional requests allowed to wait for admission | `16` |
| `--pending-timeout-ms N` | maximum preparation-plus-admission wait | `30000` |
| `--prefill-chunk N` | text-prefill chunk | `1024` |
| `--host-kv-mib N` | exact FP16 history in host memory (retained conversation roots) | `max(8192, max-context × 64 KiB, 40% of physical memory)` |
| `--host-state-slots N`, `--device-state-slots N` | host/device checkpoint slots | `8` / `max-concurrency` |
| `--max-private-continuations N` | conversation roots kept for follow-up turns (with shared prefixes at most 64) | `max(2 × max-concurrency, 32)` |
| `--max-shared-prefixes N` | cross-conversation shared-prefix roots kept for reuse (not evicted by conversation turnover) | `max(max-concurrency, 16)` |
| `--max-long-anchors-per-continuation N`, `--max-cache-markers-per-request N` | continuation descriptor bounds | `2` / `4` |
| `--no-prefix-reuse` | serve every request without reusing a cached prefix | reuse on |
| `--max-request-mib N` | body-size limit before JSON parsing | `384` |
| `--media-cache-mib N` | LRU-retained prepared BF16 media payloads; `0` disables retention | `1024` |
| `--media-live-mib N` | all live prepared BF16 media payloads | `2048` |
| `--media-preprocess-threads N` | media preprocessing workers; `0` selects up to 16 | `0` |
| `--default-max-tokens N` | output limit when a request omits one | `262144` (clamped to the remaining context) |
| `--default-thinking-budget N` | thinking cap for thinking-enabled requests (currently rejected by the Engine) | unset |
| `--no-thinking` | disable thinking by default | thinking on |
| `--no-preserve-thinking` | drop closed-turn reasoning from later prompts | keep (template default) |
| `--temperature F`, `--top-p F`, `--top-k N`, `--min-p F` | process-level sampler overrides | model preset |
| `--seed N` | fixed seed when a request omits one | fresh random seed |
| `--greedy` | force exact argmax for all requests | off |
| `--request-log-jsonl FILE` | append full-precision server/request records | disabled |
| `--response-store-max-records N`, `--response-store-max-mib N` | local Responses store bounds | `1024` / `256` |
| `--log-stats-interval-ms N` | aggregate throughput report interval; `0` disables it | `5000` |
| `--cors` | permissive browser CORS headers | off |
| `--webui`, `--webui-dir DIR` | download and serve the llama.cpp web UI at `/` | off / `<target>/webui` |

Sampler defaults follow the model card: thinking requests take the target's `generation_config.json`
(temperature 1.0, top-p 0.95, top-k 20), non-thinking requests the instruct preset (temperature 0.7,
top-p 0.8, top-k 20, presence penalty 1.5). Process flags override them, request fields override
process flags, and `--greedy` finally forces temperature `0` (a preset or requested presence penalty
still applies). Sampling runs on device with VeriCache-coupled selection: drafting, decode rounds and
the exact verification draw with the same noise and the same penalties.

Run `voidinfer-serve --help` for the exact option contract.

## Structured request log

`--request-log-jsonl FILE` enables the machine-readable measurement log. The server opens `FILE`
in append mode and flushes every event, so successive runs may share one campaign file. The parent
directory must already exist; failure to open the file aborts startup.

```powershell
.\build\apps\voidinfer-serve.exe --target $target --draft $draft `
  --request-log-jsonl profiles\bench\run\server.requests.jsonl
```

Every line is one `voidinfer_serve_request_log` schema-v18 JSON object. All events carry
`timestamp_unix_ms` and a process-unique `server_instance_id`; request IDs are monotonic only within
that server instance. Successful request-start records include request-scoped acquisition,
media-preprocessing wall/work, tokenizer, cache hit/miss/single-flight, and payload-size fields;
they do not infer request behavior from process-global counter deltas.

| Event | Contents |
|---|---|
| `server_start` | `server` settings, `model` package (target/draft/vision directories, identity, weights id, bytes read, load seconds), resolved `engine` and context-cache capacities, sampler defaults plus process overrides, Device memory arenas, Host State/KV capacity and occupancy, CUDA/GPU environment, and redacted argv |
| `request_start` | protocol, resolved sampler and seed, thinking mode and optional budget, Responses semantic-change flag, output budget, stream/message/tool shape |
| `request_rejected` | parsed request shape, media-item count, `phase: "prepare"`, and the exact HTTP status/type/code/parameter/message for a synchronous preparation rejection |
| `request_done` | finish reason, prompt/completion/cache/computed-prefill tokens, prefix reuse path, thinking counters, unrounded request-stage seconds, and complete speculative-decoding counters |
| `request_error` | the resolved request configuration and generation error message |
| `throughput` | interval token/decode/context-cache pressure counter deltas, authoritative worker Host-work deltas, current scheduler/resource gauges, and decode-round batch statistics |

`request_done.timings_seconds` contains `prepare`, `ttft`, `vision`, `prefill`, `decode`, and `total`
as full-precision JSON numbers. Its `speculative` object contains `backend` (`dflash2`), `draft_window`, `rounds`,
`drafted_tokens`, `accepted_tokens`, `fallback_steps`, and `accepted_per_position`. Rates can be
derived downstream from raw token counts and seconds instead of rounded stderr strings.

The JSONL file contains no generated response text and never records an API-key value; `argv`
replaces that value with `<redacted>`. The existing stderr summaries remain available for operators
but are rounded and are not the aggregation source. Console lines use local
`[YYYY-MM-DD HH:MM:SS.mmm] [level]` timestamps. OpenAI Responses, OpenAI Chat, and Anthropic
generation requests receive a request ID when they enter synchronous preparation. Successful
preparation produces `request_start`; a preparation failure produces `request_rejected` without a
matching start. Later generation failures produce `request_error`. Schema/model validation
rejections before preparation and token-count-only calls are not measurement requests and do not
receive request IDs.

By default the server also reports aggregate activity every five seconds: computed prefill tokens
(excluding reused prefixes), committed decode tokens (verified output, not drafts), decode rounds,
running and waiting requests, and prefix-reuse deltas. The JSONL `throughput` event carries the same
values as raw counters.

## Execution behavior

The server owns one resident Engine with `--max-concurrency` lanes (`1..8`). Every decode round
batches all decode-ready lanes into one DFlash2 round: the draft proposes seven tokens per lane and
one target pass verifies every lane's rows. A lane is admitted only when its L0 history codes fit in
free VRAM; a request that does not fit waits in FIFO order instead of oversubscribing device memory.
Exact verification runs asynchronously on a second exact context (the exact line), shared by all
lanes: every request with one lane, sampled requests with several lanes. Other requests verify in
exclusive passes.

`--max-pending-requests` bounds the requests waiting behind the active lanes. The total request
capacity is `max_concurrency + max_pending_requests`, including requests still in CPU/media
preparation and completed results whose response has not been released. A full capacity returns
HTTP 429 `server_overloaded`. The absolute `--pending-timeout-ms` deadline starts before preparation,
covers media acquisition and Engine FIFO waiting, and returns HTTP 503 `request_queue_timeout` if
admission does not occur in time.

Each lane keeps its L0 INT4 history codes, hot rows and exact sink/recent window on the device and its
exact FP16 history in a plane set in host memory (`--host-kv-mib`). A finished conversation's plane
set stays bound to its root, so a follow-up turn resumes by remapping that set behind the lane's
window instead of copying history. Idle sets are losslessly packed and, under host memory pressure,
spilled to disk. Text prompts reuse the longest compatible prefix root (conversation continuations
and up to `--max-shared-prefixes` shared prefixes); the completion log reports the reused token
count as `cache=` and the reuse path. Media prompts are served without prefix reuse.

Speculative decoding does not change protocol output shapes, stop behavior or usage accounting:
only tokens verified against exact history are published. Output-limit and context-capacity
finishes map to `length` / `max_tokens`; model stops map to `stop` / `end_turn`.

Function tools are rendered into the model prompt and generated calls are parsed into protocol
responses. VoidInfer does not execute tools and does not enforce client JSON Schema through constrained
decoding.

Prompt-token usage includes chat-template and expanded media tokens. Generated-token usage comes
from accepted output token IDs, including a stop token whose decoded text may be withheld.
