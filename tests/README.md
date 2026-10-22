# Tests

The tests protect the EXL3 engine's numerics and state transitions, its resource and lifetime
contracts, the Qwen frontend and the external OpenAI/Anthropic protocol behavior. Verification
principles are in [`../AGENTS.md`](../AGENTS.md).

## Organization

| Group | Tests | What they protect |
|---|---|---|
| EXL3 model numerics | `voidinfer_exl3_{safetensors,single_linear,cuda_linear,full_attention_layer,gdn_layer,full_model}_test`, `voidinfer_exl3_large_down_oracle`, `voidinfer_exl3_projection_benchmark` | weight framing and trellis decode, linears, attention, GDN and the full model against independent oracles |
| DFlash2 | `voidinfer_exl3_dflash2_{minimal,state,xctx,xctx2,wrap,accept}_test` | draft execution, ring state, exact accept/commit/rollback; `accept` is the exact DFlash2 oracle |
| L0 OSCAR / KV tiers | `voidinfer_exl3_oscar_*`, `voidinfer_oscar_history_attention_test`, `voidinfer_oscar_int2_g128_test`, `voidinfer_kv_cache_test` | rotation assets, history attention, INT2 codec (golden from `tools/oscar/generate_d2_1_int2_golden.py`) |
| Engine | `voidinfer_exl3_engine_device_test`, `voidinfer_exl3_greedy_device_test`, `voidinfer_exl3_recurrent_pool_test` | the public Engine on the real model, device greedy selection, pinned recurrent pools |
| Host contracts (no CUDA linkage) | `voidinfer_exl3_{resource_inventory,retirement_state,state_machine_scenarios,owner_ledger_reference,sampling_*,sampled_*,candidate_contracts,environment_options,verifier_horizon_policy,…}_test` | resource accounting, retirement, sampling policy and transactions, environment switch parsing |
| Shared Ops | `voidinfer_{add_bias,gelu,residual_add,rmsnorm,gated_rmsnorm,layer_norm,rope,vision_pos_embed}_test` | the Ops the Vision tower and gated RMSNorm launch, against FP64 oracles |
| Frontend | `voidinfer_qwen3_6_frontend_test`, `voidinfer_exl3_frontend_reference_test`, `voidinfer_media_decode_test`, `voidinfer_prepared_identity_*` | template rendering, media preprocessing, M-RoPE, prepared-prompt identity |
| Serving | `voidinfer_{openai_schema,responses_schema,response_store,anthropic_schema,tool_call_parser,serve_options,request_log,http_error_handler}_test`, `voidinfer_public_api_test` | protocol translation, Responses state and SSE, tool calls, CLI options, request log records |
| Core | `voidinfer_{device,tensor,arena,decode_graph}_test` | device primitives, arenas, graph RAII |

`research_exl3_*` are measurement tools (KV-tier capture, long-context NLL, VeriCache, overlap,
L0 agents), not regression tests; they apply the engine profile and skip without inputs.

## Build and run

```powershell
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120a -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Tests that need the model skip with exit code 77 unless the environment names it:

```powershell
$env:VOIDINFER_EXL3_TARGET_PATH  = 'C:\models\Qwen3.8-27B-exl3-6bpw-H6-V6'
$env:VOIDINFER_EXL3_DFLASH2_PATH = 'C:\models\Qwen3.8-27B-DFlash2-EXL3-5bpw'
ctest --test-dir build -R voidinfer_exl3_dflash2_accept_test --output-on-failure
```

Fixture-driven validators (`voidinfer_oscar_int2_g128_test`, `voidinfer_exl3_frontend_reference_test`)
also skip when run without their arguments. `VOIDINFER_OP_REPORT_STATS=1` makes the Op tests print
one `OP_ERROR_STATS` record per comparison (case, error, limit, ratio).
