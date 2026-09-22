find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if("${arg_LIBRARIES}" MATCHES "ninfer_engine|ninfer_serve|ninfer_media_decode")
    ninfer_copy_runtime_dlls(${name})
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Header-contract executables exercise only model-independent host state. Keep
# them independent of ninfer_core (and therefore CUDA runtime linkage) so they
# remain valid inputs to the strict CPU-only qualification allowlist.
function(ninfer_add_host_contract_test name)
  cmake_parse_arguments(arg "NEEDS_CUDA_HEADERS" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_NEEDS_CUDA_HEADERS)
    target_include_directories(${name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  endif()
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_exl3_safetensors_test
  SOURCES test_exl3_safetensors.cpp
  LIBRARIES ninfer_exl3_format)
ninfer_add_test(ninfer_exl3_mtp_manifest_test
  SOURCES test_exl3_mtp_manifest.cpp
  LIBRARIES ninfer_exl3_format)
ninfer_add_test(ninfer_exl3_mtp_runner_test
  SOURCES test_exl3_mtp_runner.cpp
  LIBRARIES ninfer_exl3_format)
ninfer_add_host_contract_test(ninfer_exl3_mtp_proposal_economics_test
  SOURCES test_exl3_mtp_proposal_economics.cpp)
ninfer_add_test(ninfer_exl3_single_linear_test
  SOURCES test_exl3_single_linear.cpp
  LIBRARIES ninfer_exl3_format)
set_tests_properties(ninfer_exl3_single_linear_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_cuda_linear_test
  SOURCES test_exl3_cuda_linear.cpp
  LIBRARIES ninfer_exl3_cuda ninfer_exl3_format)
set_tests_properties(ninfer_exl3_cuda_linear_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_k4_linear_test
  SOURCES test_exl3_k4_linear.cpp
  LIBRARIES ninfer_exl3_cuda ninfer_exl3_format)
set_tests_properties(ninfer_exl3_k4_linear_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_projection_benchmark
  SOURCES test_exl3_projection_benchmark.cpp
  LIBRARIES ninfer_exl3_cuda ninfer_exl3_format)
set_tests_properties(ninfer_exl3_projection_benchmark PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_large_down_oracle
  SOURCES test_exl3_large_down_oracle.cpp
  LIBRARIES ninfer_exl3_cuda ninfer_exl3_format)
set_tests_properties(ninfer_exl3_large_down_oracle PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_full_attention_layer_test
  SOURCES test_exl3_full_attention_layer.cpp
  LIBRARIES ninfer_exl3_cuda ninfer_exl3_format)
set_tests_properties(ninfer_exl3_full_attention_layer_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_gdn_layer_test
  SOURCES test_exl3_gdn_layer.cpp
  LIBRARIES ninfer_exl3_gdn ninfer_exl3_format)
set_tests_properties(ninfer_exl3_gdn_layer_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_full_model_test
  SOURCES test_exl3_full_model.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_full_model_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_mtp_hidden_handoff_test
  SOURCES test_exl3_mtp_hidden_handoff.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_mtp_hidden_handoff_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_dflash2_minimal_test
  SOURCES test_exl3_dflash2_minimal.cpp
  LIBRARIES ninfer_exl3_dflash2)
set_tests_properties(ninfer_exl3_dflash2_minimal_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_dflash2_state_test
  SOURCES test_exl3_dflash2_state.cpp
  LIBRARIES ninfer_exl3_dflash2)
set_tests_properties(ninfer_exl3_dflash2_state_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_compatibility_test
  SOURCES test_exl3_oscar_compatibility.cpp
  LIBRARIES ninfer_exl3_text_model ninfer_ops)
set_tests_properties(ninfer_exl3_oscar_compatibility_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_asset_compatibility_test
  SOURCES test_exl3_oscar_asset_compatibility.cpp
  LIBRARIES ninfer_exl3_text_model ninfer_ops)
set_tests_properties(ninfer_exl3_oscar_asset_compatibility_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_adapter_test
  SOURCES test_exl3_oscar_adapter.cpp
  LIBRARIES ninfer_exl3_text_model ninfer_ops)
set_tests_properties(ninfer_exl3_oscar_adapter_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_eager_model_test
  SOURCES test_exl3_oscar_eager_model.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_oscar_eager_model_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_short_bench
  SOURCES test_exl3_oscar_short_bench.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_oscar_short_bench PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_context_sweep
  SOURCES test_exl3_oscar_context_sweep.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_oscar_context_sweep PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_profile
  SOURCES test_exl3_oscar_profile.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_oscar_profile PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_oscar_graph_test
  SOURCES test_exl3_oscar_graph.cpp
  LIBRARIES ninfer_exl3_text_model)
set_tests_properties(ninfer_exl3_oscar_graph_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ninfer_media_decode)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_host_contract_test(ninfer_admission_policy_test
  NEEDS_CUDA_HEADERS
  SOURCES test_admission_policy.cpp
          ${PROJECT_SOURCE_DIR}/src/runtime/engine/admission_policy.cpp)
ninfer_add_host_contract_test(ninfer_exl3_ready_work_test SOURCES test_exl3_ready_work.cpp)
ninfer_add_test(ninfer_exl3_public_media_qualification_test
  SOURCES test_exl3_public_media_qualification.cpp)
ninfer_add_test(ninfer_media_client_loopback_test
  SOURCES test_media_client_loopback.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_media_client_loopback_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
set_tests_properties(ninfer_media_client_loopback_test PROPERTIES DISABLED TRUE)
ninfer_add_test(ninfer_context_cost_test SOURCES test_context_cost.cpp)
ninfer_add_test(ninfer_resource_manager_test SOURCES test_resource_manager.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ninfer_add_test(ninfer_oscar_mixed_cache_layout_test
  SOURCES test_oscar_mixed_cache_layout.cpp
  LIBRARIES ninfer_core ninfer_ops)
ninfer_add_test(ninfer_oscar_mixed_cache_aging_test
  SOURCES test_oscar_mixed_cache_aging.cpp
  LIBRARIES ninfer_core ninfer_ops)
ninfer_add_test(ninfer_oscar_mixed_state_transitions_test
  SOURCES test_oscar_mixed_state_transitions.cpp
  LIBRARIES ninfer_core ninfer_ops)
ninfer_add_test(ninfer_oscar_mixed_attention_reference_test
  SOURCES test_oscar_mixed_attention_reference.cpp
  LIBRARIES ninfer_core ninfer_ops)
ninfer_add_test(ninfer_oscar_history_attention_test
  SOURCES test_oscar_history_attention.cpp
  LIBRARIES ninfer_core ninfer_ops)
set_tests_properties(ninfer_oscar_history_attention_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_oscar_mixed_gpu_attention_test
  SOURCES test_oscar_mixed_gpu_attention.cpp
  LIBRARIES ninfer_core ninfer_ops)
set_tests_properties(ninfer_oscar_mixed_gpu_attention_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_cli_options_test
  SOURCES test_cli_options.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp)
target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_hierarchical_vericache_test
  SOURCES test_hierarchical_vericache.cpp
  LIBRARIES ninfer_core)
set_tests_properties(
  ninfer_artifact_materialization_test
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_state_image_test
  SOURCES targets/qwen3_6/test_state_image.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_state_image_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_state_image_layout_test
  SOURCES targets/qwen3_6/test_state_image_layout.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_layout_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_context_store_test
  SOURCES targets/qwen3_6/test_context_store.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_context_store_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_context_store_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE ${NINFER_CUDART_TARGET})
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_oscar_capture_test
  SOURCES targets/qwen3_6_27b/test_oscar_capture.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_6_27b_oscar_capture_test PRIVATE ${NINFER_CUDART_TARGET})
set_tests_properties(ninfer_qwen3_6_27b_oscar_capture_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_oscar_runtime_test
  SOURCES targets/qwen3_6_27b/test_oscar_runtime.cpp
  LIBRARIES ninfer_engine)
target_link_libraries(ninfer_qwen3_6_27b_oscar_runtime_test PRIVATE ${NINFER_CUDART_TARGET})
set_tests_properties(ninfer_qwen3_6_27b_oscar_runtime_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_oscar_live_reference_validator
  SOURCES targets/qwen3_6_27b/test_oscar_live_reference_validator.cpp
  LIBRARIES ninfer_core ninfer_ops)
ninfer_add_test(ninfer_qwen3_6_27b_model_load_probe
  SOURCES targets/qwen3_6_27b/test_model_load_probe.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_model_load_probe PROPERTIES SKIP_RETURN_CODE 77)
# The full OSCAR validator is intentionally fail-closed, but it must not be
# benchmarked as a Debug scalar oracle. Keep these flags target-local so the
# rest of the Debug developer build retains its normal diagnostics.
if(MSVC)
  target_compile_options(ninfer_qwen3_6_27b_oscar_live_reference_validator PRIVATE
    /O2 /Ob3 /Oi /Ot /arch:AVX2 /fp:precise /UNDEBUG)
else()
  target_compile_options(ninfer_qwen3_6_27b_oscar_live_reference_validator PRIVATE
    -O3 -mavx2 -fno-fast-math -ffp-contract=off)
endif()
target_compile_definitions(ninfer_qwen3_6_27b_oscar_live_reference_validator PRIVATE
  NINFER_OSCAR_FULL_VALIDATION_OPTIMIZED=1)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE ${NINFER_CUDART_TARGET})
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_host_timing_test
  SOURCES test_host_timing.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Pure host-side qualification of the fixed-dimension non-negative fit and the exact coefficient
# quantization used by the context-cost calibrator. It deliberately links no Engine/CUDA target.
add_executable(ninfer_context_cost_measure_test
  test_context_cost_measure.cpp
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)
target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)
add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  scatter
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES ops/softmax_attention/main.cpp
          ops/softmax_attention/causal_cache.cpp
          ops/softmax_attention/plain_and_packed.cpp
          ops/softmax_attention/context.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES ops/test_sliding_window_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_oscar_kv_test
  SOURCES ops/test_oscar_kv.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_oscar_int2_g128_test
  SOURCES ops/test_oscar_int2_g128.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES ops/test_kv_cache_append.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_dflash2_test
  SOURCES ops/test_dflash2.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()
ninfer_add_test(ninfer_exl3_dflash2_xctx_test
  SOURCES test_exl3_dflash2_xctx.cpp
  LIBRARIES ninfer_exl3_dflash2)
set_tests_properties(ninfer_exl3_dflash2_xctx_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_dflash2_xctx2_test
  SOURCES test_exl3_dflash2_xctx2.cpp
  LIBRARIES ninfer_exl3_dflash2)
set_tests_properties(ninfer_exl3_dflash2_xctx2_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_dflash2_wrap_test
  SOURCES test_exl3_dflash2_wrap.cpp
  LIBRARIES ninfer_exl3_dflash2)
set_tests_properties(ninfer_exl3_dflash2_wrap_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_dflash2_accept_test
  SOURCES test_exl3_dflash2_accept.cpp
  LIBRARIES ninfer_exl3_dflash2)
target_compile_options(ninfer_exl3_dflash2_accept_test PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
set_tests_properties(ninfer_exl3_dflash2_accept_test PROPERTIES SKIP_RETURN_CODE 77)

# Explicitly invoked, test-only oracle tools. Not part of default build/CTest.
ninfer_add_test(ninfer_exl3_engine_lifecycle_test
  SOURCES test_exl3_engine_lifecycle.cpp
  LIBRARIES ninfer_engine CUDA::cudart)
target_include_directories(ninfer_exl3_engine_lifecycle_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_options(ninfer_exl3_engine_lifecycle_test PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/STACK:8388608>)
set_tests_properties(ninfer_exl3_engine_lifecycle_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_frontend_reference_test
  SOURCES test_exl3_frontend_reference.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_exl3_frontend_reference_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_exl3_recurrent_pool_test
  SOURCES test_exl3_recurrent_pool.cpp
  LIBRARIES CUDA::cudart Psapi)

ninfer_add_test(ninfer_exl3_vision_test
  SOURCES test_exl3_vision.cpp
  LIBRARIES ninfer_exl3_vision ninfer_engine ninfer_core CUDA::cudart)
set_tests_properties(ninfer_exl3_vision_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_exl3_prepared_request_test
  SOURCES test_exl3_prepared_request.cpp
  LIBRARIES ninfer_engine ninfer_exl3_dflash2 ninfer_exl3_vision CUDA::cudart)

# These binaries are fixture-driven validators/tools. Their required artifact
# paths come from dedicated qualification invocations, not default CTest.
set_tests_properties(
  ninfer_oscar_mixed_attention_reference_test
  ninfer_qwen3_6_27b_oscar_live_reference_validator
  ninfer_qwen3_6_27b_model_load_probe
  ninfer_oscar_int2_g128_test
  ninfer_exl3_frontend_reference_test
  ninfer_exl3_vision_test
  ninfer_exl3_prepared_request_test
  PROPERTIES DISABLED TRUE)
ninfer_add_test(ninfer_prepared_identity_transactions_test
  SOURCES test_prepared_identity_transactions.cpp
  LIBRARIES ninfer_qwen_prefix_identity CUDA::nvtx3)
ninfer_add_test(ninfer_prepared_identity_allocation_failure_test
  SOURCES test_prepared_identity_allocation_failure.cpp
  LIBRARIES ninfer_qwen_prefix_identity CUDA::nvtx3)
ninfer_add_host_contract_test(ninfer_exl3_host_resident_set_test
  SOURCES test_exl3_host_resident_set.cpp
  LIBRARIES Psapi)
ninfer_add_host_contract_test(ninfer_exl3_greedy_packet_test
  SOURCES test_exl3_greedy_packet.cpp)
ninfer_add_host_contract_test(ninfer_exl3_packed_projection_test
  SOURCES test_exl3_packed_projection.cpp)
ninfer_add_host_contract_test(ninfer_exl3_suffix_proposer_test
  SOURCES test_exl3_suffix_proposer.cpp)
ninfer_add_test(ninfer_exl3_vision_reference_manifest_test
  SOURCES test_exl3_vision_reference_manifest.cpp)
ninfer_add_test(ninfer_exl3_vision_heldout_fixtures_test
  SOURCES test_exl3_vision_heldout_fixtures.cpp)
ninfer_add_test(ninfer_exl3_vision_qualification_plan_test
  SOURCES test_exl3_vision_qualification_plan.cpp)
foreach(vision_contract_target IN ITEMS
    ninfer_exl3_vision_reference_manifest_test
    ninfer_exl3_vision_heldout_fixtures_test
    ninfer_exl3_vision_qualification_plan_test)
  target_include_directories(${vision_contract_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()
ninfer_add_host_contract_test(ninfer_exl3_request_sampling_state_test
  SOURCES test_exl3_request_sampling_state.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampling_policy_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampling_policy.cpp)
ninfer_add_host_contract_test(ninfer_exl3_draft_distribution_contract_test
  SOURCES test_exl3_draft_distribution_contract.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampled_acceptance_reference_test
  SOURCES test_exl3_sampled_acceptance_reference.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampling_window_transaction_test
  SOURCES test_exl3_sampling_window_transaction.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampled_authority_boundary_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampled_authority_boundary.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampling_workspace_reservation_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampling_workspace_reservation.cpp)
ninfer_add_host_contract_test(ninfer_exl3_sampling_control_stop_semantics_test
  SOURCES test_exl3_sampling_control_stop_semantics.cpp)
ninfer_add_host_contract_test(ninfer_exl3_owner_ledger_reference_test
  SOURCES test_exl3_owner_ledger_reference.cpp)
ninfer_add_host_contract_test(ninfer_exl3_state_machine_scenarios_test
  SOURCES test_exl3_state_machine_scenarios.cpp)
ninfer_add_host_contract_test(ninfer_exl3_physical_resource_attribution_test
  SOURCES test_exl3_physical_resource_attribution.cpp)
ninfer_add_host_contract_test(ninfer_exl3_candidate_contracts_test
  SOURCES test_exl3_candidate_contracts.cpp)
ninfer_add_host_contract_test(ninfer_exl3_continuation_graph_drain_policy_test
  SOURCES test_exl3_continuation_graph_drain_policy.cpp)
ninfer_add_host_contract_test(ninfer_exl3_resource_inventory_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_resource_inventory.cpp)
ninfer_add_host_contract_test(ninfer_exl3_retirement_state_test
  SOURCES test_exl3_retirement_state.cpp)
ninfer_add_host_contract_test(ninfer_exl3_verifier_horizon_policy_test
  SOURCES test_exl3_verifier_horizon_policy.cpp)
ninfer_add_test(ninfer_exl3_greedy_device_test
  SOURCES test_exl3_greedy_device.cu
  LIBRARIES CUDA::cudart)

foreach(v1_mode IN ITEMS export consume)
  set(v1_target ninfer_exl3_v1_oracle_${v1_mode})
  add_executable(${v1_target} EXCLUDE_FROM_ALL test_exl3_dflash2_accept.cpp)
  set_property(TARGET ${v1_target} PROPERTY CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${v1_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  target_link_libraries(${v1_target} PRIVATE ninfer_exl3_dflash2)
  if(MSVC)
    target_compile_options(${v1_target} PRIVATE /bigobj)
  endif()
  string(TOUPPER ${v1_mode} v1_define)
  target_compile_definitions(${v1_target} PRIVATE NINFER_V1_${v1_define}=1)
endforeach()
