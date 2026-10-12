find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(voidinfer_public_api_test test_public_api.cpp)
target_include_directories(voidinfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME voidinfer_public_api_test COMMAND voidinfer_public_api_test)

function(voidinfer_add_test name)
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
    target_link_libraries(${name} PRIVATE voidinfer_core)
  endif()
  if("${arg_LIBRARIES}" MATCHES "voidinfer_engine|voidinfer_serve|voidinfer_media_decode")
    voidinfer_copy_runtime_dlls(${name})
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      VOIDINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      VOIDINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Header-contract executables exercise only model-independent host state. Keep
# them independent of voidinfer_core (and therefore CUDA runtime linkage) so they
# remain valid inputs to the strict CPU-only qualification allowlist.
function(voidinfer_add_host_contract_test name)
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

function(voidinfer_add_op_test name)
  voidinfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

# Gaming-preparation campaign: CPU contract executable has no CUDA linkage.
voidinfer_add_host_contract_test(voidinfer_exl3_gaming_contract_test
  SOURCES test_exl3_gaming_contract.cpp)
# Phase B only: explicit GPU leaf differentials and represented-state fixtures.
voidinfer_add_test(voidinfer_exl3_gaming_operators_test
  SOURCES test_exl3_gaming_operators.cu
  LIBRARIES voidinfer_exl3_dflash2)

# Artifact framing and model-independent runtime contracts.
voidinfer_add_test(voidinfer_exl3_safetensors_test
  SOURCES test_exl3_safetensors.cpp
  LIBRARIES voidinfer_exl3_format)
voidinfer_add_test(voidinfer_exl3_single_linear_test
  SOURCES test_exl3_single_linear.cpp
  LIBRARIES voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_single_linear_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_cuda_linear_test
  SOURCES test_exl3_cuda_linear.cpp
  LIBRARIES voidinfer_exl3_cuda voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_cuda_linear_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_projection_benchmark
  SOURCES test_exl3_projection_benchmark.cpp
  LIBRARIES voidinfer_exl3_cuda voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_projection_benchmark PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_large_down_oracle
  SOURCES test_exl3_large_down_oracle.cpp
  LIBRARIES voidinfer_exl3_cuda voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_large_down_oracle PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_full_attention_layer_test
  SOURCES test_exl3_full_attention_layer.cpp
  LIBRARIES voidinfer_exl3_cuda voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_full_attention_layer_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_gdn_layer_test
  SOURCES test_exl3_gdn_layer.cpp
  LIBRARIES voidinfer_exl3_gdn voidinfer_exl3_format)
set_tests_properties(voidinfer_exl3_gdn_layer_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_full_model_test
  SOURCES test_exl3_full_model.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_full_model_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_dflash2_minimal_test
  SOURCES test_exl3_dflash2_minimal.cpp
  LIBRARIES voidinfer_exl3_dflash2)
set_tests_properties(voidinfer_exl3_dflash2_minimal_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_dflash2_state_test
  SOURCES test_exl3_dflash2_state.cpp
  LIBRARIES voidinfer_exl3_dflash2)
set_tests_properties(voidinfer_exl3_dflash2_state_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_compatibility_test
  SOURCES test_exl3_oscar_compatibility.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(voidinfer_exl3_oscar_compatibility_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_asset_compatibility_test
  SOURCES test_exl3_oscar_asset_compatibility.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(voidinfer_exl3_oscar_asset_compatibility_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_adapter_test
  SOURCES test_exl3_oscar_adapter.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(voidinfer_exl3_oscar_adapter_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_eager_model_test
  SOURCES test_exl3_oscar_eager_model.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_oscar_eager_model_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_short_bench
  SOURCES test_exl3_oscar_short_bench.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_oscar_short_bench PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_context_sweep
  SOURCES test_exl3_oscar_context_sweep.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_oscar_context_sweep PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_profile
  SOURCES test_exl3_oscar_profile.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_oscar_profile PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_oscar_graph_test
  SOURCES test_exl3_oscar_graph.cpp
  LIBRARIES voidinfer_exl3_text_model)
set_tests_properties(voidinfer_exl3_oscar_graph_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES voidinfer_media_decode)
voidinfer_add_test(voidinfer_device_test       SOURCES test_device.cpp)
voidinfer_add_test(voidinfer_decode_graph_test SOURCES test_decode_graph.cpp)
voidinfer_add_test(voidinfer_tensor_test       SOURCES test_tensor.cpp)
voidinfer_add_test(voidinfer_arena_test        SOURCES test_arena.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_ready_work_test SOURCES test_exl3_ready_work.cpp)
voidinfer_add_test(voidinfer_media_client_loopback_test
  SOURCES test_media_client_loopback.cpp
  LIBRARIES voidinfer_serve)
target_include_directories(voidinfer_media_client_loopback_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
set_tests_properties(voidinfer_media_client_loopback_test PROPERTIES DISABLED TRUE)
voidinfer_add_test(voidinfer_oscar_history_attention_test
  SOURCES test_oscar_history_attention.cpp
  LIBRARIES voidinfer_core voidinfer_ops)
set_tests_properties(voidinfer_oscar_history_attention_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES voidinfer_core)
# Family frontend.
voidinfer_add_test(voidinfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES voidinfer_engine voidinfer_core)
target_include_directories(voidinfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
# The full OSCAR validator is intentionally fail-closed, but it must not be
# benchmarked as a Debug scalar oracle. Keep these flags target-local so the
# rest of the Debug developer build retains its normal diagnostics.
if(MSVC)
else()
endif()

# Product protocol contracts.
voidinfer_add_test(voidinfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_responses_schema_test
  SOURCES test_responses_schema.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_response_store_test
  SOURCES test_response_store.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES voidinfer_serve)
voidinfer_add_test(voidinfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES voidinfer_serve)
target_include_directories(voidinfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)

# Pure host-side qualification of the fixed-dimension non-negative fit and the exact coefficient
# quantization used by the context-cost calibrator. It deliberately links no Engine/CUDA target.

# Numerical tests for the shared Ops the engine still launches (Vision tower, gated RMSNorm, OSCAR INT2).
set(voidinfer_op_tests
  add_bias
  gelu
  residual_add
  rmsnorm
  gated_rmsnorm
  layer_norm
  rope
  vision_pos_embed)
foreach(op IN LISTS voidinfer_op_tests)
  voidinfer_add_op_test(voidinfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES voidinfer_ops)
endforeach()
voidinfer_add_op_test(voidinfer_oscar_int2_g128_test
  SOURCES ops/test_oscar_int2_g128.cpp
  LIBRARIES voidinfer_ops)


voidinfer_add_test(voidinfer_exl3_dflash2_xctx_test
  SOURCES test_exl3_dflash2_xctx.cpp
  LIBRARIES voidinfer_exl3_dflash2)
set_tests_properties(voidinfer_exl3_dflash2_xctx_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_dflash2_xctx2_test
  SOURCES test_exl3_dflash2_xctx2.cpp
  LIBRARIES voidinfer_exl3_dflash2)
set_tests_properties(voidinfer_exl3_dflash2_xctx2_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_dflash2_wrap_test
  SOURCES test_exl3_dflash2_wrap.cpp
  LIBRARIES voidinfer_exl3_dflash2)
set_tests_properties(voidinfer_exl3_dflash2_wrap_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_dflash2_accept_test
  SOURCES test_exl3_dflash2_accept.cpp
  LIBRARIES voidinfer_exl3_dflash2)
target_compile_options(voidinfer_exl3_dflash2_accept_test PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)
set_tests_properties(voidinfer_exl3_dflash2_accept_test PROPERTIES SKIP_RETURN_CODE 77)

# Explicitly invoked, test-only oracle tools. Not part of default build/CTest.
voidinfer_add_test(voidinfer_exl3_engine_device_test
  SOURCES test_exl3_engine_device.cpp
  LIBRARIES voidinfer_engine CUDA::cudart)
target_include_directories(voidinfer_exl3_engine_device_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(voidinfer_exl3_engine_device_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_frontend_reference_test
  SOURCES test_exl3_frontend_reference.cpp
  LIBRARIES voidinfer_engine voidinfer_core)
target_include_directories(voidinfer_exl3_frontend_reference_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(voidinfer_exl3_frontend_reference_test PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(voidinfer_exl3_recurrent_pool_test
  SOURCES test_exl3_recurrent_pool.cpp
  LIBRARIES CUDA::cudart Psapi)


# These binaries are fixture-driven validators/tools. Their required artifact
# paths come from dedicated qualification invocations, not default CTest.
voidinfer_add_test(voidinfer_prepared_identity_transactions_test
  SOURCES test_prepared_identity_transactions.cpp
  LIBRARIES voidinfer_qwen_prefix_identity CUDA::nvtx3)
voidinfer_add_test(voidinfer_prepared_identity_allocation_failure_test
  SOURCES test_prepared_identity_allocation_failure.cpp
  LIBRARIES voidinfer_qwen_prefix_identity CUDA::nvtx3)
voidinfer_add_host_contract_test(voidinfer_exl3_host_resident_set_test
  SOURCES test_exl3_host_resident_set.cpp
  LIBRARIES Psapi)
voidinfer_add_host_contract_test(voidinfer_exl3_greedy_packet_test
  SOURCES test_exl3_greedy_packet.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_packed_projection_test
  SOURCES test_exl3_packed_projection.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_suffix_proposer_test
  SOURCES test_exl3_suffix_proposer.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_request_sampling_state_test
  SOURCES test_exl3_request_sampling_state.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampling_policy_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampling_policy.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_draft_distribution_contract_test
  SOURCES test_exl3_draft_distribution_contract.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampled_acceptance_reference_test
  SOURCES test_exl3_sampled_acceptance_reference.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampling_window_transaction_test
  SOURCES test_exl3_sampling_window_transaction.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampled_authority_boundary_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampled_authority_boundary.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampling_workspace_reservation_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_sampling_workspace_reservation.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_sampling_control_stop_semantics_test
  SOURCES test_exl3_sampling_control_stop_semantics.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_owner_ledger_reference_test
  SOURCES test_exl3_owner_ledger_reference.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_state_machine_scenarios_test
  SOURCES test_exl3_state_machine_scenarios.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_physical_resource_attribution_test
  SOURCES test_exl3_physical_resource_attribution.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_candidate_contracts_test
  SOURCES test_exl3_candidate_contracts.cpp)
if(WIN32)
  voidinfer_add_host_contract_test(voidinfer_exl3_environment_options_test
    SOURCES test_exl3_environment_options.cpp)
endif()
voidinfer_add_host_contract_test(voidinfer_exl3_continuation_graph_drain_policy_test
  SOURCES test_exl3_continuation_graph_drain_policy.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_resource_inventory_test
  NEEDS_CUDA_HEADERS
  SOURCES test_exl3_resource_inventory.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_retirement_state_test
  SOURCES test_exl3_retirement_state.cpp)
voidinfer_add_host_contract_test(voidinfer_exl3_verifier_horizon_policy_test
  SOURCES test_exl3_verifier_horizon_policy.cpp)
voidinfer_add_test(voidinfer_exl3_greedy_device_test
  SOURCES test_exl3_greedy_device.cu
  LIBRARIES CUDA::cudart)
voidinfer_add_test(research_exl3_kv_pack
  SOURCES research_exl3_kv_pack.cu
  LIBRARIES CUDA::cudart)
set_tests_properties(research_exl3_kv_pack PROPERTIES SKIP_RETURN_CODE 77)

foreach(v1_mode IN ITEMS export consume)
  set(v1_target voidinfer_exl3_v1_oracle_${v1_mode})
  add_executable(${v1_target} EXCLUDE_FROM_ALL test_exl3_dflash2_accept.cpp)
  set_property(TARGET ${v1_target} PROPERTY CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${v1_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  target_link_libraries(${v1_target} PRIVATE voidinfer_exl3_dflash2)
  if(MSVC)
    target_compile_options(${v1_target} PRIVATE /bigobj)
  endif()
  string(TOUPPER ${v1_mode} v1_define)
  target_compile_definitions(${v1_target} PRIVATE VOIDINFER_V1_${v1_define}=1)
endforeach()

voidinfer_add_test(research_exl3_kv_tier_capture
  SOURCES research_exl3_kv_tier_capture.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(research_exl3_kv_tier_capture PROPERTIES SKIP_RETURN_CODE 77)

voidinfer_add_test(research_exl3_long_nll
  SOURCES research_exl3_long_nll.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(research_exl3_long_nll PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(research_exl3_vericache
  SOURCES research_exl3_vericache.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(research_exl3_vericache PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(research_exl3_overlap
  SOURCES research_exl3_overlap.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops CUDA::cuda_driver)
set_tests_properties(research_exl3_overlap PROPERTIES SKIP_RETURN_CODE 77)
voidinfer_add_test(research_exl3_l0_agents
  SOURCES research_exl3_l0_agents.cpp
  LIBRARIES voidinfer_exl3_text_model voidinfer_ops)
set_tests_properties(research_exl3_l0_agents PROPERTIES SKIP_RETURN_CODE 77)
