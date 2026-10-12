function(voidinfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(voidinfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  if(WIN32)
    set_target_properties(${target} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(voidinfer_core STATIC
  core/arena.cu
  core/device.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/oscar_mixed_cache_layout.cpp
  core/oscar_mixed_state_transitions.cpp
  core/oscar_mixed_attention_reference.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/public_types.cpp)
voidinfer_internal_includes(voidinfer_core)
voidinfer_cuda_archive(voidinfer_core)
target_link_libraries(voidinfer_core PUBLIC ${VOIDINFER_CUDART_TARGET} CUDA::nvtx3 Threads::Threads)

# Safetensors framing, dual-mirror reads, the host EXL3 reference linear and the engine profile. No CUDA.
add_library(voidinfer_exl3_format STATIC
  exl3/engine_profile.cpp
  exl3/safetensors.cpp
  exl3/linear_reference.cpp)
voidinfer_internal_includes(voidinfer_exl3_format)

# Isolated E2B native CUDA leaf. This target is intentionally separate from both the
# qualified NVFP4/OSCAR route and the E2 CPU/reference implementation.
add_library(voidinfer_exl3_cuda STATIC
  exl3/linear_cuda.cu
  exl3/prefill_attention_chain_graph.cu
  exl3/prefill_projection_chain_graph.cu
  exl3/target_projection_timing.cu
  exl3/full_attention_layer.cu)
add_library(voidinfer_exl3_oscar STATIC
  exl3/oscar_runtime.cu)
voidinfer_internal_includes(voidinfer_exl3_oscar)
voidinfer_cuda_archive(voidinfer_exl3_oscar)
target_link_libraries(voidinfer_exl3_oscar PUBLIC voidinfer_ops ${VOIDINFER_CUDART_TARGET})
voidinfer_internal_includes(voidinfer_exl3_cuda)
voidinfer_cuda_archive(voidinfer_exl3_cuda)
target_include_directories(voidinfer_exl3_cuda PUBLIC
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(voidinfer_exl3_cuda PUBLIC voidinfer_exl3_oscar ${VOIDINFER_CUDART_TARGET}
  PRIVATE CUDA::cublas CUDA::cublasLt CUDA::cuda_driver)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(voidinfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/gelu.cu
  ops/launcher/layer_norm.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rope.cu
  ops/launcher/vision_pos_embed.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/oscar_mixed/launch.cu
  ops/softmax_attention/oscar_history/launch.cu
  ops/kv_cache/oscar_int2_g128.cpp
  ops/kv_cache/oscar_conversion.cu
  ops/wrapper/add_bias.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/vision_pos_embed.cpp)
voidinfer_internal_includes(voidinfer_ops)
voidinfer_cuda_archive(voidinfer_ops)
target_link_libraries(voidinfer_ops PUBLIC voidinfer_core)

# Native EXL3 Qwen3.5 GDN layer milestone.  Kept separate from the qualified
# full-attention leaf so this phase cannot alter the existing EXL3 surface.
add_library(voidinfer_exl3_gdn STATIC
  exl3/gdn_layer.cu)
voidinfer_internal_includes(voidinfer_exl3_gdn)
voidinfer_cuda_archive(voidinfer_exl3_gdn)
target_link_libraries(voidinfer_exl3_gdn PUBLIC voidinfer_exl3_cuda voidinfer_ops)

# E4A complete native Qwen3.5 text-model assembly.  This remains isolated from
# the product engine and the qualified D4.8B/NVFP4 route.
add_library(voidinfer_exl3_text_model STATIC
  exl3/text_model.cu)
voidinfer_internal_includes(voidinfer_exl3_text_model)
voidinfer_cuda_archive(voidinfer_exl3_text_model)
target_include_directories(voidinfer_exl3_text_model PUBLIC ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(voidinfer_exl3_text_model PUBLIC voidinfer_exl3_gdn voidinfer_exl3_format voidinfer_ops)

# The BF16 Vision tower (weights streamed from pinned host memory).
add_library(voidinfer_exl3_vision STATIC exl3/vision_dense.cu)
voidinfer_internal_includes(voidinfer_exl3_vision)
voidinfer_cuda_archive(voidinfer_exl3_vision)
target_include_directories(voidinfer_exl3_vision PUBLIC ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(voidinfer_exl3_vision PUBLIC voidinfer_exl3_cuda voidinfer_exl3_format voidinfer_core voidinfer_ops PRIVATE CUDA::cublas)
# E5A2 minimal native DFlash2 draft adapter. Separate 5-layer draft model bound to
# the qualified EXL3 Qwen3.8-27B target taps/head. Correctness-first and isolated
# from the product engine; not part of the E4C1 qualified target decode path.


add_library(voidinfer_exl3_dflash2 STATIC
  exl3/dflash2_draft.cu)
voidinfer_internal_includes(voidinfer_exl3_dflash2)
voidinfer_cuda_archive(voidinfer_exl3_dflash2)
target_link_libraries(voidinfer_exl3_dflash2 PUBLIC voidinfer_exl3_text_model ${VOIDINFER_CUDART_TARGET})


# Product-independent text and media mechanisms.
add_library(voidinfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
voidinfer_internal_includes(voidinfer_text)
target_compile_definitions(voidinfer_text PUBLIC UTF8PROC_STATIC)

add_library(voidinfer_media_decode STATIC
  media/decode/decode.cpp)
voidinfer_internal_includes(voidinfer_media_decode)
target_link_libraries(voidinfer_media_decode PRIVATE ${VOIDINFER_FFMPEG_TARGET})

if(VOIDINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(voidinfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  voidinfer_internal_includes(voidinfer_media_acquire)
  target_link_libraries(voidinfer_media_acquire PRIVATE ${VOIDINFER_CURL_TARGET})
  if(WIN32)
    target_link_libraries(voidinfer_media_acquire PRIVATE ws2_32)
  endif()
endif()

if(VOIDINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(voidinfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  voidinfer_internal_includes(voidinfer_product_prompt_input)
  target_link_libraries(voidinfer_product_prompt_input PRIVATE voidinfer_media_acquire)
endif()

if(VOIDINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(voidinfer_product_load_progress STATIC
    product/load_progress/load_progress.cpp)
  voidinfer_internal_includes(voidinfer_product_load_progress)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(voidinfer_engine STATIC

  exl3/exl3_frontend_resources.cpp
  exl3/exl3_engine_core.cpp
  runtime/contract/sampling.cpp
  runtime/engine/engine.cpp)
voidinfer_internal_includes(voidinfer_engine)
target_include_directories(voidinfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)

target_link_libraries(voidinfer_engine
  PRIVATE
    voidinfer_exl3_dflash2
    voidinfer_exl3_vision
    voidinfer_core
    voidinfer_ops
    voidinfer_text
    voidinfer_media_decode)
add_library(voidinfer::engine ALIAS voidinfer_engine)

if(VOIDINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(voidinfer_serve STATIC
    serve/openai_schema.cpp
    serve/responses_schema.cpp
    serve/anthropic_schema.cpp
    serve/response_store.cpp
    serve/tool_call_parser.cpp
    serve/translate.cpp
    serve/console_log.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/responses_http.cpp
    serve/serve_options.cpp
    serve/webui_update.cpp)
  voidinfer_internal_includes(voidinfer_serve)
  target_include_directories(voidinfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(voidinfer_serve
    PUBLIC voidinfer_engine Threads::Threads
    PRIVATE voidinfer_media_acquire ${VOIDINFER_CUDART_TARGET})
  if(WIN32)
    target_link_libraries(voidinfer_serve PRIVATE winhttp)
  endif()
endif()
