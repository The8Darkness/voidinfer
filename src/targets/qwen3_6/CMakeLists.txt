# Passive prepared-input identity has a single compile owner below both the
# Engine and the EXL3 request/cache code; it has no execution dependency.
add_library(voidinfer_qwen_prefix_identity STATIC impl/runtime/prefix_identity.cpp)
voidinfer_internal_includes(voidinfer_qwen_prefix_identity)
target_link_libraries(voidinfer_qwen_prefix_identity PRIVATE voidinfer_core)
target_include_directories(voidinfer_qwen_prefix_identity PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/export)
target_link_libraries(voidinfer_engine PRIVATE voidinfer_qwen_prefix_identity)
target_link_libraries(voidinfer_exl3_dflash2 PUBLIC voidinfer_qwen_prefix_identity)

target_sources(voidinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/chat_template.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/digest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/frontend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/media_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/processor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/frontend/tokenizer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/impl/vision/control.cpp)

target_include_directories(voidinfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/export)
