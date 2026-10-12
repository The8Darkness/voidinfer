add_executable(voidinfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(voidinfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(voidinfer PRIVATE
  voidinfer_engine
  voidinfer_product_load_progress
  voidinfer_product_prompt_input)
voidinfer_copy_runtime_dlls(voidinfer)

add_executable(voidinfer-serve serve/main.cpp)
target_include_directories(voidinfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(voidinfer-serve PRIVATE voidinfer_serve voidinfer_product_load_progress)
voidinfer_copy_runtime_dlls(voidinfer-serve)
