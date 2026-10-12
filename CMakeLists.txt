cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "VoidInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(voidinfer LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
option(VOIDINFER_BUILD_APPS "Build the VoidInfer CLI and server" ON)
option(BUILD_TESTING "Build VoidInfer tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
  # CUDA's static runtime requests LIBCMT even when the application and vcpkg dependencies use
  # the project-wide /MD runtime. Keep one CRT in the final process.
  add_link_options(/NODEFAULTLIB:LIBCMT)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "VoidInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()

set(VOIDINFER_BUILD_MEDIA_ACQUIRE OFF)
if(VOIDINFER_BUILD_APPS OR BUILD_TESTING)
  set(VOIDINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(VOIDINFER_BUILD_PROMPT_INPUT OFF)
if(VOIDINFER_BUILD_APPS)
  set(VOIDINFER_BUILD_PROMPT_INPUT ON)
endif()

set(VOIDINFER_BUILD_SERVE OFF)
if(VOIDINFER_BUILD_APPS OR BUILD_TESTING)
  set(VOIDINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
if(WIN32)
  set(VOIDINFER_CUDART_TARGET CUDA::cudart_static)
else()
  set(VOIDINFER_CUDART_TARGET CUDA::cudart)
endif()
if(WIN32 OR DEFINED VCPKG_TARGET_TRIPLET)
  find_package(FFMPEG REQUIRED)
  add_library(voidinfer_ffmpeg_dependencies INTERFACE)
  target_include_directories(voidinfer_ffmpeg_dependencies INTERFACE ${FFMPEG_INCLUDE_DIRS})
  target_link_directories(voidinfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARY_DIRS})
  target_link_libraries(voidinfer_ffmpeg_dependencies INTERFACE ${FFMPEG_LIBRARIES})
  set(VOIDINFER_FFMPEG_TARGET voidinfer_ffmpeg_dependencies)
  if(VOIDINFER_BUILD_MEDIA_ACQUIRE)
    find_package(CURL 7.85 REQUIRED)
    set(VOIDINFER_CURL_TARGET CURL::libcurl)
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  set(VOIDINFER_FFMPEG_TARGET PkgConfig::FFMPEG)
  if(VOIDINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
    set(VOIDINFER_CURL_TARGET PkgConfig::LIBCURL)
  endif()
endif()

# Windows executables that use the media decoder carry FFmpeg's import table. Keep their runtime
# self-contained in the target output directory so direct smoke tests and CTest do not depend on
# whichever unrelated build happens to be present on PATH.
if(WIN32 AND FFMPEG_INCLUDE_DIRS)
  list(GET FFMPEG_INCLUDE_DIRS 0 VOIDINFER_FFMPEG_INCLUDE_ROOT)
  get_filename_component(VOIDINFER_FFMPEG_RUNTIME_ROOT
    "${VOIDINFER_FFMPEG_INCLUDE_ROOT}" DIRECTORY)
endif()

function(voidinfer_copy_runtime_dlls target)
  if(NOT WIN32 OR NOT VOIDINFER_FFMPEG_RUNTIME_ROOT)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      "${VOIDINFER_FFMPEG_RUNTIME_ROOT}/$<IF:$<CONFIG:Debug>,debug/bin,bin>"
      "$<TARGET_FILE_DIR:${target}>"
    COMMENT "Copying FFmpeg runtime DLLs for ${target}")
endfunction()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(VOIDINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
