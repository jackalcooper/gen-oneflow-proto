cmake_minimum_required(VERSION 3.18.0)
project(gen-oneflow-proto)
include(FetchContent)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
set(CMAKE_C_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)
set(CMAKE_CUDA_COMPILER_LAUNCHER ccache CACHE STRING "" FORCE)

function(use_mirror)
  cmake_parse_arguments(
    PARSED_ARGS
    ""
    "VARIABLE;URL"
    ""
    ${ARGN}
  )
  string(REPLACE "https://" "https://oneflow-static.oss-cn-beijing.aliyuncs.com/third_party_mirror/https/" ${PARSED_ARGS_VARIABLE} ${PARSED_ARGS_URL})
endfunction()

set(INJA_URL https://github.com/pantor/inja/archive/refs/tags/v3.3.0.zip CACHE STRING "")
use_mirror(VARIABLE INJA_URL URL ${INJA_URL})
set(INJA_MD5 611e6b7206d0fb89728a3879f78b4775 CACHE STRING "")
set(JSON_URL https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip CACHE STRING "")
use_mirror(VARIABLE JSON_URL URL ${JSON_URL})
set(JSON_MD5 fb96f95cdf609143e998db401ca4f324 CACHE STRING "")

FetchContent_Declare(
  inja
)
FetchContent_GetProperties(inja)
if(NOT inja_POPULATED)
  FetchContent_Populate(inja
    URL ${INJA_URL}
    URL_HASH MD5=${INJA_MD5}
  )
endif()


FetchContent_Declare(
  json
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json
    URL ${JSON_URL}
    URL_HASH MD5=${JSON_MD5}
  )
endif()


set(protobuf_BUILD_TESTS OFF CACHE STRING "" FORCE)
set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
set(PROTOBUF_URL "https://oneflow-static.oss-cn-beijing.aliyuncs.com/third_party_mirror/https/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
# set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/refs/tags/v3.19.1.zip")
FetchContent_Declare(
    protobuf
    URL ${PROTOBUF_URL}
)
FetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
    FetchContent_Populate(protobuf)
    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
    include_directories("${protobuf_SOURCE_DIR}/src")
endif()

add_executable(gen-td
    gen-td.cpp
)
target_link_libraries(gen-td libprotobuf libprotoc)
target_include_directories(gen-td
    PRIVATE ${inja_SOURCE_DIR}/include/inja
    ${json_SOURCE_DIR}/include
)
add_custom_target(gen-sys-op-td)
set(ONEFLOW_SRC_DIR /home/caishenghang/oneflow-conda)
add_custom_command(
    TARGET gen-sys-op-td
    COMMAND protoc --plugin=protoc-gen-oneflow=$<TARGET_FILE:gen-td>
        --oneflow_out=${CMAKE_CURRENT_BINARY_DIR}
        --proto_path ${ONEFLOW_SRC_DIR}
        ${ONEFLOW_SRC_DIR}/oneflow/core/operator/op_conf.proto
    DEPENDS gen-td
)
