cmake_minimum_required(VERSION 3.18.0)
project(gen-oneflow-proto)
include(FetchContent)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(protobuf_BUILD_TESTS OFF CACHE STRING "" FORCE)

set(PROTOBUF_URL "https://github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
set(PROTOBUF_URL "https://oneflow-static.oss-cn-beijing.aliyuncs.com/third_party_mirror/https/github.com/protocolbuffers/protobuf/archive/v3.9.2.zip")
set(PROTOBUF_MD5 cf02c32870a1f78c860039e0f63a6343)
FetchContent_Declare(
    protobuf
    URL ${PROTOBUF_URL}
    URL_HASH MD5=${PROTOBUF_MD5}
)
FetchContent_GetProperties(protobuf)
if(NOT protobuf_POPULATED)
    FetchContent_Populate(protobuf)
    add_subdirectory(${protobuf_SOURCE_DIR}/cmake ${protobuf_BINARY_DIR})
    include_directories("${protobuf_SOURCE_DIR}/include")
endif()
