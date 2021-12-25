cmake_minimum_required(VERSION 3.18.0)
project(gen-oneflow-proto)
include(FetchContent)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "" FORCE)
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
add_custom_target(gen-sys-op-td)
add_custom_command(
    TARGET gen-sys-op-td
    COMMAND $<TARGET_FILE:gen-td>
    DEPENDS gen-td
)
