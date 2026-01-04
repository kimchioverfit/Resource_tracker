cmake_minimum_required(VERSION 3.25)

project(
    PcCare365_
    LANGUAGES CXX
)

if (MSVC)
    add_compile_options(
        /utf-8
        /permissive-
        /W4
        /EHsc
    )
endif()

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_CMD_RESULT
)

if (NOT GIT_CMD_RESULT EQUAL 0 OR GIT_COMMIT_HASH STREQUAL "")
    set(GIT_COMMIT_HASH nogit)
endif()

set(EXE_NAME "PcCare365_${GIT_COMMIT_HASH}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(
    Torch_bin_archive_url_release
    "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.3.1%2Bcpu.zip"
)

set(
    Torch_bin_archive_url_debug
    "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-2.3.1%2Bcpu.zip"
)

option(
    TORCH_USE_DEBUG_BIN
    "Download debug libtorch archive (otherwise release)"
    OFF
)

set(_torch_use_debug "${TORCH_USE_DEBUG_BIN}")

if (NOT _torch_use_debug AND CMAKE_BUILD_TYPE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_torch_use_debug ON)
    endif()
endif()

if (_torch_use_debug)
    set(Torch_bin_archive_url "${Torch_bin_archive_url_debug}")
else()
    set(Torch_bin_archive_url "${Torch_bin_archive_url_release}")
endif()

FetchContent_Declare(
    Torch
    URL ${Torch_bin_archive_url}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(Torch)

set(Torch_DIR "${torch_SOURCE_DIR}/share/cmake/Torch/")

find_package(Torch REQUIRED)

set(TORCH_LIB_DIR "${torch_SOURCE_DIR}/lib")

set(EMBED_RC "${CMAKE_BINARY_DIR}/embedded_dlls.rc")

file(
    WRITE "${EMBED_RC}"
    "#include \"${CMAKE_SOURCE_DIR}/src/resources.h\"\n"
)

list(LENGTH EMBED_DLLS EMBED_DLLS_LEN)
math(EXPR EMBED_DLLS_LAST "${EMBED_DLLS_LEN}-1")

file(
    GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.h"
    "${CMAKE_SOURCE_DIR}/src/*.hpp"
)

add_executable(${EXE_NAME} ${SRC_FILES})

target_sources(${EXE_NAME} PRIVATE "${EMBED_RC}")

target_link_libraries(
    ${EXE_NAME}
    PRIVATE
    ${TORCH_LIBRARIES}
)

target_include_directories(
    ${EXE_NAME}
    PRIVATE
    src
)

if (WIN32)
    target_compile_definitions(
        ${EXE_NAME}
        PRIVATE
        UNICODE
        _UNICODE
        WIN32_LEAN_AND_MEAN
        _WIN32_WINNT=0x0A00
    )

    target_link_libraries(
        ${EXE_NAME}
        PRIVATE
        Pdh
        Iphlpapi
        Ws2_32
        Wbemuuid
        Ole32
        OleAut32
    )
endif()
