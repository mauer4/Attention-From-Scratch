cmake_minimum_required(VERSION 3.22)

project(AttentionFromScratch
    VERSION 0.1.0
    DESCRIPTION "From-scratch transformer inference engine"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ATTN_BUILD_PYTHON "Build Python bindings via pybind11" ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

file(GLOB_RECURSE ATTN_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

add_library(attention_core STATIC ${ATTN_CORE_SOURCES})
target_include_directories(attention_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_definitions(attention_core
    PRIVATE $<$<CONFIG:Debug>:ATTN_ENABLE_ASSERTS>
)

if(ATTN_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(attention_bindings MODULE
            python_bindings/module.cpp
        )
        target_link_libraries(attention_bindings PRIVATE attention_core)
    else()
        message(WARNING "pybind11 not found, skipping Python bindings target")
    endif()
endif()

enable_testing()
add_subdirectory(tests)
