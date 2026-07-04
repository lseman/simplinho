cmake_minimum_required(VERSION 3.16)

project(simplinho VERSION 0.1 LANGUAGES CXX)

# ==============================================================================
# Project defaults
# ==============================================================================

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# force clang
set(CMAKE_CXX_COMPILER "/data/toolchain/llvm/stage2-prof-use-lto/install/bin/clang++" CACHE STRING "C++ compiler" FORCE)

set(SIMPLEX_LOCAL_EIGEN_DIR "" CACHE PATH "Optional local Eigen checkout")
set(SIMPLEX_LOCAL_PYBIND11_DIR "" CACHE PATH "Optional local pybind11 checkout")

option(SIMPLEX_ENABLE_PERF_FLAGS "Enable perf-friendly compile and link flags" ON)
option(SIMPLEX_ENABLE_CCACHE "Enable ccache/sccache for faster builds" ON)
option(SIMPLEX_ENABLE_UNITY_BUILD "Enable Unity (jumbo) builds for faster compilation" ON)
option(SIMPLEX_ENABLE_PCH "Enable precompiled headers" OFF)
option(SIMPLEX_ENABLE_BNB "Build the branch-and-bound MIP solver" OFF)
option(SIMPLEX_ENABLE_IPM "Build the interior-point solver" OFF)
option(BNB_PROFILE_TIMERS "Enable per-phase BnB timing counters" OFF)
option(SIMPLEX_BNB_DEBUG_LOCKS "Enable BnB lock tracing" OFF)
option(SIMPLEX_BNB_DEBUG_TIMING "Enable BnB phase timing debug output" OFF)

# ==============================================================================
# Build metadata
# ==============================================================================

set(SIMPLEX_GIT_DESCRIBE "unknown")
set(SIMPLEX_GIT_BRANCH "unknown")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} describe --tags --always --dirty
        OUTPUT_VARIABLE SIMPLEX_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE SIMPLEX_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

string(STRIP "${SIMPLEX_GIT_DESCRIBE}" SIMPLEX_GIT_DESCRIBE)
string(STRIP "${SIMPLEX_GIT_BRANCH}" SIMPLEX_GIT_BRANCH)

# ==============================================================================
# Build helpers
# ==============================================================================

function(simplinho_enable_compiler_cache)
    if(NOT SIMPLEX_ENABLE_CCACHE)
        return()
    endif()

    find_program(CCACHE_PROGRAM ccache)
    if(NOT CCACHE_PROGRAM)
        find_program(CCACHE_PROGRAM sccache)
    endif()

    if(CCACHE_PROGRAM)
        message(STATUS "Using compiler cache: ${CCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE FILEPATH "C++ compiler launcher")
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE FILEPATH "C compiler launcher")
    endif()
endfunction()

function(simplinho_apply_perf_flags target_name)
    if(NOT SIMPLEX_ENABLE_PERF_FLAGS OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()

    target_compile_options(${target_name} INTERFACE
        $<$<CONFIG:Release>:-O3 -fno-semantic-interposition -fomit-frame-pointer>
        $<$<CONFIG:RelWithDebInfo>:-O3 -fno-semantic-interposition -fomit-frame-pointer>
    )
endfunction()

function(simplinho_apply_module_link_flags target_name)
    if(NOT SIMPLEX_ENABLE_PERF_FLAGS OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        return()
    endif()

    # Use lld for faster linking
    find_program(LLD_PROGRAM lld)
    if(LLD_PROGRAM)
        set_target_properties(${target_name} PROPERTIES
            LINK_FLAGS "-fuse-ld=lld"
        )
    endif()

    target_link_options(${target_name} PRIVATE
        $<$<CONFIG:Release>:-Wl,--gc-sections>
    )
endfunction()

simplinho_enable_compiler_cache()

# ==============================================================================
# Dependencies
# ==============================================================================

include(FetchContent)

# Disable TBB (you had a good reason)
set(CMAKE_DISABLE_FIND_PACKAGE_TBB ON CACHE BOOL "Disable TBB" FORCE)

if(NOT COMMAND CPMAddPackage)
    FetchContent_Declare(
        CPM
        GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
        GIT_TAG v0.42.1
    )
    FetchContent_MakeAvailable(CPM)
    include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()

cpmaddpackage(
    NAME stdexec
    GITHUB_REPOSITORY NVIDIA/stdexec
    GIT_TAG main
    OPTIONS
    "STDEXEC_BUILD_TESTS OFF"
    "STDEXEC_BUILD_EXAMPLES OFF"
    "STDEXEC_BUILD_BENCHMARKS OFF"
    "STDEXEC_BUILD_DOCS OFF"
)

cpmaddpackage(
    NAME ankerl_unordered_dense
    GITHUB_REPOSITORY martinus/unordered_dense
    GIT_TAG origin/main
)

if(NOT TARGET pybind11::module)
    if(EXISTS "${SIMPLEX_LOCAL_PYBIND11_DIR}/CMakeLists.txt")
        add_subdirectory("${SIMPLEX_LOCAL_PYBIND11_DIR}" pybind11-build)
    else()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v3.0.4
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

if(NOT TARGET Eigen3::Eigen)
    add_library(simplex_eigen INTERFACE)

    if(EXISTS "${SIMPLEX_LOCAL_EIGEN_DIR}/Eigen/Core")
        target_include_directories(simplex_eigen INTERFACE "${SIMPLEX_LOCAL_EIGEN_DIR}")
    else()
        FetchContent_Declare(
            Eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.1
        )
        FetchContent_MakeAvailable(Eigen)
        target_include_directories(simplex_eigen INTERFACE "${eigen_SOURCE_DIR}")
    endif()

    add_library(Eigen3::Eigen ALIAS simplex_eigen)
endif()

set(PYBIND11_FINDPYTHON ON)
set(PYBIND11_PYTHON_VERSION 3.14 CACHE STRING "Python version to use when building pybind11 modules" FORCE)

find_package(Python 3.14 REQUIRED COMPONENTS Interpreter Development)
find_package(Threads REQUIRED)

# ==============================================================================
# Sources
# ==============================================================================

set(SIMPLINHO_BINDING_SOURCES
    bindings/model_bindings.cpp
    bindings/module.cpp
    bindings/simplex_bindings.cpp
)

set(SIMPLINHO_EXTRA_SOURCES "")

# NLA layer — HiGHS HSimplexNla equivalent (ftran/btran chain, PF, SimplexIterate)
list(APPEND SIMPLINHO_EXTRA_SOURCES src/nla/simplex_nla.cpp)

if(SIMPLEX_ENABLE_BNB)
    list(APPEND SIMPLINHO_EXTRA_SOURCES
        src/bnb/async_heuristic_manager.cpp
        src/bnb/branching.cpp
        src/bnb/conflict_engine.cpp
        src/bnb/core.cpp
        src/bnb/cuts.cpp
        src/bnb/diving.cpp
        src/bnb/heuristic.cpp
        src/bnb/mip_presolve.cpp
    )
    list(APPEND SIMPLINHO_BINDING_SOURCES bindings/bnb_bindings.cpp)
endif()

if(SIMPLEX_ENABLE_IPM)
    list(APPEND SIMPLINHO_EXTRA_SOURCES
        src/ipm/IPSolver.cpp
        src/ipm/ipm_linear_solver.cpp
    )
    list(APPEND SIMPLINHO_BINDING_SOURCES bindings/ipm_bindings.cpp)
endif()

# ==============================================================================
# Core interface target
# ==============================================================================

add_library(simplex_core INTERFACE)

target_compile_features(simplex_core INTERFACE cxx_std_23)

target_include_directories(simplex_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvers"
    "${ankerl_unordered_dense_SOURCE_DIR}/include"
)

target_compile_definitions(simplex_core INTERFACE
    SIMPLEX_PROJECT_VERSION="${PROJECT_VERSION}"
    SIMPLEX_GIT_DESCRIBE="${SIMPLEX_GIT_DESCRIBE}"
    SIMPLEX_GIT_BRANCH="${SIMPLEX_GIT_BRANCH}"
    _GLIBCXX_USE_TBB_PAR_BACKEND=0
    $<$<BOOL:${SIMPLEX_ENABLE_BNB}>:SIMPLEX_ENABLE_BNB>
    $<$<BOOL:${SIMPLEX_ENABLE_IPM}>:SIMPLEX_ENABLE_IPM>
)

target_link_libraries(simplex_core INTERFACE
    Eigen3::Eigen
    STDEXEC::stdexec
)

simplinho_apply_perf_flags(simplex_core)

if(SIMPLEX_ENABLE_PCH)
    target_precompile_headers(simplex_core INTERFACE
        <Eigen/Core>
        <vector>
        <string>
        <memory>
        <unordered_map>
        <ankerl/unordered_dense.h>
    )
endif()

# ==============================================================================
# Python extension module
# ==============================================================================

pybind11_add_module(simplinho
    ${SIMPLINHO_BINDING_SOURCES}
    ${SIMPLINHO_EXTRA_SOURCES}
)

target_compile_features(simplinho PRIVATE cxx_std_23)

target_link_libraries(simplinho PRIVATE
    simplex_core
    pybind11::module
    Threads::Threads
)

if(SIMPLEX_ENABLE_UNITY_BUILD)
    set_target_properties(simplinho PROPERTIES
        UNITY_BUILD ON
        UNITY_BUILD_BATCH_SIZE 4
    )
endif()

simplinho_apply_module_link_flags(simplinho)

message(STATUS "simplinho build configured with ccache=${SIMPLEX_ENABLE_CCACHE}, "
    "PCH=${SIMPLEX_ENABLE_PCH}, unity=${SIMPLEX_ENABLE_UNITY_BUILD}, "
    "BnB=${SIMPLEX_ENABLE_BNB}, IPM=${SIMPLEX_ENABLE_IPM}")
