cmake_minimum_required(VERSION 3.16)

project(simplinho VERSION 0.1 LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(SIMPLEX_GIT_DESCRIBE "unknown")
set(SIMPLEX_GIT_BRANCH "unknown")

find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} describe --tags --always --dirty
        OUTPUT_VARIABLE SIMPLEX_GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} -C ${CMAKE_CURRENT_SOURCE_DIR} rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE SIMPLEX_GIT_BRANCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
endif()

if(SIMPLEX_GIT_DESCRIBE STREQUAL "")
    set(SIMPLEX_GIT_DESCRIBE "unknown")
endif()
if(SIMPLEX_GIT_BRANCH STREQUAL "")
    set(SIMPLEX_GIT_BRANCH "unknown")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(SIMPLEX_ENABLE_PERF_FLAGS
    "Enable perf-friendly compile and link flags"
    ON)

option(SIMPLEX_ENABLE_CCACHE
    "Enable ccache for faster incremental local builds when available"
    ON)

option(BNB_PROFILE_TIMERS
    "Enable per-phase BnB timing counters (set SIMPLEX_BNB_PROFILE_TIMERS=1)"
    OFF)

option(SIMPLEX_BNB_DEBUG_LOCKS
    "Enable BnB lock tracing for debugging deadlocks and contention"
    OFF)

option(SIMPLEX_BNB_DEBUG_TIMING
    "Enable BnB phase timing debug output"
    OFF)

include(FetchContent)

# Keep stdexec on its portable static-thread-pool backend. On this system,
# auto-enabling TBB pulls libtbb into the Python extension and can trigger
# glibc pthread priority assertions under long benchmark runs.
set(CMAKE_DISABLE_FIND_PACKAGE_TBB ON CACHE BOOL
    "Disable TBB discovery while configuring bundled dependencies" FORCE)

# CPM.cmake
if(NOT COMMAND CPMAddPackage)
    FetchContent_Declare(
        CPM
        GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake.git
        GIT_TAG v0.42.1)
    FetchContent_MakeAvailable(CPM)
    include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)
endif()


cpmaddpackage(
    NAME stdexec
    GITHUB_REPOSITORY NVIDIA/stdexec
    GIT_TAG main
    OPTIONS "STDEXEC_BUILD_TESTS OFF" "STDEXEC_BUILD_EXAMPLES OFF" "STDEXEC_BUILD_BENCHMARKS OFF" "STDEXEC_BUILD_DOCS OFF" "BUILD_TESTING OFF"
)

cpmaddpackage(
    NAME ankerl_unordered_dense
    GITHUB_REPOSITORY martinus/unordered_dense
    GIT_TAG origin/main
)

if(NOT TARGET pybind11::module)
    if(EXISTS "${SIMPLEX_LOCAL_PYBIND11_DIR}/CMakeLists.txt")
        add_subdirectory(
            "${SIMPLEX_LOCAL_PYBIND11_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/pybind11-build"
        )
    else()
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

if(NOT TARGET Eigen3::Eigen)
    if(EXISTS "${SIMPLEX_LOCAL_EIGEN_DIR}/Eigen/Core")
        add_library(simplex_eigen INTERFACE)
        target_include_directories(simplex_eigen INTERFACE
            "${SIMPLEX_LOCAL_EIGEN_DIR}")
        add_library(Eigen3::Eigen ALIAS simplex_eigen)
    else()
        FetchContent_Declare(
            Eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 3.4.1
        )
        FetchContent_MakeAvailable(Eigen)
        add_library(simplex_eigen INTERFACE)
        target_include_directories(simplex_eigen INTERFACE
            "${eigen_SOURCE_DIR}")
        add_library(Eigen3::Eigen ALIAS simplex_eigen)
    endif()
endif()

cpmaddpackage(
    NAME ankerl_unordered_dense
    GITHUB_REPOSITORY martinus/unordered_dense
    GIT_TAG origin/main
)

include_directories(${ankerl_unordered_dense_SOURCE_DIR}/include)

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use CMake's FindPython for pybind11" FORCE)
set(PYBIND11_PYTHON_VERSION 3.13 CACHE STRING
    "Python version to use when building pybind11 modules" FORCE)
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development)
find_package(Threads REQUIRED)

if(SIMPLEX_ENABLE_CCACHE)
    find_program(SIMPLEX_CCACHE_PROGRAM ccache)
    if(SIMPLEX_CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SIMPLEX_CCACHE_PROGRAM}" CACHE STRING
            "C++ compiler launcher" FORCE)
    endif()
endif()

add_library(simplex_core INTERFACE)
target_include_directories(simplex_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(simplex_core INTERFACE cxx_std_23)
target_compile_definitions(simplex_core INTERFACE
    SIMPLEX_PROJECT_VERSION="${PROJECT_VERSION}"
    SIMPLEX_GIT_DESCRIBE="${SIMPLEX_GIT_DESCRIBE}"
    SIMPLEX_GIT_BRANCH="${SIMPLEX_GIT_BRANCH}"
    _GLIBCXX_USE_TBB_PAR_BACKEND=0)
target_link_libraries(simplex_core INTERFACE Eigen3::Eigen STDEXEC::stdexec)
if(SIMPLEX_ENABLE_PERF_FLAGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(simplex_core INTERFACE
            $<$<CONFIG:Release>:-O3;-fno-semantic-interposition;-fomit-frame-pointer>
            $<$<CONFIG:RelWithDebInfo>:-O3;-fno-semantic-interposition;-fomit-frame-pointer>)
    endif()
endif()


pybind11_add_module(simplinho bindings/bnb_bindings.cpp bindings/model_bindings.cpp bindings/module.cpp bindings/simplex_bindings.cpp
    src/bnb/async_heuristic_manager.cpp src/bnb/branching.cpp src/bnb/conflict_engine.cpp src/bnb/core.cpp src/bnb/cuts.cpp src/bnb/diving.cpp src/bnb/heuristic.cpp src/bnb/mip_presolve.cpp)
target_compile_features(simplinho PRIVATE cxx_std_23)
target_link_libraries(simplinho PRIVATE simplex_core Eigen3::Eigen pybind11::module Threads::Threads STDEXEC::stdexec)
if(SIMPLEX_ENABLE_PERF_FLAGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(simplinho PRIVATE
            $<$<CONFIG:Release>:-Wl,--gc-sections>
            $<$<CONFIG:RelWithDebInfo>:-Wl,--gc-sections>)
    endif()
endif()


set(SIMPLEX_LOCAL_DEPS_ROOT "")
