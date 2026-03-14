cmake_minimum_required(VERSION 3.16)

project(simplinho VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

set(PYBIND11_FINDPYTHON ON CACHE BOOL "Use CMake's FindPython for pybind11" FORCE)
set(PYBIND11_PYTHON_VERSION 3.13 CACHE STRING
    "Python version to use when building pybind11 modules" FORCE)
find_package(Python 3.13 EXACT REQUIRED COMPONENTS Interpreter Development)

set(SIMPLEX_LOCAL_DEPS_ROOT "")
foreach(SIMPLEX_DEPS_CANDIDATE
    "${CMAKE_CURRENT_LIST_DIR}/build-local/_deps"
    "${CMAKE_CURRENT_LIST_DIR}/build/_deps"
    "${CMAKE_CURRENT_LIST_DIR}/../build/_deps"
    "${CMAKE_CURRENT_LIST_DIR}/../build-local/_deps")
    if(EXISTS "${SIMPLEX_DEPS_CANDIDATE}")
        set(SIMPLEX_LOCAL_DEPS_ROOT "${SIMPLEX_DEPS_CANDIDATE}")
        break()
    endif()
endforeach()

set(SIMPLEX_LOCAL_EIGEN_DIR
    "${SIMPLEX_LOCAL_DEPS_ROOT}/eigen-src")
set(SIMPLEX_LOCAL_PYBIND11_DIR
    "${SIMPLEX_LOCAL_DEPS_ROOT}/pybind11-src")

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

add_library(simplex_core INTERFACE)
target_include_directories(simplex_core INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(simplex_core INTERFACE Eigen3::Eigen)

pybind11_add_module(simplinho bindings/simplex_bindings.cpp)
target_link_libraries(simplinho PRIVATE simplex_core pybind11::module)
