#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_simplex_bindings(py::module_& m);
void bind_model_bindings(py::module_& m);
void bind_sparse_lu_bindings(py::module_& m);
