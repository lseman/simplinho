#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bindings.h"
#include "simplex/ekk/ekk_dual.h"

namespace py = pybind11;

void bind_ekk_bindings(py::module_& m) {
    using namespace simplex::ekk;

    py::enum_<ModelStatus>(m, "EkkStatus")
        .value("NotSet", ModelStatus::NotSet)
        .value("Optimal", ModelStatus::Optimal)
        .value("Infeasible", ModelStatus::Infeasible)
        .value("Unbounded", ModelStatus::Unbounded)
        .value("UnboundedOrInfeasible", ModelStatus::UnboundedOrInfeasible)
        .value("ObjectiveBound", ModelStatus::ObjectiveBound)
        .value("IterationLimit", ModelStatus::IterationLimit)
        .value("SolveError", ModelStatus::SolveError);

    py::class_<Options>(m, "EkkOptions")
        .def(py::init<>())
        .def_readwrite("primal_feasibility_tolerance", &Options::primal_feasibility_tolerance)
        .def_readwrite("dual_feasibility_tolerance", &Options::dual_feasibility_tolerance)
        .def_readwrite("dual_simplex_cost_perturbation_multiplier",
                       &Options::dual_simplex_cost_perturbation_multiplier)
        .def_readwrite("iteration_limit", &Options::iteration_limit)
        .def_readwrite("objective_bound", &Options::objective_bound)
        .def_readwrite("scale", &Options::scale);

    py::class_<Basis>(m, "EkkBasis")
        .def(py::init<>())
        .def_readwrite("basic_index", &Basis::basic_index)
        .def_readwrite("nonbasic_flag", &Basis::nonbasic_flag)
        .def_readwrite("nonbasic_move", &Basis::nonbasic_move);

    // HiGHS-style dual simplex on min c'x s.t. row_lower <= Ax <= row_upper,
    // col_lower <= x <= col_upper. Keeps its basis and factorization across
    // bound changes for hot starts.
    py::class_<DualSimplex>(m, "EkkDualSimplex")
        .def(py::init<Options>(), py::arg("options") = Options{})
        .def(
            "load",
            [](DualSimplex& self, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A,
               std::vector<double> c, std::vector<double> col_lower,
               std::vector<double> col_upper, std::vector<double> row_lower,
               std::vector<double> row_upper, double offset) {
                self.load(LpData::from_sparse(A, std::move(c), std::move(col_lower),
                                              std::move(col_upper), std::move(row_lower),
                                              std::move(row_upper), offset));
            },
            py::arg("A"), py::arg("c"), py::arg("col_lower"), py::arg("col_upper"),
            py::arg("row_lower"), py::arg("row_upper"), py::arg("offset") = 0.0)
        .def("solve", &DualSimplex::solve, py::call_guard<py::gil_scoped_release>())
        .def("change_col_bounds", &DualSimplex::change_col_bounds)
        .def("change_row_bounds", &DualSimplex::change_row_bounds)
        .def("set_objective_bound", &DualSimplex::set_objective_bound)
        .def_property("basis", &DualSimplex::basis, &DualSimplex::set_basis)
        .def_property_readonly("status", &DualSimplex::status)
        .def_property_readonly("iterations", &DualSimplex::iterations)
        .def_property_readonly("objective",
                               [](const DualSimplex& self) { return self.solution().objective; })
        .def_property_readonly("x", [](const DualSimplex& self) { return self.solution().col_value; })
        .def_property_readonly("reduced_costs",
                               [](const DualSimplex& self) { return self.solution().col_dual; })
        .def_property_readonly("row_activity",
                               [](const DualSimplex& self) { return self.solution().row_value; })
        .def_property_readonly("row_duals",
                               [](const DualSimplex& self) { return self.solution().row_dual; });
}
