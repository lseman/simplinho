#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bindings.h"
#include "ipm/IPSolver.h"
#include "simplex/simplex.h"

namespace py = pybind11;

namespace {

struct IPMSolution {
    std::vector<double> primals;
    std::vector<double> duals;
    double objective = std::numeric_limits<double>::quiet_NaN();
    std::string status;
};

Eigen::VectorXd default_sense(int rows) { return Eigen::VectorXd::Ones(rows); }

Eigen::VectorXd numeric_sense(const Eigen::VectorXd& raw, int rows) {
    if (raw.size() != rows) {
        throw std::invalid_argument("ipm: sense length must match the number of rows in A");
    }
    return raw;
}

Eigen::VectorXd parse_sense(py::object sense, int rows, Eigen::VectorXd& row_scale) {
    row_scale = Eigen::VectorXd::Ones(rows);
    if (sense.is_none()) {
        return default_sense(rows);
    }
    if (py::isinstance<py::array>(sense)) {
        return numeric_sense(sense.cast<Eigen::VectorXd>(), rows);
    }

    std::vector<std::string> tokens = sense.cast<std::vector<std::string>>();
    if (static_cast<int>(tokens.size()) != rows) {
        throw std::invalid_argument("ipm: sense length must match the number of rows in A");
    }

    Eigen::VectorXd out(rows);
    for (int i = 0; i < rows; ++i) {
        std::string token = tokens[i];
        for (char& ch : token) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        if (token == "=" || token == "==" || token == "eq" || token == "e") {
            out[i] = 1.0;
        } else if (token == "<" || token == "<=" || token == "le" || token == "l") {
            out[i] = 0.0;
        } else if (token == ">" || token == ">=" || token == "ge" || token == "g") {
            out[i] = 0.0;
            row_scale[i] = -1.0;
        } else {
            throw std::invalid_argument("ipm: sense entries must be '=', '<=', or '>='");
        }
    }
    return out;
}

Eigen::SparseMatrix<double> normalize_rows(const Eigen::SparseMatrix<double>& A,
                                           const Eigen::VectorXd& row_scale) {
    Eigen::SparseMatrix<double> normalized(A.rows(), A.cols());
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(A.nonZeros());

    for (int col = 0; col < A.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, col); it; ++it) {
            trips.emplace_back(it.row(), it.col(), row_scale[it.row()] * it.value());
        }
    }
    normalized.setFromTriplets(trips.begin(), trips.end());
    normalized.makeCompressed();
    return normalized;
}

void validate_lp_inputs(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                        const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                        const Eigen::VectorXd& ub) {
    if (A.rows() != b.size()) {
        throw std::invalid_argument("ipm: b length must match the number of rows in A");
    }
    if (A.cols() != c.size()) {
        throw std::invalid_argument("ipm: c length must match the number of columns in A");
    }
    if (lb.size() != c.size() || ub.size() != c.size()) {
        throw std::invalid_argument("ipm: lb and ub lengths must match c");
    }
}

IPMSolution solve_ipm(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                      const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                      const Eigen::VectorXd& ub, py::object sense, double tol) {
    validate_lp_inputs(A, b, c, lb, ub);

    Eigen::VectorXd row_scale;
    Eigen::VectorXd parsed_sense = parse_sense(std::move(sense), A.rows(), row_scale);
    Eigen::SparseMatrix<double> normalized_A = normalize_rows(A, row_scale);
    Eigen::VectorXd normalized_b = row_scale.array() * b.array();

    IPSolver solver;
    solver.solve(normalized_A, normalized_b, c, lb, ub, parsed_sense, tol);

    IPMSolution out{solver.getPrimals(), solver.getDuals(), solver.getObjective(), "ipm"};
    if (std::isfinite(out.objective)) {
        return out;
    }

    const bool equality_form = (parsed_sense.array() == 1.0).all();
    if (!equality_form) {
        out.status = "ipm_nonfinite";
        return out;
    }

    RevisedSimplex fallback;
    LPSolution fallback_solution = fallback.solve(normalized_A, normalized_b, c, lb, ub);
    out.primals.assign(fallback_solution.x.data(),
                       fallback_solution.x.data() + fallback_solution.x.size());
    out.duals.clear();
    out.objective = fallback_solution.obj;
    out.status = std::string("fallback_") + to_string(fallback_solution.status);
    return out;
}

IPMSolution solve_ipm_dense(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                            const Eigen::VectorXd& ub, py::object sense, double tol) {
    return solve_ipm(A.sparseView(), b, c, lb, ub, std::move(sense), tol);
}

} // namespace

void bind_ipm_bindings(py::module_& m) {
    py::class_<IPMSolution>(m, "IPMSolution")
        .def_property_readonly("x", [](const IPMSolution& self) { return self.primals; })
        .def_property_readonly("primals", [](const IPMSolution& self) { return self.primals; })
        .def_property_readonly("duals", [](const IPMSolution& self) { return self.duals; })
        .def_property_readonly("status", [](const IPMSolution& self) { return self.status; })
        .def_property_readonly("obj", [](const IPMSolution& self) { return self.objective; })
        .def_property_readonly("objective", [](const IPMSolution& self) { return self.objective; });

    py::class_<IPSolver>(m, "IPSolver")
        .def(py::init<>())
        .def(
            "solve",
            [](IPSolver&, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               py::object sense,
               double tol) { return solve_ipm_dense(A, b, c, lb, ub, std::move(sense), tol); },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"), py::arg("ub"),
            py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
            "Solve min c^T x subject to A x =/<=/>= b and lb <= x <= ub.")
        .def(
            "solve",
            [](IPSolver&, const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               py::object sense,
               double tol) { return solve_ipm(A, b, c, lb, ub, std::move(sense), tol); },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"), py::arg("ub"),
            py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
            "Solve min c^T x subject to sparse A x =/<=/>= b and lb <= x <= ub.");

    m.def("solve_ipm", &solve_ipm_dense, py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"),
          py::arg("ub"), py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a dense matrix.");
    m.def("solve_ipm", &solve_ipm, py::arg("A"), py::arg("b"), py::arg("c"), py::arg("lb"),
          py::arg("ub"), py::arg("sense") = py::none(), py::arg("tol") = 1e-8,
          "Convenience wrapper for IPSolver.solve using a sparse matrix.");
}
