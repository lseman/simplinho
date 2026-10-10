#include "simplex/engine/simplex.h"
#include "simplex/presolve/sparse_presolver.h"

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using SparseMatrix = RevisedSimplex::SparseMatrix;

bool near(double lhs, double rhs, double tolerance = 1e-8) {
    return std::abs(lhs - rhs) <= tolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

bool test_sparse_equilibration_contract() {
    presolve::SparseLP lp;
    lp.A.resize(2, 3);
    std::vector<Eigen::Triplet<double>> entries{{0, 0, 1e-8}, {1, 0, 2e4},
                                                {0, 1, -3e7}, {1, 2, 5e-5}};
    lp.A.setFromTriplets(entries.begin(), entries.end());
    lp.A.makeCompressed();
    lp.b = (Eigen::VectorXd(2) << 4.0, -7.0).finished();
    lp.c = (Eigen::VectorXd(3) << 1e-4, -8.0, 2e5).finished();
    lp.l = (Eigen::VectorXd(3) << 2.0, -1.0, 0.0).finished();
    lp.u = (Eigen::VectorXd(3) << 9.0, 6.0, presolve::inf()).finished();
    const presolve::SparseLP original = lp;

    presolve::SparsePresolver::Options options;
    options.max_passes = 0;
    options.enable_singleton_rows = false;
    options.enable_activity_tightening = false;
    options.enable_zero_columns = false;
    options.force_equilibration = true;
    options.scaling_passes = 6;
    presolve::SparsePresolver presolver(options);
    const auto result = presolver.run(lp);

    if (result.row_scale_count == 0 || result.col_scale_count == 0) {
        std::cerr << "equilibration did not produce nontrivial row and column factors\n";
        return false;
    }
    for (int j = 0; j < original.A.outerSize(); ++j) {
        for (SparseMatrix::InnerIterator it(original.A, j); it; ++it) {
            const double expected =
                it.value() / (result.row_scale(it.row()) * result.col_scale(j));
            if (!near(result.reduced.A.coeff(it.row(), j), expected, 5e-13)) {
                std::cerr << "scaled matrix violates the row/column scale contract\n";
                return false;
            }
        }
        if (!near(result.reduced.c(j), original.c(j) / result.col_scale(j), 5e-13) ||
            !near(result.reduced.l(j), original.l(j) * result.col_scale(j), 5e-13) ||
            (std::isfinite(original.u(j)) &&
             !near(result.reduced.u(j), original.u(j) * result.col_scale(j), 5e-13))) {
            std::cerr << "scaled objective or bounds violate the column scale contract\n";
            return false;
        }
    }
    for (int i = 0; i < original.b.size(); ++i) {
        if (!near(result.reduced.b(i), original.b(i) / result.row_scale(i), 5e-13)) {
            std::cerr << "scaled RHS violates the row scale contract\n";
            return false;
        }
    }
    return true;
}

bool test_scaled_exports_and_warm_factorization() {
    constexpr int rows = 3;
    constexpr int cols = 7;
    SparseMatrix A(rows, cols);
    std::vector<Eigen::Triplet<double>> entries{{0, 0, 1e-6}, {0, 3, 1.0},
                                                {1, 1, 1e6},  {1, 4, 1.0},
                                                {2, 2, 2.0},  {2, 5, 1.0}};
    A.setFromTriplets(entries.begin(), entries.end());
    A.makeCompressed();

    Eigen::VectorXd b(rows);
    b << 3e-6, 1.0, 1.0;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(cols);
    c.segment<3>(3).setOnes();
    Eigen::VectorXd lower = Eigen::VectorXd::Zero(cols);
    Eigen::VectorXd upper =
        Eigen::VectorXd::Constant(cols, std::numeric_limits<double>::infinity());
    // Exercise the sign-flip and nonzero-anchor path as well as equilibration.
    lower(0) = -std::numeric_limits<double>::infinity();
    upper(0) = 5.0;

    RevisedSimplexOptions options;
    options.mode = SimplexMode::Dual;
    options.simplex_scaling = true;
    options.force_equilibration = true;
    options.compute_tableau = true;
    options.compute_reduced_costs = true;
    RevisedSimplex solver(options);

    const std::vector<int> initial_basis{0, 1, 2};
    const LPSolution first = solver.solve(A, b, c, lower, upper, initial_basis);
    if (first.status != LPSolution::Status::Optimal) {
        std::cerr << "scaled sparse solve did not reach optimality: " << to_string(first.status)
                  << "\n";
        return false;
    }
    if (!near(first.x(0), 3.0, 1e-7) || !near(first.x(1), 1e-6, 1e-7) ||
        !near(first.x(2), 0.5, 1e-7) || (A * first.x - b).lpNorm<Eigen::Infinity>() > 1e-8) {
        std::cerr << "scaled primal solution was not restored to original coordinates\n";
        return false;
    }
    if (first.info.find("simplex_scaled") == first.info.end() ||
        first.info.at("simplex_scaled") != "1") {
        std::cerr << "scaled solve did not expose scaling telemetry\n";
        return false;
    }
    if (!first.has_internal_tableau || first.tableau.rows() != rows ||
        first.tableau.cols() != cols ||
        (first.tableau * first.x - first.tableau_rhs).lpNorm<Eigen::Infinity>() > 1e-8) {
        std::cerr << "tableau was not restored to original variable coordinates\n";
        return false;
    }
    if (first.dual_values.size() != rows || first.reduced_costs_internal.size() != cols ||
        (c - A.transpose() * first.dual_values - first.reduced_costs_internal)
                .lpNorm<Eigen::Infinity>() >
            1e-8) {
        std::cerr << "duals and reduced costs were not restored consistently\n";
        return false;
    }

    const LPSolution second = solver.solve(A, b, c, lower, upper, first.basis_state);
    if (second.status != LPSolution::Status::Optimal ||
        second.solve_stats.warm_start_attempted != 1 ||
        second.solve_stats.warm_start_accepted != 1 ||
        second.solve_stats.warm_factorization_reused != 1 ||
        (A * second.x - b).lpNorm<Eigen::Infinity>() > 1e-8) {
        std::cerr << "scaled warm solve did not reuse a valid factorization\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    const bool ok = test_sparse_equilibration_contract() &&
                    test_scaled_exports_and_warm_factorization();
    if (!ok)
        return 1;
    std::cout << "LP scaling and warm-start tests passed\n";
    return 0;
}
