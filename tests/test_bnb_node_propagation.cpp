#include "bnb/presolve/mip_presolve.h"

#include <cmath>
#include <iostream>

using namespace simplex::bnb;

int main() {
    Problem problem;
    constexpr int variable_count = 5;
    problem.lower_bounds = Eigen::VectorXd::Zero(variable_count);
    problem.upper_bounds = Eigen::VectorXd::Ones(variable_count);
    problem.upper_bounds(variable_count - 1) = 0.0;
    problem.objective_coefficients = Eigen::VectorXd::Zero(variable_count);
    problem.variable_types.assign(variable_count, VariableType::Continuous);

    // The rows are deliberately ordered against the direction of propagation.
    // Two scan rounds reach only x2 <= 0; a changed-variable queue reaches the
    // fixpoint x0..x4 <= 0.
    for (int i = 0; i + 1 < variable_count; ++i) {
        SparseLinearConstraint row;
        row.indices = {i, i + 1};
        row.values = {1.0, -1.0};
        row.rhs = 0.0;
        row.sense = LinearConstraintSense::LessEqual;
        problem.base_constraints.push_back(std::move(row));
    }

    const presolve::NodeBoundPresolveResult result = presolve::presolve_mip_node_bounds(
        problem, problem.lower_bounds, problem.upper_bounds, {}, 1e-9, 8);
    if (result.infeasible) {
        std::cerr << "propagation incorrectly declared the feasible chain infeasible\n";
        return 1;
    }
    for (int i = 0; i < variable_count; ++i) {
        if (std::abs(result.upper(i)) > 1e-12) {
            std::cerr << "propagation stopped before fixpoint at variable " << i << ": "
                      << result.upper(i) << '\n';
            return 1;
        }
    }
    return 0;
}
