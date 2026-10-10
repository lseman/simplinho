#include "bnb/cuts/cuts.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace simplex::bnb;

namespace {

bool test_tableau_mir_does_not_require_gomory() {
    Problem problem;
    problem.lower_bounds = Eigen::VectorXd::Zero(3);
    problem.upper_bounds = Eigen::VectorXd::Ones(3);
    problem.objective_coefficients = Eigen::VectorXd::Zero(3);
    problem.variable_types = {VariableType::Integer, VariableType::Continuous,
                              VariableType::Continuous};

    RelaxationSolution relaxation;
    relaxation.status = RelaxationStatus::Optimal;
    relaxation.primal = Eigen::VectorXd(3);
    relaxation.primal << 0.75, 0.0, 0.0;

    LPSolution lp;
    lp.has_internal_tableau = true;
    lp.tableau = Eigen::MatrixXd(1, 3);
    lp.tableau << 1.0, -0.5, -0.25;
    lp.tableau_rhs = Eigen::VectorXd::Constant(1, 0.75);
    lp.basis_internal = {0};
    lp.internal_column_labels = {"x_orig_0", "x_orig_1", "x_orig_2"};
    lp.basis_state.column_status = {LPBasisStatus::Basic, LPBasisStatus::AtLower,
                                    LPBasisStatus::AtLower};
    lp.basis_state.basis_columns = {0};
    relaxation.lp_solution = std::move(lp);

    Options options;
    options.use_gomory_cuts = false;
    options.use_mir_cuts = true;
    options.min_cut_violation = 1e-7;

    const std::vector<Cut> cuts = detail::generate_mir_cuts(problem, relaxation, options);
    for (const Cut& cut : cuts) {
        if ((cut.cut_type == "TMIR" || cut.cut_type == "TCMIR") &&
            detail::cut_violation(cut, relaxation.primal) > options.min_cut_violation) {
            return true;
        }
    }
    std::cerr << "MIR-only separation did not produce a tableau MIR cut\n";
    return false;
}

bool test_zero_half_combines_more_than_two_rows() {
    Problem problem;
    constexpr int variable_count = 5;
    problem.lower_bounds = Eigen::VectorXd::Zero(variable_count);
    problem.upper_bounds = Eigen::VectorXd::Ones(variable_count);
    problem.objective_coefficients = Eigen::VectorXd::Ones(variable_count);
    problem.variable_types.assign(variable_count, VariableType::Binary);
    for (int i = 0; i < variable_count; ++i) {
        SparseLinearConstraint edge;
        edge.indices = {i, (i + 1) % variable_count};
        edge.values = {1.0, 1.0};
        edge.rhs = 1.0;
        edge.sense = LinearConstraintSense::LessEqual;
        problem.base_constraints.push_back(std::move(edge));
    }

    RelaxationSolution relaxation;
    relaxation.status = RelaxationStatus::Optimal;
    relaxation.primal = Eigen::VectorXd::Constant(variable_count, 0.5);

    Options options;
    options.use_zero_half_cuts = true;
    options.min_cut_violation = 1e-7;
    const std::vector<Cut> cuts = detail::generate_zero_half_cuts(problem, relaxation, options);

    const Cut* separating_cut = nullptr;
    for (const Cut& cut : cuts) {
        if (cut.cut_type == "ZeroHalf" &&
            detail::cut_violation(cut, relaxation.primal) > options.min_cut_violation) {
            separating_cut = &cut;
            break;
        }
    }
    if (separating_cut == nullptr) {
        std::cerr << "zero-half separation missed the five-row odd-cycle cut\n";
        return false;
    }

    // Exhaustively verify validity on every binary point satisfying the cycle rows.
    for (int mask = 0; mask < (1 << variable_count); ++mask) {
        bool feasible = true;
        for (int i = 0; i < variable_count; ++i) {
            const int lhs = ((mask >> i) & 1) + ((mask >> ((i + 1) % variable_count)) & 1);
            if (lhs > 1) {
                feasible = false;
                break;
            }
        }
        if (!feasible)
            continue;

        Eigen::VectorXd point(variable_count);
        for (int i = 0; i < variable_count; ++i)
            point(i) = static_cast<double>((mask >> i) & 1);
        if (detail::cut_violation(*separating_cut, point) > 1e-9) {
            std::cerr << "zero-half separator emitted a cut that excludes a feasible point\n";
            return false;
        }
    }
    return true;
}

bool test_modk_uses_nonunit_row_weights() {
    Problem problem;
    problem.lower_bounds = Eigen::VectorXd::Zero(2);
    problem.upper_bounds = Eigen::VectorXd::Ones(2);
    problem.objective_coefficients = Eigen::VectorXd::Ones(2);
    problem.variable_types = {VariableType::Binary, VariableType::Binary};

    SparseLinearConstraint row;
    row.indices = {0, 1};
    row.values = {3.0, 3.0};
    row.rhs = 1.0;
    row.sense = LinearConstraintSense::LessEqual;
    problem.base_constraints.push_back(row);

    RelaxationSolution relaxation;
    relaxation.status = RelaxationStatus::Optimal;
    relaxation.primal = Eigen::VectorXd::Constant(2, 0.15);

    Options options;
    options.use_zero_half_cuts = true;
    options.min_cut_violation = 1e-7;
    const std::vector<Cut> cuts = detail::generate_zero_half_cuts(problem, relaxation, options);

    const Cut* separating_cut = nullptr;
    for (const Cut& cut : cuts) {
        if (cut.cut_type == "ModK" &&
            detail::cut_violation(cut, relaxation.primal) > options.min_cut_violation) {
            separating_cut = &cut;
            break;
        }
    }
    if (separating_cut == nullptr) {
        std::cerr << "Mod-k separation did not use the required weight two modulo three\n";
        return false;
    }

    for (int mask = 0; mask < 4; ++mask) {
        const int x0 = mask & 1;
        const int x1 = (mask >> 1) & 1;
        if (3 * x0 + 3 * x1 > 1)
            continue;
        Eigen::VectorXd point(2);
        point << static_cast<double>(x0), static_cast<double>(x1);
        if (detail::cut_violation(*separating_cut, point) > 1e-9) {
            std::cerr << "Mod-k separator emitted a cut that excludes a feasible point\n";
            return false;
        }
    }
    return true;
}

bool test_implied_bound_cut_is_valid() {
    Problem problem;
    problem.lower_bounds = Eigen::VectorXd::Zero(2);
    problem.upper_bounds = Eigen::VectorXd(2);
    problem.upper_bounds << 1.0, 10.0;
    problem.objective_coefficients = Eigen::VectorXd::Zero(2);
    problem.variable_types = {VariableType::Binary, VariableType::Continuous};

    SparseLinearConstraint row;
    row.indices = {1, 0};
    row.values = {1.0, 8.0};
    row.rhs = 10.0;
    row.sense = LinearConstraintSense::LessEqual;
    problem.base_constraints.push_back(row);

    RelaxationSolution relaxation;
    relaxation.status = RelaxationStatus::Optimal;
    relaxation.primal = Eigen::VectorXd(2);
    relaxation.primal << 0.75, 8.0;

    Options options;
    options.use_implied_bound_cuts = true;
    options.min_cut_violation = 1e-7;
    const std::vector<Cut> cuts = detail::generate_implied_bound_cuts(problem, relaxation, options);
    if (cuts.empty()) {
        std::cerr << "implied-bound separation did not emit the expected interpolation cut\n";
        return false;
    }

    const Cut& cut = cuts.front();
    if (cut.cut_type != "ImpliedBound" ||
        detail::cut_violation(cut, relaxation.primal) <= options.min_cut_violation) {
        std::cerr << "implied-bound separation emitted a nonseparating cut\n";
        return false;
    }

    // y=0 permits x in [0,10], while y=1 permits x in [0,2]. Check the complete
    // interval endpoints and a grid of interior feasible points for both disjunctions.
    for (int y = 0; y <= 1; ++y) {
        const double upper = y == 0 ? 10.0 : 2.0;
        for (int step = 0; step <= 20; ++step) {
            Eigen::VectorXd point(2);
            point << static_cast<double>(y), upper * static_cast<double>(step) / 20.0;
            if (detail::cut_violation(cut, point) > 1e-9) {
                std::cerr << "implied-bound cut excludes a feasible disjunctive point\n";
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    if (!test_tableau_mir_does_not_require_gomory())
        return 1;
    if (!test_zero_half_combines_more_than_two_rows())
        return 1;
    if (!test_modk_uses_nonunit_row_weights())
        return 1;
    if (!test_implied_bound_cut_is_valid())
        return 1;
    return 0;
}
