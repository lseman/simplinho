#include "bnb/core/core.h"

#include <iostream>
#include <limits>
#include <utility>
#include <vector>

using namespace simplex::bnb;

namespace {

bool test_exhausted_lp_failure_is_not_certified() {
    Problem problem;
    problem.lower_bounds = Eigen::VectorXd::Zero(1);
    problem.upper_bounds = Eigen::VectorXd::Ones(1);
    problem.objective_coefficients = Eigen::VectorXd::Ones(1);
    problem.variable_types = {VariableType::Binary};

    Options options;
    options.parallel_workers = 1;
    options.max_nodes = 16;
    options.use_cut_pool = false;
    options.use_node_presolve = false;
    options.use_async_heuristics = false;

    int solve_calls = 0;
    auto failing_relaxation = [&](const Eigen::VectorXd& lower, const Eigen::VectorXd&,
                                  const LPBasis*, const std::vector<Cut>&) {
        ++solve_calls;
        RelaxationSolution out;
        out.status = RelaxationStatus::Infeasible;
        out.primal =
            Eigen::VectorXd::Constant(lower.size(), std::numeric_limits<double>::quiet_NaN());
        out.objective = std::numeric_limits<double>::infinity();
        out.lp_failed = true;
        return out;
    };

    const SolveResult result = Solver(std::move(problem), options).solve(failing_relaxation);
    if (solve_calls == 0) {
        std::cerr << "the failing relaxation callback was never invoked\n";
        return false;
    }
    if (result.status != Status::NodeLimit) {
        std::cerr << "an exhausted LP failure was incorrectly certified with status "
                  << to_string(result.status) << '\n';
        return false;
    }
    if (result.incumbent_updates != 0) {
        std::cerr << "an exhausted LP failure manufactured an incumbent\n";
        return false;
    }
    return true;
}

bool test_node_cut_gate_matches_node_separators() {
    Options options;
    options.use_gomory_cuts = false;
    options.use_mir_cuts = false;
    options.use_cover_cuts = false;
    options.use_zero_half_cuts = false;
    options.use_implied_bound_cuts = false;
    options.use_clique_cuts = false;
    options.use_odd_cycle_cuts = false;
    options.use_probing_implications = false;
    options.use_conflict_cuts = false;
    options.use_dual_proof_cuts = false;

    if (detail::has_enabled_node_cut_separator(options)) {
        std::cerr << "disabled node separators unexpectedly enabled node cut rounds\n";
        return false;
    }

    options.use_gomory_cuts = true;
    options.use_mir_cuts = true;
    options.use_probing_implications = true;
    options.use_conflict_cuts = true;
    options.use_dual_proof_cuts = true;
    if (detail::has_enabled_node_cut_separator(options)) {
        std::cerr << "root-only or non-phase separators enabled empty node cut rounds\n";
        return false;
    }

    options.use_zero_half_cuts = true;
    if (!detail::has_enabled_node_cut_separator(options)) {
        std::cerr << "zero-half-only separation did not enable node cut rounds\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    if (!test_exhausted_lp_failure_is_not_certified())
        return 1;
    if (!test_node_cut_gate_matches_node_separators())
        return 1;
    return 0;
}
