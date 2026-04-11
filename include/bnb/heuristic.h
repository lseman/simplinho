#pragma once

#include <Eigen/Dense>

#include <functional>
#include <optional>
#include <vector>

#include "bnb/diving.h"

namespace simplex::bnb::detail {

struct NeighborhoodHeuristicResult {
    std::optional<RelaxationSolution> incumbent;
    int lp_iterations = 0;
    int successes = 0;
};

using SubproblemSolveCallback =
    std::function<SolveResult(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
using SubproblemSolveWithCutsCallback = std::function<SolveResult(
    const Eigen::VectorXd&, const Eigen::VectorXd&, const std::vector<Cut>&)>;

std::optional<RelaxationSolution> run_rounding_heuristic(const Problem& problem,
                                                         const Options& options,
                                                         const RelaxationSolution& lp_relaxation,
                                                         const std::vector<Cut>& active_cuts);

NeighborhoodHeuristicResult
run_feasibility_jump_heuristic(const Problem& problem, const Options& options,
                               const RelaxationSolution& lp_relaxation,
                               const SubproblemSolveCallback& solve_submip);

NeighborhoodHeuristicResult
run_feasibility_pump_heuristic(const Problem& problem, const Options& options,
                               const RelaxationSolution& lp_relaxation,
                               const SubproblemSolveCallback& solve_submip);

NeighborhoodHeuristicResult run_rens_heuristic(const Problem& problem, const Options& options,
                                               const RelaxationSolution& lp_relaxation,
                                               const SubproblemSolveCallback& solve_submip);

NeighborhoodHeuristicResult run_rins_heuristic(const Problem& problem, const Options& options,
                                               const RelaxationSolution& lp_relaxation,
                                               const Eigen::VectorXd& incumbent_primal,
                                               double incumbent_objective,
                                               const SubproblemSolveCallback& solve_submip);

NeighborhoodHeuristicResult run_local_search_heuristic(const Problem& problem,
                                                       const Options& options,
                                                       const RelaxationSolution& lp_relaxation,
                                                       const Eigen::VectorXd& incumbent_primal,
                                                       double incumbent_objective,
                                                       const SubproblemSolveCallback& solve_submip);

NeighborhoodHeuristicResult
run_local_branching_heuristic(const Problem& problem, const Options& options,
                              const RelaxationSolution& lp_relaxation,
                              const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
                              const SubproblemSolveWithCutsCallback& solve_submip_with_cuts);

} // namespace simplex::bnb::detail
