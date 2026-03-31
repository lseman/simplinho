#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include "simplex/simplex_types.h"

namespace simplex::bnb {

enum class VariableType { Continuous, Integer, Binary };
enum class Status { Optimal, Infeasible, Unbounded, NodeLimit };
enum class NodeSelectionStrategy {
  DepthFirst,
  BreadthFirst,
  BestBound,
  BestEstimate,
  Hybrid
};
enum class BranchingStrategy { MostFractional, PseudoCost, StrongBranching };
enum class DivingStrategy {
  Disabled,
  Fractional,
  VectorLength,
  ObjectiveValue,
  Coefficient,
  Guided,
  Adaptive
};
enum class RelaxationStatus { Optimal, Infeasible, Unbounded };
enum class LinearConstraintSense { LessEqual, GreaterEqual, Equal };
enum class TreeNodeStatus {
  Created,
  Fractional,
  Integral,
  Infeasible,
  Unbounded,
  PrunedByBound,
  Branched,
  Fathomed
};

inline const char *to_string(Status status) {
  switch (status) {
  case Status::Optimal:
    return "optimal";
  case Status::Infeasible:
    return "infeasible";
  case Status::Unbounded:
    return "unbounded";
  case Status::NodeLimit:
    return "nodelimit";
  }
  return "unknown";
}

inline const char *to_string(BranchingStrategy strategy) {
  switch (strategy) {
  case BranchingStrategy::MostFractional:
    return "most_fractional";
  case BranchingStrategy::PseudoCost:
    return "pseudocost";
  case BranchingStrategy::StrongBranching:
    return "strong_branching";
  }
  return "unknown";
}

inline const char *to_string(DivingStrategy strategy) {
  switch (strategy) {
  case DivingStrategy::Disabled:
    return "disabled";
  case DivingStrategy::Fractional:
    return "fractional_diving";
  case DivingStrategy::VectorLength:
    return "vector_length_diving";
  case DivingStrategy::ObjectiveValue:
    return "objective_diving";
  case DivingStrategy::Coefficient:
    return "coefficient_diving";
  case DivingStrategy::Guided:
    return "guided_diving";
  case DivingStrategy::Adaptive:
    return "adaptive_diving";
  }
  return "unknown";
}

inline const char *to_string(NodeSelectionStrategy strategy) {
  switch (strategy) {
  case NodeSelectionStrategy::DepthFirst:
    return "depth_first";
  case NodeSelectionStrategy::BreadthFirst:
    return "breadth_first";
  case NodeSelectionStrategy::BestBound:
    return "best_bound";
  case NodeSelectionStrategy::BestEstimate:
    return "best_estimate";
  case NodeSelectionStrategy::Hybrid:
    return "hybrid";
  }
  return "unknown";
}

inline const char *to_string(TreeNodeStatus status) {
  switch (status) {
  case TreeNodeStatus::Created:
    return "created";
  case TreeNodeStatus::Fractional:
    return "fractional";
  case TreeNodeStatus::Integral:
    return "integral";
  case TreeNodeStatus::Infeasible:
    return "infeasible";
  case TreeNodeStatus::Unbounded:
    return "unbounded";
  case TreeNodeStatus::PrunedByBound:
    return "pruned_by_bound";
  case TreeNodeStatus::Branched:
    return "branched";
  case TreeNodeStatus::Fathomed:
    return "fathomed";
  }
  return "unknown";
}

struct Options {
  int max_nodes = 10'000;
  int parallel_workers =
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  double integrality_tol = 1e-6;
  bool verbose = false;
  int log_frequency = 100;
  NodeSelectionStrategy node_selection = NodeSelectionStrategy::DepthFirst;
  int hybrid_depth_bias = 5;
  BranchingStrategy branching_strategy = BranchingStrategy::MostFractional;
  DivingStrategy diving_strategy = DivingStrategy::Disabled;
  int strong_branching_candidates = 6;
  int strong_branching_max_depth = 4;
  int pseudocost_reliability = 2;
  int max_dive_depth = 25;
  int max_dive_lp_solves = 64;
  int heuristic_frequency = 8;
  int heuristic_max_depth = 12;
  bool use_rins = false;
  double rins_fix_ratio = 0.7;
  double rins_tolerance = 1e-4;
  bool use_rens = false;
  double rens_fix_ratio = 0.75;
  bool use_local_search = false;
  int local_search_iterations = 8;
  int local_search_max_free_vars = 3;
  bool use_local_branching = false;
  double local_branching_neighborhood_ratio = 0.15;
  int local_branching_min_radius = 4;
  int local_branching_max_radius = 20;
  double local_branching_fix_agree_ratio = 0.35;
  double local_branching_lp_agreement_tol = 1e-4;
  bool use_feasibility_pump = false;
  int feasibility_pump_iterations = 6;
  double feasibility_pump_fix_ratio = 0.6;
  bool use_feasibility_jump = false;
  int feasibility_jump_iterations = 12;
  int feasibility_jump_max_free_vars = 4;
  double feasibility_jump_objective_weight = 0.05;
  int heuristic_subproblem_max_nodes = 64;
  bool use_cut_pool = false;
  int max_cut_rounds_per_node = 2;
  int max_cuts_added_per_round = 8;
  int max_cut_pool_size = 256;
  double min_cut_violation = 1e-4;
  int max_cut_age = 5;
  bool use_gomory_cuts = true;
  bool use_cover_cuts = true;
  bool use_implied_bound_cuts = true;
  bool use_clique_cuts = true;
  bool use_probing_implications = true;
  int probing_max_candidates = 4;
  bool use_conflict_cuts = true;
  int max_conflict_cuts_per_round = 4;
  int max_cuts_per_type = 4;
  double cut_max_parallelism = 0.98;
  bool use_dual_proof_cuts = true;
  // HiGHS-inspired separate tolerances
  double feasibility_tol = 1e-7;   // Feasibility checks in relaxation
  double optimality_tol = 1e-7;    // Optimality gap checks
  // integrality_tol is already defined above (line 138) with default 1e-6
};

struct SparseLinearConstraint {
  std::vector<int> indices;
  std::vector<double> values;
  double rhs = 0.0;
  LinearConstraintSense sense = LinearConstraintSense::LessEqual;
};

struct Cut {
  std::vector<int> indices;
  std::vector<double> values;
  double rhs = 0.0;
  LinearConstraintSense sense = LinearConstraintSense::LessEqual;
  std::string cut_type;
  double strength = 0.0;
  int times_used = 0;
  int age = 0;
};

struct Problem {
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  Eigen::VectorXd objective_coefficients;
  double objective_constant = 0.0;
  std::vector<VariableType> variable_types;
  std::vector<SparseLinearConstraint> base_constraints;
  bool maximize = false;
};

struct RelaxationSolution {
  RelaxationStatus status = RelaxationStatus::Infeasible;
  Eigen::VectorXd primal;
  double objective = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
  std::optional<LPBasis> basis;
  std::optional<LPSolution> lp_solution;
};

struct TreeNode {
  int id = -1;
  int parent_id = -1;
  int depth = 0;
  std::uint64_t order = 0;
  TreeNodeStatus status = TreeNodeStatus::Created;
  double bound = std::numeric_limits<double>::quiet_NaN();
  double estimate = std::numeric_limits<double>::quiet_NaN();
  int branch_var = -1;
  double branch_value = std::numeric_limits<double>::quiet_NaN();
};

struct SolveResult {
  Status status = Status::Infeasible;
  Eigen::VectorXd primal;
  double objective = std::numeric_limits<double>::quiet_NaN();
  double best_bound = std::numeric_limits<double>::quiet_NaN();
  double root_relaxation_objective = std::numeric_limits<double>::quiet_NaN();
  int root_presolve_tightened_bounds = 0;
  int root_presolve_removed_rows = 0;
  int root_presolve_removed_coeffs = 0;
  int root_presolve_aggregations = 0;
  int node_count = 0;
  int lp_iterations = 0;
  int incumbent_updates = 0;
  int heuristic_lp_iterations = 0;
  int heuristic_successes = 0;
  int feasibility_jump_successes = 0;
  int feasibility_pump_successes = 0;
  int rens_successes = 0;
  int rins_successes = 0;
  int local_search_successes = 0;
  int local_branching_successes = 0;
  int cuts_generated = 0;
  int cuts_applied = 0;
  int duplicate_cuts = 0;
  int cut_pool_size = 0;
  bool has_solution = false;
  std::vector<TreeNode> tree_nodes;
};

} // namespace simplex::bnb
