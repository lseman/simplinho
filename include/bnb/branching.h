#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/diving.h"
#include "bnb/parallel.h"

namespace simplex::bnb::detail {

// ============================================================================
// Core data structures
// ============================================================================

struct PseudoCostStats {
  double up_sum = 0.0;
  double down_sum = 0.0;
  int up_count = 0;
  int down_count = 0;

  double up_avg() const { return up_count > 0 ? up_sum / up_count : 0.0; }
  double down_avg() const {
    return down_count > 0 ? down_sum / down_count : 0.0;
  }
};

struct BranchSignalStats {
  int inference_up = 0;
  int inference_down = 0;

  double conflict_score_up = 0.0;
  double conflict_score_down = 0.0;

  int cutoff_up = 0;
  int cutoff_down = 0;
};

struct PseudoCost {
  static constexpr double kConflictWeight = 1.02;
  static constexpr int kMinReliable = 5;

  PseudoCostStats cost;
  BranchSignalStats signal;

  double up_avg() const { return cost.up_avg(); }
  double down_avg() const { return cost.down_avg(); }

  bool is_reliable() const {
    return std::min(cost.up_count, cost.down_count) >= kMinReliable;
  }

  void add_inference(bool up) {
    if (up) {
      ++signal.inference_up;
    } else {
      ++signal.inference_down;
    }
  }

  void add_cutoff(bool up) {
    if (up) {
      ++signal.cutoff_up;
    } else {
      ++signal.cutoff_down;
    }
  }

  void record_cutoff() {
    ++signal.cutoff_up;
    ++signal.cutoff_down;
  }

  void increase_conflict_score(bool up) {
    if (up) {
      signal.conflict_score_up *= kConflictWeight;
      signal.conflict_score_up += kConflictWeight;
    } else {
      signal.conflict_score_down *= kConflictWeight;
      signal.conflict_score_down += kConflictWeight;
    }
  }
};

struct ChildEvaluation {
  ChildState state;
  std::optional<RelaxationSolution> relaxation;
};

struct BranchDecision {
  int variable = -1;
  double value = std::numeric_limits<double>::quiet_NaN();
  ChildEvaluation down_child;
  ChildEvaluation up_child;
};

struct RankedPseudoCostCandidate {
  const FractionalCandidate *candidate = nullptr;
  double score = 0.0;
  bool reliable = false;
  int cutoff_up = 0;
  int cutoff_down = 0;
};

struct EvaluatedBranchCandidate {
  BranchDecision decision;
  double score = -std::numeric_limits<double>::infinity();
  const FractionalCandidate *candidate = nullptr;
};

struct PseudoCostAverages {
  double cost_avg = 0.0;
  double inference_avg = 0.0;
  double conflict_avg = 0.0;
  double cutoff_avg = 0.0;
};

// ============================================================================
// Numerical helpers
// ============================================================================

inline double safe_max(double x, double eps = 1e-12) {
  return std::max(eps, x);
}

inline double map_score(double score, double eps = 1e-12) {
  return 1.0 - 1.0 / (1.0 + safe_max(score, eps));
}

// ============================================================================
// Objective / branch scoring
// ============================================================================

inline double objective_degradation(double parent_objective,
                                    double child_objective, bool maximize) {
  const double raw = maximize ? (parent_objective - child_objective)
                              : (child_objective - parent_objective);
  return std::max(0.0, raw);
}

inline double child_branch_score(const RelaxationSolution &child,
                                 double parent_objective, double distance,
                                 bool maximize, double feasibility_tol,
                                 double integrality_tol) {
  // Use feasibility_tol for infeasibility detection (HiGHS-inspired)
  if (child.status == RelaxationStatus::Infeasible) {
    return std::numeric_limits<double>::infinity();
  }
  if (child.status != RelaxationStatus::Optimal) {
    return 0.0;
  }

  return objective_degradation(parent_objective, child.objective, maximize) /
         safe_max(distance, integrality_tol);
}

inline double combine_branch_scores(double down_score, double up_score) {
  if (std::isinf(down_score) || std::isinf(up_score)) {
    return std::numeric_limits<double>::infinity();
  }

  const double min_score = std::min(down_score, up_score);
  const double max_score = std::max(down_score, up_score);
  const double geometric_score =
      std::sqrt(std::max(0.0, down_score) * std::max(0.0, up_score));

  return 0.5 * min_score + 0.15 * max_score + 0.35 * geometric_score;
}

inline double branch_score(const FractionalCandidate &candidate,
                           double parent_objective,
                           const RelaxationSolution &down,
                           const RelaxationSolution &up, bool maximize,
                           double feasibility_tol, double integrality_tol) {
  const double down_score =
      child_branch_score(down, parent_objective, candidate.down_distance,
                         maximize, feasibility_tol, integrality_tol);
  const double up_score =
      child_branch_score(up, parent_objective, candidate.up_distance, maximize,
                         feasibility_tol, integrality_tol);
  return combine_branch_scores(down_score, up_score);
}

// ============================================================================
// Pseudocost feature extraction
// ============================================================================

inline double compute_cost_score(const PseudoCost &pc, double cost_avg,
                                 double eps = 1e-12) {
  return safe_max(pc.up_avg(), eps) * safe_max(pc.down_avg(), eps) /
         safe_max(cost_avg * cost_avg, eps);
}

inline double compute_conflict_score(const PseudoCost &pc, double conflict_avg,
                                     double eps = 1e-12) {
  return safe_max(pc.signal.conflict_score_up, eps) *
         safe_max(pc.signal.conflict_score_down, eps) /
         safe_max(conflict_avg * conflict_avg, eps);
}

inline double compute_inference_score(const PseudoCost &pc,
                                      double inference_avg,
                                      double eps = 1e-12) {
  return safe_max(static_cast<double>(pc.signal.inference_up), eps) *
         safe_max(static_cast<double>(pc.signal.inference_down), eps) /
         safe_max(inference_avg * inference_avg, eps);
}

inline double compute_cutoff_score(const PseudoCost &pc, double cutoff_avg,
                                   double eps = 1e-12) {
  return safe_max(static_cast<double>(pc.signal.cutoff_up), eps) *
         safe_max(static_cast<double>(pc.signal.cutoff_down), eps) /
         safe_max(cutoff_avg * cutoff_avg, eps);
}

inline double get_combined_pseudocost_score(const PseudoCost &pc,
                                            const PseudoCostAverages &avg,
                                            double eps = 1e-12) {
  const double cost_s =
      map_score(compute_cost_score(pc, avg.cost_avg, eps), eps);
  const double conflict_s =
      map_score(compute_conflict_score(pc, avg.conflict_avg, eps), eps);
  const double inference_s =
      map_score(compute_inference_score(pc, avg.inference_avg, eps), eps);
  const double cutoff_s =
      map_score(compute_cutoff_score(pc, avg.cutoff_avg, eps), eps);

  return 0.85 * cost_s + 0.02 * conflict_s + 0.01 * inference_s +
         0.01 * cutoff_s;
}

inline double
pseudocost_candidate_score_simple(const PseudoCost &pc,
                                  const FractionalCandidate &candidate) {
  constexpr double default_cost = 1.0;

  const double down_score =
      (pc.cost.down_count > 0 ? pc.down_avg() : default_cost) *
      candidate.down_distance;
  const double up_score = (pc.cost.up_count > 0 ? pc.up_avg() : default_cost) *
                          candidate.up_distance;

  return combine_branch_scores(down_score, up_score);
}

inline PseudoCostAverages
compute_pseudocost_averages(const std::vector<FractionalCandidate> &candidates,
                            const std::vector<PseudoCost> &pseudocosts) {
  PseudoCostAverages avg{};

  if (candidates.empty()) {
    return avg;
  }

  for (const auto &candidate : candidates) {
    const auto &pc = pseudocosts[candidate.variable];
    avg.cost_avg += pc.up_avg() + pc.down_avg();
    avg.inference_avg +=
        static_cast<double>(pc.signal.inference_up + pc.signal.inference_down);
    avg.conflict_avg +=
        pc.signal.conflict_score_up + pc.signal.conflict_score_down;
    avg.cutoff_avg +=
        static_cast<double>(pc.signal.cutoff_up + pc.signal.cutoff_down);
  }

  const double denom = static_cast<double>(candidates.size());
  avg.cost_avg /= denom;
  avg.inference_avg /= denom;
  avg.conflict_avg /= denom;
  avg.cutoff_avg /= denom;
  return avg;
}

inline double pseudocost_candidate_score(const PseudoCost &pc,
                                         const PseudoCostAverages &avg,
                                         double eps = 1e-12) {
  return get_combined_pseudocost_score(pc, avg, eps);
}

// ============================================================================
// Pseudocost updates
// ============================================================================

inline void record_cutoff(std::vector<PseudoCost> &pseudocosts, int var,
                          bool up) {
  pseudocosts[var].add_cutoff(up);
}

inline void update_pseudocosts(std::vector<PseudoCost> &pseudocosts,
                               const FractionalCandidate &candidate,
                               double parent_objective,
                               const RelaxationSolution &down,
                               const RelaxationSolution &up, bool maximize,
                               double feasibility_tol, double integrality_tol) {
  auto update_one = [&](bool branch_up, const RelaxationSolution &child,
                        double distance) {
    double cost_value = 0.0;

    // Use feasibility_tol for infeasibility detection (HiGHS-inspired)
    if (child.status == RelaxationStatus::Infeasible) {
      cost_value = 4.0 / safe_max(distance, integrality_tol);
    } else if (child.status == RelaxationStatus::Optimal) {
      cost_value =
          objective_degradation(parent_objective, child.objective, maximize) /
          safe_max(distance, integrality_tol);
    } else {
      return;
    }

    auto &pc = pseudocosts[candidate.variable];

    if (branch_up) {
      pc.cost.up_sum += cost_value;
      ++pc.cost.up_count;
      pc.add_inference(true);

      if (child.status == RelaxationStatus::Infeasible) {
        pc.increase_conflict_score(true);
      }
    } else {
      pc.cost.down_sum += cost_value;
      ++pc.cost.down_count;
      pc.add_inference(false);

      if (child.status == RelaxationStatus::Infeasible) {
        pc.increase_conflict_score(false);
      }
    }
  };

  update_one(false, down, candidate.down_distance);
  update_one(true, up, candidate.up_distance);
}

// ============================================================================
// Node estimate
// ============================================================================

inline double node_estimate(const RelaxationSolution &relaxation,
                            const std::vector<VariableType> &variable_types,
                            const std::vector<PseudoCost> &pseudocosts,
                            double integrality_tol, bool maximize) {
  if (relaxation.status != RelaxationStatus::Optimal) {
    return relaxation.objective;
  }

  const auto fractional = collect_fractional_candidates(
      relaxation.primal, variable_types, integrality_tol);
  if (fractional.empty()) {
    return relaxation.objective;
  }

  double degradation = 0.0;
  constexpr double default_cost = 1.0;

  for (const auto &candidate : fractional) {
    const auto &pc = pseudocosts[candidate.variable];

    const double down_score =
        (pc.cost.down_count > 0 ? pc.down_avg() : default_cost) *
        candidate.down_distance;
    const double up_score =
        (pc.cost.up_count > 0 ? pc.up_avg() : default_cost) *
        candidate.up_distance;

    degradation += std::min(down_score, up_score);
  }

  return maximize ? (relaxation.objective - degradation)
                  : (relaxation.objective + degradation);
}

// ============================================================================
// Branch construction
// ============================================================================

inline BranchDecision
build_decision_from_candidate(const ActiveNode &node,
                              const FractionalCandidate &candidate) {
  BranchDecision decision;
  decision.variable = candidate.variable;
  decision.value = candidate.value;
  decision.down_child.state =
      make_child_state(node, candidate.variable, false, candidate.value);
  decision.up_child.state =
      make_child_state(node, candidate.variable, true, candidate.value);
  return decision;
}

// ============================================================================
// Pseudocost-only choice without probing
// ============================================================================

inline BranchDecision choose_pseudocost_without_probing(
    const ActiveNode &node, const std::vector<FractionalCandidate> &candidates,
    const std::vector<PseudoCost> &pseudocosts) {
  if (candidates.empty()) {
    return {};
  }

  const FractionalCandidate *best = &candidates.front();
  double best_score = -std::numeric_limits<double>::infinity();
  bool saw_reliable = false;

  for (const auto &candidate : candidates) {
    if (candidate.variable < 0 ||
        candidate.variable >= static_cast<int>(pseudocosts.size())) {
      continue;
    }

    const auto &pc = pseudocosts[candidate.variable];
    const bool reliable = (pc.cost.up_count > 0 || pc.cost.down_count > 0);

    if (!reliable && saw_reliable) {
      continue;
    }

    const double score = pseudocost_candidate_score_simple(pc, candidate);

    if (!saw_reliable || reliable) {
      if (!saw_reliable && reliable) {
        best = &candidate;
        best_score = score;
        saw_reliable = true;
        continue;
      }
      if (score > best_score + 1e-12) {
        best = &candidate;
        best_score = score;
      }
    }
  }

  return build_decision_from_candidate(node, *best);
}

// ============================================================================
// Strong branching
// ============================================================================

template <typename RelaxationSolver>
inline BranchDecision choose_strong_branching(
    const ActiveNode &node, const RelaxationSolution &relaxation,
    const std::vector<FractionalCandidate> &candidates, int candidate_limit,
    int parallel_workers, bool maximize, double feasibility_tol,
    double integrality_tol, std::vector<PseudoCost> &pseudocosts,
    ParallelDispatcher *parallel_dispatcher,
    RelaxationSolver &&relaxation_solver) {
  BranchDecision best;
  double best_score = -std::numeric_limits<double>::infinity();

  const int limit = candidate_limit > 0
                        ? std::min<int>(candidate_limit, candidates.size())
                        : static_cast<int>(candidates.size());

  if (parallel_workers > 1 && limit > 1 && parallel_dispatcher != nullptr) {
    std::vector<std::optional<EvaluatedBranchCandidate>> evaluated(limit);

    parallel_dispatcher->run(limit, [&](int i) {
      EvaluatedBranchCandidate eval;
      eval.candidate = &candidates[i];
      eval.decision = build_decision_from_candidate(node, candidates[i]);

      eval.decision.down_child.relaxation =
          relaxation_solver(eval.decision.down_child.state.lower_bounds,
                            eval.decision.down_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      eval.decision.up_child.relaxation =
          relaxation_solver(eval.decision.up_child.state.lower_bounds,
                            eval.decision.up_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      eval.score = branch_score(candidates[i], relaxation.objective,
                                eval.decision.down_child.relaxation.value(),
                                eval.decision.up_child.relaxation.value(),
                                maximize, feasibility_tol, integrality_tol);

      evaluated[i] = std::move(eval);
    });

    for (auto &item : evaluated) {
      EvaluatedBranchCandidate eval = std::move(item).value();

      update_pseudocosts(pseudocosts, *eval.candidate, relaxation.objective,
                         eval.decision.down_child.relaxation.value(),
                         eval.decision.up_child.relaxation.value(), maximize,
                         feasibility_tol, integrality_tol);

      if (eval.score > best_score + 1e-12) {
        best_score = eval.score;
        best = std::move(eval.decision);
      }
    }

    if (best.variable >= 0) {
      return best;
    }
    return build_decision_from_candidate(node, candidates.front());
  }

  for (int i = 0; i < limit; ++i) {
    BranchDecision decision =
        build_decision_from_candidate(node, candidates[i]);

    decision.down_child.relaxation =
        relaxation_solver(decision.down_child.state.lower_bounds,
                          decision.down_child.state.upper_bounds,
                          node.basis ? &*node.basis : nullptr);

    decision.up_child.relaxation =
        relaxation_solver(decision.up_child.state.lower_bounds,
                          decision.up_child.state.upper_bounds,
                          node.basis ? &*node.basis : nullptr);

    update_pseudocosts(pseudocosts, candidates[i], relaxation.objective,
                       decision.down_child.relaxation.value(),
                       decision.up_child.relaxation.value(), maximize,
                       feasibility_tol, integrality_tol);

    const double score =
        branch_score(candidates[i], relaxation.objective,
                     decision.down_child.relaxation.value(),
                     decision.up_child.relaxation.value(), maximize,
                     feasibility_tol, integrality_tol);

    if (score > best_score + 1e-12) {
      best_score = score;
      best = std::move(decision);
    }
  }

  if (best.variable >= 0) {
    return best;
  }
  return build_decision_from_candidate(node, candidates.front());
}

// ============================================================================
// Pseudocost branching with optional strong branching on unreliable candidates
// ============================================================================

template <typename RelaxationSolver>
inline BranchDecision choose_pseudocost_branching(
    const ActiveNode &node, const RelaxationSolution &relaxation,
    const std::vector<FractionalCandidate> &candidates, int reliability,
    int strong_branching_candidates, int strong_branching_max_depth,
    int parallel_workers, bool maximize, double feasibility_tol,
    double integrality_tol, std::vector<PseudoCost> &pseudocosts,
    ParallelDispatcher *parallel_dispatcher,
    RelaxationSolver &&relaxation_solver) {
  if (candidates.empty()) {
    return {};
  }

  const PseudoCostAverages avg =
      compute_pseudocost_averages(candidates, pseudocosts);

  std::vector<RankedPseudoCostCandidate> ranked;
  ranked.reserve(candidates.size());

  for (const auto &candidate : candidates) {
    const auto &pc = pseudocosts[candidate.variable];
    ranked.push_back(RankedPseudoCostCandidate{
        .candidate = &candidate,
        .score = pseudocost_candidate_score(pc, avg),
        .reliable = (pc.cost.up_count >= reliability &&
                     pc.cost.down_count >= reliability),
        .cutoff_up = pc.signal.cutoff_up,
        .cutoff_down = pc.signal.cutoff_down});
  }

  std::sort(ranked.begin(), ranked.end(), [](const auto &lhs, const auto &rhs) {
    if (std::abs(lhs.score - rhs.score) > 1e-12) {
      return lhs.score > rhs.score;
    }
    if (lhs.reliable != rhs.reliable) {
      return lhs.reliable;
    }
    return lhs.candidate->variable < rhs.candidate->variable;
  });

  if (strong_branching_max_depth >= 0 &&
      node.depth > strong_branching_max_depth) {
    return build_decision_from_candidate(node, *ranked.front().candidate);
  }

  std::optional<BranchDecision> best_decision;
  double best_score = -std::numeric_limits<double>::infinity();

  for (const auto &item : ranked) {
    if (item.reliable) {
      best_decision = build_decision_from_candidate(node, *item.candidate);
      best_score = item.score;
      break;
    }
  }

  std::vector<const RankedPseudoCostCandidate *> to_evaluate;
  to_evaluate.reserve(ranked.size());

  for (const auto &item : ranked) {
    if (item.reliable) {
      continue;
    }
    if (strong_branching_candidates > 0 &&
        static_cast<int>(to_evaluate.size()) >= strong_branching_candidates) {
      break;
    }
    to_evaluate.push_back(&item);
  }

  if (parallel_workers > 1 && to_evaluate.size() > 1 &&
      parallel_dispatcher != nullptr) {
    std::vector<std::optional<EvaluatedBranchCandidate>> evaluated(
        to_evaluate.size());

    parallel_dispatcher->run(static_cast<int>(to_evaluate.size()), [&](int i) {
      const auto *item = to_evaluate[static_cast<std::size_t>(i)];

      EvaluatedBranchCandidate eval;
      eval.candidate = item->candidate;
      eval.decision = build_decision_from_candidate(node, *item->candidate);

      eval.decision.down_child.relaxation =
          relaxation_solver(eval.decision.down_child.state.lower_bounds,
                            eval.decision.down_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      eval.decision.up_child.relaxation =
          relaxation_solver(eval.decision.up_child.state.lower_bounds,
                            eval.decision.up_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      eval.score = branch_score(*item->candidate, relaxation.objective,
                                eval.decision.down_child.relaxation.value(),
                                eval.decision.up_child.relaxation.value(),
                                maximize, feasibility_tol, integrality_tol);

      evaluated[static_cast<std::size_t>(i)] = std::move(eval);
    });

    for (auto &item : evaluated) {
      EvaluatedBranchCandidate eval = std::move(item).value();

      update_pseudocosts(pseudocosts, *eval.candidate, relaxation.objective,
                         eval.decision.down_child.relaxation.value(),
                         eval.decision.up_child.relaxation.value(), maximize,
                         feasibility_tol, integrality_tol);

      if (eval.score > best_score + 1e-12) {
        best_score = eval.score;
        best_decision = std::move(eval.decision);
      }
    }
  } else {
    for (const auto *item : to_evaluate) {
      BranchDecision decision =
          build_decision_from_candidate(node, *item->candidate);

      decision.down_child.relaxation =
          relaxation_solver(decision.down_child.state.lower_bounds,
                            decision.down_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      decision.up_child.relaxation =
          relaxation_solver(decision.up_child.state.lower_bounds,
                            decision.up_child.state.upper_bounds,
                            node.basis ? &*node.basis : nullptr);

      update_pseudocosts(pseudocosts, *item->candidate, relaxation.objective,
                         decision.down_child.relaxation.value(),
                         decision.up_child.relaxation.value(), maximize,
                         feasibility_tol, integrality_tol);

      const double exact_score =
          branch_score(*item->candidate, relaxation.objective,
                       decision.down_child.relaxation.value(),
                       decision.up_child.relaxation.value(), maximize,
                       feasibility_tol, integrality_tol);

      if (exact_score > best_score + 1e-12) {
        best_score = exact_score;
        best_decision = std::move(decision);
      }
    }
  }

  if (best_decision.has_value()) {
    return std::move(*best_decision);
  }

  return build_decision_from_candidate(node, *ranked.front().candidate);
}

// ============================================================================
// Simplified fallback score
// ============================================================================

inline double get_pseudocost_score(const PseudoCost &pc, double cost_total,
                                   double eps = 1e-12) {
  constexpr double default_cost = 1.0;

  const double up_cost = pc.cost.up_count > 0 ? pc.up_avg() : default_cost;
  const double down_cost =
      pc.cost.down_count > 0 ? pc.down_avg() : default_cost;

  return map_score(up_cost * down_cost / safe_max(cost_total * cost_total, eps),
                   eps);
}

// ============================================================================
// Top-level branching strategy selection
// ============================================================================

template <typename RelaxationSolver>
inline BranchDecision choose_branching_variable(
    const ActiveNode &node, const RelaxationSolution &relaxation,
    const std::vector<FractionalCandidate> &candidates, const Options &options,
    bool maximize, std::vector<PseudoCost> &pseudocosts,
    ParallelDispatcher *parallel_dispatcher,
    RelaxationSolver &&relaxation_solver) {
  if (options.branching_strategy == BranchingStrategy::MostFractional) {
    return build_decision_from_candidate(node, candidates.front());
  }

  if (options.branching_strategy == BranchingStrategy::StrongBranching) {
    return choose_strong_branching(
        node, relaxation, candidates, options.strong_branching_candidates,
        options.parallel_workers, maximize, options.feasibility_tol,
        options.integrality_tol, pseudocosts, parallel_dispatcher,
        std::forward<RelaxationSolver>(relaxation_solver));
  }

  return choose_pseudocost_branching(
      node, relaxation, candidates, options.pseudocost_reliability,
      options.strong_branching_candidates, options.strong_branching_max_depth,
      options.parallel_workers, maximize, options.feasibility_tol,
      options.integrality_tol, pseudocosts, parallel_dispatcher,
      std::forward<RelaxationSolver>(relaxation_solver));
}

} // namespace simplex::bnb::detail
