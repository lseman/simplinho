#include "bnb/branching.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/parallel.h"

namespace simplex::bnb::detail {
namespace {

[[nodiscard]] double safe_max(double x, double eps = 1e-12) { return std::max(eps, x); }

[[nodiscard]] double map_score(double score, double eps = 1e-12) {
    return 1.0 - 1.0 / (1.0 + safe_max(score, eps));
}

[[nodiscard]] double objective_degradation(double parent_objective, double child_objective,
                                           bool maximize) {
    const double raw =
        maximize ? (parent_objective - child_objective) : (child_objective - parent_objective);
    return std::max(0.0, raw);
}

[[nodiscard]] double child_branch_score(const RelaxationSolution& child, double parent_objective,
                                        double distance, bool maximize, double feasibility_tol,
                                        double integrality_tol) {
    (void)feasibility_tol;
    if (child.status == RelaxationStatus::Infeasible) {
        return std::numeric_limits<double>::infinity();
    }
    if (child.status != RelaxationStatus::Optimal) {
        return 0.0;
    }

    return objective_degradation(parent_objective, child.objective, maximize) /
           safe_max(distance, integrality_tol);
}

[[nodiscard]] double combine_branch_scores(double down_score, double up_score) {
    if (std::isinf(down_score) || std::isinf(up_score)) {
        return std::numeric_limits<double>::infinity();
    }

    const double min_score = std::min(down_score, up_score);
    const double max_score = std::max(down_score, up_score);
    const double geometric_score = std::sqrt(std::max(0.0, down_score) * std::max(0.0, up_score));

    return 0.5 * min_score + 0.15 * max_score + 0.35 * geometric_score;
}

[[nodiscard]] double exact_branch_score(const FractionalCandidate& candidate,
                                        double parent_objective, const RelaxationSolution& down,
                                        const RelaxationSolution& up, bool maximize,
                                        double feasibility_tol, double integrality_tol) {
    const double down_score = child_branch_score(down, parent_objective, candidate.down_distance,
                                                 maximize, feasibility_tol, integrality_tol);
    const double up_score = child_branch_score(up, parent_objective, candidate.up_distance,
                                               maximize, feasibility_tol, integrality_tol);
    return combine_branch_scores(down_score, up_score);
}

[[nodiscard]] double compute_cost_score(double up_cost, double down_cost, double cost_avg,
                                        double eps = 1e-12) {
    return safe_max(up_cost, eps) * safe_max(down_cost, eps) / safe_max(cost_avg * cost_avg, eps);
}

[[nodiscard]] double compute_conflict_score(const PseudoCost& pseudocost, double conflict_avg,
                                            double eps = 1e-12) {
    return safe_max(pseudocost.signal.conflict_score_up, eps) *
           safe_max(pseudocost.signal.conflict_score_down, eps) /
           safe_max(conflict_avg * conflict_avg, eps);
}

[[nodiscard]] double compute_inference_score(const PseudoCost& pseudocost, double inference_avg,
                                             double eps = 1e-12) {
    return safe_max(pseudocost.signal.inference_up, eps) *
           safe_max(pseudocost.signal.inference_down, eps) /
           safe_max(inference_avg * inference_avg, eps);
}

[[nodiscard]] double compute_cutoff_score(const PseudoCost& pseudocost, double cutoff_avg,
                                          double eps = 1e-12) {
    const double up_denom = std::max(1.0, pseudocost.signal.cutoff_up + pseudocost.cost.up_count);
    const double down_denom =
        std::max(1.0, pseudocost.signal.cutoff_down + pseudocost.cost.down_count);
    const double cutoff_up_rate = pseudocost.signal.cutoff_up / up_denom;
    const double cutoff_down_rate = pseudocost.signal.cutoff_down / down_denom;
    return safe_max(cutoff_up_rate, eps) * safe_max(cutoff_down_rate, eps) /
           safe_max(cutoff_avg * cutoff_avg, eps);
}

[[nodiscard]] double get_combined_pseudocost_score(const PseudoCost& pseudocost, double up_cost,
                                                   double down_cost, double cost_avg,
                                                   double inference_avg, double conflict_avg,
                                                   double cutoff_avg, double eps = 1e-12) {
    const double cost_score = map_score(compute_cost_score(up_cost, down_cost, cost_avg, eps), eps);
    const double conflict_score =
        map_score(compute_conflict_score(pseudocost, conflict_avg, eps), eps);
    const double inference_score =
        map_score(compute_inference_score(pseudocost, inference_avg, eps), eps);
    const double cutoff_score = map_score(compute_cutoff_score(pseudocost, cutoff_avg, eps), eps);

    // HiGHS HighsPseudocost::getScore weighting: the cost term dominates, with
    // conflict ~1% and cutoff+inference ~0.01% as tie-breakers. See
    // highs-source/highs/mip/HighsPseudocost.h:285-289.
    return cost_score + 1e-2 * conflict_score + 1e-4 * (cutoff_score + inference_score);
}

[[nodiscard]] double blended_unit_pseudocost(const PseudoCost& pseudocost, bool branch_up,
                                             int reliability, double average_cost,
                                             double eps = 1e-12) {
    const int samples = branch_up ? pseudocost.cost.up_count : pseudocost.cost.down_count;
    const double observed = branch_up ? pseudocost.cost.up_value() : pseudocost.cost.down_value();
    const double fallback = safe_max(average_cost, 1.0);

    if (samples <= 0) {
        return fallback;
    }
    if (reliability > 0 && samples < reliability) {
        const double weight =
            0.9 + 0.1 * static_cast<double>(samples) / static_cast<double>(reliability);
        return weight * safe_max(observed, eps) + (1.0 - weight) * fallback;
    }
    return safe_max(observed, eps);
}

[[nodiscard]] double pseudocost_candidate_score_simple(const PseudoCost& pseudocost,
                                                       const FractionalCandidate& candidate,
                                                       int reliability, double average_cost) {
    const double down_score =
        blended_unit_pseudocost(pseudocost, false, reliability, average_cost) *
        candidate.down_distance;
    const double up_score = blended_unit_pseudocost(pseudocost, true, reliability, average_cost) *
                            candidate.up_distance;
    return combine_branch_scores(down_score, up_score);
}

[[nodiscard]] PseudoCostAverages
compute_pseudocost_averages(const std::vector<FractionalCandidate>& candidates,
                            const std::vector<PseudoCost>& pseudocosts) {
    PseudoCostAverages averages{};
    if (candidates.empty()) {
        return averages;
    }

    double cost_sum = 0.0;
    double inference_sum = 0.0;
    double conflict_sum = 0.0;
    double cutoff_sum = 0.0;
    int count = 0;
    int cost_observations = 0;
    for (const auto& candidate : candidates) {
        if (candidate.variable < 0 || candidate.variable >= static_cast<int>(pseudocosts.size())) {
            continue;
        }
        const auto& pseudocost = pseudocosts[candidate.variable];
        cost_sum += pseudocost.cost.up_sum + pseudocost.cost.down_sum;
        inference_sum += pseudocost.signal.inference_up + pseudocost.signal.inference_down;
        conflict_sum += pseudocost.signal.conflict_score_up + pseudocost.signal.conflict_score_down;
        cutoff_sum += pseudocost.signal.cutoff_up + pseudocost.signal.cutoff_down;
        cost_observations += pseudocost.cost.up_count + pseudocost.cost.down_count;
        ++count;
    }
    if (count <= 0) {
        return averages;
    }

    const double denom = static_cast<double>(count);
    averages.cost = cost_observations > 0
                        ? std::max(cost_sum / static_cast<double>(cost_observations), 1e-12)
                        : 1.0;
    averages.inference = std::max(inference_sum / denom, 1e-12);
    averages.conflict = std::max(conflict_sum / denom, 1e-12);
    averages.cutoff = std::max(cutoff_sum / denom, 1e-12);
    averages.cutoff_rate = std::max(
        cutoff_sum / std::max(1.0, cutoff_sum + static_cast<double>(cost_observations)), 1e-12);
    return averages;
}

[[nodiscard]] double pseudocost_candidate_score(const PseudoCost& pseudocost,
                                                const FractionalCandidate& candidate,
                                                const PseudoCostAverages& averages,
                                                int reliability) {
    const double down_cost =
        blended_unit_pseudocost(pseudocost, false, reliability, averages.cost) *
        candidate.down_distance;
    const double up_cost = blended_unit_pseudocost(pseudocost, true, reliability, averages.cost) *
                           candidate.up_distance;
    const double rich_score =
        get_combined_pseudocost_score(pseudocost, up_cost, down_cost, averages.cost,
                                      averages.inference, averages.conflict, averages.cutoff_rate);
    const double simple_score =
        pseudocost_candidate_score_simple(pseudocost, candidate, reliability, averages.cost);
    return rich_score + 1e-6 * simple_score;
}

void record_cutoff(std::vector<PseudoCost>& pseudocosts, int variable, bool branch_up) {
    if (variable < 0 || variable >= static_cast<int>(pseudocosts.size())) {
        return;
    }
    pseudocosts[variable].record_cutoff(branch_up);
}

void update_pseudocosts(std::vector<PseudoCost>& pseudocosts, const FractionalCandidate& candidate,
                        double parent_objective, const RelaxationSolution& down,
                        const RelaxationSolution& up, bool maximize, double feasibility_tol,
                        double integrality_tol) {
    (void)feasibility_tol;
    if (candidate.variable < 0 || candidate.variable >= static_cast<int>(pseudocosts.size())) {
        return;
    }

    auto update_one = [&](bool branch_up, const RelaxationSolution& child, double distance) {
        auto& pseudocost = pseudocosts[candidate.variable];
        if (child.status == RelaxationStatus::Infeasible) {
            const double gain = 4.0 / safe_max(distance, integrality_tol);
            if (branch_up) {
                pseudocost.cost.record_up(gain, 1.0);
            } else {
                pseudocost.cost.record_down(gain, 1.0);
            }
            pseudocost.record_inference(branch_up, 1.0);
            pseudocost.record_conflict(branch_up, 1.0);
            pseudocost.record_cutoff(branch_up);
            return;
        }
        if (child.status != RelaxationStatus::Optimal) {
            return;
        }

        const double gain = objective_degradation(parent_objective, child.objective, maximize);
        if (branch_up) {
            pseudocost.cost.record_up(gain, distance);
        } else {
            pseudocost.cost.record_down(gain, distance);
        }
        pseudocost.record_inference(branch_up, 1.0);
    };

    update_one(false, down, candidate.down_distance);
    update_one(true, up, candidate.up_distance);
}

[[nodiscard]] BranchDecision build_decision_from_candidate(const ActiveNode& node,
                                                           const FractionalCandidate& candidate) {
    BranchDecision decision;
    decision.variable = candidate.variable;
    decision.value = candidate.value;
    decision.down_child.state = make_child_state(node, candidate.variable, false, candidate.value);
    decision.up_child.state = make_child_state(node, candidate.variable, true, candidate.value);
    return decision;
}

[[nodiscard]] BranchDecision
choose_pseudocost_without_probing(const ActiveNode& node,
                                  const std::vector<FractionalCandidate>& candidates,
                                  const std::vector<PseudoCost>& pseudocosts, int reliability) {
    if (candidates.empty()) {
        return {};
    }

    const PseudoCostAverages averages = compute_pseudocost_averages(candidates, pseudocosts);
    const FractionalCandidate* best = &candidates.front();
    double best_score = -std::numeric_limits<double>::infinity();
    bool saw_reliable = false;

    for (const auto& candidate : candidates) {
        if (candidate.variable < 0 || candidate.variable >= static_cast<int>(pseudocosts.size())) {
            continue;
        }

        const auto& pseudocost = pseudocosts[candidate.variable];
        const bool reliable = pseudocost.cost.up_count > 0 || pseudocost.cost.down_count > 0;
        if (!reliable && saw_reliable) {
            continue;
        }

        const double score =
            pseudocost_candidate_score(pseudocost, candidate, averages, reliability);
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

[[nodiscard]] int automatic_strong_branching_limit(const ActiveNode& node, int parallel_workers,
                                                   std::size_t candidate_count) {
    if (candidate_count == 0) {
        return 0;
    }

    // HiGHS-style spirit: use strong branching as a bounded calibration step,
    // not as "probe every fractional variable" when the user leaves the limit
    // at zero. Later nodes are already limited elsewhere, so keep the root cap
    // modest and parallel-aware.
    if (node.depth > 0) {
        return std::min<int>(2, static_cast<int>(candidate_count));
    }

    const int worker_hint = std::max(1, parallel_workers);
    const int auto_limit = std::clamp(worker_hint * 2, 4, 12);
    return std::min<int>(auto_limit, static_cast<int>(candidate_count));
}

[[nodiscard]] int resolve_strong_branching_limit(const ActiveNode& node, int candidate_limit,
                                                 int strong_branching_k, // NEW: reduced limit
                                                 int parallel_workers,
                                                 std::size_t candidate_count) {
    int limit = candidate_limit > 0
                    ? candidate_limit
                    : automatic_strong_branching_limit(node, parallel_workers, candidate_count);
    if (strong_branching_k > 0) {
        // Treat k as a hard cap on the actual probing budget, even when the
        // broader candidate prefilter is larger.
        limit = std::min(limit, strong_branching_k);
    }
    return std::min<int>(limit, static_cast<int>(candidate_count));
}

[[nodiscard]] ChildEvaluation evaluate_child(const ActiveNode& node,
                                             const FractionalCandidate& candidate, bool branch_up,
                                             int lp_iteration_limit,
                                             const RelaxationSolveCallback& relaxation_solver) {
    ChildEvaluation evaluation;
    evaluation.state = make_child_state(node, candidate.variable, branch_up, candidate.value);
    const RelaxationSolveContext context{true, lp_iteration_limit, true};
    const ScopedRelaxationSolveContext scoped_context(context);
    evaluation.relaxation =
        relaxation_solver(evaluation.state, node.basis ? &*node.basis : nullptr);
    evaluation.relaxation_is_probe_only =
        evaluation.relaxation.has_value() && evaluation.relaxation->lp_solution.has_value() &&
        evaluation.relaxation->lp_solution->status == LPSolution::Status::IterLimit;
    evaluation.cutoff = !evaluation.relaxation.has_value() ||
                        evaluation.relaxation->status == RelaxationStatus::Infeasible ||
                        evaluation.relaxation->status == RelaxationStatus::Unbounded;
    return evaluation;
}

[[nodiscard]] BranchDecision choose_strong_branching(
    const ActiveNode& node, const RelaxationSolution& relaxation,
    const std::vector<FractionalCandidate>& candidates, int candidate_limit,
    int strong_branching_k, // NEW: reduced limit for Highs-like behavior
    int lp_iteration_limit, int parallel_workers, bool maximize, double feasibility_tol,
    double integrality_tol, std::vector<PseudoCost>& pseudocosts,
    ParallelDispatcher* parallel_dispatcher, const RelaxationSolveCallback& relaxation_solver) {
    BranchDecision best;
    double best_score = -std::numeric_limits<double>::infinity();
    int probe_count = 0;
    int probe_iterations = 0;
    std::uint64_t probe_core_solve_time_ns = 0;
    std::uint64_t probe_lp_assembly_time_ns = 0;
    std::uint64_t probe_lp_internal_presolve_ns = 0;
    std::uint64_t probe_lp_internal_crash_ns = 0;
    std::uint64_t probe_lp_internal_iters_ns = 0;
    std::uint64_t probe_lp_internal_serialize_ns = 0;

    const int limit = resolve_strong_branching_limit(node, candidate_limit, strong_branching_k,
                                                     parallel_workers, candidates.size());

    auto accumulate_probe_stats = [&](const ChildEvaluation& eval) {
        if (!eval.relaxation.has_value()) {
            return;
        }
        const RelaxationSolution& child = *eval.relaxation;
        ++probe_count;
        probe_iterations += child.iterations;
        probe_core_solve_time_ns += child.core_solve_time_ns;
        probe_lp_assembly_time_ns += child.lp_assembly_time_ns;
        probe_lp_internal_presolve_ns += child.lp_internal_presolve_ns;
        probe_lp_internal_crash_ns += child.lp_internal_crash_ns;
        probe_lp_internal_iters_ns += child.lp_internal_iters_ns;
        probe_lp_internal_serialize_ns += child.lp_internal_serialize_ns;
    };

    auto consume_eval = [&](const FractionalCandidate& candidate, ChildEvaluation&& down_child,
                            ChildEvaluation&& up_child) {
        update_pseudocosts(pseudocosts, candidate, relaxation.objective,
                           down_child.relaxation.value(), up_child.relaxation.value(), maximize,
                           feasibility_tol, integrality_tol);

        const double score = exact_branch_score(
            candidate, relaxation.objective, down_child.relaxation.value(),
            up_child.relaxation.value(), maximize, feasibility_tol, integrality_tol);
        if (score > best_score + 1e-12) {
            best_score = score;
            best.variable = candidate.variable;
            best.value = candidate.value;
            best.down_child = std::move(down_child);
            best.up_child = std::move(up_child);
            best.down_child.score = score;
            best.up_child.score = score;
        }
    };

    if (parallel_workers > 1 && limit > 1 && parallel_dispatcher != nullptr) {
        struct EvaluationSlot {
            ChildEvaluation down_child;
            ChildEvaluation up_child;
        };

        std::vector<std::optional<EvaluationSlot>> evaluated(limit);
        parallel_dispatcher->run(limit, [&](int i) {
            EvaluationSlot slot;
            slot.down_child =
                evaluate_child(node, candidates[i], false, lp_iteration_limit, relaxation_solver);
            slot.up_child =
                evaluate_child(node, candidates[i], true, lp_iteration_limit, relaxation_solver);
            evaluated[i] = std::move(slot);
        });

        for (int i = 0; i < limit; ++i) {
            if (!evaluated[i].has_value()) {
                continue;
            }
            auto slot = std::move(*evaluated[i]);
            accumulate_probe_stats(slot.down_child);
            accumulate_probe_stats(slot.up_child);
            if (!slot.down_child.relaxation.has_value() || !slot.up_child.relaxation.has_value()) {
                continue;
            }
            consume_eval(candidates[i], std::move(slot.down_child), std::move(slot.up_child));
        }
    } else {
        for (int i = 0; i < limit; ++i) {
            auto down_child =
                evaluate_child(node, candidates[i], false, lp_iteration_limit, relaxation_solver);
            auto up_child =
                evaluate_child(node, candidates[i], true, lp_iteration_limit, relaxation_solver);
            accumulate_probe_stats(down_child);
            accumulate_probe_stats(up_child);
            if (!down_child.relaxation.has_value() || !up_child.relaxation.has_value()) {
                continue;
            }
            consume_eval(candidates[i], std::move(down_child), std::move(up_child));
        }
    }

    auto attach_probe_stats = [&](BranchDecision& decision) {
        decision.strong_branching_probe_count = probe_count;
        decision.strong_branching_probe_iterations = probe_iterations;
        decision.strong_branching_probe_core_solve_time_ns = probe_core_solve_time_ns;
        decision.strong_branching_probe_lp_assembly_time_ns = probe_lp_assembly_time_ns;
        decision.strong_branching_probe_lp_internal_presolve_ns = probe_lp_internal_presolve_ns;
        decision.strong_branching_probe_lp_internal_crash_ns = probe_lp_internal_crash_ns;
        decision.strong_branching_probe_lp_internal_iters_ns = probe_lp_internal_iters_ns;
        decision.strong_branching_probe_lp_internal_serialize_ns = probe_lp_internal_serialize_ns;
    };

    if (best.variable >= 0) {
        attach_probe_stats(best);
        return best;
    }
    BranchDecision fallback = build_decision_from_candidate(node, candidates.front());
    attach_probe_stats(fallback);
    return fallback;
}

[[nodiscard]] BranchDecision choose_pseudocost_branching(
    const ActiveNode& node, const RelaxationSolution& relaxation,
    const std::vector<FractionalCandidate>& candidates, int reliability,
    int strong_branching_candidates, int strong_branching_k, // NEW
    int strong_branching_max_depth, int lp_iteration_limit, int parallel_workers, bool maximize,
    double feasibility_tol, double integrality_tol, std::vector<PseudoCost>& pseudocosts,
    ParallelDispatcher* parallel_dispatcher, const RelaxationSolveCallback& relaxation_solver) {
    if (candidates.empty()) {
        return {};
    }

    const PseudoCostAverages averages = compute_pseudocost_averages(candidates, pseudocosts);
    std::vector<RankedPseudoCostCandidate> ranked;
    ranked.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        RankedPseudoCostCandidate item;
        item.candidate = candidate;
        if (candidate.variable >= 0 && candidate.variable < static_cast<int>(pseudocosts.size())) {
            item.score = pseudocost_candidate_score(pseudocosts[candidate.variable], candidate,
                                                    averages, reliability);
        } else {
            item.score = candidate.fractionality;
        }
        ranked.push_back(item);
    }

    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (std::abs(lhs.score - rhs.score) > 1e-12) {
            return lhs.score > rhs.score;
        }
        return lhs.candidate.variable < rhs.candidate.variable;
    });

    if (strong_branching_max_depth >= 0 && node.depth > strong_branching_max_depth) {
        return build_decision_from_candidate(node, ranked.front().candidate);
    }
    if (strong_branching_max_depth < -1) {
        return choose_pseudocost_without_probing(node, candidates, pseudocosts, reliability);
    }

    for (const auto& item : ranked) {
        if (item.candidate.variable >= 0 &&
            item.candidate.variable < static_cast<int>(pseudocosts.size()) &&
            pseudocosts[item.candidate.variable].cost.is_reliable(reliability)) {
            return build_decision_from_candidate(node, item.candidate);
        }
    }

    std::vector<FractionalCandidate> evaluate_candidates;
    evaluate_candidates.reserve(ranked.size());
    const int evaluate_limit =
        resolve_strong_branching_limit(node, strong_branching_candidates,
                                       strong_branching_k, // Pass k parameter
                                       parallel_workers, ranked.size());
    for (const auto& item : ranked) {
        if (static_cast<int>(evaluate_candidates.size()) >= evaluate_limit) {
            break;
        }
        evaluate_candidates.push_back(item.candidate);
    }

    if (evaluate_candidates.empty()) {
        return choose_pseudocost_without_probing(node, candidates, pseudocosts, reliability);
    }

    return choose_strong_branching(
        node, relaxation, evaluate_candidates, static_cast<int>(evaluate_candidates.size()),
        strong_branching_k, lp_iteration_limit, parallel_workers, maximize, feasibility_tol,
        integrality_tol, pseudocosts, parallel_dispatcher, relaxation_solver);
}

} // namespace

BranchDecision choose_sos_branching_constraint(const ActiveNode& node,
                                               const Eigen::VectorXd& primal,
                                               const std::vector<SOSConstraint>& sos_constraints,
                                               double feasibility_tol) {
    const double tol = std::max(feasibility_tol, 1e-9);
    for (const SOSConstraint& sos : sos_constraints) {
        std::vector<int> active_positions;
        active_positions.reserve(sos.variables.size());
        for (int pos = 0; pos < static_cast<int>(sos.variables.size()); ++pos) {
            const int variable = sos.variables[pos];
            if (variable >= 0 && variable < primal.size() && std::abs(primal(variable)) > tol) {
                active_positions.push_back(pos);
            }
        }

        bool violated = false;
        if (sos.type == SOSType::SOS1) {
            violated = active_positions.size() > 1;
        } else if (active_positions.size() > 2) {
            violated = true;
        } else if (active_positions.size() == 2) {
            violated = active_positions[1] != active_positions[0] + 1;
        }
        if (!violated) {
            continue;
        }

        const int first = active_positions.front();
        const int last = active_positions.back();
        const int split = sos.type == SOSType::SOS1 ? (first + last) / 2
                                                    : std::max(first + 1, (first + last) / 2);

        std::vector<int> left_zero;
        std::vector<int> right_zero;
        left_zero.reserve(sos.variables.size());
        right_zero.reserve(sos.variables.size());
        for (int pos = 0; pos < static_cast<int>(sos.variables.size()); ++pos) {
            if (sos.type == SOSType::SOS1) {
                if (pos <= split) {
                    left_zero.push_back(sos.variables[pos]);
                }
                if (pos > split) {
                    right_zero.push_back(sos.variables[pos]);
                }
            } else {
                if (pos < split) {
                    left_zero.push_back(sos.variables[pos]);
                }
                if (pos > split) {
                    right_zero.push_back(sos.variables[pos]);
                }
            }
        }

        BranchDecision decision;
        decision.variable = sos.variables[split];
        decision.value = primal(decision.variable);
        decision.down_child.state = make_upper_zero_child_state(node, right_zero);
        decision.up_child.state = make_upper_zero_child_state(node, left_zero);
        return decision;
    }
    return {};
}

void PseudoCostStats::record_up(double gain, double distance) {
    up_sum += std::max(0.0, gain) / std::max(distance, 1e-12);
    ++up_count;
}

void PseudoCostStats::record_down(double gain, double distance) {
    down_sum += std::max(0.0, gain) / std::max(distance, 1e-12);
    ++down_count;
}

[[nodiscard]] double PseudoCostStats::up_value() const {
    return up_count > 0 ? up_sum / static_cast<double>(up_count) : 0.0;
}

[[nodiscard]] double PseudoCostStats::down_value() const {
    return down_count > 0 ? down_sum / static_cast<double>(down_count) : 0.0;
}

[[nodiscard]] bool PseudoCostStats::is_reliable(int reliability) const {
    return up_count >= reliability && down_count >= reliability;
}

void BranchSignalStats::record_inference(bool branch_up, double amount) {
    if (branch_up) {
        inference_up += amount;
    } else {
        inference_down += amount;
    }
}

void BranchSignalStats::record_conflict(bool branch_up, double amount) {
    if (branch_up) {
        conflict_score_up += amount;
    } else {
        conflict_score_down += amount;
    }
}

void BranchSignalStats::record_cutoff(bool branch_up) {
    if (branch_up) {
        cutoff_up += 1.0;
    } else {
        cutoff_down += 1.0;
    }
}

void BranchSignalStats::record_cutoff() {
    cutoff_up += 1.0;
    cutoff_down += 1.0;
}

[[nodiscard]] double BranchSignalStats::up_score() const {
    return inference_up + conflict_score_up + 2.0 * cutoff_up;
}

[[nodiscard]] double BranchSignalStats::down_score() const {
    return inference_down + conflict_score_down + 2.0 * cutoff_down;
}

void PseudoCost::record_observation(bool branch_up, double parent_objective, double child_objective,
                                    double parent_value, double child_value, bool maximize) {
    const double gain =
        maximize ? (parent_objective - child_objective) : (child_objective - parent_objective);
    const double distance = std::abs(child_value - parent_value);
    if (branch_up) {
        cost.record_up(gain, distance);
    } else {
        cost.record_down(gain, distance);
    }
}

void PseudoCost::record_inference(bool branch_up, double amount) {
    signal.record_inference(branch_up, amount);
}

void PseudoCost::record_conflict(bool branch_up, double amount) {
    signal.record_conflict(branch_up, amount);
}

void PseudoCost::record_cutoff(bool branch_up) { signal.record_cutoff(branch_up); }

void PseudoCost::record_cutoff() { signal.record_cutoff(); }

[[nodiscard]] double node_estimate(const RelaxationSolution& relaxation,
                                   const std::vector<VariableType>& variable_types,
                                   const std::vector<PseudoCost>& pseudocosts,
                                   double integrality_tol, bool maximize) {
    if (relaxation.status != RelaxationStatus::Optimal) {
        return relaxation.objective;
    }

    const auto fractional =
        collect_fractional_candidates(relaxation.primal, variable_types, integrality_tol);
    if (fractional.empty()) {
        return relaxation.objective;
    }

    double degradation = 0.0;
    constexpr double default_cost = 1.0;
    for (const auto& candidate : fractional) {
        if (candidate.variable < 0 || candidate.variable >= static_cast<int>(pseudocosts.size())) {
            degradation += default_cost * std::min(candidate.down_distance, candidate.up_distance);
            continue;
        }
        const auto& pseudocost = pseudocosts[candidate.variable];
        const double down_score =
            (pseudocost.cost.down_count > 0 ? pseudocost.cost.down_value() : default_cost) *
            candidate.down_distance;
        const double up_score =
            (pseudocost.cost.up_count > 0 ? pseudocost.cost.up_value() : default_cost) *
            candidate.up_distance;
        degradation += std::min(down_score, up_score);
    }

    return maximize ? (relaxation.objective - degradation) : (relaxation.objective + degradation);
}

BranchDecision choose_branching_variable(const ActiveNode& node,
                                         const RelaxationSolution& relaxation,
                                         const std::vector<FractionalCandidate>& fractional,
                                         const Options& options, bool maximize,
                                         std::vector<PseudoCost>& pseudocosts,
                                         ParallelDispatcher* parallel_dispatcher,
                                         const RelaxationSolveCallback& relaxation_solver) {
    if (fractional.empty()) {
        return {};
    }
    const int effective_parallel_workers =
        parallel_dispatcher != nullptr ? parallel_dispatcher->worker_count() : 1;

    if (options.branching_strategy == BranchingStrategy::MostFractional) {
        return build_decision_from_candidate(node, fractional.front());
    }
    if (options.strong_branching_max_depth < -1) {
        return choose_pseudocost_without_probing(node, fractional, pseudocosts,
                                                 options.pseudocost_reliability);
    }

    if (options.branching_strategy == BranchingStrategy::StrongBranching) {
        // HiGHS-like behavior: use strong branching mainly to seed pseudocosts
        // at the root, then branch by pseudocost and probe only shallow,
        // unreliable candidates afterwards.
        if (node.depth == 0) {
            return choose_strong_branching(
                node, relaxation, fractional, options.strong_branching_candidates,
                options.strong_branching_k, // Pass k for reduced strong branching
                options.strong_branching_lp_iter_limit, effective_parallel_workers, maximize,
                options.feasibility_tol, options.integrality_tol, pseudocosts, parallel_dispatcher,
                relaxation_solver);
        }

        const int limited_candidates = options.strong_branching_candidates > 0
                                           ? std::min(options.strong_branching_candidates, 2)
                                           : 2;
        return choose_pseudocost_branching(
            node, relaxation, fractional, options.pseudocost_reliability, limited_candidates,
            options.strong_branching_k, std::min(options.strong_branching_max_depth, 1),
            options.strong_branching_lp_iter_limit, effective_parallel_workers, maximize,
            options.feasibility_tol, options.integrality_tol, pseudocosts, parallel_dispatcher,
            relaxation_solver);
    }

    return choose_pseudocost_branching(
        node, relaxation, fractional, options.pseudocost_reliability,
        options.strong_branching_candidates,
        options.strong_branching_k, // Pass k parameter
        options.strong_branching_max_depth, options.strong_branching_lp_iter_limit,
        effective_parallel_workers, maximize, options.feasibility_tol, options.integrality_tol,
        pseudocosts, parallel_dispatcher, relaxation_solver);
}

} // namespace simplex::bnb::detail
