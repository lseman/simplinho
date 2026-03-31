#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/search.h"

namespace simplex::bnb::detail {

inline bool is_effectively_integral(double value, double tol) {
    return std::isfinite(value) && std::abs(value - std::round(value)) <= tol;
}

inline Eigen::VectorXd project_to_integral_lattice(
    const Eigen::VectorXd& primal, const std::vector<VariableType>& variable_types) {
    Eigen::VectorXd rounded = primal;
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous) continue;
        rounded(j) = std::round(primal(j));
    }
    return rounded;
}

struct FractionalCandidate {
    int variable = -1;
    double value = std::numeric_limits<double>::quiet_NaN();
    double fractionality = 0.0;
    double down_distance = 0.0;
    double up_distance = 0.0;
};

struct ChildState {
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    std::shared_ptr<NodeReasonStore> reasons;
};

inline std::vector<FractionalCandidate> collect_fractional_candidates(
    const Eigen::VectorXd& primal, const std::vector<VariableType>& variable_types,
    double integrality_tol) {
    std::vector<FractionalCandidate> out;
    out.reserve(variable_types.size());
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous) continue;
        const double value = primal(j);
        if (!std::isfinite(value)) continue;
        const double floor_value = std::floor(value);
        const double ceil_value = std::ceil(value);
        const double down_distance = value - floor_value;
        const double up_distance = ceil_value - value;
        const double fractionality = std::min(down_distance, up_distance);
        if (fractionality > integrality_tol) {
            out.push_back(FractionalCandidate{j, value, fractionality, down_distance,
                                              up_distance});
        }
    }

    std::sort(out.begin(), out.end(), [](const auto& lhs, const auto& rhs) {
        if (std::abs(lhs.fractionality - rhs.fractionality) > 1e-12) {
            return lhs.fractionality > rhs.fractionality;
        }
        return lhs.variable < rhs.variable;
    });
    return out;
}

inline ChildState make_child_state(const ActiveNode& node, int variable, bool branch_up,
                                   double value) {
    ChildState child{node.lower_bounds, node.upper_bounds, node.reasons};
    if (branch_up) {
        child.lower_bounds(variable) = std::max(child.lower_bounds(variable), std::ceil(value));
    } else {
        child.upper_bounds(variable) = std::min(child.upper_bounds(variable), std::floor(value));
    }
    return child;
}

inline ChildState make_fixed_child_state(const ActiveNode& node, int variable, bool branch_up,
                                         double value) {
    ChildState child{node.lower_bounds, node.upper_bounds, node.reasons};
    const double fixed_value = branch_up ? std::ceil(value) : std::floor(value);
    child.lower_bounds(variable) = fixed_value;
    child.upper_bounds(variable) = fixed_value;
    return child;
}

struct DivingHeuristicResult {
    std::optional<RelaxationSolution> incumbent;
    int lp_iterations = 0;
    int lp_solves = 0;
    int successes = 0;
};

struct DivingChoice {
    ChildState state;
    std::optional<RelaxationSolution> relaxation;
};

struct DivingStrategyStats {
    int attempts = 0;
    int successes = 0;
    int lp_iterations = 0;
    int lp_solves = 0;
};

inline bool objective_improves_for_problem(double candidate, double incumbent,
                                           bool maximize, double tol) {
    return maximize ? (candidate > incumbent + tol) : (candidate < incumbent - tol);
}

inline bool is_integer_feasible_solution(const Eigen::VectorXd& primal,
                                         const std::vector<VariableType>& variable_types,
                                         double tol) {
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous) continue;
        if (!is_effectively_integral(primal(j), tol)) {
            return false;
        }
    }
    return true;
}

inline int diving_strategy_index(DivingStrategy strategy) {
    switch (strategy) {
        case DivingStrategy::Fractional:
            return 0;
        case DivingStrategy::VectorLength:
            return 1;
        case DivingStrategy::ObjectiveValue:
            return 2;
        case DivingStrategy::Coefficient:
            return 3;
        case DivingStrategy::Guided:
            return 4;
        case DivingStrategy::Disabled:
        case DivingStrategy::Adaptive:
            break;
    }
    return -1;
}

template <typename RelaxationSolver>
inline std::optional<RelaxationSolution> solve_child_relaxation(
    const ChildState& state, const LPBasis* basis, int& lp_iterations, int& lp_solves,
    RelaxationSolver&& relaxation_solver) {
    const RelaxationSolution child =
        relaxation_solver(state.lower_bounds, state.upper_bounds, basis);
    lp_iterations += child.iterations;
    ++lp_solves;
    if (child.status != RelaxationStatus::Optimal) {
        return child;
    }
    return child;
}

template <typename RelaxationSolver>
inline DivingChoice select_objective_diving_choice(
    const ActiveNode& node, const FractionalCandidate& candidate, const Problem& problem,
    int& lp_iterations, int& lp_solves, RelaxationSolver&& relaxation_solver) {
    DivingChoice floor_choice{
        make_child_state(node, candidate.variable, false, candidate.value), std::nullopt};
    DivingChoice ceil_choice{
        make_child_state(node, candidate.variable, true, candidate.value), std::nullopt};

    floor_choice.relaxation = solve_child_relaxation(floor_choice.state,
                                                     node.basis ? &*node.basis : nullptr,
                                                     lp_iterations, lp_solves,
                                                     relaxation_solver);
    ceil_choice.relaxation = solve_child_relaxation(ceil_choice.state,
                                                    node.basis ? &*node.basis : nullptr,
                                                    lp_iterations, lp_solves,
                                                    relaxation_solver);

    const bool floor_opt = floor_choice.relaxation->status == RelaxationStatus::Optimal;
    const bool ceil_opt = ceil_choice.relaxation->status == RelaxationStatus::Optimal;
    if (floor_opt && ceil_opt) {
        return objective_improves_for_problem(floor_choice.relaxation->objective,
                                              ceil_choice.relaxation->objective,
                                              problem.maximize, 1e-12)
                   ? floor_choice
                   : ceil_choice;
    }
    if (floor_opt) return floor_choice;
    if (ceil_opt) return ceil_choice;
    return candidate.down_distance <= candidate.up_distance ? floor_choice : ceil_choice;
}

template <typename RelaxationSolver>
inline DivingChoice select_coefficient_diving_choice(
    const ActiveNode& node, const FractionalCandidate& candidate, const Problem& problem,
    int& lp_iterations, int& lp_solves, RelaxationSolver&& relaxation_solver) {
    DivingChoice floor_choice{
        make_fixed_child_state(node, candidate.variable, false, candidate.value), std::nullopt};
    DivingChoice ceil_choice{
        make_fixed_child_state(node, candidate.variable, true, candidate.value), std::nullopt};

    floor_choice.relaxation = solve_child_relaxation(floor_choice.state,
                                                     node.basis ? &*node.basis : nullptr,
                                                     lp_iterations, lp_solves,
                                                     relaxation_solver);
    ceil_choice.relaxation = solve_child_relaxation(ceil_choice.state,
                                                    node.basis ? &*node.basis : nullptr,
                                                    lp_iterations, lp_solves,
                                                    relaxation_solver);

    const bool floor_opt = floor_choice.relaxation->status == RelaxationStatus::Optimal;
    const bool ceil_opt = ceil_choice.relaxation->status == RelaxationStatus::Optimal;
    if (floor_opt && ceil_opt) {
        return objective_improves_for_problem(floor_choice.relaxation->objective,
                                              ceil_choice.relaxation->objective,
                                              problem.maximize, 1e-12)
                   ? floor_choice
                   : ceil_choice;
    }
    if (floor_opt) return floor_choice;
    if (ceil_opt) return ceil_choice;

    if (candidate.variable < problem.objective_coefficients.size()) {
        const double coeff = problem.objective_coefficients(candidate.variable);
        const bool prefer_ceil = problem.maximize ? (coeff >= 0.0) : (coeff < 0.0);
        return prefer_ceil ? ceil_choice : floor_choice;
    }
    return candidate.down_distance <= candidate.up_distance ? floor_choice : ceil_choice;
}

template <typename RelaxationSolver>
inline DivingChoice select_fractional_diving_choice(
    const ActiveNode& node, const FractionalCandidate& candidate,
    int& lp_iterations, int& lp_solves, RelaxationSolver&& relaxation_solver) {
    const bool prefer_ceil = candidate.down_distance > candidate.up_distance;
    DivingChoice preferred{
        make_child_state(node, candidate.variable, prefer_ceil, candidate.value), std::nullopt};
    preferred.relaxation = solve_child_relaxation(preferred.state,
                                                  node.basis ? &*node.basis : nullptr,
                                                  lp_iterations, lp_solves,
                                                  relaxation_solver);
    if (preferred.relaxation->status == RelaxationStatus::Optimal) {
        return preferred;
    }

    DivingChoice alternate{make_child_state(node, candidate.variable, !prefer_ceil,
                                            candidate.value),
                           std::nullopt};
    alternate.relaxation = solve_child_relaxation(alternate.state,
                                                  node.basis ? &*node.basis : nullptr,
                                                  lp_iterations, lp_solves,
                                                  relaxation_solver);
    return alternate;
}

template <typename RelaxationSolver>
inline DivingChoice select_vector_length_diving_choice(
    const ActiveNode& node, const FractionalCandidate& candidate,
    int& lp_iterations, int& lp_solves, RelaxationSolver&& relaxation_solver) {
    const double floor_score = candidate.down_distance * candidate.down_distance;
    const double ceil_score = candidate.up_distance * candidate.up_distance;
    const bool prefer_ceil = ceil_score < floor_score;
    DivingChoice preferred{
        make_child_state(node, candidate.variable, prefer_ceil, candidate.value), std::nullopt};
    preferred.relaxation = solve_child_relaxation(preferred.state,
                                                  node.basis ? &*node.basis : nullptr,
                                                  lp_iterations, lp_solves,
                                                  relaxation_solver);
    if (preferred.relaxation->status == RelaxationStatus::Optimal) {
        return preferred;
    }

    DivingChoice alternate{make_child_state(node, candidate.variable, !prefer_ceil,
                                            candidate.value),
                           std::nullopt};
    alternate.relaxation = solve_child_relaxation(alternate.state,
                                                  node.basis ? &*node.basis : nullptr,
                                                  lp_iterations, lp_solves,
                                                  relaxation_solver);
    return alternate;
}

inline const FractionalCandidate* select_guided_diving_candidate(
    const std::vector<FractionalCandidate>& candidates,
    const Eigen::VectorXd* incumbent_primal) {
    if (incumbent_primal == nullptr) {
        return candidates.empty() ? nullptr : &candidates.front();
    }

    const FractionalCandidate* best = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const auto& candidate : candidates) {
        if (candidate.variable < 0 || candidate.variable >= incumbent_primal->size()) {
            continue;
        }
        const double incumbent_value = std::round((*incumbent_primal)(candidate.variable));
        const double distance = std::abs(candidate.value - incumbent_value);
        if (distance + 1e-12 < best_distance) {
            best_distance = distance;
            best = &candidate;
        }
    }
    return best != nullptr ? best : (candidates.empty() ? nullptr : &candidates.front());
}

template <typename RelaxationSolver>
inline DivingChoice select_guided_diving_choice(
    const ActiveNode& node, const FractionalCandidate& candidate,
    const Eigen::VectorXd* incumbent_primal, int& lp_iterations, int& lp_solves,
    RelaxationSolver&& relaxation_solver) {
    if (incumbent_primal != nullptr && candidate.variable >= 0 &&
        candidate.variable < incumbent_primal->size()) {
        const double target = std::round((*incumbent_primal)(candidate.variable));
        const bool branch_up = target > candidate.value;
        DivingChoice preferred{
            make_child_state(node, candidate.variable, branch_up, candidate.value), std::nullopt};
        preferred.relaxation = solve_child_relaxation(preferred.state,
                                                      node.basis ? &*node.basis : nullptr,
                                                      lp_iterations, lp_solves,
                                                      relaxation_solver);
        if (preferred.relaxation->status == RelaxationStatus::Optimal) {
            return preferred;
        }

        DivingChoice alternate{make_child_state(node, candidate.variable, !branch_up,
                                                candidate.value),
                               std::nullopt};
        alternate.relaxation = solve_child_relaxation(alternate.state,
                                                      node.basis ? &*node.basis : nullptr,
                                                      lp_iterations, lp_solves,
                                                      relaxation_solver);
        return alternate;
    }

    return select_fractional_diving_choice(node, candidate, lp_iterations, lp_solves,
                                           relaxation_solver);
}

inline DivingStrategy choose_adaptive_diving_strategy(
    const std::vector<FractionalCandidate>& candidates, const Problem& problem,
    const Eigen::VectorXd* incumbent_primal, const std::vector<DivingStrategyStats>& stats) {
    const std::vector<DivingStrategy> strategies = {
        DivingStrategy::Fractional,    DivingStrategy::VectorLength,
        DivingStrategy::ObjectiveValue, DivingStrategy::Coefficient,
        DivingStrategy::Guided};

    const double top_fractionality =
        candidates.empty() ? 0.0 : candidates.front().fractionality;
    const bool has_incumbent = incumbent_primal != nullptr;
    const double objective_magnitude =
        (!candidates.empty() && candidates.front().variable >= 0 &&
         candidates.front().variable < problem.objective_coefficients.size())
            ? std::abs(problem.objective_coefficients(candidates.front().variable))
            : 0.0;

    int total_attempts = 0;
    for (const auto& stat : stats) total_attempts += stat.attempts;

    DivingStrategy best_strategy = has_incumbent ? DivingStrategy::Guided
                                                 : DivingStrategy::Fractional;
    double best_score = -std::numeric_limits<double>::infinity();

    for (const DivingStrategy strategy : strategies) {
        if (strategy == DivingStrategy::Guided && !has_incumbent) {
            continue;
        }
        const int index = diving_strategy_index(strategy);
        if (index < 0 || index >= static_cast<int>(stats.size())) {
            continue;
        }

        const auto& stat = stats[index];
        const double success_rate =
            (stat.successes + 1.0) / (stat.attempts + 2.0);
        const double exploration =
            std::sqrt(std::log(static_cast<double>(total_attempts + 2)) /
                      static_cast<double>(stat.attempts + 1));
        const double avg_iterations =
            stat.lp_solves > 0 ? static_cast<double>(stat.lp_iterations) / stat.lp_solves : 0.0;

        double context_bonus = 0.0;
        switch (strategy) {
            case DivingStrategy::Fractional:
                context_bonus = top_fractionality >= 0.35 ? 0.35 : 0.15;
                break;
            case DivingStrategy::VectorLength:
                context_bonus = (top_fractionality > 0.15 && top_fractionality < 0.4) ? 0.25
                                                                                      : 0.1;
                break;
            case DivingStrategy::ObjectiveValue:
                context_bonus = candidates.size() <= 6 ? 0.3 : 0.05;
                break;
            case DivingStrategy::Coefficient:
                context_bonus = objective_magnitude > 1e-9 ? 0.25 : 0.05;
                break;
            case DivingStrategy::Guided:
                context_bonus = has_incumbent ? 0.45 : -1e9;
                break;
            case DivingStrategy::Disabled:
            case DivingStrategy::Adaptive:
                break;
        }

        const double score =
            context_bonus + 0.85 * success_rate + 0.35 * exploration - 0.01 * avg_iterations;
        if (score > best_score + 1e-12) {
            best_score = score;
            best_strategy = strategy;
        }
    }

    return best_strategy;
}

template <typename RelaxationSolver>
inline DivingHeuristicResult run_diving_heuristic(
    const ActiveNode& start_node, const RelaxationSolution& start_relaxation,
    const Problem& problem, const Options& options, const Eigen::VectorXd* incumbent_primal,
    std::vector<DivingStrategyStats>& strategy_stats, RelaxationSolver&& relaxation_solver) {
    DivingHeuristicResult result;
    if (options.diving_strategy == DivingStrategy::Disabled || options.max_dive_depth <= 0 ||
        options.max_dive_lp_solves <= 0) {
        return result;
    }

    ActiveNode node = start_node;
    node.basis = start_relaxation.basis;
    RelaxationSolution current = start_relaxation;
    std::optional<DivingStrategy> last_used_strategy;

    for (int depth = 0; depth < options.max_dive_depth; ++depth) {
        const auto candidates = collect_fractional_candidates(current.primal, problem.variable_types,
                                                              options.integrality_tol);
        if (candidates.empty()) {
            result.incumbent = current;
            ++result.successes;
            return result;
        }
        if (result.lp_solves >= options.max_dive_lp_solves) {
            break;
        }

        DivingStrategy active_strategy = options.diving_strategy;
        if (active_strategy == DivingStrategy::Adaptive) {
            active_strategy = choose_adaptive_diving_strategy(candidates, problem,
                                                              incumbent_primal, strategy_stats);
        }

        const FractionalCandidate* candidate = &candidates.front();
        if (active_strategy == DivingStrategy::Guided) {
            candidate = select_guided_diving_candidate(candidates, incumbent_primal);
        }
        if (candidate == nullptr) {
            break;
        }

        const int strategy_index = diving_strategy_index(active_strategy);
        if (strategy_index >= 0 && strategy_index < static_cast<int>(strategy_stats.size())) {
            ++strategy_stats[strategy_index].attempts;
        }
        last_used_strategy = active_strategy;

        const int lp_iterations_before = result.lp_iterations;
        const int lp_solves_before = result.lp_solves;
        DivingChoice choice;
        switch (active_strategy) {
            case DivingStrategy::Disabled:
                return result;
            case DivingStrategy::Fractional:
                choice = select_fractional_diving_choice(node, *candidate, result.lp_iterations,
                                                         result.lp_solves, relaxation_solver);
                break;
            case DivingStrategy::VectorLength:
                choice = select_vector_length_diving_choice(node, *candidate, result.lp_iterations,
                                                            result.lp_solves, relaxation_solver);
                break;
            case DivingStrategy::ObjectiveValue:
                choice = select_objective_diving_choice(node, *candidate, problem,
                                                        result.lp_iterations, result.lp_solves,
                                                        relaxation_solver);
                break;
            case DivingStrategy::Coefficient:
                choice = select_coefficient_diving_choice(node, *candidate, problem,
                                                          result.lp_iterations, result.lp_solves,
                                                          relaxation_solver);
                break;
            case DivingStrategy::Guided:
                choice = select_guided_diving_choice(node, *candidate, incumbent_primal,
                                                     result.lp_iterations, result.lp_solves,
                                                     relaxation_solver);
                break;
            case DivingStrategy::Adaptive:
                return result;
        }

        if (strategy_index >= 0 && strategy_index < static_cast<int>(strategy_stats.size())) {
            strategy_stats[strategy_index].lp_iterations +=
                result.lp_iterations - lp_iterations_before;
            strategy_stats[strategy_index].lp_solves += result.lp_solves - lp_solves_before;
        }

        if (!choice.relaxation.has_value() ||
            choice.relaxation->status != RelaxationStatus::Optimal) {
            break;
        }

        current = *choice.relaxation;
        node.lower_bounds = choice.state.lower_bounds;
        node.upper_bounds = choice.state.upper_bounds;
        node.basis = current.basis;
    }

    const auto final_candidates =
        collect_fractional_candidates(current.primal, problem.variable_types, options.integrality_tol);
    if (final_candidates.empty() && current.status == RelaxationStatus::Optimal) {
        result.incumbent = current;
        ++result.successes;
        const int strategy_index =
            diving_strategy_index(last_used_strategy.value_or(options.diving_strategy));
        if (strategy_index >= 0 && strategy_index < static_cast<int>(strategy_stats.size())) {
            ++strategy_stats[strategy_index].successes;
        }
    }
    return result;
}

}  // namespace simplex::bnb::detail
