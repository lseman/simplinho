#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/diving.h"

namespace simplex::bnb::detail {

struct NeighborhoodHeuristicResult {
    std::optional<RelaxationSolution> incumbent;
    int lp_iterations = 0;
    int successes = 0;
};

inline double compute_problem_objective(const Problem& problem, const Eigen::VectorXd& primal) {
    const int n = std::min<int>(problem.objective_coefficients.size(), primal.size());
    return (n > 0 ? problem.objective_coefficients.head(n).dot(primal.head(n)) : 0.0) +
           problem.objective_constant;
}

inline bool satisfies_linear_constraint(const SparseLinearConstraint& row,
                                        const Eigen::VectorXd& primal, double tol) {
    double lhs = 0.0;
    for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                    k < static_cast<int>(row.values.size());
         ++k) {
        const int index = row.indices[k];
        if (index >= 0 && index < primal.size()) {
            lhs += row.values[k] * primal(index);
        }
    }
    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            return lhs <= row.rhs + tol;
        case LinearConstraintSense::GreaterEqual:
            return lhs >= row.rhs - tol;
        case LinearConstraintSense::Equal:
            return std::abs(lhs - row.rhs) <= tol;
    }
    return false;
}

inline bool satisfies_cut_constraint(const Cut& cut, const Eigen::VectorXd& primal, double tol) {
    double lhs = 0.0;
    for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                    k < static_cast<int>(cut.values.size());
         ++k) {
        const int index = cut.indices[k];
        if (index >= 0 && index < primal.size()) {
            lhs += cut.values[k] * primal(index);
        }
    }
    switch (cut.sense) {
        case LinearConstraintSense::LessEqual:
            return lhs <= cut.rhs + tol;
        case LinearConstraintSense::GreaterEqual:
            return lhs >= cut.rhs - tol;
        case LinearConstraintSense::Equal:
            return std::abs(lhs - cut.rhs) <= tol;
    }
    return false;
}

inline std::optional<RelaxationSolution> run_rounding_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const std::vector<Cut>& active_cuts) {
    if (lp_relaxation.primal.size() == 0) {
        return std::nullopt;
    }

    auto build_candidate = [&](bool objective_guided) {
        Eigen::VectorXd candidate = lp_relaxation.primal;
        for (int j = 0; j < candidate.size() &&
                        j < static_cast<int>(problem.variable_types.size());
             ++j) {
            if (problem.variable_types[j] == VariableType::Continuous) {
                candidate(j) = std::min(problem.upper_bounds(j),
                                        std::max(problem.lower_bounds(j), candidate(j)));
                continue;
            }

            double rounded = std::round(candidate(j));
            if (objective_guided && j < problem.objective_coefficients.size()) {
                const double coeff = problem.objective_coefficients(j);
                if (problem.maximize) {
                    rounded = coeff >= 0.0 ? std::ceil(candidate(j)) : std::floor(candidate(j));
                } else {
                    rounded = coeff >= 0.0 ? std::floor(candidate(j)) : std::ceil(candidate(j));
                }
            }
            rounded = std::min(problem.upper_bounds(j), std::max(problem.lower_bounds(j), rounded));
            candidate(j) = std::round(rounded);
        }
        return candidate;
    };

    for (const bool objective_guided : {false, true}) {
        const Eigen::VectorXd candidate = build_candidate(objective_guided);
        if (!is_integer_feasible_solution(candidate, problem.variable_types,
                                          options.integrality_tol)) {
            continue;
        }

        bool feasible = true;
        for (const auto& row : problem.base_constraints) {
            if (!satisfies_linear_constraint(row, candidate, options.integrality_tol)) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;
        for (const auto& cut : active_cuts) {
            if (!satisfies_cut_constraint(cut, candidate, options.integrality_tol)) {
                feasible = false;
                break;
            }
        }
        if (!feasible) continue;

        RelaxationSolution incumbent;
        incumbent.status = RelaxationStatus::Optimal;
        incumbent.primal = candidate;
        incumbent.objective = compute_problem_objective(problem, candidate);
        incumbent.iterations = 0;
        return incumbent;
    }

    return std::nullopt;
}

template <typename SubMIPSolver>
inline std::optional<RelaxationSolution> try_integer_subproblem(
    const Problem& problem, const Options& options, const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper, int& lp_iterations, int& successes,
    SubMIPSolver&& solve_submip) {
    const SolveResult subproblem = solve_submip(lower, upper);
    lp_iterations += subproblem.lp_iterations;
    if (!subproblem.has_solution ||
        !is_integer_feasible_solution(subproblem.primal, problem.variable_types,
                                      options.integrality_tol)) {
        return std::nullopt;
    }

    ++successes;
    RelaxationSolution incumbent;
    incumbent.status = RelaxationStatus::Optimal;
    incumbent.primal = subproblem.primal;
    incumbent.objective = subproblem.objective;
    incumbent.iterations = subproblem.lp_iterations;
    return incumbent;
}

inline double linear_constraint_violation_amount(const SparseLinearConstraint& row,
                                                 const Eigen::VectorXd& primal) {
    double lhs = 0.0;
    for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                    k < static_cast<int>(row.values.size());
         ++k) {
        const int index = row.indices[k];
        if (index >= 0 && index < primal.size()) {
            lhs += row.values[k] * primal(index);
        }
    }
    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            return std::max(0.0, lhs - row.rhs);
        case LinearConstraintSense::GreaterEqual:
            return std::max(0.0, row.rhs - lhs);
        case LinearConstraintSense::Equal:
            return std::abs(lhs - row.rhs);
    }
    return 0.0;
}

inline Eigen::VectorXd project_candidate_to_bounds(
    const Eigen::VectorXd& primal, const Problem& problem) {
    Eigen::VectorXd candidate = primal;
    const int n = std::min<int>(candidate.size(), problem.variable_types.size());
    for (int j = 0; j < n; ++j) {
        candidate(j) = std::min(problem.upper_bounds(j),
                                std::max(problem.lower_bounds(j), candidate(j)));
        if (problem.variable_types[j] != VariableType::Continuous) {
            candidate(j) = std::round(candidate(j));
        }
    }
    return candidate;
}

inline std::vector<double> violated_row_weights(const Problem& problem,
                                                const Eigen::VectorXd& candidate,
                                                const std::vector<double>& row_weights,
                                                double tol) {
    std::vector<double> weights(problem.base_constraints.size(), 0.0);
    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const double violation =
            linear_constraint_violation_amount(problem.base_constraints[row_index], candidate);
        if (violation > tol) {
            weights[row_index] = row_weights[row_index];
        }
    }
    return weights;
}

inline double weighted_violation_score(const Problem& problem, const Eigen::VectorXd& candidate,
                                       const std::vector<double>& row_weights) {
    double score = 0.0;
    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const double violation =
            linear_constraint_violation_amount(problem.base_constraints[row_index], candidate);
        if (violation <= 0.0) continue;
        score += row_weights[row_index] * violation;
    }
    return score;
}

inline double feasibility_jump_score(const Problem& problem, const Options& options,
                                     const RelaxationSolution& lp_relaxation,
                                     const Eigen::VectorXd& candidate,
                                     const std::vector<double>& row_weights) {
    const double violation = weighted_violation_score(problem, candidate, row_weights);
    double objective_term = compute_problem_objective(problem, candidate);
    const double obj_scale = std::max(1.0, std::abs(lp_relaxation.objective));
    objective_term =
        (problem.maximize ? -objective_term : objective_term) / obj_scale;
    return violation + options.feasibility_jump_objective_weight * objective_term;
}

template <typename SubMIPSolver>
inline std::optional<RelaxationSolution> try_feasibility_jump_repair_subproblem(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const Eigen::VectorXd& candidate, const std::vector<int>& integer_indices,
    const std::vector<double>& row_weights, int& lp_iterations, int& successes,
    SubMIPSolver&& solve_submip) {
    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;

    struct RankedVariable {
        int index = -1;
        double score = 0.0;
    };
    std::vector<RankedVariable> ranked;
    ranked.reserve(integer_indices.size());
    for (const int index : integer_indices) {
        double score = std::abs(candidate(index) - lp_relaxation.primal(index));
        for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
             ++row_index) {
            if (row_weights[row_index] <= 0.0) continue;
            const auto& row = problem.base_constraints[row_index];
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                if (row.indices[k] != index) continue;
                score += row_weights[row_index] * std::abs(row.values[k]);
            }
        }
        ranked.push_back({index, score});
    }

    std::sort(ranked.begin(), ranked.end(), [](const RankedVariable& lhs,
                                               const RankedVariable& rhs) {
        if (std::abs(lhs.score - rhs.score) > 1e-12) return lhs.score > rhs.score;
        return lhs.index < rhs.index;
    });

    const int free_limit =
        std::min<int>(options.feasibility_jump_max_free_vars, ranked.size());
    std::vector<char> free_mask(problem.variable_types.size(), 0);
    for (int i = 0; i < free_limit; ++i) {
        free_mask[ranked[i].index] = 1;
    }

    for (const int index : integer_indices) {
        if (!free_mask[index]) {
            lower(index) = candidate(index);
            upper(index) = candidate(index);
            continue;
        }
        if (problem.variable_types[index] == VariableType::Binary) continue;
        lower(index) = std::max(lower(index), candidate(index) - 1.0);
        upper(index) = std::min(upper(index), candidate(index) + 1.0);
    }

    return try_integer_subproblem(problem, options, lower, upper, lp_iterations, successes,
                                  solve_submip);
}

template <typename SubMIPSolver>
inline NeighborhoodHeuristicResult run_feasibility_jump_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    SubMIPSolver&& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_feasibility_jump || options.feasibility_jump_iterations <= 0) {
        return result;
    }

    std::vector<int> integer_indices;
    integer_indices.reserve(problem.variable_types.size());
    for (int j = 0; j < static_cast<int>(problem.variable_types.size()) &&
                    j < lp_relaxation.primal.size();
         ++j) {
        if (problem.variable_types[j] != VariableType::Continuous) {
            integer_indices.push_back(j);
        }
    }
    if (integer_indices.empty()) return result;

    Eigen::VectorXd current = lp_relaxation.primal;
    for (const int index : integer_indices) {
        const double value = lp_relaxation.primal(index);
        double rounded = std::round(value);
        if (index < problem.objective_coefficients.size()) {
            const double coeff = problem.objective_coefficients(index);
            if (problem.maximize) {
                rounded = coeff >= 0.0 ? std::ceil(value) : std::floor(value);
            } else {
                rounded = coeff >= 0.0 ? std::floor(value) : std::ceil(value);
            }
        }
        current(index) = rounded;
    }
    current = project_candidate_to_bounds(current, problem);

    std::vector<double> row_weights(problem.base_constraints.size(), 1.0);
    for (int iteration = 0; iteration < options.feasibility_jump_iterations; ++iteration) {
        auto exact = try_feasibility_jump_repair_subproblem(
            problem, options, lp_relaxation, current, integer_indices,
            std::vector<double>(problem.base_constraints.size(), 0.0), result.lp_iterations,
            result.successes, solve_submip);
        if (exact.has_value()) {
            result.incumbent = std::move(exact);
            return result;
        }

        const std::vector<double> active_row_weights =
            violated_row_weights(problem, current, row_weights, options.integrality_tol);
        auto repaired = try_feasibility_jump_repair_subproblem(
            problem, options, lp_relaxation, current, integer_indices, active_row_weights,
            result.lp_iterations, result.successes, solve_submip);
        if (repaired.has_value()) {
            result.incumbent = std::move(repaired);
            return result;
        }

        const double current_score =
            feasibility_jump_score(problem, options, lp_relaxation, current, row_weights);
        double best_score = current_score;
        Eigen::VectorXd best_candidate = current;
        bool found_move = false;

        std::vector<int> ordered = integer_indices;
        std::sort(ordered.begin(), ordered.end(), [&](int lhs, int rhs) {
            double lhs_score = std::abs(current(lhs) - lp_relaxation.primal(lhs));
            double rhs_score = std::abs(current(rhs) - lp_relaxation.primal(rhs));
            for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
                 ++row_index) {
                if (row_weights[row_index] <= 0.0) continue;
                const auto& row = problem.base_constraints[row_index];
                for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                                k < static_cast<int>(row.values.size());
                     ++k) {
                    if (row.indices[k] == lhs) lhs_score += row_weights[row_index] * std::abs(row.values[k]);
                    if (row.indices[k] == rhs) rhs_score += row_weights[row_index] * std::abs(row.values[k]);
                }
            }
            if (std::abs(lhs_score - rhs_score) > 1e-12) return lhs_score > rhs_score;
            return lhs < rhs;
        });

        const int evaluation_limit = std::min<int>(
            std::max(1, 2 * std::max(1, options.feasibility_jump_max_free_vars)),
            ordered.size());
        for (int p = 0; p < evaluation_limit; ++p) {
            const int index = ordered[p];
            std::vector<double> move_values;
            if (problem.variable_types[index] == VariableType::Binary) {
                move_values.push_back(current(index) >= 0.5 ? 0.0 : 1.0);
            } else {
                move_values.push_back(std::max(problem.lower_bounds(index), current(index) - 1.0));
                move_values.push_back(std::min(problem.upper_bounds(index), current(index) + 1.0));
                move_values.push_back(std::round(lp_relaxation.primal(index)));
            }
            std::sort(move_values.begin(), move_values.end());
            move_values.erase(std::unique(move_values.begin(), move_values.end(),
                                          [](double lhs, double rhs) {
                                              return std::abs(lhs - rhs) <= 1e-12;
                                          }),
                              move_values.end());

            for (const double move_value : move_values) {
                if (std::abs(move_value - current(index)) <= 1e-12) continue;
                Eigen::VectorXd candidate = current;
                candidate(index) = move_value;
                candidate = project_candidate_to_bounds(candidate, problem);
                const double candidate_score =
                    feasibility_jump_score(problem, options, lp_relaxation, candidate,
                                           row_weights);
                if (candidate_score + 1e-9 < best_score) {
                    best_score = candidate_score;
                    best_candidate = std::move(candidate);
                    found_move = true;
                }
            }
        }

        if (found_move) {
            current = std::move(best_candidate);
            continue;
        }

        bool bumped = false;
        for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
             ++row_index) {
            const double violation =
                linear_constraint_violation_amount(problem.base_constraints[row_index], current);
            if (violation <= options.integrality_tol) continue;
            row_weights[row_index] =
                std::min(1.0e6, row_weights[row_index] + 1.0 + 0.5 * violation);
            bumped = true;
        }
        if (!bumped) {
            break;
        }
    }

    return result;
}

template <typename SubMIPSolver>
inline NeighborhoodHeuristicResult run_feasibility_pump_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    SubMIPSolver&& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_feasibility_pump || options.feasibility_pump_iterations <= 0) {
        return result;
    }

    Eigen::VectorXd reference = lp_relaxation.primal;
    std::vector<int> integer_indices;
    integer_indices.reserve(problem.variable_types.size());
    for (int j = 0; j < static_cast<int>(problem.variable_types.size()) && j < reference.size();
         ++j) {
        if (problem.variable_types[j] != VariableType::Continuous) {
            integer_indices.push_back(j);
        }
    }
    if (integer_indices.empty()) return result;

    const int target_fixed = std::max(
        1, static_cast<int>(options.feasibility_pump_fix_ratio * integer_indices.size()));

    for (int round = 0; round < options.feasibility_pump_iterations; ++round) {
        const Eigen::VectorXd rounded =
            project_to_integral_lattice(reference, problem.variable_types);
        Eigen::VectorXd lower = problem.lower_bounds;
        Eigen::VectorXd upper = problem.upper_bounds;

        std::vector<std::pair<double, int>> ranked;
        ranked.reserve(integer_indices.size());
        for (const int index : integer_indices) {
            ranked.emplace_back(std::abs(reference(index) - rounded(index)), index);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

        int fixed = 0;
        for (const auto& [_, index] : ranked) {
            if (fixed >= target_fixed) break;
            const double value =
                std::min(upper(index), std::max(lower(index), rounded(index)));
            lower(index) = value;
            upper(index) = value;
            ++fixed;
        }

        for (const int index : integer_indices) {
            if (lower(index) == upper(index)) continue;
            if (problem.variable_types[index] == VariableType::Binary) continue;

            const double center = std::round(rounded(index));
            lower(index) = std::max(lower(index), center - 1.0);
            upper(index) = std::min(upper(index), center + 1.0);
        }

        auto incumbent = try_integer_subproblem(problem, options, lower, upper,
                                                result.lp_iterations, result.successes,
                                                solve_submip);
        if (incumbent.has_value()) {
            result.incumbent = std::move(incumbent);
            return result;
        }

        reference = 0.5 * reference + 0.5 * rounded;
        if (!ranked.empty()) {
            const int pivot = ranked[round % ranked.size()].second;
            if (problem.variable_types[pivot] == VariableType::Binary) {
                reference(pivot) = 1.0 - std::round(rounded(pivot));
            } else {
                reference(pivot) = rounded(pivot) +
                                   ((round % 2 == 0) ? 1.0 : -1.0);
                reference(pivot) = std::min(problem.upper_bounds(pivot),
                                            std::max(problem.lower_bounds(pivot),
                                                     reference(pivot)));
            }
        }
    }

    return result;
}

template <typename SubMIPSolver>
inline NeighborhoodHeuristicResult run_rens_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    SubMIPSolver&& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_rens) {
        return result;
    }

    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;
    std::vector<std::pair<double, int>> fix_candidates;
    int integer_count = 0;
    int fixed_count = 0;

    for (int j = 0; j < lp_relaxation.primal.size() &&
                    j < static_cast<int>(problem.variable_types.size());
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) continue;
        ++integer_count;
        const double value = lp_relaxation.primal(j);
        const double rounded = std::round(value);
        if (std::abs(value - rounded) <= options.integrality_tol) {
            lower(j) = rounded;
            upper(j) = rounded;
            ++fixed_count;
        } else {
            const double tightened_lower =
                std::max(lower(j), std::floor(value + options.integrality_tol));
            const double tightened_upper =
                std::min(upper(j), std::ceil(value - options.integrality_tol));
            lower(j) = tightened_lower;
            upper(j) = tightened_upper;
            if (upper(j) <= lower(j) + options.integrality_tol) {
                const double fixed_value = std::round(0.5 * (lower(j) + upper(j)));
                lower(j) = fixed_value;
                upper(j) = fixed_value;
                ++fixed_count;
            }
            fix_candidates.emplace_back(std::abs(value - rounded), j);
        }
    }

    if (integer_count == 0) {
        return result;
    }

    const int target_fixed =
        std::max(1, static_cast<int>(options.rens_fix_ratio * integer_count));

    auto solve_attempt = [&](const Eigen::VectorXd& attempt_lower,
                             const Eigen::VectorXd& attempt_upper) -> bool {
        auto incumbent = try_integer_subproblem(problem, options, attempt_lower, attempt_upper,
                                                result.lp_iterations, result.successes,
                                                solve_submip);
        if (!incumbent.has_value()) return false;
        result.incumbent = std::move(incumbent);
        return true;
    };

    if (solve_attempt(lower, upper)) {
        return result;
    }

    std::sort(fix_candidates.begin(), fix_candidates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    for (const double extra_fix_ratio : {options.rens_fix_ratio, std::min(0.95, options.rens_fix_ratio + 0.15)}) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        int fixed_in_attempt = fixed_count;
        const int target_for_round =
            std::max(target_fixed, static_cast<int>(std::ceil(extra_fix_ratio * integer_count)));
        const int need = std::min<int>(std::max(0, target_for_round - fixed_in_attempt),
                                       fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int index = fix_candidates[i].second;
            const double rounded = std::round(lp_relaxation.primal(index));
            attempt_lower(index) = rounded;
            attempt_upper(index) = rounded;
            ++fixed_in_attempt;
        }
        if (solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }

    if (!fix_candidates.empty()) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        const int need =
            std::min<int>(std::max(1, target_fixed - fixed_count), fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int index = fix_candidates[i].second;
            double guided = std::round(lp_relaxation.primal(index));
            if (index < problem.objective_coefficients.size()) {
                const double coeff = problem.objective_coefficients(index);
                guided = problem.maximize ? (coeff >= 0.0 ? std::ceil(lp_relaxation.primal(index))
                                                          : std::floor(lp_relaxation.primal(index)))
                                          : (coeff >= 0.0 ? std::floor(lp_relaxation.primal(index))
                                                          : std::ceil(lp_relaxation.primal(index)));
            }
            guided = std::min(problem.upper_bounds(index),
                              std::max(problem.lower_bounds(index), guided));
            attempt_lower(index) = guided;
            attempt_upper(index) = guided;
        }
        if (solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }
    return result;
}

template <typename SubMIPSolver>
inline NeighborhoodHeuristicResult run_rins_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
    SubMIPSolver&& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_rins || !std::isfinite(incumbent_objective)) {
        return result;
    }

    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;
    std::vector<std::pair<int, double>> extra_fix_candidates;
    int integer_count = 0;
    int fixed_count = 0;

    for (int j = 0; j < lp_relaxation.primal.size() &&
                    j < static_cast<int>(problem.variable_types.size());
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) continue;
        ++integer_count;
        const double incumbent_value = std::round(incumbent_primal(j));
        const double lp_value = lp_relaxation.primal(j);
        if (std::abs(lp_value - incumbent_value) <= options.rins_tolerance) {
            lower(j) = incumbent_value;
            upper(j) = incumbent_value;
            ++fixed_count;
            continue;
        }
        if (problem.variable_types[j] != VariableType::Binary) {
            lower(j) = std::max(lower(j), std::floor(lp_value + options.integrality_tol));
            upper(j) = std::min(upper(j), std::ceil(lp_value - options.integrality_tol));
            if (upper(j) <= lower(j) + options.integrality_tol) {
                const double fixed_value = std::round(0.5 * (lower(j) + upper(j)));
                lower(j) = fixed_value;
                upper(j) = fixed_value;
                ++fixed_count;
                continue;
            }
        }
        const double agreement_distance = std::abs(lp_value - incumbent_value);
        const double fractionality = std::abs(lp_value - std::round(lp_value));
        extra_fix_candidates.emplace_back(
            j, 0.65 * agreement_distance + 0.35 * fractionality);
    }

    if (integer_count == 0) {
        return result;
    }

    const int target_fixed = std::max(1, static_cast<int>(options.rins_fix_ratio * integer_count));

    auto solve_attempt = [&](const Eigen::VectorXd& attempt_lower,
                             const Eigen::VectorXd& attempt_upper) -> bool {
        auto incumbent = try_integer_subproblem(problem, options, attempt_lower, attempt_upper,
                                                result.lp_iterations, result.successes,
                                                solve_submip);
        if (!incumbent.has_value()) return false;
        if (!objective_improves_for_problem(incumbent->objective, incumbent_objective,
                                            problem.maximize, options.integrality_tol)) {
            return false;
        }
        result.incumbent = std::move(incumbent);
        return true;
    };

    if (solve_attempt(lower, upper)) {
        return result;
    }

    std::sort(extra_fix_candidates.begin(), extra_fix_candidates.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });

    for (const double extra_fix_ratio : {options.rins_fix_ratio, std::min(0.95, options.rins_fix_ratio + 0.15)}) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        int fixed_in_attempt = fixed_count;
        const int target_for_round =
            std::max(target_fixed, static_cast<int>(std::ceil(extra_fix_ratio * integer_count)));
        const int need = std::min<int>(std::max(0, target_for_round - fixed_in_attempt),
                                       extra_fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int var = extra_fix_candidates[i].first;
            const double incumbent_value = std::round(incumbent_primal(var));
            attempt_lower(var) = incumbent_value;
            attempt_upper(var) = incumbent_value;
            ++fixed_in_attempt;
        }
        if (solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }

    if (!extra_fix_candidates.empty()) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        const int need =
            std::min<int>(std::max(1, target_fixed - fixed_count), extra_fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int var = extra_fix_candidates[i].first;
            double guided = std::round(incumbent_primal(var));
            if (std::abs(lp_relaxation.primal(var) - guided) >
                std::abs(lp_relaxation.primal(var) - std::round(lp_relaxation.primal(var)))) {
                guided = std::round(lp_relaxation.primal(var));
            }
            guided = std::min(problem.upper_bounds(var),
                              std::max(problem.lower_bounds(var), guided));
            attempt_lower(var) = guided;
            attempt_upper(var) = guided;
        }
        if (solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }

    return result;
}

template <typename SubMIPSolver>
inline NeighborhoodHeuristicResult run_local_search_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
    SubMIPSolver&& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_local_search || !std::isfinite(incumbent_objective) ||
        options.local_search_iterations <= 0 || options.local_search_max_free_vars <= 0) {
        return result;
    }

    std::vector<std::pair<int, double>> ranked_integer_vars;
    ranked_integer_vars.reserve(problem.variable_types.size());
    for (int j = 0; j < static_cast<int>(problem.variable_types.size()) &&
                    j < lp_relaxation.primal.size();
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) continue;
        const double disagreement =
            std::abs(lp_relaxation.primal(j) - std::round(incumbent_primal(j)));
        ranked_integer_vars.emplace_back(j, disagreement);
    }
    std::sort(ranked_integer_vars.begin(), ranked_integer_vars.end(),
              [](const auto& lhs, const auto& rhs) {
                  if (std::abs(lhs.second - rhs.second) > 1e-12) {
                      return lhs.second > rhs.second;
                  }
                  return lhs.first < rhs.first;
              });

    if (ranked_integer_vars.empty()) {
        return result;
    }

    const int window =
        std::min<int>(options.local_search_max_free_vars, ranked_integer_vars.size());
    double best_objective = incumbent_objective;

    for (int iteration = 0; iteration < options.local_search_iterations; ++iteration) {
        Eigen::VectorXd lower = problem.lower_bounds;
        Eigen::VectorXd upper = problem.upper_bounds;
        std::vector<char> free_mask(problem.variable_types.size(), 0);
        for (int offset = 0; offset < window; ++offset) {
            const int ranked_index = (iteration + offset) % ranked_integer_vars.size();
            free_mask[ranked_integer_vars[ranked_index].first] = 1;
        }

        for (int j = 0; j < static_cast<int>(problem.variable_types.size()) &&
                        j < incumbent_primal.size();
             ++j) {
            if (problem.variable_types[j] == VariableType::Continuous || free_mask[j]) continue;
            const double fixed_value = std::round(incumbent_primal(j));
            lower(j) = fixed_value;
            upper(j) = fixed_value;
        }

        for (int j = 0; j < static_cast<int>(problem.variable_types.size()) &&
                        j < incumbent_primal.size();
             ++j) {
            if (!free_mask[j] || problem.variable_types[j] == VariableType::Continuous) continue;
            if (problem.variable_types[j] == VariableType::Binary) {
                continue;
            }
            const double center = std::round(incumbent_primal(j));
            lower(j) = std::max(lower(j), center - 1.0);
            upper(j) = std::min(upper(j), center + 1.0);
        }

        auto incumbent = try_integer_subproblem(problem, options, lower, upper,
                                                result.lp_iterations, result.successes,
                                                solve_submip);
        if (!incumbent.has_value() ||
            !objective_improves_for_problem(incumbent->objective, best_objective,
                                            problem.maximize, options.integrality_tol)) {
            continue;
        }

        best_objective = incumbent->objective;
        result.incumbent = std::move(incumbent);
    }

    return result;
}

template <typename SubMIPSolverWithCuts>
inline NeighborhoodHeuristicResult run_local_branching_heuristic(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
    SubMIPSolverWithCuts&& solve_submip_with_cuts) {
    NeighborhoodHeuristicResult result;
    if (!options.use_local_branching || !std::isfinite(incumbent_objective)) {
        return result;
    }

    std::vector<int> binary_indices;
    std::vector<int> agreeing_integer_indices;
    binary_indices.reserve(problem.variable_types.size());
    agreeing_integer_indices.reserve(problem.variable_types.size());
    int incumbent_ones = 0;

    for (int j = 0; j < static_cast<int>(problem.variable_types.size()) &&
                    j < incumbent_primal.size() && j < lp_relaxation.primal.size();
         ++j) {
        if (problem.variable_types[j] == VariableType::Binary) {
            binary_indices.push_back(j);
            if (std::round(incumbent_primal(j)) >= 1.0) {
                ++incumbent_ones;
            }
        } else if (problem.variable_types[j] != VariableType::Continuous) {
            const double incumbent_value = std::round(incumbent_primal(j));
            if (std::abs(lp_relaxation.primal(j) - incumbent_value) <=
                options.local_branching_lp_agreement_tol) {
                agreeing_integer_indices.push_back(j);
            }
        }
    }

    if (binary_indices.size() < 2) {
        return result;
    }

    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;

    const int max_fix_agree = static_cast<int>(
        options.local_branching_fix_agree_ratio * binary_indices.size());
    int fixed_agree = 0;
    for (const int index : agreeing_integer_indices) {
        if (fixed_agree >= max_fix_agree) break;
        const double incumbent_value = std::round(incumbent_primal(index));
        lower(index) = incumbent_value;
        upper(index) = incumbent_value;
        ++fixed_agree;
    }

    int radius = static_cast<int>(std::round(options.local_branching_neighborhood_ratio *
                                             static_cast<double>(binary_indices.size())));
    radius = std::max(options.local_branching_min_radius, radius);
    radius = std::min(options.local_branching_max_radius, radius);
    radius = std::min<int>(radius, binary_indices.size());
    if (radius <= 0) {
        return result;
    }

    std::vector<Cut> local_cuts;
    local_cuts.reserve(2);

    Cut neighborhood;
    neighborhood.sense = LinearConstraintSense::LessEqual;
    neighborhood.cut_type = "LocalBranching";
    neighborhood.rhs = static_cast<double>(radius - incumbent_ones);
    for (const int index : binary_indices) {
        const double incumbent_value = std::round(incumbent_primal(index));
        if (incumbent_value >= 1.0) {
            neighborhood.indices.push_back(index);
            neighborhood.values.push_back(-1.0);
        } else {
            neighborhood.indices.push_back(index);
            neighborhood.values.push_back(1.0);
        }
    }
    local_cuts.push_back(std::move(neighborhood));

    Cut improving_cut;
    improving_cut.cut_type = "LocalBranchingObjective";
    improving_cut.sense = problem.maximize ? LinearConstraintSense::GreaterEqual
                                           : LinearConstraintSense::LessEqual;
    improving_cut.rhs = incumbent_objective - problem.objective_constant +
                        (problem.maximize ? 1e-6 : -1e-6);
    for (int j = 0; j < problem.objective_coefficients.size(); ++j) {
        const double coeff = problem.objective_coefficients(j);
        if (std::abs(coeff) <= 1e-12) continue;
        improving_cut.indices.push_back(j);
        improving_cut.values.push_back(coeff);
    }
    if (!improving_cut.indices.empty()) {
        local_cuts.push_back(std::move(improving_cut));
    }

    const SolveResult subproblem = solve_submip_with_cuts(lower, upper, local_cuts);
    result.lp_iterations += subproblem.lp_iterations;
    if (!subproblem.has_solution ||
        !is_integer_feasible_solution(subproblem.primal, problem.variable_types,
                                      options.integrality_tol) ||
        !objective_improves_for_problem(subproblem.objective, incumbent_objective,
                                        problem.maximize, options.integrality_tol)) {
        return result;
    }

    ++result.successes;
    RelaxationSolution incumbent;
    incumbent.status = RelaxationStatus::Optimal;
    incumbent.primal = subproblem.primal;
    incumbent.objective = subproblem.objective;
    incumbent.iterations = subproblem.lp_iterations;
    result.incumbent = std::move(incumbent);
    return result;
}

}  // namespace simplex::bnb::detail
