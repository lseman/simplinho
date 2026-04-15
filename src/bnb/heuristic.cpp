#include "bnb/heuristic.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace simplex::bnb::detail {
namespace {

enum class ProjectionStage {
    BinaryOnly,
    AllIntegers,
};

double objective_guided_round_value(const Problem& problem, int index, double value,
                                    bool objective_guided, ProjectionStage stage,
                                    double perturbation = 0.0);

struct ColumnEntry {
    int row = -1;
    double coeff = 0.0;
};

struct LagrangianState {
    std::vector<std::vector<ColumnEntry>> columns;
    std::vector<double> multipliers;
    std::vector<double> row_activities;
    std::vector<double> row_violations;
    double weighted_violation = 0.0;
    double raw_violation = 0.0;
};

struct MoveEvaluation {
    int index = -1;
    double value = 0.0;
    double score = std::numeric_limits<double>::infinity();
    double raw_violation = std::numeric_limits<double>::infinity();
    double weighted_violation = std::numeric_limits<double>::infinity();
    double objective = std::numeric_limits<double>::infinity();
};

double compute_problem_objective(const Problem& problem, const Eigen::VectorXd& primal) {
    const int n = std::min<int>(problem.objective_coefficients.size(), primal.size());
    return (n > 0 ? problem.objective_coefficients.head(n).dot(primal.head(n)) : 0.0) +
           problem.objective_constant;
}

bool satisfies_linear_constraint(const SparseLinearConstraint& row, const Eigen::VectorXd& primal,
                                 double tol) {
    double lhs = 0.0;
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
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

bool satisfies_cut_constraint(const Cut& cut, const Eigen::VectorXd& primal, double tol) {
    double lhs = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
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

bool satisfies_all_constraints(const Problem& problem, const Eigen::VectorXd& primal, double tol,
                               const std::vector<Cut>& cuts = {}) {
    for (const auto& row : problem.base_constraints) {
        if (!satisfies_linear_constraint(row, primal, tol)) {
            return false;
        }
    }
    for (const auto& cut : cuts) {
        if (!satisfies_cut_constraint(cut, primal, tol)) {
            return false;
        }
    }
    return true;
}

std::optional<RelaxationSolution> make_incumbent_if_feasible(const Problem& problem,
                                                             const Options& options,
                                                             const Eigen::VectorXd& candidate,
                                                             const std::vector<Cut>& cuts = {}) {
    if (!is_integer_feasible_solution(candidate, problem.variable_types, options.integrality_tol)) {
        return std::nullopt;
    }
    if (!satisfies_all_constraints(problem, candidate, options.integrality_tol, cuts)) {
        return std::nullopt;
    }

    RelaxationSolution incumbent;
    incumbent.status = RelaxationStatus::Optimal;
    incumbent.primal = candidate;
    incumbent.objective = compute_problem_objective(problem, candidate);
    incumbent.iterations = 0;
    return incumbent;
}

double linear_constraint_violation_amount(const SparseLinearConstraint& row,
                                          const Eigen::VectorXd& primal) {
    double lhs = 0.0;
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
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

double row_violation_from_activity(const SparseLinearConstraint& row, double activity) {
    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            return std::max(0.0, activity - row.rhs);
        case LinearConstraintSense::GreaterEqual:
            return std::max(0.0, row.rhs - activity);
        case LinearConstraintSense::Equal:
            return std::abs(activity - row.rhs);
    }
    return 0.0;
}

Eigen::VectorXd project_candidate_to_bounds(const Eigen::VectorXd& primal, const Problem& problem,
                                            ProjectionStage stage = ProjectionStage::AllIntegers) {
    Eigen::VectorXd candidate = primal;
    const int n = std::min<int>(candidate.size(), problem.variable_types.size());
    for (int j = 0; j < n; ++j) {
        candidate(j) =
            std::min(problem.upper_bounds(j), std::max(problem.lower_bounds(j), candidate(j)));
        if (problem.variable_types[j] == VariableType::Continuous) {
            continue;
        }
        if (stage == ProjectionStage::BinaryOnly &&
            problem.variable_types[j] != VariableType::Binary) {
            continue;
        }
        candidate(j) = std::round(candidate(j));
    }
    return candidate;
}

bool is_stage_integer_variable(VariableType type, ProjectionStage stage) {
    if (type == VariableType::Continuous) {
        return false;
    }
    if (stage == ProjectionStage::BinaryOnly) {
        return type == VariableType::Binary;
    }
    return true;
}

void tighten_discrete_bounds(VariableType type, double* lower, double* upper, double tol) {
    if (lower == nullptr || upper == nullptr || type == VariableType::Continuous) {
        return;
    }
    *lower = std::ceil(*lower - tol);
    *upper = std::floor(*upper + tol);
    if (type == VariableType::Binary) {
        *lower = std::max(0.0, *lower);
        *upper = std::min(1.0, *upper);
    }
}

double nearest_integer_distance(double value) { return std::abs(value - std::round(value)); }

bool has_objective_rounding_tie(double value) {
    const double floor_distance = std::abs(value - std::floor(value));
    const double ceil_distance = std::abs(std::ceil(value) - value);
    return std::abs(floor_distance - ceil_distance) <= 1e-9;
}

double choose_tie_broken_fix_value(const Problem& problem, int index, double value) {
    if (!has_objective_rounding_tie(value)) {
        return std::round(value);
    }
    return objective_guided_round_value(problem, index, value, true, ProjectionStage::AllIntegers);
}

double objective_fixing_priority(const Problem& problem, int index, double value) {
    double priority = 0.0;
    if (index >= 0 && index < problem.objective_coefficients.size()) {
        priority = std::abs(problem.objective_coefficients(index));
    }
    if (has_objective_rounding_tie(value)) {
        priority += 1.0;
    }
    return priority;
}

int count_integer_variables(const Problem& problem) {
    int count = 0;
    for (const VariableType type : problem.variable_types) {
        if (type != VariableType::Continuous) {
            ++count;
        }
    }
    return count;
}

bool is_large_binary_problem(const Problem& problem) {
    return problem.variable_types.size() >= 256 &&
           std::all_of(problem.variable_types.begin(), problem.variable_types.end(),
                       [](VariableType type) { return type == VariableType::Binary; });
}

double min_submip_fixing_rate(const Problem& problem, int integer_count) {
    if (integer_count <= 0) {
        return 1.0;
    }
    if (is_large_binary_problem(problem)) {
        return 0.80;
    }
    if (integer_count >= 256) {
        return 0.70;
    }
    if (integer_count >= 128) {
        return 0.55;
    }
    return 0.35;
}

int max_submip_free_integers(const Problem& problem, int integer_count) {
    if (is_large_binary_problem(problem)) {
        return 64;
    }
    if (integer_count >= 256) {
        return 96;
    }
    if (integer_count >= 128) {
        return 80;
    }
    return 128;
}

bool should_attempt_submip_neighborhood(const Problem& problem, int integer_count,
                                        int fixed_count) {
    if (integer_count <= 0) {
        return false;
    }
    const int free_count = std::max(0, integer_count - fixed_count);
    const double fixing_rate =
        static_cast<double>(fixed_count) / static_cast<double>(integer_count);
    return fixing_rate >= min_submip_fixing_rate(problem, integer_count) &&
           free_count <= max_submip_free_integers(problem, integer_count);
}

std::vector<double> dynamic_fix_ratios(double base_ratio) {
    std::vector<double> ratios;
    const auto append_unique = [&ratios](double ratio) {
        const double clipped = std::min(0.95, std::max(0.35, ratio));
        for (const double existing : ratios) {
            if (std::abs(existing - clipped) <= 1e-9) {
                return;
            }
        }
        ratios.push_back(clipped);
    };

    append_unique(base_ratio);
    append_unique(base_ratio + 0.10);
    append_unique(base_ratio + 0.20);
    append_unique(base_ratio - 0.15);
    return ratios;
}

std::optional<Eigen::VectorXd>
extract_reduced_costs_if_aligned(const Problem& problem, const RelaxationSolution& lp_relaxation) {
    if (!lp_relaxation.lp_solution.has_value()) {
        return std::nullopt;
    }
    const Eigen::VectorXd& reduced_costs = lp_relaxation.lp_solution->reduced_costs_internal;
    const int variable_count = static_cast<int>(problem.variable_types.size());
    if (reduced_costs.size() == variable_count) {
        return reduced_costs;
    }
    return std::nullopt;
}

bool apply_leq_row_propagation(const Problem& problem, const Options& options,
                               const std::vector<int>& indices, const std::vector<double>& values,
                               double rhs, Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                               int* tightened_bounds) {
    if (lower == nullptr || upper == nullptr || tightened_bounds == nullptr) {
        return true;
    }

    double row_min = 0.0;
    for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
         ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12) {
            continue;
        }
        const double bound = coeff >= 0.0 ? (*lower)(index) : (*upper)(index);
        if (!std::isfinite(bound)) {
            return true;
        }
        row_min += coeff * bound;
    }

    if (row_min > rhs + options.integrality_tol) {
        return false;
    }

    for (int pivot = 0;
         pivot < static_cast<int>(indices.size()) && pivot < static_cast<int>(values.size());
         ++pivot) {
        const int index = indices[pivot];
        const double coeff = values[pivot];
        if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12) {
            continue;
        }

        double other_min = 0.0;
        for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
             ++k) {
            if (k == pivot) {
                continue;
            }
            const int other_index = indices[k];
            const double other_coeff = values[k];
            if (other_index < 0 || other_index >= lower->size() || std::abs(other_coeff) <= 1e-12) {
                continue;
            }
            const double bound = other_coeff >= 0.0 ? (*lower)(other_index) : (*upper)(other_index);
            if (!std::isfinite(bound)) {
                other_min = std::numeric_limits<double>::quiet_NaN();
                break;
            }
            other_min += other_coeff * bound;
        }
        if (!std::isfinite(other_min)) {
            continue;
        }

        double new_lower = (*lower)(index);
        double new_upper = (*upper)(index);
        const double candidate = (rhs - other_min) / coeff;
        if (coeff > 0.0) {
            if (std::isfinite(candidate) && candidate < new_upper - options.integrality_tol) {
                new_upper = candidate;
            }
        } else {
            if (std::isfinite(candidate) && candidate > new_lower + options.integrality_tol) {
                new_lower = candidate;
            }
        }

        tighten_discrete_bounds(problem.variable_types[index], &new_lower, &new_upper,
                                options.integrality_tol);
        if (new_upper + options.integrality_tol < new_lower) {
            return false;
        }
        if (new_lower > (*lower)(index) + options.integrality_tol) {
            (*lower)(index) = new_lower;
            ++(*tightened_bounds);
        }
        if (new_upper < (*upper)(index)-options.integrality_tol) {
            (*upper)(index) = new_upper;
            ++(*tightened_bounds);
        }
    }

    return true;
}

bool propagate_row_bounds(const Problem& problem, const Options& options,
                          const SparseLinearConstraint& row, Eigen::VectorXd* lower,
                          Eigen::VectorXd* upper, int* tightened_bounds) {
    if (row.sense == LinearConstraintSense::LessEqual) {
        return apply_leq_row_propagation(problem, options, row.indices, row.values, row.rhs, lower,
                                         upper, tightened_bounds);
    }
    std::vector<double> negated(row.values.size(), 0.0);
    for (int k = 0; k < static_cast<int>(row.values.size()); ++k) {
        negated[k] = -row.values[k];
    }
    if (row.sense == LinearConstraintSense::GreaterEqual) {
        return apply_leq_row_propagation(problem, options, row.indices, negated, -row.rhs, lower,
                                         upper, tightened_bounds);
    }
    return apply_leq_row_propagation(problem, options, row.indices, row.values, row.rhs, lower,
                                     upper, tightened_bounds) &&
           apply_leq_row_propagation(problem, options, row.indices, negated, -row.rhs, lower, upper,
                                     tightened_bounds);
}

bool propagate_fixed_neighborhood_bounds(const Problem& problem, const Options& options,
                                         Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                                         int* tightened_bounds) {
    if (lower == nullptr || upper == nullptr || tightened_bounds == nullptr) {
        return true;
    }

    for (int j = 0; j < lower->size() && j < static_cast<int>(problem.variable_types.size()); ++j) {
        double new_lower = (*lower)(j);
        double new_upper = (*upper)(j);
        tighten_discrete_bounds(problem.variable_types[j], &new_lower, &new_upper,
                                options.integrality_tol);
        if (new_upper + options.integrality_tol < new_lower) {
            return false;
        }
        (*lower)(j) = new_lower;
        (*upper)(j) = new_upper;
    }

    for (int round = 0; round < 4; ++round) {
        const int tightened_before = *tightened_bounds;
        for (const auto& row : problem.base_constraints) {
            if (!propagate_row_bounds(problem, options, row, lower, upper, tightened_bounds)) {
                return false;
            }
        }
        if (*tightened_bounds == tightened_before) {
            break;
        }
    }
    return true;
}

std::vector<int> collect_integer_indices(const Problem& problem, const Eigen::VectorXd& primal,
                                         ProjectionStage stage = ProjectionStage::AllIntegers) {
    std::vector<int> indices;
    for (int j = 0; j < primal.size() && j < static_cast<int>(problem.variable_types.size()); ++j) {
        if (is_stage_integer_variable(problem.variable_types[j], stage)) {
            indices.push_back(j);
        }
    }
    return indices;
}

double objective_guided_round_value(const Problem& problem, int index, double value,
                                    bool objective_guided, ProjectionStage stage,
                                    double perturbation) {
    const double lower = problem.lower_bounds(index);
    const double upper = problem.upper_bounds(index);
    const VariableType type = problem.variable_types[index];
    if (!is_stage_integer_variable(type, stage)) {
        return std::min(upper, std::max(lower, value));
    }

    if (type == VariableType::Binary) {
        if (!objective_guided) {
            return value >= 0.5 ? 1.0 : 0.0;
        }
        const double coeff = index < problem.objective_coefficients.size()
                                 ? problem.objective_coefficients(index)
                                 : 0.0;
        const bool prefer_up = problem.maximize ? (coeff >= 0.0) : (coeff < 0.0);
        const double biased = value + perturbation + (prefer_up ? 0.1 : -0.1);
        return biased >= 0.5 ? 1.0 : 0.0;
    }

    double rounded = std::round(value);
    if (objective_guided && index < problem.objective_coefficients.size()) {
        const double coeff = problem.objective_coefficients(index);
        const bool prefer_up = problem.maximize ? (coeff >= 0.0) : (coeff < 0.0);
        const double fractional = value - std::floor(value);
        if (fractional > 1e-9 && fractional < 1.0 - 1e-9) {
            rounded = prefer_up ? std::ceil(value) : std::floor(value);
        }
    }
    rounded = std::round(rounded + perturbation);
    return std::min(upper, std::max(lower, rounded));
}

Eigen::VectorXd build_projection(const Problem& problem, const RelaxationSolution& lp_relaxation,
                                 const Eigen::VectorXd& reference, ProjectionStage stage,
                                 bool objective_guided, double perturb_scale, int variant) {
    Eigen::VectorXd candidate = reference;
    const int n = std::min<int>(candidate.size(), problem.variable_types.size());
    for (int j = 0; j < n; ++j) {
        const bool active = is_stage_integer_variable(problem.variable_types[j], stage);
        const double base_value = active ? reference(j) : lp_relaxation.primal(j);
        const double perturbation =
            active ? perturb_scale *
                         ((variant + j) % 3 == 0 ? 1.0 : ((variant + j) % 3 == 1 ? -1.0 : 0.0))
                   : 0.0;
        candidate(j) = objective_guided_round_value(problem, j, base_value, objective_guided, stage,
                                                    perturbation);
    }
    return project_candidate_to_bounds(candidate, problem, stage);
}

std::vector<std::vector<ColumnEntry>> build_columns(const Problem& problem) {
    std::vector<std::vector<ColumnEntry>> columns(problem.variable_types.size());
    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const auto& row = problem.base_constraints[row_index];
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= static_cast<int>(columns.size())) {
                continue;
            }
            columns[index].push_back({row_index, row.values[k]});
        }
    }
    return columns;
}

LagrangianState
initialize_lagrangian_state(const Problem& problem, const Eigen::VectorXd& candidate,
                            std::vector<double> multipliers,
                            const std::vector<std::vector<ColumnEntry>>* columns = nullptr) {
    LagrangianState state;
    state.columns = columns != nullptr ? *columns : build_columns(problem);
    state.multipliers = std::move(multipliers);
    state.row_activities.assign(problem.base_constraints.size(), 0.0);
    state.row_violations.assign(problem.base_constraints.size(), 0.0);

    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const auto& row = problem.base_constraints[row_index];
        double activity = 0.0;
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index >= 0 && index < candidate.size()) {
                activity += row.values[k] * candidate(index);
            }
        }
        const double violation = row_violation_from_activity(row, activity);
        state.row_activities[row_index] = activity;
        state.row_violations[row_index] = violation;
        state.raw_violation += violation;
        state.weighted_violation += state.multipliers[row_index] * violation;
    }
    return state;
}

double normalized_objective_term(const Problem& problem, const RelaxationSolution& lp_relaxation,
                                 const Eigen::VectorXd& candidate) {
    const double value = compute_problem_objective(problem, candidate);
    const double scale = std::max(1.0, std::abs(lp_relaxation.objective));
    return (problem.maximize ? -value : value) / scale;
}

double lagrangian_score(const Problem& problem, const Options& options,
                        const RelaxationSolution& lp_relaxation, double weighted_violation,
                        double raw_violation, const Eigen::VectorXd& candidate,
                        bool objective_chasing) {
    const double objective_term = normalized_objective_term(problem, lp_relaxation, candidate);
    const double chasing_weight = objective_chasing
                                      ? (0.15 + 3.0 * options.feasibility_jump_objective_weight)
                                      : options.feasibility_jump_objective_weight;
    return weighted_violation + (objective_chasing ? 0.05 * raw_violation : 0.0) +
           chasing_weight * objective_term;
}

MoveEvaluation evaluate_move(const Problem& problem, const Options& options,
                             const RelaxationSolution& lp_relaxation,
                             const Eigen::VectorXd& candidate, const LagrangianState& state,
                             int index, double new_value, bool objective_chasing, double jitter) {
    MoveEvaluation evaluation;
    evaluation.index = index;
    evaluation.value = new_value;
    if (index < 0 || index >= candidate.size() || std::abs(new_value - candidate(index)) <= 1e-12) {
        return evaluation;
    }

    const double delta = new_value - candidate(index);
    double weighted_violation = state.weighted_violation;
    double raw_violation = state.raw_violation;
    for (const auto& entry : state.columns[index]) {
        const double old_violation = state.row_violations[entry.row];
        const double new_activity = state.row_activities[entry.row] + entry.coeff * delta;
        const double new_violation =
            row_violation_from_activity(problem.base_constraints[entry.row], new_activity);
        raw_violation += new_violation - old_violation;
        weighted_violation += state.multipliers[entry.row] * (new_violation - old_violation);
    }

    Eigen::VectorXd moved = candidate;
    moved(index) = new_value;
    moved = project_candidate_to_bounds(moved, problem, ProjectionStage::AllIntegers);

    evaluation.raw_violation = raw_violation;
    evaluation.weighted_violation = weighted_violation;
    evaluation.objective = compute_problem_objective(problem, moved);
    evaluation.score = lagrangian_score(problem, options, lp_relaxation, weighted_violation,
                                        raw_violation, moved, objective_chasing) +
                       jitter;
    return evaluation;
}

void apply_move(const Problem& problem, Eigen::VectorXd* candidate, LagrangianState* state,
                int index, double new_value) {
    const double delta = new_value - (*candidate)(index);
    if (std::abs(delta) <= 1e-12) {
        return;
    }

    for (const auto& entry : state->columns[index]) {
        const double old_violation = state->row_violations[entry.row];
        state->row_activities[entry.row] += entry.coeff * delta;
        const double new_violation = row_violation_from_activity(
            problem.base_constraints[entry.row], state->row_activities[entry.row]);
        state->row_violations[entry.row] = new_violation;
        state->raw_violation += new_violation - old_violation;
        state->weighted_violation +=
            state->multipliers[entry.row] * (new_violation - old_violation);
    }

    (*candidate)(index) = new_value;
    *candidate = project_candidate_to_bounds(*candidate, problem, ProjectionStage::AllIntegers);
}

uint64_t candidate_signature(const Eigen::VectorXd& candidate, const std::vector<int>& indices) {
    uint64_t hash = 1469598103934665603ULL;
    for (const int index : indices) {
        const int64_t value = static_cast<int64_t>(std::llround(candidate(index) * 1024.0));
        hash ^= static_cast<uint64_t>(value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));
        hash *= 1099511628211ULL;
    }
    return hash;
}

void perturb_multipliers(LagrangianState* state, const Problem& problem, std::mt19937_64* rng,
                         double integrality_tol, bool aggressive) {
    std::uniform_real_distribution<double> noise(0.0, aggressive ? 0.35 : 0.15);
    state->weighted_violation = 0.0;
    for (int row = 0; row < static_cast<int>(problem.base_constraints.size()); ++row) {
        const double violation = state->row_violations[row];
        if (violation > integrality_tol) {
            const double factor =
                (aggressive ? 1.2 : 1.05) + std::min(2.0, violation) * (aggressive ? 0.45 : 0.15);
            state->multipliers[row] =
                std::min(1.0e6, state->multipliers[row] * factor + noise(*rng));
        } else {
            state->multipliers[row] =
                std::max(1.0e-4, state->multipliers[row] * (aggressive ? 0.98 : 0.995));
        }
        state->weighted_violation += state->multipliers[row] * state->row_violations[row];
    }
}

void diversify_candidate(const Problem& problem, const RelaxationSolution& lp_relaxation,
                         const std::vector<int>& integer_indices, Eigen::VectorXd* candidate,
                         LagrangianState* state, std::mt19937_64* rng) {
    if (integer_indices.empty()) {
        return;
    }

    int selected_row = -1;
    double best_weighted_violation = -1.0;
    for (int row = 0; row < static_cast<int>(problem.base_constraints.size()); ++row) {
        const double weighted = state->multipliers[row] * state->row_violations[row];
        if (weighted > best_weighted_violation + 1e-12) {
            best_weighted_violation = weighted;
            selected_row = row;
        }
    }

    std::vector<int> candidates;
    if (selected_row >= 0) {
        for (const int index : problem.base_constraints[selected_row].indices) {
            if (index >= 0 && index < static_cast<int>(problem.variable_types.size()) &&
                problem.variable_types[index] != VariableType::Continuous) {
                candidates.push_back(index);
            }
        }
    }
    if (candidates.empty()) {
        candidates = integer_indices;
    }

    std::uniform_int_distribution<int> pick(0, static_cast<int>(candidates.size()) - 1);
    const int flips = std::min<int>(2, candidates.size());
    for (int t = 0; t < flips; ++t) {
        const int index = candidates[pick(*rng)];
        double new_value = (*candidate)(index);
        if (problem.variable_types[index] == VariableType::Binary) {
            new_value = new_value >= 0.5 ? 0.0 : 1.0;
        } else {
            const double target = std::round(lp_relaxation.primal(index));
            new_value = std::abs(target - new_value) <= 1e-12
                            ? new_value + (((*rng)() & 1ULL) ? 1.0 : -1.0)
                            : target;
            new_value = std::round(std::min(problem.upper_bounds(index),
                                            std::max(problem.lower_bounds(index), new_value)));
        }
        apply_move(problem, candidate, state, index, new_value);
    }
}

std::vector<double> violated_row_weights(const Problem& problem, const Eigen::VectorXd& candidate,
                                         const std::vector<double>& row_weights, double tol) {
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

std::optional<RelaxationSolution>
try_integer_subproblem(const Problem& problem, const Options& options, const Eigen::VectorXd& lower,
                       const Eigen::VectorXd& upper, NeighborhoodHeuristicResult* result,
                       const SubproblemSolveCallback& solve_submip) {
    const SolveResult subproblem = solve_submip(lower, upper);
    result->lp_iterations += subproblem.lp_iterations;
    if (!subproblem.has_solution ||
        !is_integer_feasible_solution(subproblem.primal, problem.variable_types,
                                      options.integrality_tol)) {
        return std::nullopt;
    }

    ++result->successes;
    RelaxationSolution incumbent;
    incumbent.status = RelaxationStatus::Optimal;
    incumbent.primal = subproblem.primal;
    incumbent.objective = subproblem.objective;
    incumbent.iterations = subproblem.lp_iterations;
    return incumbent;
}

std::optional<RelaxationSolution> try_propagated_integer_subproblem(
    const Problem& problem, const Options& options, Eigen::VectorXd lower, Eigen::VectorXd upper,
    NeighborhoodHeuristicResult* result, const SubproblemSolveCallback& solve_submip) {
    int tightened_bounds = 0;
    if (!propagate_fixed_neighborhood_bounds(problem, options, &lower, &upper, &tightened_bounds)) {
        return std::nullopt;
    }
    return try_integer_subproblem(problem, options, lower, upper, result, solve_submip);
}

std::optional<RelaxationSolution> try_feasibility_jump_repair_subproblem(
    const Problem& problem, const Options& options, const RelaxationSolution& lp_relaxation,
    const Eigen::VectorXd& candidate, const std::vector<int>& integer_indices,
    const std::vector<double>& row_weights, NeighborhoodHeuristicResult* result,
    const SubproblemSolveCallback& solve_submip) {
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
            if (row_weights[row_index] <= 0.0) {
                continue;
            }
            const auto& row = problem.base_constraints[row_index];
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                if (row.indices[k] == index) {
                    score += row_weights[row_index] * std::abs(row.values[k]);
                }
            }
        }
        ranked.push_back({index, score});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedVariable& lhs, const RankedVariable& rhs) {
                  if (std::abs(lhs.score - rhs.score) > 1e-12) {
                      return lhs.score > rhs.score;
                  }
                  return lhs.index < rhs.index;
              });

    const int free_limit = std::min<int>(options.feasibility_jump_max_free_vars, ranked.size());
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
        if (problem.variable_types[index] == VariableType::Binary) {
            continue;
        }
        lower(index) = std::max(lower(index), candidate(index) - 1.0);
        upper(index) = std::min(upper(index), candidate(index) + 1.0);
    }

    return try_integer_subproblem(problem, options, lower, upper, result, solve_submip);
}

std::optional<RelaxationSolution> choose_better(const std::optional<RelaxationSolution>& lhs,
                                                const std::optional<RelaxationSolution>& rhs,
                                                bool maximize) {
    if (!lhs.has_value()) {
        return rhs;
    }
    if (!rhs.has_value()) {
        return lhs;
    }
    return objective_improves_for_problem(rhs->objective, lhs->objective, maximize, 1e-12) ? rhs
                                                                                           : lhs;
}

std::vector<Eigen::VectorXd> feasibility_jump_starts(const Problem& problem,
                                                     const RelaxationSolution& lp_relaxation) {
    std::vector<Eigen::VectorXd> starts;
    starts.push_back(build_projection(problem, lp_relaxation, lp_relaxation.primal,
                                      ProjectionStage::AllIntegers, false, 0.0, 0));
    starts.push_back(build_projection(problem, lp_relaxation, lp_relaxation.primal,
                                      ProjectionStage::AllIntegers, true, 0.0, 1));
    starts.push_back(build_projection(problem, lp_relaxation, lp_relaxation.primal,
                                      ProjectionStage::AllIntegers, true, 0.49, 2));

    std::vector<Eigen::VectorXd> unique_starts;
    const std::vector<int> indices = collect_integer_indices(problem, lp_relaxation.primal);
    for (auto& start : starts) {
        const uint64_t signature = candidate_signature(start, indices);
        bool duplicate = false;
        for (const auto& existing : unique_starts) {
            if (candidate_signature(existing, indices) == signature) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            unique_starts.push_back(std::move(start));
        }
    }
    return unique_starts;
}

void build_pump_subproblem_bounds(const Problem& problem, const Options& options,
                                  const Eigen::VectorXd& reference, const Eigen::VectorXd& rounded,
                                  const std::vector<int>& stage_indices, ProjectionStage stage,
                                  Eigen::VectorXd* lower, Eigen::VectorXd* upper) {
    *lower = problem.lower_bounds;
    *upper = problem.upper_bounds;
    if (stage_indices.empty()) {
        return;
    }

    const int target_fixed =
        std::max(1, static_cast<int>(options.feasibility_pump_fix_ratio * stage_indices.size()));
    std::vector<std::pair<double, int>> ranked;
    ranked.reserve(stage_indices.size());
    for (const int index : stage_indices) {
        double score = std::abs(reference(index) - rounded(index));
        if (index < problem.objective_coefficients.size()) {
            score -= 0.05 * std::abs(problem.objective_coefficients(index));
        }
        ranked.emplace_back(score, index);
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
        if (std::abs(lhs.first - rhs.first) > 1e-12) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    int fixed = 0;
    for (const auto& [_, index] : ranked) {
        if (fixed >= target_fixed) {
            break;
        }
        (*lower)(index) = rounded(index);
        (*upper)(index) = rounded(index);
        ++fixed;
    }

    if (stage == ProjectionStage::AllIntegers) {
        for (const int index : stage_indices) {
            if ((*lower)(index) == (*upper)(index) ||
                problem.variable_types[index] == VariableType::Binary) {
                continue;
            }
            const double center = rounded(index);
            (*lower)(index) = std::max((*lower)(index), center - 1.0);
            (*upper)(index) = std::min((*upper)(index), center + 1.0);
        }
    }
}

void perturb_pump_point(const Problem& problem, const RelaxationSolution& lp_relaxation,
                        const std::vector<int>& stage_indices, Eigen::VectorXd* rounded,
                        std::mt19937_64* rng) {
    if (stage_indices.empty()) {
        return;
    }

    int chosen = -1;
    double best_score = -1.0;
    for (const int index : stage_indices) {
        double score = 0.0;
        for (const auto& row : problem.base_constraints) {
            const double violation = linear_constraint_violation_amount(row, *rounded);
            if (violation <= 1e-9) {
                continue;
            }
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                if (row.indices[k] == index) {
                    score += violation * std::abs(row.values[k]);
                }
            }
        }
        score += 0.25 * std::abs((*rounded)(index)-lp_relaxation.primal(index));
        if (score > best_score + 1e-12) {
            best_score = score;
            chosen = index;
        }
    }
    if (chosen < 0) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(stage_indices.size()) - 1);
        chosen = stage_indices[pick(*rng)];
    }

    if (problem.variable_types[chosen] == VariableType::Binary) {
        (*rounded)(chosen) = (*rounded)(chosen) >= 0.5 ? 0.0 : 1.0;
        return;
    }

    const double target = std::round(lp_relaxation.primal(chosen));
    if (std::abs((*rounded)(chosen)-target) <= 1e-12) {
        (*rounded)(chosen) += ((*rng)() & 1ULL) ? 1.0 : -1.0;
    } else {
        (*rounded)(chosen) = target;
    }
    (*rounded)(chosen) = std::round(std::min(
        problem.upper_bounds(chosen), std::max(problem.lower_bounds(chosen), (*rounded)(chosen))));
}

} // namespace

static std::optional<RelaxationSolution>
try_greedy_repair_rounding(const Problem& problem, const Options& options,
                           Eigen::VectorXd candidate, const std::vector<Cut>& active_cuts) {
    const int n = std::min<int>(candidate.size(), static_cast<int>(problem.variable_types.size()));

    struct FlipCandidate {
        int j;
        double violation_score;
        double objective_score;
        double combined_score;
    };
    std::vector<FlipCandidate> flippable;
    flippable.reserve(n);

    const double base_objective = compute_problem_objective(problem, candidate);
    for (int j = 0; j < n; ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) {
            continue;
        }
        const double lb = problem.lower_bounds(j);
        const double ub = problem.upper_bounds(j);
        if (!std::isfinite(lb) || !std::isfinite(ub)) {
            continue;
        }
        const double current = candidate(j);
        if (std::abs(current - std::round(current)) <= options.integrality_tol) {
            continue;
        }

        const double target = std::round(current);
        const double candidate_value = std::min(ub, std::max(lb, target));
        const double coeff = j < static_cast<int>(problem.objective_coefficients.size())
                                 ? problem.objective_coefficients(j)
                                 : 0.0;
        const double objective_impact = problem.maximize ? -coeff * (candidate_value - current)
                                                         : coeff * (candidate_value - current);

        double violation_score = 0.0;
        for (const auto& row : problem.base_constraints) {
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                if (row.indices[k] != j) {
                    continue;
                }
                const double old_activity = row.values[k] * current;
                const double new_activity = row.values[k] * candidate_value;
                const double old_violation = linear_constraint_violation_amount(row, candidate);
                Eigen::VectorXd modified = candidate;
                modified(j) = candidate_value;
                const double new_violation = linear_constraint_violation_amount(row, modified);
                violation_score += std::max(0.0, old_violation - new_violation);
            }
        }
        const double combined = 0.75 * violation_score - 0.25 * objective_impact;
        flippable.push_back({j, violation_score, objective_impact, combined});
    }

    if (flippable.empty()) {
        return std::nullopt;
    }
    std::sort(flippable.begin(), flippable.end(),
              [](const FlipCandidate& a, const FlipCandidate& b) {
                  if (std::abs(a.combined_score - b.combined_score) > 1e-12)
                      return a.combined_score > b.combined_score;
                  if (std::abs(a.violation_score - b.violation_score) > 1e-12)
                      return a.violation_score > b.violation_score;
                  return a.objective_score < b.objective_score;
              });

    for (const auto& fc : flippable) {
        const double target = std::round(candidate(fc.j));
        const double value =
            std::min(problem.upper_bounds(fc.j), std::max(problem.lower_bounds(fc.j), target));
        candidate(fc.j) = value;
        if (auto sol = make_incumbent_if_feasible(problem, options, candidate, active_cuts)) {
            return sol;
        }
    }
    return std::nullopt;
}

std::optional<RelaxationSolution> run_rounding_heuristic(const Problem& problem,
                                                         const Options& options,
                                                         const RelaxationSolution& lp_relaxation,
                                                         const std::vector<Cut>& active_cuts) {
    if (lp_relaxation.primal.size() == 0) {
        return std::nullopt;
    }

    const std::array<bool, 2> objective_guided_choices = {false, true};
    const std::array<ProjectionStage, 2> stages = {ProjectionStage::BinaryOnly,
                                                   ProjectionStage::AllIntegers};
    const std::array<double, 3> perturb_scales = {0.0, 0.25, 0.5};

    for (const ProjectionStage stage : stages) {
        for (const bool objective_guided : objective_guided_choices) {
            for (double perturb_scale : perturb_scales) {
                for (int variant = 0; variant < 3; ++variant) {
                    const Eigen::VectorXd candidate =
                        build_projection(problem, lp_relaxation, lp_relaxation.primal, stage,
                                         objective_guided, perturb_scale, variant);
                    if (auto incumbent =
                            make_incumbent_if_feasible(problem, options, candidate, active_cuts);
                        incumbent.has_value()) {
                        return incumbent;
                    }
                    if (auto repaired =
                            try_greedy_repair_rounding(problem, options, candidate, active_cuts)) {
                        return repaired;
                    }
                }
            }
        }
    }

    return std::nullopt;
}

NeighborhoodHeuristicResult
run_feasibility_jump_heuristic(const Problem& problem, const Options& options,
                               const RelaxationSolution& lp_relaxation,
                               const SubproblemSolveCallback& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_feasibility_jump || options.feasibility_jump_iterations <= 0) {
        return result;
    }

    const std::vector<int> integer_indices =
        collect_integer_indices(problem, lp_relaxation.primal, ProjectionStage::AllIntegers);
    if (integer_indices.empty()) {
        return result;
    }

    const std::vector<Eigen::VectorXd> starts = feasibility_jump_starts(problem, lp_relaxation);
    const int start_count = std::max<int>(1, starts.size());
    const int per_start_iterations = std::max(
        3, static_cast<int>(std::ceil(static_cast<double>(options.feasibility_jump_iterations) /
                                      static_cast<double>(start_count))));
    const double low_violation_threshold =
        std::max(25.0 * options.integrality_tol,
                 1e-4 * std::max(1, static_cast<int>(problem.base_constraints.size())));
    const auto columns = build_columns(problem);

    std::optional<RelaxationSolution> best_incumbent;
    std::optional<RelaxationSolution> best_repair_incumbent;

    for (int start_index = 0; start_index < static_cast<int>(starts.size()); ++start_index) {
        Eigen::VectorXd current = starts[start_index];
        if (auto incumbent = make_incumbent_if_feasible(problem, options, current);
            incumbent.has_value()) {
            result.incumbent = incumbent;
            result.successes = std::max(result.successes, 1);
            return result;
        }

        LagrangianState state = initialize_lagrangian_state(
            problem, current, std::vector<double>(problem.base_constraints.size(), 1.0), &columns);
        std::mt19937_64 rng(0xC0FFEEULL + static_cast<uint64_t>(start_index) * 1315423911ULL);
        std::vector<uint64_t> recent_signatures;
        recent_signatures.reserve(8);
        int stuck_count = 0;

        for (int iteration = 0; iteration < per_start_iterations; ++iteration) {
            const bool objective_chasing = state.raw_violation <= low_violation_threshold;
            if (objective_chasing) {
                if (auto incumbent = make_incumbent_if_feasible(problem, options, current);
                    incumbent.has_value()) {
                    best_incumbent = choose_better(best_incumbent, incumbent, problem.maximize);
                    result.incumbent = best_incumbent;
                    result.successes = std::max(result.successes, 1);
                    return result;
                }
            }

            if (iteration % 2 == 0 || objective_chasing || stuck_count > 0) {
                const std::vector<double> active_weights = violated_row_weights(
                    problem, current, state.multipliers, options.integrality_tol);
                if (auto repaired = try_feasibility_jump_repair_subproblem(
                        problem, options, lp_relaxation, current, integer_indices, active_weights,
                        &result, solve_submip);
                    repaired.has_value()) {
                    best_repair_incumbent =
                        choose_better(best_repair_incumbent, repaired, problem.maximize);
                    result.incumbent =
                        choose_better(best_incumbent, best_repair_incumbent, problem.maximize);
                    return result;
                }
            }
            const double current_score =
                lagrangian_score(problem, options, lp_relaxation, state.weighted_violation,
                                 state.raw_violation, current, objective_chasing);
            MoveEvaluation best_move;
            std::uniform_real_distribution<double> jitter(0.0, objective_chasing ? 1e-6 : 5e-5);
            for (const int index : integer_indices) {
                std::vector<double> move_values;
                if (problem.variable_types[index] == VariableType::Binary) {
                    move_values.push_back(current(index) >= 0.5 ? 0.0 : 1.0);
                } else {
                    move_values.push_back(std::round(lp_relaxation.primal(index)));
                    move_values.push_back(
                        std::min(problem.upper_bounds(index), current(index) + 1.0));
                    move_values.push_back(
                        std::max(problem.lower_bounds(index), current(index) - 1.0));
                    move_values.push_back(objective_guided_round_value(
                        problem, index, current(index), true, ProjectionStage::AllIntegers));
                }
                std::sort(move_values.begin(), move_values.end());
                move_values.erase(std::unique(move_values.begin(), move_values.end(),
                                              [](double lhs, double rhs) {
                                                  return std::abs(lhs - rhs) <= 1e-12;
                                              }),
                                  move_values.end());
                for (const double move_value : move_values) {
                    const MoveEvaluation evaluated =
                        evaluate_move(problem, options, lp_relaxation, current, state, index,
                                      move_value, objective_chasing, jitter(rng));
                    if (evaluated.score + 1e-9 < best_move.score ||
                        (std::abs(evaluated.score - best_move.score) <= 1e-9 &&
                         evaluated.raw_violation + 1e-9 < best_move.raw_violation)) {
                        best_move = evaluated;
                    }
                }
            }
            const bool improved =
                best_move.index >= 0 &&
                (best_move.score + 1e-9 < current_score ||
                 (objective_chasing && best_move.raw_violation <= state.raw_violation + 1e-9 &&
                  objective_improves_for_problem(best_move.objective,
                                                 compute_problem_objective(problem, current),
                                                 problem.maximize, 1e-9)));
            if (improved) {
                apply_move(problem, &current, &state, best_move.index, best_move.value);
                stuck_count = 0;
            } else {
                ++stuck_count;
                perturb_multipliers(&state, problem, &rng, options.integrality_tol,
                                    stuck_count > 1);
                diversify_candidate(problem, lp_relaxation, integer_indices, &current, &state,
                                    &rng);
            }
            const uint64_t signature = candidate_signature(current, integer_indices);
            const bool cycled = std::find(recent_signatures.begin(), recent_signatures.end(),
                                          signature) != recent_signatures.end();
            recent_signatures.push_back(signature);
            if (recent_signatures.size() > 8) {
                recent_signatures.erase(recent_signatures.begin());
            }
            if (cycled) {
                perturb_multipliers(&state, problem, &rng, options.integrality_tol, true);
                diversify_candidate(problem, lp_relaxation, integer_indices, &current, &state,
                                    &rng);
            }
        }
        const std::vector<double> active_weights =
            violated_row_weights(problem, current, state.multipliers, options.integrality_tol);
        if (auto repaired = try_feasibility_jump_repair_subproblem(
                problem, options, lp_relaxation, current, integer_indices, active_weights, &result,
                solve_submip);
            repaired.has_value()) {
            best_repair_incumbent =
                choose_better(best_repair_incumbent, repaired, problem.maximize);
        }
    }
    result.incumbent = choose_better(best_incumbent, best_repair_incumbent, problem.maximize);
    return result;
}
NeighborhoodHeuristicResult
run_feasibility_pump_heuristic(const Problem& problem, const Options& options,
                               const RelaxationSolution& lp_relaxation,
                               const SubproblemSolveCallback& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_feasibility_pump || options.feasibility_pump_iterations <= 0) {
        return result;
    }
    const std::vector<int> binary_indices =
        collect_integer_indices(problem, lp_relaxation.primal, ProjectionStage::BinaryOnly);
    const std::vector<int> all_integer_indices =
        collect_integer_indices(problem, lp_relaxation.primal, ProjectionStage::AllIntegers);
    if (all_integer_indices.empty()) {
        return result;
    }
    std::mt19937_64 rng(0xF00DFACEULL);
    std::optional<RelaxationSolution> best_incumbent;
    Eigen::VectorXd reference = lp_relaxation.primal;
    std::vector<uint64_t> recent_signatures;
    recent_signatures.reserve(10);
    const int binary_rounds =
        binary_indices.empty() ? 0 : std::max(1, options.feasibility_pump_iterations / 2);
    const int integer_rounds = std::max(1, options.feasibility_pump_iterations - binary_rounds);
    const std::vector<std::pair<ProjectionStage, int>> stages = {
        {ProjectionStage::BinaryOnly, binary_rounds},
        {ProjectionStage::AllIntegers, integer_rounds},
    };
    for (const auto& [stage, rounds] : stages) {
        const std::vector<int>& stage_indices =
            stage == ProjectionStage::BinaryOnly ? binary_indices : all_integer_indices;
        if (stage_indices.empty() || rounds <= 0) {
            continue;
        }
        for (int round = 0; round < rounds; ++round) {
            Eigen::VectorXd rounded =
                build_projection(problem, lp_relaxation, reference, stage, true, 0.0, round);
            if (auto incumbent = make_incumbent_if_feasible(problem, options, rounded);
                incumbent.has_value()) {
                result.incumbent = choose_better(best_incumbent, incumbent, problem.maximize);
                result.successes = std::max(result.successes, 1);
                return result;
            }
            const uint64_t signature = candidate_signature(rounded, stage_indices);
            const bool cycle = std::find(recent_signatures.begin(), recent_signatures.end(),
                                         signature) != recent_signatures.end();
            recent_signatures.push_back(signature);
            if (recent_signatures.size() > 10) {
                recent_signatures.erase(recent_signatures.begin());
            }
            if (cycle) {
                perturb_pump_point(problem, lp_relaxation, stage_indices, &rounded, &rng);
            }
            Eigen::VectorXd lower;
            Eigen::VectorXd upper;
            build_pump_subproblem_bounds(problem, options, reference, rounded, stage_indices, stage,
                                         &lower, &upper);
            const int stage_target_fixed =
                std::max(1, static_cast<int>(options.feasibility_pump_fix_ratio *
                                             static_cast<double>(stage_indices.size())));
            if (should_attempt_submip_neighborhood(problem, static_cast<int>(stage_indices.size()),
                                                   stage_target_fixed)) {
                if (auto incumbent = try_integer_subproblem(problem, options, lower, upper, &result,
                                                            solve_submip);
                    incumbent.has_value()) {
                    best_incumbent = choose_better(best_incumbent, incumbent, problem.maximize);
                    result.incumbent = best_incumbent;
                    return result;
                }
            }
            const std::vector<double> ones(problem.base_constraints.size(), 1.0);
            const std::vector<double> active_weights =
                violated_row_weights(problem, rounded, ones, options.integrality_tol);
            if (auto repaired = try_feasibility_jump_repair_subproblem(
                    problem, options, lp_relaxation, rounded, stage_indices, active_weights,
                    &result, solve_submip);
                repaired.has_value()) {
                best_incumbent = choose_better(best_incumbent, repaired, problem.maximize);
                result.incumbent = best_incumbent;
                return result;
            }
            const double blend = stage == ProjectionStage::BinaryOnly ? 0.65 : 0.55;
            reference = (1.0 - blend) * reference + blend * rounded;
            if (cycle) {
                perturb_pump_point(problem, lp_relaxation, stage_indices, &reference, &rng);
            }
            reference = project_candidate_to_bounds(reference, problem, stage);
        }
    }
    result.incumbent = best_incumbent;
    return result;
}
NeighborhoodHeuristicResult run_rens_heuristic(const Problem& problem, const Options& options,
                                               const RelaxationSolution& lp_relaxation,
                                               const SubproblemSolveCallback& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_rens) {
        return result;
    }
    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;

    struct RensFixCandidate {
        int index = -1;
        double distance = 0.0;
        double objective_priority = 0.0;
    };

    std::vector<RensFixCandidate> fix_candidates;
    int integer_count = 0;
    int fixed_count = 0;
    for (int j = 0;
         j < lp_relaxation.primal.size() && j < static_cast<int>(problem.variable_types.size());
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) {
            continue;
        }
        ++integer_count;
        const double value = lp_relaxation.primal(j);
        const double rounded = std::round(value);
        if (std::abs(value - rounded) <= options.integrality_tol) {
            lower(j) = rounded;
            upper(j) = rounded;
            ++fixed_count;
        } else {
            lower(j) = std::max(lower(j), std::floor(value + options.integrality_tol));
            upper(j) = std::min(upper(j), std::ceil(value - options.integrality_tol));
            if (upper(j) <= lower(j) + options.integrality_tol) {
                const double fixed_value = std::round(0.5 * (lower(j) + upper(j)));
                lower(j) = fixed_value;
                upper(j) = fixed_value;
                ++fixed_count;
            }
            fix_candidates.push_back(
                {j, nearest_integer_distance(value), objective_fixing_priority(problem, j, value)});
        }
    }
    if (integer_count == 0) {
        return result;
    }
    const int target_fixed = std::max(1, static_cast<int>(options.rens_fix_ratio * integer_count));
    auto solve_attempt = [&](Eigen::VectorXd attempt_lower, Eigen::VectorXd attempt_upper) -> bool {
        auto incumbent =
            try_propagated_integer_subproblem(problem, options, std::move(attempt_lower),
                                              std::move(attempt_upper), &result, solve_submip);
        if (!incumbent.has_value()) {
            return false;
        }
        result.incumbent = std::move(incumbent);
        return true;
    };
    if (should_attempt_submip_neighborhood(problem, integer_count, fixed_count) &&
        solve_attempt(lower, upper)) {
        return result;
    }
    std::sort(fix_candidates.begin(), fix_candidates.end(),
              [](const RensFixCandidate& lhs, const RensFixCandidate& rhs) {
                  if (std::abs(lhs.distance - rhs.distance) > 1e-12) {
                      return lhs.distance < rhs.distance;
                  }
                  if (std::abs(lhs.objective_priority - rhs.objective_priority) > 1e-12) {
                      return lhs.objective_priority > rhs.objective_priority;
                  }
                  return lhs.index < rhs.index;
              });
    for (const double extra_fix_ratio : dynamic_fix_ratios(options.rens_fix_ratio)) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        int fixed_in_attempt = fixed_count;
        const int target_for_round =
            std::max(target_fixed, static_cast<int>(std::ceil(extra_fix_ratio * integer_count)));
        const int need =
            std::min<int>(std::max(0, target_for_round - fixed_in_attempt), fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int index = fix_candidates[i].index;
            const double fixed_value =
                choose_tie_broken_fix_value(problem, index, lp_relaxation.primal(index));
            attempt_lower(index) = fixed_value;
            attempt_upper(index) = fixed_value;
            ++fixed_in_attempt;
        }
        if (should_attempt_submip_neighborhood(problem, integer_count, fixed_in_attempt) &&
            solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }
    if (!fix_candidates.empty()) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        const int need =
            std::min<int>(std::max(1, target_fixed - fixed_count), fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int index = fix_candidates[i].index;
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
        if (should_attempt_submip_neighborhood(problem, integer_count, fixed_count + need) &&
            solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }
    return result;
}
NeighborhoodHeuristicResult run_rins_heuristic(const Problem& problem, const Options& options,
                                               const RelaxationSolution& lp_relaxation,
                                               const Eigen::VectorXd& incumbent_primal,
                                               double incumbent_objective,
                                               const SubproblemSolveCallback& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_rins || !std::isfinite(incumbent_objective)) {
        return result;
    }
    Eigen::VectorXd lower = problem.lower_bounds;
    Eigen::VectorXd upper = problem.upper_bounds;

    struct RinsFixCandidate {
        int index = -1;
        double agreement_distance = 0.0;
        double fractionality = 0.0;
        double reduced_cost_magnitude = 0.0;
    };

    const std::optional<Eigen::VectorXd> reduced_costs =
        extract_reduced_costs_if_aligned(problem, lp_relaxation);
    std::vector<RinsFixCandidate> extra_fix_candidates;
    int integer_count = 0;
    int fixed_count = 0;
    for (int j = 0;
         j < lp_relaxation.primal.size() && j < static_cast<int>(problem.variable_types.size());
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) {
            continue;
        }
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
        const double rc_magnitude = reduced_costs.has_value() && j < reduced_costs->size()
                                        ? std::abs((*reduced_costs)(j))
                                        : 0.0;
        extra_fix_candidates.push_back({j, agreement_distance, fractionality, rc_magnitude});
    }
    if (integer_count == 0) {
        return result;
    }
    const int target_fixed = std::max(1, static_cast<int>(options.rins_fix_ratio * integer_count));
    auto solve_attempt = [&](Eigen::VectorXd attempt_lower, Eigen::VectorXd attempt_upper) -> bool {
        auto incumbent =
            try_propagated_integer_subproblem(problem, options, std::move(attempt_lower),
                                              std::move(attempt_upper), &result, solve_submip);
        if (!incumbent.has_value()) {
            return false;
        }
        if (!objective_improves_for_problem(incumbent->objective, incumbent_objective,
                                            problem.maximize, options.integrality_tol)) {
            return false;
        }
        result.incumbent = std::move(incumbent);
        return true;
    };
    if (should_attempt_submip_neighborhood(problem, integer_count, fixed_count) &&
        solve_attempt(lower, upper)) {
        return result;
    }
    std::sort(extra_fix_candidates.begin(), extra_fix_candidates.end(),
              [](const RinsFixCandidate& lhs, const RinsFixCandidate& rhs) {
                  if (std::abs(lhs.agreement_distance - rhs.agreement_distance) > 1e-12) {
                      return lhs.agreement_distance < rhs.agreement_distance;
                  }
                  if (std::abs(lhs.reduced_cost_magnitude - rhs.reduced_cost_magnitude) > 1e-12) {
                      return lhs.reduced_cost_magnitude > rhs.reduced_cost_magnitude;
                  }
                  if (std::abs(lhs.fractionality - rhs.fractionality) > 1e-12) {
                      return lhs.fractionality < rhs.fractionality;
                  }
                  return lhs.index < rhs.index;
              });
    for (const double extra_fix_ratio : dynamic_fix_ratios(options.rins_fix_ratio)) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        int fixed_in_attempt = fixed_count;
        const int target_for_round =
            std::max(target_fixed, static_cast<int>(std::ceil(extra_fix_ratio * integer_count)));
        const int need = std::min<int>(std::max(0, target_for_round - fixed_in_attempt),
                                       extra_fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int var = extra_fix_candidates[i].index;
            const double incumbent_value = std::round(incumbent_primal(var));
            attempt_lower(var) = incumbent_value;
            attempt_upper(var) = incumbent_value;
            ++fixed_in_attempt;
        }
        if (should_attempt_submip_neighborhood(problem, integer_count, fixed_in_attempt) &&
            solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }
    if (!extra_fix_candidates.empty()) {
        Eigen::VectorXd attempt_lower = lower;
        Eigen::VectorXd attempt_upper = upper;
        const int need =
            std::min<int>(std::max(1, target_fixed - fixed_count), extra_fix_candidates.size());
        for (int i = 0; i < need; ++i) {
            const int var = extra_fix_candidates[i].index;
            double guided = std::round(incumbent_primal(var));
            if (std::abs(lp_relaxation.primal(var) - guided) >
                std::abs(lp_relaxation.primal(var) - std::round(lp_relaxation.primal(var)))) {
                guided = std::round(lp_relaxation.primal(var));
            }
            guided =
                std::min(problem.upper_bounds(var), std::max(problem.lower_bounds(var), guided));
            attempt_lower(var) = guided;
            attempt_upper(var) = guided;
        }
        if (should_attempt_submip_neighborhood(problem, integer_count, fixed_count + need) &&
            solve_attempt(attempt_lower, attempt_upper)) {
            return result;
        }
    }
    return result;
}
NeighborhoodHeuristicResult
run_local_search_heuristic(const Problem& problem, const Options& options,
                           const RelaxationSolution& lp_relaxation,
                           const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
                           const SubproblemSolveCallback& solve_submip) {
    NeighborhoodHeuristicResult result;
    if (!options.use_local_search || !std::isfinite(incumbent_objective) ||
        options.local_search_iterations <= 0 || options.local_search_max_free_vars <= 0) {
        return result;
    }
    const int integer_count = count_integer_variables(problem);
    if (integer_count <= 0) {
        return result;
    }
    const bool expensive_integer_model = integer_count >= 64;
    std::vector<std::pair<int, double>> ranked_integer_vars;
    ranked_integer_vars.reserve(problem.variable_types.size());
    for (int j = 0;
         j < static_cast<int>(problem.variable_types.size()) && j < lp_relaxation.primal.size();
         ++j) {
        if (problem.variable_types[j] == VariableType::Continuous) {
            continue;
        }
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
        std::min<int>(expensive_integer_model ? std::min(options.local_search_max_free_vars, 2)
                                              : options.local_search_max_free_vars,
                      ranked_integer_vars.size());
    if (!should_attempt_submip_neighborhood(problem, integer_count, integer_count - window)) {
        return result;
    }
    const int max_iterations = expensive_integer_model
                                   ? std::min(options.local_search_iterations, 2)
                                   : options.local_search_iterations;
    double best_objective = incumbent_objective;
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        Eigen::VectorXd lower = problem.lower_bounds;
        Eigen::VectorXd upper = problem.upper_bounds;
        std::vector<char> free_mask(problem.variable_types.size(), 0);
        for (int offset = 0; offset < window; ++offset) {
            const int ranked_index = (iteration + offset) % ranked_integer_vars.size();
            free_mask[ranked_integer_vars[ranked_index].first] = 1;
        }
        for (int j = 0;
             j < static_cast<int>(problem.variable_types.size()) && j < incumbent_primal.size();
             ++j) {
            if (problem.variable_types[j] == VariableType::Continuous || free_mask[j]) {
                continue;
            }
            const double fixed_value = std::round(incumbent_primal(j));
            lower(j) = fixed_value;
            upper(j) = fixed_value;
        }
        for (int j = 0;
             j < static_cast<int>(problem.variable_types.size()) && j < incumbent_primal.size();
             ++j) {
            if (!free_mask[j] || problem.variable_types[j] == VariableType::Continuous) {
                continue;
            }
            if (problem.variable_types[j] == VariableType::Binary) {
                continue;
            }
            const double center = std::round(incumbent_primal(j));
            lower(j) = std::max(lower(j), center - 1.0);
            upper(j) = std::min(upper(j), center + 1.0);
        }
        auto incumbent =
            try_integer_subproblem(problem, options, lower, upper, &result, solve_submip);
        if (!incumbent.has_value() ||
            !objective_improves_for_problem(incumbent->objective, best_objective, problem.maximize,
                                            options.integrality_tol)) {
            continue;
        }
        best_objective = incumbent->objective;
        result.incumbent = std::move(incumbent);
        if (expensive_integer_model) {
            break;
        }
    }
    return result;
}
NeighborhoodHeuristicResult
run_local_branching_heuristic(const Problem& problem, const Options& options,
                              const RelaxationSolution& lp_relaxation,
                              const Eigen::VectorXd& incumbent_primal, double incumbent_objective,
                              const SubproblemSolveWithCutsCallback& solve_submip_with_cuts) {
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
        std::round(options.local_branching_fix_agree_ratio * agreeing_integer_indices.size()));
    int fixed_agree = 0;
    for (const int index : agreeing_integer_indices) {
        if (fixed_agree >= max_fix_agree) {
            break;
        }
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
        neighborhood.indices.push_back(index);
        neighborhood.values.push_back(incumbent_value >= 1.0 ? -1.0 : 1.0);
    }
    local_cuts.push_back(std::move(neighborhood));
    Cut improving_cut;
    improving_cut.cut_type = "LocalBranchingObjective";
    improving_cut.sense =
        problem.maximize ? LinearConstraintSense::GreaterEqual : LinearConstraintSense::LessEqual;
    improving_cut.rhs = incumbent_objective - problem.objective_constant +
                        (problem.maximize ? options.integrality_tol : -options.integrality_tol);
    for (int j = 0; j < problem.objective_coefficients.size(); ++j) {
        const double coeff = problem.objective_coefficients(j);
        if (std::abs(coeff) <= 1e-12) {
            continue;
        }
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
        !objective_improves_for_problem(subproblem.objective, incumbent_objective, problem.maximize,
                                        options.integrality_tol)) {
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
} // namespace simplex::bnb::detail
