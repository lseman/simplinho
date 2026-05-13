#pragma once

#include <Eigen/Dense>
#include "../../extern/pdqsort/pdqsort.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/search.h"

namespace simplex::bnb::detail {

inline bool is_effectively_integral(double value, double tol) {
    return std::isfinite(value) && std::abs(value - std::round(value)) <= tol;
}

inline Eigen::VectorXd
project_to_integral_lattice(const Eigen::VectorXd& primal,
                            const std::vector<VariableType>& variable_types) {
    Eigen::VectorXd rounded = primal;
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous)
            continue;
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
    std::shared_ptr<const NodeDomain> domain;
    int domain_change_count = 0;
    bool bounds_presolved = false;
    std::uint64_t presolve_cuts_revision = 0;
    std::uint64_t presolve_conflicts_revision = 0;
    std::uint64_t presolve_implications_revision = 0;
    std::shared_ptr<NodeReasonStore> reasons;
    std::vector<int> changed_variables_hint;
};

inline void materialize_child_state(ChildState* child) {
    if (child == nullptr || has_materialized_bounds(child->lower_bounds, child->upper_bounds)) {
        return;
    }
    materialize_domain_bounds(child->domain, &child->lower_bounds, &child->upper_bounds);
}

inline void prepare_child_state_for_relaxation(ChildState* child) {
    if (child == nullptr) {
        return;
    }

    // Each child now owns its bound vectors eagerly. The only remaining
    // detachment work is making sure inferred reasons are not shared across
    // concurrent evaluations or later presolve updates.
    if (child->reasons != nullptr && child->reasons.use_count() > 1) {
        child->reasons = std::make_shared<NodeReasonStore>(*child->reasons);
    }
    if (child->changed_variables_hint.size() > 1) {
        pdqsort(child->changed_variables_hint.begin(), child->changed_variables_hint.end());
        child->changed_variables_hint.erase(
            std::unique(child->changed_variables_hint.begin(), child->changed_variables_hint.end()),
            child->changed_variables_hint.end());
    }
}

inline std::vector<FractionalCandidate> collect_fractional_candidates(
    const Eigen::VectorXd& primal, const std::vector<VariableType>& variable_types,
    double integrality_tol, std::size_t max_candidates = std::numeric_limits<std::size_t>::max()) {
    std::vector<FractionalCandidate> out;
    out.reserve(variable_types.size());
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous)
            continue;
        const double value = primal(j);
        if (!std::isfinite(value))
            continue;
        const double floor_value = std::floor(value);
        const double ceil_value = std::ceil(value);
        const double down_distance = value - floor_value;
        const double up_distance = ceil_value - value;
        const double fractionality = std::min(down_distance, up_distance);
        if (fractionality > integrality_tol) {
            out.push_back(FractionalCandidate{j, value, fractionality, down_distance, up_distance});
        }
    }

    auto cmp = [](const auto& lhs, const auto& rhs) {
        if (std::abs(lhs.fractionality - rhs.fractionality) > 1e-12) {
            return lhs.fractionality > rhs.fractionality;
        }
        return lhs.variable < rhs.variable;
    };

    if (max_candidates < out.size()) {
        std::nth_element(out.begin(), out.begin() + max_candidates, out.end(), cmp);
        out.resize(max_candidates);
        pdqsort(out.begin(), out.end(), cmp);
    } else {
        pdqsort(out.begin(), out.end(), cmp);
    }
    return out;
}

inline ChildState make_child_state(const ActiveNode& node, int variable, bool branch_up,
                                   double value) {
    ChildState child;
    child.domain_change_count = node.domain_change_count;
    child.domain = node.domain;
    child.reasons = node.reasons;
    if (has_materialized_bounds(node.lower_bounds, node.upper_bounds)) {
        child.lower_bounds = node.lower_bounds;
        child.upper_bounds = node.upper_bounds;
    }

    const bool has_node_bounds = has_materialized_bounds(node.lower_bounds, node.upper_bounds);
    double base_lower = std::numeric_limits<double>::quiet_NaN();
    double base_upper = std::numeric_limits<double>::quiet_NaN();
    if (has_node_bounds) {
        if (variable >= 0 && variable < node.lower_bounds.size() &&
            variable < node.upper_bounds.size()) {
            base_lower = node.lower_bounds(variable);
            base_upper = node.upper_bounds(variable);
        }
    } else {
        const auto [resolved_lower, resolved_upper] =
            resolve_domain_variable_bounds(node.domain, variable);
        base_lower = resolved_lower;
        base_upper = resolved_upper;
    }

    if (variable >= 0 && std::isfinite(base_lower) && std::isfinite(base_upper)) {
        bool bound_tightened = false;
        double new_lower = base_lower;
        double new_upper = base_upper;
        if (branch_up) {
            const double tightened_value = std::max(base_lower, std::ceil(value));
            if (tightened_value > base_lower + 1e-12) {
                ++child.domain_change_count;
                bound_tightened = true;
            }
            new_lower = tightened_value;
            if (has_node_bounds) {
                child.lower_bounds(variable) = tightened_value;
            }
        } else {
            const double tightened_value = std::min(base_upper, std::floor(value));
            if (tightened_value < base_upper - 1e-12) {
                ++child.domain_change_count;
                bound_tightened = true;
            }
            new_upper = tightened_value;
            if (has_node_bounds) {
                child.upper_bounds(variable) = tightened_value;
            }
        }
        if (bound_tightened) {
            child.changed_variables_hint.push_back(variable);
            auto domain = std::make_shared<NodeDomain>();
            domain->parent = node.domain;
            domain->variable_count =
                node.domain != nullptr ? node.domain->variable_count : node.lower_bounds.size();
            domain->changes.push_back(DomainChange{variable, new_lower, new_upper});
            domain->chain_depth = node.domain != nullptr ? node.domain->chain_depth + 1 : 1;
            domain->total_change_count =
                (node.domain != nullptr ? node.domain->total_change_count : 0) + 1;
            child.domain = std::move(domain);
        }
    }
    return child;
}

inline ChildState make_upper_zero_child_state(const ActiveNode& node,
                                              const std::vector<int>& variables) {
    ChildState child;
    child.domain_change_count = node.domain_change_count;
    child.domain = node.domain;
    child.reasons = node.reasons;
    if (has_materialized_bounds(node.lower_bounds, node.upper_bounds)) {
        child.lower_bounds = node.lower_bounds;
        child.upper_bounds = node.upper_bounds;
    }

    const bool has_node_bounds = has_materialized_bounds(node.lower_bounds, node.upper_bounds);
    auto domain = std::make_shared<NodeDomain>();
    domain->parent = node.domain;
    int variable_count = node.domain != nullptr ? node.domain->variable_count
                                                : static_cast<int>(node.lower_bounds.size());
    for (const int variable : variables) {
        variable_count = std::max(variable_count, variable + 1);
    }
    domain->variable_count = variable_count;
    domain->chain_depth = node.domain != nullptr ? node.domain->chain_depth + 1 : 1;
    domain->total_change_count = node.domain != nullptr ? node.domain->total_change_count : 0;

    for (const int variable : variables) {
        double base_lower = std::numeric_limits<double>::quiet_NaN();
        double base_upper = std::numeric_limits<double>::quiet_NaN();
        if (has_node_bounds) {
            if (variable >= 0 && variable < node.lower_bounds.size() &&
                variable < node.upper_bounds.size()) {
                base_lower = node.lower_bounds(variable);
                base_upper = node.upper_bounds(variable);
            }
        } else {
            const auto [resolved_lower, resolved_upper] =
                resolve_domain_variable_bounds(node.domain, variable);
            base_lower = resolved_lower;
            base_upper = resolved_upper;
        }
        if (variable < 0 || !std::isfinite(base_lower) || !std::isfinite(base_upper) ||
            base_upper <= 1e-12) {
            continue;
        }
        if (has_node_bounds) {
            child.upper_bounds(variable) = 0.0;
        }
        child.changed_variables_hint.push_back(variable);
        domain->changes.push_back(DomainChange{variable, base_lower, 0.0});
        ++child.domain_change_count;
        ++domain->total_change_count;
    }

    if (!domain->changes.empty()) {
        child.domain = std::move(domain);
    }
    return child;
}

inline ChildState make_fixed_child_state(const ActiveNode& node, int variable, bool branch_up,
                                         double value) {
    ChildState child;
    child.domain_change_count = node.domain_change_count;
    child.domain = node.domain;
    child.reasons = node.reasons;
    if (has_materialized_bounds(node.lower_bounds, node.upper_bounds)) {
        child.lower_bounds = node.lower_bounds;
        child.upper_bounds = node.upper_bounds;
    }

    const bool has_node_bounds = has_materialized_bounds(node.lower_bounds, node.upper_bounds);
    double base_lower = std::numeric_limits<double>::quiet_NaN();
    double base_upper = std::numeric_limits<double>::quiet_NaN();
    if (has_node_bounds) {
        if (variable >= 0 && variable < node.lower_bounds.size() &&
            variable < node.upper_bounds.size()) {
            base_lower = node.lower_bounds(variable);
            base_upper = node.upper_bounds(variable);
        }
    } else {
        const auto [resolved_lower, resolved_upper] =
            resolve_domain_variable_bounds(node.domain, variable);
        base_lower = resolved_lower;
        base_upper = resolved_upper;
    }

    const double fixed_value = branch_up ? std::ceil(value) : std::floor(value);
    if (variable >= 0 && std::isfinite(base_lower) && std::isfinite(base_upper)) {
        bool tightened = false;
        if (std::abs(base_lower - fixed_value) > 1e-12) {
            ++child.domain_change_count;
            tightened = true;
        }
        if (std::abs(base_upper - fixed_value) > 1e-12) {
            ++child.domain_change_count;
            tightened = true;
        }
        if (tightened) {
            if (has_node_bounds) {
                child.lower_bounds(variable) = fixed_value;
                child.upper_bounds(variable) = fixed_value;
            }
            child.changed_variables_hint.push_back(variable);
            auto domain = std::make_shared<NodeDomain>();
            domain->parent = node.domain;
            domain->variable_count =
                node.domain != nullptr ? node.domain->variable_count : child.lower_bounds.size();
            domain->changes.push_back(DomainChange{variable, fixed_value, fixed_value});
            domain->chain_depth = node.domain != nullptr ? node.domain->chain_depth + 1 : 1;
            domain->total_change_count =
                (node.domain != nullptr ? node.domain->total_change_count : 0) + 1;
            child.domain = std::move(domain);
        }
    }
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

inline bool objective_improves_for_problem(double candidate, double incumbent, bool maximize,
                                           double tol) {
    return maximize ? (candidate > incumbent + tol) : (candidate < incumbent - tol);
}

inline bool is_integer_feasible_solution(const Eigen::VectorXd& primal,
                                         const std::vector<VariableType>& variable_types,
                                         double tol) {
    for (int j = 0; j < primal.size() && j < static_cast<int>(variable_types.size()); ++j) {
        if (variable_types[j] == VariableType::Continuous)
            continue;
        if (!is_effectively_integral(primal(j), tol)) {
            return false;
        }
    }
    return true;
}

inline bool is_sos_feasible_solution(const Eigen::VectorXd& primal,
                                     const std::vector<SOSConstraint>& sos_constraints,
                                     double tol) {
    for (const SOSConstraint& sos : sos_constraints) {
        std::vector<int> active_positions;
        for (int pos = 0; pos < static_cast<int>(sos.variables.size()); ++pos) {
            const int variable = sos.variables[pos];
            if (variable >= 0 && variable < primal.size() && std::abs(primal(variable)) > tol) {
                active_positions.push_back(pos);
            }
        }
        if (sos.type == SOSType::SOS1) {
            if (active_positions.size() > 1) {
                return false;
            }
        } else if (active_positions.size() > 2) {
            return false;
        } else if (active_positions.size() == 2 && active_positions[1] != active_positions[0] + 1) {
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

struct RelaxationSolveContext {
    bool strong_branching_probe = false;
    int max_lp_iterations = 0;
    bool isolate_lp_state = false;
};

inline thread_local const RelaxationSolveContext* current_relaxation_solve_context_ = nullptr;

inline const RelaxationSolveContext* current_relaxation_solve_context() {
    return current_relaxation_solve_context_;
}

class ScopedRelaxationSolveContext {
  public:
    explicit ScopedRelaxationSolveContext(const RelaxationSolveContext& context)
        : previous_(current_relaxation_solve_context_) {
        current_relaxation_solve_context_ = &context;
    }

    ~ScopedRelaxationSolveContext() { current_relaxation_solve_context_ = previous_; }

    ScopedRelaxationSolveContext(const ScopedRelaxationSolveContext&) = delete;
    ScopedRelaxationSolveContext& operator=(const ScopedRelaxationSolveContext&) = delete;

  private:
    const RelaxationSolveContext* previous_ = nullptr;
};

using RelaxationSolveCallback =
    std::function<RelaxationSolution(const ChildState&, const LPBasis*)>;

DivingHeuristicResult run_diving_heuristic(const ActiveNode& start_node,
                                           const RelaxationSolution& start_relaxation,
                                           const Problem& problem, const Options& options,
                                           const Eigen::VectorXd* incumbent_primal,
                                           std::vector<DivingStrategyStats>& strategy_stats,
                                           const RelaxationSolveCallback& relaxation_solver);

} // namespace simplex::bnb::detail
