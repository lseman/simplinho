#include "bnb/conflict_engine.h"
#include "bnb/core.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace simplex::bnb {

bool ConflictEngine::same_progress_value(const Solver& solver, double lhs, double rhs) {
    if (std::isnan(lhs) && std::isnan(rhs))
        return true;
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
        return lhs == rhs;
    }
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= std::max(1e-9, solver.options_.integrality_tol) * scale;
}

std::optional<int>
ConflictEngine::exact_binary_literal_from_conflict_literal(const Solver& solver,
                                                           const ConflictLiteral& literal) {
    if (literal.variable < 0 ||
        literal.variable >= static_cast<int>(solver.problem_.variable_types.size()) ||
        solver.problem_.variable_types[literal.variable] != VariableType::Binary) {
        return std::nullopt;
    }
    if (literal.is_lower && literal.value >= 1.0 - solver.options_.integrality_tol) {
        return detail::ConflictGraph::literal_for(literal.variable, true);
    }
    if (!literal.is_lower && literal.value <= solver.options_.integrality_tol) {
        return detail::ConflictGraph::literal_for(literal.variable, false);
    }
    return std::nullopt;
}

ConflictLiteral ConflictEngine::conflict_literal_from_binary_literal(const Solver& solver,
                                                                     int literal) {
    return ConflictLiteral{
        detail::ConflictGraph::variable_of(literal),
        detail::ConflictGraph::value_of(literal),
        detail::ConflictGraph::value_of(literal) ? 1.0 : 0.0,
    };
}

std::vector<ConflictLiteral>
ConflictEngine::conflict_literals_from_binary_literals(const Solver& solver, int lhs, int rhs) {
    auto make_literal = [&](int literal) {
        return ConflictLiteral{
            detail::ConflictGraph::variable_of(literal),
            detail::ConflictGraph::value_of(literal),
            detail::ConflictGraph::value_of(literal) ? 1.0 : 0.0,
        };
    };
    std::vector<ConflictLiteral> literals = {make_literal(lhs), make_literal(rhs)};
    std::sort(literals.begin(), literals.end(),
              [](const ConflictLiteral& left, const ConflictLiteral& right) {
                  if (left.variable != right.variable)
                      return left.variable < right.variable;
                  if (left.is_lower != right.is_lower)
                      return left.is_lower < right.is_lower;
                  return left.value < right.value;
              });
    return literals;
}

int ConflictEngine::resolve_reason_literal(const Solver& solver, int literal,
                                           const std::vector<PropagationReason>& reasons) {
    if (literal < 0 || literal >= static_cast<int>(reasons.size()))
        return literal;
    int current = literal;
    for (int depth = 0; depth < static_cast<int>(reasons.size()); ++depth) {
        const int parent = reasons[current].parent_literal;
        if (parent < 0 || parent == current || parent >= static_cast<int>(reasons.size())) {
            break;
        }
        current = parent;
    }
    return current;
}

std::vector<ConflictLiteral>
ConflictEngine::minimize_conflict_with_reasons(const Solver& solver,
                                               const std::vector<ConflictLiteral>& literals,
                                               const std::vector<PropagationReason>& reasons) {
    std::vector<ConflictLiteral> minimized;
    minimized.reserve(literals.size());
    for (const ConflictLiteral& literal : literals) {
        if (const std::optional<int> exact =
                exact_binary_literal_from_conflict_literal(solver, literal);
            exact.has_value()) {
            minimized.push_back(conflict_literal_from_binary_literal(
                solver, resolve_reason_literal(solver, *exact, reasons)));
        } else {
            minimized.push_back(literal);
        }
    }
    std::sort(minimized.begin(), minimized.end(),
              [&](const ConflictLiteral& lhs, const ConflictLiteral& rhs) {
                  if (lhs.variable != rhs.variable)
                      return lhs.variable < rhs.variable;
                  if (lhs.is_lower != rhs.is_lower)
                      return lhs.is_lower < rhs.is_lower;
                  return lhs.value < rhs.value;
              });
    minimized.erase(std::unique(minimized.begin(), minimized.end(),
                                [&](const ConflictLiteral& lhs, const ConflictLiteral& rhs) {
                                    return lhs.variable == rhs.variable &&
                                           lhs.is_lower == rhs.is_lower &&
                                           same_progress_value(solver, lhs.value, rhs.value);
                                }),
                    minimized.end());
    return minimized;
}

std::vector<ConflictLiteral>
ConflictEngine::minimize_conflict_with_row_reasons(const Solver& solver,
                                                   const std::vector<ConflictLiteral>& literals,
                                                   const std::vector<PropagationReason>& reasons) {
    struct PendingLiteral {
        ConflictLiteral literal;
        int depth = 0;
    };

    std::vector<PendingLiteral> pending;
    pending.reserve(literals.size());
    for (const ConflictLiteral& literal : literals) {
        pending.push_back(PendingLiteral{literal, 0});
    }

    std::vector<ConflictLiteral> minimized;
    while (!pending.empty()) {
        PendingLiteral current = pending.back();
        pending.pop_back();

        const std::optional<int> exact =
            exact_binary_literal_from_conflict_literal(solver, current.literal);
        if (exact.has_value() && *exact >= 0 && *exact < static_cast<int>(reasons.size()) &&
            reasons[*exact].row_index >= 0 && !reasons[*exact].antecedents.empty() &&
            current.depth < 4) {
            for (const ConflictLiteral& antecedent : reasons[*exact].antecedents) {
                pending.push_back(PendingLiteral{antecedent, current.depth + 1});
            }
            continue;
        }
        minimized.push_back(current.literal);
    }

    return minimize_conflict_with_reasons(solver, minimized, reasons);
}

std::optional<int> ConflictEngine::fixed_binary_literal_from_bounds(const Solver& solver,
                                                                    int variable,
                                                                    const Eigen::VectorXd& lower,
                                                                    const Eigen::VectorXd& upper) {
    if (variable < 0 || variable >= static_cast<int>(solver.problem_.variable_types.size()) ||
        variable >= lower.size() || variable >= upper.size() ||
        solver.problem_.variable_types[variable] != VariableType::Binary) {
        return std::nullopt;
    }
    if (lower(variable) >= 1.0 - solver.options_.integrality_tol) {
        return detail::ConflictGraph::literal_for(variable, true);
    }
    if (upper(variable) <= solver.options_.integrality_tol) {
        return detail::ConflictGraph::literal_for(variable, false);
    }
    return std::nullopt;
}

void ConflictEngine::enqueue_fixed_binary_literal(Solver& solver, int variable,
                                                  const Eigen::VectorXd& lower,
                                                  const Eigen::VectorXd& upper,
                                                  std::vector<char>* seen, std::vector<int>* queue,
                                                  std::shared_ptr<NodeReasonStore>* reasons,
                                                  int parent_literal, bool allow_global) {
    if (seen == nullptr || queue == nullptr)
        return;
    const std::optional<int> literal =
        fixed_binary_literal_from_bounds(solver, variable, lower, upper);
    if (!literal.has_value() || *literal < 0 || *literal >= static_cast<int>(seen->size())) {
        return;
    }
    if (reasons != nullptr && *reasons != nullptr &&
        *literal < static_cast<int>((*reasons)->size())) {
        NodeReasonStore* mutable_reasons = solver.ensure_reason_store_mutable_(reasons);
        if (mutable_reasons == nullptr || *literal >= static_cast<int>(mutable_reasons->size())) {
            return;
        }
        PropagationReason& reason = (*mutable_reasons)[*literal];
        if (reason.parent_literal < 0 && reason.row_index < 0 && reason.antecedents.empty()) {
            reason.parent_literal = parent_literal >= 0 ? parent_literal : *literal;
        }
    }
    if ((*seen)[*literal])
        return;
    (*seen)[*literal] = 1;
    queue->push_back(*literal);
}

std::vector<ConflictLiteral> ConflictEngine::explain_leq_row_conflict(
    const Solver& solver, const std::vector<int>& indices, const std::vector<double>& values,
    double rhs, const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
    double current_activity = 0.0;
    double base_activity = 0.0;
    std::vector<std::pair<ConflictLiteral, double>> deltas;

    for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
         ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= lower.size() || std::abs(coeff) <= 1e-12)
            continue;

        const bool use_lower = coeff >= 0.0;
        const double current_bound = use_lower ? lower(index) : upper(index);
        const double base_bound =
            use_lower ? solver.problem_.lower_bounds(index) : solver.problem_.upper_bounds(index);
        if (!std::isfinite(current_bound) || !std::isfinite(base_bound)) {
            return {};
        }

        const double current_contribution = coeff * current_bound;
        const double base_contribution = coeff * base_bound;
        current_activity += current_contribution;
        base_activity += base_contribution;

        if (solver.problem_.variable_types[index] == VariableType::Continuous)
            continue;
        if (use_lower) {
            if (current_bound <= base_bound + solver.options_.integrality_tol)
                continue;
        } else {
            if (current_bound >= base_bound - solver.options_.integrality_tol)
                continue;
        }

        const double delta = current_contribution - base_contribution;
        if (delta > solver.options_.integrality_tol) {
            deltas.push_back({ConflictLiteral{index, use_lower, current_bound}, delta});
        }
    }

    if (current_activity <= rhs + solver.options_.integrality_tol ||
        base_activity > rhs + solver.options_.integrality_tol || deltas.empty()) {
        return {};
    }

    const double required_delta = rhs - base_activity;
    std::sort(deltas.begin(), deltas.end(),
              [](const auto& lhs, const auto& rhs_item) { return lhs.second < rhs_item.second; });

    double selected_delta = 0.0;
    for (const auto& item : deltas)
        selected_delta += item.second;
    for (auto it = deltas.begin(); it != deltas.end();) {
        if (selected_delta - it->second > required_delta + solver.options_.integrality_tol) {
            selected_delta -= it->second;
            it = deltas.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<ConflictLiteral> literals;
    literals.reserve(deltas.size());
    for (const auto& [literal, _] : deltas)
        literals.push_back(literal);
    return literals;
}

bool ConflictEngine::apply_literal_implications(
    Solver& solver, const detail::ConflictGraph* graph,
    const std::vector<std::vector<ConflictLiteral>>& learned_implications, Eigen::VectorXd* lower,
    Eigen::VectorXd* upper, int* tightened_bounds, std::vector<char>* queued_literals,
    std::vector<int>* literal_queue, int* literal_queue_head, std::vector<int>* changed_variables,
    std::shared_ptr<NodeReasonStore>* reasons, bool allow_global) {
    if (lower == nullptr || upper == nullptr || tightened_bounds == nullptr ||
        queued_literals == nullptr || literal_queue == nullptr || literal_queue_head == nullptr) {
        return true;
    }

    while (*literal_queue_head < static_cast<int>(literal_queue->size())) {
        const int literal = (*literal_queue)[(*literal_queue_head)++];
        auto apply_consequence = [&](const ConflictLiteral& consequence,
                                     std::optional<int> contradiction_literal = std::nullopt) {
            const int variable = consequence.variable;
            if (variable < 0 || variable >= lower->size() ||
                variable >= static_cast<int>(solver.problem_.variable_types.size()) ||
                solver.problem_.variable_types[variable] == VariableType::Continuous) {
                return true;
            }

            double new_lower = (*lower)(variable);
            double new_upper = (*upper)(variable);
            if (consequence.is_lower) {
                new_lower = std::max(new_lower, consequence.value);
            } else {
                new_upper = std::min(new_upper, consequence.value);
            }
            solver.tighten_discrete_bounds_(solver.problem_.variable_types[variable], &new_lower,
                                            &new_upper, solver.options_.integrality_tol);
            if (new_upper + solver.options_.integrality_tol < new_lower) {
                if (contradiction_literal.has_value()) {
                    if (reasons != nullptr && *reasons != nullptr) {
                        learn_reasoned_binary_conflict(solver, literal, *contradiction_literal,
                                                       **reasons, allow_global);
                    } else {
                        learn_conflict_literals(solver,
                                                conflict_literals_from_binary_literals(
                                                    solver, literal, *contradiction_literal),
                                                allow_global);
                    }
                }
                return false;
            }

            bool changed = false;
            if (new_lower > (*lower)(variable) + solver.options_.integrality_tol) {
                (*lower)(variable) = new_lower;
                ++(*tightened_bounds);
                changed = true;
            }
            if (new_upper < (*upper)(variable)-solver.options_.integrality_tol) {
                (*upper)(variable) = new_upper;
                ++(*tightened_bounds);
                changed = true;
            }
            if (!changed)
                return true;

            if (changed_variables != nullptr)
                changed_variables->push_back(variable);
            enqueue_fixed_binary_literal(solver, variable, *lower, *upper, queued_literals,
                                         literal_queue, reasons, literal);
            return true;
        };

        if (graph != nullptr) {
            for (const int conflicting_literal : graph->neighbors(literal)) {
                const ConflictLiteral consequence = conflict_literal_from_binary_literal(
                    solver, detail::ConflictGraph::complement_of(conflicting_literal));
                if (!apply_consequence(consequence, conflicting_literal)) {
                    return false;
                }
            }
        }

        if (literal >= 0 && literal < static_cast<int>(learned_implications.size())) {
            for (const ConflictLiteral& consequence : learned_implications[literal]) {
                std::optional<int> contradiction_literal;
                const std::optional<int> consequence_literal =
                    exact_binary_literal_from_conflict_literal(solver, consequence);
                if (consequence_literal.has_value()) {
                    contradiction_literal =
                        detail::ConflictGraph::complement_of(*consequence_literal);
                }
                if (!apply_consequence(consequence, contradiction_literal)) {
                    return false;
                }
            }
        }
    }

    return true;
}

void ConflictEngine::learn_reasoned_binary_conflict(Solver& solver, int trigger_literal,
                                                    int contradiction_literal,
                                                    const std::vector<PropagationReason>& reasons,
                                                    bool allow_global) {
    const int resolved_trigger = resolve_reason_literal(solver, trigger_literal, reasons);
    const int resolved_contradiction =
        resolve_reason_literal(solver, contradiction_literal, reasons);
    if (resolved_trigger == resolved_contradiction) {
        learn_conflict_literals(
            solver, {conflict_literal_from_binary_literal(solver, resolved_trigger)}, allow_global);
        return;
    }
    learn_conflict_literals(
        solver,
        minimize_conflict_with_row_reasons(solver,
                                           conflict_literals_from_binary_literals(
                                               solver, resolved_trigger, resolved_contradiction),
                                           reasons),
        allow_global);
}

void ConflictEngine::learn_implication_unlocked(Solver& solver, int trigger_literal,
                                                const ConflictLiteral& consequence) {
    if (trigger_literal < 0 ||
        trigger_literal >= static_cast<int>(solver.learned_implications_.size()) ||
        consequence.variable < 0 ||
        consequence.variable >= static_cast<int>(solver.problem_.variable_types.size()) ||
        solver.problem_.variable_types[consequence.variable] == VariableType::Continuous ||
        !std::isfinite(consequence.value)) {
        return;
    }

    std::vector<ConflictLiteral>& implications = solver.learned_implications_[trigger_literal];
    for (ConflictLiteral& existing : implications) {
        if (existing.variable != consequence.variable ||
            existing.is_lower != consequence.is_lower) {
            continue;
        }
        if (consequence.is_lower) {
            if (consequence.value <= existing.value + solver.options_.integrality_tol)
                return;
            existing.value = consequence.value;
            return;
        }
        if (consequence.value >= existing.value - solver.options_.integrality_tol)
            return;
        existing.value = consequence.value;
        return;
    }

    implications.push_back(consequence);
    if (implications.size() > 64) {
        implications.erase(implications.begin());
    }
}

void ConflictEngine::learn_conflict_literals(Solver& solver,
                                             const std::vector<ConflictLiteral>& literals,
                                             bool allow_global) {
    if (literals.empty() || !allow_global)
        return;

    std::lock_guard<std::mutex> lock(solver.learning_mutex_);
    for (const auto& existing : solver.learned_conflicts_) {
        if (existing.literals.size() != literals.size())
            continue;
        bool identical = true;
        for (int i = 0; i < static_cast<int>(literals.size()); ++i) {
            if (existing.literals[i].variable != literals[i].variable ||
                existing.literals[i].is_lower != literals[i].is_lower ||
                !same_progress_value(solver, existing.literals[i].value, literals[i].value)) {
                identical = false;
                break;
            }
        }
        if (identical)
            return;
    }
    LearnedConflict entry;
    entry.literals = literals;
    entry.age = 0;
    entry.hits = 0;
    solver.learned_conflicts_.push_back(std::move(entry));
    const int pool_limit = std::max(1, solver.options_.max_conflict_pool_size);
    if (static_cast<int>(solver.learned_conflicts_.size()) > pool_limit) {
        // Evict the oldest-by-age entries first, breaking ties by insertion
        // order. This matches HiGHS' aging-based conflict pool: stale
        // conflicts that haven't produced violated cuts are dropped in favor
        // of newer, still-useful ones.
        auto victim = std::max_element(solver.learned_conflicts_.begin(),
                                       solver.learned_conflicts_.end(),
                                       [](const LearnedConflict& a, const LearnedConflict& b) {
                                           if (a.age != b.age)
                                               return a.age < b.age;
                                           return a.hits > b.hits;
                                       });
        if (victim != solver.learned_conflicts_.end()) {
            solver.learned_conflicts_.erase(victim);
        } else {
            solver.learned_conflicts_.erase(solver.learned_conflicts_.begin());
        }
    }

    if (literals.size() == 2) {
        const std::optional<int> lhs =
            exact_binary_literal_from_conflict_literal(solver, literals[0]);
        const std::optional<int> rhs =
            exact_binary_literal_from_conflict_literal(solver, literals[1]);
        if (lhs.has_value() && rhs.has_value()) {
            learn_implication_unlocked(solver, *lhs,
                                       conflict_literal_from_binary_literal(
                                           solver, detail::ConflictGraph::complement_of(*rhs)));
            learn_implication_unlocked(solver, *rhs,
                                       conflict_literal_from_binary_literal(
                                           solver, detail::ConflictGraph::complement_of(*lhs)));
        }
    }
}

} // namespace simplex::bnb
