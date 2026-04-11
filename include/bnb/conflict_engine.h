#pragma once

#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "bnb/conflict_graph.h"
#include "bnb/core_types.h"

namespace simplex::bnb {
class Solver;

class ConflictEngine {
  public:
    static bool same_progress_value(const Solver& solver, double lhs, double rhs);
    static std::optional<int>
    exact_binary_literal_from_conflict_literal(const Solver& solver,
                                               const ConflictLiteral& literal);
    static ConflictLiteral conflict_literal_from_binary_literal(const Solver& solver, int literal);
    static std::vector<ConflictLiteral> conflict_literals_from_binary_literals(const Solver& solver,
                                                                               int lhs, int rhs);
    static int resolve_reason_literal(const Solver& solver, int literal,
                                      const std::vector<PropagationReason>& reasons);
    static std::vector<ConflictLiteral>
    minimize_conflict_with_reasons(const Solver& solver,
                                   const std::vector<ConflictLiteral>& literals,
                                   const std::vector<PropagationReason>& reasons);
    static std::vector<ConflictLiteral>
    minimize_conflict_with_row_reasons(const Solver& solver,
                                       const std::vector<ConflictLiteral>& literals,
                                       const std::vector<PropagationReason>& reasons);
    static std::optional<int> fixed_binary_literal_from_bounds(const Solver& solver, int variable,
                                                               const Eigen::VectorXd& lower,
                                                               const Eigen::VectorXd& upper);
    static void enqueue_fixed_binary_literal(Solver& solver, int variable,
                                             const Eigen::VectorXd& lower,
                                             const Eigen::VectorXd& upper, std::vector<char>* seen,
                                             std::vector<int>* queue,
                                             std::shared_ptr<NodeReasonStore>* reasons = nullptr,
                                             int parent_literal = -1, bool allow_global = true);
    static std::vector<ConflictLiteral>
    explain_leq_row_conflict(const Solver& solver, const std::vector<int>& indices,
                             const std::vector<double>& values, double rhs,
                             const Eigen::VectorXd& lower, const Eigen::VectorXd& upper);
    static bool apply_literal_implications(
        Solver& solver, const detail::ConflictGraph* graph,
        const std::vector<std::vector<ConflictLiteral>>& learned_implications,
        Eigen::VectorXd* lower, Eigen::VectorXd* upper, int* tightened_bounds,
        std::vector<char>* queued_literals, std::vector<int>* literal_queue,
        int* literal_queue_head, std::vector<int>* changed_variables,
        std::shared_ptr<NodeReasonStore>* reasons, bool allow_global = true);
    static void learn_reasoned_binary_conflict(Solver& solver, int trigger_literal,
                                               int contradiction_literal,
                                               const std::vector<PropagationReason>& reasons,
                                               bool allow_global = true);
    static void learn_implication_unlocked(Solver& solver, int trigger_literal,
                                           const ConflictLiteral& consequence);
    static void learn_conflict_literals(Solver& solver,
                                        const std::vector<ConflictLiteral>& literals,
                                        bool allow_global = true);
};

} // namespace simplex::bnb
