#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bnb/branching.h"
#include "bnb/cuts.h"
#include "bnb/heuristic.h"
#include "bnb/parallel.h"

namespace simplex::bnb {

class Solver {
public:
  explicit Solver(Problem problem, Options options = {},
                  std::vector<Cut> initial_cuts = {})
      : problem_(std::move(problem)), options_(std::move(options)),
        initial_cuts_(std::move(initial_cuts)) {
    validate_inputs_();
    pseudocosts_.resize(problem_.variable_types.size());
  }

  template <typename RelaxationSolver>
  SolveResult solve(RelaxationSolver &&relaxation_solver) {
    reset_state_();
    parallel_task_dispatcher_ =
        options_.parallel_workers > 1
            ? std::make_unique<detail::ParallelDispatcher>(
                  options_.parallel_workers)
            : nullptr;

    auto solve_submip_with_cuts =
        [&](const Eigen::VectorXd &lower_bounds,
            const Eigen::VectorXd &upper_bounds,
            const std::vector<Cut> &initial_cuts) -> SolveResult {
      Problem subproblem = problem_;
      subproblem.lower_bounds = lower_bounds;
      subproblem.upper_bounds = upper_bounds;

      Options sub_options = options_;
      sub_options.max_nodes = options_.heuristic_subproblem_max_nodes;
      sub_options.node_selection = NodeSelectionStrategy::DepthFirst;
      sub_options.branching_strategy = BranchingStrategy::MostFractional;
      sub_options.diving_strategy = options_.diving_strategy;
      sub_options.verbose = false;
      sub_options.strong_branching_candidates = 0;
      sub_options.strong_branching_max_depth = 0;
      sub_options.heuristic_frequency = std::numeric_limits<int>::max();
      sub_options.heuristic_max_depth = 0;
      sub_options.use_feasibility_jump = false;
      sub_options.use_feasibility_pump = false;
      sub_options.use_rens = false;
      sub_options.use_rins = false;
      sub_options.use_local_search = false;
      sub_options.use_local_branching = false;
      sub_options.use_cut_pool = options_.use_cut_pool;
      sub_options.max_cut_rounds_per_node = options_.max_cut_rounds_per_node;
      sub_options.max_cuts_added_per_round =
          std::min(4, options_.max_cuts_added_per_round);

      Solver subsolver(std::move(subproblem), sub_options, initial_cuts);
      return subsolver.solve(relaxation_solver);
    };

    auto solve_submip =
        [&](const Eigen::VectorXd &lower_bounds,
            const Eigen::VectorXd &upper_bounds) -> SolveResult {
      return solve_submip_with_cuts(lower_bounds, upper_bounds, {});
    };

    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      detail::push_active_node(active_nodes_, make_root_node_(),
                               options_.node_selection, problem_.maximize);
    }

    auto process_node = [&](detail::ActiveNode node, bool allow_root_cuts) {
      process_active_node_(std::move(node), allow_root_cuts, relaxation_solver,
                           solve_submip, solve_submip_with_cuts);
    };

    if (std::optional<detail::ActiveNode> root = pop_next_active_node_();
        root.has_value()) {
      process_node(std::move(*root), true);
    }

    if (!should_terminate_()) {
      if (options_.parallel_workers > 1) {
        process_parallel_active_nodes_(process_node);
      } else {
        while (true) {
          if (should_terminate_())
            break;
          std::optional<detail::ActiveNode> next_node = pop_next_active_node_();
          if (!next_node.has_value())
            break;
          process_node(std::move(*next_node), false);
        }
      }
    }

    Status final_status = Status::NodeLimit;
    if (found_unbounded_) {
      final_status = Status::Unbounded;
    } else if (has_incumbent_ && active_nodes_.empty() && !hit_node_limit_) {
      final_status = Status::Optimal;
    } else if (!has_incumbent_ && active_nodes_.empty() && !hit_node_limit_) {
      final_status = Status::Infeasible;
    }
    if (options_.verbose) {
      std::lock_guard<std::mutex> lock(state_mutex_);
      log_progress_unlocked_("done", true, &final_status);
    }
    return finalize_result_(final_status);
  }

private:
  using ConflictLiteral = detail::ReasonLiteral;
  using PropagationReason = detail::PropagationReason;
  using NodeReasonStore = detail::NodeReasonStore;

  struct LearnedConflict {
    std::vector<ConflictLiteral> literals;
  };

  struct NodePresolveOutcome {
    bool infeasible = false;
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    int tightened_bounds = 0;
    std::shared_ptr<NodeReasonStore> reasons;
  };

  struct IncumbentSnapshot {
    bool has_incumbent = false;
    double objective = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd primal;
  };

  struct HeuristicSchedule {
    bool run_diving = false;
    bool run_feasibility_jump = false;
    bool run_feasibility_pump = false;
    bool run_rens = false;
    bool run_rins = false;
    bool run_local_search = false;
    bool run_local_branching = false;
  };

  void validate_inputs_() const {
    const int n = static_cast<int>(problem_.lower_bounds.size());
    if (n <= 0) {
      throw std::invalid_argument(
          "simplex::bnb: problem must contain at least one variable");
    }
    if (problem_.upper_bounds.size() != n ||
        static_cast<int>(problem_.variable_types.size()) != n) {
      throw std::invalid_argument("simplex::bnb: lower/upper bounds and "
                                  "variable types must have matching sizes");
    }
    if (problem_.objective_coefficients.size() != 0 &&
        problem_.objective_coefficients.size() != n) {
      throw std::invalid_argument("simplex::bnb: objective_coefficients must "
                                  "be empty or match variable count");
    }
    if (options_.max_nodes <= 0) {
      throw std::invalid_argument("simplex::bnb: max_nodes must be >= 1");
    }
    if (options_.integrality_tol < 0.0) {
      throw std::invalid_argument("simplex::bnb: integrality_tol must be >= 0");
    }
    if (options_.heuristic_subproblem_max_nodes <= 0) {
      throw std::invalid_argument(
          "simplex::bnb: heuristic_subproblem_max_nodes must be >= 1");
    }
    if (options_.feasibility_jump_iterations < 0) {
      throw std::invalid_argument(
          "simplex::bnb: feasibility_jump_iterations must be >= 0");
    }
    if (options_.feasibility_jump_max_free_vars < 0) {
      throw std::invalid_argument(
          "simplex::bnb: feasibility_jump_max_free_vars must be >= 0");
    }
    if (options_.feasibility_jump_objective_weight < 0.0) {
      throw std::invalid_argument(
          "simplex::bnb: feasibility_jump_objective_weight must be >= 0");
    }
    if (options_.probing_max_candidates < 0) {
      throw std::invalid_argument(
          "simplex::bnb: probing_max_candidates must be >= 0");
    }
    if (options_.max_conflict_cuts_per_round < 0) {
      throw std::invalid_argument(
          "simplex::bnb: max_conflict_cuts_per_round must be >= 0");
    }
    if (options_.max_cuts_per_type < 0) {
      throw std::invalid_argument(
          "simplex::bnb: max_cuts_per_type must be >= 0");
    }
    if (options_.cut_max_parallelism < 0.0 ||
        options_.cut_max_parallelism > 1.0) {
      throw std::invalid_argument(
          "simplex::bnb: cut_max_parallelism must be in [0, 1]");
    }
  }

  void reset_state_() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    active_nodes_.clear();
    cut_pool_ = detail::CutPool(options_);
    active_cuts_ = initial_cuts_;
    active_cut_signatures_.clear();
    learned_conflicts_.clear();
    learned_implications_.assign(2 * problem_.variable_types.size(), {});
    for (const auto &cut : active_cuts_) {
      active_cut_signatures_.insert(detail::cut_signature(cut));
    }
    tree_nodes_.clear();
    node_count_ = 0;
    lp_iterations_ = 0;
    incumbent_updates_ = 0;
    heuristic_lp_iterations_ = 0;
    heuristic_successes_ = 0;
    feasibility_jump_successes_ = 0;
    feasibility_pump_successes_ = 0;
    rens_successes_ = 0;
    rins_successes_ = 0;
    local_search_successes_ = 0;
    local_branching_successes_ = 0;
    diving_stats_.assign(5, detail::DivingStrategyStats{});
    next_order_ = 0;
    hybrid_counter_ = 0;
    has_incumbent_ = false;
    hit_node_limit_ = false;
    found_unbounded_ = false;
    active_workers_ = 0;
    incumbent_objective_ = problem_.maximize
                               ? -std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::infinity();
    incumbent_primal_ = Eigen::VectorXd::Constant(
        problem_.lower_bounds.size(), std::numeric_limits<double>::quiet_NaN());
    root_relaxation_objective.reset();
    progress_header_printed_ = false;
    last_logged_node_count_ = 0;
    last_logged_best_bound_ = std::numeric_limits<double>::quiet_NaN();
    last_logged_incumbent_ = incumbent_objective_;
    last_logged_gap_ = std::numeric_limits<double>::quiet_NaN();
  }

  detail::ActiveNode make_root_node_() {
    const int id = detail::append_tree_node(tree_nodes_, -1, 0, next_order_);
    detail::ActiveNode node;
    node.id = id;
    node.parent_id = -1;
    node.depth = 0;
    node.order = next_order_++;
    node.lower_bounds = problem_.lower_bounds;
    node.upper_bounds = problem_.upper_bounds;
    return node;
  }

  bool objective_improves_(double candidate, double incumbent) const {
    return problem_.maximize
               ? (candidate > incumbent + options_.integrality_tol)
               : (candidate < incumbent - options_.integrality_tol);
  }

  bool bound_prunes_(double candidate, double incumbent) const {
    return problem_.maximize
               ? (candidate <= incumbent + options_.integrality_tol)
               : (candidate >= incumbent - options_.integrality_tol);
  }

  std::vector<Cut> current_relaxation_cuts_unlocked_() const {
    std::vector<Cut> cuts = active_cuts_;
    if (!has_incumbent_ || !std::isfinite(incumbent_objective_) ||
        problem_.objective_coefficients.size() == 0) {
      return cuts;
    }

    Cut cutoff;
    cutoff.cut_type = "IncumbentCutoff";
    cutoff.sense = problem_.maximize ? LinearConstraintSense::GreaterEqual
                                     : LinearConstraintSense::LessEqual;
    cutoff.rhs = incumbent_objective_ - problem_.objective_constant +
                 (problem_.maximize ? options_.integrality_tol
                                    : -options_.integrality_tol);

    for (int j = 0; j < problem_.objective_coefficients.size(); ++j) {
      const double coeff = problem_.objective_coefficients(j);
      if (std::abs(coeff) <= 1e-12)
        continue;
      cutoff.indices.push_back(j);
      cutoff.values.push_back(coeff);
    }
    if (!cutoff.indices.empty()) {
      cuts.push_back(std::move(cutoff));
    }
    return cuts;
  }

  std::vector<Cut> current_relaxation_cuts_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_relaxation_cuts_unlocked_();
  }

  IncumbentSnapshot incumbent_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    IncumbentSnapshot snapshot;
    snapshot.has_incumbent = has_incumbent_;
    snapshot.objective = incumbent_objective_;
    snapshot.primal = incumbent_primal_;
    return snapshot;
  }

  std::vector<detail::PseudoCost> pseudocosts_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return pseudocosts_;
  }

  std::vector<LearnedConflict> learned_conflicts_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return learned_conflicts_;
  }

  std::vector<std::vector<ConflictLiteral>>
  learned_implications_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return learned_implications_;
  }

  const detail::ConflictGraph *conflict_graph_() const {
    std::call_once(conflict_graph_once_, [&]() {
      const bool has_binary = std::any_of(
          problem_.variable_types.begin(), problem_.variable_types.end(),
          [](VariableType type) { return type == VariableType::Binary; });
      if (has_binary) {
        conflict_graph_cache_ =
            std::make_unique<detail::ConflictGraph>(problem_);
      }
    });
    return conflict_graph_cache_.get();
  }

  std::vector<detail::DivingStrategyStats> diving_stats_snapshot_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return diving_stats_;
  }

  static void tighten_discrete_bounds_(VariableType type, double *lower,
                                       double *upper, double tol) {
    if (type == VariableType::Continuous)
      return;
    *lower = std::ceil(*lower - tol);
    *upper = std::floor(*upper + tol);
    if (type == VariableType::Binary) {
      *lower = std::max(0.0, *lower);
      *upper = std::min(1.0, *upper);
    }
  }

  std::optional<int>
  fixed_binary_literal_from_bounds_(int variable, const Eigen::VectorXd &lower,
                                    const Eigen::VectorXd &upper) const {
    if (variable < 0 ||
        variable >= static_cast<int>(problem_.variable_types.size()) ||
        variable >= lower.size() || variable >= upper.size() ||
        problem_.variable_types[variable] != VariableType::Binary) {
      return std::nullopt;
    }
    if (lower(variable) >= 1.0 - options_.integrality_tol) {
      return detail::ConflictGraph::literal_for(variable, true);
    }
    if (upper(variable) <= options_.integrality_tol) {
      return detail::ConflictGraph::literal_for(variable, false);
    }
    return std::nullopt;
  }

  NodeReasonStore *
  ensure_reason_store_mutable_(std::shared_ptr<NodeReasonStore> *reasons,
                               int required_size = 0) const {
    if (reasons == nullptr)
      return nullptr;
    if (*reasons == nullptr) {
      *reasons = std::make_shared<NodeReasonStore>(required_size);
      return reasons->get();
    }
    if (required_size > 0 &&
        static_cast<int>((*reasons)->size()) != required_size) {
      *reasons = std::make_shared<NodeReasonStore>(required_size);
      return reasons->get();
    }
    if (!(*reasons).unique()) {
      *reasons = std::make_shared<NodeReasonStore>(**reasons);
    }
    return reasons->get();
  }

  void
  seed_fixed_literal_reason_(int literal,
                             std::shared_ptr<NodeReasonStore> *reasons) const {
    if (reasons == nullptr || *reasons == nullptr || literal < 0 ||
        literal >= static_cast<int>((*reasons)->size())) {
      return;
    }
    NodeReasonStore *mutable_reasons = ensure_reason_store_mutable_(reasons);
    if (mutable_reasons == nullptr ||
        literal >= static_cast<int>(mutable_reasons->size())) {
      return;
    }
    PropagationReason &reason = (*mutable_reasons)[literal];
    reason.parent_literal = literal;
    reason.row_index = -1;
    reason.antecedents.clear();
  }

  void enqueue_fixed_binary_literal_(
      int variable, const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
      std::vector<char> *seen, std::vector<int> *queue,
      std::shared_ptr<NodeReasonStore> *reasons = nullptr,
      int parent_literal = -1) const {
    if (seen == nullptr || queue == nullptr)
      return;
    const std::optional<int> literal =
        fixed_binary_literal_from_bounds_(variable, lower, upper);
    if (!literal.has_value() || *literal < 0 ||
        *literal >= static_cast<int>(seen->size())) {
      return;
    }
    if (reasons != nullptr && *reasons != nullptr &&
        *literal < static_cast<int>((*reasons)->size())) {
      NodeReasonStore *mutable_reasons = ensure_reason_store_mutable_(reasons);
      if (mutable_reasons == nullptr ||
          *literal >= static_cast<int>(mutable_reasons->size())) {
        return;
      }
      PropagationReason &reason = (*mutable_reasons)[*literal];
      if (reason.parent_literal < 0 && reason.row_index < 0 &&
          reason.antecedents.empty()) {
        reason.parent_literal = parent_literal >= 0 ? parent_literal : *literal;
      }
    }
    if ((*seen)[*literal])
      return;
    (*seen)[*literal] = 1;
    queue->push_back(*literal);
  }

  std::vector<ConflictLiteral>
  conflict_literals_from_binary_literals_(int lhs, int rhs) const {
    auto make_literal = [](int literal) {
      return ConflictLiteral{
          detail::ConflictGraph::variable_of(literal),
          detail::ConflictGraph::value_of(literal),
          detail::ConflictGraph::value_of(literal) ? 1.0 : 0.0,
      };
    };
    std::vector<ConflictLiteral> literals = {make_literal(lhs),
                                             make_literal(rhs)};
    std::sort(literals.begin(), literals.end(),
              [](const ConflictLiteral &left, const ConflictLiteral &right) {
                if (left.variable != right.variable)
                  return left.variable < right.variable;
                if (left.is_lower != right.is_lower)
                  return left.is_lower < right.is_lower;
                return left.value < right.value;
              });
    return literals;
  }

  ConflictLiteral conflict_literal_from_binary_literal_(int literal) const {
    return ConflictLiteral{
        detail::ConflictGraph::variable_of(literal),
        detail::ConflictGraph::value_of(literal),
        detail::ConflictGraph::value_of(literal) ? 1.0 : 0.0,
    };
  }

  std::optional<int> exact_binary_literal_from_conflict_literal_(
      const ConflictLiteral &literal) const {
    if (literal.variable < 0 ||
        literal.variable >= static_cast<int>(problem_.variable_types.size()) ||
        problem_.variable_types[literal.variable] != VariableType::Binary) {
      return std::nullopt;
    }
    if (literal.is_lower && literal.value >= 1.0 - options_.integrality_tol) {
      return detail::ConflictGraph::literal_for(literal.variable, true);
    }
    if (!literal.is_lower && literal.value <= options_.integrality_tol) {
      return detail::ConflictGraph::literal_for(literal.variable, false);
    }
    return std::nullopt;
  }

  int resolve_reason_literal_(
      int literal, const std::vector<PropagationReason> &reasons) const {
    if (literal < 0 || literal >= static_cast<int>(reasons.size()))
      return literal;
    int current = literal;
    for (int depth = 0; depth < static_cast<int>(reasons.size()); ++depth) {
      const int parent = reasons[current].parent_literal;
      if (parent < 0 || parent == current ||
          parent >= static_cast<int>(reasons.size())) {
        break;
      }
      current = parent;
    }
    return current;
  }

  std::vector<ConflictLiteral> minimize_conflict_with_reasons_(
      const std::vector<ConflictLiteral> &literals,
      const std::vector<PropagationReason> &reasons) const {
    std::vector<ConflictLiteral> minimized;
    minimized.reserve(literals.size());
    for (const ConflictLiteral &literal : literals) {
      if (const std::optional<int> exact =
              exact_binary_literal_from_conflict_literal_(literal);
          exact.has_value()) {
        minimized.push_back(conflict_literal_from_binary_literal_(
            resolve_reason_literal_(*exact, reasons)));
      } else {
        minimized.push_back(literal);
      }
    }
    std::sort(minimized.begin(), minimized.end(),
              [](const ConflictLiteral &lhs, const ConflictLiteral &rhs) {
                if (lhs.variable != rhs.variable)
                  return lhs.variable < rhs.variable;
                if (lhs.is_lower != rhs.is_lower)
                  return lhs.is_lower < rhs.is_lower;
                return lhs.value < rhs.value;
              });
    minimized.erase(std::unique(minimized.begin(), minimized.end(),
                                [&](const ConflictLiteral &lhs,
                                    const ConflictLiteral &rhs) {
                                  return lhs.variable == rhs.variable &&
                                         lhs.is_lower == rhs.is_lower &&
                                         same_progress_value_(lhs.value,
                                                              rhs.value);
                                }),
                    minimized.end());
    return minimized;
  }

  std::vector<ConflictLiteral> explain_row_fixing_literal_(
      const std::vector<int> &indices, const std::vector<double> &values,
      double rhs, LinearConstraintSense sense, int literal,
      const Eigen::VectorXd &lower, const Eigen::VectorXd &upper) const {
    const int variable = detail::ConflictGraph::variable_of(literal);
    if (variable < 0 || variable >= lower.size() || variable >= upper.size())
      return {};

    Eigen::VectorXd probe_lower = lower;
    Eigen::VectorXd probe_upper = upper;
    if (detail::ConflictGraph::value_of(literal)) {
      probe_lower(variable) = 0.0;
      probe_upper(variable) = 0.0;
    } else {
      probe_lower(variable) = 1.0;
      probe_upper(variable) = 1.0;
    }

    int tightened_bounds = 0;
    std::vector<ConflictLiteral> conflict_literals;
    const bool feasible = propagate_row_bounds_(
        indices, values, rhs, sense, &probe_lower, &probe_upper,
        &tightened_bounds, nullptr, &conflict_literals);
    if (feasible || conflict_literals.empty())
      return {};

    conflict_literals.erase(
        std::remove_if(conflict_literals.begin(), conflict_literals.end(),
                       [&](const ConflictLiteral &candidate) {
                         return candidate.variable == variable &&
                                candidate.is_lower ==
                                    detail::ConflictGraph::value_of(literal);
                       }),
        conflict_literals.end());
    return conflict_literals;
  }

  std::vector<ConflictLiteral> minimize_conflict_with_row_reasons_(
      const std::vector<ConflictLiteral> &literals,
      const std::vector<PropagationReason> &reasons) const {
    struct PendingLiteral {
      ConflictLiteral literal;
      int depth = 0;
    };

    std::vector<PendingLiteral> pending;
    pending.reserve(literals.size());
    for (const ConflictLiteral &literal : literals) {
      pending.push_back(PendingLiteral{literal, 0});
    }

    std::vector<ConflictLiteral> minimized;
    while (!pending.empty()) {
      PendingLiteral current = pending.back();
      pending.pop_back();

      const std::optional<int> exact =
          exact_binary_literal_from_conflict_literal_(current.literal);
      if (exact.has_value() && *exact >= 0 &&
          *exact < static_cast<int>(reasons.size()) &&
          reasons[*exact].row_index >= 0 &&
          !reasons[*exact].antecedents.empty() && current.depth < 4) {
        for (const ConflictLiteral &antecedent : reasons[*exact].antecedents) {
          pending.push_back(PendingLiteral{antecedent, current.depth + 1});
        }
        continue;
      }

      minimized.push_back(current.literal);
    }

    return minimize_conflict_with_reasons_(minimized, reasons);
  }

  void learn_reasoned_binary_conflict_(
      int trigger_literal, int contradiction_literal,
      const std::vector<PropagationReason> &reasons) {
    const int resolved_trigger =
        resolve_reason_literal_(trigger_literal, reasons);
    const int resolved_contradiction =
        resolve_reason_literal_(contradiction_literal, reasons);
    if (resolved_trigger == resolved_contradiction) {
      learn_conflict_literals_(
          {conflict_literal_from_binary_literal_(resolved_trigger)});
      return;
    }
    learn_conflict_literals_(minimize_conflict_with_row_reasons_(
        conflict_literals_from_binary_literals_(resolved_trigger,
                                                resolved_contradiction),
        reasons));
  }

  void learn_implication_unlocked_(int trigger_literal,
                                   const ConflictLiteral &consequence) {
    if (trigger_literal < 0 ||
        trigger_literal >= static_cast<int>(learned_implications_.size()) ||
        consequence.variable < 0 ||
        consequence.variable >=
            static_cast<int>(problem_.variable_types.size()) ||
        problem_.variable_types[consequence.variable] ==
            VariableType::Continuous ||
        !std::isfinite(consequence.value)) {
      return;
    }

    std::vector<ConflictLiteral> &implications =
        learned_implications_[trigger_literal];
    for (ConflictLiteral &existing : implications) {
      if (existing.variable != consequence.variable ||
          existing.is_lower != consequence.is_lower) {
        continue;
      }
      if (consequence.is_lower) {
        if (consequence.value <= existing.value + options_.integrality_tol)
          return;
        existing.value = consequence.value;
        return;
      }
      if (consequence.value >= existing.value - options_.integrality_tol)
        return;
      existing.value = consequence.value;
      return;
    }

    implications.push_back(consequence);
    if (implications.size() > 64) {
      implications.erase(implications.begin());
    }
  }

  bool apply_literal_implications_(
      const detail::ConflictGraph *graph,
      const std::vector<std::vector<ConflictLiteral>> &learned_implications,
      Eigen::VectorXd *lower, Eigen::VectorXd *upper, int *tightened_bounds,
      std::vector<char> *queued_literals, std::vector<int> *literal_queue,
      int *literal_queue_head, std::vector<int> *changed_variables,
      std::shared_ptr<NodeReasonStore> *reasons) {
    if (lower == nullptr || upper == nullptr || tightened_bounds == nullptr ||
        queued_literals == nullptr || literal_queue == nullptr ||
        literal_queue_head == nullptr) {
      return true;
    }

    while (*literal_queue_head < static_cast<int>(literal_queue->size())) {
      const int literal = (*literal_queue)[(*literal_queue_head)++];
      auto apply_consequence = [&](const ConflictLiteral &consequence,
                                   std::optional<int> contradiction_literal =
                                       std::nullopt) {
        const int variable = consequence.variable;
        if (variable < 0 || variable >= lower->size() ||
            variable >= static_cast<int>(problem_.variable_types.size()) ||
            problem_.variable_types[variable] == VariableType::Continuous) {
          return true;
        }

        double new_lower = (*lower)(variable);
        double new_upper = (*upper)(variable);
        if (consequence.is_lower) {
          new_lower = std::max(new_lower, consequence.value);
        } else {
          new_upper = std::min(new_upper, consequence.value);
        }
        tighten_discrete_bounds_(problem_.variable_types[variable], &new_lower,
                                 &new_upper, options_.integrality_tol);
        if (new_upper + options_.integrality_tol < new_lower) {
          if (contradiction_literal.has_value()) {
            if (reasons != nullptr && *reasons != nullptr) {
              learn_reasoned_binary_conflict_(literal, *contradiction_literal,
                                              **reasons);
            } else {
              learn_conflict_literals_(conflict_literals_from_binary_literals_(
                  literal, *contradiction_literal));
            }
          }
          return false;
        }

        bool changed = false;
        if (new_lower > (*lower)(variable) + options_.integrality_tol) {
          (*lower)(variable) = new_lower;
          ++(*tightened_bounds);
          changed = true;
        }
        if (new_upper < (*upper)(variable)-options_.integrality_tol) {
          (*upper)(variable) = new_upper;
          ++(*tightened_bounds);
          changed = true;
        }
        if (!changed)
          return true;

        if (changed_variables != nullptr)
          changed_variables->push_back(variable);
        enqueue_fixed_binary_literal_(variable, *lower, *upper, queued_literals,
                                      literal_queue, reasons, literal);
        return true;
      };

      if (graph != nullptr) {
        for (const int conflicting_literal : graph->neighbors(literal)) {
          const ConflictLiteral consequence =
              conflict_literal_from_binary_literal_(
                  detail::ConflictGraph::complement_of(conflicting_literal));
          if (!apply_consequence(consequence, conflicting_literal)) {
            return false;
          }
        }
      }

      if (literal >= 0 &&
          literal < static_cast<int>(learned_implications.size())) {
        for (const ConflictLiteral &consequence :
             learned_implications[literal]) {
          std::optional<int> contradiction_literal;
          const std::optional<int> consequence_literal =
              exact_binary_literal_from_conflict_literal_(consequence);
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

  std::vector<ConflictLiteral>
  explain_leq_row_conflict_(const std::vector<int> &indices,
                            const std::vector<double> &values, double rhs,
                            const Eigen::VectorXd &lower,
                            const Eigen::VectorXd &upper) const {
    double current_activity = 0.0;
    double base_activity = 0.0;
    std::vector<std::pair<ConflictLiteral, double>> deltas;

    for (int k = 0; k < static_cast<int>(indices.size()) &&
                    k < static_cast<int>(values.size());
         ++k) {
      const int index = indices[k];
      const double coeff = values[k];
      if (index < 0 || index >= lower.size() || std::abs(coeff) <= 1e-12)
        continue;

      const bool use_lower = coeff >= 0.0;
      const double current_bound = use_lower ? lower(index) : upper(index);
      const double base_bound = use_lower ? problem_.lower_bounds(index)
                                          : problem_.upper_bounds(index);
      if (!std::isfinite(current_bound) || !std::isfinite(base_bound)) {
        return {};
      }

      const double current_contribution = coeff * current_bound;
      const double base_contribution = coeff * base_bound;
      current_activity += current_contribution;
      base_activity += base_contribution;

      if (problem_.variable_types[index] == VariableType::Continuous)
        continue;
      if (use_lower) {
        if (current_bound <= base_bound + options_.integrality_tol)
          continue;
      } else {
        if (current_bound >= base_bound - options_.integrality_tol)
          continue;
      }

      const double delta = current_contribution - base_contribution;
      if (delta > options_.integrality_tol) {
        deltas.push_back(
            {ConflictLiteral{index, use_lower, current_bound}, delta});
      }
    }

    if (current_activity <= rhs + options_.integrality_tol ||
        base_activity > rhs + options_.integrality_tol || deltas.empty()) {
      return {};
    }

    const double required_delta = rhs - base_activity;
    std::sort(deltas.begin(), deltas.end(),
              [](const auto &lhs, const auto &rhs_item) {
                return lhs.second < rhs_item.second;
              });

    double selected_delta = 0.0;
    for (const auto &item : deltas)
      selected_delta += item.second;
    for (auto it = deltas.begin(); it != deltas.end();) {
      if (selected_delta - it->second >
          required_delta + options_.integrality_tol) {
        selected_delta -= it->second;
        it = deltas.erase(it);
      } else {
        ++it;
      }
    }

    std::vector<ConflictLiteral> literals;
    literals.reserve(deltas.size());
    for (const auto &[literal, _] : deltas)
      literals.push_back(literal);
    return literals;
  }

  void learn_conflict_literals_(const std::vector<ConflictLiteral> &literals) {
    if (literals.empty())
      return;

    std::lock_guard<std::mutex> lock(state_mutex_);
    for (const auto &existing : learned_conflicts_) {
      if (existing.literals.size() != literals.size())
        continue;
      bool identical = true;
      for (int i = 0; i < static_cast<int>(literals.size()); ++i) {
        if (existing.literals[i].variable != literals[i].variable ||
            existing.literals[i].is_lower != literals[i].is_lower ||
            !same_progress_value_(existing.literals[i].value,
                                  literals[i].value)) {
          identical = false;
          break;
        }
      }
      if (identical)
        return;
    }
    learned_conflicts_.push_back(LearnedConflict{literals});
    if (learned_conflicts_.size() > 256) {
      learned_conflicts_.erase(learned_conflicts_.begin());
    }

    if (literals.size() == 2) {
      const std::optional<int> lhs =
          exact_binary_literal_from_conflict_literal_(literals[0]);
      const std::optional<int> rhs =
          exact_binary_literal_from_conflict_literal_(literals[1]);
      if (lhs.has_value() && rhs.has_value()) {
        learn_implication_unlocked_(
            *lhs, conflict_literal_from_binary_literal_(
                      detail::ConflictGraph::complement_of(*rhs)));
        learn_implication_unlocked_(
            *rhs, conflict_literal_from_binary_literal_(
                      detail::ConflictGraph::complement_of(*lhs)));
      }
    }
  }

  void learn_implication_(int trigger_literal,
                          const ConflictLiteral &consequence) {
    if (trigger_literal < 0)
      return;
    std::lock_guard<std::mutex> lock(state_mutex_);
    learn_implication_unlocked_(trigger_literal, consequence);
  }

  void learn_probing_implications_(int trigger_literal,
                                   const Eigen::VectorXd &base_lower,
                                   const Eigen::VectorXd &base_upper,
                                   const NodePresolveOutcome &presolved) {
    if (trigger_literal < 0 || presolved.infeasible)
      return;
    for (int j = 0;
         j < static_cast<int>(problem_.variable_types.size()) &&
         j < base_lower.size() && j < base_upper.size() &&
         j < presolved.lower_bounds.size() && j < presolved.upper_bounds.size();
         ++j) {
      if (problem_.variable_types[j] == VariableType::Continuous)
        continue;
      if (base_lower(j) + options_.integrality_tol <
          presolved.lower_bounds(j)) {
        learn_implication_(trigger_literal,
                           ConflictLiteral{j, true, presolved.lower_bounds(j)});
      }
      if (base_upper(j) - options_.integrality_tol >
          presolved.upper_bounds(j)) {
        learn_implication_(
            trigger_literal,
            ConflictLiteral{j, false, presolved.upper_bounds(j)});
      }
    }
  }

  bool apply_leq_row_propagation_(
      const std::vector<int> &indices, const std::vector<double> &values,
      double rhs, Eigen::VectorXd *lower, Eigen::VectorXd *upper,
      int *tightened_bounds, std::vector<int> *changed_variables = nullptr,
      std::vector<ConflictLiteral> *conflict_literals = nullptr) const {
    if (!lower || !upper || !tightened_bounds)
      return true;

    auto accumulate_activity = [&](bool use_upper_for_negative) {
      double activity = 0.0;
      bool finite = true;
      for (int k = 0; k < static_cast<int>(indices.size()) &&
                      k < static_cast<int>(values.size());
           ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12)
          continue;

        const double bound =
            coeff >= 0.0
                ? (*lower)(index)
                : (use_upper_for_negative ? (*upper)(index) : (*lower)(index));
        if (!std::isfinite(bound)) {
          finite = false;
          break;
        }
        activity += coeff * bound;
      }
      return std::pair<double, bool>{activity, finite};
    };

    const auto [row_min, row_min_finite] = accumulate_activity(true);
    if (row_min_finite && row_min > rhs + options_.integrality_tol) {
      if (conflict_literals != nullptr) {
        *conflict_literals =
            explain_leq_row_conflict_(indices, values, rhs, *lower, *upper);
      }
      return false;
    }

    for (int pivot = 0; pivot < static_cast<int>(indices.size()) &&
                        pivot < static_cast<int>(values.size());
         ++pivot) {
      const int index = indices[pivot];
      const double coeff = values[pivot];
      if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12)
        continue;

      double other_min = 0.0;
      bool other_min_finite = true;
      for (int k = 0; k < static_cast<int>(indices.size()) &&
                      k < static_cast<int>(values.size());
           ++k) {
        if (k == pivot)
          continue;
        const int other_index = indices[k];
        const double other_coeff = values[k];
        if (other_index < 0 || other_index >= lower->size() ||
            std::abs(other_coeff) <= 1e-12) {
          continue;
        }

        const double bound =
            other_coeff >= 0.0 ? (*lower)(other_index) : (*upper)(other_index);
        if (!std::isfinite(bound)) {
          other_min_finite = false;
          break;
        }
        other_min += other_coeff * bound;
      }
      if (!other_min_finite)
        continue;

      double new_lower = (*lower)(index);
      double new_upper = (*upper)(index);
      if (coeff > 0.0) {
        const double candidate = (rhs - other_min) / coeff;
        if (std::isfinite(candidate) &&
            candidate < new_upper - options_.integrality_tol) {
          new_upper = candidate;
        }
      } else {
        const double candidate = (rhs - other_min) / coeff;
        if (std::isfinite(candidate) &&
            candidate > new_lower + options_.integrality_tol) {
          new_lower = candidate;
        }
      }

      tighten_discrete_bounds_(problem_.variable_types[index], &new_lower,
                               &new_upper, options_.integrality_tol);
      if (new_upper + options_.integrality_tol < new_lower) {
        if (conflict_literals != nullptr) {
          *conflict_literals =
              explain_leq_row_conflict_(indices, values, rhs, *lower, *upper);
        }
        return false;
      }
      if (new_lower > (*lower)(index) + options_.integrality_tol) {
        (*lower)(index) = new_lower;
        ++(*tightened_bounds);
        if (changed_variables != nullptr)
          changed_variables->push_back(index);
      }
      if (new_upper < (*upper)(index)-options_.integrality_tol) {
        (*upper)(index) = new_upper;
        ++(*tightened_bounds);
        if (changed_variables != nullptr)
          changed_variables->push_back(index);
      }
    }

    return true;
  }

  bool propagate_row_bounds_(
      const std::vector<int> &indices, const std::vector<double> &values,
      double rhs, LinearConstraintSense sense, Eigen::VectorXd *lower,
      Eigen::VectorXd *upper, int *tightened_bounds,
      std::vector<int> *changed_variables = nullptr,
      std::vector<ConflictLiteral> *conflict_literals = nullptr) const {
    if (sense == LinearConstraintSense::LessEqual) {
      return apply_leq_row_propagation_(indices, values, rhs, lower, upper,
                                        tightened_bounds, changed_variables,
                                        conflict_literals);
    }
    if (sense == LinearConstraintSense::GreaterEqual) {
      std::vector<double> negated(values.size(), 0.0);
      for (int k = 0; k < static_cast<int>(values.size()); ++k) {
        negated[k] = -values[k];
      }
      return apply_leq_row_propagation_(indices, negated, -rhs, lower, upper,
                                        tightened_bounds, changed_variables,
                                        conflict_literals);
    }

    std::vector<double> negated(values.size(), 0.0);
    for (int k = 0; k < static_cast<int>(values.size()); ++k) {
      negated[k] = -values[k];
    }
    return apply_leq_row_propagation_(indices, values, rhs, lower, upper,
                                      tightened_bounds, changed_variables,
                                      conflict_literals) &&
           apply_leq_row_propagation_(indices, negated, -rhs, lower, upper,
                                      tightened_bounds, changed_variables,
                                      conflict_literals);
  }

  bool conflict_applies_(const LearnedConflict &conflict,
                         const Eigen::VectorXd &lower,
                         const Eigen::VectorXd &upper) const {
    for (const ConflictLiteral &literal : conflict.literals) {
      if (literal.variable < 0 || literal.variable >= lower.size())
        return false;
      if (literal.is_lower) {
        if (lower(literal.variable) + options_.integrality_tol <
            literal.value) {
          return false;
        }
      } else {
        if (upper(literal.variable) - options_.integrality_tol >
            literal.value) {
          return false;
        }
      }
    }
    return true;
  }

  std::vector<ConflictLiteral>
  conflict_literals_from_bounds_(const Eigen::VectorXd &lower,
                                 const Eigen::VectorXd &upper) const {
    std::vector<ConflictLiteral> literals;
    literals.reserve(problem_.variable_types.size());
    for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) &&
                    j < lower.size() && j < upper.size();
         ++j) {
      if (problem_.variable_types[j] == VariableType::Continuous)
        continue;

      double lower_value = lower(j);
      double upper_value = upper(j);
      tighten_discrete_bounds_(problem_.variable_types[j], &lower_value,
                               &upper_value, options_.integrality_tol);

      if (lower_value > problem_.lower_bounds(j) + options_.integrality_tol) {
        literals.push_back(ConflictLiteral{j, true, lower_value});
      }
      if (upper_value < problem_.upper_bounds(j) - options_.integrality_tol) {
        literals.push_back(ConflictLiteral{j, false, upper_value});
      }
    }

    std::sort(literals.begin(), literals.end(),
              [](const ConflictLiteral &lhs, const ConflictLiteral &rhs) {
                if (lhs.variable != rhs.variable)
                  return lhs.variable < rhs.variable;
                if (lhs.is_lower != rhs.is_lower)
                  return lhs.is_lower < rhs.is_lower;
                return lhs.value < rhs.value;
              });
    literals.erase(std::unique(literals.begin(), literals.end(),
                               [&](const ConflictLiteral &lhs,
                                   const ConflictLiteral &rhs) {
                                 return lhs.variable == rhs.variable &&
                                        lhs.is_lower == rhs.is_lower &&
                                        same_progress_value_(lhs.value,
                                                             rhs.value);
                               }),
                   literals.end());
    if (literals.size() > 16) {
      literals.clear();
    }
    return literals;
  }

  void maybe_learn_conflict_from_bounds_(const Eigen::VectorXd &lower,
                                         const Eigen::VectorXd &upper) {
    const std::vector<ConflictLiteral> literals =
        conflict_literals_from_bounds_(lower, upper);
    learn_conflict_literals_(literals);
  }

  NodePresolveOutcome presolve_node_bounds_(
      const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
      const std::vector<Cut> &cuts,
      const std::shared_ptr<NodeReasonStore> &initial_reasons = nullptr) {
    struct RowRef {
      const std::vector<int> *indices = nullptr;
      const std::vector<double> *values = nullptr;
      double rhs = 0.0;
      LinearConstraintSense sense = LinearConstraintSense::LessEqual;
    };

    NodePresolveOutcome out;
    out.lower_bounds = lower;
    out.upper_bounds = upper;

    for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) &&
                    j < out.lower_bounds.size() && j < out.upper_bounds.size();
         ++j) {
      tighten_discrete_bounds_(problem_.variable_types[j], &out.lower_bounds(j),
                               &out.upper_bounds(j), options_.integrality_tol);
      if (out.upper_bounds(j) + options_.integrality_tol <
          out.lower_bounds(j)) {
        out.infeasible = true;
        return out;
      }
    }

    std::vector<RowRef> rows;
    rows.reserve(problem_.base_constraints.size() + cuts.size());
    std::vector<std::vector<int>> column_to_rows(out.lower_bounds.size());

    auto register_row = [&](const std::vector<int> &indices,
                            const std::vector<double> &values, double rhs,
                            LinearConstraintSense sense) {
      const int row_index = static_cast<int>(rows.size());
      rows.push_back(RowRef{&indices, &values, rhs, sense});
      for (int k = 0; k < static_cast<int>(indices.size()) &&
                      k < static_cast<int>(values.size());
           ++k) {
        const int index = indices[k];
        if (index < 0 || index >= static_cast<int>(column_to_rows.size()) ||
            std::abs(values[k]) <= 1e-12) {
          continue;
        }
        column_to_rows[index].push_back(row_index);
      }
    };

    for (const SparseLinearConstraint &row : problem_.base_constraints) {
      register_row(row.indices, row.values, row.rhs, row.sense);
    }
    for (const Cut &cut : cuts) {
      register_row(cut.indices, cut.values, cut.rhs, cut.sense);
    }

    const detail::ConflictGraph *graph = conflict_graph_();
    const std::vector<std::vector<ConflictLiteral>> learned_implications =
        learned_implications_snapshot_();
    std::vector<char> queued_literals;
    std::vector<int> literal_queue;
    std::shared_ptr<NodeReasonStore> reasons;
    int literal_queue_head = 0;
    if (graph != nullptr) {
      queued_literals.assign(graph->literal_count(), 0);
      if (initial_reasons != nullptr &&
          static_cast<int>(initial_reasons->size()) == graph->literal_count()) {
        reasons = initial_reasons;
      } else {
        reasons = std::make_shared<NodeReasonStore>(graph->literal_count());
      }
      literal_queue.reserve(std::max(4, graph->literal_count() / 4));
      for (int j = 0;
           j < static_cast<int>(problem_.variable_types.size()) &&
           j < out.lower_bounds.size() && j < out.upper_bounds.size();
           ++j) {
        enqueue_fixed_binary_literal_(j, out.lower_bounds, out.upper_bounds,
                                      &queued_literals, &literal_queue,
                                      &reasons);
      }
    }

    const std::vector<LearnedConflict> learned_conflicts =
        learned_conflicts_snapshot_();
    for (const LearnedConflict &conflict : learned_conflicts) {
      if (conflict_applies_(conflict, out.lower_bounds, out.upper_bounds)) {
        out.infeasible = true;
        return out;
      }
    }

    if (graph != nullptr) {
      std::vector<int> graph_changed;
      if (!apply_literal_implications_(
              graph, learned_implications, &out.lower_bounds, &out.upper_bounds,
              &out.tightened_bounds, &queued_literals, &literal_queue,
              &literal_queue_head, &graph_changed, &reasons)) {
        out.infeasible = true;
        return out;
      }
    }

    std::vector<char> row_queued(rows.size(), 1);
    std::vector<int> row_queue(rows.size(), 0);
    int queue_head = 0;
    for (int row_index = 0; row_index < static_cast<int>(rows.size());
         ++row_index) {
      row_queue[row_index] = row_index;
    }

    while (queue_head < static_cast<int>(row_queue.size())) {
      const int row_index = row_queue[queue_head++];
      row_queued[row_index] = 0;
      const RowRef &row = rows[row_index];
      std::vector<int> changed_variables;
      std::vector<ConflictLiteral> row_conflict;
      if (!propagate_row_bounds_(*row.indices, *row.values, row.rhs, row.sense,
                                 &out.lower_bounds, &out.upper_bounds,
                                 &out.tightened_bounds, &changed_variables,
                                 &row_conflict)) {
        out.infeasible = true;
        learn_conflict_literals_(minimize_conflict_with_row_reasons_(
            row_conflict, reasons != nullptr ? *reasons : NodeReasonStore{}));
        return out;
      }

      for (const int index : changed_variables) {
        if (index < 0 || index >= static_cast<int>(column_to_rows.size()))
          continue;
        for (const int affected_row : column_to_rows[index]) {
          if (row_queued[affected_row])
            continue;
          row_queued[affected_row] = 1;
          row_queue.push_back(affected_row);
        }
        if (graph != nullptr) {
          enqueue_fixed_binary_literal_(index, out.lower_bounds,
                                        out.upper_bounds, &queued_literals,
                                        &literal_queue, &reasons);
          const std::optional<int> fixed_literal =
              fixed_binary_literal_from_bounds_(index, out.lower_bounds,
                                                out.upper_bounds);
          if (fixed_literal.has_value() && reasons != nullptr &&
              *fixed_literal >= 0 &&
              *fixed_literal < static_cast<int>(reasons->size())) {
            NodeReasonStore *mutable_reasons =
                ensure_reason_store_mutable_(&reasons, graph->literal_count());
            if (mutable_reasons != nullptr &&
                (*mutable_reasons)[*fixed_literal].row_index < 0 &&
                (*mutable_reasons)[*fixed_literal].antecedents.empty()) {
              (*mutable_reasons)[*fixed_literal].row_index = row_index;
              (*mutable_reasons)[*fixed_literal].antecedents =
                  explain_row_fixing_literal_(
                      *row.indices, *row.values, row.rhs, row.sense,
                      *fixed_literal, out.lower_bounds, out.upper_bounds);
            }
          }
        }
      }

      if (graph != nullptr) {
        std::vector<int> graph_changed;
        if (!apply_literal_implications_(
                graph, learned_implications, &out.lower_bounds,
                &out.upper_bounds, &out.tightened_bounds, &queued_literals,
                &literal_queue, &literal_queue_head, &graph_changed,
                &reasons)) {
          out.infeasible = true;
          return out;
        }
        for (const int index : graph_changed) {
          if (index < 0 || index >= static_cast<int>(column_to_rows.size()))
            continue;
          for (const int affected_row : column_to_rows[index]) {
            if (row_queued[affected_row])
              continue;
            row_queued[affected_row] = 1;
            row_queue.push_back(affected_row);
          }
        }
        changed_variables.insert(changed_variables.end(), graph_changed.begin(),
                                 graph_changed.end());
      }

      if (!changed_variables.empty()) {
        for (const LearnedConflict &conflict : learned_conflicts) {
          if (conflict_applies_(conflict, out.lower_bounds, out.upper_bounds)) {
            out.infeasible = true;
            return out;
          }
        }
      }
    }

    if (graph != nullptr) {
      out.reasons = reasons;
    }
    return out;
  }

  double relative_gap_from_bound_(double bound) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!has_incumbent_ || !std::isfinite(incumbent_objective_) ||
        !std::isfinite(bound)) {
      return std::numeric_limits<double>::infinity();
    }
    const double raw_gap = problem_.maximize ? (bound - incumbent_objective_)
                                             : (incumbent_objective_ - bound);
    if (raw_gap <= options_.integrality_tol) {
      return 0.0;
    }
    const double scale = std::max(1.0, std::abs(incumbent_objective_));
    return raw_gap / scale;
  }

  std::vector<Cut> generate_probing_implied_bound_cuts_(
      const detail::ActiveNode &node, const RelaxationSolution &relaxation,
      const std::vector<detail::FractionalCandidate> &fractional,
      const std::vector<Cut> &relaxation_cuts) {
    std::vector<Cut> cuts;
    if (!options_.use_probing_implications ||
        options_.probing_max_candidates == 0 ||
        relaxation.status != RelaxationStatus::Optimal || fractional.empty()) {
      return cuts;
    }

    const bool shallow = node.depth <= 2;
    const bool periodic = options_.heuristic_frequency <= 1 ||
                          (options_.heuristic_frequency > 0 &&
                           (node.order % static_cast<std::uint64_t>(
                                             options_.heuristic_frequency) ==
                            0));
    if (!shallow && !periodic) {
      return cuts;
    }

    std::unordered_set<std::string> signatures;
    int probes = 0;
    for (const auto &candidate : fractional) {
      if (probes >= options_.probing_max_candidates)
        break;
      if (candidate.variable < 0 ||
          candidate.variable >=
              static_cast<int>(problem_.variable_types.size()) ||
          problem_.variable_types[candidate.variable] != VariableType::Binary) {
        continue;
      }
      ++probes;

      for (int fixed_value : {0, 1}) {
        Eigen::VectorXd lower = node.lower_bounds;
        Eigen::VectorXd upper = node.upper_bounds;
        lower(candidate.variable) = static_cast<double>(fixed_value);
        upper(candidate.variable) = static_cast<double>(fixed_value);

        const NodePresolveOutcome presolved =
            presolve_node_bounds_(lower, upper, relaxation_cuts, node.reasons);
        if (presolved.infeasible) {
          Cut cut;
          cut.cut_type = "ProbeFix";
          cut.indices = {candidate.variable};
          cut.values = {1.0};
          cut.sense = fixed_value == 1 ? LinearConstraintSense::LessEqual
                                       : LinearConstraintSense::GreaterEqual;
          cut.rhs = fixed_value == 1 ? 0.0 : 1.0;
          const double violation =
              detail::cut_violation(cut, relaxation.primal);
          if (violation > options_.min_cut_violation) {
            cut.strength = violation;
            const std::string signature = detail::cut_signature(cut);
            if (!signatures.contains(signature)) {
              signatures.insert(signature);
              cuts.push_back(std::move(cut));
            }
          }
          continue;
        }

        learn_probing_implications_(detail::ConflictGraph::literal_for(
                                        candidate.variable, fixed_value == 1),
                                    node.lower_bounds, node.upper_bounds,
                                    presolved);

        for (int j = 0; j < problem_.lower_bounds.size(); ++j) {
          if (j == candidate.variable)
            continue;

          const double base_upper = node.upper_bounds(j);
          const double tightened_upper = presolved.upper_bounds(j);
          if (std::isfinite(base_upper) &&
              tightened_upper < base_upper - options_.integrality_tol) {
            Cut cut;
            cut.cut_type = "ProbeImpliedBound";
            cut.indices = {j, candidate.variable};
            cut.values = {1.0, fixed_value == 1
                                   ? (base_upper - tightened_upper)
                                   : -(base_upper - tightened_upper)};
            cut.sense = LinearConstraintSense::LessEqual;
            cut.rhs = fixed_value == 1 ? base_upper : tightened_upper;
            const double violation =
                detail::cut_violation(cut, relaxation.primal);
            if (violation > options_.min_cut_violation) {
              cut.strength = violation;
              const std::string signature = detail::cut_signature(cut);
              if (!signatures.contains(signature)) {
                signatures.insert(signature);
                cuts.push_back(std::move(cut));
              }
            }
          }

          const double base_lower = node.lower_bounds(j);
          const double tightened_lower = presolved.lower_bounds(j);
          if (std::isfinite(base_lower) &&
              tightened_lower > base_lower + options_.integrality_tol) {
            Cut cut;
            cut.cut_type = "ProbeImpliedBound";
            cut.indices = {j, candidate.variable};
            cut.values = {1.0, fixed_value == 1
                                   ? -(tightened_lower - base_lower)
                                   : (tightened_lower - base_lower)};
            cut.sense = LinearConstraintSense::GreaterEqual;
            cut.rhs = fixed_value == 1 ? base_lower : tightened_lower;
            const double violation =
                detail::cut_violation(cut, relaxation.primal);
            if (violation > options_.min_cut_violation) {
              cut.strength = violation;
              const std::string signature = detail::cut_signature(cut);
              if (!signatures.contains(signature)) {
                signatures.insert(signature);
                cuts.push_back(std::move(cut));
              }
            }
          }
        }
      }
    }

    return cuts;
  }

  std::vector<Cut>
  generate_conflict_cuts_(const RelaxationSolution &relaxation) const {
    std::vector<Cut> cuts;
    if (!options_.use_conflict_cuts ||
        options_.max_conflict_cuts_per_round == 0 ||
        relaxation.status != RelaxationStatus::Optimal) {
      return cuts;
    }

    const detail::ConflictGraph *graph = conflict_graph_();
    const std::vector<LearnedConflict> conflicts =
        learned_conflicts_snapshot_();
    std::unordered_set<std::string> signatures;
    for (const LearnedConflict &conflict : conflicts) {
      if (static_cast<int>(cuts.size()) >= options_.max_conflict_cuts_per_round)
        break;
      if (conflict.literals.size() < 2 || conflict.literals.size() > 12)
        continue;

      Cut cut;
      cut.sense = LinearConstraintSense::LessEqual;
      cut.cut_type = "Conflict";
      int zero_literals = 0;
      bool valid = true;
      std::vector<int> base_literals;
      for (const ConflictLiteral &literal : conflict.literals) {
        if (literal.variable < 0 ||
            literal.variable >=
                static_cast<int>(problem_.variable_types.size()) ||
            problem_.variable_types[literal.variable] != VariableType::Binary) {
          valid = false;
          break;
        }

        if (literal.is_lower &&
            literal.value >= 1.0 - options_.integrality_tol) {
          cut.indices.push_back(literal.variable);
          cut.values.push_back(1.0);
          base_literals.push_back(
              detail::ConflictGraph::literal_for(literal.variable, true));
        } else if (!literal.is_lower &&
                   literal.value <= options_.integrality_tol) {
          cut.indices.push_back(literal.variable);
          cut.values.push_back(-1.0);
          ++zero_literals;
          base_literals.push_back(
              detail::ConflictGraph::literal_for(literal.variable, false));
        } else {
          valid = false;
          break;
        }
      }
      if (!valid || cut.indices.empty())
        continue;

      if (graph != nullptr && base_literals.size() >= 2) {
        bool clique_conflict = true;
        for (int i = 0;
             i < static_cast<int>(base_literals.size()) && clique_conflict;
             ++i) {
          for (int j = i + 1; j < static_cast<int>(base_literals.size()); ++j) {
            if (!graph->are_conflicting(base_literals[i], base_literals[j])) {
              clique_conflict = false;
              break;
            }
          }
        }

        if (clique_conflict) {
          std::vector<int> lifted_literals = base_literals;
          std::vector<int> candidates = graph->neighbors(base_literals.front());
          for (int i = 1; i < static_cast<int>(base_literals.size()) &&
                          !candidates.empty();
               ++i) {
            const std::vector<int> neighbors =
                graph->neighbors(base_literals[i]);
            std::vector<int> intersection;
            intersection.reserve(std::min(candidates.size(), neighbors.size()));
            std::set_intersection(candidates.begin(), candidates.end(),
                                  neighbors.begin(), neighbors.end(),
                                  std::back_inserter(intersection));
            candidates = std::move(intersection);
          }

          std::unordered_set<int> used_variables;
          for (const int literal : lifted_literals) {
            used_variables.insert(detail::ConflictGraph::variable_of(literal));
          }
          candidates.erase(
              std::remove_if(
                  candidates.begin(), candidates.end(),
                  [&](int literal) {
                    return std::find(lifted_literals.begin(),
                                     lifted_literals.end(),
                                     literal) != lifted_literals.end() ||
                           used_variables.contains(
                               detail::ConflictGraph::variable_of(literal));
                  }),
              candidates.end());
          std::sort(
              candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
                const double lhs_weight = detail::ConflictGraph::literal_weight(
                    relaxation.primal, lhs);
                const double rhs_weight = detail::ConflictGraph::literal_weight(
                    relaxation.primal, rhs);
                if (std::abs(lhs_weight - rhs_weight) > 1e-12) {
                  return lhs_weight > rhs_weight;
                }
                return graph->degree(lhs) > graph->degree(rhs);
              });
          for (const int literal : candidates) {
            bool compatible = true;
            for (const int chosen : lifted_literals) {
              if (!graph->are_conflicting(literal, chosen)) {
                compatible = false;
                break;
              }
            }
            if (!compatible)
              continue;
            lifted_literals.push_back(literal);
          }

          Cut lifted_cut = detail::clique_cut_from_literals(
              problem_, lifted_literals, options_, "ConflictClique");
          if (!lifted_cut.indices.empty()) {
            const double violation =
                detail::cut_violation(lifted_cut, relaxation.primal);
            if (violation > options_.min_cut_violation) {
              lifted_cut.strength = violation;
              const std::string signature = detail::cut_signature(lifted_cut);
              if (!signatures.contains(signature)) {
                signatures.insert(signature);
                cuts.push_back(std::move(lifted_cut));
                continue;
              }
            }
          }
        }
      }

      cut.rhs = static_cast<double>(cut.indices.size() - 1 - zero_literals);
      if (!detail::canonicalize_cut(&cut, options_.min_cut_violation * 1e-3))
        continue;
      const double violation = detail::cut_violation(cut, relaxation.primal);
      if (violation <= options_.min_cut_violation)
        continue;
      cut.strength = violation;
      const std::string signature = detail::cut_signature(cut);
      if (signatures.contains(signature))
        continue;
      signatures.insert(signature);
      cuts.push_back(std::move(cut));
    }

    return cuts;
  }

  std::vector<Cut> generate_cut_candidates_(
      const detail::ActiveNode &node, const RelaxationSolution &relaxation,
      const std::vector<detail::FractionalCandidate> &fractional,
      const std::vector<Cut> &relaxation_cuts) {
    std::vector<Cut> cuts =
        detail::generate_cuts(problem_, relaxation, options_);
    std::vector<Cut> probing_cuts = generate_probing_implied_bound_cuts_(
        node, relaxation, fractional, relaxation_cuts);
    cuts.insert(cuts.end(), std::make_move_iterator(probing_cuts.begin()),
                std::make_move_iterator(probing_cuts.end()));
    std::vector<Cut> conflict_cuts = generate_conflict_cuts_(relaxation);
    cuts.insert(cuts.end(), std::make_move_iterator(conflict_cuts.begin()),
                std::make_move_iterator(conflict_cuts.end()));
    return cuts;
  }

  bool should_try_node_cuts_(
      const detail::ActiveNode &node, const RelaxationSolution &relaxation,
      const std::vector<detail::FractionalCandidate> &fractional) const {
    if (!options_.use_cut_pool || node.depth <= 0 ||
        relaxation.status != RelaxationStatus::Optimal) {
      return false;
    }
    if ((!options_.use_gomory_cuts && !options_.use_cover_cuts &&
         !options_.use_implied_bound_cuts && !options_.use_clique_cuts &&
         !options_.use_probing_implications && !options_.use_conflict_cuts) ||
        fractional.empty()) {
      return false;
    }
    if (node.depth > std::max(2, options_.strong_branching_max_depth + 1)) {
      return false;
    }

    const bool shallow = node.depth <= 2;
    const bool periodic = options_.heuristic_frequency <= 1 ||
                          (options_.heuristic_frequency > 0 &&
                           (node.order % static_cast<std::uint64_t>(
                                             options_.heuristic_frequency) ==
                            0));
    if (!shallow && !periodic) {
      return false;
    }
    if (fractional.size() > 24 && !shallow) {
      return false;
    }

    const double gap = relative_gap_from_bound_(relaxation.objective);
    return !std::isfinite(gap) || gap > 0.01 || shallow;
  }

  HeuristicSchedule
  build_heuristic_schedule_(const detail::ActiveNode &node,
                            const RelaxationSolution &relaxation) const {
    HeuristicSchedule schedule;
    if (node.depth > options_.heuristic_max_depth) {
      return schedule;
    }

    const bool periodic = options_.heuristic_frequency <= 1 ||
                          (options_.heuristic_frequency > 0 &&
                           (node.order % static_cast<std::uint64_t>(
                                             options_.heuristic_frequency) ==
                            0));
    const bool shallow = node.depth <= 2;
    const bool medium_depth = node.depth <= 6;
    const double gap = relative_gap_from_bound_(relaxation.objective);
    const bool has_incumbent = std::isfinite(gap);
    const bool large_gap = gap > 0.20;
    const bool medium_gap = gap > 0.05;

    if (options_.diving_strategy != DivingStrategy::Disabled) {
      schedule.run_diving = shallow || (!has_incumbent && medium_depth) ||
                            (large_gap && medium_depth) ||
                            (medium_gap && (shallow || periodic)) ||
                            (!medium_gap && shallow && periodic);
    }

    schedule.run_feasibility_jump =
        options_.use_feasibility_jump &&
        (!has_incumbent
             ? (shallow || (node.depth <= 3 && periodic))
             : (node.depth == 0 || (large_gap && shallow && periodic)));
    schedule.run_feasibility_pump =
        options_.use_feasibility_pump &&
        (!has_incumbent
             ? (shallow || (node.depth <= 4 && periodic))
             : (node.depth == 0 || (large_gap && shallow && periodic)));
    schedule.run_rens =
        options_.use_rens &&
        (!has_incumbent ? (shallow || (node.depth <= 3 && periodic))
                        : ((large_gap || medium_gap) && (shallow || periodic)));
    schedule.run_rins =
        options_.use_rins && has_incumbent &&
        ((large_gap && medium_depth) || (medium_gap && (shallow || periodic)) ||
         (!medium_gap && shallow && periodic));
    schedule.run_local_search =
        options_.use_local_search && has_incumbent &&
        ((large_gap && medium_depth && periodic) ||
         (medium_gap && (node.depth <= 8) && periodic) ||
         (!medium_gap && shallow && periodic));
    schedule.run_local_branching =
        options_.use_local_branching && has_incumbent &&
        ((large_gap && medium_depth) ||
         (medium_gap && (node.depth <= 5) && periodic) ||
         (!medium_gap && shallow && periodic));
    return schedule;
  }
  void merge_pseudocosts_(const std::vector<detail::PseudoCost> &before,
                          const std::vector<detail::PseudoCost> &after) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const int n = std::min<int>(std::min(before.size(), after.size()),
                                pseudocosts_.size());

    for (int i = 0; i < n; ++i) {
      // Classical pseudocost statistics
      pseudocosts_[i].cost.up_sum +=
          after[i].cost.up_sum - before[i].cost.up_sum;
      pseudocosts_[i].cost.down_sum +=
          after[i].cost.down_sum - before[i].cost.down_sum;
      pseudocosts_[i].cost.up_count +=
          after[i].cost.up_count - before[i].cost.up_count;
      pseudocosts_[i].cost.down_count +=
          after[i].cost.down_count - before[i].cost.down_count;

      // HiGHS-style auxiliary signals
      pseudocosts_[i].signal.inference_up +=
          after[i].signal.inference_up - before[i].signal.inference_up;
      pseudocosts_[i].signal.inference_down +=
          after[i].signal.inference_down - before[i].signal.inference_down;

      pseudocosts_[i].signal.conflict_score_up +=
          after[i].signal.conflict_score_up -
          before[i].signal.conflict_score_up;
      pseudocosts_[i].signal.conflict_score_down +=
          after[i].signal.conflict_score_down -
          before[i].signal.conflict_score_down;

      pseudocosts_[i].signal.cutoff_up +=
          after[i].signal.cutoff_up - before[i].signal.cutoff_up;
      pseudocosts_[i].signal.cutoff_down +=
          after[i].signal.cutoff_down - before[i].signal.cutoff_down;
    }
  }

  void
  merge_diving_stats_(const std::vector<detail::DivingStrategyStats> &before,
                      const std::vector<detail::DivingStrategyStats> &after) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    const int n = std::min<int>(std::min(before.size(), after.size()),
                                diving_stats_.size());
    for (int i = 0; i < n; ++i) {
      diving_stats_[i].attempts += after[i].attempts - before[i].attempts;
      diving_stats_[i].successes += after[i].successes - before[i].successes;
      diving_stats_[i].lp_iterations +=
          after[i].lp_iterations - before[i].lp_iterations;
      diving_stats_[i].lp_solves += after[i].lp_solves - before[i].lp_solves;
    }
  }

  void note_lp_work_(int iterations) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    ++node_count_;
    lp_iterations_ += iterations;
    if (node_count_ >= options_.max_nodes) {
      hit_node_limit_ = true;
    }
  }

  void note_heuristic_result_(int lp_iterations, int successes) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    heuristic_lp_iterations_ += lp_iterations;
    heuristic_successes_ += successes;
  }

  void note_heuristic_family_successes_(int *counter, int successes) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    *counter += successes;
  }

  void mark_root_relaxation_(double objective) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!root_relaxation_objective.has_value()) {
      root_relaxation_objective = objective;
      log_progress_unlocked_("root", true);
    }
  }

  void update_tree_node_(int id,
                         const std::function<void(TreeNode &)> &updater) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    updater(tree_nodes_[id]);
  }

  std::optional<detail::ActiveNode> pop_next_active_node_() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (active_nodes_.empty()) {
      return std::nullopt;
    }
    return detail::pop_next_node(active_nodes_, options_.node_selection,
                                 problem_.maximize, options_.hybrid_depth_bias,
                                 &hybrid_counter_);
  }

  bool should_terminate_() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return hit_node_limit_ || found_unbounded_;
  }

  void mark_unbounded_() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    found_unbounded_ = true;
  }

  void maybe_update_incumbent_(const Eigen::VectorXd &primal,
                               double objective) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!has_incumbent_ ||
        objective_improves_(objective, incumbent_objective_)) {
      incumbent_objective_ = objective;
      incumbent_primal_ = primal;
      has_incumbent_ = true;
      ++incumbent_updates_;
      log_progress_unlocked_("incumbent", true);
    }
  }

  static std::string format_progress_value_(double value) {
    if (std::isnan(value))
      return "nan";
    if (!std::isfinite(value))
      return value > 0.0 ? "+inf" : "-inf";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
  }

  bool same_progress_value_(double lhs, double rhs) const {
    if (std::isnan(lhs) && std::isnan(rhs))
      return true;
    if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
      return lhs == rhs;
    }
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <=
           std::max(1e-9, options_.integrality_tol) * scale;
  }

  double current_best_bound_unlocked_() const {
    return detail::compute_best_bound(active_nodes_, has_incumbent_,
                                      incumbent_objective_, problem_.maximize,
                                      root_relaxation_objective);
  }

  std::optional<double> current_gap_unlocked_(double best_bound) const {
    if (!has_incumbent_ || !std::isfinite(incumbent_objective_) ||
        !std::isfinite(best_bound)) {
      return std::nullopt;
    }
    const double raw_gap = problem_.maximize
                               ? (best_bound - incumbent_objective_)
                               : (incumbent_objective_ - best_bound);
    if (raw_gap <= options_.integrality_tol)
      return 0.0;
    const double denom = std::max(1.0, std::abs(incumbent_objective_));
    return raw_gap / denom;
  }

  bool gap_reduced_(double current_gap, double previous_gap) const {
    if (std::isnan(previous_gap) || !std::isfinite(previous_gap)) {
      return true;
    }
    const double scale =
        std::max({1.0, std::abs(current_gap), std::abs(previous_gap)});
    return current_gap <
           previous_gap - std::max(1e-9, options_.integrality_tol) * scale;
  }

  void log_progress_unlocked_(const char *event, bool force = false,
                              const Status *final_status = nullptr) {
    if (!options_.verbose)
      return;

    if (!progress_header_printed_) {
      std::cout << "MIP Progress | Nodes  Active  Incumbent  BestBd  Gap  Event"
                << std::endl;
      progress_header_printed_ = true;
    }

    const double best_bound = current_best_bound_unlocked_();
    if (!force && !root_relaxation_objective.has_value() && !has_incumbent_ &&
        !std::isfinite(best_bound)) {
      return;
    }
    const bool incumbent_changed =
        !same_progress_value_(incumbent_objective_, last_logged_incumbent_);
    const std::optional<double> gap = current_gap_unlocked_(best_bound);
    const bool gap_reduced =
        gap.has_value() && gap_reduced_(*gap, last_logged_gap_);
    if (!force && !incumbent_changed && !gap_reduced) {
      return;
    }

    std::ostringstream oss;
    oss << "MIP Progress | " << std::setw(6) << node_count_ << "  "
        << std::setw(6) << active_nodes_.size() << "  " << std::setw(11)
        << format_progress_value_(
               has_incumbent_ ? incumbent_objective_
                              : std::numeric_limits<double>::quiet_NaN())
        << "  " << std::setw(11) << format_progress_value_(best_bound) << "  ";
    if (gap.has_value()) {
      oss << std::fixed << std::setprecision(2) << std::setw(6)
          << (100.0 * *gap) << "%";
    } else {
      oss << std::setw(6) << "--";
    }
    oss << "  " << event;
    if (final_status) {
      oss << " [" << to_string(*final_status) << "]";
    }

    std::cout << oss.str() << std::endl;
    last_logged_node_count_ = node_count_;
    last_logged_best_bound_ = best_bound;
    last_logged_incumbent_ = incumbent_objective_;
    last_logged_gap_ = gap.value_or(std::numeric_limits<double>::quiet_NaN());
  }

  void maybe_log_progress_(const char *event, bool force = false,
                           const Status *final_status = nullptr) {
    if (!options_.verbose)
      return;
    std::lock_guard<std::mutex> lock(state_mutex_);
    log_progress_unlocked_(event, force, final_status);
  }

  template <typename ProcessNode>
  void process_parallel_active_nodes_(ProcessNode &&process_node) {
    const int worker_count = std::max(1, options_.parallel_workers);
    auto worker = [&]() {
      while (true) {
        std::optional<detail::ActiveNode> node;
        {
          std::unique_lock<std::mutex> lock(state_mutex_);
          state_cv_.wait(lock, [&]() {
            return found_unbounded_ || hit_node_limit_ ||
                   !active_nodes_.empty() ||
                   (active_workers_ == 0 && active_nodes_.empty());
          });
          if (found_unbounded_ || hit_node_limit_) {
            break;
          }
          if (active_nodes_.empty()) {
            if (active_workers_ == 0) {
              break;
            }
            continue;
          }
          node = detail::pop_next_node(
              active_nodes_, options_.node_selection, problem_.maximize,
              options_.hybrid_depth_bias, &hybrid_counter_);
          ++active_workers_;
        }

        process_node(std::move(*node), false);

        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          --active_workers_;
        }
        state_cv_.notify_all();
      }
    };

    std::vector<std::jthread> workers;
    workers.reserve(std::max(0, worker_count - 1));
    for (int i = 1; i < worker_count; ++i) {
      workers.emplace_back(worker);
    }
    worker();
  }

  template <typename RelaxationSolver, typename SubMIPSolver,
            typename SubMIPSolverWithCuts>
  void process_active_node_(detail::ActiveNode node, bool allow_root_cuts,
                            RelaxationSolver &&relaxation_solver,
                            SubMIPSolver &&solve_submip,
                            SubMIPSolverWithCuts &&solve_submip_with_cuts) {
    if (should_terminate_()) {
      return;
    }

    auto solve_relaxation_with_cuts = [&](detail::ActiveNode &current_node,
                                          const std::vector<Cut> &extra_cuts) {
      std::vector<Cut> relaxation_cuts = current_relaxation_cuts_snapshot_();
      relaxation_cuts.insert(relaxation_cuts.end(), extra_cuts.begin(),
                             extra_cuts.end());
      const NodePresolveOutcome presolved = presolve_node_bounds_(
          current_node.lower_bounds, current_node.upper_bounds, relaxation_cuts,
          current_node.reasons);
      current_node.lower_bounds = presolved.lower_bounds;
      current_node.upper_bounds = presolved.upper_bounds;
      current_node.reasons = presolved.reasons;
      if (presolved.infeasible) {
        maybe_learn_conflict_from_bounds_(current_node.lower_bounds,
                                          current_node.upper_bounds);
        RelaxationSolution out;
        out.status = RelaxationStatus::Infeasible;
        out.primal =
            Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                      std::numeric_limits<double>::quiet_NaN());
        out.objective = problem_.maximize
                            ? -std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::infinity();
        return out;
      }
      return relaxation_solver(
          current_node.lower_bounds, current_node.upper_bounds,
          current_node.basis ? &*current_node.basis : nullptr, relaxation_cuts);
    };
    auto current_relaxation = [&](detail::ActiveNode &current_node) {
      return solve_relaxation_with_cuts(current_node, {});
    };

    RelaxationSolution relaxation = current_relaxation(node);
    note_lp_work_(relaxation.iterations);

    const auto estimate = detail::node_estimate(
        relaxation, problem_.variable_types, pseudocosts_snapshot_(),
        options_.integrality_tol, problem_.maximize);
    update_tree_node_(node.id, [&](TreeNode &tree_node) {
      tree_node.bound = relaxation.objective;
      tree_node.estimate = estimate;
    });
    maybe_log_progress_("node");

    if (relaxation.status == RelaxationStatus::Unbounded) {
      update_tree_node_(node.id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Unbounded;
      });
      mark_unbounded_();
      state_cv_.notify_all();
      return;
    }
    if (relaxation.status == RelaxationStatus::Infeasible) {
      maybe_learn_conflict_from_bounds_(node.lower_bounds, node.upper_bounds);
      update_tree_node_(node.id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Infeasible;
      });
      return;
    }

    mark_root_relaxation_(relaxation.objective);

    {
      const auto incumbent = incumbent_snapshot_();
      if (incumbent.has_incumbent &&
          bound_prunes_(relaxation.objective, incumbent.objective)) {
        update_tree_node_(node.id, [&](TreeNode &tree_node) {
          tree_node.status = TreeNodeStatus::PrunedByBound;
        });
        return;
      }
    }

    if (allow_root_cuts && options_.use_cut_pool && node.depth == 0) {
      bool re_solved_with_cuts = false;
      for (int round = 0; round < options_.max_cut_rounds_per_node; ++round) {
        const auto cut_fractional = detail::collect_fractional_candidates(
            relaxation.primal, problem_.variable_types,
            options_.integrality_tol);
        const std::vector<Cut> relaxation_cuts =
            current_relaxation_cuts_snapshot_();
        std::vector<Cut> generated = generate_cut_candidates_(
            node, relaxation, cut_fractional, relaxation_cuts);
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          for (const Cut &cut : generated) {
            cut_pool_.add_cut(cut);
          }
        }
        std::vector<Cut> selected;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          selected = cut_pool_.select_violated_cuts(
              relaxation.primal, options_.max_cuts_added_per_round);
        }
        if (selected.empty())
          break;

        bool added = false;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          for (const Cut &cut : selected) {
            const std::string signature = detail::cut_signature(cut);
            if (active_cut_signatures_.contains(signature))
              continue;
            active_cut_signatures_.insert(signature);
            active_cuts_.push_back(cut);
            added = true;
          }
          if (added) {
            for (auto &active_node : active_nodes_) {
              active_node.basis.reset();
            }
          }
        }
        if (!added)
          break;

        node.basis.reset();
        relaxation = current_relaxation(node);
        note_lp_work_(relaxation.iterations);
        const auto cut_estimate = detail::node_estimate(
            relaxation, problem_.variable_types, pseudocosts_snapshot_(),
            options_.integrality_tol, problem_.maximize);
        update_tree_node_(node.id, [&](TreeNode &tree_node) {
          tree_node.bound = relaxation.objective;
          tree_node.estimate = cut_estimate;
        });
        maybe_log_progress_("cut");
        re_solved_with_cuts = true;

        if (relaxation.status == RelaxationStatus::Unbounded) {
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::Unbounded;
          });
          mark_unbounded_();
          state_cv_.notify_all();
          return;
        }
        if (relaxation.status == RelaxationStatus::Infeasible) {
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::Infeasible;
          });
          return;
        }
      }

      if (re_solved_with_cuts) {
        const auto incumbent = incumbent_snapshot_();
        if (incumbent.has_incumbent &&
            bound_prunes_(relaxation.objective, incumbent.objective)) {
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::PrunedByBound;
          });
          return;
        }
      }
    }

    auto fractional = detail::collect_fractional_candidates(
        relaxation.primal, problem_.variable_types, options_.integrality_tol);
    if (should_try_node_cuts_(node, relaxation, fractional)) {
      std::vector<Cut> local_cuts;
      std::unordered_set<std::string> local_cut_signatures;
      const int node_cut_rounds = options_.max_cut_rounds_per_node;
      for (int round = 0; round < node_cut_rounds; ++round) {
        std::vector<Cut> probing_relaxation_cuts =
            current_relaxation_cuts_snapshot_();
        probing_relaxation_cuts.insert(probing_relaxation_cuts.end(),
                                       local_cuts.begin(), local_cuts.end());
        std::vector<Cut> generated = generate_cut_candidates_(
            node, relaxation, fractional, probing_relaxation_cuts);
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          for (const Cut &cut : generated) {
            cut_pool_.add_cut(cut);
          }
        }

        std::vector<Cut> selected;
        std::unordered_set<std::string> global_cut_signatures;
        {
          std::lock_guard<std::mutex> lock(state_mutex_);
          selected = cut_pool_.select_violated_cuts(
              relaxation.primal, options_.max_cuts_added_per_round);
          global_cut_signatures = active_cut_signatures_;
        }

        bool added = false;
        for (const Cut &cut : selected) {
          const std::string signature = detail::cut_signature(cut);
          if (global_cut_signatures.contains(signature) ||
              local_cut_signatures.contains(signature)) {
            continue;
          }
          local_cut_signatures.insert(signature);
          local_cuts.push_back(cut);
          added = true;
        }
        if (!added)
          break;

        node.basis.reset();
        relaxation = solve_relaxation_with_cuts(node, local_cuts);
        note_lp_work_(relaxation.iterations);
        const auto cut_estimate = detail::node_estimate(
            relaxation, problem_.variable_types, pseudocosts_snapshot_(),
            options_.integrality_tol, problem_.maximize);
        update_tree_node_(node.id, [&](TreeNode &tree_node) {
          tree_node.bound = relaxation.objective;
          tree_node.estimate = cut_estimate;
        });
        maybe_log_progress_("node-cut");

        if (relaxation.status == RelaxationStatus::Unbounded) {
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::Unbounded;
          });
          mark_unbounded_();
          state_cv_.notify_all();
          return;
        }
        if (relaxation.status == RelaxationStatus::Infeasible) {
          maybe_learn_conflict_from_bounds_(node.lower_bounds,
                                            node.upper_bounds);
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::Infeasible;
          });
          return;
        }

        {
          const auto incumbent = incumbent_snapshot_();
          if (incumbent.has_incumbent &&
              bound_prunes_(relaxation.objective, incumbent.objective)) {
            update_tree_node_(node.id, [&](TreeNode &tree_node) {
              tree_node.status = TreeNodeStatus::PrunedByBound;
            });
            return;
          }
        }

        fractional = detail::collect_fractional_candidates(
            relaxation.primal, problem_.variable_types,
            options_.integrality_tol);
        if (fractional.empty()) {
          update_tree_node_(node.id, [&](TreeNode &tree_node) {
            tree_node.status = TreeNodeStatus::Integral;
          });
          maybe_update_incumbent_(relaxation.primal, relaxation.objective);
          state_cv_.notify_all();
          return;
        }
      }
    }
    if (fractional.empty()) {
      update_tree_node_(node.id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Integral;
      });
      maybe_update_incumbent_(relaxation.primal, relaxation.objective);
      state_cv_.notify_all();
      return;
    }

    if (const auto rounded =
            detail::run_rounding_heuristic(problem_, options_, relaxation,
                                           current_relaxation_cuts_snapshot_());
        rounded.has_value()) {
      note_heuristic_result_(0, 1);
      maybe_update_incumbent_(rounded->primal, rounded->objective);
    }

    auto node_relaxation_solver = [&](const Eigen::VectorXd &lower_bounds,
                                      const Eigen::VectorXd &upper_bounds,
                                      const LPBasis *basis) {
      const std::vector<Cut> relaxation_cuts =
          current_relaxation_cuts_snapshot_();
      const NodePresolveOutcome presolved = presolve_node_bounds_(
          lower_bounds, upper_bounds, relaxation_cuts, nullptr);
      if (presolved.infeasible) {
        maybe_learn_conflict_from_bounds_(presolved.lower_bounds,
                                          presolved.upper_bounds);
        RelaxationSolution out;
        out.status = RelaxationStatus::Infeasible;
        out.primal =
            Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                      std::numeric_limits<double>::quiet_NaN());
        out.objective = problem_.maximize
                            ? -std::numeric_limits<double>::infinity()
                            : std::numeric_limits<double>::infinity();
        return out;
      }
      RelaxationSolution out =
          relaxation_solver(presolved.lower_bounds, presolved.upper_bounds,
                            basis, relaxation_cuts);
      if (out.status == RelaxationStatus::Infeasible) {
        maybe_learn_conflict_from_bounds_(presolved.lower_bounds,
                                          presolved.upper_bounds);
      }
      return out;
    };

    const HeuristicSchedule schedule =
        build_heuristic_schedule_(node, relaxation);

    if (schedule.run_feasibility_jump) {
      const detail::NeighborhoodHeuristicResult feasibility_jump =
          detail::run_feasibility_jump_heuristic(problem_, options_, relaxation,
                                                 solve_submip);
      note_heuristic_result_(feasibility_jump.lp_iterations,
                             feasibility_jump.successes);
      note_heuristic_family_successes_(&feasibility_jump_successes_,
                                       feasibility_jump.successes);
      if (feasibility_jump.incumbent.has_value()) {
        maybe_update_incumbent_(feasibility_jump.incumbent->primal,
                                feasibility_jump.incumbent->objective);
      }
    }

    if (schedule.run_feasibility_pump) {
      const detail::NeighborhoodHeuristicResult feasibility_pump =
          detail::run_feasibility_pump_heuristic(problem_, options_, relaxation,
                                                 solve_submip);
      note_heuristic_result_(feasibility_pump.lp_iterations,
                             feasibility_pump.successes);
      note_heuristic_family_successes_(&feasibility_pump_successes_,
                                       feasibility_pump.successes);
      if (feasibility_pump.incumbent.has_value()) {
        maybe_update_incumbent_(feasibility_pump.incumbent->primal,
                                feasibility_pump.incumbent->objective);
      }
    }

    if (schedule.run_diving) {
      const auto incumbent = incumbent_snapshot_();
      const auto diving_before = diving_stats_snapshot_();
      auto diving_stats = diving_before;
      const detail::DivingHeuristicResult diving = detail::run_diving_heuristic(
          node, relaxation, problem_, options_,
          incumbent.has_incumbent ? &incumbent.primal : nullptr, diving_stats,
          node_relaxation_solver);
      merge_diving_stats_(diving_before, diving_stats);
      note_heuristic_result_(diving.lp_iterations, diving.successes);
      if (diving.incumbent.has_value()) {
        maybe_update_incumbent_(diving.incumbent->primal,
                                diving.incumbent->objective);
      }
    }

    if (schedule.run_rens) {
      const detail::NeighborhoodHeuristicResult rens =
          detail::run_rens_heuristic(problem_, options_, relaxation,
                                     solve_submip);
      note_heuristic_result_(rens.lp_iterations, rens.successes);
      note_heuristic_family_successes_(&rens_successes_, rens.successes);
      if (rens.incumbent.has_value()) {
        maybe_update_incumbent_(rens.incumbent->primal,
                                rens.incumbent->objective);
      }
    }

    const HeuristicSchedule incumbent_schedule =
        build_heuristic_schedule_(node, relaxation);
    const auto incumbent = incumbent_snapshot_();
    if (incumbent.has_incumbent && incumbent_schedule.run_rins) {
      const detail::NeighborhoodHeuristicResult rins =
          detail::run_rins_heuristic(problem_, options_, relaxation,
                                     incumbent.primal, incumbent.objective,
                                     solve_submip);
      note_heuristic_result_(rins.lp_iterations, rins.successes);
      note_heuristic_family_successes_(&rins_successes_, rins.successes);
      if (rins.incumbent.has_value()) {
        maybe_update_incumbent_(rins.incumbent->primal,
                                rins.incumbent->objective);
      }
    }

    if (incumbent.has_incumbent && incumbent_schedule.run_local_search) {
      const detail::NeighborhoodHeuristicResult local_search =
          detail::run_local_search_heuristic(problem_, options_, relaxation,
                                             incumbent.primal,
                                             incumbent.objective, solve_submip);
      note_heuristic_result_(local_search.lp_iterations,
                             local_search.successes);
      note_heuristic_family_successes_(&local_search_successes_,
                                       local_search.successes);
      if (local_search.incumbent.has_value()) {
        maybe_update_incumbent_(local_search.incumbent->primal,
                                local_search.incumbent->objective);
      }
    }

    if (incumbent.has_incumbent && incumbent_schedule.run_local_branching) {
      const detail::NeighborhoodHeuristicResult local_branching =
          detail::run_local_branching_heuristic(
              problem_, options_, relaxation, incumbent.primal,
              incumbent.objective, solve_submip_with_cuts);
      note_heuristic_result_(local_branching.lp_iterations,
                             local_branching.successes);
      note_heuristic_family_successes_(&local_branching_successes_,
                                       local_branching.successes);
      if (local_branching.incumbent.has_value()) {
        maybe_update_incumbent_(local_branching.incumbent->primal,
                                local_branching.incumbent->objective);
      }
    }

    update_tree_node_(node.id, [&](TreeNode &tree_node) {
      tree_node.status = TreeNodeStatus::Fractional;
    });

    const auto pseudocost_before = pseudocosts_snapshot_();
    auto local_pseudocosts = pseudocost_before;
    detail::BranchDecision decision = detail::choose_branching_variable(
        node, relaxation, fractional, options_, problem_.maximize,
        local_pseudocosts, parallel_task_dispatcher_.get(),
        node_relaxation_solver);
    merge_pseudocosts_(pseudocost_before, local_pseudocosts);
    if (decision.variable < 0) {
      update_tree_node_(node.id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Fathomed;
      });
      return;
    }

    update_tree_node_(node.id, [&](TreeNode &tree_node) {
      tree_node.status = TreeNodeStatus::Branched;
      tree_node.branch_var = decision.variable;
      tree_node.branch_value = relaxation.primal(decision.variable);
    });

    const LPBasis *parent_basis = node.basis ? &*node.basis : nullptr;
    auto first_child = decision.down_child;
    auto second_child = decision.up_child;
    if (options_.node_selection == NodeSelectionStrategy::DepthFirst &&
        first_child.relaxation.has_value() &&
        second_child.relaxation.has_value()) {
      const bool first_better =
          problem_.maximize ? (first_child.relaxation->objective >
                               second_child.relaxation->objective + 1e-12)
                            : (first_child.relaxation->objective <
                               second_child.relaxation->objective - 1e-12);
      if (first_better) {
        std::swap(first_child, second_child);
      }
    }
    if (parallel_task_dispatcher_ != nullptr &&
        !first_child.relaxation.has_value() &&
        !second_child.relaxation.has_value()) {
      std::array<std::optional<RelaxationSolution>, 2> child_relaxations;
      parallel_task_dispatcher_->run(2, [&](int child_index) {
        auto &target = child_index == 0 ? first_child : second_child;
        child_relaxations[static_cast<std::size_t>(child_index)] =
            node_relaxation_solver(target.state.lower_bounds,
                                   target.state.upper_bounds, parent_basis);
      });
      first_child.relaxation = std::move(child_relaxations[0]);
      second_child.relaxation = std::move(child_relaxations[1]);
    }

    process_child_(node.id, node.depth + 1, decision.variable, decision.value,
                   parent_basis, std::move(first_child),
                   node_relaxation_solver);
    if (should_terminate_()) {
      state_cv_.notify_all();
      return;
    }
    process_child_(node.id, node.depth + 1, decision.variable, decision.value,
                   parent_basis, std::move(second_child),
                   node_relaxation_solver);
    state_cv_.notify_all();
  }

  template <typename RelaxationSolver>
  void process_child_(int parent_id, int depth, int branch_variable,
                      double branch_value, const LPBasis *parent_basis,
                      detail::ChildEvaluation child,
                      RelaxationSolver &&relaxation_solver) {
    int child_id = -1;
    std::uint64_t order = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      order = next_order_++;
      child_id = detail::append_tree_node(tree_nodes_, parent_id, depth, order);
      TreeNode &tree_node = tree_nodes_[child_id];
      tree_node.branch_var = branch_variable;
      tree_node.branch_value = branch_value;
    }

    if (child.state.upper_bounds(branch_variable) + options_.integrality_tol <
        child.state.lower_bounds(branch_variable)) {
      maybe_learn_conflict_from_bounds_(child.state.lower_bounds,
                                        child.state.upper_bounds);
      update_tree_node_(child_id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Infeasible;
      });
      return;
    }

    if (const detail::ConflictGraph *graph = conflict_graph_();
        graph != nullptr) {
      if (child.state.reasons == nullptr ||
          static_cast<int>(child.state.reasons->size()) !=
              graph->literal_count()) {
        child.state.reasons =
            std::make_shared<NodeReasonStore>(graph->literal_count());
      }
      const std::optional<int> fixed_literal =
          fixed_binary_literal_from_bounds_(branch_variable,
                                            child.state.lower_bounds,
                                            child.state.upper_bounds);
      if (fixed_literal.has_value()) {
        seed_fixed_literal_reason_(*fixed_literal, &child.state.reasons);
      }
    }

    if (!child.relaxation.has_value()) {
      child.relaxation = relaxation_solver(
          child.state.lower_bounds, child.state.upper_bounds, parent_basis);
    }

    note_lp_work_(child.relaxation->iterations);
    const auto estimate = detail::node_estimate(
        *child.relaxation, problem_.variable_types, pseudocosts_snapshot_(),
        options_.integrality_tol, problem_.maximize);
    update_tree_node_(child_id, [&](TreeNode &tree_node) {
      tree_node.bound = child.relaxation->objective;
      tree_node.estimate = estimate;
    });
    maybe_log_progress_("child");

    if (child.relaxation->status == RelaxationStatus::Infeasible) {
      maybe_learn_conflict_from_bounds_(child.state.lower_bounds,
                                        child.state.upper_bounds);
      update_tree_node_(child_id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Infeasible;
      });
      return;
    }
    if (child.relaxation->status == RelaxationStatus::Unbounded) {
      update_tree_node_(child_id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Unbounded;
      });
      mark_unbounded_();
      return;
    }

    // Collect fractional candidates before bound check for cutoff tracking
    const auto fractional = detail::collect_fractional_candidates(
        child.relaxation->primal, problem_.variable_types,
        options_.integrality_tol);

    {
      const auto incumbent = incumbent_snapshot_();
      if (incumbent.has_incumbent &&
          bound_prunes_(child.relaxation->objective, incumbent.objective)) {
        // Record cutoff for fractional variables - they caused this node to be
        // pruned
        if (!fractional.empty()) {
          std::lock_guard<std::mutex> lock(state_mutex_);
          for (const auto &fc : fractional) {
            if (fc.variable >= 0 &&
                fc.variable < static_cast<int>(pseudocosts_.size())) {
              // Record cutoff for both branches (HiGHS-style)
              pseudocosts_[fc.variable].record_cutoff();
            }
          }
        }
        update_tree_node_(child_id, [&](TreeNode &tree_node) {
          tree_node.status = TreeNodeStatus::PrunedByBound;
        });
        return;
      }
    }
    if (fractional.empty()) {
      update_tree_node_(child_id, [&](TreeNode &tree_node) {
        tree_node.status = TreeNodeStatus::Integral;
      });
      maybe_update_incumbent_(child.relaxation->primal,
                              child.relaxation->objective);
      state_cv_.notify_all();
      return;
    }
    update_tree_node_(child_id, [&](TreeNode &tree_node) {
      tree_node.status = TreeNodeStatus::Created;
    });
    detail::ActiveNode active;
    active.id = child_id;
    active.parent_id = parent_id;
    active.depth = depth;
    active.order = order;
    active.bound = child.relaxation->objective;
    active.estimate = estimate;
    active.lower_bounds = child.state.lower_bounds;
    active.upper_bounds = child.state.upper_bounds;
    active.basis = child.relaxation->basis;
    active.reasons = child.state.reasons;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      detail::push_active_node(active_nodes_, std::move(active),
                               options_.node_selection, problem_.maximize);
      log_progress_unlocked_("bound");
    }
    state_cv_.notify_one();
  }

  SolveResult finalize_result_(Status status) const {
    SolveResult result;
    result.status = status;
    result.objective = has_incumbent_
                           ? incumbent_objective_
                           : std::numeric_limits<double>::quiet_NaN();
    result.primal = has_incumbent_
                        ? incumbent_primal_
                        : Eigen::VectorXd::Constant(
                              problem_.lower_bounds.size(),
                              std::numeric_limits<double>::quiet_NaN());
    result.best_bound = detail::compute_best_bound(
        active_nodes_, has_incumbent_, incumbent_objective_, problem_.maximize,
        root_relaxation_objective);
    result.root_relaxation_objective = root_relaxation_objective.value_or(
        std::numeric_limits<double>::quiet_NaN());
    result.node_count = node_count_;
    result.lp_iterations = lp_iterations_ + heuristic_lp_iterations_;
    result.incumbent_updates = incumbent_updates_;
    result.heuristic_lp_iterations = heuristic_lp_iterations_;
    result.heuristic_successes = heuristic_successes_;
    result.feasibility_jump_successes = feasibility_jump_successes_;
    result.feasibility_pump_successes = feasibility_pump_successes_;
    result.rens_successes = rens_successes_;
    result.rins_successes = rins_successes_;
    result.local_search_successes = local_search_successes_;
    result.local_branching_successes = local_branching_successes_;
    result.cuts_generated = cut_pool_.cuts_generated();
    result.cuts_applied = cut_pool_.cuts_applied();
    result.duplicate_cuts = cut_pool_.duplicate_cuts();
    result.cut_pool_size = cut_pool_.size();
    result.has_solution = has_incumbent_;
    result.tree_nodes = tree_nodes_;
    return result;
  }

  Problem problem_;
  Options options_;
  std::vector<detail::PseudoCost> pseudocosts_;
  std::unique_ptr<detail::ParallelDispatcher> parallel_task_dispatcher_;
  std::vector<detail::ActiveNode> active_nodes_;
  detail::CutPool cut_pool_{options_};
  std::vector<Cut> active_cuts_;
  std::unordered_set<std::string> active_cut_signatures_;
  std::vector<TreeNode> tree_nodes_;
  int node_count_ = 0;
  int lp_iterations_ = 0;
  int incumbent_updates_ = 0;
  int heuristic_lp_iterations_ = 0;
  int heuristic_successes_ = 0;
  int feasibility_jump_successes_ = 0;
  int feasibility_pump_successes_ = 0;
  int rens_successes_ = 0;
  int rins_successes_ = 0;
  int local_search_successes_ = 0;
  int local_branching_successes_ = 0;
  std::vector<LearnedConflict> learned_conflicts_;
  std::vector<std::vector<ConflictLiteral>> learned_implications_;
  std::vector<detail::DivingStrategyStats> diving_stats_;
  std::uint64_t next_order_ = 0;
  std::uint64_t hybrid_counter_ = 0;
  bool has_incumbent_ = false;
  bool hit_node_limit_ = false;
  bool found_unbounded_ = false;
  int active_workers_ = 0;
  double incumbent_objective_ = std::numeric_limits<double>::quiet_NaN();
  Eigen::VectorXd incumbent_primal_;
  std::optional<double> root_relaxation_objective;
  bool progress_header_printed_ = false;
  int last_logged_node_count_ = 0;
  double last_logged_best_bound_ = std::numeric_limits<double>::quiet_NaN();
  double last_logged_incumbent_ = std::numeric_limits<double>::quiet_NaN();
  double last_logged_gap_ = std::numeric_limits<double>::quiet_NaN();
  std::vector<Cut> initial_cuts_;
  mutable std::once_flag conflict_graph_once_;
  mutable std::unique_ptr<detail::ConflictGraph> conflict_graph_cache_;
  mutable std::mutex state_mutex_;
  // mutable std::mutex pseudocosts_mutex_;
  std::condition_variable state_cv_;
};

} // namespace simplex::bnb
