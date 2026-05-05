#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bnb/async_heuristic_manager.h"
#include "bnb/branching.h"
#include "bnb/conflict_engine.h"
#include "bnb/core_types.h"
#include "bnb/cuts.h"
#include "bnb/heuristic.h"
// #include "bnb/node_pool.h"
#include "bnb/parallel.h"
#include "bnb/search_coordinator.h"

namespace simplex::bnb {

class Solver {
    friend class AsyncHeuristicManager;
    friend class ConflictEngine;

  public:
    explicit Solver(Problem problem, Options options = {}, std::vector<Cut> initial_cuts = {})
        : problem_(std::move(problem)), options_(std::move(options)),
          initial_cuts_(std::move(initial_cuts)) {
        validate_inputs_();
        pseudocosts_.resize(problem_.variable_types.size());
    }

    std::optional<LPBasis> root_warm_start_basis_state() const {
        std::lock_guard<std::mutex> lock(incumbent_mutex_);
        return root_warm_start_basis_state_;
    }

    // Public snapshot of the current incumbent so per-node LP solvers can
    // install an objective-bound cutoff and bail out early in dual phase 2.
    struct IncumbentObjectiveSnapshot {
        bool has_incumbent = false;
        double objective = std::numeric_limits<double>::quiet_NaN();
    };

    IncumbentObjectiveSnapshot incumbent_objective_snapshot() const {
        std::lock_guard<std::mutex> lock(incumbent_mutex_);
        IncumbentObjectiveSnapshot snap;
        snap.has_incumbent = has_incumbent_;
        snap.objective = incumbent_objective_;
        return snap;
    }

    bool maximize() const noexcept { return problem_.maximize; }
    double objective_constant() const noexcept { return problem_.objective_constant; }

    template <typename RelaxationSolver> SolveResult solve(RelaxationSolver&& relaxation_solver) {
        reset_state_();
        initialize_node_timing_log_();
        parallel_task_dispatcher_ =
            options_.parallel_workers > 1
                ? std::make_unique<detail::ParallelDispatcher>(options_.parallel_workers)
                : nullptr;
        start_async_heuristic_workers_();

        auto solve_submip_with_cuts = [&](const Eigen::VectorXd& lower_bounds,
                                          const Eigen::VectorXd& upper_bounds,
                                          const std::vector<Cut>& initial_cuts) -> SolveResult {
            Problem subproblem = problem_;
            subproblem.lower_bounds = lower_bounds;
            subproblem.upper_bounds = upper_bounds;

            auto heuristic_subproblem_node_limit = [&](const Eigen::VectorXd& sub_lower_bounds,
                                                       const Eigen::VectorXd& sub_upper_bounds) {
                int discrete_count = 0;
                int fixed_count = 0;
                for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) &&
                                j < sub_lower_bounds.size() && j < sub_upper_bounds.size();
                     ++j) {
                    if (problem_.variable_types[j] == VariableType::Continuous) {
                        continue;
                    }
                    ++discrete_count;
                    if (sub_upper_bounds(j) <= sub_lower_bounds(j) + options_.integrality_tol) {
                        ++fixed_count;
                    }
                }

                int budget = options_.heuristic_subproblem_max_nodes;
                if (discrete_count >= 64) {
                    budget = std::min(budget, 12);
                } else if (discrete_count >= 32) {
                    budget = std::min(budget, 16);
                }

                const int free_discrete = std::max(0, discrete_count - fixed_count);
                if (free_discrete > 24) {
                    budget = std::min(budget, 8);
                } else if (free_discrete > 12) {
                    budget = std::min(budget, 12);
                } else if (free_discrete > 8) {
                    budget = std::min(budget, 16);
                }

                return std::max(4, budget);
            };

            Options sub_options = options_;
            sub_options.max_nodes = heuristic_subproblem_node_limit(lower_bounds, upper_bounds);
            sub_options.node_selection = NodeSelectionStrategy::DepthFirst;
            sub_options.branching_strategy = BranchingStrategy::MostFractional;
            sub_options.diving_strategy = options_.diving_strategy;
            sub_options.verbose = false;
            sub_options.parallel_workers = 1;
            sub_options.node_timing_log_path.clear();
            sub_options.strong_branching_candidates = 0;
            sub_options.strong_branching_max_depth = 0;
            sub_options.heuristic_frequency = std::numeric_limits<int>::max();
            sub_options.heuristic_max_depth = 0;
            sub_options.use_async_heuristics = false;
            sub_options.use_feasibility_jump = false;
            sub_options.use_feasibility_pump = false;
            sub_options.use_rens = false;
            sub_options.use_rins = false;
            sub_options.use_local_search = false;
            sub_options.use_local_branching = false;
            sub_options.use_cut_pool = options_.use_cut_pool && sub_options.max_nodes >= 12;
            sub_options.max_cut_rounds_per_node =
                sub_options.use_cut_pool ? std::min(2, options_.max_cut_rounds_per_node) : 0;
            sub_options.max_cuts_added_per_round =
                sub_options.use_cut_pool ? std::min(4, options_.max_cuts_added_per_round) : 0;

            Solver subsolver(std::move(subproblem), sub_options, initial_cuts);
            return subsolver.solve(relaxation_solver);
        };

        auto solve_submip = [&](const Eigen::VectorXd& lower_bounds,
                                const Eigen::VectorXd& upper_bounds) -> SolveResult {
            return solve_submip_with_cuts(lower_bounds, upper_bounds, {});
        };

        search_coordinator_.push(make_root_node_(), options_.node_selection, problem_.maximize);

        auto process_node = [&](detail::ActiveNode node, bool allow_root_cuts, int worker_id) {
            reap_async_heuristics_();
            process_active_node_(std::move(node), allow_root_cuts, relaxation_solver, solve_submip,
                                 solve_submip_with_cuts, worker_id);
            reap_async_heuristics_();
        };

        if (options_.parallel_workers > 1) {
            process_parallel_active_nodes_(process_node);
        } else {
            while (true) {
                if (should_terminate_())
                    break;
                std::optional<detail::ActiveNode> next_node = pop_next_active_node_();
                if (!next_node.has_value())
                    break;
                process_node(std::move(*next_node), next_node->depth == 0, -1);
                // HiGHS-style MIP gap termination: stop once the best_bound across
                // open nodes is within the configured absolute or relative gap of
                // the incumbent.
                const IncumbentSnapshot inc = incumbent_snapshot_();
                if (inc.has_incumbent) {
                    const double best_bound = search_coordinator_.compute_best_bound(
                        inc.has_incumbent, inc.objective, problem_.maximize,
                        root_relaxation_objective);
                    if (mip_gap_closed_(best_bound)) {
                        break;
                    }
                }
            }
        }

        stop_async_heuristic_workers_();

        Status final_status = Status::NodeLimit;
        {
            const IncumbentSnapshot incumbent = incumbent_snapshot_();
            if (search_coordinator_.found_unbounded()) {
                final_status = Status::Unbounded;
            } else if (incumbent.has_incumbent && search_coordinator_.empty() &&
                       !search_coordinator_.hit_node_limit()) {
                final_status = Status::Optimal;
            } else if (!incumbent.has_incumbent && search_coordinator_.empty() &&
                       !search_coordinator_.hit_node_limit()) {
                final_status = Status::Infeasible;
            }
            if (options_.verbose) {
                maybe_log_progress_("done", true, &final_status);
                log_timing_summary_();
            }
        }
        return finalize_result_(final_status);
    }

  private:
    // Core types moved to include/bnb/core_types.h

    struct ProgressSnapshot {
        int node_count = 0;
        int active_nodes = 0;
        bool has_incumbent = false;
        double incumbent_objective = std::numeric_limits<double>::quiet_NaN();
        std::optional<double> root_relaxation_objective;
    };

    static std::uint64_t elapsed_ns_(SteadyClock::time_point start, SteadyClock::time_point end) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    static void accumulate_relaxation_timing_(RelaxationTimingBucket* bucket,
                                              const RelaxationSolution& relaxation,
                                              std::uint64_t wall_ns,
                                              std::uint64_t presolve_wall_ns) {
        ++bucket->solve_count;
        bucket->wall_ns += wall_ns;
        bucket->presolve_wall_ns += presolve_wall_ns;
        if (relaxation.attempted_warm_start_basis_state) {
            ++bucket->warm_start_attempt_count;
        }
        if (relaxation.used_warm_start_basis_state) {
            ++bucket->warm_start_accept_count;
        }
        if (relaxation.cold_retried_after_warm_start) {
            ++bucket->warm_start_cold_retry_count;
        }
        bucket->core_solve_time_ns += relaxation.core_solve_time_ns;
        bucket->lp_assembly_time_ns += relaxation.lp_assembly_time_ns;
        bucket->lp_internal_presolve_ns += relaxation.lp_internal_presolve_ns;
        bucket->lp_internal_crash_ns += relaxation.lp_internal_crash_ns;
        bucket->lp_internal_iters_ns += relaxation.lp_internal_iters_ns;
        bucket->lp_internal_serialize_ns += relaxation.lp_internal_serialize_ns;
    }

    static std::string json_escape_(std::string_view value) {
        std::string escaped;
        escaped.reserve(value.size());
        for (const char ch : value) {
            switch (ch) {
                case '\\':
                    escaped += "\\\\";
                    break;
                case '"':
                    escaped += "\\\"";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped += ch;
                    break;
            }
        }
        return escaped;
    }

    static void append_json_string_field_(std::ostringstream& oss, std::string_view key,
                                          std::string_view value, bool* first_field) {
        if (!*first_field) {
            oss << ',';
        }
        *first_field = false;
        oss << '"' << key << "\":\"" << json_escape_(value) << '"';
    }

    static void append_json_int_field_(std::ostringstream& oss, std::string_view key,
                                       long long value, bool* first_field) {
        if (!*first_field) {
            oss << ',';
        }
        *first_field = false;
        oss << '"' << key << "\":" << value;
    }

    static void append_json_uint64_field_(std::ostringstream& oss, std::string_view key,
                                          std::uint64_t value, bool* first_field) {
        append_json_int_field_(oss, key, static_cast<long long>(value), first_field);
    }

    static void append_json_bool_field_(std::ostringstream& oss, std::string_view key, bool value,
                                        bool* first_field) {
        if (!*first_field) {
            oss << ',';
        }
        *first_field = false;
        oss << '"' << key << "\":" << (value ? "true" : "false");
    }

    static void append_json_double_field_(std::ostringstream& oss, std::string_view key,
                                          double value, bool* first_field) {
        if (!*first_field) {
            oss << ',';
        }
        *first_field = false;
        oss << '"' << key << "\":";
        if (!std::isfinite(value)) {
            oss << "null";
            return;
        }
        oss << std::setprecision(17) << value;
    }

    void initialize_node_timing_log_() {
        std::lock_guard<std::mutex> lock(node_timing_log_mutex_);
        node_timing_log_stream_.close();
        if (options_.node_timing_log_path.empty()) {
            return;
        }

        const std::filesystem::path log_path(options_.node_timing_log_path);
        if (log_path.has_parent_path()) {
            std::filesystem::create_directories(log_path.parent_path());
        }

        node_timing_log_stream_.open(log_path, std::ios::out | std::ios::trunc);
        if (!node_timing_log_stream_.is_open()) {
            throw std::runtime_error("simplex::bnb: unable to open node timing log '" +
                                     log_path.string() + "'");
        }
    }

    void flush_node_timing_record_(const NodeTimingRecord& record) {
        std::lock_guard<std::mutex> lock(node_timing_log_mutex_);
        if (!node_timing_log_stream_.is_open()) {
            return;
        }

        std::ostringstream oss;
        bool first_field = true;
        oss << '{';
        append_json_int_field_(oss, "node_id", record.node_id, &first_field);
        append_json_int_field_(oss, "parent_id", record.parent_id, &first_field);
        append_json_int_field_(oss, "depth", record.depth, &first_field);
        append_json_uint64_field_(oss, "order", record.order, &first_field);
        append_json_bool_field_(oss, "allow_root_cuts", record.allow_root_cuts, &first_field);
        append_json_bool_field_(oss, "root_node", record.root_node, &first_field);
        append_json_string_field_(oss, "exit_stage", record.exit_stage, &first_field);
        append_json_string_field_(oss, "final_status", record.final_status, &first_field);
        append_json_int_field_(oss, "branch_variable", record.branch_variable, &first_field);
        append_json_double_field_(oss, "branch_value", record.branch_value, &first_field);
        append_json_double_field_(oss, "final_bound", record.final_bound, &first_field);
        append_json_double_field_(oss, "final_estimate", record.final_estimate, &first_field);
        append_json_double_field_(oss, "final_relaxation_objective",
                                  record.final_relaxation_objective, &first_field);
        append_json_int_field_(oss, "fractional_count", record.fractional_count, &first_field);
        append_json_int_field_(oss, "root_cut_rounds", record.root_cut_rounds, &first_field);
        append_json_int_field_(oss, "root_cuts_generated", record.root_cuts_generated,
                               &first_field);
        append_json_int_field_(oss, "root_cuts_selected", record.root_cuts_selected, &first_field);
        append_json_int_field_(oss, "root_cuts_applied", record.root_cuts_applied, &first_field);
        append_json_int_field_(oss, "node_cut_rounds", record.node_cut_rounds, &first_field);
        append_json_int_field_(oss, "node_cuts_generated", record.node_cuts_generated,
                               &first_field);
        append_json_int_field_(oss, "node_cuts_selected", record.node_cuts_selected, &first_field);
        append_json_int_field_(oss, "node_cuts_applied", record.node_cuts_applied, &first_field);
        append_json_int_field_(oss, "strong_branching_probe_count",
                               record.strong_branching_probe_count, &first_field);
        append_json_int_field_(oss, "strong_branching_probe_iterations",
                               record.strong_branching_probe_iterations, &first_field);
        append_json_int_field_(oss, "async_heuristic_launch_attempts",
                               record.async_heuristic_launch_attempts, &first_field);
        append_json_uint64_field_(oss, "total_wall_ns", record.total_wall_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_wall_ns", record.node_relaxation.wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "node_presolve_wall_ns",
                                  record.node_relaxation.presolve_wall_ns, &first_field);
        append_json_int_field_(oss, "node_relaxation_solve_count",
                               record.node_relaxation.solve_count, &first_field);
        append_json_int_field_(oss, "node_relaxation_warm_start_attempt_count",
                               record.node_relaxation.warm_start_attempt_count, &first_field);
        append_json_int_field_(oss, "node_relaxation_warm_start_accept_count",
                               record.node_relaxation.warm_start_accept_count, &first_field);
        append_json_int_field_(oss, "node_relaxation_warm_start_cold_retry_count",
                               record.node_relaxation.warm_start_cold_retry_count, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_core_solve_time_ns",
                                  record.node_relaxation.core_solve_time_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_lp_assembly_time_ns",
                                  record.node_relaxation.lp_assembly_time_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_lp_internal_presolve_ns",
                                  record.node_relaxation.lp_internal_presolve_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_lp_internal_crash_ns",
                                  record.node_relaxation.lp_internal_crash_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_lp_internal_iters_ns",
                                  record.node_relaxation.lp_internal_iters_ns, &first_field);
        append_json_uint64_field_(oss, "node_relaxation_lp_internal_serialize_ns",
                                  record.node_relaxation.lp_internal_serialize_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_wall_ns", record.child_relaxation.wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "child_presolve_wall_ns",
                                  record.child_relaxation.presolve_wall_ns, &first_field);
        append_json_int_field_(oss, "child_relaxation_solve_count",
                               record.child_relaxation.solve_count, &first_field);
        append_json_int_field_(oss, "child_relaxation_warm_start_attempt_count",
                               record.child_relaxation.warm_start_attempt_count, &first_field);
        append_json_int_field_(oss, "child_relaxation_warm_start_accept_count",
                               record.child_relaxation.warm_start_accept_count, &first_field);
        append_json_int_field_(oss, "child_relaxation_warm_start_cold_retry_count",
                               record.child_relaxation.warm_start_cold_retry_count, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_core_solve_time_ns",
                                  record.child_relaxation.core_solve_time_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_lp_assembly_time_ns",
                                  record.child_relaxation.lp_assembly_time_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_lp_internal_presolve_ns",
                                  record.child_relaxation.lp_internal_presolve_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_lp_internal_crash_ns",
                                  record.child_relaxation.lp_internal_crash_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_lp_internal_iters_ns",
                                  record.child_relaxation.lp_internal_iters_ns, &first_field);
        append_json_uint64_field_(oss, "child_relaxation_lp_internal_serialize_ns",
                                  record.child_relaxation.lp_internal_serialize_ns, &first_field);
        append_json_uint64_field_(oss, "root_cut_generation_wall_ns",
                                  record.root_cut_generation_wall_ns, &first_field);
        append_json_uint64_field_(oss, "root_cut_selection_wall_ns",
                                  record.root_cut_selection_wall_ns, &first_field);
        append_json_uint64_field_(oss, "root_cut_activation_wall_ns",
                                  record.root_cut_activation_wall_ns, &first_field);
        append_json_uint64_field_(oss, "root_cut_resolve_wall_ns", record.root_cut_resolve_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "node_cut_generation_wall_ns",
                                  record.node_cut_generation_wall_ns, &first_field);
        append_json_uint64_field_(oss, "node_cut_selection_wall_ns",
                                  record.node_cut_selection_wall_ns, &first_field);
        append_json_uint64_field_(oss, "node_cut_resolve_wall_ns", record.node_cut_resolve_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "rounding_heuristic_wall_ns",
                                  record.rounding_heuristic_wall_ns, &first_field);
        append_json_uint64_field_(oss, "heuristics_wall_ns", record.heuristics_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "feasibility_jump_wall_ns", record.feasibility_jump_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "feasibility_pump_wall_ns", record.feasibility_pump_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "diving_wall_ns", record.diving_wall_ns, &first_field);
        append_json_uint64_field_(oss, "rens_wall_ns", record.rens_wall_ns, &first_field);
        append_json_uint64_field_(oss, "rins_wall_ns", record.rins_wall_ns, &first_field);
        append_json_uint64_field_(oss, "local_search_wall_ns", record.local_search_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "local_branching_wall_ns", record.local_branching_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "branching_wall_ns", record.branching_wall_ns, &first_field);
        append_json_uint64_field_(oss, "child_processing_wall_ns", record.child_processing_wall_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_core_solve_time_ns",
                                  record.strong_branching_probe_core_solve_time_ns, &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_lp_assembly_time_ns",
                                  record.strong_branching_probe_lp_assembly_time_ns, &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_lp_internal_presolve_ns",
                                  record.strong_branching_probe_lp_internal_presolve_ns,
                                  &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_lp_internal_crash_ns",
                                  record.strong_branching_probe_lp_internal_crash_ns, &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_lp_internal_iters_ns",
                                  record.strong_branching_probe_lp_internal_iters_ns, &first_field);
        append_json_uint64_field_(oss, "strong_branching_probe_lp_internal_serialize_ns",
                                  record.strong_branching_probe_lp_internal_serialize_ns,
                                  &first_field);
        oss << "}\n";
        node_timing_log_stream_ << oss.str();
        node_timing_log_stream_.flush();
    }

    void note_node_timing_(const NodeTimingRecord& record) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        root_cut_generation_wall_ns_ += record.root_cut_generation_wall_ns;
        root_cut_selection_wall_ns_ += record.root_cut_selection_wall_ns;
        root_cut_activation_wall_ns_ += record.root_cut_activation_wall_ns;
        root_cut_resolve_wall_ns_ += record.root_cut_resolve_wall_ns;
        node_cut_generation_wall_ns_ += record.node_cut_generation_wall_ns;
        node_cut_selection_wall_ns_ += record.node_cut_selection_wall_ns;
        node_cut_resolve_wall_ns_ += record.node_cut_resolve_wall_ns;
        rounding_heuristic_wall_ns_ += record.rounding_heuristic_wall_ns;
        heuristics_wall_ns_ += record.heuristics_wall_ns;
        feasibility_jump_wall_ns_ += record.feasibility_jump_wall_ns;
        feasibility_pump_wall_ns_ += record.feasibility_pump_wall_ns;
        diving_wall_ns_ += record.diving_wall_ns;
        rens_wall_ns_ += record.rens_wall_ns;
        rins_wall_ns_ += record.rins_wall_ns;
        local_search_wall_ns_ += record.local_search_wall_ns;
        local_branching_wall_ns_ += record.local_branching_wall_ns;
        branching_wall_ns_ += record.branching_wall_ns;
        child_processing_wall_ns_ += record.child_processing_wall_ns;
    }

    void validate_inputs_() const {
        const int n = static_cast<int>(problem_.lower_bounds.size());
        if (n <= 0) {
            throw std::invalid_argument("simplex::bnb: problem must contain at least one variable");
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
            throw std::invalid_argument("simplex::bnb: feasibility_jump_iterations must be >= 0");
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
            throw std::invalid_argument("simplex::bnb: probing_max_candidates must be >= 0");
        }
        if (options_.max_conflict_cuts_per_round < 0) {
            throw std::invalid_argument("simplex::bnb: max_conflict_cuts_per_round must be >= 0");
        }
        if (options_.max_root_cut_rounds < 0) {
            throw std::invalid_argument("simplex::bnb: max_root_cut_rounds must be >= 0");
        }
        if (options_.max_root_cuts_added_per_round < 0) {
            throw std::invalid_argument("simplex::bnb: max_root_cuts_added_per_round must be >= 0");
        }
        if (options_.max_cuts_per_type < 0) {
            throw std::invalid_argument("simplex::bnb: max_cuts_per_type must be >= 0");
        }
        if (options_.proof_strong_branching_candidates < 0) {
            throw std::invalid_argument(
                "simplex::bnb: proof_strong_branching_candidates must be >= 0");
        }
        if (options_.proof_phase_min_nodes < 0) {
            throw std::invalid_argument("simplex::bnb: proof_phase_min_nodes must be >= 0");
        }
        if (options_.proof_strong_branching_k < 0) {
            throw std::invalid_argument("simplex::bnb: proof_strong_branching_k must be >= 0");
        }
        if (options_.proof_strong_branching_max_depth < 0) {
            throw std::invalid_argument(
                "simplex::bnb: proof_strong_branching_max_depth must be >= 0");
        }
        if (options_.proof_max_cut_rounds_per_node < 0) {
            throw std::invalid_argument("simplex::bnb: proof_max_cut_rounds_per_node must be >= 0");
        }
        if (options_.proof_max_cuts_added_per_round < 0) {
            throw std::invalid_argument(
                "simplex::bnb: proof_max_cuts_added_per_round must be >= 0");
        }
        if (options_.cut_max_parallelism < 0.0 || options_.cut_max_parallelism > 1.0) {
            throw std::invalid_argument("simplex::bnb: cut_max_parallelism must be in [0, 1]");
        }
    }

    void reset_state_() {
        stop_async_heuristic_workers_();
        search_coordinator_.configure(options_.parallel_workers);
        search_coordinator_.reset();
        {
            std::lock_guard<std::mutex> lock(cuts_mutex_);
            cut_pool_.reset(options_);
            active_cuts_ = initial_cuts_;
            active_cut_signatures_.clear();
            for (const auto& cut : active_cuts_) {
                active_cut_signatures_.insert(detail::cut_signature(cut));
            }
        }
        {
            std::lock_guard<std::mutex> lock(learning_mutex_);
            learned_conflicts_.clear();
            learned_implications_.assign(2 * problem_.variable_types.size(), {});
        }
        {
            std::lock_guard<std::mutex> lock(tree_mutex_);
            tree_nodes_.clear();
            next_order_ = 0;
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            node_count_ = 0;
            relaxation_solve_count_ = 0;
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
            warm_start_relaxation_attempt_count_ = 0;
            warm_start_relaxation_accept_count_ = 0;
            warm_start_cold_retry_count_ = 0;
            warm_start_relaxation_solve_count_ = 0;
            strong_branching_probe_count_ = 0;
            strong_branching_probe_iterations_ = 0;
            lp_refactorizations_ = 0;
            lp_eta_stack_depth_entry_sum_ = 0;
            lp_dual_pool_builds_ = 0;
            lp_primal_pool_builds_ = 0;
            lp_warm_factorization_reuse_count_ = 0;
            lp_warm_dual_weights_reuse_count_ = 0;
            relaxation_core_solve_time_ns_ = 0;
            relaxation_lp_assembly_time_ns_ = 0;
            relaxation_lp_internal_presolve_ns_ = 0;
            relaxation_lp_internal_crash_ns_ = 0;
            relaxation_lp_internal_iters_ns_ = 0;
            relaxation_lp_internal_serialize_ns_ = 0;
            relaxation_lp_lu_build_ns_ = 0;
            relaxation_lp_pricing_build_ns_ = 0;
            relaxation_lp_pivot_ns_ = 0;
            strong_branching_probe_core_solve_time_ns_ = 0;
            strong_branching_probe_lp_assembly_time_ns_ = 0;
            strong_branching_probe_lp_internal_presolve_ns_ = 0;
            strong_branching_probe_lp_internal_crash_ns_ = 0;
            strong_branching_probe_lp_internal_iters_ns_ = 0;
            strong_branching_probe_lp_internal_serialize_ns_ = 0;
            root_cut_generation_wall_ns_ = 0;
            root_cut_selection_wall_ns_ = 0;
            root_cut_activation_wall_ns_ = 0;
            root_cut_resolve_wall_ns_ = 0;
            node_cut_generation_wall_ns_ = 0;
            node_cut_selection_wall_ns_ = 0;
            node_cut_resolve_wall_ns_ = 0;
            rounding_heuristic_wall_ns_ = 0;
            heuristics_wall_ns_ = 0;
            feasibility_jump_wall_ns_ = 0;
            feasibility_pump_wall_ns_ = 0;
            diving_wall_ns_ = 0;
            rens_wall_ns_ = 0;
            rins_wall_ns_ = 0;
            local_search_wall_ns_ = 0;
            local_branching_wall_ns_ = 0;
            branching_wall_ns_ = 0;
            child_processing_wall_ns_ = 0;
            diving_stats_.assign(5, detail::DivingStrategyStats{});
            pseudocosts_.assign(problem_.variable_types.size(), {});
        }
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            root_warm_start_basis_state_.reset();
            has_incumbent_ = false;
            incumbent_objective_ = problem_.maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
            incumbent_primal_ = Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                                          std::numeric_limits<double>::quiet_NaN());
            root_relaxation_objective.reset();
            root_reduced_costs_.resize(0);
            root_basis_statuses_.clear();
            root_lp_objective_ = std::numeric_limits<double>::quiet_NaN();
        }
        {
            std::lock_guard<std::mutex> lock(progress_mutex_);
            progress_header_printed_ = false;
            last_logged_node_count_ = 0;
            last_logged_best_bound_ = std::numeric_limits<double>::quiet_NaN();
            last_logged_incumbent_ = problem_.maximize ? -std::numeric_limits<double>::infinity()
                                                       : std::numeric_limits<double>::infinity();
            last_logged_gap_ = std::numeric_limits<double>::quiet_NaN();
        }
        {
            std::lock_guard<std::mutex> lock(async_heuristic_completion_mutex_);
            async_heuristic_completions_.clear();
        }
        global_domain_.reset(problem_.lower_bounds, problem_.upper_bounds);
    }

    detail::ActiveNode make_root_node_() {
        std::lock_guard<std::mutex> lock(tree_mutex_);
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
        return problem_.maximize ? (candidate > incumbent + options_.feasibility_tol)
                                 : (candidate < incumbent - options_.feasibility_tol);
    }

    // Optimum-preserving fathoming cutoff (HiGHS HighsMipSolverData::upper_limit
    // with mip_abs_gap = mip_rel_gap = 0). A node is pruned only when its LP
    // bound is strictly worse than this cutoff, so any integer-feasible
    // solution at least as good as the incumbent remains in the search tree.
    // For minimization: cutoff = incumbent - feasibility_tol
    // For maximization: cutoff = incumbent + feasibility_tol
    double fathom_cutoff_(double incumbent) const {
        return problem_.maximize ? (incumbent + options_.feasibility_tol)
                                 : (incumbent - options_.feasibility_tol);
    }

    // Gap-aware "optimality limit" (HiGHS optimality_limit). Used only to
    // decide when the requested optimality gap has been proved -- never for
    // pruning individual nodes, since that would discard the true optimum
    // whenever a better integer solution exists within the gap window.
    double optimality_limit_(double incumbent) const {
        const double scale = std::max(1.0, std::abs(incumbent));
        const double slack = std::max(options_.mip_abs_gap, options_.mip_rel_gap * scale);
        return problem_.maximize ? (incumbent + slack) : (incumbent - slack);
    }

    bool bound_prunes_(double candidate, double incumbent) const {
        // Always use the optimum-preserving cutoff for fathoming. The
        // use_gap_aware_cutoff toggle only controls the IncumbentCutoff LP cut
        // and does NOT widen this fathoming criterion -- doing so would lose
        // the optimum on instances where a better integer solution lies within
        // the requested gap window.
        const double cutoff = fathom_cutoff_(incumbent);
        return problem_.maximize ? (candidate <= cutoff) : (candidate >= cutoff);
    }

    bool mip_gap_closed_(double bound) const {
        const IncumbentSnapshot incumbent = incumbent_snapshot_();
        if (!incumbent.has_incumbent || !std::isfinite(incumbent.objective) ||
            !std::isfinite(bound)) {
            return false;
        }
        const double raw_gap =
            problem_.maximize ? (bound - incumbent.objective) : (incumbent.objective - bound);
        if (raw_gap <= options_.mip_abs_gap) {
            return true;
        }
        const double scale = std::max(1.0, std::abs(incumbent.objective));
        return (raw_gap / scale) <= options_.mip_rel_gap;
    }

    bool adaptive_proof_phase_active_() const {
        if (!options_.use_adaptive_proof_phase) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (node_count_ < options_.proof_phase_min_nodes) {
                return false;
            }
        }
        const IncumbentSnapshot incumbent = incumbent_snapshot_();
        if (!incumbent.has_incumbent || !std::isfinite(incumbent.objective)) {
            return false;
        }
        if (search_coordinator_.empty()) {
            return false;
        }
        const double best_bound =
            search_coordinator_.compute_best_bound(incumbent.has_incumbent, incumbent.objective,
                                                   problem_.maximize, root_relaxation_objective);
        return !mip_gap_closed_(best_bound);
    }

    bool adaptive_proof_phase_active_for_bound_(double bound) const {
        if (!options_.use_adaptive_proof_phase || !std::isfinite(bound)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            if (node_count_ < options_.proof_phase_min_nodes) {
                return false;
            }
        }
        const IncumbentSnapshot incumbent = incumbent_snapshot_();
        if (!incumbent.has_incumbent || !std::isfinite(incumbent.objective)) {
            return false;
        }
        return !mip_gap_closed_(bound);
    }

    Options proof_options_() const {
        Options out = options_;
        out.node_selection = options_.proof_node_selection;
        // Pseudocost mode still performs strong-branching probes for unreliable
        // candidates up to `strong_branching_max_depth`. The dedicated
        // StrongBranching strategy is intentionally root-heavy, so proof mode
        // uses the pseudocost path with larger probing budgets instead.
        out.branching_strategy = BranchingStrategy::PseudoCost;
        out.strong_branching_candidates = std::max(options_.strong_branching_candidates,
                                                   options_.proof_strong_branching_candidates);
        out.strong_branching_k =
            std::max(options_.strong_branching_k, options_.proof_strong_branching_k);
        out.strong_branching_max_depth = std::max(options_.strong_branching_max_depth,
                                                  options_.proof_strong_branching_max_depth);
        out.max_cut_rounds_per_node =
            std::max(options_.max_cut_rounds_per_node, options_.proof_max_cut_rounds_per_node);
        out.max_cuts_added_per_round =
            std::max(options_.max_cuts_added_per_round, options_.proof_max_cuts_added_per_round);
        out.use_node_presolve_on_warm_basis = options_.use_node_presolve_on_warm_basis ||
                                              options_.proof_use_node_presolve_on_warm_basis;
        return out;
    }

    Options effective_options_() const {
        return adaptive_proof_phase_active_() ? proof_options_() : options_;
    }

    Options effective_options_for_bound_(double bound) const {
        return adaptive_proof_phase_active_for_bound_(bound) ? proof_options_() : options_;
    }

    std::vector<Cut> current_relaxation_cuts_snapshot_() const {
        std::vector<Cut> cuts;
        {
            std::lock_guard<std::mutex> lock(cuts_mutex_);
            cuts = active_cuts_;
        }

        bool has_incumbent = false;
        double incumbent_objective = std::numeric_limits<double>::quiet_NaN();
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            has_incumbent = has_incumbent_;
            incumbent_objective = incumbent_objective_;
        }
        if (!has_incumbent || !std::isfinite(incumbent_objective) ||
            problem_.objective_coefficients.size() == 0) {
            return cuts;
        }

        Cut cutoff;
        cutoff.cut_type = "IncumbentCutoff";
        cutoff.sense = problem_.maximize ? LinearConstraintSense::GreaterEqual
                                         : LinearConstraintSense::LessEqual;
        // Use the optimum-preserving fathoming cutoff so the LP-side cut and
        // bound_prunes_ agree on the same threshold. Mirrors HiGHS, which
        // applies its zero-gap upper_limit to the LP objective bound and never
        // widens the cut by the user's gap tolerance.
        cutoff.rhs = fathom_cutoff_(incumbent_objective) - problem_.objective_constant;

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

    bool contains_incumbent_cutoff_(const std::vector<Cut>& cuts) const {
        for (const auto& cut : cuts) {
            if (cut.cut_type == "IncumbentCutoff") {
                return true;
            }
        }
        return false;
    }

    IncumbentSnapshot incumbent_snapshot_() const {
        std::lock_guard<std::mutex> lock(incumbent_mutex_);
        IncumbentSnapshot snapshot;
        snapshot.has_incumbent = has_incumbent_;
        snapshot.objective = incumbent_objective_;
        snapshot.primal = incumbent_primal_;
        return snapshot;
    }

    std::vector<detail::PseudoCost> pseudocosts_snapshot_() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return pseudocosts_;
    }

    std::vector<LearnedConflict> learned_conflicts_snapshot_() const {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        return learned_conflicts_;
    }

    std::vector<std::vector<ConflictLiteral>> learned_implications_snapshot_() const {
        std::lock_guard<std::mutex> lock(learning_mutex_);
        return learned_implications_;
    }

    const detail::ConflictGraph* conflict_graph_() const {
        std::call_once(conflict_graph_once_, [&]() {
            const bool has_binary =
                std::any_of(problem_.variable_types.begin(), problem_.variable_types.end(),
                            [](VariableType type) { return type == VariableType::Binary; });
            if (has_binary) {
                conflict_graph_cache_ = std::make_unique<detail::ConflictGraph>(problem_);
            }
        });
        return conflict_graph_cache_.get();
    }

    std::vector<detail::DivingStrategyStats> diving_stats_snapshot_() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return diving_stats_;
    }

    static void tighten_discrete_bounds_(VariableType type, double* lower, double* upper,
                                         double tol) {
        if (type == VariableType::Continuous)
            return;
        *lower = std::ceil(*lower - tol);
        *upper = std::floor(*upper + tol);
        if (type == VariableType::Binary) {
            *lower = std::max(0.0, *lower);
            *upper = std::min(1.0, *upper);
        }
    }

    std::optional<int> fixed_binary_literal_from_bounds_(int variable, const Eigen::VectorXd& lower,
                                                         const Eigen::VectorXd& upper) const {
        return ConflictEngine::fixed_binary_literal_from_bounds(*this, variable, lower, upper);
    }

    NodeReasonStore* ensure_reason_store_mutable_(std::shared_ptr<NodeReasonStore>* reasons,
                                                  int required_size = 0) const {
        if (reasons == nullptr)
            return nullptr;
        if (*reasons == nullptr) {
            *reasons = std::make_shared<NodeReasonStore>(required_size);
            return reasons->get();
        }
        if (required_size > 0 && static_cast<int>((*reasons)->size()) != required_size) {
            *reasons = std::make_shared<NodeReasonStore>(required_size);
            return reasons->get();
        }
        if (!(*reasons).unique()) {
            *reasons = std::make_shared<NodeReasonStore>(**reasons);
        }
        return reasons->get();
    }

    void seed_fixed_literal_reason_(int literal, std::shared_ptr<NodeReasonStore>* reasons) const {
        if (reasons == nullptr || *reasons == nullptr || literal < 0 ||
            literal >= static_cast<int>((*reasons)->size())) {
            return;
        }
        NodeReasonStore* mutable_reasons = ensure_reason_store_mutable_(reasons);
        if (mutable_reasons == nullptr || literal >= static_cast<int>(mutable_reasons->size())) {
            return;
        }
        PropagationReason& reason = (*mutable_reasons)[literal];
        reason.parent_literal = literal;
        reason.row_index = -1;
        reason.antecedents.clear();
    }

    void enqueue_fixed_binary_literal_(int variable, const Eigen::VectorXd& lower,
                                       const Eigen::VectorXd& upper, std::vector<char>* seen,
                                       std::vector<int>* queue,
                                       std::shared_ptr<NodeReasonStore>* reasons = nullptr,
                                       int parent_literal = -1,
                                       [[maybe_unused]] bool allow_global = true) const {
        ConflictEngine::enqueue_fixed_binary_literal(const_cast<Solver&>(*this), variable, lower,
                                                     upper, seen, queue, reasons, parent_literal,
                                                     allow_global);
    }

    std::vector<ConflictLiteral> conflict_literals_from_binary_literals_(int lhs, int rhs) const {
        return ConflictEngine::conflict_literals_from_binary_literals(*this, lhs, rhs);
    }

    ConflictLiteral conflict_literal_from_binary_literal_(int literal) const {
        return ConflictEngine::conflict_literal_from_binary_literal(*this, literal);
    }

    std::optional<int>
    exact_binary_literal_from_conflict_literal_(const ConflictLiteral& literal) const {
        return ConflictEngine::exact_binary_literal_from_conflict_literal(*this, literal);
    }

    int resolve_reason_literal_(int literal, const std::vector<PropagationReason>& reasons) const {
        return ConflictEngine::resolve_reason_literal(*this, literal, reasons);
    }

    std::vector<ConflictLiteral>
    minimize_conflict_with_reasons_(const std::vector<ConflictLiteral>& literals,
                                    const std::vector<PropagationReason>& reasons) const {
        return ConflictEngine::minimize_conflict_with_reasons(*this, literals, reasons);
    }

    std::vector<ConflictLiteral>
    explain_row_fixing_literal_(const std::vector<int>& indices, const std::vector<double>& values,
                                double rhs, LinearConstraintSense sense, int literal,
                                const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) const {
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
        const bool feasible =
            propagate_row_bounds_(indices, values, rhs, sense, &probe_lower, &probe_upper,
                                  &tightened_bounds, nullptr, &conflict_literals);
        if (feasible || conflict_literals.empty())
            return {};

        conflict_literals.erase(std::remove_if(conflict_literals.begin(), conflict_literals.end(),
                                               [&](const ConflictLiteral& candidate) {
                                                   return candidate.variable == variable &&
                                                          candidate.is_lower ==
                                                              detail::ConflictGraph::value_of(
                                                                  literal);
                                               }),
                                conflict_literals.end());
        return conflict_literals;
    }

    std::vector<ConflictLiteral>
    minimize_conflict_with_row_reasons_(const std::vector<ConflictLiteral>& literals,
                                        const std::vector<PropagationReason>& reasons) const {
        return ConflictEngine::minimize_conflict_with_row_reasons(*this, literals, reasons);
    }

    void learn_reasoned_binary_conflict_(int trigger_literal, int contradiction_literal,
                                         const std::vector<PropagationReason>& reasons,
                                         bool allow_global = true) {
        ConflictEngine::learn_reasoned_binary_conflict(
            *this, trigger_literal, contradiction_literal, reasons, allow_global);
    }

    void learn_implication_unlocked_(int trigger_literal, const ConflictLiteral& consequence) {
        ConflictEngine::learn_implication_unlocked(*this, trigger_literal, consequence);
    }

    bool apply_literal_implications_(
        const detail::ConflictGraph* graph,
        const std::vector<std::vector<ConflictLiteral>>& learned_implications,
        Eigen::VectorXd* lower, Eigen::VectorXd* upper, int* tightened_bounds,
        std::vector<char>* queued_literals, std::vector<int>* literal_queue,
        int* literal_queue_head, std::vector<int>* changed_variables,
        std::shared_ptr<NodeReasonStore>* reasons, bool allow_global = true) {
        return ConflictEngine::apply_literal_implications(
            *this, graph, learned_implications, lower, upper, tightened_bounds, queued_literals,
            literal_queue, literal_queue_head, changed_variables, reasons, allow_global);
    }

    std::vector<ConflictLiteral> explain_leq_row_conflict_(const std::vector<int>& indices,
                                                           const std::vector<double>& values,
                                                           double rhs, const Eigen::VectorXd& lower,
                                                           const Eigen::VectorXd& upper) const {
        return ConflictEngine::explain_leq_row_conflict(*this, indices, values, rhs, lower, upper);
    }

    void learn_conflict_literals_(const std::vector<ConflictLiteral>& literals,
                                  bool allow_global = true) {
        ConflictEngine::learn_conflict_literals(*this, literals, allow_global);
    }

    void learn_implication_(int trigger_literal, const ConflictLiteral& consequence) {
        if (trigger_literal < 0)
            return;
        std::lock_guard<std::mutex> lock(learning_mutex_);
        learn_implication_unlocked_(trigger_literal, consequence);
    }

    void learn_probing_implications_(int trigger_literal, const Eigen::VectorXd& base_lower,
                                     const Eigen::VectorXd& base_upper,
                                     const NodePresolveOutcome& presolved) {
        if (trigger_literal < 0 || presolved.infeasible)
            return;
        for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) &&
                        j < base_lower.size() && j < base_upper.size() &&
                        j < presolved.lower_bounds.size() && j < presolved.upper_bounds.size();
             ++j) {
            if (problem_.variable_types[j] == VariableType::Continuous)
                continue;
            if (base_lower(j) + options_.integrality_tol < presolved.lower_bounds(j)) {
                learn_implication_(trigger_literal,
                                   ConflictLiteral{j, true, presolved.lower_bounds(j)});
            }
            if (base_upper(j) - options_.integrality_tol > presolved.upper_bounds(j)) {
                learn_implication_(trigger_literal,
                                   ConflictLiteral{j, false, presolved.upper_bounds(j)});
            }
        }
    }

    bool
    apply_leq_row_propagation_(const std::vector<int>& indices, const std::vector<double>& values,
                               double rhs, Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                               int* tightened_bounds, std::vector<int>* changed_variables = nullptr,
                               std::vector<ConflictLiteral>* conflict_literals = nullptr) const {
        if (!lower || !upper || !tightened_bounds)
            return true;

        auto accumulate_activity = [&](bool use_upper_for_negative) {
            double activity = 0.0;
            bool finite = true;
            for (int k = 0;
                 k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size()); ++k) {
                const int index = indices[k];
                const double coeff = values[k];
                if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12)
                    continue;

                const double bound =
                    coeff >= 0.0 ? (*lower)(index)
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

        for (int pivot = 0;
             pivot < static_cast<int>(indices.size()) && pivot < static_cast<int>(values.size());
             ++pivot) {
            const int index = indices[pivot];
            const double coeff = values[pivot];
            if (index < 0 || index >= lower->size() || std::abs(coeff) <= 1e-12)
                continue;

            double other_min = 0.0;
            bool other_min_finite = true;
            for (int k = 0;
                 k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size()); ++k) {
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
                if (std::isfinite(candidate) && candidate < new_upper - options_.integrality_tol) {
                    new_upper = candidate;
                }
            } else {
                const double candidate = (rhs - other_min) / coeff;
                if (std::isfinite(candidate) && candidate > new_lower + options_.integrality_tol) {
                    new_lower = candidate;
                }
            }

            tighten_discrete_bounds_(problem_.variable_types[index], &new_lower, &new_upper,
                                     options_.integrality_tol);
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

    bool propagate_row_bounds_(const std::vector<int>& indices, const std::vector<double>& values,
                               double rhs, LinearConstraintSense sense, Eigen::VectorXd* lower,
                               Eigen::VectorXd* upper, int* tightened_bounds,
                               std::vector<int>* changed_variables = nullptr,
                               std::vector<ConflictLiteral>* conflict_literals = nullptr) const {
        if (sense == LinearConstraintSense::LessEqual) {
            return apply_leq_row_propagation_(indices, values, rhs, lower, upper, tightened_bounds,
                                              changed_variables, conflict_literals);
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
        return apply_leq_row_propagation_(indices, values, rhs, lower, upper, tightened_bounds,
                                          changed_variables, conflict_literals) &&
               apply_leq_row_propagation_(indices, negated, -rhs, lower, upper, tightened_bounds,
                                          changed_variables, conflict_literals);
    }

    bool conflict_applies_(const LearnedConflict& conflict, const Eigen::VectorXd& lower,
                           const Eigen::VectorXd& upper) const {
        for (const ConflictLiteral& literal : conflict.literals) {
            if (literal.variable < 0 || literal.variable >= lower.size())
                return false;
            if (literal.is_lower) {
                if (lower(literal.variable) + options_.integrality_tol < literal.value) {
                    return false;
                }
            } else {
                if (upper(literal.variable) - options_.integrality_tol > literal.value) {
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<ConflictLiteral>
    conflict_literals_from_bounds_(const Eigen::VectorXd& lower,
                                   const Eigen::VectorXd& upper) const {
        std::vector<ConflictLiteral> literals;
        literals.reserve(problem_.variable_types.size());
        for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) && j < lower.size() &&
                        j < upper.size();
             ++j) {
            if (problem_.variable_types[j] == VariableType::Continuous)
                continue;

            double lower_value = lower(j);
            double upper_value = upper(j);
            tighten_discrete_bounds_(problem_.variable_types[j], &lower_value, &upper_value,
                                     options_.integrality_tol);

            if (lower_value > problem_.lower_bounds(j) + options_.integrality_tol) {
                literals.push_back(ConflictLiteral{j, true, lower_value});
            }
            if (upper_value < problem_.upper_bounds(j) - options_.integrality_tol) {
                literals.push_back(ConflictLiteral{j, false, upper_value});
            }
        }

        std::sort(literals.begin(), literals.end(),
                  [](const ConflictLiteral& lhs, const ConflictLiteral& rhs) {
                      if (lhs.variable != rhs.variable)
                          return lhs.variable < rhs.variable;
                      if (lhs.is_lower != rhs.is_lower)
                          return lhs.is_lower < rhs.is_lower;
                      return lhs.value < rhs.value;
                  });
        literals.erase(std::unique(literals.begin(), literals.end(),
                                   [&](const ConflictLiteral& lhs, const ConflictLiteral& rhs) {
                                       return lhs.variable == rhs.variable &&
                                              lhs.is_lower == rhs.is_lower &&
                                              same_progress_value_(lhs.value, rhs.value);
                                   }),
                       literals.end());
        if (literals.size() > 16) {
            literals.clear();
        }
        return literals;
    }

    void maybe_learn_conflict_from_bounds_(const Eigen::VectorXd& lower,
                                           const Eigen::VectorXd& upper, bool allow_global = true) {
        const std::vector<ConflictLiteral> literals = conflict_literals_from_bounds_(lower, upper);
        learn_conflict_literals_(literals, allow_global);
    }

    NodePresolveOutcome
    presolve_node_bounds_(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                          const std::vector<Cut>& cuts,
                          const std::shared_ptr<NodeReasonStore>& initial_reasons = nullptr) {
        struct RowRef {
            const std::vector<int>* indices = nullptr;
            const std::vector<double>* values = nullptr;
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
            if (out.upper_bounds(j) + options_.integrality_tol < out.lower_bounds(j)) {
                out.infeasible = true;
                return out;
            }
        }

        std::vector<RowRef> rows;
        rows.reserve(problem_.base_constraints.size() + cuts.size());
        std::vector<std::vector<int>> column_to_rows(out.lower_bounds.size());

        auto register_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                                double rhs, LinearConstraintSense sense) {
            const int row_index = static_cast<int>(rows.size());
            rows.push_back(RowRef{&indices, &values, rhs, sense});
            for (int k = 0;
                 k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size()); ++k) {
                const int index = indices[k];
                if (index < 0 || index >= static_cast<int>(column_to_rows.size()) ||
                    std::abs(values[k]) <= 1e-12) {
                    continue;
                }
                column_to_rows[index].push_back(row_index);
            }
        };

        for (const SparseLinearConstraint& row : problem_.base_constraints) {
            register_row(row.indices, row.values, row.rhs, row.sense);
        }
        for (const Cut& cut : cuts) {
            register_row(cut.indices, cut.values, cut.rhs, cut.sense);
        }

        const bool allow_global_conflict_learning = !contains_incumbent_cutoff_(cuts);
        const detail::ConflictGraph* graph = conflict_graph_();
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
            for (int j = 0; j < static_cast<int>(problem_.variable_types.size()) &&
                            j < out.lower_bounds.size() && j < out.upper_bounds.size();
                 ++j) {
                enqueue_fixed_binary_literal_(j, out.lower_bounds, out.upper_bounds,
                                              &queued_literals, &literal_queue, &reasons);
            }
        }

        const std::vector<LearnedConflict> learned_conflicts = learned_conflicts_snapshot_();
        for (const LearnedConflict& conflict : learned_conflicts) {
            if (conflict_applies_(conflict, out.lower_bounds, out.upper_bounds)) {
                out.infeasible = true;
                return out;
            }
        }

        if (graph != nullptr) {
            std::vector<int> graph_changed;
            if (!apply_literal_implications_(
                    graph, learned_implications, &out.lower_bounds, &out.upper_bounds,
                    &out.tightened_bounds, &queued_literals, &literal_queue, &literal_queue_head,
                    &graph_changed, &reasons, allow_global_conflict_learning)) {
                out.infeasible = true;
                return out;
            }
        }

        std::vector<char> row_queued(rows.size(), 1);
        std::vector<int> row_queue(rows.size(), 0);
        int queue_head = 0;
        for (int row_index = 0; row_index < static_cast<int>(rows.size()); ++row_index) {
            row_queue[row_index] = row_index;
        }

        while (queue_head < static_cast<int>(row_queue.size())) {
            const int row_index = row_queue[queue_head++];
            row_queued[row_index] = 0;
            const RowRef& row = rows[row_index];
            std::vector<int> changed_variables;
            std::vector<ConflictLiteral> row_conflict;
            if (!propagate_row_bounds_(*row.indices, *row.values, row.rhs, row.sense,
                                       &out.lower_bounds, &out.upper_bounds, &out.tightened_bounds,
                                       &changed_variables, &row_conflict)) {
                out.infeasible = true;
                learn_conflict_literals_(
                    minimize_conflict_with_row_reasons_(
                        row_conflict, reasons != nullptr ? *reasons : NodeReasonStore{}),
                    allow_global_conflict_learning);
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
                    enqueue_fixed_binary_literal_(index, out.lower_bounds, out.upper_bounds,
                                                  &queued_literals, &literal_queue, &reasons, -1,
                                                  allow_global_conflict_learning);
                    const std::optional<int> fixed_literal = fixed_binary_literal_from_bounds_(
                        index, out.lower_bounds, out.upper_bounds);
                    if (fixed_literal.has_value() && reasons != nullptr && *fixed_literal >= 0 &&
                        *fixed_literal < static_cast<int>(reasons->size())) {
                        NodeReasonStore* mutable_reasons =
                            ensure_reason_store_mutable_(&reasons, graph->literal_count());
                        if (mutable_reasons != nullptr &&
                            (*mutable_reasons)[*fixed_literal].row_index < 0 &&
                            (*mutable_reasons)[*fixed_literal].antecedents.empty()) {
                            (*mutable_reasons)[*fixed_literal].row_index = row_index;
                            (*mutable_reasons)[*fixed_literal].antecedents =
                                explain_row_fixing_literal_(*row.indices, *row.values, row.rhs,
                                                            row.sense, *fixed_literal,
                                                            out.lower_bounds, out.upper_bounds);
                        }
                    }
                }
            }

            if (graph != nullptr) {
                std::vector<int> graph_changed;
                if (!apply_literal_implications_(graph, learned_implications, &out.lower_bounds,
                                                 &out.upper_bounds, &out.tightened_bounds,
                                                 &queued_literals, &literal_queue,
                                                 &literal_queue_head, &graph_changed, &reasons,
                                                 allow_global_conflict_learning)) {
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
                for (const LearnedConflict& conflict : learned_conflicts) {
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
        const IncumbentSnapshot incumbent = incumbent_snapshot_();
        if (!incumbent.has_incumbent || !std::isfinite(incumbent.objective) ||
            !std::isfinite(bound)) {
            return std::numeric_limits<double>::infinity();
        }
        const double raw_gap =
            problem_.maximize ? (bound - incumbent.objective) : (incumbent.objective - bound);
        if (raw_gap <= options_.integrality_tol) {
            return 0.0;
        }
        const double scale = std::max(1.0, std::abs(incumbent.objective));
        return raw_gap / scale;
    }

    std::vector<Cut>
    generate_probing_implied_bound_cuts_(const detail::ActiveNode& node,
                                         const RelaxationSolution& relaxation,
                                         const std::vector<detail::FractionalCandidate>& fractional,
                                         const std::vector<Cut>& relaxation_cuts) {
        std::vector<Cut> cuts;
        if (!options_.use_probing_implications || options_.probing_max_candidates == 0 ||
            relaxation.status != RelaxationStatus::Optimal || fractional.empty()) {
            return cuts;
        }

        const bool shallow = node.depth <= 2;
        const bool periodic =
            options_.heuristic_frequency <= 1 ||
            (options_.heuristic_frequency > 0 &&
             (node.order % static_cast<std::uint64_t>(options_.heuristic_frequency) == 0));
        if (!shallow && !periodic) {
            return cuts;
        }

        std::unordered_set<detail::CutSignature, detail::CutSignatureHash> signatures;
        int probes = 0;
        for (const auto& candidate : fractional) {
            if (probes >= options_.probing_max_candidates)
                break;
            if (candidate.variable < 0 ||
                candidate.variable >= static_cast<int>(problem_.variable_types.size()) ||
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
                    const double violation = detail::cut_violation(cut, relaxation.primal);
                    if (violation > options_.min_cut_violation) {
                        cut.strength = violation;
                        const detail::CutSignature signature = detail::cut_signature(cut);
                        if (!signatures.contains(signature)) {
                            signatures.insert(signature);
                            cuts.push_back(std::move(cut));
                        }
                    }
                    continue;
                }

                learn_probing_implications_(
                    detail::ConflictGraph::literal_for(candidate.variable, fixed_value == 1),
                    node.lower_bounds, node.upper_bounds, presolved);

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
                        cut.values = {1.0, fixed_value == 1 ? (base_upper - tightened_upper)
                                                            : -(base_upper - tightened_upper)};
                        cut.sense = LinearConstraintSense::LessEqual;
                        cut.rhs = fixed_value == 1 ? base_upper : tightened_upper;
                        const double violation = detail::cut_violation(cut, relaxation.primal);
                        if (violation > options_.min_cut_violation) {
                            cut.strength = violation;
                            const detail::CutSignature signature = detail::cut_signature(cut);
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
                        cut.values = {1.0, fixed_value == 1 ? -(tightened_lower - base_lower)
                                                            : (tightened_lower - base_lower)};
                        cut.sense = LinearConstraintSense::GreaterEqual;
                        cut.rhs = fixed_value == 1 ? base_lower : tightened_lower;
                        const double violation = detail::cut_violation(cut, relaxation.primal);
                        if (violation > options_.min_cut_violation) {
                            cut.strength = violation;
                            const detail::CutSignature signature = detail::cut_signature(cut);
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

    std::vector<Cut> generate_conflict_cuts_(const RelaxationSolution& relaxation) {
        std::vector<Cut> cuts;
        if (!options_.use_conflict_cuts || options_.max_conflict_cuts_per_round == 0 ||
            relaxation.status != RelaxationStatus::Optimal) {
            return cuts;
        }

        const detail::ConflictGraph* graph = conflict_graph_();
        const std::vector<LearnedConflict> conflicts = learned_conflicts_snapshot_();
        std::unordered_set<detail::CutSignature, detail::CutSignatureHash> signatures;
        // Track which pool entries produced a cut, so their age can be reset
        // (and hits incremented) after this round. Index matches `conflicts`.
        std::vector<char> produced_cut(conflicts.size(), 0);
        auto mark_cut_produced = [&](std::size_t idx) {
            if (idx < produced_cut.size())
                produced_cut[idx] = 1;
        };
        std::size_t conflict_index = static_cast<std::size_t>(-1);
        for (const LearnedConflict& conflict : conflicts) {
            ++conflict_index;
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
            for (const ConflictLiteral& literal : conflict.literals) {
                if (literal.variable < 0 ||
                    literal.variable >= static_cast<int>(problem_.variable_types.size()) ||
                    problem_.variable_types[literal.variable] != VariableType::Binary) {
                    valid = false;
                    break;
                }

                if (literal.is_lower && literal.value >= 1.0 - options_.integrality_tol) {
                    cut.indices.push_back(literal.variable);
                    cut.values.push_back(1.0);
                    base_literals.push_back(
                        detail::ConflictGraph::literal_for(literal.variable, true));
                } else if (!literal.is_lower && literal.value <= options_.integrality_tol) {
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
                for (int i = 0; i < static_cast<int>(base_literals.size()) && clique_conflict;
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
                    for (int i = 1;
                         i < static_cast<int>(base_literals.size()) && !candidates.empty(); ++i) {
                        const std::vector<int> neighbors = graph->neighbors(base_literals[i]);
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
                        std::remove_if(candidates.begin(), candidates.end(),
                                       [&](int literal) {
                                           return std::find(lifted_literals.begin(),
                                                            lifted_literals.end(),
                                                            literal) != lifted_literals.end() ||
                                                  used_variables.contains(
                                                      detail::ConflictGraph::variable_of(literal));
                                       }),
                        candidates.end());
                    std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
                        const double lhs_weight =
                            detail::ConflictGraph::literal_weight(relaxation.primal, lhs);
                        const double rhs_weight =
                            detail::ConflictGraph::literal_weight(relaxation.primal, rhs);
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

                    Cut lifted_cut = detail::clique_cut_from_literals(problem_, lifted_literals,
                                                                      options_, "ConflictClique");
                    if (!lifted_cut.indices.empty()) {
                        const double violation =
                            detail::cut_violation(lifted_cut, relaxation.primal);
                        if (violation > options_.min_cut_violation) {
                            lifted_cut.strength = violation;
                            const detail::CutSignature signature =
                                detail::cut_signature(lifted_cut);
                            if (!signatures.contains(signature)) {
                                signatures.insert(signature);
                                cuts.push_back(std::move(lifted_cut));
                                mark_cut_produced(conflict_index);
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
            const detail::CutSignature signature = detail::cut_signature(cut);
            if (signatures.contains(signature))
                continue;
            signatures.insert(signature);
            cuts.push_back(std::move(cut));
            mark_cut_produced(conflict_index);
        }

        // Age-based conflict-pool grooming (HiGHS-style). Entries that produced
        // a violated cut have their age reset and hits incremented; everything
        // else ages by one. Conflicts older than `max_conflict_age` without
        // ever producing a useful cut are evicted.
        {
            std::lock_guard<std::mutex> lock(learning_mutex_);
            const std::size_t n = std::min(produced_cut.size(), learned_conflicts_.size());
            for (std::size_t i = 0; i < n; ++i) {
                if (produced_cut[i]) {
                    learned_conflicts_[i].age = 0;
                    ++learned_conflicts_[i].hits;
                } else {
                    ++learned_conflicts_[i].age;
                }
            }
            const int age_limit = std::max(1, options_.max_conflict_age);
            learned_conflicts_.erase(std::remove_if(learned_conflicts_.begin(),
                                                    learned_conflicts_.end(),
                                                    [age_limit](const LearnedConflict& c) {
                                                        return c.hits == 0 && c.age > age_limit;
                                                    }),
                                     learned_conflicts_.end());
        }

        return cuts;
    }

    std::vector<Cut>
    generate_cut_candidates_(const detail::ActiveNode& node, const RelaxationSolution& relaxation,
                             const std::vector<detail::FractionalCandidate>& fractional,
                             const std::vector<Cut>& relaxation_cuts) {
        std::vector<Cut> cuts = detail::generate_cuts(problem_, relaxation, options_);
        std::vector<Cut> probing_cuts =
            generate_probing_implied_bound_cuts_(node, relaxation, fractional, relaxation_cuts);
        cuts.insert(cuts.end(), std::make_move_iterator(probing_cuts.begin()),
                    std::make_move_iterator(probing_cuts.end()));
        std::vector<Cut> conflict_cuts = generate_conflict_cuts_(relaxation);
        cuts.insert(cuts.end(), std::make_move_iterator(conflict_cuts.begin()),
                    std::make_move_iterator(conflict_cuts.end()));

        return cuts;
    }

    bool should_try_node_cuts_(const detail::ActiveNode& node, const RelaxationSolution& relaxation,
                               const std::vector<detail::FractionalCandidate>& fractional,
                               const Options& effective) const {
        if (!effective.use_cut_pool || node.depth <= 0 ||
            relaxation.status != RelaxationStatus::Optimal) {
            return false;
        }
        if ((!effective.use_gomory_cuts && !effective.use_mir_cuts && !effective.use_cover_cuts &&
             !effective.use_implied_bound_cuts && !effective.use_clique_cuts &&
             !effective.use_odd_cycle_cuts && !effective.use_probing_implications &&
             !effective.use_conflict_cuts && !effective.use_dual_proof_cuts) ||
            fractional.empty()) {
            return false;
        }
        if (node.depth > std::max(2, effective.strong_branching_max_depth + 1)) {
            return false;
        }

        const bool proof = adaptive_proof_phase_active_for_bound_(relaxation.objective);
        const bool shallow = node.depth <= 2;
        const bool periodic =
            effective.heuristic_frequency <= 1 ||
            (effective.heuristic_frequency > 0 &&
             (node.order % static_cast<std::uint64_t>(effective.heuristic_frequency) == 0));
        if (!proof && !shallow && !periodic) {
            return false;
        }
        if (fractional.size() > 24 && !shallow && !proof) {
            return false;
        }

        const double gap = relative_gap_from_bound_(relaxation.objective);
        if (!proof && !shallow && gap <= 0.05) {
            return false;
        }
        return proof || !std::isfinite(gap) || gap > 0.01 || shallow;
    }

    HeuristicSchedule build_heuristic_schedule_(const detail::ActiveNode& node,
                                                const RelaxationSolution& relaxation) const {
        HeuristicSchedule schedule;
        if (node.depth > options_.heuristic_max_depth) {
            return schedule;
        }

        const bool periodic =
            options_.heuristic_frequency <= 1 ||
            (options_.heuristic_frequency > 0 &&
             (node.order % static_cast<std::uint64_t>(options_.heuristic_frequency) == 0));
        const bool shallow = node.depth <= 2;
        const bool medium_depth = node.depth <= 6;
        const double gap = relative_gap_from_bound_(relaxation.objective);
        const bool has_incumbent = std::isfinite(gap);
        const bool large_gap = gap > 0.20;
        const bool medium_gap = gap > 0.05;

        if (options_.use_diving && options_.diving_strategy != DivingStrategy::Disabled) {
            schedule.run_diving =
                shallow || (!has_incumbent && medium_depth) || (large_gap && medium_depth) ||
                (medium_gap && (shallow || periodic)) || (!medium_gap && shallow && periodic);
        }

        schedule.run_feasibility_jump =
            options_.use_feasibility_jump &&
            (!has_incumbent ? (shallow || (node.depth <= 3 && periodic))
                            : (node.depth == 0 || (large_gap && shallow && periodic)));
        schedule.run_feasibility_pump =
            options_.use_feasibility_pump &&
            (!has_incumbent ? (shallow || (node.depth <= 4 && periodic))
                            : (node.depth == 0 || (large_gap && shallow && periodic)));
        schedule.run_rens = options_.use_rens &&
                            (!has_incumbent ? (shallow || (node.depth <= 3 && periodic))
                                            : ((large_gap || medium_gap) && (shallow || periodic)));
        schedule.run_rins = options_.use_rins && has_incumbent &&
                            ((large_gap && medium_depth) || (medium_gap && (shallow || periodic)) ||
                             (!medium_gap && shallow && periodic));
        schedule.run_local_search =
            options_.use_local_search && has_incumbent &&
            ((large_gap && medium_depth && periodic) ||
             (medium_gap && (node.depth <= 8) && periodic) || (!medium_gap && shallow && periodic));
        schedule.run_local_branching =
            options_.use_local_branching && has_incumbent &&
            ((large_gap && medium_depth) || (medium_gap && (node.depth <= 5) && periodic) ||
             (!medium_gap && shallow && periodic));
        return schedule;
    }
    void merge_pseudocosts_(const std::vector<detail::PseudoCost>& before,
                            const std::vector<detail::PseudoCost>& after) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        const int n = std::min<int>(std::min(before.size(), after.size()), pseudocosts_.size());

        for (int i = 0; i < n; ++i) {
            // Classical pseudocost statistics
            pseudocosts_[i].cost.up_sum += after[i].cost.up_sum - before[i].cost.up_sum;
            pseudocosts_[i].cost.down_sum += after[i].cost.down_sum - before[i].cost.down_sum;
            pseudocosts_[i].cost.up_count += after[i].cost.up_count - before[i].cost.up_count;
            pseudocosts_[i].cost.down_count += after[i].cost.down_count - before[i].cost.down_count;

            // HiGHS-style auxiliary signals
            pseudocosts_[i].signal.inference_up +=
                after[i].signal.inference_up - before[i].signal.inference_up;
            pseudocosts_[i].signal.inference_down +=
                after[i].signal.inference_down - before[i].signal.inference_down;

            pseudocosts_[i].signal.conflict_score_up +=
                after[i].signal.conflict_score_up - before[i].signal.conflict_score_up;
            pseudocosts_[i].signal.conflict_score_down +=
                after[i].signal.conflict_score_down - before[i].signal.conflict_score_down;

            pseudocosts_[i].signal.cutoff_up +=
                after[i].signal.cutoff_up - before[i].signal.cutoff_up;
            pseudocosts_[i].signal.cutoff_down +=
                after[i].signal.cutoff_down - before[i].signal.cutoff_down;
        }
    }

    void merge_diving_stats_(const std::vector<detail::DivingStrategyStats>& before,
                             const std::vector<detail::DivingStrategyStats>& after) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        const int n = std::min<int>(std::min(before.size(), after.size()), diving_stats_.size());
        for (int i = 0; i < n; ++i) {
            diving_stats_[i].attempts += after[i].attempts - before[i].attempts;
            diving_stats_[i].successes += after[i].successes - before[i].successes;
            diving_stats_[i].lp_iterations += after[i].lp_iterations - before[i].lp_iterations;
            diving_stats_[i].lp_solves += after[i].lp_solves - before[i].lp_solves;
        }
    }

    void note_lp_work_(const RelaxationSolution& relaxation) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        ++node_count_;
        ++relaxation_solve_count_;
        lp_iterations_ += relaxation.iterations;
        if (relaxation.attempted_warm_start_basis_state) {
            ++warm_start_relaxation_attempt_count_;
        }
        if (relaxation.used_warm_start_basis_state) {
            ++warm_start_relaxation_accept_count_;
            ++warm_start_relaxation_solve_count_;
        }
        if (relaxation.cold_retried_after_warm_start) {
            ++warm_start_cold_retry_count_;
        }
        relaxation_core_solve_time_ns_ += relaxation.core_solve_time_ns;
        relaxation_lp_assembly_time_ns_ += relaxation.lp_assembly_time_ns;
        relaxation_lp_internal_presolve_ns_ += relaxation.lp_internal_presolve_ns;
        relaxation_lp_internal_crash_ns_ += relaxation.lp_internal_crash_ns;
        relaxation_lp_internal_iters_ns_ += relaxation.lp_internal_iters_ns;
        relaxation_lp_internal_serialize_ns_ += relaxation.lp_internal_serialize_ns;
        if (relaxation.lp_solution.has_value()) {
            const LPSolveStats& lp_stats = relaxation.lp_solution->solve_stats;
            lp_refactorizations_ += lp_stats.refactorizations;
            lp_eta_stack_depth_entry_sum_ += lp_stats.eta_stack_depth_entry;
            lp_dual_pool_builds_ += lp_stats.dual_pool_builds;
            lp_primal_pool_builds_ += lp_stats.primal_pool_builds;
            lp_warm_factorization_reuse_count_ += lp_stats.warm_factorization_reused;
            lp_warm_dual_weights_reuse_count_ += lp_stats.warm_dual_weights_reused;
            relaxation_lp_lu_build_ns_ += lp_stats.lu_build_ns;
            relaxation_lp_pricing_build_ns_ += lp_stats.pricing_build_ns;
            relaxation_lp_pivot_ns_ += lp_stats.pivot_ns;
        }
        if (node_count_ >= options_.max_nodes) {
            search_coordinator_.mark_node_limit_reached();
        }
    }

    void note_strong_branching_work_(const detail::BranchDecision& decision) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        strong_branching_probe_count_ += decision.strong_branching_probe_count;
        strong_branching_probe_iterations_ += decision.strong_branching_probe_iterations;
        strong_branching_probe_core_solve_time_ns_ +=
            decision.strong_branching_probe_core_solve_time_ns;
        strong_branching_probe_lp_assembly_time_ns_ +=
            decision.strong_branching_probe_lp_assembly_time_ns;
        strong_branching_probe_lp_internal_presolve_ns_ +=
            decision.strong_branching_probe_lp_internal_presolve_ns;
        strong_branching_probe_lp_internal_crash_ns_ +=
            decision.strong_branching_probe_lp_internal_crash_ns;
        strong_branching_probe_lp_internal_iters_ns_ +=
            decision.strong_branching_probe_lp_internal_iters_ns;
        strong_branching_probe_lp_internal_serialize_ns_ +=
            decision.strong_branching_probe_lp_internal_serialize_ns;
    }

    void note_heuristic_result_(int lp_iterations, int successes) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        heuristic_lp_iterations_ += lp_iterations;
        heuristic_successes_ += successes;
    }

    void note_heuristic_family_successes_(int* counter, int successes) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        *counter += successes;
    }

    bool more_heuristics_allowed_() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        const long total_lp = std::max<long>(1, lp_iterations_);
        const long heuristic_lp = heuristic_lp_iterations_;
        if (node_count_ <= 4) {
            return heuristic_lp < total_lp / 3 + 1500;
        }
        const long search_lp = std::max<long>(0, total_lp - heuristic_lp);
        return heuristic_lp < 4000 + search_lp / 4;
    }

    bool async_heuristics_enabled_() const { return AsyncHeuristicManager::enabled(*this); }

    int async_heuristic_worker_count_() const { return AsyncHeuristicManager::worker_count(*this); }

    std::uint64_t async_heuristic_staleness_window_() const {
        return AsyncHeuristicManager::staleness_window();
    }

    int max_async_heuristic_tasks_() const { return AsyncHeuristicManager::max_tasks(*this); }

    void reap_async_heuristics_(bool wait_all = false) {
        drain_async_heuristic_completions_();
        AsyncHeuristicManager::reap(*this, wait_all);
        drain_async_heuristic_completions_();
    }

    void start_async_heuristic_workers_() { AsyncHeuristicManager::start_workers(*this); }

    void stop_async_heuristic_workers_() {
        AsyncHeuristicManager::stop_workers(*this);
        drain_async_heuristic_completions_();
    }

    template <typename Task>
    void dispatch_async_heuristic_(std::uint64_t launch_order, Task&& task) {
        AsyncHeuristicManager::dispatch(*this, launch_order, std::forward<Task>(task));
    }

    void mark_root_relaxation_(double objective) {
        bool first_root_relaxation = false;
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            if (!root_relaxation_objective.has_value()) {
                root_relaxation_objective = objective;
                first_root_relaxation = true;
            }
        }
        if (first_root_relaxation) {
            maybe_log_progress_("root", true);
        }
    }

    void update_tree_node_(int id, const std::function<void(TreeNode&)>& updater) {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        updater(tree_nodes_[id]);
    }

    std::optional<detail::ActiveNode> pop_next_active_node_() {
        reap_async_heuristics_();
        const Options effective = effective_options_();
        return search_coordinator_.try_pop(effective.node_selection, problem_.maximize,
                                           effective.hybrid_depth_bias,
                                           effective.plunging_bestfreq);
    }

    std::uint64_t current_search_order_() const {
        std::lock_guard<std::mutex> lock(tree_mutex_);
        return next_order_;
    }

    bool should_terminate_() const { return search_coordinator_.should_terminate(); }

    void mark_unbounded_() { search_coordinator_.mark_unbounded(); }

    void maybe_update_incumbent_(const Eigen::VectorXd& primal, double objective) {
        if (!detail::is_integer_feasible_solution(primal, problem_.variable_types,
                                                  options_.integrality_tol) ||
            !detail::is_sos_feasible_solution(primal, problem_.sos_constraints,
                                              options_.integrality_tol)) {
            return;
        }
        // Verify linear constraint feasibility. Heuristics can produce solutions
        // that satisfy integrality but violate constraints due to rounding or
        // numerical tolerance. An infeasible incumbent causes incorrect pruning
        // and false-positive "optimal" results.
        const double ctol = options_.feasibility_tol * 100.0;
        for (const auto& c : problem_.base_constraints) {
            double lhs = 0.0;
            for (int k = 0; k < static_cast<int>(c.indices.size()); ++k)
                lhs += c.values[k] * primal[c.indices[k]];
            if (c.sense == LinearConstraintSense::LessEqual && lhs > c.rhs + ctol)
                return;
            if (c.sense == LinearConstraintSense::GreaterEqual && lhs < c.rhs - ctol)
                return;
            if (c.sense == LinearConstraintSense::Equal && std::abs(lhs - c.rhs) > ctol)
                return;
        }
        bool updated = false;
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            if (!has_incumbent_ || objective_improves_(objective, incumbent_objective_)) {
                incumbent_objective_ = objective;
                incumbent_primal_ = primal;
                has_incumbent_ = true;
                updated = true;
            }
        }
        if (updated) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            ++incumbent_updates_;
        }
        if (updated) {
            propagate_root_redcosts_to_global_domain_(objective);
            maybe_log_progress_("incumbent", true);
            search_coordinator_.notify_all();
        }
    }

    // SCIP/HiGHS-style global reduced-cost fixing (lurking bounds / addRootRedcost).
    // When the incumbent improves to `new_cutoff`, variables whose root LP reduced
    // cost satisfies |rc_j| > gap / range can have their global bounds tightened.
    // The formula is the same as tighten_bounds_from_reduced_costs but uses the
    // ROOT LP data (valid globally) instead of per-node data, so tightenings go
    // into global_domain_ and benefit every node subsequently popped.
    void propagate_root_redcosts_to_global_domain_(double new_incumbent_obj) {
        Eigen::VectorXd root_rc;
        std::vector<LPBasisStatus> root_status;
        double root_obj = std::numeric_limits<double>::quiet_NaN();
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            if (root_reduced_costs_.size() == 0) return;
            root_rc = root_reduced_costs_;
            root_status = root_basis_statuses_;
            root_obj = root_lp_objective_;
        }
        if (!std::isfinite(root_obj) || root_rc.size() == 0) return;

        const double cutoff = fathom_cutoff_(new_incumbent_obj);
        const double objective_sign = problem_.maximize ? -1.0 : 1.0;
        const double gap = objective_sign * (cutoff - root_obj);
        if (!(gap > 0.0)) return;

        const double rc_tol = std::max(10.0 * options_.feasibility_tol,
                                       std::numeric_limits<double>::epsilon() * gap);
        const int n = std::min<int>(static_cast<int>(problem_.lower_bounds.size()),
                                    static_cast<int>(root_rc.size()));

        for (int j = 0; j < n; ++j) {
            if (j >= static_cast<int>(problem_.variable_types.size()) ||
                problem_.variable_types[j] == VariableType::Continuous)
                continue;
            if (j >= static_cast<int>(root_status.size())) continue;

            const double rc = root_rc[j];
            if (!std::isfinite(rc) || std::abs(rc) <= rc_tol) continue;

            if (root_status[j] == LPBasisStatus::AtLower && rc > rc_tol) {
                // x_j at lower bound, rc > 0: upper bound can be tightened.
                const double new_ub_cont = problem_.lower_bounds(j) + gap / rc;
                if (!std::isfinite(new_ub_cont)) continue;
                const double new_ub = std::floor(new_ub_cont + 1e-12);
                if (new_ub < problem_.upper_bounds(j) - 1e-12 &&
                    new_ub >= problem_.lower_bounds(j))
                    global_domain_.tighten(j, problem_.lower_bounds(j), new_ub);
            } else if (root_status[j] == LPBasisStatus::AtUpper && rc < -rc_tol) {
                // x_j at upper bound, rc < 0: lower bound can be tightened.
                const double new_lb_cont = problem_.upper_bounds(j) + gap / rc;
                if (!std::isfinite(new_lb_cont)) continue;
                const double new_lb = std::ceil(new_lb_cont - 1e-12);
                if (new_lb > problem_.lower_bounds(j) + 1e-12 &&
                    new_lb <= problem_.upper_bounds(j))
                    global_domain_.tighten(j, new_lb, problem_.upper_bounds(j));
            }
        }
    }

    void enqueue_async_heuristic_completion_(AsyncHeuristicCompletion completion) {
        std::lock_guard<std::mutex> lock(async_heuristic_completion_mutex_);
        async_heuristic_completions_.push_back(std::move(completion));
    }

    void apply_async_heuristic_completion_(AsyncHeuristicCompletion completion) {
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            heuristic_lp_iterations_ += completion.heuristic_lp_iterations;
            heuristic_successes_ += completion.heuristic_successes;
            feasibility_jump_successes_ += completion.feasibility_jump_successes;
            feasibility_pump_successes_ += completion.feasibility_pump_successes;
            rens_successes_ += completion.rens_successes;
            rins_successes_ += completion.rins_successes;
            local_search_successes_ += completion.local_search_successes;
            local_branching_successes_ += completion.local_branching_successes;
            const int n = std::min<int>(completion.diving_stats_delta.size(), diving_stats_.size());
            for (int i = 0; i < n; ++i) {
                diving_stats_[i].attempts += completion.diving_stats_delta[i].attempts;
                diving_stats_[i].successes += completion.diving_stats_delta[i].successes;
                diving_stats_[i].lp_iterations += completion.diving_stats_delta[i].lp_iterations;
                diving_stats_[i].lp_solves += completion.diving_stats_delta[i].lp_solves;
            }
        }

        if (completion.incumbent.has_value() && completion.incumbent->has_incumbent) {
            maybe_update_incumbent_(completion.incumbent->primal, completion.incumbent->objective);
        }
    }

    void drain_async_heuristic_completions_() {
        std::deque<AsyncHeuristicCompletion> pending;
        {
            std::lock_guard<std::mutex> lock(async_heuristic_completion_mutex_);
            if (async_heuristic_completions_.empty()) {
                return;
            }
            pending.swap(async_heuristic_completions_);
        }

        while (!pending.empty()) {
            apply_async_heuristic_completion_(std::move(pending.front()));
            pending.pop_front();
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
        return std::abs(lhs - rhs) <= std::max(1e-9, options_.integrality_tol) * scale;
    }

    ProgressSnapshot progress_snapshot_() const {
        ProgressSnapshot snapshot;
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            snapshot.node_count = node_count_;
        }
        {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            snapshot.has_incumbent = has_incumbent_;
            snapshot.incumbent_objective = incumbent_objective_;
            snapshot.root_relaxation_objective = root_relaxation_objective;
        }
        snapshot.active_nodes = search_coordinator_.size();
        return snapshot;
    }

    double current_best_bound_(const ProgressSnapshot& snapshot) const {
        return search_coordinator_.compute_best_bound(
            snapshot.has_incumbent, snapshot.incumbent_objective, problem_.maximize,
            snapshot.root_relaxation_objective);
    }

    std::optional<double> current_gap_(const ProgressSnapshot& snapshot, double best_bound) const {
        if (!snapshot.has_incumbent || !std::isfinite(snapshot.incumbent_objective) ||
            !std::isfinite(best_bound)) {
            return std::nullopt;
        }
        const double raw_gap = problem_.maximize ? (best_bound - snapshot.incumbent_objective)
                                                 : (snapshot.incumbent_objective - best_bound);
        if (raw_gap <= options_.integrality_tol)
            return 0.0;
        const double denom = std::max(1.0, std::abs(snapshot.incumbent_objective));
        return raw_gap / denom;
    }

    bool gap_reduced_(double current_gap, double previous_gap) const {
        if (std::isnan(previous_gap) || !std::isfinite(previous_gap)) {
            return true;
        }
        const double scale = std::max({1.0, std::abs(current_gap), std::abs(previous_gap)});
        return current_gap < previous_gap - std::max(1e-9, options_.integrality_tol) * scale;
    }

    // Gurobi-style progress header:
    // Expl  Unexpl  Obj     Bound   Gap    It/Node  Time
    // ----  ------  ------  ------  -----  -------  ----
    static void print_progress_header_() {
        // clang-format off
        std::cout
            << "   "                                 // marker column (1 char + 2 pad)
            << std::left  << std::setw(7)  << "Nodes"
            << std::right << std::setw(7)  << "Left"
            << std::setw(14) << "Incumbent"
            << std::setw(14) << "BestBound"
            << std::setw(8)  << "Gap"
            << "  " << "Event"
            << "\n";
        // clang-format on
        std::cout << "   " << std::string(7, '-') << std::string(7, '-') << std::string(14, '-')
                  << std::string(14, '-') << std::string(8, '-') << "\n";
    }

    // Map event name → single-char marker (Gurobi style)
    static char event_marker_(const char* event) {
        std::string_view ev(event);
        if (ev == "incumbent")
            return 'H'; // heuristic incumbent
        if (ev == "root")
            return 'R'; // root relaxation
        if (ev == "cut")
            return 'C'; // cutting plane tightened bound
        if (ev == "node-cut")
            return 'c';
        if (ev == "bound")
            return 'B';
        if (ev == "done")
            return ' ';
        return ' ';
    }

    void log_progress_locked_(const ProgressSnapshot& snapshot, const char* event, bool force,
                              const Status* final_status) {
        if (!progress_header_printed_) {
            print_progress_header_();
            progress_header_printed_ = true;
        }

        const double best_bound = current_best_bound_(snapshot);
        if (!force && !snapshot.root_relaxation_objective.has_value() && !snapshot.has_incumbent &&
            !std::isfinite(best_bound)) {
            return;
        }
        const bool incumbent_changed =
            !same_progress_value_(snapshot.incumbent_objective, last_logged_incumbent_);
        const std::optional<double> gap = current_gap_(snapshot, best_bound);
        const bool gap_reduced = gap.has_value() && gap_reduced_(*gap, last_logged_gap_);
        if (!force && !incumbent_changed && !gap_reduced) {
            return;
        }

        const char marker = event_marker_(event);
        std::ostringstream oss;
        // marker column
        oss << ' ' << marker << ' ';
        // Nodes explored / left
        oss << std::left << std::setw(7) << snapshot.node_count;
        oss << std::right << std::setw(7) << snapshot.active_nodes;
        // Incumbent
        oss << std::setw(14)
            << format_progress_value_(snapshot.has_incumbent
                                          ? snapshot.incumbent_objective
                                          : std::numeric_limits<double>::quiet_NaN());
        // Best bound
        oss << std::setw(14) << format_progress_value_(best_bound);
        // Gap
        if (gap.has_value()) {
            std::ostringstream gap_str;
            gap_str << std::fixed << std::setprecision(2) << (100.0 * *gap) << "%";
            oss << std::setw(8) << gap_str.str();
        } else {
            oss << std::setw(8) << "--";
        }
        // Event label
        oss << "  " << event;
        if (final_status) {
            oss << " [" << to_string(*final_status) << "]";
        }

        std::cout << oss.str() << "\n";
        last_logged_node_count_ = snapshot.node_count;
        last_logged_best_bound_ = best_bound;
        last_logged_incumbent_ = snapshot.incumbent_objective;
        last_logged_gap_ = gap.value_or(std::numeric_limits<double>::quiet_NaN());
    }

    static std::string format_time_ms_(std::uint64_t ns) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (static_cast<double>(ns) / 1'000'000.0);
        return oss.str();
    }

    static std::string format_time_cell_(std::optional<std::uint64_t> ns) {
        if (!ns.has_value()) {
            return "--";
        }
        return format_time_ms_(*ns) + " ms";
    }

    static std::string format_count_cell_(std::optional<int> value) {
        if (!value.has_value()) {
            return "--";
        }
        return std::to_string(*value);
    }

    // Print a section heading like:
    //   --- Timing Summary ---
    static void print_summary_section_(const char* title) {
        // 79 chars wide: "--- Title ---" padded with dashes
        const std::string t(title);
        const int total = 79;
        const int inner = static_cast<int>(t.size()) + 2; // " title "
        const int left_dashes = std::max(3, (total - inner) / 2);
        const int right_dashes = std::max(3, total - inner - left_dashes);
        std::cout << "\n"
                  << std::string(left_dashes, '-') << " " << t << " "
                  << std::string(right_dashes, '-') << "\n";
    }

    template <typename Left, typename Mid, typename Right>
    static void print_three_column_summary_row_(const char* /*section*/, Left&& left, Mid&& mid,
                                                Right&& right, int left_width = 22,
                                                int value_width = 14) {
        std::cout << "  " << std::left << std::setw(left_width) << std::forward<Left>(left)
                  << std::right << std::setw(value_width) << std::forward<Mid>(mid)
                  << std::setw(value_width) << std::forward<Right>(right) << "\n";
    }

    template <typename Left, typename Col1, typename Col2, typename Col3>
    static void print_four_column_summary_row_(const char* /*section*/, Left&& left, Col1&& col1,
                                               Col2&& col2, Col3&& col3, int left_width = 18,
                                               int value_width = 12) {
        std::cout << "  " << std::left << std::setw(left_width) << std::forward<Left>(left)
                  << std::right << std::setw(value_width) << std::forward<Col1>(col1)
                  << std::setw(value_width) << std::forward<Col2>(col2) << std::setw(value_width)
                  << std::forward<Col3>(col3) << "\n";
    }

    static std::vector<std::tuple<std::string, int, int>>
    collect_cut_family_stats_(const std::unordered_map<std::string, int>& generated,
                              const std::unordered_map<std::string, int>& applied) {
        const std::array<std::string_view, 9> preferred = {
            "GMI",
            "MIR",
            "Cover",
            "ImpliedBound",
            "Clique",
            "OddCycle",
            "DualProof",
            "ProbingImpliedBound",
            "ConflictClique",
        };

        std::vector<std::tuple<std::string, int, int>> rows;
        std::unordered_set<std::string> seen;
        rows.reserve(generated.size() + applied.size());

        auto append = [&](std::string_view name_view) {
            const std::string name(name_view);
            const auto generated_it = generated.find(name);
            const auto applied_it = applied.find(name);
            const int generated_count = generated_it != generated.end() ? generated_it->second : 0;
            const int applied_count = applied_it != applied.end() ? applied_it->second : 0;
            if (generated_count == 0 && applied_count == 0) {
                return;
            }
            seen.insert(name);
            rows.emplace_back(name, generated_count, applied_count);
        };

        for (std::string_view name : preferred) {
            append(name);
        }
        for (const auto& [name, count] : generated) {
            (void)count;
            if (!seen.contains(name)) {
                append(name);
            }
        }
        for (const auto& [name, count] : applied) {
            (void)count;
            if (!seen.contains(name)) {
                append(name);
            }
        }

        return rows;
    }

    void log_timing_summary_() const {
        std::lock_guard<std::mutex> output_lock(progress_mutex_);
        std::scoped_lock lock(stats_mutex_, cuts_mutex_);

        print_summary_section_("Timing");
        print_three_column_summary_row_("", "Phase", "Root", "Node");
        print_three_column_summary_row_("", "Cut generation",
                                        format_time_cell_(root_cut_generation_wall_ns_),
                                        format_time_cell_(node_cut_generation_wall_ns_));
        print_three_column_summary_row_("", "Cut selection",
                                        format_time_cell_(root_cut_selection_wall_ns_),
                                        format_time_cell_(node_cut_selection_wall_ns_));
        print_three_column_summary_row_("", "Cut activation",
                                        format_time_cell_(root_cut_activation_wall_ns_),
                                        format_time_cell_(std::nullopt));
        print_three_column_summary_row_("", "Cut resolve",
                                        format_time_cell_(root_cut_resolve_wall_ns_),
                                        format_time_cell_(node_cut_resolve_wall_ns_));
        print_three_column_summary_row_("", "Rounding", format_time_cell_(std::nullopt),
                                        format_time_cell_(rounding_heuristic_wall_ns_));
        print_three_column_summary_row_("", "Async heuristics", format_time_cell_(std::nullopt),
                                        format_time_cell_(heuristics_wall_ns_));
        print_three_column_summary_row_("", "Branching", format_time_cell_(std::nullopt),
                                        format_time_cell_(branching_wall_ns_));
        print_three_column_summary_row_("", "Child processing", format_time_cell_(std::nullopt),
                                        format_time_cell_(child_processing_wall_ns_));

        const int round_and_diving_successes =
            std::max(0, heuristic_successes_ - feasibility_jump_successes_ -
                            feasibility_pump_successes_ - rens_successes_ - rins_successes_ -
                            local_search_successes_ - local_branching_successes_);
        print_summary_section_("Heuristics");
        print_three_column_summary_row_("", "Method", "Successes", "Time");
        print_three_column_summary_row_(
            "", "Round + diving", format_count_cell_(round_and_diving_successes),
            format_time_cell_(rounding_heuristic_wall_ns_ + diving_wall_ns_));
        print_three_column_summary_row_("", "Feasibility jump",
                                        format_count_cell_(feasibility_jump_successes_),
                                        format_time_cell_(feasibility_jump_wall_ns_));
        print_three_column_summary_row_("", "Feasibility pump",
                                        format_count_cell_(feasibility_pump_successes_),
                                        format_time_cell_(feasibility_pump_wall_ns_));
        print_three_column_summary_row_("", "RENS", format_count_cell_(rens_successes_),
                                        format_time_cell_(rens_wall_ns_));
        print_three_column_summary_row_("", "RINS", format_count_cell_(rins_successes_),
                                        format_time_cell_(rins_wall_ns_));
        print_three_column_summary_row_("", "Local search",
                                        format_count_cell_(local_search_successes_),
                                        format_time_cell_(local_search_wall_ns_));
        print_three_column_summary_row_("", "Local branching",
                                        format_count_cell_(local_branching_successes_),
                                        format_time_cell_(local_branching_wall_ns_));
        print_three_column_summary_row_(
            "", "Total", format_count_cell_(heuristic_successes_),
            format_time_cell_(rounding_heuristic_wall_ns_ + heuristics_wall_ns_));

        print_summary_section_("Cuts");
        print_four_column_summary_row_("", "Family", "Generated", "Applied", "Share");
        const int total_generated_cuts = cut_pool_.cuts_generated();
        const std::string all_share = total_generated_cuts > 0 ? "100.0%" : "--";
        print_four_column_summary_row_("", "All", format_count_cell_(total_generated_cuts),
                                       format_count_cell_(cut_pool_.cuts_applied()), all_share);
        print_four_column_summary_row_("", "Duplicates",
                                       format_count_cell_(cut_pool_.duplicate_cuts()),
                                       format_count_cell_(std::nullopt), "--");
        print_four_column_summary_row_("", "Pool size", format_count_cell_(cut_pool_.size()),
                                       format_count_cell_(std::nullopt), "--");

        const auto cut_rows =
            collect_cut_family_stats_(cut_pool_.generated_counts(), cut_pool_.applied_counts());
        const double total_generated = std::max(1, total_generated_cuts);
        for (const auto& [name, generated_count, applied_count] : cut_rows) {
            std::ostringstream share;
            share << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(generated_count) / total_generated) << "%";
            print_four_column_summary_row_("", name, format_count_cell_(generated_count),
                                           format_count_cell_(applied_count), share.str());
        }
        std::cout << "\n" << std::string(79, '=') << "\n";
    }

    void maybe_log_progress_(const char* event, bool force = false,
                             const Status* final_status = nullptr) {
        if (!options_.verbose)
            return;
        const ProgressSnapshot snapshot = progress_snapshot_();
        std::lock_guard<std::mutex> lock(progress_mutex_);
        log_progress_locked_(snapshot, event, force, final_status);
    }

    template <typename ProcessNode>
    void process_parallel_active_nodes_(ProcessNode&& process_node) {
        const int worker_count = std::max(1, options_.parallel_workers);
        struct WorkerFinishedGuard {
            detail::SearchCoordinator* coordinator;
            bool active = true;
            ~WorkerFinishedGuard() {
                if (active) {
                    coordinator->on_worker_finished();
                }
            }
            void release() { active = false; }
        };

        auto worker = [&](int worker_id) {
            while (true) {
                reap_async_heuristics_();
                const Options effective = effective_options_();
                detail::PopResult pop_result = search_coordinator_.wait_pop(
                    effective.node_selection, problem_.maximize, effective.hybrid_depth_bias,
                    effective.plunging_bestfreq, worker_id);
                if (pop_result.terminated) {
                    break;
                }
                if (!pop_result.node.has_value()) {
                    continue;
                }

                WorkerFinishedGuard guard{&search_coordinator_, true};
                process_node(std::move(*pop_result.node), false, worker_id);

                // Direct continuation: after processing a node this worker may have just
                // pushed children.  Try to pick up local work immediately (once) before
                // yielding back to wait_pop.  This avoids an extra lock-acquire-sleep-wake
                // cycle and keeps the hot cache lines on the same core.
                {
                    const Options continuation_effective = effective_options_();
                    auto next = search_coordinator_.try_pop(
                        continuation_effective.node_selection, problem_.maximize,
                        continuation_effective.hybrid_depth_bias,
                        continuation_effective.plunging_bestfreq, worker_id);
                    if (next.has_value()) {
                        process_node(std::move(*next), false, worker_id);
                    }
                }

                guard.release();
                search_coordinator_.on_worker_finished();
            }
        };

        std::vector<std::jthread> workers;
        workers.reserve(std::max(0, worker_count - 1));
        for (int i = 1; i < worker_count; ++i) {
            workers.emplace_back([&, i] { worker(i); });
        }
        worker(0);
    }

    template <typename RelaxationSolver, typename SubMIPSolver, typename SubMIPSolverWithCuts>
    void process_active_node_(detail::ActiveNode node, bool allow_root_cuts,
                              RelaxationSolver&& relaxation_solver, SubMIPSolver&& solve_submip,
                              SubMIPSolverWithCuts&& solve_submip_with_cuts, int worker_id) {
        if (should_terminate_()) {
            return;
        }

        detail::materialize_active_node(&node);

        const auto node_start = SteadyClock::now();
        NodeTimingRecord timing;
        timing.node_id = node.id;
        timing.parent_id = node.parent_id;
        timing.depth = node.depth;
        timing.order = node.order;
        timing.allow_root_cuts = allow_root_cuts;
        timing.root_node = node.depth == 0;
        std::mutex timing_mutex;
        auto finalize_node_timing = [&](const char* exit_stage, const char* final_status) {
            timing.exit_stage = exit_stage;
            timing.final_status = final_status;
            timing.total_wall_ns = elapsed_ns_(node_start, SteadyClock::now());
            TreeNode snapshot;
            bool has_snapshot = false;
            {
                std::lock_guard<std::mutex> lock(tree_mutex_);
                if (timing.node_id >= 0 && timing.node_id < static_cast<int>(tree_nodes_.size())) {
                    snapshot = tree_nodes_[timing.node_id];
                    has_snapshot = true;
                }
            }
            if (has_snapshot) {
                timing.final_bound = snapshot.bound;
                timing.final_estimate = snapshot.estimate;
                if (timing.branch_variable < 0) {
                    timing.branch_variable = snapshot.branch_var;
                }
                if (!std::isfinite(timing.branch_value)) {
                    timing.branch_value = snapshot.branch_value;
                }
            }
            note_node_timing_(timing);
            flush_node_timing_record_(timing);
        };

        auto should_run_node_presolve = [&](const LPBasis* basis, const Options& effective) {
            return effective.use_node_presolve &&
                   (!basis || effective.use_node_presolve_on_warm_basis);
        };

        bool current_relaxation_allows_global_conflicts = true;
        auto tighten_bounds_from_reduced_costs = [&](detail::ActiveNode& current_node,
                                                     const RelaxationSolution& out) {
            if (out.status != RelaxationStatus::Optimal || !out.lp_solution.has_value() ||
                out.lp_solution->reduced_costs_internal.size() == 0 ||
                out.lp_solution->basis_state.column_status.empty()) {
                return;
            }

            const auto incumbent = incumbent_snapshot_();
            if (!incumbent.has_incumbent) {
                return;
            }

            // Use the optimum-preserving fathoming cutoff (HiGHS upper_limit
            // semantics) so the gap matches the value used by bound_prunes_.
            const double cutoff = fathom_cutoff_(incumbent.objective);
            const double objective_sign = problem_.maximize ? -1.0 : 1.0;
            const double gap = objective_sign * (cutoff - out.objective);
            if (!(gap > 0.0)) {
                return;
            }

            const Eigen::VectorXd& reduced_costs = out.lp_solution->reduced_costs_internal;
            const auto& status = out.lp_solution->basis_state.column_status;
            const int var_count =
                std::min<int>(static_cast<int>(current_node.lower_bounds.size()),
                              std::min<int>(static_cast<int>(reduced_costs.size()),
                                            static_cast<int>(status.size())));
            const double tol = 1e-12;
            // Reduced-cost tolerance matching HiGHS HighsRedcostFixing::propagateRedCost:
            // refuse to fix on tiny reduced costs that may be numerical noise.
            const double rc_tolerance = std::max(10.0 * options_.feasibility_tol,
                                                 std::numeric_limits<double>::epsilon() * gap);

            Eigen::VectorXd adjusted_reduced_costs = reduced_costs;
            if (out.lp_solution->has_internal_tableau &&
                out.lp_solution->tableau.rows() ==
                    static_cast<int>(out.lp_solution->basis_state.basis_columns.size()) &&
                out.lp_solution->tableau.cols() >= var_count &&
                static_cast<int>(out.lp_solution->x.size()) >= var_count) {
                std::vector<int> basis_row_index(var_count, -1);
                for (int row = 0;
                     row < static_cast<int>(out.lp_solution->basis_state.basis_columns.size());
                     ++row) {
                    int col = out.lp_solution->basis_state.basis_columns[row];
                    if (col >= 0 && col < var_count) {
                        basis_row_index[col] = row;
                    }
                }

                for (int j = 0; j < var_count; ++j) {
                    if (j >= static_cast<int>(problem_.variable_types.size()) ||
                        problem_.variable_types[j] == VariableType::Continuous ||
                        status[j] != LPBasisStatus::Basic) {
                        continue;
                    }
                    const double xj = out.lp_solution->x[j];
                    const double lb = current_node.lower_bounds[j];
                    const double ub = current_node.upper_bounds[j];
                    const bool at_lower = xj <= lb + tol;
                    const bool at_upper = xj >= ub - tol;
                    if (!at_lower && !at_upper) {
                        continue;
                    }

                    const int row = basis_row_index[j];
                    if (row < 0 || row >= out.lp_solution->tableau.rows()) {
                        continue;
                    }

                    const Eigen::VectorXd row_coeffs = out.lp_solution->tableau.row(row);
                    const double sign = at_lower ? 1.0 : -1.0;
                    double degenerate_dual = std::numeric_limits<double>::infinity();

                    for (int k = 0; k < var_count; ++k) {
                        if (k == j) {
                            continue;
                        }
                        const double val = sign * row_coeffs[k];
                        if (!std::isfinite(val) || std::abs(val) <= tol) {
                            continue;
                        }
                        const double xk = out.lp_solution->x[k];
                        const double lbk = current_node.lower_bounds[k];
                        const double ubk = current_node.upper_bounds[k];
                        if (!std::isfinite(xk) || !std::isfinite(lbk) || !std::isfinite(ubk)) {
                            continue;
                        }
                        const double rc_k = adjusted_reduced_costs[k];
                        if (!std::isfinite(rc_k)) {
                            continue;
                        }
                        double candidate = std::numeric_limits<double>::infinity();
                        if (val > 0.0) {
                            if (xk - lbk > tol) {
                                candidate = -rc_k / val;
                            }
                        } else {
                            if (ubk - xk > tol) {
                                candidate = -rc_k / val;
                            }
                        }
                        if (candidate < degenerate_dual) {
                            degenerate_dual = candidate;
                        }
                    }

                    if (!std::isfinite(degenerate_dual) || degenerate_dual <= tol) {
                        continue;
                    }
                    const double candidate_rc = sign * degenerate_dual;
                    if (std::abs(candidate_rc) > std::abs(adjusted_reduced_costs[j]) + tol) {
                        adjusted_reduced_costs[j] = candidate_rc;
                    }
                }
            }

            const double feastol = options_.feasibility_tol;
            for (int j = 0; j < var_count; ++j) {
                if (j >= static_cast<int>(problem_.variable_types.size())) {
                    continue;
                }
                const bool is_integer = problem_.variable_types[j] != VariableType::Continuous;
                const double rc = adjusted_reduced_costs[j];
                if (!std::isfinite(rc) || std::abs(rc) <= rc_tolerance) {
                    continue;
                }
                // Skip fixed variables (HiGHS HighsRedcostFixing::propagateRedCost L97).
                if (current_node.upper_bounds[j] - current_node.lower_bounds[j] <= tol) {
                    continue;
                }
                if (status[j] == LPBasisStatus::AtLower) {
                    if (rc <= 0.0) {
                        continue;
                    }
                    const double bound = current_node.lower_bounds[j] + gap / rc;
                    if (!std::isfinite(bound)) {
                        continue;
                    }
                    const double tightened_upper =
                        is_integer ? std::floor(bound + feastol) : (bound + feastol);
                    if (tightened_upper + tol < current_node.upper_bounds[j] &&
                        tightened_upper >= current_node.lower_bounds[j]) {
                        current_node.upper_bounds[j] =
                            std::min(current_node.upper_bounds[j], tightened_upper);
                    }
                } else if (status[j] == LPBasisStatus::AtUpper) {
                    if (rc >= 0.0) {
                        continue;
                    }
                    const double bound = current_node.upper_bounds[j] + gap / rc;
                    if (!std::isfinite(bound)) {
                        continue;
                    }
                    const double tightened_lower =
                        is_integer ? std::ceil(bound - feastol) : (bound - feastol);
                    if (tightened_lower > current_node.lower_bounds[j] + tol &&
                        tightened_lower <= current_node.upper_bounds[j]) {
                        current_node.lower_bounds[j] =
                            std::max(current_node.lower_bounds[j], tightened_lower);
                    }
                }
            }
        };

        auto solve_relaxation_with_cuts = [&](detail::ActiveNode& current_node,
                                              const std::vector<Cut>& extra_cuts) {
            const auto solve_start = SteadyClock::now();
            std::vector<Cut> relaxation_cuts = current_relaxation_cuts_snapshot_();
            relaxation_cuts.insert(relaxation_cuts.end(), extra_cuts.begin(), extra_cuts.end());
            const bool allow_global_conflict_learning =
                !contains_incumbent_cutoff_(relaxation_cuts);
            current_relaxation_allows_global_conflicts = allow_global_conflict_learning;
            const LPBasis* warm_basis = current_node.basis ? &*current_node.basis : nullptr;
            const Options effective = effective_options_();
            const auto presolve_start = SteadyClock::now();
            NodePresolveOutcome presolved;
            if (should_run_node_presolve(warm_basis, effective)) {
                presolved =
                    presolve_node_bounds_(current_node.lower_bounds, current_node.upper_bounds,
                                          relaxation_cuts, current_node.reasons);
            } else {
                presolved.lower_bounds = current_node.lower_bounds;
                presolved.upper_bounds = current_node.upper_bounds;
                presolved.reasons = current_node.reasons;
            }
            const std::uint64_t presolve_wall_ns = elapsed_ns_(presolve_start, SteadyClock::now());
            current_node.lower_bounds = presolved.lower_bounds;
            current_node.upper_bounds = presolved.upper_bounds;
            current_node.reasons = presolved.reasons;
            // Apply global domain: intersect node bounds with globally-valid tightest bounds.
            // This is the SCIP/HiGHS domain-application step — lurking bounds from root RC
            // fixing and other globally-valid deductions become visible to every node here.
            global_domain_.apply(current_node.lower_bounds, current_node.upper_bounds);
            // Check for infeasibility introduced by global domain tightening.
            bool global_domain_infeasible = false;
            if (!presolved.infeasible) {
                const int n = std::min<int>(current_node.lower_bounds.size(),
                                            current_node.upper_bounds.size());
                for (int j = 0; j < n; ++j) {
                    if (current_node.lower_bounds(j) > current_node.upper_bounds(j) + options_.feasibility_tol) {
                        global_domain_infeasible = true;
                        break;
                    }
                }
            }
            if (presolved.infeasible || global_domain_infeasible) {
                maybe_learn_conflict_from_bounds_(current_node.lower_bounds,
                                                  current_node.upper_bounds,
                                                  allow_global_conflict_learning);
                RelaxationSolution out;
                out.status = RelaxationStatus::Infeasible;
                out.primal = Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                                       std::numeric_limits<double>::quiet_NaN());
                out.objective = problem_.maximize ? -std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::infinity();
                std::lock_guard<std::mutex> timing_lock(timing_mutex);
                accumulate_relaxation_timing_(&timing.node_relaxation, out,
                                              elapsed_ns_(solve_start, SteadyClock::now()),
                                              presolve_wall_ns);
                return out;
            }
            RelaxationSolution out = relaxation_solver(
                current_node.lower_bounds, current_node.upper_bounds, warm_basis, relaxation_cuts);
            if (out.status == RelaxationStatus::Infeasible) {
                maybe_learn_conflict_from_bounds_(current_node.lower_bounds,
                                                  current_node.upper_bounds,
                                                  allow_global_conflict_learning);
                // HiGHS-style dual proof cut extraction: when the node LP is infeasible,
                // extract a Farkas-certificate-based proof cut. Continuous variables are
                // substituted at GLOBAL problem bounds, so the resulting cut is valid for
                // the entire remaining tree (not just this subtree). Only cuts that are
                // also violated at global bounds (i.e., globally infeasibility-proving) are
                // kept; those pass through cut_pool_ and benefit subsequent nodes.
                if (options_.use_dual_proof_cuts) {
                    std::vector<Cut> proof_cuts = detail::generate_dual_proof_cuts(
                        problem_, relaxation_cuts, out,
                        problem_.lower_bounds, problem_.upper_bounds, options_);
                    for (Cut& cut : proof_cuts)
                        cut_pool_.add_cut(problem_, cut);
                }
            } else {
                tighten_bounds_from_reduced_costs(current_node, out);
            }
            std::lock_guard<std::mutex> timing_lock(timing_mutex);
            accumulate_relaxation_timing_(&timing.node_relaxation, out,
                                          elapsed_ns_(solve_start, SteadyClock::now()),
                                          presolve_wall_ns);
            return out;
        };
        auto current_relaxation = [&](detail::ActiveNode& current_node) {
            return solve_relaxation_with_cuts(current_node, {});
        };

        auto tighten_bounds_from_implied_cuts = [&](const std::vector<Cut>& cuts,
                                                    detail::ActiveNode& current_node) {
            int tightened_bounds = 0;
            for (const Cut& cut : cuts) {
                if (!propagate_row_bounds_(cut.indices, cut.values, cut.rhs, cut.sense,
                                           &current_node.lower_bounds, &current_node.upper_bounds,
                                           &tightened_bounds, nullptr, nullptr)) {
                    return -1;
                }
            }
            return tightened_bounds;
        };

        RelaxationSolution relaxation = current_relaxation(node);
        timing.final_relaxation_objective = relaxation.objective;
        node.basis = relaxation.basis;
        note_lp_work_(relaxation);

        const auto estimate =
            detail::node_estimate(relaxation, problem_.variable_types, pseudocosts_snapshot_(),
                                  options_.integrality_tol, problem_.maximize);
        update_tree_node_(node.id, [&](TreeNode& tree_node) {
            tree_node.bound = relaxation.objective;
            tree_node.estimate = estimate;
        });
        maybe_log_progress_("node");

        if (relaxation.status == RelaxationStatus::Unbounded) {
            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Unbounded;
            });
            mark_unbounded_();
            search_coordinator_.notify_all();
            finalize_node_timing("initial_relaxation", "unbounded");
            return;
        }
        if (relaxation.status == RelaxationStatus::Infeasible) {
            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Infeasible;
            });
            finalize_node_timing("initial_relaxation", "infeasible");
            return;
        }

        mark_root_relaxation_(relaxation.objective);
        if (node.depth == 0 && relaxation.status == RelaxationStatus::Optimal &&
            relaxation.basis.has_value()) {
            std::lock_guard<std::mutex> lock(incumbent_mutex_);
            if (!root_warm_start_basis_state_.has_value()) {
                root_warm_start_basis_state_ = relaxation.basis;
            }
            // Capture root LP reduced costs for global reduced-cost fixing (lurking bounds).
            // These are valid globally: whenever the incumbent improves, any variable whose
            // reduced cost exceeds gap/range can have its bound tightened for ALL nodes.
            if (root_reduced_costs_.size() == 0 && relaxation.lp_solution.has_value() &&
                relaxation.lp_solution->reduced_costs_internal.size() > 0 &&
                !relaxation.lp_solution->basis_state.column_status.empty()) {
                root_reduced_costs_ = relaxation.lp_solution->reduced_costs_internal;
                root_basis_statuses_ = relaxation.lp_solution->basis_state.column_status;
                root_lp_objective_ = relaxation.objective;
            }
        }

        {
            const auto incumbent = incumbent_snapshot_();
            if (incumbent.has_incumbent &&
                bound_prunes_(relaxation.objective, incumbent.objective)) {
                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                    tree_node.status = TreeNodeStatus::PrunedByBound;
                });
                finalize_node_timing("bound_prune", "pruned_by_bound");
                return;
            }
        }

        if (allow_root_cuts && options_.use_cut_pool && node.depth == 0) {
            bool re_solved_with_cuts = false;
            const int root_cut_rounds = options_.max_root_cut_rounds > 0
                                            ? options_.max_root_cut_rounds
                                            : options_.max_cut_rounds_per_node;
            const int root_cuts_added_per_round = options_.max_root_cuts_added_per_round > 0
                                                      ? options_.max_root_cuts_added_per_round
                                                      : options_.max_cuts_added_per_round;
            for (int round = 0; round < root_cut_rounds; ++round) {
                ++timing.root_cut_rounds;
                const auto cut_fractional = detail::collect_fractional_candidates(
                    relaxation.primal, problem_.variable_types, options_.integrality_tol);
                const std::vector<Cut> relaxation_cuts = current_relaxation_cuts_snapshot_();
                const auto selection_start = SteadyClock::now();
                std::vector<Cut> selected;
                // Note: CutSeparatorPhase::Proof is intentionally excluded here.
                // DualProof cuts require an infeasible LP (Farkas certificate), but cut
                // rounds only run when the LP is optimal. Proof cuts are extracted
                // directly at infeasibility detection points in solve_relaxation_with_cuts
                // and node_relaxation_solver instead (HiGHS-style).
                const std::array<detail::CutSeparatorPhase, 4> phase_order = {
                    detail::CutSeparatorPhase::ImpliedBound, detail::CutSeparatorPhase::Clique,
                    detail::CutSeparatorPhase::OddCycle,     detail::CutSeparatorPhase::LP,
                };
                bool any_cuts_applied = false;
                for (detail::CutSeparatorPhase phase : phase_order) {
                    const auto phase_generation_start = SteadyClock::now();
                    std::vector<Cut> phase_generated = detail::generate_cuts(
                        problem_, relaxation, options_, phase, nullptr, &relaxation_cuts);
                    timing.root_cut_generation_wall_ns +=
                        elapsed_ns_(phase_generation_start, SteadyClock::now());
                    timing.root_cuts_generated += static_cast<int>(phase_generated.size());
                    if (phase == detail::CutSeparatorPhase::ImpliedBound) {
                        const int tightened =
                            tighten_bounds_from_implied_cuts(phase_generated, node);
                        if (tightened < 0) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Infeasible;
                            });
                            finalize_node_timing("root_bound_tighten", "infeasible");
                            return;
                        }
                        if (tightened > 0) {
                            any_cuts_applied = true;
                            re_solved_with_cuts = true;
                            const auto resolve_start = SteadyClock::now();
                            relaxation = current_relaxation(node);
                            timing.root_cut_resolve_wall_ns +=
                                elapsed_ns_(resolve_start, SteadyClock::now());
                            timing.final_relaxation_objective = relaxation.objective;
                            node.basis = relaxation.basis;
                            note_lp_work_(relaxation);
                            const auto cut_estimate = detail::node_estimate(
                                relaxation, problem_.variable_types, pseudocosts_snapshot_(),
                                options_.integrality_tol, problem_.maximize);
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.bound = relaxation.objective;
                                tree_node.estimate = cut_estimate;
                            });
                            if (relaxation.status == RelaxationStatus::Unbounded) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::Unbounded;
                                });
                                mark_unbounded_();
                                search_coordinator_.notify_all();
                                finalize_node_timing("root_bound_tighten", "unbounded");
                                return;
                            }
                            if (relaxation.status == RelaxationStatus::Infeasible) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::Infeasible;
                                });
                                finalize_node_timing("root_bound_tighten", "infeasible");
                                return;
                            }
                            const auto incumbent = incumbent_snapshot_();
                            if (incumbent.has_incumbent &&
                                bound_prunes_(relaxation.objective, incumbent.objective)) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::PrunedByBound;
                                });
                                finalize_node_timing("root_bound_tighten", "pruned_by_bound");
                                return;
                            }
                        }
                    }
                    for (const Cut& cut : phase_generated) {
                        cut_pool_.add_cut(problem_, cut);
                    }

                    const auto phase_selection_start = SteadyClock::now();
                    selected = cut_pool_.select_violated_cuts(
                        relaxation.primal, node.lower_bounds, node.upper_bounds,
                        root_cuts_added_per_round, 1.0, &problem_.objective_coefficients,
                        problem_.maximize);
                    timing.root_cut_selection_wall_ns +=
                        elapsed_ns_(phase_selection_start, SteadyClock::now());
                    timing.root_cuts_selected += static_cast<int>(selected.size());
                    cut_pool_.perform_aging();

                    if (selected.empty())
                        continue;

                    int added_count = 0;
                    const auto activation_start = SteadyClock::now();
                    {
                        std::lock_guard<std::mutex> lock(cuts_mutex_);
                        for (const Cut& cut : selected) {
                            const detail::CutSignature signature = detail::cut_signature(cut);
                            if (active_cut_signatures_.contains(signature))
                                continue;
                            active_cut_signatures_.insert(signature);
                            active_cuts_.push_back(cut);
                            ++added_count;
                        }
                    }
                    timing.root_cut_activation_wall_ns +=
                        elapsed_ns_(activation_start, SteadyClock::now());
                    timing.root_cuts_applied += added_count;
                    if (added_count > 0) {
                        any_cuts_applied = true;
                        re_solved_with_cuts = true;
                        const auto resolve_start = SteadyClock::now();
                        relaxation = current_relaxation(node);
                        timing.root_cut_resolve_wall_ns +=
                            elapsed_ns_(resolve_start, SteadyClock::now());
                        timing.final_relaxation_objective = relaxation.objective;
                        node.basis = relaxation.basis;
                        note_lp_work_(relaxation);
                        const auto cut_estimate = detail::node_estimate(
                            relaxation, problem_.variable_types, pseudocosts_snapshot_(),
                            options_.integrality_tol, problem_.maximize);
                        update_tree_node_(node.id, [&](TreeNode& tree_node) {
                            tree_node.bound = relaxation.objective;
                            tree_node.estimate = cut_estimate;
                        });
                        maybe_log_progress_("cut");
                        if (relaxation.status == RelaxationStatus::Unbounded) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Unbounded;
                            });
                            mark_unbounded_();
                            search_coordinator_.notify_all();
                            finalize_node_timing("root_cut_resolve", "unbounded");
                            return;
                        }
                        if (relaxation.status == RelaxationStatus::Infeasible) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Infeasible;
                            });
                            finalize_node_timing("root_cut_resolve", "infeasible");
                            return;
                        }
                        const auto incumbent = incumbent_snapshot_();
                        if (incumbent.has_incumbent &&
                            bound_prunes_(relaxation.objective, incumbent.objective)) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::PrunedByBound;
                            });
                            finalize_node_timing("root_cut_bound_prune", "pruned_by_bound");
                            return;
                        }
                    }
                }
                if (!any_cuts_applied)
                    break;
            }

            if (re_solved_with_cuts) {
                const auto incumbent = incumbent_snapshot_();
                if (incumbent.has_incumbent &&
                    bound_prunes_(relaxation.objective, incumbent.objective)) {
                    update_tree_node_(node.id, [&](TreeNode& tree_node) {
                        tree_node.status = TreeNodeStatus::PrunedByBound;
                    });
                    finalize_node_timing("root_cut_bound_prune", "pruned_by_bound");
                    return;
                }
            }
        }

        auto fractional = detail::collect_fractional_candidates(
            relaxation.primal, problem_.variable_types, options_.integrality_tol);
        const Options node_effective_options = effective_options_for_bound_(relaxation.objective);
        if (should_try_node_cuts_(node, relaxation, fractional, node_effective_options)) {
            std::vector<Cut> local_cuts;
            std::unordered_set<detail::CutSignature, detail::CutSignatureHash> local_cut_signatures;
            const int node_cut_rounds = node_effective_options.max_cut_rounds_per_node;
            for (int round = 0; round < node_cut_rounds; ++round) {
                ++timing.node_cut_rounds;
                std::vector<Cut> probing_relaxation_cuts = current_relaxation_cuts_snapshot_();
                probing_relaxation_cuts.insert(probing_relaxation_cuts.end(), local_cuts.begin(),
                                               local_cuts.end());
                const std::array<detail::CutSeparatorPhase, 4> phase_order = {
                    detail::CutSeparatorPhase::ImpliedBound, detail::CutSeparatorPhase::Clique,
                    detail::CutSeparatorPhase::OddCycle,     detail::CutSeparatorPhase::LP,
                };
                bool any_cuts_applied = false;
                for (detail::CutSeparatorPhase phase : phase_order) {
                    const auto phase_generation_start = SteadyClock::now();
                    std::vector<Cut> phase_generated =
                        detail::generate_cuts(problem_, relaxation, node_effective_options, phase,
                                              nullptr, &probing_relaxation_cuts);
                    timing.node_cut_generation_wall_ns +=
                        elapsed_ns_(phase_generation_start, SteadyClock::now());
                    timing.node_cuts_generated += static_cast<int>(phase_generated.size());
                    if (phase == detail::CutSeparatorPhase::ImpliedBound) {
                        const int tightened =
                            tighten_bounds_from_implied_cuts(phase_generated, node);
                        if (tightened < 0) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Infeasible;
                            });
                            finalize_node_timing("node_bound_tighten", "infeasible");
                            return;
                        }
                        if (tightened > 0) {
                            any_cuts_applied = true;
                            const auto resolve_start = SteadyClock::now();
                            relaxation = solve_relaxation_with_cuts(node, local_cuts);
                            timing.node_cut_resolve_wall_ns +=
                                elapsed_ns_(resolve_start, SteadyClock::now());
                            timing.final_relaxation_objective = relaxation.objective;
                            node.basis = relaxation.basis;
                            note_lp_work_(relaxation);
                            const auto cut_estimate = detail::node_estimate(
                                relaxation, problem_.variable_types, pseudocosts_snapshot_(),
                                options_.integrality_tol, problem_.maximize);
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.bound = relaxation.objective;
                                tree_node.estimate = cut_estimate;
                            });
                            if (relaxation.status == RelaxationStatus::Unbounded) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::Unbounded;
                                });
                                mark_unbounded_();
                                search_coordinator_.notify_all();
                                finalize_node_timing("node_bound_tighten", "unbounded");
                                return;
                            }
                            if (relaxation.status == RelaxationStatus::Infeasible) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::Infeasible;
                                });
                                finalize_node_timing("node_bound_tighten", "infeasible");
                                return;
                            }
                            const auto incumbent = incumbent_snapshot_();
                            if (incumbent.has_incumbent &&
                                bound_prunes_(relaxation.objective, incumbent.objective)) {
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::PrunedByBound;
                                });
                                finalize_node_timing("node_bound_tighten", "pruned_by_bound");
                                return;
                            }
                            fractional = detail::collect_fractional_candidates(
                                relaxation.primal, problem_.variable_types,
                                options_.integrality_tol);
                            if (fractional.empty() &&
                                detail::choose_sos_branching_constraint(node, relaxation.primal,
                                                                        problem_.sos_constraints,
                                                                        options_.integrality_tol)
                                        .variable < 0) {
                                timing.fractional_count = 0;
                                update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                    tree_node.status = TreeNodeStatus::Integral;
                                });
                                maybe_update_incumbent_(relaxation.primal, relaxation.objective);
                                search_coordinator_.notify_all();
                                finalize_node_timing("node_bound_tighten", "integral");
                                return;
                            }
                        }
                    }
                    // All cuts from the loop phases go to the global pool.
                    // DualProof (Proof-phase) cuts are NOT generated here: that phase is
                    // excluded from phase_order above. They are extracted at LP infeasibility
                    // detection points in solve_relaxation_with_cuts / node_relaxation_solver.
                    for (const Cut& cut : phase_generated) {
                        cut_pool_.add_cut(problem_, cut);
                    }

                    const auto selection_start = SteadyClock::now();
                    std::vector<Cut> selected = cut_pool_.select_violated_cuts(
                        relaxation.primal, node.lower_bounds, node.upper_bounds,
                        node_effective_options.max_cuts_added_per_round, 1.6,
                        &problem_.objective_coefficients, problem_.maximize);
                    timing.node_cut_selection_wall_ns +=
                        elapsed_ns_(selection_start, SteadyClock::now());
                    timing.node_cuts_selected += static_cast<int>(selected.size());
                    cut_pool_.perform_aging();

                    if (selected.empty())
                        continue;

                    int added_count = 0;
                    {
                        std::lock_guard<std::mutex> lock(cuts_mutex_);
                        for (const Cut& cut : selected) {
                            const detail::CutSignature signature = detail::cut_signature(cut);
                            if (active_cut_signatures_.contains(signature) ||
                                local_cut_signatures.contains(signature)) {
                                continue;
                            }
                            local_cut_signatures.insert(signature);
                            local_cuts.push_back(cut);
                            ++added_count;
                        }
                    }
                    timing.node_cuts_applied += added_count;
                    if (added_count > 0) {
                        any_cuts_applied = true;
                        const auto resolve_start = SteadyClock::now();
                        relaxation = solve_relaxation_with_cuts(node, local_cuts);
                        timing.node_cut_resolve_wall_ns +=
                            elapsed_ns_(resolve_start, SteadyClock::now());
                        timing.final_relaxation_objective = relaxation.objective;
                        node.basis = relaxation.basis;
                        note_lp_work_(relaxation);
                        const auto cut_estimate = detail::node_estimate(
                            relaxation, problem_.variable_types, pseudocosts_snapshot_(),
                            options_.integrality_tol, problem_.maximize);
                        update_tree_node_(node.id, [&](TreeNode& tree_node) {
                            tree_node.bound = relaxation.objective;
                            tree_node.estimate = cut_estimate;
                        });
                        maybe_log_progress_("node-cut");
                        if (relaxation.status == RelaxationStatus::Unbounded) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Unbounded;
                            });
                            mark_unbounded_();
                            search_coordinator_.notify_all();
                            finalize_node_timing("node_cut_resolve", "unbounded");
                            return;
                        }
                        if (relaxation.status == RelaxationStatus::Infeasible) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Infeasible;
                            });
                            finalize_node_timing("node_cut_resolve", "infeasible");
                            return;
                        }
                        const auto incumbent = incumbent_snapshot_();
                        if (incumbent.has_incumbent &&
                            bound_prunes_(relaxation.objective, incumbent.objective)) {
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::PrunedByBound;
                            });
                            finalize_node_timing("node_cut_bound_prune", "pruned_by_bound");
                            return;
                        }
                        fractional = detail::collect_fractional_candidates(
                            relaxation.primal, problem_.variable_types, options_.integrality_tol);
                        if (fractional.empty() &&
                            detail::choose_sos_branching_constraint(node, relaxation.primal,
                                                                    problem_.sos_constraints,
                                                                    options_.integrality_tol)
                                    .variable < 0) {
                            timing.fractional_count = 0;
                            update_tree_node_(node.id, [&](TreeNode& tree_node) {
                                tree_node.status = TreeNodeStatus::Integral;
                            });
                            maybe_update_incumbent_(relaxation.primal, relaxation.objective);
                            search_coordinator_.notify_all();
                            finalize_node_timing("node_cut_integral", "integral");
                            return;
                        }
                    }
                }
                if (!any_cuts_applied)
                    break;

                {
                    const auto incumbent = incumbent_snapshot_();
                    if (incumbent.has_incumbent &&
                        bound_prunes_(relaxation.objective, incumbent.objective)) {
                        update_tree_node_(node.id, [&](TreeNode& tree_node) {
                            tree_node.status = TreeNodeStatus::PrunedByBound;
                        });
                        finalize_node_timing("node_cut_bound_prune", "pruned_by_bound");
                        return;
                    }
                }

                fractional = detail::collect_fractional_candidates(
                    relaxation.primal, problem_.variable_types, options_.integrality_tol);
                if (fractional.empty() &&
                    detail::choose_sos_branching_constraint(
                        node, relaxation.primal, problem_.sos_constraints, options_.integrality_tol)
                            .variable < 0) {
                    timing.fractional_count = 0;
                    update_tree_node_(node.id, [&](TreeNode& tree_node) {
                        tree_node.status = TreeNodeStatus::Integral;
                    });
                    maybe_update_incumbent_(relaxation.primal, relaxation.objective);
                    search_coordinator_.notify_all();
                    finalize_node_timing("node_cut_integral", "integral");
                    return;
                }
            }
        }
        detail::BranchDecision sos_decision = detail::choose_sos_branching_constraint(
            node, relaxation.primal, problem_.sos_constraints, options_.integrality_tol);
        if (fractional.empty() && sos_decision.variable < 0) {
            timing.fractional_count = 0;
            update_tree_node_(
                node.id, [&](TreeNode& tree_node) { tree_node.status = TreeNodeStatus::Integral; });
            maybe_update_incumbent_(relaxation.primal, relaxation.objective);
            search_coordinator_.notify_all();
            finalize_node_timing("integral", "integral");
            return;
        }
        timing.fractional_count =
            static_cast<int>(fractional.size()) + (sos_decision.variable >= 0 ? 1 : 0);
        const HeuristicSchedule schedule = build_heuristic_schedule_(node, relaxation);

        const auto rounding_start = SteadyClock::now();
        if (options_.use_rounding) {
            if (const auto rounded = detail::run_rounding_heuristic(
                    problem_, options_, relaxation, current_relaxation_cuts_snapshot_());
                rounded.has_value()) {
                note_heuristic_result_(0, 1);
                maybe_update_incumbent_(rounded->primal, rounded->objective);
            }
        }
        timing.rounding_heuristic_wall_ns += elapsed_ns_(rounding_start, SteadyClock::now());

        auto node_relaxation_solver = [&](const detail::ChildState& child_state,
                                          const LPBasis* basis) {
            const auto solve_start = SteadyClock::now();
            detail::ChildState prepared = child_state;
            detail::materialize_child_state(&prepared);
            detail::prepare_child_state_for_relaxation(&prepared);
            const std::vector<Cut> relaxation_cuts = current_relaxation_cuts_snapshot_();
            const bool allow_global_conflict_learning =
                !contains_incumbent_cutoff_(relaxation_cuts);
            const Options effective = effective_options_();
            const auto presolve_start = SteadyClock::now();
            NodePresolveOutcome presolved;
            if (should_run_node_presolve(basis, effective)) {
                presolved = presolve_node_bounds_(prepared.lower_bounds, prepared.upper_bounds,
                                                  relaxation_cuts, prepared.reasons);
            } else {
                presolved.lower_bounds = prepared.lower_bounds;
                presolved.upper_bounds = prepared.upper_bounds;
                presolved.reasons = prepared.reasons;
            }
            const std::uint64_t presolve_wall_ns = elapsed_ns_(presolve_start, SteadyClock::now());
            if (presolved.infeasible) {
                maybe_learn_conflict_from_bounds_(presolved.lower_bounds, presolved.upper_bounds,
                                                  allow_global_conflict_learning);
                RelaxationSolution out;
                out.status = RelaxationStatus::Infeasible;
                out.primal = Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                                       std::numeric_limits<double>::quiet_NaN());
                out.objective = problem_.maximize ? -std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::infinity();
                std::lock_guard<std::mutex> timing_lock(timing_mutex);
                accumulate_relaxation_timing_(&timing.child_relaxation, out,
                                              elapsed_ns_(solve_start, SteadyClock::now()),
                                              presolve_wall_ns);
                return out;
            }
            RelaxationSolution out = relaxation_solver(
                presolved.lower_bounds, presolved.upper_bounds, basis, relaxation_cuts);
            if (out.status == RelaxationStatus::Infeasible) {
                maybe_learn_conflict_from_bounds_(presolved.lower_bounds, presolved.upper_bounds,
                                                  allow_global_conflict_learning);
                if (options_.use_dual_proof_cuts) {
                    std::vector<Cut> proof_cuts = detail::generate_dual_proof_cuts(
                        problem_, relaxation_cuts, out,
                        problem_.lower_bounds, problem_.upper_bounds, options_);
                    for (Cut& cut : proof_cuts)
                        cut_pool_.add_cut(problem_, cut);
                }
            }
            std::lock_guard<std::mutex> timing_lock(timing_mutex);
            accumulate_relaxation_timing_(&timing.child_relaxation, out,
                                          elapsed_ns_(solve_start, SteadyClock::now()),
                                          presolve_wall_ns);
            return out;
        };

        auto async_node_relaxation_solver = [this, &relaxation_solver, &should_run_node_presolve](
                                                const detail::ChildState& child_state,
                                                const LPBasis* basis) {
            detail::ChildState prepared = child_state;
            detail::materialize_child_state(&prepared);
            detail::prepare_child_state_for_relaxation(&prepared);
            const std::vector<Cut> relaxation_cuts = current_relaxation_cuts_snapshot_();
            const Options effective = effective_options_();
            NodePresolveOutcome presolved;
            if (should_run_node_presolve(basis, effective)) {
                presolved = presolve_node_bounds_(prepared.lower_bounds, prepared.upper_bounds,
                                                  relaxation_cuts, prepared.reasons);
            } else {
                presolved.lower_bounds = prepared.lower_bounds;
                presolved.upper_bounds = prepared.upper_bounds;
                presolved.reasons = prepared.reasons;
            }
            if (presolved.infeasible) {
                RelaxationSolution out;
                out.status = RelaxationStatus::Infeasible;
                out.primal = Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                                       std::numeric_limits<double>::quiet_NaN());
                out.objective = problem_.maximize ? -std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::infinity();
                return out;
            }
            RelaxationSolution out = relaxation_solver(presolved.lower_bounds,
                                                       presolved.upper_bounds, basis,
                                                       relaxation_cuts);
            if (out.status == RelaxationStatus::Infeasible && options_.use_dual_proof_cuts) {
                std::vector<Cut> proof_cuts = detail::generate_dual_proof_cuts(
                    problem_, relaxation_cuts, out,
                    problem_.lower_bounds, problem_.upper_bounds, options_);
                for (Cut& cut : proof_cuts)
                    cut_pool_.add_cut(problem_, cut);
            }
            return out;
        };

        const auto incumbent = incumbent_snapshot_();
        auto heuristic_budget_available = [this]() { return this->more_heuristics_allowed_(); };
        auto maybe_run_async_submip_heuristic = [this, &node, &timing](auto&& task) {
            ++timing.async_heuristic_launch_attempts;
            if (async_heuristics_enabled_()) {
                dispatch_async_heuristic_(
                    node.order, [this, task = std::forward<decltype(task)>(task)]() mutable {
                        enqueue_async_heuristic_completion_(task());
                    });
            } else {
                apply_async_heuristic_completion_(task());
            }
        };

        const auto heuristics_start = SteadyClock::now();
        if (schedule.run_feasibility_jump && heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto feasibility_jump_task = [this, relaxation, solve_submip]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult feasibility_jump =
                    detail::run_feasibility_jump_heuristic(problem_, options_, relaxation,
                                                           solve_submip);
                completion.heuristic_lp_iterations = feasibility_jump.lp_iterations;
                completion.heuristic_successes = feasibility_jump.successes;
                completion.feasibility_jump_successes = feasibility_jump.successes;
                if (feasibility_jump.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = feasibility_jump.incumbent->objective,
                        .primal = feasibility_jump.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(feasibility_jump_task));
            timing.feasibility_jump_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        if (schedule.run_feasibility_pump && heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto feasibility_pump_task = [this, relaxation, solve_submip]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult feasibility_pump =
                    detail::run_feasibility_pump_heuristic(problem_, options_, relaxation,
                                                           solve_submip);
                completion.heuristic_lp_iterations = feasibility_pump.lp_iterations;
                completion.heuristic_successes = feasibility_pump.successes;
                completion.feasibility_pump_successes = feasibility_pump.successes;
                if (feasibility_pump.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = feasibility_pump.incumbent->objective,
                        .primal = feasibility_pump.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(feasibility_pump_task));
            timing.feasibility_pump_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        if (schedule.run_diving && heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            const auto diving_before = diving_stats_snapshot_();
            auto diving_task = [this, node, relaxation, diving_before, incumbent,
                                async_node_relaxation_solver]() mutable {
                AsyncHeuristicCompletion completion;
                auto diving_stats = diving_before;
                const detail::DivingHeuristicResult diving = detail::run_diving_heuristic(
                    node, relaxation, problem_, options_,
                    incumbent.has_incumbent ? &incumbent.primal : nullptr, diving_stats,
                    async_node_relaxation_solver);
                completion.heuristic_lp_iterations = diving.lp_iterations;
                completion.heuristic_successes = diving.successes;
                completion.diving_stats_delta.resize(diving_stats.size());
                for (int i = 0; i < static_cast<int>(diving_stats.size()); ++i) {
                    completion.diving_stats_delta[i].attempts =
                        diving_stats[i].attempts - diving_before[i].attempts;
                    completion.diving_stats_delta[i].successes =
                        diving_stats[i].successes - diving_before[i].successes;
                    completion.diving_stats_delta[i].lp_iterations =
                        diving_stats[i].lp_iterations - diving_before[i].lp_iterations;
                    completion.diving_stats_delta[i].lp_solves =
                        diving_stats[i].lp_solves - diving_before[i].lp_solves;
                }
                if (diving.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = diving.incumbent->objective,
                        .primal = diving.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(diving_task));
            timing.diving_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        if (schedule.run_rens && heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto rens_task = [this, relaxation, solve_submip]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult rens =
                    detail::run_rens_heuristic(problem_, options_, relaxation, solve_submip);
                completion.heuristic_lp_iterations = rens.lp_iterations;
                completion.heuristic_successes = rens.successes;
                completion.rens_successes = rens.successes;
                if (rens.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = rens.incumbent->objective,
                        .primal = rens.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(rens_task));
            timing.rens_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        const auto incumbent_after_rens = incumbent_snapshot_();

        if (incumbent_after_rens.has_incumbent && schedule.run_rins &&
            heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto rins_task = [this, relaxation, incumbent_after_rens, solve_submip]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult rins = detail::run_rins_heuristic(
                    problem_, options_, relaxation, incumbent_after_rens.primal,
                    incumbent_after_rens.objective, solve_submip);
                completion.heuristic_lp_iterations = rins.lp_iterations;
                completion.heuristic_successes = rins.successes;
                completion.rins_successes = rins.successes;
                if (rins.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = rins.incumbent->objective,
                        .primal = rins.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(rins_task));
            timing.rins_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        if (incumbent_after_rens.has_incumbent && schedule.run_local_search &&
            heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto local_search_task = [this, relaxation, incumbent_after_rens, solve_submip]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult local_search =
                    detail::run_local_search_heuristic(
                        problem_, options_, relaxation, incumbent_after_rens.primal,
                        incumbent_after_rens.objective, solve_submip);
                completion.heuristic_lp_iterations = local_search.lp_iterations;
                completion.heuristic_successes = local_search.successes;
                completion.local_search_successes = local_search.successes;
                if (local_search.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = local_search.incumbent->objective,
                        .primal = local_search.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(local_search_task));
            timing.local_search_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }

        if (incumbent_after_rens.has_incumbent && schedule.run_local_branching &&
            heuristic_budget_available()) {
            const auto step_start = SteadyClock::now();
            auto local_branching_task = [this, relaxation, incumbent_after_rens,
                                         solve_submip_with_cuts]() {
                AsyncHeuristicCompletion completion;
                const detail::NeighborhoodHeuristicResult local_branching =
                    detail::run_local_branching_heuristic(
                        problem_, options_, relaxation, incumbent_after_rens.primal,
                        incumbent_after_rens.objective, solve_submip_with_cuts);
                completion.heuristic_lp_iterations = local_branching.lp_iterations;
                completion.heuristic_successes = local_branching.successes;
                completion.local_branching_successes = local_branching.successes;
                if (local_branching.incumbent.has_value()) {
                    completion.incumbent = IncumbentSnapshot{
                        .has_incumbent = true,
                        .objective = local_branching.incumbent->objective,
                        .primal = local_branching.incumbent->primal,
                    };
                }
                return completion;
            };
            maybe_run_async_submip_heuristic(std::move(local_branching_task));
            timing.local_branching_wall_ns += elapsed_ns_(step_start, SteadyClock::now());
        }
        timing.heuristics_wall_ns += elapsed_ns_(heuristics_start, SteadyClock::now());

        update_tree_node_(
            node.id, [&](TreeNode& tree_node) { tree_node.status = TreeNodeStatus::Fractional; });

        const auto pseudocost_before = pseudocosts_snapshot_();
        auto local_pseudocosts = pseudocost_before;
        const auto branching_start = SteadyClock::now();
        const bool branch_on_sos = sos_decision.variable >= 0;
        const Options branching_effective_options =
            effective_options_for_bound_(relaxation.objective);
        detail::BranchDecision decision =
            branch_on_sos
                ? std::move(sos_decision)
                : detail::choose_branching_variable(
                      node, relaxation, fractional, branching_effective_options, problem_.maximize,
                      local_pseudocosts, parallel_task_dispatcher_.get(), node_relaxation_solver);
        if (branch_on_sos) {
            const LPBasis* basis = node.basis ? &*node.basis : nullptr;
            decision.down_child.relaxation =
                node_relaxation_solver(decision.down_child.state, basis);
            decision.up_child.relaxation = node_relaxation_solver(decision.up_child.state, basis);
        }
        timing.branching_wall_ns += elapsed_ns_(branching_start, SteadyClock::now());
        timing.strong_branching_probe_count = decision.strong_branching_probe_count;
        timing.strong_branching_probe_iterations = decision.strong_branching_probe_iterations;
        timing.strong_branching_probe_core_solve_time_ns =
            decision.strong_branching_probe_core_solve_time_ns;
        timing.strong_branching_probe_lp_assembly_time_ns =
            decision.strong_branching_probe_lp_assembly_time_ns;
        timing.strong_branching_probe_lp_internal_presolve_ns =
            decision.strong_branching_probe_lp_internal_presolve_ns;
        timing.strong_branching_probe_lp_internal_crash_ns =
            decision.strong_branching_probe_lp_internal_crash_ns;
        timing.strong_branching_probe_lp_internal_iters_ns =
            decision.strong_branching_probe_lp_internal_iters_ns;
        timing.strong_branching_probe_lp_internal_serialize_ns =
            decision.strong_branching_probe_lp_internal_serialize_ns;
        note_strong_branching_work_(decision);
        merge_pseudocosts_(pseudocost_before, local_pseudocosts);
        if (decision.variable < 0) {
            update_tree_node_(
                node.id, [&](TreeNode& tree_node) { tree_node.status = TreeNodeStatus::Fathomed; });
            finalize_node_timing("branching", "fathomed");
            return;
        }

        timing.branch_variable = decision.variable;
        timing.branch_value = relaxation.primal(decision.variable);
        update_tree_node_(node.id, [&](TreeNode& tree_node) {
            tree_node.status = TreeNodeStatus::Branched;
            tree_node.branch_var = decision.variable;
            tree_node.branch_value = relaxation.primal(decision.variable);
        });

        const LPBasis* parent_basis = node.basis ? &*node.basis : nullptr;
        auto first_child = decision.down_child;
        auto second_child = decision.up_child;
        const Options child_effective_options = effective_options_();
        if (child_effective_options.node_selection == NodeSelectionStrategy::DepthFirst &&
            first_child.relaxation.has_value() && second_child.relaxation.has_value()) {
            const bool first_better = problem_.maximize
                                          ? (first_child.relaxation->objective >
                                             second_child.relaxation->objective + 1e-12)
                                          : (first_child.relaxation->objective <
                                             second_child.relaxation->objective - 1e-12);
            if (first_better) {
                std::swap(first_child, second_child);
            }
        }
        // HiGHS NodeData-style: record the sibling's LB when both were evaluated
        // in strong branching. Children inherit it so the parent's global bound
        // can later be tightened to min(child_lb, sibling_lb) in the cutoff logic.
        const auto sibling_bound_or_nan = [](const detail::ChildEvaluation& eval) {
            return eval.relaxation.has_value() &&
                           eval.relaxation->status == RelaxationStatus::Optimal
                       ? eval.relaxation->objective
                       : std::numeric_limits<double>::quiet_NaN();
        };
        const double first_sibling_bound = sibling_bound_or_nan(second_child);
        const double second_sibling_bound = sibling_bound_or_nan(first_child);
        process_child_(node.id, node.depth + 1, decision.variable, decision.value, parent_basis,
                       relaxation.objective, first_sibling_bound, std::move(first_child),
                       node_relaxation_solver, &timing, worker_id);
        if (should_terminate_()) {
            search_coordinator_.notify_all();
            finalize_node_timing("first_child", "branched");
            return;
        }
        process_child_(node.id, node.depth + 1, decision.variable, decision.value, parent_basis,
                       relaxation.objective, second_sibling_bound, std::move(second_child),
                       node_relaxation_solver, &timing, worker_id);
        search_coordinator_.notify_all();
        finalize_node_timing("children", "branched");
    }

    template <typename RelaxationSolver>
    void process_child_(int parent_id, int depth, int branch_variable, double branch_value,
                        const LPBasis* parent_basis, double inherited_bound, double sibling_bound,
                        detail::ChildEvaluation child, RelaxationSolver&& relaxation_solver,
                        NodeTimingRecord* parent_timing, int worker_id) {
        const auto child_start = SteadyClock::now();
        auto finalize_child_timing = [&]() {
            if (parent_timing != nullptr) {
                parent_timing->child_processing_wall_ns +=
                    elapsed_ns_(child_start, SteadyClock::now());
            }
        };
        detail::materialize_child_state(&child.state);
        detail::prepare_child_state_for_relaxation(&child.state);

        int child_id = -1;
        std::uint64_t order = 0;
        {
            std::lock_guard<std::mutex> lock(tree_mutex_);
            order = next_order_++;
            child_id = detail::append_tree_node(tree_nodes_, parent_id, depth, order);
            TreeNode& tree_node = tree_nodes_[child_id];
            tree_node.branch_var = branch_variable;
            tree_node.branch_value = branch_value;
            tree_node.sibling_bound = sibling_bound;
        }

        if (child.state.upper_bounds(branch_variable) + options_.integrality_tol <
            child.state.lower_bounds(branch_variable)) {
            maybe_learn_conflict_from_bounds_(child.state.lower_bounds, child.state.upper_bounds);
            update_tree_node_(child_id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Infeasible;
            });
            finalize_child_timing();
            return;
        }

        if (const detail::ConflictGraph* graph = conflict_graph_(); graph != nullptr) {
            if (child.state.reasons == nullptr ||
                static_cast<int>(child.state.reasons->size()) != graph->literal_count()) {
                child.state.reasons = std::make_shared<NodeReasonStore>(graph->literal_count());
            }
            const std::optional<int> fixed_literal = fixed_binary_literal_from_bounds_(
                branch_variable, child.state.lower_bounds, child.state.upper_bounds);
            if (fixed_literal.has_value()) {
                seed_fixed_literal_reason_(*fixed_literal, &child.state.reasons);
            }
        }

        if (!child.relaxation.has_value()) {
            update_tree_node_(child_id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Created;
                tree_node.bound = inherited_bound;
                tree_node.estimate = inherited_bound;
            });

            detail::ActiveNode active;
            active.id = child_id;
            active.parent_id = parent_id;
            active.depth = depth;
            active.order = order;
            active.bound = inherited_bound;
            active.estimate = inherited_bound;
            active.lower_bounds = child.state.lower_bounds;
            active.upper_bounds = child.state.upper_bounds;
            active.domain = child.state.domain;
            active.domain_change_count = child.state.domain_change_count;
            active.bounds_presolved = child.state.bounds_presolved;
            active.presolve_cuts_revision = child.state.presolve_cuts_revision;
            active.presolve_conflicts_revision = child.state.presolve_conflicts_revision;
            active.presolve_implications_revision = child.state.presolve_implications_revision;
            if (parent_basis != nullptr) {
                active.basis = *parent_basis;
            }
            active.reasons = child.state.reasons;
            finalize_child_timing();
            const Options effective = effective_options_for_bound_(inherited_bound);
            search_coordinator_.push(std::move(active), effective.node_selection, problem_.maximize,
                                     worker_id);
            maybe_log_progress_("child");
            return;
        }

        finalize_child_timing();

        note_lp_work_(*child.relaxation);
        const auto estimate = detail::node_estimate(*child.relaxation, problem_.variable_types,
                                                    pseudocosts_snapshot_(),
                                                    options_.integrality_tol, problem_.maximize);
        update_tree_node_(child_id, [&](TreeNode& tree_node) {
            tree_node.bound = child.relaxation->objective;
            tree_node.estimate = estimate;
        });
        maybe_log_progress_("child");

        if (child.relaxation->status == RelaxationStatus::Infeasible) {
            maybe_learn_conflict_from_bounds_(child.state.lower_bounds, child.state.upper_bounds);
            update_tree_node_(child_id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Infeasible;
            });
            return;
        }
        if (child.relaxation->status == RelaxationStatus::Unbounded) {
            update_tree_node_(child_id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Unbounded;
            });
            mark_unbounded_();
            return;
        }

        // Collect fractional candidates before bound check for cutoff tracking
        const auto fractional = detail::collect_fractional_candidates(
            child.relaxation->primal, problem_.variable_types, options_.integrality_tol);

        {
            const auto incumbent = incumbent_snapshot_();
            if (incumbent.has_incumbent &&
                bound_prunes_(child.relaxation->objective, incumbent.objective)) {
                // Record cutoff for fractional variables - they caused this node to be
                // pruned
                if (!fractional.empty()) {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    for (const auto& fc : fractional) {
                        if (fc.variable >= 0 &&
                            fc.variable < static_cast<int>(pseudocosts_.size())) {
                            // Record cutoff for both branches (HiGHS-style)
                            pseudocosts_[fc.variable].record_cutoff();
                        }
                    }
                }
                update_tree_node_(child_id, [&](TreeNode& tree_node) {
                    tree_node.status = TreeNodeStatus::PrunedByBound;
                });
                return;
            }
        }
        if (fractional.empty() &&
            detail::is_sos_feasible_solution(child.relaxation->primal, problem_.sos_constraints,
                                             options_.integrality_tol)) {
            update_tree_node_(child_id, [&](TreeNode& tree_node) {
                tree_node.status = TreeNodeStatus::Integral;
            });
            maybe_update_incumbent_(child.relaxation->primal, child.relaxation->objective);
            search_coordinator_.notify_all();
            return;
        }
        update_tree_node_(child_id,
                          [&](TreeNode& tree_node) { tree_node.status = TreeNodeStatus::Created; });
        detail::ActiveNode active;
        active.id = child_id;
        active.parent_id = parent_id;
        active.depth = depth;
        active.order = order;
        active.bound = child.relaxation->objective;
        active.estimate = estimate;
        active.lower_bounds = child.state.lower_bounds;
        active.upper_bounds = child.state.upper_bounds;
        active.domain = child.state.domain;
        active.domain_change_count = child.state.domain_change_count;
        active.bounds_presolved = child.state.bounds_presolved;
        active.presolve_cuts_revision = child.state.presolve_cuts_revision;
        active.presolve_conflicts_revision = child.state.presolve_conflicts_revision;
        active.presolve_implications_revision = child.state.presolve_implications_revision;
        active.basis = child.relaxation->basis;
        active.reasons = child.state.reasons;
        if (!active.basis.has_value() && parent_basis != nullptr) {
            // Preserve the parent tableau for warm-start reoptimization when
            // the child relaxation solver did not record a basis.
            active.basis = *parent_basis;
        }
        const Options effective = effective_options_for_bound_(active.bound);
        search_coordinator_.push(std::move(active), effective.node_selection, problem_.maximize,
                                 worker_id);
        maybe_log_progress_("bound");
        search_coordinator_.notify_all();
    }

    SolveResult finalize_result_(Status status) const {
        std::scoped_lock lock(stats_mutex_, incumbent_mutex_, tree_mutex_, cuts_mutex_);

        SolveResult result;
        result.status = status;
        result.objective =
            has_incumbent_ ? incumbent_objective_ : std::numeric_limits<double>::quiet_NaN();
        result.primal = has_incumbent_
                            ? incumbent_primal_
                            : Eigen::VectorXd::Constant(problem_.lower_bounds.size(),
                                                        std::numeric_limits<double>::quiet_NaN());
        result.best_bound = search_coordinator_.compute_best_bound(
            has_incumbent_, incumbent_objective_, problem_.maximize, root_relaxation_objective);
        // When optimally solved (tree exhausted), best_bound must equal the incumbent objective.
        if (status == Status::Optimal && has_incumbent_) {
            result.best_bound = incumbent_objective_;
        }
        result.root_relaxation_objective =
            root_relaxation_objective.value_or(std::numeric_limits<double>::quiet_NaN());
        result.node_count = node_count_;
        result.relaxation_solve_count = relaxation_solve_count_;
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
        result.warm_start_relaxation_attempt_count = warm_start_relaxation_attempt_count_;
        result.warm_start_relaxation_accept_count = warm_start_relaxation_accept_count_;
        result.warm_start_cold_retry_count = warm_start_cold_retry_count_;
        result.warm_start_relaxation_solve_count = warm_start_relaxation_solve_count_;
        result.strong_branching_probe_count = strong_branching_probe_count_;
        result.strong_branching_probe_iterations = strong_branching_probe_iterations_;
        result.lp_refactorizations = lp_refactorizations_;
        result.lp_eta_stack_depth_entry_sum = lp_eta_stack_depth_entry_sum_;
        result.lp_dual_pool_builds = lp_dual_pool_builds_;
        result.lp_primal_pool_builds = lp_primal_pool_builds_;
        result.lp_warm_factorization_reuse_count = lp_warm_factorization_reuse_count_;
        result.lp_warm_dual_weights_reuse_count = lp_warm_dual_weights_reuse_count_;
        result.relaxation_core_solve_time_ns = relaxation_core_solve_time_ns_;
        result.relaxation_lp_assembly_time_ns = relaxation_lp_assembly_time_ns_;
        result.relaxation_lp_internal_presolve_ns = relaxation_lp_internal_presolve_ns_;
        result.relaxation_lp_internal_crash_ns = relaxation_lp_internal_crash_ns_;
        result.relaxation_lp_internal_iters_ns = relaxation_lp_internal_iters_ns_;
        result.relaxation_lp_internal_serialize_ns = relaxation_lp_internal_serialize_ns_;
        result.relaxation_lp_lu_build_ns = relaxation_lp_lu_build_ns_;
        result.relaxation_lp_pricing_build_ns = relaxation_lp_pricing_build_ns_;
        result.relaxation_lp_pivot_ns = relaxation_lp_pivot_ns_;
        result.strong_branching_probe_core_solve_time_ns =
            strong_branching_probe_core_solve_time_ns_;
        result.strong_branching_probe_lp_assembly_time_ns =
            strong_branching_probe_lp_assembly_time_ns_;
        result.strong_branching_probe_lp_internal_presolve_ns =
            strong_branching_probe_lp_internal_presolve_ns_;
        result.strong_branching_probe_lp_internal_crash_ns =
            strong_branching_probe_lp_internal_crash_ns_;
        result.strong_branching_probe_lp_internal_iters_ns =
            strong_branching_probe_lp_internal_iters_ns_;
        result.strong_branching_probe_lp_internal_serialize_ns =
            strong_branching_probe_lp_internal_serialize_ns_;
        result.root_cut_generation_wall_ns = root_cut_generation_wall_ns_;
        result.root_cut_selection_wall_ns = root_cut_selection_wall_ns_;
        result.root_cut_activation_wall_ns = root_cut_activation_wall_ns_;
        result.root_cut_resolve_wall_ns = root_cut_resolve_wall_ns_;
        result.node_cut_generation_wall_ns = node_cut_generation_wall_ns_;
        result.node_cut_selection_wall_ns = node_cut_selection_wall_ns_;
        result.node_cut_resolve_wall_ns = node_cut_resolve_wall_ns_;
        result.rounding_heuristic_wall_ns = rounding_heuristic_wall_ns_;
        result.heuristics_wall_ns = heuristics_wall_ns_;
        result.feasibility_jump_wall_ns = feasibility_jump_wall_ns_;
        result.feasibility_pump_wall_ns = feasibility_pump_wall_ns_;
        result.diving_wall_ns = diving_wall_ns_;
        result.rens_wall_ns = rens_wall_ns_;
        result.rins_wall_ns = rins_wall_ns_;
        result.local_search_wall_ns = local_search_wall_ns_;
        result.local_branching_wall_ns = local_branching_wall_ns_;
        result.branching_wall_ns = branching_wall_ns_;
        result.child_processing_wall_ns = child_processing_wall_ns_;
        result.has_solution = has_incumbent_;
        result.tree_nodes = tree_nodes_;
        return result;
    }

    Problem problem_;
    Options options_;
    std::vector<detail::PseudoCost> pseudocosts_;
    std::unique_ptr<detail::ParallelDispatcher> parallel_task_dispatcher_;
    detail::SearchCoordinator search_coordinator_;
    detail::CutPool cut_pool_{options_};
    std::vector<Cut> active_cuts_;
    std::unordered_set<detail::CutSignature, detail::CutSignatureHash> active_cut_signatures_;
    std::vector<TreeNode> tree_nodes_;
    int node_count_ = 0;
    int relaxation_solve_count_ = 0;
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
    int warm_start_relaxation_attempt_count_ = 0;
    int warm_start_relaxation_accept_count_ = 0;
    int warm_start_cold_retry_count_ = 0;
    int warm_start_relaxation_solve_count_ = 0;
    int strong_branching_probe_count_ = 0;
    int strong_branching_probe_iterations_ = 0;
    int lp_refactorizations_ = 0;
    int lp_eta_stack_depth_entry_sum_ = 0;
    int lp_dual_pool_builds_ = 0;
    int lp_primal_pool_builds_ = 0;
    int lp_warm_factorization_reuse_count_ = 0;
    int lp_warm_dual_weights_reuse_count_ = 0;
    std::uint64_t relaxation_core_solve_time_ns_ = 0;
    std::uint64_t relaxation_lp_assembly_time_ns_ = 0;
    std::uint64_t relaxation_lp_internal_presolve_ns_ = 0;
    std::uint64_t relaxation_lp_internal_crash_ns_ = 0;
    std::uint64_t relaxation_lp_internal_iters_ns_ = 0;
    std::uint64_t relaxation_lp_internal_serialize_ns_ = 0;
    std::uint64_t relaxation_lp_lu_build_ns_ = 0;
    std::uint64_t relaxation_lp_pricing_build_ns_ = 0;
    std::uint64_t relaxation_lp_pivot_ns_ = 0;
    std::uint64_t strong_branching_probe_core_solve_time_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_assembly_time_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_crash_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_iters_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns_ = 0;
    std::uint64_t root_cut_generation_wall_ns_ = 0;
    std::uint64_t root_cut_selection_wall_ns_ = 0;
    std::uint64_t root_cut_activation_wall_ns_ = 0;
    std::uint64_t root_cut_resolve_wall_ns_ = 0;
    std::uint64_t node_cut_generation_wall_ns_ = 0;
    std::uint64_t node_cut_selection_wall_ns_ = 0;
    std::uint64_t node_cut_resolve_wall_ns_ = 0;
    std::uint64_t rounding_heuristic_wall_ns_ = 0;
    std::uint64_t heuristics_wall_ns_ = 0;
    std::uint64_t feasibility_jump_wall_ns_ = 0;
    std::uint64_t feasibility_pump_wall_ns_ = 0;
    std::uint64_t diving_wall_ns_ = 0;
    std::uint64_t rens_wall_ns_ = 0;
    std::uint64_t rins_wall_ns_ = 0;
    std::uint64_t local_search_wall_ns_ = 0;
    std::uint64_t local_branching_wall_ns_ = 0;
    std::uint64_t branching_wall_ns_ = 0;
    std::uint64_t child_processing_wall_ns_ = 0;
    std::unique_ptr<detail::AsyncTaskDispatcher> async_heuristic_dispatcher_;
    std::mutex async_heuristic_completion_mutex_;
    std::deque<AsyncHeuristicCompletion> async_heuristic_completions_;
    std::vector<LearnedConflict> learned_conflicts_;
    std::vector<std::vector<ConflictLiteral>> learned_implications_;
    std::vector<detail::DivingStrategyStats> diving_stats_;
    std::ofstream node_timing_log_stream_;
    std::uint64_t next_order_ = 0;
    bool has_incumbent_ = false;
    double incumbent_objective_ = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd incumbent_primal_;
    std::optional<LPBasis> root_warm_start_basis_state_;
    std::optional<double> root_relaxation_objective;
    // Root LP data for global reduced-cost fixing (lurking bounds).
    // Protected by incumbent_mutex_.
    Eigen::VectorXd root_reduced_costs_;
    std::vector<LPBasisStatus> root_basis_statuses_;
    double root_lp_objective_ = std::numeric_limits<double>::quiet_NaN();
    // Global domain: tightest bounds valid for the entire remaining tree.
    detail::GlobalDomain global_domain_;
    bool progress_header_printed_ = false;
    int last_logged_node_count_ = 0;
    double last_logged_best_bound_ = std::numeric_limits<double>::quiet_NaN();
    double last_logged_incumbent_ = std::numeric_limits<double>::quiet_NaN();
    double last_logged_gap_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<Cut> initial_cuts_;
    mutable std::once_flag conflict_graph_once_;
    mutable std::unique_ptr<detail::ConflictGraph> conflict_graph_cache_;
    mutable std::mutex tree_mutex_;
    mutable std::mutex incumbent_mutex_;
    mutable std::mutex cuts_mutex_;
    mutable std::mutex learning_mutex_;
    mutable std::mutex stats_mutex_;
    mutable std::mutex progress_mutex_;
    mutable std::mutex node_timing_log_mutex_;
};

#include "bnb/async_heuristic_manager.tpp"

} // namespace simplex::bnb
