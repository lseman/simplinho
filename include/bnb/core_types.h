#pragma once

#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "bnb/types.h"

namespace simplex::bnb::detail {
struct ReasonLiteral;
struct PropagationReason;
using NodeReasonStore = std::vector<PropagationReason>;
} // namespace simplex::bnb::detail

namespace simplex::bnb {

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

struct AsyncHeuristicCompletion {
    struct DivingStatsDelta {
        int attempts = 0;
        int successes = 0;
        int lp_iterations = 0;
        int lp_solves = 0;
    };

    int heuristic_lp_iterations = 0;
    int heuristic_successes = 0;
    int feasibility_jump_successes = 0;
    int feasibility_pump_successes = 0;
    int rens_successes = 0;
    int rins_successes = 0;
    int local_search_successes = 0;
    int local_branching_successes = 0;
    std::vector<DivingStatsDelta> diving_stats_delta;
    std::optional<IncumbentSnapshot> incumbent;
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

using SteadyClock = std::chrono::steady_clock;

struct RelaxationTimingBucket {
    int solve_count = 0;
    int warm_start_attempt_count = 0;
    int warm_start_accept_count = 0;
    int warm_start_cold_retry_count = 0;
    std::uint64_t wall_ns = 0;
    std::uint64_t presolve_wall_ns = 0;
    std::uint64_t core_solve_time_ns = 0;
    std::uint64_t lp_assembly_time_ns = 0;
    std::uint64_t lp_internal_presolve_ns = 0;
    std::uint64_t lp_internal_crash_ns = 0;
    std::uint64_t lp_internal_iters_ns = 0;
    std::uint64_t lp_internal_serialize_ns = 0;
};

struct NodeTimingRecord {
    int node_id = -1;
    int parent_id = -1;
    int depth = 0;
    std::uint64_t order = 0;
    bool allow_root_cuts = false;
    bool root_node = false;
    int branch_variable = -1;
    double branch_value = std::numeric_limits<double>::quiet_NaN();
    double final_bound = std::numeric_limits<double>::quiet_NaN();
    double final_estimate = std::numeric_limits<double>::quiet_NaN();
    double final_relaxation_objective = std::numeric_limits<double>::quiet_NaN();
    int fractional_count = 0;
    int root_cut_rounds = 0;
    int root_cuts_generated = 0;
    int root_cuts_selected = 0;
    int root_cuts_applied = 0;
    int node_cut_rounds = 0;
    int node_cuts_generated = 0;
    int node_cuts_selected = 0;
    int node_cuts_applied = 0;
    int strong_branching_probe_count = 0;
    int strong_branching_probe_iterations = 0;
    int async_heuristic_launch_attempts = 0;
    const char* exit_stage = "unknown";
    const char* final_status = "unknown";
    RelaxationTimingBucket node_relaxation;
    RelaxationTimingBucket child_relaxation;
    std::uint64_t total_wall_ns = 0;
    std::uint64_t root_cut_generation_wall_ns = 0;
    std::uint64_t root_cut_selection_wall_ns = 0;
    std::uint64_t root_cut_activation_wall_ns = 0;
    std::uint64_t root_cut_resolve_wall_ns = 0;
    std::uint64_t node_cut_generation_wall_ns = 0;
    std::uint64_t node_cut_selection_wall_ns = 0;
    std::uint64_t node_cut_resolve_wall_ns = 0;
    std::uint64_t rounding_heuristic_wall_ns = 0;
    std::uint64_t heuristics_wall_ns = 0;
    std::uint64_t feasibility_jump_wall_ns = 0;
    std::uint64_t feasibility_pump_wall_ns = 0;
    std::uint64_t diving_wall_ns = 0;
    std::uint64_t rens_wall_ns = 0;
    std::uint64_t rins_wall_ns = 0;
    std::uint64_t local_search_wall_ns = 0;
    std::uint64_t local_branching_wall_ns = 0;
    std::uint64_t branching_wall_ns = 0;
    std::uint64_t child_processing_wall_ns = 0;
    std::uint64_t strong_branching_probe_core_solve_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_assembly_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_crash_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_iters_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns = 0;
};

} // namespace simplex::bnb
