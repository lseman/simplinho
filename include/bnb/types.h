#pragma once

#include "../simplex/simplex_types.h"
#include <Eigen/Dense>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

#include "simplex/simplex_types.h"

namespace simplex::bnb {

enum class VariableType { Continuous, Integer, Binary };
enum class Status { Optimal, Infeasible, Unbounded, NodeLimit };
enum class NodeSelectionStrategy {
    DepthFirst,
    BreadthFirst,
    BestBound,
    BestEstimate,
    Hybrid,
    BestFirstPlunging,
    BestEstimatePlunging,
    InterleavedBestFirstBestEstimatePlunging
};
enum class BranchingStrategy { MostFractional, PseudoCost, StrongBranching };
enum class DivingStrategy {
    Disabled,
    Fractional,
    VectorLength,
    ObjectiveValue,
    Coefficient,
    Guided,
    Adaptive
};
enum class RelaxationStatus { Optimal, Infeasible, Unbounded };
enum class LinearConstraintSense { LessEqual, GreaterEqual, Equal };
enum class SOSType { SOS1, SOS2 };
enum class TreeNodeStatus {
    Created,
    Fractional,
    Integral,
    Infeasible,
    Unbounded,
    PrunedByBound,
    Branched,
    Fathomed
};

inline const char* to_string(Status status) {
    switch (status) {
        case Status::Optimal:
            return "optimal";
        case Status::Infeasible:
            return "infeasible";
        case Status::Unbounded:
            return "unbounded";
        case Status::NodeLimit:
            return "nodelimit";
    }
    return "unknown";
}

inline const char* to_string(RelaxationStatus status) {
    switch (status) {
        case RelaxationStatus::Optimal:
            return "optimal";
        case RelaxationStatus::Infeasible:
            return "infeasible";
        case RelaxationStatus::Unbounded:
            return "unbounded";
    }
    return "unknown";
}

inline const char* to_string(BranchingStrategy strategy) {
    switch (strategy) {
        case BranchingStrategy::MostFractional:
            return "most_fractional";
        case BranchingStrategy::PseudoCost:
            return "pseudocost";
        case BranchingStrategy::StrongBranching:
            return "strong_branching";
    }
    return "unknown";
}

inline const char* to_string(DivingStrategy strategy) {
    switch (strategy) {
        case DivingStrategy::Disabled:
            return "disabled";
        case DivingStrategy::Fractional:
            return "fractional_diving";
        case DivingStrategy::VectorLength:
            return "vector_length_diving";
        case DivingStrategy::ObjectiveValue:
            return "objective_diving";
        case DivingStrategy::Coefficient:
            return "coefficient_diving";
        case DivingStrategy::Guided:
            return "guided_diving";
        case DivingStrategy::Adaptive:
            return "adaptive_diving";
    }
    return "unknown";
}

inline const char* to_string(NodeSelectionStrategy strategy) {
    switch (strategy) {
        case NodeSelectionStrategy::DepthFirst:
            return "depth_first";
        case NodeSelectionStrategy::BreadthFirst:
            return "breadth_first";
        case NodeSelectionStrategy::BestBound:
            return "best_bound";
        case NodeSelectionStrategy::BestEstimate:
            return "best_estimate";
        case NodeSelectionStrategy::Hybrid:
            return "hybrid";
        case NodeSelectionStrategy::BestFirstPlunging:
            return "best_first_plunging";
        case NodeSelectionStrategy::BestEstimatePlunging:
            return "best_estimate_plunging";
        case NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging:
            return "interleaved_best_first_best_estimate_plunging";
    }
    return "unknown";
}

inline const char* to_string(TreeNodeStatus status) {
    switch (status) {
        case TreeNodeStatus::Created:
            return "created";
        case TreeNodeStatus::Fractional:
            return "fractional";
        case TreeNodeStatus::Integral:
            return "integral";
        case TreeNodeStatus::Infeasible:
            return "infeasible";
        case TreeNodeStatus::Unbounded:
            return "unbounded";
        case TreeNodeStatus::PrunedByBound:
            return "pruned_by_bound";
        case TreeNodeStatus::Branched:
            return "branched";
        case TreeNodeStatus::Fathomed:
            return "fathomed";
    }
    return "unknown";
}

struct Options {
    int max_nodes = 10'000;
    int parallel_workers = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    double integrality_tol = 1e-6;
    bool verbose = false;
    int log_frequency = 100;
    std::string node_timing_log_path;
    NodeSelectionStrategy node_selection = NodeSelectionStrategy::DepthFirst;
    int hybrid_depth_bias = 5;
    int plunging_bestfreq = 10;
    BranchingStrategy branching_strategy = BranchingStrategy::PseudoCost;
    DivingStrategy diving_strategy = DivingStrategy::Disabled;
    int strong_branching_candidates = 4; // Max candidates for strong branching
    int strong_branching_k = 2; // Number of candidates for reduced strong branching (Highs-like)
    int strong_branching_max_depth = 1;
    int strong_branching_lp_iter_limit = 24; // Per-probe LP iteration cap; <=0 means full solve
    int pseudocost_reliability = 2;
    int max_dive_depth = 15;     // Reduced from 25 for faster heuristics
    int max_dive_lp_solves = 32; // Reduced from 64 for faster heuristics
    int heuristic_frequency = 8;
    int heuristic_max_depth = 12;
    bool use_rounding = true;
    bool use_diving = true; // enable/disable all diving heuristics (sync and async)
    bool use_rins = false;
    double rins_fix_ratio = 0.7;
    double rins_tolerance = 1e-4;
    bool use_rens = false;
    double rens_fix_ratio = 0.75;
    bool use_local_search = false;
    int local_search_iterations = 8;
    int local_search_max_free_vars = 3;
    bool use_local_branching = false;
    double local_branching_neighborhood_ratio = 0.15;
    bool use_async_heuristics = true;
    int local_branching_min_radius = 4;
    int local_branching_max_radius = 20;
    double local_branching_fix_agree_ratio = 0.35;
    double local_branching_lp_agreement_tol = 1e-4;
    bool use_feasibility_pump = false;
    int feasibility_pump_iterations = 6;
    double feasibility_pump_fix_ratio = 0.6;
    bool use_feasibility_jump = false;
    int feasibility_jump_iterations = 12;
    int feasibility_jump_max_free_vars = 4;
    double feasibility_jump_objective_weight = 0.05;
    int heuristic_subproblem_max_nodes = 64;
    bool use_cut_pool = true;
    int max_cut_rounds_per_node = 1;       // Reduced from 2 for faster solving
    int max_root_cut_rounds = 2;           // Allow a dedicated root repeat pass
    int max_cuts_added_per_round = 4;      // Reduced from 8 for faster cut selection
    int max_root_cuts_added_per_round = 8; // Higher budget for root cut selection
    int max_cut_pool_size = 128;           // Reduced from 256 for faster cut management
    double min_cut_violation = 1e-4;
    int max_cut_age = 5;
    double cut_age_decay = 0.08;           // decay factor used when scoring and retaining cuts
    double cut_selection_age_bonus = 0.10; // age bonus factor used when scoring cut candidates
    bool use_gomory_cuts = true;
    bool use_mir_cuts = true; // Enable MIR cuts (more powerful than Gomory cover)
    bool use_cover_cuts = true;
    bool use_zero_half_cuts = false;
    bool use_implied_bound_cuts = true;
    bool use_clique_cuts = true;
    bool use_graph_clique_cuts = true; // enable the new graph-based clique separator
    bool use_odd_cycle_cuts = true;
    bool use_probing_implications = true;
    int probing_max_candidates = 4;
    bool use_conflict_cuts = true;
    int max_conflict_cuts_per_round = 8;
    // HiGHS-style conflict pool sizing.
    int max_conflict_pool_size = 512;
    int max_conflict_age = 10;
    int max_cuts_per_type = 4;
    double cut_max_parallelism = 0.98;
    bool use_dual_proof_cuts = true;
    bool use_lp_reoptimization_profile = true;
    bool use_quadratic_warm_start_repair = false;
    bool use_node_presolve = true;
    bool use_node_presolve_on_warm_basis = false;
    // Adaptive proof phase: after an incumbent exists, switch effort from
    // primal search toward proving the dual bound.
    bool use_adaptive_proof_phase = false;
    int proof_phase_min_nodes = 8;
    NodeSelectionStrategy proof_node_selection = NodeSelectionStrategy::BestBound;
    int proof_strong_branching_candidates = 8;
    int proof_strong_branching_k = 4;
    int proof_strong_branching_max_depth = 4;
    int proof_max_cut_rounds_per_node = 2;
    int proof_max_cuts_added_per_round = 8;
    bool proof_use_node_presolve_on_warm_basis = false;
    // HiGHS-inspired separate tolerances
    double feasibility_tol = 1e-7; // Feasibility checks in relaxation (LP)
    double optimality_tol = 1e-7;  // LP optimality checks (reduced-cost dual feasibility)
    // integrality_tol is already defined above with default 1e-6
    // Absolute MIP gap: stop once |best_bound - incumbent| <= mip_abs_gap.
    double mip_abs_gap = 1e-6;
    // Relative MIP gap: stop once |best_bound - incumbent| / max(1, |incumbent|) <= mip_rel_gap.
    // HiGHS default is 1e-4. The gap is used for early termination only
    // (HiGHS-style optimality_limit); it never widens the per-node fathoming
    // cutoff, which would risk losing the true optimum.
    double mip_rel_gap = 1e-4;
    // Cache control for parallel mode
    bool enable_parallel_cache = true; // Enable thread-local cache for reduced lock contention
};

struct SparseLinearConstraint {
    std::vector<int> indices;
    std::vector<double> values;
    double rhs = 0.0;
    LinearConstraintSense sense = LinearConstraintSense::LessEqual;
};

struct SOSConstraint {
    SOSType type = SOSType::SOS1;
    std::vector<int> variables;
    std::vector<double> weights;
};

struct Cut {
    std::vector<int> indices;
    std::vector<double> values;
    double rhs = 0.0;
    LinearConstraintSense sense = LinearConstraintSense::LessEqual;
    std::string cut_type;
    double strength = 0.0;
    int times_used = 0;
    int age = 0;
};

struct Problem {
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    Eigen::VectorXd objective_coefficients;
    double objective_constant = 0.0;
    std::vector<VariableType> variable_types;
    std::vector<SparseLinearConstraint> base_constraints;
    std::vector<SOSConstraint> sos_constraints;
    bool maximize = false;
    std::vector<int> warm_start_basis; // Basis column indices from root relaxation for warm-start
    std::optional<LPBasis> warm_start_basis_state;
};

struct RelaxationSolution {
    RelaxationStatus status = RelaxationStatus::Infeasible;
    Eigen::VectorXd primal;
    double objective = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    std::optional<LPBasis> basis;
    std::optional<LPSolution> lp_solution;
    bool attempted_warm_start_basis_state = false;
    bool used_warm_start_basis_state = false;
    bool cold_retried_after_warm_start = false;
    std::uint64_t core_solve_time_ns = 0;
    std::uint64_t lp_assembly_time_ns = 0;
    std::uint64_t lp_internal_presolve_ns = 0;
    std::uint64_t lp_internal_crash_ns = 0;
    std::uint64_t lp_internal_iters_ns = 0;
    std::uint64_t lp_internal_serialize_ns = 0;
};

struct TreeNode {
    int id = -1;
    int parent_id = -1;
    int depth = 0;
    std::uint64_t order = 0;
    TreeNodeStatus status = TreeNodeStatus::Created;
    double bound = std::numeric_limits<double>::quiet_NaN();
    double estimate = std::numeric_limits<double>::quiet_NaN();
    // HiGHS NodeData-style: the LB of the sibling branch evaluated during
    // strong branching / probing. Used to tighten the parent's global bound
    // when both children have been evaluated.
    double sibling_bound = std::numeric_limits<double>::quiet_NaN();
    int branch_var = -1;
    double branch_value = std::numeric_limits<double>::quiet_NaN();
};

struct SolveResult {
    Status status = Status::Infeasible;
    Eigen::VectorXd primal;
    double objective = std::numeric_limits<double>::quiet_NaN();
    double best_bound = std::numeric_limits<double>::quiet_NaN();
    double root_relaxation_objective = std::numeric_limits<double>::quiet_NaN();
    int root_presolve_tightened_bounds = 0;
    int root_presolve_removed_rows = 0;
    int root_presolve_removed_coeffs = 0;
    int root_presolve_aggregations = 0;
    int node_count = 0;
    int relaxation_solve_count = 0;
    int lp_iterations = 0;
    int incumbent_updates = 0;
    int heuristic_lp_iterations = 0;
    int heuristic_successes = 0;
    int feasibility_jump_successes = 0;
    int feasibility_pump_successes = 0;
    int rens_successes = 0;
    int rins_successes = 0;
    int local_search_successes = 0;
    int local_branching_successes = 0;
    int cuts_generated = 0;
    int cuts_applied = 0;
    int duplicate_cuts = 0;
    int cut_pool_size = 0;
    int warm_start_relaxation_attempt_count = 0;
    int warm_start_relaxation_accept_count = 0;
    int warm_start_cold_retry_count = 0;
    int warm_start_relaxation_solve_count = 0;
    int strong_branching_probe_count = 0;
    int strong_branching_probe_iterations = 0;
    int lp_refactorizations = 0;
    int lp_eta_stack_depth_entry_sum = 0;
    int lp_dual_pool_builds = 0;
    int lp_primal_pool_builds = 0;
    int lp_warm_factorization_reuse_count = 0;
    int lp_warm_dual_weights_reuse_count = 0;
    std::uint64_t relaxation_core_solve_time_ns = 0;
    std::uint64_t relaxation_lp_assembly_time_ns = 0;
    std::uint64_t relaxation_lp_internal_presolve_ns = 0;
    std::uint64_t relaxation_lp_internal_crash_ns = 0;
    std::uint64_t relaxation_lp_internal_iters_ns = 0;
    std::uint64_t relaxation_lp_internal_serialize_ns = 0;
    std::uint64_t relaxation_lp_lu_build_ns = 0;
    std::uint64_t relaxation_lp_pricing_build_ns = 0;
    std::uint64_t relaxation_lp_pivot_ns = 0;
    std::uint64_t strong_branching_probe_core_solve_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_assembly_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_crash_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_iters_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns = 0;
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
    std::string lp_profile = "model_options";
    bool warm_start_basis_state_used = false;
    std::string lp_mode = "auto";
    bool lp_partial_pricing = false;
    std::string lp_dual_pricing = "row";
    bool has_solution = false;
    std::vector<TreeNode> tree_nodes;
    bool has_incumbent = false;
};

} // namespace simplex::bnb
