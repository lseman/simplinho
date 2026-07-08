#pragma once

#include <Eigen/Dense>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Public result container
// ============================================================================
enum class LPBasisStatus { Basic, AtLower, AtUpper, Fixed };

struct LPWarmStateData;

struct LPBasis {
    std::vector<LPBasisStatus> column_status;
    std::vector<int> basis_columns;
    std::shared_ptr<LPWarmStateData> warm_state;
};

struct RevisedSimplexTiming {
    std::uint64_t presolve_ns = 0;
    std::uint64_t crash_ns = 0;
    std::uint64_t simplex_iters_ns = 0;
    std::uint64_t serialization_ns = 0;
};

// Per-solve counters for the LP hot path. Populated by RevisedSimplex::solve*()
// so callers (e.g. branch-and-bound) can aggregate across many solves and
// identify wasted work (full refactorizations, DSE rebuilds) that should have
// been avoided via warm-start reuse.
struct LPSolveStats {
    int refactorizations = 0;           // full LU rebuilds (dense_refactor_/sparse_refactor_)
    int eta_stack_depth_entry = 0;      // inherited FT/eta depth when the solve started
    int ft_updates = 0;                 // final FT/eta chain length at end of solve
    int dual_pool_builds = 0;           // full DSE/Devex weight rebuilds
    int primal_pool_builds = 0;         // full primal edge-weight rebuilds
    int warm_start_attempted = 0;       // 0/1: a prior basis was supplied
    int warm_start_accepted = 0;        // 0/1: warm basis was used to seed the pivot loop
    int warm_start_cold_retry = 0;      // 0/1: warm basis failed, fell back to cold start
    int warm_factorization_reused = 0;  // 0/1: reused cached LU/eta state
    int warm_dual_weights_reused = 0;   // 0/1: reused cached dual pricing weights
    int dual_row_price_calls = 0;       // sparse row-wise dual PRICE calls
    int dual_col_price_calls = 0;       // column-wise dual PRICE calls
    int dual_price_switches = 0;        // runtime row/column pricing switches
    double dual_row_ep_density = 0.0;   // running density of BTRAN pivotal rows
    double dual_row_ap_density = 0.0;   // running density of priced tableau rows
    double dual_col_aq_density = 0.0;   // running density of FTRAN pivotal columns
    std::uint64_t lu_build_ns = 0;      // cumulative time in refactor calls
    std::uint64_t pricing_build_ns = 0; // cumulative time in build_*_pool calls
    std::uint64_t pivot_ns = 0;         // cumulative time in basis update / pivot maintenance
};

struct LPSolution {
    enum class Status {
        Optimal,
        Unbounded,
        Infeasible,
        IterLimit,
        Singular,
        NeedPhase1,
        ObjectiveBound,
    };

    Status status{};
    Eigen::VectorXd x; // primal solution (original space)
    double obj = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> basis;                          // basis indices in original problem
    std::vector<int> basis_internal;                 // basis indices in solved internal model
    std::vector<int> nonbasis_internal;              // nonbasic indices in solved internal model
    std::vector<std::string> internal_column_labels; // labels for internal cols
    std::vector<std::string> internal_row_labels;    // labels for internal rows
    Eigen::MatrixXd tableau;                         // B^{-1} A for the final internal basis
    Eigen::VectorXd tableau_rhs;                     // B^{-1} b for the final internal basis
    Eigen::VectorXd reduced_costs_internal;          // c - A^T y on the internal model
    Eigen::VectorXd dual_values;                     // duals/shadow prices on the original rows
    Eigen::VectorXd shadow_prices;                   // alias of dual_values
    Eigen::VectorXd dual_values_internal;            // y = B^{-T} c_B on the internal rows
    Eigen::VectorXd shadow_prices_internal;          // alias of dual_values_internal
    bool has_internal_tableau = false;
    int iters = 0;                                     // total iterations (Phase I + II)
    std::unordered_map<std::string, std::string> info; // telemetry
    std::vector<std::string> trace;                    // verbose trace, if enabled
    LPBasis basis_state;                 // reusable warm start in the original column space
    Eigen::VectorXd farkas_y;            // Farkas certificate of infeasibility (if any)
    Eigen::VectorXd farkas_y_internal;   // Farkas certificate on internal rows
    bool farkas_has_cert = false;        // whether farkas_y is valid
    Eigen::VectorXd primal_ray;          // primal unbounded ray (original space)
    Eigen::VectorXd primal_ray_internal; // primal unbounded ray (internal cols)
    bool primal_ray_has_cert = false;    // whether primal_ray is valid
    RevisedSimplexTiming timing;         // profiling
    LPSolveStats solve_stats;            // per-solve counters for LP hot path
};

inline const char* to_string(LPSolution::Status s) {
    switch (s) {
        case LPSolution::Status::Optimal:
            return "optimal";
        case LPSolution::Status::Unbounded:
            return "unbounded";
        case LPSolution::Status::Infeasible:
            return "infeasible";
        case LPSolution::Status::IterLimit:
            return "iterlimit";
        case LPSolution::Status::Singular:
            return "singular";
        case LPSolution::Status::NeedPhase1:
            return "need_phase1";
        case LPSolution::Status::ObjectiveBound:
            return "objective_bound";
    }
    return "unknown";
}

// ============================================================================
// Options
// ============================================================================
enum class SimplexMode { Auto, Primal, Dual };

struct RevisedSimplexOptions {
    // Global
    int max_iters = 50'000;
    double tol = 1e-9;
    bool bland = false;
    double svd_tol = 1e-8;
    double ratio_delta = 1e-12;
    double ratio_eta = 1e-7;
    double deg_step_tol = 1e-12;
    double epsilon_cost = 1e-10;
    int rng_seed = 13;

    // Basis / LU
    int refactor_every = 64; // FT hard cap
    int compress_every = 32; // FT soft cap
    double lu_pivot_rel = 1e-12;
    double lu_abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;
    double ft_multiplier_guard = 1e8;
    std::string basis_update = "hybrid"; // "forrest_tomlin" | "eta" | "hybrid"
    int ft_bandwidth_cap = 12;
    double max_growth_tol = 1e3;
    double min_dynamic_growth_tol = 500;
    double max_condition_estimate = 1e13;
    int basis_refinement_steps = 3;
    double basis_residual_refactor_tol = 1e-9;
    double basis_refinement_stall_progress_ratio = 0.8;
    int basis_refinement_stall_limit = 3;
    int basis_max_eta_count = 128;
    double basis_column_residual_tol = 1e-8;
    bool basis_aggressive_residual_rebuild = true;
    std::string basis_sparse_backend = "auto"; // "auto"/"ft" | "pf" | "eigen"
    bool basis_sparse_equilibration = true;
    double basis_sparse_rhs_density_threshold = 0.40;

    // Pricing
    int devex_reset = 100;
    std::string pricing_rule = "adaptive"; // or "devex" / "most_negative"
    int adaptive_reset_freq = 400;
    bool partial_pricing = true;         // HiGHS-inspired partial pricing
    std::string dual_pricing = "switch"; // "row" | "col" | "switch" (HiGHS-inspired)
    int row_pricing_threshold = 40;      // switch if row density < this
    std::string primal_edge_weight_strategy =
        "dense_diagonal"; // "dense" | "diagonal" | "dense_diagonal"
    std::string dual_edge_weight_strategy =
        "dense_diagonal";           // "dense" | "diagonal" | "dense_diagonal"
    int dual_flip_max_per_iter = 4; // avoid pathological flip storms
    double primal_steepest_edge_weight_log_error_threshold =
        1.3862943611198906; // log(4), equivalent to 25% acceptance
    double dual_steepest_edge_weight_log_error_threshold =
        1.3862943611198906; // log(4), equivalent to 25% acceptance
    int parallel_pricing_workers = 1;
    int parallel_pricing_min_cols = 2048;

    // NLA framework switching — Devex weight error accumulation
    double framework_switch_threshold = 1.3862943611198906; // log(4)
    int framework_switch_consecutive = 3;
    bool allow_framework_switch = true;
    // Price strategy: "col" | "row_switch" | "row_switch_col_switch"
    std::string price_strategy = "col";

    // Native bounded-variable simplex: solve l <= x <= u directly in the
    // engines (nonbasic-at-lower/upper, two-sided ratio test). When false,
    // finite bounds are converted to extra rows/columns via the standard-form
    // reformulation. Free variables (both bounds infinite) always use the
    // reformulation.
    bool native_bounds = true;

    // Opt-in explicit dualization for sparse nonnegative equality-form LPs.
    // "off" keeps the current path; "on" always tries; "auto" uses the row/col ratio.
    std::string dualization = "off";
    double dualization_min_row_col_ratio = 4.0;
    int dualization_max_recovery_cols = 8192;

    // Refinement safeguards
    double residual_abs_refactor_tol = 1e-10;
    int refinement_max_steps = 6;
    double refinement_slow_progress_ratio = 0.5;
    double primal_simplex_cost_perturbation_multiplier = 1.0;
    double dual_simplex_cost_perturbation_multiplier = 1.0;
    // Bound perturbation for the primal simplex (HiGHS-style).
    // When enabled, basic variables that exceed their bounds due to numerical
    // drift get their bounds shifted outward instead of triggering a Phase-1
    // re-solve. Multiplier scales the perturbation slack magnitude.
    bool primal_simplex_bound_perturbation = true;
    double primal_simplex_bound_perturbation_multiplier = 1.0;
    // PAMI-style parallel dual simplex: process up to this many leaving rows
    // per outer iteration using std::async workers. 1 = serial (default).
    int dual_pami_rows = 1;

    // Recovery
    int max_basis_rebuilds = 3;
    int crash_attempts = 4;
    double crash_markowitz_tol = 0.2;
    std::string crash_strategy = "hybrid";
    bool repair_mapped_basis = true;
    bool use_quadratic_warm_start_repair = false;

    // Algorithm selection/tuning
    bool dual_allow_bound_flip = true;  // enable Beale bound-flipping
    double dual_flip_pivot_tol = 1e-10; // |pN(e)| below this ⇒ consider flip
    double dual_flip_rc_tol = 1e-10;    // |rN(e)| “near dual-feasible”

    // Algorithm selection
    SimplexMode mode = SimplexMode::Auto; // Auto | Primal | Dual

    // Objective-bound early termination (HiGHS-style).
    // When set, the dual engine bails out if the internal primal objective
    // (in the engine's c_work space) provably exceeds objective_bound_internal.
    // The check runs every objective_bound_check_freq iterations.
    // Disabled when objective_bound_internal is +inf.
    double objective_bound_internal = std::numeric_limits<double>::infinity();
    int objective_bound_check_freq = 16;

    // BNB warm-start hint: when true, the dual engine will not apply its
    // reactive cost perturbation. HiGHS skips perturbation entirely when the
    // basis is "near-optimal" (the typical warm-start state in MIP); the
    // BNB layer flips this on for the warm solver only. Default false to
    // preserve cold-solve behavior.
    bool dual_suppress_perturbation_when_warm = false;
    bool dual_warm_start_near_optimal = false;

    // Verbose diagnostics
    bool verbose = false;
    int verbose_every = 1;
    bool verbose_include_basis = true;
    bool verbose_include_presolve = true;
    bool disable_presolve = false;

    // Heavy diagnostic expansion. Disabled by default so production solves
    // return only primal/dual/basis/certificates on the hot path.
    bool compute_tableau = false;
    bool compute_reduced_costs = false;
};
