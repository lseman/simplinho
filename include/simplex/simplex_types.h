#pragma once

#include <Eigen/Dense>

#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Public result container
// ============================================================================
enum class LPBasisStatus { Basic, AtLower, AtUpper, Fixed };

struct LPBasis {
    std::vector<LPBasisStatus> column_status;
};

struct LPSolution {
    enum class Status {
        Optimal,
        Unbounded,
        Infeasible,
        IterLimit,
        Singular,
        NeedPhase1
    };

    Status status{};
    Eigen::VectorXd x;  // primal solution (original space)
    double obj = std::numeric_limits<double>::quiet_NaN();
    std::vector<int> basis;  // basis indices in original problem
    std::vector<int> basis_internal;  // basis indices in solved internal model
    std::vector<int> nonbasis_internal;  // nonbasic indices in solved internal model
    std::vector<std::string> internal_column_labels;  // labels for internal cols
    std::vector<std::string> internal_row_labels;  // labels for internal rows
    Eigen::MatrixXd tableau;  // B^{-1} A for the final internal basis
    Eigen::VectorXd tableau_rhs;  // B^{-1} b for the final internal basis
    Eigen::VectorXd reduced_costs_internal;  // c - A^T y on the internal model
    Eigen::VectorXd dual_values;  // duals/shadow prices on the original rows
    Eigen::VectorXd shadow_prices;  // alias of dual_values
    Eigen::VectorXd dual_values_internal;  // y = B^{-T} c_B on the internal rows
    Eigen::VectorXd shadow_prices_internal;  // alias of dual_values_internal
    bool has_internal_tableau = false;
    int iters = 0;           // total iterations (Phase I + II)
    std::unordered_map<std::string, std::string> info;  // telemetry
    std::vector<std::string> trace;  // verbose trace, if enabled
    LPBasis basis_state;  // reusable warm start in the original column space
    Eigen::VectorXd farkas_y;  // Farkas certificate of infeasibility (if any)
    Eigen::VectorXd farkas_y_internal;  // Farkas certificate on internal rows
    bool farkas_has_cert = false;  // whether farkas_y is valid
    Eigen::VectorXd primal_ray;  // primal unbounded ray (original space)
    Eigen::VectorXd primal_ray_internal;  // primal unbounded ray (internal cols)
    bool primal_ray_has_cert = false;  // whether primal_ray is valid
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
    int refactor_every = 128;  // FT hard cap
    int compress_every = 64;   // FT soft cap
    double lu_pivot_rel = 1e-12;
    double lu_abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;
    std::string basis_update = "forrest_tomlin";  // or "eta"
    int ft_bandwidth_cap = 16;

    // Pricing
    int devex_reset = 200;
    std::string pricing_rule = "adaptive";  // or "devex" / "most_negative"
    int adaptive_reset_freq = 1000;

    // Recovery
    int max_basis_rebuilds = 3;
    int crash_attempts = 4;
    double crash_markowitz_tol = 0.2;
    std::string crash_strategy = "hybrid";
    bool repair_mapped_basis = true;

    // Algorithm selection/tuning
    bool dual_allow_bound_flip = true;   // enable Beale bound-flipping
    double dual_flip_pivot_tol = 1e-10;  // |pN(e)| below this ⇒ consider flip
    double dual_flip_rc_tol = 1e-10;     // |rN(e)| “near dual-feasible”
    int dual_flip_max_per_iter = 2;      // avoid pathological flip storms

    // Algorithm selection
    SimplexMode mode = SimplexMode::Auto;  // Auto | Primal | Dual

    // Verbose diagnostics
    bool verbose = false;
    int verbose_every = 1;
    bool verbose_include_basis = true;
    bool verbose_include_presolve = true;
};
