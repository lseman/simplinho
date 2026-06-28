#pragma once

// -----------------------------------------------------------------------------
// Revised Simplex (header-only, drop-in compatible) — now with Dual Simplex
// Public API preserved: LPSolution, to_string, RevisedSimplexOptions,
//                       RevisedSimplex{ ctor, solve(...) }.
// Internals tidied without behavioral changes, plus a dual simplex phase:
//   - Options::mode = {Auto, Primal, Dual}
//   - Auto tries primal, and if primal reports negative basic variables,
//     falls back to dual before Phase I.
//   - You can force Dual by setting options.mode = SimplexMode::Dual.
// -----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simplex/engine/degeneracy.h"    // DegeneracyManager + perturbation helpers
#include "simplex/presolve/presolver.h"     // presolve::LP, Presolver
#include "simplex/engine/pricer.h"        // pricing + degeneracy helpers
#include "simplex/factorization/simplex_lu.h"    // FTBasis implementation (solve_B, solve_BT, replace_column, refactor)
#include "simplex/types/simplex_types.h" // public result/status/options types
#include "simplex/presolve/sparse_presolver.h" // sparse-native LP presolve
#include "simplex/nla/simplex_nla.h"     // SimplexNLA — PF updates, iterate snapshots, framework switching
#include <atomic>
#include <cstdio>
#include <cstdlib>

// Forward decls from degeneracy/pricer header (kept external)
static inline std::unordered_map<std::string, std::string>
dm_stats_to_map(const DegeneracyManager::Stats& s) {
    std::unordered_map<std::string, std::string> info;
    info["deg_streak"] = std::to_string(s.degeneracy_streak);
    info["deg_total"] = std::to_string(s.degeneracy_total);
    info["cycle_len"] = std::to_string(s.suspected_cycling);
    info["basis_repeat_hits"] = std::to_string(s.repeated_basis_hits);
    info["basis_cycle_hits"] = std::to_string(s.basis_cycle_hits);
    info["cond_est"] = std::to_string(s.cond_est);
    info["deg_thresh"] = std::to_string(s.adaptive_deg_threshold);
    info["deg_epoch"] = std::to_string(s.epoch);
    return info;
}

struct LPDualPricingWarmState {
    enum class Rule { None, SteepestEdge, Devex, RowPricing, MostInfeasible };

    Rule active_rule = Rule::None;
    std::vector<double> row_weights;
    std::vector<char> prefer_row_pricing;
};

struct LPWarmStateData {
    std::uint64_t matrix_signature = 0;
    std::uint64_t basis_matrix_signature = 0;
    int rows = -1;
    int cols = -1;
    bool matrix_is_sparse = false;
    std::vector<int> basis_columns;
    std::shared_ptr<simplex::nla::SimplexNLA> nla;
    std::optional<LPDualPricingWarmState> dual_pricing_state;
};

// ============================================================================
// RevisedSimplex
// ============================================================================
class RevisedSimplexPrimalEngine;
class RevisedSimplexDualEngine;

class RevisedSimplex {
  public:
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using PhaseResult = std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
                                   std::unordered_map<std::string, std::string>>;

    explicit RevisedSimplex(RevisedSimplexOptions opt = {})
        : opt_(std::move(opt)), rng_(opt_.rng_seed), degen_(opt_.rng_seed),
          adaptive_pricer_(1) // initialized to a dummy size; rebuilt per solve
    {}

    struct SolveTraceScope {
        RevisedSimplex& self;
        bool root = false;

        explicit SolveTraceScope(RevisedSimplex& owner)
            : self(owner), root(owner.solve_depth_++ == 0) {
            if (root)
                self.trace_.clear();
        }

        ~SolveTraceScope() { --self.solve_depth_; }
    };

    // Main entry (drop-in compatible)
    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        trace_line_("[solve dense] disable_presolve=" + std::to_string(opt_.disable_presolve));
        const int n = static_cast<int>(A_in.cols());
        const LPBasis* implicit_basis = nullptr;
        if (!basis_opt && has_cached_basis_state(A_in)) {
            implicit_basis = &*cached_basis_state_;
        }
        LPSolution sol =
            solve_impl_(A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
                        Eigen::VectorXd::Constant(n, presolve::inf()), basis_opt, implicit_basis);
        update_cached_basis_(sol, A_in.rows(), Eigen::VectorXd::Zero(n),
                             Eigen::VectorXd::Constant(n, presolve::inf()));
        return sol;
    }

    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const LPBasis& warm_start) {
        const int n = static_cast<int>(A_in.cols());
        LPSolution sol =
            solve_impl_(A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
                        Eigen::VectorXd::Constant(n, presolve::inf()), std::nullopt, &warm_start);
        update_cached_basis_(sol, A_in.rows(), Eigen::VectorXd::Zero(n),
                             Eigen::VectorXd::Constant(n, presolve::inf()));
        return sol;
    }

    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());
        const LPBasis* implicit_basis = nullptr;
        const Eigen::VectorXd default_l = Eigen::VectorXd::Zero(n);
        const Eigen::VectorXd default_u = Eigen::VectorXd::Constant(n, presolve::inf());
        if (!basis_opt && has_cached_basis_state(A_in) && cached_basis_bounds_match_(l_in, u_in)) {
            implicit_basis = &*cached_basis_state_;
        }
        LPSolution sol = solve_impl_(A_in, b_in, c_in, l_in, u_in, basis_opt, implicit_basis);
        update_cached_basis_(sol, A_in.rows(), l_in, u_in);
        return sol;
    }

    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in, const LPBasis& warm_start) {
        LPSolution sol = solve_impl_(A_in, b_in, c_in, l_in, u_in, std::nullopt, &warm_start);
        update_cached_basis_(sol, A_in.rows(), l_in, u_in);
        return sol;
    }

    LPSolution solve(const SparseMatrix& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        trace_line_("[solve sparse] disable_presolve=" + std::to_string(opt_.disable_presolve));
        const int n = static_cast<int>(A_in.cols());
        const LPBasis* implicit_basis = nullptr;
        const Eigen::VectorXd default_l = Eigen::VectorXd::Zero(n);
        const Eigen::VectorXd default_u = Eigen::VectorXd::Constant(n, presolve::inf());
        if (!basis_opt && has_cached_basis_state(A_in) &&
            cached_basis_bounds_match_(default_l, default_u)) {
            implicit_basis = &*cached_basis_state_;
        }
        LPSolution sol = solve_impl_sparse_(A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
                                            Eigen::VectorXd::Constant(n, presolve::inf()),
                                            basis_opt, implicit_basis);
        update_cached_basis_(sol, A_in.rows(), Eigen::VectorXd::Zero(n),
                             Eigen::VectorXd::Constant(n, presolve::inf()));
        return sol;
    }

    LPSolution solve(const SparseMatrix& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const LPBasis& warm_start) {
        const int n = static_cast<int>(A_in.cols());
        LPSolution sol = solve_impl_sparse_(A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
                                            Eigen::VectorXd::Constant(n, presolve::inf()),
                                            std::nullopt, &warm_start);
        update_cached_basis_(sol, A_in.rows(), Eigen::VectorXd::Zero(n),
                             Eigen::VectorXd::Constant(n, presolve::inf()));
        return sol;
    }

    LPSolution solve(const SparseMatrix& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());
        const LPBasis* implicit_basis = nullptr;
        if (!basis_opt && has_cached_basis_state(A_in)) {
            // has_cached_basis_state already requires opt_.mode == Dual.
            // In Dual mode pass the cached basis even when bounds changed:
            // rebase_basis_state_for_bounds_ adjusts non-basic statuses, then the dual
            // simplex restores primal feasibility in O(pivots).  This is the HiGHS/SCIP
            // hot-restart pattern and avoids a full crash+simplex on every BnB node.
            implicit_basis = &*cached_basis_state_;
        }
        LPSolution sol =
            solve_impl_sparse_(A_in, b_in, c_in, l_in, u_in, basis_opt, implicit_basis);
        update_cached_basis_(sol, A_in.rows(), l_in, u_in);
        return sol;
    }

    LPSolution solve(const SparseMatrix& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in, const LPBasis& warm_start) {
        LPSolution sol =
            solve_impl_sparse_(A_in, b_in, c_in, l_in, u_in, std::nullopt, &warm_start);
        update_cached_basis_(sol, A_in.rows(), l_in, u_in);
        return sol;
    }

    void clear_basis_cache() {
        cached_basis_state_.reset();
        cached_basis_rows_ = -1;
        cached_basis_cols_ = -1;
        cached_basis_l_.resize(0);
        cached_basis_u_.resize(0);
    }

    bool cached_basis_bounds_match_(const Eigen::VectorXd& l,
                                    const Eigen::VectorXd& u) const noexcept {
        if (cached_basis_l_.size() != l.size() || cached_basis_u_.size() != u.size()) {
            return false;
        }
        for (int j = 0; j < l.size(); ++j) {
            if (cached_basis_l_(j) != l(j) || cached_basis_u_(j) != u(j)) {
                return false;
            }
        }
        return true;
    }

    bool has_cached_basis_state(int rows, int cols) const noexcept {
        return cached_basis_state_ && cached_basis_rows_ == rows && cached_basis_cols_ == cols &&
               basis_state_matches_problem_(*cached_basis_state_, rows, cols);
    }

    // HiGHS-style per-call cutoff. The bound is interpreted in this solver's
    // internal c-space (i.e. the c the caller passes to solve()), so the BNB
    // layer is responsible for mapping incumbent_obj <-> internal cutoff.
    // Pass +inf to disable.
    void set_objective_bound_internal(double bound) noexcept {
        opt_.objective_bound_internal = bound;
    }

    double objective_bound_internal() const noexcept { return opt_.objective_bound_internal; }

  private:
    LPSolution solve_impl_(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                           const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                           const Eigen::VectorXd& u_in, std::optional<std::vector<int>> basis_opt,
                           const LPBasis* basis_state_opt) {
        auto t0_presolve = std::chrono::steady_clock::now();
        SolveTraceScope trace_scope(*this);
        degen_.reset();
        const int n = static_cast<int>(A_in.cols());
        if (b_in.size() != A_in.rows()) {
            throw std::invalid_argument("simplex: b size mismatch with rows(A)");
        }
        if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
            throw std::invalid_argument("simplex: c/l/u sizes must equal cols(A)");
        }
        begin_solve_(matrix_signature_(A_in), static_cast<int>(A_in.rows()), n, false,
                     basis_state_opt);

        trace_line_("[solve] start m=" + std::to_string(A_in.rows()) + " n=" + std::to_string(n));
        trace_line_("[solve] disable_presolve=" + std::to_string(opt_.disable_presolve));

        const auto sanitized_bounds = canonicalize_inactive_huge_bounds_(A_in, b_in, l_in, u_in);
        const Eigen::VectorXd& l_use = sanitized_bounds.l;
        const Eigen::VectorXd& u_use = sanitized_bounds.u;
        if (sanitized_bounds.relaxed_upper > 0 || sanitized_bounds.relaxed_lower > 0) {
            trace_line_("[solve] relaxed huge inactive bounds upper=" +
                        std::to_string(sanitized_bounds.relaxed_upper) +
                        " lower=" + std::to_string(sanitized_bounds.relaxed_lower));
        }

        std::optional<LPBasis> rebased_basis_state_opt = std::nullopt;
        if (basis_state_opt && !basis_state_opt->column_status.empty()) {
            rebased_basis_state_opt =
                rebase_basis_state_for_bounds_(*basis_state_opt, l_use, u_use, opt_.tol);
            basis_state_opt = &*rebased_basis_state_opt;
        }

        // When a reusable factorization is in hand (same matrix, Dual mode), prefer
        // its basis over any passed-in seed: starting the dual re-solve from exactly
        // that basis lets try_reuse_factorization_ adopt the existing LU and FT-pivot
        // instead of refactoring. Otherwise the passed-in basis_opt (a different
        // basis) forces a cold factorization on every node solve.
        if (auto factorized_seed = warm_factorization_basis_seed_()) {
            basis_opt = std::move(factorized_seed);
        }

        if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
            !basis_state_opt->column_status.empty()) {
            if ((int)basis_state_opt->column_status.size() != n) {
                throw std::invalid_argument(
                    "simplex: warm-start basis column_status size mismatch");
            }
        }

        if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
            !basis_state_opt->column_status.empty()) {
            basis_opt =
                basis_columns_from_basis_state_(*basis_state_opt, static_cast<int>(A_in.rows()));
        }

        bool is_nonnegative_standard = true;
        for (int j = 0; j < n; ++j) {
            const bool l_is_zero = std::isfinite(l_use(j)) && std::abs(l_use(j)) <= opt_.tol;
            const bool u_is_inf = !std::isfinite(u_use(j));
            if (!l_is_zero || !u_is_inf) {
                is_nonnegative_standard = false;
                break;
            }
        }

        if (!is_nonnegative_standard) {
            struct ReformVar {
                int y = -1;
                int y_pos = -1;
                int y_neg = -1;
                int upper_slack = -1;
                double shift = 0.0;
                int sign = 1;
                bool uses_single_var = false;
                bool has_upper_row = false;
            };

            std::vector<ReformVar> map(n);
            std::vector<int> single_y(n, -1);
            std::vector<int> upper_slack(n, -1);
            std::vector<int> split_pos(n, -1);
            std::vector<int> split_neg(n, -1);
            int nv = 0;
            int upper_rows = 0;
            double obj_shift = 0.0;

            for (int j = 0; j < n; ++j) {
                const bool has_l = std::isfinite(l_use(j));
                const bool has_u = std::isfinite(u_use(j));

                if (has_l && has_u && u_use(j) < l_use(j) - opt_.tol) {
                    Eigen::VectorXd xnan =
                        Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                    return finalize_solution_(
                        make_solution_(LPSolution::Status::Infeasible, xnan,
                                       std::numeric_limits<double>::infinity(), {}, 0,
                                       {{"reason", "invalid_bounds"}}));
                }

                const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= opt_.tol;
                if (fixed) {
                    map[j].uses_single_var = true;
                    map[j].y = -1;
                    single_y[j] = -1;
                    map[j].shift = l_use(j);
                    map[j].sign = 0;
                    obj_shift += c_in(j) * l_use(j);
                } else if (has_l) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    single_y[j] = map[j].y;
                    map[j].shift = l_use(j);
                    map[j].sign = 1;
                    obj_shift += c_in(j) * l_use(j);
                    if (has_u) {
                        map[j].has_upper_row = true;
                        ++upper_rows;
                    }
                } else if (has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    single_y[j] = map[j].y;
                    map[j].shift = u_use(j);
                    map[j].sign = -1;
                    obj_shift += c_in(j) * u_use(j);
                } else {
                    map[j].y_pos = nv++;
                    map[j].y_neg = nv++;
                    split_pos[j] = map[j].y_pos;
                    split_neg[j] = map[j].y_neg;
                }
            }

            const int m_eq = static_cast<int>(A_in.rows());
            const int n_total = nv + upper_rows;
            const int m_total = m_eq + upper_rows;
            trace_line_("[solve] bound reformulation nv=" + std::to_string(nv) + " upper_rows=" +
                        std::to_string(upper_rows) + " total_m=" + std::to_string(m_total) +
                        " total_n=" + std::to_string(n_total));

            Eigen::MatrixXd A_std = Eigen::MatrixXd::Zero(m_total, n_total);
            Eigen::VectorXd b_std = Eigen::VectorXd::Zero(m_total);
            Eigen::VectorXd c_std = Eigen::VectorXd::Zero(n_total);
            Eigen::VectorXd l_std = Eigen::VectorXd::Zero(n_total);
            Eigen::VectorXd u_std = Eigen::VectorXd::Constant(n_total, presolve::inf());

            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    if (map[j].y >= 0)
                        c_std(map[j].y) += static_cast<double>(map[j].sign) * c_in(j);
                } else {
                    c_std(map[j].y_pos) += c_in(j);
                    c_std(map[j].y_neg) += -c_in(j);
                }
            }

            int row = 0;
            for (int i = 0; i < m_eq; ++i, ++row) {
                double rhs = b_in(i);
                for (int j = 0; j < n; ++j) {
                    const double aij = A_in(i, j);
                    if (std::abs(aij) <= 1e-16)
                        continue;
                    if (map[j].uses_single_var) {
                        rhs -= aij * map[j].shift;
                        if (map[j].y >= 0)
                            A_std(row, map[j].y) += static_cast<double>(map[j].sign) * aij;
                    } else {
                        A_std(row, map[j].y_pos) += aij;
                        A_std(row, map[j].y_neg) += -aij;
                    }
                }
                b_std(row) = rhs;
            }

            int upper_row = 0;
            for (int j = 0; j < n; ++j) {
                if (!map[j].has_upper_row)
                    continue;
                const int slack = nv + upper_row;
                map[j].upper_slack = slack;
                upper_slack[j] = slack;
                A_std(row, map[j].y) = 1.0;
                A_std(row, slack) = 1.0;
                b_std(row) = u_use(j) - l_use(j);
                ++upper_row;
                ++row;
            }

            std::optional<std::vector<int>> basis_std = std::nullopt;
            std::optional<LPBasis> basis_state_std = std::nullopt;
            if (basis_opt && !basis_opt->empty()) {
                std::vector<int> cand;
                cand.reserve(std::min(m_eq, (int)basis_opt->size()) + upper_rows);
                for (int jorig : *basis_opt) {
                    if (jorig < 0 || jorig >= n)
                        continue;
                    if (map[jorig].uses_single_var) {
                        if (map[jorig].y >= 0)
                            cand.push_back(map[jorig].y);
                    } else if (map[jorig].y_pos >= 0) {
                        cand.push_back(map[jorig].y_pos);
                    }
                    if ((int)cand.size() == m_eq)
                        break;
                }
                for (int j = 0; j < n; ++j) {
                    if (map[j].upper_slack >= 0)
                        cand.push_back(map[j].upper_slack);
                }
                if ((int)cand.size() == m_total)
                    basis_std = std::move(cand);
            }
            if (basis_state_opt && !basis_state_opt->column_status.empty() &&
                (int)basis_state_opt->column_status.size() == n) {
                const bool exact_basis_state =
                    basis_state_matches_problem_(*basis_state_opt, m_eq, n);
                basis_state_std =
                    exact_basis_state
                        ? map_reformulated_basis_state_(*basis_state_opt, l_use, u_use, n_total,
                                                        single_y, upper_slack, split_pos, split_neg)
                        : map_reformulated_basis_seed_state_(*basis_state_opt, n_total, single_y,
                                                             upper_slack, split_pos, split_neg);
            }

            // When no warm-start basis is available, construct a logical basis from the
            // reformulation structure. Upper-bound slack columns are identity columns for their
            // rows; original rows are covered by picking the first y/y_pos with a nonzero entry.
            if (!basis_std && (!basis_state_std || basis_state_std->column_status.empty())) {
                std::vector<int> cand;
                cand.reserve(m_total);
                std::vector<bool> col_used(n_total, false);
                // Original rows: pick first available y/y_pos with nonzero in that row
                for (int i = 0; i < m_eq; ++i) {
                    int chosen = -1;
                    for (int j = 0; j < n && chosen < 0; ++j) {
                        if (map[j].uses_single_var) {
                            const int col = map[j].y;
                            if (col >= 0 && !col_used[col] && std::abs(A_std(i, col)) > 1e-14) {
                                chosen = col;
                            }
                        } else if (map[j].y_pos >= 0) {
                            const int col = map[j].y_pos;
                            if (!col_used[col] && std::abs(A_std(i, col)) > 1e-14) {
                                chosen = col;
                            }
                        }
                    }
                    if (chosen < 0)
                        break;
                    col_used[chosen] = true;
                    cand.push_back(chosen);
                }
                // Upper-bound rows: each has a dedicated slack that is an identity column
                for (int j = 0; j < n; ++j) {
                    if (map[j].upper_slack >= 0)
                        cand.push_back(map[j].upper_slack);
                }
                if ((int)cand.size() == m_total)
                    basis_std = std::move(cand);
            }

            LPSolution std_sol;
            bool reformulated_retry_used = false;
            std::optional<std::vector<int>> reformulated_basis_guess = basis_std;
            if ((!reformulated_basis_guess || reformulated_basis_guess->empty()) &&
                basis_state_std && !basis_state_std->column_status.empty()) {
                reformulated_basis_guess =
                    basis_columns_from_basis_state_(*basis_state_std, m_total);
            }
            std::optional<BasisQuality> reformulated_warm_basis_quality = std::nullopt;
            if (reformulated_basis_guess && !reformulated_basis_guess->empty()) {
                reformulated_warm_basis_quality = evaluate_basis_quality_(
                    A_std, b_std, c_std, *reformulated_basis_guess, opt_.tol);
            }
            const bool use_dual_first = opt_.mode != SimplexMode::Primal &&
                                        reformulated_warm_basis_quality &&
                                        reformulated_warm_basis_quality->valid &&
                                        reformulated_warm_basis_quality->dual_feasible;
            const char* reformulated_initial_mode =
                use_dual_first ? "dual"
                : (opt_.mode == SimplexMode::Primal ? "primal" : "auto");
            auto solve_reformulated = [&](SimplexMode mode) {
                RevisedSimplexOptions solve_opt = opt_;
                solve_opt.mode = (mode == SimplexMode::Auto ? SimplexMode::Primal : mode);
                solve_opt.disable_presolve = true;
                trace_line_("[solve_reformulated] disable_presolve=" +
                            std::to_string(solve_opt.disable_presolve));
                RevisedSimplex reformulated_solver(solve_opt);
                return basis_state_std ? reformulated_solver.solve(A_std, b_std, c_std, l_std,
                                                                   u_std, *basis_state_std)
                                       : reformulated_solver.solve(A_std, b_std, c_std, l_std,
                                                                   u_std, basis_std);
            };
            try {
                std_sol = solve_reformulated(use_dual_first ? SimplexMode::Dual : opt_.mode);
            } catch (const std::exception& e) {
                std_sol.status = LPSolution::Status::Singular;
                std_sol.info["where"] = "bound_reformulation_recursive_solve";
                std_sol.info["what"] = e.what();
            }
            if (use_dual_first && (std_sol.status == LPSolution::Status::Singular ||
                                   std_sol.status == LPSolution::Status::NeedPhase1 ||
                                   std_sol.status == LPSolution::Status::IterLimit)) {
                std_sol = solve_reformulated(SimplexMode::Auto);
                reformulated_retry_used = true;
            }
            if (opt_.mode == SimplexMode::Dual && !use_dual_first &&
                (std_sol.status == LPSolution::Status::NeedPhase1 ||
                 std_sol.status == LPSolution::Status::Singular)) {
                std_sol = solve_reformulated(SimplexMode::Auto);
                reformulated_retry_used = true;
            }
            if (std_sol.status == LPSolution::Status::Singular &&
                (basis_std.has_value() || basis_state_std.has_value())) {
                basis_std.reset();
                basis_state_std.reset();
                std_sol = solve_reformulated(SimplexMode::Auto);
                reformulated_retry_used = true;
            }

            Eigen::VectorXd x =
                Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            if (std_sol.x.size() == n_total && std_sol.x.array().isFinite().all()) {
                for (int j = 0; j < n; ++j) {
                    if (map[j].uses_single_var) {
                        x(j) = map[j].y >= 0
                                    ? map[j].shift + static_cast<double>(map[j].sign) * std_sol.x(map[j].y)
                                    : map[j].shift; // fixed variable: x = l = u
                    } else {
                        const double yp = std_sol.x(map[j].y_pos);
                        const double yn = std_sol.x(map[j].y_neg);
                        x(j) = yp - yn;
                    }
                }
            }

            std::vector<int> basis_out;
            std::vector<char> seen(n, 0);
            for (int idx : std_sol.basis) {
                for (int j = 0; j < n; ++j) {
                    const bool matches_single = map[j].uses_single_var && map[j].y == idx;
                    const bool matches_split =
                        !map[j].uses_single_var && (map[j].y_pos == idx || map[j].y_neg == idx);
                    if ((matches_single || matches_split) && !seen[j]) {
                        seen[j] = 1;
                        basis_out.push_back(j);
                        break;
                    }
                }
            }

            auto info = std_sol.info;
            info["bound_reformulation"] = "1";
            info["bound_reformulation_initial_mode"] = reformulated_initial_mode;
            if (reformulated_warm_basis_quality) {
                info["bound_reformulation_warm_start_valid"] =
                    reformulated_warm_basis_quality->valid ? "1" : "0";
                info["bound_reformulation_warm_start_primal_feasible"] =
                    reformulated_warm_basis_quality->primal_feasible ? "1" : "0";
                info["bound_reformulation_warm_start_dual_feasible"] =
                    reformulated_warm_basis_quality->dual_feasible ? "1" : "0";
            }
            info["original_m"] = std::to_string(m_eq);
            info["original_l"] = serialize_double_vec_(l_in);
            info["original_u"] = serialize_double_vec_(u_in);
            if (sanitized_bounds.relaxed_upper > 0) {
                info["input_upper_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_upper);
            }
            if (sanitized_bounds.relaxed_lower > 0) {
                info["input_lower_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_lower);
            }
            if (reformulated_retry_used) {
                info["bound_reformulation_retry_mode"] = "auto";
            }
            const double obj =
                x.array().isFinite().all()
                    ? c_in.dot(x)
                    : (std::isfinite(std_sol.obj) ? (std_sol.obj + obj_shift) : std_sol.obj);
            auto sol = make_solution_(std_sol.status, std::move(x), obj, std::move(basis_out),
                                      std_sol.iters, std::move(info), std_sol.farkas_y,
                                      std_sol.farkas_has_cert, std_sol.primal_ray,
                                      std_sol.primal_ray_has_cert);
            sol.basis_internal = std_sol.basis_internal;
            sol.nonbasis_internal = std_sol.nonbasis_internal;
            sol.internal_column_labels = std_sol.internal_column_labels;
            sol.internal_row_labels = std_sol.internal_row_labels;
            sol.tableau = std_sol.tableau;
            sol.tableau_rhs = std_sol.tableau_rhs;
            sol.reduced_costs_internal = std_sol.reduced_costs_internal;
            if (std_sol.dual_values.size() >= m_eq) {
                sol.dual_values = std_sol.dual_values.head(m_eq);
                sol.shadow_prices = sol.dual_values;
            }
            sol.dual_values_internal = std_sol.dual_values_internal;
            sol.shadow_prices_internal = std_sol.shadow_prices_internal;
            sol.farkas_y_internal = std_sol.farkas_y_internal;
            sol.primal_ray_internal = std_sol.primal_ray_internal;
            const LPBasis warm_basis =
                compute_basis_state_(sol.basis, sol.x, l_in, u_in, opt_.tol, m_eq);
            const std::string warm_basis_serialized =
                serialize_basis_state_from_primal_(sol.basis, sol.x, l_in, u_in, opt_.tol, m_eq);
            sol.basis_state = warm_basis;
            sol.info["warm_start_basis_state"] = warm_basis_serialized;
            if (std_sol.primal_ray_has_cert && std_sol.primal_ray.size() == n_total) {
                Eigen::VectorXd ray = Eigen::VectorXd::Zero(n);
                for (int j = 0; j < n; ++j) {
                    if (map[j].uses_single_var) {
                        ray(j) = static_cast<double>(map[j].sign) * std_sol.primal_ray(map[j].y);
                    } else {
                        const double pos =
                            (map[j].y_pos >= 0) ? std_sol.primal_ray(map[j].y_pos) : 0.0;
                        const double neg =
                            (map[j].y_neg >= 0) ? std_sol.primal_ray(map[j].y_neg) : 0.0;
                        ray(j) = pos - neg;
                    }
                }
                sol.primal_ray = clip_small_(ray);
            }
            sol.has_internal_tableau = std_sol.has_internal_tableau;
            return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
        }

        Eigen::MatrixXd A_model = A_in;
        Eigen::VectorXd b_model = b_in;
        Eigen::VectorXd c_model = c_in;
        Eigen::VectorXd l_model = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd u_model = Eigen::VectorXd::Constant(n, presolve::inf());
        Eigen::VectorXd anchor = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd sign = Eigen::VectorXd::Ones(n);

        for (int j = 0; j < n; ++j) {
            const bool has_l = std::isfinite(l_use(j));
            const bool has_u = std::isfinite(u_use(j));
            if (!has_l && !has_u) {
                throw std::invalid_argument(
                    "simplex: free variables are unsupported in solve(A,b,c,l,u)");
            }

            if (has_l) {
                anchor(j) = l_use(j);
                l_model(j) = 0.0;
                u_model(j) = has_u ? (u_use(j) - l_use(j)) : presolve::inf();
            } else {
                anchor(j) = u_use(j);
                sign(j) = -1.0;
                l_model(j) = 0.0;
                u_model(j) = presolve::inf();
                A_model.col(j) = -A_model.col(j);
                c_model(j) = -c_model(j);
            }

            if (anchor(j) != 0.0)
                b_model.noalias() -= A_model.col(j) * anchor(j);
        }

        // ---- (0) Wrap into presolve LP: Ax=b, default bounds, costs=c ----
        presolve::LP lp;
        lp.A = A_model;
        lp.b = b_model;
        lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
        lp.c = c_model;
        lp.l = l_model;
        lp.u = u_model;
        lp.c0 = c_in.dot(anchor);

        const bool warm_start_requested =
            (basis_opt && !basis_opt->empty()) ||
            (basis_state_opt && !basis_state_opt->column_status.empty());

        // ---- (1) Presolve ----
        presolve::Presolver::Options popt;
        popt.enable_rowreduce = !warm_start_requested;
        popt.enable_scaling = !warm_start_requested;
        popt.enable_objective_probing =
            !warm_start_requested && A_in.rows() <= 200 && A_in.cols() <= 200;
        popt.non_destructive = warm_start_requested;
        popt.allow_structural_changes = false;
        // Reoptimization should keep the LP matrix/basis mapping intact.
        // Even non-destructive fixed-variable presolve can zero columns after a
        // branch bound change and destabilize dual warm starts on basic
        // variables. HiGHS/SCIP-style hot starts are much happier when the
        // matrix is left alone and only bounds change.
        popt.max_passes = warm_start_requested ? 0 : 8;
        popt.probing_max_rounds = warm_start_requested ? 0 : 1;
        popt.probing_max_vars = warm_start_requested ? 0 : 8;
        if (opt_.disable_presolve) {
            trace_line_("[solve] disable_presolve=1");
            popt.enable_rowreduce = false;
            popt.enable_scaling = false;
            popt.enable_col_scaling = false;
            popt.enable_objective_probing = false;
            popt.max_passes = 0;
            popt.probing_max_rounds = 0;
            popt.probing_max_vars = 0;
        }
        if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
            popt.conservative_mode = true;
        }

        presolve::Presolver P(popt);
        const auto pres = P.run(lp);
        trace_presolve_(pres);

        if (pres.proven_infeasible) {
            return finalize_solution_(make_solution_(
                LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::infinity(), {}, 0, {{"presolve", "infeasible"}}));
        }
        if (pres.proven_unbounded) {
            Eigen::VectorXd xnan =
                Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            return finalize_solution_(make_solution_(LPSolution::Status::Unbounded, xnan,
                                                     -std::numeric_limits<double>::infinity(), {},
                                                     0, {{"presolve", "unbounded"}}));
        }

        const Eigen::MatrixXd& Atil = pres.reduced.A;
        const Eigen::VectorXd& btil = pres.reduced.b;
        const Eigen::VectorXd& ctil = pres.reduced.c;
        const Eigen::VectorXd& lred = pres.reduced.l;
        const Eigen::VectorXd& ured = pres.reduced.u;

        // ---- (2) m==0 fast path: optimize over bounds only ----
        if (Atil.rows() == 0) {
            Eigen::VectorXd vred = Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
            bool is_bounded = true;
            for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
                if (ctil(j) > opt_.tol) {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                } else if (ctil(j) < -opt_.tol) {
                    if (std::isfinite(ured(j)))
                        vred(j) = ured(j);
                    else {
                        is_bounded = false;
                        break;
                    }
                } else {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                }
            }
            if (!is_bounded) {
                Eigen::VectorXd xnan =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Unbounded, xnan, -std::numeric_limits<double>::infinity(),
                    {}, 0, {{"presolve", "m=0 neg cost & +inf upper"}}));
            }
            auto [z_full, obj_corr] = P.postsolve(vred);
            z_full =
                repair_nan_primal_(A_model, b_model, l_model, u_model, std::move(z_full), opt_.tol);
            Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
            const double total_obj = c_in.dot(x_full);
            auto sol = make_solution_(LPSolution::Status::Optimal, std::move(x_full), total_obj, {},
                                      0, {{"presolve", "m=0 optimized over bounds"}});
            sol.dual_values = clip_small_vec_(P.postsolve_dual(Eigen::VectorXd::Zero(0)), opt_.tol);
            sol.shadow_prices = sol.dual_values;
            return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
        }

        // ---- (3) Solve reduced problem directly with explicit bounds ----
        const bool bypass_postsolve = false;
        Eigen::MatrixXd Ared = Atil;
        Eigen::VectorXd bred = btil;
        Eigen::VectorXd cred = ctil;
        std::vector<int> col_orig_map = pres.orig_col_index;
        std::vector<int> row_orig_map = pres.orig_row_index;
        const std::vector<std::string> internal_column_labels =
            make_internal_column_labels_(col_orig_map);
        const std::vector<std::string> internal_row_labels =
            make_internal_row_labels_(row_orig_map);
        const int m_eff = static_cast<int>(Ared.rows());
        const int n_eff = static_cast<int>(Ared.cols());

        // Effective bounds (reduced space)
        Eigen::VectorXd l_eff = lred;
        Eigen::VectorXd u_eff = ured;

        const auto postsolve_primal = [&](const Eigen::VectorXd& v) {
            if (bypass_postsolve) {
                return std::make_pair(v, 0.0);
            }
            auto out = P.postsolve(v);
            out.first = repair_nan_primal_(A_model, b_model, l_model, u_model, std::move(out.first),
                                           opt_.tol);
            return out;
        };

        // ---- (4) Map incoming basis into reduced space (optional) ----
        std::optional<std::vector<int>> red_basis_opt = std::nullopt;
        std::optional<LPBasis> red_basis_state_opt = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::unordered_map<int, int> orig2red;
            orig2red.reserve(n_eff);
            for (int jr = 0; jr < n_eff; ++jr) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    orig2red[jorig] = jr;
            }
            std::vector<int> cand;
            cand.reserve(std::min(m_eff, (int)basis_opt->size()));
            std::vector<char> seen_red(n_eff, 0);
            for (int jorig : *basis_opt) {
                auto it = orig2red.find(jorig);
                if (it != orig2red.end() && !seen_red[it->second]) {
                    seen_red[it->second] = 1;
                    cand.push_back(it->second);
                    if ((int)cand.size() == m_eff)
                        break;
                }
            }
            if (!cand.empty())
                red_basis_opt = std::move(cand);
        }
        if (basis_state_opt && !basis_state_opt->column_status.empty()) {
            red_basis_state_opt =
                map_reduced_basis_state_(*basis_state_opt, col_orig_map, l_eff, u_eff, opt_.tol);
            if (!red_basis_opt || red_basis_opt->empty()) {
                red_basis_opt = basis_columns_from_basis_state_(*red_basis_state_opt, m_eff);
            }
        }

        auto t1_presolve = std::chrono::steady_clock::now();
        current_timing_.presolve_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve - t0_presolve).count();

        // ---- (5) Try Phase II directly on reduced problem (Primal/Dual per
        // mode) ----
        std::vector<int> basis_guess;
        std::optional<std::vector<int>> crash_seed_basis_opt = red_basis_opt;
        const bool seed_basis_from_state = (!basis_opt || basis_opt->empty()) &&
                                           red_basis_state_opt &&
                                           !red_basis_state_opt->column_status.empty();
        if ((!crash_seed_basis_opt || crash_seed_basis_opt->empty()) && red_basis_state_opt &&
            !red_basis_state_opt->column_status.empty()) {
            auto partial_seed = basis_columns_from_basis_state_(*red_basis_state_opt, -1);
            if (partial_seed && !partial_seed->empty()) {
                crash_seed_basis_opt = std::move(partial_seed);
            }
        }
        const bool allow_direct_warm_start =
            !seed_basis_from_state ||
            (crash_seed_basis_opt && static_cast<int>(crash_seed_basis_opt->size()) == m_eff);
        auto t0_crash = std::chrono::steady_clock::now();
        CrashSelection basis_choice = choose_initial_basis_(
            Ared, bred, cred, opt_, crash_seed_basis_opt, allow_direct_warm_start);
        auto t1_crash = std::chrono::steady_clock::now();
        current_timing_.crash_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_crash - t0_crash).count();
        basis_guess = basis_choice.basis;
        const bool basis_guess_from_crash = (basis_choice.source == "crash");
        const bool basis_guess_from_warm_start =
            (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start");

        const auto add_info = [&](std::unordered_map<std::string, std::string> info) {
            info["presolve_actions"] = std::to_string(pres.stack.size());
            info["presolve_implied_bound_updates"] = std::to_string(pres.implied_bound_updates);
            info["original_m"] = std::to_string(A_in.rows());
            info["original_l"] = serialize_double_vec_(l_in);
            info["original_u"] = serialize_double_vec_(u_in);
            info["reduced_m"] = std::to_string(m_eff);
            info["reduced_n"] = std::to_string(n_eff);
            info["obj_shift"] = std::to_string(pres.obj_shift);
            if (sanitized_bounds.relaxed_upper > 0) {
                info["input_upper_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_upper);
            }
            if (sanitized_bounds.relaxed_lower > 0) {
                info["input_lower_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_lower);
            }
            if (!basis_choice.source.empty() && basis_choice.source != "none") {
                info["basis_start"] = basis_choice.source;
                info["basis_start_style"] = basis_choice.style;
                info["basis_start_attempt"] = std::to_string(basis_choice.attempt);
                info["basis_start_primal_feasible"] =
                    basis_choice.quality.primal_feasible ? "1" : "0";
                info["basis_start_dual_feasible"] = basis_choice.quality.dual_feasible ? "1" : "0";
                info["basis_start_primal_violation"] =
                    std::to_string(basis_choice.quality.primal_violation);
                info["basis_start_dual_violation"] =
                    std::to_string(basis_choice.quality.dual_violation);
            }
            return info;
        };

        const auto parse_serialized_vec =
            [](const std::unordered_map<std::string, std::string>& info, const char* key,
               int expected_dim) -> std::optional<Eigen::VectorXd> {
            auto it = info.find(key);
            if (it == info.end())
                return std::nullopt;
            std::vector<double> vals;
            std::stringstream ss(it->second);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty())
                    vals.push_back(std::stod(tok));
            }
            if (expected_dim >= 0 && (int)vals.size() != expected_dim) {
                return std::nullopt;
            }
            if (vals.empty() && expected_dim == 0)
                return Eigen::VectorXd::Zero(0);
            if (vals.empty())
                return std::nullopt;
            return Eigen::Map<const Eigen::VectorXd>(vals.data(), static_cast<int>(vals.size()));
        };

        const bool basis_valid = ((int)basis_guess.size() == m_eff) && basis_choice.quality.valid;
        const bool allow_direct_primal =
            basis_valid && (basis_choice.quality.primal_feasible || basis_guess_from_warm_start);
        const bool allow_direct_dual = basis_valid && basis_choice.quality.dual_feasible;
        const bool allow_direct_from_guess = allow_direct_primal || allow_direct_dual;

        if (allow_direct_from_guess) {
            if (basis_guess_from_warm_start) {
                solve_stats_.warm_start_accepted = 1;
            }
            LPSolution::Status st;
            Eigen::VectorXd v2;
            std::vector<int> red_basis2;
            int it2;
            std::unordered_map<std::string, std::string> info2;

            auto run_primal = [&] {
                auto t0_iter = std::chrono::steady_clock::now();
                try {
                    auto res = phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
                    auto t1_iter = std::chrono::steady_clock::now();
                    current_timing_.simplex_iters_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter)
                            .count();
                    return res;
                } catch (const std::runtime_error& e) {
                    auto t1_iter = std::chrono::steady_clock::now();
                    current_timing_.simplex_iters_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter)
                            .count();
                    if (!is_recoverable_basis_runtime_(e.what()))
                        throw;
                    return PhaseResult{LPSolution::Status::Singular,
                                       Eigen::VectorXd{},
                                       {},
                                       0,
                                       {{"reason", "basis_factorization_failure"},
                                        {"what", e.what()},
                                        {"where", "dense_direct_primal"}}};
                }
            };
            auto run_dual = [&] {
                const bool use_warm_status =
                    red_basis_state_opt &&
                    (basis_choice.source == "warm_start" ||
                     basis_choice.source == "repaired_warm_start") &&
                    red_basis_state_opt->column_status.size() == static_cast<std::size_t>(n_eff);
                auto t0_iter = std::chrono::steady_clock::now();
                try {
                    auto res =
                        dual_phase_(Ared, bred, cred, basis_guess, l_eff, u_eff,
                                    use_warm_status ? std::optional<std::vector<LPBasisStatus>>(
                                                          red_basis_state_opt->column_status)
                                                    : std::nullopt);
                    auto t1_iter = std::chrono::steady_clock::now();
                    current_timing_.simplex_iters_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter)
                            .count();
                    return res;
                } catch (const std::runtime_error& e) {
                    auto t1_iter = std::chrono::steady_clock::now();
                    current_timing_.simplex_iters_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter)
                            .count();
                    if (!is_recoverable_basis_runtime_(e.what()))
                        throw;
                    return PhaseResult{LPSolution::Status::Singular,
                                       Eigen::VectorXd{},
                                       {},
                                       0,
                                       {{"reason", "basis_factorization_failure"},
                                        {"what", e.what()},
                                        {"where", "dense_direct_dual"}}};
                }
            };

            if (opt_.mode == SimplexMode::Dual) {
                if (allow_direct_dual) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                } else {
                    st = LPSolution::Status::NeedPhase1;
                    info2["reason"] = "no_dual_feasible_start_basis";
                }
            } else if (opt_.mode == SimplexMode::Primal) {
                if (allow_direct_primal) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                } else {
                    st = LPSolution::Status::NeedPhase1;
                    info2["reason"] = "no_primal_feasible_start_basis";
                }
            } else {
                if (allow_direct_primal) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                } else if (allow_direct_dual) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
                if (allow_direct_dual && st == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
            }

            if (st == LPSolution::Status::Optimal || st == LPSolution::Status::Unbounded ||
                st == LPSolution::Status::IterLimit || st == LPSolution::Status::ObjectiveBound ||
                (st == LPSolution::Status::Infeasible && !basis_guess_from_warm_start)) {
                auto [z_full, obj_corr] = postsolve_primal(v2);
                Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
                const double total_obj = c_in.dot(x_full);
                const bool has_primal_ray =
                    info2.count("primal_ray_has_cert") && info2.at("primal_ray_has_cert") == "1";
                const auto primal_ray_internal =
                    has_primal_ray ? parse_serialized_vec(info2, "primal_ray", n_eff)
                                   : std::nullopt;

                std::vector<int> basis_full;
                basis_full.reserve(red_basis2.size());
                for (int jr : red_basis2) {
                    if (jr >= 0 && jr < (int)col_orig_map.size()) {
                        const int jorig = col_orig_map[jr];
                        if (jorig >= 0)
                            basis_full.push_back(jorig);
                    }
                }
                auto info = add_info(std::move(info2));
                if (st == LPSolution::Status::Optimal &&
                    !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
                    info["reason"] = "invalid_returned_primal";
                    return finalize_solution_(attach_internal_basis_(
                        make_solution_(LPSolution::Status::Singular, std::move(x_full), total_obj,
                                       basis_full, it2, std::move(info)),
                        red_basis2, internal_column_labels));
                }
                return finalize_solution_(attach_basis_state_(
                    attach_mapped_primal_ray_(
                        attach_postsolved_farkas_(
                            attach_postsolved_row_duals_(
                                attach_internal_tableau_(
                                    make_solution_(st, std::move(x_full), total_obj, basis_full,
                                                   it2, std::move(info), std::nullopt, std::nullopt,
                                                   primal_ray_internal, has_primal_ray),
                                    Ared, bred, cred, red_basis2, internal_column_labels,
                                    internal_row_labels, opt_.tol, opt_.compute_tableau,
                                    opt_.compute_reduced_costs),
                                P, opt_.tol),
                            P, opt_.tol),
                        col_orig_map, sign, A_model.cols(), opt_.tol),
                    l_in, u_in, opt_.tol));
            }
            if (st == LPSolution::Status::Singular) {
                auto info = add_info({});
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
                    std::numeric_limits<double>::quiet_NaN(), {}, 0, std::move(info)));
            }
        }

        auto t1_presolve2 = std::chrono::steady_clock::now();
        current_timing_.presolve_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve2 - t1_crash).count();
        // ---- (6) Phase I on reduced problem ----
        auto [A1, b1, c1, basis1, n_orig_eff, m_rows] = make_phase1_(Ared, bred);
        auto t0_p1 = std::chrono::steady_clock::now();
        auto [status1, v1, basis1_out, it1, info1] =
            phase_(A1, b1, c1, basis1, Eigen::VectorXd::Zero(A1.cols()),
                   Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));
        auto t1_p1 = std::chrono::steady_clock::now();
        current_timing_.simplex_iters_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_p1 - t0_p1).count();

        if (status1 == LPSolution::Status::NeedPhase1 && info1.count("reason") &&
            info1.at("reason") == std::string("negative_basic_vars")) {
            auto t0_d1 = std::chrono::steady_clock::now();
            std::tie(status1, v1, basis1_out, it1, info1) =
                dual_phase_(A1, b1, c1, basis1_out.empty() ? basis1 : basis1_out,
                            Eigen::VectorXd::Zero(A1.cols()),
                            Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));
            auto t1_d1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1_d1 - t0_d1).count();
        }

        // If phase I fails or artificial cost > tol ⇒ infeasible
        if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
            auto info = add_info({{"phase1_status", to_string(status1)}});
            const auto s = degen_.get_stats();
            auto more = dm_stats_to_map(s);
            info.insert(more.begin(), more.end());
            return finalize_solution_(
                make_solution_(LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                               std::numeric_limits<double>::infinity(), {}, it1, std::move(info)));
        }

        // Warm-start Phase II basis by removing artificials
        std::vector<int> red_basis2;
        red_basis2.reserve(m_rows);
        for (int j : basis1_out)
            if (j < (int)n_orig_eff)
                red_basis2.push_back(j);

        // Basis completion if needed
        if ((int)red_basis2.size() < m_rows) {
            std::vector<int> fallback_basis = red_basis2;
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if ((int)red_basis2.size() == m_rows)
                    break;
                if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end())
                    continue;
                std::vector<int> cand = red_basis2;
                cand.push_back(j);
                if ((int)cand.size() > m_rows)
                    continue;
                const Eigen::MatrixXd Btest =
                    Ared(Eigen::all, Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                if (!(lu.rank() == (int)cand.size() && lu.isInvertible())) {
                    continue;
                }
                if (basis_is_primal_feasible_(Ared, bred, cand, opt_.tol)) {
                    red_basis2 = std::move(cand);
                    continue;
                }
                if ((int)fallback_basis.size() < (int)cand.size()) {
                    fallback_basis = cand;
                }
            }
            if ((int)red_basis2.size() < m_rows && (int)fallback_basis.size() == m_rows) {
                red_basis2 = std::move(fallback_basis);
            }
        }
        if ((int)red_basis2.size() == m_rows &&
            !basis_is_primal_feasible_(Ared, bred, red_basis2, opt_.tol)) {
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end()) {
                    continue;
                }
                bool improved = false;
                for (int r = 0; r < m_rows; ++r) {
                    std::vector<int> cand = red_basis2;
                    cand[r] = j;
                    const Eigen::MatrixXd Btest =
                        Ared(Eigen::all, Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                    Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                    if (!(lu.rank() == m_rows && lu.isInvertible()))
                        continue;
                    if (!basis_is_primal_feasible_(Ared, bred, cand, opt_.tol)) {
                        continue;
                    }
                    red_basis2 = std::move(cand);
                    improved = true;
                    break;
                }
                if (improved)
                    break;
            }
        }

        // Final Phase II on reduced problem (respect mode)
        LPSolution::Status status2;
        Eigen::VectorXd v2;
        std::vector<int> red_basis_out;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if ((int)red_basis2.size() == m_rows) {
            if (opt_.mode == SimplexMode::Dual) {
                const auto phase2_basis_quality =
                    evaluate_basis_quality_(Ared, bred, cred, red_basis2, opt_.tol);
                if (phase2_basis_quality.valid && phase2_basis_quality.dual_feasible) {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                    if (status2 == LPSolution::Status::Infeasible) {
                        auto it = info2.find("farkas_has_cert");
                        if (it != info2.end() && it->second == "1") {
                            auto yF = parse_serialized_vec(info2, "farkas_y", m_eff);
                            return finalize_solution_(attach_postsolved_farkas_(
                                attach_internal_basis_(
                                    make_solution_(LPSolution::Status::Infeasible,
                                                   Eigen::VectorXd::Zero(n),
                                                   std::numeric_limits<double>::infinity(), {}, it2,
                                                   add_info(std::move(info2)), yF, true),
                                    red_basis_out, internal_column_labels),
                                P, opt_.tol));
                        }
                    }
                } else {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                    info2["phase2_mode"] = "primal";
                    info2["phase2_dual_requested_but_basis_not_dual_feasible"] = "1";
                }

            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
            } else {
                // Auto: primal first; if negative basics → dual
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::NeedPhase1 && info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                }
            }
        } else {
            // Fall back to find a basis internally
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, std::nullopt, l_eff, u_eff);
            if (status2 == LPSolution::Status::NeedPhase1) {
                status2 = LPSolution::Status::Singular;
                info2["note"] = "reduced matrix cannot form a proper basis";
            }
        }

        const int total_iters = it1 + it2;
        auto merged_info = add_info(std::move(info2));
        merged_info.insert({"phase1_iters", std::to_string(it1)});
        const bool has_primal_ray = merged_info.count("primal_ray_has_cert") &&
                                    merged_info.at("primal_ray_has_cert") == "1";
        const auto primal_ray_internal =
            has_primal_ray ? parse_serialized_vec(merged_info, "primal_ray", n_eff) : std::nullopt;
        if (v2.size() != n_eff) {
            merged_info["where"] = "dense_phase2_finalize";
            merged_info["reason"] = "invalid_primal_dimension";
            merged_info["expected_primal_dim"] = std::to_string(n_eff);
            merged_info["actual_primal_dim"] = std::to_string(v2.size());
            return finalize_solution_(make_solution_(
                status2, Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN()),
                std::numeric_limits<double>::quiet_NaN(), {}, total_iters, std::move(merged_info)));
        }

        auto [z_full, obj_correction] = postsolve_primal(v2);
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
        const double total_obj = c_in.dot(x_full);

        std::vector<int> basis_full;
        basis_full.reserve(red_basis_out.size());
        for (int jr : red_basis_out) {
            if (jr >= 0 && jr < (int)col_orig_map.size()) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0)
                    basis_full.push_back(jorig);
            }
        }

        if (status2 == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
            merged_info["reason"] = "invalid_returned_primal";
            return finalize_solution_(attach_basis_state_(
                attach_mapped_primal_ray_(
                    attach_postsolved_farkas_(
                        attach_postsolved_row_duals_(
                            attach_internal_tableau_(
                                make_solution_(LPSolution::Status::Singular, x_full, total_obj,
                                               basis_full, total_iters, std::move(merged_info),
                                               std::nullopt, std::nullopt, primal_ray_internal,
                                               has_primal_ray),
                                Ared, bred, cred, red_basis_out, internal_column_labels,
                                internal_row_labels, opt_.tol, opt_.compute_tableau,
                                opt_.compute_reduced_costs),
                            P, opt_.tol),
                        P, opt_.tol),
                    col_orig_map, sign, A_model.cols(), opt_.tol),
                l_in, u_in, opt_.tol));
        }

        if (status2 == LPSolution::Status::Optimal) {
            return finalize_solution_(attach_basis_state_(
                attach_mapped_primal_ray_(
                    attach_postsolved_farkas_(
                        attach_postsolved_row_duals_(
                            attach_internal_tableau_(
                                make_solution_(LPSolution::Status::Optimal, x_full, total_obj,
                                               basis_full, total_iters, std::move(merged_info),
                                               std::nullopt, std::nullopt, primal_ray_internal,
                                               has_primal_ray),
                                Ared, bred, cred, red_basis_out, internal_column_labels,
                                internal_row_labels, opt_.tol, opt_.compute_tableau,
                                opt_.compute_reduced_costs),
                            P, opt_.tol),
                        P, opt_.tol),
                    col_orig_map, sign, A_model.cols(), opt_.tol),
                l_in, u_in, opt_.tol));
        }
        if (status2 == LPSolution::Status::Unbounded) {
            return finalize_solution_(attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(LPSolution::Status::Unbounded, x_full,
                                           -std::numeric_limits<double>::infinity(), basis_full,
                                           total_iters, std::move(merged_info), std::nullopt,
                                           std::nullopt, primal_ray_internal, has_primal_ray),
                            Ared, bred, cred, red_basis_out, internal_column_labels,
                            internal_row_labels, opt_.tol, opt_.compute_tableau,
                            opt_.compute_reduced_costs),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol));
        }

        const double obj_fallback =
            x_full.array().isFinite().all() ? total_obj : std::numeric_limits<double>::quiet_NaN();
        return finalize_solution_(attach_basis_state_(
            attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(status2, x_full, obj_fallback, basis_full, total_iters,
                                           std::move(merged_info), std::nullopt, std::nullopt,
                                           primal_ray_internal, has_primal_ray),
                            Ared, bred, cred, red_basis_out, internal_column_labels,
                            internal_row_labels, opt_.tol, opt_.compute_tableau,
                            opt_.compute_reduced_costs),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol),
            l_in, u_in, opt_.tol));
    }

    LPSolution solve_impl_sparse_(const SparseMatrix& A_in, const Eigen::VectorXd& b_in,
                                  const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                                  const Eigen::VectorXd& u_in,
                                  std::optional<std::vector<int>> basis_opt,
                                  const LPBasis* basis_state_opt);

  private:
    friend class RevisedSimplexPrimalEngine;
    friend class RevisedSimplexDualEngine;

    // =========================================================================
    // Helpers (private; signatures preserved where externally referenced)
    // =========================================================================

    static Eigen::VectorXd clip_small_(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i)
            if (std::abs(x(i)) < tol)
                x(i) = 0.0;
        return x;
    }

    void trace_line_(const std::string& line) const {
        if (!opt_.verbose)
            return;
        trace_.push_back(line);
        std::cout << line << std::endl;
    }

    bool should_trace_iter_(int iter) const {
        if (!opt_.verbose)
            return false;
        const int freq = std::max(1, opt_.verbose_every);
        return iter <= 1 || (iter % freq) == 0;
    }

    static std::string format_basis_(const std::vector<int>& basis) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < basis.size(); ++i) {
            if (i)
                oss << ", ";
            oss << basis[i];
        }
        oss << "]";
        return oss.str();
    }

    static std::string format_status_(LPSolution::Status status) {
        return std::string(to_string(status));
    }

    static Eigen::MatrixXd dense_copy_(const Eigen::MatrixXd& A) { return A; }
    static Eigen::MatrixXd dense_copy_(const SparseMatrix& A) { return Eigen::MatrixXd(A); }

    static Eigen::MatrixXd dense_basis_copy_(const SparseMatrix& A, const std::vector<int>& basis);

    static SparseMatrix sparse_basis_copy_(const SparseMatrix& A, const std::vector<int>& basis);

    static bool sparse_basis_has_full_rank_(const SparseMatrix& A, const std::vector<int>& basis);

    static Eigen::VectorXd sparse_solveT_from_lu_(const SparseMatrix& B,
                                                  const Eigen::SparseLU<SparseMatrix>& lu_B,
                                                  const Eigen::VectorXd& c);

    static std::vector<std::string>
    make_internal_column_labels_(const std::vector<int>& col_orig_map) {
        std::vector<std::string> labels;
        labels.reserve(col_orig_map.size());
        for (int jr = 0; jr < (int)col_orig_map.size(); ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0) {
                labels.push_back("x_orig_" + std::to_string(jorig));
            } else {
                labels.push_back("internal_" + std::to_string(jr));
            }
        }
        return labels;
    }

    static std::optional<Eigen::VectorXd>
    parse_serialized_vec_(const std::unordered_map<std::string, std::string>& info, const char* key,
                          int expected_dim) {
        auto it = info.find(key);
        if (it == info.end())
            return std::nullopt;
        std::vector<double> vals;
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty())
                vals.push_back(std::stod(tok));
        }
        if (expected_dim >= 0 && (int)vals.size() != expected_dim) {
            return std::nullopt;
        }
        if (vals.empty() && expected_dim == 0)
            return Eigen::VectorXd::Zero(0);
        if (vals.empty())
            return std::nullopt;
        return Eigen::Map<const Eigen::VectorXd>(vals.data(), static_cast<int>(vals.size()));
    }

    static std::vector<std::string>
    make_internal_row_labels_(const std::vector<int>& row_orig_map) {
        std::vector<std::string> labels;
        labels.reserve(row_orig_map.size());
        for (int ir = 0; ir < (int)row_orig_map.size(); ++ir) {
            const int iorig = row_orig_map[ir];
            if (iorig >= 0) {
                labels.push_back("row_orig_" + std::to_string(iorig));
            } else {
                labels.push_back("internal_row_" + std::to_string(ir));
            }
        }
        return labels;
    }

    static std::vector<int> make_nonbasis_internal_(int n, const std::vector<int>& basis) {
        std::vector<int> nonbasis;
        if (n <= 0)
            return nonbasis;
        std::vector<char> in_basis(n, 0);
        for (int j : basis) {
            if (j >= 0 && j < n)
                in_basis[j] = 1;
        }
        nonbasis.reserve(std::max(0, n - (int)basis.size()));
        for (int j = 0; j < n; ++j) {
            if (!in_basis[j])
                nonbasis.push_back(j);
        }
        return nonbasis;
    }

    static LPBasisStatus default_basis_status_for_bounds_(int j, const Eigen::VectorXd& l,
                                                          const Eigen::VectorXd& u,
                                                          double tol = 1e-12) {
        const bool has_l = (j < l.size()) && std::isfinite(l(j));
        const bool has_u = (j < u.size()) && std::isfinite(u(j));
        if (has_l && has_u && std::abs(u(j) - l(j)) <= tol) {
            return LPBasisStatus::Fixed;
        }
        if (has_u && !has_l)
            return LPBasisStatus::AtUpper;
        return LPBasisStatus::AtLower;
    }

    static std::optional<std::vector<int>>
    basis_columns_from_basis_state_(const LPBasis& basis_state, int expected_rows) {
        if (!basis_state.basis_columns.empty()) {
            bool ordered_matches_status = true;
            for (int j : basis_state.basis_columns) {
                if (j < 0 || j >= static_cast<int>(basis_state.column_status.size()) ||
                    basis_state.column_status[j] != LPBasisStatus::Basic) {
                    ordered_matches_status = false;
                    break;
                }
            }
            if (ordered_matches_status &&
                (expected_rows < 0 ||
                 static_cast<int>(basis_state.basis_columns.size()) == expected_rows)) {
                return basis_state.basis_columns;
            }
        }
        std::vector<int> basis;
        basis.reserve(basis_state.column_status.size());
        for (int j = 0; j < (int)basis_state.column_status.size(); ++j) {
            if (basis_state.column_status[j] == LPBasisStatus::Basic) {
                basis.push_back(j);
            }
        }
        if (expected_rows >= 0 && (int)basis.size() != expected_rows) {
            return std::nullopt;
        }
        return basis;
    }

    static LPBasis compute_basis_state_(const std::vector<int>& basis, const Eigen::VectorXd& x,
                                        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                        double tol, int basic_target = -1) {
        LPBasis basis_state;
        if (x.size() <= 0)
            return basis_state;

        basis_state.column_status.resize(x.size(), LPBasisStatus::AtLower);
        basis_state.basis_columns = basis;
        std::vector<char> in_basis(x.size(), 0);
        for (int j : basis) {
            if (j >= 0 && j < x.size()) {
                in_basis[j] = 1;
            }
        }
        std::vector<char> eligible_basic(x.size(), 1);
        for (int j = 0; j < x.size(); ++j) {
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool fixed = has_l && has_u && std::abs(u(j) - l(j)) <= tol;
            if (fixed) {
                basis_state.column_status[j] = LPBasisStatus::Fixed;
                eligible_basic[j] = in_basis[j];
                continue;
            }

            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            if (near_u && !near_l) {
                basis_state.column_status[j] = LPBasisStatus::AtUpper;
            } else if (near_l) {
                basis_state.column_status[j] = LPBasisStatus::AtLower;
            } else if (has_u && !has_l) {
                basis_state.column_status[j] = LPBasisStatus::AtUpper;
            } else if (has_l && has_u) {
                const double dl = std::abs(x(j) - l(j));
                const double du = std::abs(x(j) - u(j));
                basis_state.column_status[j] =
                    (du + tol < dl) ? LPBasisStatus::AtUpper : LPBasisStatus::AtLower;
            } else {
                basis_state.column_status[j] = LPBasisStatus::AtLower;
            }
        }

        const int target = (basic_target >= 0) ? basic_target : static_cast<int>(basis.size());
        if (target <= 0)
            return basis_state;

        std::vector<char> chosen(x.size(), 0);
        auto choose_if = [&](int j) {
            if (j < 0 || j >= x.size() || chosen[j] || !eligible_basic[j])
                return false;
            chosen[j] = 1;
            basis_state.column_status[j] = LPBasisStatus::Basic;
            return true;
        };

        int chosen_count = 0;
        for (int j : basis) {
            if (chosen_count == target)
                break;
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            const bool interior = !near_l && !near_u;
            if (interior && choose_if(j))
                ++chosen_count;
        }
        for (int j = 0; j < x.size() && chosen_count < target; ++j) {
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            const bool interior = !near_l && !near_u;
            if (interior && choose_if(j))
                ++chosen_count;
        }
        for (int j : basis) {
            if (chosen_count == target)
                break;
            if (choose_if(j))
                ++chosen_count;
        }
        for (int j = 0; j < x.size() && chosen_count < target; ++j) {
            if (choose_if(j)) {
                ++chosen_count;
            }
        }
        return basis_state;
    }

    static bool basis_state_matches_problem_(const LPBasis& basis_state, int rows, int cols) {
        if ((int)basis_state.column_status.size() != cols)
            return false;
        int basic_count = 0;
        for (const auto status : basis_state.column_status) {
            if (status == LPBasisStatus::Basic)
                ++basic_count;
        }
        if (!basis_state.basis_columns.empty() &&
            static_cast<int>(basis_state.basis_columns.size()) != rows) {
            return false;
        }
        return basic_count == rows;
    }

    static LPBasis map_reformulated_basis_state_(const LPBasis& original_basis_state,
                                                 const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                                 int n_total, const std::vector<int>& single_y,
                                                 const std::vector<int>& upper_slack,
                                                 const std::vector<int>& split_pos,
                                                 const std::vector<int>& split_neg) {
        LPBasis mapped;
        mapped.column_status.resize(n_total, LPBasisStatus::AtLower);
        for (int j = 0; j < (int)original_basis_state.column_status.size(); ++j) {
            const LPBasisStatus status = original_basis_state.column_status[j];
            const bool has_l = j < l.size() && std::isfinite(l(j));
            const bool has_u = j < u.size() && std::isfinite(u(j));
            const bool fixed = has_l && has_u && std::abs(u(j) - l(j)) <= 1e-12;
            if (single_y[j] >= 0) {
                const int y = single_y[j];
                const int slack = upper_slack[j];
                if (slack >= 0) {
                    if (fixed) {
                        // Tight bound changes turn the original column into a
                        // zero-width variable. Keeping its transformed column
                        // basic together with the upper-row slack often yields
                        // a singular warm basis after branching. Anchor the
                        // variable at its bound and let the upper-row slack
                        // carry the basis instead.
                        mapped.column_status[y] = LPBasisStatus::AtLower;
                        mapped.column_status[slack] = LPBasisStatus::Basic;
                    } else if (status == LPBasisStatus::Basic) {
                        // Original variable is basic: y covers an original row, but
                        // the upper-bound row (y + s = u-l) still needs a basic
                        // variable. Since y is basic at some interior value, s must
                        // also be basic (at u-l-y_val > 0). Both rows are covered.
                        mapped.column_status[y] = LPBasisStatus::Basic;
                        mapped.column_status[slack] = LPBasisStatus::Basic;
                    } else if (status == LPBasisStatus::AtUpper) {
                        // In the reformulated model the single variable y is
                        // nonnegative with no finite upper bound, and the
                        // finite upper bound is represented by y + s = u - l.
                        // Mapping an original AtUpper status therefore means
                        // keeping the upper slack at zero and making y basic
                        // at the transformed RHS value.
                        mapped.column_status[y] = LPBasisStatus::Basic;
                        mapped.column_status[slack] = LPBasisStatus::AtLower;
                    } else {
                        mapped.column_status[y] = LPBasisStatus::AtLower;
                        mapped.column_status[slack] = LPBasisStatus::Basic;
                    }
                } else {
                    mapped.column_status[y] = (status == LPBasisStatus::Basic)
                                                  ? LPBasisStatus::Basic
                                                  : LPBasisStatus::AtLower;
                }
                continue;
            }

            if (split_pos[j] >= 0) {
                mapped.column_status[split_pos[j]] = (status == LPBasisStatus::Basic)
                                                         ? LPBasisStatus::Basic
                                                         : LPBasisStatus::AtLower;
            }
            if (split_neg[j] >= 0 && split_neg[j] < n_total) {
                mapped.column_status[split_neg[j]] = LPBasisStatus::AtLower;
            }
        }
        if (!original_basis_state.basis_columns.empty()) {
            for (int j : original_basis_state.basis_columns) {
                if (j < 0 || j >= static_cast<int>(single_y.size())) {
                    continue;
                }
                if (single_y[j] >= 0 && mapped.column_status[single_y[j]] == LPBasisStatus::Basic) {
                    mapped.basis_columns.push_back(single_y[j]);
                } else if (split_pos[j] >= 0 &&
                           mapped.column_status[split_pos[j]] == LPBasisStatus::Basic) {
                    mapped.basis_columns.push_back(split_pos[j]);
                }
            }
            for (int slack : upper_slack) {
                if (slack >= 0 && slack < n_total &&
                    mapped.column_status[slack] == LPBasisStatus::Basic &&
                    std::find(mapped.basis_columns.begin(), mapped.basis_columns.end(), slack) ==
                        mapped.basis_columns.end()) {
                    mapped.basis_columns.push_back(slack);
                }
            }
        }
        return mapped;
    }

    static LPBasis map_reformulated_basis_seed_state_(const LPBasis& original_basis_state,
                                                      int n_total, const std::vector<int>& single_y,
                                                      const std::vector<int>& upper_slack,
                                                      const std::vector<int>& split_pos,
                                                      const std::vector<int>& split_neg) {
        LPBasis mapped;
        mapped.column_status.resize(n_total, LPBasisStatus::AtLower);
        for (int j = 0; j < (int)original_basis_state.column_status.size(); ++j) {
            const LPBasisStatus status = original_basis_state.column_status[j];
            if (single_y[j] >= 0) {
                const int y = single_y[j];
                if (status == LPBasisStatus::Basic || status == LPBasisStatus::AtUpper) {
                    mapped.column_status[y] = LPBasisStatus::Basic;
                    // Upper-bound row (y + s = u-l) needs a basic variable too.
                    // Mark the slack Basic so the mapped basis covers all rows.
                    if (j < (int)upper_slack.size() && upper_slack[j] >= 0)
                        mapped.column_status[upper_slack[j]] = LPBasisStatus::Basic;
                }
                continue;
            }

            if (split_pos[j] >= 0) {
                mapped.column_status[split_pos[j]] = (status == LPBasisStatus::Basic)
                                                         ? LPBasisStatus::Basic
                                                         : LPBasisStatus::AtLower;
            }
            if (split_neg[j] >= 0 && split_neg[j] < n_total) {
                mapped.column_status[split_neg[j]] = LPBasisStatus::AtLower;
            }
        }
        if (!original_basis_state.basis_columns.empty()) {
            for (int j : original_basis_state.basis_columns) {
                if (j < 0 || j >= static_cast<int>(single_y.size())) {
                    continue;
                }
                if (single_y[j] >= 0 && mapped.column_status[single_y[j]] == LPBasisStatus::Basic) {
                    mapped.basis_columns.push_back(single_y[j]);
                } else if (split_pos[j] >= 0 &&
                           mapped.column_status[split_pos[j]] == LPBasisStatus::Basic) {
                    mapped.basis_columns.push_back(split_pos[j]);
                }
            }
        }
        return mapped;
    }

    static LPBasis map_reduced_basis_state_(const LPBasis& original_basis_state,
                                            const std::vector<int>& col_orig_map,
                                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                            double tol) {
        LPBasis mapped;
        mapped.column_status.resize(col_orig_map.size(), LPBasisStatus::AtLower);
        for (int jr = 0; jr < (int)col_orig_map.size(); ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0 && jorig < (int)original_basis_state.column_status.size()) {
                mapped.column_status[jr] = original_basis_state.column_status[jorig];
            } else {
                mapped.column_status[jr] = default_basis_status_for_bounds_(jr, l, u, tol);
            }
        }
        if (!original_basis_state.basis_columns.empty()) {
            for (int jorig : original_basis_state.basis_columns) {
                for (int jr = 0; jr < static_cast<int>(col_orig_map.size()); ++jr) {
                    if (col_orig_map[jr] == jorig &&
                        mapped.column_status[jr] == LPBasisStatus::Basic) {
                        mapped.basis_columns.push_back(jr);
                        break;
                    }
                }
            }
        }
        return mapped;
    }

    static LPBasis rebase_basis_state_for_bounds_(const LPBasis& basis_state,
                                                  const Eigen::VectorXd& l,
                                                  const Eigen::VectorXd& u, double tol) {
        LPBasis rebased = basis_state;
        for (int j = 0; j < static_cast<int>(rebased.column_status.size()); ++j) {
            const bool has_l = j < l.size() && std::isfinite(l(j));
            const bool has_u = j < u.size() && std::isfinite(u(j));
            const bool fixed = has_l && has_u && std::abs(u(j) - l(j)) <= tol;
            auto& status = rebased.column_status[j];

            if (status == LPBasisStatus::Basic) {
                continue;
            }

            if (fixed) {
                status = LPBasisStatus::Fixed;
                continue;
            }

            switch (status) {
                case LPBasisStatus::AtUpper:
                    if (!has_u) {
                        status = default_basis_status_for_bounds_(j, l, u, tol);
                    }
                    break;
                case LPBasisStatus::Fixed:
                    status = default_basis_status_for_bounds_(j, l, u, tol);
                    break;
                case LPBasisStatus::AtLower:
                    if (!has_l && has_u) {
                        status = LPBasisStatus::AtUpper;
                    }
                    break;
                case LPBasisStatus::Basic:
                default:
                    break;
            }
        }
        return rebased;
    }

    static std::string serialize_double_vec_(const Eigen::VectorXd& v) {
        std::ostringstream oss;
        oss.setf(std::ios::scientific);
        oss << std::setprecision(17);
        for (int i = 0; i < v.size(); ++i) {
            if (i)
                oss << ",";
            oss << v(i);
        }
        return oss.str();
    }

    static std::string serialize_basis_state_from_primal_(const std::vector<int>& basis,
                                                          const Eigen::VectorXd& x,
                                                          const Eigen::VectorXd& l,
                                                          const Eigen::VectorXd& u, double tol,
                                                          int basic_target = -1) {
        if (x.size() <= 0)
            return "";

        std::vector<int> status(x.size(), 1);
        std::vector<char> eligible_basic(x.size(), 1);
        for (int j = 0; j < x.size(); ++j) {
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool fixed = has_l && has_u && std::abs(u(j) - l(j)) <= tol;
            if (fixed) {
                status[j] = 3;
                eligible_basic[j] = 0;
                continue;
            }

            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            if (near_u && !near_l) {
                status[j] = 2;
            } else if (near_l) {
                status[j] = 1;
            } else if (has_u && !has_l) {
                status[j] = 2;
            } else if (has_l && has_u) {
                const double dl = std::abs(x(j) - l(j));
                const double du = std::abs(x(j) - u(j));
                status[j] = (du + tol < dl) ? 2 : 1;
            } else {
                status[j] = 1;
            }
        }

        const int target = (basic_target >= 0) ? basic_target : static_cast<int>(basis.size());
        std::vector<char> chosen(x.size(), 0);
        auto choose_if = [&](int j) {
            if (j < 0 || j >= x.size() || chosen[j] || !eligible_basic[j])
                return false;
            chosen[j] = 1;
            status[j] = 0;
            return true;
        };

        int chosen_count = 0;
        for (int j : basis) {
            if (chosen_count == target)
                break;
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            if (!near_l && !near_u && choose_if(j))
                ++chosen_count;
        }
        for (int j = 0; j < x.size() && chosen_count < target; ++j) {
            const bool has_l = (j < l.size()) && std::isfinite(l(j));
            const bool has_u = (j < u.size()) && std::isfinite(u(j));
            const bool near_l = has_l && std::abs(x(j) - l(j)) <= 10.0 * tol;
            const bool near_u = has_u && std::abs(x(j) - u(j)) <= 10.0 * tol;
            if (!near_l && !near_u && choose_if(j))
                ++chosen_count;
        }
        for (int j : basis) {
            if (chosen_count == target)
                break;
            if (choose_if(j))
                ++chosen_count;
        }
        for (int j = 0; j < x.size() && chosen_count < target; ++j) {
            if (choose_if(j))
                ++chosen_count;
        }

        std::ostringstream oss;
        for (int j = 0; j < x.size(); ++j) {
            if (j)
                oss << ",";
            oss << status[j];
        }
        return oss.str();
    }

    static Eigen::VectorXd clip_small_vec_(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i) {
            if (std::abs(x(i)) < tol)
                x(i) = 0.0;
        }
        return x;
    }

    static Eigen::MatrixXd clip_small_mat_(Eigen::MatrixXd X, double tol = 1e-12) {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (std::abs(X(i, j)) < tol)
                    X(i, j) = 0.0;
            }
        }
        return X;
    }

    static std::string describe_presolve_action_(const presolve::Action& action) {
        return std::visit(
            [](const auto& act) -> std::string {
                using T = std::decay_t<decltype(act)>;
                std::ostringstream oss;
                if constexpr (std::is_same_v<T, presolve::ActRowReduce>) {
                    oss << "row_reduce old_m=" << act.old_m << " keep=" << act.keep.size();
                } else if constexpr (std::is_same_v<T, presolve::ActRemoveRow>) {
                    oss << "remove_row i=" << act.i << " rhs=" << act.rhs;
                } else if constexpr (std::is_same_v<T, presolve::ActRemoveCol>) {
                    oss << "remove_col j=" << act.j << " c=" << act.c_j;
                } else if constexpr (std::is_same_v<T, presolve::ActFixVar>) {
                    oss << "fix_var j=" << act.j << " x=" << act.x_fix;
                } else if constexpr (std::is_same_v<T, presolve::ActTightenBound>) {
                    oss << "tighten_bound j=" << act.j << " old_l=" << act.old_l
                        << " old_u=" << act.old_u;
                } else if constexpr (std::is_same_v<T, presolve::ActScaleRow>) {
                    oss << "scale_row i=" << act.i << " scale=" << act.scale;
                } else if constexpr (std::is_same_v<T, presolve::ActScaleCol>) {
                    oss << "scale_col j=" << act.j << " scale=" << act.scale;
                } else if constexpr (std::is_same_v<T, presolve::ActSingletonRowElim>) {
                    oss << "singleton_row_elim i=" << act.i << " j=" << act.j << " rhs=" << act.rhs;
                } else if constexpr (std::is_same_v<T, presolve::ActSingletonColElim>) {
                    oss << "singleton_col_elim j=" << act.j << " i=" << act.i << " aij=" << act.aij;
                } else if constexpr (std::is_same_v<T, presolve::ActDualFix>) {
                    oss << "dual_fix j=" << act.j << " x=" << act.x_fix;
                }
                return oss.str();
            },
            action);
    }

    void trace_presolve_(const presolve::PresolveResult& pres) const {
        if (!opt_.verbose || !opt_.verbose_include_presolve)
            return;
        trace_line_("[presolve] actions=" + std::to_string(pres.stack.size()) +
                    " reduced_m=" + std::to_string(pres.reduced.A.rows()) +
                    " reduced_n=" + std::to_string(pres.reduced.A.cols()) +
                    " infeasible=" + std::string(pres.proven_infeasible ? "1" : "0") +
                    " unbounded=" + std::string(pres.proven_unbounded ? "1" : "0"));
        for (std::size_t i = 0; i < pres.stack.size(); ++i) {
            trace_line_("[presolve] #" + std::to_string(i + 1) + " " +
                        describe_presolve_action_(pres.stack[i]));
        }
    }

    LPSolution finalize_solution_(LPSolution sol) {
        sol.timing.presolve_ns += current_timing_.presolve_ns;
        sol.timing.crash_ns += current_timing_.crash_ns;
        sol.timing.simplex_iters_ns += current_timing_.simplex_iters_ns;
        if (solve_output_warm_state_ && !sol.basis_state.column_status.empty()) {
            sol.basis_state.warm_state = solve_output_warm_state_;
        }
        sol.solve_stats = solve_stats_;
        if (opt_.verbose)
            sol.trace = trace_;
        return sol;
    }

    static std::uint64_t mix_signature_(std::uint64_t seed, std::uint64_t value) noexcept {
        seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }

    static std::uint64_t hash_double_(double value) noexcept {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if (bits == 0x8000000000000000ULL) {
            bits = 0;
        }
        return bits;
    }

    static std::uint64_t matrix_signature_(const Eigen::MatrixXd& A) noexcept {
        std::uint64_t sig = 0xcbf29ce484222325ULL;
        sig = mix_signature_(sig, static_cast<std::uint64_t>(A.rows()));
        sig = mix_signature_(sig, static_cast<std::uint64_t>(A.cols()));
        for (Eigen::Index j = 0; j < A.cols(); ++j) {
            for (Eigen::Index i = 0; i < A.rows(); ++i) {
                sig = mix_signature_(sig, static_cast<std::uint64_t>(i));
                sig = mix_signature_(sig, static_cast<std::uint64_t>(j));
                sig = mix_signature_(sig, hash_double_(A(i, j)));
            }
        }
        return sig;
    }

    static std::uint64_t matrix_signature_(const SparseMatrix& A) noexcept {
        if (!A.isCompressed()) {
            SparseMatrix compressed = A;
            compressed.makeCompressed();
            return matrix_signature_(compressed);
        }
        std::uint64_t sig = 0xcbf29ce484222325ULL;
        sig = mix_signature_(sig, static_cast<std::uint64_t>(A.rows()));
        sig = mix_signature_(sig, static_cast<std::uint64_t>(A.cols()));
        sig = mix_signature_(sig, static_cast<std::uint64_t>(A.nonZeros()));
        const int* outer = A.outerIndexPtr();
        const int* inner = A.innerIndexPtr();
        const double* value = A.valuePtr();
        for (int j = 0; j <= A.cols(); ++j) {
            sig = mix_signature_(sig, static_cast<std::uint64_t>(outer[j]));
        }
        for (int k = 0; k < A.nonZeros(); ++k) {
            sig = mix_signature_(sig, static_cast<std::uint64_t>(inner[k]));
            sig = mix_signature_(sig, hash_double_(value[k]));
        }
        return sig;
    }

    void begin_solve_(std::uint64_t matrix_signature, int rows, int cols, bool matrix_is_sparse,
                      const LPBasis* basis_state_opt) {
        current_timing_ = RevisedSimplexTiming{};
        solve_stats_ = LPSolveStats{};
        current_matrix_signature_ = matrix_signature;
        current_matrix_rows_ = rows;
        current_matrix_cols_ = cols;
        current_matrix_is_sparse_ = matrix_is_sparse;
        solve_input_warm_state_ =
            basis_state_opt ? basis_state_opt->warm_state : std::shared_ptr<LPWarmStateData>{};
        // The persistent reformulated solver used for BnB node LPs re-solves the
        // same matrix repeatedly without an explicit basis_state (it relies on
        // has_cached_basis_state and passes no basis). That internal cache holds
        // the FTBasis warm_state, but it was dropped here -- so factorization
        // reuse never fired. Recover it: when no warm state was passed in and the
        // cached basis is for the current matrix, adopt its factorization.
        if (!solve_input_warm_state_ && cached_basis_state_ && cached_basis_state_->warm_state &&
            cached_matrix_signature_ == current_matrix_signature_ && cached_basis_rows_ == rows &&
            cached_basis_cols_ == cols && cached_matrix_is_sparse_ == matrix_is_sparse) {
            solve_input_warm_state_ = cached_basis_state_->warm_state;
        }
        solve_output_warm_state_.reset();
        if (basis_state_opt && !basis_state_opt->column_status.empty()) {
            solve_stats_.warm_start_attempted = 1;
        }
    }

    std::shared_ptr<LPWarmStateData>
    try_reuse_factorization_(const std::vector<int>& basis_columns) const {
        if (!solve_input_warm_state_ || !solve_input_warm_state_->nla) {
            return nullptr;
        }
        if (solve_input_warm_state_->matrix_signature != current_matrix_signature_ ||
            solve_input_warm_state_->rows != current_matrix_rows_ ||
            solve_input_warm_state_->cols != current_matrix_cols_ ||
            solve_input_warm_state_->matrix_is_sparse != current_matrix_is_sparse_) {
            return nullptr;
        }
        if (solve_input_warm_state_->nla->factor().basis() != basis_columns) {
            return nullptr;
        }
        solve_stats_.warm_start_accepted = 1;
        solve_stats_.warm_factorization_reused = 1;
        solve_stats_.eta_stack_depth_entry =
            solve_input_warm_state_->nla->factor().stats().eta_count;
        return solve_input_warm_state_;
    }

    std::optional<std::vector<int>> warm_factorization_basis_seed_() const {
        if (opt_.mode != SimplexMode::Dual || !solve_input_warm_state_ ||
            !solve_input_warm_state_->nla) {
            return std::nullopt;
        }
        if (solve_input_warm_state_->matrix_signature != current_matrix_signature_ ||
            solve_input_warm_state_->rows != current_matrix_rows_ ||
            solve_input_warm_state_->cols != current_matrix_cols_ ||
            solve_input_warm_state_->matrix_is_sparse != current_matrix_is_sparse_) {
            return std::nullopt;
        }
        const std::vector<int>& factorized_basis =
            solve_input_warm_state_->nla->factor().basis();
        if (static_cast<int>(factorized_basis.size()) != current_matrix_rows_) {
            return std::nullopt;
        }
        for (int col : factorized_basis) {
            if (col < 0 || col >= current_matrix_cols_) {
                return std::nullopt;
            }
        }
        return factorized_basis;
    }

    void
    remember_warm_state_(const std::vector<int>& basis_columns,
                         const std::shared_ptr<simplex::nla::SimplexNLA>& nla,
                         std::optional<LPDualPricingWarmState> dual_pricing_state = std::nullopt) {
        if (!nla || basis_columns.empty()) {
            solve_output_warm_state_.reset();
            return;
        }
        auto warm_state = std::make_shared<LPWarmStateData>();
        warm_state->matrix_signature = current_matrix_signature_;
        warm_state->basis_matrix_signature = nla->factor().basis_matrix_signature();
        warm_state->rows = current_matrix_rows_;
        warm_state->cols = current_matrix_cols_;
        warm_state->matrix_is_sparse = current_matrix_is_sparse_;
        warm_state->basis_columns = basis_columns;
        warm_state->nla = nla;
        warm_state->dual_pricing_state = std::move(dual_pricing_state);
        solve_output_warm_state_ = std::move(warm_state);
        solve_stats_.ft_updates = nla->factor().stats().eta_count;
    }

    template <typename Fn> void measure_pricing_build_(bool dual_pool, Fn&& builder) {
        const auto t0 = std::chrono::steady_clock::now();
        builder();
        solve_stats_.pricing_build_ns +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - t0)
                                           .count());
        if (dual_pool) {
            ++solve_stats_.dual_pool_builds;
        } else {
            ++solve_stats_.primal_pool_builds;
        }
    }

    static LPSolution attach_postsolved_row_duals_(LPSolution sol, const presolve::Presolver& P,
                                                   double tol);

    static LPSolution attach_postsolved_farkas_(LPSolution sol, const presolve::Presolver& P,
                                                double tol);

    static LPSolution attach_mapped_primal_ray_(LPSolution sol,
                                                const std::vector<int>& col_orig_map,
                                                const Eigen::VectorXd& sign, int original_num_cols,
                                                double tol);

    static LPSolution attach_internal_basis_(LPSolution sol, std::vector<int> basis_internal,
                                             std::vector<std::string> internal_column_labels);

    static LPSolution attach_internal_tableau_(LPSolution sol, const Eigen::MatrixXd& A_internal,
                                               const Eigen::VectorXd& b_internal,
                                               const Eigen::VectorXd& c_internal,
                                               std::vector<int> basis_internal,
                                               std::vector<std::string> internal_column_labels,
                                               std::vector<std::string> internal_row_labels,
                                               double tol, bool compute_tableau,
                                               bool compute_reduced_costs);

    static LPSolution attach_internal_tableau_(LPSolution sol, const SparseMatrix& A_internal,
                                               const Eigen::VectorXd& b_internal,
                                               const Eigen::VectorXd& c_internal,
                                               std::vector<int> basis_internal,
                                               std::vector<std::string> internal_column_labels,
                                               std::vector<std::string> internal_row_labels,
                                               double tol, bool compute_tableau,
                                               bool compute_reduced_costs);

    static LPSolution attach_basis_state_(LPSolution sol, const Eigen::VectorXd& l,
                                          const Eigen::VectorXd& u, double tol,
                                          int basic_target = -1);

    struct SanitizedBounds {
        Eigen::VectorXd l;
        Eigen::VectorXd u;
        int relaxed_lower = 0;
        int relaxed_upper = 0;
    };

    SanitizedBounds canonicalize_inactive_huge_bounds_(const Eigen::MatrixXd& A,
                                                       const Eigen::VectorXd& b,
                                                       const Eigen::VectorXd& l,
                                                       const Eigen::VectorXd& u) const {
        presolve::LP problem;
        problem.A = A;
        problem.b = b;
        problem.l = l;
        problem.u = u;
        SanitizedBounds out{problem.l, problem.u, 0, 0};
        const presolve::BoundRelaxationSummary relaxed =
            presolve::canonicalize_inactive_huge_bounds(&problem, opt_.tol);
        out.l = std::move(problem.l);
        out.u = std::move(problem.u);
        out.relaxed_lower = relaxed.relaxed_lower;
        out.relaxed_upper = relaxed.relaxed_upper;
        return out;
    }

    static bool primal_feasible_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& x, const Eigen::VectorXd& l,
                                 const Eigen::VectorXd& u, double tol) {
        if (x.size() == 0 || !x.array().isFinite().all())
            return false;
        if (A.rows() > 0) {
            const Eigen::VectorXd resid = A * x - b;
            if (resid.size() > 0 && resid.lpNorm<Eigen::Infinity>() > 100.0 * tol) {
                return false;
            }
        }
        for (int j = 0; j < x.size(); ++j) {
            if (j < l.size() && std::isfinite(l(j)) && x(j) < l(j) - 100.0 * tol) {
                return false;
            }
            if (j < u.size() && std::isfinite(u(j)) && x(j) > u(j) + 100.0 * tol) {
                return false;
            }
        }
        return true;
    }

    static bool primal_feasible_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& x, const Eigen::VectorXd& l,
                                 const Eigen::VectorXd& u, double tol) {
        if (x.size() == 0 || !x.array().isFinite().all())
            return false;
        if (A.rows() > 0) {
            const Eigen::VectorXd resid = A * x - b;
            if (resid.size() > 0 && resid.lpNorm<Eigen::Infinity>() > 100.0 * tol) {
                return false;
            }
        }
        for (int j = 0; j < x.size(); ++j) {
            if (j < l.size() && std::isfinite(l(j)) && x(j) < l(j) - 100.0 * tol) {
                return false;
            }
            if (j < u.size() && std::isfinite(u(j)) && x(j) > u(j) + 100.0 * tol) {
                return false;
            }
        }
        return true;
    }

    static bool can_increase_from_lower_(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                         double tol) {
        const double x_at_bound = (j >= 0 && j < l.size() && std::isfinite(l(j))) ? l(j) : 0.0;
        if (j >= 0 && j < u.size() && std::isfinite(u(j))) {
            return (u(j) - x_at_bound) > tol;
        }
        return true;
    }

    FTBasis::Options make_basis_options_() const {
        FTBasis::Options bopt;
        bopt.refactor_every = opt_.refactor_every;
        bopt.compress_every = opt_.compress_every;
        bopt.pivot_rel = opt_.lu_pivot_rel;
        bopt.abs_floor = opt_.lu_abs_floor;
        bopt.alpha_tol = opt_.alpha_tol;
        bopt.z_inf_guard = opt_.z_inf_guard;
        bopt.ft_multiplier_guard = opt_.ft_multiplier_guard;
        bopt.ft_bandwidth_cap = opt_.ft_bandwidth_cap;
        bopt.max_growth_tol = opt_.max_growth_tol;
        bopt.min_dynamic_growth_tol = opt_.min_dynamic_growth_tol;
        bopt.max_condition_estimate = opt_.max_condition_estimate;
        bopt.refinement_steps = opt_.basis_refinement_steps;
        bopt.residual_refactor_tol = opt_.basis_residual_refactor_tol;
        bopt.residual_abs_refactor_tol = opt_.residual_abs_refactor_tol;
        bopt.refinement_max_steps = opt_.refinement_max_steps;
        bopt.refinement_slow_progress_ratio = opt_.refinement_slow_progress_ratio;
        bopt.refinement_stall_progress_ratio = opt_.basis_refinement_stall_progress_ratio;
        bopt.refinement_stall_limit = opt_.basis_refinement_stall_limit;
        bopt.max_eta_count = opt_.basis_max_eta_count;
        bopt.column_residual_tol = opt_.basis_column_residual_tol;
        bopt.aggressive_refactor_on_suspicious_residual = opt_.basis_aggressive_residual_rebuild;
        bopt.sparse_backend = opt_.basis_sparse_backend;
        bopt.sparse_equilibration = opt_.basis_sparse_equilibration;
        bopt.sparse_rhs_density_threshold = opt_.basis_sparse_rhs_density_threshold;

        std::string mode = opt_.basis_update;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (mode == "eta" || mode == "eta_stack") {
            bopt.update_mode = FTBasis::Options::UpdateMode::EtaStack;
        } else if (mode == "hybrid") {
            bopt.update_mode = FTBasis::Options::UpdateMode::Hybrid;
        } else {
            bopt.update_mode = FTBasis::Options::UpdateMode::ForrestTomlin;
        }

        bopt.ext_refactor_counter = &solve_stats_.refactorizations;
        bopt.ext_refactor_ns = &solve_stats_.lu_build_ns;
        bopt.ext_pivot_ns = &solve_stats_.pivot_ns;
        return bopt;
    }

    static Eigen::VectorXd assemble_primal_(int n, const std::vector<int>& basis,
                                            const Eigen::VectorXd& xB, const Eigen::VectorXd& l,
                                            const Eigen::VectorXd& u,
                                            const std::vector<int>* sigma = nullptr) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j >= 0 && j < n) {
                inB[j] = 1;
                if (i < xB.size())
                    x(j) = xB(i);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (inB[j])
                continue;

            const bool upper_view = sigma && j < (int)sigma->size() && (*sigma)[j] < 0;
            if (upper_view && j < u.size() && std::isfinite(u(j))) {
                x(j) = u(j);
            } else if (j < l.size() && std::isfinite(l(j))) {
                x(j) = l(j);
            } else {
                x(j) = 0.0;
            }
        }

        return clip_small_(x);
    }

    static Eigen::VectorXd repair_nan_primal_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                              const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                              Eigen::VectorXd x, double tol = 1e-9) {
        if (x.size() == 0 || x.array().isFinite().all())
            return x;

        std::vector<int> unknown;
        std::vector<int> known;
        unknown.reserve(x.size());
        known.reserve(x.size());
        for (int j = 0; j < x.size(); ++j) {
            if (std::isfinite(x(j)))
                known.push_back(j);
            else
                unknown.push_back(j);
        }

        if (!unknown.empty() && A.rows() > 0) {
            Eigen::VectorXd rhs = b;
            for (int j : known)
                rhs.noalias() -= A.col(j) * x(j);

            Eigen::MatrixXd AU(A.rows(), static_cast<int>(unknown.size()));
            for (int k = 0; k < (int)unknown.size(); ++k) {
                AU.col(k) = A.col(unknown[k]);
            }

            if (AU.size() > 0) {
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(AU);
                const Eigen::VectorXd z = qr.solve(rhs);
                if (z.size() == (int)unknown.size() && z.array().isFinite().all()) {
                    for (int k = 0; k < (int)unknown.size(); ++k) {
                        x(unknown[k]) = z(k);
                    }
                }
            }
        }

        for (int j = 0; j < x.size(); ++j) {
            if (!std::isfinite(x(j))) {
                if (j < l.size() && std::isfinite(l(j)))
                    x(j) = l(j);
                else if (j < u.size() && std::isfinite(u(j)))
                    x(j) = u(j);
                else
                    x(j) = 0.0;
            }
            if (j < l.size() && std::isfinite(l(j)) && x(j) < l(j) - tol)
                x(j) = l(j);
            if (j < u.size() && std::isfinite(u(j)) && x(j) > u(j) + tol)
                x(j) = u(j);
        }

        return clip_small_(x);
    }

    struct CrashCandidate {
        int col = -1;
        int pivot_row = -1;
        double score = -std::numeric_limits<double>::infinity();
    };

    enum class CrashStyle { Hybrid, Repair, Sprint, CrashII, CrashIII };

    struct CrashAttemptConfig {
        CrashStyle style = CrashStyle::Hybrid;
        std::string style_name = "hybrid";
        double markowitz_threshold = 0.2;
        double cost_penalty = 0.05;
        double rhs_bonus = 0.25;
        double dense_penalty = 0.5;
        double coverage_weight = 1.0;
        double seed_penalty = 0.0;
        double jitter = 0.0;
        int local_search_passes = 0;
        int max_swap_candidates = 8;
        bool prefer_seed_columns = false;
    };

    struct BasisQuality {
        bool valid = false;
        bool primal_feasible = false;
        bool dual_feasible = false;
        int rank = 0;
        double primal_violation = std::numeric_limits<double>::infinity();
        double dual_violation = std::numeric_limits<double>::infinity();
        double solve_residual = std::numeric_limits<double>::infinity();
        double density = std::numeric_limits<double>::infinity();
    };

    struct CrashSelection {
        std::vector<int> basis;
        BasisQuality quality;
        std::string source = "none";
        std::string style = "none";
        int attempt = -1;
    };

    static double positive_violation_max_(const Eigen::VectorXd& x, double tol);

    static BasisQuality evaluate_basis_quality_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                                const Eigen::VectorXd& c,
                                                const std::vector<int>& basis, double tol);

    static BasisQuality evaluate_basis_quality_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                const Eigen::VectorXd& c,
                                                const std::vector<int>& basis, double tol);

    static bool better_basis_quality_(const CrashSelection& lhs, const CrashSelection& rhs,
                                      SimplexMode mode);

    static std::string lower_copy_(std::string value);

    static CrashStyle parse_crash_style_(const std::string& strategy);

    static const char* crash_style_name_(CrashStyle style);

    static CrashAttemptConfig crash_attempt_config_(const RevisedSimplexOptions& opt, int attempt);
    static CrashAttemptConfig
    quadratic_warm_start_repair_attempt_config_(const RevisedSimplexOptions& opt, int attempt);

    static void mark_pivot_row_(const Eigen::MatrixXd& A, int col, int pivot_row_hint,
                                std::vector<char>& used_row);

    static void mark_pivot_row_(const SparseMatrix& A, int col, int pivot_row_hint,
                                std::vector<char>& used_row);

    static bool try_add_basis_column_(const Eigen::MatrixXd& A, std::vector<int>& basis,
                                      std::vector<char>& used_row, std::vector<char>& used_col,
                                      int& current_rank, int col, int pivot_row_hint, double tol);

    static bool try_add_basis_column_(const SparseMatrix& A, std::vector<int>& basis,
                                      std::vector<char>& used_row, std::vector<char>& used_col,
                                      int& current_rank, int col, int pivot_row_hint, double tol);

    static double seed_column_bonus_(int col, const std::vector<char>& seeded,
                                     const CrashAttemptConfig& cfg);

    static CrashCandidate
    choose_slack_like_column_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                              const Eigen::VectorXd& c, const std::vector<char>& used_row,
                              const std::vector<char>& used_col, const std::vector<char>& seeded,
                              const CrashAttemptConfig& cfg);

    static CrashCandidate choose_slack_like_column_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                    const Eigen::VectorXd& c,
                                                    const std::vector<char>& used_row,
                                                    const std::vector<char>& used_col,
                                                    const std::vector<char>& seeded,
                                                    const CrashAttemptConfig& cfg);

    static CrashCandidate
    choose_free_like_column_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                             const Eigen::VectorXd& c, const std::vector<char>& used_row,
                             const std::vector<char>& used_col, const std::vector<char>& seeded,
                             const CrashAttemptConfig& cfg);

    static CrashCandidate choose_free_like_column_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                   const Eigen::VectorXd& c,
                                                   const std::vector<char>& used_row,
                                                   const std::vector<char>& used_col,
                                                   const std::vector<char>& seeded,
                                                   const CrashAttemptConfig& cfg);

    static CrashCandidate choose_sprint_column_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                                const Eigen::VectorXd& c,
                                                const std::vector<char>& used_row,
                                                const std::vector<char>& used_col,
                                                const std::vector<char>& seeded,
                                                const CrashAttemptConfig& cfg);

    static CrashCandidate
    choose_sprint_column_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                          const std::vector<char>& used_row, const std::vector<char>& used_col,
                          const std::vector<char>& seeded, const CrashAttemptConfig& cfg);

    static std::vector<int> find_logical_basis_(const Eigen::MatrixXd& A);

    static std::vector<int> find_logical_basis_(const SparseMatrix& A);

    static CrashCandidate
    choose_triangular_column_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                              const Eigen::VectorXd& c, const std::vector<char>& used_row,
                              const std::vector<char>& used_col, const std::vector<char>& seeded,
                              const CrashAttemptConfig& cfg);

    static CrashCandidate choose_triangular_column_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                    const Eigen::VectorXd& c,
                                                    const std::vector<char>& used_row,
                                                    const std::vector<char>& used_col,
                                                    const std::vector<char>& seeded,
                                                    const CrashAttemptConfig& cfg);

    static std::vector<int> rank_remaining_columns_(const Eigen::MatrixXd& A,
                                                    const Eigen::VectorXd& c,
                                                    const std::vector<char>& used_col,
                                                    const std::vector<char>& seeded,
                                                    const CrashAttemptConfig& cfg);

    static std::vector<int> rank_remaining_columns_(const SparseMatrix& A, const Eigen::VectorXd& c,
                                                    const std::vector<char>& used_col,
                                                    const std::vector<char>& seeded,
                                                    const CrashAttemptConfig& cfg);

    static std::vector<int>
    improve_basis_by_swaps_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::vector<int> basis,
                            const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                            std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    improve_basis_by_swaps_(const SparseMatrix& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::vector<int> basis,
                            const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                            std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    build_basis_attempt_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                         const Eigen::VectorXd& c, const CrashAttemptConfig& cfg, double tol,
                         SimplexMode mode,
                         std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    build_basis_attempt_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                         const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                         std::optional<std::vector<int>> seed_basis = std::nullopt);

    static CrashSelection
    choose_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                          std::optional<std::vector<int>> seed_basis = std::nullopt,
                          bool allow_direct_warm_start = true);

    static CrashSelection
    choose_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                          const RevisedSimplexOptions& opt,
                          std::optional<std::vector<int>> seed_basis = std::nullopt,
                          bool allow_direct_warm_start = true);

    static std::optional<std::vector<int>>
    find_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                        const Eigen::VectorXd& c,
                        const RevisedSimplexOptions& opt = RevisedSimplexOptions{},
                        std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::optional<std::vector<int>>
    find_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                        const RevisedSimplexOptions& opt = RevisedSimplexOptions{},
                        std::optional<std::vector<int>> seed_basis = std::nullopt);

    static bool basis_is_primal_feasible_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                          const std::vector<int>& basis, double tol);

    static bool basis_is_primal_feasible_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                          const std::vector<int>& basis, double tol);

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, std::vector<int>,
                      std::size_t, int>
    make_phase1_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

    static std::tuple<SparseMatrix, Eigen::VectorXd, Eigen::VectorXd, std::vector<int>, std::size_t,
                      int>
    make_phase1_(const SparseMatrix& A, const Eigen::VectorXd& b);

    // --------------------------- PRIMAL PHASE ---------------------------
    PhaseResult phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u);

    PhaseResult phase_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u);

    // --------------------------- DUAL PHASE ---------------------------
    PhaseResult dual_phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status = std::nullopt);

    PhaseResult dual_phase_(const SparseMatrix& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status = std::nullopt);

    bool should_reuse_cached_basis_(int rows, int cols) const {
        return opt_.mode == SimplexMode::Dual && cached_basis_state_ &&
               cached_basis_rows_ == rows && cached_basis_cols_ == cols &&
               basis_state_matches_problem_(*cached_basis_state_, rows, cols);
    }

  public:
    bool has_cached_basis_state(const Eigen::MatrixXd& A) const noexcept {
        return has_cached_basis_state(static_cast<int>(A.rows()), static_cast<int>(A.cols()),
                                      matrix_signature_(A), false);
    }

    bool has_cached_basis_state(const SparseMatrix& A) const noexcept {
        // Fast path: reuse cached signature when the same compressed matrix pointer is seen.
        const double* ptr = A.isCompressed() ? A.valuePtr() : nullptr;
        const std::uint64_t sig = (ptr != nullptr && ptr == last_sparse_a_value_ptr_ &&
                                   static_cast<int>(A.rows()) == last_sparse_a_rows_ &&
                                   static_cast<int>(A.cols()) == last_sparse_a_cols_)
                                      ? last_sparse_a_signature_
                                      : matrix_signature_(A);
        return has_cached_basis_state(static_cast<int>(A.rows()), static_cast<int>(A.cols()), sig,
                                      true);
    }

    bool has_cached_basis_factorization_state(const SparseMatrix& A) const noexcept {
        const double* ptr = A.isCompressed() ? A.valuePtr() : nullptr;
        const std::uint64_t sig = (ptr != nullptr && ptr == last_sparse_a_value_ptr_ &&
                                   static_cast<int>(A.rows()) == last_sparse_a_rows_ &&
                                   static_cast<int>(A.cols()) == last_sparse_a_cols_)
                                      ? last_sparse_a_signature_
                                      : matrix_signature_(A);
        return has_cached_basis_state(static_cast<int>(A.rows()), static_cast<int>(A.cols()), sig,
                                      true) &&
               cached_basis_state_->warm_state &&
               cached_basis_state_->warm_state->nla;
    }

  private:
    bool has_cached_basis_state(int rows, int cols, std::uint64_t signature,
                                bool matrix_is_sparse) const noexcept {
        // Allow cached basis reuse in Auto as well as Dual so warm-started
        // re-solves can benefit from the stored basis state. The solver still
        // preserves its normal mode fallback logic (Auto may fall back from
        // primal to dual if needed).
        return cached_basis_state_ && cached_basis_rows_ == rows && cached_basis_cols_ == cols &&
               cached_matrix_signature_ == signature &&
               cached_matrix_is_sparse_ == matrix_is_sparse &&
               basis_state_matches_problem_(*cached_basis_state_, rows, cols);
    }

    void update_cached_basis_(const LPSolution& sol, int rows, const Eigen::VectorXd& l,
                              const Eigen::VectorXd& u) {
        const int cols = static_cast<int>(l.size());
        if (sol.x.size() == cols && sol.x.array().isFinite().all()) {
            LPBasis rebuilt = compute_basis_state_(sol.basis, sol.x, l, u, opt_.tol, rows);
            rebuilt.warm_state = sol.basis_state.warm_state;
            if (basis_state_matches_problem_(rebuilt, rows, cols)) {
                cached_basis_state_ = rebuilt;
                cached_basis_rows_ = rows;
                cached_basis_cols_ = cols;
                cached_matrix_signature_ = current_matrix_signature_;
                cached_matrix_is_sparse_ = current_matrix_is_sparse_;
                cached_basis_l_ = l;
                cached_basis_u_ = u;
                return;
            }
        }
        if (basis_state_matches_problem_(sol.basis_state, rows, cols)) {
            cached_basis_state_ = sol.basis_state;
            cached_basis_rows_ = rows;
            cached_basis_cols_ = cols;
            cached_matrix_signature_ = current_matrix_signature_;
            cached_matrix_is_sparse_ = current_matrix_is_sparse_;
            cached_basis_l_ = l;
            cached_basis_u_ = u;
        } else {
            clear_basis_cache();
        }
    }

    // Reformulation variable descriptor: maps original x_j into the standard-form
    // variable(s) y_j used in the bound-shifted problem.  Lives here (not inside
    // solve_impl_sparse_) so that SparseBoundOnlyCache can pre-allocate a reusable
    // scratch vector and avoid a heap allocation on every BnB node solve.
    struct ReformVar {
        int y = -1;           // single shifted var (uses_single_var path)
        int y_pos = -1;       // positive part (free var split path)
        int y_neg = -1;       // negative part
        int upper_slack = -1; // upper-bound row slack column index
        double shift = 0.0;   // x_j = shift + sign * y
        int sign = 1;         // +1 for lb-shift, -1 for ub-shift, 0 for fixed
        bool uses_single_var = false;
        bool has_upper_row = false;
    };

    struct SparseBoundOnlyCache {
        int rows = 0;
        int cols = 0;
        SparseMatrix A_in;
        Eigen::VectorXd b_in;
        Eigen::VectorXd c_in;
        // Pointer identity of the last A seen — if same pointer is passed we skip
        // the O(nnz) comparison. Valid only because in BnB the same A_sparse object
        // (owned by NodeLPCacheEntry) is reused across all node solves.
        const double* cached_A_value_ptr = nullptr;
        std::vector<char> has_lower;
        std::vector<char> has_upper;
        std::vector<char> fixed_bound;
        std::vector<int> single_y;
        std::vector<int> split_pos;
        std::vector<int> split_neg;
        std::vector<int> upper_slack;
        SparseMatrix A_std;
        Eigen::VectorXd c_std;
        // Pre-allocated scratch buffer for reconstruct_sparse_reformulated_rhs_
        mutable Eigen::VectorXd b_std_scratch;
        // Standard-form l/u: always 0 / +inf for shifted/reformulated variables.
        // Cached here to avoid re-allocation on every node solve.
        Eigen::VectorXd l_std;
        Eigen::VectorXd u_std;
        // Pre-allocated map scratch: reused each node solve to avoid the
        // std::vector<ReformVar>(n) heap allocation in the hot path.
        mutable std::vector<ReformVar> map_scratch;
        // Scaled data_scale = max(1, max_abs(A), max_abs(b)), computed once
        // at cache build to skip the O(nnz) matrix scan in canonicalize_inactive_huge_bounds_.
        double cached_data_scale = 0.0;
        // Incremental RHS: store previous bounds so we can update only changed columns.
        mutable Eigen::VectorXd l_prev_scratch;
        mutable Eigen::VectorXd u_prev_scratch;
        mutable bool b_std_scratch_valid = false; // true after first reconstruct post-build
        // Persistent solver for the reformulated subproblem.  Keeps the LU factorization
        // alive across BnB node solves (HiGHS/SCIP hot-restart pattern).
        // unique_ptr because RevisedSimplex is an incomplete type at this point in the header.
        mutable std::unique_ptr<RevisedSimplex> reformulated_solver_cache;
        // Last optimal basis_state returned by the reformulated solver. Feeding
        // it back into the next solve (rather than relying on the solver's
        // internal no-basis cache path, which loses the nonbasic bound views and
        // goes singular) is what lets the FTBasis factorization actually be
        // reused -- a dual warm restart in a few pivots instead of a cold refactor.
        mutable std::optional<LPBasis> last_reformulated_basis_state;
        int m_eq = 0;
        int nv = 0;
        int upper_rows = 0;
        int n_total = 0;
        int m_total = 0;
        bool valid = false;

        bool same_problem(const SparseMatrix& A, const Eigen::VectorXd& b,
                          const Eigen::VectorXd& c) const {
            if (!valid)
                return false;
            if (A.rows() != rows || A.cols() != cols)
                return false;
            if (b.size() != rows || c.size() != cols)
                return false;
            // Fast path: if the same compressed matrix object (same value pointer),
            // skip the O(nnz) element-by-element comparison.
            if (A.isCompressed() && A_in.isCompressed() && A.valuePtr() == cached_A_value_ptr &&
                A.nonZeros() == A_in.nonZeros()) {
                if (!b.isApprox(b_in) || !c.isApprox(c_in))
                    return false;
                return true;
            }
            if (!b.isApprox(b_in) || !c.isApprox(c_in))
                return false;
            if (A.nonZeros() != A_in.nonZeros())
                return false;
            if (!A.isCompressed() || !A_in.isCompressed())
                return false;
            const int* outerA = A.outerIndexPtr();
            const int* outerCache = A_in.outerIndexPtr();
            if (outerA[cols] != outerCache[cols])
                return false;
            for (int j = 0; j <= cols; ++j) {
                if (outerA[j] != outerCache[j])
                    return false;
            }
            const int* innerA = A.innerIndexPtr();
            const int* innerCache = A_in.innerIndexPtr();
            for (int k = 0; k < A.nonZeros(); ++k) {
                if (innerA[k] != innerCache[k])
                    return false;
            }
            const double* valueA = A.valuePtr();
            const double* valueCache = A_in.valuePtr();
            for (int k = 0; k < A.nonZeros(); ++k) {
                if (valueA[k] != valueCache[k])
                    return false;
            }
            return true;
        }

        bool orientation_matches(const Eigen::VectorXd& l_use, const Eigen::VectorXd& u_use) const {
            if (!valid || l_use.size() != cols || u_use.size() != cols)
                return false;
            for (int j = 0; j < cols; ++j) {
                const bool has_l = std::isfinite(l_use(j));
                const bool has_u = std::isfinite(u_use(j));
                const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= 1e-12;
                if (has_lower[j] != static_cast<char>(has_l) ||
                    has_upper[j] != static_cast<char>(has_u) ||
                    fixed_bound[j] != static_cast<char>(fixed)) {
                    return false;
                }
            }
            return true;
        }

        bool same_problem(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                          const Eigen::VectorXd& l_use, const Eigen::VectorXd& u_use) const {
            if (!valid || !same_problem(A, b, c) || !orientation_matches(l_use, u_use)) {
                return false;
            }
            return true;
        }

        void reset() {
            rows = cols = 0;
            A_in.resize(0, 0);
            b_in.resize(0);
            c_in.resize(0);
            has_lower.clear();
            has_upper.clear();
            fixed_bound.clear();
            single_y.clear();
            split_pos.clear();
            split_neg.clear();
            upper_slack.clear();
            A_std.resize(0, 0);
            c_std.resize(0);
            map_scratch.clear();
            cached_data_scale = 0.0;
            b_std_scratch_valid = false;
            reformulated_solver_cache.reset();
            last_reformulated_basis_state.reset();
            m_eq = nv = upper_rows = n_total = m_total = 0;
            valid = false;
        }
    };

    SparseBoundOnlyCache sparse_bound_only_cache_;

    // Reformulation methods defined out-of-line in simplex_reformulation.h.
    void build_sparse_bound_only_cache_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c, const Eigen::VectorXd& l_use,
                                        const Eigen::VectorXd& u_use);

    const Eigen::VectorXd& reconstruct_sparse_reformulated_rhs_(const Eigen::VectorXd& l_use,
                                                                const Eigen::VectorXd& u_use) const;

    SanitizedBounds canonicalize_inactive_huge_bounds_(const SparseMatrix& A,
                                                       const Eigen::VectorXd& b,
                                                       const Eigen::VectorXd& l,
                                                       const Eigen::VectorXd& u,
                                                       double precomputed_data_scale = 0.0) const;

    // --------------------------- Utilities ---------------------------
    static LPSolution make_solution_(LPSolution::Status st, Eigen::VectorXd x, double obj,
                                     std::vector<int> basis, int iters,
                                     std::unordered_map<std::string, std::string> info,
                                     std::optional<Eigen::VectorXd> farkas_y = std::nullopt,
                                     std::optional<bool> farkas_has_cert = std::nullopt,
                                     std::optional<Eigen::VectorXd> primal_ray = std::nullopt,
                                     std::optional<bool> primal_ray_has_cert = std::nullopt) {
        LPSolution sol;
        sol.status = st;
        sol.x = std::move(x);
        sol.obj = obj;
        sol.basis = std::move(basis);
        sol.iters = iters;
        sol.info = std::move(info);
        sol.farkas_y = farkas_y ? std::move(*farkas_y) : Eigen::VectorXd{};
        sol.farkas_has_cert = farkas_has_cert.value_or(false);
        sol.primal_ray = primal_ray ? std::move(*primal_ray) : Eigen::VectorXd{};
        sol.primal_ray_has_cert = primal_ray_has_cert.value_or(false);
        return sol;
    }

    static bool is_recoverable_basis_runtime_(std::string_view msg) noexcept {
        return msg.find("MarkowitzLU:") != std::string_view::npos ||
               msg.find("SparseForrestTomlinLU:") != std::string_view::npos ||
               msg.find("FTBasis::refine_solve_B") != std::string_view::npos ||
               msg.find("FTBasis::refine_solve_BT") != std::string_view::npos ||
               msg.find("FTBasis::solve_B") != std::string_view::npos ||
               msg.find("FTBasis::solve_BT") != std::string_view::npos ||
               msg.find("Forrest-Tomlin:") != std::string_view::npos;
    }

  private:
    // Options and state
    RevisedSimplexOptions opt_;
    std::mt19937 rng_;

    // NLA layer (PF updates, iterate snapshots, framework switching, price strategy)
    simplex::nla::SimplexNLA nla_;

    // Degeneracy + pricing
    DegeneracyManager degen_;
    AdaptivePricer adaptive_pricer_{1};
    std::unique_ptr<PrimalPricingBridge<AdaptivePricer>> bridge_;
    std::optional<LPBasis> cached_basis_state_;
    int cached_basis_rows_ = -1;
    int cached_basis_cols_ = -1;
    Eigen::VectorXd cached_basis_l_;
    Eigen::VectorXd cached_basis_u_;
    std::uint64_t cached_matrix_signature_ = 0;
    bool cached_matrix_is_sparse_ = false;
    std::uint64_t current_matrix_signature_ = 0;
    int current_matrix_rows_ = -1;
    int current_matrix_cols_ = -1;
    bool current_matrix_is_sparse_ = false;
    // Cache the last sparse-matrix value pointer + its signature to avoid
    // recomputing the O(nnz) hash when the same A_sparse object is reused
    // across consecutive solves (e.g., successive BnB node LP solves).
    const double* last_sparse_a_value_ptr_ = nullptr;
    int last_sparse_a_rows_ = -1;
    int last_sparse_a_cols_ = -1;
    std::uint64_t last_sparse_a_signature_ = 0;
    std::shared_ptr<LPWarmStateData> solve_input_warm_state_;
    std::shared_ptr<LPWarmStateData> solve_output_warm_state_;
    mutable std::vector<std::string> trace_;
    mutable int solve_depth_ = 0;
    RevisedSimplexTiming current_timing_;
    mutable LPSolveStats solve_stats_;
};

#include "simplex/factorization/crash.h"
#include "simplex/types/dual.h"
#include "simplex/engine/phase1.h"
#include "simplex/engine/postsolve.h"
#include "simplex/primal.h"
#include "simplex/engine/simplex_reformulation.h"
#include "simplex/core/sparse_utils.h"

inline RevisedSimplex::PhaseResult
RevisedSimplex::phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u) {
    return RevisedSimplexPrimalEngine::run(*this, A, b, c, std::move(basis_opt), l, u);
}

inline LPSolution RevisedSimplex::solve_impl_sparse_(
    const SparseMatrix& A_in, const Eigen::VectorXd& b_in, const Eigen::VectorXd& c_in,
    const Eigen::VectorXd& l_in, const Eigen::VectorXd& u_in,
    std::optional<std::vector<int>> basis_opt, const LPBasis* basis_state_opt) {
    auto t0_presolve = std::chrono::steady_clock::now();
    SolveTraceScope trace_scope(*this);
    degen_.reset();
    const int m_in = static_cast<int>(A_in.rows());
    const int n = static_cast<int>(A_in.cols());
    if (b_in.size() != m_in) {
        throw std::invalid_argument("simplex: b size mismatch with rows(A)");
    }
    if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
        throw std::invalid_argument("simplex: c/l/u sizes must equal cols(A)");
    }
    // Fast-path: if the same compressed A_in is passed as last time, reuse the
    // cached signature and skip the O(nnz) hash computation.
    const double* a_value_ptr = A_in.isCompressed() ? A_in.valuePtr() : nullptr;
    const bool same_sparse_ptr =
        (a_value_ptr != nullptr && a_value_ptr == last_sparse_a_value_ptr_ &&
         m_in == last_sparse_a_rows_ && n == last_sparse_a_cols_);
    const std::uint64_t sig = same_sparse_ptr ? last_sparse_a_signature_ : matrix_signature_(A_in);
    if (a_value_ptr != nullptr) {
        last_sparse_a_value_ptr_ = a_value_ptr;
        last_sparse_a_rows_ = m_in;
        last_sparse_a_cols_ = n;
        last_sparse_a_signature_ = sig;
    }
    begin_solve_(sig, m_in, n, true, basis_state_opt);

    trace_line_("[solve] sparse start m=" + std::to_string(m_in) + " n=" + std::to_string(n));
    trace_line_("[solve] sparse disable_presolve=" + std::to_string(opt_.disable_presolve));

    // Pass cached data_scale to skip the O(nnz) max-abs matrix scan when A hasn't changed.
    const double precomputed_ds = (same_sparse_ptr && sparse_bound_only_cache_.valid &&
                                   sparse_bound_only_cache_.cached_data_scale > 0.0)
                                      ? sparse_bound_only_cache_.cached_data_scale
                                      : 0.0;
    const auto sanitized_bounds =
        canonicalize_inactive_huge_bounds_(A_in, b_in, l_in, u_in, precomputed_ds);
    const Eigen::VectorXd& l_use = sanitized_bounds.l;
    const Eigen::VectorXd& u_use = sanitized_bounds.u;
    bool reused_sparse_bound_cache = false;

    std::optional<LPBasis> rebased_basis_state_opt = std::nullopt;
    if (basis_state_opt && !basis_state_opt->column_status.empty()) {
        rebased_basis_state_opt =
            rebase_basis_state_for_bounds_(*basis_state_opt, l_use, u_use, opt_.tol);
        basis_state_opt = &*rebased_basis_state_opt;
    }

    // When a reusable factorization is in hand (same matrix, Dual mode), prefer
    // its basis over any passed-in seed: starting the dual re-solve from exactly
    // that basis lets try_reuse_factorization_ adopt the existing LU and FT-pivot
    // instead of refactoring. Otherwise the passed-in basis_opt (a different
    // basis) forces a cold factorization on every node solve.
    const std::optional<std::vector<int>> factorized_basis_seed_opt =
        warm_factorization_basis_seed_();
    if (factorized_basis_seed_opt) {
        basis_opt = *factorized_basis_seed_opt;
    }
    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        if ((int)basis_state_opt->column_status.size() != n) {
            throw std::invalid_argument("simplex: warm-start basis column_status size mismatch");
        }
    }

    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        basis_opt = basis_columns_from_basis_state_(*basis_state_opt, m_in);
    }

    bool is_nonnegative_standard = true;
    for (int j = 0; j < n; ++j) {
        const bool l_is_zero = std::isfinite(l_use(j)) && std::abs(l_use(j)) <= opt_.tol;
        const bool u_is_inf = !std::isfinite(u_use(j));
        if (!l_is_zero || !u_is_inf) {
            is_nonnegative_standard = false;
            break;
        }
    }

    const bool has_warm_basis_for_dual = opt_.mode == SimplexMode::Dual && basis_state_opt &&
                                         !basis_state_opt->column_status.empty();

    auto dualization_requested = [&]() {
        std::string mode = opt_.dualization;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (mode == "on" || mode == "true" || mode == "1")
            return true;
        if (mode != "auto")
            return false;
        if (n <= 0)
            return false;
        return static_cast<double>(m_in) / static_cast<double>(n) >=
               opt_.dualization_min_row_col_ratio;
    };

    if (is_nonnegative_standard && !has_warm_basis_for_dual && !basis_opt &&
        dualization_requested() && n <= opt_.dualization_max_recovery_cols) {
        try {
            std::vector<Eigen::Triplet<double>> dual_trips;
            dual_trips.reserve(static_cast<std::size_t>(2 * A_in.nonZeros() + n));
            for (int j = 0; j < A_in.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_in, j); it; ++it) {
                    dual_trips.emplace_back(j, it.row(), it.value());
                    dual_trips.emplace_back(j, m_in + it.row(), -it.value());
                }
            }
            for (int j = 0; j < n; ++j)
                dual_trips.emplace_back(j, 2 * m_in + j, 1.0);

            SparseMatrix A_dual(n, 2 * m_in + n);
            A_dual.setFromTriplets(dual_trips.begin(), dual_trips.end());
            A_dual.makeCompressed();
            Eigen::VectorXd b_dual = c_in;
            Eigen::VectorXd c_dual = Eigen::VectorXd::Zero(2 * m_in + n);
            for (int i = 0; i < m_in; ++i) {
                c_dual(i) = -b_in(i);
                c_dual(m_in + i) = b_in(i);
            }
            Eigen::VectorXd l_dual = Eigen::VectorXd::Zero(2 * m_in + n);
            Eigen::VectorXd u_dual = Eigen::VectorXd::Constant(2 * m_in + n, presolve::inf());

            RevisedSimplexOptions dual_opt = opt_;
            dual_opt.dualization = "off";
            dual_opt.mode = SimplexMode::Dual;
            dual_opt.compute_tableau = false;
            dual_opt.compute_reduced_costs = false;
            RevisedSimplex dual_solver(dual_opt);
            LPSolution dual_sol = dual_solver.solve(A_dual, b_dual, c_dual, l_dual, u_dual);
            if (dual_sol.status == LPSolution::Status::Optimal &&
                dual_sol.x.size() == 2 * m_in + n) {
                Eigen::VectorXd restricted_u = u_in;
                int fixed_by_dual_slack = 0;
                const double active_tol = std::max(1e-8, 100.0 * opt_.tol);
                for (int j = 0; j < n; ++j) {
                    const double slack = dual_sol.x(2 * m_in + j);
                    if (std::isfinite(slack) && slack > active_tol) {
                        restricted_u(j) = 0.0;
                        ++fixed_by_dual_slack;
                    }
                }
                if (fixed_by_dual_slack > 0) {
                    RevisedSimplexOptions recovery_opt = opt_;
                    recovery_opt.dualization = "off";
                    recovery_opt.mode = SimplexMode::Dual;
                    RevisedSimplex recovery_solver(recovery_opt);
                    LPSolution recovered =
                        recovery_solver.solve(A_in, b_in, c_in, l_in, restricted_u);
                    if (recovered.status == LPSolution::Status::Optimal &&
                        primal_feasible_(A_in, b_in, recovered.x, l_in, u_in, opt_.tol)) {
                        recovered.info["dualization"] = "explicit_dual_recovery";
                        recovered.info["dualization_fixed_by_slack"] =
                            std::to_string(fixed_by_dual_slack);
                        recovered.info["dualization_dual_iters"] = std::to_string(dual_sol.iters);
                        return finalize_solution_(std::move(recovered));
                    }
                }
            }
        } catch (const std::exception& e) {
            trace_line_(std::string("[dualization] fallback after failure: ") + e.what());
        }
    }

    if (!is_nonnegative_standard) {
        const bool cache_reuse = sparse_bound_only_cache_.same_problem(A_in, b_in, c_in) &&
                                 sparse_bound_only_cache_.orientation_matches(l_use, u_use);
        if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
            static std::atomic<long> reuse_y{0}, reuse_n{0};
            (cache_reuse ? reuse_y : reuse_n)++;
            if (((reuse_y + reuse_n) % 200) == 0)
                std::fprintf(stderr, "[reform] cache_reuse yes=%ld no=%ld\n", reuse_y.load(),
                             reuse_n.load());
        }
        // Reuse pre-allocated map_scratch to avoid heap allocation on every BnB node solve.
        // ReformVar is defined at class scope so SparseBoundOnlyCache can own this buffer.
        auto& map = sparse_bound_only_cache_.map_scratch;
        map.assign(n, ReformVar{});
        // The 4 int index vectors live in the cache; references are set after the
        // cache-hit / cache-miss block below (both paths ensure the cache is up-to-date).
        int nv = 0;
        int upper_rows = 0;
        double obj_shift = 0.0;
        const int m_eq = m_in;
        int n_total = 0;
        int m_total = 0;
        // Use pointers to dispatch between cached data (no allocation) and locally
        // owned buffers (non-cache path only). In the cache path, we point directly
        // into SparseBoundOnlyCache to avoid all per-node copies.
        const SparseMatrix* A_std_ptr = nullptr;
        const Eigen::VectorXd* b_std_ptr = nullptr;
        const Eigen::VectorXd* c_std_ptr = nullptr;
        const Eigen::VectorXd* l_std_ptr = nullptr;
        const Eigen::VectorXd* u_std_ptr = nullptr;
        // Owned storage — only used in the non-cache path.
        SparseMatrix A_std_owned;
        Eigen::VectorXd b_std_owned;
        Eigen::VectorXd c_std_owned;
        Eigen::VectorXd l_std_owned;
        Eigen::VectorXd u_std_owned;

        if (cache_reuse) {
            reused_sparse_bound_cache = true;
            const auto& cache = sparse_bound_only_cache_;
            n_total = cache.n_total;
            m_total = cache.m_total;
            upper_rows = cache.upper_rows;
            // Fill b_std_scratch in-place (no allocation); use by reference below.
            reconstruct_sparse_reformulated_rhs_(l_use, u_use);
            // Point directly into cached data — no copies.
            A_std_ptr = &cache.A_std;
            b_std_ptr = &cache.b_std_scratch;
            c_std_ptr = &cache.c_std;
            l_std_ptr = &cache.l_std;
            u_std_ptr = &cache.u_std;
            for (int j = 0; j < n; ++j) {
                const bool has_l = static_cast<bool>(cache.has_lower[j]);
                const bool has_u = static_cast<bool>(cache.has_upper[j]);
                const bool fixed = static_cast<bool>(cache.fixed_bound[j]);
                if (has_l || has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = cache.single_y[j];
                    map[j].shift = has_l ? l_use(j) : u_use(j);
                    map[j].sign = fixed ? 0 : (has_l ? 1 : -1);
                    map[j].has_upper_row = has_l && has_u && !fixed;
                } else {
                    map[j].y_pos = cache.split_pos[j];
                    map[j].y_neg = cache.split_neg[j];
                }
                map[j].upper_slack = cache.upper_slack[j];
                // single_y/split_pos/split_neg/upper_slack are NOT copied to local vectors;
                // use sparse_bound_only_cache_.* directly via references below.
                if (has_l) {
                    obj_shift += c_in(j) * l_use(j);
                } else if (has_u) {
                    obj_shift += c_in(j) * u_use(j);
                }
            }
        } else {
            for (int j = 0; j < n; ++j) {
                const bool has_l = std::isfinite(l_use(j));
                const bool has_u = std::isfinite(u_use(j));
                if (has_l && has_u && u_use(j) < l_use(j) - opt_.tol) {
                    Eigen::VectorXd xnan =
                        Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                    return finalize_solution_(
                        make_solution_(LPSolution::Status::Infeasible, xnan,
                                       std::numeric_limits<double>::infinity(), {}, 0,
                                       {{"reason", "invalid_bounds"}}));
                }
                const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= opt_.tol;
                if (fixed) {
                    map[j].uses_single_var = true;
                    map[j].y = -1;
                    map[j].shift = l_use(j);
                    map[j].sign = 0;
                    obj_shift += c_in(j) * l_use(j);
                } else if (has_l) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = l_use(j);
                    map[j].sign = 1;
                    obj_shift += c_in(j) * l_use(j);
                    if (has_u) {
                        map[j].has_upper_row = true;
                        ++upper_rows;
                    }
                } else if (has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = u_use(j);
                    map[j].sign = -1;
                    obj_shift += c_in(j) * u_use(j);
                } else {
                    map[j].y_pos = nv++;
                    map[j].y_neg = nv++;
                }
            }

            n_total = nv + upper_rows;
            m_total = m_eq + upper_rows;
            b_std_owned = Eigen::VectorXd::Zero(m_total);
            c_std_owned = Eigen::VectorXd::Zero(n_total);
            l_std_owned = Eigen::VectorXd::Zero(n_total);
            u_std_owned = Eigen::VectorXd::Constant(n_total, presolve::inf());

            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    if (map[j].y < 0) {
                        continue;
                    }
                    c_std_owned(map[j].y) += static_cast<double>(map[j].sign) * c_in(j);
                } else {
                    c_std_owned(map[j].y_pos) += c_in(j);
                    c_std_owned(map[j].y_neg) += -c_in(j);
                }
            }

            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(static_cast<std::size_t>(A_in.nonZeros() * 2 + upper_rows * 2));
            for (int j = 0; j < A_in.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_in, j); it; ++it) {
                    const int row = it.row();
                    const double aij = it.value();
                    if (map[j].uses_single_var) {
                        b_std_owned(row) -= aij * map[j].shift;
                        if (map[j].y >= 0) {
                            trips.emplace_back(row, map[j].y,
                                               static_cast<double>(map[j].sign) * aij);
                        }
                    } else {
                        trips.emplace_back(row, map[j].y_pos, aij);
                        trips.emplace_back(row, map[j].y_neg, -aij);
                    }
                }
            }
            for (int i = 0; i < m_eq; ++i)
                b_std_owned(i) += b_in(i);

            int upper_row = 0;
            for (int j = 0; j < n; ++j) {
                if (!map[j].has_upper_row)
                    continue;
                const int slack = nv + upper_row;
                const int row = m_eq + upper_row;
                map[j].upper_slack = slack;
                // upper_slack vector lives in the cache (filled by build_sparse_bound_only_cache_).
                trips.emplace_back(row, map[j].y, 1.0);
                trips.emplace_back(row, slack, 1.0);
                b_std_owned(row) = u_use(j) - l_use(j);
                ++upper_row;
            }

            A_std_owned = SparseMatrix(m_total, n_total);
            if (!trips.empty())
                A_std_owned.setFromTriplets(trips.begin(), trips.end());
            A_std_owned.makeCompressed();
            build_sparse_bound_only_cache_(A_in, b_in, c_in, l_use, u_use);

            A_std_ptr = &A_std_owned;
            b_std_ptr = &b_std_owned;
            c_std_ptr = &c_std_owned;
            l_std_ptr = &l_std_owned;
            u_std_ptr = &u_std_owned;
        }
        // Uniform reference bindings — used from here on in both paths.
        const SparseMatrix& A_std = *A_std_ptr;
        const Eigen::VectorXd& b_std = *b_std_ptr;
        const Eigen::VectorXd& c_std = *c_std_ptr;
        const Eigen::VectorXd& l_std = *l_std_ptr;
        const Eigen::VectorXd& u_std = *u_std_ptr;
        // Cache int-index vectors are now authoritative in both paths.
        // Cache path: never wrote to them (used cache directly).
        // Non-cache path: build_sparse_bound_only_cache_ populated them.
        const std::vector<int>& single_y = sparse_bound_only_cache_.single_y;
        const std::vector<int>& upper_slack = sparse_bound_only_cache_.upper_slack;
        const std::vector<int>& split_pos = sparse_bound_only_cache_.split_pos;
        const std::vector<int>& split_neg = sparse_bound_only_cache_.split_neg;
        std::optional<std::vector<int>> basis_std = std::nullopt;
        std::optional<LPBasis> basis_state_std = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::vector<int> cand;
            cand.reserve(std::min(m_eq, (int)basis_opt->size()) + upper_rows);
            for (int jorig : *basis_opt) {
                if (jorig < 0 || jorig >= n)
                    continue;
                if (map[jorig].uses_single_var) {
                    if (map[jorig].y >= 0)
                        cand.push_back(map[jorig].y);
                } else if (map[jorig].y_pos >= 0) {
                    cand.push_back(map[jorig].y_pos);
                }
                if ((int)cand.size() == m_eq)
                    break;
            }
            for (int j = 0; j < n; ++j) {
                if (map[j].upper_slack >= 0)
                    cand.push_back(map[j].upper_slack);
            }
            if ((int)cand.size() == m_total)
                basis_std = std::move(cand);
        }
        if (basis_state_opt && !basis_state_opt->column_status.empty() &&
            (int)basis_state_opt->column_status.size() == n) {
            const bool exact_basis_state = basis_state_matches_problem_(*basis_state_opt, m_eq, n);
            basis_state_std =
                exact_basis_state
                    ? map_reformulated_basis_state_(*basis_state_opt, l_use, u_use, n_total,
                                                    single_y, upper_slack, split_pos, split_neg)
                    : map_reformulated_basis_seed_state_(*basis_state_opt, n_total, single_y,
                                                         upper_slack, split_pos, split_neg);
        }

        // When no warm-start basis is available, construct a logical basis from the
        // reformulation structure. Upper-bound slack columns are identity columns for their
        // rows; original rows are covered by picking the first y/y_pos with a nonzero entry.
        if (!basis_std && (!basis_state_std || basis_state_std->column_status.empty())) {
            // Build a row->first_nonzero_col map from the sparse A_std for original rows.
            std::vector<int> row_col(m_eq, -1);
            std::vector<bool> col_used(n_total, false);
            for (int j = 0; j < A_std.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_std, j); it; ++it) {
                    const int row = static_cast<int>(it.row());
                    if (row < m_eq && row_col[row] < 0 && !col_used[j] &&
                        std::abs(it.value()) > 1e-14) {
                        row_col[row] = j;
                        col_used[j] = true;
                    }
                }
            }
            std::vector<int> cand;
            cand.reserve(m_total);
            bool ok = true;
            for (int i = 0; i < m_eq; ++i) {
                if (row_col[i] < 0) { ok = false; break; }
                cand.push_back(row_col[i]);
            }
            if (ok) {
                for (int j = 0; j < n; ++j) {
                    if (map[j].upper_slack >= 0)
                        cand.push_back(map[j].upper_slack);
                }
                if ((int)cand.size() == m_total)
                    basis_std = std::move(cand);
            }
        }

        LPSolution std_sol;
        bool reformulated_retry_used = false;
        std::optional<std::vector<int>> reformulated_basis_guess = basis_std;
        if ((!reformulated_basis_guess || reformulated_basis_guess->empty()) && basis_state_std &&
            !basis_state_std->column_status.empty()) {
            reformulated_basis_guess = basis_columns_from_basis_state_(*basis_state_std, m_total);
        }
        std::optional<BasisQuality> reformulated_warm_basis_quality = std::nullopt;
        // SCIP/HiGHS insight: on a cache-hit (same A/b/c, only bounds changed), the warm basis
        // from an optimal dual-simplex solve is ALWAYS dual feasible after a bound tightening.
        // Skip the O(m^2) SparseLU quality check in this common BnB case.
        const bool skip_quality_check = cache_reuse && opt_.mode == SimplexMode::Dual &&
                                        basis_state_std.has_value() &&
                                        !basis_state_std->column_status.empty();
        if (skip_quality_check) {
            // Assume dual feasible — the dual simplex will correct any violations in O(pivots).
            BasisQuality assumed_quality;
            assumed_quality.valid = true;
            assumed_quality.dual_feasible = true;
            assumed_quality.primal_feasible = false; // conservative; dual simplex handles this
            reformulated_warm_basis_quality = assumed_quality;
        } else if (reformulated_basis_guess && !reformulated_basis_guess->empty()) {
            reformulated_warm_basis_quality =
                evaluate_basis_quality_(A_std, b_std, c_std, *reformulated_basis_guess, opt_.tol);
        }
        // Prefer dual when the mapped warm basis is dual-feasible, regardless of opt_.mode.
        // A dual-feasible warm basis makes dual simplex O(pivots) to optimality; primal
        // from the same basis often hits numerical issues on the reformulated matrix.
        const bool use_dual_first = opt_.mode != SimplexMode::Primal &&
                                    reformulated_warm_basis_quality &&
                                    reformulated_warm_basis_quality->valid &&
                                    reformulated_warm_basis_quality->dual_feasible;
        const char* reformulated_initial_mode =
            use_dual_first ? "dual"
            : (opt_.mode == SimplexMode::Primal ? "primal" : "auto");
        bool reformulated_inner_cache_used = false;
        auto solve_reformulated = [&](SimplexMode mode) {
            RevisedSimplexOptions solve_opt = opt_;
            solve_opt.mode = (mode == SimplexMode::Auto ? SimplexMode::Primal : mode);
            solve_opt.disable_presolve = true;
            trace_line_("[solve_reformulated] disable_presolve=" +
                        std::to_string(solve_opt.disable_presolve));
            // HiGHS/SCIP hot-restart: reuse the persistent reformulated solver so the
            // FTBasis factorization (and signature cache) survive across BnB node solves.
            // Created fresh only after a cache miss (when reformulated structure changed).
            if (!sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache =
                    std::make_unique<RevisedSimplex>(solve_opt);
            }
            RevisedSimplex& reformulated_solver =
                *sparse_bound_only_cache_.reformulated_solver_cache;
            reformulated_solver.opt_ = solve_opt;
            const RevisedSimplex::SparseMatrix& A_std_sparse = A_std;
            // Hot path: feed back the previous solve's optimal basis_state. Unlike
            // the no-basis internal-cache route (which loses the nonbasic bound
            // views and returns Singular), passing the full basis_state -- which
            // carries the FTBasis warm_state -- lets the solver adopt the existing
            // factorization and dual-pivot to the new optimum in a few iterations.
            auto& prev_state = sparse_bound_only_cache_.last_reformulated_basis_state;
            if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                static std::atomic<long> hit{0}, no_prev{0}, no_fact{0}, sz{0};
                if (!prev_state)
                    ++no_prev;
                else if (!reformulated_solver.has_cached_basis_factorization_state(A_std_sparse))
                    ++no_fact;
                else if (static_cast<int>(prev_state->column_status.size()) != n_total)
                    ++sz;
                else
                    ++hit;
                if (((hit + no_prev + no_fact + sz) % 200) == 0)
                    std::fprintf(stderr, "[reformpath] hit=%ld no_prev=%ld no_fact=%ld sz=%ld\n",
                                 hit.load(), no_prev.load(), no_fact.load(), sz.load());
            }
            if (prev_state &&
                reformulated_solver.has_cached_basis_factorization_state(A_std_sparse) &&
                static_cast<int>(prev_state->column_status.size()) == n_total) {
                reformulated_inner_cache_used = true;
                return reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std, u_std,
                                                 *prev_state);
            }
            return basis_state_std ? reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std,
                                                               u_std, *basis_state_std)
                                   : reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std,
                                                               u_std, basis_std);
        };
        try {
            std_sol = solve_reformulated(use_dual_first ? SimplexMode::Dual : opt_.mode);
        } catch (const std::exception& e) {
            std_sol.status = LPSolution::Status::Singular;
            std_sol.info["where"] = "bound_reformulation_recursive_solve";
            std_sol.info["what"] = e.what();
        }
        if (use_dual_first && (std_sol.status == LPSolution::Status::Singular ||
                               std_sol.status == LPSolution::Status::NeedPhase1 ||
                               std_sol.status == LPSolution::Status::IterLimit)) {
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        if (opt_.mode == SimplexMode::Dual && !use_dual_first &&
            (std_sol.status == LPSolution::Status::NeedPhase1 ||
             std_sol.status == LPSolution::Status::Singular)) {
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        if (std_sol.status == LPSolution::Status::Singular &&
            (basis_std.has_value() || basis_state_std.has_value())) {
            basis_std.reset();
            basis_state_std.reset();
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        // When skip_quality_check assumed dual feasibility without verification, an Infeasible
        // result is unreliable: if the cached basis was not truly dual-feasible, the dual simplex
        // can terminate with a spurious Farkas certificate. A wrong Infeasible propagates through
        // the BnB tree as a learned conflict, incorrectly pruning feasible nodes (false-positive
        // optimal). Verify by clearing the inner cache and re-solving cold.
        if (skip_quality_check && std_sol.status == LPSolution::Status::Infeasible) {
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            const LPSolution cold_sol = solve_reformulated(SimplexMode::Auto);
            if (cold_sol.status != LPSolution::Status::Infeasible) {
                std_sol = cold_sol;
                reformulated_retry_used = true;
            }
        }
        // Also verify a warm-started Optimal: check primal feasibility ||Ax - b||_inf.
        // A wrong Optimal (too-low LP bound from a bad warm start) causes incorrect pruning
        // in BnB, giving false-positive optimal results.
        if (skip_quality_check && std_sol.status == LPSolution::Status::Optimal &&
            std_sol.x.size() == n_total) {
            const double verify_tol = opt_.tol * 1e4;
            const Eigen::VectorXd residual = (*A_std_ptr) * std_sol.x - (*b_std_ptr);
            if (residual.lpNorm<Eigen::Infinity>() > verify_tol) {
                if (sparse_bound_only_cache_.reformulated_solver_cache) {
                    sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
                }
                const LPSolution cold_sol = solve_reformulated(SimplexMode::Auto);
                if (cold_sol.status == LPSolution::Status::Optimal) {
                    std_sol = cold_sol;
                    reformulated_retry_used = true;
                }
            }
        }

        auto reconstruct_original_x = [&]() {
            Eigen::VectorXd out =
                Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            if (std_sol.x.size() != n_total || !std_sol.x.array().isFinite().all()) {
                return out;
            }
            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    out(j) = map[j].y >= 0 ? map[j].shift + static_cast<double>(map[j].sign) *
                                                                 std_sol.x(map[j].y)
                                            : map[j].shift;
                } else {
                    out(j) = std_sol.x(map[j].y_pos) - std_sol.x(map[j].y_neg);
                }
            }
            return out;
        };

        Eigen::VectorXd x = reconstruct_original_x();
        if (std_sol.status == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, x, l_in, u_in, opt_.tol)) {
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            basis_std.reset();
            basis_state_std.reset();
            std_sol = solve_reformulated(SimplexMode::Primal);
            reformulated_retry_used = true;
            x = reconstruct_original_x();
        }

        // Persist the reformulated optimal basis_state (incl. its FTBasis
        // warm_state) so the next node's solve_reformulated can dual-restart from
        // it. Only on a clean Optimal -- a bad/partial state would force singular
        // warm restarts downstream.
        if (std_sol.status == LPSolution::Status::Optimal &&
            primal_feasible_(A_in, b_in, x, l_in, u_in, opt_.tol) &&
            !std_sol.basis_state.column_status.empty() &&
            static_cast<int>(std_sol.basis_state.column_status.size()) == n_total) {
            sparse_bound_only_cache_.last_reformulated_basis_state = std_sol.basis_state;
        } else {
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
        }

        std::vector<int> basis_out;
        std::vector<char> seen(n, 0);
        for (int idx : std_sol.basis) {
            if (static_cast<int>(basis_out.size()) == m_in)
                break;
            for (int j = 0; j < n; ++j) {
                const bool matches_single = map[j].uses_single_var && map[j].y == idx;
                const bool matches_split =
                    !map[j].uses_single_var && (map[j].y_pos == idx || map[j].y_neg == idx);
                if ((matches_single || matches_split) && !seen[j]) {
                    seen[j] = 1;
                    basis_out.push_back(j);
                    break;
                }
            }
        }

        auto info = std_sol.info;
        info["bound_reformulation"] = "1";
        info["sparse_pipeline"] = "1";
        if (reused_sparse_bound_cache) {
            info["sparse_bound_only_fast_path"] = "1";
        }
        if (reformulated_inner_cache_used) {
            info["bound_reformulation_inner_cache"] = "1";
        }
        info["bound_reformulation_initial_mode"] = reformulated_initial_mode;
        if (reformulated_warm_basis_quality) {
            info["bound_reformulation_warm_start_valid"] =
                reformulated_warm_basis_quality->valid ? "1" : "0";
            info["bound_reformulation_warm_start_primal_feasible"] =
                reformulated_warm_basis_quality->primal_feasible ? "1" : "0";
            info["bound_reformulation_warm_start_dual_feasible"] =
                reformulated_warm_basis_quality->dual_feasible ? "1" : "0";
        }
        if (reformulated_retry_used) {
            info["bound_reformulation_retry_mode"] = "auto";
        }
        const double obj =
            x.array().isFinite().all()
                ? c_in.dot(x)
                : (std::isfinite(std_sol.obj) ? (std_sol.obj + obj_shift) : std_sol.obj);
        auto sol =
            make_solution_(std_sol.status, std::move(x), obj, std::move(basis_out), std_sol.iters,
                           std::move(info), std_sol.farkas_y, std_sol.farkas_has_cert,
                           std_sol.primal_ray, std_sol.primal_ray_has_cert);
        sol.basis_state.warm_state = std_sol.basis_state.warm_state;
        sol.basis_internal = std_sol.basis_internal;
        sol.nonbasis_internal = std_sol.nonbasis_internal;
        sol.internal_column_labels = std_sol.internal_column_labels;
        sol.internal_row_labels = std_sol.internal_row_labels;
        sol.tableau = std_sol.tableau;
        sol.tableau_rhs = std_sol.tableau_rhs;
        sol.reduced_costs_internal = std_sol.reduced_costs_internal;
        if (std_sol.dual_values.size() >= m_eq) {
            sol.dual_values = std_sol.dual_values.head(m_eq);
            sol.shadow_prices = sol.dual_values;
        }
        sol.dual_values_internal = std_sol.dual_values_internal;
        sol.shadow_prices_internal = std_sol.shadow_prices_internal;
        sol.farkas_y_internal = std_sol.farkas_y_internal;
        sol.primal_ray_internal = std_sol.primal_ray_internal;
        const int outer_warm_start_attempted = solve_stats_.warm_start_attempted;
        solve_stats_ = std_sol.solve_stats;
        solve_stats_.warm_start_attempted =
            std::max(solve_stats_.warm_start_attempted, outer_warm_start_attempted);
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol, m_in));
    }

    SparseMatrix A_model = A_in;
    Eigen::VectorXd b_model = b_in;
    Eigen::VectorXd c_model = c_in;
    Eigen::VectorXd l_model = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd u_model = Eigen::VectorXd::Constant(n, presolve::inf());
    Eigen::VectorXd anchor = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd sign = Eigen::VectorXd::Ones(n);

    for (int j = 0; j < n; ++j) {
        const bool has_l = std::isfinite(l_use(j));
        const bool has_u = std::isfinite(u_use(j));
        if (!has_l && !has_u) {
            throw std::invalid_argument(
                "simplex: free variables are unsupported in solve(A,b,c,l,u)");
        }

        if (has_l) {
            anchor(j) = l_use(j);
            l_model(j) = 0.0;
            u_model(j) = has_u ? (u_use(j) - l_use(j)) : presolve::inf();
        } else {
            anchor(j) = u_use(j);
            sign(j) = -1.0;
            l_model(j) = 0.0;
            u_model(j) = presolve::inf();
            RevisedSimplexDualEngine::scale_column(A_model, j, -1.0);
            c_model(j) = -c_model(j);
        }

        if (anchor(j) != 0.0) {
            const Eigen::VectorXd model_col = A_model.col(j);
            b_model.noalias() -= model_col * anchor(j);
        }
    }

    presolve::SparsePresolveResult sparse_pres;
    if (opt_.disable_presolve) {
        sparse_pres.reduced = {A_model, b_model, c_model, l_model, u_model};
        sparse_pres.orig_col_index.resize(n);
        sparse_pres.orig_row_index.resize(m_in);
        std::iota(sparse_pres.orig_col_index.begin(), sparse_pres.orig_col_index.end(), 0);
        std::iota(sparse_pres.orig_row_index.begin(), sparse_pres.orig_row_index.end(), 0);
    } else {
        presolve::SparsePresolver::Options spopt;
        spopt.zero_tol = opt_.tol * 1e-3;
        spopt.infeas_tol = opt_.tol;
        spopt.min_delta = std::max(opt_.tol * 10.0, 1e-12);
        spopt.max_passes = (basis_state_opt && !basis_state_opt->column_status.empty()) ? 2 : 4;
        presolve::SparsePresolver sp(spopt);
        sparse_pres = sp.run({A_model, b_model, c_model, l_model, u_model});
    }
    if (sparse_pres.proven_infeasible) {
        return finalize_solution_(make_solution_(
            LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
            std::numeric_limits<double>::infinity(), {}, 0,
            {{"sparse_presolve", "infeasible"},
             {"sparse_presolve_bound_updates", std::to_string(sparse_pres.bound_updates)}}));
    }
    if (sparse_pres.proven_unbounded) {
        Eigen::VectorXd xnan =
            Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        return finalize_solution_(make_solution_(LPSolution::Status::Unbounded, xnan,
                                                 -std::numeric_limits<double>::infinity(), {}, 0,
                                                 {{"sparse_presolve", "unbounded"}}));
    }

    const SparseMatrix& Ared = sparse_pres.reduced.A;
    const Eigen::VectorXd& bred = sparse_pres.reduced.b;
    const Eigen::VectorXd& cred = sparse_pres.reduced.c;
    const Eigen::VectorXd& l_eff = sparse_pres.reduced.l;
    const Eigen::VectorXd& u_eff = sparse_pres.reduced.u;
    std::vector<int> col_orig_map(n);
    std::iota(col_orig_map.begin(), col_orig_map.end(), 0);
    std::vector<int> row_orig_map(m_in);
    std::iota(row_orig_map.begin(), row_orig_map.end(), 0);
    const std::vector<std::string> internal_column_labels =
        make_internal_column_labels_(col_orig_map);
    const std::vector<std::string> internal_row_labels = make_internal_row_labels_(row_orig_map);

    std::optional<std::vector<int>> red_basis_opt = basis_opt;
    std::optional<LPBasis> red_basis_state_opt = std::nullopt;
    if (basis_state_opt && !basis_state_opt->column_status.empty()) {
        red_basis_state_opt =
            map_reduced_basis_state_(*basis_state_opt, col_orig_map, l_eff, u_eff, opt_.tol);
        if (!red_basis_opt || red_basis_opt->empty()) {
            red_basis_opt = basis_columns_from_basis_state_(*red_basis_state_opt, m_in);
        }
    }

    auto t1_presolve = std::chrono::steady_clock::now();
    current_timing_.presolve_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve - t0_presolve).count();

    std::optional<std::vector<int>> crash_seed_basis_opt = red_basis_opt;
    if ((!crash_seed_basis_opt || crash_seed_basis_opt->empty()) && red_basis_state_opt &&
        !red_basis_state_opt->column_status.empty()) {
        auto partial_seed = basis_columns_from_basis_state_(*red_basis_state_opt, -1);
        if (partial_seed && !partial_seed->empty()) {
            crash_seed_basis_opt = std::move(partial_seed);
        }
    }

    // evaluate_basis_quality_ checks dual feasibility assuming all non-basics
    // are at lower bound (standard form). For problems with finite upper bounds,
    // non-basics at their upper bound have negative reduced costs which the
    // check incorrectly flags as violations. When Dual mode is requested and
    // a warm-start with column_status is available, override choose_initial_basis_
    // to use the warm-start directly — column_status encodes AtLower/AtUpper
    // correctly so the dual simplex can set views from it.
    const bool has_warm_column_status_for_dual =
        opt_.mode == SimplexMode::Dual && crash_seed_basis_opt.has_value() &&
        static_cast<int>(crash_seed_basis_opt->size()) == m_in && red_basis_state_opt &&
        !red_basis_state_opt->column_status.empty() &&
        static_cast<int>(red_basis_state_opt->column_status.size()) == n;

    const bool seed_basis_from_state = (!basis_opt || basis_opt->empty()) && red_basis_state_opt &&
                                       !red_basis_state_opt->column_status.empty();
    const bool allow_direct_warm_start =
        !seed_basis_from_state ||
        (crash_seed_basis_opt && static_cast<int>(crash_seed_basis_opt->size()) == m_in);
    auto t0_crash = std::chrono::steady_clock::now();
    CrashSelection basis_choice = choose_initial_basis_(
        Ared, bred, cred, opt_, crash_seed_basis_opt, allow_direct_warm_start);
    if (factorized_basis_seed_opt && crash_seed_basis_opt &&
        static_cast<int>(crash_seed_basis_opt->size()) == m_in &&
        basis_choice.basis != *crash_seed_basis_opt) {
        basis_choice.basis = *crash_seed_basis_opt;
        basis_choice.source = "warm_start";
        basis_choice.style = "factorization_reuse";
        basis_choice.quality.valid = true;
        basis_choice.quality.dual_feasible = true;
        basis_choice.quality.primal_feasible = false;
    }
    auto t1_crash = std::chrono::steady_clock::now();
    current_timing_.crash_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_crash - t0_crash).count();

    std::vector<int> basis_guess = basis_choice.basis;
    const bool basis_guess_from_warm_start =
        (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start");
    const bool basis_valid = ((int)basis_guess.size() == m_in) && basis_choice.quality.valid;
    const bool allow_direct_primal =
        basis_valid && (basis_choice.quality.primal_feasible || basis_guess_from_warm_start);
    const bool warm_has_column_status =
        red_basis_state_opt && !red_basis_state_opt->column_status.empty() &&
        static_cast<int>(red_basis_state_opt->column_status.size()) == n;
    const bool allow_direct_dual =
        basis_valid &&
        (basis_choice.quality.dual_feasible ||
         (opt_.mode == SimplexMode::Dual && basis_guess_from_warm_start && warm_has_column_status));
    if (std::getenv("SIMPLINHO_TRACE_REFORM") && opt_.mode == SimplexMode::Dual &&
        !allow_direct_dual) {
        std::fprintf(stderr,
                     "[reformdispatch] dual-warm-gate fail source=%s valid=%d dual_feasible=%d "
                     "primal_feasible=%d basis_guess_from_warm_start=%d warm_has_column_status=%d "
                     "basis_size=%d m_in=%d factorized_basis_seed_opt=%d\n",
                     basis_choice.source.c_str(), static_cast<int>(basis_choice.quality.valid),
                     static_cast<int>(basis_choice.quality.dual_feasible),
                     static_cast<int>(basis_choice.quality.primal_feasible),
                     static_cast<int>(basis_guess_from_warm_start),
                     static_cast<int>(warm_has_column_status), static_cast<int>(basis_guess.size()),
                     m_in, static_cast<int>(factorized_basis_seed_opt.has_value()));
    }

    auto add_sparse_info = [&](std::unordered_map<std::string, std::string> info) {
        info["sparse_pipeline"] = "1";
        info["sparse_presolve"] = opt_.disable_presolve ? "disabled" : "sparse";
        info["sparse_presolve_passes"] = std::to_string(sparse_pres.passes);
        info["sparse_presolve_bound_updates"] = std::to_string(sparse_pres.bound_updates);
        info["sparse_presolve_singleton_rows"] = std::to_string(sparse_pres.singleton_rows);
        info["sparse_presolve_zero_rows"] = std::to_string(sparse_pres.zero_rows);
        info["sparse_presolve_zero_columns"] = std::to_string(sparse_pres.zero_columns);
        info["original_m"] = std::to_string(m_in);
        info["reduced_m"] = std::to_string(m_in);
        info["reduced_n"] = std::to_string(n);
        if (!basis_choice.source.empty() && basis_choice.source != "none") {
            info["basis_start"] = basis_choice.source;
            info["basis_start_style"] = basis_choice.style;
        }
        return info;
    };

    auto finalize_sparse_solution = [&](LPSolution::Status status, const Eigen::VectorXd& z,
                                        const std::vector<int>& red_basis, int iters,
                                        std::unordered_map<std::string, std::string> info) {
        if (z.size() != sign.size()) {
            info["where"] = "finalize_sparse_solution";
            info["reason"] = "invalid_primal_dimension";
            info["expected_primal_dim"] = std::to_string(sign.size());
            info["actual_primal_dim"] = std::to_string(z.size());
            return finalize_solution_(make_solution_(
                status, Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN()),
                std::numeric_limits<double>::quiet_NaN(), {}, iters,
                add_sparse_info(std::move(info))));
        }
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z);
        std::vector<int> basis_full = red_basis;
        const bool has_primal_ray =
            info.count("primal_ray_has_cert") && info.at("primal_ray_has_cert") == "1";
        const auto primal_ray_internal =
            has_primal_ray ? parse_serialized_vec_(info, "primal_ray", n) : std::nullopt;
        double obj = x_full.array().isFinite().all() ? c_in.dot(x_full)
                                                     : std::numeric_limits<double>::quiet_NaN();
        if (status == LPSolution::Status::Unbounded) {
            obj = -std::numeric_limits<double>::infinity();
        }
        auto sol =
            make_solution_(status, x_full, obj, basis_full, iters, add_sparse_info(std::move(info)),
                           std::nullopt, std::nullopt, primal_ray_internal, has_primal_ray);
        sol = attach_internal_tableau_(std::move(sol), Ared, bred, cred, red_basis,
                                       internal_column_labels, internal_row_labels, opt_.tol,
                                       opt_.compute_tableau, opt_.compute_reduced_costs);
        if (sol.dual_values_internal.size() == m_in) {
            sol.dual_values = sol.dual_values_internal;
            sol.shadow_prices = sol.dual_values;
        }
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
    };

    auto run_phase2_p = [&](std::optional<std::vector<int>> b) {
        auto t0 = std::chrono::steady_clock::now();
        try {
            auto res = phase_(Ared, bred, cred, std::move(b), l_eff, u_eff);
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            return res;
        } catch (const std::runtime_error& e) {
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            return PhaseResult{LPSolution::Status::Singular,
                               Eigen::VectorXd{},
                               {},
                               0,
                               {{"reason", "basis_factorization_failure"},
                                {"what", e.what()},
                                {"where", "sparse_direct_primal"}}};
        }
    };
    auto run_phase2_d = [&](std::optional<std::vector<int>> b) {
        auto t0 = std::chrono::steady_clock::now();
        const bool use_warm_status =
            red_basis_state_opt &&
            (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start") &&
            red_basis_state_opt->column_status.size() == static_cast<std::size_t>(n);
        try {
            auto res = dual_phase_(Ared, bred, cred, std::move(b), l_eff, u_eff,
                                   use_warm_status ? std::optional<std::vector<LPBasisStatus>>(
                                                         red_basis_state_opt->column_status)
                                                   : std::nullopt);
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            return res;
        } catch (const std::runtime_error& e) {
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            return PhaseResult{LPSolution::Status::Singular,
                               Eigen::VectorXd{},
                               {},
                               0,
                               {{"reason", "basis_factorization_failure"},
                                {"what", e.what()},
                                {"where", "sparse_direct_dual"}}};
        }
    };

    if (allow_direct_primal || allow_direct_dual) {
        if (basis_guess_from_warm_start) {
            solve_stats_.warm_start_accepted = 1;
        }
        if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
            static std::atomic<long> trace_reform_dispatch{0};
            static std::atomic<long> trace_reform_primal{0};
            static std::atomic<long> trace_reform_dual{0};
            static std::atomic<long> trace_reform_auto_dual{0};
            const long count = ++trace_reform_dispatch;
            if ((count % 200) == 0) {
                std::fprintf(stderr,
                             "[reformdispatch] count=%ld mode=%d allow_primal=%d allow_dual=%d "
                             "basis_guess_from_warm_start=%d allow_direct_dual=%d\n",
                             count, static_cast<int>(opt_.mode),
                             static_cast<int>(allow_direct_primal),
                             static_cast<int>(allow_direct_dual),
                             static_cast<int>(basis_guess_from_warm_start),
                             static_cast<int>(allow_direct_dual));
            }
        }
        LPSolution::Status st = LPSolution::Status::NeedPhase1;
        Eigen::VectorXd v2;
        std::vector<int> red_basis2;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if (opt_.mode == SimplexMode::Dual) {
            if (allow_direct_dual) {
                if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                    static std::atomic<long> dual_branch_count{0};
                    const long c = ++dual_branch_count;
                    if ((c % 100) == 0)
                        std::fprintf(stderr,
                                     "[reformdispatch] dual-branch direct run_phase2_d count=%ld\n",
                                     c);
                }
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_d(basis_guess);
            } else if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                static std::atomic<long> dual_branch_skipped{0};
                const long c = ++dual_branch_skipped;
                if ((c % 100) == 0)
                    std::fprintf(
                        stderr,
                        "[reformdispatch] dual-branch skipped allow_direct_dual=false count=%ld\n",
                        c);
            }
        } else if (opt_.mode == SimplexMode::Primal) {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_p(basis_guess);
            }
        } else {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_p(basis_guess);
            }
            if (allow_direct_dual && st == LPSolution::Status::NeedPhase1 &&
                info2.count("reason") && info2.at("reason") == std::string("negative_basic_vars")) {
                if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                    static std::atomic<long> auto_dual_branch_count{0};
                    const long c = ++auto_dual_branch_count;
                    if ((c % 100) == 0)
                        std::fprintf(stderr,
                                     "[reformdispatch] auto fallback dual run_phase2_d count=%ld\n",
                                     c);
                }
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_d(basis_guess);
            }
        }

        if (st == LPSolution::Status::Optimal) {
            const Eigen::VectorXd x_full = anchor + sign.cwiseProduct(v2);
            if (primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
                return finalize_sparse_solution(st, v2, red_basis2, it2, std::move(info2));
            }
            info2["reason"] = "invalid_returned_primal";
        } else if (st == LPSolution::Status::Unbounded || st == LPSolution::Status::IterLimit ||
                   st == LPSolution::Status::ObjectiveBound ||
                   (st == LPSolution::Status::Infeasible && !basis_guess_from_warm_start)) {
            return finalize_sparse_solution(st, v2, red_basis2, it2, std::move(info2));
        }
    }

    auto t1_presolve2 = std::chrono::steady_clock::now();
    current_timing_.presolve_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve2 - t1_crash).count();

    auto [A1, b1, c1, basis1, n_orig_eff, m_rows] = make_phase1_(Ared, bred);
    PhaseResult phase1_result;
    try {
        phase1_result = phase_(A1, b1, c1, basis1, Eigen::VectorXd::Zero(A1.cols()),
                               Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));
    } catch (const std::runtime_error& e) {
        if (!is_recoverable_basis_runtime_(e.what()))
            throw;
        phase1_result = PhaseResult{LPSolution::Status::Singular,
                                    Eigen::VectorXd{},
                                    {},
                                    0,
                                    {{"reason", "basis_factorization_failure"},
                                     {"what", e.what()},
                                     {"where", "sparse_phase1_primal"}}};
    }
    auto [status1, v1, basis1_out, it1, info1] = std::move(phase1_result);
    if (status1 == LPSolution::Status::NeedPhase1 && info1.count("reason") &&
        info1.at("reason") == std::string("negative_basic_vars")) {
        try {
            std::tie(status1, v1, basis1_out, it1, info1) =
                dual_phase_(A1, b1, c1, basis1_out.empty() ? basis1 : basis1_out,
                            Eigen::VectorXd::Zero(A1.cols()),
                            Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));
        } catch (const std::runtime_error& e) {
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            status1 = LPSolution::Status::Singular;
            v1.resize(0);
            basis1_out.clear();
            it1 = 0;
            info1 = {{"reason", "basis_factorization_failure"},
                     {"what", e.what()},
                     {"where", "sparse_phase1_dual"}};
        }
    }
    if (status1 == LPSolution::Status::Singular || status1 == LPSolution::Status::IterLimit) {
        auto info = add_sparse_info(std::move(info1));
        info["phase1_status"] = to_string(status1);
        return finalize_solution_(make_solution_(status1, Eigen::VectorXd::Zero(n),
                                                 std::numeric_limits<double>::quiet_NaN(), {}, it1,
                                                 std::move(info)));
    }
    if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
        auto info = add_sparse_info({{"phase1_status", to_string(status1)}});
        return finalize_solution_(
            make_solution_(LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                           std::numeric_limits<double>::infinity(), {}, it1, std::move(info)));
    }

    std::vector<int> red_basis2;
    red_basis2.reserve(m_rows);
    for (int j : basis1_out)
        if (j < (int)n_orig_eff)
            red_basis2.push_back(j);

    if ((int)red_basis2.size() < m_rows) {
        std::vector<int> fallback_basis = red_basis2;
        for (int j = 0; j < (int)n_orig_eff; ++j) {
            if ((int)red_basis2.size() == m_rows)
                break;
            if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end())
                continue;
            std::vector<int> cand = red_basis2;
            cand.push_back(j);
            if ((int)cand.size() > m_rows)
                continue;
            if (!sparse_basis_has_full_rank_(Ared, cand))
                continue;
            if (basis_is_primal_feasible_(Ared, bred, cand, opt_.tol)) {
                red_basis2 = std::move(cand);
                continue;
            }
            if ((int)fallback_basis.size() < (int)cand.size()) {
                fallback_basis = cand;
            }
        }
        if ((int)red_basis2.size() < m_rows && (int)fallback_basis.size() == m_rows) {
            red_basis2 = std::move(fallback_basis);
        }
    }
    if ((int)red_basis2.size() == m_rows) {
        const BasisQuality phase2_start_quality =
            evaluate_basis_quality_(Ared, bred, cred, red_basis2, opt_.tol);
        const double solve_residual_guard = std::max(1e-7, 100.0 * opt_.tol);
        if (!phase2_start_quality.valid || !std::isfinite(phase2_start_quality.solve_residual) ||
            phase2_start_quality.solve_residual > solve_residual_guard) {
            const CrashSelection repaired_phase2_start =
                choose_initial_basis_(Ared, bred, cred, opt_, red_basis2);
            if (repaired_phase2_start.quality.valid &&
                better_basis_quality_(
                    repaired_phase2_start,
                    CrashSelection{red_basis2, phase2_start_quality, "phase1_basis", "phase1", -1},
                    opt_.mode)) {
                red_basis2 = repaired_phase2_start.basis;
            }
        }
    }

    LPSolution::Status status2;
    Eigen::VectorXd v2;
    std::vector<int> red_basis_out;
    int it2 = 0;
    std::unordered_map<std::string, std::string> info2;
    auto phase2_result_needs_basis_repair = [&](LPSolution::Status status,
                                                const Eigen::VectorXd& primal) {
        return status == LPSolution::Status::Singular || status == LPSolution::Status::NeedPhase1 ||
               (status == LPSolution::Status::Optimal && primal.size() != n) ||
               (status == LPSolution::Status::Unbounded && primal.size() != n) ||
               (status == LPSolution::Status::IterLimit && primal.size() != 0 &&
                primal.size() != n);
    };
    auto run_sparse_phase2_from_basis = [&](const std::vector<int>& basis) {
        if (opt_.mode == SimplexMode::Dual) {
            const auto phase2_basis_quality =
                evaluate_basis_quality_(Ared, bred, cred, basis, opt_.tol);
            if (phase2_basis_quality.valid && phase2_basis_quality.dual_feasible) {
                auto res = run_phase2_d(basis);
                std::get<4>(res)["phase2_mode"] = "dual";
                return res;
            }
            auto res = run_phase2_p(basis);
            std::get<4>(res)["phase2_mode"] = "primal";
            std::get<4>(res)["phase2_dual_requested_but_basis_not_dual_feasible"] = "1";
            return res;
        }
        if (opt_.mode == SimplexMode::Primal) {
            auto res = run_phase2_p(basis);
            std::get<4>(res)["phase2_mode"] = "primal";
            return res;
        }
        auto res = run_phase2_p(basis);
        std::get<4>(res)["phase2_mode"] = "primal";
        if (std::get<0>(res) == LPSolution::Status::NeedPhase1 &&
            std::get<4>(res).count("reason") &&
            std::get<4>(res).at("reason") == std::string("negative_basic_vars")) {
            res = run_phase2_d(basis);
            std::get<4>(res)["phase2_mode"] = "dual";
        }
        return res;
    };

    if ((int)red_basis2.size() == m_rows) {
        std::tie(status2, v2, red_basis_out, it2, info2) = run_sparse_phase2_from_basis(red_basis2);
    } else {
        std::tie(status2, v2, red_basis_out, it2, info2) = run_phase2_p(std::nullopt);
    }

    if (phase2_result_needs_basis_repair(status2, v2)) {
        const std::optional<std::vector<int>> seeded_repair_basis =
            ((int)red_basis2.size() == m_rows) ? std::optional<std::vector<int>>(red_basis2)
                                               : std::nullopt;
        const CrashSelection repaired_seed =
            choose_initial_basis_(Ared, bred, cred, opt_, seeded_repair_basis);
        const CrashSelection repaired_cold =
            choose_initial_basis_(Ared, bred, cred, opt_, std::nullopt);

        const CrashSelection* repaired = nullptr;
        if (repaired_seed.quality.valid &&
            (!repaired_cold.quality.valid ||
             better_basis_quality_(repaired_seed, repaired_cold, opt_.mode))) {
            repaired = &repaired_seed;
        } else if (repaired_cold.quality.valid) {
            repaired = &repaired_cold;
        }

        if (repaired && (int)repaired->basis.size() == m_rows) {
            LPSolution::Status repaired_status;
            Eigen::VectorXd repaired_v;
            std::vector<int> repaired_basis_out;
            int repaired_iters = 0;
            std::unordered_map<std::string, std::string> repaired_info;
            std::tie(repaired_status, repaired_v, repaired_basis_out, repaired_iters,
                     repaired_info) = run_sparse_phase2_from_basis(repaired->basis);
            if (!phase2_result_needs_basis_repair(repaired_status, repaired_v) ||
                (repaired_status == LPSolution::Status::Optimal && repaired_v.size() == n)) {
                status2 = repaired_status;
                v2 = std::move(repaired_v);
                red_basis_out = std::move(repaired_basis_out);
                it2 += repaired_iters;
                repaired_info["phase2_basis_repair"] = "1";
                repaired_info["phase2_basis_repair_source"] = repaired->source;
                repaired_info["phase2_basis_repair_style"] = repaired->style;
                repaired_info["phase2_basis_repair_attempt"] = std::to_string(repaired->attempt);
                info2 = std::move(repaired_info);
            } else {
                info2["phase2_basis_repair_attempted"] = "1";
                info2["phase2_basis_repair_source"] = repaired->source;
                info2["phase2_basis_repair_style"] = repaired->style;
            }
        }
    }

    return finalize_sparse_solution(status2, v2, red_basis_out, it1 + it2, std::move(info2));
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::dual_phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexDualEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                         std::move(warm_status));
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::phase_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u) {
    return RevisedSimplexPrimalEngine::run(*this, A, b, c, std::move(basis_opt), l, u);
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::dual_phase_(const SparseMatrix& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexDualEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                         std::move(warm_status));
}
