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
                           const LPBasis* basis_state_opt);
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

    struct RowRankReduction {
        bool needed = false;
        bool inconsistent = false;
        int original_rows = 0;
        int rank = 0;
        std::vector<int> keep_rows;
    };

    static RowRankReduction dependent_row_reduction_(const Eigen::MatrixXd& A,
                                                     const Eigen::VectorXd& b, double tol) {
        RowRankReduction out;
        out.original_rows = static_cast<int>(A.rows());
        if (A.rows() == 0)
            return out;

        const double threshold = std::max(1e-12, tol * 100.0);
        Eigen::FullPivLU<Eigen::MatrixXd> rank_lu(A);
        rank_lu.setThreshold(threshold);
        out.rank = rank_lu.rank();
        if (out.rank >= A.rows())
            return out;
        out.needed = true;

        Eigen::MatrixXd augmented(A.rows(), A.cols() + 1);
        augmented.leftCols(A.cols()) = A;
        augmented.col(A.cols()) = b;
        Eigen::FullPivLU<Eigen::MatrixXd> aug_lu(augmented);
        aug_lu.setThreshold(threshold);
        if (aug_lu.rank() > out.rank) {
            out.inconsistent = true;
            return out;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> row_qr(A.transpose());
        row_qr.setThreshold(threshold);
        const int qr_rank = std::min<int>(row_qr.rank(), out.rank);
        const Eigen::VectorXi perm = row_qr.colsPermutation().indices();
        out.keep_rows.reserve(qr_rank);
        for (int k = 0; k < qr_rank; ++k)
            out.keep_rows.push_back(perm(k));
        std::sort(out.keep_rows.begin(), out.keep_rows.end());
        return out;
    }

    static Eigen::MatrixXd select_dense_rows_(const Eigen::MatrixXd& A,
                                              const std::vector<int>& rows) {
        Eigen::MatrixXd out(rows.size(), A.cols());
        for (int ir = 0; ir < static_cast<int>(rows.size()); ++ir)
            out.row(ir) = A.row(rows[ir]);
        return out;
    }

    static Eigen::VectorXd select_vector_rows_(const Eigen::VectorXd& b,
                                               const std::vector<int>& rows) {
        Eigen::VectorXd out(rows.size());
        for (int ir = 0; ir < static_cast<int>(rows.size()); ++ir)
            out(ir) = b(rows[ir]);
        return out;
    }

    static SparseMatrix select_sparse_rows_(const SparseMatrix& A, const std::vector<int>& rows) {
        std::vector<int> old_to_new(A.rows(), -1);
        for (int ir = 0; ir < static_cast<int>(rows.size()); ++ir)
            old_to_new[rows[ir]] = ir;
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(static_cast<std::size_t>(A.nonZeros()));
        for (int j = 0; j < A.outerSize(); ++j) {
            for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
                const int nr = old_to_new[it.row()];
                if (nr >= 0)
                    trips.emplace_back(nr, j, it.value());
            }
        }
        SparseMatrix out(static_cast<int>(rows.size()), A.cols());
        if (!trips.empty())
            out.setFromTriplets(trips.begin(), trips.end());
        out.makeCompressed();
        return out;
    }

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
        bopt.ext_ft_update_counter = &solve_stats_.ft_updates;
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
                                                const std::vector<int>& basis,
                                                const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                                double tol);

    static BasisQuality evaluate_basis_quality_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                const Eigen::VectorXd& c,
                                                const std::vector<int>& basis,
                                                const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                                double tol);

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
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    improve_basis_by_swaps_(const SparseMatrix& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::vector<int> basis,
                            const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    build_basis_attempt_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                         const Eigen::VectorXd& c, const CrashAttemptConfig& cfg, double tol,
                         SimplexMode mode, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                         std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::vector<int>
    build_basis_attempt_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                         const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                         const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                         std::optional<std::vector<int>> seed_basis = std::nullopt);

    static CrashSelection
    choose_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                          const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                          const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                          std::optional<std::vector<int>> seed_basis = std::nullopt,
                          bool allow_direct_warm_start = true);

    static CrashSelection
    choose_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                          const RevisedSimplexOptions& opt, const Eigen::VectorXd& l,
                          const Eigen::VectorXd& u,
                          std::optional<std::vector<int>> seed_basis = std::nullopt,
                          bool allow_direct_warm_start = true);

    static std::optional<std::vector<int>>
    find_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                        const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                        std::optional<std::vector<int>> seed_basis = std::nullopt);

    static std::optional<std::vector<int>>
    find_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                        const RevisedSimplexOptions& opt, const Eigen::VectorXd& l,
                        const Eigen::VectorXd& u,
                        std::optional<std::vector<int>> seed_basis = std::nullopt);

    static bool basis_is_primal_feasible_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                          const std::vector<int>& basis, const Eigen::VectorXd& l,
                                          const Eigen::VectorXd& u, double tol);

    static bool basis_is_primal_feasible_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                          const std::vector<int>& basis, const Eigen::VectorXd& l,
                                          const Eigen::VectorXd& u, double tol);

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd, std::vector<int>,
                      std::size_t, int>
    make_phase1_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

    static std::tuple<SparseMatrix, Eigen::VectorXd, Eigen::VectorXd, std::vector<int>, std::size_t,
                      int>
    make_phase1_(const SparseMatrix& A, const Eigen::VectorXd& b);

    // --------------------------- PRIMAL PHASE ---------------------------
    PhaseResult phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u,
                       std::optional<std::vector<LPBasisStatus>> warm_status = std::nullopt);

    PhaseResult phase_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u,
                       std::optional<std::vector<LPBasisStatus>> warm_status = std::nullopt);

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

#include "simplex/engine/simplex_dense_impl.h"
#include "simplex/engine/simplex_sparse_impl.h"
