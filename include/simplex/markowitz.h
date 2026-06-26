#pragma once

#include "../../extern/pdqsort/pdqsort.h"
#include <Eigen/Dense>
#include <Eigen/LU> // for FullPivLU
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

// ======================================================
// MarkowitzLU — dense Markowitz LU with rook fallback
//
// Main improvements:
//
// 1. Removes duplicate Schur-complement updates.
//    The trailing block is updated exactly once in factorize_().
//
// 2. update_tracking_() is now metadata-only.
//    It refreshes row/column degrees and column maxima without
//    touching U_ again.
//
// 3. Lazy transpose fallback.
//    fallback_lu_t_ is only computed if solveT() actually needs it.
//
// 4. Reusable work buffers for solve/refinement/permutations.
//    Reduces repeated allocations on hot solve paths.
//
// 5. Pivot search is slightly more selective.
//    Early singleton checks, row-degree-biased sampling, and
//    metadata-aware acceptance.
//
// Interface preserved:
//   - L() and U() remain full dense n x n matrices.
//   - supports_inplace_updates() remains tied to non-fallback path.
// ======================================================

class MarkowitzLU {
  public:
    MarkowitzLU() = default;

    MarkowitzLU(const Eigen::MatrixXd& A, double pivot_rel = 1e-12, double abs_floor = 1e-16,
                int rook_iters = 2, bool use_threshold_piv = false)
        : use_threshold_piv_(use_threshold_piv) {
        factor(A, pivot_rel, abs_floor, rook_iters);
    }

    void factor(const Eigen::MatrixXd& A, double pivot_rel = 1e-12, double abs_floor = 1e-16,
                int rook_iters = 2) {
        if (A.rows() != A.cols() || A.rows() == 0) {
            throw std::invalid_argument("MarkowitzLU: square non-empty matrix required");
        }

        n_ = static_cast<int>(A.rows());
        pivot_rel_ = pivot_rel;
        abs_floor_ = abs_floor;
        rook_iters_ = std::max(1, rook_iters);
        use_fallback_full_piv_ = false;
        fallback_lu_t_ready_ = false;

        L_.setZero(n_, n_);
        U_ = A;

        A_orig_ = A;

        Pr_.resize(n_);
        Pc_.resize(n_);
        std::ranges::iota(Pr_, 0);
        std::ranges::iota(Pc_, 0);

        init_tracking_();
        init_workspaces_();

        try {
            factorize_();
        } catch (const std::runtime_error&) {
            fallback_lu_.compute(A);
            if (!fallback_lu_.isInvertible() || fallback_lu_.rank() != n_) {
                throw std::runtime_error("MarkowitzLU: matrix singular");
            }
            use_fallback_full_piv_ = true;
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        if (b.size() != n_) {
            throw std::invalid_argument("solve: size mismatch");
        }

        if (use_fallback_full_piv_) {
            return fallback_lu_.solve(b);
        }

        // w = P_r^T b, stored as b(Pr_[i])
        apply_perm_inplace_(b, Pr_, perm_buf1_);

        // Solve L z = P_r^T b
        triangular_solve_(L_, TriangularSolveMode::UnitLower, perm_buf1_);

        // Solve U w = z
        triangular_solve_(U_, TriangularSolveMode::Upper, perm_buf1_);

        // Optional iterative refinement in factorized coordinates
        iterative_refine_(perm_buf1_, b, /*is_transpose=*/false);

        // x = P_c * w
        apply_perm_inplace_(perm_buf1_, Pc_, perm_buf2_);
        return perm_buf2_;
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd& c) const {
        if (c.size() != n_) {
            throw std::invalid_argument("solveT: size mismatch");
        }

        if (use_fallback_full_piv_) {
            ensure_fallback_transpose_ready_();
            return fallback_lu_t_.solve(c);
        }

        // s = P_c^T c
        apply_perm_inplace_(c, Pc_, perm_buf1_);

        // Solve U^T y = P_c^T c
        triangular_solve_(U_, TriangularSolveMode::TransposeUpper, perm_buf1_);

        // Solve L^T s = y
        triangular_solve_(L_, TriangularSolveMode::TransposeUnitLower, perm_buf1_);

        iterative_refine_(perm_buf1_, c, /*is_transpose=*/true);

        // x = P_r^{-T} s = inverse row permutation
        apply_inv_perm_inplace_(perm_buf1_, Pr_, perm_buf2_);
        return perm_buf2_;
    }

    int n() const noexcept { return n_; }
    bool supports_inplace_updates() const noexcept { return !use_fallback_full_piv_; }

    Eigen::MatrixXd& L() { return L_; }
    Eigen::MatrixXd& U() { return U_; }
    const Eigen::MatrixXd& L() const { return L_; }
    const Eigen::MatrixXd& U() const { return U_; }

  private:
    enum class TriangularSolveMode {
        UnitLower,
        Upper,
        TransposeUpper,
        TransposeUnitLower,
    };

    static constexpr int kMaxRefine_ = 4;
    static constexpr double kRefineTol_ = 1e-14;
    static constexpr long kEarlyScore_ = 1;
    static constexpr double kEarlyRatio_ = 0.92;

    // --------------------------------------------------
    // Workspace
    // --------------------------------------------------
    void init_workspaces_() const {
        perm_buf1_.resize(n_);
        perm_buf2_.resize(n_);
        rhs_buf_.resize(n_);
        residual_buf_.resize(n_);
        corr_buf_.resize(n_);
    }

    // --------------------------------------------------
    // Permutations
    // --------------------------------------------------
    void apply_perm_inplace_(const Eigen::VectorXd& in, const std::vector<int>& perm,
                             Eigen::VectorXd& out) const {
        out.resize(n_);
        for (int i = 0; i < n_; ++i) {
            out(i) = in(perm[i]);
        }
    }

    void apply_inv_perm_inplace_(const Eigen::VectorXd& in, const std::vector<int>& perm,
                                 Eigen::VectorXd& out) const {
        out.resize(n_);
        for (int i = 0; i < n_; ++i) {
            out(perm[i]) = in(i);
        }
    }

    // --------------------------------------------------
    // Triangular solve
    // --------------------------------------------------
    void triangular_solve_(const Eigen::MatrixXd& mat, TriangularSolveMode mode,
                           Eigen::VectorXd& x) const {
        switch (mode) {
            case TriangularSolveMode::UnitLower:
                mat.template triangularView<Eigen::UnitLower>().solveInPlace(x);
                return;
            case TriangularSolveMode::Upper:
                mat.template triangularView<Eigen::Upper>().solveInPlace(x);
                return;
            case TriangularSolveMode::TransposeUpper:
                mat.transpose().template triangularView<Eigen::Lower>().solveInPlace(x);
                return;
            case TriangularSolveMode::TransposeUnitLower:
                mat.transpose().template triangularView<Eigen::UnitUpper>().solveInPlace(x);
                return;
        }
        throw std::logic_error("triangular_solve_: unsupported mode");
    }

    // --------------------------------------------------
    // Iterative refinement
    // --------------------------------------------------
    void iterative_refine_(Eigen::VectorXd& x, const Eigen::VectorXd& rhs_orig,
                           bool is_transpose) const {
        if (A_orig_.rows() != n_ || A_orig_.cols() != n_) {
            return;
        }

        const double rhs_norm = std::max(1.0, rhs_orig.lpNorm<Eigen::Infinity>());
        if (!std::isfinite(rhs_norm)) {
            return;
        }

        for (int it = 0; it < kMaxRefine_; ++it) {
            Eigen::VectorXd x_full;
            if (!is_transpose) {
                apply_perm_inplace_(x, Pc_, perm_buf2_);
                x_full = perm_buf2_;
                residual_buf_.noalias() = rhs_orig - A_orig_ * x_full;
            } else {
                apply_inv_perm_inplace_(x, Pr_, perm_buf2_);
                x_full = perm_buf2_;
                residual_buf_.noalias() = rhs_orig - A_orig_.transpose() * x_full;
            }

            const double berr = residual_buf_.lpNorm<Eigen::Infinity>() / rhs_norm;
            if (!std::isfinite(berr) || berr < kRefineTol_) {
                break;
            }

            // Solve correction in the original coordinate system but through
            // the current factorization coordinates.
            // Need: L*U*corr = P_r * residual
            if (!is_transpose) {
                apply_inv_perm_inplace_(residual_buf_, Pr_, corr_buf_);
                triangular_solve_(L_, TriangularSolveMode::UnitLower, corr_buf_);
                triangular_solve_(U_, TriangularSolveMode::Upper, corr_buf_);
            } else {
                apply_inv_perm_inplace_(residual_buf_, Pc_, corr_buf_);
                triangular_solve_(U_, TriangularSolveMode::TransposeUpper, corr_buf_);
                triangular_solve_(L_, TriangularSolveMode::TransposeUnitLower, corr_buf_);
            }

            if (!corr_buf_.array().isFinite().all() ||
                corr_buf_.lpNorm<Eigen::Infinity>() < 1e-16) {
                break;
            }

            x += corr_buf_;
        }
    }

    // --------------------------------------------------
    // Swaps
    // --------------------------------------------------
    void swap_rows_(int a, int b) {
        if (a == b) {
            return;
        }

        U_.row(a).swap(U_.row(b));

        if (a > 0 || b > 0) {
            const int upto = std::min(a, b);
            if (upto > 0) {
                L_.row(a).head(upto).swap(L_.row(b).head(upto));
            }
            if (a != b) {
                if (a > b) {
                    if (b < a) {
                        const double tmp = L_(a, b);
                        L_(a, b) = L_(b, b);
                        L_(b, b) = tmp;
                    }
                } else {
                    if (a < b) {
                        const double tmp = L_(a, a);
                        L_(a, a) = L_(b, a);
                        L_(b, a) = tmp;
                    }
                }
            }
        }

        std::swap(row_deg_[a], row_deg_[b]);
        std::swap(Pr_[a], Pr_[b]);
    }

    void swap_cols_(int a, int b) {
        if (a == b) {
            return;
        }

        U_.col(a).swap(U_.col(b));
        std::swap(col_deg_[a], col_deg_[b]);
        std::swap(col_max_[a], col_max_[b]);
        std::swap(col_max_dirty_[a], col_max_dirty_[b]);
        std::swap(Pc_[a], Pc_[b]);
    }

    // --------------------------------------------------
    // Tracking initialization
    // --------------------------------------------------
    void init_tracking_() {
        row_deg_.assign(n_, 0);
        col_deg_.assign(n_, 0);
        col_max_.assign(n_, 0.0);
        col_max_dirty_.assign(n_, false);

        for (int i = 0; i < n_; ++i) {
            int cnt = 0;
            for (int j = 0; j < n_; ++j) {
                if (std::abs(U_(i, j)) > abs_floor_) {
                    ++cnt;
                }
            }
            row_deg_[i] = cnt;
        }

        for (int j = 0; j < n_; ++j) {
            int cnt = 0;
            double mx = 0.0;
            for (int i = 0; i < n_; ++i) {
                const double a = std::abs(U_(i, j));
                if (a > abs_floor_) {
                    ++cnt;
                }
                mx = std::max(mx, a);
            }
            col_deg_[j] = cnt;
            col_max_[j] = mx;
        }
    }

    double col_max_active_(int j, int k) const {
        if (!col_max_dirty_[j]) {
            return col_max_[j];
        }

        double mx = 0.0;
        for (int i = k; i < n_; ++i) {
            mx = std::max(mx, std::abs(U_(i, j)));
        }
        col_max_[j] = mx;
        col_max_dirty_[j] = false;
        return mx;
    }

    // --------------------------------------------------
    // Metadata-only refresh after elimination step k
    // --------------------------------------------------
    void update_tracking_(int k) {
        row_deg_[k] = 0;
        col_deg_[k] = 0;
        col_max_[k] = 0.0;
        col_max_dirty_[k] = false;

        const int tail = n_ - k - 1;
        if (tail <= 0) {
            return;
        }

        // Refresh active rows.
        for (int i = k + 1; i < n_; ++i) {
            int cnt = 0;
            for (int j = k + 1; j < n_; ++j) {
                if (std::abs(U_(i, j)) > abs_floor_) {
                    ++cnt;
                }
            }
            row_deg_[i] = cnt;
        }

        // Refresh active columns.
        for (int j = k + 1; j < n_; ++j) {
            int cnt = 0;
            double mx = 0.0;
            for (int i = k + 1; i < n_; ++i) {
                const double a = std::abs(U_(i, j));
                if (a > abs_floor_) {
                    ++cnt;
                }
                mx = std::max(mx, a);
            }
            col_deg_[j] = cnt;
            col_max_[j] = mx;
            col_max_dirty_[j] = false;
        }
    }

    // --------------------------------------------------
    // Pivot selection
    // --------------------------------------------------
    std::pair<int, int> choose_pivot_(int k) {
        const int tail = n_ - k;
        if (tail <= 3) {
            return choose_pivot_brute_(k);
        }

        // Strong singleton preference.
        for (int i = k; i < n_; ++i) {
            if (row_deg_[i] != 1) {
                continue;
            }
            for (int j = k; j < n_; ++j) {
                if (std::abs(U_(i, j)) > abs_floor_ && col_deg_[j] == 1) {
                    return {i, j};
                }
            }
        }

        return choose_pivot_brute_(k);
    }

    std::pair<int, int> choose_pivot_brute_(int k) {
        const int tail = n_ - k;

        if (tail > 40) {
            if (const auto sample = choose_pivot_sample_(k); sample.first >= 0) {
                // Good sample found; accept if strong enough.
                const int i = sample.first;
                const int j = sample.second;
                const double ab = std::abs(U_(i, j));
                const double cm = std::max(col_max_active_(j, k), abs_floor_);
                const long score = static_cast<long>(std::max(0, row_deg_[i] - 1)) *
                                   static_cast<long>(std::max(0, col_deg_[j] - 1));
                if (score <= kEarlyScore_ || ab >= kEarlyRatio_ * cm) {
                    return sample;
                }
            }
        }

        int best_i = -1;
        int best_j = -1;
        long best_score = std::numeric_limits<long>::max();
        double best_abs = -1.0;

        // Visit rows in increasing degree order for faster early acceptance.
        std::vector<int> rows;
        rows.reserve(tail);
        for (int i = k; i < n_; ++i) {
            if (row_deg_[i] > 0) {
                rows.push_back(i);
            }
        }

        pdqsort(rows.begin(), rows.end(), [&](int a, int b) { return row_deg_[a] < row_deg_[b]; });

        for (const int i : rows) {
            const int rd = row_deg_[i];
            if (rd == 0) {
                continue;
            }

            for (int j = k; j < n_; ++j) {
                const double ab = std::abs(U_(i, j));
                if (ab <= abs_floor_) {
                    continue;
                }

                const double cm = col_max_active_(j, k);
                const double floor = std::max(cm, abs_floor_);
                if (ab < pivot_rel_ * floor) {
                    continue;
                }

                if (use_threshold_piv_) {
                    return {i, j};
                }

                const long score = static_cast<long>(std::max(0, rd - 1)) *
                                   static_cast<long>(std::max(0, col_deg_[j] - 1));

                if (score <= kEarlyScore_ && ab >= kEarlyRatio_ * floor) {
                    return {i, j};
                }

                if (score < best_score || (score == best_score && ab > best_abs)) {
                    best_score = score;
                    best_abs = ab;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i >= 0) {
            return {best_i, best_j};
        }

        return rook_pivot_(k);
    }

    std::pair<int, int> rook_pivot_(int k) {
        Eigen::Index ridx = 0;
        U_.col(k).segment(k, n_ - k).cwiseAbs().maxCoeff(&ridx);
        int i = k + static_cast<int>(ridx);
        int j = k;

        for (int t = 0; t < rook_iters_; ++t) {
            Eigen::Index cidx = 0;
            U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&cidx);
            j = k + static_cast<int>(cidx);

            U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff(&ridx);
            const int ni = k + static_cast<int>(ridx);
            if (ni == i) {
                break;
            }
            i = ni;
        }

        return {i, j};
    }

    std::pair<int, int> choose_pivot_sample_(int k) {
        const int tail = n_ - k;
        const int sample_rows = std::min(40, tail);

        std::vector<int> rows;
        rows.reserve(tail);
        for (int i = k; i < n_; ++i) {
            if (row_deg_[i] > 0) {
                rows.push_back(i);
            }
        }
        if (rows.empty()) {
            return {-1, -1};
        }

        pdqsort(rows.begin(), rows.end(), [&](int a, int b) { return row_deg_[a] < row_deg_[b]; });

        const int stride = std::max(1, static_cast<int>(rows.size()) / sample_rows);

        int best_i = -1;
        int best_j = -1;
        long best_score = std::numeric_limits<long>::max();
        double best_abs = -1.0;

        for (int idx = 0; idx < static_cast<int>(rows.size()); idx += stride) {
            const int i = rows[idx];

            for (int j = k; j < n_; ++j) {
                const double ab = std::abs(U_(i, j));
                if (ab <= abs_floor_) {
                    continue;
                }

                const double cm = col_max_active_(j, k);
                const double floor = std::max(cm, abs_floor_);
                if (ab < pivot_rel_ * floor) {
                    continue;
                }

                if (use_threshold_piv_) {
                    return {i, j};
                }

                const long score = static_cast<long>(std::max(0, row_deg_[i] - 1)) *
                                   static_cast<long>(std::max(0, col_deg_[j] - 1));

                if (score < best_score || (score == best_score && ab > best_abs)) {
                    best_score = score;
                    best_abs = ab;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        return {best_i, best_j};
    }

    // --------------------------------------------------
    // Factorization
    // --------------------------------------------------
    void factorize_() {
        double inf_norm = 0.0;
        for (int i = 0; i < n_; ++i) {
            inf_norm = std::max(inf_norm, U_.row(i).cwiseAbs().sum());
        }

        for (int k = 0; k < n_; ++k) {
            auto [pi, pj] = choose_pivot_(k);
            if (pi < 0 || pj < 0) {
                throw std::runtime_error("MarkowitzLU: no acceptable pivot found");
            }

            swap_rows_(k, pi);
            swap_cols_(k, pj);

            const double piv = U_(k, k);
            const double floor_adapt =
                std::max(abs_floor_, 10.0 * std::numeric_limits<double>::epsilon() * inf_norm);

            if (!std::isfinite(piv) || std::abs(piv) < floor_adapt) {
                throw std::runtime_error("MarkowitzLU: singular or too small pivot");
            }

            L_(k, k) = 1.0;

            const int tail = n_ - k - 1;
            if (tail > 0) {
                auto multipliers = U_.col(k).segment(k + 1, tail) * (1.0 / piv);
                L_.col(k).segment(k + 1, tail) = multipliers;

                // Single Schur update.
                U_.block(k + 1, k + 1, tail, tail).noalias() -=
                    multipliers * U_.row(k).segment(k + 1, tail);

                U_.col(k).segment(k + 1, tail).setZero();
            }

            update_tracking_(k);
        }
    }

    // --------------------------------------------------
    // Lazy transpose fallback
    // --------------------------------------------------
    void ensure_fallback_transpose_ready_() const {
        if (fallback_lu_t_ready_) {
            return;
        }
        fallback_lu_t_.compute(A_orig_.transpose());
        if (!fallback_lu_t_.isInvertible() || fallback_lu_t_.rank() != n_) {
            throw std::runtime_error("MarkowitzLU: transpose singular");
        }
        fallback_lu_t_ready_ = true;
    }

    // --------------------------------------------------
    // Members
    // --------------------------------------------------
    int n_{0};
    double pivot_rel_{1e-12};
    double abs_floor_{1e-16};
    int rook_iters_{2};
    bool use_threshold_piv_{false};
    bool use_fallback_full_piv_{false};

    Eigen::MatrixXd L_;
    Eigen::MatrixXd U_;
    Eigen::MatrixXd A_orig_;

    Eigen::FullPivLU<Eigen::MatrixXd> fallback_lu_;
    mutable Eigen::FullPivLU<Eigen::MatrixXd> fallback_lu_t_;
    mutable bool fallback_lu_t_ready_{false};

    std::vector<int> Pr_;
    std::vector<int> Pc_;

    mutable std::vector<int> row_deg_;
    mutable std::vector<int> col_deg_;
    mutable std::vector<double> col_max_;
    mutable std::vector<std::uint8_t> col_max_dirty_;

    mutable Eigen::VectorXd perm_buf1_;
    mutable Eigen::VectorXd perm_buf2_;
    mutable Eigen::VectorXd rhs_buf_;
    mutable Eigen::VectorXd residual_buf_;
    mutable Eigen::VectorXd corr_buf_;
};
