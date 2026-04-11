#pragma once

#include <Eigen/Dense>
#include <Eigen/LU> // for FullPivLU
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// ======================================================
// MarkowitzLU — Dense Markowitz + Rook fallback (C++23 enhanced)
//
// Major upgrades vs previous version:
//
// 1. Explicit singleton priority + improved early-accept.
// 2. Hybrid pivot search: full Markowitz when tail is small/medium,
//    candidate sampling + refinement when tail is large.
// 3. Incremental degree updates after rank-1 (cheaper than full rescans).
// 4. Stronger cache-friendly column-major access patterns.
// 5. C++23: std::span, ranges::iota, better move semantics.
// 6. Cleaner refinement loop with helper.
// 7. Optional threshold pivoting flag for even faster (but less stable) mode.
//
// Interface unchanged — L()/U() still full n×n for Forrest-Tomlin
// compatibility.
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
        if (A.rows() != A.cols() || A.rows() == 0)
            throw std::invalid_argument("MarkowitzLU: square non-empty matrix required");

        n_ = static_cast<int>(A.rows());
        pivot_rel_ = pivot_rel;
        abs_floor_ = abs_floor;
        rook_iters_ = rook_iters;
        use_fallback_full_piv_ = false;

        L_.setZero(n_, n_);
        U_ = A; // copy

        Pr_.resize(n_);
        Pc_.resize(n_);
        std::ranges::iota(Pr_, 0);
        std::ranges::iota(Pc_, 0);

        init_tracking_();

        try {
            factorize_();
        } catch (const std::runtime_error&) {
            // Fallback to full pivoting
            fallback_lu_.compute(A);
            if (!fallback_lu_.isInvertible() || fallback_lu_.rank() != n_)
                throw std::runtime_error("MarkowitzLU: matrix singular");

            fallback_lu_t_.compute(A.transpose());
            if (!fallback_lu_t_.isInvertible() || fallback_lu_t_.rank() != n_)
                throw std::runtime_error("MarkowitzLU: transpose singular");

            use_fallback_full_piv_ = true;
        }
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        if (b.size() != n_)
            throw std::invalid_argument("solve: size mismatch");
        if (use_fallback_full_piv_)
            return fallback_lu_.solve(b);

        Eigen::VectorXd w = apply_perm_(b, Pr_); // Pb
        triangular_solve_(L_, TriangularSolveMode::UnitLower, w);
        triangular_solve_(U_, TriangularSolveMode::Upper, w);

        iterative_refine_(w, b, /*is_transpose=*/false);
        return apply_perm_(w, Pc_);
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd& c) const {
        if (c.size() != n_)
            throw std::invalid_argument("solveT: size mismatch");
        if (use_fallback_full_piv_)
            return fallback_lu_t_.solve(c);

        Eigen::VectorXd s = apply_perm_(c, Pc_); // Pc^T c (inverse perm on right)
        triangular_solve_(U_, TriangularSolveMode::TransposeUpper, s);
        triangular_solve_(L_, TriangularSolveMode::TransposeUnitLower, s);

        iterative_refine_(s, c, /*is_transpose=*/true);
        return apply_inv_perm_(s, Pr_);
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

    static constexpr int kMaxRefine_ = 4; // slightly more aggressive
    static constexpr double kRefineTol_ = 1e-14;
    static constexpr long kEarlyScore_ = 1;
    static constexpr double kEarlyRatio_ = 0.92; // tightened a bit

    // ── Permutation helpers ─────────────────────────────────────────────
    Eigen::VectorXd apply_perm_(const Eigen::VectorXd& v, const std::vector<int>& perm) const {
        Eigen::VectorXd o(n_);
        for (int i = 0; i < n_; ++i)
            o(i) = v(perm[i]);
        return o;
    }

    Eigen::VectorXd apply_inv_perm_(const Eigen::VectorXd& y, const std::vector<int>& perm) const {
        Eigen::VectorXd o(n_);
        for (int i = 0; i < n_; ++i)
            o(perm[i]) = y(i);
        return o;
    }

    // ── Triangular solve helper (BLAS dtrsv) ───────────────────────────
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

    // ── Iterative refinement (dtrmv style) ─────────────────────────────
    void iterative_refine_(Eigen::VectorXd& x, const Eigen::VectorXd& rhs_orig,
                           bool is_transpose) const {
        const Eigen::VectorXd& rhs =
            is_transpose ? apply_perm_(rhs_orig, Pc_) : apply_perm_(rhs_orig, Pr_);

        for (int it = 0; it < kMaxRefine_; ++it) {
            Eigen::VectorXd r = rhs;
            if (is_transpose) {
                r.noalias() -= U_.triangularView<Eigen::Upper>().transpose() *
                               (L_.triangularView<Eigen::UnitLower>().transpose() * x);
            } else {
                r.noalias() -=
                    L_.triangularView<Eigen::UnitLower>() * (U_.triangularView<Eigen::Upper>() * x);
            }

            const double berr =
                r.lpNorm<Eigen::Infinity>() / std::max(1.0, rhs.lpNorm<Eigen::Infinity>());

            if (!std::isfinite(berr) || berr < kRefineTol_)
                break;

            Eigen::VectorXd dx = r;
            if (is_transpose) {
                triangular_solve_(U_, TriangularSolveMode::TransposeUpper, dx);
                triangular_solve_(L_, TriangularSolveMode::TransposeUnitLower, dx);
            } else {
                triangular_solve_(L_, TriangularSolveMode::UnitLower, dx);
                triangular_solve_(U_, TriangularSolveMode::Upper, dx);
            }

            if (!dx.array().isFinite().all() || dx.lpNorm<Eigen::Infinity>() < 1e-16)
                break;
            x += dx;
        }
    }

    // ── Row/Col swaps ───────────────────────────────────────────────────
    void swap_rows_(int a, int b) {
        if (a == b)
            return;
        U_.row(a).swap(U_.row(b));
        if (a > 0)
            L_.row(a).head(a).swap(L_.row(b).head(a));
        std::swap(row_deg_[a], row_deg_[b]);
        std::swap(Pr_[a], Pr_[b]);
    }

    void swap_cols_(int a, int b) {
        if (a == b)
            return;
        U_.col(a).swap(U_.col(b));
        std::swap(col_deg_[a], col_deg_[b]);
        std::swap(col_max_[a], col_max_[b]);
        std::swap(col_max_dirty_[a], col_max_dirty_[b]);
        std::swap(Pc_[a], Pc_[b]);
    }

    // ── Tracking ────────────────────────────────────────────────────────
    void init_tracking_() {
        row_deg_.resize(n_);
        col_deg_.resize(n_);
        col_max_.resize(n_);
        col_max_dirty_.assign(n_, true);

        for (int i = 0; i < n_; ++i)
            row_deg_[i] = static_cast<int>((U_.row(i).cwiseAbs().array() > abs_floor_).count());

        for (int j = 0; j < n_; ++j) {
            auto abs_col = U_.col(j).cwiseAbs();
            col_deg_[j] = static_cast<int>((abs_col.array() > abs_floor_).count());
            col_max_[j] = abs_col.maxCoeff();
        }
    }

    double col_max_active_(int j, int k) const {
        if (!col_max_dirty_[j])
            return col_max_[j];
        col_max_[j] = U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff();
        col_max_dirty_[j] = false;
        return col_max_[j];
    }

    // Incremental degree update after elimination at step k (cheaper than full
    // scan)
    void update_tracking_(int k) {
        const int tail = n_ - k - 1;
        if (tail <= 0) {
            row_deg_[k] = col_deg_[k] = 0;
            col_max_[k] = 0.0;
            return;
        }

        const double piv = U_(k, k);
        std::vector<int> pivot_cols;
        std::vector<double> pivot_vals;
        pivot_cols.reserve(tail);
        pivot_vals.reserve(tail);
        for (int j = k + 1; j < n_; ++j) {
            const double val = U_(k, j);
            if (std::abs(val) > abs_floor_) {
                pivot_cols.push_back(j);
                pivot_vals.push_back(val);
            }
        }

        std::vector<int> affected_rows;
        std::vector<double> multipliers;
        affected_rows.reserve(tail);
        multipliers.reserve(tail);
        for (int i = k + 1; i < n_; ++i) {
            const double uik = U_(i, k);
            if (std::abs(uik) > abs_floor_) {
                affected_rows.push_back(i);
                multipliers.push_back(uik / piv);
            }
        }

        // The pivot column becomes zero below the diagonal.
        for (int i = k + 1; i < n_; ++i)
            U_(i, k) = 0.0;

        for (int idx = 0; idx < static_cast<int>(affected_rows.size()); ++idx) {
            const int i = affected_rows[idx];
            const double lik = multipliers[idx];
            for (int t = 0; t < static_cast<int>(pivot_cols.size()); ++t) {
                const int j = pivot_cols[t];
                const double old_val = U_(i, j);
                const double new_val = old_val - lik * pivot_vals[t];
                const bool old_nz = std::abs(old_val) > abs_floor_;
                const bool new_nz = std::abs(new_val) > abs_floor_;
                if (old_nz != new_nz) {
                    row_deg_[i] += new_nz ? 1 : -1;
                    col_deg_[j] += new_nz ? 1 : -1;
                }
                U_(i, j) = new_val;
                col_max_dirty_[j] = true;
            }
        }

        row_deg_[k] = col_deg_[k] = 0;
        col_max_[k] = 0.0;
        col_max_dirty_[k] = false;
    }

    // ── Pivot selection (improved) ──────────────────────────────────────
    std::pair<int, int> choose_pivot_(int k) {
        const int tail = n_ - k;
        if (tail <= 3) { // small tail → brute force is fine
            return choose_pivot_brute_(k);
        }

        // Try early singletons first
        for (int i = k; i < n_; ++i) {
            if (row_deg_[i] == 1) {
                for (int j = k; j < n_; ++j) {
                    if (std::abs(U_(i, j)) > abs_floor_ && col_deg_[j] == 1)
                        return {i, j};
                }
            }
        }

        return choose_pivot_brute_(k); // fallback to full (still fast with early accept)
    }

    std::pair<int, int> choose_pivot_brute_(int k) {
        const int tail = n_ - k;
        if (tail > 40) {
            if (const auto sample = choose_pivot_sample_(k); sample.first >= 0)
                return sample;
        }

        int best_i = -1, best_j = -1;
        long best_score = std::numeric_limits<long>::max();
        double best_abs = -1.0;

        for (int i = k; i < n_; ++i) {
            if (row_deg_[i] == 0)
                continue;

            for (int j = k; j < n_; ++j) {
                const double ab = std::abs(U_(i, j));
                if (ab <= abs_floor_)
                    continue;

                const double cm = col_max_active_(j, k);
                const double floor = std::max(cm, abs_floor_);
                if (ab < pivot_rel_ * floor)
                    continue;

                if (use_threshold_piv_)
                    return {i, j};

                const long score = static_cast<long>(std::max(0, row_deg_[i] - 1)) *
                                   static_cast<long>(std::max(0, col_deg_[j] - 1));

                if (score <= kEarlyScore_ && ab >= kEarlyRatio_ * floor)
                    return {i, j}; // strong early accept

                if (score < best_score || (score == best_score && ab > best_abs)) {
                    best_score = score;
                    best_abs = ab;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i >= 0)
            return {best_i, best_j};

        // Rook fallback
        return rook_pivot_(k);
    }

    std::pair<int, int> rook_pivot_(int k) {
        Eigen::Index ridx;
        U_.col(k).segment(k, n_ - k).cwiseAbs().maxCoeff(&ridx);
        int i = k + static_cast<int>(ridx);
        int j = k;

        for (int t = 0; t < std::max(1, rook_iters_); ++t) {
            Eigen::Index cidx;
            U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&cidx);
            j = k + static_cast<int>(cidx);

            U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff(&ridx);
            const int ni = k + static_cast<int>(ridx);
            if (ni == i)
                break;
            i = ni;
        }
        return {i, j};
    }

    std::pair<int, int> choose_pivot_sample_(int k) const {
        const int tail = n_ - k;
        const int sample_rows = std::min(40, tail);
        const int step = std::max(1, tail / sample_rows);

        int best_i = -1, best_j = -1;
        long best_score = std::numeric_limits<long>::max();
        double best_abs = -1.0;

        for (int ii = k; ii < n_; ii += step) {
            if (row_deg_[ii] == 0)
                continue;

            for (int j = k; j < n_; ++j) {
                const double ab = std::abs(U_(ii, j));
                if (ab <= abs_floor_)
                    continue;

                const double cm = col_max_active_(j, k);
                const double floor = std::max(cm, abs_floor_);
                if (ab < pivot_rel_ * floor)
                    continue;

                if (use_threshold_piv_)
                    return {ii, j};

                const long score = static_cast<long>(std::max(0, row_deg_[ii] - 1)) *
                                   static_cast<long>(std::max(0, col_deg_[j] - 1));

                if (score < best_score || (score == best_score && ab > best_abs)) {
                    best_score = score;
                    best_abs = ab;
                    best_i = ii;
                    best_j = j;
                }
            }
        }

        return {best_i, best_j};
    }

    // ── Factorization core ──────────────────────────────────────────────
    void factorize_() {
        double inf_norm = 0.0;
        for (int i = 0; i < n_; ++i)
            inf_norm = std::max(inf_norm, U_.row(i).cwiseAbs().sum());

        for (int k = 0; k < n_; ++k) {
            auto [pi, pj] = choose_pivot_(k);
            if (pi < 0)
                throw std::runtime_error("MarkowitzLU: no acceptable pivot found");

            swap_rows_(k, pi);
            swap_cols_(k, pj);

            const double piv = U_(k, k);
            const double floor_adapt =
                std::max(abs_floor_, 10.0 * std::numeric_limits<double>::epsilon() * inf_norm);

            if (std::abs(piv) < floor_adapt || !std::isfinite(piv))
                throw std::runtime_error("MarkowitzLU: singular or too small pivot");

            L_(k, k) = 1.0;

            const int tail = n_ - k - 1;
            if (tail > 0) {
                // Multipliers
                auto multipliers = U_.col(k).segment(k + 1, tail) * (1.0 / piv);
                L_.col(k).segment(k + 1, tail) = multipliers;

                // BLAS dger via .noalias()
                U_.block(k + 1, k + 1, tail, tail).noalias() -=
                    multipliers * U_.row(k).segment(k + 1, tail);

                U_.col(k).segment(k + 1, tail).setZero();
            }

            update_tracking_(k);
        }
    }

    // ── Members ─────────────────────────────────────────────────────────
    int n_{0};
    double pivot_rel_{1e-12};
    double abs_floor_{1e-16};
    int rook_iters_{2};
    bool use_threshold_piv_{false};
    bool use_fallback_full_piv_{false};

    Eigen::MatrixXd L_, U_;
    Eigen::FullPivLU<Eigen::MatrixXd> fallback_lu_, fallback_lu_t_;

    std::vector<int> Pr_, Pc_;
    mutable std::vector<int> row_deg_, col_deg_;
    mutable std::vector<double> col_max_;
    mutable std::vector<std::uint8_t> col_max_dirty_;
};
