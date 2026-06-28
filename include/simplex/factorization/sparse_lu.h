#pragma once

#include "../../../extern/pdqsort/pdqsort.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <ankerl/unordered_dense.h>

#include "simplex/factorization/amd.h"
#include "simplex/core/markowitz.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <execution>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__has_cpp_attribute)
#    if __has_cpp_attribute(assume)
#        define SIMPLEX_ASSUME(expr) [[assume(expr)]]
#    else
#        define SIMPLEX_ASSUME(expr)
#    endif
#else
#    define SIMPLEX_ASSUME(expr)
#endif

template <class T, std::size_t Alignment> struct AlignedAllocator {
    using value_type = T;
    using is_always_equal = std::true_type;

    AlignedAllocator() noexcept = default;

    template <class U> constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_array_new_length();
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{Alignment}));
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        ::operator delete(ptr, std::align_val_t{Alignment});
    }

    template <class U> struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template <class T, class U, std::size_t Alignment>
constexpr bool operator==(const AlignedAllocator<T, Alignment>&,
                          const AlignedAllocator<U, Alignment>&) noexcept {
    return true;
}

template <class T, class U, std::size_t Alignment>
constexpr bool operator!=(const AlignedAllocator<T, Alignment>&,
                          const AlignedAllocator<U, Alignment>&) noexcept {
    return false;
}

// ======================================================
// Safe sparse base LU backend
// ======================================================
class SparseForrestTomlinLU {
  public:
    using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

    struct Config {
        bool use_amd_ordering{true};
        bool fallback_to_legacy_symbolic{true};
        bool diagonal_equilibration{true};
        int equilibration_passes{4};
        double equilibration_floor{1e-12};
        bool iterative_refinement{true};
        int iterative_refinement_steps{3};
        double iterative_refinement_tol{1e-10};
        double max_norm_growth_before_refactor{1e6};
        int max_parallel_update_size{64};
        bool enable_hyper_sparse_rhs{true};
        bool use_product_form_updates{true};
        bool force_eigen_sparse_lu{false};
    };

    struct UpdateStats {
        int count{0};
        double max_z_inf{0.0};
        double max_w_inf{0.0};
        double avg_z_density{0.0};
        double cumulative_z_inf{0.0};
        double norm_growth_estimate{1.0};
    };

    enum class UpdateFailureReason {
        None,
        BadDimensions,
        AlphaTooSmall,
        NonFiniteInput,
    };

    SparseForrestTomlinLU() = default;

    void factor(const SparseMat& A, double pivot_rel = 1e-12, double abs_floor = 1e-16,
                int refactor_rook_iters = 2, int /*ft_bandwidth_cap*/ = 0,
                const std::vector<int>* initial_row_perm = nullptr,
                const std::vector<int>* initial_col_perm = nullptr) {
        factor(A, pivot_rel, abs_floor, refactor_rook_iters, 0, initial_row_perm, initial_col_perm,
               Config{});
    }

    void factor(const SparseMat& A, double pivot_rel, double abs_floor, int refactor_rook_iters,
                int /*ft_bandwidth_cap*/, const std::vector<int>* initial_row_perm,
                const std::vector<int>* initial_col_perm, const Config& config) {
        if (A.rows() != A.cols())
            throw std::invalid_argument("SparseForrestTomlinLU: square only");

        n_ = static_cast<int>(A.rows());
        pivot_rel_ = pivot_rel;
        abs_floor_ = abs_floor;
        rook_iters_ = refactor_rook_iters;
        config_ = config;
        update_method_ = config_.use_product_form_updates ? UpdateMethod::PF : UpdateMethod::FT;
        base_matrix_original_ = A;
        base_matrix_original_.makeCompressed();
        base_matrix_one_norm_ = matrix_one_norm_(base_matrix_original_);
        use_fallback_sparse_lu_ = false;
        row_scale_.assign(n_, 1.0);
        col_scale_.assign(n_, 1.0);
        norm_growth_estimate_ = 1.0;

        Pr_.resize(n_);
        Pc_.resize(n_);
        if (initial_row_perm != nullptr) {
            if (static_cast<int>(initial_row_perm->size()) != n_)
                throw std::invalid_argument("SparseForrestTomlinLU: bad row permutation");
            Pr_ = *initial_row_perm;
        } else {
            std::iota(Pr_.begin(), Pr_.end(), 0);
        }
        if (initial_col_perm != nullptr) {
            if (static_cast<int>(initial_col_perm->size()) != n_)
                throw std::invalid_argument("SparseForrestTomlinLU: bad col permutation");
            Pc_ = *initial_col_perm;
        } else {
            std::iota(Pc_.begin(), Pc_.end(), 0);
        }
        rebuild_perm_inverses_();

        U_rows_.assign(n_, {});
        U_cols_.assign(n_, {});
        L_rows_.assign(n_, {});
        L_cols_.assign(n_, {});
        row_map_.resize(n_);
        col_map_.resize(n_);
        row_inv_.resize(n_);
        col_inv_.resize(n_);
        std::iota(row_map_.begin(), row_map_.end(), 0);
        std::iota(col_map_.begin(), col_map_.end(), 0);
        std::iota(row_inv_.begin(), row_inv_.end(), 0);
        std::iota(col_inv_.begin(), col_inv_.end(), 0);

        if (config_.force_eigen_sparse_lu) {
            activate_sparse_lu_fallback_(base_matrix_original_);
            clear_updates();
            return;
        }

        SparseMat factor_matrix = A;
        if (config_.diagonal_equilibration)
            factor_matrix = equilibrate_inf_norm_(A);
        load_initial_U_(factor_matrix);
        ensure_U_cols_ready_();
        symbolic_analyze_();
        initialize_active_stats_();

        // P1-2: Reset refactor cache at start of factorization
        refactor_info_.clear();
        refactor_info_.build_synthetic_tick = synthetic_tick_;

        try {
            factorize_sparse_();
            // P2-2: After factorization, complete any remaining logical injections
            //       for rows that were not yet processed (rank deficiency).
            complete_rank_deficient_basis_();
            build_solve_metadata_();
            // Mark refactor cache as valid after successful factorization
            if (!refactor_info_.pivot_row.empty() && !use_fallback_sparse_lu_) {
                refactor_info_.use = true;
            }
        } catch (const std::runtime_error&) {
            activate_sparse_lu_fallback_(base_matrix_original_);
        }
        clear_updates();
    }

    bool supports_inplace_updates() const noexcept { return n_ > 0 && !use_fallback_sparse_lu_; }

    bool has_updates() const noexcept { return !updates_.empty(); }

    // P2-2: Return true if the factorization had rank deficiency.
    bool is_rank_deficient() const noexcept { return rank_deficiency_ < 0; }
    int rank_deficiency() const noexcept { return -rank_deficiency_; }

    // P0-2: Refactor cache (Highs-style pivot records for fast rebuild)
    // ================================================================
    struct RefactorInfo {
        bool use{false};
        double build_synthetic_tick{0.0};
        std::vector<int> pivot_row;     // pivot row for each factor step
        std::vector<int> pivot_var;     // pivot column variable index
        std::vector<int8_t> pivot_type; // pivot type: 0=Markowitz, 1=Unit, 2=ColSingleton

        void clear() {
            use = false;
            build_synthetic_tick = 0.0;
            pivot_row.clear();
            pivot_var.clear();
            pivot_type.clear();
        }
    } refactor_info_;

    void clear_updates() noexcept {
        updates_.clear();
        clear_pf_updates();
        last_update_failure_reason_ = UpdateFailureReason::None;
        norm_growth_estimate_ = 1.0;
        updates_count_ = 0;
        updates_max_z_inf_ = 0.0;
        updates_max_w_inf_ = 0.0;
        updates_cumulative_z_inf_ = 0.0;
        updates_density_sum_ = 0.0;
    }

    UpdateFailureReason last_update_failure_reason() const noexcept {
        return last_update_failure_reason_;
    }

    const char* last_update_failure_reason_message() const noexcept {
        switch (last_update_failure_reason_) {
            case UpdateFailureReason::None:
                return "Sparse FT update accepted";
            case UpdateFailureReason::BadDimensions:
                return "Sparse FT update rejected due to dimension or index mismatch";
            case UpdateFailureReason::AlphaTooSmall:
                return "Sparse FT update rejected due to unstable alpha";
            case UpdateFailureReason::NonFiniteInput:
                return "Sparse FT update rejected due to non-finite update vectors";
        }

        return "Sparse FT update rejected for an unknown reason";
    }

    UpdateStats update_stats() const noexcept {
        UpdateStats stats;
        stats.count = updates_count_;
        stats.max_z_inf = updates_max_z_inf_;
        stats.max_w_inf = updates_max_w_inf_;
        stats.cumulative_z_inf = updates_cumulative_z_inf_;
        stats.avg_z_density =
            updates_count_ > 0 ? updates_density_sum_ / static_cast<double>(updates_count_) : 0.0;
        stats.norm_growth_estimate = norm_growth_estimate_;
        return stats;
    }

    bool append_forrest_tomlin_update(int j, const Eigen::VectorXd& u, const Eigen::VectorXd& z,
                                      const Eigen::VectorXd& w, double alpha, double eps = 1e-14) {
        if (j < 0 || j >= n_ || u.size() != n_ || z.size() != n_ || w.size() != n_) {
            last_update_failure_reason_ = UpdateFailureReason::BadDimensions;
            return false;
        }
        if (!std::isfinite(alpha) || std::abs(alpha) <= eps) {
            last_update_failure_reason_ = UpdateFailureReason::AlphaTooSmall;
            return false;
        }
        if (!u.array().isFinite().all() || !z.array().isFinite().all() ||
            !w.array().isFinite().all()) {
            last_update_failure_reason_ = UpdateFailureReason::NonFiniteInput;
            return false;
        }

        last_update_failure_reason_ = UpdateFailureReason::None;

        // If PF update method is selected, build packed column and use PF instead of FT.
        if (update_method_ == UpdateMethod::PF) {
            // Build packed column from z (excluding pivot row j).
            // PF stores: pivot_row=j, packed column entries, pivot_val=alpha.
            std::vector<int> col_idx;
            std::vector<double> col_val;
            col_idx.reserve(static_cast<size_t>(z.cwiseAbs().sum() / z.norm() + 1));
            col_val.reserve(col_idx.size());
            for (int i = 0; i < n_; ++i) {
                if (i != j && std::abs(z(i)) > eps) {
                    col_idx.push_back(i);
                    col_val.push_back(z(i));
                }
            }
            const bool pf_ok = append_pf_update(j, col_idx, col_val, alpha);
            // Preserve regular update vectors for transpose solves, while using PF
            // packed updates for forward solves.
            updates_.push_back(SparseUpdate{j, dense_to_sparse_update_(u, eps),
                                            dense_to_sparse_update_(z, eps),
                                            dense_to_sparse_update_(w, eps), alpha});
            update_norm_growth_estimate_(updates_.back());
            update_cached_stats_(updates_.back());
            return pf_ok;
        }

        updates_.push_back(SparseUpdate{j, dense_to_sparse_update_(u, eps),
                                        dense_to_sparse_update_(z, eps),
                                        dense_to_sparse_update_(w, eps), alpha});
        update_norm_growth_estimate_(updates_.back());
        update_cached_stats_(updates_.back());
        return true;
    }

    // Product Form (PF) update — stores the column in packed form for efficient updates.
    // Stores aq->packIndex/packValue excluding the pivot row.
    bool append_pf_update(int pivot_row, const std::vector<int>& col_index,
                          const std::vector<double>& col_value, double pivot_val) {
        pf_start_.push_back(static_cast<int>(pf_index_.size()));
        pf_pivot_index_.push_back(pivot_row);
        pf_pivot_value_.push_back(pivot_val);
        for (size_t i = 0; i < col_index.size(); ++i) {
            if (col_index[i] != pivot_row) {
                pf_index_.push_back(col_index[i]);
                pf_value_.push_back(col_value[i]);
            }
        }
        pf_total_fill_ += static_cast<int>(col_index.size());
        return true;
    }

    // Modified Product Form (MPF) update — stores full column + row for authenticated updates.
    bool append_mpf_update(int pivot_row, const std::vector<int>& col_index,
                           const std::vector<double>& col_value, double pivot_val,
                           const std::vector<int>& row_index,
                           const std::vector<double>& row_value) {
        pf_start_.push_back(static_cast<int>(pf_index_.size()));
        // Store column entries
        for (size_t i = 0; i < col_index.size(); ++i) {
            pf_index_.push_back(col_index[i]);
            pf_value_.push_back(col_value[i]);
        }
        // Store row entries with negated values
        for (size_t i = 0; i < row_index.size(); ++i) {
            pf_index_.push_back(row_index[i]);
            pf_value_.push_back(-row_value[i]);
        }
        pf_pivot_index_.push_back(pivot_row);
        pf_pivot_value_.push_back(pivot_val);
        pf_total_fill_ += static_cast<int>(col_index.size() + row_index.size());
        return true;
    }

    // Authenticated Product Form (APF) update — stores column plus original column for auth.
    bool append_apf_update(int pivot_row, const std::vector<int>& col_index,
                           const std::vector<double>& col_value, double pivot_val,
                           const std::vector<int>& orig_col_index,
                           const std::vector<double>& orig_col_value) {
        pf_start_.push_back(static_cast<int>(pf_index_.size()));
        // Store the new column
        for (size_t i = 0; i < col_index.size(); ++i) {
            pf_index_.push_back(col_index[i]);
            pf_value_.push_back(col_value[i]);
        }
        // Store original column for authentication
        for (size_t i = 0; i < orig_col_index.size(); ++i) {
            pf_index_.push_back(orig_col_index[i]);
            pf_value_.push_back(-orig_col_value[i]);
        }
        pf_pivot_index_.push_back(pivot_row);
        pf_pivot_value_.push_back(pivot_val);
        pf_total_fill_ += static_cast<int>(col_index.size() + orig_col_index.size());
        return true;
    }

    // Clear PF update structures.
    void clear_pf_updates() noexcept {
        pf_start_.clear();
        pf_index_.clear();
        pf_pivot_index_.clear();
        pf_value_.clear();
        pf_pivot_value_.clear();
        pf_total_fill_ = 0;
    }

    // Check if PF refactor is needed based on fill-in.
    bool pf_needs_refactor() const noexcept { return pf_total_fill_ > pf_merit_threshold_; }

    bool needs_refactor() const noexcept {
        return !std::isfinite(norm_growth_estimate_) ||
               norm_growth_estimate_ > config_.max_norm_growth_before_refactor;
    }

    // expected_density: HiGHS-style EWMA of (count/n_) for this TRAN class.
    // 1.0 (the default) means "treat as dense" — every per-stage gate falls
    // through to the indexed sparse path. Lower values let the per-stage gate
    // pick the hyper-sparse (etree-DFS reach) kernel.
    Eigen::VectorXd solve(const Eigen::VectorXd& b, double expected_density = 1.0) const {
        return solve_impl_(b, config_.iterative_refinement, expected_density);
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd& c, double expected_density = 1.0) const {
        return solveT_impl_(c, config_.iterative_refinement, expected_density);
    }

    // Sparse RHS interface (Item 1): caller provides the nonzero positions and values
    // of b directly, avoiding the O(n) scan inside the triangular solves.
    // seed_idx: original (pre-permutation) nonzero positions in b.
    Eigen::VectorXd solve_sparse(const std::vector<int>& seed_idx,
                                 const std::vector<double>& seed_val,
                                 double expected_density = 0.0) const {
        return solve_sparse_impl_(seed_idx, seed_val, config_.iterative_refinement,
                                  expected_density);
    }

    Eigen::VectorXd solveT_sparse(const std::vector<int>& seed_idx,
                                  const std::vector<double>& seed_val,
                                  double expected_density = 0.0) const {
        return solveT_sparse_impl_(seed_idx, seed_val, config_.iterative_refinement,
                                   expected_density);
    }

    // Returns the list of original-space row indices that may be nonzero in
    // the result of the most recent solve_sparse / solveT_sparse call. Only
    // meaningful when last_solve_pattern_valid() is true — otherwise the
    // caller must treat the solve result as dense.
    //
    // The pattern is invalidated (valid() returns false) when:
    //   - the solve fell back to the dense path (reach not tracked);
    //   - Forrest-Tomlin updates extended the pattern beyond the reach;
    //   - iterative refinement ran (may introduce new nonzeros);
    //   - the solve was a dense-RHS solve.
    const std::vector<int>& last_solve_reach_original() const noexcept {
        return last_solve_reach_original_;
    }

    bool last_solve_pattern_valid() const noexcept { return last_solve_pattern_valid_; }

  private:
    // FTRAN B*x = b. HiGHS-style per-stage gating:
    //   L stage: hyper iff (rhs_density <= kHyperCancel) && (expected <= kHyperFtranL)
    //   U stage: hyper iff (z_density   <= kHyperCancel) && (expected <= kHyperFtranU)
    // ema_reach_ratio_ remains a watchdog: if reach has been blowing up across
    // recent solves we force the sparse path even when current_density is low.
    Eigen::VectorXd solve_impl_(const Eigen::VectorXd& b, bool enable_refinement,
                                double expected_density) const {
        if (b.size() != n_) [[unlikely]]
            throw std::invalid_argument("SparseForrestTomlinLU::solve size mismatch");
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        last_solve_pattern_valid_ = false;

        if (use_fallback_sparse_lu_) {
            Eigen::VectorXd x = fallback_sparse_lu_.solve(b);
            if (!x.array().isFinite().all())
                throw std::runtime_error("SparseForrestTomlinLU: fallback solve failed");
            if (enable_refinement)
                x = iterative_refine_(b, x);
            return x;
        }

        permute_and_scale_rhs_(b, permuted_rhs_scratch_, Pr_, row_scale_);
        Eigen::VectorXd& Pb = permuted_rhs_scratch_;

        const int rhs_nz = count_nz_(Pb);
        const double rhs_density = static_cast<double>(rhs_nz) / static_cast<double>(n_);
        const bool watchdog_ok = ema_reach_ratio_ < kHyperSparseFallbackRatio_;

        const bool L_hyper = config_.enable_hyper_sparse_rhs && watchdog_ok &&
                             rhs_density <= kHyperCancel_ && expected_density <= kHyperFtranL_;
        if (L_hyper) {
            try {
                // Reset output scratch only at positions touched by the previous solve.
                clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
                last_solve_reach_original_.clear();

                forward_solve_L_sparse_inplace_(Pb, nullptr, sparse_l_scratch_);
                l_reach_seeds_scratch_ = std::move(reach_scratch_);

                // Per-stage U decision uses the post-L count (HiGHS pattern).
                const double z_density =
                    static_cast<double>(l_reach_seeds_scratch_.size()) / static_cast<double>(n_);
                const bool U_hyper =
                    z_density <= kHyperCancel_ && expected_density <= kHyperFtranU_;
                if (U_hyper) {
                    back_solve_U_sparse_inplace_(sparse_l_scratch_, &l_reach_seeds_scratch_,
                                                 sparse_u_scratch_);
                    hyper_solve_reach_valid_ = true;
                    last_solve_reach_original_.reserve(reach_scratch_.size());
                    for (const int i : reach_scratch_) {
                        const int xi = Pc_[i];
                        output_scratch_(xi) =
                            sparse_u_scratch_(i) * col_scale_[static_cast<size_t>(xi)];
                        last_solve_reach_original_.push_back(xi);
                    }
                    // Clear scratches at the positions we just touched; both kernels
                    // wrote only into reach positions.
                    clear_scratch_at_indices_(sparse_u_scratch_, reach_scratch_);
                    clear_scratch_at_indices_(sparse_l_scratch_, l_reach_seeds_scratch_);
                } else {
                    // Dense U on hyper-L result: lose the reach pattern. Need to
                    // promote sparse_l_scratch_ to a dense buffer for back_solve_U_,
                    // then clear it at L-reach positions afterwards.
                    Eigen::VectorXd w = back_solve_U_(sparse_l_scratch_);
                    clear_scratch_at_indices_(sparse_l_scratch_, l_reach_seeds_scratch_);
                    reach_scratch_.clear();
                    hyper_solve_reach_valid_ = false;
                    for (int i = 0; i < n_; ++i)
                        output_scratch_(Pc_[i]) = w(i);
                    apply_col_unscaling_(output_scratch_);
                    // Whole vector is now potentially nonzero — track every index
                    // so the next solve clears completely.
                    last_solve_reach_original_.resize(n_);
                    for (int i = 0; i < n_; ++i)
                        last_solve_reach_original_[i] = i;
                }
                if (!updates_.empty()) {
                    output_scratch_ = apply_updates_solve_(std::move(output_scratch_));
                    mark_output_scratch_dense_();
                }
                if (!validate_sparse_rhs_solution_(b, output_scratch_))
                    throw std::runtime_error(
                        "SparseForrestTomlinLU: hyper-sparse RHS residual check failed");
                Eigen::VectorXd x = output_scratch_;
                if (enable_refinement)
                    x = iterative_refine_(b, x);
                return x;
            } catch (const std::exception&) {
                hyper_solve_reach_valid_ = false;
                // Best-effort scratch cleanup so the dense fallback below sees a
                // clean output_scratch_ on the next call. We may have partially
                // written some positions — the safest option is full setZero on
                // both the value scratches and the output scratch.
                sparse_l_scratch_.setZero(n_);
                sparse_u_scratch_.setZero(n_);
                output_scratch_.setZero(n_);
                last_solve_reach_original_.clear();
            }
        }
        hyper_solve_reach_valid_ = false;
        Eigen::VectorXd w = back_solve_U_(forward_solve_L_(Pb));
        Eigen::VectorXd x(n_);
        for (int i = 0; i < n_; ++i)
            x(Pc_[i]) = w(i);
        apply_col_unscaling_(x);
        if (!updates_.empty())
            x = apply_updates_solve_(x);
        if (enable_refinement)
            x = iterative_refine_(b, x);
        return x;
    }

    // BTRAN B^T y = c. HiGHS-style per-stage gating (U^T first, then L^T):
    //   U^T stage: hyper iff rhs_density   <= kHyperCancel && expected <= kHyperBtranU
    //   L^T stage: hyper iff t_density     <= kHyperCancel && expected <= kHyperBtranL
    Eigen::VectorXd solveT_impl_(const Eigen::VectorXd& c, bool enable_refinement,
                                 double expected_density) const {
        if (c.size() != n_) [[unlikely]]
            throw std::invalid_argument("SparseForrestTomlinLU::solveT size mismatch");
        last_solve_pattern_valid_ = false;
        if (n_ == 0) [[unlikely]]
            return c;
        SIMPLEX_ASSUME(n_ > 0);

        if (use_fallback_sparse_lu_) {
            Eigen::VectorXd y = fallback_sparse_lu_t_.solve(c);
            if (!y.array().isFinite().all())
                throw std::runtime_error("SparseForrestTomlinLU: fallback transpose solve failed");
            if (enable_refinement)
                y = iterative_refine_T_(c, y);
            return y;
        }

        permute_and_scale_rhs_(c, permuted_transpose_rhs_scratch_, Pc_, col_scale_);
        Eigen::VectorXd& PcTc = permuted_transpose_rhs_scratch_;

        const int rhs_nz = count_nz_(PcTc);
        const double rhs_density = static_cast<double>(rhs_nz) / static_cast<double>(n_);
        const bool watchdog_ok = ema_reach_ratio_ < kHyperSparseFallbackRatio_;

        const bool UT_hyper = config_.enable_hyper_sparse_rhs && watchdog_ok &&
                              rhs_density <= kHyperCancel_ && expected_density <= kHyperBtranU_;
        if (UT_hyper) {
            try {
                clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
                last_solve_reach_original_.clear();

                forward_solve_UT_sparse_inplace_(PcTc, nullptr, sparse_u_scratch_);
                l_reach_seeds_scratch_ = std::move(reach_scratch_);

                const double t_density =
                    static_cast<double>(l_reach_seeds_scratch_.size()) / static_cast<double>(n_);
                const bool LT_hyper =
                    t_density <= kHyperCancel_ && expected_density <= kHyperBtranL_;
                if (LT_hyper) {
                    back_solve_LT_sparse_inplace_(sparse_u_scratch_, &l_reach_seeds_scratch_,
                                                  sparse_l_scratch_);
                    hyper_solve_reach_valid_ = true;
                    last_solve_reach_original_.reserve(reach_scratch_.size());
                    for (const int i : reach_scratch_) {
                        const int yi = Pr_[i];
                        output_scratch_(yi) =
                            sparse_l_scratch_(i) * row_scale_[static_cast<size_t>(yi)];
                        last_solve_reach_original_.push_back(yi);
                    }
                    clear_scratch_at_indices_(sparse_l_scratch_, reach_scratch_);
                    clear_scratch_at_indices_(sparse_u_scratch_, l_reach_seeds_scratch_);
                } else {
                    Eigen::VectorXd s = back_solve_LT_(sparse_u_scratch_);
                    clear_scratch_at_indices_(sparse_u_scratch_, l_reach_seeds_scratch_);
                    reach_scratch_.clear();
                    hyper_solve_reach_valid_ = false;
                    for (int i = 0; i < n_; ++i)
                        output_scratch_(Pr_[i]) = s(i);
                    apply_row_unscaling_(output_scratch_);
                    last_solve_reach_original_.resize(n_);
                    for (int i = 0; i < n_; ++i)
                        last_solve_reach_original_[i] = i;
                }
                if (!updates_.empty()) {
                    output_scratch_ = apply_updates_solve_T_(std::move(output_scratch_));
                    mark_output_scratch_dense_();
                }
                if (!validate_sparse_transpose_rhs_solution_(c, output_scratch_))
                    throw std::runtime_error(
                        "SparseForrestTomlinLU: hyper-sparse RHS transpose residual check failed");
                Eigen::VectorXd y = output_scratch_;
                if (enable_refinement)
                    y = iterative_refine_T_(c, y);
                return y;
            } catch (const std::exception&) {
                hyper_solve_reach_valid_ = false;
                sparse_l_scratch_.setZero(n_);
                sparse_u_scratch_.setZero(n_);
                output_scratch_.setZero(n_);
                last_solve_reach_original_.clear();
            }
        }
        hyper_solve_reach_valid_ = false;
        Eigen::VectorXd s = back_solve_LT_(forward_solve_UT_(PcTc));
        Eigen::VectorXd y(n_);
        for (int i = 0; i < n_; ++i)
            y(Pr_[i]) = s(i);
        apply_row_unscaling_(y);
        if (!updates_.empty())
            y = apply_updates_solve_T_(y);
        if (enable_refinement)
            y = iterative_refine_T_(c, y);
        return y;
    }

    // Sparse RHS implementations (Item 1): seed known nonzero positions directly,
    // bypassing the O(n) scan entirely for both L and U solves.
    //
    // expected_density is the HiGHS-style EWMA passed by the caller. The seed
    // path always feeds seeds into forward_solve_L_sparse_ (otherwise the
    // sparse-RHS API is meaningless), but the U stage falls back to a dense
    // back-solve when the post-L count exceeds the FtranU threshold or the
    // EWMA expects a dense result.
    Eigen::VectorXd solve_sparse_impl_(const std::vector<int>& seed_idx,
                                       const std::vector<double>& seed_val, bool enable_refinement,
                                       double expected_density) const {
        if (n_ == 0) [[unlikely]]
            return Eigen::VectorXd::Zero(0);
        if (use_fallback_sparse_lu_) {
            Eigen::VectorXd b = Eigen::VectorXd::Zero(n_);
            for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k)
                b(seed_idx[k]) = seed_val[k];
            return solve_impl_(b, enable_refinement, expected_density);
        }
        if (permuted_rhs_scratch_.size() < static_cast<Eigen::Index>(n_))
            permuted_rhs_scratch_.resize(n_);
        permuted_rhs_scratch_.setZero();
        perm_seeds_scratch_.clear();
        for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k) {
            const int orig = seed_idx[k];
            const int perm = Pr_inv_[static_cast<size_t>(orig)];
            permuted_rhs_scratch_(perm) = seed_val[k] * row_scale_[static_cast<size_t>(orig)];
            perm_seeds_scratch_.push_back(perm);
        }
        Eigen::VectorXd& Pb = permuted_rhs_scratch_;
        clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
        last_solve_reach_original_.clear();

        forward_solve_L_sparse_inplace_(Pb, &perm_seeds_scratch_, sparse_l_scratch_);
        l_reach_seeds_scratch_ = std::move(reach_scratch_);

        const double z_density =
            static_cast<double>(l_reach_seeds_scratch_.size()) / static_cast<double>(n_);
        const bool U_hyper = z_density <= kHyperCancel_ && expected_density <= kHyperFtranU_;
        bool pattern_via_reach = false;
        if (U_hyper) {
            back_solve_U_sparse_inplace_(sparse_l_scratch_, &l_reach_seeds_scratch_,
                                         sparse_u_scratch_);
            pattern_via_reach = true;
            last_solve_reach_original_.reserve(reach_scratch_.size());
            for (const int i : reach_scratch_) {
                const int xi = Pc_[i];
                output_scratch_(xi) = sparse_u_scratch_(i) * col_scale_[static_cast<size_t>(xi)];
                last_solve_reach_original_.push_back(xi);
            }
            clear_scratch_at_indices_(sparse_u_scratch_, reach_scratch_);
            clear_scratch_at_indices_(sparse_l_scratch_, l_reach_seeds_scratch_);
        } else {
            Eigen::VectorXd w = back_solve_U_(sparse_l_scratch_);
            clear_scratch_at_indices_(sparse_l_scratch_, l_reach_seeds_scratch_);
            reach_scratch_.clear();
            for (int i = 0; i < n_; ++i)
                output_scratch_(Pc_[i]) = w(i);
            apply_col_unscaling_(output_scratch_);
            last_solve_reach_original_.resize(n_);
            for (int i = 0; i < n_; ++i)
                last_solve_reach_original_[i] = i;
        }
        hyper_solve_reach_valid_ = pattern_via_reach;
        const bool pattern_preserved = pattern_via_reach && updates_.empty();
        if (!updates_.empty()) {
            output_scratch_ = apply_updates_solve_(std::move(output_scratch_));
            mark_output_scratch_dense_();
        }
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_);
        for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k)
            b(seed_idx[k]) = seed_val[k];
        if (!validate_sparse_rhs_solution_(b, output_scratch_)) {
            last_solve_pattern_valid_ = false;
            // Reset scratches before delegating to the dense path.
            clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
            last_solve_reach_original_.clear();
            return solve_impl_(b, enable_refinement, expected_density);
        }
        (void)enable_refinement;
        last_solve_pattern_valid_ = pattern_preserved;
        return output_scratch_;
    }

    Eigen::VectorXd solveT_sparse_impl_(const std::vector<int>& seed_idx,
                                        const std::vector<double>& seed_val, bool enable_refinement,
                                        double expected_density) const {
        if (n_ == 0) [[unlikely]]
            return Eigen::VectorXd::Zero(0);
        if (use_fallback_sparse_lu_) {
            Eigen::VectorXd c = Eigen::VectorXd::Zero(n_);
            for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k)
                c(seed_idx[k]) = seed_val[k];
            return solveT_impl_(c, enable_refinement, expected_density);
        }
        if (permuted_transpose_rhs_scratch_.size() < static_cast<Eigen::Index>(n_))
            permuted_transpose_rhs_scratch_.resize(n_);
        permuted_transpose_rhs_scratch_.setZero();
        perm_seeds_scratch_.clear();
        for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k) {
            const int orig = seed_idx[k];
            const int perm = Pc_inv_[static_cast<size_t>(orig)];
            permuted_transpose_rhs_scratch_(perm) =
                seed_val[k] * col_scale_[static_cast<size_t>(orig)];
            perm_seeds_scratch_.push_back(perm);
        }
        Eigen::VectorXd& PcTc = permuted_transpose_rhs_scratch_;
        clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
        last_solve_reach_original_.clear();

        forward_solve_UT_sparse_inplace_(PcTc, &perm_seeds_scratch_, sparse_u_scratch_);
        l_reach_seeds_scratch_ = std::move(reach_scratch_);

        const double t_density =
            static_cast<double>(l_reach_seeds_scratch_.size()) / static_cast<double>(n_);
        const bool LT_hyper = t_density <= kHyperCancel_ && expected_density <= kHyperBtranL_;
        bool pattern_via_reach = false;
        if (LT_hyper) {
            back_solve_LT_sparse_inplace_(sparse_u_scratch_, &l_reach_seeds_scratch_,
                                          sparse_l_scratch_);
            pattern_via_reach = true;
            last_solve_reach_original_.reserve(reach_scratch_.size());
            for (const int i : reach_scratch_) {
                const int yi = Pr_[i];
                output_scratch_(yi) = sparse_l_scratch_(i) * row_scale_[static_cast<size_t>(yi)];
                last_solve_reach_original_.push_back(yi);
            }
            clear_scratch_at_indices_(sparse_l_scratch_, reach_scratch_);
            clear_scratch_at_indices_(sparse_u_scratch_, l_reach_seeds_scratch_);
        } else {
            Eigen::VectorXd s = back_solve_LT_(sparse_u_scratch_);
            clear_scratch_at_indices_(sparse_u_scratch_, l_reach_seeds_scratch_);
            reach_scratch_.clear();
            for (int i = 0; i < n_; ++i)
                output_scratch_(Pr_[i]) = s(i);
            apply_row_unscaling_(output_scratch_);
            last_solve_reach_original_.resize(n_);
            for (int i = 0; i < n_; ++i)
                last_solve_reach_original_[i] = i;
        }
        hyper_solve_reach_valid_ = pattern_via_reach;
        const bool pattern_preserved = pattern_via_reach && updates_.empty();
        if (!updates_.empty()) {
            output_scratch_ = apply_updates_solve_T_(std::move(output_scratch_));
            mark_output_scratch_dense_();
        }
        Eigen::VectorXd c = Eigen::VectorXd::Zero(n_);
        for (int k = 0; k < static_cast<int>(seed_idx.size()); ++k)
            c(seed_idx[k]) = seed_val[k];
        if (!validate_sparse_transpose_rhs_solution_(c, output_scratch_)) {
            last_solve_pattern_valid_ = false;
            clear_scratch_at_indices_(output_scratch_, last_solve_reach_original_);
            last_solve_reach_original_.clear();
            return solveT_impl_(c, enable_refinement, expected_density);
        }
        (void)enable_refinement;
        last_solve_pattern_valid_ = pattern_preserved;
        return output_scratch_;
    }

    struct IndexedValue {
        int idx;
        double val;

        bool operator<(const IndexedValue& other) const noexcept { return idx < other.idx; }
    };

    using SparseRow = std::vector<IndexedValue>;
    using PatternSet = ankerl::unordered_dense::set<int>;

    static auto lower_bound_entry_(SparseRow& entries, int idx) {
        return std::lower_bound(entries.begin(), entries.end(), IndexedValue{idx, 0.0});
    }

    static auto lower_bound_entry_(const SparseRow& entries, int idx) {
        return std::lower_bound(entries.begin(), entries.end(), IndexedValue{idx, 0.0});
    }

    struct alignas(64) SparseUpdateVector {
        std::vector<int, AlignedAllocator<int, 64>> idx;
        std::vector<double, AlignedAllocator<double, 64>> val;

        double inf_norm() const {
            double out = 0.0;
            for (const double entry : val)
                out = std::max(out, std::abs(entry));
            return out;
        }

        double density(int n) const {
            if (n <= 0)
                return 0.0;
            return static_cast<double>(idx.size()) / static_cast<double>(n);
        }

        double one_norm() const {
            double out = 0.0;
            for (const double entry : val)
                out += std::abs(entry);
            return out;
        }

        double dot(const Eigen::VectorXd& x) const {
            double out = 0.0;
            const bool same_size = idx.size() == val.size();
            SIMPLEX_ASSUME(same_size);
            for (size_t pos = 0; pos < idx.size(); ++pos)
                out += val[pos] * x(idx[pos]);
            return out;
        }

        void axpy(Eigen::VectorXd& x, double alpha) const {
            const bool same_size = idx.size() == val.size();
            SIMPLEX_ASSUME(same_size);
            for (size_t pos = 0; pos < idx.size(); ++pos)
                x(idx[pos]) += alpha * val[pos];
        }
    };

    struct SparseUpdate {
        int j{-1};
        SparseUpdateVector u;
        SparseUpdateVector z;
        SparseUpdateVector w;
        double alpha{0.0};
    };

    struct RowCandidate {
        long score;
        double abs;
        int row;
        int col;
        int version;
    };

    struct RowCandidateGreater {
        bool operator()(const RowCandidate& lhs, const RowCandidate& rhs) const noexcept {
            if (lhs.score != rhs.score)
                return lhs.score > rhs.score;
            if (lhs.abs != rhs.abs)
                return lhs.abs < rhs.abs;
            if (lhs.row != rhs.row)
                return lhs.row > rhs.row;
            return lhs.col > rhs.col;
        }
    };

    static constexpr double kZeroTol_ = 1e-16;
    static constexpr double kHyperSparseDensityThreshold_ = 0.02;
    // Hyper-sparse thresholds from Highs HFactorConst.h — calibrated for LP basis factors.
    // kHyperCancel: RHS density threshold to switch from hyper-sparse to sparse solve.
    // kHyperFtranL/U, kHyperBtranL/U: expected density thresholds for hyper-sparse TRANs.
    static constexpr double kHyperSparseFallbackRatio_ = 0.05;
    static constexpr double kHyperCancel_ = 0.05;
    static constexpr double kHyperFtranL_ = 0.15;
    static constexpr double kHyperFtranU_ = 0.10;
    static constexpr double kHyperBtranL_ = 0.10;
    static constexpr double kHyperBtranU_ = 0.15;
    // kHighsTiny equivalent: threshold for zeroing small values in solves.
    static constexpr double kSparseTiny_ = 1e-14;
    static constexpr long kEarlyAcceptMarkowitzScore_ = 1;
    static constexpr double kEarlyAcceptPivotRatio_ = 0.9;

    static int count_nz_(const Eigen::VectorXd& v) noexcept {
        int nz = 0;
        for (Eigen::Index i = 0; i < v.size(); ++i)
            if (std::abs(v(i)) > kSparseTiny_)
                ++nz;
        return nz;
    }

    static bool is_hyper_sparse_rhs_(const Eigen::VectorXd& rhs) {
        if (rhs.size() == 0)
            return false;
        // Highs-style: use kHyperCancel threshold for hyper-sparse detection.
        const int threshold = std::max(1, static_cast<int>(rhs.size() * kHyperCancel_));
        int nz = 0;
        for (int i = 0; i < rhs.size() && nz <= threshold; ++i) {
            if (std::abs(rhs(i)) > kSparseTiny_)
                ++nz;
        }
        return nz <= threshold;
    }

    static double get_entry_(const SparseRow& entries, int idx) {
        const auto it = lower_bound_entry_(entries, idx);
        return it != entries.end() && it->idx == idx ? it->val : 0.0;
    }

    static void set_entry_(SparseRow& entries, int idx, double val) {
        const auto it = lower_bound_entry_(entries, idx);
        if (std::abs(val) <= kZeroTol_ || !std::isfinite(val)) {
            if (it != entries.end() && it->idx == idx)
                entries.erase(it);
            return;
        }

        if (it != entries.end() && it->idx == idx) {
            it->val = val;
            return;
        }

        entries.insert(it, IndexedValue{idx, val});
    }

    static std::vector<IndexedValue> sorted_entries_(const SparseRow& entries) { return entries; }

    static std::vector<IndexedValue> logical_sorted_entries_(const SparseRow& entries,
                                                             const std::vector<int>& inv) {
        std::vector<IndexedValue> ordered;
        ordered.reserve(entries.size());
        for (const auto& entry : entries)
            ordered.push_back(IndexedValue{inv[entry.idx], entry.val});
        pdqsort(ordered.begin(), ordered.end());
        return ordered;
    }

    static SparseUpdateVector dense_to_sparse_update_(const Eigen::VectorXd& dense, double eps) {
        SparseUpdateVector sparse;
        sparse.idx.reserve(static_cast<size_t>(std::min<int>(dense.size(), 32)));
        sparse.val.reserve(static_cast<size_t>(std::min<int>(dense.size(), 32)));
        for (int i = 0; i < dense.size(); ++i) {
            const double value = dense(i);
            if (std::abs(value) <= eps)
                continue;
            sparse.idx.push_back(i);
            sparse.val.push_back(value);
        }
        return sparse;
    }

    static double matrix_one_norm_(const SparseMat& A) {
        if (A.cols() == 0)
            return 0.0;

        double best = 0.0;
        for (int col = 0; col < A.outerSize(); ++col) {
            double sum = 0.0;
            for (typename SparseMat::InnerIterator it(A, col); it; ++it)
                sum += std::abs(it.value());
            best = std::max(best, sum);
        }
        return best;
    }

    static bool is_valid_permutation_(const std::vector<int>& perm, int n) {
        if (static_cast<int>(perm.size()) != n)
            return false;
        std::vector<char> seen(static_cast<size_t>(n), 0);
        for (const int entry : perm) {
            if (entry < 0 || entry >= n || seen[static_cast<size_t>(entry)])
                return false;
            seen[static_cast<size_t>(entry)] = 1;
        }
        return true;
    }

    static ::CSR sparse_to_amd_csr_(const SparseMat& A, double drop_tol) {
        ::CSR csr(A.rows());
        csr.indptr.assign(A.rows() + 1, 0);

        std::vector<std::vector<int>> rows(static_cast<size_t>(A.rows()));
        for (int col = 0; col < A.outerSize(); ++col) {
            for (typename SparseMat::InnerIterator it(A, col); it; ++it) {
                if (std::abs(it.value()) <= drop_tol)
                    continue;
                rows[static_cast<size_t>(it.row())].push_back(col);
            }
        }

        for (int row = 0; row < A.rows(); ++row) {
            auto& cols = rows[static_cast<size_t>(row)];
            pdqsort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            csr.indptr[row + 1] = csr.indptr[row] + static_cast<int>(cols.size());
            csr.indices.insert(csr.indices.end(), cols.begin(), cols.end());
        }
        return csr;
    }

    SparseMat equilibrate_inf_norm_(const SparseMat& A) {
        SparseMat scaled = A;
        scaled.makeCompressed();

        for (int pass = 0; pass < std::max(0, config_.equilibration_passes); ++pass) {
            std::vector<double> row_max(static_cast<size_t>(n_), 0.0);
            std::vector<double> col_max(static_cast<size_t>(n_), 0.0);

            for (int col = 0; col < scaled.outerSize(); ++col) {
                for (typename SparseMat::InnerIterator it(scaled, col); it; ++it) {
                    const double ab = std::abs(it.value());
                    row_max[static_cast<size_t>(it.row())] =
                        std::max(row_max[static_cast<size_t>(it.row())], ab);
                    col_max[static_cast<size_t>(col)] =
                        std::max(col_max[static_cast<size_t>(col)], ab);
                }
            }

            std::vector<double> row_factor(static_cast<size_t>(n_), 1.0);
            std::vector<double> col_factor(static_cast<size_t>(n_), 1.0);
            bool changed = false;

            for (int i = 0; i < n_; ++i) {
                const double vmax = row_max[static_cast<size_t>(i)];
                if (vmax > config_.equilibration_floor) {
                    row_factor[static_cast<size_t>(i)] = 1.0 / std::sqrt(vmax);
                    row_scale_[static_cast<size_t>(i)] *= row_factor[static_cast<size_t>(i)];
                    changed = changed || std::abs(row_factor[static_cast<size_t>(i)] - 1.0) > 1e-6;
                }
            }
            for (int j = 0; j < n_; ++j) {
                const double vmax = col_max[static_cast<size_t>(j)];
                if (vmax > config_.equilibration_floor) {
                    col_factor[static_cast<size_t>(j)] = 1.0 / std::sqrt(vmax);
                    col_scale_[static_cast<size_t>(j)] *= col_factor[static_cast<size_t>(j)];
                    changed = changed || std::abs(col_factor[static_cast<size_t>(j)] - 1.0) > 1e-6;
                }
            }

            if (!changed)
                break;

            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(static_cast<size_t>(scaled.nonZeros()));
            for (int col = 0; col < scaled.outerSize(); ++col) {
                const double col_scale = col_factor[static_cast<size_t>(col)];
                for (typename SparseMat::InnerIterator it(scaled, col); it; ++it) {
                    const double row_scale = row_factor[static_cast<size_t>(it.row())];
                    const double value = row_scale * it.value() * col_scale;
                    if (std::abs(value) > kZeroTol_)
                        trips.emplace_back(it.row(), col, value);
                }
            }

            SparseMat next(n_, n_);
            if (!trips.empty())
                next.setFromTriplets(trips.begin(), trips.end());
            next.makeCompressed();
            scaled = std::move(next);
        }

        return scaled;
    }

    void apply_row_scaling_(Eigen::VectorXd& rhs) const {
        for (int i = 0; i < rhs.size(); ++i)
            rhs(i) *= row_scale_[static_cast<size_t>(i)];
    }

    void apply_col_scaling_(Eigen::VectorXd& rhs) const {
        for (int i = 0; i < rhs.size(); ++i)
            rhs(i) *= col_scale_[static_cast<size_t>(i)];
    }

    template <typename ScaleVector>
    void permute_and_scale_rhs_(const Eigen::VectorXd& src, Eigen::VectorXd& dst,
                                const std::vector<int>& perm, const ScaleVector& scale) const {
        if (dst.size() < n_)
            dst.resize(n_);
        for (int i = 0; i < n_; ++i)
            dst(i) = src(perm[i]) * scale[static_cast<size_t>(perm[i])];
    }

    void rebuild_perm_inverses_() {
        Pr_inv_.resize(n_);
        Pc_inv_.resize(n_);
        for (int i = 0; i < n_; ++i)
            Pr_inv_[Pr_[i]] = i;
        for (int i = 0; i < n_; ++i)
            Pc_inv_[Pc_[i]] = i;
    }

    void apply_col_unscaling_(Eigen::VectorXd& x) const {
        for (int i = 0; i < x.size(); ++i)
            x(i) *= col_scale_[static_cast<size_t>(i)];
    }

    void apply_row_unscaling_(Eigen::VectorXd& y) const {
        for (int i = 0; i < y.size(); ++i)
            y(i) *= row_scale_[static_cast<size_t>(i)];
    }

    void clear_reach_flags_scratch_(const std::vector<int>* extra = nullptr) const {
        for (const int j : reach_scratch_)
            reach_flag_scratch_[j] = false;
        if (extra) {
            for (const int j : *extra)
                reach_flag_scratch_[j] = false;
        }
        reach_scratch_.clear();
    }

    void clear_sparse_solve_scratch_entries_(Eigen::VectorXd& x) const {
        for (const int i : reach_scratch_)
            x(i) = 0.0;
    }

    // Zero positions in `x` listed in `indices`. Used to reset sparse_l/u_scratch_
    // and output_scratch_ between solves without a full setZero().
    static void clear_scratch_at_indices_(Eigen::VectorXd& x,
                                          const std::vector<int>& indices) noexcept {
        for (const int i : indices)
            x(i) = 0.0;
    }

    void mark_output_scratch_dense_() const {
        last_solve_reach_original_.resize(n_);
        for (int i = 0; i < n_; ++i)
            last_solve_reach_original_[i] = i;
    }

    void ensure_sparse_scratch_size_(Eigen::VectorXd& x) const {
        if (x.size() != n_) {
            x.resize(n_);
            x.setZero();
        }
    }

    void symbolic_analyze_() {
        if (config_.use_amd_ordering && amd_symbolic_analyze_())
            return;
        legacy_symbolic_analyze_();
    }

    bool amd_symbolic_analyze_() {
        symbolic_row_hint_phys_.assign(n_, -1);
        symbolic_col_hint_phys_.assign(n_, -1);

        const SparseMat pattern = build_pattern_matrix_();
        const ::CSR csr = sparse_to_amd_csr_(pattern, abs_floor_);
        AMDReorderingArray amd(/*aggressive_absorption=*/true,
                               /*dense_cutoff=*/-1);
        auto [perm, stats] = amd.compute_fill_reducing_permutation(csr, /*symmetrize=*/true);
        (void)stats;

        if (!is_valid_permutation_(perm, n_))
            return false;

        for (int k = 0; k < n_; ++k) {
            symbolic_row_hint_phys_[k] = perm[static_cast<size_t>(k)];
            symbolic_col_hint_phys_[k] = perm[static_cast<size_t>(k)];
        }
        return true;
    }

    SparseMat build_pattern_matrix_() const {
        std::vector<Eigen::Triplet<double>> trips;
        for (int phys_row = 0; phys_row < n_; ++phys_row) {
            for (const auto& [phys_col, val] : U_rows_[phys_row]) {
                if (std::abs(val) <= abs_floor_)
                    continue;
                trips.emplace_back(phys_row, phys_col, val);
            }
        }

        SparseMat out(n_, n_);
        if (!trips.empty())
            out.setFromTriplets(trips.begin(), trips.end());
        out.makeCompressed();
        return out;
    }

    void legacy_symbolic_analyze_() {
        symbolic_row_hint_phys_.assign(n_, -1);
        symbolic_col_hint_phys_.assign(n_, -1);

        if (legacy_pattern_rows_scratch_.size() < static_cast<size_t>(n_))
            legacy_pattern_rows_scratch_.resize(n_);
        if (legacy_pattern_cols_scratch_.size() < static_cast<size_t>(n_))
            legacy_pattern_cols_scratch_.resize(n_);
        auto& pattern_rows = legacy_pattern_rows_scratch_;
        auto& pattern_cols = legacy_pattern_cols_scratch_;

        for (int phys_row = 0; phys_row < n_; ++phys_row) {
            pattern_rows[static_cast<size_t>(phys_row)].clear();
            pattern_cols[static_cast<size_t>(phys_row)].clear();
        }

        for (int phys_row = 0; phys_row < n_; ++phys_row) {
            pattern_rows[phys_row].reserve(U_rows_[phys_row].size());
            for (const auto& [phys_col, val] : U_rows_[phys_row]) {
                if (std::abs(val) <= abs_floor_)
                    continue;
                pattern_rows[phys_row].insert(phys_col);
                pattern_cols[phys_col].insert(phys_row);
            }
        }

        if (sym_row_map_scratch_.size() < static_cast<size_t>(n_)) {
            sym_row_map_scratch_.resize(n_);
            sym_col_map_scratch_.resize(n_);
            sym_row_inv_scratch_.resize(n_);
            sym_col_inv_scratch_.resize(n_);
        }
        auto& sym_row_map = sym_row_map_scratch_;
        auto& sym_col_map = sym_col_map_scratch_;
        auto& sym_row_inv = sym_row_inv_scratch_;
        auto& sym_col_inv = sym_col_inv_scratch_;
        std::iota(sym_row_map.begin(), sym_row_map.end(), 0);
        std::iota(sym_col_map.begin(), sym_col_map.end(), 0);
        std::iota(sym_row_inv.begin(), sym_row_inv.end(), 0);
        std::iota(sym_col_inv.begin(), sym_col_inv.end(), 0);

        for (int k = 0; k < n_; ++k) {
            int best_i = -1;
            int best_j = -1;
            long best_score = std::numeric_limits<long>::max();

            for (int i = k; i < n_; ++i) {
                const int phys_row = sym_row_map[i];
                int row_degree = 0;
                for (const int phys_col : pattern_rows[phys_row]) {
                    if (sym_col_inv[phys_col] >= k)
                        ++row_degree;
                }
                if (row_degree == 0)
                    continue;

                for (const int phys_col : pattern_rows[phys_row]) {
                    const int j = sym_col_inv[phys_col];
                    if (j < k)
                        continue;

                    int col_degree = 0;
                    for (const int phys_row_in_col : pattern_cols[phys_col]) {
                        if (sym_row_inv[phys_row_in_col] >= k)
                            ++col_degree;
                    }

                    const long score = static_cast<long>(std::max(0, row_degree - 1)) *
                                       static_cast<long>(std::max(0, col_degree - 1));
                    if (score < best_score) {
                        best_score = score;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if (best_i < 0 || best_j < 0)
                break;

            std::swap(sym_row_map[k], sym_row_map[best_i]);
            sym_row_inv[sym_row_map[k]] = k;
            sym_row_inv[sym_row_map[best_i]] = best_i;
            std::swap(sym_col_map[k], sym_col_map[best_j]);
            sym_col_inv[sym_col_map[k]] = k;
            sym_col_inv[sym_col_map[best_j]] = best_j;

            const int pivot_phys_row = sym_row_map[k];
            const int pivot_phys_col = sym_col_map[k];
            symbolic_row_hint_phys_[k] = pivot_phys_row;
            symbolic_col_hint_phys_[k] = pivot_phys_col;

            pivot_row_cols_scratch_.clear();
            pivot_row_cols_scratch_.reserve(pattern_rows[pivot_phys_row].size());
            for (const int phys_col : pattern_rows[pivot_phys_row]) {
                if (sym_col_inv[phys_col] > k)
                    pivot_row_cols_scratch_.push_back(phys_col);
            }

            symbolic_affected_rows_scratch_.clear();
            symbolic_affected_rows_scratch_.reserve(pattern_cols[pivot_phys_col].size());
            for (const int phys_row : pattern_cols[pivot_phys_col]) {
                if (sym_row_inv[phys_row] > k)
                    symbolic_affected_rows_scratch_.push_back(phys_row);
            }

            for (const int phys_row : symbolic_affected_rows_scratch_) {
                pattern_rows[phys_row].erase(pivot_phys_col);
                pattern_cols[pivot_phys_col].erase(phys_row);
                for (const int phys_col : pivot_row_cols_scratch_) {
                    if (pattern_rows[phys_row].insert(phys_col).second)
                        pattern_cols[phys_col].insert(phys_row);
                }
            }
        }
    }

    std::optional<std::pair<int, int>> symbolic_hint_pivot_(int k) const {
        if (k < 0 || k >= n_ || static_cast<int>(symbolic_row_hint_phys_.size()) != n_ ||
            static_cast<int>(symbolic_col_hint_phys_.size()) != n_ ||
            symbolic_row_hint_phys_[k] < 0 || symbolic_col_hint_phys_[k] < 0)
            return std::nullopt;

        const int row = row_inv_[symbolic_row_hint_phys_[k]];
        const int col = col_inv_[symbolic_col_hint_phys_[k]];
        if (row < k || col < k)
            return std::nullopt;

        const double aij =
            get_entry_(U_rows_[symbolic_row_hint_phys_[k]], symbolic_col_hint_phys_[k]);
        const double ab = std::abs(aij);
        if (ab <= abs_floor_)
            return std::nullopt;

        const double colmax = active_col_max_(col);
        if (ab < pivot_rel_ * std::max(colmax, abs_floor_))
            return std::nullopt;

        return std::pair<int, int>{row, col};
    }

    void reset_row_candidate_heap_() {
        row_candidate_heap_ =
            std::priority_queue<RowCandidate, std::vector<RowCandidate>, RowCandidateGreater>();
    }

    void queue_column_candidate_invalidation_(int col) {
        if (col < active_k_ || col_candidate_dirty_[col])
            return;
        col_candidate_dirty_[col] = true;
        dirty_cols_scratch_.push_back(col);
    }

    void invalidate_row_candidate_(int row) {
        if (row < active_k_ || row_candidate_dirty_[row])
            return;
        row_candidate_dirty_[row] = true;
        row_candidate_heap_.push(RowCandidate{std::numeric_limits<long>::min(),
                                              std::numeric_limits<double>::infinity(), row, -1,
                                              row_candidate_version_[row]});
    }

    void flush_column_candidate_invalidations_() {
        ensure_U_cols_ready_();
        for (const int col : dirty_cols_scratch_) {
            col_candidate_dirty_[col] = false;
            for (const auto& [phys_row, val] : U_cols_[col_map_[col]]) {
                const int logical_row = row_inv_[phys_row];
                if (logical_row >= active_k_ && std::abs(val) > abs_floor_)
                    invalidate_row_candidate_(logical_row);
            }
        }
        dirty_cols_scratch_.clear();
    }

    bool recompute_row_candidate_(int row) {
        row_candidate_dirty_[row] = false;
        row_candidate_best_col_[row] = -1;
        row_candidate_best_score_[row] = std::numeric_limits<long>::max();
        row_candidate_best_abs_[row] = -1.0;

        if (row < active_k_ || row_degree_[row] == 0)
            return false;

        int best_col = -1;
        long best_score = std::numeric_limits<long>::max();
        double best_abs = -1.0;

        for (const auto& [phys_col, aij] : U_rows_[row_map_[row]]) {
            const int col = col_inv_[phys_col];
            if (col < active_k_)
                continue;

            const double ab = std::abs(aij);
            if (ab <= abs_floor_)
                continue;

            const double colmax = active_col_max_(col);
            if (ab < pivot_rel_ * std::max(colmax, abs_floor_))
                continue;

            const long score = static_cast<long>(std::max(0, row_degree_[row] - 1)) *
                               static_cast<long>(std::max(0, col_degree_[col] - 1));

            if (score < best_score || (score == best_score && ab > best_abs)) {
                best_score = score;
                best_abs = ab;
                best_col = col;
            }
        }

        if (best_col < 0)
            return false;

        ++row_candidate_version_[row];
        row_candidate_best_col_[row] = best_col;
        row_candidate_best_score_[row] = best_score;
        row_candidate_best_abs_[row] = best_abs;
        row_candidate_heap_.push(
            RowCandidate{best_score, best_abs, row, best_col, row_candidate_version_[row]});
        return true;
    }

    bool is_active_significant_(int row, int col, double val) const {
        return row >= active_k_ && col >= active_k_ && std::abs(val) > abs_floor_;
    }

    void initialize_active_stats_() {
        active_k_ = 0;
        row_degree_.assign(n_, 0);
        col_degree_.assign(n_, 0);
        col_max_abs_.assign(n_, 0.0);
        col_max_dirty_.assign(n_, false);
        row_candidate_best_col_.assign(n_, -1);
        row_candidate_best_score_.assign(n_, std::numeric_limits<long>::max());
        row_candidate_best_abs_.assign(n_, -1.0);
        row_candidate_version_.assign(n_, 0);
        row_candidate_dirty_.assign(n_, false);
        col_candidate_dirty_.assign(n_, false);
        dirty_cols_scratch_.clear();
        reset_row_candidate_heap_();

        for (int i = 0; i < n_; ++i) {
            for (const auto& [phys_col, val] : U_rows_[i]) {
                const int logical_col = col_inv_[phys_col];
                if (std::abs(val) <= abs_floor_ || logical_col < active_k_)
                    continue;
                ++row_degree_[i];
            }
        }

        for (int j = 0; j < n_; ++j) {
            double col_max = 0.0;
            for (const auto& [phys_row, val] : U_cols_[j]) {
                const int logical_row = row_inv_[phys_row];
                if (std::abs(val) <= abs_floor_ || logical_row < active_k_)
                    continue;
                ++col_degree_[j];
                col_max = std::max(col_max, std::abs(val));
            }
            col_max_abs_[j] = col_max;
        }

        for (int i = 0; i < n_; ++i)
            recompute_row_candidate_(i);
    }

    void note_U_entry_change_(int row, int col, double old_val, double new_val) {
        const bool old_active = is_active_significant_(row, col, old_val);
        const bool new_active = is_active_significant_(row, col, new_val);

        if (old_active && !new_active) {
            --row_degree_[row];
            --col_degree_[col];
            if (std::abs(old_val) >= col_max_abs_[col])
                col_max_dirty_[col] = true;
        } else if (!old_active && new_active) {
            ++row_degree_[row];
            ++col_degree_[col];
            col_max_abs_[col] = std::max(col_max_abs_[col], std::abs(new_val));
        } else if (old_active && new_active) {
            if (std::abs(new_val) > col_max_abs_[col]) {
                col_max_abs_[col] = std::abs(new_val);
                col_max_dirty_[col] = false;
            } else if (std::abs(old_val) >= col_max_abs_[col] &&
                       std::abs(new_val) < std::abs(old_val)) {
                col_max_dirty_[col] = true;
            }
        }
    }

    void set_U_active_(int i, int j, double v) {
        const double old_v = get_U_(i, j);
        note_U_entry_change_(i, j, old_v, v);
        const int phys_row = row_map_[i];
        const int phys_col = col_map_[j];
        set_entry_(U_rows_[phys_row], phys_col, v);
        U_cols_dirty_ = true;
        invalidate_row_candidate_(i);
        queue_column_candidate_invalidation_(j);
    }

    void merge_update_U_row_active_(int row, int pivot_col_phys, double lik,
                                    const SparseRow& pivot_row_phys) {
        const int phys_row = row_map_[row];
        const SparseRow& target_row = U_rows_[phys_row];

        SparseRow& new_row_entries = merge_scratch_;
        new_row_entries.clear();
        new_row_entries.reserve(target_row.size() + pivot_row_phys.size());

        std::size_t target_pos = 0;
        std::size_t pivot_pos = 0;
        while (target_pos < target_row.size() || pivot_pos < pivot_row_phys.size()) {
            const int target_col = target_pos < target_row.size() ? target_row[target_pos].idx
                                                                  : std::numeric_limits<int>::max();
            const int pivot_col = pivot_pos < pivot_row_phys.size()
                                      ? pivot_row_phys[pivot_pos].idx
                                      : std::numeric_limits<int>::max();

            if (target_col < pivot_col) {
                new_row_entries.push_back(target_row[target_pos]);
                ++target_pos;
                continue;
            }

            if (pivot_col < target_col) {
                const double new_val = -lik * pivot_row_phys[pivot_pos].val;
                if (std::abs(new_val) > kZeroTol_) {
                    new_row_entries.push_back(IndexedValue{pivot_col, new_val});
                    const int logical_col = col_inv_[pivot_col];
                    note_U_entry_change_(row, logical_col, 0.0, new_val);
                    queue_column_candidate_invalidation_(logical_col);
                }
                ++pivot_pos;
                continue;
            }

            const double old_val = target_row[target_pos].val;
            const double new_val = old_val - lik * pivot_row_phys[pivot_pos].val;
            if (std::abs(new_val) > kZeroTol_)
                new_row_entries.push_back(IndexedValue{target_col, new_val});
            if (std::abs(new_val - old_val) > kZeroTol_) {
                const int logical_col = col_inv_[target_col];
                note_U_entry_change_(row, logical_col, old_val, new_val);
                queue_column_candidate_invalidation_(logical_col);
            }
            ++target_pos;
            ++pivot_pos;
        }

        U_rows_[phys_row] = std::move(new_row_entries);
        U_cols_dirty_ = true;
        invalidate_row_candidate_(row);
    }

    double active_col_max_(int col) const {
        if (col_max_dirty_[col]) {
            ensure_U_cols_ready_();
            double col_max = 0.0;
            for (const auto& [phys_row, val] : U_cols_[col_map_[col]]) {
                const int logical_row = row_inv_[phys_row];
                if (logical_row >= active_k_ && std::abs(val) > abs_floor_)
                    col_max = std::max(col_max, std::abs(val));
            }
            col_max_abs_[col] = col_max;
            col_max_dirty_[col] = false;
        }
        return col_max_abs_[col];
    }

    void finalize_pivot_step_(int k) {
        for (const auto& [phys_col, val] : U_rows_[row_map_[k]]) {
            const int col = col_inv_[phys_col];
            if (col <= k || std::abs(val) <= abs_floor_)
                continue;
            --col_degree_[col];
            if (std::abs(val) >= col_max_abs_[col])
                col_max_dirty_[col] = true;
        }

        ensure_U_cols_ready_();
        for (const auto& [phys_row, val] : U_cols_[col_map_[k]]) {
            const int row = row_inv_[phys_row];
            if (row <= k || std::abs(val) <= abs_floor_)
                continue;
            --row_degree_[row];
        }

        row_degree_[k] = 0;
        col_degree_[k] = 0;
        col_max_abs_[k] = 0.0;
        col_max_dirty_[k] = false;
        active_k_ = k + 1;
        queue_column_candidate_invalidation_(k);
    }

    void build_solve_metadata_() {
        L_diag_.assign(n_, 0.0);
        U_diag_.assign(n_, 0.0);
        L_lower_ptr_.assign(n_ + 1, 0);
        U_upper_ptr_.assign(n_ + 1, 0);
        UT_lower_ptr_.assign(n_ + 1, 0);
        LT_upper_ptr_.assign(n_ + 1, 0);
        L_lower_idx_.clear();
        L_lower_val_.clear();
        U_upper_idx_.clear();
        U_upper_val_.clear();
        UT_lower_idx_.clear();
        UT_lower_val_.clear();
        LT_upper_idx_.clear();
        LT_upper_val_.clear();

        // Ensure column structures are ready once before the loop (Item 4).
        // Previously ensure_U_cols_ready_() / ensure_L_cols_ready_() were called inside
        // the loop on every iteration — this moves each to a single call.
        ensure_U_cols_ready_();
        ensure_L_cols_ready_();

        for (int i = 0; i < n_; ++i) {
            const int phys_row = row_map_[i];
            const int phys_col = col_map_[i];

            L_lower_ptr_[i] = static_cast<int>(L_lower_idx_.size());
            for (const auto& entry : L_rows_[i]) {
                if (entry.idx < i) {
                    L_lower_idx_.push_back(entry.idx);
                    L_lower_val_.push_back(entry.val);
                } else if (entry.idx == i) {
                    L_diag_[i] = entry.val;
                } else {
                    break;
                }
            }
            L_lower_ptr_[i + 1] = static_cast<int>(L_lower_idx_.size());

            // U upper: remap physical→logical column indices using scratch, sort once (Item 4).
            // Replaces logical_sorted_entries_() which allocated a fresh vector per row.
            U_upper_ptr_[i] = static_cast<int>(U_upper_idx_.size());
            build_tmp_.clear();
            for (const auto& entry : U_rows_[phys_row])
                build_tmp_.push_back({col_inv_[entry.idx], entry.val});
            pdqsort(build_tmp_.begin(), build_tmp_.end());
            for (const auto& entry : build_tmp_) {
                if (entry.idx == i) {
                    U_diag_[i] = entry.val;
                } else if (entry.idx > i) {
                    U_upper_idx_.push_back(entry.idx);
                    U_upper_val_.push_back(entry.val);
                }
            }
            U_upper_ptr_[i + 1] = static_cast<int>(U_upper_idx_.size());

            // UT lower: remap physical→logical row indices using scratch, sort once (Item 4).
            UT_lower_ptr_[i] = static_cast<int>(UT_lower_idx_.size());
            build_tmp_.clear();
            for (const auto& entry : U_cols_[phys_col])
                build_tmp_.push_back({row_inv_[entry.idx], entry.val});
            pdqsort(build_tmp_.begin(), build_tmp_.end());
            for (const auto& entry : build_tmp_) {
                if (entry.idx < i) {
                    UT_lower_idx_.push_back(entry.idx);
                    UT_lower_val_.push_back(entry.val);
                } else {
                    break;
                }
            }
            UT_lower_ptr_[i + 1] = static_cast<int>(UT_lower_idx_.size());

            LT_upper_ptr_[i] = static_cast<int>(LT_upper_idx_.size());
            for (const auto& entry : L_cols_[i]) {
                if (entry.idx > i) {
                    LT_upper_idx_.push_back(entry.idx);
                    LT_upper_val_.push_back(entry.val);
                }
            }
            LT_upper_ptr_[i + 1] = static_cast<int>(LT_upper_idx_.size());
        }

        // Build elimination trees (Item 2): one entry per node encodes the primary
        // column-structure parent, enabling O(|reach|) path-tracing for tree-like factors.
        // l_etree_[j]  = min{i>j : L[i,j]!=0}  (= first entry in LT_upper row j, ascending)
        // u_etree_[j]  = max{i<j : U[i,j]!=0}  (= last  entry in UT_lower row j, ascending)
        // ut_etree_[j] = min{k>j : U[j,k]!=0}  (= first entry in U_upper row j, ascending)
        // lt_etree_[j] = max{k<j : L[j,k]!=0}  (= last  entry in L_lower row j, ascending)
        l_etree_.resize(n_);
        u_etree_.resize(n_);
        ut_etree_.resize(n_);
        lt_etree_.resize(n_);
        for (int j = 0; j < n_; ++j) {
            l_etree_[j] =
                (LT_upper_ptr_[j + 1] > LT_upper_ptr_[j]) ? LT_upper_idx_[LT_upper_ptr_[j]] : -1;
            u_etree_[j] = (UT_lower_ptr_[j + 1] > UT_lower_ptr_[j])
                              ? UT_lower_idx_[UT_lower_ptr_[j + 1] - 1]
                              : -1;
            ut_etree_[j] =
                (U_upper_ptr_[j + 1] > U_upper_ptr_[j]) ? U_upper_idx_[U_upper_ptr_[j]] : -1;
            lt_etree_[j] = (L_lower_ptr_[j + 1] > L_lower_ptr_[j])
                               ? L_lower_idx_[L_lower_ptr_[j + 1] - 1]
                               : -1;
        }

        // Pre-allocate hyper-sparse solve scratch; reset adaptive EMA on refactor (Item 9).
        reach_flag_scratch_.assign(n_, false);
        reach_scratch_.reserve(n_);
        dfs_stack_scratch_.reserve(n_);
        l_reach_seeds_scratch_.reserve(n_);
        perm_seeds_scratch_.reserve(n_);
        ema_reach_ratio_ = kHyperSparseDensityThreshold_;
        hyper_solve_reach_valid_ = false;

        // Pre-allocate persistent solve scratches so the per-solve hot path never
        // hits the heap. sparse_l/u_scratch_ are written by the kernels; output_scratch_
        // holds the final unpermuted result. last_solve_reach_original_ tracks which
        // entries of output_scratch_ were touched by the previous solve so we can
        // zero only those next time.
        sparse_l_scratch_.setZero(n_);
        sparse_u_scratch_.setZero(n_);
        output_scratch_.setZero(n_);
        last_solve_reach_original_.clear();
        last_solve_reach_original_.reserve(n_);

        // ================================================================
        // P0-2: Build LR (row-wise L^T) storage — HiGHS-style explicit row CSR
        //       for efficient BTRAN L hyper-sparse solves.
        //       LR_ptr_[i] = start offset (like HiGHS lr_start)
        //       LR_idx_[k]  = row index in L^T (= column index in L^T)
        //       LR_val_[k]  = corresponding value
        //       Row i of LR_ = Column i of L = L^T row i.
        //       This matches HiGHS's lr_start/lr_index/lr_value layout.
        // ================================================================
        LR_ptr_.assign(n_ + 1, 0);
        LR_idx_.clear();
        LR_val_.clear();
        for (int i = 0; i < n_; ++i) {
            LR_ptr_[i] = static_cast<int>(LR_idx_.size());
            // L_cols_[i] = column i of L: IndexedValue{row_idx, val}
            // For lower triangular L, row_idx >= i. Skip diagonal.
            for (const auto& entry : L_cols_[i]) {
                if (entry.idx > i) {
                    LR_idx_.push_back(entry.idx);
                    LR_val_.push_back(entry.val);
                }
            }
            LR_ptr_[i + 1] = static_cast<int>(LR_idx_.size());
        }

        // Build pivot lookup tables for hyper-sparse solves.
        // In our representation, L and U are stored row-wise with diagonals separate.
        // For hyper-sparse solves, we need pivot_lookup: physical → logical position.
        // Since L_diag_[i] and U_diag_[i] correspond to physical row i, the lookup is identity.
        L_pivot_lookup_.resize(n_);
        U_pivot_lookup_.resize(n_);
        for (int i = 0; i < n_; ++i) {
            L_pivot_lookup_[i] = i;
            U_pivot_lookup_[i] = i;
        }

        // Allocate hyper-sparse working buffers (Highs-style cwork/iwork).
        // cwork: char mark array for visited nodes
        // iwork: int array for indices and stack (size 4*n for safety)
        if (hyper_sparse_cwork_.size() < static_cast<size_t>(n_))
            hyper_sparse_cwork_.assign(n_, 0);
        if (hyper_sparse_iwork_.size() < static_cast<size_t>(n_ * 4))
            hyper_sparse_iwork_.assign(n_ * 4, 0);

        // Build CSC (column-oriented) H-factor structures for solveHyper_.
        // Convert L_lower (row CSR) → L_lower_col (column CSC)
        {
            const int nnz = static_cast<int>(L_lower_idx_.size());
            L_lower_col_ptr_.assign(n_ + 1, 0);
            // Count entries per column
            for (int pos = 0; pos < nnz; ++pos) {
                ++L_lower_col_ptr_[L_lower_idx_[pos] + 1];
            }
            // Prefix sum to get column offsets
            for (int j = 1; j <= n_; ++j) {
                L_lower_col_ptr_[j] += L_lower_col_ptr_[j - 1];
            }
            L_lower_col_idx_.resize(nnz);
            L_lower_col_val_.resize(nnz);
            std::vector<int> write_pos = L_lower_col_ptr_; // copy
            for (int row = 0; row < n_; ++row) {
                const int row_start = L_lower_ptr_[row];
                const int row_end = L_lower_ptr_[row + 1];
                for (int pos = row_start; pos < row_end; ++pos) {
                    const int col = L_lower_idx_[pos];
                    const int write_idx = write_pos[col]++;
                    L_lower_col_idx_[write_idx] = row;
                    L_lower_col_val_[write_idx] = L_lower_val_[pos];
                }
            }
        }
        // Convert U_upper (row CSR) → U_upper_col (column CSC)
        {
            const int nnz = static_cast<int>(U_upper_idx_.size());
            U_upper_col_ptr_.assign(n_ + 1, 0);
            for (int pos = 0; pos < nnz; ++pos) {
                ++U_upper_col_ptr_[U_upper_idx_[pos] + 1];
            }
            for (int j = 1; j <= n_; ++j) {
                U_upper_col_ptr_[j] += U_upper_col_ptr_[j - 1];
            }
            U_upper_col_idx_.resize(nnz);
            U_upper_col_val_.resize(nnz);
            std::vector<int> write_pos = U_upper_col_ptr_;
            for (int row = 0; row < n_; ++row) {
                const int row_start = U_upper_ptr_[row];
                const int row_end = U_upper_ptr_[row + 1];
                for (int pos = row_start; pos < row_end; ++pos) {
                    const int col = U_upper_idx_[pos];
                    const int write_idx = write_pos[col]++;
                    U_upper_col_idx_[write_idx] = row;
                    U_upper_col_val_[write_idx] = U_upper_val_[pos];
                }
            }
        }
    }

    void load_initial_U_(const SparseMat& A) {
        for (int k = 0; k < A.outerSize(); ++k) {
            for (typename SparseMat::InnerIterator it(A, k); it; ++it) {
                if (std::abs(it.value()) > kZeroTol_)
                    set_U_(it.row(), it.col(), it.value());
            }
        }
    }

    void rebuild_U_cols_() const {
        auto& cols = const_cast<std::vector<SparseRow>&>(U_cols_);
        cols.assign(n_, {});
        for (int phys_row = 0; phys_row < n_; ++phys_row) {
            for (const auto& entry : U_rows_[phys_row])
                set_entry_(cols[entry.idx], phys_row, entry.val);
        }
        const_cast<bool&>(U_cols_dirty_) = false;
    }

    void ensure_U_cols_ready_() const {
        if (!U_cols_dirty_)
            return;
        rebuild_U_cols_();
    }

    void rebuild_L_cols_() const {
        auto& cols = const_cast<std::vector<SparseRow>&>(L_cols_);
        cols.assign(n_, {});
        for (int phys_row = 0; phys_row < n_; ++phys_row) {
            for (const auto& entry : L_rows_[phys_row])
                set_entry_(cols[entry.idx], phys_row, entry.val);
        }
        const_cast<bool&>(L_cols_dirty_) = false;
    }

    void ensure_L_cols_ready_() const {
        if (!L_cols_dirty_)
            return;
        rebuild_L_cols_();
    }

    double get_U_(int i, int j) const { return get_entry_(U_rows_[row_map_[i]], col_map_[j]); }

    double get_L_(int i, int j) const { return get_entry_(L_rows_[i], j); }

    void set_U_(int i, int j, double v) {
        const int phys_row = row_map_[i];
        const int phys_col = col_map_[j];
        set_entry_(U_rows_[phys_row], phys_col, v);
        U_cols_dirty_ = true;
    }

    void set_L_(int i, int j, double v) {
        set_entry_(L_rows_[i], j, v);
        L_cols_dirty_ = true;
    }

    void swap_U_rows_(int a, int b) {
        if (a == b)
            return;

        std::swap(row_map_[a], row_map_[b]);
        row_inv_[row_map_[a]] = a;
        row_inv_[row_map_[b]] = b;
        std::swap(row_degree_[a], row_degree_[b]);
        invalidate_row_candidate_(a);
        invalidate_row_candidate_(b);
    }

    void swap_U_cols_(int a, int b) {
        if (a == b)
            return;

        std::swap(col_map_[a], col_map_[b]);
        col_inv_[col_map_[a]] = a;
        col_inv_[col_map_[b]] = b;
        std::swap(col_degree_[a], col_degree_[b]);
        std::swap(col_max_abs_[a], col_max_abs_[b]);
        const bool dirty_a = col_max_dirty_[a];
        col_max_dirty_[a] = col_max_dirty_[b];
        col_max_dirty_[b] = dirty_a;
        queue_column_candidate_invalidation_(a);
        queue_column_candidate_invalidation_(b);
    }

    // P2-2: Helper to inject identity column/row for rank deficiency.
    // Clears all entries in row k and column k of U (except diagonal),
    // sets diagonal to 1.0. Used when no acceptable pivot exists.
    void clear_U_row_col_(int k) {
        // Clear row k of U (all entries in U_rows_[row_map_[k]])
        const int phys_row = row_map_[k];
        U_rows_[phys_row].clear();
        U_cols_dirty_ = true;

        // Clear column k of U
        // U_cols_[col_map_[k]] contains (row, val) entries for column k
        const int phys_col = col_map_[k];
        U_cols_[phys_col].clear();

        // Set diagonal to 1
        set_U_(k, k, 1.0);

        // Also clear L column k (should be zero except diagonal)
        // L_rows_[k] contains entries L[k][j] for j < k (lower triangular)
        // For injected logical, L[k][j] = 0 for all j < k
        L_rows_[k].clear();
        L_cols_dirty_ = true;
    }

    void swap_L_prefix_rows_(int a, int b, int prefix_cols) {
        if (a == b || prefix_cols <= 0)
            return;

        for (int j = 0; j < prefix_cols; ++j) {
            const double va = get_L_(a, j);
            const double vb = get_L_(b, j);
            set_L_(a, j, vb);
            set_L_(b, j, va);
        }
    }

    std::pair<int, int> choose_pivot_sparse_(int k) {
        flush_column_candidate_invalidations_();

        if (const auto hint = symbolic_hint_pivot_(k); hint.has_value())
            return *hint;

        while (!row_candidate_heap_.empty()) {
            const RowCandidate candidate = row_candidate_heap_.top();
            row_candidate_heap_.pop();

            if (candidate.row < k || row_degree_[candidate.row] == 0)
                continue;

            if (row_candidate_dirty_[candidate.row]) {
                recompute_row_candidate_(candidate.row);
                continue;
            }

            if (candidate.version != row_candidate_version_[candidate.row] ||
                candidate.col != row_candidate_best_col_[candidate.row] ||
                candidate.score != row_candidate_best_score_[candidate.row]) {
                continue;
            }

            const double colmax = active_col_max_(candidate.col);
            if (row_candidate_best_abs_[candidate.row] <
                pivot_rel_ * std::max(colmax, abs_floor_)) {
                invalidate_row_candidate_(candidate.row);
                recompute_row_candidate_(candidate.row);
                continue;
            }

            if (candidate.score <= kEarlyAcceptMarkowitzScore_ &&
                row_candidate_best_abs_[candidate.row] >=
                    kEarlyAcceptPivotRatio_ * std::max(colmax, abs_floor_)) {
                return {candidate.row, candidate.col};
            }

            return {candidate.row, candidate.col};
        }

        ensure_U_cols_ready_();
        int i = k;
        double best_in_col = -1.0;
        for (const auto& [phys_row, val] : U_cols_[col_map_[k]]) {
            const int logical_row = row_inv_[phys_row];
            if (logical_row < k)
                continue;
            const double ab = std::abs(val);
            if (ab > best_in_col) {
                best_in_col = ab;
                i = logical_row;
            }
        }

        if (best_in_col <= abs_floor_)
            return {-1, -1};

        int j = k;
        for (int t = 0; t < std::max(1, rook_iters_); ++t) {
            double best_row = -1.0;
            for (const auto& [phys_col, val] : U_rows_[row_map_[i]]) {
                const int logical_col = col_inv_[phys_col];
                if (logical_col < k)
                    continue;
                const double ab = std::abs(val);
                if (ab > best_row) {
                    best_row = ab;
                    j = logical_col;
                }
            }

            int new_i = i;
            double best_col = -1.0;
            ensure_U_cols_ready_();
            for (const auto& [phys_row, val] : U_cols_[col_map_[j]]) {
                const int logical_row = row_inv_[phys_row];
                if (logical_row < k)
                    continue;
                const double ab = std::abs(val);
                if (ab > best_col) {
                    best_col = ab;
                    new_i = logical_row;
                }
            }

            if (new_i == i)
                break;
            i = new_i;
        }

        return {i, j};
    }

    void factorize_sparse_() {
        for (int k = 0; k < n_; ++k) {
            // P1-1: Check for column singletons before full Markowitz search.
            // A column with exactly 1 active entry can be pivoted immediately,
            // skipping the O(n) Markowitz search. This matches HiGHS's
            // buildSimple() singleton extraction pass.
            bool found_singleton = false;
            int singleton_row = -1;
            int singleton_col = -1;
            ensure_U_cols_ready_();
            for (int col = k; col < n_; ++col) {
                if (col_degree_[col] == 1) {
                    // Column singleton: find the row with the active entry
                    for (const auto& [phys_row, val] : U_cols_[col]) {
                        const int logical_row = row_inv_[phys_row];
                        if (logical_row >= k && std::abs(val) > abs_floor_) {
                            singleton_row = logical_row;
                            singleton_col = col;
                            found_singleton = true;
                            break;
                        }
                    }
                    if (found_singleton)
                        break;
                }
            }
            int pi, pj;
            if (found_singleton) {
                pi = singleton_row;
                pj = singleton_col;
                ++num_col_singleton_pivots_;
            } else {
                auto [candidate_pi, candidate_pj] = choose_pivot_sparse_(k);
                pi = candidate_pi;
                pj = candidate_pj;
                ++num_markowitz_pivots_;
            }

            // P2-2: Rank deficiency handling. If no pivot found or singular,
            // inject a logical (identity) column for this row. This matches
            // HiGHS's buildHandleRankDeficiency: reorder basis so singular
            // columns are replaced by logicals.
            bool injected_logical = false;
            if (pi < 0 || pj < 0) {
                // No pivot found — inject identity column for row k
                // Find an unused column (one not yet in basis)
                injected_logical = true;
                --rank_deficiency_;
                // Record this row as having no pivot
                row_with_no_pivot_.push_back(k);
                col_with_no_pivot_.push_back(-1); // no column to pair
                // Swap row k with itself, column k with itself
                pi = k;
                pj = k;
            }

            swap_U_rows_(k, pi);
            swap_U_cols_(k, pj);
            swap_L_prefix_rows_(k, pi, k);
            const int old_Pr_k = Pr_[k];
            const int old_Pr_pi = Pr_[pi];
            const int old_Pc_k = Pc_[k];
            const int old_Pc_pj = Pc_[pj];

            std::swap(Pr_[k], Pr_[pi]);
            std::swap(Pc_[k], Pc_[pj]);
            std::swap(Pr_inv_[old_Pr_k], Pr_inv_[old_Pr_pi]);
            std::swap(Pc_inv_[old_Pc_k], Pc_inv_[old_Pc_pj]);

            // P1-2: Record pivot info for refactor cache (HiGHS-style)
            // pivot_row[k] = logical row index after swap (= k)
            // pivot_var[k] = original variable index = Pc_[k] (after swap)
            // pivot_type[k] = 0 = Markowitz, 1 = Unit, 2 = ColSingleton, 3 = Logical(injected)
            if (refactor_info_.pivot_row.size() < static_cast<size_t>(n_)) {
                refactor_info_.pivot_row.push_back(k);
                refactor_info_.pivot_var.push_back(Pc_[k]);
                if (injected_logical)
                    refactor_info_.pivot_type.push_back(3); // 3 = Logical injection
                else if (found_singleton)
                    refactor_info_.pivot_type.push_back(2);
                else
                    refactor_info_.pivot_type.push_back(0);
            }

            const double piv = get_U_(k, k);
            // If pivot is singular after injection (or was already singular),
            // inject identity: set diagonal to 1, clear off-diagonals
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) {
                if (!injected_logical) {
                    // Pivot is singular — inject identity for this row/column
                    injected_logical = true;
                    --rank_deficiency_;
                    row_with_no_pivot_.push_back(k);
                    col_with_no_pivot_.push_back(-1);
                    // Clear row k and column k of U, set diagonal to 1
                    clear_U_row_col_(k);
                    set_U_(k, k, 1.0);
                }
                // If we already injected logical, U_(k,k) is already 1.0
            }

            set_L_(k, k, 1.0);
            ensure_U_cols_ready_();
            affected_rows_scratch_.clear();
            for (const auto& [phys_row, val] : U_cols_[col_map_[k]]) {
                const int logical_row = row_inv_[phys_row];
                if (logical_row > k && std::abs(val) > kZeroTol_)
                    affected_rows_scratch_.push_back(logical_row);
            }

            for (const int i : affected_rows_scratch_) {
                const double uik = get_U_(i, k);
                if (std::abs(uik) <= kZeroTol_)
                    continue;

                const double lik = uik / piv;
                set_L_(i, k, lik);
                merge_update_U_row_active_(i, col_map_[k], lik, U_rows_[row_map_[k]]);
            }

            finalize_pivot_step_(k);
        }
        rebuild_perm_inverses_();
    }

    // P2-2: Complete rank-deficient basis by injecting identity columns for
    //        any remaining rows without pivots. This matches HiGHS's
    //        buildHandleRankDeficiency: inject logical columns for singular rows.
    void complete_rank_deficient_basis_() {
        if (rank_deficiency_ <= 0)
            return;

        // Find rows that still need pivots (those not in row_with_no_pivot_)
        std::vector<bool> has_pivot(n_, false);
        for (int k = 0; k < n_; ++k) {
            if (!refactor_info_.pivot_row.empty() &&
                static_cast<size_t>(k) < refactor_info_.pivot_row.size()) {
                has_pivot[refactor_info_.pivot_row[k]] = true;
            }
        }

        // For each row without a pivot, inject an identity column
        int injected = 0;
        for (int k = 0; k < n_ && rank_deficiency_ > 0; ++k) {
            if (!has_pivot[k]) {
                // Inject identity for row k
                has_pivot[k] = true;
                clear_U_row_col_(k);
                set_L_(k, k, 1.0);

                // Record in refactor info
                if (refactor_info_.pivot_row.size() < static_cast<size_t>(n_)) {
                    refactor_info_.pivot_row.push_back(k);
                    refactor_info_.pivot_var.push_back(-1); // no variable
                    refactor_info_.pivot_type.push_back(3); // 3 = Logical injection
                }

                row_with_no_pivot_.push_back(k);
                col_with_no_pivot_.push_back(-1);
                ++injected;
                --rank_deficiency_;
            }
        }
        (void)injected;
    }

    void activate_sparse_lu_fallback_(const SparseMat& A) {
        fallback_sparse_lu_.analyzePattern(A);
        fallback_sparse_lu_.factorize(A);
        if (fallback_sparse_lu_.info() != Eigen::Success)
            throw std::runtime_error("SparseForrestTomlinLU: sparse fallback factorization failed");

        const SparseMat AT = A.transpose();
        fallback_sparse_lu_t_.analyzePattern(AT);
        fallback_sparse_lu_t_.factorize(AT);
        if (fallback_sparse_lu_t_.info() != Eigen::Success) {
            throw std::runtime_error(
                "SparseForrestTomlinLU: sparse transpose fallback factorization failed");
        }

        use_fallback_sparse_lu_ = true;
    }

    // Highs-style hyper-sparse solve using iterative deepening with domination checking.
    // Uses cwork (char mark array) and iwork (int stack/index array) like Highs.
    // Template parameter Phase: 0 = forward L (ascending), 1 = backward U (descending).
    // h_size: dimension of the factor (n_)
    // pivot_lookup: maps from H-index to position (inverse of pivot_index)
    // pivot_index: the pivot row/col indices
    // pivot_value: diagonal values (nullptr for L which has implicit 1s)
    // h_ptr: column pointer array of length h_size+1 (CSC format)
    // h_index, h_value: the sparse column data
    // x: RHS/solution vector (modified in place for RHS, returned as solution)
    template <int Phase>
    void solveHyper_(int h_size, const int* pivot_lookup, const int* pivot_index,
                     const double* pivot_value, const int* h_ptr, const int* h_index,
                     const double* h_value, double* x_array,
                     const std::vector<int>* seed_index_hint = nullptr) const {
        // Build reach set using Highs-style iterative deepening
        char* list_mark = hyper_sparse_cwork_.data();
        int* list_index = hyper_sparse_iwork_.data();
        int* list_stack = &hyper_sparse_iwork_[h_size];
        int* seed_index = &hyper_sparse_iwork_[3 * h_size];
        int list_count = 0;

        // Collect RHS nonzeros as seeds
        int rhs_count = 0;
        if (seed_index_hint != nullptr) {
            for (const int i : *seed_index_hint) {
                if (i >= 0 && i < h_size && std::abs(x_array[i]) > kSparseTiny_)
                    seed_index[rhs_count++] = i;
            }
        } else {
            for (int i = 0; i < h_size; ++i) {
                if (std::abs(x_array[i]) > kSparseTiny_)
                    seed_index[rhs_count++] = i;
            }
        }
        reach_scratch_.clear();

        int count_pivot = 0;
        int count_entry = 0;

        // Iterative deepening traversal with domination checking
        for (int i = 0; i < rhs_count; ++i) {
            int i_trans = pivot_lookup[seed_index[i]];
            if (list_mark[i_trans])
                continue; // Domination check: already visited

            int Hi = i_trans;
            int Hk = h_ptr[Hi];
            int n_stack = -1;

            list_mark[Hi] = 1;

            for (;;) {
                if (Hk < h_ptr[Hi + 1]) {
                    int Hi_sub = pivot_lookup[h_index[Hk++]];
                    if (list_mark[Hi_sub] == 0) {
                        list_mark[Hi_sub] = 1;
                        list_stack[++n_stack] = Hi;
                        list_stack[++n_stack] = Hk;
                        Hi = Hi_sub;
                        Hk = h_ptr[Hi];
                        if (Hi >= h_size) {
                            count_pivot++;
                            count_entry += h_ptr[Hi + 1] - h_ptr[Hi];
                        }
                    }
                } else {
                    list_index[list_count++] = Hi;
                    if (n_stack == -1)
                        break;
                    Hk = list_stack[n_stack--];
                    Hi = list_stack[n_stack--];
                }
            }
        }

        // Update synthetic tick with Highs-style weights
        synthetic_tick_ += count_pivot * 20 + count_entry * 10;

        // Solve with the collected list
        if (pivot_value == nullptr) {
            // L factor: diagonal is implicitly 1
            int new_count = 0;
            for (int iList = list_count - 1; iList >= 0; --iList) {
                int i = list_index[iList];
                list_mark[i] = 0;
                int pivot_row = pivot_index[i];
                double pivot_multiplier = x_array[pivot_row];
                if (std::abs(pivot_multiplier) > kSparseTiny_) {
                    reach_scratch_.push_back(pivot_row);
                    list_index[new_count++] = pivot_row;
                    const int start = h_ptr[i];
                    const int end = h_ptr[i + 1];
                    for (int k = start; k < end; ++k)
                        x_array[h_index[k]] -= pivot_multiplier * h_value[k];
                } else {
                    x_array[pivot_row] = 0;
                }
            }
        } else {
            // U factor: has explicit diagonal
            int new_count = 0;
            for (int iList = list_count - 1; iList >= 0; --iList) {
                int i = list_index[iList];
                list_mark[i] = 0;
                int pivot_row = pivot_index[i];
                double pivot_multiplier = x_array[pivot_row];
                if (std::abs(pivot_multiplier) > kSparseTiny_) {
                    pivot_multiplier /= pivot_value[i];
                    x_array[pivot_row] = pivot_multiplier;
                    reach_scratch_.push_back(pivot_row);
                    list_index[new_count++] = pivot_row;
                    const int start = h_ptr[i];
                    const int end = h_ptr[i + 1];
                    for (int k = start; k < end; ++k)
                        x_array[h_index[k]] -= pivot_multiplier * h_value[k];
                } else {
                    x_array[pivot_row] = 0;
                }
            }
        }
    }

    Eigen::VectorXd forward_solve_L_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = b;
        for (int i = 0; i < n_; ++i) {
            double s = 0.0;
            for (int pos = L_lower_ptr_[i]; pos < L_lower_ptr_[i + 1]; ++pos) {
                s += L_lower_val_[pos] * x(L_lower_idx_[pos]);
            }
            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }

    // True hyper-sparse forward solve Lx = b.
    // Hyper-sparse forward solve Lx = b.
    //
    // Reach computation (Item 2): etree path tracing instead of full-column DFS.
    //   Phase 1 — for each seed, walk l_etree_ upward until hitting a visited node.
    //             This is O(|reach|) for tree-like L (single entry per column), which
    //             covers the common case in LP basis factors.
    //   Phase 2 — handle fill-in: columns with >1 entry push their extra entries back
    //             through the same tracer.  Falls back gracefully for dense columns.
    //
    // seeds param (Item 1): when non-null, seeds the reach directly without O(n) scan.
    //   The L-reach is left in reach_scratch_ so the caller can chain it into U solve.
    //
    // EMA update (Item 9): tracks reach density to drive the adaptive fallback.
    //
    // In-place variant: writes the solve result into x_out (must be sized n_ and zero
    // at every position the caller hasn't already populated). Caller is responsible
    // for clearing touched entries after consumption — see clear_scratch_at_reach_.
    void forward_solve_L_sparse_inplace_(const Eigen::VectorXd& b, const std::vector<int>* seeds,
                                         Eigen::VectorXd& x_out) const {
        if (n_ == 0) [[unlikely]]
            return;
        SIMPLEX_ASSUME(n_ > 0);

        clear_reach_flags_scratch_(seeds);
        dfs_stack_scratch_.clear();

        // Etree path tracer: walk l_etree_ from start, queueing fill-in nodes (Phase 2).
        const auto trace_L = [&](int start) {
            for (int j = start; j >= 0 && !reach_flag_scratch_[j]; j = l_etree_[j]) {
                reach_flag_scratch_[j] = true;
                reach_scratch_.push_back(j);
                if (LT_upper_ptr_[j + 1] - LT_upper_ptr_[j] > 1)
                    dfs_stack_scratch_.push_back(j); // has extra fill-in entries
            }
        };

        // Phase 1: etree paths from seeds.
        if (seeds) {
            for (const int k : *seeds)
                trace_L(k);
        } else {
            for (int k = 0; k < n_; ++k)
                if (std::abs(b(k)) > kSparseTiny_)
                    trace_L(k);
        }

        // Phase 2: expand non-etree fill-in (extra column entries beyond the single parent).
        while (!dfs_stack_scratch_.empty()) {
            const int j = dfs_stack_scratch_.back();
            dfs_stack_scratch_.pop_back();
            for (int pos = LT_upper_ptr_[j] + 1; pos < LT_upper_ptr_[j + 1]; ++pos)
                trace_L(LT_upper_idx_[pos]);
        }

        pdqsort(reach_scratch_.begin(), reach_scratch_.end());

        // Item 9: update EMA of reach density for adaptive threshold.
        ema_reach_ratio_ +=
            0.1 * (static_cast<double>(reach_scratch_.size()) / n_ - ema_reach_ratio_);

        for (const int i : reach_scratch_) {
            double rhs = b(i);
            for (int pos = L_lower_ptr_[i]; pos < L_lower_ptr_[i + 1]; ++pos)
                rhs -= L_lower_val_[pos] * x_out(L_lower_idx_[pos]);
            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            x_out(i) = rhs / piv;
        }
    }

    Eigen::VectorXd back_solve_U_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = b;
        for (int i = n_ - 1; i >= 0; --i) {
            double s = 0.0;
            for (int pos = U_upper_ptr_[i]; pos < U_upper_ptr_[i + 1]; ++pos) {
                s += U_upper_val_[pos] * x(U_upper_idx_[pos]);
            }

            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }

    // Hyper-sparse back solve Ux = b.
    // u_etree_[j] = max{i<j : U[i,j]!=0} — backward propagation follows decreasing indices.
    // Phase 1: walk u_etree_ downward from each seed.
    // Phase 2: extra entries in column j of U (all except the last = etree parent) are below
    //          u_etree_[j] and also need tracing.
    // seeds: when provided (L-reach from prior solve), no O(n) scan needed (Item 1).
    //
    // In-place variant: writes into x_out (must be zero at uninitialized positions).
    void back_solve_U_sparse_inplace_(const Eigen::VectorXd& b, const std::vector<int>* seeds,
                                      Eigen::VectorXd& x_out) const {
        if (n_ == 0) [[unlikely]]
            return;
        SIMPLEX_ASSUME(n_ > 0);

        clear_reach_flags_scratch_(seeds);
        dfs_stack_scratch_.clear();

        const auto trace_U = [&](int start) {
            for (int j = start; j >= 0 && !reach_flag_scratch_[j]; j = u_etree_[j]) {
                reach_flag_scratch_[j] = true;
                reach_scratch_.push_back(j);
                if (UT_lower_ptr_[j + 1] - UT_lower_ptr_[j] > 1)
                    dfs_stack_scratch_.push_back(j);
            }
        };

        if (seeds) {
            for (const int k : *seeds)
                trace_U(k);
        } else {
            for (int k = 0; k < n_; ++k)
                if (std::abs(b(k)) > kSparseTiny_)
                    trace_U(k);
        }

        // Extra fill-in: all entries in column j of U *except* the last (= u_etree_[j]).
        while (!dfs_stack_scratch_.empty()) {
            const int j = dfs_stack_scratch_.back();
            dfs_stack_scratch_.pop_back();
            for (int pos = UT_lower_ptr_[j]; pos < UT_lower_ptr_[j + 1] - 1; ++pos)
                trace_U(UT_lower_idx_[pos]);
        }

        pdqsort(reach_scratch_.begin(), reach_scratch_.end(), std::greater<int>{});

        for (const int i : reach_scratch_) {
            double rhs = b(i);
            for (int pos = U_upper_ptr_[i]; pos < U_upper_ptr_[i + 1]; ++pos)
                rhs -= U_upper_val_[pos] * x_out(U_upper_idx_[pos]);
            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            x_out(i) = rhs / piv;
        }
    }

    Eigen::VectorXd forward_solve_UT_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = b;
        for (int i = 0; i < n_; ++i) {
            double s = 0.0;
            for (int pos = UT_lower_ptr_[i]; pos < UT_lower_ptr_[i + 1]; ++pos) {
                s += UT_lower_val_[pos] * x(UT_lower_idx_[pos]);
            }
            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }

    // Hyper-sparse forward solve U^T x = b (U^T lower triangular).
    // ut_etree_[j] = min{k>j : U[j,k]!=0} — forward propagation follows increasing indices.
    // Phase 2 extra fill-in: entries in row j of U beyond the first (= ut_etree_[j]).
    // seeds (Item 1): UT-reach left in reach_scratch_ for chaining into LT solve.
    //
    // In-place variant: writes into x_out (must be zero at uninitialized positions).
    void forward_solve_UT_sparse_inplace_(const Eigen::VectorXd& b, const std::vector<int>* seeds,
                                          Eigen::VectorXd& x_out) const {
        if (n_ == 0) [[unlikely]]
            return;
        SIMPLEX_ASSUME(n_ > 0);

        clear_reach_flags_scratch_(seeds);
        dfs_stack_scratch_.clear();

        const auto trace_UT = [&](int start) {
            for (int j = start; j >= 0 && !reach_flag_scratch_[j]; j = ut_etree_[j]) {
                reach_flag_scratch_[j] = true;
                reach_scratch_.push_back(j);
                if (U_upper_ptr_[j + 1] - U_upper_ptr_[j] > 1)
                    dfs_stack_scratch_.push_back(j);
            }
        };

        if (seeds) {
            for (const int k : *seeds)
                trace_UT(k);
        } else {
            for (int k = 0; k < n_; ++k)
                if (std::abs(b(k)) > kSparseTiny_)
                    trace_UT(k);
        }

        // Extra fill-in: entries in row j of U beyond the first (ut_etree_[j]).
        while (!dfs_stack_scratch_.empty()) {
            const int j = dfs_stack_scratch_.back();
            dfs_stack_scratch_.pop_back();
            for (int pos = U_upper_ptr_[j] + 1; pos < U_upper_ptr_[j + 1]; ++pos)
                trace_UT(U_upper_idx_[pos]);
        }

        pdqsort(reach_scratch_.begin(), reach_scratch_.end());

        ema_reach_ratio_ +=
            0.1 * (static_cast<double>(reach_scratch_.size()) / n_ - ema_reach_ratio_);

        for (const int i : reach_scratch_) {
            double rhs = b(i);
            for (int pos = UT_lower_ptr_[i]; pos < UT_lower_ptr_[i + 1]; ++pos)
                rhs -= UT_lower_val_[pos] * x_out(UT_lower_idx_[pos]);
            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            x_out(i) = rhs / piv;
        }
    }

    Eigen::VectorXd back_solve_LT_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = b;
        for (int i = n_ - 1; i >= 0; --i) {
            double s = 0.0;
            for (int pos = LT_upper_ptr_[i]; pos < LT_upper_ptr_[i + 1]; ++pos) {
                s += LT_upper_val_[pos] * x(LT_upper_idx_[pos]);
            }

            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            x(i) = (x(i) - s) / piv;
        }
        return x;
    }

    // Hyper-sparse back solve L^T x = b (L^T upper triangular).
    // lt_etree_[j] = max{k<j : L[j,k]!=0} — backward propagation follows decreasing indices.
    // Phase 2 extra fill-in: entries in row j of L *except* the last (= lt_etree_[j]).
    // seeds (Item 1): UT-reach from prior solve, no O(n) scan needed.
    //
    // In-place variant: writes into x_out (must be zero at uninitialized positions).
    void back_solve_LT_sparse_inplace_(const Eigen::VectorXd& b, const std::vector<int>* seeds,
                                       Eigen::VectorXd& x_out) const {
        if (n_ == 0) [[unlikely]]
            return;
        SIMPLEX_ASSUME(n_ > 0);

        clear_reach_flags_scratch_(seeds);
        dfs_stack_scratch_.clear();

        const auto trace_LT = [&](int start) {
            for (int j = start; j >= 0 && !reach_flag_scratch_[j]; j = lt_etree_[j]) {
                reach_flag_scratch_[j] = true;
                reach_scratch_.push_back(j);
                if (L_lower_ptr_[j + 1] - L_lower_ptr_[j] > 1)
                    dfs_stack_scratch_.push_back(j);
            }
        };

        if (seeds) {
            for (const int k : *seeds)
                trace_LT(k);
        } else {
            for (int k = 0; k < n_; ++k)
                if (std::abs(b(k)) > kSparseTiny_)
                    trace_LT(k);
        }

        // Extra fill-in: entries in row j of L except the last (= lt_etree_[j]).
        while (!dfs_stack_scratch_.empty()) {
            const int j = dfs_stack_scratch_.back();
            dfs_stack_scratch_.pop_back();
            for (int pos = L_lower_ptr_[j]; pos < L_lower_ptr_[j + 1] - 1; ++pos)
                trace_LT(L_lower_idx_[pos]);
        }

        pdqsort(reach_scratch_.begin(), reach_scratch_.end(), std::greater<int>{});

        for (const int i : reach_scratch_) {
            double rhs = b(i);
            for (int pos = LT_upper_ptr_[i]; pos < LT_upper_ptr_[i + 1]; ++pos)
                rhs -= LT_upper_val_[pos] * x_out(LT_upper_idx_[pos]);
            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            x_out(i) = rhs / piv;
        }
    }

    Eigen::VectorXd apply_updates_solve_(Eigen::VectorXd x) const {
        if (!pf_pivot_index_.empty())
            return solve_with_PF_(x);
        for (const auto& update : updates_) {
            // If we have a valid hyper-sparse reach, skip updates where x(j) is
            // guaranteed zero (j not in reach set).
            if (hyper_solve_reach_valid_ && !reach_flag_scratch_[update.j])
                continue;
            const double xj = x(update.j);
            if (xj != 0.0)
                update.z.axpy(x, -(xj / update.alpha));
        }
        return x;
    }

    // Apply updates using Product Form (PF) storage.
    // Each PF update stores: pivot_row, packed column (excluding pivot row), pivot_value.
    // For PF: x[pivot_row] /= pivot_val, then x[row] -= multiplier * PF_value[row].
    Eigen::VectorXd solve_with_PF_(Eigen::VectorXd x) const {
        const int num_updates = static_cast<int>(pf_pivot_index_.size());
        if (num_updates == 0)
            return x;
        for (int k = 0; k < num_updates; ++k) {
            const int pivot_row = pf_pivot_index_[k];
            const double pivot_val = pf_pivot_value_[k];
            if (std::abs(pivot_val) < kSparseTiny_)
                continue;
            const int start = pf_start_[k];
            const int end =
                (k + 1 < num_updates) ? pf_start_[k + 1] : static_cast<int>(pf_index_.size());
            double xj = x(pivot_row);
            if (std::abs(xj) < kSparseTiny_)
                continue;
            const double multiplier = xj / pivot_val;
            x(pivot_row) = multiplier;
            for (int p = start; p < end; ++p) {
                const int row = pf_index_[p];
                if (row != pivot_row) {
                    x(row) -= multiplier * pf_value_[p];
                }
            }
        }
        return x;
    }

    Eigen::VectorXd apply_updates_solve_T_(Eigen::VectorXd y) const {
        // Use regular updates for transpose solves even when PF forward updates
        // are also present, to avoid relying on incomplete PF transpose logic.
        for (const auto& update : updates_) {
            // For the transposed case, skip if none of u's support overlaps the
            // reach set — u.dot(y) must be zero if all u(i) positions are outside
            // the reach.
            if (hyper_solve_reach_valid_) {
                bool any_in_reach = false;
                for (int k = 0; k < static_cast<int>(update.u.idx.size()); ++k) {
                    if (reach_flag_scratch_[update.u.idx[k]]) {
                        any_in_reach = true;
                        break;
                    }
                }
                if (!any_in_reach)
                    continue;
            }
            const double uy = update.u.dot(y);
            if (uy != 0.0)
                update.w.axpy(y, -(uy / update.alpha));
        }
        return y;
    }

    // Transpose solve with Product Form updates.
    // PF update: B_new = B * (I - (1/alpha) * z * e_r^T)
    // Transpose: B_new^T = (I - (1/alpha) * e_r * z^T) * B^T
    // For PF (packed column form): y_new[pivot] = y[pivot]/pivot_val,
    //   y_new[row] = y[row] - y_new[pivot] * PF_value[row]
    Eigen::VectorXd solve_with_PF_T_(Eigen::VectorXd y) const {
        const int num_updates = static_cast<int>(pf_pivot_index_.size());
        if (num_updates == 0)
            return y;
        // Apply in reverse order (reverse sweep matches transpose operation order)
        for (int k = num_updates - 1; k >= 0; --k) {
            const int pivot_row = pf_pivot_index_[k];
            const double pivot_val = pf_pivot_value_[k];
            if (std::abs(pivot_val) < kSparseTiny_)
                continue;
            const int start = pf_start_[k];
            const int end =
                (k + 1 < num_updates) ? pf_start_[k + 1] : static_cast<int>(pf_index_.size());
            // First compute the contribution from all non-pivot rows to pivot row
            // (these were accumulated in y[pivot_row] during forward solve)
            double yp = y(pivot_row);
            // Compute the correction for pivot row
            double correction = 0.0;
            for (int p = start; p < end; ++p) {
                const int row = pf_index_[p];
                if (row != pivot_row) {
                    correction += pf_value_[p] * y(row);
                }
            }
            y(pivot_row) = (yp - correction) / pivot_val;
        }
        return y;
    }

    bool validate_sparse_rhs_solution_(const Eigen::VectorXd& rhs, const Eigen::VectorXd& x) const {
        if (!x.array().isFinite().all())
            return false;
        const Eigen::VectorXd residual = rhs - multiply_current_matrix_(x);
        const double max_rhs = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
        return residual.array().isFinite().all() &&
               residual.lpNorm<Eigen::Infinity>() <= 1e-8 * max_rhs;
    }

    bool validate_sparse_transpose_rhs_solution_(const Eigen::VectorXd& rhs,
                                                 const Eigen::VectorXd& y) const {
        if (!y.array().isFinite().all())
            return false;
        const Eigen::VectorXd residual = rhs - multiply_current_matrix_T_(y);
        const double max_rhs = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
        return residual.array().isFinite().all() &&
               residual.lpNorm<Eigen::Infinity>() <= 1e-8 * max_rhs;
    }

    Eigen::VectorXd multiply_current_matrix_(const Eigen::VectorXd& x) const {
        Eigen::VectorXd out = base_matrix_original_ * x;
        for (const auto& update : updates_) {
            const double xj = x(update.j);
            if (xj != 0.0)
                update.u.axpy(out, xj);
        }
        return out;
    }

    Eigen::VectorXd multiply_current_matrix_T_(const Eigen::VectorXd& y) const {
        Eigen::VectorXd out = base_matrix_original_.transpose() * y;
        for (const auto& update : updates_)
            out(update.j) += update.u.dot(y);
        return out;
    }

    Eigen::VectorXd iterative_refine_(const Eigen::VectorXd& rhs, Eigen::VectorXd x) const {
        const int max_steps = std::max(0, config_.iterative_refinement_steps);
        double previous_rel_residual = std::numeric_limits<double>::infinity();
        for (int step = 0; step < max_steps; ++step) {
            const Eigen::VectorXd residual = rhs - multiply_current_matrix_(x);
            const double rel_residual =
                residual.lpNorm<Eigen::Infinity>() / std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
            if (!std::isfinite(rel_residual) || rel_residual <= config_.iterative_refinement_tol) {
                break;
            }
            if (std::isfinite(previous_rel_residual) &&
                rel_residual > previous_rel_residual * 0.95) {
                break;
            }
            // Refinement residuals: force the dense-result path so iterative
            // refinement stays robust regardless of the caller's expected_density.
            Eigen::VectorXd dx = solve_impl_(residual, false, 1.0);
            if (!dx.array().isFinite().all() || dx.lpNorm<Eigen::Infinity>() < 1e-16)
                break;
            x += dx;
            previous_rel_residual = rel_residual;
        }
        return x;
    }

    Eigen::VectorXd iterative_refine_T_(const Eigen::VectorXd& rhs, Eigen::VectorXd y) const {
        const int max_steps = std::max(0, config_.iterative_refinement_steps);
        double previous_rel_residual = std::numeric_limits<double>::infinity();
        for (int step = 0; step < max_steps; ++step) {
            const Eigen::VectorXd residual = rhs - multiply_current_matrix_T_(y);
            const double rel_residual =
                residual.lpNorm<Eigen::Infinity>() / std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
            if (!std::isfinite(rel_residual) || rel_residual <= config_.iterative_refinement_tol) {
                break;
            }
            if (std::isfinite(previous_rel_residual) &&
                rel_residual > previous_rel_residual * 0.95) {
                break;
            }
            Eigen::VectorXd dy = solveT_impl_(residual, false, 1.0);
            if (!dy.array().isFinite().all() || dy.lpNorm<Eigen::Infinity>() < 1e-16)
                break;
            y += dy;
            previous_rel_residual = rel_residual;
        }
        return y;
    }

    void update_norm_growth_estimate_(const SparseUpdate& update) {
        const double denom = std::max({1.0, base_matrix_one_norm_, std::abs(update.alpha)});
        const double proxy =
            1.0 + std::min(1e3, (update.u.one_norm() + update.z.one_norm() + update.w.one_norm()) /
                                    denom);
        norm_growth_estimate_ *= proxy;
    }

    void update_cached_stats_(const SparseUpdate& update) {
        const double z_inf = update.z.inf_norm();
        const double w_inf = update.w.inf_norm();
        updates_count_ += 1;
        updates_max_z_inf_ = std::max(updates_max_z_inf_, z_inf);
        updates_max_w_inf_ = std::max(updates_max_w_inf_, w_inf);
        updates_cumulative_z_inf_ += z_inf;
        updates_density_sum_ += update.z.density(n_);
    }

  private:
    int n_{0};
    double pivot_rel_{1e-12};
    double abs_floor_{1e-16};
    int rook_iters_{2};
    Config config_{};
    double base_matrix_one_norm_{1.0};
    double norm_growth_estimate_{1.0};
    bool use_fallback_sparse_lu_{false};
    std::vector<double> row_scale_, col_scale_;
    SparseMat base_matrix_original_;
    Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> fallback_sparse_lu_;
    Eigen::SparseLU<SparseMat, Eigen::COLAMDOrdering<int>> fallback_sparse_lu_t_;
    mutable Eigen::VectorXd permuted_rhs_scratch_;
    mutable Eigen::VectorXd permuted_transpose_rhs_scratch_;
    mutable int active_k_{0};
    mutable std::vector<int> row_degree_, col_degree_;
    mutable std::vector<double> col_max_abs_;
    mutable std::vector<bool> col_max_dirty_;
    std::vector<int> row_candidate_best_col_;
    std::vector<long> row_candidate_best_score_;
    std::vector<double> row_candidate_best_abs_;
    std::vector<int> row_candidate_version_;
    std::vector<bool> row_candidate_dirty_;
    std::vector<bool> col_candidate_dirty_;
    std::vector<int> dirty_cols_scratch_;
    std::vector<PatternSet> legacy_pattern_rows_scratch_;
    std::vector<PatternSet> legacy_pattern_cols_scratch_;
    std::vector<int> sym_row_map_scratch_;
    std::vector<int> sym_col_map_scratch_;
    std::vector<int> sym_row_inv_scratch_;
    std::vector<int> sym_col_inv_scratch_;
    std::vector<int> pivot_row_cols_scratch_;
    std::vector<int> symbolic_affected_rows_scratch_;
    std::vector<int> symbolic_row_hint_phys_, symbolic_col_hint_phys_;
    std::priority_queue<RowCandidate, std::vector<RowCandidate>, RowCandidateGreater>
        row_candidate_heap_;
    std::vector<int> row_map_, col_map_;
    std::vector<int> row_inv_, col_inv_;
    std::vector<double> L_diag_, U_diag_;
    std::vector<int> L_lower_ptr_, U_upper_ptr_, UT_lower_ptr_, LT_upper_ptr_;
    std::vector<int> L_lower_idx_, U_upper_idx_, UT_lower_idx_, LT_upper_idx_;
    std::vector<double> L_lower_val_, U_upper_val_, UT_lower_val_, LT_upper_val_;
    // Column-oriented (CSC) H-factor structures for solveHyper_.
    // Built from L_lower_* and U_upper_* after build_solve_metadata_().
    std::vector<int> L_lower_col_ptr_, U_upper_col_ptr_;
    std::vector<int> L_lower_col_idx_, U_upper_col_idx_;
    std::vector<double> L_lower_col_val_, U_upper_col_val_;
    // Pivot lookup tables for hyper-sparse solves (Highs-style).
    // L_pivot_lookup_[pivot_index] = logical_position (inverse of L_pivot_index)
    // U_pivot_lookup_[pivot_index] = logical_position (inverse of U_pivot_index)
    std::vector<int> L_pivot_lookup_, U_pivot_lookup_;
    // Update method enum (Highs-style: FT, PF, MPF, APF).
    enum class UpdateMethod { FT = 1, PF = 2, MPF = 3, APF = 4 };
    UpdateMethod update_method_{UpdateMethod::FT};
    // Synthetic tick for timing model.
    mutable double synthetic_tick_{0.0};
    std::vector<int> affected_rows_scratch_;
    // Elimination trees (Item 2): first/last column-structure entry per node.
    // l_etree_[j]  = min{i>j : L[i,j]!=0}  (-1 if none) — forward L solve
    // u_etree_[j]  = max{i<j : U[i,j]!=0}  (-1 if none) — backward U solve
    // ut_etree_[j] = min{k>j : U[j,k]!=0}  (-1 if none) — forward U^T solve
    // lt_etree_[j] = max{k<j : L[j,k]!=0}  (-1 if none) — backward L^T solve
    std::vector<int> l_etree_, u_etree_, ut_etree_, lt_etree_;
    // Build scratch for build_solve_metadata_ — eliminates per-row heap allocations (Item 4).
    std::vector<IndexedValue> build_tmp_;
    // Permutation inverses — sparse RHS permute and FT update reach filter (Items 1, 6).
    std::vector<int> Pr_inv_, Pc_inv_;
    // Scratch for hyper-sparse triangular solves — reused across calls to avoid allocation.
    mutable std::vector<bool> reach_flag_scratch_;
    mutable std::vector<int> reach_scratch_;
    mutable std::vector<int> dfs_stack_scratch_;
    // Hyper-sparse solve working buffers (Highs-style cwork/iwork).
    mutable std::vector<char> hyper_sparse_cwork_;
    mutable std::vector<int> hyper_sparse_iwork_;
    // L-reach saved between L and U solve for seed chaining (Item 1).
    mutable std::vector<int> l_reach_seeds_scratch_;
    // Permuted seed positions for sparse RHS interface (Item 1).
    mutable std::vector<int> perm_seeds_scratch_;
    mutable Eigen::VectorXd sparse_l_scratch_;
    mutable Eigen::VectorXd sparse_u_scratch_;
    // Persistent output scratch for the hyper-sparse path. Holds the unpermuted,
    // unscaled, post-update solution. Reset between solves at the positions
    // touched by the previous solve (tracked via last_solve_reach_original_).
    mutable Eigen::VectorXd output_scratch_;
    mutable SparseRow merge_scratch_;
    // Adaptive reach-density EMA — drives hyper-sparse fallback decision (Item 9).
    mutable double ema_reach_ratio_{kHyperSparseDensityThreshold_};
    // True when reach_flag_scratch_ reflects the last hyper-sparse solve output (Item 6).
    mutable bool hyper_solve_reach_valid_{false};
    // Original-space row pattern of the last sparse-path solve (HVector source).
    mutable std::vector<int> last_solve_reach_original_;
    mutable bool last_solve_pattern_valid_{false};
    std::vector<SparseUpdate> updates_;

    // Product Form (PF) update structures — Highs-style for multiple update methods.
    // pf_start[i] = start of PF column i in pf_index/pf_value
    // pf_pivot_index[i] = row that PF column i pivots on
    // pf_pivot_value[i] = pivot value for PF column i
    std::vector<int> pf_start_, pf_index_, pf_pivot_index_;
    std::vector<double> pf_value_, pf_pivot_value_;
    // Total fill-in for PF (used to decide when to refactor).
    int pf_total_fill_{0};
    // Merit threshold for PF refactor decision.
    int pf_merit_threshold_{1000};

    // Singleton detection statistics for factorization analysis (Highs-style).
    // These count pivot types during factorization for analysis/reporting.
    int num_logical_pivots_{0};       // Unit logical columns (slacks)
    int num_unit_pivots_{0};          // Structural unit columns
    int num_row_singleton_pivots_{0}; // Row singletons
    int num_col_singleton_pivots_{0}; // Column singletons
    int num_markowitz_pivots_{0};     // General Markowitz pivots
    int num_kernel_pivots_{0};        // Kernel/remaining pivots

    // Singleton detection result structure.
    enum class PivotType { Logical, Unit, RowSingleton, ColSingleton, Markowitz };
    struct SingletonResult {
        PivotType type;
        int pivot_row{-1};
        int pivot_col{-1};
        double pivot_val{0.0};
    };

    // Detect singleton patterns in a column. Returns info about any singleton found.
    // row_count[i] = number of nonzeros in row i (used for singleton detection).
    SingletonResult detect_singleton_(int col, const SparseRow& column_entries,
                                      const std::vector<int>& row_degree) const {
        SingletonResult result;
        if (column_entries.size() == 0)
            return result;

        // Column singleton: exactly one entry in column
        if (column_entries.size() == 1) {
            result.type = PivotType::ColSingleton;
            result.pivot_row = column_entries[0].idx;
            result.pivot_col = col;
            result.pivot_val = column_entries[0].val;
            return result;
        }

        // Check if any row has exactly one entry (row singleton)
        for (const auto& entry : column_entries) {
            if (row_degree[entry.idx] == 1) {
                result.type = PivotType::RowSingleton;
                result.pivot_row = entry.idx;
                result.pivot_col = col;
                result.pivot_val = entry.val;
                return result;
            }
        }

        result.type = PivotType::Markowitz;
        return result;
    }

    // Reset singleton counters.
    void reset_singleton_counters_() noexcept {
        num_logical_pivots_ = 0;
        num_unit_pivots_ = 0;
        num_row_singleton_pivots_ = 0;
        num_col_singleton_pivots_ = 0;
        num_markowitz_pivots_ = 0;
        num_kernel_pivots_ = 0;
    }

    // Record a pivot type for statistics.
    void record_pivot_type_(PivotType type) {
        switch (type) {
            case PivotType::Logical:
                ++num_logical_pivots_;
                break;
            case PivotType::Unit:
                ++num_unit_pivots_;
                break;
            case PivotType::RowSingleton:
                ++num_row_singleton_pivots_;
                break;
            case PivotType::ColSingleton:
                ++num_col_singleton_pivots_;
                break;
            case PivotType::Markowitz:
                ++num_markowitz_pivots_;
                break;
        }
    }
    int updates_count_{0};
    double updates_max_z_inf_{0.0};
    double updates_max_w_inf_{0.0};
    double updates_cumulative_z_inf_{0.0};
    double updates_density_sum_{0.0};

    // P0-2: LR (row-wise L^T) storage — HiGHS-style explicit CSR for hyper-sparse BTRAN L.
    //       LR_ptr_ = start offset per row (like HiGHS lr_start)
    //       LR_idx_  = column indices per row of L^T (= row indices of L)
    //       LR_val_  = corresponding values
    std::vector<int> LR_ptr_;
    std::vector<int> LR_idx_;
    std::vector<double> LR_val_;

    // P2-2: Rank deficiency tracking (HiGHS-style). When no acceptable pivot is found,
    // we inject identity/logical columns to complete the factorization. The rows/columns
    // that had no valid pivot are recorded here.
    int rank_deficiency_{0};
    std::vector<int> row_with_no_pivot_; // row indices that had no pivot
    std::vector<int> col_with_no_pivot_; // column indices that had no pivot
    // Maps from pivot step k → whether it was a logical injection (true) or real pivot (false)
    std::vector<bool> pivot_was_logical_;

    std::vector<int> Pr_, Pc_;
    std::vector<SparseRow> U_rows_, L_rows_, U_cols_, L_cols_;
    mutable bool U_cols_dirty_{false};
    mutable bool L_cols_dirty_{false};
    UpdateFailureReason last_update_failure_reason_{UpdateFailureReason::None};
};
