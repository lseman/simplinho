#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <ankerl/unordered_dense.h>

#include "amd.h"
#include "markowitz.h"
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
        bool iterative_refinement{false};
        int iterative_refinement_steps{1};
        double iterative_refinement_tol{1e-10};
        double max_norm_growth_before_refactor{1e6};
        int max_parallel_update_size{64};
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

        SparseMat factor_matrix = A;
        if (config_.diagonal_equilibration)
            factor_matrix = equilibrate_inf_norm_(A);
        load_initial_U_(factor_matrix);
        ensure_U_cols_ready_();
        symbolic_analyze_();
        initialize_active_stats_();
        try {
            factorize_sparse_();
            build_solve_metadata_();
        } catch (const std::runtime_error&) {
            activate_sparse_lu_fallback_(base_matrix_original_);
        }
        clear_updates();
    }

    bool supports_inplace_updates() const noexcept { return n_ > 0 && !use_fallback_sparse_lu_; }

    bool has_updates() const noexcept { return !updates_.empty(); }

    void clear_updates() noexcept {
        updates_.clear();
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
        updates_.push_back(SparseUpdate{j, dense_to_sparse_update_(u, eps),
                                        dense_to_sparse_update_(z, eps),
                                        dense_to_sparse_update_(w, eps), alpha});
        update_norm_growth_estimate_(updates_.back());
        update_cached_stats_(updates_.back());
        return true;
    }

    bool needs_refactor() const noexcept {
        return !std::isfinite(norm_growth_estimate_) ||
               norm_growth_estimate_ > config_.max_norm_growth_before_refactor;
    }

    Eigen::VectorXd solve(const Eigen::VectorXd& b) const {
        return solve_impl_(b, config_.iterative_refinement);
    }

    Eigen::VectorXd solveT(const Eigen::VectorXd& c) const {
        return solveT_impl_(c, config_.iterative_refinement);
    }

  private:
    Eigen::VectorXd solve_impl_(const Eigen::VectorXd& b, bool enable_refinement) const {
        if (b.size() != n_) [[unlikely]]
            throw std::invalid_argument("SparseForrestTomlinLU::solve size mismatch");
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);

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

        const bool use_sparse_rhs = is_hyper_sparse_rhs_(Pb);
        Eigen::VectorXd z = use_sparse_rhs ? forward_solve_L_sparse_(Pb) : forward_solve_L_(Pb);
        Eigen::VectorXd w = use_sparse_rhs ? back_solve_U_sparse_(z) : back_solve_U_(z);

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

    Eigen::VectorXd solveT_impl_(const Eigen::VectorXd& c, bool enable_refinement) const {
        if (c.size() != n_) [[unlikely]]
            throw std::invalid_argument("SparseForrestTomlinLU::solveT size mismatch");
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

        const bool use_sparse_rhs = is_hyper_sparse_rhs_(PcTc);
        Eigen::VectorXd t =
            use_sparse_rhs ? forward_solve_UT_sparse_(PcTc) : forward_solve_UT_(PcTc);
        Eigen::VectorXd s = use_sparse_rhs ? back_solve_LT_sparse_(t) : back_solve_LT_(t);

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
            SIMPLEX_ASSUME(idx.size() == val.size());
            for (size_t pos = 0; pos < idx.size(); ++pos)
                out += val[pos] * x(idx[pos]);
            return out;
        }

        void axpy(Eigen::VectorXd& x, double alpha) const {
            SIMPLEX_ASSUME(idx.size() == val.size());
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
    static constexpr long kEarlyAcceptMarkowitzScore_ = 1;
    static constexpr double kEarlyAcceptPivotRatio_ = 0.9;

    static bool is_hyper_sparse_rhs_(const Eigen::VectorXd& rhs) {
        if (rhs.size() == 0)
            return false;
        const int threshold =
            std::max(1, static_cast<int>(rhs.size() * kHyperSparseDensityThreshold_));
        int nz = 0;
        for (int i = 0; i < rhs.size() && nz <= threshold; ++i) {
            if (std::abs(rhs(i)) > kZeroTol_)
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
        std::sort(ordered.begin(), ordered.end());
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
            std::sort(cols.begin(), cols.end());
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

    void apply_col_unscaling_(Eigen::VectorXd& x) const {
        for (int i = 0; i < x.size(); ++i)
            x(i) *= col_scale_[static_cast<size_t>(i)];
    }

    void apply_row_unscaling_(Eigen::VectorXd& y) const {
        for (int i = 0; i < y.size(); ++i)
            y(i) *= row_scale_[static_cast<size_t>(i)];
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

        SparseRow new_row_entries;
        new_row_entries.reserve(target_row.size() + pivot_row_phys.size());

        std::size_t target_pos = 0;
        std::size_t pivot_pos = 0;
        while (target_pos < target_row.size() || pivot_pos < pivot_row_phys.size()) {
            const int target_col = target_pos < target_row.size()
                                       ? target_row[target_pos].idx
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

        for (int i = 0; i < n_; ++i) {
            const int phys_row = row_map_[i];
            const int phys_col = col_map_[i];

            L_lower_ptr_[i] = static_cast<int>(L_lower_idx_.size());
            for (const auto& entry : sorted_entries_(L_rows_[i])) {
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

            U_upper_ptr_[i] = static_cast<int>(U_upper_idx_.size());
            for (const auto& entry : logical_sorted_entries_(U_rows_[phys_row], col_inv_)) {
                if (entry.idx == i) {
                    U_diag_[i] = entry.val;
                } else if (entry.idx > i) {
                    U_upper_idx_.push_back(entry.idx);
                    U_upper_val_.push_back(entry.val);
                }
            }
            U_upper_ptr_[i + 1] = static_cast<int>(U_upper_idx_.size());

            UT_lower_ptr_[i] = static_cast<int>(UT_lower_idx_.size());
            ensure_U_cols_ready_();
            for (const auto& entry : logical_sorted_entries_(U_cols_[phys_col], row_inv_)) {
                if (entry.idx < i) {
                    UT_lower_idx_.push_back(entry.idx);
                    UT_lower_val_.push_back(entry.val);
                } else {
                    break;
                }
            }
            UT_lower_ptr_[i + 1] = static_cast<int>(UT_lower_idx_.size());

            LT_upper_ptr_[i] = static_cast<int>(LT_upper_idx_.size());
            ensure_L_cols_ready_();
            for (const auto& entry : sorted_entries_(L_cols_[i])) {
                if (entry.idx > i) {
                    LT_upper_idx_.push_back(entry.idx);
                    LT_upper_val_.push_back(entry.val);
                }
            }
            LT_upper_ptr_[i + 1] = static_cast<int>(LT_upper_idx_.size());
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
            auto [pi, pj] = choose_pivot_sparse_(k);
            if (pi < 0 || pj < 0)
                throw std::runtime_error("SparseForrestTomlinLU: no pivot found");

            swap_U_rows_(k, pi);
            swap_U_cols_(k, pj);
            swap_L_prefix_rows_(k, pi, k);
            std::swap(Pr_[k], Pr_[pi]);
            std::swap(Pc_[k], Pc_[pj]);

            const double piv = get_U_(k, k);
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
                throw std::runtime_error("SparseForrestTomlinLU: singular pivot");

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

    Eigen::VectorXd forward_solve_L_sparse_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_);
        for (int i = 0; i < n_; ++i) {
            double rhs_i = b(i);
            for (int pos = L_lower_ptr_[i]; pos < L_lower_ptr_[i + 1]; ++pos) {
                const int idx = L_lower_idx_[pos];
                const double xidx = x(idx);
                if (xidx != 0.0)
                    rhs_i -= L_lower_val_[pos] * xidx;
            }
            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            const double xi = rhs_i / piv;
            if (std::abs(xi) > kZeroTol_)
                x(i) = xi;
        }
        return x;
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

    Eigen::VectorXd back_solve_U_sparse_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            double rhs_i = b(i);
            for (int pos = U_upper_ptr_[i]; pos < U_upper_ptr_[i + 1]; ++pos) {
                const int idx = U_upper_idx_[pos];
                const double xidx = x(idx);
                if (xidx != 0.0)
                    rhs_i -= U_upper_val_[pos] * xidx;
            }

            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            const double xi = rhs_i / piv;
            if (std::abs(xi) > kZeroTol_)
                x(i) = xi;
        }
        return x;
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

    Eigen::VectorXd forward_solve_UT_sparse_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_);
        for (int i = 0; i < n_; ++i) {
            double rhs_i = b(i);
            for (int pos = UT_lower_ptr_[i]; pos < UT_lower_ptr_[i + 1]; ++pos) {
                const int idx = UT_lower_idx_[pos];
                const double xidx = x(idx);
                if (xidx != 0.0)
                    rhs_i -= UT_lower_val_[pos] * xidx;
            }
            const double piv = U_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");
            const double xi = rhs_i / piv;
            if (std::abs(xi) > kZeroTol_)
                x(i) = xi;
        }
        return x;
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

    Eigen::VectorXd back_solve_LT_sparse_(const Eigen::VectorXd& b) const {
        if (n_ == 0) [[unlikely]]
            return b;
        SIMPLEX_ASSUME(n_ > 0);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            double rhs_i = b(i);
            for (int pos = LT_upper_ptr_[i]; pos < LT_upper_ptr_[i + 1]; ++pos) {
                const int idx = LT_upper_idx_[pos];
                const double xidx = x(idx);
                if (xidx != 0.0)
                    rhs_i -= LT_upper_val_[pos] * xidx;
            }
            const double piv = L_diag_[i];
            if (!std::isfinite(piv) || std::abs(piv) < abs_floor_) [[unlikely]]
                throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");
            const double xi = rhs_i / piv;
            if (std::abs(xi) > kZeroTol_)
                x(i) = xi;
        }
        return x;
    }

    Eigen::VectorXd apply_updates_solve_(Eigen::VectorXd x) const {
        for (const auto& update : updates_) {
            const double xj = x(update.j);
            if (xj != 0.0)
                update.z.axpy(x, -(xj / update.alpha));
        }
        return x;
    }

    Eigen::VectorXd apply_updates_solve_T_(Eigen::VectorXd y) const {
        for (const auto& update : updates_) {
            const double uy = update.u.dot(y);
            if (uy != 0.0)
                update.w.axpy(y, -(uy / update.alpha));
        }
        return y;
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
            Eigen::VectorXd dx = solve_impl_(residual, false);
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
            Eigen::VectorXd dy = solveT_impl_(residual, false);
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
    std::vector<int> affected_rows_scratch_;
    std::vector<SparseUpdate> updates_;
    int updates_count_{0};
    double updates_max_z_inf_{0.0};
    double updates_max_w_inf_{0.0};
    double updates_cumulative_z_inf_{0.0};
    double updates_density_sum_{0.0};

    std::vector<int> Pr_, Pc_;
    std::vector<SparseRow> U_rows_, L_rows_, U_cols_, L_cols_;
    mutable bool U_cols_dirty_{false};
    mutable bool L_cols_dirty_{false};
    UpdateFailureReason last_update_failure_reason_{UpdateFailureReason::None};
};
