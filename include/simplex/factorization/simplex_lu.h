#pragma once

#include "../../../extern/pdqsort/pdqsort.h"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <ankerl/unordered_dense.h>

#include "simplex/core/hvector.h"
#include "simplex/core/markowitz.h"
#include "simplex/factorization/amd.h"
#include "simplex/factorization/sparse_lu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// ======================================================
// FTBasis v4
// - faster update path
// - true solve-based residual checks
// - lazy sparse basis rebuild
// - incremental eta statistics
// ======================================================
class FTBasis {
  public:
    using DenseMat = Eigen::MatrixXd;
    using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;

    struct Options {
        // Hard backstops only — the synthetic reinversion clock (HiGHS
        // economics: rebuild when update-chain solve cost exceeds build cost)
        // is the primary refactor trigger. HiGHS's update limit is 5000.
        int refactor_every = 1024;
        int compress_every = 512;
        double pivot_rel = 1e-12;
        double abs_floor = 1e-16;
        double alpha_tol = 1e-10;
        double z_inf_guard = 1e6;
        bool sparse_amd = true;
        double sparse_drop_tol = 0.0;
        std::string sparse_backend = "auto"; // "auto" | "pf" | "ft" | "eigen"
        bool sparse_equilibration = true;
        // HiGHS-style: no per-solve residual validation — numerical trouble
        // is caught by the engines' alpha cross-check at pivot time. The
        // Eigen SparseLU oracle stays on: besides recovery it is the
        // factor-time singularity detector (our Markowitz build "succeeds" on
        // rank-deficient bases via logical injection, and the engines rely on
        // the oracle's factorize failure to trigger early basis repair).
        bool sparse_solve_oracle = true;
        bool sparse_validate_solves = false;
        // HiGHS uses indexed vector loops well beyond the hyper-sparse regime.
        // Keep sparse RHS solves active up to this density so FTRAN/BTRAN can
        // preserve useful reach patterns for pricing.
        double sparse_rhs_density_threshold = 0.40;

        enum class UpdateMode { EtaStack, ForrestTomlin, Hybrid };
        UpdateMode update_mode = UpdateMode::Hybrid;

        int ft_bandwidth_cap = 12;
        double max_growth_tol = 5e7;
        double min_dynamic_growth_tol = 500;
        double min_refactor_interval_fraction = 0.35;
        double max_condition_estimate = 1e13;
        double ft_multiplier_guard = 1e8;
        int rook_iters = 2;

        // safeguards
        bool enable_iterative_refinement = true;
        int refinement_steps = 2;
        double residual_refactor_tol = 1e-9;
        double residual_abs_refactor_tol = 1e-10;
        int refinement_max_steps = 6;
        double refinement_slow_progress_ratio = 0.5;
        double refinement_stall_progress_ratio = 0.8;
        int refinement_stall_limit = 3;
        int max_eta_count = 128;
        bool refactor_on_solve_failure = true;
        bool aggressive_refactor_on_suspicious_residual = true;

        // update-chain health
        double eta_max_inf_norm = 1e7;
        double eta_avg_density_guard = 0.35;
        double eta_cumulative_inf_norm_guard = 1e8;

        // true factorization residual tolerance
        double column_residual_tol = 1e-8;

        // only check solve-based basis residual periodically
        int residual_check_frequency = 8;

        // if update chain already under stress, skip expensive update attempt and refactor
        double early_refactor_stability_score = 1.0;

        // Optional external counters — when non-null, FTBasis bumps these on
        // every full refactor and accumulates time spent. Threaded through
        // Options to avoid changing the public solve_B/solve_BT/replace_column
        // signatures. Ownership stays with the caller.
        int* ext_refactor_counter = nullptr;
        int* ext_ft_update_counter = nullptr;
        std::uint64_t* ext_refactor_ns = nullptr;
        std::uint64_t* ext_pivot_ns = nullptr;
    };

    struct Eta {
        int j{-1};
        Eigen::VectorXd u;
        Eigen::VectorXd z;
        Eigen::VectorXd w;
        double alpha{0.0};

        double z_inf_norm() const { return z.lpNorm<Eigen::Infinity>(); }
        double w_inf_norm() const { return w.lpNorm<Eigen::Infinity>(); }

        double z_density(double eps = 1e-14) const {
            if (z.size() == 0)
                return 0.0;
            int nnz = 0;
            for (int i = 0; i < z.size(); ++i)
                if (std::abs(z(i)) > eps)
                    ++nnz;
            return static_cast<double>(nnz) / static_cast<double>(z.size());
        }
    };

    struct Stats {
        int update_count{0};
        int eta_count{0};
        double max_eta_z_inf{0.0};
        double max_eta_w_inf{0.0};
        double avg_eta_density{0.0};
        double cumulative_eta_z_inf{0.0};
        double last_column_residual{0.0};
        double growth_factor{1.0};
        double growth_from_last_refactor{1.0};
        double growth_from_initial_refactor{1.0};
        double estimated_condition{1.0};
        double stability_score{0.0};
        double sparse_norm_growth_estimate{1.0};
        double last_refactor_sanity_residual{0.0};
    };

    // HiGHS-style per-class FTRAN/BTRAN expected-density tracking.
    // ColAq := FTRAN of an entering column A_q (solve_B(a_e)).
    // RowEp := BTRAN of a unit row e_p or basic-cost vector cB (solve_BT).
    // RowAp := dense reduced-cost row after BTRAN — currently unused but reserved for symmetry.
    // The solver feeds `count/m_` back after each call; the kernel reads `expected(kind)` to
    // pick hyper vs sparse vs dense per triangular stage. EWMA mirrors HEkk:
    //   density = (1-mult)*density + mult*local, mult = 0.05.
    enum class TranKind { Unknown, ColAq, RowEp, RowAp };

    struct DensityTracker {
        static constexpr double kRunningAverageMultiplier = 0.05;
        double col_aq{0.0};
        double row_ep{0.0};
        double row_ap{0.0};

        double expected(TranKind k) const noexcept {
            switch (k) {
                case TranKind::ColAq:
                    return col_aq;
                case TranKind::RowEp:
                    return row_ep;
                case TranKind::RowAp:
                    return row_ap;
                default:
                    return 1.0; // unknown ⇒ assume dense (force sparse_solve path)
            }
        }

        void update(TranKind k, int count, int m) noexcept {
            if (m <= 0 || k == TranKind::Unknown)
                return;
            const double local =
                std::clamp(static_cast<double>(count) / static_cast<double>(m), 0.0, 1.0);
            double* slot = nullptr;
            switch (k) {
                case TranKind::ColAq:
                    slot = &col_aq;
                    break;
                case TranKind::RowEp:
                    slot = &row_ep;
                    break;
                case TranKind::RowAp:
                    slot = &row_ap;
                    break;
                default:
                    return;
            }
            *slot = (1.0 - kRunningAverageMultiplier) * (*slot) + kRunningAverageMultiplier * local;
        }
    };

    FTBasis(const DenseMat& A, const std::vector<int>& basis) : FTBasis(A, basis, Options{}) {}

    FTBasis(const DenseMat& A, const std::vector<int>& basis, const Options& opt)
        : A_dense_(&A), A_sparse_(nullptr), A_is_sparse_(false), m_(static_cast<int>(A.rows())),
          basis_(basis), opt_(opt) {
        if (static_cast<int>(basis_.size()) != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");
        // Validate basis columns: no duplicates, no out-of-range
        {
            std::vector<char> used(static_cast<size_t>(A.cols()), 0);
            for (int j : basis_) {
                if (j < 0 || j >= static_cast<int>(A.cols()))
                    throw std::invalid_argument("FTBasis: basis has out-of-range column index " +
                                                std::to_string(j));
                if (used[static_cast<size_t>(j)])
                    throw std::invalid_argument("FTBasis: basis has duplicate column index " +
                                                std::to_string(j));
                used[static_cast<size_t>(j)] = 1;
            }
        }
        Bcols_dense_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_dense_[i] = A.col(basis_[i]);
        initialize_dense_basis_cache_();
        dense_refactor_();
    }

    FTBasis(const SparseMat& A, const std::vector<int>& basis) : FTBasis(A, basis, Options{}) {}

    FTBasis(const SparseMat& A, const std::vector<int>& basis, const Options& opt)
        : A_dense_(nullptr), A_sparse_(&A), A_is_sparse_(true), m_(static_cast<int>(A.rows())),
          basis_(basis), opt_(opt) {
        if (static_cast<int>(basis_.size()) != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");
        // Validate basis columns: no duplicates, no out-of-range
        {
            std::vector<char> used(static_cast<size_t>(A.cols()), 0);
            for (int j : basis_) {
                if (j < 0 || j >= static_cast<int>(A.cols()))
                    throw std::invalid_argument("FTBasis: basis has out-of-range column index " +
                                                std::to_string(j));
                if (used[static_cast<size_t>(j)])
                    throw std::invalid_argument("FTBasis: basis has duplicate column index " +
                                                std::to_string(j));
                used[static_cast<size_t>(j)] = 1;
            }
        }
        Bcols_sparse_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_sparse_[i] = A.col(basis_[i]);
        current_B_sparse_dirty_ = true;
        sparse_refactor_();
    }

    int rows() const noexcept { return m_; }
    const std::vector<int>& basis() const noexcept { return basis_; }
    const std::vector<Eta>& etas() const noexcept { return etas_; }
    int update_count() const noexcept { return update_count_; }
    Stats stats() const noexcept { return stats_; }
    const std::string& last_update_diagnostic() const noexcept { return last_update_diagnostic_; }
    // Compute a hash of the B matrix for warm-reuse verification
    std::uint64_t basis_matrix_signature_() const;
    std::uint64_t basis_matrix_signature() const { return basis_matrix_signature_(); }

    // Attach the sparse pattern captured by the last sparse-path solve in
    // lu_sparse_, when that path was used and the pattern wasn't invalidated
    // by Forrest-Tomlin updates or iterative refinement.
    HVector wrap_with_pattern_(Eigen::VectorXd v, bool sparse_path_used) const {
        if (sparse_path_used && A_is_sparse_ && lu_sparse_.last_solve_pattern_valid()) {
            return HVector(std::move(v), lu_sparse_.last_solve_reach_original());
        }
        return HVector(std::move(v));
    }

    HVector solve_B(const Eigen::VectorXd& b, TranKind kind = TranKind::Unknown) const {
        if (b.size() != m_)
            throw std::invalid_argument("FTBasis::solve_B size mismatch");

        auto* self = const_cast<FTBasis*>(this);
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            Eigen::VectorXd x = self->solve_B_fast_(b, expected);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                x = self->refine_solve_B_(b, x);
            return x;
        };

        try {
            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b, x, "FTBasis::solve_B");
            HVector out = wrap_with_pattern_(std::move(x), A_is_sparse_);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b, x, "FTBasis::solve_B after refactor");
            HVector out = wrap_with_pattern_(std::move(x), A_is_sparse_);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    HVector solve_B(const HVector& b, TranKind kind = TranKind::Unknown) const {
        if (b.size() != m_)
            throw std::invalid_argument("FTBasis::solve_B HVector size mismatch");
        if (!b.has_pattern())
            return solve_B(b.value, kind);

        auto* self = const_cast<FTBasis*>(this);
        auto [seed_idx, seed_val] = hvector_to_seed_data_(b);
        const double rhs_density =
            m_ > 0 ? static_cast<double>(seed_idx.size()) / static_cast<double>(m_) : 0.0;
        const bool use_sparse_rhs = A_is_sparse_ && rhs_density < sparse_rhs_density_threshold_();
        if (!use_sparse_rhs)
            return solve_B(b.value, kind);

        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            return self->lu_sparse_.solve_sparse(seed_idx, seed_val, expected);
        };

        try {
            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b.value, x, "FTBasis::solve_B HVector");
            HVector out = wrap_with_pattern_(std::move(x), true);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b.value, x, "FTBasis::solve_B HVector after refactor");
            HVector out = wrap_with_pattern_(std::move(x), true);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    template <typename Derived>
    HVector solve_B(const Eigen::SparseMatrixBase<Derived>& b_sparse,
                    TranKind kind = TranKind::Unknown) const {
        if (b_sparse.rows() != m_ || b_sparse.cols() != 1)
            throw std::invalid_argument("FTBasis::solve_B sparse size mismatch");
        auto* self = const_cast<FTBasis*>(this);
        auto [seed_idx, seed_val] = sparse_vector_to_seed_data_(b_sparse.derived());
        const double rhs_density =
            m_ > 0 ? static_cast<double>(seed_idx.size()) / static_cast<double>(m_) : 0.0;
        const bool use_sparse_rhs = A_is_sparse_ && rhs_density < sparse_rhs_density_threshold_();
        Eigen::VectorXd b = sparse_vector_to_dense_(b_sparse.derived(), m_);
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            if (use_sparse_rhs)
                return self->lu_sparse_.solve_sparse(seed_idx, seed_val, expected);
            Eigen::VectorXd x = self->solve_B_fast_(b, expected);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                x = self->refine_solve_B_(b, x);
            return x;
        };

        try {
            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b, x, "FTBasis::solve_B sparse");
            HVector out = wrap_with_pattern_(std::move(x), use_sparse_rhs);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd x = do_solve();
            self->verify_solve_quality_B_(b, x, "FTBasis::solve_B sparse after refactor");
            HVector out = wrap_with_pattern_(std::move(x), use_sparse_rhs);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    HVector solve_BT(const Eigen::VectorXd& c, TranKind kind = TranKind::Unknown) const {
        if (c.size() != m_)
            throw std::invalid_argument("FTBasis::solve_BT size mismatch");

        auto* self = const_cast<FTBasis*>(this);
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            Eigen::VectorXd y = self->solve_BT_fast_(c, expected);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                y = self->refine_solve_BT_(c, y);
            return y;
        };

        try {
            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c, y, "FTBasis::solve_BT");
            HVector out = wrap_with_pattern_(std::move(y), A_is_sparse_);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c, y, "FTBasis::solve_BT after refactor");
            HVector out = wrap_with_pattern_(std::move(y), A_is_sparse_);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    HVector solve_BT(const HVector& c, TranKind kind = TranKind::Unknown) const {
        if (c.size() != m_)
            throw std::invalid_argument("FTBasis::solve_BT HVector size mismatch");
        if (!c.has_pattern())
            return solve_BT(c.value, kind);

        auto* self = const_cast<FTBasis*>(this);
        auto [seed_idx, seed_val] = hvector_to_seed_data_(c);
        const double rhs_density =
            m_ > 0 ? static_cast<double>(seed_idx.size()) / static_cast<double>(m_) : 0.0;
        const bool use_sparse_rhs = A_is_sparse_ && rhs_density < sparse_rhs_density_threshold_();
        if (!use_sparse_rhs)
            return solve_BT(c.value, kind);

        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            return self->lu_sparse_.solveT_sparse(seed_idx, seed_val, expected);
        };

        try {
            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c.value, y, "FTBasis::solve_BT HVector");
            HVector out = wrap_with_pattern_(std::move(y), true);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c.value, y, "FTBasis::solve_BT HVector after refactor");
            HVector out = wrap_with_pattern_(std::move(y), true);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    // Fast path for canonical unit-vector RHS (most common case: row picks in pricing).
    // Routes directly to the sparse triangular solve using a single-seed index, avoiding
    // the O(m) zero-scan in the dense solve path.
    HVector solve_B_unit(int i, TranKind kind = TranKind::Unknown) const {
        if (i < 0 || i >= m_)
            throw std::invalid_argument("FTBasis::solve_B_unit index out of range");
        if (!A_is_sparse_) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(m_);
            e(i) = 1.0;
            return solve_B(e, kind);
        }
        auto* self = const_cast<FTBasis*>(this);
        static const std::vector<double> one_val{1.0};
        std::vector<int> seed{i};
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() { return self->lu_sparse_.solve_sparse(seed, one_val, expected); };
        try {
            HVector out = wrap_with_pattern_(do_solve(), true);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;
            self->last_update_diagnostic_ = err.what();
            self->refactor();
            HVector out = wrap_with_pattern_(do_solve(), true);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    HVector solve_BT_unit(int i, TranKind kind = TranKind::Unknown) const {
        if (i < 0 || i >= m_)
            throw std::invalid_argument("FTBasis::solve_BT_unit index out of range");
        if (!A_is_sparse_) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(m_);
            e(i) = 1.0;
            return solve_BT(e, kind);
        }
        auto* self = const_cast<FTBasis*>(this);
        static const std::vector<double> one_val{1.0};
        std::vector<int> seed{i};
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() { return self->lu_sparse_.solveT_sparse(seed, one_val, expected); };
        try {
            HVector out = wrap_with_pattern_(do_solve(), true);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;
            self->last_update_diagnostic_ = err.what();
            self->refactor();
            HVector out = wrap_with_pattern_(do_solve(), true);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    template <typename Derived>
    HVector solve_BT(const Eigen::SparseMatrixBase<Derived>& c_sparse,
                     TranKind kind = TranKind::Unknown) const {
        if (c_sparse.rows() != m_ || c_sparse.cols() != 1)
            throw std::invalid_argument("FTBasis::solve_BT sparse size mismatch");
        auto* self = const_cast<FTBasis*>(this);
        auto [seed_idx, seed_val] = sparse_vector_to_seed_data_(c_sparse.derived());
        const double rhs_density =
            m_ > 0 ? static_cast<double>(seed_idx.size()) / static_cast<double>(m_) : 0.0;
        const bool use_sparse_rhs = A_is_sparse_ && rhs_density < sparse_rhs_density_threshold_();
        Eigen::VectorXd c = sparse_vector_to_dense_(c_sparse.derived(), m_);
        const double expected = density_tracker_.expected(kind);
        auto do_solve = [&]() {
            if (use_sparse_rhs)
                return self->lu_sparse_.solveT_sparse(seed_idx, seed_val, expected);
            Eigen::VectorXd y = self->solve_BT_fast_(c, expected);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                y = self->refine_solve_BT_(c, y);
            return y;
        };

        try {
            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c, y, "FTBasis::solve_BT sparse");
            HVector out = wrap_with_pattern_(std::move(y), use_sparse_rhs);
            self->update_density_tracker_(kind, out);
            return out;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd y = do_solve();
            self->verify_solve_quality_BT_(c, y, "FTBasis::solve_BT sparse after refactor");
            HVector out = wrap_with_pattern_(std::move(y), use_sparse_rhs);
            self->update_density_tracker_(kind, out);
            return out;
        }
    }

    // Read-only access to the density tracker (debug + tests).
    const DensityTracker& density_tracker() const noexcept { return density_tracker_; }

    void replace_column(int j, const Eigen::VectorXd& new_col_dense) {
        replace_column_impl_(j, std::nullopt, new_col_dense);
    }

    void replace_column(int j, int entering_col, const Eigen::VectorXd& new_col_dense) {
        replace_column_impl_(j, entering_col, new_col_dense);
    }

    template <typename Derived>
    void replace_column(int j, const Eigen::SparseMatrixBase<Derived>& new_col_sparse) {
        const auto& sparse = new_col_sparse.derived();
        const SparseMat sparse_copy = sparse;
        replace_column_impl_(j, std::nullopt, sparse_vector_to_dense_(sparse, m_), &sparse_copy);
    }

    template <typename Derived>
    void replace_column(int j, int entering_col,
                        const Eigen::SparseMatrixBase<Derived>& new_col_sparse) {
        const auto& sparse = new_col_sparse.derived();
        const SparseMat sparse_copy = sparse;
        replace_column_impl_(j, entering_col, sparse_vector_to_dense_(sparse, m_), &sparse_copy);
    }

    void refactor() {
        if (A_is_sparse_)
            sparse_refactor_();
        else
            dense_refactor_();
    }

    // Fast rebuild using HiGHS-style RefactorInfo pivot records.
    // Reconstructs LU factorization from cached pivot sequence without
    // starting from scratch. Handles special pivot types: logical, unit.
    // Returns 0 on success, non-zero if rank deficiency detected.
    int rebuild() {
        if (!A_is_sparse_) {
            // Dense rebuild not implemented - fall back to full refactor
            dense_refactor_();
            return 0;
        }
        // For now, sparse rebuild is not fully implemented for the sparse LU backend.
        // The HiGHS HFactor::rebuild() logic requires specific data structures
        // that SparseForrestTomlinLU does not have.
        //
        // TODO: Implement full sparse rebuild logic mirroring HiGHS HFactor::rebuild()
        // for Markowitz pivots using forward_solve_L_ and forming L/U factors incrementally.

        sparse_refactor_();
        return 0;
    }

    // Restore the last successfully-refactored basis snapshot and refactor it.
    // Returns true if a snapshot existed and the restored basis factored
    // cleanly. On success, `engine_basis` is overwritten with the restored
    // basis indices so the caller can resync its parallel basis vector.
    // Halves the adaptive refactor budget (HiGHS-style) to force more frequent
    // refactors on the recovered chain.
    bool try_backtrack_to_last_good(std::vector<int>& engine_basis) {
        if (!backtracking_snapshot_.has_value())
            return false;
        const Snapshot& snap = *backtracking_snapshot_;
        if (static_cast<int>(snap.basis.size()) != m_)
            return false;

        basis_ = snap.basis;
        if (A_is_sparse_) {
            Bcols_sparse_ = snap.Bcols_sparse;
            current_B_sparse_dirty_ = true;
        } else {
            Bcols_dense_ = snap.Bcols_dense;
            initialize_dense_basis_cache_();
        }

        try {
            if (A_is_sparse_)
                sparse_refactor_();
            else
                dense_refactor_();
        } catch (...) {
            // The snapshot itself is now unusable — drop it so we don't loop.
            backtracking_snapshot_.reset();
            return false;
        }

        // Halve the adaptive refactor cadence on the recovered chain. Floor at 1
        // so we don't disable updates entirely. We halve current_refactor_every_
        // (not opt_.refactor_every) so the original setting is restored on the
        // next clean refactor.
        current_refactor_every_ = std::max(1, current_refactor_every_ / 2);
        engine_basis = basis_;
        return true;
    }

    bool has_backtracking_snapshot() const noexcept { return backtracking_snapshot_.has_value(); }

    Eigen::MatrixXd explicit_B_dense() const {
        if (A_is_sparse_) {
            ensure_current_B_sparse_();
            return Eigen::MatrixXd(current_B_sparse_);
        }
        return current_B_dense_;
    }

  private:
    static std::vector<int> permutation_to_vector_(const Permutation& perm) {
        std::vector<int> out(static_cast<size_t>(perm.indices().size()));
        for (int i = 0; i < perm.indices().size(); ++i)
            out[static_cast<size_t>(i)] = perm.indices()(i);
        return out;
    }

    template <typename Derived>
    static Eigen::VectorXd sparse_vector_to_dense_(const Eigen::SparseMatrixBase<Derived>& mat,
                                                   int rows) {
        Eigen::VectorXd out = Eigen::VectorXd::Zero(rows);
        const auto& derived = mat.derived();
        for (int k = 0; k < derived.outerSize(); ++k) {
            for (typename Derived::InnerIterator it(derived, k); it; ++it)
                out(it.row()) = it.value();
        }
        return out;
    }

    template <typename Derived>
    static std::pair<std::vector<int>, std::vector<double>>
    sparse_vector_to_seed_data_(const Eigen::SparseMatrixBase<Derived>& mat) {
        std::vector<int> seed_idx;
        std::vector<double> seed_val;
        const auto& derived = mat.derived();
        const int reserve_hint = std::max(0, static_cast<int>(derived.nonZeros()));
        seed_idx.reserve(static_cast<size_t>(reserve_hint));
        seed_val.reserve(static_cast<size_t>(reserve_hint));
        for (int k = 0; k < derived.outerSize(); ++k) {
            for (typename Derived::InnerIterator it(derived, k); it; ++it) {
                seed_idx.push_back(it.row());
                seed_val.push_back(it.value());
            }
        }
        return {std::move(seed_idx), std::move(seed_val)};
    }

    static std::pair<std::vector<int>, std::vector<double>>
    hvector_to_seed_data_(const HVector& v, double eps = 1e-14) {
        std::vector<int> seed_idx;
        std::vector<double> seed_val;
        if (!v.has_pattern())
            return {std::move(seed_idx), std::move(seed_val)};
        seed_idx.reserve(static_cast<std::size_t>(std::max(0, v.count)));
        seed_val.reserve(static_cast<std::size_t>(std::max(0, v.count)));
        for (int k = 0; k < v.count; ++k) {
            const int i = v.index[static_cast<std::size_t>(k)];
            if (i < 0 || i >= v.value.size())
                throw std::invalid_argument("FTBasis: HVector pattern index out of range");
            const double value = v.value(i);
            if (std::abs(value) > eps) {
                seed_idx.push_back(i);
                seed_val.push_back(value);
            }
        }
        return {std::move(seed_idx), std::move(seed_val)};
    }

    static CSR sparse_to_amd_csr_(const SparseMat& B) {
        CSR csr(B.rows());
        csr.indptr[0] = 0;
        std::vector<std::vector<int>> row_cols(static_cast<size_t>(B.rows()));
        for (int col = 0; col < B.outerSize(); ++col) {
            for (SparseMat::InnerIterator it(B, col); it; ++it)
                row_cols[static_cast<size_t>(it.row())].push_back(col);
        }
        for (int row = 0; row < B.rows(); ++row) {
            auto& cols = row_cols[static_cast<size_t>(row)];
            pdqsort(cols.begin(), cols.end());
            cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
            csr.indptr[row + 1] = csr.indptr[row] + static_cast<int>(cols.size());
            csr.indices.insert(csr.indices.end(), cols.begin(), cols.end());
        }
        return csr;
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

    double sparse_rhs_density_threshold_() const noexcept {
        if (!std::isfinite(opt_.sparse_rhs_density_threshold))
            return 0.40;
        return std::clamp(opt_.sparse_rhs_density_threshold, 0.0, 1.0);
    }

    void compute_sparse_ordering_(const SparseMat& B) {
        if (!opt_.sparse_amd || m_ <= 1 || sparse_ordering_cached_)
            return;

        const CSR csr = sparse_to_amd_csr_(B);
        AMDReorderingArray amd(/*aggressive_absorption=*/true, /*dense_cutoff=*/-1);
        auto [perm, stats] = amd.compute_fill_reducing_permutation(csr, /*symmetrize=*/true);
        (void)stats;

        if (!is_valid_permutation_(perm, m_))
            return;

        sparse_row_perm_.assign(perm.begin(), perm.end());
        sparse_col_perm_ = sparse_row_perm_;
        sparse_ordering_cached_ = true;
    }

    // ----------------------------
    // Update-state reset / diagnostics
    // ----------------------------
    void reset_update_state_() {
        etas_.clear();
        update_count_ = 0;
        stats_ = Stats{};

        eta_density_sum_ = 0.0;
        eta_max_z_inf_ = 0.0;
        eta_max_w_inf_ = 0.0;
        eta_cumulative_z_inf_ = 0.0;
    }

    double adaptive_alpha_tol_(double reference_scale) const noexcept {
        return std::max(opt_.alpha_tol, 1e-12 * std::max(reference_scale, opt_.abs_floor));
    }

    double dynamic_pivot_floor_(const Eigen::VectorXd& incoming_col,
                                double reference_scale) const noexcept {
        return std::max({opt_.abs_floor,
                         opt_.pivot_rel * std::max(1.0, incoming_col.lpNorm<Eigen::Infinity>()),
                         1e-14 * std::max(reference_scale, opt_.abs_floor)});
    }

    static std::string format_metric_(double value) {
        if (!std::isfinite(value))
            return "nan";
        std::ostringstream oss;
        oss.setf(std::ios::scientific);
        oss.precision(3);
        oss << value;
        return oss.str();
    }

    std::string
    make_update_diagnostic_(const std::string& reason, double alpha,
                            const Eigen::VectorXd* z = nullptr, const Eigen::VectorXd* w = nullptr,
                            double growth = std::numeric_limits<double>::quiet_NaN(),
                            double residual = std::numeric_limits<double>::quiet_NaN()) const {
        std::ostringstream oss;
        oss << reason;
        oss << " alpha=" << format_metric_(alpha);
        if (z != nullptr)
            oss << " z_inf=" << format_metric_(z->lpNorm<Eigen::Infinity>());
        if (w != nullptr)
            oss << " w_inf=" << format_metric_(w->lpNorm<Eigen::Infinity>());
        if (std::isfinite(growth))
            oss << " growth=" << format_metric_(growth);
        if (std::isfinite(residual))
            oss << " residual=" << format_metric_(residual);
        return oss.str();
    }

    double dense_condition_estimate_() const {
        if (A_is_sparse_ || lu_dense_.U().rows() == 0)
            return 1.0;
        const Eigen::ArrayXd diag = lu_dense_.U().diagonal().array().abs();
        double min_diag = std::numeric_limits<double>::infinity();
        double max_diag = 0.0;
        for (int i = 0; i < diag.size(); ++i) {
            const double value = diag(i);
            if (!std::isfinite(value))
                return std::numeric_limits<double>::infinity();
            min_diag = std::min(min_diag, value);
            max_diag = std::max(max_diag, value);
        }
        if (!std::isfinite(min_diag) || min_diag <= opt_.abs_floor)
            return std::numeric_limits<double>::infinity();
        return max_diag / min_diag;
    }

    double stability_score_() const noexcept {
        auto ratio = [](double value, double guard) noexcept {
            if (!(guard > 0.0) || !std::isfinite(value))
                return 1.0;
            return std::clamp(value / guard, 0.0, 4.0);
        };

        double score = 0.0;
        score = std::max(score, ratio(stats_.last_column_residual, opt_.column_residual_tol));
        score = std::max(score, ratio(stats_.growth_from_last_refactor, opt_.max_growth_tol));
        score =
            std::max(score, ratio(stats_.cumulative_eta_z_inf, opt_.eta_cumulative_inf_norm_guard));
        score = std::max(score, ratio(stats_.estimated_condition, opt_.max_condition_estimate));
        return score;
    }

    double dynamic_growth_tol_() const noexcept {
        const double chain_progress = (opt_.refactor_every > 0)
                                          ? std::clamp(static_cast<double>(update_count_) /
                                                           static_cast<double>(opt_.refactor_every),
                                                       0.0, 1.0)
                                          : 1.0;
        const double eta_pressure =
            (opt_.eta_cumulative_inf_norm_guard > 0.0)
                ? std::clamp(stats_.cumulative_eta_z_inf / opt_.eta_cumulative_inf_norm_guard, 0.0,
                             1.0)
                : 0.0;
        const double pressure = std::max(chain_progress, eta_pressure);
        const double tightened = opt_.max_growth_tol * (1.0 - 0.75 * pressure);
        return std::max(opt_.min_dynamic_growth_tol, tightened);
    }

    int adaptive_refactor_limit_() const noexcept {
        const int base = std::max(1, current_refactor_every_);
        const double pressure = std::clamp(stats_.stability_score, 0.0, 1.0);
        const double scaled =
            1.0 - (1.0 - std::clamp(opt_.min_refactor_interval_fraction, 0.1, 1.0)) * pressure;
        return std::max(1, static_cast<int>(std::lround(static_cast<double>(base) * scaled)));
    }

    int adaptive_compress_limit_() const noexcept {
        const int base = std::max(1, opt_.compress_every);
        const double pressure = std::clamp(stats_.stability_score, 0.0, 1.0);
        const double scaled =
            1.0 - (1.0 - std::clamp(opt_.min_refactor_interval_fraction, 0.1, 1.0)) * pressure;
        return std::max(1, static_cast<int>(std::lround(static_cast<double>(base) * scaled)));
    }

    SparseForrestTomlinLU::Config make_sparse_lu_config_() const {
        SparseForrestTomlinLU::Config config;
        config.use_amd_ordering = opt_.sparse_amd;
        config.diagonal_equilibration = opt_.sparse_equilibration;
        config.iterative_refinement = opt_.enable_iterative_refinement;
        config.iterative_refinement_steps = std::max(1, opt_.refinement_steps);
        config.iterative_refinement_tol = opt_.residual_refactor_tol;
        config.max_norm_growth_before_refactor = std::max(1e4, opt_.max_growth_tol * 100.0);
        config.enable_solve_oracle = opt_.sparse_solve_oracle;
        config.validate_solves = opt_.sparse_validate_solves;
        std::string backend = opt_.sparse_backend;
        std::transform(backend.begin(), backend.end(), backend.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (backend == "eigen" || backend == "sparse_lu") {
            config.force_eigen_sparse_lu = true;
        } else if (backend == "pf" || backend == "product_form") {
            config.use_product_form_updates = true;
            config.update_method = SparseForrestTomlinLU::UpdateMethod::PF;
        } else if (backend == "mpf" || backend == "middle_product_form") {
            config.use_product_form_updates = false;
            config.update_method = SparseForrestTomlinLU::UpdateMethod::MPF;
        } else if (backend == "apf" || backend == "alternate_product_form") {
            config.use_product_form_updates = false;
            config.update_method = SparseForrestTomlinLU::UpdateMethod::APF;
        } else {
            // HiGHS' HFactor defaults to Forrest-Tomlin updates. Product form
            // remains available through basis_sparse_backend="pf".
            config.use_product_form_updates = false;
            config.update_method = SparseForrestTomlinLU::UpdateMethod::FT;
        }
        return config;
    }

    void set_dense_basis_column_(int col_j, const Eigen::VectorXd& dense) {
        Bcols_dense_[col_j] = dense;
        current_B_dense_.col(col_j) = dense;
    }

    double compute_refactor_sanity_residual_() const {
        if (m_ <= 0)
            return 0.0;

        Eigen::VectorXd rhs(m_);
        for (int i = 0; i < m_; ++i)
            rhs(i) = (i % 2 == 0) ? 1.0 : -1.0;

        const Eigen::VectorXd x = base_solve_B_(rhs);
        const Eigen::VectorXd residual = rhs - multiply_B_(x);
        const double denom = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
        return residual.lpNorm<Eigen::Infinity>() / denom;
    }

    bool should_verify_solve_quality_() const noexcept {
        if (m_ <= 0)
            return false;
        if (A_is_sparse_) {
            if (!lu_sparse_.has_updates())
                return false;
            if (lu_sparse_.needs_refactor())
                return true;
            if (update_count_ >= std::max(1, adaptive_refactor_limit_() / 2))
                return true;
            return stats_.stability_score >= 0.5;
        }
        if (etas_.empty())
            return true; // verify full refactor results even when no dense update chain exists
        if (update_count_ >= std::max(1, adaptive_refactor_limit_() / 2))
            return true;
        if (stats_.stability_score >= 0.5)
            return true;
        return stats_.estimated_condition > 0.5 * opt_.max_condition_estimate;
    }

    static double relative_residual_(const Eigen::VectorXd& residual,
                                     const Eigen::VectorXd& rhs) noexcept {
        const double denom = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
        return residual.lpNorm<Eigen::Infinity>() / denom;
    }

    bool solve_residual_ok_(double berr, double abs_residual) const noexcept {
        return std::isfinite(berr) && std::isfinite(abs_residual) &&
               (berr <= opt_.residual_refactor_tol ||
                abs_residual <= opt_.residual_abs_refactor_tol);
    }

    void verify_solve_quality_B_(const Eigen::VectorXd& rhs, const Eigen::VectorXd& x,
                                 const char* label) {
        if (!should_verify_solve_quality_())
            return;
        const Eigen::VectorXd residual = rhs - multiply_B_(x);
        const double abs_residual = residual.lpNorm<Eigen::Infinity>();
        const double berr = relative_residual_(residual, rhs);
        if (solve_residual_ok_(berr, abs_residual))
            return;
        throw std::runtime_error(std::string(label) +
                                 " suspicious residual berr=" + format_metric_(berr) +
                                 " abs_residual=" + format_metric_(abs_residual));
    }

    void verify_solve_quality_BT_(const Eigen::VectorXd& rhs, const Eigen::VectorXd& y,
                                  const char* label) {
        if (!should_verify_solve_quality_())
            return;
        const Eigen::VectorXd residual = rhs - multiply_BT_(y);
        const double abs_residual = residual.lpNorm<Eigen::Infinity>();
        const double berr = relative_residual_(residual, rhs);
        if (solve_residual_ok_(berr, abs_residual))
            return;
        throw std::runtime_error(std::string(label) +
                                 " suspicious residual berr=" + format_metric_(berr) +
                                 " abs_residual=" + format_metric_(abs_residual));
    }

    void refresh_refactor_diagnostics_() {
        stats_.growth_factor = 1.0;
        stats_.growth_from_last_refactor = 1.0;
        stats_.growth_from_initial_refactor = 1.0;
        stats_.estimated_condition = A_is_sparse_ ? 1.0 : dense_condition_estimate_();
        stats_.sparse_norm_growth_estimate = 1.0;
        try {
            stats_.last_refactor_sanity_residual = compute_refactor_sanity_residual_();
        } catch (...) {
            stats_.last_refactor_sanity_residual = std::numeric_limits<double>::infinity();
        }
        stats_.stability_score = stability_score_();
        if (!A_is_sparse_ && stats_.estimated_condition > opt_.max_condition_estimate) {
            last_update_diagnostic_ =
                "Dense refactor produced a poorly conditioned basis est_cond=" +
                format_metric_(stats_.estimated_condition);
        }
    }

    void initialize_dense_basis_cache_() {
        current_B_dense_.resize(m_, m_);
        for (int k = 0; k < m_; ++k)
            current_B_dense_.col(k) = Bcols_dense_[k];
    }

    void rebuild_sparse_basis_cache_() {
        std::vector<Eigen::Triplet<double>> trips;
        size_t nnz_hint = 0;
        for (const auto& col : Bcols_sparse_)
            nnz_hint += static_cast<size_t>(col.nonZeros());
        trips.reserve(std::max<size_t>(1, nnz_hint));

        for (int k = 0; k < m_; ++k) {
            const auto& col = Bcols_sparse_[k];
            for (SparseMat::InnerIterator it(col, 0); it; ++it)
                trips.emplace_back(it.row(), k, it.value());
        }

        current_B_sparse_.resize(m_, m_);
        current_B_sparse_.setFromTriplets(trips.begin(), trips.end());
        if (opt_.sparse_drop_tol > 0.0)
            current_B_sparse_.prune(opt_.sparse_drop_tol);
        current_B_sparse_.makeCompressed();
        current_B_sparse_dirty_ = false;
    }

    void ensure_current_B_sparse_() const {
        if (!A_is_sparse_)
            return;
        if (!current_B_sparse_dirty_)
            return;
        auto* self = const_cast<FTBasis*>(this);
        self->rebuild_sparse_basis_cache_();
    }

    void mark_sparse_basis_cache_dirty_() noexcept { current_B_sparse_dirty_ = true; }

    void dense_refactor_() {
        const auto t0 = std::chrono::steady_clock::now();
        lu_dense_.factor(current_B_dense_, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters);
        max_element_ = lu_dense_.U().cwiseAbs().maxCoeff();
        refactor_baseline_max_element_ = std::max(1.0, max_element_);
        if (initial_refactor_max_element_ == 0.0)
            initial_refactor_max_element_ = refactor_baseline_max_element_;
        current_refactor_every_ = opt_.refactor_every; // restore after any backtrack halving
        reset_update_state_();
        refresh_refactor_diagnostics_();
        report_refactor_telemetry_(t0);
        save_backtracking_basis_();
    }

    void sparse_build_B_(SparseMat& B) const {
        ensure_current_B_sparse_();
        B = current_B_sparse_;
    }

    void sparse_refactor_() {
        const auto t0 = std::chrono::steady_clock::now();
        SparseMat B;
        sparse_build_B_(B);
        const auto config = make_sparse_lu_config_();

        // The AMD pre-permutation fast path (factor row_perm*B*col_perm while
        // passing the permutations as initial Pr/Pc) returned solutions in a
        // wrong index order for every non-identity ordering: solve_B produced
        // garbage xB, so any sparse solve whose first factorization had a
        // non-trivial AMD ordering (e.g. every phase-2 start after phase 1)
        // failed with NeedPhase1/Singular. No row/col direct-vs-inverse
        // convention of the pre-permutation satisfies the solve contract, so
        // the initial-permutation pathway of SparseForrestTomlinLU::factor is
        // unusable as-is. Factor B directly — the Markowitz kernel performs
        // its own fill-reducing pivoting.
        lu_sparse_.factor(B, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters,
                          opt_.ft_bandwidth_cap, nullptr, nullptr, config);
        current_refactor_every_ = opt_.refactor_every; // restore after any backtrack halving
        reset_update_state_();
        refresh_refactor_diagnostics_();
        report_refactor_telemetry_(t0);
        save_backtracking_basis_();
    }

    void report_refactor_telemetry_(const std::chrono::steady_clock::time_point& t0) noexcept {
        if (opt_.ext_refactor_counter)
            ++(*opt_.ext_refactor_counter);
        if (opt_.ext_refactor_ns) {
            const auto dt = std::chrono::steady_clock::now() - t0;
            *opt_.ext_refactor_ns += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count());
        }
    }

    // ----------------------------
    // Sparse rebuild using HiGHS-style RefactorInfo pivot records.
    // Reconstructs LU factorization from cached pivot sequence without
    // starting from scratch. Handles special pivot types: logical, unit.
    // Returns 0 on success, non-zero if rank deficiency detected.
    int sparse_rebuild_() {
        // For now, sparse rebuild is not fully implemented for the sparse LU backend.
        // The HiGHS HFactor::rebuild() logic requires specific data structures
        // (l_start, l_index, l_value, u_start, u_index, u_value) that SparseForrestTomlinLU
        // does not have. SparseForrestTomlinLU uses U_rows_, L_rows_, U_cols_, L_cols_.
        //
        // For logical/unit pivots only, we could implement a simple rebuild, but for
        // now we fall back to full refactor to ensure correctness.
        //
        // TODO: Implement full sparse rebuild logic mirroring HiGHS HFactor::rebuild()
        // for Markowitz pivots using forward_solve_L_ and forming L/U factors incrementally.

        lu_sparse_.refactor_info_.clear();
        sparse_refactor_();
        return 0;
    }

    // ----------------------------
    // Base solves / fast solves
    //
    // expected_density is the HiGHS-style EWMA of (count/m_) for this TRAN
    // class. It propagates into the sparse LU's per-stage hyper/sparse gating.
    // For dense LU it's ignored (Markowitz LU uses dense back-/forward-solves
    // with no hyper-sparse path to gate).
    // ----------------------------
    Eigen::VectorXd base_solve_B_(const Eigen::VectorXd& b, double expected_density = 1.0) const {
        return A_is_sparse_ ? lu_sparse_.solve(b, expected_density) : lu_dense_.solve(b);
    }

    Eigen::VectorXd base_solve_BT_(const Eigen::VectorXd& c, double expected_density = 1.0) const {
        return A_is_sparse_ ? lu_sparse_.solveT(c, expected_density) : lu_dense_.solveT(c);
    }

    Eigen::VectorXd solve_B_fast_(const Eigen::VectorXd& b, double expected_density = 1.0) const {
        Eigen::VectorXd x = base_solve_B_(b, expected_density);
        if (!A_is_sparse_ && !etas_.empty())
            x = apply_etas_solve_(x);
        return x;
    }

    Eigen::VectorXd solve_BT_fast_(const Eigen::VectorXd& c, double expected_density = 1.0) const {
        Eigen::VectorXd y = base_solve_BT_(c, expected_density);
        if (!A_is_sparse_ && !etas_.empty())
            y = apply_etas_solve_T_(y);
        return y;
    }

    // ----------------------------
    // Explicit multiplications
    // ----------------------------
    Eigen::VectorXd multiply_B_(const Eigen::VectorXd& x) const {
        if (x.size() != m_)
            throw std::invalid_argument("FTBasis::multiply_B_ size mismatch");

        if (A_is_sparse_) {
            ensure_current_B_sparse_();
            return current_B_sparse_ * x;
        }
        return current_B_dense_ * x;
    }

    Eigen::VectorXd multiply_BT_(const Eigen::VectorXd& y) const {
        if (y.size() != m_)
            throw std::invalid_argument("FTBasis::multiply_BT_ size mismatch");

        if (A_is_sparse_) {
            ensure_current_B_sparse_();
            return current_B_sparse_.transpose() * y;
        }
        return current_B_dense_.transpose() * y;
    }

    // ----------------------------
    // Refinement
    // ----------------------------
    Eigen::VectorXd refine_solve_B_(const Eigen::VectorXd& rhs, Eigen::VectorXd x) const {
        int max_steps = std::max(0, opt_.refinement_steps);
        if (max_steps == 0)
            return x;

        const int hard_cap = std::max(max_steps, opt_.refinement_max_steps);
        double previous_berr = std::numeric_limits<double>::infinity();
        double final_berr = std::numeric_limits<double>::infinity();
        double final_abs_residual = std::numeric_limits<double>::infinity();
        int stall_steps = 0;

        for (int it = 0; it < max_steps; ++it) {
            const Eigen::VectorXd Bx = multiply_B_(x);
            const Eigen::VectorXd r = rhs - Bx;
            final_abs_residual = r.lpNorm<Eigen::Infinity>();
            const double denom = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
            const double berr = final_abs_residual / denom;
            final_berr = berr;
            if (!std::isfinite(berr) || (berr < opt_.residual_refactor_tol &&
                                         final_abs_residual < opt_.residual_abs_refactor_tol))
                break;
            if (std::isfinite(previous_berr)) {
                if (berr > previous_berr * opt_.refinement_stall_progress_ratio) {
                    ++stall_steps;
                } else {
                    stall_steps = 0;
                }
                if (stall_steps >= std::max(1, opt_.refinement_stall_limit)) {
                    throw std::runtime_error(
                        "FTBasis::refine_solve_B stalled berr=" + format_metric_(berr) +
                        " abs_residual=" + format_metric_(final_abs_residual));
                }
            }

            Eigen::VectorXd dx = solve_B_fast_(r);
            if (!dx.array().isFinite().all())
                break;
            x += dx;

            if (it + 1 == max_steps && max_steps < hard_cap &&
                berr > previous_berr * opt_.refinement_slow_progress_ratio) {
                ++max_steps;
            }
            previous_berr = berr;
        }

        if (!std::isfinite(final_berr) || (final_berr > opt_.residual_refactor_tol &&
                                           final_abs_residual > opt_.residual_abs_refactor_tol)) {
            throw std::runtime_error(
                "FTBasis::refine_solve_B residual remained large after refinement berr=" +
                format_metric_(final_berr) + " abs_residual=" + format_metric_(final_abs_residual));
        }
        return x;
    }

    Eigen::VectorXd refine_solve_BT_(const Eigen::VectorXd& rhs, Eigen::VectorXd y) const {
        int max_steps = std::max(0, opt_.refinement_steps);
        if (max_steps == 0)
            return y;

        const int hard_cap = std::max(max_steps, opt_.refinement_max_steps);
        double previous_berr = std::numeric_limits<double>::infinity();
        double final_berr = std::numeric_limits<double>::infinity();
        double final_abs_residual = std::numeric_limits<double>::infinity();
        int stall_steps = 0;

        for (int it = 0; it < max_steps; ++it) {
            const Eigen::VectorXd r = rhs - multiply_BT_(y);
            final_abs_residual = r.lpNorm<Eigen::Infinity>();
            const double denom = std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
            const double berr = final_abs_residual / denom;
            final_berr = berr;
            if (!std::isfinite(berr) || (berr < opt_.residual_refactor_tol &&
                                         final_abs_residual < opt_.residual_abs_refactor_tol))
                break;
            if (std::isfinite(previous_berr)) {
                if (berr > previous_berr * opt_.refinement_stall_progress_ratio) {
                    ++stall_steps;
                } else {
                    stall_steps = 0;
                }
                if (stall_steps >= std::max(1, opt_.refinement_stall_limit)) {
                    throw std::runtime_error(
                        "FTBasis::refine_solve_BT stalled berr=" + format_metric_(berr) +
                        " abs_residual=" + format_metric_(final_abs_residual));
                }
            }

            Eigen::VectorXd dy = solve_BT_fast_(r);
            if (!dy.array().isFinite().all())
                break;
            y += dy;

            if (it + 1 == max_steps && max_steps < hard_cap &&
                berr > previous_berr * opt_.refinement_slow_progress_ratio) {
                ++max_steps;
            }
            previous_berr = berr;
        }

        if (!std::isfinite(final_berr) || (final_berr > opt_.residual_refactor_tol &&
                                           final_abs_residual > opt_.residual_abs_refactor_tol)) {
            throw std::runtime_error(
                "FTBasis::refine_solve_BT residual remained large after refinement berr=" +
                format_metric_(final_berr) + " abs_residual=" + format_metric_(final_abs_residual));
        }
        return y;
    }

    // ----------------------------
    // Dense FT
    // ----------------------------
    void forrest_tomlin_update_dense_(int j, const Eigen::VectorXd& incoming_col,
                                      const Eigen::VectorXd& z, double alpha) {
        Eigen::MatrixXd& L = lu_dense_.L();
        Eigen::MatrixXd& U = lu_dense_.U();
        const int n = static_cast<int>(U.rows());
        const double old_pivot = std::abs(U(j, j));
        const double alpha_floor = adaptive_alpha_tol_(old_pivot);

        if (!std::isfinite(alpha) || std::abs(alpha) < alpha_floor)
            throw std::runtime_error("Forrest-Tomlin: alpha too small/unstable");

        Eigen::VectorXd contrib = U * z;
        const double contrib_floor =
            std::max(opt_.abs_floor, 1e-14 * std::max({1.0, U.col(j).lpNorm<Eigen::Infinity>(),
                                                       incoming_col.lpNorm<Eigen::Infinity>(),
                                                       z.lpNorm<Eigen::Infinity>()}));
        for (int i = 0; i < contrib.size(); ++i) {
            if (std::abs(contrib(i)) < contrib_floor)
                contrib(i) = 0.0;
        }
        U.col(j) += contrib;

        const double pivot = U(j, j);
        const double pivot_floor = dynamic_pivot_floor_(incoming_col, old_pivot);
        if (!std::isfinite(pivot) || std::abs(pivot) < pivot_floor ||
            std::abs(pivot) < std::abs(contrib(j)) * opt_.pivot_rel) {
            throw std::runtime_error("Forrest-Tomlin: new pivot too small");
        }

        const int band = (opt_.ft_bandwidth_cap > 0) ? opt_.ft_bandwidth_cap : n;
        const int i_lo = j + 1;
        const int i_hi = std::min(n - 1, j + band);

        for (int i = i_lo; i <= i_hi; ++i) {
            const double factor = U(i, j) / pivot;
            if (!std::isfinite(factor) || std::abs(factor) > opt_.ft_multiplier_guard) {
                throw std::runtime_error("Forrest-Tomlin: elimination multiplier too large");
            }
            if (std::abs(factor) > 1e-16) {
                L(i, j) = factor;
                U.row(i).segment(j, n - j).noalias() -= factor * U.row(j).segment(j, n - j);
                U(i, j) = 0.0;
            }
        }

        L(j, j) = 1.0;
        if (std::abs(U(j, j)) < pivot_floor)
            throw std::runtime_error("Forrest-Tomlin: pivot collapsed");
    }

    // ----------------------------
    // Column helpers
    // ----------------------------
    static Eigen::VectorXd sparse_column_to_dense_(const SparseMat& col) {
        Eigen::VectorXd dense = Eigen::VectorXd::Zero(col.rows());
        for (SparseMat::InnerIterator it(col, 0); it; ++it)
            dense[it.row()] = it.value();
        return dense;
    }

    void set_sparse_column_(int col_j, const Eigen::VectorXd& dense) {
        std::vector<Eigen::Triplet<double>> tr;
        tr.reserve(static_cast<size_t>(std::min<int>(dense.size(), 32)));
        for (int r = 0; r < dense.size(); ++r) {
            const double v = dense[r];
            if (std::abs(v) > 0.0)
                tr.emplace_back(r, 0, v);
        }
        SparseMat col(m_, 1);
        if (!tr.empty())
            col.setFromTriplets(tr.begin(), tr.end());
        col.makeCompressed();
        Bcols_sparse_[col_j] = std::move(col);
        mark_sparse_basis_cache_dirty_();
    }

    void set_sparse_column_(int col_j, const SparseMat& sparse) {
        Bcols_sparse_[col_j] = sparse;
        mark_sparse_basis_cache_dirty_();
    }

    // ----------------------------
    // Eta application
    // ----------------------------
    Eigen::VectorXd apply_etas_solve_(Eigen::VectorXd x) const {
        for (const auto& eta : etas_) {
            const double xj = x(eta.j);
            if (xj != 0.0)
                x.noalias() -= eta.z * (xj / eta.alpha);
        }
        return x;
    }

    Eigen::VectorXd apply_etas_solve_T_(Eigen::VectorXd y) const {
        for (const auto& eta : etas_) {
            const double uy = eta.u.dot(y);
            if (uy != 0.0)
                y.noalias() -= eta.w * (uy / eta.alpha);
        }
        return y;
    }

    // ----------------------------
    // Monitoring
    // ----------------------------
    void reset_incremental_eta_stats_() noexcept {
        eta_density_sum_ = 0.0;
        eta_cumulative_z_inf_ = 0.0;
        eta_max_z_inf_ = 0.0;
        eta_max_w_inf_ = 0.0;
    }

    void absorb_eta_stats_(const Eta& e) noexcept {
        const double zinf = e.z_inf_norm();
        const double winf = e.w_inf_norm();
        eta_density_sum_ += e.z_density();
        eta_cumulative_z_inf_ += zinf;
        eta_max_z_inf_ = std::max(eta_max_z_inf_, zinf);
        eta_max_w_inf_ = std::max(eta_max_w_inf_, winf);
    }

    void recompute_incremental_eta_stats_from_chain_() noexcept {
        reset_incremental_eta_stats_();
        for (const auto& e : etas_)
            absorb_eta_stats_(e);
    }

    void ensure_sparse_basis_current_() const {
        if (!A_is_sparse_)
            return;
        if (!current_B_sparse_dirty_)
            return;
        auto* self = const_cast<FTBasis*>(this);
        self->rebuild_sparse_basis_cache_();
        self->current_B_sparse_dirty_ = false;
    }

    double factorization_column_residual_(int j) const {
        if (j < 0 || j >= m_)
            throw std::out_of_range("FTBasis::factorization_column_residual_ bad j");

        Eigen::VectorXd e = unit_basis_vector_(j);
        Eigen::VectorXd x = fast_solve_B_(e);
        Eigen::VectorXd r = multiply_B_(x) - e;
        const double denom = std::max(1.0, e.lpNorm<Eigen::Infinity>());
        return r.lpNorm<Eigen::Infinity>() / denom;
    }

    bool should_check_factorization_column_residual_() const noexcept {
        if (update_count_ <= 0)
            return false;
        if (update_count_ == 1)
            return true;
        if (stats_.stability_score >= 0.50)
            return true;
        if (update_count_ >= std::max(2, adaptive_refactor_limit_() / 2))
            return true;
        return (update_count_ % residual_check_period_()) == 0;
    }

    int residual_check_period_() const noexcept {
        const int base = std::max(4, std::min(32, opt_.compress_every));
        if (stats_.stability_score >= 0.75)
            return std::max(2, base / 2);
        return base;
    }

    void refresh_stats_() {
        stats_.update_count = update_count_;

        if (A_is_sparse_) {
            const auto sparse_stats = lu_sparse_.update_stats();
            stats_.eta_count = sparse_stats.count;
            stats_.max_eta_z_inf = sparse_stats.max_z_inf;
            stats_.max_eta_w_inf = sparse_stats.max_w_inf;
            stats_.avg_eta_density = sparse_stats.avg_z_density;
            stats_.cumulative_eta_z_inf = sparse_stats.cumulative_z_inf;
            stats_.growth_factor = sparse_stats.norm_growth_estimate;
            stats_.growth_from_last_refactor = sparse_stats.norm_growth_estimate;
            stats_.growth_from_initial_refactor = sparse_stats.norm_growth_estimate;
            stats_.estimated_condition = 1.0;
            stats_.sparse_norm_growth_estimate = sparse_stats.norm_growth_estimate;
            stats_.stability_score = stability_score_();
            return;
        }

        stats_.eta_count = static_cast<int>(etas_.size());
        stats_.max_eta_z_inf = eta_max_z_inf_;
        stats_.max_eta_w_inf = eta_max_w_inf_;
        stats_.avg_eta_density =
            etas_.empty() ? 0.0 : eta_density_sum_ / static_cast<double>(etas_.size());
        stats_.cumulative_eta_z_inf = eta_cumulative_z_inf_;
        stats_.growth_from_last_refactor =
            max_element_ / std::max(1.0, refactor_baseline_max_element_);
        stats_.growth_from_initial_refactor =
            max_element_ / std::max(1.0, initial_refactor_max_element_);
        stats_.growth_factor = stats_.growth_from_last_refactor;
        stats_.estimated_condition = dense_condition_estimate_();
        stats_.sparse_norm_growth_estimate = 1.0;
        stats_.stability_score = stability_score_();
    }

    bool bad_factorization_column_residual_(int j) {
        if (!should_check_factorization_column_residual_()) {
            stats_.last_column_residual = 0.0;
            return false;
        }
        stats_.last_column_residual = factorization_column_residual_(j);
        return (!std::isfinite(stats_.last_column_residual)) ||
               (stats_.last_column_residual > opt_.column_residual_tol);
    }

    bool sparse_update_looks_doomed_(const Eigen::VectorXd& old_col,
                                     const Eigen::VectorXd& new_col) const noexcept {
        if (!lu_sparse_.supports_inplace_updates())
            return true;
        if (lu_sparse_.needs_refactor())
            return true;
        if (update_count_ >= adaptive_refactor_limit_())
            return true;
        if (stats_.stability_score >= 0.95)
            return true;

        const double old_norm = std::max(1.0, old_col.lpNorm<Eigen::Infinity>());
        const double new_norm = new_col.lpNorm<Eigen::Infinity>();
        if (!std::isfinite(new_norm))
            return true;
        if (new_norm > old_norm * std::max(100.0, opt_.ft_multiplier_guard))
            return true;

        return false;
    }

    bool need_compress_() const noexcept {
        const int update_chain_size =
            A_is_sparse_ ? lu_sparse_.update_stats().count : static_cast<int>(etas_.size());

        if (update_chain_size >= adaptive_compress_limit_())
            return true;
        if (update_chain_size >= opt_.max_eta_count)
            return true;
        if (update_count_ >= adaptive_refactor_limit_())
            return true;
        // HiGHS reinversion economics: rebuild once the update chain costs
        // more per solve than a fresh factorization would.
        if (A_is_sparse_ && lu_sparse_.synthetic_clock_says_refactor())
            return true;

        const bool hard_guard =
            (stats_.max_eta_z_inf > opt_.eta_max_inf_norm) ||
            (stats_.cumulative_eta_z_inf > opt_.eta_cumulative_inf_norm_guard) ||
            (stats_.growth_from_last_refactor > dynamic_growth_tol_()) ||
            (stats_.growth_from_initial_refactor >
             dynamic_growth_tol_() * std::max(10.0, opt_.max_growth_tol)) ||
            (stats_.estimated_condition > opt_.max_condition_estimate);

        if (hard_guard)
            return true;

        int soft_hits = 0;
        if (stats_.avg_eta_density > opt_.eta_avg_density_guard)
            ++soft_hits;
        if (stats_.max_eta_z_inf > opt_.z_inf_guard)
            ++soft_hits;
        if (stats_.stability_score > 0.9)
            ++soft_hits;

        return soft_hits >= 2;
    }

    // ----------------------------
    // Fast internal solves for update construction
    // ----------------------------
    Eigen::VectorXd fast_solve_B_(const Eigen::VectorXd& b) const {
        Eigen::VectorXd x = base_solve_B_(b);
        if (!A_is_sparse_ && !etas_.empty())
            x = apply_etas_solve_(x);
        return x;
    }

    Eigen::VectorXd fast_solve_BT_(const Eigen::VectorXd& c) const {
        Eigen::VectorXd y = base_solve_BT_(c);
        if (!A_is_sparse_ && !etas_.empty())
            y = apply_etas_solve_T_(y);
        return y;
    }

    const Eigen::VectorXd& unit_basis_vector_(int j) const {
        if (workspace_ej_.size() != m_)
            workspace_ej_ = Eigen::VectorXd::Zero(m_);
        else
            workspace_ej_.setZero();
        workspace_ej_(j) = 1.0;
        return workspace_ej_;
    }

    // ----------------------------
    // Replace column
    // ----------------------------
    void replace_column_impl_(int j, std::optional<int> entering_col,
                              const Eigen::VectorXd& new_col_dense,
                              const SparseMat* new_col_sparse = nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        auto report_pivot_telemetry = [&]() noexcept {
            if (opt_.ext_pivot_ns) {
                *opt_.ext_pivot_ns +=
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                   std::chrono::steady_clock::now() - t0)
                                                   .count());
            }
        };
        if (j < 0 || j >= m_)
            throw std::out_of_range("FTBasis::replace_column bad j");
        if (new_col_dense.size() != m_)
            throw std::invalid_argument("FTBasis::replace_column size mismatch");

        last_update_diagnostic_.clear();
        const std::optional<int> pending_basis = entering_col;
        struct BasisCommit {
            FTBasis* self;
            int j;
            const std::optional<int> pending_basis;
            ~BasisCommit() {
                if (pending_basis.has_value())
                    self->basis_[j] = *pending_basis;
            }
        } basis_commit{this, j, pending_basis};

        if (A_is_sparse_) {
            ensure_sparse_basis_current_();
            const Eigen::VectorXd old_dense = Eigen::VectorXd(current_B_sparse_.col(j));

            if (sparse_update_looks_doomed_(old_dense, new_col_dense)) {
                if (new_col_sparse)
                    set_sparse_column_(j, *new_col_sparse);
                else
                    set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }

            Eigen::VectorXd u = new_col_dense - old_dense;
            const double alpha_floor = adaptive_alpha_tol_(old_dense.lpNorm<Eigen::Infinity>());

            Eigen::VectorXd z, w;
            double alpha = 0.0;

            try {
                z = fast_solve_B_(u);
                w = fast_solve_BT_(unit_basis_vector_(j));
                alpha = 1.0 + z(j);
            } catch (...) {
                if (new_col_sparse)
                    set_sparse_column_(j, *new_col_sparse);
                else
                    set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }

            const bool unstable = (!std::isfinite(alpha)) || (std::abs(alpha) < alpha_floor) ||
                                  (!z.array().isFinite().all()) || (!w.array().isFinite().all()) ||
                                  (z.lpNorm<Eigen::Infinity>() > opt_.z_inf_guard);

            if (unstable) {
                last_update_diagnostic_ = make_update_diagnostic_(
                    "Sparse FT update rejected by adaptive alpha or finite guards", alpha, &z, &w);
                if (new_col_sparse)
                    set_sparse_column_(j, *new_col_sparse);
                else
                    set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }

            if (!lu_sparse_.append_forrest_tomlin_update(j, u, z, w, alpha,
                                                         std::max(opt_.abs_floor, 1e-14))) {
                last_update_diagnostic_ = make_update_diagnostic_(
                    lu_sparse_.last_update_failure_reason_message(), alpha, &z, &w);
                if (new_col_sparse)
                    set_sparse_column_(j, *new_col_sparse);
                else
                    set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }

            if (new_col_sparse)
                set_sparse_column_(j, *new_col_sparse);
            else
                set_sparse_column_(j, new_col_dense);
            ++update_count_;
            if (opt_.ext_ft_update_counter)
                ++(*opt_.ext_ft_update_counter);
            refresh_stats_();

            if (bad_factorization_column_residual_(j) || need_compress_() ||
                lu_sparse_.needs_refactor())
                sparse_refactor_();
            report_pivot_telemetry();
            return;
        }

        const Eigen::VectorXd old = current_B_dense_.col(j);
        Eigen::VectorXd u = new_col_dense - old;
        const double alpha_floor = adaptive_alpha_tol_(std::abs(lu_dense_.U()(j, j)));

        Eigen::VectorXd z, w;
        double alpha = 0.0;

        try {
            z = fast_solve_B_(u);
            w = fast_solve_BT_(unit_basis_vector_(j));
            alpha = 1.0 + z(j);
        } catch (...) {
            set_dense_basis_column_(j, new_col_dense);
            dense_refactor_();
            return;
        }

        const bool allow_ft_mode = opt_.update_mode == Options::UpdateMode::ForrestTomlin ||
                                   opt_.update_mode == Options::UpdateMode::Hybrid;
        const bool hybrid_prefers_ft = opt_.update_mode != Options::UpdateMode::Hybrid ||
                                       (stats_.estimated_condition <= opt_.max_condition_estimate &&
                                        stats_.stability_score < 0.75);

        const bool try_ft =
            lu_dense_.supports_inplace_updates() && allow_ft_mode && hybrid_prefers_ft &&
            std::abs(alpha) >= alpha_floor && update_count_ < adaptive_refactor_limit_() &&
            z.cwiseAbs().maxCoeff() <= opt_.z_inf_guard && w.array().isFinite().all();

        if (try_ft) {
            bool ok = true;
            try {
                forrest_tomlin_update_dense_(j, new_col_dense, z, alpha);
                const double current_max = lu_dense_.U().cwiseAbs().maxCoeff();
                const double dynamic_growth_tol = dynamic_growth_tol_();
                const double baseline_growth =
                    current_max / std::max(1.0, refactor_baseline_max_element_);
                const double initial_growth =
                    current_max / std::max(1.0, initial_refactor_max_element_);
                const double cond_estimate = dense_condition_estimate_();

                if (current_max > std::max(1.0, max_element_) * dynamic_growth_tol ||
                    baseline_growth > dynamic_growth_tol ||
                    initial_growth > dynamic_growth_tol * std::max(dynamic_growth_tol, 10.0) ||
                    cond_estimate > opt_.max_condition_estimate) {
                    last_update_diagnostic_ = make_update_diagnostic_(
                        "Dense FT update exceeded stability guards", alpha, &z, &w, baseline_growth,
                        stats_.last_column_residual);
                    ok = false;
                } else {
                    max_element_ = std::max(max_element_, current_max);
                }
            } catch (const std::exception& err) {
                last_update_diagnostic_ = err.what();
                ok = false;
            }

            if (!ok) {
                set_dense_basis_column_(j, new_col_dense);
                dense_refactor_();
                return;
            }

            set_dense_basis_column_(j, new_col_dense);
            ++update_count_;
            refresh_stats_();

            if (bad_factorization_column_residual_(j) ||
                (opt_.aggressive_refactor_on_suspicious_residual &&
                 stats_.last_column_residual > opt_.column_residual_tol) ||
                need_compress_()) {
                dense_refactor_();
            }
            report_pivot_telemetry();
            return;
        }

        const bool refactor_now =
            (std::abs(alpha) < alpha_floor) || (update_count_ >= adaptive_refactor_limit_()) ||
            (!std::isfinite(alpha)) || (stats_.estimated_condition > opt_.max_condition_estimate);

        if (refactor_now) {
            last_update_diagnostic_ = make_update_diagnostic_(
                "Dense update fell back to refactor due to stability guards", alpha, &z, &w);
            set_dense_basis_column_(j, new_col_dense);
            dense_refactor_();
            return;
        }

        set_dense_basis_column_(j, new_col_dense);
        etas_.push_back(Eta{j, std::move(u), std::move(z), std::move(w), alpha});
        absorb_eta_stats_(etas_.back());

        ++update_count_;
        refresh_stats_();

        if ((opt_.aggressive_refactor_on_suspicious_residual &&
             bad_factorization_column_residual_(j)) ||
            need_compress_()) {
            dense_refactor_();
        }
        report_pivot_telemetry();
    }

    // NLA hook — save current basis for backtracking (public API for SimplexNLA)
  public:
    void save_backtracking_basis_() {
        Snapshot snap;
        snap.basis = basis_;
        if (A_is_sparse_)
            snap.Bcols_sparse = Bcols_sparse_;
        else
            snap.Bcols_dense = Bcols_dense_;
        backtracking_snapshot_ = std::move(snap);
    }

  private:
    const DenseMat* A_dense_{nullptr};
    const SparseMat* A_sparse_{nullptr};
    bool A_is_sparse_{false};

    std::vector<Eigen::VectorXd> Bcols_dense_;
    std::vector<SparseMat> Bcols_sparse_;
    DenseMat current_B_dense_;
    mutable SparseMat current_B_sparse_;
    mutable bool current_B_sparse_dirty_{false};

    bool sparse_ordering_cached_{false};
    std::vector<int> sparse_row_perm_, sparse_col_perm_;

    MarkowitzLU lu_dense_;
    SparseForrestTomlinLU lu_sparse_;

    int m_{0};
    std::vector<int> basis_;
    Options opt_;
    std::vector<Eta> etas_;
    int update_count_{0};
    int current_refactor_every_{
        32}; // mirrors opt_.refactor_every; halved on backtrack, restored on clean refactor
    double max_element_{0.0};
    double refactor_baseline_max_element_{0.0};
    double initial_refactor_max_element_{0.0};
    Stats stats_{};
    std::string last_update_diagnostic_;

    // Incremental dense eta-chain stats
    double eta_density_sum_{0.0};
    double eta_cumulative_z_inf_{0.0};
    double eta_max_z_inf_{0.0};
    double eta_max_w_inf_{0.0};

    // Small reusable workspace
    mutable Eigen::VectorXd workspace_ej_;

    // Last successfully-refactored basis snapshot. Captured at the end of
    // dense_refactor_/sparse_refactor_; consumed by try_backtrack_to_last_good
    // when an FT-updated basis later goes singular.
    struct Snapshot {
        std::vector<int> basis;
        std::vector<Eigen::VectorXd> Bcols_dense;
        std::vector<SparseMat> Bcols_sparse;
    };
    std::optional<Snapshot> backtracking_snapshot_;

    // HiGHS-style per-class FTRAN/BTRAN density EWMA. Survives refactor() —
    // density is treated as a property of the LP, not the basis.
    DensityTracker density_tracker_;

    // Feed the result count back into the EWMA after a tagged solve.
    // When the result has a known sparse pattern (HVector::has_pattern()),
    // we use that count directly; otherwise scan the dense vector for
    // entries above kSparsePatternTol_.
    static constexpr double kDensityScanTol_ = 1e-14;
    void update_density_tracker_(TranKind kind, const HVector& out) {
        if (kind == TranKind::Unknown || m_ <= 0)
            return;
        int count;
        if (out.has_pattern()) {
            count = out.count;
        } else {
            count = 0;
            for (Eigen::Index i = 0; i < out.value.size(); ++i)
                if (std::abs(out.value(i)) > kDensityScanTol_)
                    ++count;
        }
        density_tracker_.update(kind, count, m_);
    }
};

// Compute a hash of the B matrix for warm-reuse verification
inline std::uint64_t FTBasis::basis_matrix_signature_() const {
    std::uint64_t sig = 0xcbf29ce484222325ULL;
    sig ^= (static_cast<std::uint64_t>(m_) + 0xc6b1a7c3d5e9f0a2ULL);
    sig = (sig << 31) | (sig >> 33);
    if (A_is_sparse_) {
        ensure_current_B_sparse_();
        const auto& B = current_B_sparse_;
        sig ^= static_cast<std::uint64_t>(B.nonZeros());
        sig = (sig << 31) | (sig >> 33);
        const int* outer = B.outerIndexPtr();
        const int* inner = B.innerIndexPtr();
        const double* value = B.valuePtr();
        for (int j = 0; j < m_; ++j) {
            sig ^= static_cast<std::uint64_t>(outer[j]);
            sig = (sig << 31) | (sig >> 33);
            for (int k = outer[j]; k < (j < m_ - 1 ? outer[j + 1] : B.nonZeros()); ++k) {
                sig ^= static_cast<std::uint64_t>(inner[k]);
                sig = (sig << 31) | (sig >> 33);
                std::uint64_t hv = std::bit_cast<std::uint64_t>(value[k]);
                sig ^= hv;
                sig = (sig << 31) | (sig >> 33);
            }
        }
    } else {
        for (int i = 0; i < m_; ++i)
            for (int j = 0; j < m_; ++j) {
                std::uint64_t hv = std::bit_cast<std::uint64_t>(current_B_dense_(i, j));
                sig ^= hv;
                sig = (sig << 31) | (sig >> 33);
            }
    }
    return sig;
}
