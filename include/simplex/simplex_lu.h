#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>

#include <ankerl/unordered_dense.h>

#include "amd.h"
#include "markowitz.h"
#include "sparse_lu.h"

#include <algorithm>
#include <cmath>
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
// FTBasis v3
// ======================================================
class FTBasis {
  public:
    using DenseMat = Eigen::MatrixXd;
    using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using Permutation = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int>;

    struct Options {
        int refactor_every = 64;
        int compress_every = 32;
        double pivot_rel = 1e-12;
        double abs_floor = 1e-16;
        double alpha_tol = 1e-10;
        double z_inf_guard = 1e6;
        bool sparse_amd = true;
        double sparse_drop_tol = 0.0;

        enum class UpdateMode { EtaStack, ForrestTomlin, Hybrid };
        UpdateMode update_mode = UpdateMode::ForrestTomlin;

        int ft_bandwidth_cap = 16;
        double max_growth_tol = 1e4;
        double min_dynamic_growth_tol = 1e3;
        double min_refactor_interval_fraction = 0.35;
        double max_condition_estimate = 1e13;
        double ft_multiplier_guard = 1e8;
        int rook_iters = 2;

        // v2/v3 safeguards
        bool enable_iterative_refinement = true;
        int refinement_steps = 2;
        double residual_refactor_tol = 1e-9;
        double residual_abs_refactor_tol = 1e-10;
        int refinement_max_steps = 6;
        double refinement_slow_progress_ratio = 0.5;
        double refinement_stall_progress_ratio = 0.1;
        int refinement_stall_limit = 2;
        int max_eta_count = 128;
        bool refactor_on_solve_failure = true;
        bool aggressive_refactor_on_suspicious_residual = true;

        // v3 update-chain health
        double eta_max_inf_norm = 1e7;
        double eta_avg_density_guard = 0.35;
        double eta_cumulative_inf_norm_guard = 1e8;
        double column_residual_tol = 1e-8;
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

    FTBasis(const DenseMat& A, const std::vector<int>& basis) : FTBasis(A, basis, Options{}) {}

    FTBasis(const DenseMat& A, const std::vector<int>& basis, const Options& opt)
        : A_dense_(&A), A_sparse_(nullptr), A_is_sparse_(false), m_(static_cast<int>(A.rows())),
          basis_(basis), opt_(opt) {
        if (static_cast<int>(basis_.size()) != m_)
            throw std::invalid_argument("FTBasis: basis size must equal m");

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

        Bcols_sparse_.resize(m_);
        for (int i = 0; i < m_; ++i)
            Bcols_sparse_[i] = A.col(basis_[i]);
        rebuild_sparse_basis_cache_();

        sparse_refactor_();
    }

    int rows() const noexcept { return m_; }
    const std::vector<int>& basis() const noexcept { return basis_; }
    const std::vector<Eta>& etas() const noexcept { return etas_; }
    int update_count() const noexcept { return update_count_; }
    Stats stats() const noexcept { return stats_; }
    const std::string& last_update_diagnostic() const noexcept { return last_update_diagnostic_; }

    Eigen::VectorXd solve_B(const Eigen::VectorXd& b) const {
        if (b.size() != m_)
            throw std::invalid_argument("FTBasis::solve_B size mismatch");

        try {
            Eigen::VectorXd x = base_solve_B_(b);
            if (!A_is_sparse_ && !etas_.empty()) {
                x = apply_etas_solve_(x);
            }
            if (opt_.enable_iterative_refinement && !A_is_sparse_)
                x = refine_solve_B_(b, x);
            return x;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            auto* self = const_cast<FTBasis*>(this);
            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd x = self->base_solve_B_(b);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                x = self->refine_solve_B_(b, x);
            return x;
        }
    }

    Eigen::VectorXd solve_BT(const Eigen::VectorXd& c) const {
        if (c.size() != m_)
            throw std::invalid_argument("FTBasis::solve_BT size mismatch");

        try {
            Eigen::VectorXd y = base_solve_BT_(c);
            if (!A_is_sparse_ && !etas_.empty()) {
                y = apply_etas_solve_T_(y);
            }
            if (opt_.enable_iterative_refinement && !A_is_sparse_)
                y = refine_solve_BT_(c, y);
            return y;
        } catch (const std::exception& err) {
            if (!opt_.refactor_on_solve_failure)
                throw;

            auto* self = const_cast<FTBasis*>(this);
            self->last_update_diagnostic_ = err.what();
            self->refactor();

            Eigen::VectorXd y = self->base_solve_BT_(c);
            if (self->opt_.enable_iterative_refinement && !self->A_is_sparse_)
                y = self->refine_solve_BT_(c, y);
            return y;
        }
    }

    void replace_column(int j, const Eigen::VectorXd& new_col_dense) {
        replace_column_impl_(j, std::nullopt, new_col_dense);
    }

    void replace_column(int j, int entering_col, const Eigen::VectorXd& new_col_dense) {
        replace_column_impl_(j, entering_col, new_col_dense);
    }

    template <typename Derived>
    void replace_column(int j, const Eigen::SparseMatrixBase<Derived>& new_col_sparse) {
        Eigen::SparseMatrix<double> tmp = new_col_sparse.derived().eval();
        Eigen::VectorXd dense(m_);
        dense.setZero();
        for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, 0); it; ++it)
            dense[it.row()] = it.value();
        replace_column_impl_(j, std::nullopt, dense);
    }

    template <typename Derived>
    void replace_column(int j, int entering_col,
                        const Eigen::SparseMatrixBase<Derived>& new_col_sparse) {
        Eigen::SparseMatrix<double> tmp = new_col_sparse.derived().eval();
        Eigen::VectorXd dense(m_);
        dense.setZero();
        for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, 0); it; ++it)
            dense[it.row()] = it.value();
        replace_column_impl_(j, entering_col, dense);
    }

    void refactor() {
        if (A_is_sparse_)
            sparse_refactor_();
        else
            dense_refactor_();
    }

    Eigen::MatrixXd explicit_B_dense() const {
        return A_is_sparse_ ? Eigen::MatrixXd(current_B_sparse_) : current_B_dense_;
    }

  private:
    static std::vector<int> permutation_to_vector_(const Permutation& perm) {
        std::vector<int> out(static_cast<size_t>(perm.indices().size()));
        for (int i = 0; i < perm.indices().size(); ++i)
            out[static_cast<size_t>(i)] = perm.indices()(i);
        return out;
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
            std::sort(cols.begin(), cols.end());
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

    void compute_sparse_ordering_(const SparseMat& B) {
        if (!opt_.sparse_amd || m_ <= 1 || sparse_ordering_cached_)
            return;

        const CSR csr = sparse_to_amd_csr_(B);
        AMDReorderingArray amd(/*aggressive_absorption=*/true,
                               /*dense_cutoff=*/-1);
        auto [perm, stats] = amd.compute_fill_reducing_permutation(csr, /*symmetrize=*/true);
        (void)stats;

        if (!is_valid_permutation_(perm, m_))
            return;

        sparse_row_perm_.assign(perm.begin(), perm.end());
        sparse_col_perm_ = sparse_row_perm_;
        sparse_ordering_cached_ = true;
    }

    // ----------------------------
    // Refactors
    // ----------------------------
    void reset_update_state_() {
        etas_.clear();
        update_count_ = 0;
        stats_ = Stats{};
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

    std::string make_update_diagnostic_(const std::string& reason, double alpha,
                                        const Eigen::VectorXd* z = nullptr,
                                        const Eigen::VectorXd* w = nullptr,
                                        double growth = std::numeric_limits<double>::quiet_NaN(),
                                        double residual =
                                            std::numeric_limits<double>::quiet_NaN()) const {
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
        score = std::max(score, ratio(stats_.cumulative_eta_z_inf, opt_.eta_cumulative_inf_norm_guard));
        score = std::max(score, ratio(stats_.estimated_condition, opt_.max_condition_estimate));
        return score;
    }

    double dynamic_growth_tol_() const noexcept {
        const double chain_progress =
            (opt_.refactor_every > 0)
                ? std::clamp(static_cast<double>(update_count_) / static_cast<double>(opt_.refactor_every),
                             0.0, 1.0)
                : 1.0;
        const double eta_pressure =
            (opt_.eta_cumulative_inf_norm_guard > 0.0)
                ? std::clamp(stats_.cumulative_eta_z_inf / opt_.eta_cumulative_inf_norm_guard, 0.0, 1.0)
                : 0.0;
        const double pressure = std::max(chain_progress, eta_pressure);
        const double tightened = opt_.max_growth_tol * (1.0 - 0.75 * pressure);
        return std::max(opt_.min_dynamic_growth_tol, tightened);
    }

    int adaptive_refactor_limit_() const noexcept {
        const int base = std::max(1, opt_.refactor_every);
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
        config.iterative_refinement = opt_.enable_iterative_refinement;
        config.iterative_refinement_steps = std::max(1, opt_.refinement_steps);
        config.iterative_refinement_tol = opt_.residual_refactor_tol;
        config.max_norm_growth_before_refactor = std::max(1e4, opt_.max_growth_tol * 100.0);
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
            last_update_diagnostic_ = "Dense refactor produced a poorly conditioned basis est_cond=" +
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
    }

    void update_sparse_basis_cache_column_(int col_j, const SparseMat& col) {
        if (current_B_sparse_.rows() != m_ || current_B_sparse_.cols() != m_) {
            rebuild_sparse_basis_cache_();
            return;
        }

        current_B_sparse_.col(col_j) = col;
        if (opt_.sparse_drop_tol > 0.0)
            current_B_sparse_.prune(opt_.sparse_drop_tol);
        current_B_sparse_.makeCompressed();
    }

    void dense_refactor_() {
        lu_dense_.factor(current_B_dense_, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters);
        max_element_ = lu_dense_.U().cwiseAbs().maxCoeff();
        refactor_baseline_max_element_ = std::max(1.0, max_element_);
        if (initial_refactor_max_element_ == 0.0)
            initial_refactor_max_element_ = refactor_baseline_max_element_;
        reset_update_state_();
        refresh_refactor_diagnostics_();
    }

    void sparse_build_B_(SparseMat& B) const { B = current_B_sparse_; }

    void sparse_refactor_() {
        SparseMat B;
        sparse_build_B_(B);
        compute_sparse_ordering_(B);
        const auto config = make_sparse_lu_config_();

        if (sparse_ordering_cached_) {
            Permutation row_perm(m_);
            Permutation col_perm(m_);
            for (int i = 0; i < m_; ++i) {
                row_perm.indices()(i) = sparse_row_perm_[static_cast<size_t>(i)];
                col_perm.indices()(i) = sparse_col_perm_[static_cast<size_t>(i)];
            }

            const SparseMat B_perm = row_perm * B * col_perm;

            lu_sparse_.factor(B_perm, opt_.pivot_rel, opt_.abs_floor, std::min(opt_.rook_iters, 1),
                              opt_.ft_bandwidth_cap, &sparse_row_perm_, &sparse_col_perm_, config);
        } else {
            lu_sparse_.factor(B, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters,
                              opt_.ft_bandwidth_cap, nullptr, nullptr, config);
        }
        reset_update_state_();
        refresh_refactor_diagnostics_();
    }

    // ----------------------------
    // Base solves
    // ----------------------------
    Eigen::VectorXd base_solve_B_(const Eigen::VectorXd& b) const {
        return A_is_sparse_ ? lu_sparse_.solve(b) : lu_dense_.solve(b);
    }

    Eigen::VectorXd base_solve_BT_(const Eigen::VectorXd& c) const {
        return A_is_sparse_ ? lu_sparse_.solveT(c) : lu_dense_.solveT(c);
    }

    // ----------------------------
    // Explicit multiplications
    // ----------------------------
    Eigen::VectorXd multiply_B_(const Eigen::VectorXd& x) const {
        if (x.size() != m_)
            throw std::invalid_argument("FTBasis::multiply_B_ size mismatch");

        if (A_is_sparse_)
            return current_B_sparse_ * x;
        return current_B_dense_ * x;
    }

    Eigen::VectorXd multiply_BT_(const Eigen::VectorXd& y) const {
        if (y.size() != m_)
            throw std::invalid_argument("FTBasis::multiply_BT_ size mismatch");

        if (A_is_sparse_)
            return current_B_sparse_.transpose() * y;
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
            const Eigen::VectorXd r = rhs - multiply_B_(x);
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
                    throw std::runtime_error("FTBasis::refine_solve_B stalled berr=" +
                                             format_metric_(berr) +
                                             " abs_residual=" + format_metric_(final_abs_residual));
                }
            }

            Eigen::VectorXd dx = base_solve_B_(r);
            if (!A_is_sparse_ && !etas_.empty()) {
                dx = apply_etas_solve_(dx);
            }

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
            throw std::runtime_error("FTBasis::refine_solve_B residual remained large after refinement"
                                     " berr=" + format_metric_(final_berr) +
                                     " abs_residual=" + format_metric_(final_abs_residual));
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
                    throw std::runtime_error("FTBasis::refine_solve_BT stalled berr=" +
                                             format_metric_(berr) +
                                             " abs_residual=" + format_metric_(final_abs_residual));
                }
            }

            Eigen::VectorXd dy = base_solve_BT_(r);
            if (!A_is_sparse_ && !etas_.empty()) {
                dy = apply_etas_solve_T_(dy);
            }

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
                format_metric_(final_berr) +
                " abs_residual=" + format_metric_(final_abs_residual));
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
            std::max(opt_.abs_floor,
                     1e-14 * std::max({1.0, U.col(j).lpNorm<Eigen::Infinity>(),
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
        Eigen::VectorXd dense(col.rows());
        dense.setZero();
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
        update_sparse_basis_cache_column_(col_j, Bcols_sparse_[col_j]);
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
        stats_.max_eta_z_inf = 0.0;
        stats_.max_eta_w_inf = 0.0;
        stats_.avg_eta_density = 0.0;
        stats_.cumulative_eta_z_inf = 0.0;

        double density_sum = 0.0;
        for (const auto& e : etas_) {
            const double zinf = e.z_inf_norm();
            const double winf = e.w_inf_norm();
            stats_.max_eta_z_inf = std::max(stats_.max_eta_z_inf, zinf);
            stats_.max_eta_w_inf = std::max(stats_.max_eta_w_inf, winf);
            stats_.cumulative_eta_z_inf += zinf;
            density_sum += e.z_density();
        }
        stats_.avg_eta_density =
            etas_.empty() ? 0.0 : density_sum / static_cast<double>(etas_.size());
        stats_.growth_from_last_refactor = max_element_ / std::max(1.0, refactor_baseline_max_element_);
        stats_.growth_from_initial_refactor = max_element_ / std::max(1.0, initial_refactor_max_element_);
        stats_.growth_factor = stats_.growth_from_last_refactor;
        stats_.estimated_condition = dense_condition_estimate_();
        stats_.sparse_norm_growth_estimate = 1.0;
        stats_.stability_score = stability_score_();
    }

    double column_residual_(int j) const {
        Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
        ej(j) = 1.0;

        const Eigen::VectorXd Bj = multiply_B_(ej);
        const Eigen::VectorXd target =
            A_is_sparse_ ? Eigen::VectorXd(current_B_sparse_.col(j)) : current_B_dense_.col(j);

        const Eigen::VectorXd r = target - Bj;
        const double denom = std::max(1.0, target.lpNorm<Eigen::Infinity>());
        return r.lpNorm<Eigen::Infinity>() / denom;
    }

    bool bad_column_residual_(int j) {
        stats_.last_column_residual = column_residual_(j);
        return (!std::isfinite(stats_.last_column_residual)) ||
               (stats_.last_column_residual > opt_.column_residual_tol);
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
        if (stats_.max_eta_z_inf > opt_.eta_max_inf_norm)
            return true;
        if (stats_.cumulative_eta_z_inf > opt_.eta_cumulative_inf_norm_guard)
            return true;
        if (stats_.avg_eta_density > opt_.eta_avg_density_guard)
            return true;
        if (stats_.max_eta_z_inf > opt_.z_inf_guard)
            return true;
        if (stats_.growth_from_last_refactor > dynamic_growth_tol_())
            return true;
        if (stats_.growth_from_initial_refactor > dynamic_growth_tol_() * std::max(10.0, opt_.max_growth_tol))
            return true;
        if (stats_.estimated_condition > opt_.max_condition_estimate)
            return true;
        return false;
    }

    // ----------------------------
    // Replace column
    // ----------------------------
    void replace_column_impl_(int j, std::optional<int> entering_col,
                              const Eigen::VectorXd& new_col_dense) {
        if (j < 0 || j >= m_)
            throw std::out_of_range("FTBasis::replace_column bad j");
        if (new_col_dense.size() != m_)
            throw std::invalid_argument("FTBasis::replace_column size mismatch");

        last_update_diagnostic_.clear();

        if (entering_col.has_value())
            basis_[j] = *entering_col;

        if (A_is_sparse_) {
            const Eigen::VectorXd old_dense = Eigen::VectorXd(current_B_sparse_.col(j));
            if (lu_sparse_.needs_refactor()) {
                last_update_diagnostic_ = "Sparse FT update chain exceeded norm-growth guard";
                set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }
            Eigen::VectorXd u = new_col_dense - old_dense;
            const double alpha_floor = adaptive_alpha_tol_(old_dense.lpNorm<Eigen::Infinity>());

            Eigen::VectorXd z, w;
            double alpha = 0.0;

            try {
                z = solve_B(u);
                Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
                ej(j) = 1.0;
                w = solve_BT(ej);
                alpha = 1.0 + z(j);
            } catch (...) {
                set_sparse_column_(j, new_col_dense);
                sparse_refactor_();
                return;
            }

            set_sparse_column_(j, new_col_dense);

            const bool unstable = (!std::isfinite(alpha)) || (std::abs(alpha) < alpha_floor) ||
                                  (!z.array().isFinite().all()) || (!w.array().isFinite().all()) ||
                                  (z.lpNorm<Eigen::Infinity>() > opt_.z_inf_guard);

            if (unstable) {
                last_update_diagnostic_ = make_update_diagnostic_(
                    "Sparse FT update rejected by adaptive alpha or finite guards", alpha, &z, &w);
                sparse_refactor_();
                return;
            }

            if (!lu_sparse_.append_forrest_tomlin_update(j, u, z, w, alpha,
                                                         std::max(opt_.abs_floor, 1e-14))) {
                last_update_diagnostic_ =
                    make_update_diagnostic_(lu_sparse_.last_update_failure_reason_message(), alpha, &z, &w);
                sparse_refactor_();
                return;
            }
            ++update_count_;
            refresh_stats_();

            if (bad_column_residual_(j) || need_compress_() ||
                update_count_ >= opt_.refactor_every) {
                sparse_refactor_();
            }
            return;
        }

        const Eigen::VectorXd old = current_B_dense_.col(j);
        Eigen::VectorXd u = new_col_dense - old;
        const double alpha_floor = adaptive_alpha_tol_(std::abs(lu_dense_.U()(j, j)));

        Eigen::VectorXd z, w;
        double alpha = 0.0;

        try {
            z = solve_B(u);
            Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
            ej(j) = 1.0;
            w = solve_BT(ej);
            alpha = 1.0 + z(j);
        } catch (...) {
            set_dense_basis_column_(j, new_col_dense);
            dense_refactor_();
            return;
        }

        const bool allow_ft_mode =
            opt_.update_mode == Options::UpdateMode::ForrestTomlin ||
            opt_.update_mode == Options::UpdateMode::Hybrid;
        const bool hybrid_prefers_ft =
            opt_.update_mode != Options::UpdateMode::Hybrid ||
            (stats_.estimated_condition <= opt_.max_condition_estimate &&
             stats_.stability_score < 0.75);
        const bool try_ft = lu_dense_.supports_inplace_updates() && allow_ft_mode && hybrid_prefers_ft &&
                            std::abs(alpha) >= alpha_floor &&
                            update_count_ < adaptive_refactor_limit_() &&
                            z.cwiseAbs().maxCoeff() <= opt_.z_inf_guard &&
                            w.array().isFinite().all();

        if (try_ft) {
            bool ok = true;
            try {
                forrest_tomlin_update_dense_(j, new_col_dense, z, alpha);
                const double current_max = lu_dense_.U().cwiseAbs().maxCoeff();
                const double dynamic_growth_tol = dynamic_growth_tol_();
                const double baseline_growth = current_max / std::max(1.0, refactor_baseline_max_element_);
                const double initial_growth = current_max / std::max(1.0, initial_refactor_max_element_);
                const double cond_estimate = dense_condition_estimate_();
                if (current_max > std::max(1.0, max_element_) * dynamic_growth_tol ||
                    baseline_growth > dynamic_growth_tol ||
                    initial_growth > dynamic_growth_tol * std::max(dynamic_growth_tol, 10.0) ||
                    cond_estimate > opt_.max_condition_estimate) {
                    last_update_diagnostic_ = make_update_diagnostic_(
                        "Dense FT update exceeded stability guards", alpha, &z, &w,
                        baseline_growth, stats_.last_column_residual);
                    ok = false;
                } else {
                    max_element_ = std::max(max_element_, current_max);
                }
            } catch (const std::exception& err) {
                last_update_diagnostic_ = err.what();
                ok = false;
            }

            set_dense_basis_column_(j, new_col_dense);
            const bool bad_residual = bad_column_residual_(j);

            if (!ok || (opt_.aggressive_refactor_on_suspicious_residual && bad_residual)) {
                if (!ok && last_update_diagnostic_.empty())
                    last_update_diagnostic_ =
                        make_update_diagnostic_("Dense FT update failed numerical stability checks",
                                                alpha, &z, &w, stats_.growth_from_last_refactor,
                                                stats_.last_column_residual);
                dense_refactor_();
                return;
            }

            ++update_count_;
            refresh_stats_();
            if (need_compress_())
                dense_refactor_();
            return;
        }

        const bool refactor_now = (std::abs(alpha) < alpha_floor) ||
                                  (update_count_ >= adaptive_refactor_limit_()) ||
                                  (!std::isfinite(alpha)) ||
                                  (stats_.estimated_condition > opt_.max_condition_estimate);

        if (refactor_now) {
            last_update_diagnostic_ =
                make_update_diagnostic_("Dense update fell back to refactor due to stability guards",
                                        alpha, &z, &w);
            set_dense_basis_column_(j, new_col_dense);
            dense_refactor_();
            return;
        }

        set_dense_basis_column_(j, new_col_dense);
        etas_.push_back(Eta{j, std::move(u), std::move(z), std::move(w), alpha});
        ++update_count_;
        refresh_stats_();
        const bool bad_residual = bad_column_residual_(j);

        if ((opt_.aggressive_refactor_on_suspicious_residual && bad_residual) || need_compress_())
            dense_refactor_();
    }

  private:
    const DenseMat* A_dense_{nullptr};
    const SparseMat* A_sparse_{nullptr};
    bool A_is_sparse_{false};

    std::vector<Eigen::VectorXd> Bcols_dense_;
    std::vector<SparseMat> Bcols_sparse_;
    DenseMat current_B_dense_;
    SparseMat current_B_sparse_;
    bool sparse_ordering_cached_{false};
    std::vector<int> sparse_row_perm_, sparse_col_perm_;

    MarkowitzLU lu_dense_;
    SparseForrestTomlinLU lu_sparse_;

    int m_{0};
    std::vector<int> basis_;
    Options opt_;
    std::vector<Eta> etas_;
    int update_count_{0};
    double max_element_{0.0};
    double refactor_baseline_max_element_{0.0};
    double initial_refactor_max_element_{0.0};
    Stats stats_{};
    std::string last_update_diagnostic_;
};
