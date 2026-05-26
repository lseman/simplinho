#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "presolve_types.h"

namespace presolve {


class Presolver {
  public:
    struct Options {
        // numerics
        double svd_tol = 1e-8;
        double zero_tol = 1e-12;
        double infeas_tol = 1e-9;
        double rrqr_pivot_tol = 1e-8;
        double rr_infeas_mult = 1e3;
        double cond_max = 1e10;
        double dual_fix_tol = 1e-10;

        // passes
        int max_passes = 3;             // Reduced from 5
        bool enable_rowreduce = false;  // DISABLED by default for speed (turn ON for max reduction)
        bool enable_scaling = true;     // row scaling only
        bool enable_col_scaling = true; // OFF in non-destructive mode
        bool enable_dual_fixing = true;

        // behavior
        bool non_destructive = true;           // DEFAULT: keep ranking stable for BNB
        bool allow_structural_changes = false; // DEFAULT: OFF for speed (turn ON for max reduction)

        // row-reduce
        int max_ruiz_iters = 10;
        RowReduceMethod row_reduce_method = RowReduceMethod::Auto;
        bool use_iter_refine = true;

        // conservative extras
        bool classic_row_reduce = true;
        bool conservative_mode = false;
        bool enable_huge_bound_relaxation = false; // DISABLED by default for speed
        double huge_bound_factor = 1e6;
        double huge_bound_relax_gap = 1e6;

        // light objective-guided probing
        bool enable_objective_probing = false; // DISABLED by default for speed
        int probing_max_vars = 3;              // Reduce from 6
        int probing_max_rounds = 1;            // Reduce from 2
        double probing_obj_tol = 1e-6;         // Relax tolerance

        // doubleton equation elimination: x_elim = (b - a_keep*x_keep) / a_elim
        // Only active when allow_structural_changes && !non_destructive
        bool enable_doubleton_elim = false;

        // HiGHS-style reduced-cost lurking bound tightening:
        // For a continuous column with one-sided reduced cost implied by its
        // lock structure, we derive implied dual row bounds and use them to
        // tighten primal bounds on other columns in those rows.
        bool enable_reduced_cost_lurking = true;

        // Row aggregation through continuous free variables:
        // Pair rows containing a common continuous free variable and sum them
        // with appropriate scaling to eliminate the variable, deriving a new
        // aggregate constraint on the remaining variables.  New bounds
        // propagated without filling in the matrix.
        bool enable_row_aggregation = false; // off by default (may increase constraint count)
        int  row_agg_max_pairs = 20;         // max (row_i, row_k) pairs per pass
    };

    Presolver() : opt_() { domprop_min_delta_ = 1e3 * opt_.infeas_tol; }
    explicit Presolver(const Options& opt) : opt_(opt) {
        domprop_min_delta_ = 1e3 * opt_.infeas_tol;
        if (opt_.non_destructive) {
            opt_.enable_col_scaling = false;
        }
    }

    using Result = PresolveResult;

    PresolveResult run(const LP& in) {
        res_.stack.clear();
        res_.obj_shift = 0.0;
        res_.proven_infeasible = false;
        res_.proven_unbounded = false;
        res_.implied_bound_updates = 0;
        res_.relaxed_huge_lower_bounds = 0;
        res_.relaxed_huge_upper_bounds = 0;

        LP P = in;
        const int m0 = (int)P.A.rows();
        const int n0 = (int)P.A.cols();
        res_.original_num_rows = m0;
        res_.original_num_cols = n0;
        res_.orig_row_index.resize(m0);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        res_.orig_col_index.resize(n0);
        std::iota(res_.orig_col_index.begin(), res_.orig_col_index.end(), 0);

        // sanity
        if ((int)P.sense.size() != (int)P.b.size())
            throw std::invalid_argument("presolve: sense size mismatch with b");
        if ((int)P.l.size() != n0 || (int)P.u.size() != n0 || (int)P.c.size() != n0)
            throw std::invalid_argument("presolve: vector sizes must equal n");

        if (opt_.enable_huge_bound_relaxation) {
            const BoundRelaxationSummary relaxed = canonicalize_inactive_huge_bounds(
                &P, opt_.zero_tol, opt_.huge_bound_factor, opt_.huge_bound_relax_gap);
            res_.relaxed_huge_lower_bounds = relaxed.relaxed_lower;
            res_.relaxed_huge_upper_bounds = relaxed.relaxed_upper;
        }

        if (!check_and_fix_bounds(P)) {
            res_.reduced = std::move(P);
            res_.proven_infeasible = true;
            return res_;
        }
        if (detect_unboundedness(P)) {
            res_.reduced = std::move(P);
            res_.proven_unbounded = true;
            return res_;
        }

        if (opt_.enable_scaling)
            scale_rows_unit_inf(P);
        if (opt_.enable_col_scaling && !opt_.non_destructive)
            scale_cols_unit_inf(P);

        if (opt_.enable_rowreduce) {
            if (!row_reduce(P)) {
                res_.reduced = std::move(P);
                res_.proven_infeasible = true;
                return res_;
            }
        }

        // Build sparse index once; maintained incrementally throughout the pass loop
        build_sparse_index(P);

        int pass = 0;
        bool changed = true;
        while (changed && pass < opt_.max_passes) {
            changed = false;

            if (detect_unboundedness(P)) {
                res_.reduced = std::move(P);
                res_.proven_unbounded = true;
                return res_;
            }

            // Zero-rows only (safe)
            changed |= remove_free_zero_rows(P);
            if (res_.proven_infeasible)
                break;

            // Zero-columns with no constraints can often be fixed directly.
            changed |= remove_free_zero_columns(P);
            if (res_.proven_unbounded || res_.proven_infeasible)
                break;

            // Fixed variable handling:
            //  - non_destructive => "fix-and-zero": keep column, zero A(:,j),
            //  keep c_j, set l=u=x*
            //  - structural      => erase column and shift
            changed |= fixed_variable_detection(P);

            // Tighten singleton rows first, then general row-based bound tightening.
            changed |= singleton_row_elimination(P);
            if (res_.proven_infeasible)
                break;

            // Tighten bounds by row activities (no c changes)
            changed |= tighten_bounds_by_rows(P);
            if (res_.proven_infeasible)
                break;

            // Multi-round guarded domain propagation with dirty row/column
            // tracking.
            changed |= domain_propagation_rounds(P);
            if (res_.proven_infeasible)
                break;

            // Cheap reduced-cost-style fixing from objective sign and row locks.
            changed |= dual_fix_by_locks(P);
            if (res_.proven_infeasible)
                break;

            // HiGHS-style reduced-cost lurking bounds: implied primal bounds
            // from dual bounds derived via reduced-cost certificates.
            changed |= reduced_cost_lurking_bounds(P);
            if (res_.proven_infeasible)
                break;

            // Row aggregation through continuous free variables.
            changed |= row_aggregation(P);
            if (res_.proven_infeasible)
                break;

            // Singleton column substitution / implied free detection.
            changed |= singleton_column_substitution(P);
            if (res_.proven_infeasible)
                break;

            // Budgeted probing on high-impact objective variables.
            changed |= objective_guided_probing(P);
            if (res_.proven_infeasible)
                break;

            // Exact duplicate row removal (safe)
            changed |= redundancy_duplicate_rows(P);

            // Doubleton equation elimination (structural, opt-in)
            changed |= doubleton_equation_elimination(P);
            if (res_.proven_infeasible)
                break;

            // No structural or objective-changing passes unless explicitly
            // enabled
            ++pass;
        }

        if (detect_unboundedness(P)) {
            res_.reduced = std::move(P);
            res_.proven_unbounded = true;
            return res_;
        }

        prune_zero_rows(P);

        res_.reduced = std::move(P);
        return res_;
    }

    std::pair<Eigen::VectorXd, double> postsolve(const Eigen::VectorXd& x_red) const {
        const int n_full_guess = std::max(res_.original_num_cols, (int)res_.orig_col_index.size());
        Eigen::VectorXd x_full =
            Eigen::VectorXd::Constant(n_full_guess, std::numeric_limits<double>::quiet_NaN());
        for (int jr = 0; jr < (int)x_red.size(); ++jr) {
            int jorig = res_.orig_col_index[jr];
            if (jorig >= 0 && jorig < n_full_guess)
                x_full(jorig) = x_red(jr);
        }
        double obj_correction = res_.obj_shift;

        for (int k = (int)res_.stack.size() - 1; k >= 0; --k) {
            const auto& act = res_.stack[k];
            std::visit([&](auto const& a) { undo_action(a, x_full, obj_correction); }, act);
        }

        return {x_full, obj_correction};
    }

    Eigen::VectorXd postsolve_dual(const Eigen::VectorXd& y_red) const {
        Eigen::VectorXd y_full = y_red;
        for (int k = (int)res_.stack.size() - 1; k >= 0; --k) {
            const auto& act = res_.stack[k];
            std::visit([&](auto const& a) { undo_dual_action(a, y_full); }, act);
        }
        return y_full;
    }

    const PresolveResult& result() const noexcept { return res_; }

  private:
    // ---------- sparse index (maintained during run()) ----------
    // row_nz_[i] = sorted column indices with A(i,j) != 0
    // col_nz_[j] = sorted row indices with A(i,j) != 0
    std::vector<std::vector<int>> row_nz_;
    std::vector<std::vector<int>> col_nz_;

    // ---------- sparse index management ----------
    void build_sparse_index(const LP& P) {
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        row_nz_.assign(m, {});
        col_nz_.assign(n, {});
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (std::abs(P.A(i, j)) > opt_.zero_tol) {
                    row_nz_[i].push_back(j);
                    col_nz_[j].push_back(i);
                }
            }
        }
    }

    // Update sparse index when row i is about to be physically erased
    // Must be called BEFORE the actual erase so row_nz_[i] is still valid
    void update_index_erase_row(int i) {
        // Remove row i from each col's list
        for (int j : row_nz_[i]) {
            auto& cv = col_nz_[j];
            cv.erase(std::remove(cv.begin(), cv.end(), i), cv.end());
        }
        // Remove row_nz_[i]
        row_nz_.erase(row_nz_.begin() + i);
        // Decrement all row indices > i in all col_nz_ entries
        for (auto& cv : col_nz_)
            for (int& r : cv)
                if (r > i)
                    --r;
    }

    // Update sparse index when col j is about to be physically erased
    // Must be called BEFORE the actual erase so col_nz_[j] is still valid
    void update_index_erase_col(int j) {
        // Remove col j from each row's list
        for (int i : col_nz_[j]) {
            auto& rv = row_nz_[i];
            rv.erase(std::remove(rv.begin(), rv.end(), j), rv.end());
        }
        // Remove col_nz_[j]
        col_nz_.erase(col_nz_.begin() + j);
        // Decrement all col indices > j in all row_nz_ entries
        for (auto& rv : row_nz_)
            for (int& c : rv)
                if (c > j)
                    --c;
    }

    // Remove all entries for column j (when A(:,j) is zeroed out)
    void zero_col_in_index(int j) {
        if (j >= (int)col_nz_.size())
            return;
        for (int i : col_nz_[j]) {
            auto& rv = row_nz_[i];
            rv.erase(std::remove(rv.begin(), rv.end(), j), rv.end());
        }
        col_nz_[j].clear();
    }

    // Remove a single entry (i,j) from the index (when A(i,j) is set to zero)
    void zero_entry_in_index(int i, int j) {
        if (i < (int)row_nz_.size()) {
            auto& rv = row_nz_[i];
            rv.erase(std::remove(rv.begin(), rv.end(), j), rv.end());
        }
        if (j < (int)col_nz_.size()) {
            auto& cv = col_nz_[j];
            cv.erase(std::remove(cv.begin(), cv.end(), i), cv.end());
        }
    }

    // Add entry (i,j) to the index (when a new nonzero appears at A(i,j))
    void add_entry_to_index(int i, int j) {
        if (i >= (int)row_nz_.size() || j >= (int)col_nz_.size())
            return;
        auto& rv = row_nz_[i];
        auto it = std::lower_bound(rv.begin(), rv.end(), j);
        if (it == rv.end() || *it != j)
            rv.insert(it, j);
        auto& cv = col_nz_[j];
        auto it2 = std::lower_bound(cv.begin(), cv.end(), i);
        if (it2 == cv.end() || *it2 != i)
            cv.insert(it2, i);
    }

    // ---------- numerics helpers ----------
    static double cond2_estimate_upper(const Eigen::MatrixXd& R11) {
        if (R11.size() == 0)
            return 0.0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(R11, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto s = svd.singularValues();
        const double smax = s(0);
        const double smin = s.tail(1)(0);
        return (smin > 0.0) ? (smax / smin) : std::numeric_limits<double>::infinity();
    }

    static std::pair<Eigen::VectorXd, Eigen::VectorXd> ruiz_balance(Eigen::MatrixXd& A, int iters,
                                                                    double floor = 1e-12) {
        const int m = (int)A.rows(), n = (int)A.cols();
        Eigen::VectorXd Dr = Eigen::VectorXd::Ones(m), Dc = Eigen::VectorXd::Ones(n);
        for (int k = 0; k < iters; ++k) {
            for (int i = 0; i < m; ++i) {
                double s = std::sqrt(A.row(i).cwiseAbs().mean());
                if (!std::isfinite(s) || s < floor)
                    s = 1.0;
                A.row(i) /= s;
                Dr(i) *= s;
            }
            for (int j = 0; j < n; ++j) {
                double s = std::sqrt(A.col(j).cwiseAbs().mean());
                if (!std::isfinite(s) || s < floor)
                    s = 1.0;
                A.col(j) /= s;
                Dc(j) *= s;
            }
        }
        return {Dr, Dc};
    }
    bool detect_unboundedness(const LP& P) const {
        const int n = (int)P.A.cols(), m = (int)P.A.rows();
        for (int j = 0; j < n; ++j) {
            const bool can_inc = !is_finite(P.u(j));
            const bool can_dec = !is_finite(P.l(j));
            if (!can_inc && !can_dec)
                continue;

            if (m == 0) {
                if (P.c(j) < -opt_.zero_tol && can_inc)
                    return true;
                if (P.c(j) > opt_.zero_tol && can_dec)
                    return true;
                continue;
            }

            // guard: if column is numerically zero, don’t use it to declare
            // unbounded
            if (safe_abs_max(P.A.col(j)) <= opt_.zero_tol) {
                if ((can_inc || can_dec) && std::abs(P.c(j)) > opt_.zero_tol) {
                    // only suspicious if literally free and improving, but we
                    // still skip declaring here
                    continue; // let the solver (not presolve) decide
                }
                continue;
            }

            bool blocks_plus = false, blocks_minus = false;
            for (int i = 0; i < m; ++i) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    continue;
                if (P.sense[i] == RowSense::EQ) {
                    blocks_plus = blocks_minus = true;
                    break;
                }
                if (P.sense[i] == RowSense::LE) {
                    if (aij > 0)
                        blocks_plus = true;
                    if (aij < 0)
                        blocks_minus = true;
                } else if (P.sense[i] == RowSense::GE) {
                    if (aij < 0)
                        blocks_plus = true;
                    if (aij > 0)
                        blocks_minus = true;
                }
            }
            if (P.c(j) < -opt_.zero_tol && can_inc && !blocks_plus)
                return true;
            if (P.c(j) > opt_.zero_tol && can_dec && !blocks_minus)
                return true;
        }
        return false;
    }

    void scale_rows_unit_inf(LP& P) {
        const int m = (int)P.A.rows();
        for (int i = 0; i < m; ++i) {
            const double s = nearest_power_of_two_magnitude(safe_abs_max(P.A.row(i)));
            if (s > 0 && !nearly_zero(s, opt_.zero_tol) && std::isfinite(s)) {
                P.A.row(i) /= s;
                P.b(i) /= s;
                res_.stack.emplace_back(ActScaleRow{i, s});
            }
        }
    }
    void scale_cols_unit_inf(LP& P) {
        // disabled by default to preserve ranking
        if (opt_.non_destructive)
            return;
        const int n = (int)P.A.cols();
        for (int j = 0; j < n; ++j) {
            const double s = nearest_power_of_two_magnitude(safe_abs_max(P.A.col(j)));
            if (s > 0 && !nearly_zero(s, opt_.zero_tol) && std::isfinite(s)) {
                P.A.col(j) /= s;
                P.c(j) /= s;
                if (is_finite(P.l(j)))
                    P.l(j) *= s;
                if (is_finite(P.u(j)))
                    P.u(j) *= s;
                res_.stack.emplace_back(ActScaleCol{j, s});
            }
        }
    }

    // --- row reduction: choose RRQR by default, fallback to SVD ---
    bool row_reduce(LP& P) {
        if (!opt_.enable_rowreduce)
            return true;
        if (!all_rows_equal_(P))
            return true;
        switch (opt_.row_reduce_method) {
            case RowReduceMethod::RRQR:
                return row_reduce_rrqr(P);
            case RowReduceMethod::SVD:
                return row_reduce_svd(P);
            case RowReduceMethod::Auto:
            default:
                return row_reduce_rrqr(P);
        }
    }

    static bool all_rows_equal_(const LP& P) {
        for (RowSense s : P.sense)
            if (s != RowSense::EQ)
                return false;
        return true;
    }

    bool row_reduce_svd(LP& P) {
        using SVD = Eigen::BDCSVD<Eigen::MatrixXd>;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0)
            return true;

        SVD svd(P.A, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& S = svd.singularValues();
        const double eps = std::numeric_limits<double>::epsilon();
        const double smax = (S.size() > 0 ? S(0) : 0.0);
        const double thr = std::max(opt_.svd_tol * smax, 100.0 * eps * smax);

        int r = 0;
        for (int i = 0; i < S.size(); ++i)
            if (S(i) > thr)
                ++r;

        const double ainfn = P.A.cwiseAbs().rowwise().sum().maxCoeff();
        const bool matrix_not_tiny = (smax > 1e3 * eps * std::max(1.0, ainfn));
        if (r == 0) {
            if (!matrix_not_tiny) {
                if (P.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                    res_.stack.emplace_back(
                        ActRowReduce{Eigen::MatrixXd::Zero(m, 0), Eigen::VectorXi(), m});
                    P.A.resize(0, n);
                    P.b.resize(0);
                    P.sense.clear();
                    res_.orig_row_index.clear();
                    return true;
                } else {
                    res_.proven_infeasible = true;
                    return false;
                }
            } else
                return true;
        }

        const Eigen::MatrixXd Ur = svd.matrixU().leftCols(r);
        if (r < m) {
            const Eigen::VectorXd resid = P.b - Ur * (Ur.transpose() * P.b);
            const double res_inf = resid.lpNorm<Eigen::Infinity>();
            const double allowed = opt_.rr_infeas_mult * opt_.infeas_tol *
                                   std::max(1.0, P.b.lpNorm<Eigen::Infinity>());
            if (res_inf > allowed) {
                res_.proven_infeasible = true;
                return false;
            }
        }

        const Eigen::VectorXd Sr = S.head(r);
        const Eigen::MatrixXd Vr = svd.matrixV().leftCols(r);
        const Eigen::MatrixXd Atil = Sr.asDiagonal() * Vr.transpose();
        const Eigen::VectorXd btil = Ur.transpose() * P.b;

        Eigen::VectorXi keep(r);
        for (int i = 0; i < r; ++i)
            keep(i) = i;
        res_.stack.emplace_back(ActRowReduce{Ur, keep, m});
        P.A = Atil;
        P.b = btil;
        P.sense.assign(r, RowSense::EQ);
        res_.orig_row_index.resize(r);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        return true;
    }

    bool row_reduce_rrqr(LP& P) {
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0)
            return true;

        Eigen::MatrixXd A = P.A;
        Eigen::VectorXd b = P.b;
        if (opt_.max_ruiz_iters > 0) {
            auto [Dr, Dc] = ruiz_balance(A, opt_.max_ruiz_iters);
            for (int i = 0; i < m; ++i)
                b(i) /= Dr(i);
            (void)Dc;
        }

        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrA(A);
        if (qrA.info() != Eigen::Success)
            return true;

        const int kmax = std::min(m, n);
        Eigen::MatrixXd R =
            qrA.matrixR().topLeftCorner(kmax, kmax).template triangularView<Eigen::Upper>();
        Eigen::VectorXd diagR = R.diagonal().cwiseAbs();
        const double eps = std::numeric_limits<double>::epsilon();
        const double dmax = (diagR.size() ? diagR.maxCoeff() : 0.0);
        const double rthr =
            std::max({opt_.rrqr_pivot_tol, opt_.svd_tol * dmax, 100.0 * eps * dmax});

        int r = 0;
        for (; r < diagR.size(); ++r)
            if (diagR(r) < rthr)
                break;
        if (r == 0) {
            const double ainfn = P.A.cwiseAbs().rowwise().sum().maxCoeff();
            if (ainfn > 1e3 * eps)
                return true;
            if (P.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                res_.stack.emplace_back(
                    ActRowReduce{Eigen::MatrixXd::Zero(m, 0), Eigen::VectorXi(), m});
                P.A.resize(0, n);
                P.b.resize(0);
                P.sense.clear();
                res_.orig_row_index.clear();
                return true;
            } else {
                res_.proven_infeasible = true;
                return false;
            }
        }

        auto cond_ok = [&](int rr) -> bool {
            if (rr <= 0)
                return false;
            Eigen::MatrixXd R11 = R.topLeftCorner(rr, rr);
            const double kappa = cond2_estimate_upper(R11);
            return (kappa <= opt_.cond_max || !std::isfinite(kappa));
        };
        int r_try = r;
        while (r_try <= (int)diagR.size()) {
            if (!cond_ok(r_try))
                break;
            ++r_try;
        }
        r = std::max(1, std::min(r_try - 1, (int)diagR.size()));

        Eigen::HouseholderQR<Eigen::MatrixXd> qra_full(P.A);
        Eigen::MatrixXd Ur = Eigen::MatrixXd::Identity(m, r);
        Ur = qra_full.householderQ() * Ur;

        Eigen::VectorXd resid = P.b - Ur * (Ur.transpose() * P.b);
        const double allowed =
            opt_.rr_infeas_mult * opt_.infeas_tol * std::max(1.0, P.b.lpNorm<Eigen::Infinity>());
        if (resid.lpNorm<Eigen::Infinity>() > allowed)
            return row_reduce_svd(P);

        Eigen::MatrixXd Atil = Ur.transpose() * P.A;
        Eigen::VectorXd btil = Ur.transpose() * P.b;
        if (opt_.use_iter_refine) {
            Eigen::VectorXd corr = Ur.transpose() * (P.b - Ur * btil);
            btil += corr;
        }
        Eigen::VectorXi keep(r);
        for (int i = 0; i < r; ++i)
            keep(i) = i;
        res_.stack.emplace_back(ActRowReduce{Ur, keep, m});
        P.A = std::move(Atil);
        P.b = std::move(btil);
        P.sense.assign(r, RowSense::EQ);
        res_.orig_row_index.resize(r);
        std::iota(res_.orig_row_index.begin(), res_.orig_row_index.end(), 0);
        return true;
    }

    // ---------- passes that do NOT touch columns or c ----------
    bool remove_free_zero_rows(LP& P) {
        bool changed = false;
        for (int i = 0; i < (int)P.A.rows();) {
            if (i >= (int)P.A.rows())
                break;
            const auto row = P.A.row(i);
            if (safe_abs_max(row) <= opt_.zero_tol) {
                const double rhs = P.b(i);
                if ((P.sense[i] == RowSense::EQ && std::abs(rhs) > opt_.infeas_tol) ||
                    (P.sense[i] == RowSense::LE && rhs < -opt_.infeas_tol) ||
                    (P.sense[i] == RowSense::GE && rhs > opt_.infeas_tol)) {
                    res_.proven_infeasible = true;
                    return true;
                }
                res_.stack.emplace_back(ActRemoveRow{i, P.sense[i], rhs, row.transpose()});
                erase_row(P, i);
                changed = true;
            } else
                ++i;
        }
        return changed;
    }

    bool remove_free_zero_columns(LP& P) {
        bool changed = false;
        for (int j = 0; j < (int)P.A.cols();) {
            if (j >= (int)P.A.cols())
                break;
            const auto col = P.A.col(j);
            if (safe_abs_max(col) <= opt_.zero_tol) {
                const double cj = P.c(j);
                const bool has_l = is_finite(P.l(j));
                const bool has_u = is_finite(P.u(j));
                double xfix = std::numeric_limits<double>::quiet_NaN();
                if (cj > opt_.zero_tol && has_l) {
                    xfix = P.l(j);
                } else if (cj < -opt_.zero_tol && has_u) {
                    xfix = P.u(j);
                } else if (std::abs(cj) <= opt_.zero_tol) {
                    if (has_l)
                        xfix = P.l(j);
                    else if (has_u)
                        xfix = P.u(j);
                }
                if (!is_finite(xfix)) {
                    if ((cj > opt_.zero_tol && !has_l) || (cj < -opt_.zero_tol && !has_u)) {
                        res_.proven_unbounded = true;
                        return true;
                    }
                    ++j;
                    continue;
                }

                if (opt_.allow_structural_changes && !opt_.non_destructive) {
                    if (apply_structural_fix(P, j, xfix))
                        return true;
                    changed = true;
                    --j;
                    continue;
                }

                const double oldL = P.l(j);
                const double oldU = P.u(j);
                res_.stack.emplace_back(ActTightenBound{j, oldL, oldU});
                P.l(j) = xfix;
                P.u(j) = xfix;
                zero_col_in_index(j);
                P.A.col(j).setZero();
                changed = true;
                ++j;
            } else {
                ++j;
            }
        }
        return changed;
    }

    // non_destructive fix-and-zero
    bool fixed_variable_detection(LP& P) {
        bool changed = false;
        for (int j = 0; j < (int)P.A.cols(); ++j) {
            const double lj = P.l(j), uj = P.u(j);
            if (!(is_finite(lj) && is_finite(uj) && std::abs(lj - uj) <= opt_.zero_tol))
                continue;

            const double xfix = 0.5 * (lj + uj);
            // b <- b - A(:,j)*xfix
            P.b.noalias() -= P.A.col(j) * xfix;

            if (opt_.allow_structural_changes && !opt_.non_destructive) {
                // old behavior: remove column, shift objective
                res_.obj_shift += P.c(j) * xfix;
                res_.stack.emplace_back(ActFixVar{j, xfix, P.c(j), P.A.col(j)});
                erase_col(P, j);
                --j;
                changed = true;
            } else {
                // keep column for ranking: zero it out, keep c_j, set l=u=xfix
                res_.stack.emplace_back(ActTightenBound{j, P.l(j), P.u(j)});
                zero_col_in_index(j);
                P.A.col(j).setZero();
                P.l(j) = xfix;
                P.u(j) = xfix;
                changed = true;
            }
        }
        return changed;
    }

    bool fix_singleton_row_variable(LP& P, int j, double xfix) {
        if (!is_finite(xfix)) {
            res_.proven_infeasible = true;
            return true;
        }
        if ((is_finite(P.l(j)) && xfix < P.l(j) - opt_.infeas_tol) ||
            (is_finite(P.u(j)) && xfix > P.u(j) + opt_.infeas_tol)) {
            res_.proven_infeasible = true;
            return true;
        }
        P.b.noalias() -= P.A.col(j) * xfix;
        res_.stack.emplace_back(ActTightenBound{j, P.l(j), P.u(j)});
        zero_col_in_index(j);
        P.A.col(j).setZero();
        P.l(j) = xfix;
        P.u(j) = xfix;
        return false;
    }

    bool singleton_row_elimination(LP& P) {
        bool changed = false;
        const int m = (int)P.A.rows();
        const int n = (int)P.A.cols();
        for (int i = 0; i < m; ++i) {
            int j = -1;
            int count = 0;
            // Use sparse index for fast singleton check when available
            if (i < (int)row_nz_.size()) {
                count = (int)row_nz_[i].size();
                if (count == 1)
                    j = row_nz_[i][0];
            } else {
                for (int k = 0; k < n; ++k) {
                    if (std::abs(P.A(i, k)) <= opt_.zero_tol)
                        continue;
                    j = k;
                    ++count;
                    if (count > 1)
                        break;
                }
            }
            if (count != 1)
                continue;

            const double aij = P.A(i, j);
            const double rhs = P.b(i);
            double bound_value = std::numeric_limits<double>::quiet_NaN();
            bool tighten_lower = false;
            bool tighten_upper = false;
            bool fix_value = false;

            if (P.sense[i] == RowSense::EQ) {
                bound_value = rhs / aij;
                fix_value = true;
            } else if (P.sense[i] == RowSense::LE) {
                if (aij > 0.0) {
                    bound_value = rhs / aij;
                    tighten_upper = true;
                } else {
                    bound_value = rhs / aij;
                    tighten_lower = true;
                }
            } else if (P.sense[i] == RowSense::GE) {
                if (aij > 0.0) {
                    bound_value = rhs / aij;
                    tighten_lower = true;
                } else {
                    bound_value = rhs / aij;
                    tighten_upper = true;
                }
            }

            if (fix_value) {
                if (opt_.allow_structural_changes && !opt_.non_destructive) {
                    if (apply_structural_fix(P, j, bound_value))
                        return true;
                    changed = true;
                    continue;
                }
                if (fix_singleton_row_variable(P, j, bound_value))
                    return true;
                changed = true;
                continue;
            }

            const double oldL = P.l(j);
            const double oldU = P.u(j);
            double newL = oldL;
            double newU = oldU;
            if (tighten_lower) {
                newL = std::max(newL, bound_value);
            }
            if (tighten_upper) {
                newU = std::min(newU, bound_value);
            }
            if (newL > newU + opt_.infeas_tol) {
                res_.proven_infeasible = true;
                return true;
            }
            const bool lower_unchanged =
                (is_finite(newL) && is_finite(oldL) ? std::abs(newL - oldL) <= opt_.zero_tol
                                                    : (!is_finite(newL) && !is_finite(oldL)));
            const bool upper_unchanged =
                (is_finite(newU) && is_finite(oldU) ? std::abs(newU - oldU) <= opt_.zero_tol
                                                    : (!is_finite(newU) && !is_finite(oldU)));
            if (lower_unchanged && upper_unchanged)
                continue;

            if (is_finite(newL) && is_finite(newU) && std::abs(newU - newL) <= opt_.zero_tol) {
                const double xfix = 0.5 * (newL + newU);
                if (opt_.allow_structural_changes && !opt_.non_destructive) {
                    if (apply_structural_fix(P, j, xfix))
                        return true;
                    changed = true;
                    continue;
                }
                if (fix_singleton_row_variable(P, j, xfix))
                    return true;
                changed = true;
                continue;
            }

            if (!is_finite(newL) && is_finite(newU)) {
                if (!is_finite(oldL) || newU < oldU - opt_.zero_tol) {
                    res_.stack.emplace_back(ActTightenBound{j, oldL, oldU});
                    P.u(j) = newU;
                    res_.implied_bound_updates += 1;
                    changed = true;
                }
            } else if (is_finite(newL) && !is_finite(newU)) {
                if (!is_finite(oldU) || newL > oldL + opt_.zero_tol) {
                    res_.stack.emplace_back(ActTightenBound{j, oldL, oldU});
                    P.l(j) = newL;
                    res_.implied_bound_updates += 1;
                    changed = true;
                }
            } else if (is_finite(newL) && is_finite(newU)) {
                if (newL > oldL + opt_.zero_tol || newU < oldU - opt_.zero_tol) {
                    res_.stack.emplace_back(ActTightenBound{j, oldL, oldU});
                    P.l(j) = newL;
                    P.u(j) = newU;
                    res_.implied_bound_updates += 1;
                    changed = true;
                }
            }
        }
        return changed;
    }

    bool tighten_bounds_by_rows(LP& P) {
        if (!rows_feasible_(P, nullptr))
            return true;
        const ImpliedBoundsSummary implied = collect_implied_bounds_(P, nullptr);
        return apply_implied_bounds_(P, implied, nullptr, false);
    }

    bool domain_propagation_rounds(LP& P) {
        bool changed_any = false;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0 || n == 0)
            return false;

        // Use persistent sparse index if available; otherwise build locally
        const bool use_global = ((int)row_nz_.size() == m && (int)col_nz_.size() == n);

        std::vector<std::vector<int>> local_row_cols, local_col_rows;
        const std::vector<std::vector<int>>& row_cols = use_global ? row_nz_ : local_row_cols;
        const std::vector<std::vector<int>>& col_rows = use_global ? col_nz_ : local_col_rows;

        if (!use_global) {
            local_row_cols.assign(m, {});
            local_col_rows.assign(n, {});
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (std::abs(P.A(i, j)) <= opt_.zero_tol)
                        continue;
                    local_row_cols[i].push_back(j);
                    local_col_rows[j].push_back(i);
                }
            }
        }
        (void)row_cols; // silence unused-variable warning (used via references above)

        std::vector<char> dirty_rows(m, 1), dirty_cols(n, 0);
        const int max_rounds = std::max(2, opt_.max_passes);
        for (int round = 0; round < max_rounds; ++round) {
            bool any_dirty_row = false;
            std::fill(dirty_cols.begin(), dirty_cols.end(), 0);

            for (int i = 0; i < m; ++i) {
                if (dirty_rows[i])
                    any_dirty_row = true;
            }

            if (!any_dirty_row)
                break;
            if (!rows_feasible_(P, &dirty_rows))
                return true;

            const ImpliedBoundsSummary implied = collect_implied_bounds_(P, &dirty_rows);
            const bool round_changed = apply_implied_bounds_(P, implied, &dirty_cols, true);
            if (res_.proven_infeasible)
                return true;
            if (!round_changed)
                break;
            changed_any = true;

            std::fill(dirty_rows.begin(), dirty_rows.end(), 0);
            for (int j = 0; j < n; ++j) {
                if (!dirty_cols[j])
                    continue;
                for (int idx = 0; idx < (int)col_rows[j].size(); ++idx) {
                    dirty_rows[col_rows[j][idx]] = 1;
                }
            }
        }
        return changed_any;
    }

    bool domain_propagation_once(LP& P) {
        if (!rows_feasible_(P, nullptr))
            return true;
        const ImpliedBoundsSummary implied = collect_implied_bounds_(P, nullptr);
        return apply_implied_bounds_(P, implied, nullptr, true);
    }

    ImpliedInterval singleton_implied_interval(const LP& P, int row_idx, int j) const {
        ImpliedInterval implied;
        const Eigen::RowVectorXd row = P.A.row(row_idx);
        const double rhs = P.b(row_idx);
        const double aij = row(j);
        const auto other = row_activity_range_excluding(row, P.l, P.u, j, opt_.zero_tol);

        if (P.sense[row_idx] == RowSense::LE) {
            if (aij > 0 && other.min_finite) {
                implied.upper = (rhs - other.min_act) / aij;
                implied.has_upper = true;
            } else if (aij < 0 && other.min_finite) {
                implied.lower = (rhs - other.min_act) / aij;
                implied.has_lower = true;
            }
        } else if (P.sense[row_idx] == RowSense::GE) {
            if (aij > 0 && other.max_finite) {
                implied.lower = (rhs - other.max_act) / aij;
                implied.has_lower = true;
            } else if (aij < 0 && other.max_finite) {
                implied.upper = (rhs - other.max_act) / aij;
                implied.has_upper = true;
            }
        } else {
            if (aij > 0) {
                if (other.max_finite) {
                    implied.lower = (rhs - other.max_act) / aij;
                    implied.has_lower = true;
                }
                if (other.min_finite) {
                    implied.upper = (rhs - other.min_act) / aij;
                    implied.has_upper = true;
                }
            } else {
                if (other.min_finite) {
                    implied.lower = (rhs - other.min_act) / aij;
                    implied.has_lower = true;
                }
                if (other.max_finite) {
                    implied.upper = (rhs - other.max_act) / aij;
                    implied.has_upper = true;
                }
            }
        }

        if (implied.has_lower && implied.has_upper && implied.lower > implied.upper) {
            std::swap(implied.lower, implied.upper);
        }
        return implied;
    }

    bool rows_feasible_(const LP& P, const std::vector<char>* active_rows) {
        const int m = (int)P.A.rows();
        const int n = (int)P.A.cols();
        for (int i = 0; i < m; ++i) {
            if (active_rows && (i >= (int)active_rows->size() || !(*active_rows)[i]))
                continue;

            // Compute row activity using sparse index when available
            double min_finite = 0.0, max_finite = 0.0;
            int min_inf_cnt = 0, max_inf_cnt = 0;

            auto process_row_col = [&](int j) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    return;
                if (aij > 0.0) {
                    if (is_finite(P.l(j)))
                        min_finite += aij * P.l(j);
                    else
                        ++min_inf_cnt;
                    if (is_finite(P.u(j)))
                        max_finite += aij * P.u(j);
                    else
                        ++max_inf_cnt;
                } else {
                    if (is_finite(P.u(j)))
                        min_finite += aij * P.u(j);
                    else
                        ++min_inf_cnt;
                    if (is_finite(P.l(j)))
                        max_finite += aij * P.l(j);
                    else
                        ++max_inf_cnt;
                }
            };

            if (i < (int)row_nz_.size()) {
                for (int j : row_nz_[i])
                    process_row_col(j);
            } else {
                for (int j = 0; j < n; ++j)
                    process_row_col(j);
            }

            const double rhs = P.b(i);
            const bool min_f = (min_inf_cnt == 0);
            const bool max_f = (max_inf_cnt == 0);

            if (P.sense[i] == RowSense::LE) {
                if (min_f && min_finite > rhs + opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return false;
                }
            } else if (P.sense[i] == RowSense::GE) {
                if (max_f && max_finite < rhs - opt_.infeas_tol) {
                    res_.proven_infeasible = true;
                    return false;
                }
            } else {
                if ((min_f && min_finite > rhs + opt_.infeas_tol) ||
                    (max_f && max_finite < rhs - opt_.infeas_tol)) {
                    res_.proven_infeasible = true;
                    return false;
                }
            }
        }
        return true;
    }

    void merge_implied_interval_(ImpliedBoundsSummary& summary, int j,
                                 const ImpliedInterval& implied) const {
        if (j < 0 || j >= summary.impl_col_lower.size())
            return;
        if (implied.has_lower) {
            if (!summary.has_lower[j] || implied.lower > summary.impl_col_lower(j)) {
                summary.impl_col_lower(j) = implied.lower;
            }
            summary.has_lower[j] = 1;
        }
        if (implied.has_upper) {
            if (!summary.has_upper[j] || implied.upper < summary.impl_col_upper(j)) {
                summary.impl_col_upper(j) = implied.upper;
            }
            summary.has_upper[j] = 1;
        }
    }

    ImpliedBoundsSummary collect_implied_bounds_(const LP& P,
                                                 const std::vector<char>* active_rows) const {
        const int m = (int)P.A.rows();
        const int n = (int)P.A.cols();
        ImpliedBoundsSummary summary(n);

        for (int i = 0; i < m; ++i) {
            if (active_rows && (i >= (int)active_rows->size() || !(*active_rows)[i]))
                continue;

            // Use sparse index when available, else scan densely
            const std::vector<int>* nz_cols_ptr = (i < (int)row_nz_.size()) ? &row_nz_[i] : nullptr;

            // Build local sparse list + row activity summary in one pass
            struct NzEntry {
                int j;
                double aij;
            };
            std::vector<NzEntry> entries;

            double min_finite = 0.0, max_finite = 0.0;
            int min_inf_cnt = 0, max_inf_cnt = 0;

            auto process_col = [&](int j) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    return;
                entries.push_back({j, aij});
                if (aij > 0.0) {
                    if (is_finite(P.l(j)))
                        min_finite += aij * P.l(j);
                    else
                        ++min_inf_cnt;
                    if (is_finite(P.u(j)))
                        max_finite += aij * P.u(j);
                    else
                        ++max_inf_cnt;
                } else {
                    if (is_finite(P.u(j)))
                        min_finite += aij * P.u(j);
                    else
                        ++min_inf_cnt;
                    if (is_finite(P.l(j)))
                        max_finite += aij * P.l(j);
                    else
                        ++max_inf_cnt;
                }
            };

            if (nz_cols_ptr) {
                entries.reserve(nz_cols_ptr->size());
                for (int j : *nz_cols_ptr)
                    process_col(j);
            } else {
                for (int j = 0; j < n; ++j)
                    process_col(j);
            }

            if (entries.empty())
                continue;

            const double rhs = P.b(i);

            for (const auto& [j, aij] : entries) {
                // Subtract column j's contribution from row activity to get "other" activity
                double other_min = min_finite;
                int other_min_inf = min_inf_cnt;
                const bool j_contributes_to_min_inf =
                    (aij > 0.0) ? !is_finite(P.l(j)) : !is_finite(P.u(j));
                if (j_contributes_to_min_inf) {
                    --other_min_inf;
                } else {
                    other_min -= (aij > 0.0) ? aij * P.l(j) : aij * P.u(j);
                }

                double other_max = max_finite;
                int other_max_inf = max_inf_cnt;
                const bool j_contributes_to_max_inf =
                    (aij > 0.0) ? !is_finite(P.u(j)) : !is_finite(P.l(j));
                if (j_contributes_to_max_inf) {
                    --other_max_inf;
                } else {
                    other_max -= (aij > 0.0) ? aij * P.u(j) : aij * P.l(j);
                }

                const bool omin_finite = (other_min_inf == 0);
                const bool omax_finite = (other_max_inf == 0);

                ImpliedInterval impl;
                if (P.sense[i] == RowSense::LE) {
                    if (aij > 0.0 && omin_finite) {
                        impl.upper = (rhs - other_min) / aij;
                        impl.has_upper = true;
                    } else if (aij < 0.0 && omin_finite) {
                        impl.lower = (rhs - other_min) / aij;
                        impl.has_lower = true;
                    }
                } else if (P.sense[i] == RowSense::GE) {
                    if (aij > 0.0 && omax_finite) {
                        impl.lower = (rhs - other_max) / aij;
                        impl.has_lower = true;
                    } else if (aij < 0.0 && omax_finite) {
                        impl.upper = (rhs - other_max) / aij;
                        impl.has_upper = true;
                    }
                } else { // EQ
                    if (aij > 0.0) {
                        if (omax_finite) {
                            impl.lower = (rhs - other_max) / aij;
                            impl.has_lower = true;
                        }
                        if (omin_finite) {
                            impl.upper = (rhs - other_min) / aij;
                            impl.has_upper = true;
                        }
                    } else {
                        if (omin_finite) {
                            impl.lower = (rhs - other_min) / aij;
                            impl.has_lower = true;
                        }
                        if (omax_finite) {
                            impl.upper = (rhs - other_max) / aij;
                            impl.has_upper = true;
                        }
                    }
                }
                if (impl.has_lower && impl.has_upper && impl.lower > impl.upper)
                    std::swap(impl.lower, impl.upper);
                merge_implied_interval_(summary, j, impl);
            }
        }
        return summary;
    }

    bool apply_implied_bounds_(LP& P, const ImpliedBoundsSummary& summary,
                               std::vector<char>* dirty_cols, bool require_big_delta) {
        bool changed = false;
        int j = 0;
        while (j < (int)P.A.cols()) {
            double newL = P.l(j);
            double newU = P.u(j);
            if (j < (int)summary.has_lower.size() && summary.has_lower[j]) {
                newL = std::max(newL, summary.impl_col_lower(j));
            }
            if (j < (int)summary.has_upper.size() && summary.has_upper[j]) {
                newU = std::min(newU, summary.impl_col_upper(j));
            }

            if (newL > newU + opt_.infeas_tol) {
                res_.proven_infeasible = true;
                return true;
            }
            if (is_finite(newL) && is_finite(newU) && std::abs(newU - newL) <= opt_.zero_tol) {
                const double xfix = 0.5 * (newL + newU);
                newL = xfix;
                newU = xfix;
                if (opt_.allow_structural_changes && !opt_.non_destructive) {
                    if (apply_structural_fix(P, j, xfix))
                        return true;
                    changed = true;
                    continue; // current index now holds next column
                }
            }

            const double oldL = P.l(j);
            const double oldU = P.u(j);
            const bool tightenL =
                is_finite(newL) && (!is_finite(oldL) || newL > oldL + opt_.zero_tol) &&
                (!require_big_delta || !is_finite(oldL) || (newL - oldL) > domprop_min_delta_);
            const bool tightenU =
                is_finite(newU) && (!is_finite(oldU) || newU < oldU - opt_.zero_tol) &&
                (!require_big_delta || !is_finite(oldU) || (oldU - newU) > domprop_min_delta_);
            if (!tightenL && !tightenU) {
                ++j;
                continue;
            }

            res_.stack.emplace_back(ActTightenBound{j, oldL, oldU});
            P.l(j) = tightenL ? newL : oldL;
            P.u(j) = tightenU ? newU : oldU;
            ++res_.implied_bound_updates;
            if (dirty_cols && j < (int)dirty_cols->size()) {
                (*dirty_cols)[j] = 1;
            }
            changed = true;
            ++j;
        }
        return changed;
    }

    bool apply_structural_fix(LP& P, int j, double xfix) {
        if ((is_finite(P.l(j)) && xfix < P.l(j) - opt_.infeas_tol) ||
            (is_finite(P.u(j)) && xfix > P.u(j) + opt_.infeas_tol)) {
            res_.proven_infeasible = true;
            return true;
        }
        res_.obj_shift += P.c(j) * xfix;
        res_.stack.emplace_back(ActFixVar{j, xfix, P.c(j), P.A.col(j)});
        P.b.noalias() -= P.A.col(j) * xfix;
        erase_col(P, j);
        return false;
    }

    struct ProbeSnapshot {
        std::size_t stack_size = 0;
        double obj_shift = 0.0;
        bool proven_infeasible = false;
        bool proven_unbounded = false;
    };

    ProbeSnapshot save_probe_snapshot_() const {
        ProbeSnapshot s;
        s.stack_size = res_.stack.size();
        s.obj_shift = res_.obj_shift;
        s.proven_infeasible = res_.proven_infeasible;
        s.proven_unbounded = res_.proven_unbounded;
        return s;
    }

    void restore_probe_snapshot_(const ProbeSnapshot& s) {
        res_.stack.resize(s.stack_size);
        res_.obj_shift = s.obj_shift;
        res_.proven_infeasible = s.proven_infeasible;
        res_.proven_unbounded = s.proven_unbounded;
    }

    bool run_probe_fix_(const LP& P, int j, double xfix) {
        ProbeSnapshot snapshot = save_probe_snapshot_();

        LP Q = P;
        Q.l(j) = xfix;
        Q.u(j) = xfix;

        bool feasible = true;
        if (!check_and_fix_bounds(Q)) {
            feasible = false;
        } else {
            for (int round = 0; round < std::max(1, opt_.probing_max_rounds); ++round) {
                bool changed = false;
                changed |= tighten_bounds_by_rows(Q);
                if (res_.proven_infeasible) {
                    feasible = false;
                    break;
                }
                changed |= domain_propagation_rounds(Q);
                if (res_.proven_infeasible) {
                    feasible = false;
                    break;
                }
                if (!changed)
                    break;
            }
        }

        restore_probe_snapshot_(snapshot);
        return feasible;
    }

    bool objective_guided_probing(LP& P) {
        if (!opt_.enable_objective_probing)
            return false;
        if (opt_.probing_max_vars <= 0)
            return false;
        if (P.A.cols() == 0 || P.A.rows() == 0)
            return false;

        std::vector<std::pair<double, int>> scored;
        scored.reserve(P.A.cols());
        for (int j = 0; j < (int)P.A.cols(); ++j) {
            const double cj = std::abs(P.c(j));
            if (cj <= opt_.probing_obj_tol)
                continue;
            if (safe_abs_max(P.A.col(j)) <= opt_.zero_tol)
                continue;
            if (is_finite(P.l(j)) && is_finite(P.u(j)) &&
                std::abs(P.u(j) - P.l(j)) <= opt_.zero_tol) {
                continue;
            }

            double width_scale = 1.0;
            if (is_finite(P.l(j)) && is_finite(P.u(j))) {
                width_scale = std::max(1.0, std::abs(P.u(j) - P.l(j)));
            }
            scored.emplace_back(cj * width_scale, j);
        }

        if (scored.empty())
            return false;

        const int keep = std::min(opt_.probing_max_vars, static_cast<int>(scored.size()));
        std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                          [](const auto& a, const auto& b) {
                              if (a.first != b.first)
                                  return a.first > b.first;
                              return a.second < b.second;
                          });

        for (int idx = 0; idx < keep; ++idx) {
            const int j = scored[idx].second;
            const bool can_probe_l = is_finite(P.l(j));
            const bool can_probe_u = is_finite(P.u(j));
            if (!can_probe_l && !can_probe_u)
                continue;

            bool lower_feasible = true;
            bool upper_feasible = true;
            if (can_probe_l)
                lower_feasible = run_probe_fix_(P, j, P.l(j));
            if (can_probe_u)
                upper_feasible = run_probe_fix_(P, j, P.u(j));

            if (can_probe_l && can_probe_u && !lower_feasible && !upper_feasible) {
                res_.proven_infeasible = true;
                return true;
            }
        }

        return false;
    }

    // -----------------------------------------------------------------------
    // reduced_cost_lurking_bounds (HiGHS-style)
    // -----------------------------------------------------------------------
    // For each continuous, non-fixed column j whose objective coefficient c_j
    // is non-zero and whose every appearance is in a one-sided inequality
    // (no EQ rows), we exploit the complementary-slackness / reduced-cost
    // condition to infer that certain rows must be *tight* at any LP optimum,
    // then propagate primal bounds from those tight rows.
    //
    // Key observation:
    //   LE row i: dual y_i <= 0;  if c_j > 0 and a_ij > 0 => rc = c_j - a_ij*y_i > 0
    //     => x_j pushed to lower bound (already handled by dual_fix_by_locks).
    //   NEW case – mixed signs:
    //     For a given row i, bounding y_i from rc >= 0 using worst-case y_k=0
    //     for all other rows k gives:  y_i <= c_j / a_ij  (if a_ij > 0, LE row).
    //     If c_j / a_ij < 0 (i.e. c_j < 0 and a_ij > 0), then y_i < 0, meaning
    //     row i MUST be tight (dual > 0 in absolute value => active constraint).
    //     We then treat row i as equality and tighten all columns in it.
    // -----------------------------------------------------------------------
    bool reduced_cost_lurking_bounds(LP& P) {
        if (!opt_.enable_reduced_cost_lurking)
            return false;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0 || n == 0)
            return false;

        ImpliedBoundsSummary summary(n);

        for (int j = 0; j < n; ++j) {
            // Skip fixed variables
            if (is_finite(P.l(j)) && is_finite(P.u(j)) &&
                std::abs(P.u(j) - P.l(j)) <= opt_.zero_tol)
                continue;

            const double cj = P.c(j);
            if (std::abs(cj) <= opt_.zero_tol)
                continue;

            const std::vector<int>& rows_j =
                (j < (int)col_nz_.size()) ? col_nz_[j] : std::vector<int>{};
            if (rows_j.empty())
                continue;

            // All rows must be one-sided inequalities for this to apply
            bool all_one_sided = true;
            for (int i : rows_j) {
                if (P.sense[i] == RowSense::EQ) { all_one_sided = false; break; }
            }
            if (!all_one_sided)
                continue;

            // For each row i containing j, derive an implied dual bound and
            // check if that forces the row to be tight.
            for (int i : rows_j) {
                const double aij = P.A(i, j);
                if (std::abs(aij) <= opt_.zero_tol)
                    continue;

                // Dual sign: LE => y_i <= 0; GE => y_i >= 0
                // From rc condition (worst case other duals = 0):
                //   y_i <= c_j / a_ij   (if a_ij > 0)
                //   y_i >= c_j / a_ij   (if a_ij < 0)
                const double yi_bound = cj / aij;

                bool row_forced_tight = false;
                if (P.sense[i] == RowSense::LE) {
                    // y_i <= 0 by complementarity
                    // If yi_bound < 0 strictly: y_i is forced < 0 => row active
                    if (aij > opt_.zero_tol && yi_bound < -opt_.dual_fix_tol)
                        row_forced_tight = true;
                } else { // GE
                    // y_i >= 0 by complementarity
                    // If yi_bound > 0 strictly: y_i is forced > 0 => row active
                    if (aij < -opt_.zero_tol && yi_bound > opt_.dual_fix_tol)
                        row_forced_tight = true;
                }

                if (!row_forced_tight)
                    continue;

                // Row i is forced tight at optimality: propagate EQ-style bounds.
                const std::vector<int>* nz_row_ptr =
                    (i < (int)row_nz_.size()) ? &row_nz_[i] : nullptr;

                struct NzE { int k; double a; };
                std::vector<NzE> entries;
                double min_finite = 0.0, max_finite = 0.0;
                int min_inf_cnt = 0, max_inf_cnt = 0;

                auto process = [&](int k) {
                    const double a = P.A(i, k);
                    if (std::abs(a) <= opt_.zero_tol) return;
                    entries.push_back({k, a});
                    if (a > 0.0) {
                        if (is_finite(P.l(k))) min_finite += a * P.l(k); else ++min_inf_cnt;
                        if (is_finite(P.u(k))) max_finite += a * P.u(k); else ++max_inf_cnt;
                    } else {
                        if (is_finite(P.u(k))) min_finite += a * P.u(k); else ++min_inf_cnt;
                        if (is_finite(P.l(k))) max_finite += a * P.l(k); else ++max_inf_cnt;
                    }
                };
                if (nz_row_ptr) {
                    entries.reserve(nz_row_ptr->size());
                    for (int k : *nz_row_ptr) process(k);
                } else {
                    for (int k = 0; k < n; ++k) process(k);
                }

                const double rhs = P.b(i);
                for (const auto& [k, a] : entries) {
                    double omin = min_finite; int omin_inf = min_inf_cnt;
                    double omax = max_finite; int omax_inf = max_inf_cnt;
                    if (a > 0.0) {
                        if (is_finite(P.l(k))) omin -= a * P.l(k); else --omin_inf;
                        if (is_finite(P.u(k))) omax -= a * P.u(k); else --omax_inf;
                    } else {
                        if (is_finite(P.u(k))) omin -= a * P.u(k); else --omin_inf;
                        if (is_finite(P.l(k))) omax -= a * P.l(k); else --omax_inf;
                    }
                    const bool omin_f = (omin_inf == 0), omax_f = (omax_inf == 0);
                    ImpliedInterval impl;
                    // EQ-style: both bounds
                    if (a > 0.0) {
                        if (omax_f) { impl.lower = (rhs - omax) / a; impl.has_lower = true; }
                        if (omin_f) { impl.upper = (rhs - omin) / a; impl.has_upper = true; }
                    } else {
                        if (omin_f) { impl.lower = (rhs - omin) / a; impl.has_lower = true; }
                        if (omax_f) { impl.upper = (rhs - omax) / a; impl.has_upper = true; }
                    }
                    if (impl.has_lower && impl.has_upper && impl.lower > impl.upper)
                        std::swap(impl.lower, impl.upper);
                    merge_implied_interval_(summary, k, impl);
                }
            }
        }

        return apply_implied_bounds_(P, summary, nullptr, false);
    }

    // -----------------------------------------------------------------------
    // row_aggregation (Farkas/Chvátal aggregation through free continuous vars)
    // -----------------------------------------------------------------------
    // For a continuous free variable x_j (l_j = -inf, u_j = +inf) appearing in
    // both a LE and a GE row, eliminate x_j by Farkas aggregation:
    //   lambda * (LE row) + (GE row), lambda > 0 chosen to cancel x_j.
    // The resulting constraint is valid and may tighten bounds on other columns.
    // We propagate the new bounds immediately without modifying A.
    // -----------------------------------------------------------------------
    bool row_aggregation(LP& P) {
        if (!opt_.enable_row_aggregation)
            return false;
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (m == 0 || n == 0)
            return false;

        ImpliedBoundsSummary summary(n);
        int pairs_processed = 0;

        for (int j = 0; j < n && pairs_processed < opt_.row_agg_max_pairs; ++j) {
            // Must be free (both bounds infinite)
            if (is_finite(P.l(j)) || is_finite(P.u(j)))
                continue;

            const std::vector<int>& rows_j =
                (j < (int)col_nz_.size()) ? col_nz_[j] : std::vector<int>{};
            if (rows_j.size() < 2)
                continue;

            std::vector<int> le_rows, ge_rows;
            for (int i : rows_j) {
                if (P.sense[i] == RowSense::LE) le_rows.push_back(i);
                else if (P.sense[i] == RowSense::GE) ge_rows.push_back(i);
            }
            if (le_rows.empty() || ge_rows.empty())
                continue;

            for (int ir : le_rows) {
                for (int ig : ge_rows) {
                    if (pairs_processed >= opt_.row_agg_max_pairs) goto done_agg;
                    const double a_le = P.A(ir, j);
                    const double a_ge = P.A(ig, j);
                    if (std::abs(a_le) <= opt_.zero_tol || std::abs(a_ge) <= opt_.zero_tol)
                        continue;

                    // lambda = -a_ge / a_le must be > 0 for Farkas validity
                    const double lambda = -a_ge / a_le;
                    if (lambda <= opt_.zero_tol)
                        continue;

                    ++pairs_processed;

                    // Compute aggregated row coefficients and activity range
                    // Aggregate = lambda*(LE row) + (GE row)  -> new GE constraint
                    // Collect columns in union of the two rows
                    const auto& ri = (ir < (int)row_nz_.size()) ? row_nz_[ir] : std::vector<int>{};
                    const auto& rg = (ig < (int)row_nz_.size()) ? row_nz_[ig] : std::vector<int>{};

                    std::vector<int> cols_union;
                    cols_union.reserve(ri.size() + rg.size());
                    cols_union.insert(cols_union.end(), ri.begin(), ri.end());
                    for (int k : rg) {
                        if (std::find(ri.begin(), ri.end(), k) == ri.end())
                            cols_union.push_back(k);
                    }

                    const double agg_rhs = lambda * P.b(ir) + P.b(ig);

                    struct AggEntry { int k; double coeff; };
                    std::vector<AggEntry> agg;
                    agg.reserve(cols_union.size());
                    double min_act = 0.0, max_act = 0.0;
                    int min_inf = 0, max_inf = 0;

                    for (int k : cols_union) {
                        if (k == j) continue; // cancelled
                        const double coeff = lambda * P.A(ir, k) + P.A(ig, k);
                        if (std::abs(coeff) <= opt_.zero_tol) continue;
                        agg.push_back({k, coeff});
                        if (coeff > 0.0) {
                            if (is_finite(P.l(k))) min_act += coeff * P.l(k); else ++min_inf;
                            if (is_finite(P.u(k))) max_act += coeff * P.u(k); else ++max_inf;
                        } else {
                            if (is_finite(P.u(k))) min_act += coeff * P.u(k); else ++min_inf;
                            if (is_finite(P.l(k))) max_act += coeff * P.l(k); else ++max_inf;
                        }
                    }
                    if (agg.empty()) continue;

                    // Infeasibility: max activity < rhs for GE
                    if (max_inf == 0 && max_act < agg_rhs - opt_.infeas_tol) {
                        res_.proven_infeasible = true;
                        return true;
                    }

                    // Propagate GE bounds
                    for (const auto& [k, coeff] : agg) {
                        double other_max = max_act; int o_max_inf = max_inf;
                        if (coeff > 0.0) {
                            if (is_finite(P.u(k))) other_max -= coeff * P.u(k); else --o_max_inf;
                        } else {
                            if (is_finite(P.l(k))) other_max -= coeff * P.l(k); else --o_max_inf;
                        }
                        if (o_max_inf != 0) continue;
                        const double implied = (agg_rhs - other_max) / coeff;
                        ImpliedInterval impl;
                        if (coeff > 0.0) { impl.lower = implied; impl.has_lower = true; }
                        else             { impl.upper = implied; impl.has_upper = true; }
                        merge_implied_interval_(summary, k, impl);
                    }
                }
            }
        }
        done_agg:

        return apply_implied_bounds_(P, summary, nullptr, false);
    }

    bool dual_fix_by_locks(LP& P) {
        if (!opt_.enable_dual_fixing)
            return false;

        bool changed = false;
        const int m = (int)P.A.rows();
        for (int j = 0; j < (int)P.A.cols(); ++j) {
            const double cj = P.c(j);
            if (std::abs(cj) <= opt_.dual_fix_tol)
                continue;

            const bool can_fix_lower = (cj > opt_.dual_fix_tol) && is_finite(P.l(j));
            const bool can_fix_upper = (cj < -opt_.dual_fix_tol) && is_finite(P.u(j));
            if (!can_fix_lower && !can_fix_upper)
                continue;

            const bool already_fixed = is_finite(P.l(j)) && is_finite(P.u(j)) &&
                                       std::abs(P.u(j) - P.l(j)) <= opt_.zero_tol;
            if (already_fixed)
                continue;

            bool has_eq = false;
            bool up_relaxes = false;
            bool down_relaxes = false;

            // Use col_nz_ for sparse row iteration when available
            const bool use_nz = (j < (int)col_nz_.size());
            if (use_nz) {
                for (int i : col_nz_[j]) {
                    const double aij = P.A(i, j);
                    if (std::abs(aij) <= opt_.zero_tol)
                        continue;
                    if (P.sense[i] == RowSense::EQ) {
                        has_eq = true;
                        break;
                    }
                    if (P.sense[i] == RowSense::LE) {
                        if (aij < 0.0)
                            up_relaxes = true;
                        if (aij > 0.0)
                            down_relaxes = true;
                    } else if (P.sense[i] == RowSense::GE) {
                        if (aij > 0.0)
                            up_relaxes = true;
                        if (aij < 0.0)
                            down_relaxes = true;
                    }
                    if ((can_fix_lower && up_relaxes) || (can_fix_upper && down_relaxes))
                        break;
                }
            } else {
                for (int i = 0; i < m; ++i) {
                    const double aij = P.A(i, j);
                    if (std::abs(aij) <= opt_.zero_tol)
                        continue;

                    if (P.sense[i] == RowSense::EQ) {
                        has_eq = true;
                        break;
                    }

                    if (P.sense[i] == RowSense::LE) {
                        if (aij < 0.0)
                            up_relaxes = true;
                        if (aij > 0.0)
                            down_relaxes = true;
                    } else if (P.sense[i] == RowSense::GE) {
                        if (aij > 0.0)
                            up_relaxes = true;
                        if (aij < 0.0)
                            down_relaxes = true;
                    }

                    if ((can_fix_lower && up_relaxes) || (can_fix_upper && down_relaxes)) {
                        break;
                    }
                }
            }

            if (has_eq)
                continue;

            const bool fix_to_lower = can_fix_lower && !up_relaxes;
            const bool fix_to_upper = can_fix_upper && !down_relaxes;
            if (!fix_to_lower && !fix_to_upper)
                continue;

            const double xfix = fix_to_lower ? P.l(j) : P.u(j);
            if (!is_finite(xfix))
                continue;

            P.b.noalias() -= P.A.col(j) * xfix;
            res_.stack.emplace_back(ActDualFix{j, P.l(j), P.u(j), xfix});

            if (opt_.allow_structural_changes && !opt_.non_destructive) {
                res_.obj_shift += P.c(j) * xfix;
                erase_col(P, j);
                --j;
            } else {
                zero_col_in_index(j);
                P.A.col(j).setZero();
                P.l(j) = xfix;
                P.u(j) = xfix;
            }

            changed = true;
        }

        return changed;
    }

    bool singleton_column_substitution(LP& P) {
        if (!opt_.allow_structural_changes || opt_.non_destructive)
            return false;

        bool changed = false;
        for (int j = 0; j < (int)P.A.cols(); ++j) {
            int row_idx = -1;
            int col_nnz = 0;
            if (j < (int)col_nz_.size()) {
                col_nnz = (int)col_nz_[j].size();
                if (col_nnz == 1)
                    row_idx = col_nz_[j][0];
            } else {
                for (int i = 0; i < (int)P.A.rows(); ++i) {
                    if (std::abs(P.A(i, j)) <= opt_.zero_tol)
                        continue;
                    row_idx = i;
                    ++col_nnz;
                    if (col_nnz > 1)
                        break;
                }
            }
            if (col_nnz != 1 || row_idx < 0)
                continue;

            const Eigen::RowVectorXd row = P.A.row(row_idx);
            const double rhs = P.b(row_idx);
            const double aij = row(j);

            int row_nnz = 0;
            if (row_idx < (int)row_nz_.size()) {
                row_nnz = (int)row_nz_[row_idx].size();
            } else {
                for (int k = 0; k < row.size(); ++k)
                    if (std::abs(row(k)) > opt_.zero_tol)
                        ++row_nnz;
            }

            const ImpliedInterval implied = singleton_implied_interval(P, row_idx, j);
            double eff_l = is_finite(P.l(j)) ? P.l(j) : ninf();
            double eff_u = is_finite(P.u(j)) ? P.u(j) : inf();
            if (implied.has_lower)
                eff_l = std::max(eff_l, implied.lower);
            if (implied.has_upper)
                eff_u = std::min(eff_u, implied.upper);
            if (eff_l > eff_u + opt_.infeas_tol) {
                res_.proven_infeasible = true;
                return true;
            }

            if (is_finite(eff_l) && is_finite(eff_u) && std::abs(eff_u - eff_l) <= opt_.zero_tol) {
                if (apply_structural_fix(P, j, 0.5 * (eff_l + eff_u))) {
                    return true;
                }
                changed = true;
                --j;
                continue;
            }

            const bool eq_row = P.sense[row_idx] == RowSense::EQ;
            if (!eq_row)
                continue;

            if (row_nnz == 1) {
                const double xfix = rhs / aij;
                if ((is_finite(P.l(j)) && xfix < P.l(j) - opt_.infeas_tol) ||
                    (is_finite(P.u(j)) && xfix > P.u(j) + opt_.infeas_tol)) {
                    res_.proven_infeasible = true;
                    return true;
                }

                res_.obj_shift += P.c(j) * xfix;
                res_.stack.emplace_back(
                    ActSingletonRowElim{row_idx, j, RowSense::EQ, rhs, aij, row.transpose()});
                erase_row(P, row_idx);
                erase_col(P, j);
                changed = true;
                --j;
                continue;
            }

            const bool lower_redundant =
                !is_finite(P.l(j)) ||
                (implied.has_lower && implied.lower >= P.l(j) - opt_.infeas_tol);
            const bool upper_redundant =
                !is_finite(P.u(j)) ||
                (implied.has_upper && implied.upper <= P.u(j) + opt_.infeas_tol);
            if (!(lower_redundant && upper_redundant))
                continue;

            res_.obj_shift += P.c(j) * (rhs / aij);
            for (int k = 0; k < (int)P.A.cols(); ++k) {
                if (k == j)
                    continue;
                const double aik = row(k);
                if (std::abs(aik) <= opt_.zero_tol)
                    continue;
                P.c(k) -= P.c(j) * (aik / aij);
            }

            res_.stack.emplace_back(
                ActSingletonRowElim{row_idx, j, RowSense::EQ, rhs, aij, row.transpose()});
            erase_row(P, row_idx);
            erase_col(P, j);
            changed = true;
            --j;
        }
        return changed;
    }

    bool doubleton_equation_elimination(LP& P) {
        // For each equality row with exactly 2 nonzeros (a1*x_j1 + a2*x_j2 = b),
        // substitute x_elim = (b - a_keep * x_keep) / a_elim, eliminating one variable.
        // Only active when allow_structural_changes && !non_destructive.
        if (!opt_.enable_doubleton_elim || !opt_.allow_structural_changes || opt_.non_destructive)
            return false;

        bool changed = false;
        for (int i = 0; i < (int)P.A.rows();) {
            if (P.sense[i] != RowSense::EQ) {
                ++i;
                continue;
            }

            // Check for exactly 2 nonzeros using sparse index
            const int nnz_i = (i < (int)row_nz_.size()) ? (int)row_nz_[i].size() : -1;
            if (nnz_i != 2) {
                if (nnz_i >= 0) {
                    ++i;
                    continue;
                }
                // Fall back to dense count
                int cnt = 0;
                for (int j = 0; j < (int)P.A.cols(); ++j)
                    if (std::abs(P.A(i, j)) > opt_.zero_tol && ++cnt > 2)
                        break;
                if (cnt != 2) {
                    ++i;
                    continue;
                }
            }

            // Get the two nonzero columns
            int j1, j2;
            if (i < (int)row_nz_.size()) {
                j1 = row_nz_[i][0];
                j2 = row_nz_[i][1];
            } else {
                j1 = j2 = -1;
                for (int j = 0; j < (int)P.A.cols(); ++j) {
                    if (std::abs(P.A(i, j)) > opt_.zero_tol) {
                        if (j1 < 0)
                            j1 = j;
                        else
                            j2 = j;
                    }
                }
            }
            if (j1 < 0 || j2 < 0) {
                ++i;
                continue;
            }

            const double a1 = P.A(i, j1), a2 = P.A(i, j2), bi = P.b(i);
            if (std::abs(a1) < opt_.zero_tol || std::abs(a2) < opt_.zero_tol) {
                ++i;
                continue;
            }

            // Choose which variable to eliminate: prefer the one with fewer column nonzeros
            // (less fill-in in the substitution step)
            const int nnz1 = (j1 < (int)col_nz_.size()) ? (int)col_nz_[j1].size() : 9999;
            const int nnz2 = (j2 < (int)col_nz_.size()) ? (int)col_nz_[j2].size() : 9999;
            int elim = (nnz1 <= nnz2) ? j1 : j2;
            int keep = (nnz1 <= nnz2) ? j2 : j1;
            double a_elim = P.A(i, elim), a_keep = P.A(i, keep);

            // Compute new bounds on 'keep' implied by 'elim's bounds via the equation:
            //   x_elim = (bi - a_keep * x_keep) / a_elim
            //   x_keep = (bi - a_elim * x_elim) / a_keep
            double new_l_keep = P.l(keep), new_u_keep = P.u(keep);
            if (is_finite(P.l(elim))) {
                // x_elim >= l_elim → x_keep bounded
                double val = (bi - a_elim * P.l(elim)) / a_keep;
                if (a_keep > 0)
                    new_u_keep = std::min(new_u_keep, val);
                else
                    new_l_keep = std::max(new_l_keep, val);
            }
            if (is_finite(P.u(elim))) {
                double val = (bi - a_elim * P.u(elim)) / a_keep;
                if (a_keep > 0)
                    new_l_keep = std::max(new_l_keep, val);
                else
                    new_u_keep = std::min(new_u_keep, val);
            }
            if (new_l_keep > new_u_keep + opt_.infeas_tol) {
                res_.proven_infeasible = true;
                return true;
            }

            // Record postsolve action BEFORE modifying
            res_.stack.emplace_back(
                ActDoubletonEq{elim, keep, a_elim, a_keep, bi, P.l(elim), P.u(elim), P.c(elim)});

            // Update objective: c(elim)*x_elim = c(elim)*(bi - a_keep*x_keep)/a_elim
            res_.obj_shift += P.c(elim) * bi / a_elim;
            P.c(keep) -= P.c(elim) * a_keep / a_elim;
            P.c(elim) = 0.0;

            // Substitute into all rows containing 'elim' (except row i itself)
            // Make a copy since col_nz_[elim] will be modified during iteration
            const std::vector<int> affected =
                (elim < (int)col_nz_.size()) ? col_nz_[elim] : std::vector<int>{};
            for (int k : affected) {
                if (k == i)
                    continue;
                const double a_k_elim = P.A(k, elim);
                if (std::abs(a_k_elim) <= opt_.zero_tol)
                    continue;

                // b(k) -= a_k_elim * bi / a_elim
                P.b(k) -= a_k_elim * bi / a_elim;

                // A(k, keep) += -a_k_elim * a_keep / a_elim
                const double old_val = P.A(k, keep);
                const double new_val = old_val - a_k_elim * a_keep / a_elim;
                P.A(k, keep) = new_val;
                if (std::abs(old_val) <= opt_.zero_tol && std::abs(new_val) > opt_.zero_tol) {
                    // New nonzero: add to index
                    add_entry_to_index(k, keep);
                } else if (std::abs(old_val) > opt_.zero_tol &&
                           std::abs(new_val) <= opt_.zero_tol) {
                    // Became zero: remove from index
                    zero_entry_in_index(k, keep);
                }

                // Zero out A(k, elim)
                P.A(k, elim) = 0.0;
                zero_entry_in_index(k, elim);
            }

            // Tighten bounds on 'keep'
            if (new_l_keep > P.l(keep) + opt_.zero_tol || new_u_keep < P.u(keep) - opt_.zero_tol) {
                res_.stack.emplace_back(ActTightenBound{keep, P.l(keep), P.u(keep)});
                if (new_l_keep > P.l(keep))
                    P.l(keep) = new_l_keep;
                if (new_u_keep < P.u(keep))
                    P.u(keep) = new_u_keep;
            }

            // Remove row i (doubleton row) and column elim
            erase_row(P, i); // updates sparse index
            erase_col(P, elim);
            // After erase_col, if elim < keep, 'keep' index shifted down by 1
            // erase_col already adjusts col_nz_ entries — nothing extra needed

            changed = true;
            // Don't increment i: row i was erased, so current i now points to next row
        }
        return changed;
    }

    bool redundancy_duplicate_rows(LP& P) {
        // Hash-based duplicate detection: O(m * avg_row_nnz) vs naïve O(m^2 * n)
        // Key: (sense, rhs_rounded, sorted nonzero (col, val_rounded) pairs)
        struct RowKey {
            int sense_int;
            int64_t rhs_hash;
            std::vector<std::pair<int, int64_t>> entries; // (col, coeff_hash)
            bool operator==(const RowKey& o) const {
                return sense_int == o.sense_int && rhs_hash == o.rhs_hash && entries == o.entries;
            }
        };
        // Map hash → first row index with that hash
        std::unordered_map<std::size_t, std::vector<int>> hash_to_rows;
        hash_to_rows.reserve((int)P.A.rows());

        // Scale hash values to reduce false collisions
        const double hash_scale = 1.0 / std::max(opt_.zero_tol, 1e-15);

        auto hash_double = [&](double v) -> int64_t {
            return static_cast<int64_t>(std::round(v * hash_scale));
        };

        // Build row hashes
        std::vector<RowKey> keys;
        keys.resize(P.A.rows());
        for (int i = 0; i < (int)P.A.rows(); ++i) {
            keys[i].sense_int = static_cast<int>(P.sense[i]);
            keys[i].rhs_hash = hash_double(P.b(i));
            if (i < (int)row_nz_.size()) {
                keys[i].entries.reserve(row_nz_[i].size());
                for (int j : row_nz_[i])
                    keys[i].entries.emplace_back(j, hash_double(P.A(i, j)));
            } else {
                for (int j = 0; j < (int)P.A.cols(); ++j) {
                    if (std::abs(P.A(i, j)) > opt_.zero_tol)
                        keys[i].entries.emplace_back(j, hash_double(P.A(i, j)));
                }
            }
            // Compute combined hash
            std::size_t h = std::hash<int>{}(keys[i].sense_int);
            h ^= std::hash<int64_t>{}(keys[i].rhs_hash) + 0x9e3779b9 + (h << 6) + (h >> 2);
            for (const auto& [c, v] : keys[i].entries) {
                h ^= std::hash<int>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
                h ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            hash_to_rows[h].push_back(i);
        }

        bool changed = false;
        std::vector<bool> deleted(P.A.rows(), false);
        for (auto& [h, row_list] : hash_to_rows) {
            if (row_list.size() < 2)
                continue;
            // Within candidates with same hash, do exact comparison
            for (int a = 0; a < (int)row_list.size(); ++a) {
                int i = row_list[a];
                if (deleted[i])
                    continue;
                for (int b = a + 1; b < (int)row_list.size(); ++b) {
                    int k = row_list[b];
                    if (deleted[k])
                        continue;
                    if (keys[i] == keys[k] && std::abs(P.b(i) - P.b(k)) <= opt_.infeas_tol &&
                        safe_abs_max(P.A.row(i) - P.A.row(k)) <= opt_.zero_tol) {
                        deleted[k] = true;
                    }
                }
            }
        }

        // Erase duplicate rows in reverse order to preserve indices
        for (int i = (int)P.A.rows() - 1; i >= 0; --i) {
            if (deleted[i]) {
                res_.stack.emplace_back(
                    ActRemoveRow{i, P.sense[i], P.b(i), P.A.row(i).transpose()});
                erase_row(P, i);
                changed = true;
            }
        }
        return changed;
    }

    void prune_zero_rows(LP& P) {
        for (int i = 0; i < (int)P.A.rows();) {
            if (safe_abs_max(P.A.row(i)) <= opt_.zero_tol) {
                res_.stack.emplace_back(
                    ActRemoveRow{i, P.sense[i], P.b(i), P.A.row(i).transpose()});
                erase_row(P, i);
            } else
                ++i;
        }
    }

    // ---------- maintenance ----------
    bool check_and_fix_bounds(LP& P) const {
        const int n = (int)P.A.cols();
        for (int j = 0; j < n; ++j) {
            if (is_finite(P.l(j)) && is_finite(P.u(j)) && P.l(j) > P.u(j) + opt_.infeas_tol)
                return false;
            if (is_finite(P.l(j)) && is_finite(P.u(j)) && P.l(j) > P.u(j)) {
                const double mid = 0.5 * (P.l(j) + P.u(j));
                P.l(j) = P.u(j) = mid;
            }
        }
        return true;
    }

    void erase_row(LP& P, int i) {
        update_index_erase_row(i);
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (i < m - 1) {
            P.A.block(i, 0, m - i - 1, n) = P.A.block(i + 1, 0, m - i - 1, n);
            P.b.segment(i, m - i - 1) = P.b.segment(i + 1, m - i - 1);
            for (int k = i; k < m - 1; ++k)
                P.sense[k] = P.sense[k + 1];
        }
        P.A.conservativeResize(m - 1, n);
        P.b.conservativeResize(m - 1);
        P.sense.pop_back();
        if ((int)res_.orig_row_index.size() == m)
            res_.orig_row_index.erase(res_.orig_row_index.begin() + i);
    }

    void erase_col(LP& P, int j) {
        update_index_erase_col(j);
        const int m = (int)P.A.rows(), n = (int)P.A.cols();
        if (j < n - 1) {
            P.A.block(0, j, m, n - j - 1) = P.A.block(0, j + 1, m, n - j - 1);
            P.c.segment(j, n - j - 1) = P.c.segment(j + 1, n - j - 1);
            P.l.segment(j, n - j - 1) = P.l.segment(j + 1, n - j - 1);
            P.u.segment(j, n - j - 1) = P.u.segment(j + 1, n - j - 1);
        }
        P.A.conservativeResize(m, n - 1);
        P.c.conservativeResize(n - 1);
        P.l.conservativeResize(n - 1);
        P.u.conservativeResize(n - 1);
        if ((int)res_.orig_col_index.size() == n)
            res_.orig_col_index.erase(res_.orig_col_index.begin() + j);
    }

    // ---------- undo ----------
    static void undo_action(const ActScaleRow&, Eigen::VectorXd&, double&) {}
    static void undo_action(const ActScaleCol& a, Eigen::VectorXd& x, double&) {
        if (a.j < (int)x.size() && std::isfinite(x(a.j)))
            x(a.j) /= a.scale;
    }
    static void undo_action(const ActRemoveRow&, Eigen::VectorXd&, double&) {}
    static void undo_action(const ActRowReduce&, Eigen::VectorXd&, double&) {}
    static void undo_action(const ActFixVar& a, Eigen::VectorXd& x, double&) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        x(a.j) = a.x_fix;
    }
    static void undo_action(const ActTightenBound&, Eigen::VectorXd&, double&) {}
    static void undo_action(const ActSingletonRowElim& a, Eigen::VectorXd& x, double&) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (a.sense == RowSense::EQ && std::abs(a.aij) > 1e-12) {
            double other = 0.0;
            for (int k = 0; k < (int)a.row.size(); ++k)
                if (k != a.j && k < (int)x.size() && std::isfinite(x(k)))
                    other += a.row(k) * x(k);
            x(a.j) = (a.rhs - other) / a.aij;
        } else if (!std::isfinite(x(a.j)))
            x(a.j) = 0.0;
    }
    static void undo_action(const ActSingletonColElim& a, Eigen::VectorXd& x, double&) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j)))
            x(a.j) = 0.0;
    }
    static void undo_action(const ActDualFix& a, Eigen::VectorXd& x, double&) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j)))
            x(a.j) = a.x_fix;
    }
    static void undo_action(const ActDoubletonEq& a, Eigen::VectorXd& x, double&) {
        // x_elim = (b_row - a_keep * x_keep) / a_elim
        // We need x[col_elim]; x[col_keep] must already be filled in.
        // Expand x if needed
        const int need = std::max(a.col_elim, a.col_keep) + 1;
        if (need > (int)x.size()) {
            Eigen::VectorXd xnew(need);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (std::abs(a.a_elim) > 1e-14) {
            const double x_keep = std::isfinite(x(a.col_keep)) ? x(a.col_keep) : 0.0;
            const double x_elim_val = (a.b_row - a.a_keep * x_keep) / a.a_elim;
            // Clamp to original bounds if needed
            double xv = x_elim_val;
            if (is_finite(a.old_l_elim))
                xv = std::max(xv, a.old_l_elim);
            if (is_finite(a.old_u_elim))
                xv = std::min(xv, a.old_u_elim);
            x(a.col_elim) = xv;
        } else if (!std::isfinite(x(a.col_elim))) {
            x(a.col_elim) = is_finite(a.old_l_elim)   ? a.old_l_elim
                            : is_finite(a.old_u_elim) ? a.old_u_elim
                                                      : 0.0;
        }
    }
    static void undo_action(const ActRemoveCol& a, Eigen::VectorXd& x, double&) {
        if (a.j >= (int)x.size()) {
            Eigen::VectorXd xnew(a.j + 1);
            xnew.setConstant(std::numeric_limits<double>::quiet_NaN());
            xnew.head(x.size()) = x;
            x.swap(xnew);
        }
        if (!std::isfinite(x(a.j))) {
            if (is_finite(a.l_j) && is_finite(a.u_j))
                x(a.j) = 0.5 * (a.l_j + a.u_j);
            else if (is_finite(a.l_j))
                x(a.j) = a.l_j;
            else if (is_finite(a.u_j))
                x(a.j) = a.u_j;
            else
                x(a.j) = 0.0;
        }
    }

    static void undo_dual_action(const ActScaleRow& a, Eigen::VectorXd& y) {
        if (a.i >= 0 && a.i < y.size() && std::abs(a.scale) > 0.0) {
            y(a.i) /= a.scale;
        }
    }
    static void undo_dual_action(const ActScaleCol&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActRemoveRow& a, Eigen::VectorXd& y) {
        const int old_m = std::max(a.i + 1, static_cast<int>(y.size()) + 1);
        Eigen::VectorXd ynew = Eigen::VectorXd::Zero(old_m);
        if (a.i > 0 && y.size() > 0) {
            ynew.head(std::min(a.i, static_cast<int>(y.size()))) =
                y.head(std::min(a.i, static_cast<int>(y.size())));
        }
        if (a.i < y.size()) {
            ynew.tail(y.size() - a.i) = y.tail(y.size() - a.i);
        }
        y.swap(ynew);
    }
    static void undo_dual_action(const ActRowReduce& a, Eigen::VectorXd& y) {
        if (a.U.cols() != y.size()) {
            y = Eigen::VectorXd::Constant(a.old_m, std::numeric_limits<double>::quiet_NaN());
            return;
        }
        y = a.U * y;
    }
    static void undo_dual_action(const ActFixVar&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActTightenBound&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActSingletonRowElim&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActSingletonColElim&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActDualFix&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActDoubletonEq&, Eigen::VectorXd&) {}
    static void undo_dual_action(const ActRemoveCol&, Eigen::VectorXd&) {}

  private:
    Options opt_;
    PresolveResult res_;
    double domprop_min_delta_{1e-6};
};

} // namespace presolve
