#pragma once

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "simplex/presolve/presolve_types.h"

namespace presolve {

struct SparseLP {
    Eigen::SparseMatrix<double, Eigen::ColMajor, int> A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
};

struct SparsePresolveResult {
    SparseLP reduced;
    std::vector<int> orig_col_index;
    std::vector<int> orig_row_index;
    Eigen::VectorXd row_scale;
    Eigen::VectorXd col_scale;
    bool proven_infeasible = false;
    bool proven_unbounded = false;
    int bound_updates = 0;
    int singleton_rows = 0;
    int zero_rows = 0;
    int zero_columns = 0;
    int row_reduced = 0;
    int rows_removed = 0;
    int row_scale_count = 0;
    int col_scale_count = 0;
    int passes = 0;
};

class SparsePresolver {
  public:
    struct Options {
        int max_passes = 4;
        double zero_tol = 1e-12;
        double infeas_tol = 1e-9;
        double min_delta = 1e-9;
        double svd_tol = 1e-8;
        double rrqr_pivot_tol = 1e-8;
        double rr_infeas_mult = 1e3;
        double cond_max = 1e10;
        int max_ruiz_iters = 10;
        bool enable_singleton_rows = true;
        bool enable_activity_tightening = true;
        bool enable_zero_columns = true;
        bool enable_row_scaling = true;
        bool enable_col_scaling = true;
        bool enable_row_reduce = false;
    };

    SparsePresolver() = default;
    explicit SparsePresolver(Options opt) : opt_(opt) {}

    SparsePresolveResult run(const SparseLP& in) const {
        validate_(in);
        SparsePresolveResult out;
        out.reduced = in;
        const int m0 = static_cast<int>(in.A.rows());
        const int n0 = static_cast<int>(in.A.cols());
        out.orig_row_index.resize(m0);
        out.orig_col_index.resize(n0);
        std::iota(out.orig_row_index.begin(), out.orig_row_index.end(), 0);
        std::iota(out.orig_col_index.begin(), out.orig_col_index.end(), 0);
        out.row_scale = Eigen::VectorXd::Ones(m0);
        out.col_scale = Eigen::VectorXd::Ones(n0);

        if (!check_bounds_(out.reduced)) {
            out.proven_infeasible = true;
            return out;
        }

        // ---- Scaling ----
        if (opt_.enable_row_scaling)
            scale_rows_sparse_(out.reduced, out);
        if (!check_bounds_(out.reduced)) {
            out.proven_infeasible = true;
            return out;
        }
        if (opt_.enable_col_scaling)
            scale_cols_sparse_(out.reduced, out);
        if (!check_bounds_(out.reduced)) {
            out.proven_infeasible = true;
            return out;
        }

        // ---- Row reduction (RRQR): dense QR on sparse→dense→sparse ----
        if (opt_.enable_row_reduce && m0 > 0) {
            SparseLP pre_qr = out.reduced;
            RowIndex pre_rows = build_row_index_(pre_qr.A);
            inspect_zero_rows_(pre_qr, pre_rows, out);
            if (out.proven_infeasible)
                return out;
            if (opt_.enable_zero_columns)
                inspect_zero_columns_(pre_qr, out);
            if (out.proven_unbounded || out.proven_infeasible)
                return out;
            if (!row_reduce_rrqr_sparse_(pre_qr, out))
                return out;
        }

        // Rebuild sparse index after scaling/reduction
        RowIndex rows = build_row_index_(out.reduced.A);

        bool changed = true;
        while (changed && out.passes < opt_.max_passes) {
            changed = false;
            ++out.passes;

            if (opt_.enable_singleton_rows) {
                changed |= tighten_singleton_rows_(out.reduced, rows, out);
                if (out.proven_infeasible)
                    return out;
            }
            if (opt_.enable_activity_tightening) {
                changed |= tighten_by_row_activity_(out.reduced, rows, out);
                if (out.proven_infeasible)
                    return out;
            }
        }

        return out;
    }

  private:
    struct RowEntry {
        int col = -1;
        double value = 0.0;
    };
    using RowIndex = std::vector<std::vector<RowEntry>>;

    Options opt_;

    static bool finite_(double v) { return std::isfinite(v); }

    void validate_(const SparseLP& lp) const {
        const int m = static_cast<int>(lp.A.rows());
        const int n = static_cast<int>(lp.A.cols());
        if (lp.b.size() != m)
            throw std::invalid_argument("sparse presolve: b size mismatch");
        if (lp.c.size() != n || lp.l.size() != n || lp.u.size() != n)
            throw std::invalid_argument("sparse presolve: c/l/u size mismatch");
    }

    bool check_bounds_(const SparseLP& lp) const {
        for (int j = 0; j < lp.l.size(); ++j) {
            if (finite_(lp.l(j)) && finite_(lp.u(j)) && lp.l(j) > lp.u(j) + opt_.infeas_tol)
                return false;
        }
        return true;
    }

    RowIndex build_row_index_(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A) const {
        RowIndex rows(static_cast<std::size_t>(A.rows()));
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, j); it;
                 ++it) {
                if (std::abs(it.value()) <= opt_.zero_tol)
                    continue;
                rows[static_cast<std::size_t>(it.row())].push_back({j, it.value()});
            }
        }
        return rows;
    }

    void inspect_zero_rows_(const SparseLP& lp, const RowIndex& rows,
                            SparsePresolveResult& out) const {
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            if (!rows[static_cast<std::size_t>(i)].empty())
                continue;
            ++out.zero_rows;
            if (std::abs(lp.b(i)) > opt_.infeas_tol) {
                out.proven_infeasible = true;
                return;
            }
        }
    }

    void inspect_zero_columns_(SparseLP& lp, SparsePresolveResult& out) const {
        std::vector<char> has_nz(static_cast<std::size_t>(lp.A.cols()), 0);
        for (int j = 0; j < lp.A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it) {
                if (std::abs(it.value()) > opt_.zero_tol) {
                    has_nz[static_cast<std::size_t>(j)] = 1;
                    break;
                }
            }
        }

        for (int j = 0; j < static_cast<int>(has_nz.size()); ++j) {
            if (has_nz[static_cast<std::size_t>(j)])
                continue;
            ++out.zero_columns;
            if (lp.c(j) > opt_.zero_tol) {
                if (!finite_(lp.l(j))) {
                    out.proven_unbounded = true;
                    return;
                }
                tighten_upper_(lp, j, lp.l(j), out);
            } else if (lp.c(j) < -opt_.zero_tol) {
                if (!finite_(lp.u(j))) {
                    out.proven_unbounded = true;
                    return;
                }
                tighten_lower_(lp, j, lp.u(j), out);
            }
        }
    }

    bool tighten_lower_(SparseLP& lp, int j, double value, SparsePresolveResult& out) const {
        if (!finite_(value))
            return false;
        if (finite_(lp.u(j)) && value > lp.u(j) + opt_.infeas_tol) {
            out.proven_infeasible = true;
            return false;
        }
        if (!finite_(lp.l(j)) || value > lp.l(j) + opt_.min_delta) {
            lp.l(j) = value;
            ++out.bound_updates;
            return true;
        }
        return false;
    }

    bool tighten_upper_(SparseLP& lp, int j, double value, SparsePresolveResult& out) const {
        if (!finite_(value))
            return false;
        if (finite_(lp.l(j)) && value < lp.l(j) - opt_.infeas_tol) {
            out.proven_infeasible = true;
            return false;
        }
        if (!finite_(lp.u(j)) || value < lp.u(j) - opt_.min_delta) {
            lp.u(j) = value;
            ++out.bound_updates;
            return true;
        }
        return false;
    }

    bool tighten_singleton_rows_(SparseLP& lp, const RowIndex& rows,
                                 SparsePresolveResult& out) const {
        bool changed = false;
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            const auto& row = rows[static_cast<std::size_t>(i)];
            if (row.size() != 1)
                continue;
            const int j = row.front().col;
            const double a = row.front().value;
            if (std::abs(a) <= opt_.zero_tol)
                continue;
            const double value = lp.b(i) / a;
            ++out.singleton_rows;
            changed |= tighten_lower_(lp, j, value, out);
            if (out.proven_infeasible)
                return changed;
            changed |= tighten_upper_(lp, j, value, out);
            if (out.proven_infeasible)
                return changed;
        }
        return changed;
    }

    bool row_activity_excluding_(const SparseLP& lp, const std::vector<RowEntry>& row, int skip_col,
                                 double& min_activity, double& max_activity) const {
        min_activity = 0.0;
        max_activity = 0.0;
        for (const RowEntry& entry : row) {
            if (entry.col == skip_col)
                continue;
            const double a = entry.value;
            const int j = entry.col;
            if (a >= 0.0) {
                if (!finite_(lp.l(j)) || !finite_(lp.u(j)))
                    return false;
                min_activity += a * lp.l(j);
                max_activity += a * lp.u(j);
            } else {
                if (!finite_(lp.l(j)) || !finite_(lp.u(j)))
                    return false;
                min_activity += a * lp.u(j);
                max_activity += a * lp.l(j);
            }
        }
        return true;
    }

    bool tighten_by_row_activity_(SparseLP& lp, const RowIndex& rows,
                                  SparsePresolveResult& out) const {
        bool changed = false;
        for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
            const auto& row = rows[static_cast<std::size_t>(i)];
            if (row.size() <= 1)
                continue;
            for (const RowEntry& entry : row) {
                double other_min = 0.0;
                double other_max = 0.0;
                if (!row_activity_excluding_(lp, row, entry.col, other_min, other_max))
                    continue;
                const double a = entry.value;
                if (a > opt_.zero_tol) {
                    changed |= tighten_lower_(lp, entry.col, (lp.b(i) - other_max) / a, out);
                    if (out.proven_infeasible)
                        return changed;
                    changed |= tighten_upper_(lp, entry.col, (lp.b(i) - other_min) / a, out);
                } else if (a < -opt_.zero_tol) {
                    changed |= tighten_lower_(lp, entry.col, (lp.b(i) - other_min) / a, out);
                    if (out.proven_infeasible)
                        return changed;
                    changed |= tighten_upper_(lp, entry.col, (lp.b(i) - other_max) / a, out);
                }
                if (out.proven_infeasible)
                    return changed;
            }
        }
        return changed;
    }

    // ---- Scaling helpers ----
    void scale_rows_sparse_(SparseLP& lp, SparsePresolveResult& out) const {
        const int m = static_cast<int>(lp.A.rows());
        Eigen::VectorXd row_max = Eigen::VectorXd::Zero(m);
        for (int j = 0; j < lp.A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it)
                row_max(it.row()) = std::max(row_max(it.row()), std::abs(it.value()));
        }
        for (int i = 0; i < m; ++i) {
            const double mx = row_max(i);
            if (mx <= opt_.zero_tol || !std::isfinite(mx))
                continue;
            const double s = nearest_power_of_two_magnitude(mx);
            if (nearly_zero(s - mx, opt_.zero_tol * 100.0))
                continue;
            lp.b(i) /= s;
            out.row_scale(i) *= s;
            ++out.row_scale_count;
        }
        for (int j = 0; j < lp.A.outerSize(); ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it)
                it.valueRef() /= out.row_scale(it.row());
        }
    }

    void scale_cols_sparse_(SparseLP& lp, SparsePresolveResult& out) const {
        const int n = static_cast<int>(lp.A.cols());
        for (int j = 0; j < n; ++j) {
            double mx = 0.0;
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it) {
                const double v = std::abs(it.value());
                if (v > mx)
                    mx = v;
            }
            if (mx <= opt_.zero_tol || !std::isfinite(mx))
                continue;
            double s = nearest_power_of_two_magnitude(mx);
            if (nearly_zero(s - mx, opt_.zero_tol * 100.0))
                continue;
            // Scale column j: divide entries and c by s; multiply bounds by s
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it) {
                it.valueRef() /= s;
            }
            lp.c(j) /= s;
            if (std::isfinite(lp.l(j)))
                lp.l(j) *= s;
            if (std::isfinite(lp.u(j)))
                lp.u(j) *= s;
            out.col_scale(j) *= s;
            ++out.col_scale_count;
        }
    }

    // ---- RRQR row reduction (sparse-aware) ----
    // Converts sparse A to dense, runs ColPivHouseholderQR, determines rank,
    // computes projection U_r, reduces A and b, converts back to sparse.
    bool row_reduce_rrqr_sparse_(SparseLP& lp, SparsePresolveResult& out) const {
        const int m = static_cast<int>(lp.A.rows());
        const int n = static_cast<int>(lp.A.cols());
        if (m == 0)
            return true;

        // Convert sparse → dense for QR
        Eigen::MatrixXd Ad = Eigen::MatrixXd::Zero(m, n);
        for (int j = 0; j < n; ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(lp.A, j); it;
                 ++it) {
                Ad(it.row(), j) = it.value();
            }
        }

        // Ruiz balance
        if (opt_.max_ruiz_iters > 0) {
            auto [Dr, Dc] = ruiz_balance_sparse(Ad, opt_.max_ruiz_iters);
            for (int i = 0; i < m; ++i)
                lp.b(i) /= Dr(i);
            (void)Dc;
        }

        // ColPivHouseholderQR
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qrA(Ad);
        if (qrA.info() != Eigen::Success)
            return true;

        // Determine rank from R diagonal
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
            const double ainfn = Ad.cwiseAbs().rowwise().sum().maxCoeff();
            if (ainfn > 1e3 * eps)
                return true;
            if (lp.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                // All rows dependent, rhs compatible => no constraints
                lp.A.resize(0, n);
                lp.b.resize(0);
                out.proven_infeasible = false;
                out.rows_removed = m;
                out.row_reduced = m;
                out.orig_row_index.clear();
                return true;
            } else {
                out.proven_infeasible = true;
                return false;
            }
        }

        // Condition check on R(1:r, 1:r)
        auto cond_ok = [&](int rr) -> bool {
            if (rr <= 0)
                return false;
            Eigen::MatrixXd R11 = R.topLeftCorner(rr, rr);
            const double kappa = cond2_estimate_sparse(R11);
            return (kappa <= opt_.cond_max || !std::isfinite(kappa));
        };
        int r_try = r;
        while (r_try <= (int)diagR.size()) {
            if (!cond_ok(r_try))
                break;
            ++r_try;
        }
        r = std::max(1, std::min(r_try - 1, (int)diagR.size()));

        // Compute U_r: first r columns of Q from Householder reflections
        Eigen::MatrixXd Ur = Eigen::MatrixXd::Identity(m, r);
        Ur = qrA.householderQ() * Ur;

        // Check residual
        Eigen::VectorXd resid = lp.b - Ur * (Ur.transpose() * lp.b);
        const double allowed =
            opt_.rr_infeas_mult * opt_.infeas_tol * std::max(1.0, lp.b.lpNorm<Eigen::Infinity>());
        if (resid.lpNorm<Eigen::Infinity>() > allowed) {
            // Fallback: SVD-based reduction
            return row_reduce_svd_sparse_(lp, out, r, Ur, Ad);
        }

        // Reduce: Atil = Ur^T * Ad, btil = Ur^T * b
        Eigen::MatrixXd Adense_red = Ur.transpose() * Ad;
        Eigen::VectorXd b_red = Ur.transpose() * lp.b;

        // Convert back to sparse
        Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_red(m, n);
        A_red.setZero();
        A_red = Adense_red.sparseView();

        lp.A = std::move(A_red);
        lp.b = std::move(b_red);

        out.row_reduced = r;
        out.rows_removed = m - r;
        out.orig_row_index.resize(r);
        std::iota(out.orig_row_index.begin(), out.orig_row_index.end(), 0);

        return true;
    }

    // Fallback: SVD-based row reduction (called when QR residual too large)
    bool row_reduce_svd_sparse_(SparseLP& lp, SparsePresolveResult& out, int r_hint,
                                const Eigen::MatrixXd& /*Ur*/, const Eigen::MatrixXd& Ad) const {
        const int m = static_cast<int>(lp.A.rows());
        const int n = static_cast<int>(lp.A.cols());
        using SVD = Eigen::BDCSVD<Eigen::MatrixXd>;

        SVD svd(Ad, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& S = svd.singularValues();
        const double eps = std::numeric_limits<double>::epsilon();
        const double smax = (S.size() > 0 ? S(0) : 0.0);
        const double thr = std::max(opt_.svd_tol * smax, 100.0 * eps * smax);

        int r = 0;
        for (int i = 0; i < S.size(); ++i)
            if (S(i) > thr)
                ++r;

        if (r == 0) {
            const double ainfn = Ad.cwiseAbs().rowwise().sum().maxCoeff();
            if (ainfn > 1e3 * eps)
                return true;
            if (lp.b.lpNorm<Eigen::Infinity>() <= opt_.infeas_tol) {
                lp.A.resize(0, n);
                lp.b.resize(0);
                out.rows_removed = m;
                out.row_reduced = 0;
                out.orig_row_index.clear();
                return true;
            } else {
                out.proven_infeasible = true;
                return false;
            }
        }

        const Eigen::MatrixXd Ur = svd.matrixU().leftCols(r);
        const Eigen::VectorXd resid = lp.b - Ur * (Ur.transpose() * lp.b);
        const double allowed =
            opt_.rr_infeas_mult * opt_.infeas_tol * std::max(1.0, lp.b.lpNorm<Eigen::Infinity>());
        if (resid.lpNorm<Eigen::Infinity>() > allowed) {
            out.proven_infeasible = true;
            return false;
        }

        // Reduce
        Eigen::MatrixXd Ad_red = Ur.transpose() * Ad;
        Eigen::VectorXd b_red = Ur.transpose() * lp.b;

        Eigen::SparseMatrix<double, Eigen::ColMajor, int> A_red(m, n);
        A_red.setZero();
        A_red = Ad_red.sparseView();

        lp.A = std::move(A_red);
        lp.b = std::move(b_red);
        out.row_reduced = r;
        out.rows_removed = m - r;
        out.orig_row_index.resize(r);
        std::iota(out.orig_row_index.begin(), out.orig_row_index.end(), 0);
        return true;
    }

    // Ruiz balance for dense matrix (sparse-preserving factors)
    static std::pair<Eigen::VectorXd, Eigen::VectorXd>
    ruiz_balance_sparse(Eigen::MatrixXd& A, int iters, double floor = 1e-12) {
        const int m = static_cast<int>(A.rows()), n = static_cast<int>(A.cols());
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

    // Condition number upper bound estimate (dense only)
    static double cond2_estimate_sparse(const Eigen::MatrixXd& R11) {
        if (R11.size() == 0)
            return 0.0;
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(R11, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto s = svd.singularValues();
        const double smax = s(0);
        const double smin = s.tail(1)(0);
        return (smin > 0.0) ? (smax / smin) : std::numeric_limits<double>::infinity();
    }
};

} // namespace presolve
