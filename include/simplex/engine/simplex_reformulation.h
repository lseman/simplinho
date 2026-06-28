#pragma once

// Out-of-line definitions for RevisedSimplex's sparse standard-form
// reformulation used by branch-and-bound node solves.  Declarations live in
// simplex.h; this header is included after the class is complete (same pattern
// as crash.h / phase1.h / postsolve.h).

#include "simplex/presolve/presolver.h"
#include "simplex/engine/simplex.h"

inline void RevisedSimplex::build_sparse_bound_only_cache_(const SparseMatrix& A,
                                                           const Eigen::VectorXd& b,
                                                           const Eigen::VectorXd& c,
                                                           const Eigen::VectorXd& l_use,
                                                           const Eigen::VectorXd& u_use) {
    sparse_bound_only_cache_.rows = static_cast<int>(A.rows());
    sparse_bound_only_cache_.cols = static_cast<int>(A.cols());
    sparse_bound_only_cache_.A_in = A;
    sparse_bound_only_cache_.A_in.makeCompressed();
    // Store the value pointer of the INPUT for identity fast-path in same_problem.
    // We take it from A (before copying) since A_in may be a different allocation.
    sparse_bound_only_cache_.cached_A_value_ptr = A.isCompressed() ? A.valuePtr() : nullptr;
    sparse_bound_only_cache_.b_in = b;
    sparse_bound_only_cache_.c_in = c;
    const int n = sparse_bound_only_cache_.cols;
    sparse_bound_only_cache_.has_lower.assign(n, 0);
    sparse_bound_only_cache_.has_upper.assign(n, 0);
    sparse_bound_only_cache_.fixed_bound.assign(n, 0);
    sparse_bound_only_cache_.single_y.assign(n, -1);
    sparse_bound_only_cache_.split_pos.assign(n, -1);
    sparse_bound_only_cache_.split_neg.assign(n, -1);
    sparse_bound_only_cache_.upper_slack.assign(n, -1);
    sparse_bound_only_cache_.m_eq = sparse_bound_only_cache_.rows;
    sparse_bound_only_cache_.nv = 0;
    sparse_bound_only_cache_.upper_rows = 0;

    for (int j = 0; j < n; ++j) {
        const bool has_l = std::isfinite(l_use(j));
        const bool has_u = std::isfinite(u_use(j));
        const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= opt_.tol;
        sparse_bound_only_cache_.has_lower[j] = static_cast<char>(has_l);
        sparse_bound_only_cache_.has_upper[j] = static_cast<char>(has_u);
        sparse_bound_only_cache_.fixed_bound[j] = static_cast<char>(fixed);
        if (fixed) {
            continue;
        }
        if (has_l || has_u) {
            sparse_bound_only_cache_.single_y[j] = sparse_bound_only_cache_.nv++;
            if (has_l && has_u)
                sparse_bound_only_cache_.upper_rows++;
        } else {
            sparse_bound_only_cache_.split_pos[j] = sparse_bound_only_cache_.nv++;
            sparse_bound_only_cache_.split_neg[j] = sparse_bound_only_cache_.nv++;
        }
    }
    sparse_bound_only_cache_.n_total =
        sparse_bound_only_cache_.nv + sparse_bound_only_cache_.upper_rows;
    sparse_bound_only_cache_.m_total =
        sparse_bound_only_cache_.m_eq + sparse_bound_only_cache_.upper_rows;

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(
        static_cast<std::size_t>(A.nonZeros() * 2 + sparse_bound_only_cache_.upper_rows * 2));
    sparse_bound_only_cache_.c_std = Eigen::VectorXd::Zero(sparse_bound_only_cache_.n_total);

    for (int j = 0; j < n; ++j) {
        const int y = sparse_bound_only_cache_.single_y[j];
        const int y_pos = sparse_bound_only_cache_.split_pos[j];
        const int y_neg = sparse_bound_only_cache_.split_neg[j];
        const bool has_l = static_cast<bool>(sparse_bound_only_cache_.has_lower[j]);
        const bool has_u = static_cast<bool>(sparse_bound_only_cache_.has_upper[j]);
        const bool fixed = static_cast<bool>(sparse_bound_only_cache_.fixed_bound[j]);
        if (fixed) {
            continue;
        }
        if (has_l && has_u) {
            sparse_bound_only_cache_.c_std[y] += c(j);
        } else if (has_l) {
            sparse_bound_only_cache_.c_std[y] += c(j);
        } else if (has_u) {
            sparse_bound_only_cache_.c_std[y] -= c(j);
        } else {
            sparse_bound_only_cache_.c_std[y_pos] += c(j);
            sparse_bound_only_cache_.c_std[y_neg] -= c(j);
        }
    }

    for (int j = 0; j < n; ++j) {
        const int y = sparse_bound_only_cache_.single_y[j];
        const int y_pos = sparse_bound_only_cache_.split_pos[j];
        const int y_neg = sparse_bound_only_cache_.split_neg[j];
        const bool has_l = static_cast<bool>(sparse_bound_only_cache_.has_lower[j]);
        const bool has_u = static_cast<bool>(sparse_bound_only_cache_.has_upper[j]);
        const bool fixed = static_cast<bool>(sparse_bound_only_cache_.fixed_bound[j]);
        const bool uses_single = (has_l || has_u) && !fixed;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            const int row = it.row();
            const double aij = it.value();
            if (fixed) {
                continue;
            } else if (uses_single) {
                const double sign = has_l ? 1.0 : -1.0;
                trips.emplace_back(row, y, sign * aij);
            } else {
                trips.emplace_back(row, y_pos, aij);
                trips.emplace_back(row, y_neg, -aij);
            }
        }
    }
    int upper_row = 0;
    for (int j = 0; j < n; ++j) {
        if (!static_cast<bool>(sparse_bound_only_cache_.has_lower[j]) ||
            !static_cast<bool>(sparse_bound_only_cache_.has_upper[j]) ||
            static_cast<bool>(sparse_bound_only_cache_.fixed_bound[j])) {
            continue;
        }
        const int slack = sparse_bound_only_cache_.nv + upper_row;
        sparse_bound_only_cache_.upper_slack[j] = slack;
        const int row = sparse_bound_only_cache_.m_eq + upper_row;
        const int y = sparse_bound_only_cache_.single_y[j];
        trips.emplace_back(row, y, 1.0);
        trips.emplace_back(row, slack, 1.0);
        upper_row++;
    }

    sparse_bound_only_cache_.A_std =
        SparseMatrix(sparse_bound_only_cache_.m_total, sparse_bound_only_cache_.n_total);
    if (!trips.empty())
        sparse_bound_only_cache_.A_std.setFromTriplets(trips.begin(), trips.end());
    sparse_bound_only_cache_.A_std.makeCompressed();
    sparse_bound_only_cache_.l_std = Eigen::VectorXd::Zero(sparse_bound_only_cache_.n_total);
    sparse_bound_only_cache_.u_std =
        Eigen::VectorXd::Constant(sparse_bound_only_cache_.n_total, presolve::inf());
    // Compute data_scale once so canonicalize_inactive_huge_bounds_ can skip the O(nnz) scan.
    {
        double max_abs = 0.0;
        for (int jj = 0; jj < A.outerSize(); ++jj)
            for (SparseMatrix::InnerIterator it(A, jj); it; ++it)
                if (const double av = std::abs(it.value()); av > max_abs)
                    max_abs = av;
        if (b.size() > 0)
            max_abs = std::max(max_abs, b.cwiseAbs().maxCoeff());
        sparse_bound_only_cache_.cached_data_scale = std::max(1.0, max_abs);
    }
    // Invalidate incremental RHS state and persistent reformulated solver —
    // next call will do a full RHS recompute, and the solver must be recreated
    // since the reformulated problem structure (A_std) changed.
    sparse_bound_only_cache_.b_std_scratch_valid = false;
    sparse_bound_only_cache_.reformulated_solver_cache.reset();
    sparse_bound_only_cache_.valid = true;
}

inline const Eigen::VectorXd&
RevisedSimplex::reconstruct_sparse_reformulated_rhs_(const Eigen::VectorXd& l_use,
                                                     const Eigen::VectorXd& u_use) const {
    const auto& cache = sparse_bound_only_cache_;
    Eigen::VectorXd& b_std = cache.b_std_scratch;

    if (cache.b_std_scratch_valid) {
        // Incremental update: scan bounds for changes and apply column-delta only.
        // In BnB, typically only 1 bound changes per node → O(n + changed_col_nnz)
        // vs O(total_nnz) for a full recompute.
        const Eigen::VectorXd& l_prev = cache.l_prev_scratch;
        const Eigen::VectorXd& u_prev = cache.u_prev_scratch;
        for (int j = 0; j < cache.cols; ++j) {
            const bool has_l = static_cast<bool>(cache.has_lower[j]);
            const bool has_u = static_cast<bool>(cache.has_upper[j]);
            const bool fixed = static_cast<bool>(cache.fixed_bound[j]);
            if (!has_l && !has_u)
                continue;

            const double old_shift = has_l ? l_prev(j) : u_prev(j);
            const double new_shift = has_l ? l_use(j) : u_use(j);
            const bool shift_changed = (old_shift != new_shift);
            const bool upper_changed =
                has_l && has_u && !fixed && (u_prev(j) != u_use(j) || l_prev(j) != l_use(j));

            if (shift_changed) {
                // delta = old_shift - new_shift; b_std -= A*shift so:
                // b_new = b_old + A_col * (old_shift - new_shift)
                const double delta = old_shift - new_shift;
                for (SparseMatrix::InnerIterator it(cache.A_in, j); it; ++it)
                    b_std(it.row()) += it.value() * delta;
            }
            if (upper_changed) {
                // Upper row: b_std[m_eq + (slack_col - nv)] = u(j) - l(j)
                const int row = cache.m_eq + (cache.upper_slack[j] - cache.nv);
                b_std(row) = u_use(j) - l_use(j);
            }
        }
    } else {
        // Full recompute (first call after cache build).
        b_std.setZero(cache.m_total);
        for (int j = 0; j < cache.cols; ++j) {
            const bool has_l = static_cast<bool>(cache.has_lower[j]);
            const bool has_u = static_cast<bool>(cache.has_upper[j]);
            if (!has_l && !has_u)
                continue;
            const double shift = has_l ? l_use(j) : u_use(j);
            for (SparseMatrix::InnerIterator it(cache.A_in, j); it; ++it)
                b_std(it.row()) -= it.value() * shift;
        }
        for (int i = 0; i < cache.m_eq; ++i)
            b_std(i) += cache.b_in(i);
        int upper_row = 0;
        for (int j = 0; j < cache.cols; ++j) {
            if (!cache.has_lower[j] || !cache.has_upper[j] || cache.fixed_bound[j])
                continue;
            b_std(cache.m_eq + upper_row) = u_use(j) - l_use(j);
            upper_row++;
        }
        cache.b_std_scratch_valid = true;
    }
    // Save current bounds for the next incremental update.
    cache.l_prev_scratch = l_use;
    cache.u_prev_scratch = u_use;
    return b_std;
}

inline RevisedSimplex::SanitizedBounds
RevisedSimplex::canonicalize_inactive_huge_bounds_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                                   const Eigen::VectorXd& l,
                                                   const Eigen::VectorXd& u,
                                                   double precomputed_data_scale) const {
    double data_scale = 1.0;
    if (precomputed_data_scale > 0.0) {
        // Fast path: caller already computed data_scale from the same A and b
        // (cached in SparseBoundOnlyCache after first build).  Skip O(nnz) scan.
        data_scale = precomputed_data_scale;
    } else if (A.nonZeros() > 0) {
        double max_abs = 0.0;
        for (int j = 0; j < A.outerSize(); ++j) {
            for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
                const double abs_val = std::abs(it.value());
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }
        }
        data_scale = std::max(data_scale, max_abs);
        if (b.size() > 0)
            data_scale = std::max(data_scale, b.cwiseAbs().maxCoeff());
    }
    const double huge_bound = 1e6 * data_scale;
    bool any_huge = false;
    for (int j = 0; j < A.cols(); ++j) {
        if ((std::isfinite(u(j)) && u(j) > huge_bound) ||
            (std::isfinite(l(j)) && l(j) < -huge_bound)) {
            any_huge = true;
            break;
        }
    }
    if (!any_huge)
        return SanitizedBounds{l, u, 0, 0};
    presolve::LP problem;
    problem.A = Eigen::MatrixXd(A);
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
