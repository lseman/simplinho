#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

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
    bool proven_infeasible = false;
    bool proven_unbounded = false;
    int bound_updates = 0;
    int singleton_rows = 0;
    int zero_rows = 0;
    int zero_columns = 0;
    int passes = 0;
};

class SparsePresolver {
  public:
    struct Options {
        int max_passes = 4;
        double zero_tol = 1e-12;
        double infeas_tol = 1e-9;
        double min_delta = 1e-9;
        bool enable_singleton_rows = true;
        bool enable_activity_tightening = true;
        bool enable_zero_columns = true;
    };

    SparsePresolver() = default;
    explicit SparsePresolver(Options opt) : opt_(opt) {}

    SparsePresolveResult run(const SparseLP& in) const {
        validate_(in);
        SparsePresolveResult out;
        out.reduced = in;
        const int m = static_cast<int>(in.A.rows());
        const int n = static_cast<int>(in.A.cols());
        out.orig_row_index.resize(m);
        out.orig_col_index.resize(n);
        std::iota(out.orig_row_index.begin(), out.orig_row_index.end(), 0);
        std::iota(out.orig_col_index.begin(), out.orig_col_index.end(), 0);

        if (!check_bounds_(out.reduced)) {
            out.proven_infeasible = true;
            return out;
        }

        RowIndex rows = build_row_index_(out.reduced.A);
        inspect_zero_rows_(out.reduced, rows, out);
        if (out.proven_infeasible)
            return out;
        if (opt_.enable_zero_columns) {
            inspect_zero_columns_(out.reduced, out);
            if (out.proven_unbounded || out.proven_infeasible)
                return out;
        }

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
};

} // namespace presolve
