#pragma once

#include "simplex/core/hvector.h"

namespace simplex::engine {

class BoundUtilities {
  public:
    static double bound_range(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        const bool has_l = j < l.size() && std::isfinite(l(j));
        const bool has_u = j < u.size() && std::isfinite(u(j));
        return has_l && has_u ? std::max(0.0, u(j) - l(j))
                              : std::numeric_limits<double>::infinity();
    }

    static double nonbasic_value_(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                  bool upper) {
        const bool has_l = j < l.size() && std::isfinite(l(j));
        const bool has_u = j < u.size() && std::isfinite(u(j));
        if (upper && has_u)
            return u(j);
        if (has_l)
            return l(j);
        if (has_u)
            return u(j);
        return 0.0;
    }

    static bool clamp_basic_solution_to_bounds_(Eigen::VectorXd& xB,
                                                const std::vector<int>& basis,
                                                const Eigen::VectorXd& l,
                                                const Eigen::VectorXd& u, double tol) {
        for (int i = 0; i < xB.size(); ++i) {
            const int j = basis[i];
            const double lo = j >= 0 && j < l.size() && std::isfinite(l(j))
                                  ? l(j)
                                  : -std::numeric_limits<double>::infinity();
            const double hi = j >= 0 && j < u.size() && std::isfinite(u(j))
                                  ? u(j)
                                  : std::numeric_limits<double>::infinity();
            if (xB(i) < lo - tol || xB(i) > hi + tol)
                return false;
            xB(i) = std::clamp(xB(i), lo, hi);
        }
        return true;
    }
};

class MatrixUtilities {
  public:
    template <class MatrixType>
    static void axpy_col_(const MatrixType& A, int j, double coefficient,
                          Eigen::VectorXd& target) {
        if (coefficient == 0.0)
            return;
        if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
            for (typename MatrixType::InnerIterator it(A, j); it; ++it)
                target(it.row()) += coefficient * it.value();
        } else {
            target.noalias() += coefficient * A.col(j);
        }
    }

    static double column_dot(const Eigen::MatrixXd& A, int j, const Eigen::VectorXd& v) {
        return A.col(j).dot(v);
    }

    static double column_dot(const RevisedSimplex::SparseMatrix& A, int j,
                             const Eigen::VectorXd& v) {
        double sum = 0.0;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it)
            sum += it.value() * v(it.row());
        return sum;
    }

    static Eigen::VectorXd dense_column(const Eigen::MatrixXd& A, int j) { return A.col(j); }

    static Eigen::VectorXd dense_column(const RevisedSimplex::SparseMatrix& A, int j) {
        Eigen::VectorXd out = Eigen::VectorXd::Zero(A.rows());
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it)
            out(it.row()) = it.value();
        return out;
    }

    static int count_effective_nonzeros(const Eigen::VectorXd& v, double tol = 0.0) {
        int count = 0;
        for (int i = 0; i < v.size(); ++i)
            count += std::abs(v(i)) > tol;
        return count;
    }

    static int count_effective_nonzeros(const HVector& v, double tol = 0.0) {
        if (v.has_pattern()) {
            int count = 0;
            for (int p = 0; p < v.count; ++p)
                count += std::abs(v.value(v.index[p])) > tol;
            return count;
        }
        return count_effective_nonzeros(v.value, tol);
    }

    static double vector_density(const Eigen::VectorXd& v, double tol = 0.0) {
        return v.size() > 0 ? static_cast<double>(count_effective_nonzeros(v, tol)) / v.size()
                            : 0.0;
    }

    static double vector_density(const HVector& v, double tol = 0.0) {
        return v.value.size() > 0
                   ? static_cast<double>(count_effective_nonzeros(v, tol)) / v.value.size()
                   : 0.0;
    }
};

} // namespace simplex::engine
