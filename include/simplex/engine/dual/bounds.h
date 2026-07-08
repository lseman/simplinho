#pragma once

#include "simplex/engine/dual/ratio.h"

namespace simplex::engine {

class DualBoundModel : public DualRatioTest, public MatrixUtilities {
  public:
    using BoundView = DualRatioTest::BoundView;
    using SparseRowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;

    static BoundView default_bound_view(int j, const Eigen::VectorXd& l,
                                        const Eigen::VectorXd& u) {
        const bool has_l = j < l.size() && std::isfinite(l(j));
        const bool has_u = j < u.size() && std::isfinite(u(j));
        if (has_l && has_u && std::abs(u(j) - l(j)) <= 1e-12)
            return BoundView::Fixed;
        return has_u && !has_l ? BoundView::Upper : BoundView::Lower;
    }

    static double bound_anchor(BoundView view, int j, const Eigen::VectorXd& l,
                               const Eigen::VectorXd& u) {
        if (view == BoundView::Upper)
            return j < u.size() && std::isfinite(u(j)) ? u(j) : 0.0;
        return j < l.size() && std::isfinite(l(j)) ? l(j) : 0.0;
    }

    static int view_sign(BoundView view) { return view == BoundView::Upper ? -1 : 1; }

    template <class MatrixType>
    static Eigen::VectorXd transformed_rhs(const MatrixType& A, const std::vector<BoundView>& view,
                                           const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(A.rows());
        for (int j = 0; j < A.cols(); ++j) {
            const double anchor = bound_anchor(view[j], j, l, u);
            if (anchor == 0.0)
                continue;
            if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                for (typename MatrixType::InnerIterator it(A, j); it; ++it)
                    rhs(it.row()) += it.value() * anchor;
            } else {
                rhs.noalias() += A.col(j) * anchor;
            }
        }
        return rhs;
    }

    static void scale_column(Eigen::MatrixXd& A, int j, double scale) {
        if (scale != 1.0)
            A.col(j) *= scale;
    }

    static void scale_column(RevisedSimplex::SparseMatrix& A, int j, double scale) {
        if (scale == 1.0)
            return;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it)
            it.valueRef() *= scale;
    }

    static SparseRowMatrix rowwise_copy(const RevisedSimplex::SparseMatrix& A) {
        return SparseRowMatrix(A);
    }

    static void scale_rowwise_column(SparseRowMatrix& row_matrix,
                                     const RevisedSimplex::SparseMatrix& column_matrix, int j,
                                     double scale) {
        if (scale == 1.0)
            return;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(column_matrix, j); it; ++it)
            row_matrix.coeffRef(it.row(), j) *= scale;
    }

    template <class MatrixType>
    static MatrixType signed_matrix_copy(const MatrixType& A,
                                         const std::vector<BoundView>& view) {
        MatrixType out = A;
        for (int j = 0; j < out.cols(); ++j)
            if (view_sign(view[j]) < 0)
                scale_column(out, j, -1.0);
        return out;
    }

    static Eigen::VectorXd assemble_transformed_primal(int n, const std::vector<int>& basis,
                                                       const Eigen::VectorXd& yB,
                                                       const Eigen::VectorXd& l,
                                                       const Eigen::VectorXd& u,
                                                       const std::vector<BoundView>& view) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> in_basis(n, 0);
        for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
            const int j = basis[i];
            if (j < 0 || j >= n)
                continue;
            in_basis[j] = 1;
            const double anchor = bound_anchor(view[j], j, l, u);
            const double basic_value = i < yB.size() ? yB(i) : 0.0;
            x(j) = anchor + view_sign(view[j]) * basic_value;
        }
        for (int j = 0; j < n; ++j)
            if (!in_basis[j])
                x(j) = bound_anchor(view[j], j, l, u);
        for (int j = 0; j < x.size(); ++j)
            if (std::abs(x(j)) <= 1e-12)
                x(j) = 0.0;
        return x;
    }
};

} // namespace simplex::engine
