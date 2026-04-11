#pragma once

inline Eigen::MatrixXd RevisedSimplex::dense_basis_copy_(const SparseMatrix& A,
                                                         const std::vector<int>& basis) {
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(A.rows(), basis.size());
    for (int k = 0; k < static_cast<int>(basis.size()); ++k) {
        const int j = basis[k];
        if (j < 0 || j >= A.cols())
            continue;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            B(it.row(), k) = it.value();
        }
    }
    return B;
}

inline RevisedSimplex::SparseMatrix
RevisedSimplex::sparse_basis_copy_(const SparseMatrix& A, const std::vector<int>& basis) {
    std::size_t reserve_nnz = 0;
    if (A.isCompressed()) {
        const int* outer = A.outerIndexPtr();
        for (int j : basis) {
            if (j < 0 || j >= A.cols())
                continue;
            reserve_nnz += static_cast<std::size_t>(outer[j + 1] - outer[j]);
        }
    } else {
        reserve_nnz = static_cast<std::size_t>(std::max<int>(1, basis.size() * 8));
    }

    SparseMatrix B(A.rows(), basis.size());
    B.reserve(reserve_nnz);
    for (int k = 0; k < static_cast<int>(basis.size()); ++k) {
        const int j = basis[k];
        if (j < 0 || j >= A.cols())
            continue;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            B.insert(it.row(), k) = it.value();
        }
    }
    B.makeCompressed();
    return B;
}

inline bool RevisedSimplex::sparse_basis_has_full_rank_(const SparseMatrix& A,
                                                        const std::vector<int>& basis) {
    const int m = static_cast<int>(A.rows());
    if (static_cast<int>(basis.size()) != m)
        return false;
    if (m == 0)
        return true;
    SparseMatrix B = sparse_basis_copy_(A, basis);
    Eigen::SparseLU<SparseMatrix> lu;
    lu.analyzePattern(B);
    lu.factorize(B);
    return lu.info() == Eigen::Success;
}

inline Eigen::VectorXd RevisedSimplex::sparse_solveT_from_lu_(
    const SparseMatrix& B, const Eigen::SparseLU<SparseMatrix>& lu_B, const Eigen::VectorXd& c) {
    (void)lu_B;
    SparseMatrix BT = B.transpose();
    Eigen::SparseLU<SparseMatrix> lu_BT;
    lu_BT.analyzePattern(BT);
    lu_BT.factorize(BT);
    if (lu_BT.info() != Eigen::Success)
        return Eigen::VectorXd();
    return lu_BT.solve(c);
}
