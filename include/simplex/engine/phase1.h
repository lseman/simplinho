#pragma once

namespace simplex_detail {
inline bool basic_values_within_bounds_(const Eigen::VectorXd &xB,
                                        const std::vector<int> &basis,
                                        const Eigen::VectorXd &l,
                                        const Eigen::VectorXd &u, double tol) {
  for (int i = 0; i < (int)basis.size(); ++i) {
    const int j = basis[i];
    const double lo = (j >= 0 && j < l.size() && std::isfinite(l(j))) ? l(j)
                                                                       : -std::numeric_limits<double>::infinity();
    const double hi = (j >= 0 && j < u.size() && std::isfinite(u(j))) ? u(j)
                                                                       : std::numeric_limits<double>::infinity();
    if (xB(i) < lo - tol || xB(i) > hi + tol)
      return false;
  }
  return true;
}
}  // namespace simplex_detail

inline bool RevisedSimplex::basis_is_primal_feasible_(
    const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
    const std::vector<int> &basis, const Eigen::VectorXd &l,
    const Eigen::VectorXd &u, double tol) {
  const int m = static_cast<int>(A.rows());
  if ((int)basis.size() != m)
    return false;
  if (m == 0)
    return true;
  const Eigen::MatrixXd B =
      A(Eigen::all, Eigen::VectorXi::Map(basis.data(), m));
  Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
  if (lu.rank() != m || !lu.isInvertible())
    return false;
  const Eigen::VectorXd xB = lu.solve(b);
  return xB.allFinite() &&
         simplex_detail::basic_values_within_bounds_(xB, basis, l, u, tol);
}

inline bool RevisedSimplex::basis_is_primal_feasible_(
    const SparseMatrix &A, const Eigen::VectorXd &b,
    const std::vector<int> &basis, const Eigen::VectorXd &l,
    const Eigen::VectorXd &u, double tol) {
  const int m = static_cast<int>(A.rows());
  if ((int)basis.size() != m)
    return false;
  if (m == 0)
    return true;
  SparseMatrix B = sparse_basis_copy_(A, basis);
  Eigen::SparseLU<SparseMatrix> lu;
  lu.analyzePattern(B);
  lu.factorize(B);
  if (lu.info() != Eigen::Success)
    return false;
  const Eigen::VectorXd xB = lu.solve(b);
  if (lu.info() != Eigen::Success)
    return false;
  return xB.allFinite() &&
         simplex_detail::basic_values_within_bounds_(xB, basis, l, u, tol);
}

inline std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                  std::vector<int>, std::size_t, int>
RevisedSimplex::make_phase1_(const Eigen::MatrixXd &A,
                             const Eigen::VectorXd &b) {
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());

  Eigen::MatrixXd A1 = A;
  Eigen::VectorXd b1 = b;
  for (int i = 0; i < m; ++i) {
    if (b1(i) < 0) {
      A1.row(i) *= -1.0;
      b1(i) *= -1.0;
    }
  }

  Eigen::MatrixXd A_aux(m, n + m);
  A_aux.leftCols(n) = A1;
  A_aux.rightCols(m) = Eigen::MatrixXd::Identity(m, m);

  Eigen::VectorXd c_aux(n + m);
  c_aux.setZero();
  c_aux.tail(m).setOnes();

  std::vector<int> basis(m);
  std::iota(basis.begin(), basis.end(), n);

  return {A_aux, b1, c_aux, basis, static_cast<std::size_t>(n), m};
}

inline std::tuple<RevisedSimplex::SparseMatrix, Eigen::VectorXd,
                  Eigen::VectorXd, std::vector<int>, std::size_t, int>
RevisedSimplex::make_phase1_(const SparseMatrix &A, const Eigen::VectorXd &b) {
  const int m = static_cast<int>(A.rows());
  const int n = static_cast<int>(A.cols());

  Eigen::VectorXd b1 = b;
  Eigen::VectorXd row_sign = Eigen::VectorXd::Ones(m);
  for (int i = 0; i < m; ++i) {
    if (b1(i) < 0.0) {
      b1(i) *= -1.0;
      row_sign(i) = -1.0;
    }
  }

  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(A.nonZeros() + m));
  for (int j = 0; j < A.outerSize(); ++j) {
    for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
      trips.emplace_back(it.row(), it.col(), row_sign(it.row()) * it.value());
    }
  }
  for (int i = 0; i < m; ++i) {
    trips.emplace_back(i, n + i, 1.0);
  }

  SparseMatrix A_aux(m, n + m);
  if (!trips.empty())
    A_aux.setFromTriplets(trips.begin(), trips.end());
  A_aux.makeCompressed();

  Eigen::VectorXd c_aux(n + m);
  c_aux.setZero();
  c_aux.tail(m).setOnes();

  std::vector<int> basis(m);
  std::iota(basis.begin(), basis.end(), n);

  return {A_aux, b1, c_aux, basis, static_cast<std::size_t>(n), m};
}
