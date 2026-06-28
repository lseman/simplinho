#pragma once

inline LPSolution RevisedSimplex::attach_postsolved_row_duals_(
    LPSolution sol, const presolve::Presolver &P, double tol) {
  if (sol.dual_values_internal.size() != P.result().reduced.A.rows()) {
    return sol;
  }
  Eigen::VectorXd y = P.postsolve_dual(sol.dual_values_internal);
  sol.dual_values = clip_small_vec_(std::move(y), tol);
  sol.shadow_prices = sol.dual_values;
  return sol;
}

inline LPSolution RevisedSimplex::attach_postsolved_farkas_(
    LPSolution sol, const presolve::Presolver &P, double tol) {
  if (!sol.farkas_has_cert)
    return sol;
  if (sol.farkas_y.size() != P.result().reduced.A.rows())
    return sol;
  sol.farkas_y_internal = sol.farkas_y;
  sol.farkas_y = clip_small_vec_(P.postsolve_dual(sol.farkas_y_internal), tol);
  return sol;
}

inline LPSolution RevisedSimplex::attach_mapped_primal_ray_(
    LPSolution sol, const std::vector<int> &col_orig_map,
    const Eigen::VectorXd &sign, int original_num_cols, double tol) {
  if (!sol.primal_ray_has_cert)
    return sol;
  if (sol.primal_ray.size() != static_cast<int>(col_orig_map.size()))
    return sol;
  sol.primal_ray_internal = sol.primal_ray;
  Eigen::VectorXd mapped = Eigen::VectorXd::Zero(original_num_cols);
  for (int jr = 0; jr < static_cast<int>(col_orig_map.size()); ++jr) {
    const int jorig = col_orig_map[jr];
    if (jorig < 0 || jorig >= original_num_cols || jorig >= sign.size()) {
      continue;
    }
    mapped(jorig) += sign(jorig) * sol.primal_ray_internal(jr);
  }
  sol.primal_ray = clip_small_vec_(std::move(mapped), tol);
  return sol;
}

inline LPSolution RevisedSimplex::attach_internal_basis_(
    LPSolution sol, std::vector<int> basis_internal,
    std::vector<std::string> internal_column_labels) {
  sol.basis_internal = std::move(basis_internal);
  sol.internal_column_labels = std::move(internal_column_labels);
  return sol;
}

inline LPSolution RevisedSimplex::attach_internal_tableau_(
    LPSolution sol, const Eigen::MatrixXd &A_internal,
    const Eigen::VectorXd &b_internal, const Eigen::VectorXd &c_internal,
    std::vector<int> basis_internal,
    std::vector<std::string> internal_column_labels,
    std::vector<std::string> internal_row_labels, double tol,
    bool compute_tableau, bool compute_reduced_costs) {
  sol.basis_internal = std::move(basis_internal);
  sol.internal_column_labels = std::move(internal_column_labels);
  sol.internal_row_labels = std::move(internal_row_labels);
  sol.nonbasis_internal = make_nonbasis_internal_(
      static_cast<int>(A_internal.cols()), sol.basis_internal);

  const int m = static_cast<int>(A_internal.rows());
  const int n = static_cast<int>(A_internal.cols());
  if (m == 0) {
    sol.dual_values_internal = Eigen::VectorXd::Zero(0);
    sol.shadow_prices_internal = Eigen::VectorXd::Zero(0);
    if (compute_tableau) {
      sol.tableau = Eigen::MatrixXd::Zero(0, n);
      sol.tableau_rhs = Eigen::VectorXd::Zero(0);
      sol.has_internal_tableau = true;
    }
    if (compute_reduced_costs) {
      sol.reduced_costs_internal = clip_small_vec_(c_internal, tol);
    }
    return sol;
  }
  if (static_cast<int>(sol.basis_internal.size()) != m)
    return sol;

  Eigen::VectorXi basis_idx =
      Eigen::Map<const Eigen::VectorXi>(sol.basis_internal.data(), m);
  const Eigen::MatrixXd B = A_internal(Eigen::all, basis_idx);
  Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
  if (!(lu.rank() == m && lu.isInvertible()))
    return sol;

  Eigen::VectorXd cB(m);
  for (int i = 0; i < m; ++i)
    cB(i) = c_internal(sol.basis_internal[i]);
  Eigen::FullPivLU<Eigen::MatrixXd> lu_t(B.transpose());
  if (lu_t.rank() == m && lu_t.isInvertible()) {
    const Eigen::VectorXd y = lu_t.solve(cB);
    sol.dual_values_internal = clip_small_vec_(y, tol);
    sol.shadow_prices_internal = sol.dual_values_internal;
    if (compute_reduced_costs) {
      sol.reduced_costs_internal =
          clip_small_vec_(c_internal - A_internal.transpose() * y, tol);
    }
  }

  if (compute_tableau) {
    sol.tableau = clip_small_mat_(lu.solve(A_internal), tol);
    sol.tableau_rhs = clip_small_vec_(lu.solve(b_internal), tol);
    sol.has_internal_tableau = true;
  }
  return sol;
}

inline LPSolution RevisedSimplex::attach_internal_tableau_(
    LPSolution sol, const SparseMatrix &A_internal,
    const Eigen::VectorXd &b_internal, const Eigen::VectorXd &c_internal,
    std::vector<int> basis_internal,
    std::vector<std::string> internal_column_labels,
    std::vector<std::string> internal_row_labels, double tol,
    bool compute_tableau, bool compute_reduced_costs) {
  sol.basis_internal = std::move(basis_internal);
  sol.internal_column_labels = std::move(internal_column_labels);
  sol.internal_row_labels = std::move(internal_row_labels);
  sol.nonbasis_internal = make_nonbasis_internal_(
      static_cast<int>(A_internal.cols()), sol.basis_internal);

  const int m = static_cast<int>(A_internal.rows());
  const int n = static_cast<int>(A_internal.cols());
  if (m == 0) {
    sol.dual_values_internal = Eigen::VectorXd::Zero(0);
    sol.shadow_prices_internal = Eigen::VectorXd::Zero(0);
    if (compute_tableau) {
      sol.tableau = Eigen::MatrixXd::Zero(0, n);
      sol.tableau_rhs = Eigen::VectorXd::Zero(0);
      sol.has_internal_tableau = true;
    }
    if (compute_reduced_costs) {
      sol.reduced_costs_internal = clip_small_vec_(c_internal, tol);
    }
    return sol;
  }
  if (static_cast<int>(sol.basis_internal.size()) != m)
    return sol;

  SparseMatrix B = sparse_basis_copy_(A_internal, sol.basis_internal);
  Eigen::SparseLU<SparseMatrix> lu;
  lu.analyzePattern(B);
  lu.factorize(B);
  if (lu.info() != Eigen::Success)
    return sol;

  Eigen::VectorXd cB(m);
  for (int i = 0; i < m; ++i)
    cB(i) = c_internal(sol.basis_internal[i]);

  const Eigen::VectorXd y = sparse_solveT_from_lu_(B, lu, cB);
  if (y.allFinite()) {
    sol.dual_values_internal = clip_small_vec_(y, tol);
    sol.shadow_prices_internal = sol.dual_values_internal;
    if (compute_reduced_costs) {
      sol.reduced_costs_internal =
          clip_small_vec_(c_internal - A_internal.transpose() * y, tol);
    }
  }

  if (compute_tableau) {
    sol.tableau = Eigen::MatrixXd::Zero(m, n);
    for (int j = 0; j < n; ++j) {
      Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m);
      for (SparseMatrix::InnerIterator it(A_internal, j); it; ++it) {
        rhs(it.row()) = it.value();
      }
      Eigen::VectorXd col = lu.solve(rhs);
      if (lu.info() != Eigen::Success)
        return sol;
      sol.tableau.col(j) = col;
    }
    sol.tableau = clip_small_mat_(std::move(sol.tableau), tol);
    sol.tableau_rhs = clip_small_vec_(lu.solve(b_internal), tol);
    if (lu.info() != Eigen::Success)
      return sol;
    sol.has_internal_tableau = true;
  }
  return sol;
}

inline LPSolution RevisedSimplex::attach_basis_state_(LPSolution sol,
                                                      const Eigen::VectorXd &l,
                                                      const Eigen::VectorXd &u,
                                                      double tol,
                                                      int basic_target) {
  if (sol.x.size() > 0 && sol.x.array().isFinite().all()) {
    auto t0_ser = std::chrono::steady_clock::now();
  LPBasis warm_basis =
    compute_basis_state_(sol.basis, sol.x, l, u, tol, basic_target);
  warm_basis.warm_state = sol.basis_state.warm_state;
    sol.basis_state = warm_basis;
    sol.info["warm_start_basis_state"] =
        serialize_basis_state_from_primal_(sol.basis, sol.x, l, u, tol);
    auto t1_ser = std::chrono::steady_clock::now();
    sol.timing.serialization_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1_ser - t0_ser).count();
  }
  return sol;
}
