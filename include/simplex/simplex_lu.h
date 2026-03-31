#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

// ======================================================
// Dense Markowitz + rook LU with permutations/refinement
// ======================================================
class MarkowitzLU {
public:
  MarkowitzLU() = default;

  MarkowitzLU(const Eigen::MatrixXd &A, double pivot_rel = 1e-12,
              double abs_floor = 1e-16, int rook_iters = 2) {
    factor(A, pivot_rel, abs_floor, rook_iters);
  }

  void factor(const Eigen::MatrixXd &A, double pivot_rel = 1e-12,
              double abs_floor = 1e-16, int rook_iters = 2) {
    if (A.rows() != A.cols())
      throw std::invalid_argument("MarkowitzLU: square only");

    n_ = static_cast<int>(A.rows());
    pivot_rel_ = pivot_rel;
    abs_floor_ = abs_floor;
    rook_iters_ = rook_iters;
    use_fallback_full_piv_ = false;

    L_ = Eigen::MatrixXd::Zero(n_, n_);
    U_ = A;
    Pr_.resize(n_);
    Pc_.resize(n_);
    std::iota(Pr_.begin(), Pr_.end(), 0);
    std::iota(Pc_.begin(), Pc_.end(), 0);

    try {
      factorize_();
    } catch (const std::runtime_error &) {
      fallback_lu_.compute(A);
      if (!fallback_lu_.isInvertible() || fallback_lu_.rank() != n_)
        throw;

      fallback_lu_t_.compute(A.transpose());
      if (!fallback_lu_t_.isInvertible() || fallback_lu_t_.rank() != n_)
        throw;

      use_fallback_full_piv_ = true;
    }
  }

  Eigen::VectorXd solve(const Eigen::VectorXd &b) const {
    if (b.size() != n_)
      throw std::invalid_argument("MarkowitzLU::solve size mismatch");

    if (use_fallback_full_piv_)
      return fallback_lu_.solve(b);

    Eigen::VectorXd Pb = apply_Pr_(b);
    Eigen::VectorXd z = forward_sub_(L_, Pb);
    Eigen::VectorXd w = back_sub_(U_, z);

    const int max_refinements = 3;
    const double refinement_tol = 1e-14;

    for (int iter = 0; iter < max_refinements; ++iter) {
      Eigen::VectorXd r = Pb - L_ * (U_ * w);
      double denom = std::max(1.0, Pb.lpNorm<Eigen::Infinity>());
      double backward_err = r.lpNorm<Eigen::Infinity>() / denom;
      if (backward_err < refinement_tol)
        break;

      Eigen::VectorXd dz = forward_sub_(L_, r);
      Eigen::VectorXd dw = back_sub_(U_, dz);
      if (!dw.array().isFinite().all() || dw.lpNorm<Eigen::Infinity>() < 1e-16)
        break;
      w += dw;
    }

    return apply_Pc_(w);
  }

  Eigen::VectorXd solveT(const Eigen::VectorXd &c) const {
    if (c.size() != n_)
      throw std::invalid_argument("MarkowitzLU::solveT size mismatch");

    if (use_fallback_full_piv_)
      return fallback_lu_t_.solve(c);

    Eigen::VectorXd PcTc = apply_PcT_(c);
    Eigen::VectorXd t = forward_sub_(U_.transpose(), PcTc);
    Eigen::VectorXd s = back_sub_(L_.transpose(), t);

    const int max_refinements = 3;
    const double refinement_tol = 1e-14;

    for (int iter = 0; iter < max_refinements; ++iter) {
      Eigen::VectorXd r = PcTc - U_.transpose() * (L_.transpose() * s);
      double denom = std::max(1.0, PcTc.lpNorm<Eigen::Infinity>());
      double backward_err = r.lpNorm<Eigen::Infinity>() / denom;
      if (backward_err < refinement_tol)
        break;

      Eigen::VectorXd dt = forward_sub_(U_.transpose(), r);
      Eigen::VectorXd ds = back_sub_(L_.transpose(), dt);
      if (!ds.array().isFinite().all() || ds.lpNorm<Eigen::Infinity>() < 1e-16)
        break;
      s += ds;
    }

    return apply_PrT_inv_(s);
  }

  int n() const noexcept { return n_; }
  bool supports_inplace_updates() const noexcept {
    return !use_fallback_full_piv_;
  }

  Eigen::MatrixXd &L() { return L_; }
  Eigen::MatrixXd &U() { return U_; }

private:
  static constexpr double kSingFloor_ = 1e-18;

  static Eigen::VectorXd forward_sub_(const Eigen::MatrixXd &L,
                                      const Eigen::VectorXd &b) {
    const int n = static_cast<int>(L.rows());
    Eigen::VectorXd x = b;
    for (int i = 0; i < n; ++i) {
      double s = L.row(i).head(i).dot(x.head(i));
      double piv = L(i, i);
      if (std::abs(piv) < kSingFloor_ || !std::isfinite(piv))
        throw std::runtime_error("Singular lower triangular");
      x(i) = (x(i) - s) / piv;
    }
    return x;
  }

  static Eigen::VectorXd back_sub_(const Eigen::MatrixXd &U,
                                   const Eigen::VectorXd &b) {
    const int n = static_cast<int>(U.rows());
    Eigen::VectorXd x = b;
    for (int i = n - 1; i >= 0; --i) {
      double s = U.row(i)
                     .segment(i + 1, n - (i + 1))
                     .dot(x.segment(i + 1, n - (i + 1)));
      double piv = U(i, i);
      if (std::abs(piv) < kSingFloor_ || !std::isfinite(piv))
        throw std::runtime_error("Singular upper triangular");
      x(i) = (x(i) - s) / piv;
    }
    return x;
  }

  Eigen::VectorXd apply_Pr_(const Eigen::VectorXd &v) const {
    Eigen::VectorXd out(n_);
    for (int i = 0; i < n_; ++i)
      out(i) = v(Pr_[i]);
    return out;
  }

  Eigen::VectorXd apply_PrT_inv_(const Eigen::VectorXd &y) const {
    Eigen::VectorXd out(n_);
    for (int i = 0; i < n_; ++i)
      out(Pr_[i]) = y(i);
    return out;
  }

  Eigen::VectorXd apply_Pc_(const Eigen::VectorXd &x) const {
    Eigen::VectorXd out(n_);
    for (int i = 0; i < n_; ++i)
      out(Pc_[i]) = x(i);
    return out;
  }

  Eigen::VectorXd apply_PcT_(const Eigen::VectorXd &c) const {
    Eigen::VectorXd out(n_);
    for (int i = 0; i < n_; ++i)
      out(i) = c(Pc_[i]);
    return out;
  }

  void swap_rows_(int i, int j) {
    if (i == j)
      return;
    U_.row(i).swap(U_.row(j));
    L_.row(i).head(i).swap(L_.row(j).head(i));
    std::swap(Pr_[i], Pr_[j]);
  }

  void swap_cols_(int i, int j) {
    if (i == j)
      return;
    U_.col(i).swap(U_.col(j));
    std::swap(Pc_[i], Pc_[j]);
  }

  static std::pair<std::vector<int>, std::vector<int>>
  nnz_row_col_(const Eigen::MatrixXd &M, double eps = 1e-16) {
    const int r = static_cast<int>(M.rows());
    const int c = static_cast<int>(M.cols());
    std::vector<int> rn(r, 0), cn(c, 0);
    for (int i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j)
        if (std::abs(M(i, j)) > eps) {
          rn[i]++;
          cn[j]++;
        }
    return {rn, cn};
  }

  std::tuple<int, int, double> choose_pivot_(int k,
                                             const Eigen::VectorXd &colmax) {
    int best_i = -1, best_j = -1;
    double best_val = 0.0;
    long best_score = std::numeric_limits<long>::max();

    Eigen::MatrixXd sub = U_.block(k, k, n_ - k, n_ - k);
    auto [rn, cn] = nnz_row_col_(sub);

    for (int i = k; i < n_; ++i) {
      for (int j = k; j < n_; ++j) {
        double aij = U_(i, j);
        if (std::abs(aij) >= pivot_rel_ * std::max(colmax(j - k), abs_floor_)) {
          long score = static_cast<long>(rn[i - k] - 1) *
                       static_cast<long>(cn[j - k] - 1);
          if (score < best_score ||
              (score == best_score && std::abs(aij) > std::abs(best_val))) {
            best_score = score;
            best_i = i;
            best_j = j;
            best_val = aij;
          }
        }
      }
    }

    if (best_i >= 0)
      return {best_i, best_j, best_val};

    int i_idx = 0;
    U_.col(k).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
    int i = k + i_idx;

    int j_idx = 0;
    U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
    int j = k + j_idx;

    for (int t = 0; t < std::max(0, rook_iters_); ++t) {
      int prev_i = i, prev_j = j;
      U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff(&i_idx);
      i = k + i_idx;
      U_.row(i).segment(k, n_ - k).cwiseAbs().maxCoeff(&j_idx);
      j = k + j_idx;
      if (i == prev_i && j == prev_j)
        break;
    }

    double val = U_(i, j);
    double col_abs_max = U_.col(j).segment(k, n_ - k).cwiseAbs().maxCoeff();
    if (std::abs(val) >= std::max(abs_floor_, pivot_rel_ * col_abs_max))
      return {i, j, val};

    return {-1, -1, 0.0};
  }

  void factorize_() {
    const double inf_norm = L1_inf_norm_(U_);

    for (int k = 0; k < n_; ++k) {
      Eigen::VectorXd colmax = U_.block(k, k, n_ - k, n_ - k)
                                   .cwiseAbs()
                                   .colwise()
                                   .maxCoeff()
                                   .transpose();

      for (int t = 0; t < colmax.size(); ++t)
        if (colmax(t) < abs_floor_)
          colmax(t) = 1.0;

      auto [pi, pj, pval] = choose_pivot_(k, colmax);
      if (pi < 0) {
        Eigen::MatrixXd sub = U_.block(k, k, n_ - k, n_ - k).cwiseAbs();
        Eigen::Index rr, cc;
        sub.maxCoeff(&rr, &cc);
        pi = k + static_cast<int>(rr);
        pj = k + static_cast<int>(cc);

        const double floor_adapt = std::max(
            abs_floor_, 10 * std::numeric_limits<double>::epsilon() * inf_norm);
        if (std::abs(U_(pi, pj)) < floor_adapt)
          throw std::runtime_error("MarkowitzLU: singular matrix");
      }

      swap_rows_(k, pi);
      swap_cols_(k, pj);

      double piv = U_(k, k);
      const double floor_adapt = std::max(
          abs_floor_, 10 * std::numeric_limits<double>::epsilon() * inf_norm);
      if (std::abs(piv) < floor_adapt || !std::isfinite(piv))
        throw std::runtime_error("MarkowitzLU: numerically singular pivot");

      L_(k, k) = 1.0;
      for (int i = k + 1; i < n_; ++i) {
        double lik = U_(i, k);
        if (lik != 0.0) {
          L_(i, k) = lik / piv;
          U_.row(i).segment(k, n_ - k) -=
              L_(i, k) * U_.row(k).segment(k, n_ - k);
        }
      }
    }
  }

  static double L1_inf_norm_(const Eigen::MatrixXd &A) {
    if (A.size() == 0)
      return 0.0;
    double maxrow = 0.0;
    for (int i = 0; i < A.rows(); ++i)
      maxrow = std::max(maxrow, A.row(i).cwiseAbs().sum());
    return maxrow;
  }

private:
  int n_{0};
  double pivot_rel_{1e-12};
  double abs_floor_{1e-16};
  int rook_iters_{2};
  bool use_fallback_full_piv_{false};

  Eigen::MatrixXd L_, U_;
  Eigen::FullPivLU<Eigen::MatrixXd> fallback_lu_;
  Eigen::FullPivLU<Eigen::MatrixXd> fallback_lu_t_;
  std::vector<int> Pr_, Pc_;
};

// ======================================================
// Sparse Forrest–Tomlin LU
//
// Practical simplex-oriented implementation:
//   - explicit sparse row/column structures
//   - sparse Markowitz-style factorization
//   - sparse forward/back solves with permutations
//   - sparse FT-style in-place column update on U
//   - spike elimination stored into L
//
// This is a mutable sparse LU update engine intended for
// basis maintenance, not a full industrial sparse solver.
// ======================================================
class SparseForrestTomlinLU {
public:
  using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
  using SparseVec = Eigen::SparseVector<double, Eigen::ColMajor, int>;

  SparseForrestTomlinLU() = default;

  void factor(const SparseMat &A, double pivot_rel = 1e-12,
              double abs_floor = 1e-16, int refactor_rook_iters = 2,
              int ft_bandwidth_cap = 0) {
    if (A.rows() != A.cols())
      throw std::invalid_argument("SparseForrestTomlinLU: square only");

    n_ = static_cast<int>(A.rows());
    pivot_rel_ = pivot_rel;
    abs_floor_ = abs_floor;
    rook_iters_ = refactor_rook_iters;
    ft_bandwidth_cap_ = ft_bandwidth_cap;

    Pr_.resize(n_);
    Pc_.resize(n_);
    std::iota(Pr_.begin(), Pr_.end(), 0);
    std::iota(Pc_.begin(), Pc_.end(), 0);

    U_rows_.assign(n_, {});
    U_cols_.assign(n_, {});
    L_rows_.assign(n_, {});
    L_cols_.assign(n_, {});

    load_initial_U_(A);
    factorize_sparse_();
    max_u_abs_ = max_abs_U_();
  }

  Eigen::VectorXd solve(const Eigen::VectorXd &b) const {
    if (b.size() != n_)
      throw std::invalid_argument("SparseForrestTomlinLU::solve size mismatch");

    Eigen::VectorXd Pb(n_);
    for (int i = 0; i < n_; ++i)
      Pb(i) = b(Pr_[i]);

    Eigen::VectorXd z = forward_solve_L_(Pb);
    Eigen::VectorXd w = back_solve_U_(z);

    Eigen::VectorXd x(n_);
    for (int i = 0; i < n_; ++i)
      x(Pc_[i]) = w(i);
    return x;
  }

  Eigen::VectorXd solveT(const Eigen::VectorXd &c) const {
    if (c.size() != n_)
      throw std::invalid_argument(
          "SparseForrestTomlinLU::solveT size mismatch");

    Eigen::VectorXd PcTc(n_);
    for (int i = 0; i < n_; ++i)
      PcTc(i) = c(Pc_[i]);

    Eigen::VectorXd t = forward_solve_UT_(PcTc);
    Eigen::VectorXd s = back_solve_LT_(t);

    Eigen::VectorXd y(n_);
    for (int i = 0; i < n_; ++i)
      y(Pr_[i]) = s(i);
    return y;
  }

  void replace_column(int j, const Eigen::VectorXd &new_col_dense,
                      double alpha_tol = 1e-10, double growth_tol = 1e4) {
    if (j < 0 || j >= n_)
      throw std::out_of_range("SparseForrestTomlinLU::replace_column bad j");
    if (new_col_dense.size() != n_)
      throw std::invalid_argument(
          "SparseForrestTomlinLU::replace_column size mismatch");

    // Compute u = new_col - old_col  (in original row ordering)
    Eigen::VectorXd old_col = basis_column_from_factors_(j);
    Eigen::VectorXd u = new_col_dense - old_col;

    // Solve for z:  B z = u   →   z tells how the update affects the basis
    Eigen::VectorXd z = solve(u);

    double alpha = 1.0 + z(j);
    if (!std::isfinite(alpha) || std::abs(alpha) < alpha_tol)
      throw std::runtime_error("SparseForrestTomlinLU: unstable alpha");

    // Rank-1 update on column j of U:   U(:,j) ← U(:,j) + U * z
    apply_rank1_to_U_column_(j, z);

    // Suhl-style spike elimination with threshold pivoting
    eliminate_spike_in_column_(j);

    // Final checks
    double piv = get_U_(j, j);
    if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
      throw std::runtime_error(
          "SparseForrestTomlinLU: pivot collapsed after update");

    double curr_max = max_abs_U_();
    if (curr_max > std::max(1.0, max_u_abs_) * growth_tol)
      throw std::runtime_error("SparseForrestTomlinLU: growth too large");

    max_u_abs_ = std::max(max_u_abs_, curr_max);
  }

  int n() const noexcept { return n_; }

  SparseMat L_sparse() const { return maps_to_sparse_(L_rows_, n_, n_); }
  SparseMat U_sparse() const { return maps_to_sparse_(U_rows_, n_, n_); }

private:
  using RowMap = std::map<int, double>;
  using ColMap = std::map<int, double>;

  static constexpr double kZeroTol_ = 1e-16;

  static SparseMat maps_to_sparse_(const std::vector<RowMap> &rows, int m,
                                   int n) {
    std::vector<Eigen::Triplet<double>> trips;
    for (int i = 0; i < m; ++i)
      for (const auto &[j, v] : rows[i])
        if (std::abs(v) > kZeroTol_)
          trips.emplace_back(i, j, v);

    SparseMat out(m, n);
    out.setFromTriplets(trips.begin(), trips.end());
    out.makeCompressed();
    return out;
  }

  void load_initial_U_(const SparseMat &A) {
    for (int k = 0; k < A.outerSize(); ++k) {
      for (SparseMat::InnerIterator it(A, k); it; ++it) {
        if (std::abs(it.value()) > kZeroTol_)
          set_U_(it.row(), it.col(), it.value());
      }
    }
  }

  double get_U_(int i, int j) const {
    auto it = U_rows_[i].find(j);
    return (it == U_rows_[i].end()) ? 0.0 : it->second;
  }

  double get_L_(int i, int j) const {
    auto it = L_rows_[i].find(j);
    return (it == L_rows_[i].end()) ? 0.0 : it->second;
  }

  void set_U_(int i, int j, double v) {
    if (std::abs(v) <= kZeroTol_ || !std::isfinite(v)) {
      auto itr = U_rows_[i].find(j);
      if (itr != U_rows_[i].end())
        U_rows_[i].erase(itr);
      auto itc = U_cols_[j].find(i);
      if (itc != U_cols_[j].end())
        U_cols_[j].erase(itc);
      return;
    }
    U_rows_[i][j] = v;
    U_cols_[j][i] = v;
  }

  void set_L_(int i, int j, double v) {
    if (std::abs(v) <= kZeroTol_ || !std::isfinite(v)) {
      auto itr = L_rows_[i].find(j);
      if (itr != L_rows_[i].end())
        L_rows_[i].erase(itr);
      auto itc = L_cols_[j].find(i);
      if (itc != L_cols_[j].end())
        L_cols_[j].erase(itc);
      return;
    }
    L_rows_[i][j] = v;
    L_cols_[j][i] = v;
  }

  void swap_U_rows_(int a, int b) {
    if (a == b)
      return;

    std::vector<std::pair<int, double>> row_a(U_rows_[a].begin(),
                                              U_rows_[a].end());
    std::vector<std::pair<int, double>> row_b(U_rows_[b].begin(),
                                              U_rows_[b].end());

    for (const auto &[j, _] : row_a)
      U_cols_[j].erase(a);
    for (const auto &[j, _] : row_b)
      U_cols_[j].erase(b);

    std::swap(U_rows_[a], U_rows_[b]);

    for (const auto &[j, v] : U_rows_[a])
      U_cols_[j][a] = v;
    for (const auto &[j, v] : U_rows_[b])
      U_cols_[j][b] = v;
  }

  void swap_U_cols_(int a, int b) {
    if (a == b)
      return;

    std::vector<std::pair<int, double>> col_a(U_cols_[a].begin(),
                                              U_cols_[a].end());
    std::vector<std::pair<int, double>> col_b(U_cols_[b].begin(),
                                              U_cols_[b].end());

    for (const auto &[i, _] : col_a)
      U_rows_[i].erase(a);
    for (const auto &[i, _] : col_b)
      U_rows_[i].erase(b);

    std::swap(U_cols_[a], U_cols_[b]);

    for (const auto &[i, v] : U_cols_[a])
      U_rows_[i][a] = v;
    for (const auto &[i, v] : U_cols_[b])
      U_rows_[i][b] = v;
  }

  void swap_L_prefix_rows_(int a, int b, int prefix_cols) {
    if (a == b || prefix_cols <= 0)
      return;

    for (int j = 0; j < prefix_cols; ++j) {
      double va = get_L_(a, j);
      double vb = get_L_(b, j);
      set_L_(a, j, vb);
      set_L_(b, j, va);
    }
  }

  std::pair<int, int> choose_pivot_sparse_(int k) const {
    if (k >= n_)
      return {-1, -1};

    int best_i = -1;
    int best_j = -1;
    long best_score = std::numeric_limits<long>::max();
    double best_abs = -1.0;

    // Step 1: Count nonzeros in the active submatrix (k to n-1)
    std::vector<int> row_nnz(n_, 0);
    std::vector<int> col_nnz(n_, 0);

    for (int i = k; i < n_; ++i) {
      for (const auto &[j, v] : U_rows_[i]) {
        if (j >= k && std::abs(v) > abs_floor_) {
          row_nnz[i]++;
          col_nnz[j]++;
        }
      }
    }

    // Step 2: Markowitz search with threshold bias for stability
    const double threshold_bias =
        0.01; // small stability bias (common in simplex LU)

    for (int i = k; i < n_; ++i) {
      if (row_nnz[i] == 0)
        continue;

      for (const auto &[j, aij_val] : U_rows_[i]) {
        if (j < k)
          continue;

        double ab = std::abs(aij_val);
        if (ab <= abs_floor_)
          continue;

        // Compute column max in the remaining submatrix (more accurate than
        // before)
        double col_max = abs_floor_;
        auto it = U_cols_[j].lower_bound(k);
        for (; it != U_cols_[j].end(); ++it) {
          col_max = std::max(col_max, std::abs(it->second));
        }

        // Threshold test + stability bias
        if (ab < pivot_rel_ * col_max * (1.0 + threshold_bias))
          continue;

        long score = static_cast<long>(std::max(0, row_nnz[i] - 1)) *
                     static_cast<long>(std::max(0, col_nnz[j] - 1));

        // Tie-breaking: prefer larger pivot
        bool better =
            (score < best_score) || (score == best_score && ab > best_abs);

        if (better) {
          best_score = score;
          best_abs = ab;
          best_i = i;
          best_j = j;
        }
      }
    }

    // If we found a good Markowitz pivot, return it
    if (best_i >= 0) {
      return {best_i, best_j};
    }

    // Step 3: Stronger rook pivoting fallback (more robust)
    int i = k;
    int j = k;

    // Initial: find largest in column k (submatrix)
    double max_in_col = -1.0;
    auto itc = U_cols_[k].lower_bound(k);
    for (; itc != U_cols_[k].end(); ++itc) {
      double ab = std::abs(itc->second);
      if (ab > max_in_col) {
        max_in_col = ab;
        i = itc->first;
      }
    }

    // Rook iterations (increased max iterations)
    const int max_rook_steps = std::max(6, rook_iters_ * 2);

    for (int t = 0; t < max_rook_steps; ++t) {
      int prev_i = i;
      int prev_j = j;

      // Find max in current row i (from column k onward)
      double max_in_row = -1.0;
      int new_j = j;
      for (const auto &[cj, v] : U_rows_[i]) {
        if (cj < k)
          continue;
        double ab = std::abs(v);
        if (ab > max_in_row) {
          max_in_row = ab;
          new_j = cj;
        }
      }
      j = new_j;

      // Find max in current column j (from row k onward)
      double max_in_new_col = -1.0;
      int new_i = i;
      auto it = U_cols_[j].lower_bound(k);
      for (; it != U_cols_[j].end(); ++it) {
        double ab = std::abs(it->second);
        if (ab > max_in_new_col) {
          max_in_new_col = ab;
          new_i = it->first;
        }
      }
      i = new_i;

      // Convergence check
      if (i == prev_i && j == prev_j && t >= 1)
        break;
    }

    // Final safety: if the chosen pivot is still too small, try to find any
    // non-zero
    double final_piv = get_U_(i, j);
    if (std::abs(final_piv) < abs_floor_ && k < n_) {
      // Very last resort: any non-zero in submatrix
      for (int r = k; r < n_; ++r) {
        for (const auto &[c, v] : U_rows_[r]) {
          if (c >= k && std::abs(v) > abs_floor_) {
            return {r, c};
          }
        }
      }
    }

    return {i, j};
  }

  void factorize_sparse_() {
    for (int k = 0; k < n_; ++k) {
      auto [pi, pj] = choose_pivot_sparse_(k);
      if (pi < 0 || pj < 0)
        throw std::runtime_error("SparseForrestTomlinLU: no pivot found");

      swap_U_rows_(k, pi);
      swap_U_cols_(k, pj);
      swap_L_prefix_rows_(k, pi, k);
      std::swap(Pr_[k], Pr_[pi]);
      std::swap(Pc_[k], Pc_[pj]);

      double piv = get_U_(k, k);
      if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
        throw std::runtime_error("SparseForrestTomlinLU: singular pivot");

      set_L_(k, k, 1.0);

      std::vector<int> affected_rows;
      auto itc = U_cols_[k].upper_bound(k);
      for (; itc != U_cols_[k].end(); ++itc)
        affected_rows.push_back(itc->first);

      for (int i : affected_rows) {
        double uik = get_U_(i, k);
        if (std::abs(uik) <= kZeroTol_)
          continue;

        double lik = uik / piv;
        set_L_(i, k, lik);

        std::vector<std::pair<int, double>> pivot_row(U_rows_[k].lower_bound(k),
                                                      U_rows_[k].end());

        for (const auto &[j, ukj] : pivot_row) {
          double newv = get_U_(i, j) - lik * ukj;
          set_U_(i, j, newv);
        }
        set_U_(i, k, 0.0);
      }
    }
  }

  Eigen::VectorXd forward_solve_L_(const Eigen::VectorXd &b) const {
    Eigen::VectorXd x = b;
    for (int i = 0; i < n_; ++i) {
      double s = 0.0;
      bool has_nonzero = false;

      for (const auto &[j, v] : L_rows_[i]) {
        if (j >= i)
          break;
        if (std::abs(v) > kZeroTol_) {
          s += v * x(j);
          has_nonzero = true;
        }
      }

      double piv = get_L_(i, i);
      if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
        throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal");

      if (has_nonzero || std::abs(x(i) - s) > kZeroTol_) {
        x(i) = (x(i) - s) / piv;
      } else {
        x(i) = 0.0; // early zeroing
      }
    }
    return x;
  }

  Eigen::VectorXd back_solve_U_(const Eigen::VectorXd &b) const {
    Eigen::VectorXd x = b;
    for (int i = n_ - 1; i >= 0; --i) {
      double s = 0.0;
      auto it = U_rows_[i].upper_bound(i);
      for (; it != U_rows_[i].end(); ++it) {
        s += it->second * x(it->first);
      }

      double piv = get_U_(i, i);
      if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
        throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal");

      x(i) = (x(i) - s) / piv;

      // Optional: zero small values for better sparsity in subsequent ops
      if (std::abs(x(i)) < kZeroTol_)
        x(i) = 0.0;
    }
    return x;
  }
  Eigen::VectorXd forward_solve_UT_(const Eigen::VectorXd &b) const {
    Eigen::VectorXd x = b;
    for (int i = 0; i < n_; ++i) {
      double s = 0.0;
      for (const auto &[r, v] : U_cols_[i]) {
        if (r >= i)
          break;
        s += v * x(r);
      }

      double piv = get_U_(i, i);
      if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
        throw std::runtime_error("SparseForrestTomlinLU: bad U diagonal in UT");

      x(i) = (x(i) - s) / piv;
      if (std::abs(x(i)) < kZeroTol_)
        x(i) = 0.0;
    }
    return x;
  }

  Eigen::VectorXd back_solve_LT_(const Eigen::VectorXd &b) const {
    Eigen::VectorXd x = b;
    for (int i = n_ - 1; i >= 0; --i) {
      double s = 0.0;
      auto it = L_cols_[i].upper_bound(i);
      for (; it != L_cols_[i].end(); ++it) {
        s += it->second * x(it->first);
      }

      double piv = get_L_(i, i);
      if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
        throw std::runtime_error("SparseForrestTomlinLU: bad L diagonal in LT");

      x(i) = (x(i) - s) / piv;
      if (std::abs(x(i)) < kZeroTol_)
        x(i) = 0.0;
    }
    return x;
  }
  Eigen::VectorXd basis_column_from_factors_(int j) const {
    Eigen::VectorXd ej = Eigen::VectorXd::Zero(n_);
    ej(j) = 1.0;

    // B e_j = P_r^T L U P_c^T e_j
    // First q = P_c^T e_j
    Eigen::VectorXd q = Eigen::VectorXd::Zero(n_);
    for (int i = 0; i < n_; ++i)
      if (Pc_[i] == j)
        q(i) = 1.0;

    // r = U q
    Eigen::VectorXd r = Eigen::VectorXd::Zero(n_);
    for (int i = 0; i < n_; ++i)
      for (const auto &[col, v] : U_rows_[i])
        r(i) += v * q(col);

    // s = L r
    Eigen::VectorXd s = Eigen::VectorXd::Zero(n_);
    for (int i = 0; i < n_; ++i)
      for (const auto &[col, v] : L_rows_[i])
        s(i) += v * r(col);

    // out = P_r^T s
    Eigen::VectorXd out(n_);
    for (int i = 0; i < n_; ++i)
      out(Pr_[i]) = s(i);

    return out;
  }

  void apply_rank1_to_U_column_(int j, const Eigen::VectorXd &z) {
    std::map<int, double> delta;

    for (int k = 0; k < n_; ++k) {
      double zk = z(k);
      if (std::abs(zk) <= kZeroTol_)
        continue;

      for (const auto &[r, v] : U_cols_[k])
        delta[r] += zk * v;
    }

    for (const auto &[r, dv] : delta) {
      double newv = get_U_(r, j) + dv;
      set_U_(r, j, newv);
    }
  }
  void eliminate_spike_in_column_(int j) {
    double piv = get_U_(j, j);
    if (!std::isfinite(piv) || std::abs(piv) < abs_floor_)
      throw std::runtime_error(
          "SparseForrestTomlinLU: bad pivot during spike elimination");

    std::vector<int> spike_rows;
    auto itc = U_cols_[j].upper_bound(j);
    for (; itc != U_cols_[j].end(); ++itc)
      spike_rows.push_back(itc->first);

    if (spike_rows.empty())
      return;

    const int band = (ft_bandwidth_cap_ > 0) ? ft_bandwidth_cap_
                                             : std::numeric_limits<int>::max();

    const double mult_threshold = 0.1; // Suhl-style stability threshold
    const double drop_tol = 1e-14;     // drop tiny fill-in

    for (int i : spike_rows) {
      if (i <= j || (i - j > band))
        continue;

      double uij = get_U_(i, j);
      if (std::abs(uij) <= kZeroTol_)
        continue;

      double mult = uij / piv;

      // Threshold pivoting (core Suhl-inspired stability control)
      if (std::abs(mult) > mult_threshold) {
        continue; // skip to avoid large growth
      }

      set_L_(i, j, mult);

      // Elimination: row_i -= mult * row_j  (from column j onwards)
      for (auto it = U_rows_[j].lower_bound(j); it != U_rows_[j].end(); ++it) {
        int col = it->first;
        double vj = it->second;

        double newv = get_U_(i, col) - mult * vj;

        if (std::abs(newv) > drop_tol) {
          set_U_(i, col, newv);
        } else {
          set_U_(i, col, 0.0);
        }
      }

      set_U_(i, j, 0.0);
    }
  }
  double max_abs_U_() const {
    double mx = 0.0;
    for (const auto &row : U_rows_)
      for (const auto &[j, v] : row)
        mx = std::max(mx, std::abs(v));
    return mx;
  }

private:
  int n_{0};
  double pivot_rel_{1e-8};
  double abs_floor_{1e-12};
  int rook_iters_{3};
  int ft_bandwidth_cap_{0};
  double max_u_abs_{1.0};

  std::vector<int> Pr_, Pc_;
  std::vector<RowMap> U_rows_, L_rows_;
  std::vector<ColMap> U_cols_, L_cols_;
};

// ======================================================
// Forrest–Tomlin basis updates over LU backends
//   - Dense : MarkowitzLU + optional in-place FT update
//   - Sparse: custom SparseForrestTomlinLU
// ======================================================
class FTBasis {
public:
  using DenseMat = Eigen::MatrixXd;
  using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

  struct Options {
    int refactor_every = 20;
    int compress_every = 10;
    double pivot_rel = 1e-12;
    double abs_floor = 1e-16;
    double alpha_tol = 1e-10;
    double z_inf_guard = 1e6;
    bool sparse_amd = true;
    double sparse_drop_tol = 0.0;

    enum class UpdateMode { EtaStack, ForrestTomlin };
    UpdateMode update_mode = UpdateMode::ForrestTomlin;

    int ft_bandwidth_cap = 16;
    double max_growth_tol = 1e4;
    int rook_iters = 2;
    double ft_multiplier_threshold = 0.1; // new parameter
  };

  struct Eta {
    int j;
    Eigen::VectorXd u;
    Eigen::VectorXd z;
    Eigen::VectorXd w;
    double alpha;
  };

  FTBasis(const DenseMat &A, const std::vector<int> &basis)
      : FTBasis(A, basis, Options{}) {}

  FTBasis(const DenseMat &A, const std::vector<int> &basis, const Options &opt)
      : A_dense_(&A), A_sparse_(nullptr), A_is_sparse_(false),
        m_(static_cast<int>(A.rows())), basis_(basis), opt_(opt) {
    if (static_cast<int>(basis_.size()) != m_)
      throw std::invalid_argument("FTBasis: basis size must equal m");

    Bcols_dense_.resize(m_);
    for (int i = 0; i < m_; ++i)
      Bcols_dense_[i] = A.col(basis_[i]);

    dense_refactor_();
  }

  FTBasis(const SparseMat &A, const std::vector<int> &basis)
      : FTBasis(A, basis, Options{}) {}

  FTBasis(const SparseMat &A, const std::vector<int> &basis, const Options &opt)
      : A_dense_(nullptr), A_sparse_(&A), A_is_sparse_(true),
        m_(static_cast<int>(A.rows())), basis_(basis), opt_(opt) {
    if (static_cast<int>(basis_.size()) != m_)
      throw std::invalid_argument("FTBasis: basis size must equal m");

    Bcols_sparse_.resize(m_);
    for (int i = 0; i < m_; ++i)
      Bcols_sparse_[i] = A.col(basis_[i]);

    sparse_refactor_();
  }

  FTBasis(const DenseMat &A, const std::vector<int> &basis, int refactor_every,
          int compress_every, double pivot_rel, double abs_floor,
          double alpha_tol, double z_inf_guard)
      : FTBasis(A, basis,
                Options{refactor_every, compress_every, pivot_rel, abs_floor,
                        alpha_tol, z_inf_guard, true, 0.0,
                        Options::UpdateMode::EtaStack, 16, 1e4, 2}) {}

  int rows() const noexcept { return m_; }
  const std::vector<int> &basis() const noexcept { return basis_; }
  const std::vector<Eta> &etas() const noexcept { return etas_; }

  Eigen::VectorXd solve_B(const Eigen::VectorXd &b) const {
    Eigen::VectorXd x = A_is_sparse_ ? sparse_solve_(b) : dense_solve_(b);
    if (!A_is_sparse_ && !etas_.empty())
      x = apply_etas_solve_(x);
    return x;
  }

  Eigen::VectorXd solve_BT(const Eigen::VectorXd &c) const {
    Eigen::VectorXd y = A_is_sparse_ ? sparse_solveT_(c) : dense_solveT_(c);
    if (!A_is_sparse_ && !etas_.empty())
      y = apply_etas_solve_T_(y);
    return y;
  }

  void replace_column(int j, const Eigen::VectorXd &new_col_dense) {
    replace_column_impl_(j, new_col_dense);
  }

  template <typename Derived>
  void replace_column(int j,
                      const Eigen::SparseMatrixBase<Derived> &new_col_sparse) {
    Eigen::SparseMatrix<double> tmp = new_col_sparse.derived().eval();
    Eigen::VectorXd dense(m_);
    dense.setZero();
    for (Eigen::SparseMatrix<double>::InnerIterator it(tmp, 0); it; ++it)
      dense[it.row()] = it.value();
    replace_column_impl_(j, dense);
  }

  void refactor() {
    if (A_is_sparse_)
      sparse_refactor_();
    else
      dense_refactor_();
  }

private:
  double max_element_ = 0.0;

  void dense_refactor_() {
    Eigen::MatrixXd B(m_, m_);
    for (int k = 0; k < m_; ++k)
      B.col(k) = Bcols_dense_[k];

    lu_dense_.factor(B, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters);
    max_element_ = lu_dense_.U().cwiseAbs().maxCoeff();
    etas_.clear();
    update_count_ = 0;
  }

  Eigen::VectorXd dense_solve_(const Eigen::VectorXd &b) const {
    return lu_dense_.solve(b);
  }

  Eigen::VectorXd dense_solveT_(const Eigen::VectorXd &c) const {
    return lu_dense_.solveT(c);
  }

  void sparse_build_B_(SparseMat &B) const {
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(std::max(1, m_)) * 8);

    for (int k = 0; k < m_; ++k) {
      const auto &col = Bcols_sparse_[k];
      for (SparseMat::InnerIterator it(col, 0); it; ++it)
        trips.emplace_back(it.row(), k, it.value());
    }

    B.resize(m_, m_);
    B.setFromTriplets(trips.begin(), trips.end());
    if (opt_.sparse_drop_tol > 0.0)
      B.prune(opt_.sparse_drop_tol);
    B.makeCompressed();
  }

  void sparse_refactor_() {
    SparseMat B;
    sparse_build_B_(B);
    lu_sparse_.factor(B, opt_.pivot_rel, opt_.abs_floor, opt_.rook_iters,
                      opt_.ft_bandwidth_cap);
    update_count_ = 0;
  }

  Eigen::VectorXd sparse_solve_(const Eigen::VectorXd &b) const {
    return lu_sparse_.solve(b);
  }

  Eigen::VectorXd sparse_solveT_(const Eigen::VectorXd &c) const {
    return lu_sparse_.solveT(c);
  }

  void forrest_tomlin_update_dense_(int j, const Eigen::VectorXd &z,
                                    double alpha) {
    Eigen::MatrixXd &L = lu_dense_.L();
    Eigen::MatrixXd &U = lu_dense_.U();
    const int n = static_cast<int>(U.rows());

    if (!std::isfinite(alpha) || std::abs(alpha) < opt_.alpha_tol)
      throw std::runtime_error("Forrest-Tomlin: alpha too small/unstable");

    Eigen::VectorXd contrib = U * z;
    U.col(j) += contrib;

    double pivot = U(j, j);
    if (!std::isfinite(pivot) ||
        std::abs(pivot) <
            std::max(opt_.abs_floor, 1e-14 * std::abs(contrib(j)))) {
      throw std::runtime_error("Forrest-Tomlin: new pivot too small");
    }

    const int band = (opt_.ft_bandwidth_cap > 0) ? opt_.ft_bandwidth_cap : n;
    const int i_lo = j + 1;
    const int i_hi = std::min(n - 1, j + band);

    for (int i = i_lo; i <= i_hi; ++i) {
      double factor = U(i, j) / pivot;
      if (std::abs(factor) > 1e-16) {
        L(i, j) = factor;
        U.row(i).segment(j, n - j).noalias() -=
            factor * U.row(j).segment(j, n - j);
        U(i, j) = 0.0;
      }
    }

    L(j, j) = 1.0;
    if (std::abs(U(j, j)) < opt_.abs_floor)
      throw std::runtime_error("Forrest-Tomlin: pivot collapsed");
  }

  void replace_column_impl_(int j, const Eigen::VectorXd &new_col_dense) {
    if (j < 0 || j >= m_)
      throw std::out_of_range("FTBasis::replace_column bad j");
    if (new_col_dense.size() != m_)
      throw std::invalid_argument("FTBasis::replace_column size mismatch");

    if (A_is_sparse_) {
      set_sparse_column_(j, new_col_dense);

      try {
        lu_sparse_.replace_column(j, new_col_dense, opt_.alpha_tol,
                                  opt_.max_growth_tol);
        ++update_count_;
      } catch (...) {
        sparse_refactor_();
        return;
      }

      if (update_count_ >= opt_.refactor_every)
        sparse_refactor_();
      return;
    }

    Eigen::VectorXd old = Bcols_dense_[j];
    Eigen::VectorXd u = new_col_dense - old;

    Eigen::VectorXd z = solve_B(u);
    Eigen::VectorXd ej = Eigen::VectorXd::Zero(m_);
    ej(j) = 1.0;
    Eigen::VectorXd w = solve_BT(ej);
    double alpha = 1.0 + z(j);

    const bool try_ft =
        lu_dense_.supports_inplace_updates() &&
        opt_.update_mode == Options::UpdateMode::ForrestTomlin &&
        std::abs(alpha) >= opt_.alpha_tol &&
        update_count_ < opt_.refactor_every &&
        z.cwiseAbs().maxCoeff() <= opt_.z_inf_guard;

    Bcols_dense_[j] = new_col_dense;

    if (try_ft) {
      bool ok = true;
      try {
        forrest_tomlin_update_dense_(j, z, alpha);
        double current_max = lu_dense_.U().cwiseAbs().maxCoeff();
        if (current_max > std::max(1.0, max_element_) * opt_.max_growth_tol) {
          dense_refactor_();
          return;
        }
        max_element_ = std::max(max_element_, current_max);
      } catch (...) {
        ok = false;
      }

      if (!ok) {
        dense_refactor_();
        return;
      }

      ++update_count_;
      if (need_compress_())
        dense_refactor_();
      return;
    }

    const bool refactor_now = (std::abs(alpha) < opt_.alpha_tol) ||
                              (update_count_ >= opt_.refactor_every);
    if (refactor_now) {
      dense_refactor_();
      return;
    }

    etas_.push_back(Eta{j, std::move(u), std::move(z), std::move(w), alpha});
    ++update_count_;
    if (need_compress_())
      dense_refactor_();
  }

  void set_sparse_column_(int col_j, const Eigen::VectorXd &dense) {
    std::vector<Eigen::Triplet<double>> tr;
    tr.reserve(static_cast<size_t>(std::min<int>(dense.size(), 32)));
    for (int r = 0; r < dense.size(); ++r) {
      double v = dense[r];
      if (v != 0.0)
        tr.emplace_back(r, 0, v);
    }
    SparseMat col(m_, 1);
    if (!tr.empty())
      col.setFromTriplets(tr.begin(), tr.end());
    col.makeCompressed();
    Bcols_sparse_[col_j] = std::move(col);
  }

  Eigen::VectorXd apply_etas_solve_(Eigen::VectorXd x) const {
    for (const auto &eta : etas_) {
      double xj = x(eta.j);
      if (xj != 0.0)
        x.noalias() -= eta.z * (xj / eta.alpha);
    }
    return x;
  }

  Eigen::VectorXd apply_etas_solve_T_(Eigen::VectorXd y) const {
    for (const auto &eta : etas_) {
      double uy = eta.u.dot(y);
      if (uy != 0.0)
        y.noalias() -= eta.w * (uy / eta.alpha);
    }
    return y;
  }

  bool need_compress_() const noexcept {
    if (static_cast<int>(etas_.size()) >= opt_.compress_every)
      return true;

    double maxabsz = 0.0;
    for (const auto &e : etas_)
      maxabsz = std::max(maxabsz, e.z.cwiseAbs().maxCoeff());

    if (maxabsz > opt_.z_inf_guard)
      return true;
    return false;
  }

private:
  const DenseMat *A_dense_{nullptr};
  const SparseMat *A_sparse_{nullptr};
  bool A_is_sparse_{false};

  std::vector<Eigen::VectorXd> Bcols_dense_;
  std::vector<SparseMat> Bcols_sparse_;

  MarkowitzLU lu_dense_;
  SparseForrestTomlinLU lu_sparse_;

  int m_{0};
  std::vector<int> basis_;
  Options opt_;
  std::vector<Eta> etas_;
  int update_count_{0};
};
