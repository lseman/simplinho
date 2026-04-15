#pragma once

inline double RevisedSimplex::positive_violation_max_(const Eigen::VectorXd& x, double tol) {
    double worst = 0.0;
    const double scale = (x.size() == 0) ? 1.0 : (1.0 + x.cwiseAbs().maxCoeff());
    const double adjusted_tol = tol * scale;
    for (int i = 0; i < x.size(); ++i) {
        worst = std::max(worst, x(i) - adjusted_tol);
    }
    return worst;
}

inline RevisedSimplex::BasisQuality
RevisedSimplex::evaluate_basis_quality_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c, const std::vector<int>& basis,
                                        double tol) {
    BasisQuality q;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (b.size() != m || c.size() != n) {
        return q;
    }
    if ((int)basis.size() != m || m == 0) {
        if (m == 0 && basis.empty()) {
            q.valid = true;
            q.primal_feasible = true;
            q.dual_feasible = true;
            q.rank = 0;
            q.primal_violation = 0.0;
            q.dual_violation = 0.0;
            q.density = 0.0;
        }
        return q;
    }

    std::vector<char> in_basis(n, 0);
    for (int j : basis) {
        if (j < 0 || j >= n || in_basis[j])
            return q;
        in_basis[j] = 1;
    }

    const Eigen::MatrixXd B = A(Eigen::all, Eigen::VectorXi::Map(basis.data(), m));
    q.density = ((B.array().abs() > 1e-12).cast<double>().sum()) /
                std::max(1.0, static_cast<double>(m) * m);

    Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
    q.rank = lu.rank();
    if (q.rank != m || !lu.isInvertible())
        return q;

    q.valid = true;
    const Eigen::VectorXd xB = lu.solve(b);
    if (xB.size() != m)
        return q;
    {
        Eigen::VectorXd rhs(m);
        for (int i = 0; i < m; ++i)
            rhs(i) = (i % 2 == 0) ? 1.0 : -1.0;
        const Eigen::VectorXd sanity_x = lu.solve(rhs);
        if (sanity_x.size() != m || !sanity_x.allFinite())
            return q;
        const Eigen::VectorXd sanity_residual = rhs - B * sanity_x;
        q.solve_residual =
            sanity_residual.lpNorm<Eigen::Infinity>() / std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
    }
    q.primal_violation = positive_violation_max_(-xB, tol);
    q.primal_feasible = xB.allFinite() && q.primal_violation <= tol;

    Eigen::VectorXd cB(m);
    for (int i = 0; i < m; ++i)
        cB(i) = c(basis[i]);
    Eigen::FullPivLU<Eigen::MatrixXd> luT(B.transpose());
    const Eigen::VectorXd y = luT.solve(cB);
    if (y.size() != m || !y.allFinite())
        return q;

    Eigen::VectorXd neg_rc = Eigen::VectorXd::Zero(n - m);
    int k = 0;
    for (int j = 0; j < n; ++j) {
        if (in_basis[j])
            continue;
        neg_rc(k++) = -(c(j) - A.col(j).dot(y));
    }
    q.dual_violation = positive_violation_max_(neg_rc, tol);
    q.dual_feasible = q.dual_violation <= tol;
    return q;
}

inline RevisedSimplex::BasisQuality
RevisedSimplex::evaluate_basis_quality_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c, const std::vector<int>& basis,
                                        double tol) {
    BasisQuality q;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (b.size() != m || c.size() != n) {
        return q;
    }
    if ((int)basis.size() != m || m == 0) {
        if (m == 0 && basis.empty()) {
            q.valid = true;
            q.primal_feasible = true;
            q.dual_feasible = true;
            q.rank = 0;
            q.primal_violation = 0.0;
            q.dual_violation = 0.0;
            q.density = 0.0;
        }
        return q;
    }

    std::vector<char> in_basis(n, 0);
    for (int j : basis) {
        if (j < 0 || j >= n || in_basis[j])
            return q;
        in_basis[j] = 1;
    }

    SparseMatrix B = sparse_basis_copy_(A, basis);
    q.density = static_cast<double>(B.nonZeros()) / std::max(1.0, static_cast<double>(m) * m);

    Eigen::SparseLU<SparseMatrix> lu;
    lu.analyzePattern(B);
    lu.factorize(B);
    if (lu.info() != Eigen::Success)
        return q;
    q.rank = m;

    q.valid = true;
    const Eigen::VectorXd xB = lu.solve(b);
    if (xB.size() != m)
        return q;
    if (lu.info() != Eigen::Success)
        return q;
    {
        Eigen::VectorXd rhs(m);
        for (int i = 0; i < m; ++i)
            rhs(i) = (i % 2 == 0) ? 1.0 : -1.0;
        const Eigen::VectorXd sanity_x = lu.solve(rhs);
        if (sanity_x.size() != m || lu.info() != Eigen::Success || !sanity_x.allFinite())
            return q;
        const Eigen::VectorXd sanity_residual = rhs - B * sanity_x;
        q.solve_residual =
            sanity_residual.lpNorm<Eigen::Infinity>() / std::max(1.0, rhs.lpNorm<Eigen::Infinity>());
    }
    q.primal_violation = positive_violation_max_(-xB, tol);
    q.primal_feasible = xB.allFinite() && q.primal_violation <= tol;

    Eigen::VectorXd cB(m);
    for (int i = 0; i < m; ++i)
        cB(i) = c(basis[i]);
    const Eigen::VectorXd y = sparse_solveT_from_lu_(B, lu, cB);
    if (y.size() != m || !y.allFinite())
        return q;

    Eigen::VectorXd neg_rc = Eigen::VectorXd::Zero(n - m);
    int k = 0;
    for (int j = 0; j < n; ++j) {
        if (in_basis[j])
            continue;
        double ay = 0.0;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            ay += it.value() * y(it.row());
        }
        neg_rc(k++) = -(c(j) - ay);
    }
    q.dual_violation = positive_violation_max_(neg_rc, tol);
    q.dual_feasible = q.dual_violation <= tol;
    return q;
}

inline bool RevisedSimplex::better_basis_quality_(const CrashSelection& lhs,
                                                  const CrashSelection& rhs, SimplexMode mode) {
    const BasisQuality& a = lhs.quality;
    const BasisQuality& b = rhs.quality;
    if (a.valid != b.valid)
        return a.valid;
    if (a.rank != b.rank)
        return a.rank > b.rank;

    if (mode == SimplexMode::Auto) {
        const int a_feasible =
            static_cast<int>(a.primal_feasible) + static_cast<int>(a.dual_feasible);
        const int b_feasible =
            static_cast<int>(b.primal_feasible) + static_cast<int>(b.dual_feasible);
        if (a_feasible != b_feasible)
            return a_feasible > b_feasible;

        const double a_best = std::min(a.primal_violation, a.dual_violation);
        const double b_best = std::min(b.primal_violation, b.dual_violation);
        if (std::abs(a_best - b_best) > 1e-12)
            return a_best < b_best;

        const double a_total = a.primal_violation + a.dual_violation;
        const double b_total = b.primal_violation + b.dual_violation;
        if (std::abs(a_total - b_total) > 1e-12)
            return a_total < b_total;
    }

    if (std::abs(a.solve_residual - b.solve_residual) > 1e-12)
        return a.solve_residual < b.solve_residual;

    const bool a_primary = (mode == SimplexMode::Dual) ? a.dual_feasible : a.primal_feasible;
    const bool b_primary = (mode == SimplexMode::Dual) ? b.dual_feasible : b.primal_feasible;
    if (a_primary != b_primary)
        return a_primary;

    const bool a_secondary = (mode == SimplexMode::Dual) ? a.primal_feasible : a.dual_feasible;
    const bool b_secondary = (mode == SimplexMode::Dual) ? b.primal_feasible : b.dual_feasible;
    if (a_secondary != b_secondary)
        return a_secondary;

    const double a_primary_violation =
        (mode == SimplexMode::Dual) ? a.dual_violation : a.primal_violation;
    const double b_primary_violation =
        (mode == SimplexMode::Dual) ? b.dual_violation : b.primal_violation;
    if (std::abs(a_primary_violation - b_primary_violation) > 1e-12) {
        return a_primary_violation < b_primary_violation;
    }

    const double a_secondary_violation =
        (mode == SimplexMode::Dual) ? a.primal_violation : a.dual_violation;
    const double b_secondary_violation =
        (mode == SimplexMode::Dual) ? b.primal_violation : b.dual_violation;
    if (std::abs(a_secondary_violation - b_secondary_violation) > 1e-12) {
        return a_secondary_violation < b_secondary_violation;
    }

    if (std::abs(a.density - b.density) > 1e-12)
        return a.density < b.density;
    if (lhs.attempt != rhs.attempt)
        return lhs.attempt < rhs.attempt;
    return std::lexicographical_compare(lhs.basis.begin(), lhs.basis.end(), rhs.basis.begin(),
                                        rhs.basis.end());
}

inline std::string RevisedSimplex::lower_copy_(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline RevisedSimplex::CrashStyle RevisedSimplex::parse_crash_style_(const std::string& strategy) {
    const std::string key = lower_copy_(strategy);
    if (key.empty() || key == "hybrid" || key == "auto") {
        return CrashStyle::Hybrid;
    }
    if (key == "repair" || key == "repair_warm_start") {
        return CrashStyle::Repair;
    }
    if (key == "sprint")
        return CrashStyle::Sprint;
    if (key == "crash_ii" || key == "crash-ii" || key == "crash2") {
        return CrashStyle::CrashII;
    }
    if (key == "crash_iii" || key == "crash-iii" || key == "crash3") {
        return CrashStyle::CrashIII;
    }
    return CrashStyle::Hybrid;
}

inline const char* RevisedSimplex::crash_style_name_(CrashStyle style) {
    switch (style) {
        case CrashStyle::Repair:
            return "repair";
        case CrashStyle::Sprint:
            return "sprint";
        case CrashStyle::CrashII:
            return "crash_ii";
        case CrashStyle::CrashIII:
            return "crash_iii";
        case CrashStyle::Hybrid:
        default:
            return "hybrid";
    }
}

inline RevisedSimplex::CrashAttemptConfig
RevisedSimplex::crash_attempt_config_(const RevisedSimplexOptions& opt, int attempt) {
    CrashAttemptConfig cfg;
    const double base = std::clamp(opt.crash_markowitz_tol, 1e-3, 0.95);
    CrashStyle style = parse_crash_style_(opt.crash_strategy);
    if (style == CrashStyle::Hybrid) {
        switch (attempt % 4) {
            case 0:
                style = CrashStyle::Repair;
                break;
            case 1:
                style = CrashStyle::Sprint;
                break;
            case 2:
                style = CrashStyle::CrashII;
                break;
            default:
                style = CrashStyle::CrashIII;
                break;
        }
    }

    cfg.style = style;
    cfg.style_name = crash_style_name_(style);
    switch (style) {
        case CrashStyle::Repair:
            cfg.markowitz_threshold = std::max(1e-3, 0.45 * base);
            cfg.cost_penalty = 0.02;
            cfg.rhs_bonus = 0.45;
            cfg.dense_penalty = 0.25;
            cfg.coverage_weight = 1.20;
            cfg.seed_penalty = 6.0;
            cfg.local_search_passes = 2;
            cfg.max_swap_candidates = 12;
            cfg.prefer_seed_columns = true;
            break;
        case CrashStyle::Sprint:
            cfg.markowitz_threshold = std::max(1e-3, 0.60 * base);
            cfg.cost_penalty = 0.02;
            cfg.rhs_bonus = 0.20;
            cfg.dense_penalty = 0.35;
            cfg.coverage_weight = 1.40;
            cfg.local_search_passes = 1;
            cfg.max_swap_candidates = 8;
            break;
        case CrashStyle::CrashII:
            cfg.markowitz_threshold = base;
            cfg.cost_penalty = 0.05;
            cfg.rhs_bonus = 0.25;
            cfg.dense_penalty = 0.50;
            cfg.coverage_weight = 1.00;
            cfg.local_search_passes = 1;
            cfg.max_swap_candidates = 10;
            break;
        case CrashStyle::CrashIII:
            cfg.markowitz_threshold = std::min(0.95, 1.6 * base);
            cfg.cost_penalty = 0.08;
            cfg.rhs_bonus = 0.15;
            cfg.dense_penalty = 0.65;
            cfg.coverage_weight = 0.80;
            cfg.local_search_passes = 2;
            cfg.max_swap_candidates = 14;
            break;
        case CrashStyle::Hybrid:
        default:
            break;
    }
    cfg.jitter = 1e-6 * static_cast<double>(attempt + 1);
    return cfg;
}

inline void RevisedSimplex::mark_pivot_row_(const Eigen::MatrixXd& A, int col, int pivot_row_hint,
                                            std::vector<char>& used_row) {
    if (pivot_row_hint >= 0 && pivot_row_hint < (int)used_row.size() && !used_row[pivot_row_hint]) {
        used_row[pivot_row_hint] = 1;
        return;
    }

    int best_row = -1;
    double best_abs = 0.0;
    for (int i = 0; i < A.rows(); ++i) {
        if (used_row[i])
            continue;
        const double aa = std::abs(A(i, col));
        if (aa > best_abs) {
            best_abs = aa;
            best_row = i;
        }
    }
    if (best_row >= 0)
        used_row[best_row] = 1;
}

inline void RevisedSimplex::mark_pivot_row_(const SparseMatrix& A, int col, int pivot_row_hint,
                                            std::vector<char>& used_row) {
    if (pivot_row_hint >= 0 && pivot_row_hint < (int)used_row.size() && !used_row[pivot_row_hint]) {
        used_row[pivot_row_hint] = 1;
        return;
    }

    int best_row = -1;
    double best_abs = 0.0;
    for (SparseMatrix::InnerIterator it(A, col); it; ++it) {
        if (used_row[it.row()])
            continue;
        const double aa = std::abs(it.value());
        if (aa > best_abs) {
            best_abs = aa;
            best_row = it.row();
        }
    }
    if (best_row >= 0)
        used_row[best_row] = 1;
}

inline bool RevisedSimplex::try_add_basis_column_(const Eigen::MatrixXd& A, std::vector<int>& basis,
                                                  std::vector<char>& used_row,
                                                  std::vector<char>& used_col, int& current_rank,
                                                  int col, int pivot_row_hint, double tol) {
    const int n = static_cast<int>(A.cols());
    if (col < 0 || col >= n || used_col[col])
        return false;

    const int m = static_cast<int>(A.rows());
    const auto has_unused_row = [&](int column) {
        for (int i = 0; i < m; ++i) {
            if (!used_row[i] && std::abs(A(i, column)) > 1e-12)
                return true;
        }
        return false;
    };

    if (basis.empty() || has_unused_row(col)) {
        used_col[col] = 1;
        basis.push_back(col);
        current_rank = static_cast<int>(basis.size());
        mark_pivot_row_(A, col, pivot_row_hint, used_row);
        return true;
    }

    const Eigen::VectorXd col_vec = A.col(col);
    Eigen::MatrixXd B(m, static_cast<int>(basis.size()));
    for (int k = 0; k < static_cast<int>(basis.size()); ++k) {
        B.col(k) = A.col(basis[k]);
    }

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
    if (qr.rank() != current_rank)
        return false;

    const Eigen::VectorXd x = qr.solve(col_vec);
    const double residual = (B * x - col_vec).norm();
    if (residual <= tol * (1.0 + col_vec.lpNorm<Eigen::Infinity>()))
        return false;

    used_col[col] = 1;
    basis.push_back(col);
    current_rank = static_cast<int>(basis.size());
    mark_pivot_row_(A, col, pivot_row_hint, used_row);
    return true;
}

inline bool RevisedSimplex::try_add_basis_column_(const SparseMatrix& A, std::vector<int>& basis,
                                                  std::vector<char>& used_row,
                                                  std::vector<char>& used_col, int& current_rank,
                                                  int col, int pivot_row_hint, double tol) {
    const int n = static_cast<int>(A.cols());
    if (col < 0 || col >= n || used_col[col])
        return false;

    const int m = static_cast<int>(A.rows());
    const auto has_unused_row = [&](int column) {
        for (SparseMatrix::InnerIterator it(A, column); it; ++it) {
            if (!used_row[it.row()] && std::abs(it.value()) > 1e-12)
                return true;
        }
        return false;
    };

    if (basis.empty() || has_unused_row(col)) {
        used_col[col] = 1;
        basis.push_back(col);
        current_rank = static_cast<int>(basis.size());
        mark_pivot_row_(A, col, pivot_row_hint, used_row);
        return true;
    }

    Eigen::VectorXd col_vec = Eigen::VectorXd::Zero(m);
    for (SparseMatrix::InnerIterator it(A, col); it; ++it)
        col_vec(it.row()) = it.value();

    SparseMatrix B = sparse_basis_copy_(A, basis);
    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
    qr.compute(B);
    if (qr.info() != Eigen::Success || qr.rank() != current_rank)
        return false;

    const Eigen::VectorXd x = qr.solve(col_vec);
    if (x.size() != current_rank || qr.info() != Eigen::Success || !x.allFinite())
        return false;

    const Eigen::VectorXd residual_vec = B * x - col_vec;
    const double residual = residual_vec.lpNorm<Eigen::Infinity>();
    if (residual <= tol * (1.0 + col_vec.lpNorm<Eigen::Infinity>()))
        return false;

    used_col[col] = 1;
    basis.push_back(col);
    current_rank = static_cast<int>(basis.size());
    mark_pivot_row_(A, col, pivot_row_hint, used_row);
    return true;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_slack_like_column_(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        int pivot_row = -1;
        int nnz = 0;
        double pivot = 0.0;
        double off_sum = 0.0;
        for (int i = 0; i < m; ++i) {
            const double aij = A(i, j);
            if (std::abs(aij) <= 1e-12)
                continue;
            ++nnz;
            if (used_row[i]) {
                off_sum += std::abs(aij);
                continue;
            }
            if (std::abs(aij) > std::abs(pivot)) {
                if (pivot_row >= 0)
                    off_sum += std::abs(pivot);
                pivot_row = i;
                pivot = aij;
            } else {
                off_sum += std::abs(aij);
            }
        }
        if (pivot_row < 0 || std::abs(pivot) <= 1e-12)
            continue;

        const bool exact_unit = (nnz == 1 && std::abs(std::abs(pivot) - 1.0) <= 1e-10);
        const bool slack_like = (nnz == 1) || (off_sum <= 1e-10);
        if (!slack_like)
            continue;

        int row_nnz = 0;
        for (int jj = 0; jj < n; ++jj) {
            if (used_col[jj] || jj == j)
                continue;
            if (std::abs(A(pivot_row, jj)) > 1e-12)
                ++row_nnz;
        }

        double score = exact_unit ? 1e6 : 1e5;
        score += 1e3 / (1.0 + off_sum);
        score += 15.0 / (1.0 + static_cast<double>(nnz));
        score += 15.0 / (1.0 + std::abs(std::abs(pivot) - 1.0));
        score += 20.0 / (1.0 + std::abs(c(j)) / c_scale);
        if (b(pivot_row) >= -1e-10)
            score += 50.0;
        score += 10.0 / (1.0 + static_cast<double>(row_nnz));
        score -= 0.02 * (std::abs(c(j)) / c_scale);
        score -= 0.01 * static_cast<double>(j);

        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_slack_like_column_(
    const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col) {
    CrashCandidate best;
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        int pivot_row = -1;
        int nnz = 0;
        double pivot = 0.0;
        double off_sum = 0.0;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            const double aij = it.value();
            if (std::abs(aij) <= 1e-12)
                continue;
            ++nnz;
            if (used_row[it.row()]) {
                off_sum += std::abs(aij);
                continue;
            }
            if (std::abs(aij) > std::abs(pivot)) {
                if (pivot_row >= 0)
                    off_sum += std::abs(pivot);
                pivot_row = it.row();
                pivot = aij;
            } else {
                off_sum += std::abs(aij);
            }
        }
        if (pivot_row < 0 || std::abs(pivot) <= 1e-12)
            continue;

        const bool exact_unit = (nnz == 1 && std::abs(std::abs(pivot) - 1.0) <= 1e-10);
        const bool slack_like = (nnz == 1) || (off_sum <= 1e-10);
        if (!slack_like)
            continue;

        int row_nnz = 0;
        for (int jj = 0; jj < A.cols(); ++jj) {
            if (used_col[jj] || jj == j)
                continue;
            for (SparseMatrix::InnerIterator it(A, jj); it; ++it) {
                if (it.row() == pivot_row && std::abs(it.value()) > 1e-12) {
                    ++row_nnz;
                    break;
                }
            }
        }

        double score = exact_unit ? 1e6 : 1e5;
        score += 1e3 / (1.0 + off_sum);
        score += 15.0 / (1.0 + static_cast<double>(nnz));
        score += 15.0 / (1.0 + std::abs(std::abs(pivot) - 1.0));
        score += 20.0 / (1.0 + std::abs(c(j)) / c_scale);
        if (pivot_row < b.size() && b(pivot_row) >= -1e-10)
            score += 50.0;
        score += 10.0 / (1.0 + static_cast<double>(row_nnz));
        score -= 0.02 * (std::abs(c(j)) / c_scale);
        score -= 0.01 * static_cast<double>(j);

        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_free_like_column_(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col,
    const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        int pivot_row = -1;
        double pivot_abs = 0.0;
        int nnz = 0;
        for (int i = 0; i < m; ++i) {
            const double aij = std::abs(A(i, j));
            if (aij <= 1e-12)
                continue;
            ++nnz;
            if (used_row[i])
                continue;
            if (aij > pivot_abs) {
                pivot_abs = aij;
                pivot_row = i;
            }
        }
        if (pivot_row < 0 || pivot_abs <= 1e-12)
            continue;

        const double score = 200.0 / (1.0 + static_cast<double>(nnz)) + 30.0 * pivot_abs -
                             25.0 * (std::abs(c(j)) / c_scale) - 0.01 * static_cast<double>(j) +
                             cfg.jitter;
        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_free_like_column_(
    const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col,
    const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        int pivot_row = -1;
        double pivot_abs = 0.0;
        int nnz = 0;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            const double aij = std::abs(it.value());
            if (aij <= 1e-12)
                continue;
            ++nnz;
            if (used_row[it.row()])
                continue;
            if (aij > pivot_abs) {
                pivot_abs = aij;
                pivot_row = it.row();
            }
        }
        if (pivot_row < 0 || pivot_abs <= 1e-12)
            continue;

        const double score = 200.0 / (1.0 + static_cast<double>(nnz)) + 30.0 * pivot_abs -
                             25.0 * (std::abs(c(j)) / c_scale) - 0.01 * static_cast<double>(j) +
                             cfg.jitter;
        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline RevisedSimplex::CrashCandidate
RevisedSimplex::choose_sprint_column_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                      const Eigen::VectorXd& c, const std::vector<char>& used_row,
                                      const std::vector<char>& used_col,
                                      const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;

        int pivot_row = -1;
        double pivot_abs = 0.0;
        double uncovered_sum = 0.0;
        int total_nnz = 0;
        for (int i = 0; i < m; ++i) {
            const double aa = std::abs(A(i, j));
            if (aa <= 1e-12)
                continue;
            ++total_nnz;
            if (used_row[i])
                continue;
            uncovered_sum += aa;
            if (aa > pivot_abs) {
                pivot_abs = aa;
                pivot_row = i;
            }
        }
        if (pivot_row < 0 || pivot_abs <= 1e-12)
            continue;

        const double coverage = uncovered_sum / std::max(1e-12, A.col(j).lpNorm<1>());
        const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
        const double rhs_bonus = (b(pivot_row) >= -1e-10) ? cfg.rhs_bonus : 0.0;
        const double cost_penalty = cfg.cost_penalty * (std::abs(c(j)) / c_scale);
        const double jitter = cfg.jitter * std::cos(static_cast<double>((j + 1) * (pivot_row + 1)));
        const double score = 90.0 * cfg.coverage_weight * coverage + 25.0 * sparsity_bonus +
                             5.0 * pivot_abs + rhs_bonus - cost_penalty -
                             0.001 * static_cast<double>(j) + jitter;
        if (score > best.score)
            best = {j, pivot_row, score};
    }
    return best;
}

inline RevisedSimplex::CrashCandidate
RevisedSimplex::choose_sprint_column_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                      const Eigen::VectorXd& c, const std::vector<char>& used_row,
                                      const std::vector<char>& used_col,
                                      const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;

        int pivot_row = -1;
        double pivot_abs = 0.0;
        double uncovered_sum = 0.0;
        double total_sum = 0.0;
        int total_nnz = 0;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            const double aa = std::abs(it.value());
            if (aa <= 1e-12)
                continue;
            ++total_nnz;
            total_sum += aa;
            if (used_row[it.row()])
                continue;
            uncovered_sum += aa;
            if (aa > pivot_abs) {
                pivot_abs = aa;
                pivot_row = it.row();
            }
        }
        if (pivot_row < 0 || pivot_abs <= 1e-12)
            continue;

        const double coverage = uncovered_sum / std::max(1e-12, total_sum);
        const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
        const double rhs_bonus =
            (pivot_row < b.size() && b(pivot_row) >= -1e-10) ? cfg.rhs_bonus : 0.0;
        const double cost_penalty = cfg.cost_penalty * (std::abs(c(j)) / c_scale);
        const double jitter = cfg.jitter * std::cos(static_cast<double>((j + 1) * (pivot_row + 1)));
        const double score = 90.0 * cfg.coverage_weight * coverage + 25.0 * sparsity_bonus +
                             5.0 * pivot_abs + rhs_bonus - cost_penalty -
                             0.001 * static_cast<double>(j) + jitter;
        if (score > best.score)
            best = {j, pivot_row, score};
    }
    return best;
}

inline std::vector<int> RevisedSimplex::find_logical_basis_(const Eigen::MatrixXd& A) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (m == 0)
        return {};
    if (n < m)
        return {};

    std::vector<int> basis(m, -1);
    for (int j = 0; j < n; ++j) {
        int pivot_row = -1;
        bool exact_unit = true;
        for (int i = 0; i < m; ++i) {
            const double aij = A(i, j);
            if (std::abs(aij) <= 1e-12)
                continue;
            if (pivot_row >= 0 || std::abs(std::abs(aij) - 1.0) > 1e-10) {
                exact_unit = false;
                break;
            }
            pivot_row = i;
        }
        if (!exact_unit || pivot_row < 0 || basis[pivot_row] >= 0)
            continue;
        basis[pivot_row] = j;
    }
    if (std::find(basis.begin(), basis.end(), -1) != basis.end())
        return {};
    return basis;
}

inline std::vector<int> RevisedSimplex::find_logical_basis_(const SparseMatrix& A) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (m == 0)
        return {};
    if (n < m)
        return {};

    std::vector<int> basis(m, -1);
    for (int j = 0; j < n; ++j) {
        int pivot_row = -1;
        bool exact_unit = true;
        int nnz = 0;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            if (std::abs(it.value()) <= 1e-12)
                continue;
            ++nnz;
            if (nnz > 1 || std::abs(std::abs(it.value()) - 1.0) > 1e-10) {
                exact_unit = false;
                break;
            }
            pivot_row = it.row();
        }
        if (!exact_unit || pivot_row < 0 || basis[pivot_row] >= 0)
            continue;
        basis[pivot_row] = j;
    }
    if (std::find(basis.begin(), basis.end(), -1) != basis.end())
        return {};
    return basis;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_triangular_column_(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col,
    const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    const double tol_nz = 1e-12;
    std::vector<int> row_nnz(m, 0);
    std::vector<int> col_nnz(n, 0);
    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        for (int i = 0; i < m; ++i) {
            if (used_row[i])
                continue;
            if (std::abs(A(i, j)) > tol_nz) {
                ++row_nnz[i];
                ++col_nnz[j];
            }
        }
    }

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;

        int pivot_row = -1;
        double pivot_abs = 0.0;
        double uncovered_sum = 0.0;
        double covered_sum = 0.0;
        int uncovered_nnz = 0;
        int total_nnz = 0;
        int best_row_degree = std::numeric_limits<int>::max();
        for (int i = 0; i < m; ++i) {
            const double aij = A(i, j);
            const double aa = std::abs(aij);
            if (aa <= 1e-12)
                continue;
            ++total_nnz;
            if (used_row[i]) {
                covered_sum += aa;
                continue;
            }
            ++uncovered_nnz;
            uncovered_sum += aa;
            const int row_degree = row_nnz[i];
            if (row_degree < best_row_degree || (row_degree == best_row_degree && aa > pivot_abs)) {
                pivot_row = i;
                pivot_abs = aa;
                best_row_degree = row_degree;
            }
        }
        if (pivot_row < 0 || pivot_abs <= 1e-12)
            continue;

        if (pivot_abs + 1e-12 < cfg.markowitz_threshold * row_nnz[pivot_row])
            continue;

        const double dominance = pivot_abs / std::max(1e-12, uncovered_sum);
        const double triangularity = pivot_abs / std::max(1e-12, covered_sum + uncovered_sum);
        const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
        const double row_bonus = 5.0 / (1.0 + static_cast<double>(row_nnz[pivot_row]));
        const double col_bonus = 5.0 / (1.0 + static_cast<double>(col_nnz[j]));
        const double rhs_bonus = (b(pivot_row) >= -1e-10) ? cfg.rhs_bonus : 0.0;
        const double cost_penalty = cfg.cost_penalty * (std::abs(c(j)) / c_scale);
        const double markowitz_penalty = static_cast<double>(std::max(0, row_nnz[pivot_row] - 1) *
                                                             std::max(0, uncovered_nnz - 1));
        const double markowitz_bonus = 1.0 / (1.0 + markowitz_penalty);
        const double dense_penalty = cfg.dense_penalty * (covered_sum / std::max(1e-12, pivot_abs));
        const double jitter = cfg.jitter * std::sin(static_cast<double>((j + 1) * (pivot_row + 1)));

        const double score = 100.0 * dominance * cfg.coverage_weight + 30.0 * triangularity +
                             25.0 * markowitz_bonus + 15.0 * sparsity_bonus + 10.0 * row_bonus +
                             10.0 * col_bonus + rhs_bonus - cost_penalty - dense_penalty -
                             0.001 * static_cast<double>(j) + jitter;
        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline RevisedSimplex::CrashCandidate RevisedSimplex::choose_triangular_column_(
    const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
    const std::vector<char>& used_row, const std::vector<char>& used_col,
    const CrashAttemptConfig& cfg) {
    CrashCandidate best;
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    double c_scale = 1.0;
    if (c.size() > 0)
        c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

    const double tol_nz = 1e-12;
    std::vector<int> row_nnz(m, 0);
    std::vector<int> col_nnz(n, 0);
    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            if (used_row[it.row()] || std::abs(it.value()) <= tol_nz)
                continue;
            ++row_nnz[it.row()];
            ++col_nnz[j];
        }
    }

    for (int j = 0; j < n; ++j) {
        if (used_col[j])
            continue;

        int pivot_row = -1;
        double pivot_abs = 0.0;
        double uncovered_sum = 0.0;
        double covered_sum = 0.0;
        int uncovered_nnz = 0;
        int total_nnz = 0;
        int best_row_degree = std::numeric_limits<int>::max();
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            const double aa = std::abs(it.value());
            if (aa <= tol_nz)
                continue;
            ++total_nnz;
            if (used_row[it.row()]) {
                covered_sum += aa;
                continue;
            }
            ++uncovered_nnz;
            uncovered_sum += aa;
            const int row_degree = row_nnz[it.row()];
            if (row_degree < best_row_degree || (row_degree == best_row_degree && aa > pivot_abs)) {
                pivot_row = it.row();
                pivot_abs = aa;
                best_row_degree = row_degree;
            }
        }
        if (pivot_row < 0 || pivot_abs <= tol_nz)
            continue;

        if (pivot_abs + tol_nz < cfg.markowitz_threshold * row_nnz[pivot_row])
            continue;

        const double dominance = pivot_abs / std::max(1e-12, uncovered_sum);
        const double triangularity = pivot_abs / std::max(1e-12, covered_sum + uncovered_sum);
        const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
        const double row_bonus = 5.0 / (1.0 + static_cast<double>(row_nnz[pivot_row]));
        const double col_bonus = 5.0 / (1.0 + static_cast<double>(col_nnz[j]));
        const double rhs_bonus = (pivot_row < b.size() && b(pivot_row) >= -1e-10) ? cfg.rhs_bonus
                                                                                    : 0.0;
        const double cost_penalty = cfg.cost_penalty * (std::abs(c(j)) / c_scale);
        const double markowitz_penalty = static_cast<double>(std::max(0, row_nnz[pivot_row] - 1) *
                                                             std::max(0, uncovered_nnz - 1));
        const double markowitz_bonus = 1.0 / (1.0 + markowitz_penalty);
        const double dense_penalty = cfg.dense_penalty * (covered_sum / std::max(1e-12, pivot_abs));
        const double jitter = cfg.jitter * std::sin(static_cast<double>((j + 1) * (pivot_row + 1)));

        const double score = 100.0 * dominance * cfg.coverage_weight + 30.0 * triangularity +
                             25.0 * markowitz_bonus + 15.0 * sparsity_bonus + 10.0 * row_bonus +
                             10.0 * col_bonus + rhs_bonus - cost_penalty - dense_penalty -
                             0.001 * static_cast<double>(j) + jitter;
        if (score > best.score) {
            best = {j, pivot_row, score};
        }
    }
    return best;
}

inline std::vector<int> RevisedSimplex::rank_remaining_columns_(const Eigen::MatrixXd& A,
                                                                const Eigen::VectorXd& c,
                                                                const std::vector<char>& used_col,
                                                                const CrashAttemptConfig& cfg) {
    std::vector<int> ranked;
    ranked.reserve(A.cols());
    const int n = static_cast<int>(A.cols());
    std::vector<double> col_nnz(n, 0.0);
    for (int j = 0; j < n; ++j) {
        col_nnz[j] = (A.col(j).array().abs() > 1e-12).cast<double>().sum();
    }

    for (int j = 0; j < n; ++j) {
        if (!used_col[j])
            ranked.push_back(j);
    }
    std::sort(ranked.begin(), ranked.end(), [&](int a, int b_idx) {
        const double score_a = col_nnz[a] + cfg.cost_penalty * std::abs(c(a)) +
                               cfg.jitter * std::sin(static_cast<double>(a + 1));
        const double score_b = col_nnz[b_idx] + cfg.cost_penalty * std::abs(c(b_idx)) +
                               cfg.jitter * std::sin(static_cast<double>(b_idx + 1));
        if (std::abs(score_a - score_b) > 1e-12)
            return score_a < score_b;
        return a < b_idx;
    });
    return ranked;
}

inline std::vector<int> RevisedSimplex::rank_remaining_columns_(const SparseMatrix& A,
                                                                const Eigen::VectorXd& c,
                                                                const std::vector<char>& used_col,
                                                                const CrashAttemptConfig& cfg) {
    std::vector<int> ranked;
    ranked.reserve(A.cols());
    const int n = static_cast<int>(A.cols());
    std::vector<double> col_nnz(n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            if (std::abs(it.value()) > 1e-12)
                col_nnz[j] += 1.0;
        }
    }

    for (int j = 0; j < n; ++j) {
        if (!used_col[j])
            ranked.push_back(j);
    }
    std::sort(ranked.begin(), ranked.end(), [&](int a, int b_idx) {
        const double score_a = col_nnz[a] + cfg.cost_penalty * std::abs(c(a)) +
                               cfg.jitter * std::sin(static_cast<double>(a + 1));
        const double score_b = col_nnz[b_idx] + cfg.cost_penalty * std::abs(c(b_idx)) +
                               cfg.jitter * std::sin(static_cast<double>(b_idx + 1));
        if (std::abs(score_a - score_b) > 1e-12)
            return score_a < score_b;
        return a < b_idx;
    });
    return ranked;
}

inline std::vector<int>
RevisedSimplex::improve_basis_by_swaps_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c, std::vector<int> basis,
                                        const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                                        std::optional<std::vector<int>> seed_basis) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if ((int)basis.size() != m || cfg.local_search_passes <= 0) {
        return basis;
    }

    std::vector<int> col_nnz(n, 0);
    for (int j = 0; j < n; ++j) {
        col_nnz[j] = (A.col(j).array().abs() > 1e-12).cast<int>().sum();
    }
    std::vector<double> col_weight(n, 0.0);
    for (int j = 0; j < n; ++j) {
        col_weight[j] = cfg.cost_penalty * std::abs(c(j));
    }

    std::vector<char> seeded(n, 0);
    if (seed_basis) {
        for (int j : *seed_basis) {
            if (j >= 0 && j < n)
                seeded[j] = 1;
        }
    }

    CrashSelection best;
    best.basis = basis;
    best.quality = evaluate_basis_quality_(A, b, c, basis, tol);

    auto promising_swap = [&](int entering, int leaving) {
        const double score_enter = col_nnz[entering] + col_weight[entering];
        const double score_leave = col_nnz[leaving] + col_weight[leaving];
        return score_enter <= score_leave;
    };

    for (int pass = 0; pass < cfg.local_search_passes; ++pass) {
        std::vector<char> in_basis(n, 0);
        for (int j : basis)
            if (j >= 0 && j < n)
                in_basis[j] = 1;

        std::vector<int> nonbasic;
        nonbasic.reserve(std::max(0, n - m));
        for (int j = 0; j < n; ++j)
            if (!in_basis[j])
                nonbasic.push_back(j);
        std::sort(nonbasic.begin(), nonbasic.end(), [&](int a, int b_idx) {
            const double score_a = col_nnz[a] + col_weight[a];
            const double score_b = col_nnz[b_idx] + col_weight[b_idx];
            if (std::abs(score_a - score_b) > 1e-12)
                return score_a < score_b;
            return a < b_idx;
        });
        if ((int)nonbasic.size() > cfg.max_swap_candidates) {
            nonbasic.resize(cfg.max_swap_candidates);
        }

        std::vector<std::pair<double, int>> position_priority;
        position_priority.reserve(m);
        for (int p = 0; p < m; ++p) {
            const int col = basis[p];
            double score = col_nnz[col] + col_weight[col];
            if (seeded[col])
                score *= 0.75;
            position_priority.emplace_back(score, p);
        }
        std::sort(position_priority.begin(), position_priority.end(), std::greater<>());
        const int position_limit = std::min(m, cfg.max_swap_candidates);
        std::vector<int> positions;
        positions.reserve(position_limit);
        for (int idx = 0; idx < position_limit; ++idx)
            positions.push_back(position_priority[idx].second);

        bool improved = false;
        for (int entering : nonbasic) {
            for (int pos : positions) {
                if (!promising_swap(entering, basis[pos]))
                    continue;

                std::vector<int> cand = basis;
                cand[pos] = entering;

                CrashSelection trial;
                trial.basis = std::move(cand);
                trial.quality = evaluate_basis_quality_(A, b, c, trial.basis, tol);
                if (!trial.quality.valid)
                    continue;
                if (!better_basis_quality_(trial, best, mode))
                    continue;
                basis = trial.basis;
                best = std::move(trial);
                improved = true;
                break;
            }
            if (improved)
                break;
        }
        if (!improved)
            break;
    }

    return basis;
}

inline std::vector<int>
RevisedSimplex::improve_basis_by_swaps_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                        const Eigen::VectorXd& c, std::vector<int> basis,
                                        const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
                                        std::optional<std::vector<int>> seed_basis) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if ((int)basis.size() != m || cfg.local_search_passes <= 0) {
        return basis;
    }

    std::vector<int> col_nnz(n, 0);
    for (int j = 0; j < n; ++j) {
        for (SparseMatrix::InnerIterator it(A, j); it; ++it) {
            if (std::abs(it.value()) > 1e-12)
                ++col_nnz[j];
        }
    }
    std::vector<double> col_weight(n, 0.0);
    for (int j = 0; j < n; ++j) {
        col_weight[j] = cfg.cost_penalty * std::abs(c(j));
    }

    std::vector<char> seeded(n, 0);
    if (seed_basis) {
        for (int j : *seed_basis) {
            if (j >= 0 && j < n)
                seeded[j] = 1;
        }
    }

    CrashSelection best;
    best.basis = basis;
    best.quality = evaluate_basis_quality_(A, b, c, basis, tol);

    auto promising_swap = [&](int entering, int leaving) {
        const double score_enter = col_nnz[entering] + col_weight[entering];
        const double score_leave = col_nnz[leaving] + col_weight[leaving];
        return score_enter <= score_leave;
    };

    for (int pass = 0; pass < cfg.local_search_passes; ++pass) {
        std::vector<char> in_basis(n, 0);
        for (int j : basis)
            if (j >= 0 && j < n)
                in_basis[j] = 1;

        std::vector<int> nonbasic;
        nonbasic.reserve(std::max(0, n - m));
        for (int j = 0; j < n; ++j)
            if (!in_basis[j])
                nonbasic.push_back(j);
        std::sort(nonbasic.begin(), nonbasic.end(), [&](int a, int b_idx) {
            const double score_a = col_nnz[a] + col_weight[a];
            const double score_b = col_nnz[b_idx] + col_weight[b_idx];
            if (std::abs(score_a - score_b) > 1e-12)
                return score_a < score_b;
            return a < b_idx;
        });
        if ((int)nonbasic.size() > cfg.max_swap_candidates) {
            nonbasic.resize(cfg.max_swap_candidates);
        }

        std::vector<std::pair<double, int>> position_priority;
        position_priority.reserve(m);
        for (int p = 0; p < m; ++p) {
            const int col = basis[p];
            double score = col_nnz[col] + col_weight[col];
            if (seeded[col])
                score *= 0.75;
            position_priority.emplace_back(score, p);
        }
        std::sort(position_priority.begin(), position_priority.end(), std::greater<>());
        const int position_limit = std::min(m, cfg.max_swap_candidates);
        std::vector<int> positions;
        positions.reserve(position_limit);
        for (int idx = 0; idx < position_limit; ++idx)
            positions.push_back(position_priority[idx].second);

        bool improved = false;
        for (int entering : nonbasic) {
            for (int pos : positions) {
                if (!promising_swap(entering, basis[pos]))
                    continue;

                std::vector<int> cand = basis;
                cand[pos] = entering;

                CrashSelection trial;
                trial.basis = std::move(cand);
                trial.quality = evaluate_basis_quality_(A, b, c, trial.basis, tol);
                if (!trial.quality.valid)
                    continue;
                if (!better_basis_quality_(trial, best, mode))
                    continue;
                basis = trial.basis;
                best = std::move(trial);
                improved = true;
                break;
            }
            if (improved)
                break;
        }
        if (!improved)
            break;
    }

    return basis;
}

inline std::vector<int>
RevisedSimplex::build_basis_attempt_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                     const Eigen::VectorXd& c, const CrashAttemptConfig& cfg,
                                     double tol, SimplexMode mode,
                                     std::optional<std::vector<int>> seed_basis) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (m == 0)
        return {};
    if (n < m)
        return {};

    std::vector<int> basis;
    basis.reserve(m);
    std::vector<char> used_row(m, 0), used_col(n, 0);
    int current_rank = 0;

    if (seed_basis) {
        for (int j : *seed_basis) {
            if ((int)basis.size() == m)
                break;
            (void)try_add_basis_column_(A, basis, used_row, used_col, current_rank, j, -1, tol);
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_free_like_column_(A, b, c, used_row, used_col, cfg);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            break;
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_slack_like_column_(A, b, c, used_row, used_col);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            continue;
        }
    }

    if (cfg.style == CrashStyle::Sprint) {
        while ((int)basis.size() < m) {
            const CrashCandidate cand = choose_sprint_column_(A, b, c, used_row, used_col, cfg);
            if (cand.col < 0)
                break;
            if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                       cand.pivot_row, tol)) {
                continue;
            }
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_triangular_column_(A, b, c, used_row, used_col, cfg);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            continue;
        }
    }

    if ((int)basis.size() < m) {
        for (int j : rank_remaining_columns_(A, c, used_col, cfg)) {
            if ((int)basis.size() == m)
                break;
            (void)try_add_basis_column_(A, basis, used_row, used_col, current_rank, j, -1, tol);
        }
    }

    if ((int)basis.size() != m)
        return {};
    return improve_basis_by_swaps_(A, b, c, std::move(basis), cfg, tol, mode, seed_basis);
}

inline std::vector<int>
RevisedSimplex::build_basis_attempt_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                     const Eigen::VectorXd& c, const CrashAttemptConfig& cfg,
                                     double tol, SimplexMode mode,
                                     std::optional<std::vector<int>> seed_basis) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    if (m == 0)
        return {};
    if (n < m)
        return {};

    std::vector<int> basis;
    basis.reserve(m);
    std::vector<char> used_row(m, 0), used_col(n, 0);
    int current_rank = 0;

    if (seed_basis) {
        for (int j : *seed_basis) {
            if ((int)basis.size() == m)
                break;
            (void)try_add_basis_column_(A, basis, used_row, used_col, current_rank, j, -1, tol);
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_free_like_column_(A, b, c, used_row, used_col, cfg);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            break;
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_slack_like_column_(A, b, c, used_row, used_col);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            continue;
        }
    }

    if (cfg.style == CrashStyle::Sprint) {
        while ((int)basis.size() < m) {
            const CrashCandidate cand = choose_sprint_column_(A, b, c, used_row, used_col, cfg);
            if (cand.col < 0)
                break;
            if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                       cand.pivot_row, tol)) {
                continue;
            }
        }
    }

    while ((int)basis.size() < m) {
        const CrashCandidate cand = choose_triangular_column_(A, b, c, used_row, used_col, cfg);
        if (cand.col < 0)
            break;
        if (!try_add_basis_column_(A, basis, used_row, used_col, current_rank, cand.col,
                                   cand.pivot_row, tol)) {
            continue;
        }
    }

    if ((int)basis.size() < m) {
        for (int j : rank_remaining_columns_(A, c, used_col, cfg)) {
            if ((int)basis.size() == m)
                break;
            (void)try_add_basis_column_(A, basis, used_row, used_col, current_rank, j, -1, tol);
        }
    }

    if ((int)basis.size() != m)
        return {};
    return improve_basis_by_swaps_(A, b, c, std::move(basis), cfg, tol, mode, seed_basis);
}

inline RevisedSimplex::CrashSelection
RevisedSimplex::choose_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                      const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                                      std::optional<std::vector<int>> seed_basis) {
    CrashSelection best;

    auto consider = [&](std::vector<int> candidate, std::string source, int attempt) {
        if (candidate.empty() && A.rows() != 0)
            return;
        CrashSelection sel;
        sel.basis = std::move(candidate);
        sel.quality = evaluate_basis_quality_(A, b, c, sel.basis, opt.tol);
        sel.source = std::move(source);
        if (attempt >= 0) {
            sel.style = crash_attempt_config_(opt, attempt).style_name;
        } else if (attempt == -2) {
            sel.style = "logical";
        } else {
            sel.style = "mapped";
        }
        sel.attempt = attempt;
        if (better_basis_quality_(sel, best, opt.mode))
            best = std::move(sel);
    };

    const auto can_accept_early = [&](const BasisQuality& q) {
        if (!q.valid) {
            return false;
        }
        const double solve_residual_guard = std::max(1e-7, 100.0 * opt.tol);
        if (!std::isfinite(q.solve_residual) || q.solve_residual > solve_residual_guard)
            return false;
        switch (opt.mode) {
            case SimplexMode::Dual:
                return q.dual_feasible;
            case SimplexMode::Primal:
                return q.primal_feasible;
            case SimplexMode::Auto:
            default:
                return q.primal_feasible || q.dual_feasible;
        }
    };

    if (seed_basis && !seed_basis->empty()) {
        if ((int)seed_basis->size() == A.rows()) {
            consider(*seed_basis, "warm_start", -1);
            if (can_accept_early(best.quality)) {
                return best;
            }
        }
        if (opt.repair_mapped_basis) {
            const int attempts = std::max(1, opt.crash_attempts);
            for (int k = 0; k < attempts; ++k) {
                consider(build_basis_attempt_(A, b, c, crash_attempt_config_(opt, k), opt.tol,
                                              opt.mode, seed_basis),
                         "repaired_warm_start", k);
                if (can_accept_early(best.quality)) {
                    return best;
                }
            }
        }
    }

    consider(find_logical_basis_(A), "logical_basis", -2);
    if (can_accept_early(best.quality)) {
        return best;
    }

    const int attempts = std::max(1, opt.crash_attempts);
    for (int k = 0; k < attempts; ++k) {
        consider(build_basis_attempt_(A, b, c, crash_attempt_config_(opt, k), opt.tol, opt.mode),
                 "crash", k);
        if (can_accept_early(best.quality)) {
            return best;
        }
    }
    return best;
}

inline RevisedSimplex::CrashSelection
RevisedSimplex::choose_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                      const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                                      std::optional<std::vector<int>> seed_basis) {
    CrashSelection best;

    auto consider = [&](std::vector<int> candidate, std::string source, int attempt) {
        if (candidate.empty() && A.rows() != 0)
            return;
        CrashSelection sel;
        sel.basis = std::move(candidate);
        sel.quality = evaluate_basis_quality_(A, b, c, sel.basis, opt.tol);
        sel.source = std::move(source);
        if (attempt >= 0) {
            sel.style = crash_attempt_config_(opt, attempt).style_name;
        } else if (attempt == -2) {
            sel.style = "logical";
        } else {
            sel.style = "mapped";
        }
        sel.attempt = attempt;
        if (better_basis_quality_(sel, best, opt.mode))
            best = std::move(sel);
    };

    const auto can_accept_early = [&](const BasisQuality& q) {
        if (!q.valid) {
            return false;
        }
        const double solve_residual_guard = std::max(1e-7, 100.0 * opt.tol);
        if (!std::isfinite(q.solve_residual) || q.solve_residual > solve_residual_guard)
            return false;
        switch (opt.mode) {
            case SimplexMode::Dual:
                return q.dual_feasible;
            case SimplexMode::Primal:
                return q.primal_feasible;
            case SimplexMode::Auto:
            default:
                return q.primal_feasible || q.dual_feasible;
        }
    };

    if (seed_basis && !seed_basis->empty()) {
        if ((int)seed_basis->size() == A.rows()) {
            consider(*seed_basis, "warm_start", -1);
            if (can_accept_early(best.quality)) {
                return best;
            }
        }
        if (opt.repair_mapped_basis) {
            const int attempts = std::max(1, opt.crash_attempts);
            for (int k = 0; k < attempts; ++k) {
                consider(build_basis_attempt_(A, b, c, crash_attempt_config_(opt, k), opt.tol,
                                              opt.mode, seed_basis),
                         "repaired_warm_start", k);
                if (can_accept_early(best.quality)) {
                    return best;
                }
            }
        }
    }

    consider(find_logical_basis_(A), "logical_basis", -2);
    if (can_accept_early(best.quality)) {
        return best;
    }

    const int attempts = std::max(1, opt.crash_attempts);
    for (int k = 0; k < attempts; ++k) {
        consider(build_basis_attempt_(A, b, c, crash_attempt_config_(opt, k), opt.tol, opt.mode),
                 "crash", k);
        if (can_accept_early(best.quality)) {
            return best;
        }
    }
    return best;
}

inline std::optional<std::vector<int>>
RevisedSimplex::find_initial_basis_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                    const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                                    std::optional<std::vector<int>> seed_basis) {
    if (!seed_basis || seed_basis->empty()) {
        std::vector<int> logical = find_logical_basis_(A);
        const BasisQuality logical_quality = evaluate_basis_quality_(A, b, c, logical, opt.tol);
        if (logical_quality.valid && logical_quality.rank == static_cast<int>(A.rows())) {
            return logical;
        }
    }
    CrashSelection sel = choose_initial_basis_(A, b, c, opt, seed_basis);
    if (!sel.quality.valid)
        return std::nullopt;
    return sel.basis;
}

inline std::optional<std::vector<int>>
RevisedSimplex::find_initial_basis_(const SparseMatrix& A, const Eigen::VectorXd& b,
                                    const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
                                    std::optional<std::vector<int>> seed_basis) {
    if (!seed_basis || seed_basis->empty()) {
        std::vector<int> logical = find_logical_basis_(A);
        const BasisQuality logical_quality = evaluate_basis_quality_(A, b, c, logical, opt.tol);
        if (logical_quality.valid && logical_quality.rank == static_cast<int>(A.rows())) {
            return logical;
        }
    }
    CrashSelection sel = choose_initial_basis_(A, b, c, opt, seed_basis);
    if (!sel.quality.valid)
        return std::nullopt;
    return sel.basis;
}
