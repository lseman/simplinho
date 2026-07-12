#pragma once

// Internal inline implementation included at the end of simplex.h after
// RevisedSimplex is fully declared.

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "simplex/core/sparse_utils.h"
#include "simplex/engine/phase1.h"
#include "simplex/engine/postsolve.h"
#include "simplex/engine/simplex_reformulation.h"
#include "simplex/factorization/crash.h"
#include "simplex/primal.h"
#include "simplex/types/dual.h"

inline RevisedSimplex::PhaseResult
RevisedSimplex::phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u,
                       std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexPrimalEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                           std::move(warm_status));
}

inline LPSolution RevisedSimplex::solve_impl_sparse_(
    const SparseMatrix& A_in, const Eigen::VectorXd& b_in, const Eigen::VectorXd& c_in,
    const Eigen::VectorXd& l_in, const Eigen::VectorXd& u_in,
    std::optional<std::vector<int>> basis_opt, const LPBasis* basis_state_opt) {
    auto t0_presolve = std::chrono::steady_clock::now();
    SolveTraceScope trace_scope(*this);
    degen_.reset();
    const int m_in = static_cast<int>(A_in.rows());
    const int n = static_cast<int>(A_in.cols());
    if (b_in.size() != m_in) {
        throw std::invalid_argument("simplex: b size mismatch with rows(A)");
    }
    if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
        throw std::invalid_argument("simplex: c/l/u sizes must equal cols(A)");
    }
    // HiGHS-style kernel selection: sparse storage does not imply sparse
    // numerical work. Dense factorization is more stable and cheaper for
    // compact matrices above this density; retain sparse LU for large or
    // genuinely sparse models.
    const std::int64_t matrix_entries = static_cast<std::int64_t>(m_in) * n;
    const double matrix_density = matrix_entries > 0
                                      ? static_cast<double>(A_in.nonZeros()) /
                                            static_cast<double>(matrix_entries)
                                      : 0.0;
    bool nonnegative_equality_form = true;
    for (int j = 0; j < n; ++j) {
        if (!std::isfinite(l_in(j)) || std::abs(l_in(j)) > opt_.tol || std::isfinite(u_in(j))) {
            nonnegative_equality_form = false;
            break;
        }
    }
    if (nonnegative_equality_form && matrix_entries <= 250'000 && matrix_density >= 0.35) {
        LPSolution sol = solve_impl_(Eigen::MatrixXd(A_in), b_in, c_in, l_in, u_in,
                                     std::move(basis_opt), basis_state_opt);
        sol.info["factorization_kernel"] = "dense_from_sparse";
        return sol;
    }
    // Fast-path: if the same compressed A_in is passed as last time, reuse the
    // cached signature and skip the O(nnz) hash computation.
    const double* a_value_ptr = A_in.isCompressed() ? A_in.valuePtr() : nullptr;
    const bool same_sparse_ptr =
        (a_value_ptr != nullptr && a_value_ptr == last_sparse_a_value_ptr_ &&
         m_in == last_sparse_a_rows_ && n == last_sparse_a_cols_);
    const std::uint64_t sig = same_sparse_ptr ? last_sparse_a_signature_ : matrix_signature_(A_in);
    if (a_value_ptr != nullptr) {
        last_sparse_a_value_ptr_ = a_value_ptr;
        last_sparse_a_rows_ = m_in;
        last_sparse_a_cols_ = n;
        last_sparse_a_signature_ = sig;
    }
    begin_solve_(sig, m_in, n, true, basis_state_opt);

    trace_line_("[solve] sparse start m=" + std::to_string(m_in) + " n=" + std::to_string(n));
    trace_line_("[solve] sparse disable_presolve=" + std::to_string(opt_.disable_presolve));

    const RowRankReduction row_rank =
        dependent_row_reduction_(Eigen::MatrixXd(A_in), b_in, opt_.tol);
    if (row_rank.needed) {
        if (row_rank.inconsistent) {
            return finalize_solution_(make_solution_(
                LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::infinity(), {}, 0,
                {{"reason", "inconsistent_dependent_rows"},
                 {"row_rank_reduction_original_m", std::to_string(row_rank.original_rows)},
                 {"row_rank_reduction_rank", std::to_string(row_rank.rank)}}));
        }
        LPSolution reduced_sol = solve_impl_sparse_(select_sparse_rows_(A_in, row_rank.keep_rows),
                                                    select_vector_rows_(b_in, row_rank.keep_rows),
                                                    c_in, l_in, u_in, std::nullopt, nullptr);
        if (reduced_sol.status == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, reduced_sol.x, l_in, u_in, opt_.tol)) {
            reduced_sol.status = LPSolution::Status::Infeasible;
            reduced_sol.obj = std::numeric_limits<double>::infinity();
            reduced_sol.info["reason"] = "dependent_row_solution_failed_original_check";
        }
        reduced_sol.info["row_rank_reduction"] = "1";
        reduced_sol.info["row_rank_reduction_original_m"] = std::to_string(row_rank.original_rows);
        reduced_sol.info["row_rank_reduction_rank"] = std::to_string(row_rank.rank);
        return reduced_sol;
    }

    // Pass cached data_scale to skip the O(nnz) max-abs matrix scan when A hasn't changed.
    const double precomputed_ds = (same_sparse_ptr && sparse_bound_only_cache_.valid &&
                                   sparse_bound_only_cache_.cached_data_scale > 0.0)
                                      ? sparse_bound_only_cache_.cached_data_scale
                                      : 0.0;
    const auto sanitized_bounds =
        canonicalize_inactive_huge_bounds_(A_in, b_in, l_in, u_in, precomputed_ds);
    const Eigen::VectorXd& l_use = sanitized_bounds.l;
    const Eigen::VectorXd& u_use = sanitized_bounds.u;
    bool reused_sparse_bound_cache = false;

    std::optional<LPBasis> rebased_basis_state_opt = std::nullopt;
    if (basis_state_opt && !basis_state_opt->column_status.empty()) {
        rebased_basis_state_opt =
            rebase_basis_state_for_bounds_(*basis_state_opt, l_use, u_use, opt_.tol);
        basis_state_opt = &*rebased_basis_state_opt;
    }

    // When a reusable factorization is in hand (same matrix, Dual mode), prefer
    // its basis over any passed-in seed: starting the dual re-solve from exactly
    // that basis lets try_reuse_factorization_ adopt the existing LU and FT-pivot
    // instead of refactoring. Otherwise the passed-in basis_opt (a different
    // basis) forces a cold factorization on every node solve.
    const std::optional<std::vector<int>> factorized_basis_seed_opt =
        warm_factorization_basis_seed_();
    if (factorized_basis_seed_opt) {
        basis_opt = *factorized_basis_seed_opt;
    }
    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        if ((int)basis_state_opt->column_status.size() != n) {
            throw std::invalid_argument("simplex: warm-start basis column_status size mismatch");
        }
    }

    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        basis_opt = basis_columns_from_basis_state_(*basis_state_opt, m_in);
    }

    bool is_nonnegative_standard = true;
    bool has_free_vars = false;
    for (int j = 0; j < n; ++j) {
        const bool has_l = std::isfinite(l_use(j));
        const bool has_u = std::isfinite(u_use(j));
        const bool l_is_zero = has_l && std::abs(l_use(j)) <= opt_.tol;
        if (!l_is_zero || has_u) {
            is_nonnegative_standard = false;
        }
        if (!has_l && !has_u) {
            has_free_vars = true;
        }
    }

    const bool has_warm_basis_for_dual = opt_.mode == SimplexMode::Dual && basis_state_opt &&
                                         !basis_state_opt->column_status.empty();

    auto dualization_requested = [&]() {
        std::string mode = opt_.dualization;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (mode == "on" || mode == "true" || mode == "1")
            return true;
        if (mode != "auto")
            return false;
        if (n <= 0)
            return false;
        return static_cast<double>(m_in) / static_cast<double>(n) >=
               opt_.dualization_min_row_col_ratio;
    };

    if (is_nonnegative_standard && !has_warm_basis_for_dual && !basis_opt &&
        dualization_requested() && n <= opt_.dualization_max_recovery_cols) {
        try {
            std::vector<Eigen::Triplet<double>> dual_trips;
            dual_trips.reserve(static_cast<std::size_t>(2 * A_in.nonZeros() + n));
            for (int j = 0; j < A_in.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_in, j); it; ++it) {
                    dual_trips.emplace_back(j, it.row(), it.value());
                    dual_trips.emplace_back(j, m_in + it.row(), -it.value());
                }
            }
            for (int j = 0; j < n; ++j)
                dual_trips.emplace_back(j, 2 * m_in + j, 1.0);

            SparseMatrix A_dual(n, 2 * m_in + n);
            A_dual.setFromTriplets(dual_trips.begin(), dual_trips.end());
            A_dual.makeCompressed();
            Eigen::VectorXd b_dual = c_in;
            Eigen::VectorXd c_dual = Eigen::VectorXd::Zero(2 * m_in + n);
            for (int i = 0; i < m_in; ++i) {
                c_dual(i) = -b_in(i);
                c_dual(m_in + i) = b_in(i);
            }
            Eigen::VectorXd l_dual = Eigen::VectorXd::Zero(2 * m_in + n);
            Eigen::VectorXd u_dual = Eigen::VectorXd::Constant(2 * m_in + n, presolve::inf());

            RevisedSimplexOptions dual_opt = opt_;
            dual_opt.dualization = "off";
            dual_opt.mode = SimplexMode::Dual;
            dual_opt.compute_tableau = false;
            dual_opt.compute_reduced_costs = false;
            RevisedSimplex dual_solver(dual_opt);
            LPSolution dual_sol = dual_solver.solve(A_dual, b_dual, c_dual, l_dual, u_dual);
            if (dual_sol.status == LPSolution::Status::Optimal &&
                dual_sol.x.size() == 2 * m_in + n) {
                Eigen::VectorXd restricted_u = u_in;
                int fixed_by_dual_slack = 0;
                const double active_tol = std::max(1e-8, 100.0 * opt_.tol);
                for (int j = 0; j < n; ++j) {
                    const double slack = dual_sol.x(2 * m_in + j);
                    if (std::isfinite(slack) && slack > active_tol) {
                        restricted_u(j) = 0.0;
                        ++fixed_by_dual_slack;
                    }
                }
                if (fixed_by_dual_slack > 0) {
                    RevisedSimplexOptions recovery_opt = opt_;
                    recovery_opt.dualization = "off";
                    recovery_opt.mode = SimplexMode::Dual;
                    RevisedSimplex recovery_solver(recovery_opt);
                    LPSolution recovered =
                        recovery_solver.solve(A_in, b_in, c_in, l_in, restricted_u);
                    if (recovered.status == LPSolution::Status::Optimal &&
                        primal_feasible_(A_in, b_in, recovered.x, l_in, u_in, opt_.tol)) {
                        recovered.info["dualization"] = "explicit_dual_recovery";
                        recovered.info["dualization_fixed_by_slack"] =
                            std::to_string(fixed_by_dual_slack);
                        recovered.info["dualization_dual_iters"] = std::to_string(dual_sol.iters);
                        return finalize_solution_(std::move(recovered));
                    }
                }
            }
        } catch (const std::exception& e) {
            trace_line_(std::string("[dualization] fallback after failure: ") + e.what());
        }
    }

    // Native-bounds path: bounded variables are handled directly by the
    // engines (anchored model below). The standard-form reformulation is only
    // needed for free variables (both bounds infinite), when explicitly
    // requested via opt_.native_bounds = false, or when the dual engine is
    // explicitly requested. Auto cold solves prefer native bounded primal,
    // avoiding one extra row and slack per finite upper bound.
    bool has_upper_bounds = false;
    for (int j = 0; j < n; ++j) {
        if (std::isfinite(u_use(j))) {
            has_upper_bounds = true;
            break;
        }
    }
    const bool use_reformulation = !is_nonnegative_standard &&
                                   (has_free_vars || !opt_.native_bounds ||
                                    (opt_.mode == SimplexMode::Dual && has_upper_bounds)) &&
                                   !std::getenv("SIMPLINHO_FORCE_NATIVE_DUAL");
    if (use_reformulation) {
        const bool cache_reuse = sparse_bound_only_cache_.same_problem(A_in, b_in, c_in) &&
                                 sparse_bound_only_cache_.orientation_matches(l_use, u_use);
        if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
            static std::atomic<long> reuse_y{0}, reuse_n{0};
            (cache_reuse ? reuse_y : reuse_n)++;
            if (((reuse_y + reuse_n) % 200) == 0)
                std::fprintf(stderr, "[reform] cache_reuse yes=%ld no=%ld\n", reuse_y.load(),
                             reuse_n.load());
        }
        // Reuse pre-allocated map_scratch to avoid heap allocation on every BnB node solve.
        // ReformVar is defined at class scope so SparseBoundOnlyCache can own this buffer.
        auto& map = sparse_bound_only_cache_.map_scratch;
        map.assign(n, ReformVar{});
        // The 4 int index vectors live in the cache; references are set after the
        // cache-hit / cache-miss block below (both paths ensure the cache is up-to-date).
        int nv = 0;
        int upper_rows = 0;
        double obj_shift = 0.0;
        const int m_eq = m_in;
        int n_total = 0;
        int m_total = 0;
        // Use pointers to dispatch between cached data (no allocation) and locally
        // owned buffers (non-cache path only). In the cache path, we point directly
        // into SparseBoundOnlyCache to avoid all per-node copies.
        const SparseMatrix* A_std_ptr = nullptr;
        const Eigen::VectorXd* b_std_ptr = nullptr;
        const Eigen::VectorXd* c_std_ptr = nullptr;
        const Eigen::VectorXd* l_std_ptr = nullptr;
        const Eigen::VectorXd* u_std_ptr = nullptr;
        // Owned storage — only used in the non-cache path.
        SparseMatrix A_std_owned;
        Eigen::VectorXd b_std_owned;
        Eigen::VectorXd c_std_owned;
        Eigen::VectorXd l_std_owned;
        Eigen::VectorXd u_std_owned;

        if (cache_reuse) {
            reused_sparse_bound_cache = true;
            const auto& cache = sparse_bound_only_cache_;
            n_total = cache.n_total;
            m_total = cache.m_total;
            upper_rows = cache.upper_rows;
            // Fill b_std_scratch in-place (no allocation); use by reference below.
            reconstruct_sparse_reformulated_rhs_(l_use, u_use);
            // Point directly into cached data — no copies.
            A_std_ptr = &cache.A_std;
            b_std_ptr = &cache.b_std_scratch;
            c_std_ptr = &cache.c_std;
            l_std_ptr = &cache.l_std;
            u_std_ptr = &cache.u_std;
            for (int j = 0; j < n; ++j) {
                const bool has_l = static_cast<bool>(cache.has_lower[j]);
                const bool has_u = static_cast<bool>(cache.has_upper[j]);
                const bool fixed = static_cast<bool>(cache.fixed_bound[j]);
                if (has_l || has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = cache.single_y[j];
                    map[j].shift = has_l ? l_use(j) : u_use(j);
                    map[j].sign = fixed ? 0 : (has_l ? 1 : -1);
                    map[j].has_upper_row = has_l && has_u && !fixed;
                } else {
                    map[j].y_pos = cache.split_pos[j];
                    map[j].y_neg = cache.split_neg[j];
                }
                map[j].upper_slack = cache.upper_slack[j];
                // single_y/split_pos/split_neg/upper_slack are NOT copied to local vectors;
                // use sparse_bound_only_cache_.* directly via references below.
                if (has_l) {
                    obj_shift += c_in(j) * l_use(j);
                } else if (has_u) {
                    obj_shift += c_in(j) * u_use(j);
                }
            }
        } else {
            for (int j = 0; j < n; ++j) {
                const bool has_l = std::isfinite(l_use(j));
                const bool has_u = std::isfinite(u_use(j));
                if (has_l && has_u && u_use(j) < l_use(j) - opt_.tol) {
                    Eigen::VectorXd xnan =
                        Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                    return finalize_solution_(
                        make_solution_(LPSolution::Status::Infeasible, xnan,
                                       std::numeric_limits<double>::infinity(), {}, 0,
                                       {{"reason", "invalid_bounds"}}));
                }
                const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= opt_.tol;
                if (fixed) {
                    map[j].uses_single_var = true;
                    map[j].y = -1;
                    map[j].shift = l_use(j);
                    map[j].sign = 0;
                    obj_shift += c_in(j) * l_use(j);
                } else if (has_l) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = l_use(j);
                    map[j].sign = 1;
                    obj_shift += c_in(j) * l_use(j);
                    if (has_u) {
                        map[j].has_upper_row = true;
                        ++upper_rows;
                    }
                } else if (has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = u_use(j);
                    map[j].sign = -1;
                    obj_shift += c_in(j) * u_use(j);
                } else {
                    map[j].y_pos = nv++;
                    map[j].y_neg = nv++;
                }
            }

            n_total = nv + upper_rows;
            m_total = m_eq + upper_rows;
            b_std_owned = Eigen::VectorXd::Zero(m_total);
            c_std_owned = Eigen::VectorXd::Zero(n_total);
            l_std_owned = Eigen::VectorXd::Zero(n_total);
            u_std_owned = Eigen::VectorXd::Constant(n_total, presolve::inf());

            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    if (map[j].y < 0) {
                        continue;
                    }
                    c_std_owned(map[j].y) += static_cast<double>(map[j].sign) * c_in(j);
                } else {
                    c_std_owned(map[j].y_pos) += c_in(j);
                    c_std_owned(map[j].y_neg) += -c_in(j);
                }
            }

            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(static_cast<std::size_t>(A_in.nonZeros() * 2 + upper_rows * 2));
            for (int j = 0; j < A_in.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_in, j); it; ++it) {
                    const int row = it.row();
                    const double aij = it.value();
                    if (map[j].uses_single_var) {
                        b_std_owned(row) -= aij * map[j].shift;
                        if (map[j].y >= 0) {
                            trips.emplace_back(row, map[j].y,
                                               static_cast<double>(map[j].sign) * aij);
                        }
                    } else {
                        trips.emplace_back(row, map[j].y_pos, aij);
                        trips.emplace_back(row, map[j].y_neg, -aij);
                    }
                }
            }
            for (int i = 0; i < m_eq; ++i)
                b_std_owned(i) += b_in(i);

            int upper_row = 0;
            for (int j = 0; j < n; ++j) {
                if (!map[j].has_upper_row)
                    continue;
                const int slack = nv + upper_row;
                const int row = m_eq + upper_row;
                map[j].upper_slack = slack;
                // upper_slack vector lives in the cache (filled by build_sparse_bound_only_cache_).
                trips.emplace_back(row, map[j].y, 1.0);
                trips.emplace_back(row, slack, 1.0);
                b_std_owned(row) = u_use(j) - l_use(j);
                ++upper_row;
            }

            A_std_owned = SparseMatrix(m_total, n_total);
            if (!trips.empty())
                A_std_owned.setFromTriplets(trips.begin(), trips.end());
            A_std_owned.makeCompressed();
            build_sparse_bound_only_cache_(A_in, b_in, c_in, l_use, u_use);

            A_std_ptr = &A_std_owned;
            b_std_ptr = &b_std_owned;
            c_std_ptr = &c_std_owned;
            l_std_ptr = &l_std_owned;
            u_std_ptr = &u_std_owned;
        }
        // Uniform reference bindings — used from here on in both paths.
        const SparseMatrix& A_std = *A_std_ptr;
        const Eigen::VectorXd& b_std = *b_std_ptr;
        const Eigen::VectorXd& c_std = *c_std_ptr;
        const Eigen::VectorXd& l_std = *l_std_ptr;
        const Eigen::VectorXd& u_std = *u_std_ptr;
        // Cache int-index vectors are now authoritative in both paths.
        // Cache path: never wrote to them (used cache directly).
        // Non-cache path: build_sparse_bound_only_cache_ populated them.
        const std::vector<int>& single_y = sparse_bound_only_cache_.single_y;
        const std::vector<int>& upper_slack = sparse_bound_only_cache_.upper_slack;
        const std::vector<int>& split_pos = sparse_bound_only_cache_.split_pos;
        const std::vector<int>& split_neg = sparse_bound_only_cache_.split_neg;
        std::optional<std::vector<int>> basis_std = std::nullopt;
        std::optional<LPBasis> basis_state_std = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::vector<int> cand;
            cand.reserve(std::min(m_eq, (int)basis_opt->size()) + upper_rows);
            for (int jorig : *basis_opt) {
                if (jorig < 0 || jorig >= n)
                    continue;
                if (map[jorig].uses_single_var) {
                    if (map[jorig].y >= 0)
                        cand.push_back(map[jorig].y);
                } else if (map[jorig].y_pos >= 0) {
                    cand.push_back(map[jorig].y_pos);
                }
                if ((int)cand.size() == m_eq)
                    break;
            }
            for (int j = 0; j < n; ++j) {
                if (map[j].upper_slack >= 0)
                    cand.push_back(map[j].upper_slack);
            }
            if ((int)cand.size() == m_total)
                basis_std = std::move(cand);
        }
        if (basis_state_opt && !basis_state_opt->column_status.empty() &&
            (int)basis_state_opt->column_status.size() == n) {
            const bool exact_basis_state = basis_state_matches_problem_(*basis_state_opt, m_eq, n);
            basis_state_std =
                exact_basis_state
                    ? map_reformulated_basis_state_(*basis_state_opt, l_use, u_use, n_total,
                                                    single_y, upper_slack, split_pos, split_neg)
                    : map_reformulated_basis_seed_state_(*basis_state_opt, n_total, single_y,
                                                         upper_slack, split_pos, split_neg);
        }

        // When no warm-start basis is available, construct a logical basis from the
        // reformulation structure. Upper-bound slack columns are identity columns for their
        // rows; original rows are covered by picking the first y/y_pos with a nonzero entry.
        if (!basis_std && (!basis_state_std || basis_state_std->column_status.empty())) {
            // Build a row->first_nonzero_col map from the sparse A_std for original rows.
            std::vector<int> row_col(m_eq, -1);
            std::vector<bool> col_used(n_total, false);
            for (int j = 0; j < A_std.outerSize(); ++j) {
                for (SparseMatrix::InnerIterator it(A_std, j); it; ++it) {
                    const int row = static_cast<int>(it.row());
                    if (row < m_eq && row_col[row] < 0 && !col_used[j] &&
                        std::abs(it.value()) > 1e-14) {
                        row_col[row] = j;
                        col_used[j] = true;
                    }
                }
            }
            std::vector<int> cand;
            cand.reserve(m_total);
            bool ok = true;
            for (int i = 0; i < m_eq; ++i) {
                if (row_col[i] < 0) {
                    ok = false;
                    break;
                }
                cand.push_back(row_col[i]);
            }
            if (ok) {
                for (int j = 0; j < n; ++j) {
                    if (map[j].upper_slack >= 0)
                        cand.push_back(map[j].upper_slack);
                }
                if ((int)cand.size() == m_total)
                    basis_std = std::move(cand);
            }
        }

        LPSolution std_sol;
        bool reformulated_retry_used = false;
        std::optional<std::vector<int>> reformulated_basis_guess = basis_std;
        if ((!reformulated_basis_guess || reformulated_basis_guess->empty()) && basis_state_std &&
            !basis_state_std->column_status.empty()) {
            reformulated_basis_guess = basis_columns_from_basis_state_(*basis_state_std, m_total);
        }
        std::optional<BasisQuality> reformulated_warm_basis_quality = std::nullopt;
        // SCIP/HiGHS insight: on a cache-hit (same A/b/c, only bounds changed), the warm basis
        // from an optimal dual-simplex solve is ALWAYS dual feasible after a bound tightening.
        // Skip the O(m^2) SparseLU quality check in this common BnB case.
        const bool skip_quality_check = cache_reuse && opt_.mode == SimplexMode::Dual &&
                                        basis_state_std.has_value() &&
                                        !basis_state_std->column_status.empty();
        if (skip_quality_check) {
            // Assume dual feasible — the dual simplex will correct any violations in O(pivots).
            BasisQuality assumed_quality;
            assumed_quality.valid = true;
            assumed_quality.dual_feasible = true;
            assumed_quality.primal_feasible = false; // conservative; dual simplex handles this
            reformulated_warm_basis_quality = assumed_quality;
        } else if (reformulated_basis_guess && !reformulated_basis_guess->empty()) {
            reformulated_warm_basis_quality = evaluate_basis_quality_(
                A_std, b_std, c_std, *reformulated_basis_guess, l_std, u_std, opt_.tol);
        }
        // Prefer dual when the mapped warm basis is dual-feasible, regardless of opt_.mode.
        // A dual-feasible warm basis makes dual simplex O(pivots) to optimality; primal
        // from the same basis often hits numerical issues on the reformulated matrix.
        const bool use_dual_first = opt_.mode != SimplexMode::Primal &&
                                    reformulated_warm_basis_quality &&
                                    reformulated_warm_basis_quality->valid &&
                                    reformulated_warm_basis_quality->dual_feasible;
        const char* reformulated_initial_mode =
            use_dual_first ? "dual" : (opt_.mode == SimplexMode::Primal ? "primal" : "auto");
        bool reformulated_inner_cache_used = false;
        auto solve_reformulated = [&](SimplexMode mode) {
            RevisedSimplexOptions solve_opt = opt_;
            solve_opt.mode = mode;
            solve_opt.disable_presolve = true;
            trace_line_("[solve_reformulated] disable_presolve=" +
                        std::to_string(solve_opt.disable_presolve));
            // HiGHS/SCIP hot-restart: reuse the persistent reformulated solver so the
            // FTBasis factorization (and signature cache) survive across BnB node solves.
            // Created fresh only after a cache miss (when reformulated structure changed).
            if (!sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache =
                    std::make_unique<RevisedSimplex>(solve_opt);
            }
            RevisedSimplex& reformulated_solver =
                *sparse_bound_only_cache_.reformulated_solver_cache;
            reformulated_solver.opt_ = solve_opt;
            const RevisedSimplex::SparseMatrix& A_std_sparse = A_std;
            // Hot path: feed back the previous solve's optimal basis_state. Unlike
            // the no-basis internal-cache route (which loses the nonbasic bound
            // views and returns Singular), passing the full basis_state -- which
            // carries the FTBasis warm_state -- lets the solver adopt the existing
            // factorization and dual-pivot to the new optimum in a few iterations.
            auto& prev_state = sparse_bound_only_cache_.last_reformulated_basis_state;
            if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                static std::atomic<long> hit{0}, no_prev{0}, no_fact{0}, sz{0};
                if (!prev_state)
                    ++no_prev;
                else if (!reformulated_solver.has_cached_basis_factorization_state(A_std_sparse))
                    ++no_fact;
                else if (static_cast<int>(prev_state->column_status.size()) != n_total)
                    ++sz;
                else
                    ++hit;
                if (((hit + no_prev + no_fact + sz) % 200) == 0)
                    std::fprintf(stderr, "[reformpath] hit=%ld no_prev=%ld no_fact=%ld sz=%ld\n",
                                 hit.load(), no_prev.load(), no_fact.load(), sz.load());
            }
            if (prev_state &&
                reformulated_solver.has_cached_basis_factorization_state(A_std_sparse) &&
                static_cast<int>(prev_state->column_status.size()) == n_total) {
                reformulated_inner_cache_used = true;
                return reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std, u_std,
                                                 *prev_state);
            }
            return basis_state_std ? reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std,
                                                               u_std, *basis_state_std)
                                   : reformulated_solver.solve(A_std_sparse, b_std, c_std, l_std,
                                                               u_std, basis_std);
        };
        try {
            std_sol = solve_reformulated(use_dual_first ? SimplexMode::Dual : opt_.mode);
        } catch (const std::exception& e) {
            std_sol.status = LPSolution::Status::Singular;
            std_sol.info["where"] = "bound_reformulation_recursive_solve";
            std_sol.info["what"] = e.what();
        }
        if (use_dual_first && (std_sol.status == LPSolution::Status::Singular ||
                               std_sol.status == LPSolution::Status::NeedPhase1 ||
                               std_sol.status == LPSolution::Status::IterLimit)) {
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        if (opt_.mode == SimplexMode::Dual && !use_dual_first &&
            (std_sol.status == LPSolution::Status::NeedPhase1 ||
             std_sol.status == LPSolution::Status::Singular)) {
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        if ((std_sol.status == LPSolution::Status::Singular ||
             std_sol.status == LPSolution::Status::NeedPhase1) &&
            (basis_std.has_value() || basis_state_std.has_value())) {
            basis_std.reset();
            basis_state_std.reset();
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }
        // When skip_quality_check assumed dual feasibility without verification, an Infeasible
        // result is unreliable: if the cached basis was not truly dual-feasible, the dual simplex
        // can terminate with a spurious Farkas certificate. A wrong Infeasible propagates through
        // the BnB tree as a learned conflict, incorrectly pruning feasible nodes (false-positive
        // optimal). Verify by clearing the inner cache and re-solving cold.
        if (skip_quality_check && std_sol.status == LPSolution::Status::Infeasible) {
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            const LPSolution cold_sol = solve_reformulated(SimplexMode::Auto);
            if (cold_sol.status != LPSolution::Status::Infeasible) {
                std_sol = cold_sol;
                reformulated_retry_used = true;
            }
        }
        // Also verify a warm-started Optimal: check primal feasibility ||Ax - b||_inf.
        // A wrong Optimal (too-low LP bound from a bad warm start) causes incorrect pruning
        // in BnB, giving false-positive optimal results.
        if (skip_quality_check && std_sol.status == LPSolution::Status::Optimal &&
            std_sol.x.size() == n_total) {
            const double verify_tol = opt_.tol * 1e4;
            const Eigen::VectorXd residual = (*A_std_ptr) * std_sol.x - (*b_std_ptr);
            if (residual.lpNorm<Eigen::Infinity>() > verify_tol) {
                if (sparse_bound_only_cache_.reformulated_solver_cache) {
                    sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
                }
                const LPSolution cold_sol = solve_reformulated(SimplexMode::Auto);
                if (cold_sol.status == LPSolution::Status::Optimal) {
                    std_sol = cold_sol;
                    reformulated_retry_used = true;
                }
            }
        }

        auto reconstruct_original_x = [&]() {
            Eigen::VectorXd out =
                Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            if (std_sol.x.size() != n_total || !std_sol.x.array().isFinite().all()) {
                return out;
            }
            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    out(j) = map[j].y >= 0 ? map[j].shift + static_cast<double>(map[j].sign) *
                                                                std_sol.x(map[j].y)
                                           : map[j].shift;
                } else {
                    out(j) = std_sol.x(map[j].y_pos) - std_sol.x(map[j].y_neg);
                }
            }
            return out;
        };

        Eigen::VectorXd x = reconstruct_original_x();
        if (std_sol.status == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, x, l_in, u_in, opt_.tol)) {
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            basis_std.reset();
            basis_state_std.reset();
            std_sol = solve_reformulated(SimplexMode::Primal);
            reformulated_retry_used = true;
            x = reconstruct_original_x();
        }
        const bool reformulated_internal_failure =
            std_sol.status == LPSolution::Status::NeedPhase1 ||
            std_sol.status == LPSolution::Status::Singular ||
            std_sol.status == LPSolution::Status::IterLimit ||
            (std_sol.status == LPSolution::Status::Optimal &&
             !primal_feasible_(A_in, b_in, x, l_in, u_in, opt_.tol));
        if (reformulated_internal_failure) {
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
            if (sparse_bound_only_cache_.reformulated_solver_cache) {
                sparse_bound_only_cache_.reformulated_solver_cache->clear_basis_cache();
            }
            Eigen::MatrixXd A_dense(A_in);
            LPSolution dense_sol =
                solve_impl_(A_dense, b_in, c_in, l_in, u_in, basis_opt, basis_state_opt);
            dense_sol.info["sparse_dense_fallback"] = "1";
            dense_sol.info["sparse_dense_fallback_from_status"] = to_string(std_sol.status);
            return dense_sol;
        }

        // Persist the reformulated optimal basis_state (incl. its FTBasis
        // warm_state) so the next node's solve_reformulated can dual-restart from
        // it. Only on a clean Optimal -- a bad/partial state would force singular
        // warm restarts downstream.
        if (std_sol.status == LPSolution::Status::Optimal &&
            primal_feasible_(A_in, b_in, x, l_in, u_in, opt_.tol) &&
            !std_sol.basis_state.column_status.empty() &&
            static_cast<int>(std_sol.basis_state.column_status.size()) == n_total) {
            sparse_bound_only_cache_.last_reformulated_basis_state = std_sol.basis_state;
        } else {
            sparse_bound_only_cache_.last_reformulated_basis_state.reset();
        }

        std::vector<int> basis_out;
        std::vector<char> seen(n, 0);
        for (int idx : std_sol.basis) {
            if (static_cast<int>(basis_out.size()) == m_in)
                break;
            for (int j = 0; j < n; ++j) {
                const bool matches_single = map[j].uses_single_var && map[j].y == idx;
                const bool matches_split =
                    !map[j].uses_single_var && (map[j].y_pos == idx || map[j].y_neg == idx);
                if ((matches_single || matches_split) && !seen[j]) {
                    seen[j] = 1;
                    basis_out.push_back(j);
                    break;
                }
            }
        }

        auto info = std_sol.info;
        info["bound_reformulation"] = "1";
        info["sparse_pipeline"] = "1";
        if (reused_sparse_bound_cache) {
            info["sparse_bound_only_fast_path"] = "1";
        }
        if (reformulated_inner_cache_used) {
            info["bound_reformulation_inner_cache"] = "1";
        }
        info["bound_reformulation_initial_mode"] = reformulated_initial_mode;
        if (reformulated_warm_basis_quality) {
            info["bound_reformulation_warm_start_valid"] =
                reformulated_warm_basis_quality->valid ? "1" : "0";
            info["bound_reformulation_warm_start_primal_feasible"] =
                reformulated_warm_basis_quality->primal_feasible ? "1" : "0";
            info["bound_reformulation_warm_start_dual_feasible"] =
                reformulated_warm_basis_quality->dual_feasible ? "1" : "0";
        }
        if (reformulated_retry_used) {
            info["bound_reformulation_retry_mode"] = "auto";
        }
        const double obj =
            x.array().isFinite().all()
                ? c_in.dot(x)
                : (std::isfinite(std_sol.obj) ? (std_sol.obj + obj_shift) : std_sol.obj);
        auto sol =
            make_solution_(std_sol.status, std::move(x), obj, std::move(basis_out), std_sol.iters,
                           std::move(info), std_sol.farkas_y, std_sol.farkas_has_cert,
                           std_sol.primal_ray, std_sol.primal_ray_has_cert);
        sol.basis_state.warm_state = std_sol.basis_state.warm_state;
        sol.basis_internal = std_sol.basis_internal;
        sol.nonbasis_internal = std_sol.nonbasis_internal;
        sol.internal_column_labels = std_sol.internal_column_labels;
        sol.internal_row_labels = std_sol.internal_row_labels;
        sol.tableau = std_sol.tableau;
        sol.tableau_rhs = std_sol.tableau_rhs;
        sol.reduced_costs_internal = std_sol.reduced_costs_internal;
        if (std_sol.dual_values.size() >= m_eq) {
            sol.dual_values = std_sol.dual_values.head(m_eq);
            sol.shadow_prices = sol.dual_values;
        }
        sol.dual_values_internal = std_sol.dual_values_internal;
        sol.shadow_prices_internal = std_sol.shadow_prices_internal;
        sol.farkas_y_internal = std_sol.farkas_y_internal;
        sol.primal_ray_internal = std_sol.primal_ray_internal;
        const int outer_warm_start_attempted = solve_stats_.warm_start_attempted;
        solve_stats_ = std_sol.solve_stats;
        solve_stats_.warm_start_attempted =
            std::max(solve_stats_.warm_start_attempted, outer_warm_start_attempted);
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol, m_in));
    }

    SparseMatrix A_model = A_in;
    Eigen::VectorXd b_model = b_in;
    Eigen::VectorXd c_model = c_in;
    Eigen::VectorXd l_model = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd u_model = Eigen::VectorXd::Constant(n, presolve::inf());
    Eigen::VectorXd anchor = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd sign = Eigen::VectorXd::Ones(n);

    for (int j = 0; j < n; ++j) {
        const bool has_l = std::isfinite(l_use(j));
        const bool has_u = std::isfinite(u_use(j));
        if (!has_l && !has_u) {
            throw std::invalid_argument(
                "simplex: free variables are unsupported in solve(A,b,c,l,u)");
        }

        if (has_l) {
            anchor(j) = l_use(j);
            l_model(j) = 0.0;
            u_model(j) = has_u ? (u_use(j) - l_use(j)) : presolve::inf();
        } else {
            anchor(j) = u_use(j);
            sign(j) = -1.0;
            l_model(j) = 0.0;
            u_model(j) = presolve::inf();
            RevisedSimplexDualEngine::scale_column(A_model, j, -1.0);
            c_model(j) = -c_model(j);
        }

        if (anchor(j) != 0.0) {
            const Eigen::VectorXd model_col = A_model.col(j);
            b_model.noalias() -= model_col * anchor(j);
        }
    }

    // Keep one row orientation for crash, Phase I, Phase II and factorization.
    // Equality rows with negative RHS are multiplied by -1 once here instead
    // of only inside the auxiliary model.
    Eigen::VectorXd model_row_sign = Eigen::VectorXd::Ones(m_in);
    for (int i = 0; i < m_in; ++i) {
        if (b_model(i) < 0.0) {
            b_model(i) = -b_model(i);
            model_row_sign(i) = -1.0;
        }
    }
    if ((model_row_sign.array() < 0.0).any()) {
        for (int j = 0; j < A_model.outerSize(); ++j)
            for (SparseMatrix::InnerIterator it(A_model, j); it; ++it)
                it.valueRef() *= model_row_sign(it.row());
    }

    presolve::SparsePresolveResult sparse_pres;
    if (opt_.disable_presolve) {
        sparse_pres.reduced = {A_model, b_model, c_model, l_model, u_model};
        sparse_pres.orig_col_index.resize(n);
        sparse_pres.orig_row_index.resize(m_in);
        std::iota(sparse_pres.orig_col_index.begin(), sparse_pres.orig_col_index.end(), 0);
        std::iota(sparse_pres.orig_row_index.begin(), sparse_pres.orig_row_index.end(), 0);
        sparse_pres.row_scale = Eigen::VectorXd::Ones(m_in);
        sparse_pres.col_scale = Eigen::VectorXd::Ones(n);
    } else {
        const bool warm = basis_state_opt && !basis_state_opt->column_status.empty();
        presolve::SparsePresolver::Options spopt;
        spopt.zero_tol = opt_.tol * 1e-3;
        spopt.infeas_tol = opt_.tol;
        spopt.min_delta = std::max(opt_.tol * 10.0, 1e-12);
        spopt.max_passes = warm ? 2 : 4;
        // Phase I currently starts structural variables at their original
        // zero anchor. Passing tightened positive lower bounds only to Phase
        // II changes bound space across the transition and invalidates the
        // feasible basis. Keep presolve non-destructive until its bound shifts
        // are included in Phase-I RHS construction and postsolve state.
        spopt.enable_singleton_rows = false;
        spopt.enable_activity_tightening = false;
        // HiGHS only enables model transformations that can be undone by its
        // postsolve stack. This presolver does not yet carry row/column scale
        // and RRQR recovery through every exit path, so keep the public solve
        // pipeline non-destructive. Basis factorization has independent
        // equilibration and remains enabled.
        spopt.enable_row_scaling = false;
        spopt.enable_col_scaling = false;
        spopt.enable_row_reduce = false;
        presolve::SparsePresolver sp(spopt);
        sparse_pres = sp.run({A_model, b_model, c_model, l_model, u_model});
    }
    if (sparse_pres.proven_infeasible) {
        return finalize_solution_(make_solution_(
            LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
            std::numeric_limits<double>::infinity(), {}, 0,
            {{"sparse_presolve", "infeasible"},
             {"sparse_presolve_bound_updates", std::to_string(sparse_pres.bound_updates)}}));
    }
    if (sparse_pres.proven_unbounded) {
        Eigen::VectorXd xnan =
            Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        return finalize_solution_(make_solution_(LPSolution::Status::Unbounded, xnan,
                                                 -std::numeric_limits<double>::infinity(), {}, 0,
                                                 {{"sparse_presolve", "unbounded"}}));
    }

    const SparseMatrix& Ared = sparse_pres.reduced.A;
    const Eigen::VectorXd& bred = sparse_pres.reduced.b;
    const Eigen::VectorXd& cred = sparse_pres.reduced.c;
    const Eigen::VectorXd& l_eff = sparse_pres.reduced.l;
    const Eigen::VectorXd& u_eff = sparse_pres.reduced.u;
    std::vector<int> col_orig_map(n);
    std::iota(col_orig_map.begin(), col_orig_map.end(), 0);
    std::vector<int> row_orig_map(m_in);
    std::iota(row_orig_map.begin(), row_orig_map.end(), 0);
    const std::vector<std::string> internal_column_labels =
        make_internal_column_labels_(col_orig_map);
    const std::vector<std::string> internal_row_labels = make_internal_row_labels_(row_orig_map);

    std::optional<std::vector<int>> red_basis_opt = basis_opt;
    std::optional<LPBasis> red_basis_state_opt = std::nullopt;
    if (basis_state_opt && !basis_state_opt->column_status.empty()) {
        red_basis_state_opt =
            map_reduced_basis_state_(*basis_state_opt, col_orig_map, l_eff, u_eff, opt_.tol);
        if (!red_basis_opt || red_basis_opt->empty()) {
            red_basis_opt = basis_columns_from_basis_state_(*red_basis_state_opt, m_in);
        }
    }

    auto t1_presolve = std::chrono::steady_clock::now();
    current_timing_.presolve_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve - t0_presolve).count();

    std::optional<std::vector<int>> crash_seed_basis_opt = red_basis_opt;
    if ((!crash_seed_basis_opt || crash_seed_basis_opt->empty()) && red_basis_state_opt &&
        !red_basis_state_opt->column_status.empty()) {
        auto partial_seed = basis_columns_from_basis_state_(*red_basis_state_opt, -1);
        if (partial_seed && !partial_seed->empty()) {
            crash_seed_basis_opt = std::move(partial_seed);
        }
    }

    // evaluate_basis_quality_ checks dual feasibility assuming all non-basics
    // are at lower bound (standard form). For problems with finite upper bounds,
    // non-basics at their upper bound have negative reduced costs which the
    // check incorrectly flags as violations. When Dual mode is requested and
    // a warm-start with column_status is available, override choose_initial_basis_
    // to use the warm-start directly — column_status encodes AtLower/AtUpper
    // correctly so the dual simplex can set views from it.
    const bool has_warm_column_status_for_dual =
        opt_.mode == SimplexMode::Dual && crash_seed_basis_opt.has_value() &&
        static_cast<int>(crash_seed_basis_opt->size()) == m_in && red_basis_state_opt &&
        !red_basis_state_opt->column_status.empty() &&
        static_cast<int>(red_basis_state_opt->column_status.size()) == n;

    const bool seed_basis_from_state = (!basis_opt || basis_opt->empty()) && red_basis_state_opt &&
                                       !red_basis_state_opt->column_status.empty();
    const bool allow_direct_warm_start =
        !seed_basis_from_state ||
        (crash_seed_basis_opt && static_cast<int>(crash_seed_basis_opt->size()) == m_in);
    auto t0_crash = std::chrono::steady_clock::now();
    CrashSelection basis_choice = choose_initial_basis_(
        Ared, bred, cred, opt_, l_eff, u_eff, crash_seed_basis_opt, allow_direct_warm_start);
    if (factorized_basis_seed_opt && crash_seed_basis_opt &&
        static_cast<int>(crash_seed_basis_opt->size()) == m_in &&
        basis_choice.basis != *crash_seed_basis_opt) {
        basis_choice.basis = *crash_seed_basis_opt;
        basis_choice.source = "warm_start";
        basis_choice.style = "factorization_reuse";
        basis_choice.quality.valid = true;
        basis_choice.quality.dual_feasible = true;
        basis_choice.quality.primal_feasible = false;
    }
    auto t1_crash = std::chrono::steady_clock::now();
    current_timing_.crash_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_crash - t0_crash).count();

    std::vector<int> basis_guess = basis_choice.basis;
    const bool basis_guess_from_warm_start =
        (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start");
    const bool basis_valid = ((int)basis_guess.size() == m_in) && basis_choice.quality.valid;
    const bool allow_direct_primal =
        basis_valid && (basis_choice.quality.primal_feasible || basis_guess_from_warm_start);
    const bool warm_has_column_status =
        red_basis_state_opt && !red_basis_state_opt->column_status.empty() &&
        static_cast<int>(red_basis_state_opt->column_status.size()) == n;
    const bool allow_direct_dual =
        basis_valid &&
        (basis_choice.quality.dual_feasible ||
         (opt_.mode == SimplexMode::Dual && basis_guess_from_warm_start && warm_has_column_status));
    if (std::getenv("SIMPLINHO_TRACE_REFORM") && opt_.mode == SimplexMode::Dual &&
        !allow_direct_dual) {
        std::fprintf(stderr,
                     "[reformdispatch] dual-warm-gate fail source=%s valid=%d dual_feasible=%d "
                     "primal_feasible=%d basis_guess_from_warm_start=%d warm_has_column_status=%d "
                     "basis_size=%d m_in=%d factorized_basis_seed_opt=%d\n",
                     basis_choice.source.c_str(), static_cast<int>(basis_choice.quality.valid),
                     static_cast<int>(basis_choice.quality.dual_feasible),
                     static_cast<int>(basis_choice.quality.primal_feasible),
                     static_cast<int>(basis_guess_from_warm_start),
                     static_cast<int>(warm_has_column_status), static_cast<int>(basis_guess.size()),
                     m_in, static_cast<int>(factorized_basis_seed_opt.has_value()));
    }

    auto add_sparse_info = [&](std::unordered_map<std::string, std::string> info) {
        info["sparse_pipeline"] = "1";
        info["sparse_presolve"] = opt_.disable_presolve ? "disabled" : "sparse";
        info["sparse_presolve_passes"] = std::to_string(sparse_pres.passes);
        info["sparse_presolve_bound_updates"] = std::to_string(sparse_pres.bound_updates);
        info["sparse_presolve_singleton_rows"] = std::to_string(sparse_pres.singleton_rows);
        info["sparse_presolve_zero_rows"] = std::to_string(sparse_pres.zero_rows);
        info["sparse_presolve_zero_columns"] = std::to_string(sparse_pres.zero_columns);
        info["original_m"] = std::to_string(m_in);
        info["reduced_m"] = std::to_string(m_in);
        info["reduced_n"] = std::to_string(n);
        if (!basis_choice.source.empty() && basis_choice.source != "none") {
            info["basis_start"] = basis_choice.source;
            info["basis_start_style"] = basis_choice.style;
        }
        return info;
    };

    auto finalize_sparse_solution = [&](LPSolution::Status status, const Eigen::VectorXd& z,
                                        const std::vector<int>& red_basis, int iters,
                                        std::unordered_map<std::string, std::string> info) {
        if (z.size() != sign.size()) {
            info["where"] = "finalize_sparse_solution";
            info["reason"] = "invalid_primal_dimension";
            info["expected_primal_dim"] = std::to_string(sign.size());
            info["actual_primal_dim"] = std::to_string(z.size());
            return finalize_solution_(make_solution_(
                status, Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN()),
                std::numeric_limits<double>::quiet_NaN(), {}, iters,
                add_sparse_info(std::move(info))));
        }
        Eigen::VectorXd z_unscaled = z;
        if (sparse_pres.col_scale.size() == z.size())
            z_unscaled.array() /= sparse_pres.col_scale.array();
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_unscaled);
        if (status == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
            status = LPSolution::Status::Singular;
            info["reason"] = "original_model_primal_check_failed";
        }
        std::vector<int> basis_full = red_basis;
        const bool has_primal_ray =
            info.count("primal_ray_has_cert") && info.at("primal_ray_has_cert") == "1";
        const auto primal_ray_internal =
            has_primal_ray ? parse_serialized_vec_(info, "primal_ray", n) : std::nullopt;
        double obj = x_full.array().isFinite().all() ? c_in.dot(x_full)
                                                     : std::numeric_limits<double>::quiet_NaN();
        if (status == LPSolution::Status::Unbounded) {
            obj = -std::numeric_limits<double>::infinity();
        }
        auto sol =
            make_solution_(status, x_full, obj, basis_full, iters, add_sparse_info(std::move(info)),
                           std::nullopt, std::nullopt, primal_ray_internal, has_primal_ray);
        sol = attach_internal_tableau_(std::move(sol), Ared, bred, cred, red_basis,
                                       internal_column_labels, internal_row_labels, opt_.tol,
                                       opt_.compute_tableau, opt_.compute_reduced_costs);
        if (sol.dual_values_internal.size() == m_in) {
            sol.dual_values = sol.dual_values_internal;
            // Duals are computed in the sign-normalized row space; map them
            // back to the caller's row orientation.
            sol.dual_values.array() *= model_row_sign.array();
            sol.shadow_prices = sol.dual_values;
        }
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
    };

    // Nonbasic bound statuses to seed phase 2 with (set after phase 1: with
    // native bounds, phase 1 may finish with nonbasics resting at their upper
    // bounds, and restarting them at lower would discard feasibility).
    std::optional<std::vector<LPBasisStatus>> phase2_seed_status;

    auto run_phase2_p = [&](std::optional<std::vector<int>> b, bool ignore_seed_status = false) {
        auto t0 = std::chrono::steady_clock::now();
        const bool use_warm_status =
            red_basis_state_opt &&
            (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start") &&
            red_basis_state_opt->column_status.size() == static_cast<std::size_t>(n);
        std::optional<std::vector<LPBasisStatus>> ws;
        if (phase2_seed_status && !ignore_seed_status)
            ws = phase2_seed_status;
        else if (use_warm_status)
            ws = red_basis_state_opt->column_status;
        try {
            auto res = phase_(Ared, bred, cred, std::move(b), l_eff, u_eff, std::move(ws));
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            return res;
        } catch (const std::runtime_error& e) {
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            return PhaseResult{LPSolution::Status::Singular,
                               Eigen::VectorXd{},
                               {},
                               0,
                               {{"reason", "basis_factorization_failure"},
                                {"what", e.what()},
                                {"where", "sparse_direct_primal"}}};
        }
    };
    auto run_phase2_d = [&](std::optional<std::vector<int>> b, bool ignore_seed_status = false) {
        auto t0 = std::chrono::steady_clock::now();
        const bool use_warm_status =
            red_basis_state_opt &&
            (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start") &&
            red_basis_state_opt->column_status.size() == static_cast<std::size_t>(n);
        std::optional<std::vector<LPBasisStatus>> ws;
        if (phase2_seed_status && !ignore_seed_status)
            ws = phase2_seed_status;
        else if (use_warm_status)
            ws = red_basis_state_opt->column_status;
        try {
            auto res = dual_phase_(Ared, bred, cred, std::move(b), l_eff, u_eff, std::move(ws));
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            return res;
        } catch (const std::runtime_error& e) {
            auto t1 = std::chrono::steady_clock::now();
            current_timing_.simplex_iters_ns +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            return PhaseResult{LPSolution::Status::Singular,
                               Eigen::VectorXd{},
                               {},
                               0,
                               {{"reason", "basis_factorization_failure"},
                                {"what", e.what()},
                                {"where", "sparse_direct_dual"}}};
        }
    };

    if (allow_direct_primal || allow_direct_dual) {
        if (basis_guess_from_warm_start) {
            solve_stats_.warm_start_accepted = 1;
        }
        if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
            static std::atomic<long> trace_reform_dispatch{0};
            static std::atomic<long> trace_reform_primal{0};
            static std::atomic<long> trace_reform_dual{0};
            static std::atomic<long> trace_reform_auto_dual{0};
            const long count = ++trace_reform_dispatch;
            if ((count % 200) == 0) {
                std::fprintf(stderr,
                             "[reformdispatch] count=%ld mode=%d allow_primal=%d allow_dual=%d "
                             "basis_guess_from_warm_start=%d allow_direct_dual=%d\n",
                             count, static_cast<int>(opt_.mode),
                             static_cast<int>(allow_direct_primal),
                             static_cast<int>(allow_direct_dual),
                             static_cast<int>(basis_guess_from_warm_start),
                             static_cast<int>(allow_direct_dual));
            }
        }
        LPSolution::Status st = LPSolution::Status::NeedPhase1;
        Eigen::VectorXd v2;
        std::vector<int> red_basis2;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if (opt_.mode == SimplexMode::Dual) {
            if (allow_direct_dual) {
                if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                    static std::atomic<long> dual_branch_count{0};
                    const long c = ++dual_branch_count;
                    if ((c % 100) == 0)
                        std::fprintf(stderr,
                                     "[reformdispatch] dual-branch direct run_phase2_d count=%ld\n",
                                     c);
                }
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_d(basis_guess);
            } else if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                static std::atomic<long> dual_branch_skipped{0};
                const long c = ++dual_branch_skipped;
                if ((c % 100) == 0)
                    std::fprintf(
                        stderr,
                        "[reformdispatch] dual-branch skipped allow_direct_dual=false count=%ld\n",
                        c);
            }
        } else if (opt_.mode == SimplexMode::Primal) {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_p(basis_guess);
            }
        } else {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_p(basis_guess);
            }
            if (allow_direct_dual && st == LPSolution::Status::NeedPhase1 &&
                info2.count("reason") && info2.at("reason") == std::string("negative_basic_vars")) {
                if (std::getenv("SIMPLINHO_TRACE_REFORM")) {
                    static std::atomic<long> auto_dual_branch_count{0};
                    const long c = ++auto_dual_branch_count;
                    if ((c % 100) == 0)
                        std::fprintf(stderr,
                                     "[reformdispatch] auto fallback dual run_phase2_d count=%ld\n",
                                     c);
                }
                std::tie(st, v2, red_basis2, it2, info2) = run_phase2_d(basis_guess);
            }
        }

        const bool direct_dual_native_bounds =
            opt_.mode == SimplexMode::Dual && allow_direct_dual && has_upper_bounds;
        if (st == LPSolution::Status::Optimal) {
            Eigen::VectorXd v2_unscaled = v2;
            if (sparse_pres.col_scale.size() == v2.size())
                v2_unscaled.array() /= sparse_pres.col_scale.array();
            const Eigen::VectorXd x_full = anchor + sign.cwiseProduct(v2_unscaled);
            if (primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
                return finalize_sparse_solution(st, v2, red_basis2, it2, std::move(info2));
            }
            info2["reason"] = "invalid_returned_primal";
        } else if (st == LPSolution::Status::Unbounded ||
                   (st == LPSolution::Status::IterLimit && !direct_dual_native_bounds) ||
                   st == LPSolution::Status::ObjectiveBound ||
                   (st == LPSolution::Status::Infeasible && !basis_guess_from_warm_start &&
                    !direct_dual_native_bounds)) {
            return finalize_sparse_solution(st, v2, red_basis2, it2, std::move(info2));
        } else if (st == LPSolution::Status::IterLimit) {
            info2["direct_phase2_status"] = to_string(st);
            info2["direct_phase2_recovery"] = "phase1";
        } else if (st == LPSolution::Status::Infeasible) {
            info2["direct_phase2_status"] = to_string(st);
            info2["direct_phase2_recovery"] = "phase1";
            info2["direct_phase2_recovery_reason"] = "cross_check_infeasibility";
        }
    }

    auto t1_presolve2 = std::chrono::steady_clock::now();
    current_timing_.presolve_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve2 - t1_crash).count();

    auto [A1, b1, c1, basis1, n_orig_eff, m_rows] = make_phase1_(Ared, bred);
    // Phase-1 bounds: original columns keep their finite upper bounds so the
    // feasible point found also respects them; lower bounds stay 0 (the
    // artificial identity basis starts from x = 0). Artificials are [0, inf).
    Eigen::VectorXd l_phase1 = Eigen::VectorXd::Zero(A1.cols());
    Eigen::VectorXd u_phase1 = Eigen::VectorXd::Constant(A1.cols(), presolve::inf());
    if (u_eff.size() == static_cast<Eigen::Index>(n_orig_eff)) {
        u_phase1.head(n_orig_eff) = u_eff;
    }
    PhaseResult phase1_result;
    try {
        phase1_result = phase_(A1, b1, c1, basis1, l_phase1, u_phase1);
    } catch (const std::runtime_error& e) {
        if (!is_recoverable_basis_runtime_(e.what()))
            throw;
        phase1_result = PhaseResult{LPSolution::Status::Singular,
                                    Eigen::VectorXd{},
                                    {},
                                    0,
                                    {{"reason", "basis_factorization_failure"},
                                     {"what", e.what()},
                                     {"where", "sparse_phase1_primal"}}};
    }
    auto [status1, v1, basis1_out, it1, info1] = std::move(phase1_result);
    if (status1 == LPSolution::Status::NeedPhase1 && info1.count("reason") &&
        info1.at("reason") == std::string("negative_basic_vars")) {
        try {
            std::tie(status1, v1, basis1_out, it1, info1) = dual_phase_(
                A1, b1, c1, basis1_out.empty() ? basis1 : basis1_out, l_phase1, u_phase1);
        } catch (const std::runtime_error& e) {
            if (!is_recoverable_basis_runtime_(e.what()))
                throw;
            status1 = LPSolution::Status::Singular;
            v1.resize(0);
            basis1_out.clear();
            it1 = 0;
            info1 = {{"reason", "basis_factorization_failure"},
                     {"what", e.what()},
                     {"where", "sparse_phase1_dual"}};
        }
    }
    if (status1 == LPSolution::Status::Singular || status1 == LPSolution::Status::IterLimit) {
        auto info = add_sparse_info(std::move(info1));
        info["phase1_status"] = to_string(status1);
        return finalize_solution_(make_solution_(status1, Eigen::VectorXd::Zero(n),
                                                 std::numeric_limits<double>::quiet_NaN(), {}, it1,
                                                 std::move(info)));
    }
    if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
        auto info = add_sparse_info({{"phase1_status", to_string(status1)}});
        return finalize_solution_(
            make_solution_(LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                           std::numeric_limits<double>::infinity(), {}, it1, std::move(info)));
    }
    // Preserve the Phase-I feasible point while removing artificials.  Each
    // replacement is a zero-step tableau pivot, as in HiGHS basis repair.
    {
        auto cleanup_candidate_feasible = [&](const std::vector<int>& candidate) {
            std::vector<char> in_basis(A1.cols(), 0);
            for (int j : candidate)
                in_basis[j] = 1;
            Eigen::VectorXd rhs = b1;
            for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
                if (in_basis[j])
                    continue;
                double value = l_phase1(j);
                if (std::isfinite(u_phase1(j)) && v1.size() > j &&
                    std::abs(v1(j) - u_phase1(j)) <= 10.0 * opt_.tol)
                    value = u_phase1(j);
                if (value != 0.0)
                    for (SparseMatrix::InnerIterator it(A1, j); it; ++it)
                        rhs(it.row()) -= it.value() * value;
            }
            const SparseMatrix B = sparse_basis_copy_(A1, candidate);
            Eigen::SparseLU<SparseMatrix> test_lu;
            test_lu.analyzePattern(B);
            test_lu.factorize(B);
            if (test_lu.info() != Eigen::Success)
                return false;
            const Eigen::VectorXd xb = test_lu.solve(rhs);
            if (test_lu.info() != Eigen::Success || !xb.allFinite())
                return false;
            for (int k = 0; k < static_cast<int>(candidate.size()); ++k) {
                const int j = candidate[k];
                if (j >= static_cast<int>(n_orig_eff)) {
                    if (std::abs(xb(k)) > opt_.tol)
                        return false;
                } else if (xb(k) < l_phase1(j) - opt_.tol ||
                           xb(k) > u_phase1(j) + opt_.tol) {
                    return false;
                }
            }
            return true;
        };
        std::vector<char> basic(A1.cols(), 0);
        for (int j : basis1_out)
            basic[j] = 1;
        for (int r = 0; r < m_rows; ++r) {
            if (basis1_out[r] < static_cast<int>(n_orig_eff))
                continue;
            const SparseMatrix B = sparse_basis_copy_(A1, basis1_out);
            Eigen::SparseLU<SparseMatrix> lu;
            lu.analyzePattern(B);
            lu.factorize(B);
            bool replaced = false;
            if (lu.info() == Eigen::Success) {
                for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
                    if (basic[j])
                        continue;
                    const Eigen::VectorXd d = lu.solve(Eigen::VectorXd(A1.col(j)));
                    if (lu.info() != Eigen::Success || !d.allFinite() ||
                        std::abs(d(r)) <= opt_.alpha_tol)
                        continue;
                    std::vector<int> candidate = basis1_out;
                    candidate[r] = j;
                    if (!cleanup_candidate_feasible(candidate))
                        continue;
                    basic[basis1_out[r]] = 0;
                    basis1_out[r] = j;
                    basic[j] = 1;
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                auto info = add_sparse_info({{"reason", "phase1_artificial_cleanup_failed"},
                                             {"row", std::to_string(r)}});
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
                    std::numeric_limits<double>::quiet_NaN(), {}, it1, std::move(info)));
            }
        }
        if (!cleanup_candidate_feasible(basis1_out)) {
            auto info = add_sparse_info({{"reason", "phase1_cleanup_lost_feasibility"}});
            return finalize_solution_(make_solution_(
                LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::quiet_NaN(), {}, it1, std::move(info)));
        }
    }

    // Phase 1 may finish with nonbasics at their upper bounds; seed phase 2
    // with those statuses so its starting point matches the feasible point.
    if (v1.size() >= static_cast<Eigen::Index>(n_orig_eff) && static_cast<int>(n_orig_eff) == n) {
        std::vector<LPBasisStatus> st(n, LPBasisStatus::AtLower);
        for (int j = 0; j < n; ++j) {
            const bool near_u =
                std::isfinite(u_eff(j)) && std::abs(v1(j) - u_eff(j)) <= 10.0 * opt_.tol;
            const bool near_l =
                std::isfinite(l_eff(j)) && std::abs(v1(j) - l_eff(j)) <= 10.0 * opt_.tol;
            if (near_u && !near_l)
                st[j] = LPBasisStatus::AtUpper;
        }
        phase2_seed_status = std::move(st);
    }
    auto phase2_effective_rhs = [&](const std::vector<int>& basis) {
        Eigen::VectorXd rhs = bred;
        if (!phase2_seed_status ||
            phase2_seed_status->size() != static_cast<std::size_t>(n_orig_eff))
            return rhs;
        std::vector<char> in_basis(static_cast<std::size_t>(n_orig_eff), 0);
        for (int j : basis)
            if (j >= 0 && j < static_cast<int>(n_orig_eff))
                in_basis[static_cast<std::size_t>(j)] = 1;
        for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
            if (in_basis[static_cast<std::size_t>(j)])
                continue;
            double xj = 0.0;
            if ((*phase2_seed_status)[j] == LPBasisStatus::AtUpper && std::isfinite(u_eff(j)))
                xj = u_eff(j);
            else if (std::isfinite(l_eff(j)))
                xj = l_eff(j);
            if (xj != 0.0) {
                for (SparseMatrix::InnerIterator it(Ared, j); it; ++it)
                    rhs(it.row()) -= it.value() * xj;
            }
        }
        return rhs;
    };
    auto phase2_basis_primal_feasible = [&](const std::vector<int>& basis) {
        return basis_is_primal_feasible_(Ared, phase2_effective_rhs(basis), basis, l_eff, u_eff,
                                         opt_.tol);
    };

    std::vector<int> red_basis2;
    red_basis2.reserve(m_rows);
    for (int j : basis1_out)
        if (j < (int)n_orig_eff)
            red_basis2.push_back(j);

    if ((int)red_basis2.size() == m_rows && !phase2_basis_primal_feasible(red_basis2)) {
        auto info = add_sparse_info({{"reason", "phase1_basis_not_feasible_in_phase2_space"}});
        return finalize_solution_(make_solution_(
            LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
            std::numeric_limits<double>::quiet_NaN(), red_basis2, it1, std::move(info)));
    }

    if ((int)red_basis2.size() < m_rows) {
        std::vector<int> fallback_basis = red_basis2;
        for (int j = 0; j < (int)n_orig_eff; ++j) {
            if ((int)red_basis2.size() == m_rows)
                break;
            if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end())
                continue;
            std::vector<int> cand = red_basis2;
            cand.push_back(j);
            if ((int)cand.size() > m_rows)
                continue;
            if (!sparse_basis_has_full_rank_(Ared, cand))
                continue;
            if (phase2_basis_primal_feasible(cand)) {
                red_basis2 = std::move(cand);
                continue;
            }
            if ((int)fallback_basis.size() < (int)cand.size()) {
                fallback_basis = cand;
            }
        }
        if ((int)red_basis2.size() < m_rows && (int)fallback_basis.size() == m_rows) {
            red_basis2 = std::move(fallback_basis);
        }
    }
    if ((int)red_basis2.size() == m_rows) {
        const bool phase2_start_primal_feasible = phase2_basis_primal_feasible(red_basis2);
        const BasisQuality phase2_start_quality =
            evaluate_basis_quality_(Ared, bred, cred, red_basis2, l_eff, u_eff, opt_.tol);
        const double solve_residual_guard = std::max(1e-7, 100.0 * opt_.tol);
        // evaluate_basis_quality_ assumes all nonbasics are at their lower
        // bounds. It must not replace a basis already certified feasible with
        // Phase-I's actual AtUpper statuses.
        if (!phase2_start_primal_feasible &&
            (!phase2_start_quality.valid ||
             !std::isfinite(phase2_start_quality.solve_residual) ||
             phase2_start_quality.solve_residual > solve_residual_guard)) {
            const CrashSelection repaired_phase2_start =
                choose_initial_basis_(Ared, bred, cred, opt_, l_eff, u_eff, red_basis2);
            if (repaired_phase2_start.quality.valid &&
                better_basis_quality_(
                    repaired_phase2_start,
                    CrashSelection{red_basis2, phase2_start_quality, "phase1_basis", "phase1", -1},
                    opt_.mode)) {
                red_basis2 = repaired_phase2_start.basis;
            }
        }
    }
    if ((int)red_basis2.size() != m_rows || !phase2_basis_primal_feasible(red_basis2)) {
        std::vector<int> candidates;
        candidates.reserve(static_cast<std::size_t>(n_orig_eff));
        for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
            if (v1.size() > j && v1(j) > 10.0 * opt_.tol)
                candidates.push_back(j);
        }
        std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            const double va = (v1.size() > a) ? v1(a) : 0.0;
            const double vb = (v1.size() > b) ? v1(b) : 0.0;
            return va > vb;
        });
        for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
            if (std::find(candidates.begin(), candidates.end(), j) == candidates.end())
                candidates.push_back(j);
        }
        auto sparse_columns_independent = [&](const std::vector<int>& cols) {
            if (cols.empty())
                return true;
            Eigen::MatrixXd Btest(Ared.rows(), static_cast<int>(cols.size()));
            for (int k = 0; k < static_cast<int>(cols.size()); ++k)
                Btest.col(k) = Eigen::VectorXd(Ared.col(cols[k]));
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
            return lu.rank() == static_cast<int>(cols.size());
        };
        std::vector<int> support_basis;
        support_basis.reserve(m_rows);
        for (int j : candidates) {
            if (static_cast<int>(support_basis.size()) == m_rows)
                break;
            std::vector<int> trial = support_basis;
            trial.push_back(j);
            if (sparse_columns_independent(trial))
                support_basis = std::move(trial);
        }
        if (static_cast<int>(support_basis.size()) == m_rows &&
            phase2_basis_primal_feasible(support_basis)) {
            red_basis2 = std::move(support_basis);
        }
    }

    LPSolution::Status status2;
    Eigen::VectorXd v2;
    std::vector<int> red_basis_out;
    int it2 = 0;
    std::unordered_map<std::string, std::string> info2;
    auto phase2_result_needs_basis_repair = [&](LPSolution::Status status,
                                                const Eigen::VectorXd& primal) {
        return status == LPSolution::Status::Singular || status == LPSolution::Status::NeedPhase1 ||
               (status == LPSolution::Status::Optimal && primal.size() != n) ||
               (status == LPSolution::Status::Unbounded && primal.size() != n) ||
               (status == LPSolution::Status::IterLimit && primal.size() != 0 &&
                primal.size() != n);
    };
    auto run_sparse_phase2_from_basis = [&](const std::vector<int>& basis,
                                            bool ignore_seed_status = false) {
        if (opt_.mode == SimplexMode::Dual) {
            const auto phase2_basis_quality =
                evaluate_basis_quality_(Ared, bred, cred, basis, l_eff, u_eff, opt_.tol);
            if (phase2_basis_quality.valid && phase2_basis_quality.dual_feasible) {
                auto res = run_phase2_d(basis, ignore_seed_status);
                std::get<4>(res)["phase2_mode"] = "dual";
                const LPSolution::Status dual_status = std::get<0>(res);
                const Eigen::VectorXd& dual_primal = std::get<1>(res);
                const bool invalid_dual_optimum =
                    dual_status == LPSolution::Status::Optimal &&
                    (dual_primal.size() != n ||
                     !primal_feasible_(Ared, bred, dual_primal, l_eff, u_eff, opt_.tol));
                const bool recover_with_primal = dual_status == LPSolution::Status::Singular ||
                                                 dual_status == LPSolution::Status::IterLimit ||
                                                 dual_status == LPSolution::Status::NeedPhase1 ||
                                                 dual_status == LPSolution::Status::Infeasible ||
                                                 invalid_dual_optimum;
                if (recover_with_primal) {
                    const bool dual_basis_is_reliable =
                        dual_status == LPSolution::Status::Infeasible || invalid_dual_optimum;
                    const std::vector<int>& recovery_basis =
                        dual_basis_is_reliable &&
                                static_cast<int>(std::get<2>(res).size()) == m_rows
                            ? std::get<2>(res)
                            : basis;
                    auto primal_res = run_phase2_p(recovery_basis);
                    std::get<4>(primal_res)["phase2_mode"] = "primal";
                    std::get<4>(primal_res)["phase2_dual_recovery"] = "1";
                    std::get<4>(primal_res)["phase2_dual_recovery_status"] = to_string(dual_status);
                    if (const auto reason = std::get<4>(res).find("reason");
                        reason != std::get<4>(res).end())
                        std::get<4>(primal_res)["phase2_dual_recovery_detail"] = reason->second;
                    if (invalid_dual_optimum)
                        std::get<4>(primal_res)["phase2_dual_recovery_reason"] =
                            "invalid_returned_primal";
                    else if (dual_status == LPSolution::Status::Infeasible)
                        std::get<4>(primal_res)["phase2_dual_recovery_reason"] =
                            "cross_check_infeasibility";
                    return primal_res;
                }
                return res;
            }
            auto res = run_phase2_p(basis, ignore_seed_status);
            std::get<4>(res)["phase2_mode"] = "primal";
            std::get<4>(res)["phase2_dual_requested_but_basis_not_dual_feasible"] = "1";
            return res;
        }
        if (opt_.mode == SimplexMode::Primal) {
            auto res = run_phase2_p(basis, ignore_seed_status);
            std::get<4>(res)["phase2_mode"] = "primal";
            return res;
        }
        auto res = run_phase2_p(basis, ignore_seed_status);
        std::get<4>(res)["phase2_mode"] = "primal";
        if (std::get<0>(res) == LPSolution::Status::NeedPhase1 &&
            std::get<4>(res).count("reason") &&
            std::get<4>(res).at("reason") == std::string("negative_basic_vars")) {
            res = run_phase2_d(basis, ignore_seed_status);
            std::get<4>(res)["phase2_mode"] = "dual";
        }
        return res;
    };

    if ((int)red_basis2.size() == m_rows) {
        std::tie(status2, v2, red_basis_out, it2, info2) = run_sparse_phase2_from_basis(red_basis2);
    } else {
        std::tie(status2, v2, red_basis_out, it2, info2) = run_phase2_p(std::nullopt);
    }

    const bool certified_phase2_seed =
        static_cast<int>(red_basis2.size()) == m_rows && phase2_basis_primal_feasible(red_basis2);
    if (phase2_result_needs_basis_repair(status2, v2) &&
        !(status2 == LPSolution::Status::NeedPhase1 && certified_phase2_seed)) {
        const std::optional<std::vector<int>> seeded_repair_basis =
            ((int)red_basis2.size() == m_rows) ? std::optional<std::vector<int>>(red_basis2)
                                               : std::nullopt;
        RevisedSimplexOptions repair_opt = opt_;
        repair_opt.mode = SimplexMode::Primal;
        const CrashSelection repaired_seed =
            choose_initial_basis_(Ared, bred, cred, repair_opt, l_eff, u_eff, seeded_repair_basis);
        const CrashSelection repaired_cold =
            choose_initial_basis_(Ared, bred, cred, repair_opt, l_eff, u_eff, std::nullopt);

        const CrashSelection* repaired = nullptr;
        if (repaired_seed.quality.valid &&
            (!repaired_cold.quality.valid ||
             better_basis_quality_(repaired_seed, repaired_cold, repair_opt.mode))) {
            repaired = &repaired_seed;
        } else if (repaired_cold.quality.valid) {
            repaired = &repaired_cold;
        }

        if (repaired && (int)repaired->basis.size() == m_rows) {
            LPSolution::Status repaired_status;
            Eigen::VectorXd repaired_v;
            std::vector<int> repaired_basis_out;
            int repaired_iters = 0;
            std::unordered_map<std::string, std::string> repaired_info;
            std::tie(repaired_status, repaired_v, repaired_basis_out, repaired_iters,
                     repaired_info) = run_sparse_phase2_from_basis(repaired->basis);
            if (repaired_status == LPSolution::Status::NeedPhase1 &&
                repaired_info.count("reason") &&
                repaired_info.at("reason") == std::string("optimal_dual_check_failed")) {
                std::tie(repaired_status, repaired_v, repaired_basis_out, repaired_iters,
                         repaired_info) = run_sparse_phase2_from_basis(repaired->basis, true);
            }
            if (!phase2_result_needs_basis_repair(repaired_status, repaired_v) ||
                (repaired_status == LPSolution::Status::Optimal && repaired_v.size() == n)) {
                status2 = repaired_status;
                v2 = std::move(repaired_v);
                red_basis_out = std::move(repaired_basis_out);
                it2 += repaired_iters;
                repaired_info["phase2_basis_repair"] = "1";
                repaired_info["phase2_basis_repair_source"] = repaired->source;
                repaired_info["phase2_basis_repair_style"] = repaired->style;
                repaired_info["phase2_basis_repair_attempt"] = std::to_string(repaired->attempt);
                info2 = std::move(repaired_info);
            } else {
                info2["phase2_basis_repair_attempted"] = "1";
                info2["phase2_basis_repair_source"] = repaired->source;
                info2["phase2_basis_repair_style"] = repaired->style;
            }
        }
    }

    if (status2 == LPSolution::Status::NeedPhase1 && info2.count("reason") &&
        info2.at("reason") == std::string("optimal_dual_check_failed")) {
        RevisedSimplexOptions cold_primal_opt = opt_;
        cold_primal_opt.mode = SimplexMode::Primal;
        cold_primal_opt.disable_presolve = true;
        RevisedSimplex cold_primal(cold_primal_opt);
        LPSolution cold = cold_primal.solve(Ared, bred, cred, l_eff, u_eff);
        if (cold.status == LPSolution::Status::Optimal && cold.x.size() == n &&
            primal_feasible_(Ared, bred, cold.x, l_eff, u_eff, opt_.tol)) {
            status2 = cold.status;
            v2 = cold.x;
            red_basis_out = cold.basis;
            info2 = cold.info;
            info2["phase2_mode"] = "primal";
            info2["phase2_cold_primal_recovery"] = "1";
            info2["phase2_cold_primal_recovery_reason"] = "optimal_dual_check_failed";
        }
    }

    return finalize_sparse_solution(status2, v2, red_basis_out, it1 + it2, std::move(info2));
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::dual_phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexDualEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                         std::move(warm_status));
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::phase_(const SparseMatrix& A, const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt, const Eigen::VectorXd& l,
                       const Eigen::VectorXd& u,
                       std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexPrimalEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                           std::move(warm_status));
}

inline RevisedSimplex::PhaseResult
RevisedSimplex::dual_phase_(const SparseMatrix& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                            std::optional<std::vector<LPBasisStatus>> warm_status) {
    return RevisedSimplexDualEngine::run(*this, A, b, c, std::move(basis_opt), l, u,
                                         std::move(warm_status));
}
