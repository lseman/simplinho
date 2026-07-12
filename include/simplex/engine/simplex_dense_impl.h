#pragma once

// Internal dense solve implementation included at the end of simplex.h after
// RevisedSimplex is fully declared.

inline LPSolution
RevisedSimplex::solve_impl_(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                            const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                            const Eigen::VectorXd& u_in, std::optional<std::vector<int>> basis_opt,
                            const LPBasis* basis_state_opt) {
    auto t0_presolve = std::chrono::steady_clock::now();
    SolveTraceScope trace_scope(*this);
    degen_.reset();
    const int n = static_cast<int>(A_in.cols());
    if (b_in.size() != A_in.rows()) {
        throw std::invalid_argument("simplex: b size mismatch with rows(A)");
    }
    if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
        throw std::invalid_argument("simplex: c/l/u sizes must equal cols(A)");
    }
    begin_solve_(matrix_signature_(A_in), static_cast<int>(A_in.rows()), n, false, basis_state_opt);

    trace_line_("[solve] start m=" + std::to_string(A_in.rows()) + " n=" + std::to_string(n));
    trace_line_("[solve] disable_presolve=" + std::to_string(opt_.disable_presolve));

    const RowRankReduction row_rank = dependent_row_reduction_(A_in, b_in, opt_.tol);
    if (row_rank.needed) {
        if (row_rank.inconsistent) {
            return finalize_solution_(make_solution_(
                LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::infinity(), {}, 0,
                {{"reason", "inconsistent_dependent_rows"},
                 {"row_rank_reduction_original_m", std::to_string(row_rank.original_rows)},
                 {"row_rank_reduction_rank", std::to_string(row_rank.rank)}}));
        }
        LPSolution reduced_sol = solve_impl_(select_dense_rows_(A_in, row_rank.keep_rows),
                                             select_vector_rows_(b_in, row_rank.keep_rows), c_in,
                                             l_in, u_in, std::nullopt, nullptr);
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

    const auto sanitized_bounds = canonicalize_inactive_huge_bounds_(A_in, b_in, l_in, u_in);
    const Eigen::VectorXd& l_use = sanitized_bounds.l;
    const Eigen::VectorXd& u_use = sanitized_bounds.u;
    if (sanitized_bounds.relaxed_upper > 0 || sanitized_bounds.relaxed_lower > 0) {
        trace_line_("[solve] relaxed huge inactive bounds upper=" +
                    std::to_string(sanitized_bounds.relaxed_upper) +
                    " lower=" + std::to_string(sanitized_bounds.relaxed_lower));
    }

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
    if (auto factorized_seed = warm_factorization_basis_seed_()) {
        basis_opt = std::move(factorized_seed);
    }

    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        if ((int)basis_state_opt->column_status.size() != n) {
            throw std::invalid_argument("simplex: warm-start basis column_status size mismatch");
        }
    }

    if ((!basis_opt || basis_opt->empty()) && basis_state_opt &&
        !basis_state_opt->column_status.empty()) {
        basis_opt =
            basis_columns_from_basis_state_(*basis_state_opt, static_cast<int>(A_in.rows()));
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

    // Native-bounds path: bounded variables are handled directly by the
    // engines (anchored model below). The standard-form reformulation is
    // only needed for free variables (both bounds infinite), when
    // explicitly requested via opt_.native_bounds = false, or when the
    // dual engine might run (mode Dual or Auto): native dual handling for
    // two-sided bounds is still guarded by the reformulation path until
    // its larger sparse-LP stability issues are resolved. Only
    // Primal-forced solves use the native path when upper bounds are
    // present.
    bool has_upper_bounds = false;
    for (int j = 0; j < n; ++j) {
        if (std::isfinite(u_use(j))) {
            has_upper_bounds = true;
            break;
        }
    }
    const bool use_reformulation = !is_nonnegative_standard &&
                                   (has_free_vars || !opt_.native_bounds ||
                                    (opt_.mode != SimplexMode::Primal && has_upper_bounds)) &&
                                   !std::getenv("SIMPLINHO_FORCE_NATIVE_DUAL");
    if (use_reformulation) {
        struct ReformVar {
            int y = -1;
            int y_pos = -1;
            int y_neg = -1;
            int upper_slack = -1;
            double shift = 0.0;
            int sign = 1;
            bool uses_single_var = false;
            bool has_upper_row = false;
        };

        std::vector<ReformVar> map(n);
        std::vector<int> single_y(n, -1);
        std::vector<int> upper_slack(n, -1);
        std::vector<int> split_pos(n, -1);
        std::vector<int> split_neg(n, -1);
        int nv = 0;
        int upper_rows = 0;
        double obj_shift = 0.0;

        for (int j = 0; j < n; ++j) {
            const bool has_l = std::isfinite(l_use(j));
            const bool has_u = std::isfinite(u_use(j));

            if (has_l && has_u && u_use(j) < l_use(j) - opt_.tol) {
                Eigen::VectorXd xnan =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                return finalize_solution_(make_solution_(LPSolution::Status::Infeasible, xnan,
                                                         std::numeric_limits<double>::infinity(),
                                                         {}, 0, {{"reason", "invalid_bounds"}}));
            }

            const bool fixed = has_l && has_u && std::abs(u_use(j) - l_use(j)) <= opt_.tol;
            if (fixed) {
                map[j].uses_single_var = true;
                map[j].y = -1;
                single_y[j] = -1;
                map[j].shift = l_use(j);
                map[j].sign = 0;
                obj_shift += c_in(j) * l_use(j);
            } else if (has_l) {
                map[j].uses_single_var = true;
                map[j].y = nv++;
                single_y[j] = map[j].y;
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
                single_y[j] = map[j].y;
                map[j].shift = u_use(j);
                map[j].sign = -1;
                obj_shift += c_in(j) * u_use(j);
            } else {
                map[j].y_pos = nv++;
                map[j].y_neg = nv++;
                split_pos[j] = map[j].y_pos;
                split_neg[j] = map[j].y_neg;
            }
        }

        const int m_eq = static_cast<int>(A_in.rows());
        const int n_total = nv + upper_rows;
        const int m_total = m_eq + upper_rows;
        trace_line_("[solve] bound reformulation nv=" + std::to_string(nv) +
                    " upper_rows=" + std::to_string(upper_rows) +
                    " total_m=" + std::to_string(m_total) + " total_n=" + std::to_string(n_total));

        Eigen::MatrixXd A_std = Eigen::MatrixXd::Zero(m_total, n_total);
        Eigen::VectorXd b_std = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd c_std = Eigen::VectorXd::Zero(n_total);
        Eigen::VectorXd l_std = Eigen::VectorXd::Zero(n_total);
        Eigen::VectorXd u_std = Eigen::VectorXd::Constant(n_total, presolve::inf());

        for (int j = 0; j < n; ++j) {
            if (map[j].uses_single_var) {
                if (map[j].y >= 0)
                    c_std(map[j].y) += static_cast<double>(map[j].sign) * c_in(j);
            } else {
                c_std(map[j].y_pos) += c_in(j);
                c_std(map[j].y_neg) += -c_in(j);
            }
        }

        int row = 0;
        for (int i = 0; i < m_eq; ++i, ++row) {
            double rhs = b_in(i);
            for (int j = 0; j < n; ++j) {
                const double aij = A_in(i, j);
                if (std::abs(aij) <= 1e-16)
                    continue;
                if (map[j].uses_single_var) {
                    rhs -= aij * map[j].shift;
                    if (map[j].y >= 0)
                        A_std(row, map[j].y) += static_cast<double>(map[j].sign) * aij;
                } else {
                    A_std(row, map[j].y_pos) += aij;
                    A_std(row, map[j].y_neg) += -aij;
                }
            }
            b_std(row) = rhs;
        }

        int upper_row = 0;
        for (int j = 0; j < n; ++j) {
            if (!map[j].has_upper_row)
                continue;
            const int slack = nv + upper_row;
            map[j].upper_slack = slack;
            upper_slack[j] = slack;
            A_std(row, map[j].y) = 1.0;
            A_std(row, slack) = 1.0;
            b_std(row) = u_use(j) - l_use(j);
            ++upper_row;
            ++row;
        }

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
            std::vector<int> cand;
            cand.reserve(m_total);
            std::vector<bool> col_used(n_total, false);
            // Original rows: pick first available y/y_pos with nonzero in that row
            for (int i = 0; i < m_eq; ++i) {
                int chosen = -1;
                for (int j = 0; j < n && chosen < 0; ++j) {
                    if (map[j].uses_single_var) {
                        const int col = map[j].y;
                        if (col >= 0 && !col_used[col] && std::abs(A_std(i, col)) > 1e-14) {
                            chosen = col;
                        }
                    } else if (map[j].y_pos >= 0) {
                        const int col = map[j].y_pos;
                        if (!col_used[col] && std::abs(A_std(i, col)) > 1e-14) {
                            chosen = col;
                        }
                    }
                }
                if (chosen < 0)
                    break;
                col_used[chosen] = true;
                cand.push_back(chosen);
            }
            // Upper-bound rows: each has a dedicated slack that is an identity column
            for (int j = 0; j < n; ++j) {
                if (map[j].upper_slack >= 0)
                    cand.push_back(map[j].upper_slack);
            }
            if ((int)cand.size() == m_total)
                basis_std = std::move(cand);
        }

        LPSolution std_sol;
        bool reformulated_retry_used = false;
        std::optional<std::vector<int>> reformulated_basis_guess = basis_std;
        if ((!reformulated_basis_guess || reformulated_basis_guess->empty()) && basis_state_std &&
            !basis_state_std->column_status.empty()) {
            reformulated_basis_guess = basis_columns_from_basis_state_(*basis_state_std, m_total);
        }
        std::optional<BasisQuality> reformulated_warm_basis_quality = std::nullopt;
        if (reformulated_basis_guess && !reformulated_basis_guess->empty()) {
            reformulated_warm_basis_quality = evaluate_basis_quality_(
                A_std, b_std, c_std, *reformulated_basis_guess, l_std, u_std, opt_.tol);
        }
        const bool use_dual_first = opt_.mode != SimplexMode::Primal &&
                                    reformulated_warm_basis_quality &&
                                    reformulated_warm_basis_quality->valid &&
                                    reformulated_warm_basis_quality->dual_feasible;
        const char* reformulated_initial_mode =
            use_dual_first ? "dual" : (opt_.mode == SimplexMode::Primal ? "primal" : "auto");
        auto solve_reformulated = [&](SimplexMode mode) {
            RevisedSimplexOptions solve_opt = opt_;
            solve_opt.mode = mode;
            solve_opt.disable_presolve = true;
            trace_line_("[solve_reformulated] disable_presolve=" +
                        std::to_string(solve_opt.disable_presolve));
            RevisedSimplex reformulated_solver(solve_opt);
            return basis_state_std
                       ? reformulated_solver.solve(A_std, b_std, c_std, l_std, u_std,
                                                   *basis_state_std)
                       : reformulated_solver.solve(A_std, b_std, c_std, l_std, u_std, basis_std);
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
            std_sol = solve_reformulated(SimplexMode::Auto);
            reformulated_retry_used = true;
        }

        Eigen::VectorXd x = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        if (std_sol.x.size() == n_total && std_sol.x.array().isFinite().all()) {
            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    x(j) = map[j].y >= 0 ? map[j].shift + static_cast<double>(map[j].sign) *
                                                              std_sol.x(map[j].y)
                                         : map[j].shift; // fixed variable: x = l = u
                } else {
                    const double yp = std_sol.x(map[j].y_pos);
                    const double yn = std_sol.x(map[j].y_neg);
                    x(j) = yp - yn;
                }
            }
        }

        std::vector<int> basis_out;
        std::vector<char> seen(n, 0);
        for (int idx : std_sol.basis) {
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
        info["bound_reformulation_initial_mode"] = reformulated_initial_mode;
        if (reformulated_warm_basis_quality) {
            info["bound_reformulation_warm_start_valid"] =
                reformulated_warm_basis_quality->valid ? "1" : "0";
            info["bound_reformulation_warm_start_primal_feasible"] =
                reformulated_warm_basis_quality->primal_feasible ? "1" : "0";
            info["bound_reformulation_warm_start_dual_feasible"] =
                reformulated_warm_basis_quality->dual_feasible ? "1" : "0";
        }
        info["original_m"] = std::to_string(m_eq);
        info["original_l"] = serialize_double_vec_(l_in);
        info["original_u"] = serialize_double_vec_(u_in);
        if (sanitized_bounds.relaxed_upper > 0) {
            info["input_upper_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_upper);
        }
        if (sanitized_bounds.relaxed_lower > 0) {
            info["input_lower_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_lower);
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
        const LPBasis warm_basis =
            compute_basis_state_(sol.basis, sol.x, l_in, u_in, opt_.tol, m_eq);
        const std::string warm_basis_serialized =
            serialize_basis_state_from_primal_(sol.basis, sol.x, l_in, u_in, opt_.tol, m_eq);
        sol.basis_state = warm_basis;
        sol.info["warm_start_basis_state"] = warm_basis_serialized;
        if (std_sol.primal_ray_has_cert && std_sol.primal_ray.size() == n_total) {
            Eigen::VectorXd ray = Eigen::VectorXd::Zero(n);
            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    ray(j) = static_cast<double>(map[j].sign) * std_sol.primal_ray(map[j].y);
                } else {
                    const double pos = (map[j].y_pos >= 0) ? std_sol.primal_ray(map[j].y_pos) : 0.0;
                    const double neg = (map[j].y_neg >= 0) ? std_sol.primal_ray(map[j].y_neg) : 0.0;
                    ray(j) = pos - neg;
                }
            }
            sol.primal_ray = clip_small_(ray);
        }
        sol.has_internal_tableau = std_sol.has_internal_tableau;
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
    }

    Eigen::MatrixXd A_model = A_in;
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
            A_model.col(j) = -A_model.col(j);
            c_model(j) = -c_model(j);
        }

        if (anchor(j) != 0.0)
            b_model.noalias() -= A_model.col(j) * anchor(j);
    }

    // ---- (0) Wrap into presolve LP: Ax=b, default bounds, costs=c ----
    presolve::LP lp;
    lp.A = A_model;
    lp.b = b_model;
    lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
    lp.c = c_model;
    lp.l = l_model;
    lp.u = u_model;
    lp.c0 = c_in.dot(anchor);

    const bool warm_start_requested = (basis_opt && !basis_opt->empty()) ||
                                      (basis_state_opt && !basis_state_opt->column_status.empty());

    // ---- (1) Presolve ----
    presolve::Presolver::Options popt;
    // Row reduction (RRQR): heavy but powerful for cold solves. Skip for warm
    // starts to preserve the basis mapping (HiGHS/SCIP hot-restart pattern).
    popt.enable_rowreduce = !warm_start_requested;
    popt.enable_scaling = true;
    popt.enable_objective_probing =
        !warm_start_requested && A_in.rows() <= 300 && A_in.cols() <= 500;
    popt.non_destructive = warm_start_requested;
    popt.allow_structural_changes = false;
    // Reoptimization should keep the LP matrix/basis mapping intact.
    // Even non-destructive fixed-variable presolve can zero columns after a
    // branch bound change and destabilize dual warm starts on basic
    // variables. HiGHS/SCIP-style hot starts are much happier when the
    // matrix is left alone and only bounds change.
    popt.max_passes = warm_start_requested ? 0 : 8;
    popt.probing_max_rounds = warm_start_requested ? 0 : 1;
    popt.probing_max_vars = warm_start_requested ? 0 : 20;
    if (opt_.disable_presolve) {
        trace_line_("[solve] disable_presolve=1");
        popt.enable_rowreduce = false;
        popt.enable_scaling = false;
        popt.enable_objective_probing = false;
        popt.max_passes = 0;
        popt.probing_max_rounds = 0;
        popt.probing_max_vars = 0;
    }
    // Huge bound relaxation: catches numerically degenerate bounds (e.g. [0,1e12]).
    // Safe for cold solves; no effect on warm starts (max_passes=0).
    popt.enable_huge_bound_relaxation = !warm_start_requested;
    if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
        popt.conservative_mode = true;
    }

    presolve::Presolver P(popt);
    const auto pres = P.run(lp);
    trace_presolve_(pres);

    if (pres.proven_infeasible) {
        return finalize_solution_(make_solution_(
            LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
            std::numeric_limits<double>::infinity(), {}, 0, {{"presolve", "infeasible"}}));
    }
    if (pres.proven_unbounded) {
        Eigen::VectorXd xnan =
            Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        return finalize_solution_(make_solution_(LPSolution::Status::Unbounded, xnan,
                                                 -std::numeric_limits<double>::infinity(), {}, 0,
                                                 {{"presolve", "unbounded"}}));
    }

    const Eigen::MatrixXd& Atil = pres.reduced.A;
    const Eigen::VectorXd& btil = pres.reduced.b;
    const Eigen::VectorXd& ctil = pres.reduced.c;
    const Eigen::VectorXd& lred = pres.reduced.l;
    const Eigen::VectorXd& ured = pres.reduced.u;

    // ---- (2) m==0 fast path: optimize over bounds only ----
    if (Atil.rows() == 0) {
        Eigen::VectorXd vred = Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
        bool is_bounded = true;
        for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
            if (ctil(j) > opt_.tol) {
                vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
            } else if (ctil(j) < -opt_.tol) {
                if (std::isfinite(ured(j)))
                    vred(j) = ured(j);
                else {
                    is_bounded = false;
                    break;
                }
            } else {
                vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
            }
        }
        if (!is_bounded) {
            Eigen::VectorXd xnan =
                Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
            return finalize_solution_(make_solution_(
                LPSolution::Status::Unbounded, xnan, -std::numeric_limits<double>::infinity(), {},
                0, {{"presolve", "m=0 neg cost & +inf upper"}}));
        }
        auto [z_full, obj_corr] = P.postsolve(vred);
        z_full =
            repair_nan_primal_(A_model, b_model, l_model, u_model, std::move(z_full), opt_.tol);
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
        const double total_obj = c_in.dot(x_full);
        auto sol = make_solution_(LPSolution::Status::Optimal, std::move(x_full), total_obj, {}, 0,
                                  {{"presolve", "m=0 optimized over bounds"}});
        sol.dual_values = clip_small_vec_(P.postsolve_dual(Eigen::VectorXd::Zero(0)), opt_.tol);
        sol.shadow_prices = sol.dual_values;
        return finalize_solution_(attach_basis_state_(std::move(sol), l_in, u_in, opt_.tol));
    }

    // ---- (3) Solve reduced problem directly with explicit bounds ----
    const bool bypass_postsolve = false;
    Eigen::MatrixXd Ared = Atil;
    Eigen::VectorXd bred = btil;
    Eigen::VectorXd cred = ctil;
    std::vector<int> col_orig_map = pres.orig_col_index;
    std::vector<int> row_orig_map = pres.orig_row_index;
    const std::vector<std::string> internal_column_labels =
        make_internal_column_labels_(col_orig_map);
    const std::vector<std::string> internal_row_labels = make_internal_row_labels_(row_orig_map);
    const int m_eff = static_cast<int>(Ared.rows());
    const int n_eff = static_cast<int>(Ared.cols());

    // Effective bounds (reduced space)
    Eigen::VectorXd l_eff = lred;
    Eigen::VectorXd u_eff = ured;

    const auto postsolve_primal = [&](const Eigen::VectorXd& v) {
        if (bypass_postsolve) {
            return std::make_pair(v, 0.0);
        }
        auto out = P.postsolve(v);
        out.first =
            repair_nan_primal_(A_model, b_model, l_model, u_model, std::move(out.first), opt_.tol);
        return out;
    };

    // ---- (4) Map incoming basis into reduced space (optional) ----
    std::optional<std::vector<int>> red_basis_opt = std::nullopt;
    std::optional<LPBasis> red_basis_state_opt = std::nullopt;
    if (basis_opt && !basis_opt->empty()) {
        std::unordered_map<int, int> orig2red;
        orig2red.reserve(n_eff);
        for (int jr = 0; jr < n_eff; ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0)
                orig2red[jorig] = jr;
        }
        std::vector<int> cand;
        cand.reserve(std::min(m_eff, (int)basis_opt->size()));
        std::vector<char> seen_red(n_eff, 0);
        for (int jorig : *basis_opt) {
            auto it = orig2red.find(jorig);
            if (it != orig2red.end() && !seen_red[it->second]) {
                seen_red[it->second] = 1;
                cand.push_back(it->second);
                if ((int)cand.size() == m_eff)
                    break;
            }
        }
        if (!cand.empty())
            red_basis_opt = std::move(cand);
    }
    if (basis_state_opt && !basis_state_opt->column_status.empty()) {
        red_basis_state_opt =
            map_reduced_basis_state_(*basis_state_opt, col_orig_map, l_eff, u_eff, opt_.tol);
        if (!red_basis_opt || red_basis_opt->empty()) {
            red_basis_opt = basis_columns_from_basis_state_(*red_basis_state_opt, m_eff);
        }
    }

    auto t1_presolve = std::chrono::steady_clock::now();
    current_timing_.presolve_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_presolve - t0_presolve).count();

    // ---- (5) Try Phase II directly on reduced problem (Primal/Dual per
    // mode) ----
    std::vector<int> basis_guess;
    std::optional<std::vector<int>> crash_seed_basis_opt = red_basis_opt;
    const bool seed_basis_from_state = (!basis_opt || basis_opt->empty()) && red_basis_state_opt &&
                                       !red_basis_state_opt->column_status.empty();
    if ((!crash_seed_basis_opt || crash_seed_basis_opt->empty()) && red_basis_state_opt &&
        !red_basis_state_opt->column_status.empty()) {
        auto partial_seed = basis_columns_from_basis_state_(*red_basis_state_opt, -1);
        if (partial_seed && !partial_seed->empty()) {
            crash_seed_basis_opt = std::move(partial_seed);
        }
    }
    const bool allow_direct_warm_start =
        !seed_basis_from_state ||
        (crash_seed_basis_opt && static_cast<int>(crash_seed_basis_opt->size()) == m_eff);
    auto t0_crash = std::chrono::steady_clock::now();
    CrashSelection basis_choice = choose_initial_basis_(
        Ared, bred, cred, opt_, l_eff, u_eff, crash_seed_basis_opt, allow_direct_warm_start);
    auto t1_crash = std::chrono::steady_clock::now();
    current_timing_.crash_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_crash - t0_crash).count();
    basis_guess = basis_choice.basis;
    const bool basis_guess_from_crash = (basis_choice.source == "crash");
    const bool basis_guess_from_warm_start =
        (basis_choice.source == "warm_start" || basis_choice.source == "repaired_warm_start");

    const auto add_info = [&](std::unordered_map<std::string, std::string> info) {
        info["presolve_actions"] = std::to_string(pres.stack.size());
        info["presolve_implied_bound_updates"] = std::to_string(pres.implied_bound_updates);
        info["original_m"] = std::to_string(A_in.rows());
        info["original_l"] = serialize_double_vec_(l_in);
        info["original_u"] = serialize_double_vec_(u_in);
        info["reduced_m"] = std::to_string(m_eff);
        info["reduced_n"] = std::to_string(n_eff);
        info["obj_shift"] = std::to_string(pres.obj_shift);
        if (sanitized_bounds.relaxed_upper > 0) {
            info["input_upper_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_upper);
        }
        if (sanitized_bounds.relaxed_lower > 0) {
            info["input_lower_bounds_relaxed"] = std::to_string(sanitized_bounds.relaxed_lower);
        }
        if (!basis_choice.source.empty() && basis_choice.source != "none") {
            info["basis_start"] = basis_choice.source;
            info["basis_start_style"] = basis_choice.style;
            info["basis_start_attempt"] = std::to_string(basis_choice.attempt);
            info["basis_start_primal_feasible"] = basis_choice.quality.primal_feasible ? "1" : "0";
            info["basis_start_dual_feasible"] = basis_choice.quality.dual_feasible ? "1" : "0";
            info["basis_start_primal_violation"] =
                std::to_string(basis_choice.quality.primal_violation);
            info["basis_start_dual_violation"] =
                std::to_string(basis_choice.quality.dual_violation);
        }
        return info;
    };

    const auto parse_serialized_vec = [](const std::unordered_map<std::string, std::string>& info,
                                         const char* key,
                                         int expected_dim) -> std::optional<Eigen::VectorXd> {
        auto it = info.find(key);
        if (it == info.end())
            return std::nullopt;
        std::vector<double> vals;
        std::stringstream ss(it->second);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty())
                vals.push_back(std::stod(tok));
        }
        if (expected_dim >= 0 && (int)vals.size() != expected_dim) {
            return std::nullopt;
        }
        if (vals.empty() && expected_dim == 0)
            return Eigen::VectorXd::Zero(0);
        if (vals.empty())
            return std::nullopt;
        return Eigen::Map<const Eigen::VectorXd>(vals.data(), static_cast<int>(vals.size()));
    };

    const bool basis_valid = ((int)basis_guess.size() == m_eff) && basis_choice.quality.valid;
    const bool allow_direct_primal =
        basis_valid && (basis_choice.quality.primal_feasible || basis_guess_from_warm_start);
    const bool allow_direct_dual = basis_valid && basis_choice.quality.dual_feasible;
    const bool allow_direct_from_guess = allow_direct_primal || allow_direct_dual;

    if (allow_direct_from_guess) {
        if (basis_guess_from_warm_start) {
            solve_stats_.warm_start_accepted = 1;
        }
        LPSolution::Status st;
        Eigen::VectorXd v2;
        std::vector<int> red_basis2;
        int it2;
        std::unordered_map<std::string, std::string> info2;

        auto run_primal = [&] {
            auto t0_iter = std::chrono::steady_clock::now();
            try {
                auto res = phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
                auto t1_iter = std::chrono::steady_clock::now();
                current_timing_.simplex_iters_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter).count();
                return res;
            } catch (const std::runtime_error& e) {
                auto t1_iter = std::chrono::steady_clock::now();
                current_timing_.simplex_iters_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter).count();
                if (!is_recoverable_basis_runtime_(e.what()))
                    throw;
                return PhaseResult{LPSolution::Status::Singular,
                                   Eigen::VectorXd{},
                                   {},
                                   0,
                                   {{"reason", "basis_factorization_failure"},
                                    {"what", e.what()},
                                    {"where", "dense_direct_primal"}}};
            }
        };
        auto run_dual = [&] {
            const bool use_warm_status =
                red_basis_state_opt &&
                (basis_choice.source == "warm_start" ||
                 basis_choice.source == "repaired_warm_start") &&
                red_basis_state_opt->column_status.size() == static_cast<std::size_t>(n_eff);
            auto t0_iter = std::chrono::steady_clock::now();
            try {
                auto res = dual_phase_(Ared, bred, cred, basis_guess, l_eff, u_eff,
                                       use_warm_status ? std::optional<std::vector<LPBasisStatus>>(
                                                             red_basis_state_opt->column_status)
                                                       : std::nullopt);
                auto t1_iter = std::chrono::steady_clock::now();
                current_timing_.simplex_iters_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter).count();
                return res;
            } catch (const std::runtime_error& e) {
                auto t1_iter = std::chrono::steady_clock::now();
                current_timing_.simplex_iters_ns +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1_iter - t0_iter).count();
                if (!is_recoverable_basis_runtime_(e.what()))
                    throw;
                return PhaseResult{LPSolution::Status::Singular,
                                   Eigen::VectorXd{},
                                   {},
                                   0,
                                   {{"reason", "basis_factorization_failure"},
                                    {"what", e.what()},
                                    {"where", "dense_direct_dual"}}};
            }
        };

        if (opt_.mode == SimplexMode::Dual) {
            if (allow_direct_dual) {
                std::tie(st, v2, red_basis2, it2, info2) = run_dual();
            } else {
                st = LPSolution::Status::NeedPhase1;
                info2["reason"] = "no_dual_feasible_start_basis";
            }
        } else if (opt_.mode == SimplexMode::Primal) {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
            } else {
                st = LPSolution::Status::NeedPhase1;
                info2["reason"] = "no_primal_feasible_start_basis";
            }
        } else {
            if (allow_direct_primal) {
                std::tie(st, v2, red_basis2, it2, info2) = run_primal();
            } else if (allow_direct_dual) {
                std::tie(st, v2, red_basis2, it2, info2) = run_dual();
            }
            if (allow_direct_dual && st == LPSolution::Status::NeedPhase1 &&
                info2.count("reason") && info2.at("reason") == std::string("negative_basic_vars")) {
                std::tie(st, v2, red_basis2, it2, info2) = run_dual();
            }
        }

        const bool direct_dual_native_bounds =
            opt_.mode == SimplexMode::Dual && allow_direct_dual && has_upper_bounds;
        if (st == LPSolution::Status::Optimal || st == LPSolution::Status::Unbounded ||
            (st == LPSolution::Status::IterLimit && !direct_dual_native_bounds) ||
            st == LPSolution::Status::ObjectiveBound ||
            (st == LPSolution::Status::Infeasible && !basis_guess_from_warm_start &&
             !direct_dual_native_bounds)) {
            auto [z_full, obj_corr] = postsolve_primal(v2);
            Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
            const double total_obj = c_in.dot(x_full);
            const bool has_primal_ray =
                info2.count("primal_ray_has_cert") && info2.at("primal_ray_has_cert") == "1";
            const auto primal_ray_internal =
                has_primal_ray ? parse_serialized_vec(info2, "primal_ray", n_eff) : std::nullopt;

            std::vector<int> basis_full;
            basis_full.reserve(red_basis2.size());
            for (int jr : red_basis2) {
                if (jr >= 0 && jr < (int)col_orig_map.size()) {
                    const int jorig = col_orig_map[jr];
                    if (jorig >= 0)
                        basis_full.push_back(jorig);
                }
            }
            auto info = add_info(std::move(info2));
            if (st == LPSolution::Status::Optimal &&
                !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
                if (direct_dual_native_bounds) {
                    info["reason"] = "invalid_returned_primal";
                    info2["reason"] = "invalid_returned_primal";
                    info2["direct_phase2_status"] = to_string(st);
                    info2["direct_phase2_recovery"] = "phase1";
                } else {
                    info["reason"] = "invalid_returned_primal";
                    return finalize_solution_(attach_internal_basis_(
                        make_solution_(LPSolution::Status::Singular, std::move(x_full), total_obj,
                                       basis_full, it2, std::move(info)),
                        red_basis2, internal_column_labels));
                }
            }
            if (!(st == LPSolution::Status::Optimal && direct_dual_native_bounds &&
                  !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol))) {
                return finalize_solution_(attach_basis_state_(
                    attach_mapped_primal_ray_(
                        attach_postsolved_farkas_(
                            attach_postsolved_row_duals_(
                                attach_internal_tableau_(
                                    make_solution_(st, std::move(x_full), total_obj, basis_full,
                                                   it2, std::move(info), std::nullopt, std::nullopt,
                                                   primal_ray_internal, has_primal_ray),
                                    Ared, bred, cred, red_basis2, internal_column_labels,
                                    internal_row_labels, opt_.tol, opt_.compute_tableau,
                                    opt_.compute_reduced_costs),
                                P, opt_.tol),
                            P, opt_.tol),
                        col_orig_map, sign, A_model.cols(), opt_.tol),
                    l_in, u_in, opt_.tol));
            }
        }
        if (st == LPSolution::Status::Singular) {
            info2["direct_phase2_status"] = to_string(st);
            info2["direct_phase2_recovery"] = "phase1";
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
    // ---- (6) Phase I on reduced problem ----
    auto [A1, b1, c1, basis1, n_orig_eff, m_rows] = make_phase1_(Ared, bred);
    // Phase-1 bounds: original columns keep their finite upper bounds so
    // the feasible point found also respects them; lower bounds stay 0
    // (the artificial identity basis starts from x = 0).
    Eigen::VectorXd l_phase1 = Eigen::VectorXd::Zero(A1.cols());
    Eigen::VectorXd u_phase1 = Eigen::VectorXd::Constant(A1.cols(), presolve::inf());
    if (u_eff.size() == static_cast<Eigen::Index>(n_orig_eff)) {
        u_phase1.head(n_orig_eff) = u_eff;
    }
    auto t0_p1 = std::chrono::steady_clock::now();
    auto [status1, v1, basis1_out, it1, info1] = phase_(A1, b1, c1, basis1, l_phase1, u_phase1);
    auto t1_p1 = std::chrono::steady_clock::now();
    current_timing_.simplex_iters_ns +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_p1 - t0_p1).count();

    if (status1 == LPSolution::Status::NeedPhase1 && info1.count("reason") &&
        info1.at("reason") == std::string("negative_basic_vars")) {
        auto t0_d1 = std::chrono::steady_clock::now();
        std::tie(status1, v1, basis1_out, it1, info1) =
            dual_phase_(A1, b1, c1, basis1_out.empty() ? basis1 : basis1_out, l_phase1, u_phase1);
        auto t1_d1 = std::chrono::steady_clock::now();
        current_timing_.simplex_iters_ns +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_d1 - t0_d1).count();
    }

    // If phase I fails or artificial cost > tol ⇒ infeasible
    if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
        auto info = add_info({{"phase1_status", to_string(status1)}});
        const auto s = degen_.get_stats();
        auto more = dm_stats_to_map(s);
        info.insert(more.begin(), more.end());
        return finalize_solution_(
            make_solution_(LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                           std::numeric_limits<double>::infinity(), {}, it1, std::move(info)));
    }

    // HiGHS-style Phase-I basis cleanup: pivot zero-valued artificials out
    // degenerately, preserving the feasible Phase-I point.  Dropping them and
    // completing a basis by rank alone loses that invariant.
    {
        std::vector<char> basic(A1.cols(), 0);
        for (int j : basis1_out)
            basic[j] = 1;
        for (int r = 0; r < m_rows; ++r) {
            if (basis1_out[r] < static_cast<int>(n_orig_eff))
                continue;
            Eigen::MatrixXd B(A1.rows(), m_rows);
            for (int k = 0; k < m_rows; ++k)
                B.col(k) = A1.col(basis1_out[k]);
            Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
            bool replaced = false;
            if (lu.rank() == m_rows) {
                for (int j = 0; j < static_cast<int>(n_orig_eff); ++j) {
                    if (basic[j])
                        continue;
                    const Eigen::VectorXd d = lu.solve(A1.col(j));
                    if (!d.allFinite() || std::abs(d(r)) <= opt_.alpha_tol)
                        continue;
                    basic[basis1_out[r]] = 0;
                    basis1_out[r] = j;
                    basic[j] = 1;
                    replaced = true;
                    break;
                }
            }
            if (!replaced) {
                auto info = add_info({{"reason", "phase1_artificial_cleanup_failed"},
                                      {"row", std::to_string(r)}});
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
                    std::numeric_limits<double>::quiet_NaN(), {}, it1, std::move(info)));
            }
        }
    }

    // Phase 1 may finish with nonbasics at their upper bounds; seed
    // phase 2 with those statuses so its starting point matches the
    // feasible point found by phase 1 (native bounds).
    std::optional<std::vector<LPBasisStatus>> phase2_seed_status;
    if (v1.size() >= static_cast<Eigen::Index>(n_orig_eff) &&
        static_cast<int>(n_orig_eff) == u_eff.size()) {
        const int nred = static_cast<int>(n_orig_eff);
        std::vector<LPBasisStatus> stv(nred, LPBasisStatus::AtLower);
        for (int j = 0; j < nred; ++j) {
            const bool near_u =
                std::isfinite(u_eff(j)) && std::abs(v1(j) - u_eff(j)) <= 10.0 * opt_.tol;
            const bool near_l =
                std::isfinite(l_eff(j)) && std::abs(v1(j) - l_eff(j)) <= 10.0 * opt_.tol;
            if (near_u && !near_l)
                stv[j] = LPBasisStatus::AtUpper;
        }
        phase2_seed_status = std::move(stv);
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
            if (xj != 0.0)
                rhs.noalias() -= Ared.col(j) * xj;
        }
        return rhs;
    };
    auto phase2_basis_primal_feasible = [&](const std::vector<int>& basis) {
        return basis_is_primal_feasible_(Ared, phase2_effective_rhs(basis), basis, l_eff, u_eff,
                                         opt_.tol);
    };

    // Warm-start Phase II basis by removing artificials
    std::vector<int> red_basis2;
    red_basis2.reserve(m_rows);
    for (int j : basis1_out)
        if (j < (int)n_orig_eff)
            red_basis2.push_back(j);

    // Basis completion if needed
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
            const Eigen::MatrixXd Btest =
                Ared(Eigen::all, Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
            if (!(lu.rank() == (int)cand.size() && lu.isInvertible())) {
                continue;
            }
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
    if ((int)red_basis2.size() == m_rows && !phase2_basis_primal_feasible(red_basis2)) {
        for (int j = 0; j < (int)n_orig_eff; ++j) {
            if (std::find(red_basis2.begin(), red_basis2.end(), j) != red_basis2.end()) {
                continue;
            }
            bool improved = false;
            for (int r = 0; r < m_rows; ++r) {
                std::vector<int> cand = red_basis2;
                cand[r] = j;
                const Eigen::MatrixXd Btest =
                    Ared(Eigen::all, Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                if (!(lu.rank() == m_rows && lu.isInvertible()))
                    continue;
                if (!phase2_basis_primal_feasible(cand)) {
                    continue;
                }
                red_basis2 = std::move(cand);
                improved = true;
                break;
            }
            if (improved)
                break;
        }
    }

    // Final Phase II on reduced problem (respect mode)
    LPSolution::Status status2;
    Eigen::VectorXd v2;
    std::vector<int> red_basis_out;
    int it2 = 0;
    std::unordered_map<std::string, std::string> info2;

    if ((int)red_basis2.size() == m_rows) {
        if (opt_.mode == SimplexMode::Dual) {
            const auto phase2_basis_quality =
                evaluate_basis_quality_(Ared, bred, cred, red_basis2, l_eff, u_eff, opt_.tol);
            if (phase2_basis_quality.valid && phase2_basis_quality.dual_feasible) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff, phase2_seed_status);
                const bool invalid_dual_optimum =
                    status2 == LPSolution::Status::Optimal &&
                    (v2.size() != n_eff ||
                     !primal_feasible_(Ared, bred, v2, l_eff, u_eff, opt_.tol));
                if (status2 == LPSolution::Status::Singular ||
                    status2 == LPSolution::Status::IterLimit ||
                    status2 == LPSolution::Status::NeedPhase1 ||
                    status2 == LPSolution::Status::Infeasible || invalid_dual_optimum) {
                    std::vector<int> recovery_basis =
                        (status2 == LPSolution::Status::Infeasible || invalid_dual_optimum) &&
                                static_cast<int>(red_basis_out.size()) == m_rows
                            ? red_basis_out
                            : red_basis2;
                    const LPSolution::Status dual_status = status2;
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        phase_(Ared, bred, cred, recovery_basis, l_eff, u_eff, phase2_seed_status);
                    info2["phase2_mode"] = "primal";
                    info2["phase2_dual_recovery"] = "1";
                    info2["phase2_dual_recovery_status"] = to_string(dual_status);
                    if (invalid_dual_optimum)
                        info2["phase2_dual_recovery_reason"] = "invalid_returned_primal";
                    else if (dual_status == LPSolution::Status::Infeasible)
                        info2["phase2_dual_recovery_reason"] = "cross_check_infeasibility";
                }
                if (status2 == LPSolution::Status::Infeasible) {
                    auto it = info2.find("farkas_has_cert");
                    if (it != info2.end() && it->second == "1") {
                        auto yF = parse_serialized_vec(info2, "farkas_y", m_eff);
                        return finalize_solution_(attach_postsolved_farkas_(
                            attach_internal_basis_(
                                make_solution_(LPSolution::Status::Infeasible,
                                               Eigen::VectorXd::Zero(n),
                                               std::numeric_limits<double>::infinity(), {}, it2,
                                               add_info(std::move(info2)), yF, true),
                                red_basis_out, internal_column_labels),
                            P, opt_.tol));
                    }
                }
            } else {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff, phase2_seed_status);
                info2["phase2_mode"] = "primal";
                info2["phase2_dual_requested_but_basis_not_dual_feasible"] = "1";
            }

        } else if (opt_.mode == SimplexMode::Primal) {
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, red_basis2, l_eff, u_eff, phase2_seed_status);
        } else {
            // Auto: primal first; if negative basics → dual
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, red_basis2, l_eff, u_eff, phase2_seed_status);
            if (status2 == LPSolution::Status::NeedPhase1 && info2.count("reason") &&
                info2.at("reason") == std::string("negative_basic_vars")) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff, phase2_seed_status);
            }
        }
    } else {
        // Fall back to find a basis internally
        std::tie(status2, v2, red_basis_out, it2, info2) =
            phase_(Ared, bred, cred, std::nullopt, l_eff, u_eff);
        if (status2 == LPSolution::Status::NeedPhase1) {
            status2 = LPSolution::Status::Singular;
            info2["note"] = "reduced matrix cannot form a proper basis";
        }
    }

    if (status2 == LPSolution::Status::NeedPhase1) {
        RevisedSimplexOptions cold_primal_opt = opt_;
        cold_primal_opt.mode = SimplexMode::Primal;
        cold_primal_opt.disable_presolve = true;
        RevisedSimplex cold_primal(cold_primal_opt);
        LPSolution cold = cold_primal.solve(Ared, bred, cred, l_eff, u_eff);
        if (cold.status == LPSolution::Status::Optimal && cold.x.size() == n_eff &&
            primal_feasible_(Ared, bred, cold.x, l_eff, u_eff, opt_.tol)) {
            status2 = cold.status;
            v2 = cold.x;
            red_basis_out = cold.basis;
            info2 = cold.info;
            info2["phase2_mode"] = "primal";
            info2["phase2_cold_primal_recovery"] = "1";
        }
    }

    const int total_iters = it1 + it2;
    auto merged_info = add_info(std::move(info2));
    merged_info.insert({"phase1_iters", std::to_string(it1)});
    const bool has_primal_ray =
        merged_info.count("primal_ray_has_cert") && merged_info.at("primal_ray_has_cert") == "1";
    const auto primal_ray_internal =
        has_primal_ray ? parse_serialized_vec(merged_info, "primal_ray", n_eff) : std::nullopt;
    if (v2.size() != n_eff) {
        merged_info["where"] = "dense_phase2_finalize";
        merged_info["reason"] = "invalid_primal_dimension";
        merged_info["expected_primal_dim"] = std::to_string(n_eff);
        merged_info["actual_primal_dim"] = std::to_string(v2.size());
        return finalize_solution_(make_solution_(
            status2, Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN()),
            std::numeric_limits<double>::quiet_NaN(), {}, total_iters, std::move(merged_info)));
    }

    auto [z_full, obj_correction] = postsolve_primal(v2);
    Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
    const double total_obj = c_in.dot(x_full);

    std::vector<int> basis_full;
    basis_full.reserve(red_basis_out.size());
    for (int jr : red_basis_out) {
        if (jr >= 0 && jr < (int)col_orig_map.size()) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0)
                basis_full.push_back(jorig);
        }
    }

    if (status2 == LPSolution::Status::Optimal &&
        !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
        merged_info["reason"] = "invalid_returned_primal";
        return finalize_solution_(attach_basis_state_(
            attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(LPSolution::Status::Singular, x_full, total_obj,
                                           basis_full, total_iters, std::move(merged_info),
                                           std::nullopt, std::nullopt, primal_ray_internal,
                                           has_primal_ray),
                            Ared, bred, cred, red_basis_out, internal_column_labels,
                            internal_row_labels, opt_.tol, opt_.compute_tableau,
                            opt_.compute_reduced_costs),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol),
            l_in, u_in, opt_.tol));
    }

    if (status2 == LPSolution::Status::Optimal) {
        return finalize_solution_(attach_basis_state_(
            attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(LPSolution::Status::Optimal, x_full, total_obj,
                                           basis_full, total_iters, std::move(merged_info),
                                           std::nullopt, std::nullopt, primal_ray_internal,
                                           has_primal_ray),
                            Ared, bred, cred, red_basis_out, internal_column_labels,
                            internal_row_labels, opt_.tol, opt_.compute_tableau,
                            opt_.compute_reduced_costs),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol),
            l_in, u_in, opt_.tol));
    }
    if (status2 == LPSolution::Status::Unbounded) {
        return finalize_solution_(attach_mapped_primal_ray_(
            attach_postsolved_farkas_(
                attach_postsolved_row_duals_(
                    attach_internal_tableau_(
                        make_solution_(LPSolution::Status::Unbounded, x_full,
                                       -std::numeric_limits<double>::infinity(), basis_full,
                                       total_iters, std::move(merged_info), std::nullopt,
                                       std::nullopt, primal_ray_internal, has_primal_ray),
                        Ared, bred, cred, red_basis_out, internal_column_labels,
                        internal_row_labels, opt_.tol, opt_.compute_tableau,
                        opt_.compute_reduced_costs),
                    P, opt_.tol),
                P, opt_.tol),
            col_orig_map, sign, A_model.cols(), opt_.tol));
    }

    const double obj_fallback =
        x_full.array().isFinite().all() ? total_obj : std::numeric_limits<double>::quiet_NaN();
    return finalize_solution_(attach_basis_state_(
        attach_mapped_primal_ray_(
            attach_postsolved_farkas_(
                attach_postsolved_row_duals_(
                    attach_internal_tableau_(
                        make_solution_(status2, x_full, obj_fallback, basis_full, total_iters,
                                       std::move(merged_info), std::nullopt, std::nullopt,
                                       primal_ray_internal, has_primal_ray),
                        Ared, bred, cred, red_basis_out, internal_column_labels,
                        internal_row_labels, opt_.tol, opt_.compute_tableau,
                        opt_.compute_reduced_costs),
                    P, opt_.tol),
                P, opt_.tol),
            col_orig_map, sign, A_model.cols(), opt_.tol),
        l_in, u_in, opt_.tol));
}
