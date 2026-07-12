#pragma once

#include "simplex/core/hvector.h"
#include "simplex/engine/common/utils.h"
#include "simplex/engine/primal/components.h"
#include "simplex/nla/simplex_nla.h"

// Bounded-variable primal revised simplex engine.
//
// Handles l <= x <= u natively: nonbasic variables rest at their lower or
// upper bound (tracked in `at_upper`), the ratio test blocks basic variables
// against both bounds, and an entering variable whose own opposite bound is
// the tightest block performs a bound flip (no basis change). Free nonbasic
// variables (both bounds infinite) rest at 0 and may move in either
// direction.
class RevisedSimplexPrimalEngine : public simplex::engine::PrimalPivotSelection,
                                   public simplex::engine::MatrixUtilities {
  public:
    using RatioResult = simplex::engine::PrimalPivotSelection::RatioResult;
    using ReducedCostView = simplex::engine::PrimalPivotSelection::ReducedCostView;
    struct IterationWork {
        Eigen::VectorXd base_value;
        Eigen::VectorXd base_cost;
        Eigen::VectorXd work_dual;
        Eigen::VectorXd reduced_cost;
        Eigen::VectorXd entering_measure;
        std::optional<int> entering_rel;
        int entering_col = -1;
        double entering_direction = 0.0;
    };

    template <class MatrixType>
    static bool
    exact_optimality_check_(RevisedSimplex& self, const MatrixType& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c, const std::vector<int>& basis,
                            const std::vector<int>& nonbasis, const std::vector<char>& at_upper,
                            const Eigen::VectorXd& x, const Eigen::VectorXd& l,
                            const Eigen::VectorXd& u, FTBasis& factor, std::string& reason) {
        if (!RevisedSimplex::primal_feasible_(A, b, x, l, u, self.opt_.tol)) {
            reason = "optimal_primal_check_failed";
            return false;
        }
        Eigen::VectorXd cB(basis.size());
        for (int i = 0; i < static_cast<int>(basis.size()); ++i)
            cB(i) = c(basis[i]);
        HVector y_hvec;
        try {
            y_hvec = factor.solve_BT(cB, FTBasis::TranKind::RowEp);
        } catch (...) {
            reason = "optimal_dual_check_solve_failed";
            return false;
        }
        const ReducedCostView exact_rc =
            compute_reduced_costs_(A, c, y_hvec.value, nonbasis, at_upper, l, u, self.opt_.tol);
        if (choose_dantzig_entering_(exact_rc.entering_measure, self.opt_.tol)) {
            reason = "optimal_dual_check_failed";
            return false;
        }
        return true;
    }

    template <class MatrixType>
    static RevisedSimplex::PhaseResult
    run(RevisedSimplex& self, const MatrixType& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        std::optional<std::vector<LPBasisStatus>> warm_status = std::nullopt) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;
        Eigen::VectorXd c_work = c;
        bool costs_perturbed = false;
        Eigen::VectorXd l_work = l;
        Eigen::VectorXd u_work = u;
        bool bounds_perturbed = false;

        std::vector<int> basis;
        if (basis_opt) {
            basis = *basis_opt;
            if ((int)basis.size() != m)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "basis size != m"}}};
        } else {
            auto maybe = self.find_initial_basis_(A, b, c, self.opt_, l, u);
            if (!maybe)
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        {},
                        0,
                        {{"reason", "no_crash_basis"}}};
            basis = *maybe;
        }

        std::vector<int> N;
        N.reserve(n - m);
        {
            std::vector<char> inB(n, 0);
            for (int j : basis) {
                if (j < 0 || j >= n)
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            0,
                            {{"where", "initial basis index out of range"}}};
                inB[j] = 1;
            }
            for (int j = 0; j < n; ++j)
                if (!inB[j])
                    N.push_back(j);
        }

        if (N.empty()) {
            Eigen::MatrixXd B(m, m);
            for (int i = 0; i < m; ++i)
                B.col(i) = A.col(basis[i]);
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(B);
            qr.setThreshold(self.opt_.svd_tol);
            if (qr.rank() < m) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "full-basis rank check failed"}}};
            }
            const Eigen::VectorXd xB = qr.solve(b);
            Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l, u);
            if (self.primal_feasible_(A, b, x, l, u, self.opt_.tol)) {
                return {LPSolution::Status::Optimal, self.clip_small_(x), basis, iters,
                        dm_stats_to_map(self.degen_.get_stats())};
            }
            return {LPSolution::Status::NeedPhase1,
                    Eigen::VectorXd::Zero(n),
                    basis,
                    iters,
                    {{"reason", "full_basis_primal_infeasible"}}};
        }

        // ── Nonbasic bound status ────────────────────────────────────────────
        // at_upper[j] == 1 means nonbasic j rests at its (finite) upper bound.
        std::vector<char> at_upper(n, 0);
        if (warm_status && (int)warm_status->size() == n) {
            for (int j : N) {
                if ((*warm_status)[j] == LPBasisStatus::AtUpper && j < u_work.size() &&
                    std::isfinite(u_work(j)))
                    at_upper[j] = 1;
            }
        }
        for (int j : N) {
            const bool has_l = (j < l_work.size()) && std::isfinite(l_work(j));
            const bool has_u = (j < u_work.size()) && std::isfinite(u_work(j));
            if (!has_l && has_u)
                at_upper[j] = 1; // only an upper bound to rest at
        }

        // Effective RHS: b minus the contribution of nonbasics at nonzero values.
        Eigen::VectorXd rhs_eff = b;
        for (int j : N) {
            const double xj = nonbasic_value_(j, l_work, u_work, at_upper[j]);
            axpy_col_(A, j, -xj, rhs_eff);
        }

        std::shared_ptr<simplex::nla::SimplexNLA> nla;
        // Verify warm factorization's B matrix matches current B matrix
        if (self.solve_input_warm_state_ && self.solve_input_warm_state_->basis_matrix_signature) {
            std::uint64_t sig = 0xcbf29ce484222325ULL;
            sig ^= (static_cast<std::uint64_t>(m) + 0xc6b1a7c3d5e9f0a2ULL);
            sig = (sig << 31) | (sig >> 33);
            Eigen::MatrixXd B_dense(m, m);
            B_dense.setZero();
            if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                for (int j = 0; j < m; ++j) {
                    const int col = basis[j];
                    for (typename MatrixType::InnerIterator it(A, col); it; ++it)
                        B_dense(it.row(), j) = it.value();
                }
            } else {
                for (int j = 0; j < m; ++j)
                    B_dense.col(j) = A.col(basis[j]);
            }
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < m; ++j) {
                    std::uint64_t hv = std::bit_cast<std::uint64_t>(B_dense(i, j));
                    sig ^= hv;
                    sig = (sig << 31) | (sig >> 33);
                }
            if (sig != self.solve_input_warm_state_->basis_matrix_signature) {
                self.solve_input_warm_state_ = nullptr;
            }
        }
        auto make_nla_config = [&]() {
            simplex::nla::NLAConfig nla_cfg;
            nla_cfg.framework_switch_threshold_ = self.opt_.framework_switch_threshold;
            nla_cfg.framework_switch_consecutive_ = self.opt_.framework_switch_consecutive;
            nla_cfg.allow_framework_switch_ = self.opt_.allow_framework_switch;
            if (self.opt_.price_strategy == "row_switch_col_switch")
                nla_cfg.price_strategy_ =
                    simplex::nla::NLAConfig::PriceStrategy::RowSwitchColSwitch;
            else if (self.opt_.price_strategy == "row_switch")
                nla_cfg.price_strategy_ = simplex::nla::NLAConfig::PriceStrategy::RowSwitch;
            else
                nla_cfg.price_strategy_ = simplex::nla::NLAConfig::PriceStrategy::ColOnly;
            return nla_cfg;
        };
        auto build_nla = [&]() -> std::shared_ptr<simplex::nla::SimplexNLA> {
            auto fresh = std::make_shared<simplex::nla::SimplexNLA>();
            fresh->setup(A.rows(), 0.1, make_nla_config());
            if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                if (m <= 16) {
                    Eigen::MatrixXd B_dense(m, m);
                    for (int i = 0; i < m; ++i)
                        B_dense.col(i) = A.col(basis[i]);
                    std::vector<int> dense_basis(m);
                    std::iota(dense_basis.begin(), dense_basis.end(), 0);
                    fresh->setup_factor(B_dense, dense_basis, self.make_basis_options_());
                    return fresh;
                }
            }
            fresh->setup_factor(A, basis, self.make_basis_options_());
            return fresh;
        };
        if (const auto warm_state = self.try_reuse_factorization_(basis)) {
            nla = warm_state->nla;
        } else {
            try {
                nla = build_nla();
            } catch (const std::exception& e) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "initial basis factorization failed"}, {"what", e.what()}}};
            }
        }
        auto read_basis = [&]() -> FTBasis& { return nla->factor(); };
        auto refactor_basis = [&]() -> FTBasis& {
            if (!nla.unique()) {
                nla = build_nla();
            } else {
                nla->invert();
            }
            return nla->factor();
        };
        auto update_basis = [&](int row, int entering_col, const auto& entering_vector) {
            if (!nla.unique())
                nla = build_nla();
            nla->update_basis(row, entering_col, entering_vector);
        };
        self.degen_.start_basis_history(basis);
        self.trace_line_("[primal] start basis=" + self.format_basis_(basis));

        if (self.opt_.pricing_rule == "adaptive") {
            AdaptivePricer::PricingOptions popts;
            popts.steepest_pool_max = 0;
            popts.steepest_reset_freq = self.opt_.adaptive_reset_freq;
            popts.devex_reset_freq = self.opt_.devex_reset;
            popts.primal_edge_weight_strategy = self.opt_.primal_edge_weight_strategy;
            popts.primal_weight_log_error_threshold =
                self.opt_.primal_steepest_edge_weight_log_error_threshold;
            self.adaptive_pricer_ = AdaptivePricer(n, popts);
            self.measure_pricing_build_(
                false, [&]() { self.adaptive_pricer_.build_primal_pools(read_basis(), A, N); });
            self.bridge_ = std::make_unique<PrimalPricingBridge<AdaptivePricer>>(
                self.degen_, self.adaptive_pricer_);
        }

        auto serialize_vec = [](const Eigen::VectorXd& v) {
            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss << std::setprecision(17);
            for (int i = 0; i < v.size(); ++i) {
                if (i)
                    oss << ",";
                oss << v(i);
            }
            return oss.str();
        };

        auto unbounded_ray = [&](int entering_abs, double sigma, const Eigen::VectorXd& dB) {
            Eigen::VectorXd ray = Eigen::VectorXd::Zero(n);
            if (entering_abs >= 0 && entering_abs < n)
                ray(entering_abs) = sigma;
            for (int i = 0; i < m && i < dB.size(); ++i) {
                const int j = basis[i];
                if (j >= 0 && j < n)
                    ray(j) = -sigma * dB(i);
            }
            return self.clip_small_(ray);
        };

        auto sigma_view = [&]() {
            std::vector<int> sv(n, 1);
            for (int j = 0; j < n; ++j)
                if (at_upper[j])
                    sv[j] = -1;
            return sv;
        };

        int rebuild_attempts = 0;

        // ── Incremental xB (primal basic solution) cache ─────────────────────
        // Avoids re-solving B·xB = rhs_eff from scratch each iteration. After a
        // pivot with FTRAN column dB = B^{-1}a_e, direction sigma and step t:
        //   xB_new[i] = xB_old[i] − sigma·t·dB[i]    (i ≠ r)
        //   xB_new[r] = value of the entering variable
        // Falls back to a full solve after explicit refactors or every
        // refactor_every pivots.
        Eigen::VectorXd xB_cache;
        bool xB_cache_valid = false;
        int xB_cache_age = 0;
        const int xB_max_age = std::max(1, self.opt_.refactor_every);
        auto refresh_xB_cache = [&]() {
            HVector xB_hvec = read_basis().solve_B(rhs_eff, FTBasis::TranKind::ColAq);
            xB_cache = xB_hvec.value;
            nla->update_ema_reach(xB_hvec.count, m);
            xB_cache_valid = true;
            xB_cache_age = 0;
        };
        refresh_xB_cache(); // prime before the loop

        auto rebuild_pricing = [&]() {
            if (self.opt_.pricing_rule == "adaptive") {
                self.measure_pricing_build_(
                    false, [&]() { self.adaptive_pricer_.build_primal_pools(read_basis(), A, N); });
                self.adaptive_pricer_.clear_rebuild_flag();
            }
        };
        auto cleanup_primal_perturbations = [&](const char* reason) {
            if (!costs_perturbed && !bounds_perturbed)
                return false;
            if (costs_perturbed) {
                c_work = c;
                costs_perturbed = false;
            }
            if (bounds_perturbed) {
                l_work = l;
                u_work = u;
                bounds_perturbed = false;
            }
            xB_cache_valid = false;
            self.trace_line_("[primal] iter=" + std::to_string(iters) +
                             " cleanup true costs/bounds after " + std::string(reason));
            rebuild_pricing();
            return true;
        };
        auto finish_optimal = [&](const Eigen::VectorXd& xB_current) {
            // HiGHS-style reinversion before declaring optimality. Updated
            // factors and the incremental basic-value cache may agree with
            // each other while both have drifted from the explicit basis.
            Eigen::VectorXd xB_certified = xB_current;
            try {
                refactor_basis();
                xB_certified = read_basis().solve_B(rhs_eff, FTBasis::TranKind::ColAq).value;
                xB_cache = xB_certified;
                xB_cache_valid = true;
                xB_cache_age = 0;
            } catch (...) {
                return RevisedSimplex::PhaseResult{
                    LPSolution::Status::Singular, Eigen::VectorXd::Zero(n), basis, iters,
                    {{"reason", "optimal_reinversion_failed"}}};
            }
            const auto sv = sigma_view();
            Eigen::VectorXd x =
                self.assemble_primal_(n, basis, xB_certified, l_work, u_work, &sv);
            std::string exact_reason;
            if (!exact_optimality_check_(self, A, b, c, basis, N, at_upper, x, l, u, read_basis(),
                                         exact_reason)) {
                return RevisedSimplex::PhaseResult{
                    LPSolution::Status::NeedPhase1,
                    Eigen::VectorXd::Zero(n),
                    basis,
                    iters,
                    {{"reason", exact_reason.empty() ? "optimality_check_failed" : exact_reason}}};
            }
            self.trace_line_("[primal] optimal iter=" + std::to_string(iters) +
                             " basis=" + self.format_basis_(basis));
            self.remember_warm_state_(basis, nla);
            return RevisedSimplex::PhaseResult{LPSolution::Status::Optimal, self.clip_small_(x),
                                               basis, iters,
                                               dm_stats_to_map(self.degen_.get_stats())};
        };

        while (iters < self.opt_.max_iters) {
            ++iters;

            IterationWork work;
            try {
                if (!xB_cache_valid || xB_cache_age >= xB_max_age)
                    refresh_xB_cache();
                work.base_value = xB_cache;
            } catch (...) {
                if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                     " refactor after solve_B failure");
                    refactor_basis();
                    xB_cache_valid = false;
                    rebuild_pricing();
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "solve(B,b) repair failed"}}};
            }

            if (!clamp_basic_solution_to_bounds_(work.base_value, basis, l_work, u_work,
                                                 self.opt_.tol)) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " infeasible basic vars, handing off to phase I");
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "negative_basic_vars"}}};
            }

            work.base_cost.resize(m);
            for (int i = 0; i < m; ++i)
                work.base_cost(i) = c_work(basis[i]);

            HVector y_hvec;
            try {
                y_hvec = read_basis().solve_BT(work.base_cost, FTBasis::TranKind::RowEp);
                nla->update_ema_reach(y_hvec.count, m);
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after solve_BT failure");
                refactor_basis();
                xB_cache_valid = false;
                y_hvec = read_basis().solve_BT(work.base_cost, FTBasis::TranKind::RowEp);
                nla->update_ema_reach(y_hvec.count, m);
                rebuild_pricing();
            }
            work.work_dual = y_hvec.value;
            const ReducedCostView rc_view = compute_reduced_costs_(
                A, c_work, work.work_dual, N, at_upper, l_work, u_work, self.opt_.tol);
            work.reduced_cost = rc_view.raw;
            work.entering_measure = rc_view.entering_measure;

            // CHUZC: choose the entering nonbasic column.
            if (self.opt_.bland) {
                work.entering_rel = choose_bland_entering_(work.entering_measure, self.opt_.tol);
                if (!work.entering_rel) {
                    if (cleanup_primal_perturbations("no entering column"))
                        continue;
                    return finish_optimal(work.base_value);
                }
            } else {
                if (self.opt_.pricing_rule == "adaptive") {
                    double current_obj = 0.0;
                    {
                        std::vector<char> inB(n, 0);
                        for (int i = 0; i < (int)basis.size(); ++i) {
                            const int j = basis[i];
                            if (j >= 0 && j < n) {
                                inB[j] = 1;
                                current_obj += c_work(j) * work.base_value(i);
                            }
                        }
                        for (int j = 0; j < n; ++j) {
                            if (inB[j])
                                continue;
                            current_obj +=
                                c_work(j) * nonbasic_value_(j, l_work, u_work, at_upper[j]);
                        }
                    }
                    work.entering_rel = self.bridge_->choose_primal_entering(
                        work.entering_measure, N, self.opt_.tol, iters, current_obj, read_basis(),
                        A, self.opt_.partial_pricing);
                } else {
                    work.entering_rel =
                        choose_dantzig_entering_(work.entering_measure, self.opt_.tol);
                }

                if (!work.entering_rel) {
                    if (cleanup_primal_perturbations("no entering column"))
                        continue;
                    return finish_optimal(work.base_value);
                }
            }

            const int idxN = *work.entering_rel;
            work.entering_col = N[idxN];
            const double rc_e = work.reduced_cost(idxN);
            work.entering_direction = entering_direction_(work.entering_col, rc_e, at_upper, l_work,
                                                          u_work, self.opt_.tol);

            // FTRAN: compute the pivotal column.
            HVector dB;
            try {
                dB = read_basis().solve_B(A.col(work.entering_col), FTBasis::TranKind::ColAq);
                nla->update_ema_reach(dB.count, m);
            } catch (...) {
                refactor_basis();
                xB_cache_valid = false;
                dB = read_basis().solve_B(A.col(work.entering_col), FTBasis::TranKind::ColAq);
                nla->update_ema_reach(dB.count, m);
                rebuild_pricing();
            }

            // CHUZR: choose the leaving basic row, allowing bound flips.
            const RatioResult rt =
                ratio_test(work.base_value, dB, work.entering_direction, basis, l_work, u_work,
                           self.opt_.ratio_delta, self.opt_.ratio_eta);

            // Step allowed by the entering variable's own opposite bound.
            const double l_e =
                (work.entering_col < l_work.size()) ? l_work(work.entering_col) : 0.0;
            const double u_e =
                (work.entering_col < u_work.size()) ? u_work(work.entering_col) : presolve::inf();
            const double range_e = (std::isfinite(l_e) && std::isfinite(u_e))
                                       ? std::max(0.0, u_e - l_e)
                                       : std::numeric_limits<double>::infinity();

            const double step = std::min(rt.theta, range_e);
            if (!std::isfinite(step)) {
                if (cleanup_primal_perturbations("unbounded step"))
                    continue;
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray =
                    unbounded_ray(work.entering_col, work.entering_direction, dB.value);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) +
                                 " entering=" + std::to_string(work.entering_col));
                auto info = dm_stats_to_map(self.degen_.get_stats());
                info["certificate"] = "primal_ray";
                info["primal_ray_has_cert"] = "1";
                info["primal_ray_dim"] = std::to_string(n);
                info["primal_ray"] = serialize_vec(ray);
                self.remember_warm_state_(basis, nla);
                return {LPSolution::Status::Unbounded, x, basis, iters, std::move(info)};
            }

            // ── Bound flip: entering variable hits its opposite bound first ──
            if (range_e < rt.theta - 1e-14) {
                const double old_val =
                    nonbasic_value_(work.entering_col, l_work, u_work, at_upper[work.entering_col]);
                at_upper[work.entering_col] = at_upper[work.entering_col] ? 0 : 1;
                const double new_val =
                    nonbasic_value_(work.entering_col, l_work, u_work, at_upper[work.entering_col]);
                const double delta_x = new_val - old_val;
                if (delta_x != 0.0) {
                    axpy_col_(A, work.entering_col, -delta_x, rhs_eff);
                    if (xB_cache_valid && xB_cache_age < xB_max_age) {
                        xB_cache.noalias() -= delta_x * dB.value;
                        ++xB_cache_age;
                    } else {
                        xB_cache_valid = false;
                    }
                }
                (void)self.degen_.detect_degeneracy(step, self.opt_.deg_step_tol);
                if (self.should_trace_iter_(iters)) {
                    self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                     " bound flip var=" + std::to_string(work.entering_col) +
                                     " step=" + std::to_string(step));
                }
                continue;
            }

            if (!rt.row) {
                if (cleanup_primal_perturbations("no leaving row"))
                    continue;
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray =
                    unbounded_ray(work.entering_col, work.entering_direction, dB.value);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) + " entering=" +
                                 std::to_string(work.entering_col) + " no leaving variable");
                auto info = dm_stats_to_map(self.degen_.get_stats());
                info["certificate"] = "primal_ray";
                info["primal_ray_has_cert"] = "1";
                info["primal_ray_dim"] = std::to_string(n);
                info["primal_ray"] = serialize_vec(ray);
                self.remember_warm_state_(basis, nla);
                return {LPSolution::Status::Unbounded, x, basis, iters, std::move(info)};
            }

            const int r = *rt.row;
            const double alpha = dB(r);
            const int oldAbs = basis[r];
            const int eAbs = work.entering_col;
            const auto basis_cycle = self.degen_.register_basis_change(basis, r, eAbs, iters);
            if (basis_cycle.repeated_basis) {
                self.trace_line_(
                    "[primal] iter=" + std::to_string(iters) +
                    " repeated basis candidate leave_var=" + std::to_string(oldAbs) +
                    " enter=" + std::to_string(eAbs) +
                    (basis_cycle.cycling_detected ? " cycle_detected=1" : " cycle_detected=0"));
            }

            const bool is_degenerate = self.degen_.detect_degeneracy(step, self.opt_.deg_step_tol);
            if (basis_cycle.cycling_detected ||
                (is_degenerate && self.degen_.should_apply_perturbation())) {
                if (!costs_perturbed) {
                    const double rel_multiplier =
                        1e-8 *
                        std::max(1e-6, self.opt_.primal_simplex_cost_perturbation_multiplier);
                    const double abs_multiplier =
                        1e-10 *
                        std::max(1e-6, self.opt_.primal_simplex_cost_perturbation_multiplier);
                    degeneracy_helpers::perturbCosts(c_work, self.rng_, rel_multiplier);
                    degeneracy_helpers::perturbCostsAbsolute(c_work, self.rng_, abs_multiplier);
                    costs_perturbed = true;
                }
                // Bound perturbation (HiGHS-style): shift bounds that are
                // violated by the current basic solution outward so the iterate
                // stays feasible without returning to Phase 1.
                if (self.opt_.primal_simplex_bound_perturbation && !bounds_perturbed) {
                    bounds_perturbed = degeneracy_helpers::perturbBounds(
                        work.base_value, basis, l_work, u_work, self.rng_, self.opt_.tol,
                        self.opt_.primal_simplex_bound_perturbation_multiplier);
                    if (bounds_perturbed)
                        xB_cache_valid = false; // bounds changed, xB may shift
                }
            } else {
                if (costs_perturbed) {
                    c_work = c;
                    costs_perturbed = false;
                }
                if (bounds_perturbed) {
                    l_work = l;
                    u_work = u;
                    bounds_perturbed = false;
                    xB_cache_valid = false;
                }
                (void)self.degen_.reset_perturbation();
            }

            if (self.opt_.pricing_rule == "adaptive") {
                const double rc_impr = -work.entering_measure(idxN);
                self.bridge_->after_primal_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N, rc_impr,
                                                 is_degenerate);
            }

            // UPDATE: apply the accepted basis change and maintain work arrays.
            // New value of the entering variable and exit value of the leaver.
            const double enter_old_val =
                nonbasic_value_(work.entering_col, l_work, u_work, at_upper[work.entering_col]);
            const double enter_new_val = enter_old_val + work.entering_direction * step;
            const double leave_exit_val =
                nonbasic_value_(oldAbs, l_work, u_work, rt.leaving_to_upper);

            if (self.should_trace_iter_(iters)) {
                const auto sv = sigma_view();
                const Eigen::VectorXd xcur =
                    self.assemble_primal_(n, basis, work.base_value, l_work, u_work, &sv);
                std::ostringstream oss;
                oss << "[primal] iter=" << iters << " obj=" << c.dot(xcur) << " enter=" << eAbs
                    << " leave_row=" << r << " leave_var=" << oldAbs << " step=" << step
                    << " alpha=" << alpha;
                if (self.opt_.verbose_include_basis) {
                    oss << " basis_before=" << self.format_basis_(basis);
                }
                self.trace_line_(oss.str());
            }

            basis[r] = eAbs;
            N[idxN] = oldAbs;
            at_upper[oldAbs] = rt.leaving_to_upper ? 1 : 0;

            // rhs_eff: the entering variable stops contributing as a nonbasic;
            // the leaving variable starts contributing at its exit bound.
            axpy_col_(A, eAbs, enter_old_val, rhs_eff);
            axpy_col_(A, oldAbs, -leave_exit_val, rhs_eff);

            // NLA framework switch check — rebuild if Devex weight errors accumulate
            if (nla->needs_framework_rebuild()) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " framework rebuild triggered");
                refactor_basis();
                nla->clear_framework_rebuild();
                xB_cache_valid = false;
                rebuild_pricing();
                continue;
            }

            // ── Incremental xB update (rank-1) ─────────────────────────────
            //   xB_new = xB_old − sigma·step·dB;  xB_new[r] = enter_new_val.
            if (xB_cache_valid && xB_cache_age < xB_max_age) {
                xB_cache.noalias() -= (work.entering_direction * step) * dB.value;
                xB_cache(r) = enter_new_val;
                ++xB_cache_age;
            } else {
                xB_cache_valid = false;
            }

            try {
                update_basis(r, eAbs, A.col(work.entering_col));
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                refactor_basis();
                xB_cache_valid = false;
                rebuild_pricing();
            }

            if (self.should_trace_iter_(iters) && self.opt_.verbose_include_basis) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
            }

            if (self.opt_.pricing_rule == "adaptive") {
                // NLA Devex framework switch — pass weight error from pricer
                if (self.adaptive_pricer_.needs_rebuild()) {
                    double w_err = self.adaptive_pricer_.average_log_weight_error();
                    nla->record_framework_error(w_err);
                }
            }
            if (self.opt_.pricing_rule == "adaptive" && self.adaptive_pricer_.needs_rebuild()) {
                self.measure_pricing_build_(
                    false, [&]() { self.adaptive_pricer_.build_primal_pools(read_basis(), A, N); });
                self.adaptive_pricer_.clear_rebuild_flag();
                // Pricing rebuild signals accumulated numerical drift → refresh xB next iteration.
                xB_cache_valid = false;
            }
        }

        self.trace_line_("[primal] iterlimit basis=" + self.format_basis_(basis));
        self.remember_warm_state_(basis, nla);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis, iters,
                dm_stats_to_map(self.degen_.get_stats())};
    }
};
