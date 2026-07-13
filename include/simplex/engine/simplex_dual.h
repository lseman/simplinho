#pragma once

#include "extern/pdqsort/pdqsort.h"
#include "simplex/engine/dual/pricing.h"
#include <algorithm>
#include <future>
#include <type_traits>

class RevisedSimplexDualEngine : public simplex::engine::DualPricingOperations {
  public:
    using SparseRowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    using BoundView = simplex::engine::DualRatioTest::BoundView;
    using DualChoose = simplex::engine::DualRatioTest::DualChoose;
    using DualBFRTDecision = simplex::engine::DualRatioTest::DualBFRTDecision;

    using DualPricingTelemetry = simplex::engine::DualPricingOperations::Telemetry;

    struct DualIterationWork {
        Eigen::VectorXd base_value;
        Eigen::VectorXd base_cost;
        Eigen::VectorXd work_dual;
        Eigen::VectorXd row_price;
        Eigen::VectorXd reduced_cost;
        HVector pivot_row;
        HVector pivot_col;
        int leaving_row = -1;
        int leaving_sign = 1;
        int entering_rel = -1;
        int entering_col = -1;
        double theta = std::numeric_limits<double>::infinity();
    };

    template <class MatrixType>
    static Eigen::VectorXd compute_nonbasic_duals_(const MatrixType& Ahat,
                                                   const std::vector<int>& nonbasis,
                                                   const Eigen::VectorXd& ydual,
                                                   const Eigen::VectorXd& chat) {
        Eigen::VectorXd rN(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            const int j = nonbasis[k];
            rN(k) = chat(j) - column_dot(Ahat, j, ydual);
        }
        return rN;
    }

    static bool dual_feasible_nonbasics_(const Eigen::VectorXd& rN, double tol) {
        for (int k = 0; k < rN.size(); ++k) {
            if (rN(k) < -tol)
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
        bool cost_shift_phase1_used = false;

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
                        {{"where", "dual full-basis rank check failed"}}};
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

        std::vector<BoundView> view(n, BoundView::Lower);
        for (int j = 0; j < n; ++j)
            view[j] = default_bound_view(j, l, u);
        const bool warm_views_provided =
            warm_status && warm_status->size() == static_cast<std::size_t>(n);

        // ── HiGHS-style upfront cost perturbation ──────────────────────────
        // HEkkDual::solve() (HEkkDual.cpp) perturbs costs unconditionally at
        // the start of every dual-phase-2 solve (unless the incoming point
        // is already near-optimal), rather than waiting for a reactive
        // degeneracy streak to accumulate mid-loop (the only mechanism this
        // engine had before). That matters here: this engine's above-upper
        // leaving-row handling produces long runs of *interleaved* (not
        // consecutive) degenerate pivots on harder LPs, which never trips
        // the streak-based `should_apply_perturbation()` gate, so the
        // engine can wander for tens of thousands of iterations before a
        // numerically-blown-up ratio test declares a false Infeasible.
        // Perturbing upfront (small, sign-matched to each column's active
        // bound direction) breaks that degeneracy before it starts. Skipped
        // when warm-starting in a BnB node context — those LPs are usually
        // only a few pivots from optimal, and perturbing would just add a
        // cleanup pass the objective-bound bailout makes irrelevant.
        const bool suppress_upfront_perturbation =
            self.opt_.dual_suppress_perturbation_when_warm &&
            std::isfinite(self.opt_.objective_bound_internal);
        bool costs_perturbed_upfront = false;
        if (!suppress_upfront_perturbation && self.opt_.dual_simplex_cost_perturbation_multiplier > 0.0) {
            double max_abs_cost = 0.0;
            for (int j = 0; j < n; ++j) {
                max_abs_cost = std::max(max_abs_cost, std::abs(c_work(j)));
            }
            if (max_abs_cost > 100.0) {
                max_abs_cost = std::sqrt(std::sqrt(max_abs_cost));
            }
            int boxed_count = 0;
            for (int j = 0; j < n; ++j) {
                if (std::isfinite(l(j)) && std::isfinite(u(j))) {
                    ++boxed_count;
                }
            }
            if (n > 0 && static_cast<double>(boxed_count) / n < 0.01) {
                max_abs_cost = std::min(max_abs_cost, 1.0);
            }
            const double perturbation_base =
                self.opt_.dual_simplex_cost_perturbation_multiplier * 5e-7 * max_abs_cost;
            if (perturbation_base > 0.0) {
                std::uniform_real_distribution<double> pert01(0.0, 1.0);
                for (int j = 0; j < n; ++j) {
                    const bool has_l = std::isfinite(l(j));
                    const bool has_u = std::isfinite(u(j));
                    if (!has_l && !has_u) {
                        continue;  // free — no perturb
                    }
                    if (has_l && has_u && std::abs(u(j) - l(j)) <= self.opt_.tol) {
                        continue;  // fixed — no perturb
                    }
                    const double xpert =
                        (1.0 + pert01(self.rng_)) * (std::abs(c_work(j)) + 1.0) * perturbation_base;
                    if (has_l && has_u) {
                        c_work(j) += (c_work(j) >= 0.0) ? xpert : -xpert;  // boxed
                    } else if (has_u) {
                        c_work(j) -= xpert;  // upper-only
                    } else {
                        c_work(j) += xpert;  // lower-only
                    }
                }
                costs_perturbed_upfront = true;
            }
        }
        costs_perturbed = costs_perturbed_upfront;

        self.bridge_.reset();
        DualAdaptivePricer dual_pricer(
            self.opt_.pricing_rule, self.opt_.devex_reset, self.opt_.adaptive_reset_freq,
            self.opt_.partial_pricing, self.opt_.dual_pricing, self.opt_.row_pricing_threshold,
            self.opt_.dual_edge_weight_strategy,
            self.opt_.dual_steepest_edge_weight_log_error_threshold,
            self.opt_.dual_warm_start_near_optimal, RevisedSimplex::find_logical_basis_(A).empty());

        MatrixType Ahat = signed_matrix_copy(A, view);
        std::optional<SparseRowMatrix> Ahat_row;
        Eigen::VectorXd chat = c_work;
        if (warm_views_provided) {
            std::vector<char> inB(n, 0);
            for (int j : basis)
                if (j >= 0 && j < n)
                    inB[j] = 1;
            for (int j = 0; j < n; ++j) {
                if (inB[j])
                    continue;
                switch ((*warm_status)[j]) {
                    case LPBasisStatus::AtUpper:
                        view[j] = std::isfinite(u(j)) ? BoundView::Upper : BoundView::Lower;
                        break;
                    case LPBasisStatus::Fixed:
                        view[j] = BoundView::Fixed;
                        break;
                    case LPBasisStatus::Basic:
                    case LPBasisStatus::AtLower:
                    default:
                        view[j] = BoundView::Lower;
                        break;
                }
            }
        }
        Ahat = signed_matrix_copy(A, view);
        if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
            Ahat_row.emplace(rowwise_copy(Ahat));
        }
        for (int j = 0; j < n; ++j)
            chat(j) = (view_sign(view[j]) > 0) ? c_work(j) : -c_work(j);
        for (int j : basis) {
            if (j >= 0 && j < n)
                view[j] = BoundView::Lower;
            if (j >= 0 && j < n) {
                if (chat(j) != c_work(j)) {
                    scale_column(Ahat, j, -1.0);
                    if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                        scale_rowwise_column(*Ahat_row, Ahat, j, -1.0);
                    }
                }
                chat(j) = c_work(j);
            }
        }

        std::shared_ptr<simplex::nla::SimplexNLA> nla;
        std::shared_ptr<LPWarmStateData> reused_warm_state = self.try_reuse_factorization_(basis);
        if (reused_warm_state) {
            nla = reused_warm_state->nla;
        } else {
            nla = std::make_shared<simplex::nla::SimplexNLA>();
            try {
                simplex::nla::NLAConfig nla_cfg;
                nla_cfg.framework_switch_threshold_ = self.opt_.framework_switch_threshold;
                nla_cfg.framework_switch_consecutive_ = self.opt_.framework_switch_consecutive;
                nla_cfg.allow_framework_switch_ = self.opt_.allow_framework_switch;
                if (self.opt_.price_strategy == "row_switch")
                    nla_cfg.price_strategy_ = simplex::nla::NLAConfig::PriceStrategy::RowSwitch;
                else if (self.opt_.price_strategy == "row_switch_col_switch")
                    nla_cfg.price_strategy_ =
                        simplex::nla::NLAConfig::PriceStrategy::RowSwitchColSwitch;
                else
                    nla_cfg.price_strategy_ = simplex::nla::NLAConfig::PriceStrategy::ColOnly;
                nla->setup(Ahat.rows(), 0.1, nla_cfg);
                nla->setup_factor(Ahat, basis, self.make_basis_options_());
            } catch (const std::exception& e) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "dual initial basis factorization failed"}, {"what", e.what()}}};
            }
        }
        auto rebuild_nla = [&]() -> std::shared_ptr<simplex::nla::SimplexNLA> {
            auto fresh = std::make_shared<simplex::nla::SimplexNLA>();
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
            fresh->setup(Ahat.rows(), 0.1, nla_cfg);
            fresh->setup_factor(Ahat, basis, self.make_basis_options_());
            return fresh;
        };
        auto read_basis = [&]() -> FTBasis& { return nla->factor(); };
        auto write_basis = [&]() -> FTBasis& {
            if (!nla.unique()) {
                nla = rebuild_nla();
            }
            return nla->factor();
        };
        auto refactor_basis = [&]() -> FTBasis& {
            if (!nla.unique()) {
                nla = rebuild_nla();
            } else {
                nla->invert();
            }
            return nla->factor();
        };
        auto update_basis = [&](int row, int entering_col, const auto& entering_vector) {
            if (!nla.unique())
                nla = rebuild_nla();
            nla->update_basis(row, entering_col, entering_vector);
        };
        self.degen_.start_basis_history(basis);

        auto apply_views_to_nonbasics = [&](const Eigen::VectorXd& ydual) {
            bool changed = false;
            std::vector<char> inB(n, 0);
            for (int j : basis)
                if (j >= 0 && j < n)
                    inB[j] = 1;

            for (int j = 0; j < n; ++j) {
                if (inB[j])
                    continue;

                const double raw_rc = c_work(j) - column_dot(A, j, ydual);
                const bool has_l = (j < l.size()) && std::isfinite(l(j));
                const bool has_u = (j < u.size()) && std::isfinite(u(j));
                BoundView next = view[j];

                if (has_l && has_u) {
                    if (std::abs(u(j) - l(j)) <= self.opt_.tol) {
                        next = BoundView::Fixed;
                    } else {
                        next = (raw_rc < 0.0) ? BoundView::Upper : BoundView::Lower;
                    }
                } else if (has_u && !has_l) {
                    next = BoundView::Upper;
                } else {
                    next = BoundView::Lower;
                }

                if (next != view[j]) {
                    scale_column(Ahat, j, -1.0);
                    if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                        scale_rowwise_column(*Ahat_row, Ahat, j, -1.0);
                    }
                    chat(j) = -chat(j);
                    view[j] = next;
                    changed = true;
                }
            }
            return changed;
        };

        if (!warm_views_provided) {
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i)
                cB(i) = chat(basis[i]);
            Eigen::VectorXd ydual = read_basis().solve_BT(cB, FTBasis::TranKind::RowEp);
            apply_views_to_nonbasics(ydual);
        }
        if (!reused_warm_state) {
            try {
                read_basis().refactor();
                self.runtime_state_.mark_rebuilt();
                self.runtime_state_.validity.has_basis = true;
            } catch (const std::exception& e) {
                self.runtime_state_.request_rebuild(
                    simplex::engine::RebuildReason::SingularBasis);
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "dual initial refactor failed"}, {"what", e.what()}}};
            }
        }

        auto rebuild_dual_pool = [&](const char* where,
                                     int iter) -> std::optional<RevisedSimplex::PhaseResult> {
            if (std::getenv("SIMPLINHO_TRACE_POOL_REBUILDS"))
                std::fprintf(stderr, "[poolrebuild] iter=%d where=%s\n", iter, where);
            try {
                self.measure_pricing_build_(
                    true, [&]() { dual_pricer.build_dual_pool(read_basis(), Ahat, N); });
                if (std::getenv("SIMPLINHO_TRACE_POOL_REBUILDS"))
                    std::fprintf(stderr, "[poolrebuild] strategy=%s\n",
                                 dual_pricer.current_strategy_name());
                self.runtime_state_.validity.has_pricing_weights = true;
                return std::nullopt;
            } catch (const std::exception& e) {
                self.runtime_state_.validity.has_pricing_weights = false;
                self.runtime_state_.request_rebuild(
                    simplex::engine::RebuildReason::PricingFailure);
                std::unordered_map<std::string, std::string> info{
                    {"where", where},
                    {"what", e.what()},
                };
                if (iter > 0) {
                    info["iter"] = std::to_string(iter);
                }
                return RevisedSimplex::PhaseResult{LPSolution::Status::Singular,
                                                   Eigen::VectorXd::Zero(n), basis, iter,
                                                   std::move(info)};
            }
        };
        bool restored_dual_weights = false;
        if (reused_warm_state && reused_warm_state->dual_pricing_state.has_value()) {
            DualAdaptivePricer::WarmStartState imported;
            switch (reused_warm_state->dual_pricing_state->active_rule) {
                case LPDualPricingWarmState::Rule::SteepestEdge:
                    imported.active_rule = DualAdaptivePricer::Rule::SteepestEdge;
                    imported.steepest.row_weights =
                        reused_warm_state->dual_pricing_state->row_weights;
                    break;
                case LPDualPricingWarmState::Rule::Devex:
                    imported.active_rule = DualAdaptivePricer::Rule::Devex;
                    imported.devex.row_weights = reused_warm_state->dual_pricing_state->row_weights;
                    break;
                case LPDualPricingWarmState::Rule::RowPricing:
                    imported.active_rule = DualAdaptivePricer::Rule::RowPricing;
                    imported.row.row_weights = reused_warm_state->dual_pricing_state->row_weights;
                    break;
                case LPDualPricingWarmState::Rule::MostInfeasible:
                    imported.active_rule = DualAdaptivePricer::Rule::MostInfeasible;
                    break;
                case LPDualPricingWarmState::Rule::None:
                    imported.active_rule = DualAdaptivePricer::Rule::MostInfeasible;
                    break;
            }
            restored_dual_weights = dual_pricer.import_state(imported, m);
            if (restored_dual_weights) {
                self.solve_stats_.warm_dual_weights_reused = 1;
            }
        }
        if (!restored_dual_weights) {
            if (auto failed = rebuild_dual_pool("dual initial pricing setup failed", 0)) {
                return *failed;
            }
        }
        self.trace_line_("[dual] start basis=" + self.format_basis_(basis));

        int rebuild_attempts = 0;
        int backtrack_repairs = 0;
        int total_flips = 0;
        const int adaptive_flip_budget =
            self.opt_.dual_flip_max_per_iter > 0
                ? self.opt_.dual_flip_max_per_iter
                : (m >= 256 ? 16 : 4);
        DualPricingTelemetry pricing_telemetry;
        Eigen::VectorXd rhs_eff = b - transformed_rhs(A, view, l, u);
        bool ydual_cached = false;
        Eigen::VectorXd ydual;

        auto attach_dual_pricing_info =
            [&](std::unordered_map<std::string, std::string>& info_map) {
                info_map["dual_pricing"] = dual_pricer.current_strategy_name();
                info_map["dual_bfrt_flips"] = std::to_string(total_flips);
                info_map["dual_row_price_calls"] =
                    std::to_string(pricing_telemetry.row_price_calls);
                info_map["dual_col_price_calls"] =
                    std::to_string(pricing_telemetry.col_price_calls);
                info_map["dual_price_switches"] = std::to_string(pricing_telemetry.price_switches);
                info_map["dual_row_ep_density"] = std::to_string(pricing_telemetry.row_ep_density);
                info_map["dual_row_ap_density"] = std::to_string(pricing_telemetry.row_ap_density);
                info_map["dual_col_aq_density"] = std::to_string(pricing_telemetry.col_aq_density);
                self.solve_stats_.dual_row_price_calls = pricing_telemetry.row_price_calls;
                self.solve_stats_.dual_col_price_calls = pricing_telemetry.col_price_calls;
                self.solve_stats_.dual_price_switches = pricing_telemetry.price_switches;
                self.solve_stats_.dual_row_ep_density = pricing_telemetry.row_ep_density;
                self.solve_stats_.dual_row_ap_density = pricing_telemetry.row_ap_density;
                self.solve_stats_.dual_col_aq_density = pricing_telemetry.col_aq_density;
                auto& density = self.runtime_state_.density;
                density.row_ep = simplex::engine::DensityHistory::update(
                    density.row_ep, pricing_telemetry.row_ep_density);
                density.row_ap = simplex::engine::DensityHistory::update(
                    density.row_ap, pricing_telemetry.row_ap_density);
                density.col_aq = simplex::engine::DensityHistory::update(
                    density.col_aq, pricing_telemetry.col_aq_density);
            };

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

        auto exact_dual_optimality_check = [&](std::string& reason) {
            Eigen::VectorXd true_chat(n);
            for (int j = 0; j < n; ++j)
                true_chat(j) = (view_sign(view[j]) > 0) ? c(j) : -c(j);
            Eigen::VectorXd true_cB(m);
            for (int i = 0; i < m; ++i)
                true_cB(i) = true_chat(basis[i]);

            HVector true_ydual_hvec;
            try {
                true_ydual_hvec = read_basis().solve_BT(true_cB, FTBasis::TranKind::RowEp);
                nla->update_ema_reach(true_ydual_hvec.count, m);
            } catch (...) {
                reason = "optimal_dual_check_solve_failed";
                return false;
            }

            const Eigen::VectorXd true_rN =
                compute_nonbasic_duals_(Ahat, N, true_ydual_hvec.value, true_chat);
            if (!dual_feasible_nonbasics_(true_rN, self.opt_.tol)) {
                reason = "optimal_dual_check_failed";
                return false;
            }
            return true;
        };

        auto finish_optimal = [&](const Eigen::VectorXd& yB_current) {
            Eigen::VectorXd x =
                assemble_transformed_primal(n, basis, yB_current.cwiseMax(0.0), l, u, view);
            if (!RevisedSimplex::primal_feasible_(A, b, x, l, u, self.opt_.tol)) {
                auto info_map = dm_stats_to_map(self.degen_.get_stats());
                attach_dual_pricing_info(info_map);
                info_map["reason"] = "optimal_primal_check_failed";
                return RevisedSimplex::PhaseResult{LPSolution::Status::Singular, std::move(x),
                                                   basis, iters, std::move(info_map)};
            }
            std::string exact_reason;
            if (!exact_dual_optimality_check(exact_reason)) {
                auto info_map = dm_stats_to_map(self.degen_.get_stats());
                attach_dual_pricing_info(info_map);
                info_map["reason"] =
                    exact_reason.empty() ? "optimality_check_failed" : exact_reason;
                return RevisedSimplex::PhaseResult{LPSolution::Status::NeedPhase1, std::move(x),
                                                   basis, iters, std::move(info_map)};
            }
            auto info_map = dm_stats_to_map(self.degen_.get_stats());
            attach_dual_pricing_info(info_map);
            self.trace_line_("[dual] optimal iter=" + std::to_string(iters) +
                             " basis=" + self.format_basis_(basis));
            const auto dual_state = dual_pricer.export_state();
            LPDualPricingWarmState pricing_state;
            switch (dual_state.active_rule) {
                case DualAdaptivePricer::Rule::SteepestEdge:
                    pricing_state.active_rule = LPDualPricingWarmState::Rule::SteepestEdge;
                    pricing_state.row_weights = dual_state.steepest.row_weights;
                    break;
                case DualAdaptivePricer::Rule::Devex:
                    pricing_state.active_rule = LPDualPricingWarmState::Rule::Devex;
                    pricing_state.row_weights = dual_state.devex.row_weights;
                    break;
                case DualAdaptivePricer::Rule::RowPricing:
                    pricing_state.active_rule = LPDualPricingWarmState::Rule::RowPricing;
                    pricing_state.row_weights = dual_state.row.row_weights;
                    break;
                case DualAdaptivePricer::Rule::MostInfeasible:
                    pricing_state.active_rule = LPDualPricingWarmState::Rule::MostInfeasible;
                    break;
            }
            self.remember_warm_state_(basis, nla, pricing_state);
            return RevisedSimplex::PhaseResult{LPSolution::Status::Optimal, std::move(x), basis,
                                               iters, std::move(info_map)};
        };

        // ── Incremental primal-solution (yB) cache ────────────────────────────
        // Each outer iteration needs yB = B^{-1} rhs_eff. Recomputing it from
        // scratch costs one full BTRAN. Instead we maintain it with the rank-1
        // update formula after each standard pivot:
        //   yB_new = yB_old − tau_r·s_enter + tau_r·e_{r_leave}
        //   (tau_r = yB_old[r_leave] / s_enter[r_leave], derived via SMW)
        // Falls back to a full BTRAN when rhs_eff changes (BFRT bound-flips),
        // after explicit refactors / backtrack, or every refactor_every pivots.
        Eigen::VectorXd yB_cache;
        bool yB_cache_valid = false;
        int yB_cache_age = 0;
        const int yB_max_age = std::max(1, self.opt_.refactor_every);
        auto refresh_yB_cache = [&]() {
            {
                HVector yB_hvec = read_basis().solve_B(rhs_eff, FTBasis::TranKind::ColAq);
                yB_cache = yB_hvec.value;
                nla->update_ema_reach(yB_hvec.count, m);
            }
            yB_cache_valid = true;
            yB_cache_age = 0;
        };
        refresh_yB_cache(); // prime the cache before the loop

        // ── Two-sided basic-variable feasibility ───────────────────────────
        // A basic variable's value in `yB` terms is always measured from its
        // Lower bound (`view[basis[i]] == Lower` is an invariant enforced
        // throughout — see the `for (int j : basis) view[j] = Lower;` blocks
        // above). Feasibility is therefore two-sided:
        //   0 <= yB(i) <= range(i),  range(i) = u(basis[i]) - l(basis[i]).
        // `choose_dual_leaving` (in pricer.h, shared by all pricing rules)
        // only ever checks `yB(i) < -tol`, i.e. it is blind to a basic
        // variable that has walked above its upper bound. To reuse that
        // machinery unmodified, `basic_leaving_infeasibility` builds a
        // "folded" vector where an above-upper row `i` is remapped to
        // `range(i) - yB(i) < 0` — the same shape of violation
        // `choose_dual_leaving` already knows how to rank and select. The
        // returned `sign[i]` records which physical bound row `i` violates
        // (+1 below-lower, -1 above-upper) so every downstream step (pricing
        // direction, ratio-test filter, post-pivot nonbasic bound) can un-fold
        // it consistently.
        auto basic_leaving_infeasibility = [&](const Eigen::VectorXd& yB_in,
                                               std::vector<int>& sign_out) {
            Eigen::VectorXd folded = yB_in;
            sign_out.assign(static_cast<std::size_t>(m), 1);
            for (int i = 0; i < m && i < yB_in.size(); ++i) {
                const int j = basis[i];
                if (j < 0 || j >= n || view[j] == BoundView::Fixed) {
                    continue;
                }
                const double range = bound_range(j, l, u);
                if (std::isfinite(range) && yB_in(i) > range + self.opt_.tol) {
                    folded(i) = range - yB_in(i);
                    sign_out[static_cast<std::size_t>(i)] = -1;
                }
            }
            return folded;
        };
        auto basic_above_range_rows = [&](const Eigen::VectorXd& yB_in) {
            std::vector<int> rows;
            rows.reserve(static_cast<std::size_t>(m));
            for (int i = 0; i < m && i < yB_in.size(); ++i) {
                const int j = basis[i];
                if (j < 0 || j >= n || view[j] == BoundView::Fixed) {
                    continue;
                }
                const double range = bound_range(j, l, u);
                if (std::isfinite(range) && yB_in(i) > range + self.opt_.tol) {
                    rows.push_back(i);
                }
            }
            return rows;
        };

        while (iters < self.opt_.max_iters) {
            ++iters;
            int flips_this_iter = 0;
            DualIterationWork work;
            work.base_cost.resize(m);

            while (true) {
                try {
                    // Use cached yB when valid; otherwise solve from scratch and prime cache.
                    if (!yB_cache_valid || yB_cache_age >= yB_max_age)
                        refresh_yB_cache();
                    work.base_value = yB_cache;
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve_B failure");
                        refactor_basis();
                        yB_cache_valid = false;
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after solve_B", iters)) {
                            return *failed;
                        }
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: solve(Bhat,rhs) repair failed"}}};
                }

                for (int i = 0; i < m; ++i)
                    work.base_cost(i) = chat(basis[i]);
                if (!ydual_cached) {
                    try {
                        {
                            HVector ydual_hvec =
                                read_basis().solve_BT(work.base_cost, FTBasis::TranKind::RowEp);
                            ydual = ydual_hvec.value;
                            nla->update_ema_reach(ydual_hvec.count, m);
                            ydual_cached = true;
                        }
                    } catch (...) {
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve_BT failure");
                        refactor_basis();
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after solve_BT", iters)) {
                            return *failed;
                        }
                        {
                            HVector ydual_hvec =
                                read_basis().solve_BT(work.base_cost, FTBasis::TranKind::RowEp);
                            ydual = ydual_hvec.value;
                            nla->update_ema_reach(ydual_hvec.count, m);
                            ydual_cached = true;
                        }
                    }
                }
                work.work_dual = ydual;

                if (!(warm_views_provided && iters == 1) &&
                    apply_views_to_nonbasics(work.work_dual)) {
                    rhs_eff = b - transformed_rhs(A, view, l, u);
                    yB_cache_valid = false; // rhs_eff recomputed from scratch
                    continue;
                }

                std::vector<int> leaving_row_sign;
                const Eigen::VectorXd yB_for_leaving =
                    basic_leaving_infeasibility(work.base_value, leaving_row_sign);
                // CHUZR: choose the leaving basic row from primal infeasibilities.
                const auto leaving =
                    dual_pricer.choose_dual_leaving(read_basis(), yB_for_leaving, self.opt_.tol);
                work.leaving_row = leaving.row;
                work.leaving_sign =
                    (work.leaving_row >= 0 &&
                     work.leaving_row < static_cast<int>(leaving_row_sign.size()))
                        ? leaving_row_sign[static_cast<std::size_t>(work.leaving_row)]
                        : 1;
                if (work.leaving_row < 0) {
                    work.reduced_cost = compute_nonbasic_duals_(Ahat, N, ydual, chat);
                    bool dual_feasible =
                        dual_feasible_nonbasics_(work.reduced_cost, self.opt_.tol);
                    // HiGHS-style cleanup (HEkkDual::cleanup()): this
                    // dual-feasibility check ran against perturbed costs, so
                    // an apparent optimum here may be an artifact of the
                    // perturbation, not a real one. Remove it and recheck
                    // with the true costs before trusting the result; if it
                    // no longer holds, fall through to another outer
                    // iteration (now perturbation-free) instead of returning
                    // early.
                    if (dual_feasible && costs_perturbed) {
                        c_work = c;
                        for (int j = 0; j < n; ++j) {
                            chat(j) = (view_sign(view[j]) > 0) ? c_work(j) : -c_work(j);
                        }
                        costs_perturbed = false;
                        ydual_cached = false;
                        {
                            Eigen::VectorXd cB_clean(m);
                            for (int i = 0; i < m; ++i)
                                cB_clean(i) = chat(basis[i]);
                            HVector ydual_hvec =
                                read_basis().solve_BT(cB_clean, FTBasis::TranKind::RowEp);
                            ydual = ydual_hvec.value;
                            nla->update_ema_reach(ydual_hvec.count, m);
                            ydual_cached = true;
                        }
                        work.reduced_cost = compute_nonbasic_duals_(Ahat, N, ydual, chat);
                        dual_feasible =
                            dual_feasible_nonbasics_(work.reduced_cost, self.opt_.tol);
                        if (!dual_feasible && apply_views_to_nonbasics(ydual)) {
                            rhs_eff = b - transformed_rhs(A, view, l, u);
                            yB_cache_valid = false;
                            continue;
                        }
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after cleanup", iters)) {
                            return *failed;
                        }
                        if (!dual_feasible) {
                            self.trace_line_(
                                "[dual] iter=" + std::to_string(iters) +
                                " cleanup: not optimal with true costs, continuing");
                            continue;
                        }
                    }
                    if (dual_feasible) {
                        return finish_optimal(work.base_value);
                    }
                    self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                     " primal-feasible but dual-infeasible");
                    if (!cost_shift_phase1_used) {
                        cost_shift_phase1_used = true;
                        double max_violation = 0.0;
                        for (int k = 0; k < (int)N.size(); ++k) {
                            if (work.reduced_cost(k) < -self.opt_.tol &&
                                -work.reduced_cost(k) > max_violation) {
                                max_violation = -work.reduced_cost(k);
                            }
                        }
                        if (max_violation > 0.0 && max_violation < 1e12) {
                            Eigen::VectorXd shift = Eigen::VectorXd::Zero(n);
                            for (int k = 0; k < (int)N.size(); ++k) {
                                if (work.reduced_cost(k) < -self.opt_.tol) {
                                    const int j = N[k];
                                    shift(j) = -work.reduced_cost(k) * 1.5;
                                }
                            }
                            Eigen::VectorXd c_shifted = c_work + shift;
                            for (int k = 0; k < (int)N.size(); ++k) {
                                const int j = N[k];
                                chat(j) = (view_sign(view[j]) > 0) ? c_shifted(j) : -c_shifted(j);
                            }
                            for (int i = 0; i < m; ++i) {
                                const int j = basis[i];
                                chat(j) = view_sign(view[j]) > 0 ? c_work(j) : -c_work(j);
                            }
                            self.trace_line_("[dual] cost-shift Phase 1 applied, refactoring");
                            try {
                                refactor_basis();
                                yB_cache_valid = false;
                            } catch (const std::exception&) {
                                return {LPSolution::Status::NeedPhase1,
                                        Eigen::VectorXd::Zero(n),
                                        basis,
                                        iters,
                                        {{"reason", "cost_shift_refactor_failed"}}};
                            }
                            if (auto failed =
                                    rebuild_dual_pool("cost-shift refactor failed", iters)) {
                                return *failed;
                            }
                            continue;
                        }
                    }
                    return {LPSolution::Status::NeedPhase1,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"reason", "dual_infeasible_at_primal_feasible"}}};
                }

                // `leaving.dual_row` is B^{-T}e_{r_leave} in the raw (unfolded)
                // orientation. For an above-upper leaving row (leaving_sign<0)
                // pricing must run in the folded orientation so the existing
                // `pN(k) < -tol` candidate filter (below-lower convention)
                // also selects the entering variables that would *decrease*
                // an above-range basic value. Negate w for pricing only; the
                // pivot/update math below (`pivot`, `alpha`, `z`) intentionally
                // keeps using the raw, un-negated BTRAN quantities — see the
                // comment at the pivot-apply site for why those must stay raw.
                work.pivot_row = leaving.dual_row;
                if (work.leaving_sign < 0) {
                    work.pivot_row.value = -work.pivot_row.value;
                }
                const double local_row_ep_density = vector_density(work.pivot_row, self.opt_.tol);
                DualPricingTelemetry::update_density(local_row_ep_density,
                                                     pricing_telemetry.row_ep_density);
                // PRICE + CHUZC: price the pivot row and choose the entering column.
                if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                    const bool allow_runtime_switch =
                        self.opt_.dual_pricing == "switch" || self.opt_.dual_pricing == "col";
                    const bool use_column_price =
                        self.opt_.dual_pricing == "col" ||
                        (allow_runtime_switch &&
                         (local_row_ep_density > 0.75 || (pricing_telemetry.row_ap_density > 0.10 &&
                                                          local_row_ep_density > 0.10)));
                    pricing_telemetry.record_price_mode(use_column_price);
                    if (use_column_price) {
                        if (self.opt_.parallel_pricing_workers > 1 &&
                            static_cast<int>(N.size()) >=
                                std::max(1, self.opt_.parallel_pricing_min_cols)) {
                            compute_pricing_products_parallel(
                                Ahat, N, work.pivot_row, ydual, chat, work.row_price,
                                work.reduced_cost, self.opt_.parallel_pricing_workers,
                                self.opt_.parallel_pricing_min_cols);
                        } else {
                            compute_pricing_products_by_column(
                                Ahat, N, work.pivot_row, ydual, chat, work.row_price,
                                work.reduced_cost);
                        }
                    } else {
                        compute_pricing_products(Ahat, *Ahat_row, N, work.pivot_row, ydual, chat,
                                                 work.row_price, work.reduced_cost);
                    }
                } else {
                    pricing_telemetry.record_price_mode(true);
                    compute_pricing_products(Ahat, N, work.pivot_row, ydual, chat, work.row_price,
                                             work.reduced_cost);
                }
                DualPricingTelemetry::update_density(vector_density(work.row_price, self.opt_.tol),
                                                     pricing_telemetry.row_ap_density);

                const DualBFRTDecision bfrt =
                    dual_bfrt_decide(self.opt_, work.reduced_cost, work.row_price, N, view, l, u,
                                     -yB_for_leaving(work.leaving_row),
                                     self.opt_.dual_allow_bound_flip
                                         ? (adaptive_flip_budget - flips_this_iter)
                                         : 0,
                                     read_basis().update_count());
                if (!bfrt.pivot_rel) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after no eligible entering");
                        refactor_basis();
                        yB_cache_valid = false;
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after no eligible entering", iters)) {
                            return *failed;
                        }
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: no eligible entering"}}};
                }

                if (!bfrt.flip_rels.empty()) {
                    if (self.should_trace_iter_(iters)) {
                        std::ostringstream oss;
                        oss << "[dual] iter=" << iters << " bound flips=" << bfrt.flip_rels.size();
                        if (self.opt_.verbose_include_basis) {
                            oss << " basis=" << self.format_basis_(basis);
                        }
                        self.trace_line_(oss.str());
                    }
                    // Accumulate every anchor change and apply B^{-1} once,
                    // matching HiGHS' BFRT-column update. The old path
                    // restarted the outer loop and paid a fresh solve for the
                    // complete RHS after every group of flips.
                    Eigen::VectorXd flip_rhs = Eigen::VectorXd::Zero(m);
                    for (int rel_k : bfrt.flip_rels) {
                        const int j = N[rel_k];
                        const double old_anchor = bound_anchor(view[j], j, l, u);
                        view[j] =
                            (view[j] == BoundView::Upper) ? BoundView::Lower : BoundView::Upper;
                        const double new_anchor = bound_anchor(view[j], j, l, u);
                        const double delta_anchor = new_anchor - old_anchor;
                        if (delta_anchor != 0.0) {
                            if constexpr (std::is_same_v<MatrixType,
                                                         RevisedSimplex::SparseMatrix>) {
                                for (typename RevisedSimplex::SparseMatrix::InnerIterator it(A, j);
                                     it; ++it) {
                                    flip_rhs(it.row()) -= it.value() * delta_anchor;
                                }
                            } else {
                                const Eigen::VectorXd col_j = A.col(j);
                                flip_rhs.noalias() -= col_j * delta_anchor;
                            }
                        }
                        if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                            for (typename RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, j);
                                 it; ++it) {
                                it.valueRef() = -it.valueRef();
                            }
                        } else {
                            Ahat.col(j) = -Ahat.col(j);
                        }
                        if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                            scale_rowwise_column(*Ahat_row, Ahat, j, -1.0);
                        }
                        chat(j) = -chat(j);
                        work.row_price(rel_k) = -work.row_price(rel_k);
                        work.reduced_cost(rel_k) = -work.reduced_cost(rel_k);
                        ++flips_this_iter;
                        ++total_flips;
                    }
                    rhs_eff.noalias() += flip_rhs;
                    try {
                        HVector flip_col =
                            read_basis().solve_B(flip_rhs, FTBasis::TranKind::ColAq);
                        nla->update_ema_reach(flip_col.count, m);
                        work.base_value.noalias() += flip_col.value;
                        if (yB_cache_valid) {
                            yB_cache.noalias() += flip_col.value;
                            ++yB_cache_age;
                        }
                    } catch (...) {
                        // The signed columns and RHS are already consistent;
                        // restart with a fresh factorization/cache if the
                        // incremental BFRT solve cannot be trusted.
                        refactor_basis();
                        yB_cache_valid = false;
                        continue;
                    }
                }

                work.entering_rel = *bfrt.pivot_rel;
                work.entering_col = N[work.entering_rel];
                work.theta = bfrt.tau;
                if (self.degen_.would_repeat_basis_change(basis, work.leaving_row,
                                                           work.entering_col)) {
                    int alt_rel = -1;
                    double alt_tau = std::numeric_limits<double>::infinity();
                    for (int k = 0; k < static_cast<int>(N.size()); ++k) {
                        if (k == work.entering_rel ||
                            !(work.row_price(k) < -self.opt_.ratio_delta)) {
                            continue;
                        }
                        const double candidate_tau = work.reduced_cost(k) / (-work.row_price(k));
                        if (!std::isfinite(candidate_tau) || candidate_tau < 0.0) {
                            continue;
                        }
                        const int candidate_abs = N[k];
                        if (self.degen_.would_repeat_basis_change(basis, work.leaving_row,
                                                                  candidate_abs)) {
                            continue;
                        }
                        if (candidate_tau < alt_tau - 1e-16 ||
                            (std::abs(candidate_tau - alt_tau) <= 1e-16 &&
                             (alt_rel < 0 || candidate_abs < N[alt_rel]))) {
                            alt_rel = k;
                            alt_tau = candidate_tau;
                        }
                    }
                    if (alt_rel >= 0) {
                        work.entering_rel = alt_rel;
                        work.entering_col = N[work.entering_rel];
                        work.theta = alt_tau;
                    }
                }
                // FTRAN: compute the pivotal column for the selected entering variable.
                try {
                    {
                        work.pivot_col = read_basis().solve_B(Ahat.col(work.entering_col),
                                                              FTBasis::TranKind::ColAq);
                        nla->update_ema_reach(work.pivot_col.count, m);
                    }
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve(B,a_e) failure");
                        refactor_basis();
                        yB_cache_valid = false;
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after solve(B,a_e)", iters)) {
                            return *failed;
                        }
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: solve(Bhat,a_e) repair failed"}}};
                }
                DualPricingTelemetry::update_density(vector_density(work.pivot_col, self.opt_.tol),
                                                     pricing_telemetry.col_aq_density);
                // HiGHS reinvertOnNumericalTrouble: the pivot value is known
                // from both the BTRAN'd pivot row (row_price at the entering
                // column) and the FTRAN'd pivot column (at the leaving row).
                // A large relative difference between the two means the
                // factorization is drifting — reinvert now, before pivoting
                // on a bad alpha. This check is essentially free.
                {
                    const double alpha_col = work.pivot_col(work.leaving_row);
                    const double alpha_row =
                        (work.entering_rel >= 0 &&
                         work.entering_rel < static_cast<int>(work.row_price.size()))
                            ? work.row_price(work.entering_rel)
                            : alpha_col;
                    const double min_abs = std::min(std::abs(alpha_col), std::abs(alpha_row));
                    const double diff = std::abs(std::abs(alpha_col) - std::abs(alpha_row));
                    if (min_abs > 0.0 && diff > 1e-7 * min_abs &&
                        rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after alpha cross-check (col=" +
                                         std::to_string(alpha_col) +
                                         " row=" + std::to_string(alpha_row) + ")");
                        refactor_basis();
                        yB_cache_valid = false;
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after alpha cross-check", iters)) {
                            return *failed;
                        }
                        continue;
                    }
                }
                break;
            }

            if (!std::isfinite(work.theta)) {
                Eigen::VectorXd yF = work.pivot_row;
                if (yF.dot(rhs_eff) >= 0)
                    yF = -yF;

                auto info_map = dm_stats_to_map(self.degen_.get_stats());
                info_map["where"] = "dual: infinite step";
                attach_dual_pricing_info(info_map);
                info_map["certificate"] = "farkas";
                info_map["farkas_has_cert"] = "1";
                info_map["farkas_dim"] = std::to_string(m);
                info_map["farkas_y"] = serialize_vec(yF);
                self.trace_line_("[dual] infeasible iter=" + std::to_string(iters) +
                                 " produced Farkas certificate");
                self.remember_warm_state_(basis, nla);
                return {LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n), basis, iters,
                        std::move(info_map)};
            }

            const bool is_degenerate =
                self.degen_.detect_degeneracy(work.theta, self.opt_.deg_step_tol);
            const int oldAbs = basis[work.leaving_row];
            // A leaving row selected as below-lower (leaving_sign>0) exits to
            // its Lower bound, matching the permanent Lower-view invariant it
            // already had while basic — no view change needed. A leaving row
            // selected as above-upper (leaving_sign<0) must instead exit to
            // its Upper bound, since that's the bound it was violating.
            const BoundView leaving_current_view = view[oldAbs];
            const BoundView leaving_target_view =
                work.leaving_sign < 0
                    ? (leaving_current_view == BoundView::Upper ? BoundView::Lower
                                                               : BoundView::Upper)
                    : leaving_current_view;
            const auto basis_cycle =
                self.degen_.register_basis_change(basis, work.leaving_row, work.entering_col,
                                                  iters);
            if (basis_cycle.repeated_basis) {
                self.trace_line_(
                    "[dual] iter=" + std::to_string(iters) +
                    " repeated basis candidate leave_var=" + std::to_string(oldAbs) +
                    " enter=" + std::to_string(work.entering_col) +
                    (basis_cycle.cycling_detected ? " cycle_detected=1" : " cycle_detected=0"));
            }
            // HiGHS-style: when warm-starting in a BNB context (signalled by
            // dual_suppress_perturbation_when_warm + a finite objective bound),
            // skip the reactive cost perturbation. Node LPs typically pivot a
            // handful of times before they're optimal or pruned; perturbing
            // their costs only adds a cleanup pass that the bailout check
            // makes irrelevant.
            const bool suppress_perturbation = self.opt_.dual_suppress_perturbation_when_warm &&
                                               std::isfinite(self.opt_.objective_bound_internal);
            if (!suppress_perturbation &&
                (basis_cycle.cycling_detected ||
                 (is_degenerate && self.degen_.should_apply_perturbation()))) {
                if (!costs_perturbed) {
                    const double rel_multiplier =
                        1e-8 * std::max(1e-6, self.opt_.dual_simplex_cost_perturbation_multiplier);
                    const double abs_multiplier =
                        1e-10 * std::max(1e-6, self.opt_.dual_simplex_cost_perturbation_multiplier);
                    degeneracy_helpers::perturbCosts(c_work, self.rng_, rel_multiplier);
                    degeneracy_helpers::perturbCostsAbsolute(c_work, self.rng_, abs_multiplier);
                    for (int j = 0; j < n; ++j) {
                        chat(j) = (view_sign(view[j]) > 0) ? c_work(j) : -c_work(j);
                    }
                    costs_perturbed = true;
                    ydual_cached = false;
                }
            } else {
                // Only clear costs that this reactive branch itself applied.
                // The upfront perturbation (costs_perturbed_upfront) is
                // deliberately unconditional for the whole solve — HiGHS
                // doesn't reactively clear it either; it survives until the
                // explicit cleanup-before-Optimal step. Clearing it here on
                // the very first non-degenerate pivot defeated its purpose:
                // it never got a chance to prevent the *later* interleaved
                // degeneracy this LP exhibits.
                if (costs_perturbed && !costs_perturbed_upfront) {
                    c_work = c;
                    for (int j = 0; j < n; ++j) {
                        chat(j) = (view_sign(view[j]) > 0) ? c_work(j) : -c_work(j);
                    }
                    costs_perturbed = false;
                    ydual_cached = false;
                }
                (void)self.degen_.reset_perturbation();
            }
            self.degen_.after_pivot(
                work.leaving_row, work.entering_col, work.theta, 0.0,
                std::isfinite(work.theta) ? std::abs(work.theta) : 0.0, is_degenerate);

            if (self.should_trace_iter_(iters)) {
                Eigen::VectorXd xcur =
                    assemble_transformed_primal(n, basis, work.base_value, l, u, view);
                std::ostringstream oss;
                oss << "[dual] iter=" << iters << " obj=" << c.dot(xcur)
                    << " leave_row=" << work.leaving_row << " leave_var=" << oldAbs
                    << " enter=" << work.entering_col << " tau=" << work.theta
                    << " leaving_sign=" << work.leaving_sign;
                if (self.opt_.verbose_include_basis) {
                    oss << " basis_before=" << self.format_basis_(basis);
                }
                self.trace_line_(oss.str());
            }

            // UPDATE: apply the accepted basis change and maintain work arrays.
            // `pivot`/`z` intentionally use the raw (unflipped) BTRAN,
            // independent of `leaving_sign`: `s_enter = B^{-1}Ahat.col(eAbs)`
            // and `z = B^{-T}e_{r_leave}` are plain linear solves that never
            // depended on the pricing-only `w` flip above, and the identity
            // `alpha * (Ahat.col(k)'z) == rN_new(k) - rN(k)` that this dual
            // update relies on to keep `rN` consistent holds only with the
            // raw `z`/`pivot` — flipping them here (as an earlier attempt
            // did) breaks that identity and corrupts `ydual`/`rN` on every
            // above-upper pivot.
            HVector z = read_basis().solve_BT_unit(work.leaving_row, FTBasis::TranKind::RowEp);
            const double pivot = work.pivot_col(work.leaving_row);
            const double alpha = work.reduced_cost(work.entering_rel) / pivot;
            ydual.noalias() += alpha * z.value;
            for (int k = 0; k < static_cast<int>(N.size()); ++k) {
                if (k == work.entering_rel)
                    continue;
                work.reduced_cost(k) -= alpha * column_dot(Ahat, N[k], z);
            }
            basis[work.leaving_row] = work.entering_col;
            N[work.entering_rel] = oldAbs;
            // The leaving variable `oldAbs` was always viewed as Lower while
            // basic (the basis-membership invariant). If it's exiting to
            // Upper instead (leaving_sign<0), re-anchor it now: shift
            // `rhs_eff` by the anchor delta and flip its `Ahat`/`chat` column
            // sign, exactly like the nonbasic bound-flip machinery elsewhere
            // in this loop. Must happen before the `rN(e_rel)` recompute
            // below, which reads `Ahat.col(oldAbs)` under the new view.
            if (leaving_target_view != leaving_current_view) {
                const double old_anchor = bound_anchor(leaving_current_view, oldAbs, l, u);
                view[oldAbs] = leaving_target_view;
                const double new_anchor = bound_anchor(view[oldAbs], oldAbs, l, u);
                const double delta_anchor = new_anchor - old_anchor;
                if (delta_anchor != 0.0) {
                    if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                        for (typename RevisedSimplex::SparseMatrix::InnerIterator it(A, oldAbs); it;
                             ++it) {
                            rhs_eff(it.row()) -= it.value() * delta_anchor;
                        }
                    } else {
                        rhs_eff.noalias() -= A.col(oldAbs) * delta_anchor;
                    }
                }
                // view_sign(Lower)=+1, view_sign(Upper)=-1: always flips here.
                if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                    for (typename RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, oldAbs); it;
                         ++it) {
                        it.valueRef() = -it.valueRef();
                    }
                    scale_rowwise_column(*Ahat_row, Ahat, oldAbs, -1.0);
                } else {
                    Ahat.col(oldAbs) = -Ahat.col(oldAbs);
                }
                chat(oldAbs) = -chat(oldAbs);
            } else {
                view[oldAbs] = leaving_current_view;
            }
            work.reduced_cost(work.entering_rel) = chat(oldAbs) - column_dot(Ahat, oldAbs, ydual);

            // NLA framework switch check — rebuild if Devex weight errors accumulate
            if (nla->needs_framework_rebuild()) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " framework rebuild triggered");
                refactor_basis();
                nla->clear_framework_rebuild();
                yB_cache_valid = false;
                if (auto failed = rebuild_dual_pool(
                        "dual pricing rebuild failed after framework rebuild", iters)) {
                    return *failed;
                }
                continue;
            }

            bool backtracked_this_iter = false;
            // FTRAN-DSE: tau = B^{-1} psi_r must come from the PRE-pivot
            // factorization (HiGHS updateFtranDSE runs before updateFactor),
            // so solve it before update_basis below. On failure the pricer
            // falls back to its Devex-style weight bound.
            std::optional<Eigen::VectorXd> ftran_dse;
            if (dual_pricer.wants_ftran_dse()) {
                try {
                    ftran_dse.emplace(
                        read_basis()
                            .solve_B(Eigen::VectorXd(work.pivot_row.value), FTBasis::TranKind::ColAq)
                            .value);
                } catch (...) {
                    ftran_dse.reset();
                }
            }
            try {
                update_basis(work.leaving_row, work.entering_col, Ahat.col(work.entering_col));
            } catch (...) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                std::vector<int> restored_basis;
                if (backtrack_repairs == 0 &&
                    write_basis().try_backtrack_to_last_good(restored_basis)) {
                    ++backtrack_repairs;
                    self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                     " backtracked to last full-rank basis");
                    basis = std::move(restored_basis);

                    std::vector<char> in_basis(static_cast<size_t>(n), 0);
                    for (int b_var : basis)
                        in_basis[static_cast<size_t>(b_var)] = 1;
                    N.clear();
                    N.reserve(static_cast<size_t>(n - m));
                    for (int j = 0; j < n; ++j)
                        if (!in_basis[static_cast<size_t>(j)])
                            N.push_back(j);

                    ydual_cached = false;
                    yB_cache_valid = false; // basis and rhs_eff both changed
                    rhs_eff = b - transformed_rhs(A, view, l, u);
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after backtrack", iters)) {
                        return *failed;
                    }
                    backtracked_this_iter = true;
                } else {
                    refactor_basis();
                    yB_cache_valid = false;
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after replace_column", iters)) {
                        return *failed;
                    }
                }
            }

            if (backtracked_this_iter)
                continue; // restored basis changes N/ydual state; restart from optimality check

            // ── Incremental yB rank-1 update ─────────────────────────────────
            // Formula (derived via Sherman-Morrison-Woodbury):
            //   yB_new[i] = yB_old[i] - tau_r * s_enter[i]    (i != r_leave)
            //   yB_new[r_leave] = tau_r
            // where tau_r = yB_old[r_leave] / pivot, pivot = s_enter[r_leave].
            // s_enter was B_old^{-1} a_e (computed before replace_column).
            // This SMW shortcut assumes `rhs_eff` did NOT change this
            // iteration (only the basis did) — true when the leaving var
            // exits to Lower (leaving_sign>0). When it exits to Upper
            // (leaving_sign<0) the anchor re-shift above already mutated
            // `rhs_eff`, invalidating the premise; fall back to a full
            // solve_B next time yB is needed instead of patching the
            // formula for a case it wasn't derived for.
            if (!backtracked_this_iter && yB_cache_valid && work.leaving_sign > 0) {
                const double yb_pivot = work.pivot_col(work.leaving_row);
                if (std::abs(yb_pivot) > 1e-14 && yB_cache_age < yB_max_age) {
                    const double tau_r = yB_cache(work.leaving_row) / yb_pivot;
                    yB_cache.noalias() -= tau_r * work.pivot_col.value;
                    yB_cache(work.leaving_row) = tau_r; // override (the -= above gives 0 here)
                    ++yB_cache_age;
                } else {
                    yB_cache_valid = false; // pivot too small or cache aged out
                }
            } else if (!backtracked_this_iter) {
                yB_cache_valid = false; // rhs_eff changed (leaving var re-anchored to Upper)
            }

            {
                // Both exit directions use the incremental weight update:
                // leaving_sign folds the pivot-row sign flip of the folded
                // (above-upper) view into the DSE cross term, so no pool
                // rebuild is needed on upper-bound pivots.
                dual_pricer.update_after_dual_pivot(
                    work.leaving_row, work.entering_col, oldAbs, work.pivot_col,
                    work.pivot_col(work.leaving_row), Ahat, N, work.pivot_row, true,
                    ftran_dse ? &*ftran_dse : nullptr, work.leaving_sign);
                // NLA Devex framework switch — pass weight error from pricer
                if (dual_pricer.needs_rebuild() && nla->allow_framework_switch()) {
                    double w_err = dual_pricer.average_log_weight_error();
                    nla->record_framework_error(w_err);
                }
                if (dual_pricer.needs_rebuild()) {
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after pivot update", iters)) {
                        return *failed;
                    }
                    dual_pricer.clear_rebuild_flag();
                }
            }
            if (self.should_trace_iter_(iters) && self.opt_.verbose_include_basis) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
            }

            // ── PAMI sub-iterations ───────────────────────────────────────────
            // After committing the first pivot of this outer iteration, attempt
            // up to (dual_pami_rows - 1) additional pivots using the updated
            // yB cache. Each sub-pivot is fully sequential (basis updates cannot
            // be parallelized); the throughput gain comes from amortizing the
            // per-outer-iteration overhead (pricing pool setup, cache priming)
            // across multiple pivots.
            for (int pami_k = 1; pami_k < self.opt_.dual_pami_rows && iters < self.opt_.max_iters;
                 ++pami_k) {
                // Need a fresh primal solution to identify most-infeasible row.
                if (!yB_cache_valid || yB_cache_age >= yB_max_age)
                    refresh_yB_cache();
                const Eigen::VectorXd& yB_sub = yB_cache;

                // PAMI sub-pivots only handle the below-lower case (matching
                // the raw, unflipped choose_dual_leaving/update_after_dual_pivot
                // calls below). If any row is above its range — whether from
                // before this batch started or as a side effect of an earlier
                // sub-pivot in it — bail out to the outer loop, which runs the
                // full two-sided leaving-row logic.
                if (!basic_above_range_rows(yB_sub).empty())
                    break;

                const auto sub_leaving =
                    dual_pricer.choose_dual_leaving(read_basis(), yB_sub, self.opt_.tol);
                if (sub_leaving.row < 0)
                    break; // optimal — let the outer loop detect it cleanly

                const int sub_r = sub_leaving.row;
                HVector sub_w = sub_leaving.dual_row;

                // PRICE: compute pN_sub / rN_sub for the sub-leaving row
                Eigen::VectorXd sub_pN, sub_rN;
                if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                    const double sub_density = vector_density(sub_w, self.opt_.tol);
                    const bool use_col_price =
                        self.opt_.dual_pricing == "col" ||
                        (self.opt_.dual_pricing == "switch" && sub_density > 0.75);
                    if (use_col_price)
                        compute_pricing_products_by_column(Ahat, N, sub_w, ydual, chat, sub_pN,
                                                           sub_rN);
                    else
                        compute_pricing_products(Ahat, *Ahat_row, N, sub_w, ydual, chat, sub_pN,
                                                 sub_rN);
                } else {
                    compute_pricing_products(Ahat, N, sub_w, ydual, chat, sub_pN, sub_rN);
                }

                const DualBFRTDecision sub_bfrt = dual_bfrt_decide(
                    self.opt_, sub_rN, sub_pN, N, view, l, u, -yB_sub(sub_r),
                    self.opt_.dual_allow_bound_flip ? adaptive_flip_budget : 0,
                    read_basis().update_count());
                if (!sub_bfrt.pivot_rel || !sub_bfrt.flip_rels.empty())
                    break; // bound flip or no pivot: fall back to outer loop

                const int sub_e_rel = *sub_bfrt.pivot_rel;
                const int sub_eAbs = N[sub_e_rel];
                const double sub_tau = sub_bfrt.tau;
                if (!std::isfinite(sub_tau))
                    break; // infeasibility detected — outer loop handles it

                HVector sub_s;
                try {
                    sub_s = read_basis().solve_B(Ahat.col(sub_eAbs), FTBasis::TranKind::ColAq);
                    nla->update_ema_reach(sub_s.count, m);
                } catch (...) {
                    break; // numerical issue; let outer loop deal with it
                }

                ++iters;
                const int sub_oldAbs = basis[sub_r];

                // Apply the sub-pivot dual update
                {
                    HVector sub_z = read_basis().solve_BT_unit(sub_r, FTBasis::TranKind::RowEp);
                    nla->update_ema_reach(sub_z.count, m);
                    const double sub_pivot = sub_s(sub_r);
                    const double sub_alpha = sub_rN(sub_e_rel) / sub_pivot;
                    ydual.noalias() += sub_alpha * sub_z.value;
                    for (int k = 0; k < static_cast<int>(N.size()); ++k) {
                        if (k == sub_e_rel)
                            continue;
                        sub_rN(k) -= sub_alpha * column_dot(Ahat, N[k], sub_z);
                    }
                }

                basis[sub_r] = sub_eAbs;
                N[sub_e_rel] = sub_oldAbs;
                ydual_cached = false;

                try {
                    update_basis(sub_r, sub_eAbs, Ahat.col(sub_eAbs));
                } catch (...) {
                    refactor_basis();
                    yB_cache_valid = false;
                    break; // stop PAMI sub-iters on numerical failure
                }

                // Update yB cache with rank-1 formula
                if (yB_cache_valid) {
                    const double yb_piv = sub_s(sub_r);
                    if (std::abs(yb_piv) > 1e-14 && yB_cache_age < yB_max_age) {
                        const double tau_r = yB_cache(sub_r) / yb_piv;
                        yB_cache.noalias() -= tau_r * sub_s.value;
                        yB_cache(sub_r) = tau_r;
                        ++yB_cache_age;
                    } else {
                        yB_cache_valid = false;
                    }
                }

                dual_pricer.update_after_dual_pivot(sub_r, sub_eAbs, sub_oldAbs, sub_s,
                                                    sub_s(sub_r), Ahat, N, sub_w, true);
                if (dual_pricer.needs_rebuild()) {
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after PAMI pivot", iters)) {
                        return *failed;
                    }
                    dual_pricer.clear_rebuild_flag();
                    break; // pricing state changed; restart from outer loop
                }
            }

            // HiGHS-style objective-bound bailout: dual phase 2 obj is monotone
            // non-decreasing for a min problem. Check periodically; if we have
            // already crossed the bound the node can be pruned without solving
            // to optimality.
            if (std::isfinite(self.opt_.objective_bound_internal) &&
                self.opt_.objective_bound_check_freq > 0 &&
                (iters % self.opt_.objective_bound_check_freq) == 0) {
                if (!yB_cache_valid || yB_cache_age >= yB_max_age)
                    refresh_yB_cache();
                Eigen::VectorXd x_check =
                    assemble_transformed_primal(n, basis, yB_cache.cwiseMax(0.0), l, u, view);
                const double obj_check = c.dot(x_check);
                if (obj_check > self.opt_.objective_bound_internal) {
                    auto info_map = dm_stats_to_map(self.degen_.get_stats());
                    attach_dual_pricing_info(info_map);
                    info_map["objective_bound_bailout"] = "1";
                    info_map["objective_bound_bailout_obj"] = std::to_string(obj_check);
                    self.trace_line_(
                        "[dual] objective-bound bailout iter=" + std::to_string(iters) +
                        " obj=" + std::to_string(obj_check) +
                        " bound=" + std::to_string(self.opt_.objective_bound_internal));
                    self.remember_warm_state_(basis, nla);
                    return {LPSolution::Status::ObjectiveBound, std::move(x_check), basis, iters,
                            std::move(info_map)};
                }
            }
        }

        auto info_map = dm_stats_to_map(self.degen_.get_stats());
        attach_dual_pricing_info(info_map);
        self.trace_line_("[dual] iterlimit basis=" + self.format_basis_(basis));
        self.remember_warm_state_(basis, nla);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis, iters,
                std::move(info_map)};
    }
};
