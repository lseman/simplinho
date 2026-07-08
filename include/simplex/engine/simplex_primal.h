#pragma once

#include "simplex/core/hvector.h"
#include "simplex/nla/simplex_nla.h"

// Bounded-variable primal revised simplex engine.
//
// Handles l <= x <= u natively: nonbasic variables rest at their lower or
// upper bound (tracked in `at_upper`), the ratio test blocks basic variables
// against both bounds, and an entering variable whose own opposite bound is
// the tightest block performs a bound flip (no basis change). Free nonbasic
// variables (both bounds infinite) rest at 0 and may move in either
// direction.
class RevisedSimplexPrimalEngine {
  public:
    struct RatioResult {
        std::optional<int> row;
        double theta = std::numeric_limits<double>::infinity();
        bool leaving_to_upper = false; // leaving basic exits at its upper bound
    };

    // Two-sided Harris-style ratio test. `sigma` is the movement direction of
    // the entering variable (+1 increasing, -1 decreasing); the basic values
    // move by -sigma*t*dB. A row blocks when its basic variable hits its
    // lower bound (sigma*dB > delta) or its finite upper bound
    // (sigma*dB < -delta). Among near-minimal ratios, prefers the largest
    // |pivot| for numerical stability.
    template <class RowRange>
    static RatioResult ratio_test_core_(const Eigen::VectorXd& xB, const Eigen::VectorXd& dB,
                                        double sigma, const RowRange& rows,
                                        const std::vector<int>& basis, const Eigen::VectorXd& l,
                                        const Eigen::VectorXd& u, double delta, double eta) {
        RatioResult out;
        double theta_star = std::numeric_limits<double>::infinity();

        auto row_ratio = [&](int i, double& ratio, bool& to_upper) -> bool {
            const double d = sigma * dB(i);
            const int j = basis[i];
            if (d > delta) {
                // basic decreases toward its lower bound
                const double lo = (j >= 0 && j < l.size()) ? l(j) : 0.0;
                if (!std::isfinite(lo))
                    return false;
                ratio = std::max(0.0, xB(i) - lo) / d;
                to_upper = false;
                return true;
            }
            if (d < -delta) {
                // basic increases toward its upper bound
                const double hi =
                    (j >= 0 && j < u.size()) ? u(j) : std::numeric_limits<double>::infinity();
                if (!std::isfinite(hi))
                    return false;
                ratio = std::max(0.0, hi - xB(i)) / (-d);
                to_upper = true;
                return true;
            }
            return false;
        };

        for (int i : rows) {
            double ratio;
            bool to_upper;
            if (row_ratio(i, ratio, to_upper))
                theta_star = std::min(theta_star, ratio);
        }
        if (!std::isfinite(theta_star))
            return out; // no blocking row

        const double kappa = std::max(eta, eta * theta_star);
        int best = -1;
        double best_pivot = 0.0;
        bool best_to_upper = false;
        for (int i : rows) {
            double ratio;
            bool to_upper;
            if (!row_ratio(i, ratio, to_upper))
                continue;
            if (ratio <= theta_star + kappa) {
                const double piv = std::abs(dB(i));
                if (piv > best_pivot) {
                    best_pivot = piv;
                    best = i;
                    best_to_upper = to_upper;
                }
            }
        }
        if (best < 0)
            return out;
        double theta;
        bool to_upper;
        row_ratio(best, theta, to_upper);
        out.row = best;
        out.theta = std::max(0.0, theta);
        out.leaving_to_upper = best_to_upper;
        return out;
    }

    struct AllRows {
        int m;
        struct It {
            int i;
            int operator*() const { return i; }
            It& operator++() {
                ++i;
                return *this;
            }
            bool operator!=(const It& o) const { return i != o.i; }
        };
        It begin() const { return {0}; }
        It end() const { return {m}; }
    };

    static RatioResult ratio_test(const Eigen::VectorXd& xB, const HVector& dB, double sigma,
                                  const std::vector<int>& basis, const Eigen::VectorXd& l,
                                  const Eigen::VectorXd& u, double delta, double eta) {
        if (dB.has_pattern()) {
            std::vector<int> rows(dB.index.begin(), dB.index.begin() + dB.count);
            return ratio_test_core_(xB, dB.value, sigma, rows, basis, l, u, delta, eta);
        }
        return ratio_test_core_(xB, dB.value, sigma, AllRows{static_cast<int>(dB.value.size())},
                                basis, l, u, delta, eta);
    }

    // Value a nonbasic variable rests at.
    static double nonbasic_value_(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                  bool upper) {
        const bool has_l = (j < l.size()) && std::isfinite(l(j));
        const bool has_u = (j < u.size()) && std::isfinite(u(j));
        if (upper && has_u)
            return u(j);
        if (has_l)
            return l(j);
        if (has_u)
            return u(j);
        return 0.0;
    }

    template <class MatrixType>
    static void axpy_col_(const MatrixType& A, int j, double coef, Eigen::VectorXd& target) {
        if (coef == 0.0)
            return;
        if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
            for (typename MatrixType::InnerIterator it(A, j); it; ++it)
                target(it.row()) += coef * it.value();
        } else {
            target.noalias() += coef * A.col(j);
        }
    }

    template <class MatrixType>
    static RevisedSimplex::PhaseResult run(RevisedSimplex& self, const MatrixType& A,
                                           const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                                           std::optional<std::vector<int>> basis_opt,
                                           const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                           std::optional<std::vector<LPBasisStatus>> warm_status =
                                               std::nullopt) {
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
                nla_cfg.price_strategy_ = simplex::nla::NLAConfig::PriceStrategy::RowSwitchColSwitch;
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

        while (iters < self.opt_.max_iters) {
            ++iters;

            Eigen::VectorXd xB;
            try {
                if (!xB_cache_valid || xB_cache_age >= xB_max_age)
                    refresh_xB_cache();
                xB = xB_cache;
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

            // Two-sided primal feasibility of the basic solution.
            {
                bool infeasible = false;
                for (int i = 0; i < m; ++i) {
                    const int j = basis[i];
                    const double lo = (j >= 0 && j < l_work.size() && std::isfinite(l_work(j)))
                                          ? l_work(j)
                                          : -std::numeric_limits<double>::infinity();
                    const double hi = (j >= 0 && j < u_work.size() && std::isfinite(u_work(j)))
                                          ? u_work(j)
                                          : std::numeric_limits<double>::infinity();
                    if (xB(i) < lo - self.opt_.tol || xB(i) > hi + self.opt_.tol) {
                        infeasible = true;
                        break;
                    }
                    // Clamp tolerance-level violations onto the bound.
                    if (xB(i) < lo)
                        xB(i) = lo;
                    else if (xB(i) > hi)
                        xB(i) = hi;
                }
                if (infeasible) {
                    self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                     " infeasible basic vars, handing off to phase I");
                    return {LPSolution::Status::NeedPhase1,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"reason", "negative_basic_vars"}}};
                }
            }

            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i)
                cB(i) = c_work(basis[i]);

            HVector y_hvec;
            try {
                y_hvec = read_basis().solve_BT(cB, FTBasis::TranKind::RowEp);
                nla->update_ema_reach(y_hvec.count, m);
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after solve_BT failure");
                refactor_basis();
                xB_cache_valid = false;
                y_hvec = read_basis().solve_BT(cB, FTBasis::TranKind::RowEp);
                nla->update_ema_reach(y_hvec.count, m);
                rebuild_pricing();
            }
            Eigen::VectorXd y = y_hvec;
            Eigen::VectorXd aTy = A.transpose() * y;
            Eigen::VectorXd rN(N.size());
            Eigen::VectorXd rN_select(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = c_work(j) - aTy(j);
                const bool has_l = (j < l_work.size()) && std::isfinite(l_work(j));
                const bool has_u = (j < u_work.size()) && std::isfinite(u_work(j));
                const bool fixed = has_l && has_u && (u_work(j) - l_work(j)) <= self.opt_.tol;
                if (fixed) {
                    rN_select(k) = 0.0; // fixed variables never enter
                } else if (at_upper[j]) {
                    // at upper: profitable to decrease when rc > 0
                    rN_select(k) = -rN(k);
                } else if (has_l) {
                    // at lower: profitable to increase when rc < 0
                    rN_select(k) = rN(k);
                } else {
                    // free at 0: profitable in either direction
                    rN_select(k) = -std::abs(rN(k));
                }
            }

            std::optional<int> e_rel;

            if (self.opt_.bland) {
                int idx = -1;
                for (int k = 0; k < (int)N.size(); ++k)
                    if (rN_select(k) < -self.opt_.tol) {
                        idx = k;
                        break;
                    }
                if (idx < 0) {
                    const auto sv = sigma_view();
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l_work, u_work, &sv);
                    self.trace_line_("[primal] optimal iter=" + std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    self.remember_warm_state_(basis, nla);
                    return {LPSolution::Status::Optimal, self.clip_small_(x), basis, iters,
                            dm_stats_to_map(self.degen_.get_stats())};
                }
                e_rel = idx;
            } else {
                if (self.opt_.pricing_rule == "adaptive") {
                    double current_obj = 0.0;
                    {
                        std::vector<char> inB(n, 0);
                        for (int i = 0; i < (int)basis.size(); ++i) {
                            const int j = basis[i];
                            if (j >= 0 && j < n) {
                                inB[j] = 1;
                                current_obj += c_work(j) * xB(i);
                            }
                        }
                        for (int j = 0; j < n; ++j) {
                            if (inB[j])
                                continue;
                            current_obj +=
                                c_work(j) * nonbasic_value_(j, l_work, u_work, at_upper[j]);
                        }
                    }
                    e_rel = self.bridge_->choose_primal_entering(rN_select, N, self.opt_.tol, iters,
                                                                 current_obj, read_basis(), A,
                                                                 self.opt_.partial_pricing);
                } else {
                    int idx = -1;
                    double best = 0.0;
                    for (int k = 0; k < (int)N.size(); ++k)
                        if (rN_select(k) < -self.opt_.tol) {
                            if (idx < 0 || rN_select(k) < best) {
                                best = rN_select(k);
                                idx = k;
                            }
                        }
                    if (idx >= 0)
                        e_rel = idx;
                }

                if (!e_rel) {
                    const auto sv = sigma_view();
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l_work, u_work, &sv);
                    self.trace_line_("[primal] optimal iter=" + std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    self.remember_warm_state_(basis, nla);
                    return {LPSolution::Status::Optimal, self.clip_small_(x), basis, iters,
                            dm_stats_to_map(self.degen_.get_stats())};
                }
            }

            const int idxN = *e_rel;
            const int e = N[idxN];
            const double rc_e = rN(idxN);
            // Movement direction of the entering variable.
            const double sigma = at_upper[e] ? -1.0 : ((rc_e > 0.0 && !(e < l_work.size() &&
                                                                        std::isfinite(l_work(e))))
                                                           ? -1.0
                                                           : 1.0);

            HVector dB;
            try {
                dB = read_basis().solve_B(A.col(e), FTBasis::TranKind::ColAq);
                nla->update_ema_reach(dB.count, m);
            } catch (...) {
                refactor_basis();
                xB_cache_valid = false;
                dB = read_basis().solve_B(A.col(e), FTBasis::TranKind::ColAq);
                nla->update_ema_reach(dB.count, m);
                rebuild_pricing();
            }

            const RatioResult rt = ratio_test(xB, dB, sigma, basis, l_work, u_work,
                                              self.opt_.ratio_delta, self.opt_.ratio_eta);

            // Step allowed by the entering variable's own opposite bound.
            const double l_e = (e < l_work.size()) ? l_work(e) : 0.0;
            const double u_e = (e < u_work.size()) ? u_work(e) : presolve::inf();
            const double range_e = (std::isfinite(l_e) && std::isfinite(u_e))
                                       ? std::max(0.0, u_e - l_e)
                                       : std::numeric_limits<double>::infinity();

            const double step = std::min(rt.theta, range_e);
            if (!std::isfinite(step)) {
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray = unbounded_ray(e, sigma, dB.value);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) +
                                 " entering=" + std::to_string(e));
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
                const double old_val = nonbasic_value_(e, l_work, u_work, at_upper[e]);
                at_upper[e] = at_upper[e] ? 0 : 1;
                const double new_val = nonbasic_value_(e, l_work, u_work, at_upper[e]);
                const double delta_x = new_val - old_val;
                if (delta_x != 0.0) {
                    axpy_col_(A, e, -delta_x, rhs_eff);
                    if (xB_cache_valid && xB_cache_age < xB_max_age) {
                        xB_cache.noalias() -= delta_x * dB.value;
                        ++xB_cache_age;
                    } else {
                        xB_cache_valid = false;
                    }
                }
                (void)self.degen_.detect_degeneracy(step, self.opt_.deg_step_tol);
                if (self.should_trace_iter_(iters)) {
                    self.trace_line_("[primal] iter=" + std::to_string(iters) + " bound flip var=" +
                                     std::to_string(e) + " step=" + std::to_string(step));
                }
                continue;
            }

            if (!rt.row) {
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray = unbounded_ray(e, sigma, dB.value);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) +
                                 " entering=" + std::to_string(e) + " no leaving variable");
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
            const int eAbs = e;
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
                        xB, basis, l_work, u_work, self.rng_, self.opt_.tol,
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
                const double rc_impr = -rN_select(idxN);
                self.bridge_->after_primal_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N, rc_impr);
            }

            // New value of the entering variable and exit value of the leaver.
            const double enter_old_val = nonbasic_value_(e, l_work, u_work, at_upper[e]);
            const double enter_new_val = enter_old_val + sigma * step;
            const double leave_exit_val =
                nonbasic_value_(oldAbs, l_work, u_work, rt.leaving_to_upper);

            if (self.should_trace_iter_(iters)) {
                const auto sv = sigma_view();
                const Eigen::VectorXd xcur =
                    self.assemble_primal_(n, basis, xB, l_work, u_work, &sv);
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
                xB_cache.noalias() -= (sigma * step) * dB.value;
                xB_cache(r) = enter_new_val;
                ++xB_cache_age;
            } else {
                xB_cache_valid = false;
            }

            try {
                update_basis(r, eAbs, A.col(e));
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
