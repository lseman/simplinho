#pragma once

class RevisedSimplexPrimalEngine {
   public:
    struct BFRTStep {
        double theta_e = std::numeric_limits<double>::infinity();
        bool to_upper = false;
    };

    static std::pair<std::optional<int>, double> harris_ratio(
        const Eigen::VectorXd& xB, const Eigen::VectorXd& dB, double delta,
        double eta) {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i)
            if (dB(i) > delta) pos.push_back(i);
        if (pos.empty())
            return {std::nullopt, std::numeric_limits<double>::infinity()};

        double theta_star = std::numeric_limits<double>::infinity();
        for (int idx : pos)
            theta_star = std::min(theta_star, xB(idx) / dB(idx));

        double max_resid = 0.0;
        std::vector<int> candidates;
        for (int idx : pos) {
            const double ratio = xB(idx) / dB(idx);
            if (std::abs(ratio - theta_star) <= 1e-10)
                candidates.push_back(idx);
            const double resid = xB(idx) - theta_star * dB(idx);
            max_resid = std::max(max_resid, std::max(0.0, resid));
        }
        if (!candidates.empty()) {
            int best = candidates.front();
            for (int idx : candidates)
                if (idx < best) best = idx;
            return {best, theta_star};
        }

        const double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;
        for (int idx : pos) {
            const double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa) eligible.push_back(idx);
        }
        if (!eligible.empty()) {
            int best = eligible.front();
            for (int idx : eligible)
                if (idx < best) best = idx;
            return {best, theta_star};
        }

        int best = pos.front();
        double best_ratio = xB(best) / dB(best);
        for (int i = 1; i < (int)pos.size(); ++i) {
            const int idx = pos[i];
            const double r = xB(idx) / dB(idx);
            if (r < best_ratio) {
                best_ratio = r;
                best = idx;
            }
        }
        return {best, best_ratio};
    }

    static BFRTStep entering_bound_step(double x_e, double l_e, double u_e,
                                        double rc_e, double tol) {
        BFRTStep out;
        if (rc_e < -tol) {
            if (std::isfinite(u_e)) {
                out.theta_e = std::max(0.0, u_e - x_e);
                out.to_upper = true;
            }
        } else if (rc_e > tol) {
            if (std::isfinite(l_e)) {
                out.theta_e = std::max(0.0, x_e - l_e);
                out.to_upper = false;
            }
        }
        return out;
    }

    static RevisedSimplex::PhaseResult run(
        RevisedSimplex& self, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        int iters = 0;

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
            auto maybe = self.find_initial_basis_(A, b, c, self.opt_);
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
                if (!inB[j]) N.push_back(j);
        }

        std::optional<FTBasis> Bopt;
        try {
            Bopt.emplace(A, basis, self.make_basis_options_());
        } catch (const std::exception& e) {
            return {LPSolution::Status::Singular,
                    Eigen::VectorXd::Zero(n),
                    basis,
                    0,
                    {{"where", "initial basis factorization failed"},
                     {"what", e.what()}}};
        }
        FTBasis& B = *Bopt;
        self.trace_line_("[primal] start basis=" + self.format_basis_(basis));

        if (self.opt_.pricing_rule == "adaptive") {
            AdaptivePricer::PricingOptions popts;
            popts.steepest_pool_max = 0;
            popts.steepest_reset_freq = self.opt_.adaptive_reset_freq;
            popts.devex_reset_freq = self.opt_.devex_reset;
            self.adaptive_pricer_ = AdaptivePricer(n, popts);
            self.adaptive_pricer_.build_pools(B, A, N);
            self.bridge_ =
                std::make_unique<DegeneracyPricerBridge<AdaptivePricer>>(
                    self.degen_, self.adaptive_pricer_);
        }

        int rebuild_attempts = 0;

        while (iters < self.opt_.max_iters) {
            ++iters;

            Eigen::VectorXd xB;
            try {
                xB = B.solve_B(b);
            } catch (...) {
                if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                    ++rebuild_attempts;
                    self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                     " refactor after solve_B failure");
                    B.refactor();
                    if (self.opt_.pricing_rule == "adaptive") {
                        self.adaptive_pricer_.build_pools(B, A, N);
                        self.adaptive_pricer_.clear_rebuild_flag();
                    }
                    continue;
                }
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"where", "solve(B,b) repair failed"}}};
            }

            if ((xB.array() < -self.opt_.tol).any()) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " negative basic vars, handing off to phase I");
                return {LPSolution::Status::NeedPhase1,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        iters,
                        {{"reason", "negative_basic_vars"}}};
            }
            xB = xB.cwiseMax(0.0);

            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i) cB(i) = c(basis[i]);

            Eigen::VectorXd y;
            try {
                y = B.solve_BT(cB);
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after solve_BT failure");
                B.refactor();
                y = B.solve_BT(cB);
                if (self.opt_.pricing_rule == "adaptive") {
                    self.adaptive_pricer_.build_pools(B, A, N);
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            Eigen::VectorXd rN(N.size());
            Eigen::VectorXd rN_select(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = c(j) - A.col(j).dot(y);
                rN_select(k) =
                    self.can_increase_from_lower_(j, l, u, self.opt_.tol)
                        ? rN(k)
                        : 0.0;
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
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l, u);
                    self.trace_line_("[primal] optimal iter=" +
                                     std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    return {LPSolution::Status::Optimal, self.clip_small_(x), basis,
                            iters, dm_stats_to_map(self.degen_.get_stats())};
                }
                e_rel = idx;
            } else {
                if (self.opt_.pricing_rule == "adaptive") {
                    Eigen::VectorXd xcur =
                        self.assemble_primal_(n, basis, xB, l, u);
                    const double current_obj = c.dot(xcur);
                    e_rel = self.bridge_->choose_entering(
                        rN_select, N, self.opt_.tol, iters, current_obj, B, A);
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
                    if (idx >= 0) e_rel = idx;
                }

                if (!e_rel) {
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l, u);
                    self.trace_line_("[primal] optimal iter=" +
                                     std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    return {LPSolution::Status::Optimal, self.clip_small_(x), basis,
                            iters, dm_stats_to_map(self.degen_.get_stats())};
                }
            }

            const int e = N[*e_rel];
            const auto aE = A.col(e);

            Eigen::VectorXd dB;
            try {
                dB = B.solve_B(aE);
            } catch (...) {
                B.refactor();
                dB = B.solve_B(aE);
                if (self.opt_.pricing_rule == "adaptive") {
                    self.adaptive_pricer_.build_pools(B, A, N);
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            auto [leave_rel_opt, theta_B] =
                harris_ratio(xB, dB, self.opt_.ratio_delta, self.opt_.ratio_eta);

            const int idxN = *e_rel;
            const double rc_e = rN(idxN);
            const double l_e = (e >= 0 && e < l.size()) ? l(e) : 0.0;
            const double u_e =
                (e >= 0 && e < u.size()) ? u(e) : presolve::inf();
            const double x_e = std::isfinite(l_e) ? l_e : 0.0;
            const BFRTStep bfrt =
                entering_bound_step(x_e, l_e, u_e, rc_e, self.opt_.tol);

            double step = std::min(theta_B, bfrt.theta_e);
            if (!std::isfinite(step)) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                self.trace_line_("[primal] unbounded iter=" +
                                 std::to_string(iters) +
                                 " entering=" + std::to_string(e));
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(self.degen_.get_stats())};
            }

            const bool flip_entering = (bfrt.theta_e + 1e-14 < theta_B);
            if (flip_entering) {
                dB = -dB;
                const_cast<Eigen::VectorXd&>(rN)(idxN) = -rc_e;
            }

            if (!leave_rel_opt) {
                Eigen::VectorXd x = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                self.trace_line_("[primal] unbounded iter=" +
                                 std::to_string(iters) +
                                 " entering=" + std::to_string(e) +
                                 " no leaving variable");
                return {LPSolution::Status::Unbounded, x, basis, iters,
                        dm_stats_to_map(self.degen_.get_stats())};
            }

            const int r = *leave_rel_opt;
            const double alpha = dB(r);
            const int oldAbs = basis[r];
            const int eAbs = e;

            const bool is_degenerate =
                self.degen_.detect_degeneracy(step, self.opt_.deg_step_tol);
            if (is_degenerate && self.degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    self.degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;
            } else {
                (void)self.degen_.reset_perturbation();
            }

            if (self.opt_.pricing_rule == "adaptive") {
                const double rc_impr = -rN(idxN);
                self.bridge_->after_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N,
                                          rc_impr);
            }

            if (self.should_trace_iter_(iters)) {
                const Eigen::VectorXd xcur =
                    self.assemble_primal_(n, basis, xB, l, u);
                std::ostringstream oss;
                oss << "[primal] iter=" << iters << " obj=" << c.dot(xcur)
                    << " enter=" << eAbs << " leave_row=" << r
                    << " leave_var=" << oldAbs << " step=" << step
                    << " alpha=" << alpha;
                if (self.opt_.verbose_include_basis) {
                    oss << " basis_before=" << self.format_basis_(basis);
                }
                self.trace_line_(oss.str());
            }

            basis[r] = eAbs;
            N[idxN] = oldAbs;

            try {
                B.replace_column(r, aE);
                B.refactor();
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                B.refactor();
                if (self.opt_.pricing_rule == "adaptive") {
                    self.adaptive_pricer_.build_pools(B, A, N);
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            if (self.should_trace_iter_(iters) &&
                self.opt_.verbose_include_basis) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
            }

            if (self.opt_.pricing_rule == "adaptive" &&
                self.adaptive_pricer_.needs_rebuild()) {
                self.adaptive_pricer_.build_pools(B, A, N);
                self.adaptive_pricer_.clear_rebuild_flag();
            }
        }

        self.trace_line_("[primal] iterlimit basis=" + self.format_basis_(basis));
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, dm_stats_to_map(self.degen_.get_stats())};
    }
};
