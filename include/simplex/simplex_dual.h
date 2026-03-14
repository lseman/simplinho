#pragma once

class RevisedSimplexDualEngine {
   public:
    enum class BoundView { Lower, Upper, Fixed };

    struct DualChoose {
        std::optional<int> e_rel;
        double tau = std::numeric_limits<double>::infinity();
    };

    struct DualBFRTDecision {
        std::optional<int> pivot_rel;
        double tau = std::numeric_limits<double>::infinity();
        std::vector<int> flip_rels;
    };

    static BoundView default_bound_view(int j, const Eigen::VectorXd& l,
                                        const Eigen::VectorXd& u) {
        const bool has_l = (j < l.size()) && std::isfinite(l(j));
        const bool has_u = (j < u.size()) && std::isfinite(u(j));
        if (has_l && has_u && std::abs(u(j) - l(j)) <= 1e-12) {
            return BoundView::Fixed;
        }
        if (has_u && !has_l) return BoundView::Upper;
        return BoundView::Lower;
    }

    static double bound_anchor(BoundView view, int j, const Eigen::VectorXd& l,
                               const Eigen::VectorXd& u) {
        switch (view) {
            case BoundView::Upper:
                return (j < u.size() && std::isfinite(u(j))) ? u(j) : 0.0;
            case BoundView::Fixed:
            case BoundView::Lower:
            default:
                return (j < l.size() && std::isfinite(l(j))) ? l(j) : 0.0;
        }
    }

    static int view_sign(BoundView view) {
        return (view == BoundView::Upper) ? -1 : 1;
    }

    static double bound_range(int j, const Eigen::VectorXd& l,
                              const Eigen::VectorXd& u) {
        if (j >= l.size() || j >= u.size() || !std::isfinite(l(j)) ||
            !std::isfinite(u(j))) {
            return std::numeric_limits<double>::infinity();
        }
        return std::max(0.0, u(j) - l(j));
    }

    static Eigen::VectorXd transformed_rhs(const Eigen::MatrixXd& A,
                                           const std::vector<BoundView>& view,
                                           const Eigen::VectorXd& l,
                                           const Eigen::VectorXd& u) {
        Eigen::VectorXd rhs = A.rows() ? Eigen::VectorXd::Zero(A.rows())
                                       : Eigen::VectorXd{};
        for (int j = 0; j < A.cols(); ++j) {
            const double anchor = bound_anchor(view[j], j, l, u);
            if (anchor != 0.0) rhs.noalias() += A.col(j) * anchor;
        }
        return rhs;
    }

    static Eigen::VectorXd assemble_transformed_primal(
        int n, const std::vector<int>& basis, const Eigen::VectorXd& yB,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        const std::vector<BoundView>& view) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j < 0 || j >= n) continue;
            inB[j] = 1;
            const double anchor = bound_anchor(view[j], j, l, u);
            if (view_sign(view[j]) > 0) {
                x(j) = anchor + ((i < yB.size()) ? yB(i) : 0.0);
            } else {
                x(j) = anchor - ((i < yB.size()) ? yB(i) : 0.0);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (!inB[j]) x(j) = bound_anchor(view[j], j, l, u);
        }
        return RevisedSimplex::clip_small_(x);
    }

    static DualChoose dual_harris_choose(const Eigen::VectorXd& rN,
                                         const Eigen::VectorXd& pN, double delta,
                                         double eta) {
        std::vector<int> E;
        E.reserve((int)pN.size());
        for (int k = 0; k < pN.size(); ++k)
            if (pN(k) < -delta) E.push_back(k);
        if (E.empty()) return {};

        double tau_star = std::numeric_limits<double>::infinity();
        for (int k : E) tau_star = std::min(tau_star, rN(k) / (-pN(k)));

        const double kappa = std::max(eta, eta * std::abs(tau_star));
        std::vector<int> candidates;
        for (int k : E) {
            if ((rN(k) / (-pN(k))) <= tau_star + kappa) candidates.push_back(k);
        }
        if (!candidates.empty()) {
            int best = candidates.front();
            double best_ratio = rN(best) / (-pN(best));
            for (int kk : candidates) {
                const double val = rN(kk) / (-pN(kk));
                if ((val < best_ratio - 1e-16) ||
                    (std::abs(val - best_ratio) <= 1e-16 && kk < best)) {
                    best = kk;
                    best_ratio = val;
                }
            }
            return {best, std::max(0.0, best_ratio)};
        }

        int best = E.front();
        double best_ratio = rN(best) / (-pN(best));
        for (int i = 1; i < (int)E.size(); ++i) {
            const int k = E[i];
            const double val = rN(k) / (-pN(k));
            if (val < best_ratio) {
                best_ratio = val;
                best = k;
            }
        }
        return {best, std::max(0.0, best_ratio)};
    }

    static DualBFRTDecision dual_bfrt_decide(
        const RevisedSimplex& self, const Eigen::VectorXd& rN,
        const Eigen::VectorXd& pN, const std::vector<int>& N,
        const std::vector<BoundView>& view, const Eigen::VectorXd& l,
        const Eigen::VectorXd& u, int max_flips) {
        DualBFRTDecision out;
        DualChoose dc = dual_harris_choose(
            rN, pN, self.opt_.ratio_delta, self.opt_.ratio_eta);
        out.pivot_rel = dc.e_rel;
        out.tau = dc.tau;
        if (!dc.e_rel || !std::isfinite(dc.tau) || max_flips <= 0) return out;

        struct Event {
            double tau;
            int rel;
        };
        std::vector<Event> events;
        events.reserve(N.size());
        const double tau_cap =
            dc.tau + std::max(self.opt_.ratio_eta, 1e-12 * (1.0 + dc.tau));

        for (int k = 0; k < (int)N.size(); ++k) {
            if (k == *dc.e_rel) continue;
            if (!(pN(k) < -self.opt_.ratio_delta)) continue;

            const int j = N[k];
            const double range = bound_range(j, l, u);
            if (!std::isfinite(range) || range <= self.opt_.tol) continue;
            if (view[j] == BoundView::Fixed) continue;

            const double tau_k = rN(k) / (-pN(k));
            if (!std::isfinite(tau_k) || tau_k < 0.0 || tau_k > tau_cap) {
                continue;
            }
            events.push_back({tau_k, k});
        }

        std::sort(events.begin(), events.end(), [](const Event& a,
                                                   const Event& b) {
            if (std::abs(a.tau - b.tau) > 1e-16) return a.tau < b.tau;
            return a.rel < b.rel;
        });

        for (int i = 0; i < (int)events.size() && i < max_flips; ++i) {
            out.flip_rels.push_back(events[i].rel);
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

        std::vector<BoundView> view(n, BoundView::Lower);
        for (int j = 0; j < n; ++j) view[j] = default_bound_view(j, l, u);
        self.bridge_.reset();
        DualAdaptivePricer dual_pricer(self.opt_.pricing_rule,
                                       self.opt_.devex_reset,
                                       self.opt_.adaptive_reset_freq);

        Eigen::MatrixXd Ahat = A;
        Eigen::VectorXd chat = c;
        for (int j : basis) {
            if (j >= 0 && j < n) view[j] = BoundView::Lower;
        }

        std::optional<FTBasis> Bopt;
        try {
            Bopt.emplace(Ahat, basis, self.make_basis_options_());
        } catch (const std::exception& e) {
            return {LPSolution::Status::Singular,
                    Eigen::VectorXd::Zero(n),
                    basis,
                    0,
                    {{"where", "dual initial basis factorization failed"},
                     {"what", e.what()}}};
        }
        FTBasis& B = *Bopt;

        auto apply_views_to_nonbasics = [&](const Eigen::VectorXd& ydual) {
            bool changed = false;
            std::vector<char> inB(n, 0);
            for (int j : basis)
                if (j >= 0 && j < n) inB[j] = 1;

            for (int j = 0; j < n; ++j) {
                if (inB[j]) continue;

                const double raw_rc = c(j) - A.col(j).dot(ydual);
                const bool has_l = (j < l.size()) && std::isfinite(l(j));
                const bool has_u = (j < u.size()) && std::isfinite(u(j));
                BoundView next = view[j];

                if (has_l && has_u) {
                    if (std::abs(u(j) - l(j)) <= self.opt_.tol) {
                        next = BoundView::Fixed;
                    } else {
                        next =
                            (raw_rc < 0.0) ? BoundView::Upper : BoundView::Lower;
                    }
                } else if (has_u && !has_l) {
                    next = BoundView::Upper;
                } else {
                    next = BoundView::Lower;
                }

                if (next != view[j]) {
                    view[j] = next;
                    changed = true;
                }
            }

            if (changed) {
                for (int j = 0; j < n; ++j) {
                    const double sign = static_cast<double>(view_sign(view[j]));
                    if (sign > 0.0) {
                        Ahat.col(j) = A.col(j);
                        chat(j) = c(j);
                    } else {
                        Ahat.col(j) = -A.col(j);
                        chat(j) = -c(j);
                    }
                }
            }
            return changed;
        };

        {
            Eigen::VectorXd cB(m);
            for (int i = 0; i < m; ++i) cB(i) = chat(basis[i]);
            Eigen::VectorXd ydual = B.solve_BT(cB);
            apply_views_to_nonbasics(ydual);
        }

        for (int j : basis) {
            if (view_sign(view[j]) < 0) {
                Ahat.col(j) = -A.col(j);
                chat(j) = -c(j);
            }
        }
        B.refactor();
        dual_pricer.build_pool(B, Ahat, N);
        self.trace_line_("[dual] start basis=" + self.format_basis_(basis));

        int rebuild_attempts = 0;
        int total_flips = 0;

        auto serialize_vec = [](const Eigen::VectorXd& v) {
            std::ostringstream oss;
            oss.setf(std::ios::scientific);
            oss << std::setprecision(17);
            for (int i = 0; i < v.size(); ++i) {
                if (i) oss << ",";
                oss << v(i);
            }
            return oss.str();
        };

        while (iters < self.opt_.max_iters) {
            ++iters;
            int flips_this_iter = 0;
            Eigen::VectorXd rhs_eff = b - transformed_rhs(A, view, l, u);
            Eigen::VectorXd yB;
            Eigen::VectorXd cB(m);
            Eigen::VectorXd ydual;
            Eigen::VectorXd pN;
            Eigen::VectorXd rN;
            int r_leave = -1;
            Eigen::VectorXd w;
            int e_rel = -1;
            int eAbs = -1;
            Eigen::VectorXd s_enter;
            double tau = std::numeric_limits<double>::infinity();

            while (true) {
                try {
                    yB = B.solve_B(rhs_eff);
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve_B failure");
                        B.refactor();
                        dual_pricer.build_pool(B, Ahat, N);
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where", "dual: solve(Bhat,rhs) repair failed"}}};
                }

                for (int i = 0; i < m; ++i) cB(i) = chat(basis[i]);
                try {
                    ydual = B.solve_BT(cB);
                } catch (...) {
                    self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                     " refactor after solve_BT failure");
                    B.refactor();
                    dual_pricer.build_pool(B, Ahat, N);
                    ydual = B.solve_BT(cB);
                }

                if (apply_views_to_nonbasics(ydual)) {
                    rhs_eff = b - transformed_rhs(A, view, l, u);
                    dual_pricer.build_pool(B, Ahat, N);
                    continue;
                }

                const auto leaving =
                    dual_pricer.choose_leaving(B, yB, self.opt_.tol);
                r_leave = leaving.row;
                if (r_leave < 0) {
                    rN.resize(N.size());
                    bool dual_feasible = true;
                    for (int k = 0; k < (int)N.size(); ++k) {
                        const int j = N[k];
                        rN(k) = chat(j) - Ahat.col(j).dot(ydual);
                        if (rN(k) < -self.opt_.tol) dual_feasible = false;
                    }
                    if (dual_feasible) {
                        Eigen::VectorXd x = assemble_transformed_primal(
                            n, basis, yB.cwiseMax(0.0), l, u, view);
                        auto info_map = dm_stats_to_map(self.degen_.get_stats());
                        info_map["dual_pricing"] =
                            dual_pricer.current_strategy_name();
                        info_map["dual_bfrt_flips"] =
                            std::to_string(total_flips);
                        self.trace_line_("[dual] optimal iter=" +
                                         std::to_string(iters) +
                                         " basis=" + self.format_basis_(basis));
                        return {LPSolution::Status::Optimal, std::move(x), basis,
                                iters, std::move(info_map)};
                    }
                    self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                     " primal-feasible but dual-infeasible");
                    return {LPSolution::Status::NeedPhase1,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"reason",
                              "dual_infeasible_at_primal_feasible"}}};
                }

                w = leaving.dual_row;
                pN.resize(N.size());
                rN.resize(N.size());
                for (int k = 0; k < (int)N.size(); ++k) {
                    const int j = N[k];
                    pN(k) = w.dot(Ahat.col(j));
                    rN(k) = chat(j) - Ahat.col(j).dot(ydual);
                }

                const DualBFRTDecision bfrt = dual_bfrt_decide(
                    self, rN, pN, N, view, l, u,
                    self.opt_.dual_allow_bound_flip
                        ? (self.opt_.dual_flip_max_per_iter - flips_this_iter)
                        : 0);
                if (!bfrt.pivot_rel) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_(
                            "[dual] iter=" + std::to_string(iters) +
                            " refactor after no eligible entering");
                        B.refactor();
                        dual_pricer.build_pool(B, Ahat, N);
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
                        oss << "[dual] iter=" << iters
                            << " bound flips=" << bfrt.flip_rels.size();
                        if (self.opt_.verbose_include_basis) {
                            oss << " basis=" << self.format_basis_(basis);
                        }
                        self.trace_line_(oss.str());
                    }
                    for (int rel_k : bfrt.flip_rels) {
                        const int j = N[rel_k];
                        const double old_anchor = bound_anchor(view[j], j, l, u);
                        view[j] = (view[j] == BoundView::Upper)
                                      ? BoundView::Lower
                                      : BoundView::Upper;
                        const double new_anchor = bound_anchor(view[j], j, l, u);
                        const double delta_anchor = new_anchor - old_anchor;
                        if (delta_anchor != 0.0) {
                            rhs_eff.noalias() -= A.col(j) * delta_anchor;
                        }
                        Ahat.col(j) = -Ahat.col(j);
                        chat(j) = -chat(j);
                        ++flips_this_iter;
                        ++total_flips;
                    }
                    dual_pricer.build_pool(B, Ahat, N);
                    continue;
                }

                e_rel = *bfrt.pivot_rel;
                eAbs = N[e_rel];
                tau = bfrt.tau;
                try {
                    s_enter = B.solve_B(Ahat.col(eAbs));
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve(B,a_e) failure");
                        B.refactor();
                        dual_pricer.build_pool(B, Ahat, N);
                        continue;
                    }
                    return {LPSolution::Status::Singular,
                            Eigen::VectorXd::Zero(n),
                            basis,
                            iters,
                            {{"where",
                              "dual: solve(Bhat,a_e) repair failed"}}};
                }
                break;
            }

            if (!std::isfinite(tau)) {
                Eigen::VectorXd yF = w;
                if (yF.dot(rhs_eff) >= 0) yF = -yF;

                auto info_map = dm_stats_to_map(self.degen_.get_stats());
                info_map["where"] = "dual: infinite step";
                info_map["dual_pricing"] = dual_pricer.current_strategy_name();
                info_map["dual_bfrt_flips"] = std::to_string(total_flips);
                info_map["farkas_has_cert"] = "1";
                info_map["farkas_dim"] = std::to_string(m);
                info_map["farkas_y"] = serialize_vec(yF);
                self.trace_line_("[dual] infeasible iter=" +
                                 std::to_string(iters) +
                                 " produced Farkas certificate");

                return {LPSolution::Status::Infeasible,
                        Eigen::VectorXd::Zero(n), basis, iters,
                        std::move(info_map)};
            }

            const bool is_degenerate =
                self.degen_.detect_degeneracy(tau, self.opt_.deg_step_tol);
            if (is_degenerate && self.degen_.should_apply_perturbation()) {
                auto [Ap, bp, cp] =
                    self.degen_.apply_perturbation(A, b, c, basis, iters);
                (void)Ap;
                (void)bp;
                (void)cp;
            } else {
                (void)self.degen_.reset_perturbation();
            }

            const int oldAbs = basis[r_leave];
            if (self.should_trace_iter_(iters)) {
                Eigen::VectorXd xcur = assemble_transformed_primal(
                    n, basis, yB.cwiseMax(0.0), l, u, view);
                std::ostringstream oss;
                oss << "[dual] iter=" << iters << " obj=" << c.dot(xcur)
                    << " leave_row=" << r_leave << " leave_var=" << oldAbs
                    << " enter=" << eAbs << " tau=" << tau;
                if (self.opt_.verbose_include_basis) {
                    oss << " basis_before=" << self.format_basis_(basis);
                }
                self.trace_line_(oss.str());
            }
            basis[r_leave] = eAbs;
            N[e_rel] = oldAbs;

            try {
                B.replace_column(r_leave, Ahat.col(eAbs));
                B.refactor();
            } catch (...) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                B.refactor();
                dual_pricer.build_pool(B, Ahat, N);
            }

            dual_pricer.update_after_pivot(r_leave, eAbs, oldAbs, s_enter,
                                           s_enter(r_leave), Ahat, N, &w, true);
            if (dual_pricer.needs_rebuild()) {
                dual_pricer.build_pool(B, Ahat, N);
                dual_pricer.clear_rebuild_flag();
            }
            if (self.should_trace_iter_(iters) &&
                self.opt_.verbose_include_basis) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
            }
        }

        auto info_map = dm_stats_to_map(self.degen_.get_stats());
        info_map["dual_pricing"] = dual_pricer.current_strategy_name();
        info_map["dual_bfrt_flips"] = std::to_string(total_flips);
        self.trace_line_("[dual] iterlimit basis=" + self.format_basis_(basis));
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis,
                iters, std::move(info_map)};
    }
};
