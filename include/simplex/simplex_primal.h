#pragma once

#include "hvector.h"

class RevisedSimplexPrimalEngine {
  public:
    struct BFRTStep {
        double theta_e = std::numeric_limits<double>::infinity();
        bool to_upper = false;
    };

    // Core Harris ratio test. `candidate_rows` lists the row indices worth
    // examining (positive entries of dB above `delta`). When dB has a known
    // sparse pattern, the caller iterates only the pattern rows; otherwise
    // it scans all m rows. The arithmetic from the initial filter onward is
    // identical in both paths.
    static std::pair<std::optional<int>, double> harris_ratio_core_(const Eigen::VectorXd& xB,
                                                                    const Eigen::VectorXd& dB,
                                                                    const std::vector<int>& pos,
                                                                    double eta) {
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
                if (idx < best)
                    best = idx;
            return {best, theta_star};
        }

        const double kappa = std::max(eta, eta * max_resid);
        std::vector<int> eligible;
        for (int idx : pos) {
            const double resid = xB(idx) - theta_star * dB(idx);
            if (resid <= kappa)
                eligible.push_back(idx);
        }
        if (!eligible.empty()) {
            int best = eligible.front();
            for (int idx : eligible)
                if (idx < best)
                    best = idx;
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

    static std::pair<std::optional<int>, double>
    harris_ratio(const Eigen::VectorXd& xB, const Eigen::VectorXd& dB, double delta, double eta) {
        std::vector<int> pos;
        pos.reserve(dB.size());
        for (int i = 0; i < dB.size(); ++i)
            if (dB(i) > delta)
                pos.push_back(i);
        return harris_ratio_core_(xB, dB, pos, eta);
    }

    // Sparse-pattern-aware variant: iterates only the rows where dB *may* be
    // nonzero (from the HVector index list), collapsing the O(m) initial scan
    // to O(nnz). Correctness is preserved because dB.value is still the
    // authoritative dense store and we always check `dB(i) > delta`.
    static std::pair<std::optional<int>, double>
    harris_ratio(const Eigen::VectorXd& xB, const HVector& dB, double delta, double eta) {
        std::vector<int> pos;
        if (dB.has_pattern()) {
            pos.reserve(static_cast<std::size_t>(dB.count));
            for (int k = 0; k < dB.count; ++k) {
                const int i = dB.index[k];
                if (dB.value(i) > delta)
                    pos.push_back(i);
            }
            return harris_ratio_core_(xB, dB.value, pos, eta);
        }
        return harris_ratio(xB, dB.value, delta, eta);
    }

    static BFRTStep entering_bound_step(double x_e, double l_e, double u_e, double rc_e,
                                        double tol) {
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

    template <class MatrixType>
    static RevisedSimplex::PhaseResult run(RevisedSimplex& self, const MatrixType& A,
                                           const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                                           std::optional<std::vector<int>> basis_opt,
                                           const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
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

        std::shared_ptr<FTBasis> basis_factorization;
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
        if (const auto warm_state = self.try_reuse_factorization_(basis)) {
            basis_factorization = warm_state->basis_factorization;
        } else {
            try {
                if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                    if (m <= 16) {
                        Eigen::MatrixXd B_dense(m, m);
                        for (int i = 0; i < m; ++i)
                            B_dense.col(i) = A.col(basis[i]);
                        std::vector<int> dense_basis(m);
                        std::iota(dense_basis.begin(), dense_basis.end(), 0);
                        basis_factorization = std::make_shared<FTBasis>(B_dense, dense_basis,
                                                                        self.make_basis_options_());
                    } else {
                        basis_factorization =
                            std::make_shared<FTBasis>(A, basis, self.make_basis_options_());
                    }
                } else {
                    basis_factorization =
                        std::make_shared<FTBasis>(A, basis, self.make_basis_options_());
                }
            } catch (const std::exception& e) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "initial basis factorization failed"}, {"what", e.what()}}};
            }
        }
        auto rebuild_basis_factorization = [&]() -> std::shared_ptr<FTBasis> {
            if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                if (m <= 16) {
                    Eigen::MatrixXd B_dense(m, m);
                    for (int i = 0; i < m; ++i)
                        B_dense.col(i) = A.col(basis[i]);
                    std::vector<int> dense_basis(m);
                    std::iota(dense_basis.begin(), dense_basis.end(), 0);
                    return std::make_shared<FTBasis>(B_dense, dense_basis,
                                                     self.make_basis_options_());
                }
                return std::make_shared<FTBasis>(A, basis, self.make_basis_options_());
            }
            return std::make_shared<FTBasis>(A, basis, self.make_basis_options_());
        };
        auto read_basis = [&]() -> FTBasis& { return *basis_factorization; };
        auto write_basis = [&]() -> FTBasis& {
            if (!basis_factorization.unique()) {
                basis_factorization = rebuild_basis_factorization();
            }
            return *basis_factorization;
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

        auto unbounded_ray = [&](int entering_abs, const Eigen::VectorXd& dB) {
            Eigen::VectorXd ray = Eigen::VectorXd::Zero(n);
            if (entering_abs >= 0 && entering_abs < n)
                ray(entering_abs) = 1.0;
            for (int i = 0; i < m && i < dB.size(); ++i) {
                const int j = basis[i];
                if (j >= 0 && j < n)
                    ray(j) = -dB(i);
            }
            return self.clip_small_(ray);
        };

        int rebuild_attempts = 0;

        // ── Incremental xB (primal basic solution) cache ─────────────────────
        // Avoids re-solving B·xB = b from scratch each iteration. After a
        // pivot at leaving row r with FTRAN column dB = B^{-1}a_e and step θ:
        //   xB_new[i] = xB_old[i] − θ·dB[i]    (i ≠ r)
        //   xB_new[r] = θ
        // Falls back to a full BTRAN after explicit refactors or every
        // refactor_every pivots.
        Eigen::VectorXd xB_cache;
        bool xB_cache_valid = false;
        int xB_cache_age = 0;
        const int xB_max_age = std::max(1, self.opt_.refactor_every);
        auto refresh_xB_cache = [&]() {
            xB_cache = read_basis().solve_B(b, FTBasis::TranKind::ColAq).value;
            xB_cache_valid = true;
            xB_cache_age = 0;
        };
        refresh_xB_cache(); // prime before the loop

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
                    write_basis().refactor();
                    xB_cache_valid = false;
                    if (self.opt_.pricing_rule == "adaptive") {
                        self.measure_pricing_build_(false, [&]() {
                            self.adaptive_pricer_.build_primal_pools(read_basis(), A, N);
                        });
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
            for (int i = 0; i < m; ++i)
                cB(i) = c_work(basis[i]);

            Eigen::VectorXd y;
            try {
                y = read_basis().solve_BT(cB, FTBasis::TranKind::RowEp);
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after solve_BT failure");
                write_basis().refactor();
                xB_cache_valid = false;
                y = read_basis().solve_BT(cB, FTBasis::TranKind::RowEp);
                if (self.opt_.pricing_rule == "adaptive") {
                    self.measure_pricing_build_(false, [&]() {
                        self.adaptive_pricer_.build_primal_pools(read_basis(), A, N);
                    });
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            Eigen::VectorXd aTy = A.transpose() * y;
            Eigen::VectorXd rN(N.size());
            Eigen::VectorXd rN_select(N.size());
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                rN(k) = c_work(j) - aTy(j);
                rN_select(k) = self.can_increase_from_lower_(j, l_work, u_work, self.opt_.tol) ? rN(k) : 0.0;
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
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l_work, u_work);
                    self.trace_line_("[primal] optimal iter=" + std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    self.remember_warm_state_(basis, basis_factorization);
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
                            if (j < l_work.size() && std::isfinite(l_work(j))) {
                                current_obj += c_work(j) * l_work(j);
                            }
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
                    Eigen::VectorXd x = self.assemble_primal_(n, basis, xB, l_work, u_work);
                    self.trace_line_("[primal] optimal iter=" + std::to_string(iters) +
                                     " basis=" + self.format_basis_(basis));
                    self.remember_warm_state_(basis, basis_factorization);
                    return {LPSolution::Status::Optimal, self.clip_small_(x), basis, iters,
                            dm_stats_to_map(self.degen_.get_stats())};
                }
            }

            const int e = N[*e_rel];

            HVector dB;
            try {
                dB = read_basis().solve_B(A.col(e), FTBasis::TranKind::ColAq);
            } catch (...) {
                write_basis().refactor();
                xB_cache_valid = false;
                dB = read_basis().solve_B(A.col(e), FTBasis::TranKind::ColAq);
                if (self.opt_.pricing_rule == "adaptive") {
                    self.measure_pricing_build_(false, [&]() {
                        self.adaptive_pricer_.build_primal_pools(read_basis(), A, N);
                    });
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            auto [leave_rel_opt, theta_B] =
                harris_ratio(xB, dB, self.opt_.ratio_delta, self.opt_.ratio_eta);

            const int idxN = *e_rel;
            const double rc_e = rN(idxN);
            const double l_e = (e >= 0 && e < l_work.size()) ? l_work(e) : 0.0;
            const double u_e = (e >= 0 && e < u_work.size()) ? u_work(e) : presolve::inf();
            const double x_e = std::isfinite(l_e) ? l_e : 0.0;
            const BFRTStep bfrt = entering_bound_step(x_e, l_e, u_e, rc_e, self.opt_.tol);

            double step = std::min(theta_B, bfrt.theta_e);
            if (!std::isfinite(step)) {
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray = unbounded_ray(e, dB);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) +
                                 " entering=" + std::to_string(e));
                auto info = dm_stats_to_map(self.degen_.get_stats());
                info["certificate"] = "primal_ray";
                info["primal_ray_has_cert"] = "1";
                info["primal_ray_dim"] = std::to_string(n);
                info["primal_ray"] = serialize_vec(ray);
                self.remember_warm_state_(basis, basis_factorization);
                return {LPSolution::Status::Unbounded, x, basis, iters, std::move(info)};
            }

            const bool flip_entering = (bfrt.theta_e + 1e-14 < theta_B);
            if (flip_entering) {
                dB.value = -dB.value; // pattern unchanged, just sign-flip
                const_cast<Eigen::VectorXd&>(rN)(idxN) = -rc_e;
            }

            if (!leave_rel_opt) {
                Eigen::VectorXd x =
                    Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
                Eigen::VectorXd ray = unbounded_ray(e, dB);
                self.trace_line_("[primal] unbounded iter=" + std::to_string(iters) +
                                 " entering=" + std::to_string(e) + " no leaving variable");
                auto info = dm_stats_to_map(self.degen_.get_stats());
                info["certificate"] = "primal_ray";
                info["primal_ray_has_cert"] = "1";
                info["primal_ray_dim"] = std::to_string(n);
                info["primal_ray"] = serialize_vec(ray);
                self.remember_warm_state_(basis, basis_factorization);
                return {LPSolution::Status::Unbounded, x, basis, iters, std::move(info)};
            }

            const int r = *leave_rel_opt;
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
                        xB, basis, l_work, u_work, self.rng_,
                        self.opt_.tol,
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
                const double rc_impr = -rN(idxN);
                self.bridge_->after_primal_pivot(r, eAbs, oldAbs, dB, alpha, step, A, N, rc_impr);
            }

            if (self.should_trace_iter_(iters)) {
                const Eigen::VectorXd xcur = self.assemble_primal_(n, basis, xB, l_work, u_work);
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

            // ── Incremental xB update (rank-1) ─────────────────────────────
            // xB_new = xB_old − step·dB;  xB_new[r] = step (override, non-flip only).
            // For flip pivots dB was negated, so xB[r] is already correct after the
            // noalias update; unconditionally overriding would corrupt it.
            if (xB_cache_valid && xB_cache_age < xB_max_age) {
                xB_cache.noalias() -= step * dB.value;
                if (!flip_entering)
                    xB_cache(r) = step;
                xB_cache = xB_cache.cwiseMax(0.0);
                ++xB_cache_age;
            } else {
                xB_cache_valid = false;
            }

            try {
                write_basis().replace_column(r, eAbs, A.col(e));
            } catch (...) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                write_basis().refactor();
                xB_cache_valid = false;
                if (self.opt_.pricing_rule == "adaptive") {
                    self.measure_pricing_build_(false, [&]() {
                        self.adaptive_pricer_.build_primal_pools(read_basis(), A, N);
                    });
                    self.adaptive_pricer_.clear_rebuild_flag();
                }
            }

            if (self.should_trace_iter_(iters) && self.opt_.verbose_include_basis) {
                self.trace_line_("[primal] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
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
        self.remember_warm_state_(basis, basis_factorization);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis, iters,
                dm_stats_to_map(self.degen_.get_stats())};
    }
};
