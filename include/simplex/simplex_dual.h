#pragma once

#include <type_traits>

class RevisedSimplexDualEngine {
  public:
    using SparseRowMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor, int>;
    enum class BoundView { Lower, Upper, Fixed };

    struct SparsePricingWorkspace {
        std::vector<int> rel_of_col;
        std::vector<unsigned int> mark_of_col;
        unsigned int stamp = 1;

        void prepare(int num_cols, const std::vector<int>& nonbasic_columns) {
            if (num_cols < 0) {
                num_cols = 0;
            }
            if (static_cast<int>(rel_of_col.size()) < num_cols) {
                rel_of_col.resize(static_cast<std::size_t>(num_cols), -1);
                mark_of_col.resize(static_cast<std::size_t>(num_cols), 0);
            }

            ++stamp;
            if (stamp == 0) {
                std::fill(mark_of_col.begin(), mark_of_col.end(), 0);
                stamp = 1;
            }

            for (int k = 0; k < static_cast<int>(nonbasic_columns.size()); ++k) {
                const int col = nonbasic_columns[k];
                if (col < 0 || col >= num_cols) {
                    continue;
                }
                rel_of_col[static_cast<std::size_t>(col)] = k;
                mark_of_col[static_cast<std::size_t>(col)] = stamp;
            }
        }

        [[nodiscard]] int lookup(int col) const {
            if (col < 0 || col >= static_cast<int>(mark_of_col.size())) {
                return -1;
            }
            return mark_of_col[static_cast<std::size_t>(col)] == stamp
                       ? rel_of_col[static_cast<std::size_t>(col)]
                       : -1;
        }
    };

    struct DualChoose {
        std::optional<int> e_rel;
        double tau = std::numeric_limits<double>::infinity();
    };

    struct DualBFRTDecision {
        std::optional<int> pivot_rel;
        double tau = std::numeric_limits<double>::infinity();
        std::vector<int> flip_rels;
    };

    static BoundView default_bound_view(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        const bool has_l = (j < l.size()) && std::isfinite(l(j));
        const bool has_u = (j < u.size()) && std::isfinite(u(j));
        if (has_l && has_u && std::abs(u(j) - l(j)) <= 1e-12) {
            return BoundView::Fixed;
        }
        if (has_u && !has_l)
            return BoundView::Upper;
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

    static int view_sign(BoundView view) { return (view == BoundView::Upper) ? -1 : 1; }

    static double bound_range(int j, const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        if (j >= l.size() || j >= u.size() || !std::isfinite(l(j)) || !std::isfinite(u(j))) {
            return std::numeric_limits<double>::infinity();
        }
        return std::max(0.0, u(j) - l(j));
    }

    static Eigen::VectorXd transformed_rhs(const Eigen::MatrixXd& A,
                                           const std::vector<BoundView>& view,
                                           const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        Eigen::VectorXd rhs = A.rows() ? Eigen::VectorXd::Zero(A.rows()) : Eigen::VectorXd{};
        for (int j = 0; j < A.cols(); ++j) {
            const double anchor = bound_anchor(view[j], j, l, u);
            if (anchor != 0.0) {
                rhs.noalias() += A.col(j) * anchor;
            }
        }
        return rhs;
    }

    static Eigen::VectorXd transformed_rhs(const RevisedSimplex::SparseMatrix& A,
                                           const std::vector<BoundView>& view,
                                           const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
        Eigen::VectorXd rhs = A.rows() ? Eigen::VectorXd::Zero(A.rows()) : Eigen::VectorXd{};
        for (int j = 0; j < A.cols(); ++j) {
            const double anchor = bound_anchor(view[j], j, l, u);
            if (anchor == 0.0) {
                continue;
            }
            for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it) {
                rhs(it.row()) += it.value() * anchor;
            }
        }
        return rhs;
    }

    static void scale_column(Eigen::MatrixXd& A, int j, double scale) {
        if (scale == 1.0)
            return;
        A.col(j) *= scale;
    }

    static void scale_column(RevisedSimplex::SparseMatrix& A, int j, double scale) {
        if (scale == 1.0)
            return;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it) {
            it.valueRef() *= scale;
        }
    }

    static SparseRowMatrix rowwise_copy(const RevisedSimplex::SparseMatrix& A) {
        return SparseRowMatrix(A);
    }

    static void scale_rowwise_column(SparseRowMatrix& A_row,
                                     const RevisedSimplex::SparseMatrix& A_col, int j,
                                     double scale) {
        if (scale == 1.0)
            return;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A_col, j); it; ++it) {
            A_row.coeffRef(it.row(), j) *= scale;
        }
    }

    static double column_dot(const Eigen::MatrixXd& A, int j, const Eigen::VectorXd& v) {
        return A.col(j).dot(v);
    }

    static double column_dot(const RevisedSimplex::SparseMatrix& A, int j,
                             const Eigen::VectorXd& v) {
        double dot = 0.0;
        for (RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it) {
            dot += it.value() * v(it.row());
        }
        return dot;
    }

    static Eigen::VectorXd dense_column(const Eigen::MatrixXd& A, int j) { return A.col(j); }

    static Eigen::VectorXd dense_column(const RevisedSimplex::SparseMatrix& A, int j) {
        Eigen::VectorXd col = Eigen::VectorXd::Zero(A.rows());
        for (typename RevisedSimplex::SparseMatrix::InnerIterator it(A, j); it; ++it) {
            col(it.row()) = it.value();
        }
        return col;
    }

    static void compute_pricing_products(const Eigen::MatrixXd& Ahat, const std::vector<int>& N,
                                         const Eigen::VectorXd& w, const Eigen::VectorXd& ydual,
                                         const Eigen::VectorXd& chat, Eigen::VectorXd& pN,
                                         Eigen::VectorXd& rN) {
        pN.resize(N.size());
        rN.resize(N.size());
        for (int k = 0; k < (int)N.size(); ++k) {
            const int j = N[k];
            const Eigen::VectorXd col_j = Ahat.col(j);
            pN(k) = w.dot(col_j);
            rN(k) = chat(j) - col_j.dot(ydual);
        }
    }

    static void compute_pricing_products(const RevisedSimplex::SparseMatrix& Ahat,
                                         const SparseRowMatrix& Ahat_row, const std::vector<int>& N,
                                         const Eigen::VectorXd& w, const Eigen::VectorXd& ydual,
                                         const Eigen::VectorXd& chat, Eigen::VectorXd& pN,
                                         Eigen::VectorXd& rN) {
        pN = Eigen::VectorXd::Zero(N.size());
        rN.resize(N.size());
        for (int k = 0; k < (int)N.size(); ++k) {
            rN(k) = chat(N[k]);
        }

        thread_local SparsePricingWorkspace workspace;
        workspace.prepare(Ahat.cols(), N);

        for (int i = 0; i < Ahat_row.rows(); ++i) {
            const double wi = (i < w.size()) ? w(i) : 0.0;
            const double yi = (i < ydual.size()) ? ydual(i) : 0.0;
            if (wi == 0.0 && yi == 0.0)
                continue;
            for (SparseRowMatrix::InnerIterator it(Ahat_row, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel < 0)
                    continue;
                if (wi != 0.0)
                    pN(rel) += wi * it.value();
                if (yi != 0.0)
                    rN(rel) -= yi * it.value();
            }
        }
    }

    // HVector-aware variant: when w carries a sparse pattern, iterate only
    // the rows where w may be nonzero. `ydual` is assumed dense (typically
    // is in practice, so we still walk Ahat_row for it — but skip rows that
    // are zero in both w and ydual the same way the dense path does).
    static void compute_pricing_products(const RevisedSimplex::SparseMatrix& Ahat,
                                         const SparseRowMatrix& Ahat_row, const std::vector<int>& N,
                                         const HVector& w, const Eigen::VectorXd& ydual,
                                         const Eigen::VectorXd& chat, Eigen::VectorXd& pN,
                                         Eigen::VectorXd& rN) {
        if (!w.has_pattern()) {
            compute_pricing_products(Ahat, Ahat_row, N, w.value, ydual, chat, pN, rN);
            return;
        }

        pN = Eigen::VectorXd::Zero(N.size());
        rN.resize(N.size());
        for (int k = 0; k < (int)N.size(); ++k) {
            rN(k) = chat(N[k]);
        }

        thread_local SparsePricingWorkspace workspace;
        workspace.prepare(Ahat.cols(), N);

        // Accumulate w-contribution: iterate only w's nonzero rows.
        for (int k = 0; k < w.count; ++k) {
            const int i = w.index[k];
            const double wi = w.value(i);
            if (wi == 0.0)
                continue;
            if (i >= Ahat_row.rows())
                continue;
            for (SparseRowMatrix::InnerIterator it(Ahat_row, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel < 0)
                    continue;
                pN(rel) += wi * it.value();
            }
        }

        // Accumulate ydual-contribution: full row scan (ydual is dense).
        for (int i = 0; i < Ahat_row.rows(); ++i) {
            const double yi = (i < ydual.size()) ? ydual(i) : 0.0;
            if (yi == 0.0)
                continue;
            for (SparseRowMatrix::InnerIterator it(Ahat_row, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel < 0)
                    continue;
                rN(rel) -= yi * it.value();
            }
        }
    }

    template <class MatrixType>
    static MatrixType signed_matrix_copy(const MatrixType& A, const std::vector<BoundView>& view) {
        MatrixType out = A;
        for (int j = 0; j < out.cols(); ++j) {
            if (view_sign(view[j]) < 0)
                scale_column(out, j, -1.0);
        }
        return out;
    }

    static Eigen::VectorXd assemble_transformed_primal(int n, const std::vector<int>& basis,
                                                       const Eigen::VectorXd& yB,
                                                       const Eigen::VectorXd& l,
                                                       const Eigen::VectorXd& u,
                                                       const std::vector<BoundView>& view) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j < 0 || j >= n)
                continue;
            inB[j] = 1;
            const double anchor = bound_anchor(view[j], j, l, u);
            if (view_sign(view[j]) > 0) {
                x(j) = anchor + ((i < yB.size()) ? yB(i) : 0.0);
            } else {
                x(j) = anchor - ((i < yB.size()) ? yB(i) : 0.0);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (!inB[j])
                x(j) = bound_anchor(view[j], j, l, u);
        }
        return RevisedSimplex::clip_small_(x);
    }

    static DualChoose dual_harris_choose(const Eigen::VectorXd& rN, const Eigen::VectorXd& pN,
                                         double delta, double eta) {
        std::vector<int> E;
        E.reserve((int)pN.size());
        for (int k = 0; k < pN.size(); ++k)
            if (pN(k) < -delta)
                E.push_back(k);
        if (E.empty())
            return {};

        double tau_star = std::numeric_limits<double>::infinity();
        for (int k : E)
            tau_star = std::min(tau_star, rN(k) / (-pN(k)));

        const double kappa = std::max(eta, eta * std::abs(tau_star));
        std::vector<int> candidates;
        for (int k : E) {
            if ((rN(k) / (-pN(k))) <= tau_star + kappa)
                candidates.push_back(k);
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

    static DualBFRTDecision dual_bfrt_decide(const RevisedSimplex& self, const Eigen::VectorXd& rN,
                                             const Eigen::VectorXd& pN, const std::vector<int>& N,
                                             const std::vector<BoundView>& view,
                                             const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                             int max_flips) {
        DualBFRTDecision out;
        DualChoose dc = dual_harris_choose(rN, pN, self.opt_.ratio_delta, self.opt_.ratio_eta);
        out.pivot_rel = dc.e_rel;
        out.tau = dc.tau;
        if (!dc.e_rel || !std::isfinite(dc.tau) || max_flips <= 0)
            return out;

        struct Event {
            double tau;
            int rel;
        };
        std::vector<Event> events;
        events.reserve(N.size());
        const double tau_cap = dc.tau + std::max(self.opt_.ratio_eta, 1e-12 * (1.0 + dc.tau));

        for (int k = 0; k < (int)N.size(); ++k) {
            if (k == *dc.e_rel)
                continue;
            if (!(pN(k) < -self.opt_.ratio_delta))
                continue;

            const int j = N[k];
            const double range = bound_range(j, l, u);
            if (!std::isfinite(range) || range <= self.opt_.tol)
                continue;
            if (view[j] == BoundView::Fixed)
                continue;

            const double tau_k = rN(k) / (-pN(k));
            if (!std::isfinite(tau_k) || tau_k < 0.0 || tau_k > tau_cap) {
                continue;
            }
            events.push_back({tau_k, k});
        }

        std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
            if (std::abs(a.tau - b.tau) > 1e-16)
                return a.tau < b.tau;
            return a.rel < b.rel;
        });

        for (int i = 0; i < (int)events.size() && i < max_flips; ++i) {
            out.flip_rels.push_back(events[i].rel);
        }
        return out;
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

        std::vector<BoundView> view(n, BoundView::Lower);
        for (int j = 0; j < n; ++j)
            view[j] = default_bound_view(j, l, u);
        const bool warm_views_provided =
            warm_status && warm_status->size() == static_cast<std::size_t>(n);
        self.bridge_.reset();
        DualAdaptivePricer dual_pricer(self.opt_.pricing_rule, self.opt_.devex_reset,
                                       self.opt_.adaptive_reset_freq, self.opt_.partial_pricing,
                                       self.opt_.dual_pricing, self.opt_.row_pricing_threshold,
                                       self.opt_.dual_edge_weight_strategy,
                                       self.opt_.dual_steepest_edge_weight_log_error_threshold);

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

        std::shared_ptr<FTBasis> basis_factorization;
        std::shared_ptr<LPWarmStateData> reused_warm_state = self.try_reuse_factorization_(basis);
        if (reused_warm_state) {
            basis_factorization = reused_warm_state->basis_factorization;
        } else {
            try {
                basis_factorization =
                    std::make_shared<FTBasis>(Ahat, basis, self.make_basis_options_());
            } catch (const std::exception& e) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "dual initial basis factorization failed"}, {"what", e.what()}}};
            }
        }
        auto rebuild_basis_factorization = [&]() -> std::shared_ptr<FTBasis> {
            return std::make_shared<FTBasis>(Ahat, basis, self.make_basis_options_());
        };
        auto read_basis = [&]() -> FTBasis& { return *basis_factorization; };
        auto write_basis = [&]() -> FTBasis& {
            if (!basis_factorization.unique()) {
                basis_factorization = rebuild_basis_factorization();
            }
            return *basis_factorization;
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
            Eigen::VectorXd ydual = read_basis().solve_BT(cB);
            apply_views_to_nonbasics(ydual);
        }
        if (!reused_warm_state) {
            try {
                read_basis().refactor();
            } catch (const std::exception& e) {
                return {LPSolution::Status::Singular,
                        Eigen::VectorXd::Zero(n),
                        basis,
                        0,
                        {{"where", "dual initial refactor failed"}, {"what", e.what()}}};
            }
        }
        auto rebuild_dual_pool = [&](const char* where,
                                     int iter) -> std::optional<RevisedSimplex::PhaseResult> {
            try {
                self.measure_pricing_build_(
                    true, [&]() { dual_pricer.build_dual_pool(read_basis(), Ahat, N); });
                return std::nullopt;
            } catch (const std::exception& e) {
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
                    imported.row.prefer_row_pricing =
                        reused_warm_state->dual_pricing_state->prefer_row_pricing;
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
        int total_flips = 0;
        Eigen::VectorXd rhs_eff = b - transformed_rhs(A, view, l, u);
        bool ydual_cached = false;
        Eigen::VectorXd ydual;

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

        while (iters < self.opt_.max_iters) {
            ++iters;
            int flips_this_iter = 0;
            Eigen::VectorXd yB;
            Eigen::VectorXd cB(m);
            Eigen::VectorXd pN;
            Eigen::VectorXd rN;
            int r_leave = -1;
            HVector w;
            int e_rel = -1;
            int eAbs = -1;
            HVector s_enter;
            double tau = std::numeric_limits<double>::infinity();

            while (true) {
                try {
                    yB = read_basis().solve_B(rhs_eff);
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve_B failure");
                        write_basis().refactor();
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
                    cB(i) = chat(basis[i]);
                if (!ydual_cached) {
                    try {
                        ydual = read_basis().solve_BT(cB);
                        ydual_cached = true;
                    } catch (...) {
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve_BT failure");
                        write_basis().refactor();
                        if (auto failed = rebuild_dual_pool(
                                "dual pricing rebuild failed after solve_BT", iters)) {
                            return *failed;
                        }
                        ydual = read_basis().solve_BT(cB);
                        ydual_cached = true;
                    }
                }

                if (!(warm_views_provided && iters == 1) && apply_views_to_nonbasics(ydual)) {
                    rhs_eff = b - transformed_rhs(A, view, l, u);
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after bound view update", iters)) {
                        return *failed;
                    }
                    continue;
                }

                const auto leaving =
                    dual_pricer.choose_dual_leaving(read_basis(), yB, self.opt_.tol);
                r_leave = leaving.row;
                if (r_leave < 0) {
                    rN.resize(N.size());
                    bool dual_feasible = true;
                    for (int k = 0; k < (int)N.size(); ++k) {
                        const int j = N[k];
                        rN(k) = chat(j) - column_dot(Ahat, j, ydual);
                        if (rN(k) < -self.opt_.tol)
                            dual_feasible = false;
                    }
                    if (dual_feasible) {
                        Eigen::VectorXd x =
                            assemble_transformed_primal(n, basis, yB.cwiseMax(0.0), l, u, view);
                        auto info_map = dm_stats_to_map(self.degen_.get_stats());
                        info_map["dual_pricing"] = dual_pricer.current_strategy_name();
                        info_map["dual_bfrt_flips"] = std::to_string(total_flips);
                        self.trace_line_("[dual] optimal iter=" + std::to_string(iters) +
                                         " basis=" + self.format_basis_(basis));
                        const auto dual_state = dual_pricer.export_state();
                        LPDualPricingWarmState pricing_state;
                        switch (dual_state.active_rule) {
                            case DualAdaptivePricer::Rule::SteepestEdge:
                                pricing_state.active_rule =
                                    LPDualPricingWarmState::Rule::SteepestEdge;
                                pricing_state.row_weights = dual_state.steepest.row_weights;
                                break;
                            case DualAdaptivePricer::Rule::Devex:
                                pricing_state.active_rule = LPDualPricingWarmState::Rule::Devex;
                                pricing_state.row_weights = dual_state.devex.row_weights;
                                break;
                            case DualAdaptivePricer::Rule::RowPricing:
                                pricing_state.active_rule =
                                    LPDualPricingWarmState::Rule::RowPricing;
                                pricing_state.row_weights = dual_state.row.row_weights;
                                pricing_state.prefer_row_pricing =
                                    dual_state.row.prefer_row_pricing;
                                break;
                            case DualAdaptivePricer::Rule::MostInfeasible:
                                pricing_state.active_rule =
                                    LPDualPricingWarmState::Rule::MostInfeasible;
                                break;
                        }
                        self.remember_warm_state_(basis, basis_factorization, pricing_state);
                        return {LPSolution::Status::Optimal, std::move(x), basis, iters,
                                std::move(info_map)};
                    }
                    self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                     " primal-feasible but dual-infeasible");
                    if (!cost_shift_phase1_used) {
                        cost_shift_phase1_used = true;
                        double max_violation = 0.0;
                        for (int k = 0; k < (int)N.size(); ++k) {
                            if (rN(k) < -self.opt_.tol && -rN(k) > max_violation) {
                                max_violation = -rN(k);
                            }
                        }
                        if (max_violation > 0.0 && max_violation < 1e12) {
                            Eigen::VectorXd shift = Eigen::VectorXd::Zero(n);
                            for (int k = 0; k < (int)N.size(); ++k) {
                                if (rN(k) < -self.opt_.tol) {
                                    const int j = N[k];
                                    shift(j) = -rN(k) * 1.5;
                                }
                            }
                            Eigen::VectorXd c_shifted = c_work + shift;
                            for (int k = 0; k < (int)N.size(); ++k) {
                                const int j = N[k];
                                chat(j) = (view_sign(view[j]) > 0) ? c_shifted(j) : -c_shifted(j);
                            }
                            for (int i = 0; i < m; ++i)
                                chat(basis[i]) = c_work(basis[i]);
                            self.trace_line_("[dual] cost-shift Phase 1 applied, refactoring");
                            try {
                                write_basis().refactor();
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

                w = leaving.dual_row;
                if constexpr (std::is_same_v<MatrixType, RevisedSimplex::SparseMatrix>) {
                    compute_pricing_products(Ahat, *Ahat_row, N, w, ydual, chat, pN, rN);
                } else {
                    compute_pricing_products(Ahat, N, w, ydual, chat, pN, rN);
                }

                const DualBFRTDecision bfrt =
                    dual_bfrt_decide(self, rN, pN, N, view, l, u,
                                     self.opt_.dual_allow_bound_flip
                                         ? (self.opt_.dual_flip_max_per_iter - flips_this_iter)
                                         : 0);
                if (!bfrt.pivot_rel) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after no eligible entering");
                        write_basis().refactor();
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
                                    rhs_eff(it.row()) -= it.value() * delta_anchor;
                                }
                            } else {
                                const Eigen::VectorXd col_j = A.col(j);
                                rhs_eff.noalias() -= col_j * delta_anchor;
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
                        ++flips_this_iter;
                        ++total_flips;
                    }
                    if (auto failed = rebuild_dual_pool(
                            "dual pricing rebuild failed after bound flips", iters)) {
                        return *failed;
                    }
                    continue;
                }

                e_rel = *bfrt.pivot_rel;
                eAbs = N[e_rel];
                tau = bfrt.tau;
                if (self.degen_.would_repeat_basis_change(basis, r_leave, eAbs)) {
                    int alt_rel = -1;
                    double alt_tau = std::numeric_limits<double>::infinity();
                    for (int k = 0; k < static_cast<int>(N.size()); ++k) {
                        if (k == e_rel || !(pN(k) < -self.opt_.ratio_delta)) {
                            continue;
                        }
                        const double candidate_tau = rN(k) / (-pN(k));
                        if (!std::isfinite(candidate_tau) || candidate_tau < 0.0) {
                            continue;
                        }
                        const int candidate_abs = N[k];
                        if (self.degen_.would_repeat_basis_change(basis, r_leave, candidate_abs)) {
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
                        e_rel = alt_rel;
                        eAbs = N[e_rel];
                        tau = alt_tau;
                    }
                }
                try {
                    s_enter = read_basis().solve_B(Ahat.col(eAbs));
                } catch (...) {
                    if (rebuild_attempts < self.opt_.max_basis_rebuilds) {
                        ++rebuild_attempts;
                        self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                         " refactor after solve(B,a_e) failure");
                        write_basis().refactor();
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
                break;
            }

            if (!std::isfinite(tau)) {
                Eigen::VectorXd yF = w;
                if (yF.dot(rhs_eff) >= 0)
                    yF = -yF;

                auto info_map = dm_stats_to_map(self.degen_.get_stats());
                info_map["where"] = "dual: infinite step";
                info_map["dual_pricing"] = dual_pricer.current_strategy_name();
                info_map["dual_bfrt_flips"] = std::to_string(total_flips);
                info_map["certificate"] = "farkas";
                info_map["farkas_has_cert"] = "1";
                info_map["farkas_dim"] = std::to_string(m);
                info_map["farkas_y"] = serialize_vec(yF);
                self.trace_line_("[dual] infeasible iter=" + std::to_string(iters) +
                                 " produced Farkas certificate");
                self.remember_warm_state_(basis, basis_factorization);
                return {LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n), basis, iters,
                        std::move(info_map)};
            }

            const bool is_degenerate = self.degen_.detect_degeneracy(tau, self.opt_.deg_step_tol);
            const int oldAbs = basis[r_leave];
            const auto basis_cycle = self.degen_.register_basis_change(basis, r_leave, eAbs, iters);
            if (basis_cycle.repeated_basis) {
                self.trace_line_(
                    "[dual] iter=" + std::to_string(iters) +
                    " repeated basis candidate leave_var=" + std::to_string(oldAbs) +
                    " enter=" + std::to_string(eAbs) +
                    (basis_cycle.cycling_detected ? " cycle_detected=1" : " cycle_detected=0"));
            }
            if (basis_cycle.cycling_detected ||
                (is_degenerate && self.degen_.should_apply_perturbation())) {
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
                if (costs_perturbed) {
                    c_work = c;
                    for (int j = 0; j < n; ++j) {
                        chat(j) = (view_sign(view[j]) > 0) ? c_work(j) : -c_work(j);
                    }
                    costs_perturbed = false;
                    ydual_cached = false;
                }
                (void)self.degen_.reset_perturbation();
            }
            self.degen_.after_pivot(r_leave, eAbs, tau, 0.0,
                                    std::isfinite(tau) ? std::abs(tau) : 0.0);

            if (self.should_trace_iter_(iters)) {
                Eigen::VectorXd xcur =
                    assemble_transformed_primal(n, basis, yB.cwiseMax(0.0), l, u, view);
                std::ostringstream oss;
                oss << "[dual] iter=" << iters << " obj=" << c.dot(xcur) << " leave_row=" << r_leave
                    << " leave_var=" << oldAbs << " enter=" << eAbs << " tau=" << tau;
                if (self.opt_.verbose_include_basis) {
                    oss << " basis_before=" << self.format_basis_(basis);
                }
                self.trace_line_(oss.str());
            }

            Eigen::VectorXd unit = Eigen::VectorXd::Zero(m);
            unit(r_leave) = 1.0;
            Eigen::VectorXd z = read_basis().solve_BT(unit);
            const double pivot = s_enter(r_leave);
            const double alpha = rN(e_rel) / pivot;
            ydual.noalias() += alpha * z;
            for (int k = 0; k < static_cast<int>(N.size()); ++k) {
                if (k == e_rel)
                    continue;
                rN(k) -= alpha * column_dot(Ahat, N[k], z);
            }
            basis[r_leave] = eAbs;
            N[e_rel] = oldAbs;
            rN(e_rel) = chat(oldAbs) - column_dot(Ahat, oldAbs, ydual);

            try {
                write_basis().replace_column(r_leave, Ahat.col(eAbs));
            } catch (...) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " refactor after replace_column failure");
                write_basis().refactor();
                if (auto failed = rebuild_dual_pool(
                        "dual pricing rebuild failed after replace_column", iters)) {
                    return *failed;
                }
            }

            dual_pricer.update_after_dual_pivot(r_leave, eAbs, oldAbs, s_enter, s_enter(r_leave),
                                                Ahat, N, w, true);
            if (dual_pricer.needs_rebuild()) {
                if (auto failed = rebuild_dual_pool(
                        "dual pricing rebuild failed after pivot update", iters)) {
                    return *failed;
                }
                dual_pricer.clear_rebuild_flag();
            }
            if (self.should_trace_iter_(iters) && self.opt_.verbose_include_basis) {
                self.trace_line_("[dual] iter=" + std::to_string(iters) +
                                 " basis_after=" + self.format_basis_(basis));
            }
        }

        auto info_map = dm_stats_to_map(self.degen_.get_stats());
        info_map["dual_pricing"] = dual_pricer.current_strategy_name();
        info_map["dual_bfrt_flips"] = std::to_string(total_flips);
        self.trace_line_("[dual] iterlimit basis=" + self.format_basis_(basis));
        self.remember_warm_state_(basis, basis_factorization);
        return {LPSolution::Status::IterLimit, Eigen::VectorXd::Zero(n), basis, iters,
                std::move(info_map)};
    }
};
