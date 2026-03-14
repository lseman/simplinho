#pragma once

#include "degeneracy.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================
// PrimalPricingBridge
//  - Adapter to thread DegeneracyManager signals into the primal pricer
//  - Does ABS <-> REL mapping here (we have N)
// ============================================================================
template <class PrimalPricer>
struct PrimalPricingBridge {
    DegeneracyManager& dm;
    PrimalPricer& pricer;

    PrimalPricingBridge(DegeneracyManager& dm_, PrimalPricer& pr_)
        : dm(dm_), pricer(pr_) {}

    template <class BasisLike>
    std::optional<int> choose_primal_entering(
        const Eigen::VectorXd& rN, const std::vector<int>& N, double tol,
        int iteration, double current_objective, const BasisLike& basis,
        const Eigen::MatrixXd& A) {
        const auto& sig =
            dm.begin_pricing(current_objective, iteration, int(N.size()));

        const bool strategy_changed =
            pricer.apply_preferred_strategy(sig.preferred_strategy);

        if (strategy_changed || sig.request_pool_rebuild) {
            pricer.build_primal_pools(basis, A, N);
        }

        // Build effective reduced costs with ABS-keyed weight/lex hints
        Eigen::VectorXd rN_eff = rN;

        if (!sig.weight_overrides_abs.empty()) {
            for (int k = 0; k < (int)N.size(); ++k) {
                const int jAbs = N[k];
                auto it = sig.weight_overrides_abs.find(jAbs);
                if (it == sig.weight_overrides_abs.end()) {
                    // handle pseudo keys (kRel) if DM used kRel as placeholder
                    it = sig.weight_overrides_abs.find(k);
                }
                if (it != sig.weight_overrides_abs.end()) {
                    const double w = std::max(1.0, it->second);
                    rN_eff(k) = rN_eff(k) / std::sqrt(w);
                }
            }
        }

        if (!sig.lex_order_abs.empty()) {
            for (int k = 0; k < (int)N.size(); ++k) {
                const int jAbs = N[k];
                auto it = sig.lex_order_abs.find(jAbs);
                if (it == sig.lex_order_abs.end()) {
                    // accept pseudo kRel key
                    it = sig.lex_order_abs.find(k);
                }
                if (it != sig.lex_order_abs.end() && !it->second.empty()) {
                    // add a tiny epsilon as tie breaker
                    rN_eff(k) += 1e-16 * it->second.back();
                }
            }
        }

        auto entering_rel =
            pricer.choose_primal_entering(rN_eff, N, tol, iteration,
                                          current_objective, basis, A);

        // Forbid list: ABS -> REL mapping
        if (entering_rel && !sig.forbid_abs_candidates.empty()) {
            std::unordered_set<int> forbid(sig.forbid_abs_candidates.begin(),
                                           sig.forbid_abs_candidates.end());
            if (forbid.count(N[*entering_rel])) {
                int best = -1;
                double best_rc = 0.0;
                for (int k = 0; k < (int)N.size(); ++k) {
                    if (rN_eff(k) < -tol && !forbid.count(N[k]) &&
                        rN_eff(k) < best_rc) {
                        best_rc = rN_eff(k);
                        best = k;
                    }
                }
                if (best >= 0)
                    entering_rel = best;
                else
                    entering_rel.reset();
            }
        }
        return entering_rel;
    }

    void after_primal_pivot(int leaving_rel, int entering_abs, int old_abs,
                            const Eigen::VectorXd& pivot_column, double alpha,
                            double step_size, const Eigen::MatrixXd& A,
                            const std::vector<int>& N,
                            double rc_improvement = 0.0) {
        pricer.update_after_primal_pivot(leaving_rel, entering_abs, old_abs,
                                         pivot_column, alpha, step_size, A, N);

        dm.after_pivot(leaving_rel, entering_abs, alpha, rc_improvement,
                       step_size);

        if (pricer.needs_rebuild()) {
            pricer.clear_rebuild_flag();
        }
    }
};

// ============================================================================
// SteepestEdgePricer (true steepest-edge; FT-consistent update)
// ============================================================================
class SteepestEdgePricer {
   public:
    struct Entry {
        int jN;             // absolute col index
        Eigen::VectorXd t;  // B^{-1} a_j
        double weight;      // 1 + ||t||^2
    };

    explicit SteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_primal_pool(const BasisLike& B, const Eigen::MatrixXd& A,
                           const std::vector<int>& N) {
        pool_.clear();
        pos_.clear();
        const int take = (pool_max_ > 0)
                             ? std::min<int>(pool_max_, (int)N.size())
                             : (int)N.size();
        pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            Entry e;
            e.jN = j;
            e.t = B.solve_B(A.col(j));  // caller-provided
            e.weight = 1.0 + e.t.squaredNorm();
            pos_[j] = (int)pool_.size();
            pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    std::optional<int> choose_primal_entering(
        const Eigen::VectorXd& rcN, const std::vector<int>& N, double tol) {
        ++iter_count_;
        int best_rel = -1;
        double best_score = -1.0;

        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol) continue;
            const int j = N[k];
            double w = 1.0;
            if (auto it = pos_.find(j); it != pos_.end())
                w = pool_[it->second].weight;
            const double score = (rcN(k) * rcN(k)) / w;
            if (score > best_score) {
                best_score = score;
                best_rel = k;
            }
        }
        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_primal_pivot(int leave_rel, int e_abs, int old_abs,
                                   const Eigen::VectorXd& s, double alpha,
                                   const Eigen::MatrixXd& /*A*/,
                                   const std::vector<int>& /*N*/,
                                   bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol) {
            need_rebuild_ = true;
            return;
        }

        const double inv_alpha = 1.0 / alpha;

        // Update t, weight
        for (auto& E : pool_) {
            if (leave_rel < E.t.size()) {
                const double tr = E.t(leave_rel);
                if (tr != 0.0) {
                    E.t.noalias() -= s * (tr * inv_alpha);
                    E.weight = 1.0 + E.t.squaredNorm();
                }
            }
        }

        // Remove entering from pool
        if (auto itE = pos_.find(e_abs); itE != pos_.end()) {
            const int idx = itE->second;
            const int last = (int)pool_.size() - 1;
            if (idx != last) {
                pos_[pool_[last].jN] = idx;
                std::swap(pool_[idx], pool_[last]);
            }
            pool_.pop_back();
            pos_.erase(itE);
        }

        // Optionally add leaving
        if (insert_leaver_into_pool) {
            Entry E;
            E.jN = old_abs;
            E.t = Eigen::VectorXd::Zero(s.size());
            if (leave_rel < E.t.size()) E.t(leave_rel) = 1.0;
            E.t.noalias() -= s * inv_alpha;
            E.weight = 1.0 + E.t.squaredNorm();

            if (pool_max_ > 0 && (int)pool_.size() >= pool_max_) {
                // Evict largest-weight entry
                int evict = 0;
                double wmax = pool_[0].weight;
                for (int i = 1; i < (int)pool_.size(); ++i) {
                    if (pool_[i].weight > wmax) {
                        wmax = pool_[i].weight;
                        evict = i;
                    }
                }
                pos_.erase(pool_[evict].jN);
                pool_[evict] = std::move(E);
                pos_[pool_[evict].jN] = evict;
            } else {
                pos_[E.jN] = (int)pool_.size();
                pool_.push_back(std::move(E));
            }
        }

        ++iter_count_;
        if (need_rebuild_ || iter_count_ >= reset_freq_) need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

   private:
    std::vector<Entry> pool_;
    std::unordered_map<int, int> pos_;
    int pool_max_{0};
    int reset_freq_{1000};
    int iter_count_{0};
    bool need_rebuild_{false};
};

// ============================================================================
// DevexPricer (lightweight weights; API preserved)
// ============================================================================
class DevexPricer {
   public:
    explicit DevexPricer(double threshold = 0.99, int reset_frequency = 1000)
        : threshold_(threshold), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_primal_pool(const BasisLike& /*B*/, const Eigen::MatrixXd& /*A*/,
                           const std::vector<int>& N) {
        weights_.clear();
        for (int j : N) weights_[j] = 1.0;
        iter_count_ = 0;
    }

    std::optional<int> choose_primal_entering(
        const Eigen::VectorXd& rcN, const std::vector<int>& N, double tol) {
        ++iter_count_;
        if (iter_count_ % reset_freq_ == 0) {
            for (auto& p : weights_) p.second = 1.0;
        }

        int best_rel = -1;
        double best_crit = -1.0;
        for (int k = 0; k < (int)N.size(); ++k) {
            if (rcN(k) >= -tol) continue;
            const int j = N[k];
            const double w = (weights_.count(j) ? weights_.at(j) : 1.0);
            const double crit = (rcN(k) * rcN(k)) / w;
            if (crit > best_crit) {
                best_crit = crit;
                best_rel = k;
            }
        }
        return (best_rel >= 0) ? std::optional<int>(best_rel) : std::nullopt;
    }

    void update_after_primal_pivot(int leave_rel, int e_abs, int old_abs,
                                   const Eigen::VectorXd& pivot_column,
                                   double alpha,
                                   const Eigen::MatrixXd& /*A*/,
                                   const std::vector<int>& N,
                                   bool /*insert_leaver_into_pool*/ = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol) return;

        // Entering weight: keep bounded to avoid runaway
        const double a2 = alpha * alpha;
        weights_[e_abs] = std::min(std::max(a2, 1e-4), 1e6);

        // Update others (classic Devex-like)
        if (leave_rel < pivot_column.size()) {
            const double gamma_over_alpha = pivot_column(leave_rel) / alpha;
            const double add = gamma_over_alpha * gamma_over_alpha;
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                if (j == e_abs) continue;
                double& w = weights_[j];
                if (!weights_.count(j)) w = 1.0;
                const double nw = w + add;
                w = std::max(nw, threshold_ * w);
            }
        }

        // Ensure leaving has a slot
        (void)old_abs;
        if (!weights_.count(old_abs)) weights_[old_abs] = 1.0;
    }

    bool needs_rebuild() const { return false; }
    void clear_rebuild_flag() {}

   private:
    std::unordered_map<int, double> weights_;
    double threshold_{0.99};
    int reset_freq_{1000};
    int iter_count_{0};
};

// ============================================================================
// DualSteepestEdgePricer (exact dual row weights + maintained column weights)
// ============================================================================
class DualSteepestEdgePricer {
   public:
    struct DualEntry {
        int jN;
        Eigen::VectorXd w;   // approx B^{-T} a_j
        double dual_weight;  // ||w||^2
    };

    struct RowEntry {
        Eigen::VectorXd psi;  // exact B^{-T} e_i for current basis row i
        double weight = 1.0;
    };

    struct LeavingChoice {
        int row = -1;
        Eigen::VectorXd dual_row;
        double weight = 1.0;
    };

    explicit DualSteepestEdgePricer(int pool_max = 0,
                                    int reset_frequency = 1000)
        : pool_max_(pool_max), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_dual_pool(const BasisLike& B, const Eigen::MatrixXd& A,
                         const std::vector<int>& N) {
        dual_pool_.clear();
        dual_pos_.clear();
        row_pool_.clear();
        const int take = (pool_max_ > 0)
                             ? std::min<int>(pool_max_, (int)N.size())
                             : (int)N.size();
        dual_pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            DualEntry e;
            e.jN = j;
            const Eigen::VectorXd Aj = A.col(j);
            e.w = B.solve_BT(Aj);  // caller-provided
            e.dual_weight = e.w.squaredNorm();
            dual_pos_[j] = (int)dual_pool_.size();
            dual_pool_.push_back(std::move(e));
        }
        row_pool_.resize(A.rows());
        for (int i = 0; i < A.rows(); ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(A.rows());
            e_i(i) = 1.0;
            row_pool_[i].psi = B.solve_BT(e_i);
            row_pool_[i].weight =
                std::max(1.0, row_pool_[i].psi.squaredNorm());
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B,
                                      const Eigen::VectorXd& yB,
                                      double tol) const {
        LeavingChoice best;
        double best_score = -1.0;
        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol) continue;
            double weight = 1.0;
            if (i < (int)row_pool_.size()) weight = row_pool_[i].weight;
            const double infeas = -yB(i);
            const double score = (infeas * infeas) / weight;
            if (score > best_score) {
                best_score = score;
                best.row = i;
                if (i < (int)row_pool_.size() &&
                    row_pool_[i].psi.size() == yB.size()) {
                    best.dual_row = row_pool_[i].psi;
                } else {
                    Eigen::VectorXd e_i = Eigen::VectorXd::Zero(yB.size());
                    e_i(i) = 1.0;
                    best.dual_row = B.solve_BT(e_i);
                    weight = std::max(1.0, best.dual_row.squaredNorm());
                }
                best.weight = weight;
            }
        }
        return best;
    }

    void update_after_dual_pivot(int leave_rel, int e_abs, int old_abs,
                                 const Eigen::VectorXd& s, double alpha,
                                 const Eigen::MatrixXd& A,
                                 const std::vector<int>& /*N*/,
                                 const Eigen::VectorXd& dual_row,
                                 bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol) {
            need_rebuild_ = true;
            return;
        }

        if (A.size() == 0) {
            // A mathematically correct DSE update needs psi_r = B^{-T} e_r
            // from the pre-pivot basis.
            need_rebuild_ = true;
            return;
        }

        const Eigen::VectorXd& psi_r = dual_row;
        if (leave_rel < 0 || leave_rel >= s.size()) {
            need_rebuild_ = true;
            return;
        }

        if (!row_pool_.empty()) {
            if (leave_rel >= (int)row_pool_.size()) {
                need_rebuild_ = true;
                return;
            }

            const Eigen::VectorXd psi_before = psi_r;
            for (int i = 0; i < (int)row_pool_.size(); ++i) {
                if (i == leave_rel) continue;
                const double coeff = s(i) / alpha;
                if (coeff != 0.0) {
                    row_pool_[i].psi.noalias() -= psi_before * coeff;
                }
                row_pool_[i].weight =
                    std::max(1.0, row_pool_[i].psi.squaredNorm());
            }
            row_pool_[leave_rel].psi = psi_before / alpha;
            row_pool_[leave_rel].weight =
                std::max(1.0, row_pool_[leave_rel].psi.squaredNorm());
        }

        Eigen::VectorXd e_r = Eigen::VectorXd::Zero(s.size());
        if (leave_rel >= 0 && leave_rel < e_r.size()) e_r(leave_rel) = 1.0;
        const Eigen::VectorXd s_minus_er = s - e_r;
        const double inv_alpha = 1.0 / alpha;

        // Exact rank-one update for w_j = B^{-T} a_j under a primal pivot:
        //   w'_j = w_j - psi_r * (((s - e_r)^T a_j) / alpha)
        // where psi_r = B^{-T} e_r from the pre-pivot basis.
        for (auto& E : dual_pool_) {
            if (E.jN == e_abs) continue;
            const double beta = s_minus_er.dot(A.col(E.jN)) * inv_alpha;
            if (beta != 0.0) E.w.noalias() -= psi_r * beta;
            E.dual_weight = std::max(1.0, E.w.squaredNorm());
        }

        // Remove entering
        if (auto itE = dual_pos_.find(e_abs); itE != dual_pos_.end()) {
            const int idx = itE->second, last = (int)dual_pool_.size() - 1;
            if (idx != last) {
                dual_pos_[dual_pool_[last].jN] = idx;
                std::swap(dual_pool_[idx], dual_pool_[last]);
            }
            dual_pool_.pop_back();
            dual_pos_.erase(itE);
        }

        // Add leaving
        if (insert_leaver_into_pool) {
            DualEntry E;
            E.jN = old_abs;
            E.w = e_r;
            const double beta_old = s_minus_er.dot(A.col(old_abs)) * inv_alpha;
            if (beta_old != 0.0) E.w.noalias() -= psi_r * beta_old;
            E.dual_weight = std::max(1.0, E.w.squaredNorm());

            if (pool_max_ > 0 && (int)dual_pool_.size() >= pool_max_) {
                int evict = 0;
                double wmax = dual_pool_[0].dual_weight;
                for (int i = 1; i < (int)dual_pool_.size(); ++i) {
                    if (dual_pool_[i].dual_weight > wmax) {
                        wmax = dual_pool_[i].dual_weight;
                        evict = i;
                    }
                }
                dual_pos_.erase(dual_pool_[evict].jN);
                dual_pool_[evict] = std::move(E);
                dual_pos_[dual_pool_[evict].jN] = evict;
            } else {
                dual_pos_[E.jN] = (int)dual_pool_.size();
                dual_pool_.push_back(std::move(E));
            }
        }

        ++iter_count_;
        if (iter_count_ >= reset_freq_) need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

   private:
    std::vector<DualEntry> dual_pool_;
    std::vector<RowEntry> row_pool_;
    std::unordered_map<int, int> dual_pos_;
    int pool_max_{0};
    int reset_freq_{1000};
    int iter_count_{0};
    bool need_rebuild_{false};
};

// ============================================================================
// DualDevexPricer (dual leaving-row Devex with exact resets)
// ============================================================================
class DualDevexPricer {
   public:
    struct LeavingChoice {
        int row = -1;
        Eigen::VectorXd dual_row;
        double weight = 1.0;
    };

    explicit DualDevexPricer(double threshold = 0.99, int reset_frequency = 200)
        : threshold_(threshold), reset_freq_(reset_frequency) {}

    template <class BasisLike>
    void build_dual_pool(const BasisLike& B, const Eigen::MatrixXd& A,
                         const std::vector<int>& /*N*/) {
        row_weights_.assign(A.rows(), 1.0);
        for (int i = 0; i < A.rows(); ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(A.rows());
            e_i(i) = 1.0;
            const Eigen::VectorXd psi_i = B.solve_BT(e_i);
            row_weights_[i] = std::max(1.0, psi_i.squaredNorm());
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B,
                                      const Eigen::VectorXd& yB,
                                      double tol) const {
        LeavingChoice best;
        double best_score = -1.0;
        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol) continue;
            const double weight =
                (i < (int)row_weights_.size()) ? row_weights_[i] : 1.0;
            const double infeas = -yB(i);
            const double score = (infeas * infeas) / std::max(1.0, weight);
            if (score > best_score) {
                best_score = score;
                best.row = i;
                best.weight = std::max(1.0, weight);
            }
        }

        if (best.row >= 0) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(yB.size());
            e_i(best.row) = 1.0;
            best.dual_row = B.solve_BT(e_i);
            best.weight = std::max(1.0, best.dual_row.squaredNorm());
        }
        return best;
    }

    void update_after_dual_pivot(int leave_rel, int /*e_abs*/, int /*old_abs*/,
                                 const Eigen::VectorXd& s, double alpha,
                                 const Eigen::MatrixXd& /*A*/,
                                 const std::vector<int>& /*N*/,
                                 const Eigen::VectorXd& /*dual_row*/,
                                 bool /*insert_leaver_into_pool*/ = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol) {
            need_rebuild_ = true;
            return;
        }
        if (leave_rel < 0 || leave_rel >= (int)row_weights_.size()) {
            need_rebuild_ = true;
            return;
        }

        const double pivot_weight = std::max(1.0, row_weights_[leave_rel]);
        const double inv_alpha = 1.0 / alpha;
        for (int i = 0; i < (int)row_weights_.size() && i < s.size(); ++i) {
            const double sigma = (i == leave_rel) ? inv_alpha : s(i) * inv_alpha;
            const double candidate = sigma * sigma * pivot_weight;
            row_weights_[i] =
                std::max({1.0, threshold_ * row_weights_[i], candidate});
        }

        ++iter_count_;
        if (iter_count_ >= reset_freq_) need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

   private:
    std::vector<double> row_weights_;
    double threshold_{0.99};
    int reset_freq_{200};
    int iter_count_{0};
    bool need_rebuild_{false};
};

// ============================================================================
// DualAdaptivePricer (dual-side pricing rule selection)
// ============================================================================
class DualAdaptivePricer {
   public:
    struct LeavingChoice {
        int row = -1;
        Eigen::VectorXd dual_row;
        double weight = 1.0;
    };

    DualAdaptivePricer(std::string pricing_rule, int devex_reset_frequency,
                       int steepest_reset_frequency)
        : requested_rule_(std::move(pricing_rule)),
          steepest_pricer_(0, steepest_reset_frequency),
          devex_pricer_(0.99, devex_reset_frequency) {}

    template <class BasisLike>
    void build_dual_pool(const BasisLike& B, const Eigen::MatrixXd& A,
                         const std::vector<int>& N) {
        active_rule_ = select_rule_(A.rows());
        if (active_rule_ == Rule::SteepestEdge) {
            steepest_pricer_.build_dual_pool(B, A, N);
            devex_pricer_.clear_rebuild_flag();
        } else if (active_rule_ == Rule::Devex) {
            devex_pricer_.build_dual_pool(B, A, N);
            steepest_pricer_.clear_rebuild_flag();
        } else {
            steepest_pricer_.clear_rebuild_flag();
            devex_pricer_.clear_rebuild_flag();
        }
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B,
                                      const Eigen::VectorXd& yB,
                                      double tol) const {
        switch (active_rule_) {
            case Rule::SteepestEdge: {
                const auto choice =
                    steepest_pricer_.choose_dual_leaving(B, yB, tol);
                return {choice.row, choice.dual_row, choice.weight};
            }
            case Rule::Devex: {
                const auto choice = devex_pricer_.choose_dual_leaving(B, yB, tol);
                return {choice.row, choice.dual_row, choice.weight};
            }
            case Rule::MostInfeasible: {
                int best_row = -1;
                double best_infeas = 0.0;
                for (int i = 0; i < yB.size(); ++i) {
                    if (yB(i) >= -tol) continue;
                    const double infeas = -yB(i);
                    if (best_row < 0 || infeas > best_infeas) {
                        best_row = i;
                        best_infeas = infeas;
                    }
                }

                LeavingChoice choice;
                choice.row = best_row;
                if (best_row >= 0) {
                    Eigen::VectorXd e_i = Eigen::VectorXd::Zero(yB.size());
                    e_i(best_row) = 1.0;
                    choice.dual_row = B.solve_BT(e_i);
                    choice.weight =
                        std::max(1.0, choice.dual_row.squaredNorm());
                }
                return choice;
            }
        }
        return {};
    }

    void update_after_dual_pivot(int leave_rel, int e_abs, int old_abs,
                                 const Eigen::VectorXd& s, double alpha,
                                 const Eigen::MatrixXd& A,
                                 const std::vector<int>& N,
                                 const Eigen::VectorXd& dual_row,
                                 bool insert_leaver_into_pool = true) {
        switch (active_rule_) {
            case Rule::SteepestEdge:
                steepest_pricer_.update_after_dual_pivot(
                    leave_rel, e_abs, old_abs, s, alpha, A, N, dual_row,
                    insert_leaver_into_pool);
                need_rebuild_ = steepest_pricer_.needs_rebuild();
                break;
            case Rule::Devex:
                devex_pricer_.update_after_dual_pivot(
                    leave_rel, e_abs, old_abs, s, alpha, A, N, dual_row,
                    insert_leaver_into_pool);
                need_rebuild_ = devex_pricer_.needs_rebuild();
                break;
            case Rule::MostInfeasible:
                need_rebuild_ =
                    (std::abs(alpha) < dm_consts::kDegenerateAlphaTol);
                break;
        }
    }

    bool needs_rebuild() const {
        return need_rebuild_ ||
               (active_rule_ == Rule::SteepestEdge &&
                steepest_pricer_.needs_rebuild()) ||
               (active_rule_ == Rule::Devex && devex_pricer_.needs_rebuild());
    }

    void clear_rebuild_flag() {
        need_rebuild_ = false;
        steepest_pricer_.clear_rebuild_flag();
        devex_pricer_.clear_rebuild_flag();
    }

    const char* current_strategy_name() const {
        switch (active_rule_) {
            case Rule::SteepestEdge:
                return "dual_steepest_edge";
            case Rule::Devex:
                return "dual_devex";
            case Rule::MostInfeasible:
                return "dual_most_infeasible";
        }
        return "dual_unknown";
    }

   private:
    enum class Rule { SteepestEdge, Devex, MostInfeasible };

    Rule select_rule_(int basis_rows) const {
        if (requested_rule_ == "devex") return Rule::Devex;
        if (requested_rule_ == "most_negative") return Rule::MostInfeasible;
        if (requested_rule_ == "adaptive") {
            return (basis_rows > 256) ? Rule::Devex : Rule::SteepestEdge;
        }
        return Rule::SteepestEdge;
    }

    std::string requested_rule_;
    Rule active_rule_{Rule::SteepestEdge};
    bool need_rebuild_{false};
    DualSteepestEdgePricer steepest_pricer_;
    DualDevexPricer devex_pricer_;
};

// ============================================================================
// AdaptivePricer (strategy orchestration; API preserved)
// ============================================================================
class AdaptivePricer {
   public:
    enum Strategy {
        STEEPEST_EDGE = 0,
        DEVEX = 1,
        PARTIAL_PRICING = 2,
        MOST_NEGATIVE = 3
    };
    static constexpr int kNumStrategies = 4;

    struct PricingOptions {
        Strategy initial_strategy = STEEPEST_EDGE;
        int switch_threshold = 100;
        int performance_window = 50;
        double improvement_factor = 1.2;
        int partial_block_factor = 10;
        int min_partial_block = 10;
        bool enable_adaptive_switching = true;
        int steepest_pool_max = 0;
        int steepest_reset_freq = 1000;
        int devex_reset_freq = 1000;
    };

    struct PricingStats {
        int total_pricing_calls{0};
        int strategy_switches{0};
        double avg_improvement_per_iteration{0.0};
        std::vector<int> strategy_usage_count{
            std::vector<int>(kNumStrategies, 0)};
    };

    explicit AdaptivePricer(int n) : AdaptivePricer(n, PricingOptions{}) {}

    AdaptivePricer(int n, const PricingOptions& opts)
        : current_strategy_(opts.initial_strategy),
          options_(opts),
          n_(n),
          steepest_pricer_(opts.steepest_pool_max, opts.steepest_reset_freq),
          devex_pricer_(0.99, opts.devex_reset_freq),
          iterations_since_switch_(0),
          last_objective_(0.0),
          first_call_(true) {
        stats_.strategy_usage_count.assign(kNumStrategies, 0);
    }

    // Main pricing entry
    template <typename BasisLike>
    std::optional<int> choose_primal_entering(
        const Eigen::VectorXd& rN, const std::vector<int>& N, double tol,
        int iteration, double current_objective, const BasisLike& basis,
        const Eigen::MatrixXd& A) {
        ++stats_.total_pricing_calls;
        ++stats_.strategy_usage_count[current_strategy_];

        track_performance_(current_objective);

        if (options_.enable_adaptive_switching &&
            should_switch_strategy_(iteration)) {
            adapt_strategy_();
            rebuild_pools_(basis, A, N);
        }

        switch (current_strategy_) {
            case STEEPEST_EDGE:
                return steepest_pricer_.choose_primal_entering(rN, N, tol);
            case DEVEX:
                return devex_pricer_.choose_primal_entering(rN, N, tol);
            case PARTIAL_PRICING:
                return partial_pricing_(rN, N, tol, iteration);
            case MOST_NEGATIVE:
                return most_negative_pricing_(rN, N, tol);
        }
        return std::nullopt;
    }

    // Build pools for all (cheap; preserves API)
    template <typename BasisLike>
    void build_primal_pools(const BasisLike& basis, const Eigen::MatrixXd& A,
                            const std::vector<int>& N) {
        steepest_pricer_.build_primal_pool(basis, A, N);
        devex_pricer_.build_primal_pool(basis, A, N);
    }

    bool apply_preferred_strategy(
        std::optional<PricingStrategy> preferred_strategy) {
        if (!preferred_strategy) return false;

        const Strategy next = map_strategy_(*preferred_strategy);
        if (next == current_strategy_) return false;
        current_strategy_ = next;
        iterations_since_switch_ = 0;
        ++stats_.strategy_switches;
        return true;
    }

    void update_after_primal_pivot(int leaving_rel, int entering_abs,
                                   int old_abs,
                                   const Eigen::VectorXd& pivot_column,
                                   double alpha, double step_size,
                                   const Eigen::MatrixXd& A,
                                   const std::vector<int>& N) {
        steepest_pricer_.update_after_primal_pivot(
            leaving_rel, entering_abs, old_abs, pivot_column, alpha, A, N, true);
        devex_pricer_.update_after_primal_pivot(leaving_rel, entering_abs, old_abs,
                                                pivot_column, alpha, A, N, true);

        if ((int)performance_history_.size() >= options_.performance_window)
            performance_history_.pop_front();
        performance_history_.push_back(step_size);
    }

    bool needs_rebuild() const {
        switch (current_strategy_) {
            case STEEPEST_EDGE:
                return steepest_pricer_.needs_rebuild();
            default:
                return false;
        }
    }

    void clear_rebuild_flag() {
        steepest_pricer_.clear_rebuild_flag();
    }

    const char* get_current_strategy_name() const {
        switch (current_strategy_) {
            case STEEPEST_EDGE:
                return "steepest_edge";
            case DEVEX:
                return "devex";
            case PARTIAL_PRICING:
                return "partial_pricing";
            case MOST_NEGATIVE:
                return "most_negative";
        }
        return "unknown";
    }

    const PricingStats& get_stats() const { return stats_; }

    void reset(int new_n) {
        n_ = new_n;
        current_strategy_ = options_.initial_strategy;
        performance_history_.clear();
        recent_objectives_.clear();
        iterations_since_switch_ = 0;
        first_call_ = true;
        stats_ = PricingStats{};
        stats_.strategy_usage_count.assign(kNumStrategies, 0);
    }

   private:
    template <typename BasisLike>
    void rebuild_pools_(const BasisLike& basis, const Eigen::MatrixXd& A,
                        const std::vector<int>& N) {
        switch (current_strategy_) {
            case STEEPEST_EDGE:
                steepest_pricer_.build_primal_pool(basis, A, N);
                break;
            case DEVEX:
                devex_pricer_.build_primal_pool(basis, A, N);
                break;
            default:
                break;
        }
    }

    static Strategy map_strategy_(PricingStrategy strategy) {
        switch (strategy) {
            case PricingStrategy::STEEPEST_EDGE:
                return STEEPEST_EDGE;
            case PricingStrategy::DEVEX:
                return DEVEX;
            case PricingStrategy::PARTIAL_PRICING:
                return PARTIAL_PRICING;
            case PricingStrategy::MOST_NEGATIVE:
            default:
                return MOST_NEGATIVE;
        }
    }

    void track_performance_(double current_objective) {
        if (!first_call_) {
            double improvement = std::abs(current_objective - last_objective_);
            if ((int)recent_objectives_.size() >= 2 * dm_consts::kPerfWindow)
                recent_objectives_.pop_front();
            recent_objectives_.push_back(improvement);
        }
        last_objective_ = current_objective;
        first_call_ = false;
        // Optional: maintain average
        double sum = std::accumulate(recent_objectives_.begin(),
                                     recent_objectives_.end(), 0.0);
        const int cnt = (int)recent_objectives_.size();
        stats_.avg_improvement_per_iteration = (cnt > 0) ? (sum / cnt) : 0.0;
    }

    void adapt_strategy_() {
        if ((int)recent_objectives_.size() < 2 * dm_consts::kPerfWindow) return;

        ++stats_.strategy_switches;

        const double recent_avg =
            std::accumulate(recent_objectives_.end() - dm_consts::kPerfWindow,
                            recent_objectives_.end(), 0.0) /
            dm_consts::kPerfWindow;
        const double older_avg =
            std::accumulate(recent_objectives_.begin(),
                            recent_objectives_.begin() + dm_consts::kPerfWindow,
                            0.0) /
            dm_consts::kPerfWindow;

        if (recent_avg < older_avg / options_.improvement_factor) {
            if (n_ > 10000) {
                current_strategy_ = (current_strategy_ == PARTIAL_PRICING)
                                        ? DEVEX
                                        : PARTIAL_PRICING;
            } else if (!performance_history_.empty()) {
                const double avg_step =
                    std::accumulate(performance_history_.begin(),
                                    performance_history_.end(), 0.0) /
                    performance_history_.size();
                if (avg_step < 1e-10) {
                    current_strategy_ = STEEPEST_EDGE;
                } else {
                    current_strategy_ = static_cast<Strategy>(
                        (current_strategy_ + 1) % kNumStrategies);
                }
            } else {
                current_strategy_ = static_cast<Strategy>(
                    (current_strategy_ + 1) % kNumStrategies);
            }
        }
        iterations_since_switch_ = 0;
    }

    bool should_switch_strategy_(int /*iteration*/) {
        return (++iterations_since_switch_) >= options_.switch_threshold;
    }

    std::optional<int> partial_pricing_(const Eigen::VectorXd& rN,
                                        const std::vector<int>& N, double tol,
                                        int iteration) {
        const int block_size = std::max(
            options_.min_partial_block,
            (int)N.size() / std::max(1, options_.partial_block_factor));
        const int start_idx =
            (block_size > 0)
                ? ((iteration * block_size) % std::max(1, (int)N.size()))
                : 0;

        int best_idx = -1;
        double best_rc = 0.0;
        const int limit = std::min(block_size, (int)N.size());
        for (int k = 0; k < limit; ++k) {
            const int idx = (start_idx + k) % N.size();
            if (rN(idx) < -tol && rN(idx) < best_rc) {
                best_rc = rN(idx);
                best_idx = idx;
            }
        }
        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

    std::optional<int> most_negative_pricing_(const Eigen::VectorXd& rN,
                                              const std::vector<int>& /*N*/,
                                              double tol) {
        int best_idx = -1;
        double best_rc = 0.0;
        for (int k = 0; k < rN.size(); ++k) {
            if (rN(k) < -tol && rN(k) < best_rc) {
                best_rc = rN(k);
                best_idx = k;
            }
        }
        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

   private:
    Strategy current_strategy_;
    PricingOptions options_;
    int n_{0};
    mutable PricingStats stats_;

    // Sub-pricers
    SteepestEdgePricer steepest_pricer_;
    DevexPricer devex_pricer_;

    // Switching/perf state
    int iterations_since_switch_{0};
    double last_objective_{0.0};
    bool first_call_{true};
    std::deque<double> performance_history_;
    std::deque<double> recent_objectives_;
};
