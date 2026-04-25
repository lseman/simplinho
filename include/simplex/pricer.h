#pragma once

#include "degeneracy.h"
#include "hvector.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace pricing_detail {

inline double clamp_positive(double value, double fallback = 1.0) {
    return (std::isfinite(value) && value > 0.0) ? value : fallback;
}

inline double log_weight_error(double updated_weight, double reference_weight) {
    const double updated = clamp_positive(updated_weight);
    const double reference = clamp_positive(reference_weight);
    return std::abs(std::log(updated / reference));
}

inline bool weight_log_error_ok(double updated_weight, double reference_weight,
                                double log_error_threshold) {
    return log_weight_error(updated_weight, reference_weight) <= std::max(0.0, log_error_threshold);
}

inline double diagonal_weight_surrogate(const Eigen::VectorXd& direction) {
    return 1.0 + direction.cwiseAbs().sum();
}

inline double edge_weight_from_direction(const Eigen::VectorXd& direction,
                                         const std::string& strategy) {
    const double dense_weight = 1.0 + direction.squaredNorm();
    if (strategy == "diagonal") {
        return diagonal_weight_surrogate(direction);
    }
    if (strategy == "dense_diagonal" || strategy == "hybrid") {
        return 0.5 * (dense_weight + diagonal_weight_surrogate(direction));
    }
    return dense_weight;
}

// Column FTRAN/BTRAN that preserves sparsity when A is a CSC sparse matrix.
// Dense A paths fall through to the generic solve_B/solve_BT with a dense column.
template <class BasisLike, class MatrixLike>
inline Eigen::VectorXd solve_B_column(const BasisLike& B, const MatrixLike& A, int j) {
    return B.solve_B(A.col(j).eval());
}

inline Eigen::VectorXd
solve_B_column(const auto& B, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A, int j) {
    return B.solve_B(A.col(j));
}

template <class BasisLike, class MatrixLike>
inline Eigen::VectorXd solve_BT_column(const BasisLike& B, const MatrixLike& A, int j) {
    return B.solve_BT(A.col(j).eval());
}

inline Eigen::VectorXd
solve_BT_column(const auto& B, const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A, int j) {
    return B.solve_BT(A.col(j));
}

template <class MatrixLike>
inline double column_dot(const MatrixLike& A, int j, const Eigen::VectorXd& v) {
    return A.col(j).dot(v);
}

template <class MatrixLike>
inline double column_dot(const MatrixLike& A, int j, const HVector& v) {
    if (!v.has_pattern())
        return column_dot(A, j, v.value);

    double dot = 0.0;
    for (int k = 0; k < v.count; ++k) {
        const int i = v.index[k];
        if (i >= 0 && i < v.value.size())
            dot += A(i, j) * v.value(i);
    }
    return dot;
}

inline double column_dot(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A, int j,
                         const Eigen::VectorXd& v) {
    double dot = 0.0;
    for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, j); it; ++it) {
        dot += it.value() * v(it.row());
    }
    return dot;
}

inline double column_dot(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A, int j,
                         const HVector& v) {
    return column_dot(A, j, v.value);
}

inline HVector pivot_column_delta(const HVector& s, int leave_rel) {
    HVector out;
    out.value = s.value;
    if (leave_rel >= 0 && leave_rel < out.value.size())
        out.value(leave_rel) -= 1.0;

    if (!s.has_pattern()) {
        out.drop_pattern();
        return out;
    }

    std::vector<char> present(static_cast<std::size_t>(s.size()), 0);
    out.index.reserve(static_cast<std::size_t>(s.count) + 1);
    for (int k = 0; k < s.count; ++k) {
        const int i = s.index[k];
        if (i < 0 || i >= s.size() || present[static_cast<std::size_t>(i)])
            continue;
        present[static_cast<std::size_t>(i)] = 1;
        out.index.push_back(i);
    }
    if (leave_rel >= 0 && leave_rel < s.size() &&
        !present[static_cast<std::size_t>(leave_rel)]) {
        out.index.push_back(leave_rel);
    }
    out.count = static_cast<int>(out.index.size());
    return out;
}

inline void normalize_pattern(HVector& v) {
    if (!v.has_pattern())
        return;

    std::vector<char> present(static_cast<std::size_t>(v.size()), 0);
    std::vector<int> unique;
    unique.reserve(static_cast<std::size_t>(v.count));
    for (int k = 0; k < v.count; ++k) {
        const int i = v.index[k];
        if (i < 0 || i >= v.size() || present[static_cast<std::size_t>(i)])
            continue;
        present[static_cast<std::size_t>(i)] = 1;
        unique.push_back(i);
    }
    v.index = std::move(unique);
    v.count = static_cast<int>(v.index.size());
}

inline HVector scaled_hvector(const HVector& v, double scale) {
    HVector out;
    out.value = v.value * scale;
    if (v.has_pattern()) {
        out.index = v.index;
        out.count = v.count;
        normalize_pattern(out);
    }
    return out;
}

inline void subtract_scaled(Eigen::VectorXd& target, const HVector& delta, double coeff) {
    if (!delta.has_pattern()) {
        target.noalias() -= delta.value * coeff;
        return;
    }
    for (int k = 0; k < delta.count; ++k) {
        const int i = delta.index[k];
        if (i >= 0 && i < target.size())
            target(i) -= delta.value(i) * coeff;
    }
}

inline void subtract_scaled(HVector& target, const HVector& delta, double coeff) {
    if (!delta.has_pattern()) {
        target.value.noalias() -= delta.value * coeff;
        target.drop_pattern();
        return;
    }

    for (int k = 0; k < delta.count; ++k) {
        const int i = delta.index[k];
        if (i >= 0 && i < target.value.size())
            target.value(i) -= delta.value(i) * coeff;
    }

    if (!target.has_pattern())
        return;

    std::vector<char> present(static_cast<std::size_t>(target.size()), 0);
    std::vector<int> merged;
    merged.reserve(static_cast<std::size_t>(target.count + delta.count));
    auto add = [&](int i) {
        if (i < 0 || i >= target.size() || present[static_cast<std::size_t>(i)])
            return;
        present[static_cast<std::size_t>(i)] = 1;
        merged.push_back(i);
    };
    for (int k = 0; k < target.count; ++k)
        add(target.index[k]);
    for (int k = 0; k < delta.count; ++k)
        add(delta.index[k]);
    target.index = std::move(merged);
    target.count = static_cast<int>(target.index.size());
}

} // namespace pricing_detail

// ============================================================================
// PrimalPricingBridge
//  - Adapter to thread DegeneracyManager signals into the primal pricer
//  - Does ABS <-> REL mapping here (we have N)
// ============================================================================
template <class PrimalPricer> struct PrimalPricingBridge {
    DegeneracyManager& dm;
    PrimalPricer& pricer;

    PrimalPricingBridge(DegeneracyManager& dm_, PrimalPricer& pr_) : dm(dm_), pricer(pr_) {}

    template <class BasisLike, class MatrixLike>
    std::optional<int> choose_primal_entering(const Eigen::VectorXd& rN, const std::vector<int>& N,
                                              double tol, int iteration, double current_objective,
                                              const BasisLike& basis, const MatrixLike& A,
                                              bool encourage_partial_pricing = false) {
        const auto& sig = dm.begin_pricing(current_objective, iteration, int(N.size()));

        const bool strategy_changed = pricer.apply_preferred_strategy(sig.preferred_strategy);

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

        auto entering_rel = pricer.choose_primal_entering(
            rN_eff, N, tol, iteration, current_objective, basis, A,
            encourage_partial_pricing || sig.encourage_partial_pricing);

        // Forbid list: ABS -> REL mapping
        if (entering_rel && !sig.forbid_abs_candidates.empty()) {
            std::unordered_set<int> forbid(sig.forbid_abs_candidates.begin(),
                                           sig.forbid_abs_candidates.end());
            if (forbid.count(N[*entering_rel])) {
                int best = -1;
                double best_rc = 0.0;
                for (int k = 0; k < (int)N.size(); ++k) {
                    if (rN_eff(k) < -tol && !forbid.count(N[k]) && rN_eff(k) < best_rc) {
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

    template <class MatrixLike>
    void after_primal_pivot(int leaving_rel, int entering_abs, int old_abs,
                            const Eigen::VectorXd& pivot_column, double alpha, double step_size,
                            const MatrixLike& A, const std::vector<int>& N,
                            double rc_improvement = 0.0) {
        pricer.update_after_primal_pivot(leaving_rel, entering_abs, old_abs, pivot_column, alpha,
                                         step_size, A, N);

        dm.after_pivot(leaving_rel, entering_abs, alpha, rc_improvement, step_size);

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
        int jN;            // absolute col index
        Eigen::VectorXd t; // B^{-1} a_j
        double weight;     // 1 + ||t||^2
    };

    explicit SteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000,
                                std::string weight_strategy = "dense",
                                double log_error_threshold = 1.3862943611198906)
        : pool_max_(pool_max), reset_freq_(reset_frequency),
          weight_strategy_(std::move(weight_strategy)), log_error_threshold_(log_error_threshold) {}

    template <class BasisLike, class MatrixLike>
    void build_primal_pool(const BasisLike& B, const MatrixLike& A, const std::vector<int>& N) {
        pool_.clear();
        initialize_positions_(N);
        const int take = (pool_max_ > 0) ? std::min<int>(pool_max_, (int)N.size()) : (int)N.size();
        pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            Entry e;
            e.jN = j;
            e.t = pricing_detail::solve_B_column(B, A, j); // sparse-aware FTRAN of column
            e.weight = pricing_detail::edge_weight_from_direction(e.t, weight_strategy_);
            set_position_(j, (int)pool_.size());
            pool_.push_back(std::move(e));
        }
        iter_count_ = 0;
        no_improvement_count_ = 0;
        partial_cursor_ = 0;
        need_rebuild_ = false;
    }

    // HiGHS-inspired: partial pricing with adaptive pool size
    // Only price a subset of nonbasic variables to save computation
    // Pool size grows if no improving column found
    std::optional<int> choose_primal_entering(const Eigen::VectorXd& rcN, const std::vector<int>& N,
                                              double tol, bool partial_pricing = false) {
        ++iter_count_;
        int best_rel = -1;
        double best_score = -1.0;

        const int total_candidates = (int)N.size();
        const int pool_size = partial_pool_size_(total_candidates, partial_pricing);
        const int scan_count =
            partial_pricing ? std::min(pool_size, total_candidates) : total_candidates;
        const int start =
            partial_pricing && total_candidates > 0 ? (partial_cursor_ % total_candidates) : 0;

        for (int k = 0; k < scan_count; ++k) {
            const int idx = partial_pricing ? ((start + k) % total_candidates) : k;
            if (rcN(idx) >= -tol)
                continue;
            const int j = N[idx];
            const double w = weight_for_column_(j);
            const double score = (rcN(idx) * rcN(idx)) / w;
            if (score > best_score) {
                best_score = score;
                best_rel = idx;
            }
        }

        advance_partial_cursor_(partial_pricing, total_candidates, start, scan_count);
        if (best_rel >= 0) {
            no_improvement_count_ = 0;
            return std::optional<int>(best_rel);
        }

        if (partial_pricing) {
            for (int k = 0; k < (int)N.size(); ++k) {
                if (rcN(k) >= -tol)
                    continue;
                const int j = N[k];
                const double w = weight_for_column_(j);
                const double score = (rcN(k) * rcN(k)) / w;
                if (score > best_score) {
                    best_score = score;
                    best_rel = k;
                }
            }
            if (best_rel >= 0)
                return std::optional<int>(best_rel);
            ++no_improvement_count_;
        }

        return std::nullopt;
    }

    template <class MatrixLike>
    void update_after_primal_pivot(int leave_rel, int e_abs, int old_abs, const Eigen::VectorXd& s,
                                   double alpha, const MatrixLike& /*A*/,
                                   const std::vector<int>& /*N*/,
                                   bool insert_leaver_into_pool = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol) {
            need_rebuild_ = true;
            return;
        }

        const double inv_alpha = 1.0 / alpha;

        // Update t, weight with error checking
        for (auto& E : pool_) {
            if (leave_rel < E.t.size()) {
                const double tr = E.t(leave_rel);
                if (tr != 0.0) {
                    const double old_weight = E.weight;
                    E.t.noalias() -= s * (tr * inv_alpha);
                    const double new_weight =
                        pricing_detail::edge_weight_from_direction(E.t, weight_strategy_);
                    const double log_error =
                        pricing_detail::log_weight_error(new_weight, old_weight);
                    if (new_weight < old_weight) {
                        average_log_low_weight_error_ =
                            0.99 * average_log_low_weight_error_ + 0.01 * log_error;
                    } else {
                        average_log_high_weight_error_ =
                            0.99 * average_log_high_weight_error_ + 0.01 * log_error;
                    }
                    if (pricing_detail::weight_log_error_ok(new_weight, old_weight,
                                                            log_error_threshold_)) {
                        E.weight = new_weight;
                    } else {
                        E.t.noalias() += s * (tr * inv_alpha);
                        need_rebuild_ = true;
                    }
                }
            }
        }

        // Remove entering from pool
        const int entering_pos = position_of_(e_abs);
        if (entering_pos >= 0) {
            const int idx = entering_pos;
            const int last = (int)pool_.size() - 1;
            if (idx != last) {
                set_position_(pool_[last].jN, idx);
                std::swap(pool_[idx], pool_[last]);
            }
            pool_.pop_back();
            set_position_(e_abs, -1);
        }

        // Optionally add leaving
        if (insert_leaver_into_pool) {
            Entry E;
            E.jN = old_abs;
            E.t = Eigen::VectorXd::Zero(s.size());
            if (leave_rel < E.t.size())
                E.t(leave_rel) = 1.0;
            E.t.noalias() -= s * inv_alpha;
            E.weight = pricing_detail::edge_weight_from_direction(E.t, weight_strategy_);

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
                set_position_(pool_[evict].jN, -1);
                pool_[evict] = std::move(E);
                set_position_(pool_[evict].jN, evict);
            } else {
                set_position_(E.jN, (int)pool_.size());
                pool_.push_back(std::move(E));
            }
        }

        ++iter_count_;
        if (need_rebuild_ || iter_count_ >= reset_freq_)
            need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }
    void clear_weight_error_stats() {
        average_log_low_weight_error_ = 0.0;
        average_log_high_weight_error_ = 0.0;
    }
    double average_log_weight_error_sum() const {
        return average_log_low_weight_error_ + average_log_high_weight_error_;
    }

  private:
    static int max_column_index_(const std::vector<int>& N) {
        int max_index = -1;
        for (const int j : N)
            max_index = std::max(max_index, j);
        return max_index;
    }

    void initialize_positions_(const std::vector<int>& N) {
        const int max_index = max_column_index_(N);
        pos_.assign(static_cast<size_t>(std::max(0, max_index) + 1), -1);
    }

    int position_of_(int j) const {
        if (j < 0 || static_cast<size_t>(j) >= pos_.size())
            return -1;
        return pos_[static_cast<size_t>(j)];
    }

    void set_position_(int j, int pos) {
        if (j < 0)
            return;
        if (static_cast<size_t>(j) >= pos_.size())
            pos_.resize(static_cast<size_t>(j) + 1, -1);
        pos_[static_cast<size_t>(j)] = pos;
    }

    double weight_for_column_(int j) const {
        const int pos = position_of_(j);
        return pos >= 0 ? pool_[static_cast<size_t>(pos)].weight : 1.0;
    }

    int partial_pool_size_(int total_candidates, bool partial_pricing) const {
        int pool_size = total_candidates;
        if (partial_pricing && pool_size > 10) {
            pool_size = std::min<int>(pool_size, std::max(10, pool_size / 10));
            if (no_improvement_count_ > 5)
                pool_size = std::min<int>(pool_size * 2, pool_size + 50);
        }
        return pool_size;
    }

    void advance_partial_cursor_(bool partial_pricing, int total_candidates, int start,
                                 int scan_count) {
        if (!partial_pricing || total_candidates <= 0)
            return;
        partial_cursor_ = (start + std::max(1, scan_count)) % total_candidates;
    }

    std::vector<Entry> pool_;
    std::vector<int> pos_;
    int pool_max_{0};
    int reset_freq_{1000};
    int iter_count_{0};
    bool need_rebuild_{false};
    std::string weight_strategy_{"dense"};
    double log_error_threshold_{1.3862943611198906};
    // HiGHS-inspired: track no-improvement iterations for adaptive pool sizing
    int no_improvement_count_{0};
    int partial_cursor_{0};
    double average_log_low_weight_error_{0.0};
    double average_log_high_weight_error_{0.0};
};

// ============================================================================
// DevexPricer (lightweight weights with error checking; API preserved)
// ============================================================================
class DevexPricer {
  public:
    explicit DevexPricer(double threshold = 0.99, int reset_frequency = 1000)
        : threshold_(threshold), reset_freq_(reset_frequency) {}

    // HiGHS-inspired: minimum fraction of previous weight to accept update
    static constexpr double kMinWeightAcceptRatio = 0.25;

    template <class BasisLike, class MatrixLike>
    void build_primal_pool(const BasisLike& /*B*/, const MatrixLike& /*A*/,
                           const std::vector<int>& N) {
        initialize_weights_(N);
        iter_count_ = 0;
        no_improvement_count_ = 0;
        partial_cursor_ = 0;
        need_rebuild_ = false;
    }

    // HiGHS-inspired: partial pricing with adaptive pool
    std::optional<int> choose_primal_entering(const Eigen::VectorXd& rcN, const std::vector<int>& N,
                                              double tol, bool partial_pricing = false) {
        ++iter_count_;
        if (iter_count_ % reset_freq_ == 0) {
            reset_weights_(N);
        }

        const int total_candidates = (int)N.size();
        const int pool_size = partial_pool_size_(total_candidates, partial_pricing);
        const int scan_count =
            partial_pricing ? std::min(pool_size, total_candidates) : total_candidates;
        const int start =
            partial_pricing && total_candidates > 0 ? (partial_cursor_ % total_candidates) : 0;
        int best_rel = -1;
        double best_crit = -1.0;
        for (int k = 0; k < scan_count; ++k) {
            const int idx = partial_pricing ? ((start + k) % total_candidates) : k;
            if (rcN(idx) >= -tol)
                continue;
            const int j = N[idx];
            const double w = weight_for_(j);
            const double crit = (rcN(idx) * rcN(idx)) / w;
            if (crit > best_crit) {
                best_crit = crit;
                best_rel = idx;
            }
        }

        advance_partial_cursor_(partial_pricing, total_candidates, start, scan_count);
        if (best_rel >= 0) {
            no_improvement_count_ = 0;
            return std::optional<int>(best_rel);
        }

        if (partial_pricing) {
            for (int k = 0; k < (int)N.size(); ++k) {
                if (rcN(k) >= -tol)
                    continue;
                const int j = N[k];
                const double w = weight_for_(j);
                const double crit = (rcN(k) * rcN(k)) / w;
                if (crit > best_crit) {
                    best_crit = crit;
                    best_rel = k;
                }
            }
            if (best_rel >= 0)
                return std::optional<int>(best_rel);
            ++no_improvement_count_;
        }

        return std::nullopt;
    }

    template <class MatrixLike>
    void update_after_primal_pivot(int leave_rel, int e_abs, int old_abs,
                                   const Eigen::VectorXd& pivot_column, double alpha,
                                   const MatrixLike& /*A*/, const std::vector<int>& N,
                                   bool /*insert_leaver_into_pool*/ = true) {
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol)
            return;

        // Entering weight: keep bounded to avoid runaway (HiGHS style)
        const double a2 = alpha * alpha;
        set_weight_(e_abs, std::min(std::max(a2, 1e-4), 1e6));

        // Update others (classic Devex-like) with error checking
        if (leave_rel < pivot_column.size()) {
            const double gamma_over_alpha = pivot_column(leave_rel) / alpha;
            const double add = gamma_over_alpha * gamma_over_alpha;
            for (int k = 0; k < (int)N.size(); ++k) {
                const int j = N[k];
                if (j == e_abs)
                    continue;
                double& w = weight_ref_(j);
                const double old_w = w;
                const double nw = w + add;
                w = std::max(nw, threshold_ * w);
                // HiGHS-inspired: check for weight corruption
                if (w < kMinWeightAcceptRatio * old_w && old_w > 0) {
                    // Weight dropped too far - mark for rebuild
                    need_rebuild_ = true;
                }
            }
        }

        // Ensure leaving has a slot
        (void)old_abs;
        if (!has_weight_(old_abs))
            set_weight_(old_abs, 1.0);
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }

  private:
    static int max_column_index_(const std::vector<int>& N) {
        int max_index = -1;
        for (const int j : N)
            max_index = std::max(max_index, j);
        return max_index;
    }

    void ensure_weight_capacity_(int j) {
        if (j < 0)
            return;
        const size_t required_size = static_cast<size_t>(j) + 1;
        if (required_size <= weights_.size())
            return;
        weights_.resize(required_size, 1.0);
        weight_present_.resize(required_size, 0);
    }

    void initialize_weights_(const std::vector<int>& N) {
        const int max_index = max_column_index_(N);
        weights_.assign(static_cast<size_t>(std::max(0, max_index) + 1), 1.0);
        weight_present_.assign(weights_.size(), 0);
        for (const int j : N) {
            if (j < 0)
                continue;
            weight_present_[static_cast<size_t>(j)] = 1;
        }
    }

    bool has_weight_(int j) const {
        return j >= 0 && static_cast<size_t>(j) < weight_present_.size() &&
               weight_present_[static_cast<size_t>(j)] != 0;
    }

    double weight_for_(int j) const {
        return has_weight_(j) ? weights_[static_cast<size_t>(j)] : 1.0;
    }

    double& weight_ref_(int j) {
        ensure_weight_capacity_(j);
        weight_present_[static_cast<size_t>(j)] = 1;
        return weights_[static_cast<size_t>(j)];
    }

    void set_weight_(int j, double value) {
        ensure_weight_capacity_(j);
        weight_present_[static_cast<size_t>(j)] = 1;
        weights_[static_cast<size_t>(j)] = value;
    }

    void reset_weights_(const std::vector<int>& N) {
        for (const int j : N) {
            if (j < 0)
                continue;
            ensure_weight_capacity_(j);
            weight_present_[static_cast<size_t>(j)] = 1;
            weights_[static_cast<size_t>(j)] = 1.0;
        }
    }

    int partial_pool_size_(int total_candidates, bool partial_pricing) const {
        int pool_size = total_candidates;
        if (partial_pricing && pool_size > 10) {
            pool_size = std::min<int>(pool_size, std::max(10, pool_size / 10));
            if (no_improvement_count_ > 5)
                pool_size = std::min<int>(pool_size * 2, pool_size + 50);
        }
        return pool_size;
    }

    void advance_partial_cursor_(bool partial_pricing, int total_candidates, int start,
                                 int scan_count) {
        if (!partial_pricing || total_candidates <= 0)
            return;
        partial_cursor_ = (start + std::max(1, scan_count)) % total_candidates;
    }

    std::vector<double> weights_;
    std::vector<char> weight_present_;
    double threshold_{0.99};
    int reset_freq_{1000};
    int iter_count_{0};
    int no_improvement_count_{0};
    int partial_cursor_{0};
    bool need_rebuild_{false};
};

// ============================================================================
// DualSteepestEdgePricer (exact dual row weights + maintained column weights)
// ============================================================================
class DualSteepestEdgePricer {
  public:
    struct WarmStartState {
        std::vector<double> row_weights;
    };

    struct DualEntry {
        int jN;
        Eigen::VectorXd w;  // approx B^{-T} a_j
        double dual_weight; // ||w||^2
    };

    struct RowEntry {
        HVector psi; // exact B^{-T} e_i for current basis row i
        double weight = 1.0;
    };

    struct LeavingChoice {
        int row = -1;
        HVector dual_row;
        double weight = 1.0;
    };

    explicit DualSteepestEdgePricer(int pool_max = 0, int reset_frequency = 1000,
                                    std::string weight_strategy = "dense",
                                    double log_error_threshold = 1.3862943611198906)
        : pool_max_(pool_max), reset_freq_(reset_frequency),
          weight_strategy_(std::move(weight_strategy)), log_error_threshold_(log_error_threshold) {}

    template <class BasisLike, class MatrixLike>
    void build_dual_pool(const BasisLike& B, const MatrixLike& A, const std::vector<int>& N) {
        dual_pool_.clear();
        dual_pos_.clear();
        row_pool_.clear();
        const int take = (pool_max_ > 0) ? std::min<int>(pool_max_, (int)N.size()) : (int)N.size();
        dual_pool_.reserve(take);
        for (int k = 0; k < take; ++k) {
            const int j = N[k];
            DualEntry e;
            e.jN = j;
            e.w = pricing_detail::solve_BT_column(B, A, j); // sparse-aware BTRAN of column
            e.dual_weight = pricing_detail::edge_weight_from_direction(e.w, weight_strategy_);
            dual_pos_[j] = (int)dual_pool_.size();
            dual_pool_.push_back(std::move(e));
        }
        row_pool_.resize(A.rows());
        for (int i = 0; i < A.rows(); ++i) {
            row_pool_[i].psi = HVector();
            row_pool_[i].weight = 1.0;
        }
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B, const Eigen::VectorXd& yB,
                                      double tol) const {
        LeavingChoice best;
        double best_score = -1.0;
        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol)
                continue;
            double weight = 1.0;
            if (i < (int)row_pool_.size())
                weight = row_pool_[i].weight;
            const double infeas = -yB(i);
            const double score = (infeas * infeas) / weight;
            if (score > best_score) {
                best_score = score;
                best.row = i;
                if (i < (int)row_pool_.size() && row_pool_[i].psi.has_pattern()) {
                    best.dual_row = row_pool_[i].psi;
                } else {
                    best.dual_row = B.solve_BT_unit(i);
                    pricing_detail::normalize_pattern(best.dual_row);
                    weight = std::max(1.0, pricing_detail::edge_weight_from_direction(
                                               best.dual_row, weight_strategy_));
                }
                best.weight = weight;
            }
        }
        return best;
    }

    template <class MatrixLike>
    void update_after_dual_pivot(int leave_rel, int e_abs, int old_abs, const HVector& s,
                                 double alpha, const MatrixLike& A, const std::vector<int>& /*N*/,
                                 const HVector& dual_row,
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

        HVector psi_before = dual_row;
        pricing_detail::normalize_pattern(psi_before);
        if (leave_rel < 0 || leave_rel >= s.size()) {
            need_rebuild_ = true;
            return;
        }

        if (!row_pool_.empty()) {
            if (leave_rel >= (int)row_pool_.size()) {
                need_rebuild_ = true;
                return;
            }

            auto update_row = [&](int i) {
                if (i == leave_rel)
                    return;
                if (row_pool_[i].psi.has_pattern()) {
                    const double coeff = s(i) / alpha;
                    if (coeff != 0.0) {
                        pricing_detail::subtract_scaled(row_pool_[i].psi, psi_before, coeff);
                    }
                    row_pool_[i].weight = std::max(1.0, pricing_detail::edge_weight_from_direction(
                                                            row_pool_[i].psi, weight_strategy_));
                }
            };
            if (s.has_pattern()) {
                std::vector<char> seen(row_pool_.size(), 0);
                for (int k = 0; k < s.count; ++k) {
                    const int i = s.index[k];
                    if (i < 0 || i >= (int)row_pool_.size() ||
                        seen[static_cast<std::size_t>(i)])
                        continue;
                    seen[static_cast<std::size_t>(i)] = 1;
                    update_row(i);
                }
            } else {
                for (int i = 0; i < (int)row_pool_.size(); ++i)
                    update_row(i);
            }
            row_pool_[leave_rel].psi = pricing_detail::scaled_hvector(psi_before, 1.0 / alpha);
            row_pool_[leave_rel].weight =
                std::max(1.0, pricing_detail::edge_weight_from_direction(row_pool_[leave_rel].psi,
                                                                         weight_strategy_));
        }

        Eigen::VectorXd e_r = Eigen::VectorXd::Zero(s.size());
        if (leave_rel >= 0 && leave_rel < e_r.size())
            e_r(leave_rel) = 1.0;
        const HVector s_minus_er = pricing_detail::pivot_column_delta(s, leave_rel);
        const double inv_alpha = 1.0 / alpha;

        // Exact rank-one update for w_j = B^{-T} a_j under a primal pivot:
        //   w'_j = w_j - psi_r * (((s - e_r)^T a_j) / alpha)
        // where psi_r = B^{-T} e_r from the pre-pivot basis.
        for (auto& E : dual_pool_) {
            if (E.jN == e_abs)
                continue;
            const double beta = pricing_detail::column_dot(A, E.jN, s_minus_er) * inv_alpha;
            if (beta != 0.0) {
                const double old_weight = E.dual_weight;
                pricing_detail::subtract_scaled(E.w, psi_before, beta);
                const double new_weight = std::max(
                    1.0, pricing_detail::edge_weight_from_direction(E.w, weight_strategy_));
                const double log_error = pricing_detail::log_weight_error(new_weight, old_weight);
                if (new_weight < old_weight) {
                    average_log_low_weight_error_ =
                        0.99 * average_log_low_weight_error_ + 0.01 * log_error;
                } else {
                    average_log_high_weight_error_ =
                        0.99 * average_log_high_weight_error_ + 0.01 * log_error;
                }
                if (!pricing_detail::weight_log_error_ok(new_weight, old_weight,
                                                         log_error_threshold_)) {
                    need_rebuild_ = true;
                } else {
                    E.dual_weight = new_weight;
                }
            }
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
            const double beta_old = pricing_detail::column_dot(A, old_abs, s_minus_er) * inv_alpha;
            if (beta_old != 0.0)
                pricing_detail::subtract_scaled(E.w, psi_before, beta_old);
            E.dual_weight =
                std::max(1.0, pricing_detail::edge_weight_from_direction(E.w, weight_strategy_));

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
        if (iter_count_ >= reset_freq_)
            need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }
    void clear_weight_error_stats() {
        average_log_low_weight_error_ = 0.0;
        average_log_high_weight_error_ = 0.0;
    }
    WarmStartState export_state() const {
        WarmStartState out;
        out.row_weights.reserve(row_pool_.size());
        for (const auto& row : row_pool_) {
            out.row_weights.push_back(row.weight);
        }
        return out;
    }
    bool import_state(const WarmStartState& state, int rows) {
        if (rows < 0 || static_cast<int>(state.row_weights.size()) != rows) {
            return false;
        }
        dual_pool_.clear();
        dual_pos_.clear();
        row_pool_.assign(static_cast<std::size_t>(rows), RowEntry{});
        for (int i = 0; i < rows; ++i) {
            row_pool_[static_cast<std::size_t>(i)].weight =
                std::max(1.0, state.row_weights[static_cast<std::size_t>(i)]);
        }
        iter_count_ = 0;
        no_improvement_count_ = 0;
        need_rebuild_ = false;
        clear_weight_error_stats();
        return true;
    }
    double average_log_weight_error_sum() const {
        return average_log_low_weight_error_ + average_log_high_weight_error_;
    }

  private:
    std::vector<DualEntry> dual_pool_;
    std::vector<RowEntry> row_pool_;
    std::unordered_map<int, int> dual_pos_;
    int pool_max_{0};
    int reset_freq_{1000};
    int iter_count_{0};
    int no_improvement_count_{0};
    bool need_rebuild_{false};
    std::string weight_strategy_{"dense"};
    double log_error_threshold_{1.3862943611198906};
    double average_log_low_weight_error_{0.0};
    double average_log_high_weight_error_{0.0};
};

// ============================================================================
// DualDevexPricer (dual leaving-row Devex with exact resets)
// ============================================================================
// DualDevexPricer with weight error checking
// ============================================================================
class DualDevexPricer {
  public:
    struct WarmStartState {
        std::vector<double> row_weights;
    };

    struct LeavingChoice {
        int row = -1;
        HVector dual_row;
        double weight = 1.0;
    };

    // HiGHS-inspired: minimum fraction of previous weight to accept
    static constexpr double kMinWeightAcceptRatio = 0.25;

    explicit DualDevexPricer(double threshold = 0.99, int reset_frequency = 200)
        : threshold_(threshold), reset_freq_(reset_frequency) {}

    template <class BasisLike, class MatrixLike>
    void build_dual_pool(const BasisLike& /*B*/, const MatrixLike& A, const std::vector<int>& /*N*/) {
        row_weights_.assign(A.rows(), 1.0);
        iter_count_ = 0;
        no_improvement_count_ = 0;
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B, const Eigen::VectorXd& yB,
                                      double tol) const {
        LeavingChoice best;
        double best_score = -1.0;
        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol)
                continue;
            const double weight = (i < (int)row_weights_.size()) ? row_weights_[i] : 1.0;
            const double infeas = -yB(i);
            const double score = (infeas * infeas) / std::max(1.0, weight);
            if (score > best_score) {
                best_score = score;
                best.row = i;
                best.weight = std::max(1.0, weight);
            }
        }

        if (best.row >= 0) {
            best.dual_row = B.solve_BT_unit(best.row);
            best.weight = std::max(1.0, best.dual_row.value.squaredNorm());
        }
        return best;
    }

    template <class MatrixLike>
    void update_after_dual_pivot(int leave_rel, int /*e_abs*/, int /*old_abs*/,
                                 const HVector& s, double alpha, const MatrixLike& /*A*/,
                                 const std::vector<int>& /*N*/, const HVector& /*dual_row*/,
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
        auto update_weight = [&](int i) {
            const double old_weight = row_weights_[i];
            const double sigma = (i == leave_rel) ? inv_alpha : s(i) * inv_alpha;
            const double candidate = sigma * sigma * pivot_weight;
            row_weights_[i] = std::max({1.0, threshold_ * row_weights_[i], candidate});
            // HiGHS-inspired: check for weight corruption
            if (row_weights_[i] < kMinWeightAcceptRatio * old_weight && old_weight > 0) {
                need_rebuild_ = true;
            }
        };
        if (s.has_pattern()) {
            for (double& weight : row_weights_)
                weight = std::max(1.0, threshold_ * weight);
            std::vector<char> seen(row_weights_.size(), 0);
            auto visit = [&](int i) {
                if (i < 0 || i >= (int)row_weights_.size() || i >= s.size() ||
                    seen[static_cast<std::size_t>(i)])
                    return;
                seen[static_cast<std::size_t>(i)] = 1;
                const double sigma = (i == leave_rel) ? inv_alpha : s(i) * inv_alpha;
                const double candidate = sigma * sigma * pivot_weight;
                row_weights_[i] = std::max(row_weights_[i], candidate);
            };
            visit(leave_rel);
            for (int k = 0; k < s.count; ++k)
                visit(s.index[k]);
        } else {
            for (int i = 0; i < (int)row_weights_.size() && i < s.size(); ++i)
                update_weight(i);
        }

        ++iter_count_;
        if (iter_count_ >= reset_freq_)
            need_rebuild_ = true;
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }
    WarmStartState export_state() const { return WarmStartState{row_weights_}; }
    bool import_state(const WarmStartState& state, int rows) {
        if (rows < 0 || static_cast<int>(state.row_weights.size()) != rows) {
            return false;
        }
        row_weights_ = state.row_weights;
        for (double& weight : row_weights_) {
            weight = std::max(1.0, weight);
        }
        iter_count_ = 0;
        no_improvement_count_ = 0;
        need_rebuild_ = false;
        return true;
    }

  private:
    std::vector<double> row_weights_;
    double threshold_{0.99};
    int reset_freq_{200};
    int iter_count_{0};
    int no_improvement_count_{0};
    bool need_rebuild_{false};
};

// ============================================================================
// DualRowPricer (HiGHS-inspired row pricing for dual simplex)
// Uses row pricing when rows are sparse, column pricing when dense
// ============================================================================
class DualRowPricer {
  public:
    struct WarmStartState {
        std::vector<double> row_weights;
        std::vector<char> prefer_row_pricing;
    };

    struct LeavingChoice {
        int row = -1;
        HVector dual_row;
        double weight = 1.0;
    };

    explicit DualRowPricer(int reset_frequency = 200, int density_threshold = 10,
                           std::string weight_strategy = "dense")
        : reset_freq_(reset_frequency), density_threshold_(density_threshold),
          weight_strategy_(std::move(weight_strategy)) {}

    template <class BasisLike, class MatrixLike>
    void build_dual_pool(const BasisLike& B, const MatrixLike& A, const std::vector<int>& N) {
        update_row_weights(B, A, N);
        iter_count_ = 0;
        need_rebuild_ = false;
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B, const Eigen::VectorXd& yB,
                                      double tol) const {
        LeavingChoice best;
        double best_score = -1.0;

        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol)
                continue;

            const double infeas = -yB(i);
            const bool use_row_pricing =
                (i < (int)prefer_row_pricing_.size()) && prefer_row_pricing_[i];
            const double weight =
                use_row_pricing && i < (int)row_weights_.size() ? row_weights_[i] : 1.0;
            const double score = (infeas * infeas) / std::max(1.0, weight);

            if (score > best_score) {
                best_score = score;
                best.row = i;
                best.weight = std::max(1.0, weight);
                if (use_row_pricing) {
                    best.dual_row = B.solve_BT_unit(i);
                }
            }
        }

        if (best.row >= 0) {
            // Always compute the dual row for the leaving row
            best.dual_row = B.solve_BT_unit(best.row);
            best.weight = std::max(
                1.0, pricing_detail::edge_weight_from_direction(best.dual_row, weight_strategy_));
        }

        return best;
    }

    template <class BasisLike, class MatrixLike>
    void update_row_weights(const BasisLike& B_inv, const MatrixLike& A,
                            const std::vector<int>& N) {
        row_weights_.resize(A.rows());
        prefer_row_pricing_.resize(A.rows(), false);
        for (int i = 0; i < A.rows(); ++i) {
            // Compute row weight as ||B^{-T} e_i||^2
            const Eigen::VectorXd psi_i = B_inv.solve_BT_unit(i);
            row_weights_[i] =
                std::max(1.0, pricing_detail::edge_weight_from_direction(psi_i, weight_strategy_));
            prefer_row_pricing_[i] =
                computeRowDensity(i, A, N) < static_cast<double>(density_threshold_);
        }
        iter_count_ = 0;
    }

    template <class MatrixLike>
    void update_after_dual_pivot(int /*leave_rel*/, int /*e_abs*/, int /*old_abs*/,
                                 const HVector& /*s*/, double alpha,
                                 const MatrixLike& /*A*/, const std::vector<int>& /*N*/,
                                 const HVector& /*dual_row*/,
                                 bool /*insert_leaver_into_pool*/ = true) {
        ++iter_count_;
        if (std::abs(alpha) < dm_consts::kDegenerateAlphaTol || iter_count_ >= reset_freq_) {
            need_rebuild_ = true;
        }
    }

    bool needs_rebuild() const { return need_rebuild_; }
    void clear_rebuild_flag() { need_rebuild_ = false; }
    WarmStartState export_state() const { return WarmStartState{row_weights_, prefer_row_pricing_}; }
    bool import_state(const WarmStartState& state, int rows) {
        if (rows < 0 || static_cast<int>(state.row_weights.size()) != rows ||
            static_cast<int>(state.prefer_row_pricing.size()) != rows) {
            return false;
        }
        row_weights_ = state.row_weights;
        for (double& weight : row_weights_) {
            weight = std::max(1.0, weight);
        }
        prefer_row_pricing_ = state.prefer_row_pricing;
        iter_count_ = 0;
        need_rebuild_ = false;
        return true;
    }

  private:
    template <class MatrixLike>
    int countRowNnz(int row, const MatrixLike& A, const std::vector<int>& N) const {
        int nz_count = 0;
        if constexpr (std::is_same_v<MatrixLike,
                                     Eigen::SparseMatrix<double, Eigen::ColMajor, int>>) {
            for (int j : N) {
                for (typename MatrixLike::InnerIterator it(A, j); it; ++it) {
                    if (it.row() == row) {
                        ++nz_count;
                        break;
                    }
                }
            }
        } else {
            for (int j : N) {
                if (std::abs(A(row, j)) > 0.0)
                    ++nz_count;
            }
        }
        return nz_count;
    }

    template <class MatrixLike>
    double computeRowDensity(int row, const MatrixLike& A, const std::vector<int>& N) const {
        return static_cast<double>(countRowNnz(row, A, N));
    }

    std::vector<double> row_weights_;
    std::vector<char> prefer_row_pricing_;
    int reset_freq_{200};
    int density_threshold_{10};
    int iter_count_{0};
    bool need_rebuild_{false};
    std::string weight_strategy_{"dense"};
};

// ============================================================================
// DualAdaptivePricer (dual-side pricing rule selection)
// ============================================================================
class DualAdaptivePricer {
  public:
    enum class Rule { SteepestEdge, Devex, RowPricing, MostInfeasible };

    struct WarmStartState {
        Rule active_rule = Rule::MostInfeasible;
        DualSteepestEdgePricer::WarmStartState steepest;
        DualDevexPricer::WarmStartState devex;
        DualRowPricer::WarmStartState row;
    };

    struct LeavingChoice {
        int row = -1;
        HVector dual_row;
        double weight = 1.0;
    };

    DualAdaptivePricer(std::string pricing_rule, int devex_reset_frequency,
                       int steepest_reset_frequency, bool partial_pricing = false,
                       std::string dual_pricing = "row", int row_pricing_threshold = 10,
                       std::string dual_edge_weight_strategy = "dense",
                       double dual_weight_log_error_threshold = 1.3862943611198906)
        : requested_rule_(std::move(pricing_rule)), partial_pricing_enabled_(partial_pricing),
          dual_pricing_preference_(std::move(dual_pricing)),
          row_pricing_threshold_(row_pricing_threshold),
          steepest_pricer_(0, steepest_reset_frequency, dual_edge_weight_strategy,
                           dual_weight_log_error_threshold),
          devex_pricer_(0.99, devex_reset_frequency),
          row_pricer_(devex_reset_frequency, row_pricing_threshold, dual_edge_weight_strategy),
          dual_weight_log_error_threshold_(dual_weight_log_error_threshold) {}

    template <class BasisLike, class MatrixLike>
    void build_dual_pool(const BasisLike& B, const MatrixLike& A, const std::vector<int>& N) {
        active_rule_ = select_rule_(A, N, A.rows());
        if (active_rule_ == Rule::SteepestEdge) {
            steepest_pricer_.clear_weight_error_stats();
            steepest_pricer_.build_dual_pool(B, A, N);
            devex_pricer_.clear_rebuild_flag();
            row_pricer_.clear_rebuild_flag();
        } else if (active_rule_ == Rule::Devex) {
            devex_pricer_.build_dual_pool(B, A, N);
            steepest_pricer_.clear_rebuild_flag();
            row_pricer_.clear_rebuild_flag();
        } else if (active_rule_ == Rule::RowPricing) {
            row_pricer_.build_dual_pool(B, A, N);
            steepest_pricer_.clear_rebuild_flag();
            devex_pricer_.clear_rebuild_flag();
        } else {
            steepest_pricer_.clear_rebuild_flag();
            devex_pricer_.clear_rebuild_flag();
            row_pricer_.clear_rebuild_flag();
        }
        need_rebuild_ = false;
    }

    template <class MatrixLike>
    static double average_row_nonzeros(const MatrixLike& A, const std::vector<int>& N) {
        const int m = A.rows();
        if (m <= 0 || N.empty())
            return 0.0;

        double total_nnz = 0.0;
        if constexpr (std::is_same_v<MatrixLike,
                                     Eigen::SparseMatrix<double, Eigen::ColMajor, int>>) {
            using SparseMatrixType = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
            std::vector<int> row_counts(m, 0);
            for (int j : N) {
                for (typename SparseMatrixType::InnerIterator it(A, j); it; ++it) {
                    const int i = it.row();
                    if (i >= 0 && i < m)
                        row_counts[i]++;
                }
            }
            for (int count : row_counts)
                total_nnz += count;
        } else {
            for (int i = 0; i < m; ++i) {
                int count = 0;
                for (int j : N) {
                    if (j >= 0 && j < A.cols() && std::abs(A(i, j)) > 0.0) {
                        ++count;
                    }
                }
                total_nnz += count;
            }
        }
        return total_nnz / static_cast<double>(m);
    }

    template <class MatrixLike>
    bool row_pricing_is_beneficial(const MatrixLike& A, const std::vector<int>& N) const {
        const double avg_row_nnz = average_row_nonzeros(A, N);
        if (partial_pricing_enabled_) {
            return avg_row_nnz <= static_cast<double>(row_pricing_threshold_) * 4.0;
        }
        return avg_row_nnz <= static_cast<double>(row_pricing_threshold_);
    }

    template <class BasisLike>
    LeavingChoice choose_dual_leaving(const BasisLike& B, const Eigen::VectorXd& yB,
                                      double tol) const {
        switch (active_rule_) {
            case Rule::SteepestEdge: {
                const auto choice = steepest_pricer_.choose_dual_leaving(B, yB, tol);
                return {choice.row, choice.dual_row, choice.weight};
            }
            case Rule::Devex: {
                const auto choice = devex_pricer_.choose_dual_leaving(B, yB, tol);
                return {choice.row, choice.dual_row, choice.weight};
            }
            case Rule::RowPricing: {
                const auto choice = row_pricer_.choose_dual_leaving(B, yB, tol);
                return {choice.row, choice.dual_row, choice.weight};
            }
            case Rule::MostInfeasible: {
                int best_row = -1;
                double best_infeas = 0.0;
                for (int i = 0; i < yB.size(); ++i) {
                    if (yB(i) >= -tol)
                        continue;
                    const double infeas = -yB(i);
                    if (best_row < 0 || infeas > best_infeas) {
                        best_row = i;
                        best_infeas = infeas;
                    }
                }

                LeavingChoice choice;
                choice.row = best_row;
                if (best_row >= 0) {
                    choice.dual_row = B.solve_BT_unit(best_row);
                    choice.weight = std::max(1.0, choice.dual_row.value.squaredNorm());
                }
                return choice;
            }
        }
        return {};
    }

    template <class MatrixLike>
    void update_after_dual_pivot(int leave_rel, int e_abs, int old_abs, const HVector& s,
                                 double alpha, const MatrixLike& A, const std::vector<int>& N,
                                 const HVector& dual_row,
                                 bool insert_leaver_into_pool = true) {
        switch (active_rule_) {
            case Rule::SteepestEdge:
                steepest_pricer_.update_after_dual_pivot(leave_rel, e_abs, old_abs, s, alpha, A, N,
                                                         dual_row, insert_leaver_into_pool);
                need_rebuild_ = steepest_pricer_.needs_rebuild();
                if (!need_rebuild_ && steepest_pricer_.average_log_weight_error_sum() >
                                          dual_weight_log_error_threshold_) {
                    active_rule_ = Rule::Devex;
                    need_rebuild_ = true;
                }
                break;
            case Rule::Devex:
                devex_pricer_.update_after_dual_pivot(leave_rel, e_abs, old_abs, s, alpha, A, N,
                                                      dual_row, insert_leaver_into_pool);
                need_rebuild_ = devex_pricer_.needs_rebuild();
                break;
            case Rule::RowPricing:
                row_pricer_.update_after_dual_pivot(leave_rel, e_abs, old_abs, s, alpha, A, N,
                                                    dual_row, insert_leaver_into_pool);
                need_rebuild_ = row_pricer_.needs_rebuild();
                break;
            case Rule::MostInfeasible:
                need_rebuild_ = (std::abs(alpha) < dm_consts::kDegenerateAlphaTol);
                break;
        }
    }

    bool needs_rebuild() const {
        return need_rebuild_ ||
               (active_rule_ == Rule::SteepestEdge && steepest_pricer_.needs_rebuild()) ||
               (active_rule_ == Rule::RowPricing && row_pricer_.needs_rebuild()) ||
               (active_rule_ == Rule::Devex && devex_pricer_.needs_rebuild());
    }

    void clear_rebuild_flag() {
        need_rebuild_ = false;
        steepest_pricer_.clear_rebuild_flag();
        devex_pricer_.clear_rebuild_flag();
        row_pricer_.clear_rebuild_flag();
    }

    const char* current_strategy_name() const {
        switch (active_rule_) {
            case Rule::SteepestEdge:
                return "dual_steepest_edge";
            case Rule::Devex:
                return "dual_devex";
            case Rule::RowPricing:
                return "dual_row_pricing";
            case Rule::MostInfeasible:
                return "dual_most_infeasible";
        }
        return "dual_unknown";
    }

    WarmStartState export_state() const {
        WarmStartState out;
        out.active_rule = active_rule_;
        out.steepest = steepest_pricer_.export_state();
        out.devex = devex_pricer_.export_state();
        out.row = row_pricer_.export_state();
        return out;
    }

    bool import_state(const WarmStartState& state, int rows) {
        active_rule_ = state.active_rule;
        bool ok = false;
        switch (active_rule_) {
            case Rule::SteepestEdge:
                ok = steepest_pricer_.import_state(state.steepest, rows);
                devex_pricer_.clear_rebuild_flag();
                row_pricer_.clear_rebuild_flag();
                break;
            case Rule::Devex:
                ok = devex_pricer_.import_state(state.devex, rows);
                steepest_pricer_.clear_rebuild_flag();
                row_pricer_.clear_rebuild_flag();
                break;
            case Rule::RowPricing:
                ok = row_pricer_.import_state(state.row, rows);
                steepest_pricer_.clear_rebuild_flag();
                devex_pricer_.clear_rebuild_flag();
                break;
            case Rule::MostInfeasible:
                ok = true;
                steepest_pricer_.clear_rebuild_flag();
                devex_pricer_.clear_rebuild_flag();
                row_pricer_.clear_rebuild_flag();
                break;
        }
        need_rebuild_ = !ok;
        return ok;
    }

  private:
    template <class MatrixLike>
    Rule select_rule_(const MatrixLike& A, const std::vector<int>& N, int basis_rows) const {
        if (dual_pricing_preference_ == "row")
            return Rule::RowPricing;
        if (dual_pricing_preference_ == "switch") {
            if (row_pricing_is_beneficial(A, N)) {
                return Rule::RowPricing;
            }
            return Rule::Devex;
        }
        if (requested_rule_ == "devex")
            return Rule::Devex;
        if (requested_rule_ == "most_negative")
            return Rule::MostInfeasible;
        if (requested_rule_ == "adaptive") {
            if (row_pricing_is_beneficial(A, N)) {
                return Rule::RowPricing;
            }
            return (basis_rows > 256) ? Rule::Devex : Rule::SteepestEdge;
        }
        return Rule::SteepestEdge;
    }

    std::string requested_rule_;
    bool partial_pricing_enabled_{false};
    std::string dual_pricing_preference_{"row"};
    int row_pricing_threshold_{10};
    Rule active_rule_{Rule::SteepestEdge};
    bool need_rebuild_{false};
    DualSteepestEdgePricer steepest_pricer_;
    DualDevexPricer devex_pricer_;
    DualRowPricer row_pricer_;
    double dual_weight_log_error_threshold_{1.3862943611198906};
};

// ============================================================================
// AdaptivePricer (strategy orchestration; API preserved)
// ============================================================================
class AdaptivePricer {
  public:
    enum Strategy { STEEPEST_EDGE = 0, DEVEX = 1, PARTIAL_PRICING = 2, MOST_NEGATIVE = 3 };
    static constexpr int kNumStrategies = 4;

    struct PricingOptions {
        Strategy initial_strategy = PARTIAL_PRICING;
        int switch_threshold = 50;
        int performance_window = 50;
        double improvement_factor = 1.2;
        int partial_block_factor = 20;
        int min_partial_block = 8;
        bool enable_adaptive_switching = true;
        int steepest_pool_max = 0;
        int steepest_reset_freq = 1000;
        int devex_reset_freq = 1000;
        std::string primal_edge_weight_strategy = "dense_diagonal";
        double primal_weight_log_error_threshold = 1.3862943611198906;
    };

    struct PricingStats {
        int total_pricing_calls{0};
        int strategy_switches{0};
        double avg_improvement_per_iteration{0.0};
        std::vector<int> strategy_usage_count{std::vector<int>(kNumStrategies, 0)};
    };

    explicit AdaptivePricer(int n) : AdaptivePricer(n, PricingOptions{}) {}

    AdaptivePricer(int n, const PricingOptions& opts)
        : current_strategy_(opts.initial_strategy), options_(opts), n_(n),
          steepest_pricer_(opts.steepest_pool_max, opts.steepest_reset_freq,
                           opts.primal_edge_weight_strategy,
                           opts.primal_weight_log_error_threshold),
          devex_pricer_(0.99, opts.devex_reset_freq), iterations_since_switch_(0),
          last_objective_(0.0), first_call_(true) {
        stats_.strategy_usage_count.assign(kNumStrategies, 0);
    }

    // Main pricing entry
    template <typename BasisLike, typename MatrixLike>
    std::optional<int> choose_primal_entering(const Eigen::VectorXd& rN, const std::vector<int>& N,
                                              double tol, int iteration, double current_objective,
                                              const BasisLike& basis, const MatrixLike& A,
                                              bool encourage_partial_pricing = false) {
        ++stats_.total_pricing_calls;
        ++stats_.strategy_usage_count[current_strategy_];

        track_performance_(current_objective);

        if (options_.enable_adaptive_switching && should_switch_strategy_(iteration)) {
            adapt_strategy_();
            rebuild_pools_(basis, A, N);
        }

        switch (current_strategy_) {
            case STEEPEST_EDGE:
                return steepest_pricer_.choose_primal_entering(rN, N, tol,
                                                               encourage_partial_pricing);
            case DEVEX:
                return devex_pricer_.choose_primal_entering(rN, N, tol, encourage_partial_pricing);
            case PARTIAL_PRICING:
                return partial_pricing_(rN, N, tol, iteration);
            case MOST_NEGATIVE:
                return most_negative_pricing_(rN, N, tol);
        }
        return std::nullopt;
    }

    // Build pools for all (cheap; preserves API)
    template <typename BasisLike, typename MatrixLike>
    void build_primal_pools(const BasisLike& basis, const MatrixLike& A,
                            const std::vector<int>& N) {
        steepest_pricer_.build_primal_pool(basis, A, N);
        devex_pricer_.build_primal_pool(basis, A, N);
    }

    bool apply_preferred_strategy(std::optional<PricingStrategy> preferred_strategy) {
        if (!preferred_strategy)
            return false;

        const Strategy next = map_strategy_(*preferred_strategy);
        if (next == current_strategy_)
            return false;
        current_strategy_ = next;
        iterations_since_switch_ = 0;
        ++stats_.strategy_switches;
        return true;
    }

    template <typename MatrixLike>
    void update_after_primal_pivot(int leaving_rel, int entering_abs, int old_abs,
                                   const Eigen::VectorXd& pivot_column, double alpha,
                                   double step_size, const MatrixLike& A,
                                   const std::vector<int>& N) {
        steepest_pricer_.update_after_primal_pivot(leaving_rel, entering_abs, old_abs, pivot_column,
                                                   alpha, A, N, true);
        devex_pricer_.update_after_primal_pivot(leaving_rel, entering_abs, old_abs, pivot_column,
                                                alpha, A, N, true);

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

    void clear_rebuild_flag() { steepest_pricer_.clear_rebuild_flag(); }

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
    template <typename BasisLike, typename MatrixLike>
    void rebuild_pools_(const BasisLike& basis, const MatrixLike& A, const std::vector<int>& N) {
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
        double sum = std::accumulate(recent_objectives_.begin(), recent_objectives_.end(), 0.0);
        const int cnt = (int)recent_objectives_.size();
        stats_.avg_improvement_per_iteration = (cnt > 0) ? (sum / cnt) : 0.0;
    }

    void adapt_strategy_() {
        if ((int)recent_objectives_.size() < 2 * dm_consts::kPerfWindow)
            return;

        ++stats_.strategy_switches;

        const double recent_avg = std::accumulate(recent_objectives_.end() - dm_consts::kPerfWindow,
                                                  recent_objectives_.end(), 0.0) /
                                  dm_consts::kPerfWindow;
        const double older_avg =
            std::accumulate(recent_objectives_.begin(),
                            recent_objectives_.begin() + dm_consts::kPerfWindow, 0.0) /
            dm_consts::kPerfWindow;

        if (recent_avg < older_avg / options_.improvement_factor) {
            if (n_ > 10000) {
                current_strategy_ =
                    (current_strategy_ == PARTIAL_PRICING) ? DEVEX : PARTIAL_PRICING;
            } else if (!performance_history_.empty()) {
                const double avg_step =
                    std::accumulate(performance_history_.begin(), performance_history_.end(), 0.0) /
                    performance_history_.size();
                if (avg_step < 1e-10) {
                    current_strategy_ = STEEPEST_EDGE;
                } else {
                    current_strategy_ =
                        static_cast<Strategy>((current_strategy_ + 1) % kNumStrategies);
                }
            } else {
                current_strategy_ = static_cast<Strategy>((current_strategy_ + 1) % kNumStrategies);
            }
        }
        iterations_since_switch_ = 0;
    }

    bool should_switch_strategy_(int /*iteration*/) {
        return (++iterations_since_switch_) >= options_.switch_threshold;
    }

    std::optional<int> partial_pricing_(const Eigen::VectorXd& rN, const std::vector<int>& N,
                                        double tol, int iteration) {
        const int block_size = std::max(options_.min_partial_block,
                                        (int)N.size() / std::max(1, options_.partial_block_factor));
        const int start_idx =
            (block_size > 0) ? ((iteration * block_size) % std::max(1, (int)N.size())) : 0;

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
        if (best_idx >= 0)
            return std::optional<int>(best_idx);

        // Fall back to a full scan if the sample block contains no improving column.
        // This preserves correctness for partial pricing mode.
        for (int k = 0; k < (int)N.size(); ++k) {
            if (rN(k) < -tol && rN(k) < best_rc) {
                best_rc = rN(k);
                best_idx = k;
            }
        }
        return (best_idx >= 0) ? std::optional<int>(best_idx) : std::nullopt;
    }

    std::optional<int> most_negative_pricing_(const Eigen::VectorXd& rN,
                                              const std::vector<int>& /*N*/, double tol) {
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
