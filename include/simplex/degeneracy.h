#pragma once

#include "aux.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================
// HiGHS-inspired cost perturbation utilities for degeneracy handling
// ============================================================================
namespace degeneracy_helpers {

// Add small random perturbations to cost vector to break degeneracy
// Perturbation magnitude is relative to cost magnitude (HiGHS-inspired)
inline void perturbCosts(Eigen::VectorXd& c, std::mt19937& rng,
                         double perturbation_multiplier = 1e-8) {
    std::normal_distribution<double> pert(0.0, 1.0);
    for (int i = 0; i < c.size(); ++i) {
        if (std::abs(c(i)) > 1e-14) {
            // Relative perturbation: perturbation * |c[i]|
            c(i) += pert(rng) * perturbation_multiplier * std::abs(c(i));
        }
    }
}

// Add absolute perturbation (useful when costs are near zero)
inline void perturbCostsAbsolute(Eigen::VectorXd& c, std::mt19937& rng,
                                 double perturbation_amount = 1e-10) {
    std::uniform_real_distribution<double> pert(-1.0, 1.0);
    for (int i = 0; i < c.size(); ++i) {
        if (std::abs(c(i)) < 1e-14) {
            c(i) += pert(rng) * perturbation_amount;
        }
    }
}

}  // namespace degeneracy_helpers

// ============================================================================
// DegeneracyManager
//  - API preserved broadly
//  - Signals are ABS-indexed; bridge maps to REL when necessary
// ============================================================================
class DegeneracyManager {
   public:
    struct BasisCycleEvent {
        bool repeated_basis{false};
        bool cycling_detected{false};
    };

    // Signals sent to the primal pricer (nudge only; pricer remains
    // authoritative)
    struct DegeneracySignals {
        std::optional<PricingStrategy> preferred_strategy{};
        bool request_pool_rebuild{false};

        // === ABSOLUTE column indices ===
        std::vector<int> forbid_abs_candidates;                // abs ids
        std::unordered_map<int, double> weight_overrides_abs;  // abs -> weight
        std::unordered_map<int, std::vector<double>>
            lex_order_abs;  // abs -> lex tuple

        bool encourage_partial_pricing{false};
        bool cycling_alert{false};
        int epoch{0};
    };

    enum class Method {
        PERTURBATION,
        LEXICOGRAPHIC,
        BLAND,
        STEEPEST_EDGE,
        DEVEX,
        DUAL_SIMPLEX,
        PRIMAL_DUAL,
        HYBRID
    };

    explicit DegeneracyManager(int rng_seed = 13,
                               Method default_method = Method::HYBRID)
        : rng_(rng_seed),
          default_method_(default_method),
          current_method_(default_method) {
        reset();
        method_perf_.reserve(8);
    }

    // ---------------- Backward-compatible core ----------------

    // Heuristic: a very small primal/dual step is degeneracy
    bool detect_degeneracy(double step, double deg_step_tol) {
        push_step_(step);
        const double tol = std::max(deg_step_tol, 1e-16 * scale_hint_);
        const bool deg = (step <= tol);
        if (deg) {
            ++deg_streak_;
            ++deg_total_;
            update_cycle_signal_();
        } else {
            if (deg_streak_ > 0) ++successes_recent_;
            deg_streak_ = 0;
            cycling_len_ = 0;
        }
        return deg;
    }

    // Legacy toggle (kept for compatibility - prefer pricer-based anti-cycling)
    bool should_apply_perturbation() const {
        return (deg_streak_ > std::max(10, adaptive_deg_threshold_) &&
                !perturb_on_);
    }

    void start_basis_history(const std::vector<int>& basis) {
        visited_basis_hashes_.clear();
        current_basis_hash_ = compute_basis_hash_(basis);
        visited_basis_hashes_.insert(current_basis_hash_);
        basis_hash_initialized_ = true;
        previous_iteration_basis_repeat_ = std::numeric_limits<int>::min() / 4;
    }

    BasisCycleEvent register_basis_change(const std::vector<int>& basis,
                                          int leaving_rel, int entering_abs,
                                          int iteration) {
        BasisCycleEvent event;
        if (leaving_rel < 0 || leaving_rel >= static_cast<int>(basis.size()) ||
            entering_abs < 0) {
            return event;
        }

        if (!basis_hash_initialized_) start_basis_history(basis);

        const int leaving_abs = basis[leaving_rel];
        if (leaving_abs < 0 || leaving_abs == entering_abs) {
            return event;
        }

        const std::uint64_t next_hash =
            current_basis_hash_ ^ basis_token_(leaving_abs) ^
            basis_token_(entering_abs);
        event.repeated_basis =
            visited_basis_hashes_.find(next_hash) != visited_basis_hashes_.end();
        if (event.repeated_basis) {
            ++basis_repeat_total_;
            if (iteration == previous_iteration_basis_repeat_ + 1) {
                event.cycling_detected = true;
                ++basis_cycle_total_;
                cycling_len_ = std::max(cycling_len_, 4);
            }
            previous_iteration_basis_repeat_ = iteration;
        }

        current_basis_hash_ = next_hash;
        visited_basis_hashes_.insert(current_basis_hash_);
        return event;
    }

    bool would_repeat_basis_change(const std::vector<int>& basis,
                                   int leaving_rel,
                                   int entering_abs) const {
        if (!basis_hash_initialized_ || leaving_rel < 0 ||
            leaving_rel >= static_cast<int>(basis.size()) || entering_abs < 0) {
            return false;
        }
        const int leaving_abs = basis[leaving_rel];
        if (leaving_abs < 0 || leaving_abs == entering_abs) {
            return false;
        }
        const std::uint64_t next_hash =
            current_basis_hash_ ^ basis_token_(leaving_abs) ^ basis_token_(entering_abs);
        return visited_basis_hashes_.find(next_hash) != visited_basis_hashes_.end();
    }

    // Compatibility: does not modify A,b,c anymore
    std::tuple<std::optional<Eigen::MatrixXd>, std::optional<Eigen::VectorXd>,
               std::optional<Eigen::VectorXd>>
    reset_perturbation() {
        perturb_on_ = false;
        return {std::nullopt, std::nullopt, std::nullopt};
    }

    // Compatibility: returns inputs (no-op)
    std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>
    apply_perturbation(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       const Eigen::VectorXd& c,
                       const std::vector<int>& /*basis*/, int /*iters*/) {
        return {A, b, c};
    }

    // ---------------- New hooks for the pricer loop ----------------

    // Call at start of pricing (DM stays abs-indexed; bridge will map)
    const DegeneracySignals& begin_pricing(
        double objective, int iter, int n_nonbasic,
        std::optional<double> illcond_hint = std::nullopt) {
        ++epoch_;
        iter_ = iter;
        last_obj_impr_ = std::max(0.0, last_obj_ - objective);
        last_obj_ = objective;
        nN_ = n_nonbasic;
        if (illcond_hint) cond_est_ = *illcond_hint;

        signals_ = {};
        signals_.epoch = epoch_;

        // Cycling suspicion -> nudge for primal-side robust criteria
        if (cycling_len_ >= 3 ||
            (deg_streak_ >= 8 && small_recent_improvement_())) {
            signals_.cycling_alert = true;
            signals_.encourage_partial_pricing = true;

            if (cond_est_ > 1e9) {
                signals_.preferred_strategy = PricingStrategy::PARTIAL_PRICING;
            } else {
                signals_.preferred_strategy =
                    (iter % 2 == 0) ? PricingStrategy::DEVEX
                                    : PricingStrategy::STEEPEST_EDGE;
            }
            signals_.request_pool_rebuild = true;
        }

        // Lex tie-break seeds when degeneracy is ongoing (ABS ids unknown here;
        // bridge will translate using N; we only set epsilon "shape" directive)
        if (deg_streak_ >= 3) {
            std::uniform_real_distribution<double> eps(-1e-14, 1e-14);
            // Fill "shape" hints keyed by pseudo-abs 0..(nN_-1); bridge will
            // rewrite keys to ABS. If you prefer, you can leave this empty and
            // let pricer do its own lex.
            for (int kRel = 0; kRel < nN_; ++kRel) {
                // store by pseudo key = kRel; bridge will remap using N[kRel]
                signals_.lex_order_abs[kRel] = {static_cast<double>(kRel),
                                                eps(rng_)};
            }
        }

        // Ill-conditioning -> gently increase weights (more conservative), keyed
        // by pseudo
        if (cond_est_ > 1e10) {
            for (int kRel = 0; kRel < nN_; ++kRel) {
                signals_.weight_overrides_abs[kRel] =
                    2.0;  // heavier (bridge remaps)
            }
        }

        // Recurrent suspects (forbid), only if degeneracy persists - ABS ids
        // already tracked
        trim_repeat_window_();
        if (!repeat_abs_block_.empty() && deg_streak_ >= 4) {
            signals_.forbid_abs_candidates.assign(repeat_abs_block_.begin(),
                                                  repeat_abs_block_.end());
        }

        return signals_;
    }

    // (Deprecated) Optional filter for REL candidates - no-op now that signals
    // are ABS-only.
    void filter_candidates_in_place(
        std::vector<int>& /*rel_candidates*/) const {
        // Intentionally empty: filtering happens in the bridge using ABS->REL
        // mapping.
    }

    // Call right after pivot is applied
    void after_pivot(int /*leaving_rel*/, int entering_abs, double step_alpha,
                     double rc_improvement, double step_norm) {
        last_rc_impr_ = rc_improvement;
        last_step_norm_ = step_norm;

        if (std::abs(step_alpha) <=
            dm_consts::kDegenerateAlphaTol) {  // degenerate pivot
            ++deg_streak_;
            ++deg_total_;
            push_repeat_(entering_abs);
            update_cycle_signal_();
        } else {
            ++successes_recent_;
            deg_streak_ = 0;
            cycling_len_ = 0;
            repeat_abs_block_.clear();
        }
        tune_thresholds_();
        scale_hint_ = std::max(scale_hint_, last_step_norm_);
    }

    // Lightweight stats for logging
    struct Stats {
        int degeneracy_streak{0};
        int degeneracy_total{0};
        int suspected_cycling{0};
        int repeated_basis_hits{0};
        int basis_cycle_hits{0};
        double cond_est{0.0};
        int adaptive_deg_threshold{10};
        int epoch{0};
    };
    Stats get_stats() const {
        return Stats{deg_streak_,
                     deg_total_,
                     cycling_len_,
                     basis_repeat_total_,
                     basis_cycle_total_,
                     cond_est_,
                     adaptive_deg_threshold_,
                     epoch_};
    }

    // Manual knobs
    void set_method(Method m) { current_method_ = m; }
    Method method() const { return current_method_; }

    void reset() {
        step_hist_.clear();
        deg_streak_ = 0;
        deg_total_ = 0;
        cycling_len_ = 0;
        successes_recent_ = 0;
        adaptive_deg_threshold_ = 10;
        scale_hint_ = 1.0;
        cond_est_ = 1.0;
        epoch_ = 0;
        last_obj_ = 0.0;
        last_obj_impr_ = 0.0;
        last_rc_impr_ = 0.0;
        last_step_norm_ = 0.0;
        repeat_window_.clear();
        repeat_freq_.clear();
        repeat_abs_block_.clear();
        nN_ = 0;
        perturb_on_ = false;
        current_method_ = default_method_;
        signals_ = {};
        basis_hash_initialized_ = false;
        current_basis_hash_ = 0;
        visited_basis_hashes_.clear();
        previous_iteration_basis_repeat_ = std::numeric_limits<int>::min() / 4;
        basis_repeat_total_ = 0;
        basis_cycle_total_ = 0;
    }

   private:
    static std::uint64_t mix_u64_(std::uint64_t x) {
        x += 0x9e3779b97f4a7c15ULL;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31);
    }
    static std::uint64_t basis_token_(int col) {
        return mix_u64_(static_cast<std::uint64_t>(
            static_cast<std::int64_t>(col) + 0x100000001b3LL));
    }
    static std::uint64_t compute_basis_hash_(const std::vector<int>& basis) {
        std::uint64_t hash = 0;
        for (const int col : basis) {
            if (col >= 0) hash ^= basis_token_(col);
        }
        return hash;
    }

    // --- small helpers ---
    void push_step_(double s) {
        if ((int)step_hist_.size() >= dm_consts::kStepHistCap)
            step_hist_.pop_front();
        step_hist_.push_back(s);
    }
    bool small_recent_improvement_() const {
        const bool obj_stalled = (last_obj_impr_ < dm_consts::kObjStallTol);
        const bool rc_stalled =
            (std::abs(last_rc_impr_) < dm_consts::kRcStallTol);
        return obj_stalled && rc_stalled;
    }
    void push_repeat_(int entering_abs) {
        if ((int)repeat_window_.size() >= dm_consts::kRepeatWinCap) {
            int old = repeat_window_.front();
            repeat_window_.pop_front();
            auto it = repeat_freq_.find(old);
            if (it != repeat_freq_.end()) {
                if (--(it->second) == 0) repeat_freq_.erase(it);
            }
        }
        repeat_window_.push_back(entering_abs);
        ++repeat_freq_[entering_abs];

        repeat_abs_block_.clear();
        for (auto& kv : repeat_freq_)
            if (kv.second >= dm_consts::kRepeatMinCount)
                repeat_abs_block_.push_back(kv.first);
    }
    void update_cycle_signal_() {
        if (deg_streak_ >= 6) cycling_len_ = std::max(cycling_len_, 3);
    }
    void tune_thresholds_() {
        if (successes_recent_ >= 5) {
            adaptive_deg_threshold_ =
                std::min(50, int(adaptive_deg_threshold_ * 1.1 + 1));
            successes_recent_ = 0;
        } else if (deg_streak_ >= adaptive_deg_threshold_) {
            adaptive_deg_threshold_ =
                std::max(4, int(adaptive_deg_threshold_ * 0.8));
        }
    }
    void trim_repeat_window_() {
        if (deg_streak_ == 0 && !repeat_window_.empty() &&
            (int)repeat_window_.size() > 8) {
            repeat_window_.erase(
                repeat_window_.begin(),
                repeat_window_.begin() + int(repeat_window_.size()) / 2);
        }
    }

   private:
    // RNG & config
    std::mt19937 rng_;
    Method default_method_;
    Method current_method_;

    // State
    int iter_{0};
    int epoch_{0};
    int nN_{0};

    // Degeneracy / cycling
    int deg_streak_{0};
    int deg_total_{0};
    int cycling_len_{0};
    int adaptive_deg_threshold_{10};
    int successes_recent_{0};
    bool perturb_on_{false};

    // Scales & telemetry
    double scale_hint_{1.0};
    double cond_est_{1.0};
    double last_obj_{0.0};
    double last_obj_impr_{0.0};
    double last_rc_impr_{0.0};
    double last_step_norm_{0.0};

    // Histories
    std::deque<double> step_hist_;
    std::deque<int> repeat_window_;
    std::unordered_map<int, int> repeat_freq_;
    std::vector<int> repeat_abs_block_;

    // (future) method performance bookkeeping
    struct Perf {
        int tries{0}, wins{0};
        double delta{0.0};
    };
    std::vector<Perf> method_perf_;

    // Outgoing signals
    DegeneracySignals signals_;

    // Basis-hash based cycling detection
    bool basis_hash_initialized_{false};
    std::uint64_t current_basis_hash_{0};
    std::unordered_set<std::uint64_t> visited_basis_hashes_;
    int previous_iteration_basis_repeat_{std::numeric_limits<int>::min() / 4};
    int basis_repeat_total_{0};
    int basis_cycle_total_{0};
};
