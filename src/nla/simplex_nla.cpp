#include "simplex/nla/simplex_nla.h"

#include <algorithm>
#include <cmath>

namespace simplex::nla {

// =========================================================================
// Devex framework switching
// =========================================================================

void SimplexNLA::update_framework_stats(double new_weight, double old_weight) {
    if (!config_.allow_framework_switch_ || old_weight <= 0)
        return;
    double log_err = std::abs(std::log(new_weight / old_weight));
    record_framework_error(log_err);
}

void SimplexNLA::record_framework_error(double log_error) noexcept {
    if (!config_.allow_framework_switch_ || !std::isfinite(log_error))
        return;
    framework_stats_.total_framework_updates_++;
    double log_err = std::abs(log_error);
    framework_stats_.avg_log_error_ = 0.99 * framework_stats_.avg_log_error_ + 0.01 * log_err;
    if (framework_stats_.avg_log_error_ > config_.framework_switch_threshold_)
        ++framework_stats_.consecutive_errors_;
    else
        framework_stats_.consecutive_errors_ = 0;
}

bool SimplexNLA::needs_framework_rebuild() const noexcept {
    return framework_stats_.consecutive_errors_ >= config_.framework_switch_consecutive_;
}

// =========================================================================
// Price strategy
// =========================================================================

bool SimplexNLA::use_row_pricing_for_row(int row_idx, const Eigen::VectorXd& yB,
                                         double tol) const noexcept {
    if (config_.price_strategy_ == NLAConfig::PriceStrategy::ColOnly)
        return false;
    if (config_.price_strategy_ == NLAConfig::PriceStrategy::RowSwitch) {
        // Row pricing when duals are infeasible at this row.
        // Column pricing when duals are feasible.
        return yB(row_idx) < -tol;
    }
    // RowSwitchColSwitch — use row pricing if row density is low AND dual infeasible
    (void)row_idx;
    return false;
}

void SimplexNLA::update_price_strategy(const Eigen::VectorXd& yB, double tol) noexcept {
    if (config_.price_strategy_ != NLAConfig::PriceStrategy::RowSwitchColSwitch)
        return;
    if (config_.starting_row_pricing_) {
        int infeas_count = 0;
        int switched_count = 0;
        for (int i = 0; i < yB.size(); ++i) {
            if (yB(i) >= -tol)
                ++switched_count;
            else
                ++infeas_count;
        }
        if (infeas_count > 0 &&
            static_cast<double>(switched_count) / (infeas_count + switched_count) > 0.5) {
            config_.starting_row_pricing_ = false;
        }
    }
}

void SimplexNLA::set_price_strategy(NLAConfig::PriceStrategy strategy) noexcept {
    config_.price_strategy_ = strategy;
}

// =========================================================================
// Density tracking
// =========================================================================

void SimplexNLA::update_ema_reach(int actual_reach, int vector_size) noexcept {
    if (vector_size <= 0)
        return;
    double ratio = static_cast<double>(actual_reach) / vector_size;
    config_.ema_reach_ratio_ = 0.9 * config_.ema_reach_ratio_ + 0.1 * ratio;
}

bool SimplexNLA::should_use_hyper_sparse(int vector_size) const noexcept {
    if (vector_size <= 0)
        return false;
    // Use hyper-sparse path when EMA ratio is low (< 5%)
    return config_.ema_reach_ratio_ < 0.05;
}

} // namespace simplex::nla
