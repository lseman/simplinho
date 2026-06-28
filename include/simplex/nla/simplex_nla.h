#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "simplex/factorization/simplex_lu.h"
#include "simplex/types/simplex_types.h"

namespace simplex::nla {

// ============================================================================
// SimplexIterate — complete factorization snapshot for fast backtrack
//       Matches HiGHS SimplexIterate{basis_, InvertibleRepresentation}.
//       putInvert() copies current state into snapshot for restore on backtrack.
//       getInvert() overwrites current state with snapshot.
// ============================================================================
struct SimplexIterate {
    bool valid_{false};
    std::vector<int> basis;
    // L factor (lower triangular, row-wise)
    std::vector<int> l_pivot_lookup;
    std::vector<int> l_start, l_index;
    std::vector<double> l_value;
    // LR storage (row-wise L^T for hyper-sparse solves)
    std::vector<int> lr_start, lr_index;
    std::vector<double> lr_value;
    // U factor (upper triangular)
    std::vector<int> u_pivot_lookup;
    std::vector<double> u_pivot_value; // diagonal
    std::vector<int> u_start, u_last_p, u_index;
    std::vector<double> u_value;
    // UR storage (upper part for update)
    std::vector<int> ur_start, ur_lastp, ur_space, ur_index;
    std::vector<double> ur_value;
    // PF updates
    std::vector<int> pf_start, pf_index, pf_pivot_index;
    std::vector<double> pf_value, pf_pivot_value;

    void clear() {
        valid_ = false;
        basis.clear();
        l_pivot_lookup.clear();
        l_start.clear();
        l_index.clear();
        l_value.clear();
        lr_start.clear();
        lr_index.clear();
        lr_value.clear();
        u_pivot_lookup.clear();
        u_pivot_value.clear();
        u_start.clear();
        u_last_p.clear();
        u_index.clear();
        u_value.clear();
        ur_start.clear();
        ur_lastp.clear();
        ur_space.clear();
        ur_index.clear();
        ur_value.clear();
        pf_start.clear();
        pf_index.clear();
        pf_pivot_index.clear();
        pf_value.clear();
        pf_pivot_value.clear();
    }
};

// ============================================================================
// NLA config — framework switching, price strategy, density tracking
// ============================================================================
struct NLAConfig {
    // Devex framework switching — rebuild when errors accumulate
    double framework_switch_threshold_{1.3862943611198906}; // log(4.0)
    int framework_switch_consecutive_{3};
    bool allow_framework_switch_{true};

    // Price strategy (HiGHS-style)
    enum class PriceStrategy {
        ColOnly = 0,
        RowSwitch,
        RowSwitchColSwitch
    } price_strategy_{PriceStrategy::ColOnly};
    bool starting_row_pricing_{false};

    // Density tracking for ftran/btran
    double ema_reach_ratio_{1.0};
    int factorization_count_{0};
    double build_synthetic_tick_{0.0};
};

// ============================================================================
// DevexFrameworkStats — tracks weight error accumulation for auto-rebuild
// ============================================================================
struct DevexFrameworkStats {
    double avg_log_error_{0.0};
    int consecutive_errors_{0};
    int total_framework_updates_{0};
};

// ============================================================================
// NLAStats — numerical linear algebra statistics
// ============================================================================
struct NLAStats {
    int factorizations_{0};
    int pf_updates_{0};
    int framework_rebuilds_{0};
    double total_ftran_ns_{0};
    double total_btran_ns_{0};
    double total_update_ns_{0};
};

// ============================================================================
// SimplexNLA — HiGHS HSimplexNla equivalent
//       The glue layer between simplex engines and factorizer.
//       Owns the factorizer and NLA metadata. Update storage itself belongs to
//       FTBasis/SparseForrestTomlinLU, matching HiGHS' HFactor ownership model:
//       update() chooses FT/PF/MPF/APF internally and ftran/btran apply that same
//       representation.
//
//       Usage:
//       1. nla.setup(rows, expected_density, config)
//       2. nla.setup_factor(A, basis, options)
//       3. FTRAN/BTRAN through nla.ftran()/nla.btran()
//       4. After each pivot: nla.update_basis(row, entering_col, entering_vector)
//       5. Framework: nla.record_framework_error(log_error)
// ============================================================================
class SimplexNLA {
  public:
    explicit SimplexNLA() = default;

    ~SimplexNLA() = default;

    // Setup NLA — called once at simplex start
    void setup(int num_rows, double expected_pf_density, const NLAConfig& config) {
        (void)expected_pf_density;
        config_ = config;
        num_rows_ = num_rows;
        config_.ema_reach_ratio_ = 1.0;
        config_.factorization_count_ = 0;
        config_.build_synthetic_tick_ = 0.0;
        stats_ = NLAStats{};
    }

    // Clear all factorization metadata and iterate data.
    void clear() {
        factor_.reset();
        simplex_iterate_.clear();
        framework_stats_.avg_log_error_ = 0.0;
        framework_stats_.consecutive_errors_ = 0;
        framework_stats_.total_framework_updates_ = 0;
        config_.starting_row_pricing_ = false;
    }

    // Update count — number of active factorizer updates since the last refactor.
    int update_count() const noexcept { return factor_ ? factor_->stats().eta_count : 0; }

    // =========================================================================
    // SimplexIterate — snapshot/restore for fast backtrack
    //       Delegates to FTBasis backtracking (put_backtracking_basis_ /
    //       try_backtrack_to_last_good). simplex_iterate_ stores the basis
    //       snapshot for warm-start reuse across solves.
    // ============================================================================

    // Save current factorization state — delegates to FTBasis.
    void putInvert() {
        if (factor_) {
            factor_->save_backtracking_basis_();
            simplex_iterate_.basis = factor_->basis();
            simplex_iterate_.valid_ = true;
        }
    }

    // Restore factorization state — delegates to FTBasis.
    // Returns true if a backtrack was performed.
    bool getInvert() {
        if (!factor_ || !simplex_iterate_.valid_)
            return false;
        std::vector<int> restored_basis;
        if (factor_->try_backtrack_to_last_good(restored_basis)) {
            simplex_iterate_.basis = restored_basis;
            return true;
        }
        return false;
    }

    // =========================================================================
    // Factorizer — NLA owns FTBasis (HiGHS HSimplexNla::factor_ pattern)
    // =========================================================================

    // Construct and factorize from a dense or sparse matrix + basis.
    template <class MatrixType>
    void setup_factor(const MatrixType& A, const std::vector<int>& basis,
                      const FTBasis::Options& opts) {
        factor_ = std::make_unique<FTBasis>(A, basis, opts);
        ++stats_.factorizations_;
        sync_update_stats_();
    }

    bool has_factor() const noexcept { return factor_ != nullptr; }

    FTBasis& factor() noexcept {
        assert(factor_ && "SimplexNLA: setup_factor() not called");
        return *factor_;
    }
    const FTBasis& factor() const noexcept {
        assert(factor_ && "SimplexNLA: setup_factor() not called");
        return *factor_;
    }

    // FTRAN: x = B^{-1} b
    template <class RhsT>
    HVector ftran(RhsT&& rhs, FTBasis::TranKind kind = FTBasis::TranKind::Unknown) const {
        return factor().solve_B(std::forward<RhsT>(rhs), kind);
    }

    HVector ftran_unit(int i, FTBasis::TranKind kind = FTBasis::TranKind::Unknown) const {
        return factor().solve_B_unit(i, kind);
    }

    // BTRAN: y = B^{-T} c
    template <class RhsT>
    HVector btran(RhsT&& rhs, FTBasis::TranKind kind = FTBasis::TranKind::Unknown) const {
        return factor().solve_BT(std::forward<RhsT>(rhs), kind);
    }

    HVector btran_unit(int i, FTBasis::TranKind kind = FTBasis::TranKind::Unknown) const {
        return factor().solve_BT_unit(i, kind);
    }

    // Basis update: FT/eta update after pivot
    template <class ColT> void update_basis(int j, int entering_col, ColT&& new_col) {
        factor().replace_column(j, entering_col, std::forward<ColT>(new_col));
        sync_update_stats_();
    }

    // Full refactor
    void invert() {
        factor().refactor();
        ++stats_.factorizations_;
        sync_update_stats_();
    }

    // Backtrack to last good snapshot (returns true if successful)
    bool try_backtrack(std::vector<int>& engine_basis) {
        return factor().try_backtrack_to_last_good(engine_basis);
    }

    // Check if snapshot exists
    bool has_snapshot() const noexcept { return simplex_iterate_.valid_; }

    // Set/Get snapshot state (used by simplex.h to copy data)
    void set_snapshot(const SimplexIterate& snap) {
        simplex_iterate_ = snap;
        simplex_iterate_.valid_ = true;
    }
    const SimplexIterate& snapshot() const noexcept { return simplex_iterate_; }

    // =========================================================================
    // Devex framework switching — HiGHS-style automatic rebuild
    // When weight errors accumulate beyond threshold, mark for rebuild.
    // =========================================================================

    // Call after each pivot to update framework stats
    void update_framework_stats(double new_weight, double old_weight);
    void record_framework_error(double log_error) noexcept;

    // Check if framework needs rebuilding
    bool needs_framework_rebuild() const noexcept;

    // Clear rebuild flag after rebuild
    void clear_framework_rebuild() noexcept { framework_stats_.consecutive_errors_ = 0; }

    // Get devex stats
    const DevexFrameworkStats& framework_stats() const noexcept { return framework_stats_; }
    bool allow_framework_switch() const noexcept { return config_.allow_framework_switch_; }

    // =========================================================================
    // Price strategy — row/column pricing switching
    // =========================================================================

    // Decide whether to use row or column pricing for row i
    bool use_row_pricing_for_row(int row_idx, const Eigen::VectorXd& yB, double tol) const noexcept;

    // After a pivot, update price strategy dynamically
    void update_price_strategy(const Eigen::VectorXd& yB, double tol) noexcept;

    // Public accessor for price strategy
    void set_price_strategy(NLAConfig::PriceStrategy strategy) noexcept;
    NLAConfig::PriceStrategy price_strategy() const noexcept { return config_.price_strategy_; }

    // =========================================================================
    // Density tracking — EMA watchdog for hyper-sparse
    // =========================================================================

    // Update EMA reach ratio after ftran/btran
    void update_ema_reach(int actual_reach, int vector_size) noexcept;

    // Get current EMA reach ratio
    double ema_reach_ratio() const noexcept { return config_.ema_reach_ratio_; }

    // Check if hyper-sparse path should be used
    bool should_use_hyper_sparse(int vector_size) const noexcept;

    // =========================================================================
    // Stats
    // =========================================================================

    NLAStats& stats() noexcept { return stats_; }
    const NLAStats& stats() const noexcept { return stats_; }

  private:
    void sync_update_stats_() noexcept {
        if (!factor_)
            return;
        stats_.pf_updates_ = factor_->stats().eta_count;
    }

    std::unique_ptr<FTBasis> factor_;
    SimplexIterate simplex_iterate_;
    int num_rows_{0};
    NLAConfig config_;
    DevexFrameworkStats framework_stats_;
    NLAStats stats_;
};

} // namespace simplex::nla
