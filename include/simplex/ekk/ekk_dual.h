#pragma once
// HiGHS-style dual revised simplex ("EKK" engine).
//
// A port of the serial dual simplex in HiGHS (third_party/highs-source, MIT
// licence): HEkk (basis/work arrays, cost perturbation, phase-1 bounds,
// primal/dual computation), HEkkDual (phase 1/2 drivers, rebuild, CHUZR,
// PRICE/CHUZC, FTRAN updates, dual and primal updates), HEkkDualRow (bound
// flipping ratio test) and HEkkDualRHS (primal infeasibility bookkeeping).
// Dual steepest edge pricing; a compact primal phase 2 cleans up dual
// infeasibilities left after removing cost perturbations, where HiGHS runs
// HEkkPrimal. The basis matrix is factorized with the existing FTBasis
// (Markowitz LU with Forrest-Tomlin updates) on [A | I].

#include "simplex/ekk/ekk_lp.h"
#include "simplex/factorization/simplex_lu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef EKK_TRACE
#include <cstdio>
#define EKK_LOG(...) std::fprintf(stderr, __VA_ARGS__)
#else
#define EKK_LOG(...) ((void)0)
#endif

namespace simplex::ekk {

enum class ModelStatus {
    NotSet,
    Optimal,
    Infeasible,
    Unbounded,
    UnboundedOrInfeasible,
    ObjectiveBound,
    IterationLimit,
    SolveError,
};

inline const char* to_string(ModelStatus s) {
    switch (s) {
        case ModelStatus::NotSet: return "not_set";
        case ModelStatus::Optimal: return "optimal";
        case ModelStatus::Infeasible: return "infeasible";
        case ModelStatus::Unbounded: return "unbounded";
        case ModelStatus::UnboundedOrInfeasible: return "unbounded_or_infeasible";
        case ModelStatus::ObjectiveBound: return "objective_bound";
        case ModelStatus::IterationLimit: return "iteration_limit";
        case ModelStatus::SolveError: return "solve_error";
    }
    return "unknown";
}

struct Options {
    double primal_feasibility_tolerance = 1e-7;
    double dual_feasibility_tolerance = 1e-7;
    double dual_simplex_cost_perturbation_multiplier = 1.0;
    double pivot_growth_tolerance = 1e-9;   // dual_simplex_pivot_growth_tolerance
    double small_matrix_value = 1e-9;
    int update_limit = 5000;                // simplex_update_limit
    long long iteration_limit = std::numeric_limits<long long>::max();
    // Minimisation cutoff in the original objective units; +inf disables.
    double objective_bound = kInf;
    bool scale = true;
    int max_cleanup_rounds = 3;
    unsigned seed = 0x5eed;
};

// HiGHS SimplexBasis: basic_index per row, nonbasic_flag (1 = nonbasic) and
// nonbasic_move (+1 at lower moving up, -1 at upper moving down, 0 fixed/free/basic)
// over the num_col + num_row variables.
struct Basis {
    std::vector<int> basic_index;
    std::vector<int8_t> nonbasic_flag;
    std::vector<int8_t> nonbasic_move;

    bool valid_for(int num_col, int num_row) const {
        return static_cast<int>(basic_index.size()) == num_row &&
               static_cast<int>(nonbasic_flag.size()) == num_col + num_row &&
               static_cast<int>(nonbasic_move.size()) == num_col + num_row;
    }
};

struct Solution {
    std::vector<double> col_value;
    std::vector<double> col_dual; // reduced costs d = c - A'y
    std::vector<double> row_value;
    std::vector<double> row_dual; // y
    double objective = 0.0;
};

class DualSimplex {
  public:
    explicit DualSimplex(Options options = {}) : opt_(options) {}
    // The factorization keeps a pointer to full_, so the object stays put.
    DualSimplex(const DualSimplex&) = delete;
    DualSimplex& operator=(const DualSimplex&) = delete;

    // Load an LP; any existing basis is kept when the dimensions match.
    void load(const LpData& lp) {
        lp.validate();
        original_ = lp;
        scale_ = opt_.scale ? compute_equilibration_scale(lp) : LpScale{};
        lp_ = lp;
        apply_scale(lp_, scale_);
        n_ = lp_.num_col;
        m_ = lp_.num_row;
        tot_ = n_ + m_;
        ar_.build(lp_);
        build_full_matrix_();
        allocate_();
        if (!basis_.valid_for(n_, m_))
            basis_ = Basis{};
        factor_.reset();
        weights_valid_ = false;
    }

    void set_basis(const Basis& basis) {
        if (!basis.valid_for(n_, m_))
            throw std::invalid_argument("ekk::DualSimplex::set_basis: wrong dimensions");
        basis_ = basis;
        factor_.reset();
        weights_valid_ = false;
    }

    const Basis& basis() const noexcept { return basis_; }
    const Solution& solution() const noexcept { return solution_; }
    ModelStatus status() const noexcept { return status_; }
    long long iterations() const noexcept { return iteration_count_; }
    const LpData& lp() const noexcept { return original_; }

    // Bound changes keep the basis and factorization (HiGHS hot start).
    void change_col_bounds(int j, double lower, double upper) {
        original_.col_lower[j] = lower;
        original_.col_upper[j] = upper;
        const double c = scale_.active ? scale_.col[j] : 1.0;
        lp_.col_lower[j] = lower / c;
        lp_.col_upper[j] = upper / c;
    }
    void change_row_bounds(int i, double lower, double upper) {
        original_.row_lower[i] = lower;
        original_.row_upper[i] = upper;
        const double r = scale_.active ? scale_.row[i] : 1.0;
        lp_.row_lower[i] = lower * r;
        lp_.row_upper[i] = upper * r;
    }
    void set_objective_bound(double bound) noexcept { opt_.objective_bound = bound; }
    Options& options() noexcept { return opt_; }

    ModelStatus solve() {
        status_ = ModelStatus::NotSet;
        iteration_count_ = 0;
        visited_basis_.clear();
        bad_changes_.clear();
        previous_cycling_iteration_ = -2;
        if (m_ == 0)
            return solve_bound_only_();
        if (!basis_.valid_for(n_, m_))
            set_logical_basis_();
        if (!factor_ && !invert_()) {
            // An alien or singular basis: fall back to the logical basis.
            set_logical_basis_();
            if (!invert_()) {
                { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
                return status_;
            }
        }
        dual_then_cleanup_();
        if (status_ == ModelStatus::UnboundedOrInfeasible) {
            // As HiGHS: settle primal feasibility with the dual simplex on
            // zero costs, then let the primal simplex resolve the true costs.
            EKK_LOG("ekk: dual says unbounded_or_infeasible, solving feasibility LP\n");
            std::vector<double> cost(lp_.col_cost.size(), 0.0);
            std::swap(cost, lp_.col_cost);
            const double bound = opt_.objective_bound;
            opt_.objective_bound = kInf;
            status_ = ModelStatus::NotSet;
            dual_then_cleanup_();
            std::swap(cost, lp_.col_cost);
            opt_.objective_bound = bound;
            if (status_ == ModelStatus::Optimal) {
                status_ = ModelStatus::NotSet;
                for (int round = 0; status_ == ModelStatus::NotSet && round <= opt_.max_cleanup_rounds;
                     ++round) {
                    primal_cleanup_();
                    if (status_ == ModelStatus::NotSet)
                        dual_then_cleanup_();
                }
            }
        }
        if (status_ == ModelStatus::NotSet)
            status_ = verify_optimal_() ? ModelStatus::Optimal : ModelStatus::SolveError;
        extract_solution_();
        return status_;
    }

  private:
    void dual_then_cleanup_() {
        for (int round = 0;; ++round) {
            dual_solve_();
            if (status_ != ModelStatus::NotSet || round >= opt_.max_cleanup_rounds)
                break;
            // Dual infeasibilities remain after removing cost perturbation:
            // primal simplex from this primal feasible basis (HiGHS cleanup).
            EKK_LOG("ekk: primal cleanup round %d iter %lld\n", round, iteration_count_);
            primal_cleanup_();
            EKK_LOG("ekk: primal cleanup -> %s iter %lld\n", to_string(status_), iteration_count_);
            if (status_ != ModelStatus::NotSet)
                break;
        }
    }

    // Cleanup budget exhausted: accept the basis only if it is optimal for
    // the true costs within 10x the tolerances.
    bool verify_optimal_() {
        try {
            if (!reinvert_())
                return false;
            initialise_cost_(false);
            initialise_bound_(2);
            initialise_nonbasic_value_and_move_();
            compute_primal_();
            compute_dual_();
        } catch (const std::exception&) {
            return false;
        }
        const Infeasibility p = compute_primal_infeasible_();
        const Infeasibility d = compute_dual_infeasible_(false);
        return p.max <= 10 * opt_.primal_feasibility_tolerance &&
               d.max <= 10 * opt_.dual_feasibility_tolerance;
    }

    // ------------------------------------------------------------------ setup
    void build_full_matrix_() {
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(lp_.a_value.size() + m_);
        for (int j = 0; j < n_; ++j)
            for (int k = lp_.a_start[j]; k < lp_.a_start[j + 1]; ++k)
                trips.emplace_back(lp_.a_index[k], j, lp_.a_value[k]);
        for (int i = 0; i < m_; ++i)
            trips.emplace_back(i, n_ + i, 1.0);
        full_.resize(m_, tot_);
        full_.setFromTriplets(trips.begin(), trips.end());
        full_.makeCompressed();
    }

    void allocate_() {
        work_cost_.assign(tot_, 0.0);
        work_dual_.assign(tot_, 0.0);
        work_shift_.assign(tot_, 0.0);
        work_lower_.assign(tot_, 0.0);
        work_upper_.assign(tot_, 0.0);
        work_range_.assign(tot_, 0.0);
        work_value_.assign(tot_, 0.0);
        base_lower_.assign(m_, 0.0);
        base_upper_.assign(m_, 0.0);
        base_value_.assign(m_, 0.0);
        work_infeasibility_.assign(m_, 0.0);
        edge_weight_.assign(m_, 1.0);
        std::mt19937 rng(opt_.seed);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        random_value_.resize(tot_);
        for (double& v : random_value_)
            v = uni(rng);
        permutation_.resize(tot_);
        std::iota(permutation_.begin(), permutation_.end(), 0);
        std::shuffle(permutation_.begin(), permutation_.end(), rng);
        rng_ = std::mt19937(opt_.seed + 1);
        std::mt19937_64 hash_rng(opt_.seed + 2);
        hash_key_.resize(tot_);
        for (uint64_t& key : hash_key_)
            key = hash_rng();
        row_ep_.setup(m_);
        row_ap_.setup(n_);
        col_aq_.setup(m_);
        col_bfrt_.setup(m_);
        col_dse_.setup(m_);
        pack_index_.assign(tot_, 0);
        pack_value_.assign(tot_, 0.0);
        work_data_.assign(tot_, {0, 0.0});
        price_mark_.assign(n_, 0);
    }

    double var_lower_(int j) const {
        return j < n_ ? lp_.col_lower[j] : -lp_.row_upper[j - n_];
    }
    double var_upper_(int j) const {
        return j < n_ ? lp_.col_upper[j] : -lp_.row_lower[j - n_];
    }

    void set_logical_basis_() {
        basis_.basic_index.resize(m_);
        basis_.nonbasic_flag.assign(tot_, 1);
        basis_.nonbasic_move.assign(tot_, 0);
        for (int i = 0; i < m_; ++i) {
            basis_.basic_index[i] = n_ + i;
            basis_.nonbasic_flag[n_ + i] = 0;
        }
        set_nonbasic_move_();
        factor_.reset();
        weights_valid_ = false;
    }

    // HEkk::setNonbasicMove
    void set_nonbasic_move_() {
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j]) {
                basis_.nonbasic_move[j] = 0;
                continue;
            }
            const double lower = var_lower_(j), upper = var_upper_(j);
            int8_t move;
            if (lower == upper)
                move = 0;
            else if (std::isfinite(lower))
                move = std::isfinite(upper) ? (std::abs(lower) < std::abs(upper) ? 1 : -1) : 1;
            else if (std::isfinite(upper))
                move = -1;
            else
                move = 0;
            basis_.nonbasic_move[j] = move;
        }
    }

    bool logical_basis_() const {
        for (int i = 0; i < m_; ++i)
            if (basis_.basic_index[i] < n_)
                return false;
        return true;
    }

    // ------------------------------------------------------- factorization
    bool invert_() {
        try {
            FTBasis::Options fopt;
            if (const char* backend = std::getenv("EKK_LU_BACKEND"))
                fopt.sparse_backend = backend;
            factor_ = std::make_unique<FTBasis>(full_, basis_.basic_index, fopt);
            // FTBasis may accept a numerically singular basis; probe it.
            const Eigen::VectorXd ones = Eigen::VectorXd::Ones(m_);
            if (!factor_->solve_B(ones).value.allFinite() ||
                !factor_->solve_BT(ones).value.allFinite())
                throw std::runtime_error("ekk: singular basis");
        } catch (const std::exception& e) {
            EKK_LOG("ekk: invert failed after %d updates: %s\n", update_count_, e.what());
#ifdef EKK_TRACE
            {
                Eigen::MatrixXd B(m_, m_);
                for (int i = 0; i < m_; ++i)
                    B.col(i) = Eigen::VectorXd(full_.col(basis_.basic_index[i]));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(B);
                EKK_LOG("ekk:   dense rank %ld / %d, sigma_min %g sigma_max %g\n", (long)lu.rank(), m_,
                        svd.singularValues()(m_ - 1), svd.singularValues()(0));
                if (std::FILE* f = std::fopen("bad_basis.txt", "w")) {
                    std::fprintf(f, "%d\n", m_);
                    for (int i = 0; i < m_; ++i)
                        for (int r = 0; r < m_; ++r)
                            std::fprintf(f, "%.17g%c", B(r, i), r + 1 == m_ ? '\n' : ' ');
                    std::fclose(f);
                }
            }
#endif
            factor_.reset();
            return false;
        }
        update_count_ = 0;
        compute_basis_hash_();
        visited_basis_.insert(basis_hash_);
        backtrack_basis_ = basis_;
        backtrack_weights_ = edge_weight_;
        has_backtrack_ = true;
        return true;
    }

    // Refactorize the current basis; on singularity restore the last
    // nonsingular basis and halve the update limit (HEkk::getNonsingularInverse),
    // and as a last resort the logical basis.
    bool reinvert_() {
        if (invert_())
            return true;
        backtracking_ = true;
        if (has_backtrack_ && basis_.basic_index != backtrack_basis_.basic_index) {
            update_limit_ = std::max(1, update_count_ / 2);
            basis_ = backtrack_basis_;
            edge_weight_ = backtrack_weights_;
            if (invert_())
                return true;
        }
        set_logical_basis_();
        std::fill(edge_weight_.begin(), edge_weight_.end(), 1.0);
        weights_valid_ = true;
        return invert_();
    }

    void ftran_(WorkVector& v) { solve_(v, true); }
    void btran_(WorkVector& v) { solve_(v, false); }

    // FTRAN/BTRAN in place. A known RHS pattern (count >= 0) takes the
    // factorization's hyper-sparse path; the array moves into and out of the
    // HVector without copying and the result keeps the solve's reach pattern.
    void solve_(WorkVector& v, bool forward) {
        HVector in;
        if (v.count >= 0) {
            v.index.resize(v.count);
            in = HVector(std::move(v.array), std::move(v.index));
        } else {
            in = HVector(std::move(v.array));
        }
        HVector out;
        try {
            out = forward ? factor_->solve_B(in, FTBasis::TranKind::ColAq)
                          : factor_->solve_BT(in, FTBasis::TranKind::RowEp);
        } catch (...) {
            v.setup(m_);
            throw;
        }
        v.array = std::move(out.value);
        if (out.has_pattern()) {
            v.index = std::move(out.index);
            v.count = static_cast<int>(v.index.size());
            v.tight_pattern();
        } else {
            v.tight();
        }
    }

    // Add multiple * column j of [A | I] into the dense array of v.
    // Keeps v's pattern when it is known (count >= 0).
    void collect_column_(WorkVector& v, int j, double multiple) const {
        auto add = [&](int i, double value) {
            if (v.count >= 0 && v.array[i] == 0.0) {
                v.index.push_back(i);
                ++v.count;
            }
            v.array[i] += value;
            if (v.array[i] == 0.0)
                v.array[i] = 1e-300; // keep the slot so the pattern stays exact
        };
        if (j < n_) {
            for (int k = lp_.a_start[j]; k < lp_.a_start[j + 1]; ++k)
                add(lp_.a_index[k], multiple * lp_.a_value[k]);
        } else {
            add(j - n_, multiple);
        }
    }

    double column_dot_(int j, const WorkVector& v) const {
        if (j >= n_)
            return v.array[j - n_];
        double s = 0.0;
        for (int k = lp_.a_start[j]; k < lp_.a_start[j + 1]; ++k)
            s += lp_.a_value[k] * v.array[lp_.a_index[k]];
        return s;
    }

    // ----------------------------------------------- costs, bounds, values
    // HEkk::initialiseCost
    void initialise_cost_(bool perturb) {
        for (int j = 0; j < n_; ++j)
            work_cost_[j] = lp_.col_cost[j];
        for (int j = n_; j < tot_; ++j)
            work_cost_[j] = 0.0;
        std::fill(work_shift_.begin(), work_shift_.end(), 0.0);
        costs_shifted_ = false;
        costs_perturbed_ = false;
        if (!perturb || opt_.dual_simplex_cost_perturbation_multiplier == 0.0 ||
            !allow_cost_perturbation_)
            return;
        double max_abs_cost = 0.0;
        for (int j = 0; j < n_; ++j)
            max_abs_cost = std::max(max_abs_cost, std::abs(work_cost_[j]));
        if (max_abs_cost > 100.0)
            max_abs_cost = std::sqrt(std::sqrt(max_abs_cost));
        double boxed_rate = 0.0;
        for (int j = 0; j < tot_; ++j)
            boxed_rate += work_range_[j] < 1e30;
        boxed_rate /= tot_;
        if (boxed_rate < 0.01)
            max_abs_cost = std::min(max_abs_cost, 1.0);
        const double base = opt_.dual_simplex_cost_perturbation_multiplier * 5e-7 * max_abs_cost;
        for (int j = 0; j < n_; ++j) {
            const double lower = lp_.col_lower[j], upper = lp_.col_upper[j];
            const double xpert = (1.0 + random_value_[j]) * (std::abs(work_cost_[j]) + 1.0) * base;
            if (!std::isfinite(lower) && !std::isfinite(upper)) {
            } else if (!std::isfinite(upper)) {
                work_cost_[j] += xpert;
            } else if (!std::isfinite(lower)) {
                work_cost_[j] -= xpert;
            } else if (lower != upper) {
                work_cost_[j] += work_cost_[j] >= 0.0 ? xpert : -xpert;
            }
        }
        const double row_base = opt_.dual_simplex_cost_perturbation_multiplier * 1e-12;
        for (int j = n_; j < tot_; ++j)
            work_cost_[j] += (0.5 - random_value_[j]) * row_base;
        costs_perturbed_ = true;
    }

    // HEkk::initialiseBound for the dual simplex
    void initialise_bound_(int phase) {
        free_vars_.clear();
        for (int j = 0; j < tot_; ++j) {
            work_lower_[j] = var_lower_(j);
            work_upper_[j] = var_upper_(j);
            work_range_[j] = work_upper_[j] - work_lower_[j];
            if (!std::isfinite(work_lower_[j]) && !std::isfinite(work_upper_[j]))
                free_vars_.push_back(j);
        }
        if (phase == 2)
            return;
        free_vars_.clear(); // phase-1 bounds box every variable
        // Dual phase-1 bounds: the dual objective is minus the sum of dual
        // infeasibilities, so a dual-feasible nonbasic sits at value 0.
        for (int j = 0; j < tot_; ++j) {
            const bool has_l = std::isfinite(work_lower_[j]);
            const bool has_u = std::isfinite(work_upper_[j]);
            if (!has_l && !has_u) {
                work_lower_[j] = -1000.0;
                work_upper_[j] = 1000.0;
            } else if (!has_l) {
                work_lower_[j] = -1.0;
                work_upper_[j] = 0.0;
            } else if (!has_u) {
                work_lower_[j] = 0.0;
                work_upper_[j] = 1.0;
            } else {
                work_lower_[j] = 0.0;
                work_upper_[j] = 0.0;
            }
            work_range_[j] = work_upper_[j] - work_lower_[j];
        }
    }

    // HEkk::initialiseNonbasicValueAndMove
    void initialise_nonbasic_value_and_move_() {
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j]) {
                basis_.nonbasic_move[j] = 0;
                continue;
            }
            const double lower = work_lower_[j], upper = work_upper_[j];
            const int8_t original = basis_.nonbasic_move[j];
            double value;
            int8_t move;
            if (lower == upper) {
                value = lower;
                move = 0;
            } else if (std::isfinite(lower)) {
                if (std::isfinite(upper)) {
                    if (original == -1) {
                        value = upper;
                        move = -1;
                    } else {
                        value = lower;
                        move = 1;
                    }
                } else {
                    value = lower;
                    move = 1;
                }
            } else if (std::isfinite(upper)) {
                value = upper;
                move = -1;
            } else {
                value = 0.0;
                move = 0;
            }
            basis_.nonbasic_move[j] = move;
            work_value_[j] = value;
        }
    }

    // HEkk::computePrimal: x_B = -B^{-1} sum_{j nonbasic} a_j x_j
    void compute_primal_() {
        WorkVector v;
        v.setup(m_);
        for (int j = 0; j < tot_; ++j)
            if (basis_.nonbasic_flag[j] && work_value_[j] != 0.0)
                collect_column_(v, j, work_value_[j]);
        v.count = -1;
        ftran_(v);
        for (int i = 0; i < m_; ++i) {
            const int var = basis_.basic_index[i];
            base_value_[i] = -v.array[i];
            base_lower_[i] = work_lower_[var];
            base_upper_[i] = work_upper_[var];
        }
    }

    // HEkk::computeDual: d = c - [A I]' B^{-T} c_B
    void compute_dual_() {
        WorkVector y;
        y.setup(m_);
        y.count = -1;
        bool any = false;
        for (int i = 0; i < m_; ++i) {
            const int var = basis_.basic_index[i];
            const double value = work_cost_[var] + work_shift_[var];
            if (value != 0.0) {
                y.array[i] = value;
                any = true;
            }
        }
        for (int j = 0; j < tot_; ++j)
            work_dual_[j] = work_cost_[j] + work_shift_[j];
        if (!any)
            return;
        btran_(y);
        for (int j = 0; j < n_; ++j)
            work_dual_[j] -= column_dot_(j, y);
        for (int i = 0; i < m_; ++i)
            work_dual_[n_ + i] -= y.array[i];
    }

    struct Infeasibility {
        int num = 0;
        double max = 0.0;
        double sum = 0.0;
    };

    // HEkk::computeSimplexDualInfeasible (fixed variables contribute through
    // nonbasic_move = 0 only when free)
    Infeasibility compute_dual_infeasible_(bool ignore_fixed) const {
        Infeasibility out;
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const double dual = work_dual_[j];
            const double lower = work_lower_[j], upper = work_upper_[j];
            double infeasibility;
            if (!std::isfinite(lower) && !std::isfinite(upper))
                infeasibility = std::abs(dual);
            else if (ignore_fixed || lower != upper)
                infeasibility = -basis_.nonbasic_move[j] * dual;
            else
                infeasibility = 0.0;
            if (infeasibility > 0.0) {
                if (infeasibility >= opt_.dual_feasibility_tolerance)
                    ++out.num;
                out.max = std::max(out.max, infeasibility);
                out.sum += infeasibility;
            }
        }
        return out;
    }

    Infeasibility compute_primal_infeasible_() const {
        Infeasibility out;
        const double tol = opt_.primal_feasibility_tolerance;
        for (int i = 0; i < m_; ++i) {
            const double v = base_value_[i];
            double inf = 0.0;
            if (v < base_lower_[i] - tol)
                inf = base_lower_[i] - v;
            else if (v > base_upper_[i] + tol)
                inf = v - base_upper_[i];
            if (inf > 0.0) {
                ++out.num;
                out.max = std::max(out.max, inf);
                out.sum += inf;
            }
        }
        return out;
    }

    double compute_dual_objective_(int phase) const {
        double obj = 0.0;
        for (int j = 0; j < tot_; ++j)
            if (basis_.nonbasic_flag[j])
                obj += work_value_[j] * work_dual_[j];
        if (phase != 1)
            obj += lp_.offset;
        return obj;
    }

    void flip_bound_(int j) {
        const int8_t move = basis_.nonbasic_move[j] = static_cast<int8_t>(-basis_.nonbasic_move[j]);
        work_value_[j] = move == 1 ? work_lower_[j] : work_upper_[j];
    }

    // HEkkDual::correctDualInfeasibilities: flips for fixed/boxed, cost shifts
    // for one-sided; returns the number of free-variable dual infeasibilities.
    int correct_dual_infeasibilities_() {
        int free_infeasibility_count = 0;
        const double tol = opt_.dual_feasibility_tolerance;
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const double lower = work_lower_[j], upper = work_upper_[j];
            const double dual = work_dual_[j];
            const int move = basis_.nonbasic_move[j];
            const bool fixed = lower == upper;
            const bool boxed = std::isfinite(lower) && std::isfinite(upper);
            if (!std::isfinite(lower) && !std::isfinite(upper)) {
                if (std::abs(dual) >= tol)
                    ++free_infeasibility_count;
                continue;
            }
            const double infeasibility = -move * dual;
            if (infeasibility < tol)
                continue;
            if (fixed || (boxed && !force_phase2_)) {
                flip_bound_(j);
                continue;
            }
            costs_shifted_ = true;
            const double new_dual = (move == 1 ? 1.0 : -1.0) * (1.0 + uni(rng_)) * tol;
            const double shift = new_dual - dual;
            work_dual_[j] = new_dual;
            work_cost_[j] += shift;
        }
        force_phase2_ = false;
        return free_infeasibility_count;
    }

    // ------------------------------------------------------------- DSE
    void compute_dse_weights_() {
        WorkVector e;
        e.setup(m_);
        for (int i = 0; i < m_; ++i) {
            e.clear();
            e.set_unit(i);
            btran_(e);
            edge_weight_[i] = e.norm2();
        }
    }

    // ------------------------------------------------------- dual driver
    // HEkkDual::solve. Leaves status_ NotSet when dual infeasibilities remain
    // after removing the cost perturbation (primal cleanup required).
    void dual_solve_() {
        taboo_rows_.clear();
        allow_cost_perturbation_ = true;
        initialise_cost_(false);
        initialise_bound_(2);
        initialise_nonbasic_value_and_move_();
        compute_primal_();
        compute_dual_();
        const Infeasibility unperturbed = compute_dual_infeasible_(false);
        const bool dual_feasible_unperturbed = unperturbed.num == 0;
        force_phase2_ = unperturbed.max * unperturbed.max < opt_.dual_feasibility_tolerance;
        const Infeasibility primal_inf = compute_primal_infeasible_();
        const bool near_optimal = (dual_feasible_unperturbed || force_phase2_) &&
                                  primal_inf.num < 1000 && primal_inf.max < 1e-3;
        const bool perturb = !near_optimal;
        initialise_cost_(perturb);
        if (!weights_valid_) {
            std::fill(edge_weight_.begin(), edge_weight_.end(), 1.0);
            if (!logical_basis_())
                compute_dse_weights_();
            weights_valid_ = true;
        }
        int dual_infeas_count = 0;
        if (perturb) {
            compute_dual_();
            dual_infeas_count = compute_dual_infeasible_(true).num;
        }
        int phase = force_phase2_ ? 2 : (dual_infeas_count > 0 ? 1 : 2);
        for (int guard = 0; phase != 0 && guard < 1000; ++guard) {
            if (phase == -1) { // unknown: recompute from scratch (after backtracking)
                initialise_bound_(2);
                initialise_nonbasic_value_and_move_();
                compute_dual_();
                phase = compute_dual_infeasible_(true).num > 0 ? 1 : 2;
                if (backtracking_) {
                    initialise_bound_(phase);
                    initialise_nonbasic_value_and_move_();
                    backtracking_ = false;
                }
            }
            EKK_LOG("ekk: enter phase %d iter %lld\n", phase, iteration_count_);
            if (phase == 1)
                phase = solve_phase1_();
            else if (phase == 2)
                phase = solve_phase2_();
            EKK_LOG("ekk: -> phase %d iter %lld status %s\n", phase, iteration_count_,
                    to_string(status_));
            if (phase == kPhaseExit || phase == kPhaseCleanup)
                break;
            if (bailout_())
                return;
        }
        if (phase == kPhaseCleanup)
            status_ = ModelStatus::NotSet; // primal cleanup required
    }

    static constexpr int kPhaseExit = 100;
    static constexpr int kPhaseCleanup = 101;

    bool bailout_() {
        if (iteration_count_ >= opt_.iteration_limit) {
            status_ = ModelStatus::IterationLimit;
            return true;
        }
        return status_ == ModelStatus::ObjectiveBound || status_ == ModelStatus::SolveError;
    }

    enum RebuildReason {
        kRebuildNo = 0,
        kRebuildPossiblyOptimal,
        kRebuildPossiblyDualUnbounded,
        kRebuildUpdateLimit,
        kRebuildPossiblySingular,
        kRebuildChooseColumnFail,
        kRebuildExcessivePrimal,
    };

    // HEkkDual::rebuild
    bool rebuild_(int phase) {
        for (BadBasisChange& change : bad_changes_)
            change.taboo = false;
        const int reason = rebuild_reason_;
        rebuild_reason_ = kRebuildNo;
        (void)reason;
        if ((update_count_ > 0 || !factor_) && !reinvert_())
            return false;
        for (int attempt = 0;; ++attempt) {
            try {
                compute_dual_();
                if (!backtracking_) {
                    dual_infeas_count_ = correct_dual_infeasibilities_();
                    compute_primal_();
                }
                break;
            } catch (const std::exception&) {
                // Solves failed on a factorization that looked fine.
                if (attempt > 0)
                    return false;
                backtracking_ = true;
                set_logical_basis_();
                std::fill(edge_weight_.begin(), edge_weight_.end(), 1.0);
                weights_valid_ = true;
                if (!invert_())
                    return false;
            }
        }
        if (backtracking_)
            return true; // caller re-derives the phase
        create_infeasibility_array_();
        updated_dual_objective_ = compute_dual_objective_(phase);
        fresh_rebuild_ = true;
        return true;
    }

    void create_infeasibility_array_() {
        const double tol = opt_.primal_feasibility_tolerance;
        for (int i = 0; i < m_; ++i) {
            const double v = base_value_[i];
            double inf = 0.0;
            if (v < base_lower_[i] - tol)
                inf = base_lower_[i] - v;
            else if (v > base_upper_[i] + tol)
                inf = v - base_upper_[i];
            work_infeasibility_[i] = inf * inf;
        }
    }

    int solve_phase1_() {
        initialise_bound_(1);
        initialise_nonbasic_value_and_move_();
        for (;;) {
            if (!rebuild_(1)) {
                { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
                return kPhaseExit;
            }
            if (backtracking_)
                return -1;
            if (bailout_())
                return kPhaseExit;
            for (;;) {
                iterate_(1);
                if (bailout_())
                    return kPhaseExit;
                if (rebuild_reason_)
                    break;
            }
            const bool finished = fresh_rebuild_ && update_count_ == 0;
            if (finished || rebuild_reason_ == kRebuildPossiblyOptimal ||
                rebuild_reason_ == kRebuildPossiblyDualUnbounded ||
                rebuild_reason_ == kRebuildChooseColumnFail ||
                rebuild_reason_ == kRebuildExcessivePrimal) {
                if (!(fresh_rebuild_ && update_count_ == 0))
                    continue; // refactor and look again with fresh data
                break;
            }
        }
        if (rebuild_reason_ == kRebuildChooseColumnFail || rebuild_reason_ == kRebuildExcessivePrimal) {
            { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
            return kPhaseExit;
        }
        int next_phase;
        if (row_out_ < 0) {
            const double obj = compute_dual_objective_(1);
            if (obj == 0.0) {
                next_phase = 2;
            } else {
                // Phase-1 optimum with nonzero objective: remove the
                // perturbation and judge dual feasibility unperturbed.
                cleanup_(1);
                if (dual_infeas_count_ > 0) {
                    next_phase = 1;
                } else if (compute_dual_objective_(1) == 0.0 ||
                           lp_dual_infeasible_count_() == 0) {
                    next_phase = 2;
                } else {
                    status_ = ModelStatus::UnboundedOrInfeasible;
                    next_phase = kPhaseExit;
                }
            }
        } else {
            // Dual phase 1 unbounded: should not happen for a bounded auxiliary LP.
            if (costs_perturbed_) {
                cleanup_(1);
                next_phase = dual_infeas_count_ == 0 ? 2 : 1;
            } else {
                // Numerically impossible for the bounded auxiliary LP; let
                // solve() settle it via the feasibility LP and primal simplex.
                status_ = ModelStatus::UnboundedOrInfeasible;
                next_phase = kPhaseExit;
            }
        }
        if (next_phase == 2 || next_phase == kPhaseExit) {
            initialise_bound_(2);
            initialise_nonbasic_value_and_move_();
            if (next_phase == 2)
                allow_cost_perturbation_ = true;
        }
        return next_phase;
    }

    // Dual infeasibilities of the LP with respect to the phase-2 bounds
    int lp_dual_infeasible_count_() {
        int count = 0;
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const double lower = var_lower_(j), upper = var_upper_(j);
            const double dual = work_dual_[j];
            double inf = 0.0;
            if (!std::isfinite(lower) && !std::isfinite(upper))
                inf = std::abs(dual);
            else if (!std::isfinite(upper))
                inf = -dual;
            else if (!std::isfinite(lower))
                inf = dual;
            if (inf >= opt_.dual_feasibility_tolerance)
                ++count;
        }
        return count;
    }

    int solve_phase2_() {
        for (;;) {
            if (!rebuild_(2)) {
                { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
                return kPhaseExit;
            }
            if (backtracking_)
                return -1;
            if (bailout_())
                return kPhaseExit;
            if (check_objective_bound_())
                return kPhaseExit;
            if (dual_infeas_count_ > 0)
                return 1;
            for (;;) {
                iterate_(2);
                if (bailout_())
                    return kPhaseExit;
                if (check_objective_bound_())
                    return kPhaseExit;
                if (rebuild_reason_ == kRebuildPossiblyDualUnbounded && fresh_rebuild_ &&
                    update_count_ == 0) {
                    // Dual unbounded after a fresh rebuild: primal infeasible
                    // when the pivotal row proves it.
                    if (proof_of_primal_infeasibility_()) {
                        status_ = ModelStatus::Infeasible;
                        return kPhaseExit;
                    }
                    taboo_rows_.push_back(row_out_);
                    rebuild_reason_ = kRebuildNo;
                    continue;
                }
                if (rebuild_reason_)
                    break;
            }
            const bool fresh = fresh_rebuild_ && update_count_ == 0;
            if (!fresh)
                continue;
            if (rebuild_reason_ == kRebuildPossiblyDualUnbounded) {
                if (proof_of_primal_infeasibility_()) {
                    status_ = ModelStatus::Infeasible;
                    return kPhaseExit;
                }
                taboo_rows_.push_back(row_out_);
                if (static_cast<int>(taboo_rows_.size()) > m_) {
                    { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
                    return kPhaseExit;
                }
                continue;
            }
            break;
        }
        if (!taboo_rows_.empty() && compute_primal_infeasible_().num > 0) {
            // Rows whose infeasibility could neither be removed nor proved:
            // let solve() settle feasibility separately.
            taboo_rows_.clear();
            status_ = ModelStatus::UnboundedOrInfeasible;
            return kPhaseExit;
        }
        taboo_rows_.clear();
        if (rebuild_reason_ == kRebuildChooseColumnFail || rebuild_reason_ == kRebuildExcessivePrimal) {
            { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
            return kPhaseExit;
        }
        if (dual_infeas_count_ > 0)
            return 1;
        // No CHUZR candidate after a fresh rebuild: optimal for the perturbed
        // costs. Remove the perturbation and check dual feasibility.
        cleanup_(2);
        if (dual_infeas_count_ > 0)
            return kPhaseCleanup;
        status_ = ModelStatus::Optimal;
        return 0;
    }

    // HEkkDual::cleanup
    void cleanup_(int phase) {
        initialise_cost_(false);
        allow_cost_perturbation_ = false;
        initialise_bound_(phase);
        compute_dual_();
        dual_infeas_count_ = compute_dual_infeasible_(true).num;
        updated_dual_objective_ = compute_dual_objective_(phase);
    }

    // HEkkDual::reachedExactObjectiveBound for minimisation in phase 2
    bool check_objective_bound_() {
        if (!std::isfinite(opt_.objective_bound))
            return false;
        if (updated_dual_objective_ <= scaled_objective_bound_())
            return false;
        const double exact = exact_dual_objective_();
        if (exact > scaled_objective_bound_()) {
            status_ = ModelStatus::ObjectiveBound;
            return true;
        }
        return false;
    }

    double scaled_objective_bound_() const { return opt_.objective_bound; }

    // HEkkDual::computeExactDualObjectiveValue: dual objective of the current
    // basis under the true costs, a valid lower bound on the LP optimum.
    double exact_dual_objective_() {
        WorkVector y;
        y.setup(m_);
        y.count = -1;
        for (int i = 0; i < m_; ++i) {
            const int var = basis_.basic_index[i];
            if (var < n_)
                y.array[i] = lp_.col_cost[var];
        }
        btran_(y);
        double obj = lp_.offset;
        for (int j = 0; j < n_; ++j) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const double dual = lp_.col_cost[j] - column_dot_(j, y);
            double active;
            if (dual > opt_.small_matrix_value)
                active = lp_.col_lower[j];
            else if (dual < -opt_.small_matrix_value)
                active = lp_.col_upper[j];
            else
                active = work_value_[j];
            if (!std::isfinite(active))
                return -kInf;
            obj += active * dual;
        }
        for (int i = 0; i < m_; ++i) {
            const int var = n_ + i;
            if (!basis_.nonbasic_flag[var])
                continue;
            const double dual = y.array[i];
            double active;
            if (dual > opt_.small_matrix_value)
                active = lp_.row_lower[i];
            else if (dual < -opt_.small_matrix_value)
                active = lp_.row_upper[i];
            else
                active = -work_value_[var];
            if (!std::isfinite(active))
                return -kInf;
            obj += active * dual;
        }
        return obj;
    }

    // HEkk::proofOfPrimalInfeasibility: with ep = B^{-T} e_p, every x with
    // r = Ax satisfies ep'Ax = ep'r. Infeasible when the ranges of the two
    // sides over the column and row boxes are disjoint.
    bool proof_of_primal_infeasibility_() {
        if (row_out_ < 0)
            return false;
        WorkVector ep;
        ep.setup(m_);
        ep.set_unit(row_out_);
        btran_(ep);
        std::vector<double> y(m_, 0.0);
        for (int k = 0; k < ep.count; ++k) {
            const int i = ep.index[k];
            if (std::abs(ep.array[i]) > opt_.small_matrix_value)
                y[i] = ep.array[i];
        }
        // Range of y'r over the row box
        double r_min = 0.0, r_max = 0.0;
        for (int i = 0; i < m_; ++i) {
            if (y[i] == 0.0)
                continue;
            r_min += y[i] * (y[i] > 0 ? lp_.row_lower[i] : lp_.row_upper[i]);
            r_max += y[i] * (y[i] > 0 ? lp_.row_upper[i] : lp_.row_lower[i]);
        }
        // Range of y'Ax over the column box
        double a_min = 0.0, a_max = 0.0;
        for (int j = 0; j < n_; ++j) {
            double a = 0.0;
            for (int k = lp_.a_start[j]; k < lp_.a_start[j + 1]; ++k)
                a += lp_.a_value[k] * y[lp_.a_index[k]];
            if (std::abs(a) <= opt_.small_matrix_value)
                continue;
            a_min += a * (a > 0 ? lp_.col_lower[j] : lp_.col_upper[j]);
            a_max += a * (a > 0 ? lp_.col_upper[j] : lp_.col_lower[j]);
        }
        // NaN (inf - inf) compares false, so it never proves anything.
        const double tol = 1e3 * opt_.primal_feasibility_tolerance;
        const bool disjoint_up = r_min - a_max > tol * std::max(1.0, std::abs(r_min));
        const bool disjoint_down = a_min - r_max > tol * std::max(1.0, std::abs(a_min));
        return disjoint_up || disjoint_down;
    }

    // ------------------------------------------------------- iteration
    void iterate_(int phase) {
        try {
            iterate_body_(phase);
        } catch (const std::exception& e) {
            EKK_LOG("ekk: iterate exception (updates %d): %s\n", update_count_, e.what());
            rebuild_reason_ = kRebuildPossiblySingular;
            fresh_rebuild_ = false;
            if (update_count_ == 0) {
                // The fresh factorization itself is unusable: restart from
                // the logical basis.
                backtracking_ = true;
                set_logical_basis_();
                std::fill(edge_weight_.begin(), edge_weight_.end(), 1.0);
                weights_valid_ = true;
            }
        }
    }

    void iterate_body_(int phase) {
        choose_row_();
        if (rebuild_reason_)
            return;
        choose_column_();
        if (rebuild_reason_)
            return;
        if (is_bad_basis_change_())
            return;
        update_ftran_bfrt_();
        update_ftran_();
        update_ftran_dse_();
        update_verify_();
        if (rebuild_reason_)
            return;
        update_dual_();
        update_primal_();
        if (rebuild_reason_)
            return;
        update_pivots_(phase);
    }

    // HEkk::isBadBasisChange: a basis change that revisits a basis hash on
    // consecutive checks is cycling and becomes taboo.
    bool is_bad_basis_change_() {
        const int var_out = basis_.basic_index[row_out_];
        const uint64_t hash = basis_hash_ ^ hash_key_[var_out] ^ hash_key_[variable_in_];
        bool cycling = false;
        if (visited_basis_.count(hash)) {
            if (iteration_count_ == previous_cycling_iteration_ + 1)
                cycling = true;
            else
                previous_cycling_iteration_ = iteration_count_;
        }
        if (cycling) {
            EKK_LOG("ekk: cycling detected, row %d out %d in %d taboo\n", row_out_, var_out, variable_in_);
            for (BadBasisChange& change : bad_changes_) {
                if (change.row_out == row_out_ && change.variable_out == var_out &&
                    change.variable_in == variable_in_) {
                    change.taboo = true;
                    return true;
                }
            }
            bad_changes_.push_back({row_out_, var_out, variable_in_, true});
            return true;
        }
        for (BadBasisChange& change : bad_changes_) {
            if (change.row_out == row_out_ && change.variable_out == var_out &&
                change.variable_in == variable_in_) {
                change.taboo = true;
                return true;
            }
        }
        return false;
    }

    void compute_basis_hash_() {
        basis_hash_ = 0;
        for (int i = 0; i < m_; ++i)
            basis_hash_ ^= hash_key_[basis_.basic_index[i]];
    }

    // HEkkDual::chooseRow with DSE weight verification
    void choose_row_() {
        for (int r : taboo_rows_)
            if (r >= 0 && r < m_)
                work_infeasibility_[r] = 0.0;
        // HEkk::applyTabooRowOut for cycling records; restored afterwards.
        saved_infeasibility_.clear();
        for (const BadBasisChange& change : bad_changes_) {
            if (change.taboo) {
                saved_infeasibility_.emplace_back(change.row_out, work_infeasibility_[change.row_out]);
                work_infeasibility_[change.row_out] = 0.0;
            }
        }
        choose_row_search_();
        for (auto it = saved_infeasibility_.rbegin(); it != saved_infeasibility_.rend(); ++it)
            work_infeasibility_[it->first] = it->second;
        if (row_out_ < 0 && !saved_infeasibility_.empty()) {
            // Only cycling rows remain: drop the taboos rather than stop.
            for (BadBasisChange& change : bad_changes_)
                change.taboo = false;
            rebuild_reason_ = kRebuildNo;
            choose_row_search_();
        }
    }

    void choose_row_search_() {
        for (;;) {
            int best = -1;
            double best_merit = 0.0;
            for (int i = 0; i < m_; ++i) {
                const double inf = work_infeasibility_[i];
                if (inf > 1e-50 && best_merit * edge_weight_[i] < inf) {
                    best_merit = inf / edge_weight_[i];
                    best = i;
                }
            }
            row_out_ = best;
            if (row_out_ < 0) {
                rebuild_reason_ = kRebuildPossiblyOptimal;
                return;
            }
            row_ep_.clear();
            row_ep_.set_unit(row_out_);
            btran_(row_ep_);
            const double updated = edge_weight_[row_out_];
            edge_weight_[row_out_] = row_ep_.norm2();
            if (updated >= 0.25 * edge_weight_[row_out_])
                break;
        }
        variable_out_ = basis_.basic_index[row_out_];
        if (base_value_[row_out_] < base_lower_[row_out_])
            delta_primal_ = base_value_[row_out_] - base_lower_[row_out_];
        else
            delta_primal_ = base_value_[row_out_] - base_upper_[row_out_];
        move_out_ = delta_primal_ < 0 ? -1 : 1;
    }

    // HEkk::tableauRowPrice: row_ap = row_ep' A for nonbasic structurals
    void price_() {
        row_ap_.clear();
        const double density = static_cast<double>(row_ep_.count) / std::max(1, m_);
        if (density > 0.1) {
            for (int j = 0; j < n_; ++j) {
                row_ap_.array[j] = basis_.nonbasic_flag[j] ? column_dot_(j, row_ep_) : 0.0;
            }
            row_ap_.tight();
            return;
        }
        row_ap_.index.clear();
        for (int k = 0; k < row_ep_.count; ++k) {
            const int i = row_ep_.index[k];
            const double yi = row_ep_.array[i];
            for (int p = ar_.start[i]; p < ar_.start[i + 1]; ++p) {
                const int j = ar_.index[p];
                if (!basis_.nonbasic_flag[j])
                    continue;
                if (!price_mark_[j]) {
                    price_mark_[j] = 1;
                    row_ap_.index.push_back(j);
                }
                row_ap_.array[j] += yi * ar_.value[p];
            }
        }
        for (int j : row_ap_.index)
            price_mark_[j] = 0;
        row_ap_.count = static_cast<int>(row_ap_.index.size());
        row_ap_.tight_pattern();
    }

    // HEkkDual::chooseColumn: PRICE, then the bound-flipping ratio test
    void choose_column_() {
        price_();
        // Free nonbasic variables get a move from the sign of their alpha
        // (HEkkDualRow::createFreemove) so the ratio test can use them.
        const double ta_free = update_count_ < 10 ? 1e-9 : update_count_ < 20 ? 3e-8 : 1e-6;
        free_moved_.clear();
        for (int j : free_vars_) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const double alpha = j < n_ ? row_ap_.array[j] : row_ep_.array[j - n_];
            if (std::abs(alpha) > ta_free) {
                basis_.nonbasic_move[j] = static_cast<int8_t>(alpha * move_out_ > 0 ? 1 : -1);
                free_moved_.push_back(j);
            }
        }
        pack_count_ = 0;
        for (int k = 0; k < row_ap_.count; ++k) {
            const int j = row_ap_.index[k];
            pack_index_[pack_count_] = j;
            pack_value_[pack_count_++] = row_ap_.array[j];
        }
        for (int k = 0; k < row_ep_.count; ++k) {
            const int i = row_ep_.index[k];
            pack_index_[pack_count_] = n_ + i;
            pack_value_[pack_count_++] = row_ep_.array[i];
        }
        double max_abs = 0.0;
        for (int k = 0; k < pack_count_; ++k)
            max_abs = std::max(max_abs, std::abs(pack_value_[k]));
        const double row_ep_scale = 1.0 / nearest_power_of_two(std::max(max_abs, 1e-300));
        variable_in_ = -1;
        for (int pass = 0;; ++pass) {
            choose_possible_();
            if (work_theta_ <= 0 || work_count_ == 0) {
#ifdef EKK_TRACE
                EKK_LOG("ekk: dual unbounded row %d var_out %d x %g [%g,%g] delta %g theta %g count %d pack %d\n",
                        row_out_, variable_out_, base_value_[row_out_], base_lower_[row_out_],
                        base_upper_[row_out_], delta_primal_, work_theta_, work_count_, pack_count_);
                for (int k = 0; k < pack_count_; ++k) {
                    const int j = pack_index_[k];
                    EKK_LOG("   j %d alpha %g move %d value %g work[%g,%g] dual %g\n", j, pack_value_[k],
                            basis_.nonbasic_move[j], work_value_[j], work_lower_[j], work_upper_[j],
                            work_dual_[j]);
                }
#endif
                rebuild_reason_ = kRebuildPossiblyDualUnbounded;
                break;
            }
            if (!choose_final_()) {
                rebuild_reason_ = kRebuildChooseColumnFail;
                break;
            }
            if (work_pivot_ >= 0 &&
                std::abs(row_ep_scale * work_alpha_) <= opt_.pivot_growth_tolerance) {
                // Small pivot: remove it from the pack and choose again.
                for (int k = 0; k < pack_count_; ++k) {
                    if (pack_index_[k] == work_pivot_) {
                        pack_index_[k] = pack_index_[pack_count_ - 1];
                        pack_value_[k] = pack_value_[pack_count_ - 1];
                        --pack_count_;
                        break;
                    }
                }
                work_pivot_ = -1;
                if (pack_count_ > 0)
                    continue;
                rebuild_reason_ = kRebuildPossiblyDualUnbounded;
            }
            break;
        }
        for (int j : free_moved_)
            basis_.nonbasic_move[j] = 0;
        if (rebuild_reason_)
            return;
        variable_in_ = work_pivot_;
        alpha_row_ = work_alpha_;
        theta_dual_ = work_theta_;
    }

    // HEkkDualRow::choosePossible
    void choose_possible_() {
        const double ta = update_count_ < 10 ? 1e-9 : update_count_ < 20 ? 3e-8 : 1e-6;
        const double td = opt_.dual_feasibility_tolerance;
        work_theta_ = kInf;
        work_count_ = 0;
        for (int k = 0; k < pack_count_; ++k) {
            const int j = pack_index_[k];
            const int move = basis_.nonbasic_move[j];
            const double alpha = pack_value_[k] * move_out_ * move;
            if (alpha > ta) {
                work_data_[work_count_++] = {j, alpha};
                const double relax = work_dual_[j] * move + td;
                if (work_theta_ * alpha > relax)
                    work_theta_ = relax / alpha;
            }
        }
    }

    // HEkkDualRow::chooseFinal: large-step reduction, quadratic grouping,
    // largest-alpha selection and the bound flips before the break group.
    bool choose_final_() {
        int full_count = work_count_;
        work_count_ = 0;
        double total_change = 0.0;
        const double total_delta = std::abs(delta_primal_);
        double select_theta = 10 * work_theta_ + 1e-7;
        for (;;) {
            for (int i = work_count_; i < full_count; ++i) {
                const int j = work_data_[i].first;
                const double alpha = work_data_[i].second;
                const double tight = basis_.nonbasic_move[j] * work_dual_[j];
                if (alpha * select_theta >= tight) {
                    std::swap(work_data_[work_count_++], work_data_[i]);
                    total_change += work_range_[j] * alpha;
                }
            }
            select_theta *= 10;
            if (total_change >= total_delta || work_count_ == full_count)
                break;
        }
        // chooseFinalWorkGroupQuad
        const double td = opt_.dual_feasibility_tolerance;
        full_count = work_count_;
        work_count_ = 0;
        total_change = 1e-12;
        select_theta = work_theta_;
        work_group_.clear();
        work_group_.push_back(0);
        int prev_count = work_count_;
        double prev_remain = 1e100, prev_select = select_theta;
        while (select_theta < 1e18) {
            double remain_theta = 1e100;
            for (int i = work_count_; i < full_count; ++i) {
                const int j = work_data_[i].first;
                const double value = work_data_[i].second;
                const double dual = basis_.nonbasic_move[j] * work_dual_[j];
                if (dual <= select_theta * value) {
                    std::swap(work_data_[work_count_++], work_data_[i]);
                    total_change += value * work_range_[j];
                } else if (dual + td < remain_theta * value) {
                    remain_theta = (dual + td) / value;
                }
            }
            work_group_.push_back(work_count_);
            select_theta = remain_theta;
            if (work_count_ == prev_count && prev_select == select_theta &&
                prev_remain == remain_theta)
                return false;
            prev_count = work_count_;
            prev_remain = remain_theta;
            prev_select = select_theta;
            if (total_change >= total_delta || work_count_ == full_count)
                break;
        }
        if (work_group_.size() <= 1)
            return false;
        // chooseFinalLargeAlpha
        double final_compare = 0.0;
        for (int i = 0; i < work_count_; ++i)
            final_compare = std::max(final_compare, work_data_[i].second);
        final_compare = std::min(0.1 * final_compare, 1.0);
        const int count_group = static_cast<int>(work_group_.size()) - 1;
        int break_index = -1, break_group = -1;
        for (int g = count_group - 1; g >= 0; --g) {
            double dmax = 0.0;
            int imax = -1;
            for (int i = work_group_[g]; i < work_group_[g + 1]; ++i) {
                if (dmax < work_data_[i].second) {
                    dmax = work_data_[i].second;
                    imax = i;
                } else if (imax >= 0 && dmax == work_data_[i].second &&
                           permutation_[work_data_[i].first] <
                               permutation_[work_data_[imax].first]) {
                    imax = i;
                }
            }
            if (imax >= 0 && work_data_[imax].second > final_compare) {
                break_index = imax;
                break_group = g;
                break;
            }
        }
        if (break_index < 0)
            return false;
        work_pivot_ = work_data_[break_index].first;
        work_alpha_ =
            work_data_[break_index].second * move_out_ * basis_.nonbasic_move[work_pivot_];
        if (work_dual_[work_pivot_] * basis_.nonbasic_move[work_pivot_] > 0)
            work_theta_ = work_dual_[work_pivot_] / work_alpha_;
        else
            work_theta_ = 0.0;
        // Flip every candidate in the groups before the break group.
        work_count_ = 0;
        for (int i = 0; i < work_group_[break_group]; ++i) {
            const int j = work_data_[i].first;
            work_data_[work_count_++] = {j, basis_.nonbasic_move[j] * work_range_[j]};
        }
        if (work_theta_ == 0.0)
            work_count_ = 0;
        std::sort(work_data_.begin(), work_data_.begin() + work_count_);
        return true;
    }

    // HEkkDual::updateFtranBFRT via HEkkDualRow::updateFlip
    void update_ftran_bfrt_() {
        col_bfrt_.clear();
        if (work_count_ == 0)
            return;
        double objective_change = 0.0;
        for (int k = 0; k < work_count_; ++k) {
            const int j = work_data_[k].first;
            const double change = work_data_[k].second;
            objective_change += change * work_dual_[j];
            flip_bound_(j);
            collect_column_(col_bfrt_, j, change);
        }
        updated_dual_objective_ += objective_change;
        col_bfrt_.count = -1;
        ftran_(col_bfrt_);
    }

    void update_ftran_() {
        col_aq_.clear();
        collect_column_(col_aq_, variable_in_, 1.0);
        ftran_(col_aq_);
        alpha_col_ = col_aq_.array[row_out_];
    }

    void update_ftran_dse_() {
        col_dse_.clear();
        for (int k = 0; k < row_ep_.count; ++k)
            col_dse_.array[row_ep_.index[k]] = row_ep_.array[row_ep_.index[k]];
        col_dse_.index.assign(row_ep_.index.begin(), row_ep_.index.begin() + row_ep_.count);
        col_dse_.count = row_ep_.count;
        ftran_(col_dse_);
    }

    // HEk::reinvertOnNumericalTrouble
    void update_verify_() {
        const double abs_col = std::abs(alpha_col_);
        const double abs_row = std::abs(alpha_row_);
        const double min_abs = std::min(abs_col, abs_row);
        const double trouble = min_abs > 0.0 ? std::abs(abs_col - abs_row) / min_abs : kInf;
        if (trouble > 1e-7 && update_count_ > 0)
            rebuild_reason_ = kRebuildPossiblySingular;
        else if (alpha_col_ == 0.0)
            rebuild_reason_ = kRebuildPossiblySingular;
    }

    // HEkkDual::updateDual
    void update_dual_() {
        if (theta_dual_ == 0.0) {
            // shiftCost(variable_in, -dual)
            costs_shifted_ = true;
            work_shift_[variable_in_] = -work_dual_[variable_in_];
        } else {
            double objective_change = 0.0;
            for (int k = 0; k < pack_count_; ++k) {
                const int j = pack_index_[k];
                const double delta = theta_dual_ * pack_value_[k];
                work_dual_[j] -= delta;
                objective_change += basis_.nonbasic_flag[j] * (-work_value_[j] * delta);
            }
            updated_dual_objective_ += objective_change;
        }
        updated_dual_objective_ +=
            basis_.nonbasic_flag[variable_in_] * (-work_value_[variable_in_] * work_dual_[variable_in_]);
        work_dual_[variable_in_] = 0.0;
        work_dual_[variable_out_] = -theta_dual_;
        // shiftBack(variable_out)
        if (work_shift_[variable_out_] != 0.0) {
            work_dual_[variable_out_] -= work_shift_[variable_out_];
            work_shift_[variable_out_] = 0.0;
        }
    }

    bool update_base_values_(const WorkVector& column, double theta) {
        const double tol = opt_.primal_feasibility_tolerance;
        bool ok = true;
        auto touch = [&](int i) {
            base_value_[i] -= theta * column.array[i];
            const double v = base_value_[i];
            double inf = 0.0;
            if (v < base_lower_[i] - tol)
                inf = base_lower_[i] - v;
            else if (v > base_upper_[i] + tol)
                inf = v - base_upper_[i];
            work_infeasibility_[i] = inf * inf;
            if (std::abs(v) >= 1e25)
                ok = false;
        };
        if (column.count < 0) {
            for (int i = 0; i < m_; ++i)
                if (column.array[i] != 0.0)
                    touch(i);
        } else {
            for (int k = 0; k < column.count; ++k)
                touch(column.index[k]);
        }
        return ok;
    }

    // HEkkDual::updatePrimal with the DSE weight update
    void update_primal_() {
        update_base_values_(col_bfrt_, 1.0);
        const double x_out = base_value_[row_out_];
        const double bound = delta_primal_ < 0 ? base_lower_[row_out_] : base_upper_[row_out_];
        theta_primal_ = (x_out - bound) / alpha_col_;
        if (!update_base_values_(col_aq_, theta_primal_)) {
            rebuild_reason_ = kRebuildExcessivePrimal;
            return;
        }
        bad_changes_.erase(std::remove_if(bad_changes_.begin(), bad_changes_.end(),
                                          [&](const BadBasisChange& change) {
                                              return std::abs(col_aq_.array[change.row_out] *
                                                              theta_primal_) >=
                                                     opt_.primal_feasibility_tolerance;
                                          }),
                           bad_changes_.end());
        const double new_pivotal_weight = edge_weight_[row_out_] / (alpha_col_ * alpha_col_);
        const double kai = -2.0 / alpha_col_;
        for (int k = 0; k < col_aq_.count; ++k) {
            const int i = col_aq_.index[k];
            const double aa = col_aq_.array[i];
            edge_weight_[i] += aa * (new_pivotal_weight * aa + kai * col_dse_.array[i]);
            edge_weight_[i] = std::max(1e-4, edge_weight_[i]);
        }
        edge_weight_[row_out_] = new_pivotal_weight;
    }

    // HEkkDual::updatePivots / HEkk::updatePivots / updateFactor
    void update_pivots_(int phase) {
        (void)phase;
        EKK_LOG("ekk: pivot it %lld row %d in %d out %d alpha_col %g alpha_row %g theta_d %g theta_p %g\n",
                iteration_count_, row_out_, variable_in_, variable_out_, alpha_col_, alpha_row_,
                theta_dual_, theta_primal_);
        const int var_in = variable_in_;
        const int var_out = variable_out_;
        // Factor update needs the original column and its FTRAN/BTRAN images.
        Eigen::VectorXd aq = Eigen::VectorXd::Zero(m_);
        if (var_in < n_) {
            for (int k = lp_.a_start[var_in]; k < lp_.a_start[var_in + 1]; ++k)
                aq(lp_.a_index[k]) = lp_.a_value[k];
        } else {
            aq(var_in - n_) = 1.0;
        }

        basis_.basic_index[row_out_] = var_in;
        basis_hash_ ^= hash_key_[var_out] ^ hash_key_[var_in];
        visited_basis_.insert(basis_hash_);
        basis_.nonbasic_flag[var_in] = 0;
        basis_.nonbasic_move[var_in] = 0;
        base_lower_[row_out_] = work_lower_[var_in];
        base_upper_[row_out_] = work_upper_[var_in];
        basis_.nonbasic_flag[var_out] = 1;
        if (work_lower_[var_out] == work_upper_[var_out]) {
            work_value_[var_out] = work_lower_[var_out];
            basis_.nonbasic_move[var_out] = 0;
        } else if (move_out_ == -1) {
            work_value_[var_out] = work_lower_[var_out];
            basis_.nonbasic_move[var_out] = 1;
        } else {
            work_value_[var_out] = work_upper_[var_out];
            basis_.nonbasic_move[var_out] = -1;
        }
        updated_dual_objective_ += work_value_[var_out] * work_dual_[var_out];
        ++update_count_;
        ++iteration_count_;
        fresh_rebuild_ = false;
        try {
            factor_->replace_column_with_transforms(row_out_, var_in, aq, col_aq_.array,
                                                    row_ep_.array);
        } catch (const std::exception&) {
            rebuild_reason_ = kRebuildPossiblySingular;
        }
        if (update_count_ >= std::min(update_limit_, opt_.update_limit))
            rebuild_reason_ = kRebuildUpdateLimit;
        // New basic value at the pivot row and its infeasibility
        base_value_[row_out_] = work_value_[var_in] + theta_primal_;
        const double tol = opt_.primal_feasibility_tolerance;
        const double v = base_value_[row_out_];
        double inf = 0.0;
        if (v < base_lower_[row_out_] - tol)
            inf = base_lower_[row_out_] - v;
        else if (v > base_upper_[row_out_] + tol)
            inf = v - base_upper_[row_out_];
        work_infeasibility_[row_out_] = inf * inf;
    }

    // ------------------------------------------------- primal cleanup
    // Primal simplex phase 2 from a primal-feasible basis, removing the dual
    // infeasibilities left after removing cost perturbations. Dantzig pricing
    // with a two-pass Harris ratio test and bound flips for the entering
    // variable. Leaves status_ NotSet if primal feasibility is lost.
    void primal_cleanup_() {
        initialise_cost_(false);
        initialise_bound_(2);
        if (!reinvert_()) {
            { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
            return;
        }
        compute_primal_();
        compute_dual_();
        weights_valid_ = false;
        const double tp = opt_.primal_feasibility_tolerance;
        const double td = opt_.dual_feasibility_tolerance;
        const long long limit = iteration_count_ + 10LL * (m_ + n_) + 1000;
        for (;;) {
            if (iteration_count_ >= opt_.iteration_limit) {
                status_ = ModelStatus::IterationLimit;
                return;
            }
            if (iteration_count_ >= limit)
                return;
            // CHUZC: most dual infeasible nonbasic
            int q = -1;
            double best = td;
            for (int j = 0; j < tot_; ++j) {
                if (!basis_.nonbasic_flag[j])
                    continue;
                const double lower = work_lower_[j], upper = work_upper_[j];
                if (lower == upper)
                    continue;
                const double d = work_dual_[j];
                double inf;
                if (!std::isfinite(lower) && !std::isfinite(upper))
                    inf = std::abs(d);
                else
                    inf = -basis_.nonbasic_move[j] * d;
                if (inf > best) {
                    best = inf;
                    q = j;
                }
            }
            if (q < 0) {
                if (compute_primal_infeasible_().num == 0)
                    status_ = ModelStatus::Optimal;
                return;
            }
            const double sigma = work_dual_[q] < 0 ? 1.0 : -1.0;
            col_aq_.clear();
            collect_column_(col_aq_, q, 1.0);
            ftran_(col_aq_);
            // Harris pass 1: largest step with relaxed bounds
            double relaxed = kInf;
            for (int k = 0; k < col_aq_.count; ++k) {
                const int i = col_aq_.index[k];
                const double rate = -sigma * col_aq_.array[i];
                if (std::abs(rate) <= 1e-9)
                    continue;
                if (rate < 0 && std::isfinite(base_lower_[i]))
                    relaxed = std::min(relaxed, (base_value_[i] - base_lower_[i] + tp) / -rate);
                else if (rate > 0 && std::isfinite(base_upper_[i]))
                    relaxed = std::min(relaxed, (base_upper_[i] - base_value_[i] + tp) / rate);
            }
            const double range = work_range_[q];
            if (!std::isfinite(relaxed) && !std::isfinite(range)) {
                status_ = ModelStatus::Unbounded;
                return;
            }
            // Pass 2: among ratios within the relaxed step, the largest pivot
            int p = -1;
            double best_alpha = 0.0, theta = 0.0;
            for (int k = 0; k < col_aq_.count; ++k) {
                const int i = col_aq_.index[k];
                const double rate = -sigma * col_aq_.array[i];
                if (std::abs(rate) <= 1e-9)
                    continue;
                double t;
                if (rate < 0 && std::isfinite(base_lower_[i]))
                    t = (base_value_[i] - base_lower_[i]) / -rate;
                else if (rate > 0 && std::isfinite(base_upper_[i]))
                    t = (base_upper_[i] - base_value_[i]) / rate;
                else
                    continue;
                if (t <= relaxed && std::abs(rate) > best_alpha) {
                    best_alpha = std::abs(rate);
                    p = i;
                    theta = std::max(0.0, t);
                }
            }
            if (p < 0 || (std::isfinite(range) && range <= theta)) {
                // Bound flip of the entering variable
                if (!std::isfinite(range)) {
                    status_ = ModelStatus::Unbounded;
                    return;
                }
                for (int k = 0; k < col_aq_.count; ++k) {
                    const int i = col_aq_.index[k];
                    base_value_[i] -= sigma * range * col_aq_.array[i];
                }
                flip_bound_(q);
                ++iteration_count_;
#ifdef EKK_TRACE
                {
                    const std::vector<double> kept = base_value_;
                    compute_primal_();
                    double err = 0.0;
                    for (int i = 0; i < m_; ++i)
                        err = std::max(err, std::abs(kept[i] - base_value_[i]) / (1 + std::abs(base_value_[i])));
                    if (err > 1e-6)
                        EKK_LOG("ekk: primal flip it %lld drift %g q %d range %g\n", iteration_count_, err, q, range);
                    base_value_ = kept;
                }
#endif
                continue;
            }
            // Basis change: q enters at row p
            row_out_ = p;
            variable_in_ = q;
            variable_out_ = basis_.basic_index[p];
            const double rate_p = -sigma * col_aq_.array[p];
            move_out_ = rate_p < 0 ? -1 : 1; // leaves at lower when decreasing
            row_ep_.clear();
            row_ep_.set_unit(p);
            btran_(row_ep_);
            alpha_col_ = col_aq_.array[p];
            alpha_row_ = column_dot_(q, row_ep_);
            if (update_count_ > 0 &&
                std::abs(alpha_col_ - alpha_row_) >
                    1e-7 * std::min(std::abs(alpha_col_), std::abs(alpha_row_))) {
                // Numerical trouble in the updated factorization: refactor,
                // recompute and choose again (HEkk::reinvertOnNumericalTrouble).
                if (!reinvert_()) {
                    status_ = ModelStatus::SolveError;
                    return;
                }
                if (backtracking_) {
                    initialise_nonbasic_value_and_move_();
                    backtracking_ = false;
                }
                compute_primal_();
                compute_dual_();
                continue;
            }
            for (int k = 0; k < col_aq_.count; ++k) {
                const int i = col_aq_.index[k];
                base_value_[i] -= sigma * theta * col_aq_.array[i];
            }
            theta_primal_ = sigma * theta;
            update_pivots_(2);
            base_value_[p] = work_value_[q] + sigma * theta;
            if (rebuild_reason_ || update_count_ % 50 == 0) {
                rebuild_reason_ = kRebuildNo;
                if (!reinvert_()) {
                    { EKK_LOG("ekk: solve_error at line %d reason %d\n", __LINE__, rebuild_reason_); status_ = ModelStatus::SolveError; }
                    return;
                }
                compute_primal_();
            }
#ifdef EKK_TRACE
            {
                const std::vector<double> kept = base_value_;
                compute_primal_();
                double err = 0.0;
                for (int i = 0; i < m_; ++i)
                    err = std::max(err, std::abs(kept[i] - base_value_[i]) / (1 + std::abs(base_value_[i])));
                if (err > 1e-6) {
                    Eigen::MatrixXd B(m_, m_);
                    for (int i = 0; i < m_; ++i)
                        B.col(i) = Eigen::VectorXd(full_.col(basis_.basic_index[i]));
                    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m_);
                    for (int j = 0; j < tot_; ++j)
                        if (basis_.nonbasic_flag[j] && work_value_[j] != 0.0)
                            rhs += work_value_[j] * Eigen::VectorXd(full_.col(j));
                    const Eigen::VectorXd xb = -B.fullPivLu().solve(rhs);
                    double e_kept = 0.0, e_factor = 0.0;
                    for (int i = 0; i < m_; ++i) {
                        e_kept = std::max(e_kept, std::abs(kept[i] - xb(i)) / (1 + std::abs(xb(i))));
                        e_factor = std::max(e_factor, std::abs(base_value_[i] - xb(i)) / (1 + std::abs(xb(i))));
                    }
                    EKK_LOG("ekk: primal it %lld drift %g (q %d p %d theta %g) err maintained %g factor %g\n",
                            iteration_count_, err, q, p, theta, e_kept, e_factor);
                }
                base_value_ = kept;
            }
#endif
            compute_dual_();
            if (compute_primal_infeasible_().max > 1e3 * tp) {
                // Lost primal feasibility: hand back to the dual simplex.
                return;
            }
        }
    }

    // ------------------------------------------------- solution
    ModelStatus solve_bound_only_() {
        solution_ = Solution{};
        solution_.col_value.resize(n_);
        solution_.col_dual = original_.col_cost;
        solution_.objective = original_.offset;
        for (int j = 0; j < n_; ++j) {
            const double c = original_.col_cost[j];
            const double lo = original_.col_lower[j], up = original_.col_upper[j];
            double x;
            if (c > 0)
                x = lo;
            else if (c < 0)
                x = up;
            else
                x = std::isfinite(lo) ? lo : (std::isfinite(up) ? up : 0.0);
            if (!std::isfinite(x)) {
                status_ = ModelStatus::Unbounded;
                return status_;
            }
            if (lo > up) {
                status_ = ModelStatus::Infeasible;
                return status_;
            }
            solution_.col_value[j] = x;
            solution_.objective += c * x;
        }
        status_ = ModelStatus::Optimal;
        return status_;
    }

    void extract_solution_() {
#ifdef EKK_TRACE
        for (int j = 0; j < tot_; ++j) {
            if (!basis_.nonbasic_flag[j])
                continue;
            const int mv = basis_.nonbasic_move[j];
            const double lo = var_lower_(j), up = var_upper_(j);
            const bool bad_value = (mv == 1 && work_value_[j] != lo) || (mv == -1 && work_value_[j] != up);
            const bool bad_dual = -mv * work_dual_[j] > 1e-6 ||
                                  (mv == 0 && lo != up && std::abs(work_dual_[j]) > 1e-6);
            if (bad_value || bad_dual)
                EKK_LOG("ekk: nonbasic %d move %d value %g [%g,%g] work[%g,%g] dual %g shift %g cost %g\n", j, mv,
                        work_value_[j], lo, up, work_lower_[j], work_upper_[j], work_dual_[j], work_shift_[j], work_cost_[j]);
        }
#endif
        solution_ = Solution{};
        if (status_ != ModelStatus::Optimal && status_ != ModelStatus::ObjectiveBound &&
            status_ != ModelStatus::IterationLimit)
            return;
        // Values in the scaled space, then unscale.
        std::vector<double> value(tot_, 0.0);
        for (int j = 0; j < tot_; ++j)
            if (basis_.nonbasic_flag[j])
                value[j] = work_value_[j];
        for (int i = 0; i < m_; ++i)
            value[basis_.basic_index[i]] = base_value_[i];
        solution_.col_value.resize(n_);
        solution_.col_dual.resize(n_);
        solution_.row_value.resize(m_);
        solution_.row_dual.resize(m_);
        for (int j = 0; j < n_; ++j) {
            const double c = scale_.active ? scale_.col[j] : 1.0;
            solution_.col_value[j] = value[j] * c;
            solution_.col_dual[j] = basis_.nonbasic_flag[j] ? work_dual_[j] / c : 0.0;
        }
        for (int i = 0; i < m_; ++i) {
            const double r = scale_.active ? scale_.row[i] : 1.0;
            solution_.row_value[i] = -value[n_ + i] / r;
            solution_.row_dual[i] =
                basis_.nonbasic_flag[n_ + i] ? -work_dual_[n_ + i] * r : 0.0;
        }
        double obj = original_.offset;
        for (int j = 0; j < n_; ++j)
            obj += original_.col_cost[j] * solution_.col_value[j];
        solution_.objective = obj;
    }

    // ------------------------------------------------------------ state
    Options opt_;
    LpData original_;
    LpData lp_; // scaled
    LpScale scale_;
    RowwiseMatrix ar_;
    FTBasis::SparseMat full_;
    int n_ = 0, m_ = 0, tot_ = 0;

    Basis basis_;
    Basis backtrack_basis_;
    std::vector<double> backtrack_weights_;
    bool has_backtrack_ = false;
    bool backtracking_ = false;
    std::unique_ptr<FTBasis> factor_;
    int update_count_ = 0;
    int update_limit_ = std::numeric_limits<int>::max();
    long long iteration_count_ = 0;
    ModelStatus status_ = ModelStatus::NotSet;
    Solution solution_;

    std::vector<double> work_cost_, work_dual_, work_shift_;
    std::vector<double> work_lower_, work_upper_, work_range_, work_value_;
    std::vector<double> base_lower_, base_upper_, base_value_;
    std::vector<double> work_infeasibility_;
    std::vector<double> edge_weight_;
    bool weights_valid_ = false;
    std::vector<double> random_value_;
    std::vector<int> permutation_;
    std::mt19937 rng_;

    bool costs_shifted_ = false;
    bool costs_perturbed_ = false;
    bool allow_cost_perturbation_ = true;
    bool force_phase2_ = false;
    bool fresh_rebuild_ = false;
    int rebuild_reason_ = kRebuildNo;
    int dual_infeas_count_ = 0;
    double updated_dual_objective_ = 0.0;
    std::vector<int> taboo_rows_;
    struct BadBasisChange {
        int row_out;
        int variable_out;
        int variable_in;
        bool taboo;
    };
    std::vector<BadBasisChange> bad_changes_;
    std::vector<std::pair<int, double>> saved_infeasibility_;
    std::vector<uint64_t> hash_key_;
    uint64_t basis_hash_ = 0;
    std::unordered_set<uint64_t> visited_basis_;
    long long previous_cycling_iteration_ = -2;

    // Iteration data
    int row_out_ = -1, variable_out_ = -1, variable_in_ = -1;
    int move_out_ = 0;
    double delta_primal_ = 0.0, theta_dual_ = 0.0, theta_primal_ = 0.0;
    double alpha_col_ = 0.0, alpha_row_ = 0.0;
    WorkVector row_ep_, row_ap_, col_aq_, col_bfrt_, col_dse_;
    std::vector<int> price_mark_;
    std::vector<int> free_moved_;
    std::vector<int> free_vars_; // variables free under the current work bounds

    // Dual row (HEkkDualRow)
    std::vector<int> pack_index_;
    std::vector<double> pack_value_;
    int pack_count_ = 0;
    std::vector<std::pair<int, double>> work_data_;
    std::vector<int> work_group_;
    int work_count_ = 0;
    double work_theta_ = 0.0;
    int work_pivot_ = -1;
    double work_alpha_ = 0.0;
};

} // namespace simplex::ekk
