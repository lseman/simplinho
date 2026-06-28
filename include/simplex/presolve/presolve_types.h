#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <variant>
#include <vector>

namespace presolve {

// ------------------------------
// Problem container
// ------------------------------
enum class RowSense : int { LE = -1, EQ = 0, GE = 1 };

struct LP {
    Eigen::MatrixXd A;           // m x n
    Eigen::VectorXd b;           // m
    std::vector<RowSense> sense; // m
    Eigen::VectorXd c;           // n
    Eigen::VectorXd l;           // n  (can be -inf)
    Eigen::VectorXd u;           // n  (can be +inf)
    double c0 = 0.0;
};

struct BoundRelaxationSummary {
    int relaxed_lower = 0;
    int relaxed_upper = 0;
};

inline double inf() { return std::numeric_limits<double>::infinity(); }
inline double ninf() { return -std::numeric_limits<double>::infinity(); }
inline bool is_finite(double v) { return std::isfinite(v); }

struct ActRowReduce {
    Eigen::MatrixXd U;
    Eigen::VectorXi keep;
    int old_m = 0;
};
struct ActRemoveRow {
    int i;
    RowSense sense;
    double rhs;
    Eigen::VectorXd row;
};
struct ActRemoveCol {
    int j;
    double c_j;
    double l_j, u_j;
    Eigen::VectorXd col;
};
struct ActFixVar {
    int j;
    double x_fix;
    double c_j;
    Eigen::VectorXd col;
};
struct ActTightenBound {
    int j;
    double old_l, old_u;
};
struct ActScaleRow {
    int i;
    double scale;
};
struct ActScaleCol {
    int j;
    double scale;
};
struct ActSingletonRowElim {
    int i;
    int j;
    RowSense sense;
    double rhs;
    double aij;
    Eigen::VectorXd row;
};
struct ActSingletonColElim {
    int j;
    int i;
    double aij;
    Eigen::VectorXd col;
};
struct ActDualFix {
    int j;
    double old_l, old_u;
    double x_fix;
};

// Doubleton equality row: x_elim = (b_i - a_keep * x_keep) / a_elim
struct ActDoubletonEq {
    int col_elim;                  // column that was eliminated (expressed via col_keep)
    int col_keep;                  // column that remains in the reduced problem
    double a_elim;                 // coefficient of col_elim in the doubleton row
    double a_keep;                 // coefficient of col_keep in the doubleton row
    double b_row;                  // RHS of the doubleton equality row
    double old_l_elim, old_u_elim; // original bounds of col_elim
    double c_elim;                 // original objective coefficient of col_elim
};

using Action =
    std::variant<ActRowReduce, ActRemoveRow, ActRemoveCol, ActFixVar, ActTightenBound, ActScaleRow,
                 ActScaleCol, ActSingletonRowElim, ActSingletonColElim, ActDualFix, ActDoubletonEq>;

struct PresolveResult {
    LP reduced;
    std::vector<Action> stack;
    std::vector<int> orig_col_index;
    std::vector<int> orig_row_index;
    int original_num_cols = 0;
    int original_num_rows = 0;
    double obj_shift = 0.0;
    bool proven_infeasible = false;
    bool proven_unbounded = false;
    int implied_bound_updates = 0;
    int relaxed_huge_lower_bounds = 0;
    int relaxed_huge_upper_bounds = 0;
};

struct ActivityBounds {
    double min_act = 0.0, max_act = 0.0;
};

struct ActivityRange {
    double min_act = 0.0, max_act = 0.0;
    bool min_finite = true, max_finite = true;
};

struct ImpliedInterval {
    double lower = ninf(), upper = inf();
    bool has_lower = false, has_upper = false;
};

struct ImpliedBoundsSummary {
    Eigen::VectorXd impl_col_lower;
    Eigen::VectorXd impl_col_upper;
    std::vector<char> has_lower;
    std::vector<char> has_upper;

    explicit ImpliedBoundsSummary(int n = 0)
        : impl_col_lower(Eigen::VectorXd::Constant(n, ninf())),
          impl_col_upper(Eigen::VectorXd::Constant(n, inf())), has_lower(n, 0), has_upper(n, 0) {}
};

inline ActivityBounds row_activity_bounds(const Eigen::RowVectorXd& a, const Eigen::VectorXd& l,
                                          const Eigen::VectorXd& u) {
    ActivityBounds ab{0.0, 0.0};
    const int n = (int)a.size();
    for (int j = 0; j < n; ++j) {
        const double coeff = a(j);
        if (coeff >= 0.0) {
            ab.min_act += coeff * (is_finite(l(j)) ? l(j) : (coeff == 0.0 ? 0.0 : -inf()));
            ab.max_act += coeff * (is_finite(u(j)) ? u(j) : inf());
        } else {
            ab.min_act += coeff * (is_finite(u(j)) ? u(j) : inf());
            ab.max_act += coeff * (is_finite(l(j)) ? l(j) : -inf());
        }
    }
    return ab;
}

inline ActivityRange row_activity_range_excluding(const Eigen::RowVectorXd& a,
                                                  const Eigen::VectorXd& l,
                                                  const Eigen::VectorXd& u, int skip_j,
                                                  double zero_tol) {
    ActivityRange range;
    const int n = (int)a.size();
    for (int j = 0; j < n; ++j) {
        if (j == skip_j)
            continue;
        const double coeff = a(j);
        if (std::abs(coeff) <= zero_tol)
            continue;

        if (coeff >= 0.0) {
            if (range.min_finite) {
                if (is_finite(l(j)))
                    range.min_act += coeff * l(j);
                else
                    range.min_finite = false;
            }
            if (range.max_finite) {
                if (is_finite(u(j)))
                    range.max_act += coeff * u(j);
                else
                    range.max_finite = false;
            }
        } else {
            if (range.min_finite) {
                if (is_finite(u(j)))
                    range.min_act += coeff * u(j);
                else
                    range.min_finite = false;
            }
            if (range.max_finite) {
                if (is_finite(l(j)))
                    range.max_act += coeff * l(j);
                else
                    range.max_finite = false;
            }
        }
    }
    return range;
}

inline BoundRelaxationSummary canonicalize_inactive_huge_bounds(LP* problem, double zero_tol,
                                                                double huge_bound_factor = 1e6,
                                                                double relax_gap_factor = 1e6) {
    BoundRelaxationSummary summary;
    if (!problem || problem->A.rows() == 0 || problem->A.cols() == 0)
        return summary;

    double data_scale = 1.0;
    if (problem->A.size() > 0)
        data_scale = std::max(data_scale, problem->A.cwiseAbs().maxCoeff());
    if (problem->b.size() > 0)
        data_scale = std::max(data_scale, problem->b.cwiseAbs().maxCoeff());

    const double huge_bound = huge_bound_factor * data_scale;

    for (int j = 0; j < problem->A.cols(); ++j) {
        const Eigen::VectorXd col = problem->A.col(j);
        const double col_max = col.cwiseAbs().maxCoeff();

        // Skip zero columns - they can't constrain anything
        if (col_max <= zero_tol) {
            if (is_finite(problem->u(j)) && problem->u(j) > huge_bound)
                problem->u(j) = inf();
            if (is_finite(problem->l(j)) && problem->l(j) < -huge_bound)
                problem->l(j) = ninf();
            continue;
        }

        bool needs_upper_check = is_finite(problem->u(j)) && problem->u(j) > huge_bound;
        bool needs_lower_check = is_finite(problem->l(j)) && problem->l(j) < -huge_bound;

        if (!needs_upper_check && !needs_lower_check)
            continue;

        double implied_u = inf();
        double implied_l = ninf();
        bool has_implied_u = false;
        bool has_implied_l = false;

        int row_count = (int)problem->A.rows();
        for (int i = 0; i < row_count; ++i) {
            const double aij = col(i);
            if (std::abs(aij) <= zero_tol)
                continue;

            const auto other = row_activity_range_excluding(problem->A.row(i), problem->l,
                                                            problem->u, j, zero_tol);
            if (aij > 0.0) {
                if (other.min_finite) {
                    double val = (problem->b(i) - other.min_act) / aij;
                    if (val < implied_u) {
                        implied_u = val;
                        has_implied_u = true;
                    }
                }
                if (other.max_finite) {
                    double val = (problem->b(i) - other.max_act) / aij;
                    if (val > implied_l) {
                        implied_l = val;
                        has_implied_l = true;
                    }
                }
            } else {
                if (other.max_finite) {
                    double val = (problem->b(i) - other.max_act) / aij;
                    if (val < implied_u) {
                        implied_u = val;
                        has_implied_u = true;
                    }
                }
                if (other.min_finite) {
                    double val = (problem->b(i) - other.min_act) / aij;
                    if (val > implied_l) {
                        implied_l = val;
                        has_implied_l = true;
                    }
                }
            }

            // Early termination: if we found finite bounds, we can stop
            if (has_implied_u && has_implied_l && implied_u > implied_l) {
                if (implied_u > -inf()) {
                    const double ref = std::max({1.0, std::abs(implied_u), data_scale});
                    if (problem->u(j) > implied_u + relax_gap_factor * ref) {
                        problem->u(j) = inf();
                        ++summary.relaxed_upper;
                    }
                }
                if (implied_l < inf()) {
                    const double ref = std::max({1.0, std::abs(implied_l), data_scale});
                    if (problem->l(j) < implied_l - relax_gap_factor * ref) {
                        problem->l(j) = ninf();
                        ++summary.relaxed_lower;
                    }
                }
                break;
            }
        }

        // Final check if we didn't break early
        if (has_implied_u && is_finite(implied_u)) {
            const double ref = std::max({1.0, std::abs(implied_u), data_scale});
            if (problem->u(j) > implied_u + relax_gap_factor * ref) {
                problem->u(j) = inf();
                ++summary.relaxed_upper;
            }
        }
        if (has_implied_l && is_finite(implied_l)) {
            const double ref = std::max({1.0, std::abs(implied_l), data_scale});
            if (problem->l(j) < implied_l - relax_gap_factor * ref) {
                problem->l(j) = ninf();
                ++summary.relaxed_lower;
            }
        }
    }

    return summary;
}

inline bool nearly_zero(double v, double tol = 1e-12) { return std::abs(v) <= tol; }

inline double nearest_power_of_two_magnitude(double value) {
    if (!(std::isfinite(value)) || value <= 0.0)
        return 1.0;
    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    if (fraction < std::sqrt(0.5))
        --exponent;
    return std::ldexp(1.0, exponent);
}

template <class Derived> inline double safe_abs_max(const Eigen::MatrixBase<Derived>& x) {
    return x.size() ? x.cwiseAbs().maxCoeff() : 0.0;
}

enum class RowReduceMethod { RRQR, SVD, Auto };

}  // namespace presolve
