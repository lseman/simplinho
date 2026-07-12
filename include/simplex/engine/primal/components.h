#pragma once

#include "simplex/engine/common/utils.h"

namespace simplex::engine {

class PrimalPivotSelection : public BoundUtilities {
  public:
    struct RatioResult {
        std::optional<int> row;
        double theta = std::numeric_limits<double>::infinity();
        bool leaving_to_upper = false;
    };

    template <class RowRange>
    static RatioResult ratio_test_highs_style_(const Eigen::VectorXd& xB, const Eigen::VectorXd& dB,
                                               double sigma, const RowRange& rows,
                                               const std::vector<int>& basis,
                                               const Eigen::VectorXd& l, const Eigen::VectorXd& u,
                                               double alphaTol, double primal_feas_tol) {
        RatioResult out;
        out.row = std::nullopt;
        out.theta = std::numeric_limits<double>::infinity();
        out.leaving_to_upper = false;

        // Pass 1: compute relaxTheta
        double relaxTheta = std::numeric_limits<double>::infinity();

        for (int i : rows) {
            double d = sigma * dB(i);
            if (d > alphaTol) {
                const int j = basis[i];
                const double lo = j >= 0 && j < l.size() ? l(j) : 0.0;
                if (!std::isfinite(lo))
                    continue;
                double relaxSpace = xB(i) - lo + primal_feas_tol;
                if (relaxSpace < relaxTheta * d) {
                    relaxTheta = relaxSpace / d;
                }
            } else if (d < -alphaTol) {
                const int j = basis[i];
                const double hi =
                    j >= 0 && j < u.size() ? u(j) : std::numeric_limits<double>::infinity();
                if (!std::isfinite(hi))
                    continue;
                double relaxSpace = xB(i) - hi - primal_feas_tol;
                if (relaxSpace > relaxTheta * d) {
                    relaxTheta = relaxSpace / d;
                }
            }
        }

        if (std::isinf(relaxTheta))
            return out;

        // Pass 2: select row with largest |d| among those satisfying relaxed condition
        int best = -1;
        double bestAlpha = 0.0;

        for (int i : rows) {
            double d = sigma * dB(i);
            if (d > alphaTol) {
                const int j = basis[i];
                const double lo = j >= 0 && j < l.size() ? l(j) : 0.0;
                if (!std::isfinite(lo))
                    continue;
                double tightSpace = xB(i) - lo;
                if (tightSpace < relaxTheta * d) {
                    if (bestAlpha < d) {
                        bestAlpha = d;
                        best = i;
                    }
                }
            } else if (d < -alphaTol) {
                const int j = basis[i];
                const double hi =
                    j >= 0 && j < u.size() ? u(j) : std::numeric_limits<double>::infinity();
                if (!std::isfinite(hi))
                    continue;
                double tightSpace = xB(i) - hi;
                if (tightSpace > relaxTheta * d) {
                    if (bestAlpha < -d) {
                        bestAlpha = -d;
                        best = i;
                    }
                }
            }
        }

        if (best < 0)
            return out;

        out.row = best;
        double d = sigma * dB(best);
        if (d > alphaTol) {
            const int j = basis[best];
            const double lo = j >= 0 && j < l.size() ? l(j) : 0.0;
            out.theta = std::max(0.0, xB(best) - lo) / d;
            out.leaving_to_upper = false;
        } else if (d < -alphaTol) {
            const int j = basis[best];
            const double hi =
                j >= 0 && j < u.size() ? u(j) : std::numeric_limits<double>::infinity();
            out.theta = std::max(0.0, hi - xB(best)) / -d;
            out.leaving_to_upper = true;
        } else {
            out.theta = std::numeric_limits<double>::infinity();
        }

        return out;
    }

    struct AllRows {
        int count;
        struct Iterator {
            int value;
            int operator*() const { return value; }
            Iterator& operator++() {
                ++value;
                return *this;
            }
            bool operator!=(const Iterator& other) const { return value != other.value; }
        };
        Iterator begin() const { return {0}; }
        Iterator end() const { return {count}; }
    };

    static RatioResult ratio_test(const Eigen::VectorXd& xB, const HVector& dB, double sigma,
                                  const std::vector<int>& basis, const Eigen::VectorXd& l,
                                  const Eigen::VectorXd& u, double delta, double eta) {
        // HiGHS-style ratio test with relaxTheta
        double alphaTol = delta;      // delta is the degeneracy tolerance, used as alphaTol
        double primal_feas_tol = eta; // eta is used as feasibility tolerance for relaxation
        if (dB.has_pattern()) {
            std::vector<int> rows(dB.index.begin(), dB.index.begin() + dB.count);
            return ratio_test_highs_style_(xB, dB.value, sigma, rows, basis, l, u, alphaTol,
                                           primal_feas_tol);
        }
        return ratio_test_highs_style_(xB, dB.value, sigma,
                                       AllRows{static_cast<int>(dB.value.size())}, basis, l, u,
                                       alphaTol, primal_feas_tol);
    }

    struct ReducedCostView {
        Eigen::VectorXd raw;
        Eigen::VectorXd entering_measure;
    };

    static int nonbasic_move_(int j, const std::vector<char>& at_upper, const Eigen::VectorXd& l,
                              const Eigen::VectorXd& u, double tol) {
        const bool has_l = j < l.size() && std::isfinite(l(j));
        const bool has_u = j < u.size() && std::isfinite(u(j));
        if (has_l && has_u && u(j) - l(j) <= tol)
            return 0;
        if (at_upper[j])
            return -1;
        if (has_l)
            return 1;
        if (has_u)
            return -1;
        return 0;
    }

    template <class MatrixType>
    static ReducedCostView
    compute_reduced_costs_(const MatrixType& A, const Eigen::VectorXd& c, const Eigen::VectorXd& y,
                           const std::vector<int>& nonbasis, const std::vector<char>& at_upper,
                           const Eigen::VectorXd& l, const Eigen::VectorXd& u, double tol) {
        ReducedCostView out;
        out.raw.resize(nonbasis.size());
        out.entering_measure.resize(nonbasis.size());
        const Eigen::VectorXd aTy = A.transpose() * y;
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            const int j = nonbasis[k];
            out.raw(k) = c(j) - aTy(j);
            const int move = nonbasic_move_(j, at_upper, l, u, tol);
            if (move == 0) {
                const bool fixed = j < l.size() && j < u.size() && std::isfinite(l(j)) &&
                                   std::isfinite(u(j)) && u(j) - l(j) <= tol;
                out.entering_measure(k) = fixed ? 0.0 : -std::abs(out.raw(k));
            } else {
                out.entering_measure(k) = move * out.raw(k);
            }
        }
        return out;
    }

    static std::optional<int> choose_dantzig_entering_(const Eigen::VectorXd& measure, double tol) {
        int best = -1;
        double best_value = 0.0;
        for (int k = 0; k < measure.size(); ++k) {
            if (measure(k) < -tol && (best < 0 || measure(k) < best_value)) {
                best = k;
                best_value = measure(k);
            }
        }
        return best < 0 ? std::nullopt : std::optional<int>(best);
    }

    static std::optional<int> choose_bland_entering_(const Eigen::VectorXd& measure, double tol) {
        for (int k = 0; k < measure.size(); ++k)
            if (measure(k) < -tol)
                return k;
        return std::nullopt;
    }

    static double entering_direction_(int entering_col, double reduced_cost,
                                      const std::vector<char>& at_upper, const Eigen::VectorXd& l,
                                      const Eigen::VectorXd& u, double tol) {
        const int move = nonbasic_move_(entering_col, at_upper, l, u, tol);
        if (move)
            return move;
        const bool has_lower = entering_col < l.size() && std::isfinite(l(entering_col));
        return reduced_cost > 0.0 && !has_lower ? -1.0 : 1.0;
    }
};

} // namespace simplex::engine
