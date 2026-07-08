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
    static RatioResult ratio_test_core_(const Eigen::VectorXd& xB, const Eigen::VectorXd& dB,
                                        double sigma, const RowRange& rows,
                                        const std::vector<int>& basis, const Eigen::VectorXd& l,
                                        const Eigen::VectorXd& u, double delta, double eta) {
        RatioResult out;
        double theta_star = std::numeric_limits<double>::infinity();
        auto row_ratio = [&](int i, double& ratio, bool& to_upper) {
            const double d = sigma * dB(i);
            const int j = basis[i];
            if (d > delta) {
                const double lo = j >= 0 && j < l.size() ? l(j) : 0.0;
                if (!std::isfinite(lo))
                    return false;
                ratio = std::max(0.0, xB(i) - lo) / d;
                to_upper = false;
                return true;
            }
            if (d < -delta) {
                const double hi = j >= 0 && j < u.size()
                                      ? u(j)
                                      : std::numeric_limits<double>::infinity();
                if (!std::isfinite(hi))
                    return false;
                ratio = std::max(0.0, hi - xB(i)) / -d;
                to_upper = true;
                return true;
            }
            return false;
        };

        for (int i : rows) {
            double ratio;
            bool to_upper;
            if (row_ratio(i, ratio, to_upper))
                theta_star = std::min(theta_star, ratio);
        }
        if (!std::isfinite(theta_star))
            return out;

        const double kappa = std::max(eta, eta * theta_star);
        int best = -1;
        double best_pivot = 0.0;
        bool best_to_upper = false;
        for (int i : rows) {
            double ratio;
            bool to_upper;
            if (!row_ratio(i, ratio, to_upper) || ratio > theta_star + kappa)
                continue;
            const double pivot = std::abs(dB(i));
            if (pivot > best_pivot) {
                best = i;
                best_pivot = pivot;
                best_to_upper = to_upper;
            }
        }
        if (best < 0)
            return out;
        bool ignored;
        row_ratio(best, out.theta, ignored);
        out.theta = std::max(0.0, out.theta);
        out.row = best;
        out.leaving_to_upper = best_to_upper;
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
        if (dB.has_pattern()) {
            std::vector<int> rows(dB.index.begin(), dB.index.begin() + dB.count);
            return ratio_test_core_(xB, dB.value, sigma, rows, basis, l, u, delta, eta);
        }
        return ratio_test_core_(xB, dB.value, sigma,
                                AllRows{static_cast<int>(dB.value.size())}, basis, l, u, delta,
                                eta);
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

    static std::optional<int> choose_dantzig_entering_(const Eigen::VectorXd& measure,
                                                       double tol) {
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
