#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bnb/types.h"

namespace simplex::bnb::detail {

class ConflictGraph {
   public:
    explicit ConflictGraph(const Problem& problem, int min_clique_size = 5)
        : num_variables_(static_cast<int>(problem.variable_types.size())),
          min_clique_size_(std::max(3, min_clique_size)),
          adjacency_(2 * num_variables_),
          literal_clique_refs_(2 * num_variables_) {
        for (int j = 0; j < num_variables_; ++j) {
            if (problem.variable_types[j] != VariableType::Binary) continue;
            add_edge(literal_for(j, true), literal_for(j, false));
        }
        build(problem);
        finalize();
    }

    static int literal_for(int variable, bool value) { return 2 * variable + (value ? 1 : 0); }
    static int variable_of(int literal) { return literal / 2; }
    static bool value_of(int literal) { return (literal & 1) != 0; }
    static int complement_of(int literal) { return literal ^ 1; }

    int literal_count() const { return static_cast<int>(adjacency_.size()); }

    bool has_literal(int literal) const {
        return literal >= 0 && literal < static_cast<int>(adjacency_.size());
    }

    bool are_conflicting(int lhs, int rhs) const {
        if (lhs == rhs) return true;
        if (!has_literal(lhs) || !has_literal(rhs)) return false;
        if (std::binary_search(adjacency_[lhs].begin(), adjacency_[lhs].end(), rhs)) {
            return true;
        }
        const auto& refs = literal_clique_refs_[lhs];
        for (const int clique_id : refs) {
            const auto& clique = cliques_[clique_id];
            if (std::binary_search(clique.begin(), clique.end(), rhs)) return true;
        }
        return false;
    }

    std::vector<int> neighbors(int literal) const {
        if (!has_literal(literal)) return {};

        std::vector<int> out = adjacency_[literal];
        for (const int clique_id : literal_clique_refs_[literal]) {
            const auto& clique = cliques_[clique_id];
            out.insert(out.end(), clique.begin(), clique.end());
        }
        out.erase(std::remove(out.begin(), out.end(), literal), out.end());
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }

    int degree(int literal) const { return static_cast<int>(neighbors(literal).size()); }

    std::vector<int> fractional_literals(const Eigen::VectorXd& primal, double tol) const {
        std::vector<int> literals;
        for (int j = 0; j < num_variables_ && j < primal.size(); ++j) {
            if (primal(j) <= tol || primal(j) >= 1.0 - tol) continue;
            literals.push_back(literal_for(j, true));
            literals.push_back(literal_for(j, false));
        }
        return literals;
    }

    static double literal_weight(const Eigen::VectorXd& primal, int literal) {
        const int variable = variable_of(literal);
        if (variable < 0 || variable >= primal.size()) return 0.0;
        return value_of(literal) ? primal(variable) : (1.0 - primal(variable));
    }

   private:
    struct NormalizedRow {
        std::vector<int> literals;
        std::vector<double> coeffs;
        double rhs = 0.0;
    };

    void build(const Problem& problem) {
        for (const SparseLinearConstraint& row : problem.base_constraints) {
            std::vector<NormalizedRow> normalized =
                normalized_rows(problem, row.indices, row.values, row.rhs, row.sense);
            for (const NormalizedRow& knapsack : normalized) {
                add_knapsack_cliques(knapsack);
            }
        }
    }

    void finalize() {
        for (auto& neighbors : adjacency_) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        for (auto& refs : literal_clique_refs_) {
            std::sort(refs.begin(), refs.end());
            refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
        }
    }

    static std::vector<NormalizedRow> normalized_rows(const Problem& problem,
                                                      const std::vector<int>& indices,
                                                      const std::vector<double>& values, double rhs,
                                                      LinearConstraintSense sense) {
        std::vector<NormalizedRow> rows;
        auto add_less_equal = [&](const std::vector<double>& coeffs, double row_rhs) {
            NormalizedRow out;
            out.rhs = row_rhs;
            for (int k = 0; k < static_cast<int>(indices.size()) &&
                            k < static_cast<int>(coeffs.size());
                 ++k) {
                const int index = indices[k];
                const double coeff = coeffs[k];
                if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
                    std::abs(coeff) <= 1e-12) {
                    continue;
                }

                if (problem.variable_types[index] == VariableType::Binary) {
                    if (coeff > 0.0) {
                        out.literals.push_back(literal_for(index, true));
                        out.coeffs.push_back(coeff);
                    } else {
                        out.literals.push_back(literal_for(index, false));
                        out.coeffs.push_back(-coeff);
                        out.rhs -= coeff;
                    }
                    continue;
                }

                const double bound =
                    coeff >= 0.0 ? problem.lower_bounds(index) : problem.upper_bounds(index);
                if (!std::isfinite(bound)) {
                    out.literals.clear();
                    out.coeffs.clear();
                    out.rhs = std::numeric_limits<double>::quiet_NaN();
                    return;
                }
                out.rhs -= coeff * bound;
            }
            if (std::isfinite(out.rhs) && out.literals.size() >= 2) {
                rows.push_back(std::move(out));
            }
        };

        if (sense == LinearConstraintSense::LessEqual) {
            add_less_equal(values, rhs);
        } else if (sense == LinearConstraintSense::GreaterEqual) {
            std::vector<double> negated(values.size(), 0.0);
            for (int k = 0; k < static_cast<int>(values.size()); ++k) negated[k] = -values[k];
            add_less_equal(negated, -rhs);
        } else {
            add_less_equal(values, rhs);
            std::vector<double> negated(values.size(), 0.0);
            for (int k = 0; k < static_cast<int>(values.size()); ++k) negated[k] = -values[k];
            add_less_equal(negated, -rhs);
        }

        return rows;
    }

    void add_knapsack_cliques(const NormalizedRow& row) {
        if (row.literals.size() < 2 || !std::isfinite(row.rhs)) return;

        std::vector<int> order(row.literals.size(), 0);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int lhs, int rhs_idx) {
            if (std::abs(row.coeffs[lhs] - row.coeffs[rhs_idx]) > 1e-12) {
                return row.coeffs[lhs] < row.coeffs[rhs_idx];
            }
            return row.literals[lhs] < row.literals[rhs_idx];
        });

        std::vector<int> literals;
        std::vector<double> coeffs;
        literals.reserve(order.size());
        coeffs.reserve(order.size());
        for (const int pos : order) {
            literals.push_back(row.literals[pos]);
            coeffs.push_back(row.coeffs[pos]);
        }

        const int n = static_cast<int>(literals.size());
        if (n < 2 || coeffs[n - 2] + coeffs[n - 1] <= row.rhs + 1e-12) return;

        int left = 0;
        int right = n - 2;
        int first = n - 2;
        while (left <= right) {
            const int mid = left + (right - left) / 2;
            if (coeffs[mid] + coeffs[mid + 1] > row.rhs + 1e-12) {
                first = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        add_clique(std::vector<int>(literals.begin() + first, literals.end()));

        for (int outside = first - 1; outside >= 0; --outside) {
            int lo = first;
            int hi = n - 1;
            int suffix_start = -1;
            while (lo <= hi) {
                const int mid = lo + (hi - lo) / 2;
                if (coeffs[outside] + coeffs[mid] > row.rhs + 1e-12) {
                    suffix_start = mid;
                    hi = mid - 1;
                } else {
                    lo = mid + 1;
                }
            }
            if (suffix_start < 0) break;

            std::vector<int> clique;
            clique.reserve(n - suffix_start + 1);
            clique.push_back(literals[outside]);
            clique.insert(clique.end(), literals.begin() + suffix_start, literals.end());
            add_clique(std::move(clique));
        }
    }

    void add_edge(int lhs, int rhs) {
        if (lhs == rhs || !has_literal(lhs) || !has_literal(rhs)) return;
        adjacency_[lhs].push_back(rhs);
        adjacency_[rhs].push_back(lhs);
    }

    void add_clique(std::vector<int> clique) {
        std::sort(clique.begin(), clique.end());
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
        if (clique.size() < 2) return;

        std::string signature;
        signature.reserve(8 * clique.size());
        for (const int literal : clique) {
            signature.append(std::to_string(literal));
            signature.push_back(';');
        }
        if (clique_signatures_.contains(signature)) return;
        clique_signatures_.insert(signature);

        if (static_cast<int>(clique.size()) < min_clique_size_) {
            for (int i = 0; i < static_cast<int>(clique.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(clique.size()); ++j) {
                    add_edge(clique[i], clique[j]);
                }
            }
            return;
        }

        const int clique_id = static_cast<int>(cliques_.size());
        cliques_.push_back(clique);
        for (const int literal : cliques_.back()) {
            literal_clique_refs_[literal].push_back(clique_id);
        }
    }

    int num_variables_ = 0;
    int min_clique_size_ = 5;
    std::vector<std::vector<int>> adjacency_;
    std::vector<std::vector<int>> cliques_;
    std::vector<std::vector<int>> literal_clique_refs_;
    std::unordered_set<std::string> clique_signatures_;
};

}  // namespace simplex::bnb::detail
