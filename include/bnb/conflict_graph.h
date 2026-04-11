#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bnb/types.h"

namespace simplex::bnb::detail {

struct CliqueSignature {
    std::uint64_t lo = 0x9ae16a3b2f90404fULL;
    std::uint64_t hi = 0xc3a5c85c97cb3127ULL;

    bool operator==(const CliqueSignature&) const noexcept = default;
};

struct CliqueSignatureHash {
    std::size_t operator()(const CliqueSignature& signature) const noexcept {
        std::uint64_t combined = signature.lo ^ (signature.hi + 0x9e3779b97f4a7c15ULL +
                                                 (signature.lo << 6) + (signature.lo >> 2));
        return static_cast<std::size_t>(combined);
    }
};

inline std::uint64_t clique_signature_mix_(std::uint64_t seed, std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

inline void clique_signature_combine_(CliqueSignature* signature, std::uint64_t value) noexcept {
    if (signature == nullptr)
        return;
    signature->lo = clique_signature_mix_(signature->lo, value);
    signature->hi = clique_signature_mix_(signature->hi, value ^ 0x517cc1b727220a95ULL);
}

struct NormalizedCliqueRow {
    std::vector<int> literals;
    std::vector<double> coeffs;
    double rhs = 0.0;
};

inline void normalize_clique_less_equal_row_(const Problem& problem,
                                             const std::vector<int>& indices,
                                             const std::vector<double>& values, double rhs,
                                             std::vector<NormalizedCliqueRow>* rows) {
    if (rows == nullptr)
        return;

    NormalizedCliqueRow out;
    out.rhs = rhs;
    for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
         ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
            std::abs(coeff) <= 1e-12) {
            continue;
        }

        if (problem.variable_types[index] == VariableType::Binary) {
            if (coeff > 0.0) {
                out.literals.push_back(2 * index + 1);
                out.coeffs.push_back(coeff);
            } else {
                out.literals.push_back(2 * index);
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

    if (std::isfinite(out.rhs) && out.literals.size() >= 2)
        rows->push_back(std::move(out));
}

inline std::vector<NormalizedCliqueRow>
normalized_clique_rows(const Problem& problem, const std::vector<int>& indices,
                       const std::vector<double>& values, double rhs, LinearConstraintSense sense) {
    std::vector<NormalizedCliqueRow> rows;
    if (sense == LinearConstraintSense::LessEqual) {
        normalize_clique_less_equal_row_(problem, indices, values, rhs, &rows);
    } else if (sense == LinearConstraintSense::GreaterEqual) {
        std::vector<double> negated(values.size(), 0.0);
        for (int k = 0; k < static_cast<int>(values.size()); ++k)
            negated[k] = -values[k];
        normalize_clique_less_equal_row_(problem, indices, negated, -rhs, &rows);
    } else {
        normalize_clique_less_equal_row_(problem, indices, values, rhs, &rows);
        std::vector<double> negated(values.size(), 0.0);
        for (int k = 0; k < static_cast<int>(values.size()); ++k)
            negated[k] = -values[k];
        normalize_clique_less_equal_row_(problem, indices, negated, -rhs, &rows);
    }
    return rows;
}

inline std::vector<NormalizedCliqueRow> normalized_clique_rows(const Problem& problem,
                                                               const SparseLinearConstraint& row) {
    return normalized_clique_rows(problem, row.indices, row.values, row.rhs, row.sense);
}

inline void extract_binary_knapsack_cliques(const NormalizedCliqueRow& row,
                                            std::vector<std::vector<int>>* cliques,
                                            int max_cliques = std::numeric_limits<int>::max()) {
    if (cliques == nullptr || row.literals.size() < 2 || !std::isfinite(row.rhs) ||
        max_cliques <= 0) {
        return;
    }

    std::vector<int> order(row.literals.size(), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs_idx) {
        if (std::abs(row.coeffs[lhs] - row.coeffs[rhs_idx]) > 1e-12)
            return row.coeffs[lhs] < row.coeffs[rhs_idx];
        return row.literals[lhs] < row.literals[rhs_idx];
    });

    std::vector<int> literals;
    std::vector<double> coeffs;
    literals.reserve(order.size());
    coeffs.reserve(order.size());
    for (int pos : order) {
        literals.push_back(row.literals[pos]);
        coeffs.push_back(row.coeffs[pos]);
    }

    const int n = static_cast<int>(literals.size());
    if (n < 2 || coeffs[n - 2] + coeffs[n - 1] <= row.rhs + 1e-12)
        return;

    auto push_clique = [&](std::vector<int> clique) {
        if (static_cast<int>(cliques->size()) >= max_cliques || clique.size() < 2)
            return;
        std::sort(clique.begin(), clique.end());
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
        if (clique.size() >= 2)
            cliques->push_back(std::move(clique));
    };

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

    push_clique(std::vector<int>(literals.begin() + first, literals.end()));

    for (int outside = first - 1; outside >= 0 && static_cast<int>(cliques->size()) < max_cliques;
         --outside) {
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
        if (suffix_start < 0)
            break;

        std::vector<int> clique;
        clique.reserve(n - suffix_start + 1);
        clique.push_back(literals[outside]);
        clique.insert(clique.end(), literals.begin() + suffix_start, literals.end());
        push_clique(std::move(clique));
    }
}

class ConflictGraph {
  public:
    explicit ConflictGraph(const Problem& problem, int min_clique_size = 5)
        : problem_(&problem), num_variables_(static_cast<int>(problem.variable_types.size())),
          min_clique_size_(std::max(3, min_clique_size)), adjacency_(2 * num_variables_),
          literal_clique_refs_(2 * num_variables_) {
        for (int j = 0; j < num_variables_; ++j) {
            if (problem.variable_types[j] != VariableType::Binary)
                continue;
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
        if (lhs == rhs)
            return true;
        if (!has_literal(lhs) || !has_literal(rhs))
            return false;
        return std::binary_search(merged_neighbors_[lhs].begin(), merged_neighbors_[lhs].end(),
                                  rhs);
    }

    const std::vector<int>& neighbors(int literal) const {
        static const std::vector<int> kEmpty;
        if (!has_literal(literal))
            return kEmpty;
        return merged_neighbors_[literal];
    }

    int degree(int literal) const {
        if (!has_literal(literal))
            return 0;
        return neighbor_degrees_[literal];
    }

    const std::vector<std::vector<int>>& cliques() const { return cliques_; }

    void prepare_for_queries() { finalize(); }

    void add_conflict_edge(int lhs, int rhs) {
        if (lhs == rhs || !has_literal(lhs) || !has_literal(rhs))
            return;
        insert_sorted_unique_(&adjacency_[lhs], rhs);
        insert_sorted_unique_(&adjacency_[rhs], lhs);
    }

    void add_implication(int trigger_literal, int consequence_literal) {
        if (!has_literal(trigger_literal) || !has_literal(consequence_literal))
            return;
        add_conflict_edge(trigger_literal, complement_of(consequence_literal));
    }

    std::vector<int> fractional_literals(const Eigen::VectorXd& primal, double tol) const {
        std::vector<int> literals;
        for (int j = 0; j < num_variables_ && j < primal.size(); ++j) {
            if (primal(j) <= tol || primal(j) >= 1.0 - tol)
                continue;
            literals.push_back(literal_for(j, true));
            literals.push_back(literal_for(j, false));
        }
        return literals;
    }

    static double literal_weight(const Eigen::VectorXd& primal, int literal) {
        const int variable = variable_of(literal);
        if (variable < 0 || variable >= primal.size())
            return 0.0;
        return value_of(literal) ? primal(variable) : (1.0 - primal(variable));
    }

    void add_row_cliques(const std::vector<int>& indices, const std::vector<double>& values,
                         double rhs, LinearConstraintSense sense) {
        if (problem_ == nullptr)
            return;
        std::vector<NormalizedCliqueRow> normalized =
            normalized_clique_rows(*problem_, indices, values, rhs, sense);
        for (const NormalizedCliqueRow& row : normalized)
            add_knapsack_cliques(row);
        finalize();
    }

    void add_cut_cliques(const Cut& cut) {
        if (cut.cut_type == "IncumbentCutoff")
            return;
        add_row_cliques(cut.indices, cut.values, cut.rhs, cut.sense);
    }

  private:
    static void insert_sorted_unique_(std::vector<int>* values, int value) {
        if (values == nullptr)
            return;
        const auto pos = std::lower_bound(values->begin(), values->end(), value);
        if (pos == values->end() || *pos != value)
            values->insert(pos, value);
    }

    void build(const Problem& problem) {
        for (const SparseLinearConstraint& row : problem.base_constraints) {
            std::vector<NormalizedCliqueRow> normalized = normalized_clique_rows(problem, row);
            for (const NormalizedCliqueRow& knapsack : normalized)
                add_knapsack_cliques(knapsack);
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
        rebuild_query_cache_();
    }

    void rebuild_query_cache_() {
        merged_neighbors_.assign(adjacency_.size(), {});
        neighbor_degrees_.assign(adjacency_.size(), 0);
        for (int literal = 0; literal < static_cast<int>(adjacency_.size()); ++literal) {
            std::vector<int>& out = merged_neighbors_[literal];
            out = adjacency_[literal];
            for (const int clique_id : literal_clique_refs_[literal]) {
                const auto& clique = cliques_[clique_id];
                out.insert(out.end(), clique.begin(), clique.end());
            }
            out.erase(std::remove(out.begin(), out.end(), literal), out.end());
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            neighbor_degrees_[literal] = static_cast<int>(out.size());
        }
    }

    void add_knapsack_cliques(const NormalizedCliqueRow& row) {
        std::vector<std::vector<int>> extracted;
        extract_binary_knapsack_cliques(row, &extracted);
        for (std::vector<int>& clique : extracted)
            add_clique(std::move(clique));
    }

    void add_edge(int lhs, int rhs) {
        if (lhs == rhs || !has_literal(lhs) || !has_literal(rhs))
            return;
        adjacency_[lhs].push_back(rhs);
        adjacency_[rhs].push_back(lhs);
    }

    void add_clique(std::vector<int> clique) {
        std::sort(clique.begin(), clique.end());
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
        if (clique.size() < 2)
            return;

        CliqueSignature signature;
        clique_signature_combine_(&signature, static_cast<std::uint64_t>(clique.size()));
        for (const int literal : clique) {
            clique_signature_combine_(&signature, static_cast<std::uint64_t>(literal));
        }
        if (clique_signatures_.contains(signature))
            return;
        clique_signatures_.insert(signature);

        const int clique_id = static_cast<int>(cliques_.size());
        cliques_.push_back(clique);
        for (const int literal : cliques_.back())
            literal_clique_refs_[literal].push_back(clique_id);

        if (static_cast<int>(clique.size()) < min_clique_size_) {
            for (int i = 0; i < static_cast<int>(clique.size()); ++i) {
                for (int j = i + 1; j < static_cast<int>(clique.size()); ++j) {
                    add_edge(clique[i], clique[j]);
                }
            }
        }
    }

    const Problem* problem_ = nullptr;
    int num_variables_ = 0;
    int min_clique_size_ = 5;
    std::vector<std::vector<int>> adjacency_;
    std::vector<std::vector<int>> cliques_;
    std::vector<std::vector<int>> literal_clique_refs_;
    std::vector<std::vector<int>> merged_neighbors_;
    std::vector<int> neighbor_degrees_;
    std::unordered_set<CliqueSignature, CliqueSignatureHash> clique_signatures_;
};

} // namespace simplex::bnb::detail
