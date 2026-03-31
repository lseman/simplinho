#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bnb/conflict_graph.h"
#include "bnb/types.h"

namespace simplex::bnb::detail {

inline double cut_violation(const Cut& cut, const Eigen::VectorXd& primal) {
    double lhs = 0.0;
    for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                    k < static_cast<int>(cut.values.size());
         ++k) {
        const int index = cut.indices[k];
        if (index >= 0 && index < primal.size()) {
            lhs += cut.values[k] * primal(index);
        }
    }
    switch (cut.sense) {
        case LinearConstraintSense::LessEqual:
            return std::max(0.0, lhs - cut.rhs);
        case LinearConstraintSense::GreaterEqual:
            return std::max(0.0, cut.rhs - lhs);
        case LinearConstraintSense::Equal:
            return std::abs(lhs - cut.rhs);
    }
    return 0.0;
}

inline std::string cut_signature(const Cut& cut, int precision = 9) {
    std::vector<std::pair<int, double>> terms;
    terms.reserve(std::min(cut.indices.size(), cut.values.size()));
    for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                    k < static_cast<int>(cut.values.size());
         ++k) {
        if (std::abs(cut.values[k]) <= 1e-12) continue;
        const double scale = std::pow(10.0, precision);
        const double rounded = std::round(cut.values[k] * scale) / scale;
        terms.emplace_back(cut.indices[k], rounded);
    }
    std::sort(terms.begin(), terms.end());
    std::ostringstream oss;
    for (const auto& [index, coeff] : terms) {
        oss << index << ":" << coeff << ";";
    }
    const double scale = std::pow(10.0, precision);
    const double rounded_rhs = std::round(cut.rhs * scale) / scale;
    oss << "|rhs:" << rounded_rhs << "|sense:" << static_cast<int>(cut.sense);
    return oss.str();
}

inline bool canonicalize_cut(Cut* cut, double zero_tol = 1e-12) {
    if (cut == nullptr || !std::isfinite(cut->rhs)) return false;

    std::vector<std::pair<int, double>> merged;
    merged.reserve(std::min(cut->indices.size(), cut->values.size()));
    for (int k = 0; k < static_cast<int>(cut->indices.size()) &&
                    k < static_cast<int>(cut->values.size());
         ++k) {
        if (std::abs(cut->values[k]) <= zero_tol) continue;
        merged.emplace_back(cut->indices[k], cut->values[k]);
    }
    if (merged.empty()) return false;

    std::sort(merged.begin(), merged.end());
    std::vector<int> indices;
    std::vector<double> values;
    indices.reserve(merged.size());
    values.reserve(merged.size());
    for (const auto& [index, value] : merged) {
        if (!indices.empty() && indices.back() == index) {
            values.back() += value;
        } else {
            indices.push_back(index);
            values.push_back(value);
        }
    }

    double max_abs = 0.0;
    std::vector<int> final_indices;
    std::vector<double> final_values;
    final_indices.reserve(indices.size());
    final_values.reserve(values.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        if (std::abs(values[i]) <= zero_tol) continue;
        final_indices.push_back(indices[i]);
        final_values.push_back(values[i]);
        max_abs = std::max(max_abs, std::abs(values[i]));
    }
    if (final_indices.empty() || max_abs <= zero_tol) return false;

    for (double& value : final_values) value /= max_abs;
    cut->rhs /= max_abs;
    cut->indices = std::move(final_indices);
    cut->values = std::move(final_values);
    return std::isfinite(cut->rhs);
}

inline Cut clique_cut_from_literals(const Problem& problem,
                                    const std::vector<int>& clique_literals,
                                    const Options& options,
                                    const std::string& cut_type = "Clique") {
    Cut cut;
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = 1.0;
    cut.cut_type = cut_type;
    for (const int literal : clique_literals) {
        const int variable = ConflictGraph::variable_of(literal);
        if (variable < 0 || variable >= static_cast<int>(problem.variable_types.size())) continue;
        cut.indices.push_back(variable);
        if (ConflictGraph::value_of(literal)) {
            cut.values.push_back(1.0);
        } else {
            cut.values.push_back(-1.0);
            cut.rhs -= 1.0;
        }
    }
    if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3)) {
        cut.indices.clear();
    }
    return cut;
}

inline double cut_parallelism(const Cut& lhs, const Cut& rhs) {
    double lhs_norm_sq = 0.0;
    for (const double value : lhs.values) lhs_norm_sq += value * value;
    double rhs_norm_sq = 0.0;
    for (const double value : rhs.values) rhs_norm_sq += value * value;
    if (lhs_norm_sq <= 1e-16 || rhs_norm_sq <= 1e-16) return 0.0;

    double dot = 0.0;
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(lhs.indices.size()) && j < static_cast<int>(rhs.indices.size())) {
        if (lhs.indices[i] == rhs.indices[j]) {
            dot += lhs.values[i] * rhs.values[j];
            ++i;
            ++j;
        } else if (lhs.indices[i] < rhs.indices[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return std::abs(dot) / std::sqrt(lhs_norm_sq * rhs_norm_sq);
}

class CutPool {
   public:
    explicit CutPool(const Options& options = {})
        : max_pool_size_(options.max_cut_pool_size),
          min_violation_(options.min_cut_violation),
          max_age_(options.max_cut_age),
          max_cuts_per_type_(options.max_cuts_per_type),
          max_parallelism_(options.cut_max_parallelism) {}

    bool add_cut(const Cut& cut) {
        Cut canonical = cut;
        if (!canonicalize_cut(&canonical, min_violation_ * 1e-3)) return false;
        const std::string signature = cut_signature(canonical);
        if (signatures_.contains(signature)) {
            ++duplicate_cuts_;
            return false;
        }
        cuts_.push_back(std::move(canonical));
        signatures_.insert(signature);
        ++cuts_generated_;
        manage_pool_size_();
        return true;
    }

    std::vector<Cut> select_violated_cuts(const Eigen::VectorXd& primal, int max_cuts) {
        std::vector<std::pair<double, int>> scored;
        scored.reserve(cuts_.size());
        for (int i = 0; i < static_cast<int>(cuts_.size()); ++i) {
            const double violation = cut_violation(cuts_[i], primal);
            if (violation > min_violation_) {
                const double nnz = std::max<int>(1, cuts_[i].indices.size());
                const double efficacy = violation / std::sqrt(nnz);
                const double score =
                    0.75 * efficacy + 0.2 * cuts_[i].strength +
                    0.05 * (cuts_[i].times_used / std::max(1, cuts_[i].age + 1));
                scored.emplace_back(score, i);
                ++cuts_[i].times_used;
                cuts_[i].age = 0;
            } else {
                ++cuts_[i].age;
            }
        }

        std::sort(scored.begin(), scored.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });
        std::vector<Cut> selected;
        std::vector<int> selected_indices;
        std::unordered_map<std::string, int> type_counts;
        for (const auto& [_, index] : scored) {
            if (selected.size() >= static_cast<std::size_t>(max_cuts)) break;
            const Cut& candidate = cuts_[index];
            if (max_cuts_per_type_ > 0 &&
                type_counts[candidate.cut_type] >= max_cuts_per_type_) {
                continue;
            }
            bool too_parallel = false;
            for (const int selected_index : selected_indices) {
                if (cut_parallelism(candidate, cuts_[selected_index]) > max_parallelism_) {
                    too_parallel = true;
                    break;
                }
            }
            if (too_parallel) continue;
            selected.push_back(cuts_[index]);
            selected_indices.push_back(index);
            ++type_counts[candidate.cut_type];
        }
        cuts_applied_ += static_cast<int>(selected.size());
        manage_pool_size_();
        return selected;
    }

    int cuts_generated() const { return cuts_generated_; }
    int cuts_applied() const { return cuts_applied_; }
    int duplicate_cuts() const { return duplicate_cuts_; }
    int size() const { return static_cast<int>(cuts_.size()); }

   private:
    void manage_pool_size_() {
        if (cuts_.size() <= static_cast<std::size_t>(max_pool_size_)) return;

        std::erase_if(cuts_, [&](const Cut& cut) { return cut.age > max_age_; });
        if (cuts_.size() > static_cast<std::size_t>(max_pool_size_)) {
            std::sort(cuts_.begin(), cuts_.end(), [](const Cut& lhs, const Cut& rhs) {
                const double lhs_score =
                    lhs.times_used / static_cast<double>(lhs.age + 1) + lhs.strength;
                const double rhs_score =
                    rhs.times_used / static_cast<double>(rhs.age + 1) + rhs.strength;
                return lhs_score > rhs_score;
            });
            cuts_.resize(max_pool_size_);
        }
        signatures_.clear();
        for (const Cut& cut : cuts_) {
            signatures_.insert(cut_signature(cut));
        }
    }

    int max_pool_size_ = 256;
    double min_violation_ = 1e-4;
    int max_age_ = 5;
    int max_cuts_per_type_ = 4;
    double max_parallelism_ = 0.98;
    std::vector<Cut> cuts_;
    std::unordered_set<std::string> signatures_;
    int cuts_generated_ = 0;
    int cuts_applied_ = 0;
    int duplicate_cuts_ = 0;
};

inline std::optional<int> parse_internal_label_index(const std::string& label) {
    constexpr const char* prefix = "x_orig_";
    if (!label.starts_with(prefix)) return std::nullopt;
    try {
        return std::stoi(label.substr(std::char_traits<char>::length(prefix)));
    } catch (...) {
        return std::nullopt;
    }
}

inline double fractional_part(double value) {
    const double frac = value - std::floor(value);
    if (frac <= 1e-10) return 0.0;
    if (frac >= 1.0 - 1e-10) return 1.0;
    return frac;
}

inline std::vector<Cut> generate_gomory_cuts(const Problem& problem,
                                             const RelaxationSolution& relaxation,
                                             const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_gomory_cuts || !relaxation.lp_solution.has_value()) {
        return cuts;
    }

    const LPSolution& lp = *relaxation.lp_solution;
    if (!lp.has_internal_tableau || lp.tableau.rows() == 0 ||
        lp.tableau_rhs.size() != lp.tableau.rows()) {
        return cuts;
    }
    if (lp.internal_column_labels.size() != static_cast<std::size_t>(lp.tableau.cols()) ||
        lp.basis_internal.size() != static_cast<std::size_t>(lp.tableau.rows())) {
        return cuts;
    }

    for (int row = 0; row < lp.tableau.rows(); ++row) {
        const int basic_col = lp.basis_internal[row];
        if (basic_col < 0 || basic_col >= static_cast<int>(lp.internal_column_labels.size())) {
            continue;
        }
        const auto basic_index = parse_internal_label_index(lp.internal_column_labels[basic_col]);
        if (!basic_index.has_value() || *basic_index < 0 ||
            *basic_index >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[*basic_index] == VariableType::Continuous) {
            continue;
        }

        const double rhs = lp.tableau_rhs(row);
        const double f0 = fractional_part(rhs);
        if (std::min(f0, 1.0 - f0) <= options.min_cut_violation) {
            continue;
        }

        Cut cut;
        cut.sense = LinearConstraintSense::GreaterEqual;
        cut.rhs = f0 + 1e-9;
        cut.cut_type = "GMI";

        bool valid = true;
        for (int col = 0; col < lp.tableau.cols(); ++col) {
            if (col == basic_col) continue;
            const double tij = lp.tableau(row, col);
            if (std::abs(tij) <= 1e-10) continue;
            const auto mapped_index = parse_internal_label_index(lp.internal_column_labels[col]);
            if (!mapped_index.has_value() || *mapped_index < 0 ||
                *mapped_index >= static_cast<int>(problem.variable_types.size())) {
                valid = false;
                break;
            }

            double coefficient = 0.0;
            if (problem.variable_types[*mapped_index] != VariableType::Continuous) {
                const double fj = fractional_part(tij);
                coefficient = (fj <= f0)
                                  ? fj
                                  : ((std::abs(1.0 - f0) > 1e-10)
                                         ? (f0 * (1.0 - fj)) / (1.0 - f0)
                                         : 0.0);
            } else {
                coefficient = (tij > 0.0)
                                  ? tij
                                  : ((std::abs(1.0 - f0) > 1e-10)
                                         ? (-(f0 * tij) / (1.0 - f0))
                                         : 0.0);
            }
            if (std::abs(coefficient) > 1e-8) {
                cut.indices.push_back(*mapped_index);
                cut.values.push_back(coefficient);
            }
        }

        if (!valid || cut.indices.empty()) continue;
        const double violation = cut_violation(cut, relaxation.primal);
        if (violation <= options.min_cut_violation) continue;
        cut.strength = violation;
        cuts.push_back(std::move(cut));
    }

    return cuts;
}

inline std::vector<int> make_minimal_cover(std::vector<int> indices,
                                           const SparseLinearConstraint& row) {
    auto coeff_for = [&](int variable) {
        for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                        k < static_cast<int>(row.values.size());
             ++k) {
            if (row.indices[k] == variable) return row.values[k];
        }
        return 0.0;
    };

    double total = 0.0;
    for (const int index : indices) total += coeff_for(index);
    if (total <= row.rhs + 1e-9) return {};

    bool changed = true;
    while (changed && indices.size() > 1) {
        changed = false;
        std::sort(indices.begin(), indices.end(),
                  [&](int lhs, int rhs_idx) { return coeff_for(lhs) < coeff_for(rhs_idx); });
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            const double coeff = coeff_for(*it);
            if (total - coeff > row.rhs + 1e-9) {
                total -= coeff;
                indices.erase(it);
                changed = true;
                break;
            }
        }
    }
    return indices;
}

inline std::vector<Cut> generate_cover_cuts(const Problem& problem,
                                            const RelaxationSolution& relaxation,
                                            const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_cover_cuts) return cuts;

    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense != LinearConstraintSense::LessEqual) continue;

        struct BinaryTerm {
            int index = -1;
            double coeff = 0.0;
            double value = 0.0;
        };
        std::vector<BinaryTerm> binaries;
        for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                        k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            const double coeff = row.values[k];
            if (index < 0 || index >= static_cast<int>(problem.variable_types.size())) continue;
            if (problem.variable_types[index] != VariableType::Binary || coeff <= 1e-8) continue;
            binaries.push_back(BinaryTerm{index, coeff, relaxation.primal(index)});
        }
        if (binaries.size() < 2) continue;

        std::sort(binaries.begin(), binaries.end(), [](const auto& lhs, const auto& rhs) {
            if (std::abs(lhs.value - rhs.value) > 1e-12) return lhs.value > rhs.value;
            return lhs.coeff > rhs.coeff;
        });

        double total = 0.0;
        std::vector<int> cover;
        for (const BinaryTerm& term : binaries) {
            cover.push_back(term.index);
            total += term.coeff;
            if (total > row.rhs + 1e-9) break;
        }
        cover = make_minimal_cover(cover, row);
        if (cover.size() < 2) continue;

        Cut cut;
        cut.sense = LinearConstraintSense::LessEqual;
        cut.rhs = static_cast<double>(cover.size() - 1);
        cut.cut_type = "Cover";
        cut.indices = cover;
        cut.values.assign(cover.size(), 1.0);
        const double violation = cut_violation(cut, relaxation.primal);
        if (violation <= options.min_cut_violation) continue;
        cut.strength = violation;
        cuts.push_back(std::move(cut));
    }

    return cuts;
}

inline void append_implied_bound_cuts_from_leq(const Problem& problem,
                                               const std::vector<int>& indices,
                                               const std::vector<double>& values, double rhs,
                                               const RelaxationSolution& relaxation,
                                               const Options& options,
                                               std::vector<Cut>* cuts) {
    if (cuts == nullptr) return;

    for (int y_pos = 0; y_pos < static_cast<int>(indices.size()) &&
                        y_pos < static_cast<int>(values.size());
         ++y_pos) {
        const int y = indices[y_pos];
        const double y_coeff = values[y_pos];
        if (y < 0 || y >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[y] != VariableType::Binary ||
            std::abs(y_coeff) <= 1e-9) {
            continue;
        }

        const double tight_y_value = y_coeff > 0.0 ? 1.0 : 0.0;
        for (int x_pos = 0; x_pos < static_cast<int>(indices.size()) &&
                            x_pos < static_cast<int>(values.size());
             ++x_pos) {
            if (x_pos == y_pos) continue;
            const int x = indices[x_pos];
            const double x_coeff = values[x_pos];
            if (x < 0 || x >= problem.lower_bounds.size() || x_coeff <= 1e-9) continue;

            const double upper = problem.upper_bounds(x);
            if (!std::isfinite(upper)) continue;

            double min_other_activity = 0.0;
            bool finite = true;
            for (int k = 0; k < static_cast<int>(indices.size()) &&
                            k < static_cast<int>(values.size());
                 ++k) {
                if (k == x_pos || k == y_pos) continue;
                const int index = indices[k];
                const double coeff = values[k];
                if (index < 0 || index >= problem.lower_bounds.size() ||
                    std::abs(coeff) <= 1e-12) {
                    continue;
                }
                const double bound = coeff >= 0.0 ? problem.lower_bounds(index)
                                                  : problem.upper_bounds(index);
                if (!std::isfinite(bound)) {
                    finite = false;
                    break;
                }
                min_other_activity += coeff * bound;
            }
            if (!finite) continue;

            const double tightened_upper =
                (rhs - y_coeff * tight_y_value - min_other_activity) / x_coeff;
            if (!std::isfinite(tightened_upper) ||
                tightened_upper >= upper - options.min_cut_violation) {
                continue;
            }

            Cut cut;
            cut.sense = LinearConstraintSense::LessEqual;
            cut.rhs = tight_y_value > 0.5 ? upper : tightened_upper;
            cut.cut_type = "ImpliedBound";
            cut.indices = {x, y};
            cut.values = {1.0, tight_y_value > 0.5 ? (upper - tightened_upper)
                                                   : -(upper - tightened_upper)};
            const double violation = cut_violation(cut, relaxation.primal);
            if (violation <= options.min_cut_violation) continue;
            cut.strength = violation;
            cuts->push_back(std::move(cut));
        }
    }
}

inline std::vector<Cut> generate_implied_bound_cuts(const Problem& problem,
                                                    const RelaxationSolution& relaxation,
                                                    const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_implied_bound_cuts ||
        relaxation.primal.size() != problem.lower_bounds.size() ||
        problem.upper_bounds.size() != problem.lower_bounds.size()) {
        return cuts;
    }

    auto handle_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                          double rhs) {
        append_implied_bound_cuts_from_leq(problem, indices, values, rhs, relaxation, options,
                                           &cuts);
    };

    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense == LinearConstraintSense::LessEqual) {
            handle_row(row.indices, row.values, row.rhs);
        } else if (row.sense == LinearConstraintSense::GreaterEqual) {
            std::vector<double> negated = row.values;
            for (double& value : negated) value = -value;
            handle_row(row.indices, negated, -row.rhs);
        } else {
            handle_row(row.indices, row.values, row.rhs);
            std::vector<double> negated = row.values;
            for (double& value : negated) value = -value;
            handle_row(row.indices, negated, -row.rhs);
        }
    }

    return cuts;
}

inline void append_clique_cuts_from_leq(const Problem& problem,
                                        const std::vector<int>& indices,
                                        const std::vector<double>& values, double rhs,
                                        const RelaxationSolution& relaxation,
                                        const Options& options, std::vector<Cut>* cuts) {
    if (cuts == nullptr) return;

    struct BinaryTerm {
        int index = -1;
        double coeff = 0.0;
        double value = 0.0;
    };

    std::vector<BinaryTerm> binaries;
    for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
         ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size())) continue;
        if (problem.variable_types[index] != VariableType::Binary || coeff <= 1e-8) continue;
        binaries.push_back(BinaryTerm{index, coeff, relaxation.primal(index)});
    }
    if (binaries.size() < 3) return;

    std::sort(binaries.begin(), binaries.end(), [](const BinaryTerm& lhs, const BinaryTerm& rhs_term) {
        if (std::abs(lhs.value - rhs_term.value) > 1e-12) return lhs.value > rhs_term.value;
        if (std::abs(lhs.coeff - rhs_term.coeff) > 1e-12) return lhs.coeff > rhs_term.coeff;
        return lhs.index < rhs_term.index;
    });

    const int nbin = static_cast<int>(binaries.size());
    std::vector<std::vector<char>> conflicts(nbin, std::vector<char>(nbin, 0));
    for (int i = 0; i < nbin; ++i) {
        conflicts[i][i] = 1;
    }

    for (int i = 0; i < nbin; ++i) {
        for (int j = i + 1; j < nbin; ++j) {
            double min_other_activity = 0.0;
            bool finite = true;
            for (int k = 0; k < static_cast<int>(indices.size()) &&
                            k < static_cast<int>(values.size());
                 ++k) {
                const int index = indices[k];
                const double coeff = values[k];
                if (std::abs(coeff) <= 1e-12) continue;
                if (index == binaries[i].index || index == binaries[j].index) continue;
                if (index < 0 || index >= problem.lower_bounds.size()) continue;
                const double bound =
                    coeff >= 0.0 ? problem.lower_bounds(index) : problem.upper_bounds(index);
                if (!std::isfinite(bound)) {
                    finite = false;
                    break;
                }
                min_other_activity += coeff * bound;
            }
            if (!finite) continue;
            if (binaries[i].coeff + binaries[j].coeff + min_other_activity >
                rhs + options.min_cut_violation) {
                conflicts[i][j] = 1;
                conflicts[j][i] = 1;
            }
        }
    }

    std::unordered_set<std::string> local_signatures;
    std::vector<int> best_clique;
    double best_violation = options.min_cut_violation;

    for (int seed = 0; seed < nbin; ++seed) {
        std::vector<int> clique = {seed};
        double clique_value = binaries[seed].value;
        for (int cand = 0; cand < nbin; ++cand) {
            if (cand == seed) continue;
            bool compatible = true;
            for (const int chosen : clique) {
                if (!conflicts[cand][chosen]) {
                    compatible = false;
                    break;
                }
            }
            if (!compatible) continue;
            clique.push_back(cand);
            clique_value += binaries[cand].value;
        }

        if (clique.size() < 2) continue;
        const double violation = clique_value - 1.0;
        if (violation <= best_violation) continue;

        std::vector<int> clique_indices;
        clique_indices.reserve(clique.size());
        for (const int pos : clique) clique_indices.push_back(binaries[pos].index);
        std::sort(clique_indices.begin(), clique_indices.end());

        Cut candidate;
        candidate.sense = LinearConstraintSense::LessEqual;
        candidate.rhs = 1.0;
        candidate.cut_type = "Clique";
        candidate.indices = clique_indices;
        candidate.values.assign(clique_indices.size(), 1.0);
        const std::string signature = cut_signature(candidate);
        if (local_signatures.contains(signature)) continue;
        local_signatures.insert(signature);

        best_violation = violation;
        best_clique = std::move(clique_indices);
    }

    if (best_clique.empty()) return;

    Cut cut;
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = 1.0;
    cut.cut_type = "Clique";
    cut.indices = best_clique;
    cut.values.assign(best_clique.size(), 1.0);
    const double violation = cut_violation(cut, relaxation.primal);
    if (violation <= options.min_cut_violation) return;
    cut.strength = violation;
    cuts->push_back(std::move(cut));
}

inline std::vector<Cut> generate_clique_cuts(const Problem& problem,
                                             const RelaxationSolution& relaxation,
                                             const Options& options) {
    auto generate_graph_clique_cuts = [&]() {
        std::vector<Cut> cuts;
        if (!options.use_clique_cuts || relaxation.primal.size() != problem.lower_bounds.size()) {
            return cuts;
        }

        ConflictGraph graph(problem);
        const std::vector<int> vertices =
            graph.fractional_literals(relaxation.primal, options.integrality_tol);
        if (vertices.size() < 2) return cuts;

        std::vector<double> vertex_weights(vertices.size(), 0.0);
        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            vertex_weights[i] = ConflictGraph::literal_weight(relaxation.primal, vertices[i]);
        }

        std::vector<std::vector<char>> adjacent(vertices.size(),
                                                std::vector<char>(vertices.size(), 0));
        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            adjacent[i][i] = 1;
            for (int j = i + 1; j < static_cast<int>(vertices.size()); ++j) {
                if (graph.are_conflicting(vertices[i], vertices[j])) {
                    adjacent[i][j] = 1;
                    adjacent[j][i] = 1;
                }
            }
        }

        constexpr int kMaxCliqueCalls = 5000;
        const int max_found =
            std::max(options.max_cuts_added_per_round * 4, options.max_cuts_per_type);
        int calls = 0;
        std::vector<std::vector<int>> found;

        std::function<void(std::vector<int>, std::vector<int>, std::vector<int>, double)> search =
            [&](std::vector<int> clique, std::vector<int> candidates, std::vector<int> excluded,
                double clique_weight) {
                if (calls++ >= kMaxCliqueCalls ||
                    static_cast<int>(found.size()) >= max_found) {
                    return;
                }

                double upper_bound = clique_weight;
                for (const int v : candidates) upper_bound += vertex_weights[v];
                if (upper_bound <= 1.0 + options.min_cut_violation) return;

                if (candidates.empty() && excluded.empty()) {
                    if (clique_weight > 1.0 + options.min_cut_violation) {
                        found.push_back(std::move(clique));
                    }
                    return;
                }

                int pivot = -1;
                int best_neighbors = -1;
                std::vector<int> pivot_pool = candidates;
                pivot_pool.insert(pivot_pool.end(), excluded.begin(), excluded.end());
                for (const int u : pivot_pool) {
                    int count = 0;
                    for (const int v : candidates) count += adjacent[u][v] ? 1 : 0;
                    if (count > best_neighbors) {
                        best_neighbors = count;
                        pivot = u;
                    }
                }

                std::vector<int> expand = candidates;
                if (pivot >= 0) {
                    expand.erase(std::remove_if(expand.begin(), expand.end(),
                                                [&](int v) { return adjacent[pivot][v] != 0; }),
                                 expand.end());
                }

                for (const int v : expand) {
                    std::vector<int> next_clique = clique;
                    next_clique.push_back(v);
                    std::vector<int> next_candidates;
                    std::vector<int> next_excluded;
                    next_candidates.reserve(candidates.size());
                    next_excluded.reserve(excluded.size());
                    for (const int u : candidates) {
                        if (u != v && adjacent[v][u]) next_candidates.push_back(u);
                    }
                    for (const int u : excluded) {
                        if (adjacent[v][u]) next_excluded.push_back(u);
                    }

                    search(std::move(next_clique), std::move(next_candidates),
                           std::move(next_excluded), clique_weight + vertex_weights[v]);

                    candidates.erase(std::remove(candidates.begin(), candidates.end(), v),
                                     candidates.end());
                    excluded.push_back(v);
                }
            };

        std::vector<int> all_vertices(vertices.size(), 0);
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        search({}, all_vertices, {}, 0.0);

        std::unordered_set<std::string> signatures;
        for (std::vector<int>& clique_pos : found) {
            std::vector<int> clique_literals;
            clique_literals.reserve(clique_pos.size());
            for (const int pos : clique_pos) clique_literals.push_back(vertices[pos]);

            std::vector<int> extension_candidates;
            for (int j = 0; j < static_cast<int>(problem.variable_types.size()) && j < relaxation.primal.size();
                 ++j) {
                if (problem.variable_types[j] != VariableType::Binary) continue;
                if (relaxation.primal(j) > options.integrality_tol &&
                    relaxation.primal(j) < 1.0 - options.integrality_tol) {
                    continue;
                }
                const int literal = ConflictGraph::literal_for(j, relaxation.primal(j) >= 0.5);
                if (std::find(clique_literals.begin(), clique_literals.end(), literal) !=
                    clique_literals.end()) {
                    continue;
                }
                bool compatible = true;
                for (const int chosen : clique_literals) {
                    if (!graph.are_conflicting(literal, chosen)) {
                        compatible = false;
                        break;
                    }
                }
                if (compatible) extension_candidates.push_back(literal);
            }
            std::sort(extension_candidates.begin(), extension_candidates.end(),
                      [&](int lhs, int rhs) { return graph.degree(lhs) > graph.degree(rhs); });
            for (const int literal : extension_candidates) {
                bool compatible = true;
                for (const int chosen : clique_literals) {
                    if (!graph.are_conflicting(literal, chosen)) {
                        compatible = false;
                        break;
                    }
                }
                if (compatible) clique_literals.push_back(literal);
            }

            std::sort(clique_literals.begin(), clique_literals.end());
            clique_literals.erase(std::unique(clique_literals.begin(), clique_literals.end()),
                                 clique_literals.end());
            Cut cut = clique_cut_from_literals(problem, clique_literals, options);
            if (cut.indices.empty()) continue;
            const double violation = cut_violation(cut, relaxation.primal);
            if (violation <= options.min_cut_violation) continue;
            cut.strength = violation;
            const std::string signature = cut_signature(cut);
            if (signatures.contains(signature)) continue;
            signatures.insert(signature);
            cuts.push_back(std::move(cut));
        }

        return cuts;
    };

    std::vector<Cut> cuts;
    if (!options.use_clique_cuts || relaxation.primal.size() != problem.lower_bounds.size()) {
        return cuts;
    }

    auto handle_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                          double rhs) {
        append_clique_cuts_from_leq(problem, indices, values, rhs, relaxation, options, &cuts);
    };

    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense == LinearConstraintSense::LessEqual) {
            handle_row(row.indices, row.values, row.rhs);
        } else if (row.sense == LinearConstraintSense::Equal) {
            handle_row(row.indices, row.values, row.rhs);
            std::vector<double> negated = row.values;
            for (double& value : negated) value = -value;
            handle_row(row.indices, negated, -row.rhs);
        }
    }

    std::vector<Cut> graph_cuts = generate_graph_clique_cuts();
    cuts.insert(cuts.end(), std::make_move_iterator(graph_cuts.begin()),
                std::make_move_iterator(graph_cuts.end()));
    return cuts;
}

// ============================================================================
// Dual proof cuts
// ============================================================================
// Based on HiGHS: Uses dual values from LP solution to generate cuts
// that prove infeasibility of the current domain and tighten bounds

inline std::vector<Cut> generate_dual_proof_cuts(
    const Problem& problem, const RelaxationSolution& relaxation,
    const Options& options) {
  std::vector<Cut> cuts;

  // Only generate dual proof cuts when LP solution has dual values
  if (!relaxation.lp_solution.has_value()) {
    return cuts;
  }

  const auto& lp_sol = relaxation.lp_solution.value();
  const int n = static_cast<int>(problem.variable_types.size());

  // Collect columns with significant dual values (HiGHS-inspired)
  // We build a cut based on integer columns with significant dual values
  std::vector<int> inds;
  std::vector<double> vals;
  std::vector<double> bound_adjustments;
  double proof_rhs = 0.0;

  const double eps = options.feasibility_tol;  // Use feasibility_tol for dual value threshold

  // First pass: collect fractional integer variables with significant dual values
  for (int j = 0; j < n && static_cast<int>(inds.size()) <= 32; ++j) {
    // Only consider integer and binary variables
    if (problem.variable_types[j] != VariableType::Integer &&
        problem.variable_types[j] != VariableType::Binary) {
      continue;
    }

    const double dual = lp_sol.dual_values[j];
    const double abs_dual = std::abs(dual);

    // Skip columns with negligible dual values (HiGHS uses feasibility_tol)
    if (abs_dual < eps) {
      continue;
    }

    // Check if column is at its bound
    const double lb = problem.lower_bounds[j];
    const double ub = problem.upper_bounds[j];
    const double xj = relaxation.primal[j];

    bool at_lower = std::abs(xj - lb) < eps;
    bool at_upper = std::abs(xj - ub) < eps;

    // Add to proof if dual value is significant and column is fractional
    // (not at either bound) - this indicates the variable contributes to infeasibility
    if (abs_dual > eps && !at_lower && !at_upper) {
      // Apply the dual value contribution to the RHS
      proof_rhs += dual * xj;

      inds.push_back(j);
      vals.push_back(dual);
    }
  }

  if (inds.empty()) {
    return cuts;
  }

  // Create a cut based on the dual proof
  // This is a Gomory-style cut derived from the dual proof
  Cut cut;
  cut.cut_type = "DualProof";
  cut.sense = LinearConstraintSense::GreaterEqual;
  cut.rhs = proof_rhs;

  // Scale the coefficients for numerical stability
  const double scale = 1.0e6;
  for (size_t i = 0; i < vals.size(); ++i) {
    cut.indices.push_back(inds[i]);
    cut.values.push_back(vals[i] / scale);
  }

  // Only add if the cut is violated by the current solution
  const double violation = cut_violation(cut, relaxation.primal);
  if (violation > options.min_cut_violation) {
    cut.strength = violation;
    cuts.push_back(std::move(cut));
  }

  return cuts;
}

inline std::vector<Cut> generate_cuts(const Problem& problem,
                                      const RelaxationSolution& relaxation,
                                      const Options& options) {
  std::vector<Cut> cuts = generate_gomory_cuts(problem, relaxation, options);
  std::vector<Cut> cover_cuts = generate_cover_cuts(problem, relaxation, options);
  cuts.insert(cuts.end(), std::make_move_iterator(cover_cuts.begin()),
              std::make_move_iterator(cover_cuts.end()));
  std::vector<Cut> implied_bound_cuts =
      generate_implied_bound_cuts(problem, relaxation, options);
  cuts.insert(cuts.end(),
              std::make_move_iterator(implied_bound_cuts.begin()),
              std::make_move_iterator(implied_bound_cuts.end()));
  std::vector<Cut> clique_cuts = generate_clique_cuts(problem, relaxation, options);
  cuts.insert(cuts.end(),
              std::make_move_iterator(clique_cuts.begin()),
              std::make_move_iterator(clique_cuts.end()));
  // Dual proof cuts - HiGHS-inspired bound tightening
  if (options.use_dual_proof_cuts) {
    std::vector<Cut> dual_proof_cuts =
        generate_dual_proof_cuts(problem, relaxation, options);
    cuts.insert(cuts.end(),
                std::make_move_iterator(dual_proof_cuts.begin()),
                std::make_move_iterator(dual_proof_cuts.end()));
  }
  return cuts;
}

}  // namespace simplex::bnb::detail
