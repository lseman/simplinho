#include "bnb/mip_presolve.h"
#include "bnb/conflict_graph.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace simplex::bnb::presolve {

namespace detail {

struct SparseRowView {
    const std::vector<int>* indices = nullptr;
    const std::vector<double>* values = nullptr;
    LinearConstraintSense sense = LinearConstraintSense::Equal;
    double rhs = 0.0;
};

struct SparseVariableContribution {
    double min_value = 0.0;
    double max_value = 0.0;
    bool min_finite = true;
    bool max_finite = true;
};

struct SparseRowActivitySummary {
    double min_activity = 0.0;
    double max_activity = 0.0;
    int min_infinite_terms = 0;
    int max_infinite_terms = 0;
};

inline SparseVariableContribution sparse_variable_contribution(double coeff, int index,
                                                               const Eigen::VectorXd& lower,
                                                               const Eigen::VectorXd& upper) {
    SparseVariableContribution contribution;
    if (index < 0 || index >= lower.size() || index >= upper.size() ||
        std::abs(coeff) <= kCoeffTol) {
        return contribution;
    }

    const double lo = lower(index);
    const double up = upper(index);
    if (coeff >= 0.0) {
        contribution.min_finite = std::isfinite(lo);
        contribution.max_finite = std::isfinite(up);
        if (contribution.min_finite)
            contribution.min_value = coeff * lo;
        if (contribution.max_finite)
            contribution.max_value = coeff * up;
    } else {
        contribution.min_finite = std::isfinite(up);
        contribution.max_finite = std::isfinite(lo);
        if (contribution.min_finite)
            contribution.min_value = coeff * up;
        if (contribution.max_finite)
            contribution.max_value = coeff * lo;
    }

    return contribution;
}

inline SparseRowActivitySummary sparse_row_activity_summary(const SparseRowView& row,
                                                            const Eigen::VectorXd& lower,
                                                            const Eigen::VectorXd& upper) {
    SparseRowActivitySummary summary;
    if (!row.indices || !row.values)
        return summary;

    for (int k = 0;
         k < static_cast<int>(row.indices->size()) && k < static_cast<int>(row.values->size());
         ++k) {
        const int index = (*row.indices)[k];
        const double coeff = (*row.values)[k];
        const SparseVariableContribution contribution =
            sparse_variable_contribution(coeff, index, lower, upper);

        if (contribution.min_finite) {
            summary.min_activity += contribution.min_value;
        } else if (std::abs(coeff) > kCoeffTol) {
            ++summary.min_infinite_terms;
        }

        if (contribution.max_finite) {
            summary.max_activity += contribution.max_value;
        } else if (std::abs(coeff) > kCoeffTol) {
            ++summary.max_infinite_terms;
        }
    }

    return summary;
}

inline bool sparse_row_is_feasible(const SparseRowActivitySummary& summary,
                                   LinearConstraintSense sense, double rhs, double tol) {
    switch (sense) {
        case LinearConstraintSense::LessEqual:
            return summary.min_infinite_terms > 0 || summary.min_activity <= rhs + tol;
        case LinearConstraintSense::GreaterEqual:
            return summary.max_infinite_terms > 0 || summary.max_activity >= rhs - tol;
        case LinearConstraintSense::Equal:
            return (summary.min_infinite_terms > 0 || summary.min_activity <= rhs + tol) &&
                   (summary.max_infinite_terms > 0 || summary.max_activity >= rhs - tol);
    }
    return true;
}

inline bool sparse_row_is_redundant(const SparseRowActivitySummary& summary,
                                    LinearConstraintSense sense, double rhs, double tol) {
    switch (sense) {
        case LinearConstraintSense::LessEqual:
            return summary.max_infinite_terms == 0 && summary.max_activity <= rhs + tol;
        case LinearConstraintSense::GreaterEqual:
            return summary.min_infinite_terms == 0 && summary.min_activity >= rhs - tol;
        case LinearConstraintSense::Equal:
            return summary.min_infinite_terms == 0 && summary.max_infinite_terms == 0 &&
                   summary.min_activity >= rhs - tol && summary.max_activity <= rhs + tol;
    }
    return false;
}

inline std::string sparse_row_signature(const SparseLinearConstraint& row, int precision = 12) {
    const double scale = std::pow(10.0, precision);
    std::ostringstream oss;
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        const double coeff = row.values[k];
        if (std::abs(coeff) <= kCoeffTol)
            continue;
        const double rounded = std::round(coeff * scale) / scale;
        oss << row.indices[k] << ':' << rounded << ';';
    }
    const double rounded_rhs = std::round(row.rhs * scale) / scale;
    oss << "|rhs:" << rounded_rhs << "|sense:" << static_cast<int>(row.sense);
    return oss.str();
}

inline std::string sparse_row_lhs_signature(const SparseLinearConstraint& row, int precision = 12) {
    const double scale = std::pow(10.0, precision);
    std::ostringstream oss;
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        const double coeff = row.values[k];
        if (std::abs(coeff) <= kCoeffTol)
            continue;
        const double rounded = std::round(coeff * scale) / scale;
        oss << row.indices[k] << ':' << rounded << ';';
    }
    return oss.str();
}

inline LinearConstraintSense flipped_sense(LinearConstraintSense sense) {
    switch (sense) {
        case LinearConstraintSense::LessEqual:
            return LinearConstraintSense::GreaterEqual;
        case LinearConstraintSense::GreaterEqual:
            return LinearConstraintSense::LessEqual;
        case LinearConstraintSense::Equal:
            return LinearConstraintSense::Equal;
    }
    return sense;
}

inline bool is_integral_value(double value, double tol) {
    if (!std::isfinite(value))
        return false;
    return std::abs(value - std::round(value)) <= tol;
}

inline SparseLinearConstraint normalize_parallel_row(const SparseLinearConstraint& input,
                                                     double tol) {
    SparseLinearConstraint row = input;
    int pivot = -1;
    for (int k = 0; k < static_cast<int>(row.values.size()); ++k) {
        if (std::abs(row.values[k]) > tol) {
            pivot = k;
            break;
        }
    }
    if (pivot < 0)
        return row;

    if (row.values[pivot] < 0.0) {
        for (double& value : row.values)
            value = -value;
        row.rhs = -row.rhs;
        row.sense = flipped_sense(row.sense);
    }

    const double scale = std::abs(row.values[pivot]);
    if (!(scale > tol))
        return row;

    for (double& value : row.values) {
        value /= scale;
        if (std::abs(value) <= tol)
            value = 0.0;
    }
    row.rhs /= scale;
    if (std::abs(row.rhs) <= tol)
        row.rhs = 0.0;
    return row;
}

struct RowEnvelope {
    SparseLinearConstraint prototype;
    std::optional<double> lower_rhs;
    std::optional<double> upper_rhs;
    int source_rows = 0;
};

struct MergeRowsResult {
    bool infeasible = false;
    int removed_rows = 0;
    std::vector<SparseLinearConstraint> rows;
};

inline bool absorb_row_into_envelope(RowEnvelope* envelope, const SparseLinearConstraint& row,
                                     double tol) {
    if (envelope == nullptr)
        return true;

    if (envelope->source_rows == 0) {
        envelope->prototype = row;
        envelope->prototype.rhs = 0.0;
        envelope->prototype.sense = LinearConstraintSense::Equal;
    }
    ++envelope->source_rows;

    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            if (!envelope->upper_rhs.has_value() || row.rhs < *envelope->upper_rhs - tol) {
                envelope->upper_rhs = row.rhs;
            }
            break;
        case LinearConstraintSense::GreaterEqual:
            if (!envelope->lower_rhs.has_value() || row.rhs > *envelope->lower_rhs + tol) {
                envelope->lower_rhs = row.rhs;
            }
            break;
        case LinearConstraintSense::Equal:
            if (!envelope->lower_rhs.has_value() || row.rhs > *envelope->lower_rhs + tol) {
                envelope->lower_rhs = row.rhs;
            }
            if (!envelope->upper_rhs.has_value() || row.rhs < *envelope->upper_rhs - tol) {
                envelope->upper_rhs = row.rhs;
            }
            break;
    }

    return !(envelope->lower_rhs.has_value() && envelope->upper_rhs.has_value() &&
             *envelope->lower_rhs > *envelope->upper_rhs + tol);
}

inline MergeRowsResult merge_parallel_rows(const std::vector<std::string>& order,
                                           const std::unordered_map<std::string, RowEnvelope>& rows,
                                           double tol) {
    MergeRowsResult result;
    result.rows.reserve(order.size() * 2);

    for (const auto& signature : order) {
        const auto it = rows.find(signature);
        if (it == rows.end())
            continue;

        const RowEnvelope& envelope = it->second;
        int kept_rows = 0;
        if (envelope.lower_rhs.has_value() && envelope.upper_rhs.has_value() &&
            *envelope.lower_rhs > *envelope.upper_rhs + tol) {
            result.infeasible = true;
            result.rows.clear();
            return result;
        }

        if (envelope.lower_rhs.has_value() && envelope.upper_rhs.has_value() &&
            std::abs(*envelope.lower_rhs - *envelope.upper_rhs) <= tol) {
            SparseLinearConstraint equality = envelope.prototype;
            equality.sense = LinearConstraintSense::Equal;
            equality.rhs = 0.5 * (*envelope.lower_rhs + *envelope.upper_rhs);
            result.rows.push_back(std::move(equality));
            kept_rows = 1;
        } else {
            if (envelope.lower_rhs.has_value()) {
                SparseLinearConstraint lower = envelope.prototype;
                lower.sense = LinearConstraintSense::GreaterEqual;
                lower.rhs = *envelope.lower_rhs;
                result.rows.push_back(std::move(lower));
                ++kept_rows;
            }
            if (envelope.upper_rhs.has_value()) {
                SparseLinearConstraint upper = envelope.prototype;
                upper.sense = LinearConstraintSense::LessEqual;
                upper.rhs = *envelope.upper_rhs;
                result.rows.push_back(std::move(upper));
                ++kept_rows;
            }
        }

        result.removed_rows += std::max(0, envelope.source_rows - kept_rows);
    }

    return result;
}

inline void canonicalize_sparse_row(SparseLinearConstraint* row, const Eigen::VectorXd& lower,
                                    const Eigen::VectorXd& upper, int* removed_coeffs, double tol) {
    if (row == nullptr)
        return;

    std::vector<std::pair<int, double>> terms;
    terms.reserve(std::min(row->indices.size(), row->values.size()));
    double rhs = row->rhs;

    for (int k = 0;
         k < static_cast<int>(row->indices.size()) && k < static_cast<int>(row->values.size());
         ++k) {
        const int index = row->indices[k];
        const double coeff = row->values[k];
        if (index < 0 || index >= lower.size() || index >= upper.size() ||
            std::abs(coeff) <= kCoeffTol) {
            if (removed_coeffs)
                ++(*removed_coeffs);
            continue;
        }

        if (std::isfinite(lower(index)) && std::isfinite(upper(index)) &&
            std::abs(lower(index) - upper(index)) <= tol) {
            rhs -= coeff * lower(index);
            if (removed_coeffs)
                ++(*removed_coeffs);
            continue;
        }

        terms.emplace_back(index, coeff);
    }

    std::sort(terms.begin(), terms.end(),
              [](const auto& lhs, const auto& rhs_pair) { return lhs.first < rhs_pair.first; });

    row->indices.clear();
    row->values.clear();
    row->rhs = rhs;
    for (const auto& [index, coeff] : terms) {
        if (!row->indices.empty() && row->indices.back() == index) {
            const double merged = row->values.back() + coeff;
            if (std::abs(merged) <= kCoeffTol) {
                row->indices.pop_back();
                row->values.pop_back();
                if (removed_coeffs)
                    ++(*removed_coeffs);
            } else {
                row->values.back() = merged;
            }
            continue;
        }
        row->indices.push_back(index);
        row->values.push_back(coeff);
    }
}

inline std::optional<int> find_row_coefficient_position(const SparseLinearConstraint& row,
                                                        int index) {
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        if (row.indices[k] == index)
            return k;
    }
    return std::nullopt;
}

inline std::optional<std::pair<double, double>>
implied_interval_for_equality_pivot(const SparseLinearConstraint& row, int pivot,
                                    const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                                    double tol) {
    const auto pivot_pos = find_row_coefficient_position(row, pivot);
    if (!pivot_pos.has_value())
        return std::nullopt;

    const SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
    const SparseRowActivitySummary summary = sparse_row_activity_summary(view, lower, upper);
    const double aij = row.values[*pivot_pos];
    const SparseVariableContribution pivot_contribution =
        sparse_variable_contribution(aij, pivot, lower, upper);
    const bool other_min_finite =
        summary.min_infinite_terms - (pivot_contribution.min_finite ? 0 : 1) == 0;
    const bool other_max_finite =
        summary.max_infinite_terms - (pivot_contribution.max_finite ? 0 : 1) == 0;
    if (!other_min_finite || !other_max_finite || std::abs(aij) <= tol) {
        return std::nullopt;
    }

    const double other_min =
        summary.min_activity - (pivot_contribution.min_finite ? pivot_contribution.min_value : 0.0);
    const double other_max =
        summary.max_activity - (pivot_contribution.max_finite ? pivot_contribution.max_value : 0.0);

    double implied_lower = 0.0;
    double implied_upper = 0.0;
    if (aij > 0.0) {
        implied_lower = (row.rhs - other_max) / aij;
        implied_upper = (row.rhs - other_min) / aij;
    } else {
        implied_lower = (row.rhs - other_min) / aij;
        implied_upper = (row.rhs - other_max) / aij;
    }
    if (implied_lower > implied_upper)
        std::swap(implied_lower, implied_upper);
    return std::make_pair(implied_lower, implied_upper);
}

inline void update_row_summary_for_bound_change(const SparseVariableContribution& old_contribution,
                                                const SparseVariableContribution& new_contribution,
                                                SparseRowActivitySummary* summary) {
    if (old_contribution.min_finite) {
        summary->min_activity -= old_contribution.min_value;
    } else {
        --summary->min_infinite_terms;
    }
    if (old_contribution.max_finite) {
        summary->max_activity -= old_contribution.max_value;
    } else {
        --summary->max_infinite_terms;
    }

    if (new_contribution.min_finite) {
        summary->min_activity += new_contribution.min_value;
    } else {
        ++summary->min_infinite_terms;
    }
    if (new_contribution.max_finite) {
        summary->max_activity += new_contribution.max_value;
    } else {
        ++summary->max_infinite_terms;
    }
}

inline void tighten_discrete_bounds(VariableType type, double* lower, double* upper, double tol) {
    if (type == VariableType::Continuous)
        return;

    if (type == VariableType::Binary) {
        *lower = std::max(*lower, 0.0);
        *upper = std::min(*upper, 1.0);
    }

    if (std::isfinite(*lower))
        *lower = std::ceil(*lower - tol);
    if (std::isfinite(*upper))
        *upper = std::floor(*upper + tol);
}

inline bool apply_tightened_bounds(Problem* problem, int index, double tightened_lower,
                                   double tightened_upper, double tol, int* tightened_bounds,
                                   int* fixed_variables) {
    if (problem == nullptr || index < 0 || index >= problem->lower_bounds.size() ||
        index >= problem->upper_bounds.size() ||
        index >= static_cast<int>(problem->variable_types.size())) {
        return false;
    }

    const bool was_fixed =
        std::isfinite(problem->lower_bounds(index)) &&
        std::isfinite(problem->upper_bounds(index)) &&
        std::abs(problem->lower_bounds(index) - problem->upper_bounds(index)) <= tol;

    tighten_discrete_bounds(problem->variable_types[index], &tightened_lower, &tightened_upper,
                            tol);
    if (tightened_upper + tol < tightened_lower)
        return false;

    const bool lower_changed = tightened_lower > problem->lower_bounds(index) + tol;
    const bool upper_changed = tightened_upper < problem->upper_bounds(index) - tol;
    if (!lower_changed && !upper_changed)
        return true;

    if (lower_changed) {
        problem->lower_bounds(index) = tightened_lower;
        if (tightened_bounds)
            ++(*tightened_bounds);
    }
    if (upper_changed) {
        problem->upper_bounds(index) = tightened_upper;
        if (tightened_bounds)
            ++(*tightened_bounds);
    }

    const bool is_fixed =
        std::isfinite(problem->lower_bounds(index)) &&
        std::isfinite(problem->upper_bounds(index)) &&
        std::abs(problem->lower_bounds(index) - problem->upper_bounds(index)) <= tol;
    if (!was_fixed && is_fixed && fixed_variables)
        ++(*fixed_variables);
    return true;
}

inline bool tighten_bounds_from_sparse_row(const SparseRowView& row, const Problem& problem,
                                           Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                                           int* tightened_bounds,
                                           const std::vector<std::vector<int>>& col_to_rows,
                                           std::vector<char>* next_dirty_rows, double tol) {
    SparseRowActivitySummary summary = sparse_row_activity_summary(row, *lower, *upper);
    if (!sparse_row_is_feasible(summary, row.sense, row.rhs, tol))
        return false;
    if (!row.indices || !row.values)
        return true;

    for (int k = 0;
         k < static_cast<int>(row.indices->size()) && k < static_cast<int>(row.values->size());
         ++k) {
        const int index = (*row.indices)[k];
        if (index < 0 || index >= lower->size() || index >= upper->size())
            continue;

        const double coeff = (*row.values)[k];
        if (std::abs(coeff) <= kCoeffTol)
            continue;

        const SparseVariableContribution old_contribution =
            sparse_variable_contribution(coeff, index, *lower, *upper);
        const bool other_min_finite =
            summary.min_infinite_terms - (old_contribution.min_finite ? 0 : 1) == 0;
        const bool other_max_finite =
            summary.max_infinite_terms - (old_contribution.max_finite ? 0 : 1) == 0;
        const double other_min_activity =
            summary.min_activity - (old_contribution.min_finite ? old_contribution.min_value : 0.0);
        const double other_max_activity =
            summary.max_activity - (old_contribution.max_finite ? old_contribution.max_value : 0.0);

        double tightened_lower = (*lower)(index);
        double tightened_upper = (*upper)(index);

        const auto apply_upper = [&](double candidate) {
            if (std::isfinite(candidate) && candidate < tightened_upper - tol)
                tightened_upper = candidate;
        };
        const auto apply_lower = [&](double candidate) {
            if (std::isfinite(candidate) && candidate > tightened_lower + tol)
                tightened_lower = candidate;
        };

        switch (row.sense) {
            case LinearConstraintSense::LessEqual:
                if (coeff > 0.0 && other_min_finite) {
                    apply_upper((row.rhs - other_min_activity) / coeff);
                } else if (coeff < 0.0 && other_min_finite) {
                    apply_lower((row.rhs - other_min_activity) / coeff);
                }
                break;
            case LinearConstraintSense::GreaterEqual:
                if (coeff > 0.0 && other_max_finite) {
                    apply_lower((row.rhs - other_max_activity) / coeff);
                } else if (coeff < 0.0 && other_max_finite) {
                    apply_upper((row.rhs - other_max_activity) / coeff);
                }
                break;
            case LinearConstraintSense::Equal:
                if (coeff > 0.0) {
                    if (other_min_finite)
                        apply_upper((row.rhs - other_min_activity) / coeff);
                    if (other_max_finite)
                        apply_lower((row.rhs - other_max_activity) / coeff);
                } else {
                    if (other_min_finite)
                        apply_lower((row.rhs - other_min_activity) / coeff);
                    if (other_max_finite)
                        apply_upper((row.rhs - other_max_activity) / coeff);
                }
                break;
        }

        tighten_discrete_bounds(problem.variable_types[index], &tightened_lower, &tightened_upper,
                                tol);
        if (tightened_upper + tol < tightened_lower)
            return false;

        const bool lower_changed = tightened_lower > (*lower)(index) + tol;
        const bool upper_changed = tightened_upper < (*upper)(index)-tol;
        if (!lower_changed && !upper_changed)
            continue;

        (*lower)(index) = tightened_lower;
        (*upper)(index) = tightened_upper;
        if (lower_changed)
            ++(*tightened_bounds);
        if (upper_changed)
            ++(*tightened_bounds);

        const SparseVariableContribution new_contribution =
            sparse_variable_contribution(coeff, index, *lower, *upper);
        update_row_summary_for_bound_change(old_contribution, new_contribution, &summary);

        if (next_dirty_rows && index >= 0 && index < static_cast<int>(col_to_rows.size())) {
            for (const int affected_row : col_to_rows[index]) {
                if (affected_row >= 0 && affected_row < static_cast<int>(next_dirty_rows->size())) {
                    (*next_dirty_rows)[affected_row] = 1;
                }
            }
        }
    }

    return true;
}

inline bool tighten_singleton_row_bounds(Problem* problem, const SparseLinearConstraint& row,
                                         double tol, int* tightened_bounds, int* fixed_variables) {
    if (problem == nullptr || row.indices.empty() || row.values.empty())
        return true;
    const int index = row.indices.front();
    if (index < 0 || index >= problem->lower_bounds.size() ||
        index >= problem->upper_bounds.size() ||
        index >= static_cast<int>(problem->variable_types.size())) {
        return true;
    }

    const double coeff = row.values.front();
    if (std::abs(coeff) <= tol)
        return std::abs(row.rhs) <= tol;

    double tightened_lower = problem->lower_bounds(index);
    double tightened_upper = problem->upper_bounds(index);
    const double implied = row.rhs / coeff;
    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            if (coeff > 0.0) {
                tightened_upper = std::min(tightened_upper, implied);
            } else {
                tightened_lower = std::max(tightened_lower, implied);
            }
            break;
        case LinearConstraintSense::GreaterEqual:
            if (coeff > 0.0) {
                tightened_lower = std::max(tightened_lower, implied);
            } else {
                tightened_upper = std::min(tightened_upper, implied);
            }
            break;
        case LinearConstraintSense::Equal:
            tightened_lower = implied;
            tightened_upper = implied;
            break;
    }

    return apply_tightened_bounds(problem, index, tightened_lower, tightened_upper, tol,
                                  tightened_bounds, fixed_variables);
}

inline std::pair<int, int> relax_huge_bounds(Problem* problem, double tol,
                                             double huge_bound_factor = 1e6,
                                             double relax_gap_factor = 1e6) {
    if (problem == nullptr || problem->base_constraints.empty())
        return {0, 0};

    double data_scale = 1.0;
    if (problem->objective_coefficients.size() > 0) {
        data_scale = std::max(data_scale, problem->objective_coefficients.cwiseAbs().maxCoeff());
    }
    for (const auto& row : problem->base_constraints) {
        data_scale = std::max(data_scale, std::abs(row.rhs));
        for (double value : row.values)
            data_scale = std::max(data_scale, std::abs(value));
    }
    const double huge_bound = huge_bound_factor * data_scale;
    if (!std::isfinite(huge_bound) || huge_bound <= 0.0)
        return {0, 0};

    std::vector<std::vector<int>> col_to_rows(problem->lower_bounds.size());
    for (int row_index = 0; row_index < static_cast<int>(problem->base_constraints.size());
         ++row_index) {
        const auto& row = problem->base_constraints[row_index];
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= static_cast<int>(col_to_rows.size()) ||
                std::abs(row.values[k]) <= kCoeffTol) {
                continue;
            }
            col_to_rows[index].push_back(row_index);
        }
    }

    int relaxed_lower = 0;
    int relaxed_upper = 0;
    for (int index = 0;
         index < problem->lower_bounds.size() && index < problem->upper_bounds.size(); ++index) {
        const bool check_upper = std::isfinite(problem->upper_bounds(index)) &&
                                 problem->upper_bounds(index) > huge_bound;
        const bool check_lower = std::isfinite(problem->lower_bounds(index)) &&
                                 problem->lower_bounds(index) < -huge_bound;
        if (!check_upper && !check_lower)
            continue;

        double implied_lower = -std::numeric_limits<double>::infinity();
        double implied_upper = std::numeric_limits<double>::infinity();
        bool has_implied_lower = false;
        bool has_implied_upper = false;

        for (const int row_index : col_to_rows[index]) {
            const auto& row = problem->base_constraints[row_index];
            const auto coeff_pos = find_row_coefficient_position(row, index);
            if (!coeff_pos.has_value())
                continue;
            const double coeff = row.values[*coeff_pos];
            const SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
            const SparseRowActivitySummary summary =
                sparse_row_activity_summary(view, problem->lower_bounds, problem->upper_bounds);
            const SparseVariableContribution contribution = sparse_variable_contribution(
                coeff, index, problem->lower_bounds, problem->upper_bounds);
            const bool other_min_finite =
                summary.min_infinite_terms - (contribution.min_finite ? 0 : 1) == 0;
            const bool other_max_finite =
                summary.max_infinite_terms - (contribution.max_finite ? 0 : 1) == 0;
            const double other_min =
                summary.min_activity - (contribution.min_finite ? contribution.min_value : 0.0);
            const double other_max =
                summary.max_activity - (contribution.max_finite ? contribution.max_value : 0.0);

            const auto apply_upper = [&](double candidate) {
                if (std::isfinite(candidate)) {
                    implied_upper = std::min(implied_upper, candidate);
                    has_implied_upper = true;
                }
            };
            const auto apply_lower = [&](double candidate) {
                if (std::isfinite(candidate)) {
                    implied_lower = std::max(implied_lower, candidate);
                    has_implied_lower = true;
                }
            };

            switch (row.sense) {
                case LinearConstraintSense::LessEqual:
                    if (coeff > 0.0 && other_min_finite) {
                        apply_upper((row.rhs - other_min) / coeff);
                    } else if (coeff < 0.0 && other_min_finite) {
                        apply_lower((row.rhs - other_min) / coeff);
                    }
                    break;
                case LinearConstraintSense::GreaterEqual:
                    if (coeff > 0.0 && other_max_finite) {
                        apply_lower((row.rhs - other_max) / coeff);
                    } else if (coeff < 0.0 && other_max_finite) {
                        apply_upper((row.rhs - other_max) / coeff);
                    }
                    break;
                case LinearConstraintSense::Equal:
                    if (coeff > 0.0) {
                        if (other_min_finite)
                            apply_upper((row.rhs - other_min) / coeff);
                        if (other_max_finite)
                            apply_lower((row.rhs - other_max) / coeff);
                    } else {
                        if (other_min_finite)
                            apply_lower((row.rhs - other_min) / coeff);
                        if (other_max_finite)
                            apply_upper((row.rhs - other_max) / coeff);
                    }
                    break;
            }
        }

        if (check_upper && has_implied_upper && std::isfinite(implied_upper)) {
            const double ref = std::max({1.0, std::abs(implied_upper), data_scale});
            if (problem->upper_bounds(index) > implied_upper + relax_gap_factor * ref) {
                problem->upper_bounds(index) = std::numeric_limits<double>::infinity();
                ++relaxed_upper;
            }
        }
        if (check_lower && has_implied_lower && std::isfinite(implied_lower)) {
            const double ref = std::max({1.0, std::abs(implied_lower), data_scale});
            if (problem->lower_bounds(index) < implied_lower - relax_gap_factor * ref) {
                problem->lower_bounds(index) = -std::numeric_limits<double>::infinity();
                ++relaxed_lower;
            }
        }
    }

    return {relaxed_lower, relaxed_upper};
}

/// Coefficient strengthening: DISABLED.
///
/// Two independent derivations (single-variable saturation, then a
/// minimal-cover reduction restricted to pure 0/1 rows) were tried and both
/// produced verified wrong answers on random knapsack instances -- e.g. row
/// `3x0+10x1+8x2+13x3<=23` (all binary) strengthened to `3x0+8x1+8x2+8x3<=16`
/// wrongly excludes the feasible/optimal point x0=x1=x2=1,x3=0 (original
/// activity 21<=23, strengthened activity 19>16). The minimal-cover
/// reduction must be applied as an additional cut derived from the cover,
/// not as a wholesale replacement of the row's coefficients/rhs; getting
/// that right needs a careful port of a reference implementation (e.g.
/// HiGHS's HPresolve::strengthenInequalities) with row-by-row verification
/// against a working solver, not a from-memory re-derivation. Left as a
/// documented no-op rather than ship a silent wrong-answer generator in
/// root presolve. See git history on this function for the two failed
/// attempts if revisiting.
inline std::pair<int, int> coefficient_strengthening(Problem* problem, const std::vector<Cut>& cuts,
                                                     double tol, int* tightened_bounds,
                                                     int* strengthened_coeffs) {
    (void)cuts;
    (void)tol;
    (void)tightened_bounds;
    (void)strengthened_coeffs;
    if (problem == nullptr)
        return {0, 0};
    return {0, 0};
}

/// Probing with badge-based selection (PaPILO style)
/// Probes binary variables to infer bounds and detect substitutions
struct ProbingResult {
    bool infeasible = false;
    int tightened_bounds = 0;
    int fixed_variables = 0;
};

inline ProbingResult probe_row(const Problem* problem, const SparseRowView& row, int probe_idx,
                               const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                               double tol, double* implied_lb, double* implied_ub) {
    ProbingResult result;

    if (!row.indices || !row.values || implied_lb == nullptr || implied_ub == nullptr ||
        problem == nullptr || probe_idx < 0 || probe_idx >= lower.size() ||
        probe_idx >= upper.size() ||
        probe_idx >= static_cast<int>(problem->variable_types.size())) {
        return result;
    }

    double probe_coeff = 0.0;
    bool found_probe = false;
    for (int k = 0;
         k < static_cast<int>(row.indices->size()) && k < static_cast<int>(row.values->size());
         ++k) {
        const int index = (*row.indices)[k];
        const double coeff = (*row.values)[k];
        if (index == probe_idx && std::abs(coeff) > kCoeffTol) {
            probe_coeff = coeff;
            found_probe = true;
            break;
        }
    }

    if (!found_probe || problem->variable_types[probe_idx] != VariableType::Binary)
        return result;

    const SparseRowActivitySummary full_summary = sparse_row_activity_summary(row, lower, upper);
    if (!sparse_row_is_feasible(full_summary, row.sense, row.rhs, tol)) {
        result.infeasible = true;
        return result;
    }

    SparseRowActivitySummary summary = full_summary;

    const SparseVariableContribution probe_contribution =
        sparse_variable_contribution(probe_coeff, probe_idx, lower, upper);
    update_row_summary_for_bound_change(probe_contribution,
                                        SparseVariableContribution{0.0, 0.0, true, true}, &summary);

    const bool other_min_finite = summary.min_infinite_terms == 0;
    const bool other_max_finite = summary.max_infinite_terms == 0;

    *implied_lb = lower(probe_idx);
    *implied_ub = upper(probe_idx);

    if (other_min_finite && other_max_finite && std::abs(probe_coeff) > tol) {
        const double other_min = summary.min_activity;
        const double other_max = summary.max_activity;

        double candidate_lb = -std::numeric_limits<double>::infinity();
        double candidate_ub = std::numeric_limits<double>::infinity();
        if (probe_coeff > 0.0) {
            switch (row.sense) {
                case LinearConstraintSense::LessEqual:
                    candidate_ub = (row.rhs - other_min) / probe_coeff;
                    break;
                case LinearConstraintSense::GreaterEqual:
                    candidate_lb = (row.rhs - other_max) / probe_coeff;
                    break;
                case LinearConstraintSense::Equal:
                    candidate_lb = (row.rhs - other_max) / probe_coeff;
                    candidate_ub = (row.rhs - other_min) / probe_coeff;
                    break;
            }
        } else {
            switch (row.sense) {
                case LinearConstraintSense::LessEqual:
                    candidate_lb = (row.rhs - other_min) / probe_coeff;
                    break;
                case LinearConstraintSense::GreaterEqual:
                    candidate_ub = (row.rhs - other_max) / probe_coeff;
                    break;
                case LinearConstraintSense::Equal:
                    candidate_lb = (row.rhs - other_min) / probe_coeff;
                    candidate_ub = (row.rhs - other_max) / probe_coeff;
                    break;
            }
        }

        if (candidate_lb > candidate_ub)
            std::swap(candidate_lb, candidate_ub);

        candidate_lb = std::max(candidate_lb, 0.0);
        candidate_ub = std::min(candidate_ub, 1.0);

        *implied_lb = candidate_lb;
        *implied_ub = candidate_ub;

        if (*implied_ub + tol < *implied_lb) {
            result.infeasible = true;
        }
    }

    return result;
}

inline double probing_score_for_variable(const Problem& problem,
                                         const std::vector<SparseRowView>& row_views,
                                         const std::vector<int>& incident_rows, int variable) {
    double score = 0.0;
    for (const int row_idx : incident_rows) {
        if (row_idx < 0 || row_idx >= static_cast<int>(row_views.size()))
            continue;
        const SparseRowView& row = row_views[row_idx];
        if (!row.indices || !row.values)
            continue;

        int binary_neighbors = 0;
        double abs_coeff_sum = 0.0;
        double own_coeff = 0.0;
        for (int k = 0;
             k < static_cast<int>(row.indices->size()) && k < static_cast<int>(row.values->size());
             ++k) {
            const int index = (*row.indices)[k];
            if (index < 0 || index >= static_cast<int>(problem.variable_types.size()))
                continue;
            if (problem.variable_types[index] == VariableType::Binary)
                ++binary_neighbors;
            abs_coeff_sum += std::abs((*row.values)[k]);
            if (index == variable)
                own_coeff = std::abs((*row.values)[k]);
        }

        score += static_cast<double>(std::max(0, binary_neighbors - 1));
        if (abs_coeff_sum > kCoeffTol)
            score += own_coeff / abs_coeff_sum;
    }
    return score;
}

inline ProbingResult probe_all_binary_variables(Problem* problem, const std::vector<Cut>& cuts,
                                                Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                                                double tol, int* tightened_bounds,
                                                int* fixed_vars) {
    ProbingResult result;

    if (problem == nullptr || lower == nullptr || upper == nullptr)
        return result;

    std::vector<SparseRowView> row_views;
    row_views.reserve(problem->base_constraints.size() + cuts.size());

    std::vector<std::vector<int>> col_to_rows(problem->lower_bounds.size());

    const auto add_row = [&](const auto& source_row) {
        const int row_index = static_cast<int>(row_views.size());
        row_views.push_back(detail::SparseRowView{&source_row.indices, &source_row.values,
                                                  source_row.sense, source_row.rhs});

        for (int k = 0; k < static_cast<int>(source_row.indices.size()) &&
                        k < static_cast<int>(source_row.values.size());
             ++k) {
            const int index = source_row.indices[k];
            if (index >= 0 && index < static_cast<int>(problem->lower_bounds.size())) {
                if (problem->variable_types[index] == VariableType::Binary) {
                    col_to_rows[index].push_back(row_index);
                }
            }
        }
    };

    for (const auto& base_row : problem->base_constraints)
        add_row(base_row);

    for (const auto& cut : cuts)
        add_row(cut);

    std::vector<int> binary_vars;
    for (int j = 0; j < static_cast<int>(problem->lower_bounds.size()); ++j) {
        if (problem->variable_types[j] == VariableType::Binary) {
            binary_vars.push_back(j);
        }
    }

    if (binary_vars.empty())
        return result;

    struct ProbingScore {
        int index;
        double score = 0.0;
        int row_count = 0;
    };
    std::vector<ProbingScore> scores;
    scores.reserve(binary_vars.size());

    for (int j : binary_vars) {
        scores.push_back(
            ProbingScore{j, probing_score_for_variable(*problem, row_views, col_to_rows[j], j),
                         static_cast<int>(col_to_rows[j].size())});
    }

    std::sort(scores.begin(), scores.end(), [](const ProbingScore& a, const ProbingScore& b) {
        if (a.score != b.score)
            return a.score > b.score;
        if (a.row_count != b.row_count)
            return a.row_count > b.row_count;
        return a.index < b.index;
    });

    const int max_initial_badge_size = 1000;
    const int min_badge_size = 10;
    int initial_badge_limit = std::clamp(static_cast<int>(problem->base_constraints.size()) * 4,
                                         min_badge_size, max_initial_badge_size);

    for (const auto& score : scores) {
        if (initial_badge_limit <= 0)
            break;
        if (col_to_rows[score.index].empty())
            continue;

        const double start_lb = (*lower)(score.index);
        const double start_ub = (*upper)(score.index);
        double candidate_lb = start_lb;
        double candidate_ub = start_ub;

        for (const int row_idx : col_to_rows[score.index]) {
            if (initial_badge_limit <= 0)
                break;

            const SparseRowView& row = row_views[row_idx];
            double row_lb = candidate_lb;
            double row_ub = candidate_ub;
            ProbingResult row_result =
                probe_row(problem, row, score.index, *lower, *upper, tol, &row_lb, &row_ub);

            if (row_result.infeasible) {
                result.infeasible = true;
                return result;
            }

            candidate_lb = std::max(candidate_lb, row_lb);
            candidate_ub = std::min(candidate_ub, row_ub);

            --initial_badge_limit;
        }

        if (!detail::apply_tightened_bounds(problem, score.index, candidate_lb, candidate_ub, tol,
                                            tightened_bounds, fixed_vars)) {
            result.infeasible = true;
            return result;
        }

        if (tightened_bounds)
            result.tightened_bounds = *tightened_bounds;
        if (fixed_vars)
            result.fixed_variables = *fixed_vars;

        if (candidate_lb <= start_lb + tol && candidate_ub >= start_ub - tol) {
            initial_badge_limit = std::max(min_badge_size, initial_badge_limit - 1);
        }
    }

    return result;
}

inline ProbingResult probe_mip_root(Problem* problem, const std::vector<Cut>& cuts, double tol,
                                    int* tightened_bounds, int* fixed_vars) {
    ProbingResult result;

    if (problem == nullptr)
        return result;

    ProbingResult probe_result =
        probe_all_binary_variables(problem, cuts, &problem->lower_bounds, &problem->upper_bounds,
                                   tol, tightened_bounds, fixed_vars);

    if (probe_result.infeasible) {
        result.infeasible = true;
        return result;
    }

    const NodeBoundPresolveResult tightened = presolve_mip_node_bounds(
        *problem, problem->lower_bounds, problem->upper_bounds, cuts, tol, 2);
    if (tightened.infeasible) {
        result.infeasible = true;
        return result;
    }
    if (tightened.tightened_bounds > 0) {
        problem->lower_bounds = tightened.lower;
        problem->upper_bounds = tightened.upper;
        if (tightened_bounds)
            *tightened_bounds += tightened.tightened_bounds;
        result.tightened_bounds += tightened.tightened_bounds;
    }
    result.tightened_bounds += probe_result.tightened_bounds;
    result.fixed_variables += probe_result.fixed_variables;

    return result;
}

/// Component detection for MIP presolve
/// Identifies disconnected components in the constraint graph
struct ComponentInfo {
    int component_id = 0;
    int nintegral = 0;
    int ncontinuous = 0;
    int nnonz = 0;
};

inline std::vector<std::vector<int>> build_column_to_rows(const Problem& problem);
inline double effective_objective_coefficient(const Problem& problem, int index);
inline int singleton_column_substitution(Problem* problem, double tol, int* substitution_count);
inline int simple_substitution(Problem* problem, double tol, int* removed_coeffs,
                               int* substitution_count);
inline int dual_fix_variables(Problem* problem, double tol, int* fixed_variables);
inline int dual_inference_bound_tightening(Problem* problem, double tol, int max_iterations);
inline int free_variable_substitution(Problem* problem, double tol, int* removed_coeffs,
                                      int* substitution_count);
inline int sparsify_with_equalities(Problem* problem, double tol, double max_scale,
                                    int* removed_coeffs);
inline bool try_aggregate_implied_free_continuous_variable(Problem* problem, int row_index,
                                                           double tol, int* removed_coeffs,
                                                           int* aggregation_count);

struct SparseColumn {
    std::vector<int> rows;
    std::vector<double> values;
};

struct BinaryClique {
    std::vector<int> variables;
};

struct RootPresolveContext {
    Problem* problem = nullptr;
    RootProblemPresolveResult* result = nullptr;
    double tol = 1e-9;
    int max_passes = 1;
    bool structure_dirty = true;
    std::vector<std::vector<int>> col_to_rows;
    std::vector<SparseColumn> columns;
    std::vector<BinaryClique> cliques;
};

inline void mark_structure_dirty(RootPresolveContext* context) {
    if (context)
        context->structure_dirty = true;
}

inline std::vector<SparseColumn> build_sparse_columns(const Problem& problem) {
    std::vector<SparseColumn> columns(problem.lower_bounds.size());
    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const auto& row = problem.base_constraints[row_index];
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            const double value = row.values[k];
            if (index < 0 || index >= static_cast<int>(columns.size()) ||
                std::abs(value) <= kCoeffTol) {
                continue;
            }
            columns[index].rows.push_back(row_index);
            columns[index].values.push_back(value);
        }
    }
    return columns;
}

inline std::string exact_column_signature(const SparseColumn& column, int precision = 12) {
    const double scale = std::pow(10.0, precision);
    std::ostringstream oss;
    for (int k = 0;
         k < static_cast<int>(column.rows.size()) && k < static_cast<int>(column.values.size());
         ++k) {
        const double rounded = std::round(column.values[k] * scale) / scale;
        oss << column.rows[k] << ':' << rounded << ';';
    }
    return oss.str();
}

inline void extract_binary_knapsack_cliques_from_row(const Problem& problem,
                                                     const SparseLinearConstraint& row, double tol,
                                                     std::vector<std::vector<int>>* cliques) {
    if (cliques == nullptr || row.sense != LinearConstraintSense::LessEqual ||
        row.indices.size() < 2 || row.values.size() < 2 || !std::isfinite(row.rhs)) {
        return;
    }

    double effective_tol = std::max(tol, 1e-12);
    std::vector<std::pair<double, int>> coeffs;
    coeffs.reserve(row.indices.size());
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        const int index = row.indices[k];
        const double coeff = row.values[k];
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[index] != VariableType::Binary || coeff <= effective_tol) {
            return;
        }
        coeffs.emplace_back(coeff, index);
    }

    if (static_cast<int>(coeffs.size()) < 2)
        return;

    std::sort(coeffs.begin(), coeffs.end(), [](const auto& lhs, const auto& rhs) {
        if (std::abs(lhs.first - rhs.first) > 1e-12)
            return lhs.first < rhs.first;
        return lhs.second < rhs.second;
    });

    const int n = static_cast<int>(coeffs.size());
    if (coeffs[n - 2].first + coeffs[n - 1].first <= row.rhs + effective_tol)
        return;

    auto push_clique = [&](int suffix_start, int outside_index) {
        std::vector<int> clique;
        if (outside_index >= 0)
            clique.push_back(coeffs[outside_index].second);
        for (int pos = suffix_start; pos < n; ++pos)
            clique.push_back(coeffs[pos].second);
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
        if (coeffs[mid].first + coeffs[mid + 1].first > row.rhs + effective_tol) {
            first = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    push_clique(first, -1);

    for (int outside = first - 1; outside >= 0; --outside) {
        int lo = first;
        int hi = n - 1;
        int suffix_start = -1;
        while (lo <= hi) {
            const int mid = lo + (hi - lo) / 2;
            if (coeffs[outside].first + coeffs[mid].first > row.rhs + effective_tol) {
                suffix_start = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        if (suffix_start < 0)
            break;
        push_clique(suffix_start, outside);
    }
}

inline void ensure_structure_cache(RootPresolveContext* context) {
    if (context == nullptr || context->problem == nullptr || !context->structure_dirty)
        return;
    context->col_to_rows = build_column_to_rows(*context->problem);
    context->columns = build_sparse_columns(*context->problem);
    context->cliques.clear();
    std::unordered_set<std::string> seen_cliques;
    for (const auto& row : context->problem->base_constraints) {
        std::vector<std::vector<int>> cliques;
        extract_binary_knapsack_cliques_from_row(*context->problem, row, context->tol, &cliques);
        for (auto& clique_vars : cliques) {
            std::ostringstream key;
            for (int var : clique_vars)
                key << var << ';';
            if (seen_cliques.insert(key.str()).second) {
                context->cliques.push_back(BinaryClique{std::move(clique_vars)});
            }
        }
    }
    context->structure_dirty = false;
}

inline std::int64_t gcd64(std::int64_t lhs, std::int64_t rhs) {
    lhs = std::llabs(lhs);
    rhs = std::llabs(rhs);
    while (rhs != 0) {
        const std::int64_t tmp = lhs % rhs;
        lhs = rhs;
        rhs = tmp;
    }
    return lhs;
}

inline SparseLinearConstraint implication_to_constraint(int src, bool src_is_one, int dst,
                                                        bool dst_is_one) {
    SparseLinearConstraint row;
    row.indices = {src, dst};
    if (src_is_one && dst_is_one) {
        row.values = {-1.0, 1.0};
        row.sense = LinearConstraintSense::GreaterEqual;
        row.rhs = 0.0;
    } else if (src_is_one && !dst_is_one) {
        row.values = {1.0, 1.0};
        row.sense = LinearConstraintSense::LessEqual;
        row.rhs = 1.0;
    } else if (!src_is_one && dst_is_one) {
        row.values = {1.0, 1.0};
        row.sense = LinearConstraintSense::GreaterEqual;
        row.rhs = 1.0;
    } else {
        row.values = {1.0, -1.0};
        row.sense = LinearConstraintSense::GreaterEqual;
        row.rhs = 0.0;
    }
    return row;
}

inline std::vector<ComponentInfo> detect_components(const Problem& problem) {
    std::vector<ComponentInfo> components;
    if (problem.base_constraints.empty() || problem.lower_bounds.size() == 0)
        return components;

    const int ncols = static_cast<int>(problem.lower_bounds.size());

    std::vector<int> parent(ncols);
    std::iota(parent.begin(), parent.end(), 0);

    const auto find_root = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };

    const auto union_sets = [&](int a, int b) {
        const int root_a = find_root(a);
        const int root_b = find_root(b);
        if (root_a != root_b) {
            parent[root_a] = root_b;
        }
    };

    for (const auto& row : problem.base_constraints) {
        int first_col = -1;
        for (int index : row.indices) {
            if (index >= 0 && index < ncols) {
                first_col = index;
                break;
            }
        }
        if (first_col < 0)
            continue;

        for (int col : row.indices) {
            if (col >= 0 && col < ncols && first_col >= 0 && first_col < ncols) {
                union_sets(first_col, col);
            }
        }
    }

    std::unordered_map<int, int> root_to_component;
    std::vector<int> component_of_col(ncols, -1);
    for (int i = 0; i < ncols; ++i) {
        const int root = find_root(i);
        const auto [it, inserted] =
            root_to_component.try_emplace(root, static_cast<int>(root_to_component.size()));
        if (inserted) {
            components.push_back(ComponentInfo{it->second, 0, 0, 0});
        }
        component_of_col[i] = it->second;
    }

    for (int i = 0; i < ncols; ++i) {
        const int comp = component_of_col[i];
        if (comp >= 0 && comp < static_cast<int>(components.size())) {
            components[comp].nintegral += (problem.variable_types[i] != VariableType::Continuous);
            components[comp].ncontinuous += (problem.variable_types[i] == VariableType::Continuous);
        }
    }

    for (const auto& row : problem.base_constraints) {
        int row_component = -1;
        int row_nnz = 0;
        for (int index : row.indices) {
            if (index < 0 || index >= ncols)
                continue;
            if (row_component < 0)
                row_component = component_of_col[index];
            ++row_nnz;
        }
        if (row_component >= 0 && row_component < static_cast<int>(components.size()))
            components[row_component].nnonz += row_nnz;
    }

    return components;
}

inline std::vector<std::vector<int>> build_column_to_rows(const Problem& problem) {
    std::vector<std::vector<int>> col_to_rows(problem.lower_bounds.size());
    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const auto& row = problem.base_constraints[row_index];
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= static_cast<int>(col_to_rows.size()) ||
                std::abs(row.values[k]) <= kCoeffTol) {
                continue;
            }
            col_to_rows[index].push_back(row_index);
        }
    }
    return col_to_rows;
}

inline int count_binary_variables(const Problem& problem) {
    return static_cast<int>(std::count(problem.variable_types.begin(), problem.variable_types.end(),
                                       VariableType::Binary));
}

inline int detect_implied_integers(Problem* problem, double tol, int max_rounds = 4) {
    if (problem == nullptr)
        return 0;

    int promoted = 0;
    for (int round = 0; round < std::max(1, max_rounds); ++round) {
        const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
        bool changed = false;

        for (int col = 0; col < static_cast<int>(problem->variable_types.size()); ++col) {
            if (problem->variable_types[col] != VariableType::Continuous)
                continue;

            bool implied_integer = false;
            for (const int row_index : col_to_rows[col]) {
                if (row_index < 0 ||
                    row_index >= static_cast<int>(problem->base_constraints.size()))
                    continue;

                const auto& row = problem->base_constraints[row_index];
                if (row.sense != LinearConstraintSense::Equal)
                    continue;

                const auto coeff_pos = find_row_coefficient_position(row, col);
                if (!coeff_pos.has_value())
                    continue;

                const double pivot_coeff = row.values[*coeff_pos];
                if (std::abs(pivot_coeff) <= tol)
                    continue;

                const double scale = 1.0 / pivot_coeff;
                if (!is_integral_value(scale * row.rhs, tol))
                    continue;

                implied_integer = true;
                for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                                k < static_cast<int>(row.values.size());
                     ++k) {
                    const int other = row.indices[k];
                    if (other == col)
                        continue;
                    if (other < 0 || other >= static_cast<int>(problem->variable_types.size()) ||
                        !is_integral_value(scale * row.values[k], tol) ||
                        problem->variable_types[other] == VariableType::Continuous) {
                        implied_integer = false;
                        break;
                    }
                }
                if (implied_integer)
                    break;
            }

            if (!implied_integer)
                continue;

            const double lb = problem->lower_bounds(col);
            const double ub = problem->upper_bounds(col);
            if (std::isfinite(lb) && std::isfinite(ub) && lb >= -tol && ub <= 1.0 + tol) {
                problem->variable_types[col] = VariableType::Binary;
            } else {
                problem->variable_types[col] = VariableType::Integer;
            }
            ++promoted;
            changed = true;
            double tightened_lower = problem->lower_bounds(col);
            double tightened_upper = problem->upper_bounds(col);
            detail::tighten_discrete_bounds(problem->variable_types[col], &tightened_lower,
                                            &tightened_upper, tol);
            problem->lower_bounds(col) = tightened_lower;
            problem->upper_bounds(col) = tightened_upper;
            if (problem->upper_bounds(col) + tol < problem->lower_bounds(col))
                return -1;
        }

        if (!changed)
            break;
    }

    return promoted;
}

inline int simplify_integral_inequalities(Problem* problem, double tol) {
    if (problem == nullptr)
        return 0;

    int changed = 0;
    for (auto& row : problem->base_constraints) {
        const int nnz =
            std::min(static_cast<int>(row.indices.size()), static_cast<int>(row.values.size()));
        if (nnz <= 0)
            continue;

        std::int64_t gcd = 0;
        bool all_discrete = true;
        bool all_integral_coeffs = true;
        for (int k = 0; k < nnz; ++k) {
            const int index = row.indices[k];
            const double coeff = row.values[k];
            if (index < 0 || index >= static_cast<int>(problem->variable_types.size()) ||
                problem->variable_types[index] == VariableType::Continuous) {
                all_discrete = false;
                break;
            }
            if (!is_integral_value(coeff, tol) ||
                std::abs(std::round(coeff)) >
                    static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
                all_integral_coeffs = false;
                break;
            }
            gcd = gcd64(gcd, static_cast<std::int64_t>(std::llround(coeff)));
        }
        if (!all_discrete || !all_integral_coeffs || gcd <= 1)
            continue;

        bool row_changed = false;
        if (row.sense == LinearConstraintSense::Equal) {
            const double scaled_rhs = row.rhs / static_cast<double>(gcd);
            if (!is_integral_value(scaled_rhs, tol))
                return -1;
            for (double& coeff : row.values)
                coeff = std::llround(coeff) / static_cast<double>(gcd);
            row.rhs = std::llround(scaled_rhs);
            row_changed = true;
        } else if (row.sense == LinearConstraintSense::LessEqual) {
            for (double& coeff : row.values)
                coeff = std::llround(coeff) / static_cast<double>(gcd);
            row.rhs = std::floor(row.rhs / static_cast<double>(gcd) + tol);
            row_changed = true;
        } else if (row.sense == LinearConstraintSense::GreaterEqual) {
            for (double& coeff : row.values)
                coeff = std::llround(coeff) / static_cast<double>(gcd);
            row.rhs = std::ceil(row.rhs / static_cast<double>(gcd) - tol);
            row_changed = true;
        }

        if (row_changed)
            ++changed;
    }
    return changed;
}

inline int canonicalize_and_merge_rows(Problem* problem, double tol, int* removed_rows,
                                       int* removed_coeffs, int* tightened_bounds,
                                       int* fixed_variables) {
    if (problem == nullptr)
        return 0;

    bool changed = false;
    std::vector<std::string> row_signature_order;
    row_signature_order.reserve(problem->base_constraints.size());
    std::unordered_map<std::string, detail::RowEnvelope> row_envelopes;
    std::unordered_map<std::string, int> signature_counts;
    std::unordered_map<std::string, SparseLinearConstraint> representative_rows;

    for (const auto& base_row : problem->base_constraints) {
        SparseLinearConstraint row = base_row;
        detail::canonicalize_sparse_row(&row, problem->lower_bounds, problem->upper_bounds,
                                        removed_coeffs, tol);

        if (row.indices.size() == 1 && row.values.size() == 1) {
            if (!detail::tighten_singleton_row_bounds(problem, row, tol, tightened_bounds,
                                                      fixed_variables)) {
                return -1;
            }
            if (removed_rows)
                ++(*removed_rows);
            changed = true;
            continue;
        }

        const detail::SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
        const detail::SparseRowActivitySummary summary =
            detail::sparse_row_activity_summary(view, problem->lower_bounds, problem->upper_bounds);
        if (!detail::sparse_row_is_feasible(summary, row.sense, row.rhs, tol)) {
            return -1;
        }
        if (detail::sparse_row_is_redundant(summary, row.sense, row.rhs, tol)) {
            if (removed_rows)
                ++(*removed_rows);
            changed = true;
            continue;
        }

        const SparseLinearConstraint normalized = normalize_parallel_row(row, tol);
        const std::string lhs_signature = detail::sparse_row_lhs_signature(normalized);
        const auto [it, inserted] = row_envelopes.try_emplace(lhs_signature);
        if (inserted) {
            row_signature_order.push_back(lhs_signature);
            representative_rows.emplace(lhs_signature, row);
            signature_counts.emplace(lhs_signature, 0);
        }
        ++signature_counts[lhs_signature];
        if (!detail::absorb_row_into_envelope(&it->second, normalized, tol)) {
            return -1;
        }
    }

    std::vector<SparseLinearConstraint> final_rows;
    final_rows.reserve(problem->base_constraints.size());
    for (const auto& signature : row_signature_order) {
        const auto env_it = row_envelopes.find(signature);
        if (env_it == row_envelopes.end())
            continue;
        const detail::RowEnvelope& envelope = env_it->second;
        const int source_rows = signature_counts[signature];

        if (envelope.lower_rhs.has_value() && envelope.upper_rhs.has_value() &&
            *envelope.lower_rhs > *envelope.upper_rhs + tol) {
            return -1;
        }

        if (source_rows <= 1) {
            final_rows.push_back(representative_rows.at(signature));
            continue;
        }

        int kept_rows = 0;
        if (envelope.lower_rhs.has_value() && envelope.upper_rhs.has_value() &&
            std::abs(*envelope.lower_rhs - *envelope.upper_rhs) <= tol) {
            SparseLinearConstraint equality = envelope.prototype;
            equality.sense = LinearConstraintSense::Equal;
            equality.rhs = 0.5 * (*envelope.lower_rhs + *envelope.upper_rhs);
            final_rows.push_back(std::move(equality));
            kept_rows = 1;
        } else {
            if (envelope.lower_rhs.has_value()) {
                SparseLinearConstraint lower = envelope.prototype;
                lower.sense = LinearConstraintSense::GreaterEqual;
                lower.rhs = *envelope.lower_rhs;
                final_rows.push_back(std::move(lower));
                ++kept_rows;
            }
            if (envelope.upper_rhs.has_value()) {
                SparseLinearConstraint upper = envelope.prototype;
                upper.sense = LinearConstraintSense::LessEqual;
                upper.rhs = *envelope.upper_rhs;
                final_rows.push_back(std::move(upper));
                ++kept_rows;
            }
        }

        if (source_rows > kept_rows) {
            if (removed_rows)
                *removed_rows += source_rows - kept_rows;
            changed = true;
        }
    }
    problem->base_constraints = std::move(final_rows);
    return changed ? 1 : 0;
}

inline int clique_merge_parallel_columns(RootPresolveContext* context, int* fixed_variables) {
    if (context == nullptr || context->problem == nullptr)
        return 0;

    ensure_structure_cache(context);
    int fixed = 0;
    for (const auto& clique : context->cliques) {
        std::unordered_map<std::string, std::vector<int>> buckets;
        for (int var : clique.variables) {
            if (var < 0 || var >= static_cast<int>(context->problem->variable_types.size()) ||
                context->problem->variable_types[var] != VariableType::Binary ||
                context->problem->lower_bounds(var) > context->tol ||
                context->problem->upper_bounds(var) < 1.0 - context->tol ||
                var >= static_cast<int>(context->columns.size())) {
                continue;
            }
            buckets[exact_column_signature(context->columns[var])].push_back(var);
        }

        for (auto& [signature, bucket] : buckets) {
            if (bucket.size() < 2)
                continue;
            std::sort(bucket.begin(), bucket.end(), [&](int lhs, int rhs) {
                const double lhs_cost = effective_objective_coefficient(*context->problem, lhs);
                const double rhs_cost = effective_objective_coefficient(*context->problem, rhs);
                if (lhs_cost != rhs_cost)
                    return lhs_cost < rhs_cost;
                return lhs < rhs;
            });

            for (std::size_t i = 1; i < bucket.size(); ++i) {
                const int loser = bucket[i];
                if (!detail::apply_tightened_bounds(context->problem, loser, 0.0, 0.0, context->tol,
                                                    &fixed, fixed_variables)) {
                    return -1;
                }
            }
        }
    }
    if (fixed > 0)
        mark_structure_dirty(context);
    return fixed;
}

struct StrongProbingArtifacts {
    bool infeasible = false;
    int tightened_bounds = 0;
    int fixed_variables = 0;
    int added_constraints = 0;
};

inline StrongProbingArtifacts strong_probe_root(Problem* problem, double tol, int max_candidates,
                                                int max_passes) {
    StrongProbingArtifacts out;
    if (problem == nullptr || max_candidates <= 0)
        return out;

    std::vector<detail::SparseRowView> row_views;
    row_views.reserve(problem->base_constraints.size());
    std::vector<std::vector<int>> col_to_rows(problem->lower_bounds.size());
    for (int row_index = 0; row_index < static_cast<int>(problem->base_constraints.size());
         ++row_index) {
        const auto& row = problem->base_constraints[row_index];
        row_views.push_back(detail::SparseRowView{&row.indices, &row.values, row.sense, row.rhs});
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index >= 0 && index < static_cast<int>(col_to_rows.size()) &&
                problem->variable_types[index] == VariableType::Binary &&
                std::abs(row.values[k]) > kCoeffTol) {
                col_to_rows[index].push_back(row_index);
            }
        }
    }

    struct CandidateScore {
        int index = -1;
        double score = 0.0;
        int incidents = 0;
    };
    std::vector<CandidateScore> candidates;
    for (int j = 0; j < static_cast<int>(problem->variable_types.size()); ++j) {
        if (problem->variable_types[j] != VariableType::Binary)
            continue;
        candidates.push_back(
            CandidateScore{j, probing_score_for_variable(*problem, row_views, col_to_rows[j], j),
                           static_cast<int>(col_to_rows[j].size())});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const CandidateScore& lhs, const CandidateScore& rhs) {
                  if (lhs.score != rhs.score)
                      return lhs.score > rhs.score;
                  if (lhs.incidents != rhs.incidents)
                      return lhs.incidents > rhs.incidents;
                  return lhs.index < rhs.index;
              });
    if (static_cast<int>(candidates.size()) > max_candidates)
        candidates.resize(max_candidates);

    std::unordered_set<std::string> row_signatures;
    for (const auto& row : problem->base_constraints)
        row_signatures.insert(sparse_row_signature(row));

    for (const auto& candidate : candidates) {
        std::array<NodeBoundPresolveResult, 2> branch_bounds;
        std::array<bool, 2> branch_infeasible{false, false};

        for (int value = 0; value <= 1; ++value) {
            Eigen::VectorXd lower = problem->lower_bounds;
            Eigen::VectorXd upper = problem->upper_bounds;
            lower(candidate.index) = static_cast<double>(value);
            upper(candidate.index) = static_cast<double>(value);
            branch_bounds[value] =
                presolve_mip_node_bounds(*problem, lower, upper, {}, tol, std::max(2, max_passes));
            branch_infeasible[value] = branch_bounds[value].infeasible;
        }

        if (branch_infeasible[0] && branch_infeasible[1]) {
            out.infeasible = true;
            return out;
        }
        if (branch_infeasible[0] || branch_infeasible[1]) {
            const int forced_value = branch_infeasible[0] ? 1 : 0;
            if (!detail::apply_tightened_bounds(problem, candidate.index,
                                                static_cast<double>(forced_value),
                                                static_cast<double>(forced_value), tol,
                                                &out.tightened_bounds, &out.fixed_variables)) {
                out.infeasible = true;
                return out;
            }
            continue;
        }

        for (int other = 0; other < static_cast<int>(problem->variable_types.size()); ++other) {
            if (other == candidate.index || problem->variable_types[other] != VariableType::Binary)
                continue;

            std::array<int, 2> implied_value{-1, -1};
            for (int branch = 0; branch <= 1; ++branch) {
                if (branch_bounds[branch].lower(other) > 1.0 - tol)
                    implied_value[branch] = 1;
                else if (branch_bounds[branch].upper(other) < tol)
                    implied_value[branch] = 0;
            }

            if (implied_value[0] >= 0 && implied_value[0] == implied_value[1]) {
                if (!detail::apply_tightened_bounds(problem, other,
                                                    static_cast<double>(implied_value[0]),
                                                    static_cast<double>(implied_value[0]), tol,
                                                    &out.tightened_bounds, &out.fixed_variables)) {
                    out.infeasible = true;
                    return out;
                }
            }

            for (int branch = 0; branch <= 1; ++branch) {
                if (implied_value[branch] < 0)
                    continue;
                SparseLinearConstraint implication = implication_to_constraint(
                    candidate.index, branch == 1, other, implied_value[branch] == 1);
                detail::canonicalize_sparse_row(&implication, problem->lower_bounds,
                                                problem->upper_bounds, nullptr, tol);
                const std::string signature = sparse_row_signature(implication);
                if (row_signatures.insert(signature).second) {
                    problem->base_constraints.push_back(std::move(implication));
                    ++out.added_constraints;
                }
            }
        }
    }

    return out;
}

inline bool run_fast_root_passes(RootPresolveContext* context) {
    if (context == nullptr || context->problem == nullptr || context->result == nullptr)
        return false;

    bool changed = false;
    const auto [relaxed_lower, relaxed_upper] =
        detail::relax_huge_bounds(context->problem, context->tol);
    if (relaxed_lower > 0 || relaxed_upper > 0) {
        context->result->relaxed_huge_lower_bounds += relaxed_lower;
        context->result->relaxed_huge_upper_bounds += relaxed_upper;
        changed = true;
    }

    const int row_result = canonicalize_and_merge_rows(
        context->problem, context->tol, &context->result->removed_rows,
        &context->result->removed_coeffs, &context->result->tightened_bounds,
        &context->result->fixed_variables);
    if (row_result < 0) {
        context->result->infeasible = true;
        return false;
    }
    if (row_result > 0)
        changed = true;

    const int implied_integer_count = detect_implied_integers(context->problem, context->tol);
    if (implied_integer_count < 0) {
        context->result->infeasible = true;
        return false;
    }
    if (implied_integer_count > 0) {
        changed = true;
        mark_structure_dirty(context);
    }

    const NodeBoundPresolveResult tightened =
        presolve_mip_node_bounds(*context->problem, context->problem->lower_bounds,
                                 context->problem->upper_bounds, {}, context->tol, 2);
    if (tightened.infeasible) {
        context->result->infeasible = true;
        return false;
    }
    if (tightened.tightened_bounds > 0) {
        context->problem->lower_bounds = tightened.lower;
        context->problem->upper_bounds = tightened.upper;
        context->result->tightened_bounds += tightened.tightened_bounds;
        changed = true;
    }

    if (changed)
        mark_structure_dirty(context);
    return changed;
}

inline bool run_medium_root_passes(RootPresolveContext* context) {
    if (context == nullptr || context->problem == nullptr || context->result == nullptr)
        return false;

    bool changed = false;
    const int ncols = static_cast<int>(context->problem->variable_types.size());
    const int nrows = static_cast<int>(context->problem->base_constraints.size());
    const int nbinary = count_binary_variables(*context->problem);
    const bool enable_large_scale_reductions = nbinary >= 8 || ncols >= 12 || nrows >= 8;

    const int singleton_substitutions = detail::singleton_column_substitution(
        context->problem, context->tol, &context->result->aggregations);
    if (singleton_substitutions > 0)
        changed = true;

    const int simple_substitutions = detail::simple_substitution(context->problem, context->tol,
                                                                 &context->result->removed_coeffs,
                                                                 &context->result->aggregations);
    if (simple_substitutions > 0)
        changed = true;

    if (enable_large_scale_reductions) {
        const int simplified_ineq = simplify_integral_inequalities(context->problem, context->tol);
        if (simplified_ineq < 0) {
            context->result->infeasible = true;
            return false;
        }
        if (simplified_ineq > 0)
            changed = true;
    }

    int strengthened_coeffs = 0;
    const auto [strengthened, strengthening_tightened] = detail::coefficient_strengthening(
        context->problem, {}, context->tol, nullptr, &strengthened_coeffs);
    if (strengthened > 0 || strengthening_tightened > 0) {
        context->result->strengthened_coeffs += strengthened;
        context->result->tightened_bounds += strengthening_tightened;
        changed = true;
    }

    if (enable_large_scale_reductions) {
        const StrongProbingArtifacts probing =
            strong_probe_root(context->problem, context->tol, 12, context->max_passes);
        if (probing.infeasible) {
            context->result->infeasible = true;
            return false;
        }
        if (probing.tightened_bounds > 0 || probing.fixed_variables > 0 ||
            probing.added_constraints > 0) {
            context->result->tightened_bounds += probing.tightened_bounds;
            context->result->fixed_variables += probing.fixed_variables;
            changed = true;
        }

        const int clique_fixed =
            clique_merge_parallel_columns(context, &context->result->fixed_variables);
        if (clique_fixed < 0) {
            context->result->infeasible = true;
            return false;
        }
        if (clique_fixed > 0) {
            context->result->tightened_bounds += clique_fixed;
            changed = true;
        }
    }

    if (changed)
        mark_structure_dirty(context);
    return changed;
}

inline bool run_exhaustive_root_passes(RootPresolveContext* context) {
    if (context == nullptr || context->problem == nullptr || context->result == nullptr)
        return false;

    bool changed = false;

    const int dual_fixed = detail::dual_fix_variables(context->problem, context->tol,
                                                      &context->result->fixed_variables);
    if (dual_fixed < 0) {
        context->result->infeasible = true;
        return false;
    }
    if (dual_fixed > 0) {
        context->result->tightened_bounds += dual_fixed;
        changed = true;
    }

    const int dual_tightened = detail::dual_inference_bound_tightening(
        context->problem, context->tol, std::max(2, context->result->detected_components));
    if (dual_tightened < 0) {
        context->result->infeasible = true;
        return false;
    }
    if (dual_tightened > 0) {
        context->result->tightened_bounds += dual_tightened;
        changed = true;
    }

    const int free_substitutions = detail::free_variable_substitution(
        context->problem, context->tol, &context->result->removed_coeffs,
        &context->result->aggregations);
    if (free_substitutions > 0)
        changed = true;

    const int sparsified = detail::sparsify_with_equalities(context->problem, context->tol, 1e3,
                                                            &context->result->removed_coeffs);
    if (sparsified > 0)
        changed = true;

    for (int row_index = 0; row_index < static_cast<int>(context->problem->base_constraints.size());
         ++row_index) {
        if (detail::try_aggregate_implied_free_continuous_variable(
                context->problem, row_index, context->tol, &context->result->removed_coeffs,
                &context->result->aggregations)) {
            changed = true;
            break;
        }
    }

    if (changed)
        mark_structure_dirty(context);
    return changed;
}

inline RootProblemPresolveResult legacy_root_problem_presolve(const Problem& input, double tol,
                                                              int max_passes) {
    RootProblemPresolveResult out;
    out.problem = input;

    int n = static_cast<int>(out.problem.lower_bounds.size());
    n = std::min(n, static_cast<int>(out.problem.upper_bounds.size()));
    n = std::min(n, static_cast<int>(out.problem.variable_types.size()));
    out.problem.lower_bounds.conservativeResize(n);
    out.problem.upper_bounds.conservativeResize(n);
    if (out.problem.objective_coefficients.size() == 0) {
        out.problem.objective_coefficients = Eigen::VectorXd::Zero(n);
    } else {
        out.problem.objective_coefficients.conservativeResize(n);
    }

    for (int j = 0; j < n; ++j) {
        double tightened_lower = out.problem.lower_bounds(j);
        double tightened_upper = out.problem.upper_bounds(j);
        detail::tighten_discrete_bounds(out.problem.variable_types[j], &tightened_lower,
                                        &tightened_upper, tol);
        out.problem.lower_bounds(j) = tightened_lower;
        out.problem.upper_bounds(j) = tightened_upper;
        if (out.problem.upper_bounds(j) + tol < out.problem.lower_bounds(j)) {
            out.infeasible = true;
            return out;
        }
    }

    for (int pass = 0; pass < std::max(1, max_passes); ++pass) {
        bool changed = false;
        const auto [relaxed_lower, relaxed_upper] = detail::relax_huge_bounds(&out.problem, tol);
        if (relaxed_lower > 0 || relaxed_upper > 0) {
            out.relaxed_huge_lower_bounds += relaxed_lower;
            out.relaxed_huge_upper_bounds += relaxed_upper;
            changed = true;
        }

        std::vector<std::string> row_signature_order;
        row_signature_order.reserve(out.problem.base_constraints.size());
        std::unordered_map<std::string, detail::RowEnvelope> row_envelopes;

        for (const auto& base_row : out.problem.base_constraints) {
            SparseLinearConstraint row = base_row;
            detail::canonicalize_sparse_row(&row, out.problem.lower_bounds,
                                            out.problem.upper_bounds, &out.removed_coeffs, tol);

            if (row.indices.size() == 1 && row.values.size() == 1) {
                if (!detail::tighten_singleton_row_bounds(
                        &out.problem, row, tol, &out.tightened_bounds, &out.fixed_variables)) {
                    out.infeasible = true;
                    return out;
                }
                ++out.removed_rows;
                changed = true;
                continue;
            }

            const detail::SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
            const detail::SparseRowActivitySummary summary = detail::sparse_row_activity_summary(
                view, out.problem.lower_bounds, out.problem.upper_bounds);
            if (!detail::sparse_row_is_feasible(summary, row.sense, row.rhs, tol)) {
                out.infeasible = true;
                return out;
            }
            if (detail::sparse_row_is_redundant(summary, row.sense, row.rhs, tol)) {
                ++out.removed_rows;
                changed = true;
                continue;
            }

            const std::string lhs_signature = detail::sparse_row_lhs_signature(row);
            const auto [it, inserted] = row_envelopes.try_emplace(lhs_signature);
            if (inserted) {
                row_signature_order.push_back(lhs_signature);
            }
            if (!detail::absorb_row_into_envelope(&it->second, row, tol)) {
                out.infeasible = true;
                return out;
            }
        }

        detail::MergeRowsResult merged_rows =
            detail::merge_parallel_rows(row_signature_order, row_envelopes, tol);
        if (merged_rows.infeasible) {
            out.infeasible = true;
            return out;
        }
        if (merged_rows.removed_rows > 0) {
            out.removed_rows += merged_rows.removed_rows;
            changed = true;
        }
        out.problem.base_constraints = std::move(merged_rows.rows);

        const std::vector<detail::ComponentInfo> components =
            detail::detect_components(out.problem);
        out.detected_components =
            std::max(out.detected_components, static_cast<int>(components.size()));

        const int singleton_substitutions =
            detail::singleton_column_substitution(&out.problem, tol, &out.aggregations);
        if (singleton_substitutions > 0)
            changed = true;

        const int simple_substitutions =
            detail::simple_substitution(&out.problem, tol, &out.removed_coeffs, &out.aggregations);
        if (simple_substitutions > 0)
            changed = true;

        int strengthened_coeffs = 0;
        const auto [strengthened, strengthening_tightened] =
            detail::coefficient_strengthening(&out.problem, {}, tol, nullptr, &strengthened_coeffs);
        if (strengthened > 0 || strengthening_tightened > 0) {
            out.strengthened_coeffs += strengthened;
            out.tightened_bounds += strengthening_tightened;
            changed = true;
        }

        const NodeBoundPresolveResult tightened = presolve_mip_node_bounds(
            out.problem, out.problem.lower_bounds, out.problem.upper_bounds, {}, tol, 2);
        if (tightened.infeasible) {
            out.infeasible = true;
            return out;
        }
        if (tightened.tightened_bounds > 0) {
            out.problem.lower_bounds = tightened.lower;
            out.problem.upper_bounds = tightened.upper;
            out.tightened_bounds += tightened.tightened_bounds;
            changed = true;
        }

        const detail::ProbingResult probing = detail::probe_mip_root(
            &out.problem, {}, tol, &out.tightened_bounds, &out.fixed_variables);
        if (probing.infeasible) {
            out.infeasible = true;
            return out;
        }
        if (probing.tightened_bounds > 0 || probing.fixed_variables > 0)
            changed = true;

        const int dual_fixed = detail::dual_fix_variables(&out.problem, tol, &out.fixed_variables);
        if (dual_fixed < 0) {
            out.infeasible = true;
            return out;
        }
        if (dual_fixed > 0) {
            out.tightened_bounds += dual_fixed;
            changed = true;
        }

        const int dual_tightened = detail::dual_inference_bound_tightening(
            &out.problem, tol, std::max(2, out.detected_components));
        if (dual_tightened < 0) {
            out.infeasible = true;
            return out;
        }
        if (dual_tightened > 0) {
            out.tightened_bounds += dual_tightened;
            changed = true;
        }

        const int free_substitutions = detail::free_variable_substitution(
            &out.problem, tol, &out.removed_coeffs, &out.aggregations);
        if (free_substitutions > 0)
            changed = true;

        const int sparsified =
            detail::sparsify_with_equalities(&out.problem, tol, 1e3, &out.removed_coeffs);
        if (sparsified > 0)
            changed = true;

        bool aggregated = false;
        for (int row_index = 0; row_index < static_cast<int>(out.problem.base_constraints.size());
             ++row_index) {
            if (detail::try_aggregate_implied_free_continuous_variable(
                    &out.problem, row_index, tol, &out.removed_coeffs, &out.aggregations)) {
                aggregated = true;
                changed = true;
                break;
            }
        }

        if (!changed && !aggregated)
            break;
    }

    return out;
}

inline double effective_objective_coefficient(const Problem& problem, int index) {
    if (index < 0 || index >= problem.objective_coefficients.size())
        return 0.0;
    return problem.maximize ? -problem.objective_coefficients(index)
                            : problem.objective_coefficients(index);
}

inline int row_nonzeros(const SparseLinearConstraint& row) {
    int nnz = 0;
    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        if (std::abs(row.values[k]) > kCoeffTol)
            ++nnz;
    }
    return nnz;
}

inline void count_locks_for_column_occurrence(LinearConstraintSense sense, double coeff,
                                              int* down_locks, int* up_locks) {
    if (down_locks == nullptr || up_locks == nullptr || std::abs(coeff) <= kCoeffTol)
        return;

    switch (sense) {
        case LinearConstraintSense::LessEqual:
            if (coeff > 0.0) {
                ++(*up_locks);
            } else {
                ++(*down_locks);
            }
            break;
        case LinearConstraintSense::GreaterEqual:
            if (coeff > 0.0) {
                ++(*down_locks);
            } else {
                ++(*up_locks);
            }
            break;
        case LinearConstraintSense::Equal:
            ++(*down_locks);
            ++(*up_locks);
            break;
    }
}

inline bool substitute_objective_from_equality(Problem* problem,
                                               const SparseLinearConstraint& defining_row,
                                               int pivot_pos, double tol) {
    if (problem == nullptr || defining_row.sense != LinearConstraintSense::Equal || pivot_pos < 0 ||
        pivot_pos >= static_cast<int>(defining_row.indices.size()) ||
        pivot_pos >= static_cast<int>(defining_row.values.size())) {
        return false;
    }

    const int pivot = defining_row.indices[pivot_pos];
    if (pivot < 0 || pivot >= problem->objective_coefficients.size())
        return false;

    const double pivot_coeff = defining_row.values[pivot_pos];
    const double objective_coeff = problem->objective_coefficients(pivot);
    if (std::abs(pivot_coeff) <= tol || std::abs(objective_coeff) <= tol)
        return false;

    problem->objective_constant += objective_coeff * defining_row.rhs / pivot_coeff;
    for (int k = 0; k < static_cast<int>(defining_row.indices.size()) &&
                    k < static_cast<int>(defining_row.values.size());
         ++k) {
        if (k == pivot_pos)
            continue;
        const int index = defining_row.indices[k];
        if (index < 0 || index >= problem->objective_coefficients.size())
            continue;
        problem->objective_coefficients(index) -=
            objective_coeff * defining_row.values[k] / pivot_coeff;
    }
    problem->objective_coefficients(pivot) = 0.0;
    return true;
}

inline bool substitute_pivot_from_equality(Problem* problem, int row_index, int pivot_pos,
                                           const std::vector<std::vector<int>>& col_to_rows,
                                           double tol, int max_fill_per_row, int* removed_coeffs,
                                           int* substitution_count) {
    if (problem == nullptr || row_index < 0 ||
        row_index >= static_cast<int>(problem->base_constraints.size())) {
        return false;
    }

    const SparseLinearConstraint defining_row = problem->base_constraints[row_index];
    if (defining_row.sense != LinearConstraintSense::Equal || pivot_pos < 0 ||
        pivot_pos >= static_cast<int>(defining_row.indices.size()) ||
        pivot_pos >= static_cast<int>(defining_row.values.size())) {
        return false;
    }

    const int pivot = defining_row.indices[pivot_pos];
    if (pivot < 0 || pivot >= static_cast<int>(col_to_rows.size()))
        return false;

    const double pivot_coeff = defining_row.values[pivot_pos];
    if (std::abs(pivot_coeff) <= tol)
        return false;

    bool changed = substitute_objective_from_equality(problem, defining_row, pivot_pos, tol);
    const auto affected_rows = col_to_rows[pivot];

    for (const int other_row_index : affected_rows) {
        if (other_row_index == row_index || other_row_index < 0 ||
            other_row_index >= static_cast<int>(problem->base_constraints.size())) {
            continue;
        }

        const auto other_pivot_pos =
            find_row_coefficient_position(problem->base_constraints[other_row_index], pivot);
        if (!other_pivot_pos.has_value())
            continue;

        const double factor =
            problem->base_constraints[other_row_index].values[*other_pivot_pos] / pivot_coeff;
        if (std::abs(factor) <= tol)
            continue;

        const int old_nnz = row_nonzeros(problem->base_constraints[other_row_index]);
        SparseLinearConstraint candidate = problem->base_constraints[other_row_index];
        for (int k = 0; k < static_cast<int>(defining_row.indices.size()) &&
                        k < static_cast<int>(defining_row.values.size());
             ++k) {
            candidate.indices.push_back(defining_row.indices[k]);
            candidate.values.push_back(-factor * defining_row.values[k]);
        }
        candidate.rhs -= factor * defining_row.rhs;

        int candidate_removed_coeffs = 0;
        canonicalize_sparse_row(&candidate, problem->lower_bounds, problem->upper_bounds,
                                &candidate_removed_coeffs, tol);

        const int new_nnz = row_nonzeros(candidate);
        if (max_fill_per_row >= 0 && new_nnz > old_nnz + max_fill_per_row)
            continue;
        if (new_nnz > old_nnz && std::abs(problem->objective_coefficients(pivot)) <= tol)
            continue;

        problem->base_constraints[other_row_index] = std::move(candidate);
        if (removed_coeffs)
            *removed_coeffs += candidate_removed_coeffs;
        changed = true;
    }

    if (changed && substitution_count)
        ++(*substitution_count);
    return changed;
}

inline int singleton_column_substitution(Problem* problem, double tol, int* substitution_count) {
    if (problem == nullptr)
        return 0;

    const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
    int changed = 0;

    for (int column = 0; column < static_cast<int>(col_to_rows.size()); ++column) {
        if (col_to_rows[column].size() != 1)
            continue;
        if (column < 0 || column >= problem->objective_coefficients.size() ||
            std::abs(problem->objective_coefficients(column)) <= tol) {
            continue;
        }

        const int row_index = col_to_rows[column].front();
        if (row_index < 0 || row_index >= static_cast<int>(problem->base_constraints.size()))
            continue;
        const auto pivot_pos =
            find_row_coefficient_position(problem->base_constraints[row_index], column);
        if (!pivot_pos.has_value())
            continue;
        if (problem->base_constraints[row_index].sense != LinearConstraintSense::Equal ||
            row_nonzeros(problem->base_constraints[row_index]) <= 1) {
            continue;
        }

        if (substitute_objective_from_equality(problem, problem->base_constraints[row_index],
                                               *pivot_pos, tol)) {
            ++changed;
            if (substitution_count)
                ++(*substitution_count);
        }
    }

    return changed;
}

inline int simple_substitution(Problem* problem, double tol, int* removed_coeffs,
                               int* substitution_count) {
    if (problem == nullptr)
        return 0;

    const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
    int changed = 0;

    for (int row_index = 0; row_index < static_cast<int>(problem->base_constraints.size());
         ++row_index) {
        const auto& row = problem->base_constraints[row_index];
        if (row.sense != LinearConstraintSense::Equal || row_nonzeros(row) != 2)
            continue;

        int best_pos = -1;
        int best_incidence = -1;
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= static_cast<int>(col_to_rows.size()) ||
                std::abs(row.values[k]) <= tol) {
                continue;
            }
            const int incidence = static_cast<int>(col_to_rows[index].size());
            const double abs_obj = std::abs(problem->objective_coefficients(index));
            if (incidence > best_incidence ||
                (incidence == best_incidence &&
                 abs_obj > std::abs(problem->objective_coefficients(
                               row.indices[std::max(0, best_pos)])))) {
                best_pos = k;
                best_incidence = incidence;
            }
        }

        if (best_pos < 0)
            continue;

        if (substitute_pivot_from_equality(problem, row_index, best_pos, col_to_rows, tol, 0,
                                           removed_coeffs, substitution_count)) {
            ++changed;
        }
    }

    return changed;
}

inline int dual_fix_variables(Problem* problem, double tol, int* fixed_variables) {
    if (problem == nullptr)
        return 0;

    const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
    int tightened = 0;

    for (int column = 0;
         column < problem->lower_bounds.size() && column < problem->upper_bounds.size() &&
         column < static_cast<int>(problem->variable_types.size());
         ++column) {
        if (std::isfinite(problem->lower_bounds(column)) &&
            std::isfinite(problem->upper_bounds(column)) &&
            std::abs(problem->lower_bounds(column) - problem->upper_bounds(column)) <= tol) {
            continue;
        }

        int down_locks = 0;
        int up_locks = 0;
        for (const int row_index : col_to_rows[column]) {
            if (row_index < 0 || row_index >= static_cast<int>(problem->base_constraints.size()))
                continue;
            const auto pos =
                find_row_coefficient_position(problem->base_constraints[row_index], column);
            if (!pos.has_value())
                continue;
            count_locks_for_column_occurrence(problem->base_constraints[row_index].sense,
                                              problem->base_constraints[row_index].values[*pos],
                                              &down_locks, &up_locks);
            if (down_locks > 0 && up_locks > 0)
                break;
        }

        const double effective_cost = effective_objective_coefficient(*problem, column);
        if (down_locks == 0 && effective_cost > tol &&
            std::isfinite(problem->lower_bounds(column))) {
            if (apply_tightened_bounds(problem, column, problem->lower_bounds(column),
                                       problem->lower_bounds(column), tol, &tightened,
                                       fixed_variables)) {
                continue;
            }
            return -1;
        }
        if (up_locks == 0 && effective_cost < -tol &&
            std::isfinite(problem->upper_bounds(column))) {
            if (apply_tightened_bounds(problem, column, problem->upper_bounds(column),
                                       problem->upper_bounds(column), tol, &tightened,
                                       fixed_variables)) {
                continue;
            }
            return -1;
        }
    }

    return tightened;
}

inline int free_variable_substitution(Problem* problem, double tol, int* removed_coeffs,
                                      int* substitution_count) {
    if (problem == nullptr)
        return 0;

    const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
    int changed = 0;

    for (int row_index = 0; row_index < static_cast<int>(problem->base_constraints.size());
         ++row_index) {
        const auto& row = problem->base_constraints[row_index];
        const int row_nnz = row_nonzeros(row);
        if (row.sense != LinearConstraintSense::Equal || row_nnz < 3 || row_nnz > 10)
            continue;

        int best_pos = -1;
        int best_score = std::numeric_limits<int>::min();
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int pivot = row.indices[k];
            if (pivot < 0 || pivot >= static_cast<int>(col_to_rows.size()) ||
                std::abs(row.values[k]) <= tol) {
                continue;
            }

            const int incidence = static_cast<int>(col_to_rows[pivot].size());
            if (incidence <= 1 && std::abs(problem->objective_coefficients(pivot)) <= tol)
                continue;

            const int estimated_fill = std::max(0, incidence - 1) * std::max(0, row_nnz - 2);
            if (estimated_fill > 16)
                continue;

            const int score = 8 * std::max(0, incidence - 1) - estimated_fill;
            if (score > best_score) {
                best_score = score;
                best_pos = k;
            }
        }

        if (best_pos < 0)
            continue;

        if (substitute_pivot_from_equality(problem, row_index, best_pos, col_to_rows, tol, 4,
                                           removed_coeffs, substitution_count)) {
            ++changed;
        }
    }

    return changed;
}

inline int sparsify_with_equalities(Problem* problem, double tol, double max_scale,
                                    int* removed_coeffs) {
    if (problem == nullptr)
        return 0;

    const std::vector<std::vector<int>> col_to_rows = build_column_to_rows(*problem);
    int changed = 0;

    for (int eq_row_index = 0; eq_row_index < static_cast<int>(problem->base_constraints.size());
         ++eq_row_index) {
        const auto& eq_row = problem->base_constraints[eq_row_index];
        const int eq_nnz = row_nonzeros(eq_row);
        if (eq_row.sense != LinearConstraintSense::Equal || eq_nnz < 2 || eq_nnz > 8)
            continue;

        bool applied = false;
        for (int k = 0; k < static_cast<int>(eq_row.indices.size()) &&
                        k < static_cast<int>(eq_row.values.size()) && !applied;
             ++k) {
            const int pivot = eq_row.indices[k];
            const double pivot_coeff = eq_row.values[k];
            if (pivot < 0 || pivot >= static_cast<int>(col_to_rows.size()) ||
                std::abs(pivot_coeff) <= tol) {
                continue;
            }

            for (const int other_row_index : col_to_rows[pivot]) {
                if (other_row_index == eq_row_index || other_row_index < 0 ||
                    other_row_index >= static_cast<int>(problem->base_constraints.size())) {
                    continue;
                }

                const auto other_pos = find_row_coefficient_position(
                    problem->base_constraints[other_row_index], pivot);
                if (!other_pos.has_value())
                    continue;

                const double factor =
                    problem->base_constraints[other_row_index].values[*other_pos] / pivot_coeff;
                if (std::abs(factor) <= tol || std::abs(factor) > max_scale)
                    continue;

                const int old_nnz = row_nonzeros(problem->base_constraints[other_row_index]);
                SparseLinearConstraint candidate = problem->base_constraints[other_row_index];
                for (int t = 0; t < static_cast<int>(eq_row.indices.size()) &&
                                t < static_cast<int>(eq_row.values.size());
                     ++t) {
                    candidate.indices.push_back(eq_row.indices[t]);
                    candidate.values.push_back(-factor * eq_row.values[t]);
                }
                candidate.rhs -= factor * eq_row.rhs;

                int candidate_removed_coeffs = 0;
                canonicalize_sparse_row(&candidate, problem->lower_bounds, problem->upper_bounds,
                                        &candidate_removed_coeffs, tol);
                const int new_nnz = row_nonzeros(candidate);
                if (new_nnz >= old_nnz)
                    continue;

                problem->base_constraints[other_row_index] = std::move(candidate);
                if (removed_coeffs)
                    *removed_coeffs += candidate_removed_coeffs;
                ++changed;
                applied = true;
                break;
            }
        }
    }

    return changed;
}

/// Dual inference for bound tightening (PaPILO style)
/// Uses dual information to tighten bounds on variables
inline int dual_inference_bound_tightening(Problem* problem, double tol = 1e-9,
                                           int max_iterations = 4) {
    if (problem == nullptr)
        return 0;

    int n = static_cast<int>(problem->lower_bounds.size());
    n = std::min(n, static_cast<int>(problem->upper_bounds.size()));
    n = std::min(n, static_cast<int>(problem->variable_types.size()));
    int tightened = 0;
    int fixed_variables = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int iter_tightened = 0;

        for (const auto& row : problem->base_constraints) {
            const SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
            const SparseRowActivitySummary summary =
                sparse_row_activity_summary(view, problem->lower_bounds, problem->upper_bounds);

            if (!sparse_row_is_feasible(summary, row.sense, row.rhs, tol))
                return -1;

            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                const int index = row.indices[k];
                if (index < 0 || index >= n)
                    continue;
                if (std::abs(row.values[k]) <= kCoeffTol)
                    continue;
                if (problem->variable_types[index] == VariableType::Continuous &&
                    row.sense != LinearConstraintSense::Equal) {
                    continue;
                }

                double new_lb = problem->lower_bounds(index);
                double new_ub = problem->upper_bounds(index);

                const SparseVariableContribution var_contrib = sparse_variable_contribution(
                    row.values[k], index, problem->lower_bounds, problem->upper_bounds);

                const bool other_min_finite =
                    summary.min_infinite_terms - (var_contrib.min_finite ? 0 : 1) == 0;
                const bool other_max_finite =
                    summary.max_infinite_terms - (var_contrib.max_finite ? 0 : 1) == 0;

                const double other_min =
                    summary.min_activity - (var_contrib.min_finite ? var_contrib.min_value : 0.0);
                const double other_max =
                    summary.max_activity - (var_contrib.max_finite ? var_contrib.max_value : 0.0);

                double implied_lb = -std::numeric_limits<double>::infinity();
                double implied_ub = std::numeric_limits<double>::infinity();

                switch (row.sense) {
                    case LinearConstraintSense::LessEqual:
                        if (row.values[k] > 0 && other_min_finite)
                            implied_ub = (row.rhs - other_min) / row.values[k];
                        else if (row.values[k] < 0 && other_min_finite)
                            implied_lb = (row.rhs - other_min) / row.values[k];
                        break;
                    case LinearConstraintSense::GreaterEqual:
                        if (row.values[k] > 0 && other_max_finite)
                            implied_lb = (row.rhs - other_max) / row.values[k];
                        else if (row.values[k] < 0 && other_max_finite)
                            implied_ub = (row.rhs - other_max) / row.values[k];
                        break;
                    case LinearConstraintSense::Equal:
                        if (row.values[k] > 0) {
                            if (other_min_finite)
                                implied_ub = (row.rhs - other_min) / row.values[k];
                            if (other_max_finite)
                                implied_lb = (row.rhs - other_max) / row.values[k];
                        } else {
                            if (other_min_finite)
                                implied_lb = (row.rhs - other_min) / row.values[k];
                            if (other_max_finite)
                                implied_ub = (row.rhs - other_max) / row.values[k];
                        }
                        break;
                }

                if (problem->variable_types[index] == VariableType::Binary) {
                    implied_lb = std::max(implied_lb, 0.0);
                    implied_ub = std::min(implied_ub, 1.0);
                } else if (problem->variable_types[index] != VariableType::Continuous) {
                    implied_lb = std::ceil(implied_lb - tol);
                    implied_ub = std::floor(implied_ub + tol);
                }

                if (implied_lb > new_lb + tol)
                    new_lb = implied_lb;
                if (implied_ub < new_ub - tol)
                    new_ub = implied_ub;

                const int before = iter_tightened;
                if (!detail::apply_tightened_bounds(problem, index, new_lb, new_ub, tol,
                                                    &iter_tightened, &fixed_variables)) {
                    return -1;
                }
                if (iter_tightened != before && std::isfinite(problem->lower_bounds(index)) &&
                    std::isfinite(problem->upper_bounds(index)) &&
                    std::abs(problem->lower_bounds(index) - problem->upper_bounds(index)) <= tol) {
                    max_iterations = std::max(max_iterations, iter + 2);
                }
            }
        }

        if (iter_tightened == 0)
            break;
        tightened += iter_tightened;
    }

    return tightened;
}

inline bool try_aggregate_implied_free_continuous_variable(Problem* problem, int row_index,
                                                           double tol, int* removed_coeffs,
                                                           int* aggregation_count) {
    if (!problem || row_index < 0 ||
        row_index >= static_cast<int>(problem->base_constraints.size())) {
        return false;
    }

    auto& defining_row = problem->base_constraints[row_index];
    if (defining_row.sense != LinearConstraintSense::Equal)
        return false;

    const int row_nnz = std::min(static_cast<int>(defining_row.indices.size()),
                                 static_cast<int>(defining_row.values.size()));
    if (row_nnz < 2 || row_nnz > 8)
        return false;

    std::vector<std::vector<int>> col_to_rows(problem->lower_bounds.size());
    std::vector<double> col_max_abs(problem->lower_bounds.size(), 0.0);
    for (int r = 0; r < static_cast<int>(problem->base_constraints.size()); ++r) {
        const auto& row = problem->base_constraints[r];
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= problem->lower_bounds.size())
                continue;
            if (std::abs(row.values[k]) <= kCoeffTol)
                continue;
            col_to_rows[index].push_back(r);
            col_max_abs[index] = std::max(col_max_abs[index], std::abs(row.values[k]));
        }
    }

    double row_max_abs = 0.0;
    for (int k = 0; k < row_nnz; ++k)
        row_max_abs = std::max(row_max_abs, std::abs(defining_row.values[k]));

    struct Candidate {
        int pivot = -1;
        int pivot_pos = -1;
        int col_nnz = 0;
        int estimated_fill = 0;
        double coeff_abs = 0.0;
    };
    std::optional<Candidate> best;

    for (int k = 0; k < row_nnz; ++k) {
        const int pivot = defining_row.indices[k];
        const double aij = defining_row.values[k];
        if (pivot < 0 || pivot >= problem->lower_bounds.size())
            continue;
        if (problem->variable_types[pivot] != VariableType::Continuous)
            continue;
        if (std::abs(aij) <= kCoeffTol)
            continue;

        const auto implied = implied_interval_for_equality_pivot(
            defining_row, pivot, problem->lower_bounds, problem->upper_bounds, tol);
        if (!implied.has_value())
            continue;

        const double explicit_l = problem->lower_bounds(pivot);
        const double explicit_u = problem->upper_bounds(pivot);
        if ((std::isfinite(explicit_l) && implied->first < explicit_l - tol) ||
            (std::isfinite(explicit_u) && implied->second > explicit_u + tol)) {
            continue;
        }

        const int col_nnz = static_cast<int>(col_to_rows[pivot].size());
        if (col_nnz <= 1 || col_nnz > 4)
            continue;

        const double coeff_abs = std::abs(aij);
        if (coeff_abs < 0.01 * row_max_abs || coeff_abs < 0.01 * col_max_abs[pivot]) {
            continue;
        }

        const int estimated_fill = (col_nnz - 1) * std::max(0, row_nnz - 2);
        if (estimated_fill > 12)
            continue;

        Candidate candidate{pivot, k, col_nnz, estimated_fill, coeff_abs};
        if (!best.has_value() || candidate.estimated_fill < best->estimated_fill ||
            (candidate.estimated_fill == best->estimated_fill &&
             candidate.col_nnz < best->col_nnz) ||
            (candidate.estimated_fill == best->estimated_fill &&
             candidate.col_nnz == best->col_nnz && candidate.coeff_abs > best->coeff_abs)) {
            best = candidate;
        }
    }

    if (!best.has_value())
        return false;

    const int pivot = best->pivot;
    const int pivot_pos = best->pivot_pos;
    const double pivot_coeff = defining_row.values[pivot_pos];
    const double objective_coeff = pivot < problem->objective_coefficients.size()
                                       ? problem->objective_coefficients(pivot)
                                       : 0.0;

    if (std::abs(objective_coeff) > kCoeffTol) {
        problem->objective_constant += objective_coeff * defining_row.rhs / pivot_coeff;
        for (int k = 0; k < row_nnz; ++k) {
            if (k == pivot_pos)
                continue;
            const int index = defining_row.indices[k];
            if (index < 0 || index >= problem->objective_coefficients.size())
                continue;
            problem->objective_coefficients(index) -=
                objective_coeff * defining_row.values[k] / pivot_coeff;
        }
        problem->objective_coefficients(pivot) = 0.0;
    }

    const auto affected_rows = col_to_rows[pivot];
    for (const int other_row_index : affected_rows) {
        if (other_row_index == row_index)
            continue;
        auto& other_row = problem->base_constraints[other_row_index];
        const auto other_pivot_pos = find_row_coefficient_position(other_row, pivot);
        if (!other_pivot_pos.has_value())
            continue;
        const double factor = other_row.values[*other_pivot_pos] / pivot_coeff;
        if (std::abs(factor) <= kCoeffTol)
            continue;

        other_row.indices.push_back(pivot);
        other_row.values.push_back(-other_row.values[*other_pivot_pos]);
        for (int k = 0; k < row_nnz; ++k) {
            other_row.indices.push_back(defining_row.indices[k]);
            other_row.values.push_back(-factor * defining_row.values[k]);
        }
        other_row.rhs -= factor * defining_row.rhs;
        canonicalize_sparse_row(&other_row, problem->lower_bounds, problem->upper_bounds,
                                removed_coeffs, tol);
    }

    if (aggregation_count)
        ++(*aggregation_count);
    return true;
}

} // namespace detail

inline double cut_activity_bound(const Cut& cut, const Eigen::VectorXd& lower,
                                 const Eigen::VectorXd& upper, bool use_upper) {
    double activity = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        if (index < 0 || index >= lower.size() || index >= upper.size())
            continue;
        const double coeff = cut.values[k];
        const bool take_upper = use_upper ? (coeff >= 0.0) : (coeff < 0.0);
        activity += coeff * (take_upper ? upper(index) : lower(index));
    }
    return activity;
}

SimplifiedCutsResult simplify_cuts_for_bounds(const std::vector<Cut>& cuts,
                                              const Eigen::VectorXd& lower,
                                              const Eigen::VectorXd& upper, double tol) {
    SimplifiedCutsResult out;
    out.cuts.reserve(cuts.size());

    for (const auto& cut : cuts) {
        const double min_activity = cut_activity_bound(cut, lower, upper, false);
        const double max_activity = cut_activity_bound(cut, lower, upper, true);

        bool redundant = false;
        bool infeasible = false;
        switch (cut.sense) {
            case LinearConstraintSense::LessEqual:
                redundant = max_activity <= cut.rhs + tol;
                infeasible = min_activity > cut.rhs + tol;
                break;
            case LinearConstraintSense::GreaterEqual:
                redundant = min_activity >= cut.rhs - tol;
                infeasible = max_activity < cut.rhs - tol;
                break;
            case LinearConstraintSense::Equal:
                redundant = max_activity <= cut.rhs + tol && min_activity >= cut.rhs - tol;
                infeasible = max_activity < cut.rhs - tol || min_activity > cut.rhs + tol;
                break;
        }

        if (infeasible) {
            out.infeasible = true;
            out.cuts.clear();
            return out;
        }
        if (redundant)
            continue;

        Cut simplified;
        simplified.sense = cut.sense;
        simplified.rhs = cut.rhs;
        simplified.cut_type = cut.cut_type;
        simplified.strength = cut.strength;
        simplified.times_used = cut.times_used;
        simplified.age = cut.age;

        for (int k = 0;
             k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size());
             ++k) {
            const int index = cut.indices[k];
            if (index < 0 || index >= lower.size() || index >= upper.size())
                continue;
            const double coeff = cut.values[k];
            if (std::abs(coeff) <= kCoeffTol)
                continue;
            if (std::isfinite(lower(index)) && std::isfinite(upper(index)) &&
                std::abs(lower(index) - upper(index)) <= tol) {
                simplified.rhs -= coeff * lower(index);
                continue;
            }
            simplified.indices.push_back(index);
            simplified.values.push_back(coeff);
        }

        if (simplified.indices.empty()) {
            bool scalar_redundant = false;
            bool scalar_infeasible = false;
            switch (simplified.sense) {
                case LinearConstraintSense::LessEqual:
                    scalar_redundant = 0.0 <= simplified.rhs + tol;
                    scalar_infeasible = 0.0 > simplified.rhs + tol;
                    break;
                case LinearConstraintSense::GreaterEqual:
                    scalar_redundant = 0.0 >= simplified.rhs - tol;
                    scalar_infeasible = 0.0 < simplified.rhs - tol;
                    break;
                case LinearConstraintSense::Equal:
                    scalar_redundant = std::abs(simplified.rhs) <= tol;
                    scalar_infeasible = !scalar_redundant;
                    break;
            }
            if (scalar_infeasible) {
                out.infeasible = true;
                out.cuts.clear();
                return out;
            }
            if (scalar_redundant)
                continue;
        }

        out.cuts.push_back(std::move(simplified));
    }

    return out;
}

std::string cut_set_signature(const std::vector<Cut>& cuts) {
    std::ostringstream oss;
    for (const auto& cut : cuts) {
        const auto signature = simplex::bnb::detail::cut_signature(cut);
        oss << signature.lo << ':' << signature.hi << '\n';
    }
    return oss.str();
}

NodeBoundPresolveResult presolve_mip_node_bounds(const Problem& problem,
                                                 const Eigen::VectorXd& lower_in,
                                                 const Eigen::VectorXd& upper_in,
                                                 const std::vector<Cut>& extra_cuts, double tol,
                                                 int max_passes) {
    NodeBoundPresolveResult out;
    out.lower = lower_in;
    out.upper = upper_in;

    int n = static_cast<int>(problem.lower_bounds.size());
    n = std::min(n, static_cast<int>(problem.upper_bounds.size()));
    n = std::min(n, static_cast<int>(out.lower.size()));
    n = std::min(n, static_cast<int>(out.upper.size()));
    n = std::min(n, static_cast<int>(problem.variable_types.size()));
    out.lower.conservativeResize(n);
    out.upper.conservativeResize(n);

    for (int j = 0; j < n; ++j) {
        detail::tighten_discrete_bounds(problem.variable_types[j], &out.lower(j), &out.upper(j),
                                        tol);
        if (out.upper(j) + tol < out.lower(j)) {
            out.infeasible = true;
            return out;
        }
    }

    std::vector<detail::SparseRowView> rows;
    rows.reserve(problem.base_constraints.size() + extra_cuts.size());
    std::vector<std::vector<int>> col_to_rows(n);

    const auto add_row = [&](const auto& source_row) {
        const int row_index = static_cast<int>(rows.size());
        rows.push_back(detail::SparseRowView{&source_row.indices, &source_row.values,
                                             source_row.sense, source_row.rhs});
        for (int k = 0; k < static_cast<int>(source_row.indices.size()) &&
                        k < static_cast<int>(source_row.values.size());
             ++k) {
            const int index = source_row.indices[k];
            if (index < 0 || index >= n || std::abs(source_row.values[k]) <= kCoeffTol) {
                continue;
            }
            col_to_rows[index].push_back(row_index);
        }
    };

    for (const auto& row : problem.base_constraints)
        add_row(row);
    for (const auto& cut : extra_cuts)
        add_row(cut);

    std::vector<char> dirty_rows(rows.size(), 1);
    const int propagation_rounds = 2; // Reduced from max(2, 2*max_passes) for speed
    for (int round = 0; round < propagation_rounds; ++round) {
        bool any_dirty = false;
        bool changed = false;
        std::vector<char> next_dirty_rows(rows.size(), 0);

        for (int row_index = 0; row_index < static_cast<int>(rows.size()); ++row_index) {
            if (!dirty_rows[row_index])
                continue;
            any_dirty = true;

            const int tightened_before = out.tightened_bounds;
            if (!detail::tighten_bounds_from_sparse_row(rows[row_index], problem, &out.lower,
                                                        &out.upper, &out.tightened_bounds,
                                                        col_to_rows, &next_dirty_rows, tol)) {
                out.infeasible = true;
                return out;
            }
            if (out.tightened_bounds != tightened_before)
                changed = true;
        }

        if (!any_dirty || !changed)
            break;
        dirty_rows = std::move(next_dirty_rows);
    }

    return out;
}

RootProblemPresolveResult presolve_mip_root_problem(const Problem& input, double tol,
                                                    int max_passes) {
    int ncols = static_cast<int>(input.lower_bounds.size());
    ncols = std::min(ncols, static_cast<int>(input.upper_bounds.size()));
    ncols = std::min(ncols, static_cast<int>(input.variable_types.size()));
    const int nrows = static_cast<int>(input.base_constraints.size());
    if (ncols < 8 && nrows < 8) {
        return detail::legacy_root_problem_presolve(input, tol, max_passes);
    }

    RootProblemPresolveResult out;
    out.problem = input;

    int n = static_cast<int>(out.problem.lower_bounds.size());
    n = std::min(n, static_cast<int>(out.problem.upper_bounds.size()));
    n = std::min(n, static_cast<int>(out.problem.variable_types.size()));
    out.problem.lower_bounds.conservativeResize(n);
    out.problem.upper_bounds.conservativeResize(n);
    if (out.problem.objective_coefficients.size() == 0) {
        out.problem.objective_coefficients = Eigen::VectorXd::Zero(n);
    } else {
        out.problem.objective_coefficients.conservativeResize(n);
    }

    for (int j = 0; j < n; ++j) {
        detail::tighten_discrete_bounds(out.problem.variable_types[j], &out.problem.lower_bounds(j),
                                        &out.problem.upper_bounds(j), tol);
        if (out.problem.upper_bounds(j) + tol < out.problem.lower_bounds(j)) {
            out.infeasible = true;
            return out;
        }
    }

    detail::RootPresolveContext context;
    context.problem = &out.problem;
    context.result = &out;
    context.tol = tol;
    context.max_passes = max_passes;

    for (int pass = 0; pass < std::max(1, max_passes); ++pass) {
        bool changed = false;

        changed |= detail::run_fast_root_passes(&context);
        if (out.infeasible)
            return out;

        const std::vector<detail::ComponentInfo> components =
            detail::detect_components(out.problem);
        out.detected_components =
            std::max(out.detected_components, static_cast<int>(components.size()));

        changed |= detail::run_medium_root_passes(&context);
        if (out.infeasible)
            return out;

        changed |= detail::run_exhaustive_root_passes(&context);
        if (out.infeasible)
            return out;

        if (!changed)
            break;
    }

    return out;
}

} // namespace simplex::bnb::presolve
