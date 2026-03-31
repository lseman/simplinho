#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simplex/bnb.h"
#include "simplex/simplex.h"

namespace py = pybind11;
namespace simplex_bnb = simplex::bnb;

namespace {

constexpr double kCoeffTol = 1e-12;

enum class ConstraintSense { LessEqual, Equal, GreaterEqual };
using VarType = simplex_bnb::VariableType;
using MIPStatus = simplex_bnb::Status;
using NodeSelectionStrategy = simplex_bnb::NodeSelectionStrategy;
using BranchingStrategy = simplex_bnb::BranchingStrategy;
using DivingStrategy = simplex_bnb::DivingStrategy;
using BranchAndBoundOptions = simplex_bnb::Options;
using MIPTreeNode = simplex_bnb::TreeNode;
using MIPTreeNodeStatus = simplex_bnb::TreeNodeStatus;

struct LinearExprData {
    std::unordered_map<int, double> coeffs;
    double constant = 0.0;
};

struct ModelState;

struct VarData {
    std::uint64_t id = 0;
    std::string name;
    double lb = 0.0;
    double ub = std::numeric_limits<double>::infinity();
    VarType type = VarType::Continuous;
};

struct ConstraintData {
    std::uint64_t id = 0;
    LinearExprData expr;
    ConstraintSense sense = ConstraintSense::Equal;
    std::string name;
};

struct ModelState {
    RevisedSimplexOptions options;
    std::vector<VarData> vars;
    std::unordered_map<std::string, int> name_to_index;
    std::vector<ConstraintData> constraints;
    LinearExprData objective;
    bool maximize = false;
    std::vector<double> last_constraint_pi;
    std::optional<LPBasis> last_basis;
    std::uint64_t revision = 0;
    std::uint64_t solved_revision = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t next_var_id = 1;
    std::uint64_t next_constraint_id = 1;
};

class Var;
class LinearExpr;
class ConstraintSpec;
class ConstraintHandle;
class Model;
class ModelSolution;
class MIPSolution;

std::pair<double, double> canonicalize_var_bounds(VarType type, double lb, double ub) {
    if (type == VarType::Binary && !std::isfinite(ub)) {
        ub = 1.0;
    }
    if (std::isfinite(lb) && std::isfinite(ub) && ub < lb) {
        throw std::invalid_argument("simplex: variable upper bound cannot be below lower bound");
    }
    if (type == VarType::Binary) {
        if (lb < -kCoeffTol || ub > 1.0 + kCoeffTol) {
            throw std::invalid_argument(
                "simplex: binary variables must satisfy 0 <= lb <= ub <= 1");
        }
        lb = std::max(0.0, lb);
        ub = std::min(1.0, ub);
    }
    return {lb, ub};
}

double normalized_coeff(double value) {
    return std::abs(value) <= kCoeffTol ? 0.0 : value;
}

void add_coeff(LinearExprData& data, int index, double delta) {
    delta = normalized_coeff(delta);
    if (delta == 0.0) {
        return;
    }

    const auto it = data.coeffs.find(index);
    if (it == data.coeffs.end()) {
        data.coeffs.emplace(index, delta);
        return;
    }

    const double updated = normalized_coeff(it->second + delta);
    if (updated == 0.0) {
        data.coeffs.erase(it);
    } else {
        it->second = updated;
    }
}

void set_coeff_value(LinearExprData& data, int index, double value) {
    value = normalized_coeff(value);
    if (value == 0.0) {
        data.coeffs.erase(index);
        return;
    }
    data.coeffs[index] = value;
}

void erase_and_reindex_coeffs(LinearExprData& data, int removed_index) {
    std::unordered_map<int, double> updated;
    updated.reserve(data.coeffs.size());
    for (const auto& [index, coeff] : data.coeffs) {
        if (index == removed_index) {
            continue;
        }
        updated.emplace(index > removed_index ? index - 1 : index, coeff);
    }
    data.coeffs = std::move(updated);
}

std::shared_ptr<ModelState> merge_model_state(
    const std::shared_ptr<ModelState>& lhs,
    const std::shared_ptr<ModelState>& rhs,
    const char* context) {
    if (lhs && rhs && lhs.get() != rhs.get()) {
        throw std::invalid_argument(
            std::string("simplex: cannot combine objects from different models in ") +
            context);
    }
    return lhs ? lhs : rhs;
}

LinearExprData add_expr_data(const LinearExprData& lhs, const LinearExprData& rhs,
                             double rhs_scale = 1.0) {
    LinearExprData out;
    out.constant = lhs.constant + rhs_scale * rhs.constant;
    out.coeffs = lhs.coeffs;
    for (const auto& [index, coeff] : rhs.coeffs) {
        add_coeff(out, index, rhs_scale * coeff);
    }
    return out;
}

LinearExprData scale_expr_data(const LinearExprData& expr, double scale) {
    LinearExprData out;
    out.constant = expr.constant * scale;
    for (const auto& [index, coeff] : expr.coeffs) {
        add_coeff(out, index, coeff * scale);
    }
    return out;
}

std::string format_number(double value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

std::string format_var_name(const std::shared_ptr<ModelState>& state, int index) {
    if (!state || index < 0 || index >= static_cast<int>(state->vars.size())) {
        return "x" + std::to_string(index);
    }
    return state->vars[index].name;
}

std::string expr_repr(const LinearExprData& data,
                      const std::shared_ptr<ModelState>& state) {
    std::vector<std::pair<int, double>> ordered(data.coeffs.begin(), data.coeffs.end());
    std::sort(ordered.begin(), ordered.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::ostringstream oss;
    bool first = true;

    auto append_term = [&](double coeff, const std::string& term) {
        const bool negative = coeff < 0.0;
        const double magnitude = std::abs(coeff);
        if (!first) {
            oss << (negative ? " - " : " + ");
        } else if (negative) {
            oss << "-";
        }

        if (std::abs(magnitude - 1.0) > kCoeffTol) {
            oss << format_number(magnitude) << "*";
        }
        oss << term;
        first = false;
    };

    for (const auto& [index, coeff] : ordered) {
        append_term(coeff, format_var_name(state, index));
    }

    if (std::abs(data.constant) > kCoeffTol || first) {
        if (!first) {
            oss << (data.constant < 0.0 ? " - " : " + ");
            oss << format_number(std::abs(data.constant));
        } else {
            oss << format_number(data.constant);
        }
    }

    return oss.str();
}

std::string join_trace_lines(const std::vector<std::string>& trace) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < trace.size(); ++i) {
        if (i > 0) {
            oss << '\n';
        }
        oss << trace[i];
    }
    return oss.str();
}

std::optional<std::string> find_info_string(
    const std::unordered_map<std::string, std::string>& info,
    const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<int> find_info_int(
    const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    try {
        return std::stoi(it->second);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<double> find_info_double(
    const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    try {
        return std::stod(it->second);
    } catch (...) {
        return std::nullopt;
    }
}

std::optional<bool> find_info_bool(
    const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    if (it->second == "1" || it->second == "true" || it->second == "True") {
        return true;
    }
    if (it->second == "0" || it->second == "false" || it->second == "False") {
        return false;
    }
    return std::nullopt;
}

LPBasis parse_basis_state_from_info(
    const std::unordered_map<std::string, std::string>& info,
    const LPBasis& fallback = LPBasis{}) {
    const auto it = info.find("warm_start_basis_state");
    if (it == info.end()) {
        return fallback;
    }
    LPBasis out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        const int value = std::stoi(tok);
        switch (value) {
            case 0:
                out.column_status.push_back(LPBasisStatus::Basic);
                break;
            case 1:
                out.column_status.push_back(LPBasisStatus::AtLower);
                break;
            case 2:
                out.column_status.push_back(LPBasisStatus::AtUpper);
                break;
            case 3:
                out.column_status.push_back(LPBasisStatus::Fixed);
                break;
            default:
                out.column_status.push_back(LPBasisStatus::AtLower);
                break;
        }
    }
    return out;
}

std::optional<std::vector<double>> parse_double_list_from_info(
    const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) return std::nullopt;
    std::vector<double> out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stod(tok));
    }
    return out;
}

LPBasis rebuild_basis_from_solution(const LPSolution& sol) {
    const auto maybe_l = parse_double_list_from_info(sol.info, "original_l");
    const auto maybe_u = parse_double_list_from_info(sol.info, "original_u");
    const auto maybe_m = find_info_int(sol.info, "original_m");
    if (!maybe_l || !maybe_u || !maybe_m) {
        return parse_basis_state_from_info(sol.info, sol.basis_state);
    }
    if (sol.x.size() != static_cast<int>(maybe_l->size()) ||
        sol.x.size() != static_cast<int>(maybe_u->size())) {
        return parse_basis_state_from_info(sol.info, sol.basis_state);
    }

    std::vector<int> status(sol.x.size(), 1);
    std::vector<char> eligible(sol.x.size(), 1);
    const double tol = 1e-8;
    for (int j = 0; j < sol.x.size(); ++j) {
        const double x = sol.x(j);
        const double l = (*maybe_l)[j];
        const double u = (*maybe_u)[j];
        const bool has_l = std::isfinite(l);
        const bool has_u = std::isfinite(u);
        const bool fixed = has_l && has_u && std::abs(u - l) <= tol;
        if (fixed) {
            status[j] = 3;
            eligible[j] = 0;
            continue;
        }
        const bool near_l = has_l && std::abs(x - l) <= tol;
        const bool near_u = has_u && std::abs(x - u) <= tol;
        if (near_u && !near_l) {
            status[j] = 2;
        } else if (near_l) {
            status[j] = 1;
        } else if (has_u && !has_l) {
            status[j] = 2;
        } else if (has_l && has_u) {
            status[j] = (std::abs(x - u) + tol < std::abs(x - l)) ? 2 : 1;
        } else {
            status[j] = 1;
        }
    }

    const int target = *maybe_m;
    std::vector<char> chosen(sol.x.size(), 0);
    auto choose_if = [&](int j) {
        if (j < 0 || j >= sol.x.size() || chosen[j] || !eligible[j]) return false;
        chosen[j] = 1;
        status[j] = 0;
        return true;
    };

    int chosen_count = 0;
    for (int j : sol.basis) {
        if (chosen_count == target) break;
        if (j < 0 || j >= sol.x.size()) continue;
        const double l = (*maybe_l)[j];
        const double u = (*maybe_u)[j];
        const bool has_l = std::isfinite(l);
        const bool has_u = std::isfinite(u);
        const bool near_l = has_l && std::abs(sol.x(j) - l) <= tol;
        const bool near_u = has_u && std::abs(sol.x(j) - u) <= tol;
        if (!near_l && !near_u && choose_if(j)) ++chosen_count;
    }
    for (int j = 0; j < sol.x.size() && chosen_count < target; ++j) {
        const double l = (*maybe_l)[j];
        const double u = (*maybe_u)[j];
        const bool has_l = std::isfinite(l);
        const bool has_u = std::isfinite(u);
        const bool near_l = has_l && std::abs(sol.x(j) - l) <= tol;
        const bool near_u = has_u && std::abs(sol.x(j) - u) <= tol;
        if (!near_l && !near_u && choose_if(j)) ++chosen_count;
    }
    for (int j : sol.basis) {
        if (chosen_count == target) break;
        if (choose_if(j)) ++chosen_count;
    }
    for (int j = 0; j < sol.x.size() && chosen_count < target; ++j) {
        if (choose_if(j)) ++chosen_count;
    }

    LPBasis out;
    out.column_status.reserve(status.size());
    for (const int value : status) {
        switch (value) {
            case 0:
                out.column_status.push_back(LPBasisStatus::Basic);
                break;
            case 2:
                out.column_status.push_back(LPBasisStatus::AtUpper);
                break;
            case 3:
                out.column_status.push_back(LPBasisStatus::Fixed);
                break;
            case 1:
            default:
                out.column_status.push_back(LPBasisStatus::AtLower);
                break;
        }
    }
    return out;
}

bool basis_matches_dimensions(const LPBasis& basis, int columns, int rows) {
    if (basis.column_status.size() != static_cast<std::size_t>(columns)) {
        return false;
    }
    int basic_count = 0;
    for (const auto status : basis.column_status) {
        if (status == LPBasisStatus::Basic) ++basic_count;
    }
    return basic_count == rows;
}

simplex_bnb::LinearConstraintSense to_bnb_sense(ConstraintSense sense) {
    switch (sense) {
        case ConstraintSense::LessEqual:
            return simplex_bnb::LinearConstraintSense::LessEqual;
        case ConstraintSense::GreaterEqual:
            return simplex_bnb::LinearConstraintSense::GreaterEqual;
        case ConstraintSense::Equal:
            return simplex_bnb::LinearConstraintSense::Equal;
    }
    return simplex_bnb::LinearConstraintSense::Equal;
}

struct SimplifiedCutsResult {
    bool infeasible = false;
    std::vector<simplex_bnb::Cut> cuts;
};

struct NodeBoundPresolveResult {
    bool infeasible = false;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    int tightened_bounds = 0;
};

struct SparseActivityRange {
    double min_activity = 0.0;
    double max_activity = 0.0;
    bool min_finite = true;
    bool max_finite = true;
};

struct SparseRowView {
    const std::vector<int>* indices = nullptr;
    const std::vector<double>* values = nullptr;
    simplex_bnb::LinearConstraintSense sense = simplex_bnb::LinearConstraintSense::Equal;
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

struct RootProblemPresolveResult {
    simplex_bnb::Problem problem;
    bool infeasible = false;
    int tightened_bounds = 0;
    int removed_rows = 0;
    int removed_coeffs = 0;
    int aggregations = 0;
};

void tighten_discrete_bounds(VarType type, double* lower, double* upper, double tol);

SparseVariableContribution sparse_variable_contribution(double coeff, int index,
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
        if (contribution.min_finite) contribution.min_value = coeff * lo;
        if (contribution.max_finite) contribution.max_value = coeff * up;
    } else {
        contribution.min_finite = std::isfinite(up);
        contribution.max_finite = std::isfinite(lo);
        if (contribution.min_finite) contribution.min_value = coeff * up;
        if (contribution.max_finite) contribution.max_value = coeff * lo;
    }

    return contribution;
}

SparseRowActivitySummary sparse_row_activity_summary(const SparseRowView& row,
                                                     const Eigen::VectorXd& lower,
                                                     const Eigen::VectorXd& upper) {
    SparseRowActivitySummary summary;
    if (!row.indices || !row.values) {
        return summary;
    }

    for (int k = 0; k < static_cast<int>(row.indices->size()) &&
                    k < static_cast<int>(row.values->size());
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

bool sparse_row_is_feasible(const SparseRowActivitySummary& summary,
                            simplex_bnb::LinearConstraintSense sense, double rhs, double tol) {
    switch (sense) {
        case simplex_bnb::LinearConstraintSense::LessEqual:
            return summary.min_infinite_terms > 0 || summary.min_activity <= rhs + tol;
        case simplex_bnb::LinearConstraintSense::GreaterEqual:
            return summary.max_infinite_terms > 0 || summary.max_activity >= rhs - tol;
        case simplex_bnb::LinearConstraintSense::Equal:
            return (summary.min_infinite_terms > 0 || summary.min_activity <= rhs + tol) &&
                   (summary.max_infinite_terms > 0 || summary.max_activity >= rhs - tol);
    }
    return true;
}

bool sparse_row_is_redundant(const SparseRowActivitySummary& summary,
                             simplex_bnb::LinearConstraintSense sense, double rhs, double tol) {
    switch (sense) {
        case simplex_bnb::LinearConstraintSense::LessEqual:
            return summary.max_infinite_terms == 0 && summary.max_activity <= rhs + tol;
        case simplex_bnb::LinearConstraintSense::GreaterEqual:
            return summary.min_infinite_terms == 0 && summary.min_activity >= rhs - tol;
        case simplex_bnb::LinearConstraintSense::Equal:
            return summary.min_infinite_terms == 0 && summary.max_infinite_terms == 0 &&
                   summary.min_activity >= rhs - tol && summary.max_activity <= rhs + tol;
    }
    return false;
}

std::string sparse_row_signature(const simplex_bnb::SparseLinearConstraint& row,
                                 int precision = 12) {
    const double scale = std::pow(10.0, precision);
    std::ostringstream oss;
    for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                    k < static_cast<int>(row.values.size());
         ++k) {
        const double coeff = row.values[k];
        if (std::abs(coeff) <= kCoeffTol) continue;
        const double rounded = std::round(coeff * scale) / scale;
        oss << row.indices[k] << ":" << rounded << ";";
    }
    const double rounded_rhs = std::round(row.rhs * scale) / scale;
    oss << "|rhs:" << rounded_rhs << "|sense:" << static_cast<int>(row.sense);
    return oss.str();
}

void canonicalize_sparse_row(simplex_bnb::SparseLinearConstraint* row,
                             const Eigen::VectorXd& lower, const Eigen::VectorXd& upper,
                             int* removed_coeffs, double tol) {
    if (!row) return;

    std::vector<std::pair<int, double>> terms;
    terms.reserve(std::min(row->indices.size(), row->values.size()));
    double rhs = row->rhs;

    for (int k = 0; k < static_cast<int>(row->indices.size()) &&
                    k < static_cast<int>(row->values.size());
         ++k) {
        const int index = row->indices[k];
        const double coeff = row->values[k];
        if (index < 0 || index >= lower.size() || index >= upper.size() ||
            std::abs(coeff) <= kCoeffTol) {
            if (removed_coeffs) ++(*removed_coeffs);
            continue;
        }

        if (std::isfinite(lower(index)) && std::isfinite(upper(index)) &&
            std::abs(lower(index) - upper(index)) <= tol) {
            rhs -= coeff * lower(index);
            if (removed_coeffs) ++(*removed_coeffs);
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
                if (removed_coeffs) ++(*removed_coeffs);
            } else {
                row->values.back() = merged;
            }
            continue;
        }
        row->indices.push_back(index);
        row->values.push_back(coeff);
    }
}

std::optional<int> find_row_coefficient_position(const simplex_bnb::SparseLinearConstraint& row,
                                                 int index) {
    for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                    k < static_cast<int>(row.values.size());
         ++k) {
        if (row.indices[k] == index) return k;
    }
    return std::nullopt;
}

std::optional<std::pair<double, double>> implied_interval_for_equality_pivot(
    const simplex_bnb::SparseLinearConstraint& row, int pivot, const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper, double tol) {
    const auto pivot_pos = find_row_coefficient_position(row, pivot);
    if (!pivot_pos.has_value()) return std::nullopt;

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
        summary.min_activity -
        (pivot_contribution.min_finite ? pivot_contribution.min_value : 0.0);
    const double other_max =
        summary.max_activity -
        (pivot_contribution.max_finite ? pivot_contribution.max_value : 0.0);

    double implied_lower = 0.0;
    double implied_upper = 0.0;
    if (aij > 0.0) {
        implied_lower = (row.rhs - other_max) / aij;
        implied_upper = (row.rhs - other_min) / aij;
    } else {
        implied_lower = (row.rhs - other_min) / aij;
        implied_upper = (row.rhs - other_max) / aij;
    }
    if (implied_lower > implied_upper) std::swap(implied_lower, implied_upper);
    return std::make_pair(implied_lower, implied_upper);
}

bool try_aggregate_implied_free_continuous_variable(
    simplex_bnb::Problem* problem, int row_index, double tol, int* removed_coeffs,
    int* aggregation_count) {
    if (!problem || row_index < 0 ||
        row_index >= static_cast<int>(problem->base_constraints.size())) {
        return false;
    }

    auto& defining_row = problem->base_constraints[row_index];
    if (defining_row.sense != simplex_bnb::LinearConstraintSense::Equal) {
        return false;
    }
    const int row_nnz =
        std::min(static_cast<int>(defining_row.indices.size()),
                 static_cast<int>(defining_row.values.size()));
    if (row_nnz < 2 || row_nnz > 8) {
        return false;
    }

    std::vector<std::vector<int>> col_to_rows(problem->lower_bounds.size());
    std::vector<double> col_max_abs(problem->lower_bounds.size(), 0.0);
    for (int r = 0; r < static_cast<int>(problem->base_constraints.size()); ++r) {
        const auto& row = problem->base_constraints[r];
        for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                        k < static_cast<int>(row.values.size());
             ++k) {
            const int index = row.indices[k];
            if (index < 0 || index >= problem->lower_bounds.size()) continue;
            if (std::abs(row.values[k]) <= kCoeffTol) continue;
            col_to_rows[index].push_back(r);
            col_max_abs[index] = std::max(col_max_abs[index], std::abs(row.values[k]));
        }
    }

    double row_max_abs = 0.0;
    for (int k = 0; k < row_nnz; ++k) {
        row_max_abs = std::max(row_max_abs, std::abs(defining_row.values[k]));
    }

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
        if (pivot < 0 || pivot >= problem->lower_bounds.size()) continue;
        if (problem->variable_types[pivot] != VarType::Continuous) continue;
        if (std::abs(aij) <= kCoeffTol) continue;

        const auto implied = implied_interval_for_equality_pivot(
            defining_row, pivot, problem->lower_bounds, problem->upper_bounds, tol);
        if (!implied.has_value()) continue;

        const double explicit_l = problem->lower_bounds(pivot);
        const double explicit_u = problem->upper_bounds(pivot);
        if ((std::isfinite(explicit_l) && implied->first < explicit_l - tol) ||
            (std::isfinite(explicit_u) && implied->second > explicit_u + tol)) {
            continue;
        }

        const int col_nnz = static_cast<int>(col_to_rows[pivot].size());
        if (col_nnz <= 1 || col_nnz > 4) continue;

        const double coeff_abs = std::abs(aij);
        if (coeff_abs < 0.01 * row_max_abs || coeff_abs < 0.01 * col_max_abs[pivot]) {
            continue;
        }

        const int estimated_fill = (col_nnz - 1) * std::max(0, row_nnz - 2);
        if (estimated_fill > 12) continue;

        Candidate candidate{pivot, k, col_nnz, estimated_fill, coeff_abs};
        if (!best.has_value() || candidate.estimated_fill < best->estimated_fill ||
            (candidate.estimated_fill == best->estimated_fill &&
             candidate.col_nnz < best->col_nnz) ||
            (candidate.estimated_fill == best->estimated_fill &&
             candidate.col_nnz == best->col_nnz && candidate.coeff_abs > best->coeff_abs)) {
            best = candidate;
        }
    }

    if (!best.has_value()) return false;

    const int pivot = best->pivot;
    const int pivot_pos = best->pivot_pos;
    const double pivot_coeff = defining_row.values[pivot_pos];
    const double objective_coeff =
        pivot < problem->objective_coefficients.size() ? problem->objective_coefficients(pivot) : 0.0;

    if (std::abs(objective_coeff) > kCoeffTol) {
        problem->objective_constant += objective_coeff * defining_row.rhs / pivot_coeff;
        for (int k = 0; k < row_nnz; ++k) {
            if (k == pivot_pos) continue;
            const int index = defining_row.indices[k];
            if (index < 0 || index >= problem->objective_coefficients.size()) continue;
            problem->objective_coefficients(index) -=
                objective_coeff * defining_row.values[k] / pivot_coeff;
        }
        problem->objective_coefficients(pivot) = 0.0;
    }

    const auto affected_rows = col_to_rows[pivot];
    for (const int other_row_index : affected_rows) {
        if (other_row_index == row_index) continue;
        auto& other_row = problem->base_constraints[other_row_index];
        const auto other_pivot_pos = find_row_coefficient_position(other_row, pivot);
        if (!other_pivot_pos.has_value()) continue;
        const double factor = other_row.values[*other_pivot_pos] / pivot_coeff;
        if (std::abs(factor) <= kCoeffTol) continue;

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

    if (aggregation_count) ++(*aggregation_count);
    return true;
}

void update_row_summary_for_bound_change(const SparseVariableContribution& old_contribution,
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

bool tighten_bounds_from_sparse_row(const SparseRowView& row, const simplex_bnb::Problem& problem,
                                    Eigen::VectorXd* lower, Eigen::VectorXd* upper,
                                    int* tightened_bounds,
                                    const std::vector<std::vector<int>>& col_to_rows,
                                    std::vector<char>* next_dirty_rows, double tol) {
    SparseRowActivitySummary summary = sparse_row_activity_summary(row, *lower, *upper);
    if (!sparse_row_is_feasible(summary, row.sense, row.rhs, tol)) {
        return false;
    }
    if (!row.indices || !row.values) {
        return true;
    }

    for (int k = 0; k < static_cast<int>(row.indices->size()) &&
                    k < static_cast<int>(row.values->size());
         ++k) {
        const int index = (*row.indices)[k];
        if (index < 0 || index >= lower->size() || index >= upper->size()) {
            continue;
        }

        const double coeff = (*row.values)[k];
        if (std::abs(coeff) <= kCoeffTol) {
            continue;
        }

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
            if (std::isfinite(candidate) && candidate < tightened_upper - tol) {
                tightened_upper = candidate;
            }
        };
        const auto apply_lower = [&](double candidate) {
            if (std::isfinite(candidate) && candidate > tightened_lower + tol) {
                tightened_lower = candidate;
            }
        };

        switch (row.sense) {
            case simplex_bnb::LinearConstraintSense::LessEqual:
                if (coeff > 0.0 && other_min_finite) {
                    apply_upper((row.rhs - other_min_activity) / coeff);
                } else if (coeff < 0.0 && other_min_finite) {
                    apply_lower((row.rhs - other_min_activity) / coeff);
                }
                break;
            case simplex_bnb::LinearConstraintSense::GreaterEqual:
                if (coeff > 0.0 && other_max_finite) {
                    apply_lower((row.rhs - other_max_activity) / coeff);
                } else if (coeff < 0.0 && other_max_finite) {
                    apply_upper((row.rhs - other_max_activity) / coeff);
                }
                break;
            case simplex_bnb::LinearConstraintSense::Equal:
                if (coeff > 0.0) {
                    if (other_min_finite) {
                        apply_upper((row.rhs - other_min_activity) / coeff);
                    }
                    if (other_max_finite) {
                        apply_lower((row.rhs - other_max_activity) / coeff);
                    }
                } else {
                    if (other_min_finite) {
                        apply_lower((row.rhs - other_min_activity) / coeff);
                    }
                    if (other_max_finite) {
                        apply_upper((row.rhs - other_max_activity) / coeff);
                    }
                }
                break;
        }

        tighten_discrete_bounds(problem.variable_types[index], &tightened_lower, &tightened_upper,
                                tol);
        if (tightened_upper + tol < tightened_lower) {
            return false;
        }

        const bool lower_changed = tightened_lower > (*lower)(index) + tol;
        const bool upper_changed = tightened_upper < (*upper)(index) - tol;
        if (!lower_changed && !upper_changed) {
            continue;
        }

        (*lower)(index) = tightened_lower;
        (*upper)(index) = tightened_upper;
        if (lower_changed) ++(*tightened_bounds);
        if (upper_changed) ++(*tightened_bounds);

        const SparseVariableContribution new_contribution =
            sparse_variable_contribution(coeff, index, *lower, *upper);
        update_row_summary_for_bound_change(old_contribution, new_contribution, &summary);

        if (next_dirty_rows && index >= 0 && index < static_cast<int>(col_to_rows.size())) {
            for (const int affected_row : col_to_rows[index]) {
                if (affected_row >= 0 &&
                    affected_row < static_cast<int>(next_dirty_rows->size())) {
                    (*next_dirty_rows)[affected_row] = 1;
                }
            }
        }
    }

    return true;
}

void tighten_discrete_bounds(VarType type, double* lower, double* upper, double tol) {
    if (type == VarType::Continuous) {
        return;
    }

    if (type == VarType::Binary) {
        *lower = std::max(*lower, 0.0);
        *upper = std::min(*upper, 1.0);
    }

    if (std::isfinite(*lower)) {
        *lower = std::ceil(*lower - tol);
    }
    if (std::isfinite(*upper)) {
        *upper = std::floor(*upper + tol);
    }
}

NodeBoundPresolveResult presolve_mip_node_bounds(
    const simplex_bnb::Problem& problem, const Eigen::VectorXd& lower_in,
    const Eigen::VectorXd& upper_in,
    const std::vector<simplex_bnb::Cut>& extra_cuts = {}, double tol = 1e-9,
    int max_passes = 2) {
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
        tighten_discrete_bounds(problem.variable_types[j], &out.lower(j), &out.upper(j), tol);
        if (out.upper(j) + tol < out.lower(j)) {
            out.infeasible = true;
            return out;
        }
    }

    std::vector<SparseRowView> rows;
    rows.reserve(problem.base_constraints.size() + extra_cuts.size());
    std::vector<std::vector<int>> col_to_rows(n);

    const auto add_row = [&](const auto& source_row) {
        const int row_index = static_cast<int>(rows.size());
        rows.push_back(SparseRowView{&source_row.indices, &source_row.values, source_row.sense,
                                     source_row.rhs});
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

    for (const auto& row : problem.base_constraints) add_row(row);
    for (const auto& cut : extra_cuts) add_row(cut);

    std::vector<char> dirty_rows(rows.size(), 1);
    const int propagation_rounds = std::max(2, 2 * std::max(1, max_passes));
    for (int round = 0; round < propagation_rounds; ++round) {
        bool any_dirty = false;
        bool changed = false;
        std::vector<char> next_dirty_rows(rows.size(), 0);

        for (int row_index = 0; row_index < static_cast<int>(rows.size()); ++row_index) {
            if (!dirty_rows[row_index]) continue;
            any_dirty = true;

            const int tightened_before = out.tightened_bounds;
            if (!tighten_bounds_from_sparse_row(rows[row_index], problem, &out.lower, &out.upper,
                                                &out.tightened_bounds, col_to_rows,
                                                &next_dirty_rows, tol)) {
                out.infeasible = true;
                return out;
            }
            if (out.tightened_bounds != tightened_before) {
                changed = true;
            }
        }

        if (!any_dirty || !changed) {
            break;
        }
        dirty_rows = std::move(next_dirty_rows);
    }

    return out;
}

RootProblemPresolveResult presolve_mip_root_problem(const simplex_bnb::Problem& input,
                                                    double tol = 1e-9, int max_passes = 4) {
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
        tighten_discrete_bounds(out.problem.variable_types[j], &out.problem.lower_bounds(j),
                                &out.problem.upper_bounds(j), tol);
        if (out.problem.upper_bounds(j) + tol < out.problem.lower_bounds(j)) {
            out.infeasible = true;
            return out;
        }
    }

    for (int pass = 0; pass < std::max(1, max_passes); ++pass) {
        bool changed = false;
        std::vector<simplex_bnb::SparseLinearConstraint> kept_rows;
        kept_rows.reserve(out.problem.base_constraints.size());
        std::unordered_map<std::string, int> seen_rows;

        for (const auto& base_row : out.problem.base_constraints) {
            simplex_bnb::SparseLinearConstraint row = base_row;
            canonicalize_sparse_row(&row, out.problem.lower_bounds, out.problem.upper_bounds,
                                    &out.removed_coeffs, tol);

            const SparseRowView view{&row.indices, &row.values, row.sense, row.rhs};
            const SparseRowActivitySummary summary =
                sparse_row_activity_summary(view, out.problem.lower_bounds, out.problem.upper_bounds);
            if (!sparse_row_is_feasible(summary, row.sense, row.rhs, tol)) {
                out.infeasible = true;
                return out;
            }
            if (sparse_row_is_redundant(summary, row.sense, row.rhs, tol)) {
                ++out.removed_rows;
                changed = true;
                continue;
            }

            const std::string signature = sparse_row_signature(row);
            if (seen_rows.contains(signature)) {
                ++out.removed_rows;
                changed = true;
                continue;
            }
            seen_rows.emplace(signature, static_cast<int>(kept_rows.size()));
            kept_rows.push_back(std::move(row));
        }

        out.problem.base_constraints = std::move(kept_rows);

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

        bool aggregated = false;
        for (int row_index = 0; row_index < static_cast<int>(out.problem.base_constraints.size());
             ++row_index) {
            if (try_aggregate_implied_free_continuous_variable(
                    &out.problem, row_index, tol, &out.removed_coeffs, &out.aggregations)) {
                aggregated = true;
                changed = true;
                break;
            }
        }

        if (!changed) break;
    }

    return out;
}

double cut_activity_bound(const simplex_bnb::Cut& cut, const Eigen::VectorXd& lower,
                          const Eigen::VectorXd& upper, bool use_upper) {
    double activity = 0.0;
    for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                    k < static_cast<int>(cut.values.size());
         ++k) {
        const int index = cut.indices[k];
        if (index < 0 || index >= lower.size() || index >= upper.size()) {
            continue;
        }
        const double coeff = cut.values[k];
        const bool take_upper = use_upper ? (coeff >= 0.0) : (coeff < 0.0);
        activity += coeff * (take_upper ? upper(index) : lower(index));
    }
    return activity;
}

SimplifiedCutsResult simplify_cuts_for_bounds(
    const std::vector<simplex_bnb::Cut>& cuts, const Eigen::VectorXd& lower,
    const Eigen::VectorXd& upper, double tol = 1e-9) {
    SimplifiedCutsResult out;
    out.cuts.reserve(cuts.size());

    for (const auto& cut : cuts) {
        const double min_activity = cut_activity_bound(cut, lower, upper, false);
        const double max_activity = cut_activity_bound(cut, lower, upper, true);

        bool redundant = false;
        bool infeasible = false;
        switch (cut.sense) {
            case simplex_bnb::LinearConstraintSense::LessEqual:
                redundant = max_activity <= cut.rhs + tol;
                infeasible = min_activity > cut.rhs + tol;
                break;
            case simplex_bnb::LinearConstraintSense::GreaterEqual:
                redundant = min_activity >= cut.rhs - tol;
                infeasible = max_activity < cut.rhs - tol;
                break;
            case simplex_bnb::LinearConstraintSense::Equal:
                redundant = max_activity <= cut.rhs + tol && min_activity >= cut.rhs - tol;
                infeasible = max_activity < cut.rhs - tol || min_activity > cut.rhs + tol;
                break;
        }

        if (infeasible) {
            out.infeasible = true;
            out.cuts.clear();
            return out;
        }
        if (redundant) {
            continue;
        }

        simplex_bnb::Cut simplified;
        simplified.sense = cut.sense;
        simplified.rhs = cut.rhs;
        simplified.cut_type = cut.cut_type;
        simplified.strength = cut.strength;
        simplified.times_used = cut.times_used;
        simplified.age = cut.age;

        for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                        k < static_cast<int>(cut.values.size());
             ++k) {
            const int index = cut.indices[k];
            if (index < 0 || index >= lower.size() || index >= upper.size()) {
                continue;
            }
            const double coeff = cut.values[k];
            if (std::abs(coeff) <= kCoeffTol) {
                continue;
            }
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
                case simplex_bnb::LinearConstraintSense::LessEqual:
                    scalar_redundant = 0.0 <= simplified.rhs + tol;
                    scalar_infeasible = 0.0 > simplified.rhs + tol;
                    break;
                case simplex_bnb::LinearConstraintSense::GreaterEqual:
                    scalar_redundant = 0.0 >= simplified.rhs - tol;
                    scalar_infeasible = 0.0 < simplified.rhs - tol;
                    break;
                case simplex_bnb::LinearConstraintSense::Equal:
                    scalar_redundant = std::abs(simplified.rhs) <= tol;
                    scalar_infeasible = !scalar_redundant;
                    break;
            }
            if (scalar_infeasible) {
                out.infeasible = true;
                out.cuts.clear();
                return out;
            }
            if (scalar_redundant) {
                continue;
            }
        }

        out.cuts.push_back(std::move(simplified));
    }

    return out;
}

std::string cut_set_signature(const std::vector<simplex_bnb::Cut>& cuts) {
    std::ostringstream oss;
    for (const auto& cut : cuts) {
        oss << simplex_bnb::detail::cut_signature(cut) << '\n';
    }
    return oss.str();
}

struct SolveStats {
    std::string status;
    int iterations = 0;
    int phase2_iterations = 0;
    std::optional<int> phase1_iterations;
    std::optional<int> presolve_actions;
    std::optional<int> presolve_implied_bound_updates;
    std::optional<int> reduced_rows;
    std::optional<int> reduced_cols;
    std::optional<double> objective_shift;
    std::optional<int> input_upper_bounds_relaxed;
    std::optional<int> input_lower_bounds_relaxed;
    std::optional<std::string> basis_start;
    std::optional<std::string> basis_start_style;
    std::optional<int> basis_start_attempt;
    std::optional<bool> basis_start_primal_feasible;
    std::optional<bool> basis_start_dual_feasible;
    std::optional<double> basis_start_primal_violation;
    std::optional<double> basis_start_dual_violation;
    std::optional<std::string> phase1_status;
    std::optional<std::string> reason;
    std::optional<std::string> note;
    std::optional<std::string> certificate;
    std::optional<std::string> dual_pricing;
    std::optional<int> dual_bfrt_flips;
    std::optional<int> degeneracy_streak;
    std::optional<int> degeneracy_total;
    std::optional<int> suspected_cycle_length;
    std::optional<double> condition_estimate;
    std::optional<double> degeneracy_threshold;
    std::optional<int> degeneracy_epoch;
    bool farkas_has_cert = false;
    bool primal_ray_has_cert = false;
    int trace_lines = 0;
    std::unordered_map<std::string, std::string> raw_info;

    py::dict as_dict() const {
        py::dict out;
        out["status"] = status;
        out["iterations"] = iterations;
        out["phase2_iterations"] = phase2_iterations;
        out["phase1_iterations"] =
            phase1_iterations ? py::cast(*phase1_iterations) : py::none();
        out["presolve_actions"] =
            presolve_actions ? py::cast(*presolve_actions) : py::none();
        out["presolve_implied_bound_updates"] = presolve_implied_bound_updates
                                                   ? py::cast(*presolve_implied_bound_updates)
                                                   : py::none();
        out["reduced_rows"] = reduced_rows ? py::cast(*reduced_rows) : py::none();
        out["reduced_cols"] = reduced_cols ? py::cast(*reduced_cols) : py::none();
        out["objective_shift"] =
            objective_shift ? py::cast(*objective_shift) : py::none();
        out["input_upper_bounds_relaxed"] = input_upper_bounds_relaxed
                                                ? py::cast(*input_upper_bounds_relaxed)
                                                : py::none();
        out["input_lower_bounds_relaxed"] = input_lower_bounds_relaxed
                                                ? py::cast(*input_lower_bounds_relaxed)
                                                : py::none();
        out["basis_start"] = basis_start ? py::cast(*basis_start) : py::none();
        out["basis_start_style"] =
            basis_start_style ? py::cast(*basis_start_style) : py::none();
        out["basis_start_attempt"] =
            basis_start_attempt ? py::cast(*basis_start_attempt) : py::none();
        out["basis_start_primal_feasible"] = basis_start_primal_feasible
                                                 ? py::cast(*basis_start_primal_feasible)
                                                 : py::none();
        out["basis_start_dual_feasible"] = basis_start_dual_feasible
                                               ? py::cast(*basis_start_dual_feasible)
                                               : py::none();
        out["basis_start_primal_violation"] = basis_start_primal_violation
                                                  ? py::cast(*basis_start_primal_violation)
                                                  : py::none();
        out["basis_start_dual_violation"] = basis_start_dual_violation
                                                ? py::cast(*basis_start_dual_violation)
                                                : py::none();
        out["phase1_status"] = phase1_status ? py::cast(*phase1_status) : py::none();
        out["reason"] = reason ? py::cast(*reason) : py::none();
        out["note"] = note ? py::cast(*note) : py::none();
        out["certificate"] = certificate ? py::cast(*certificate) : py::none();
        out["dual_pricing"] = dual_pricing ? py::cast(*dual_pricing) : py::none();
        out["dual_bfrt_flips"] =
            dual_bfrt_flips ? py::cast(*dual_bfrt_flips) : py::none();
        out["degeneracy_streak"] =
            degeneracy_streak ? py::cast(*degeneracy_streak) : py::none();
        out["degeneracy_total"] =
            degeneracy_total ? py::cast(*degeneracy_total) : py::none();
        out["suspected_cycle_length"] =
            suspected_cycle_length ? py::cast(*suspected_cycle_length) : py::none();
        out["condition_estimate"] =
            condition_estimate ? py::cast(*condition_estimate) : py::none();
        out["degeneracy_threshold"] =
            degeneracy_threshold ? py::cast(*degeneracy_threshold) : py::none();
        out["degeneracy_epoch"] =
            degeneracy_epoch ? py::cast(*degeneracy_epoch) : py::none();
        out["farkas_has_cert"] = farkas_has_cert;
        out["primal_ray_has_cert"] = primal_ray_has_cert;
        out["trace_lines"] = trace_lines;
        out["raw_info"] = raw_info;
        return out;
    }
};

SolveStats build_solve_stats(const LPSolution& sol) {
    SolveStats stats;
    stats.status = to_string(sol.status);
    stats.iterations = sol.iters;
    stats.phase1_iterations = find_info_int(sol.info, "phase1_iters");
    stats.phase2_iterations =
        sol.iters - stats.phase1_iterations.value_or(0);
    stats.presolve_actions = find_info_int(sol.info, "presolve_actions");
    stats.presolve_implied_bound_updates =
        find_info_int(sol.info, "presolve_implied_bound_updates");
    stats.reduced_rows = find_info_int(sol.info, "reduced_m");
    stats.reduced_cols = find_info_int(sol.info, "reduced_n");
    stats.objective_shift = find_info_double(sol.info, "obj_shift");
    stats.input_upper_bounds_relaxed =
        find_info_int(sol.info, "input_upper_bounds_relaxed");
    stats.input_lower_bounds_relaxed =
        find_info_int(sol.info, "input_lower_bounds_relaxed");
    stats.basis_start = find_info_string(sol.info, "basis_start");
    stats.basis_start_style = find_info_string(sol.info, "basis_start_style");
    stats.basis_start_attempt = find_info_int(sol.info, "basis_start_attempt");
    stats.basis_start_primal_feasible =
        find_info_bool(sol.info, "basis_start_primal_feasible");
    stats.basis_start_dual_feasible =
        find_info_bool(sol.info, "basis_start_dual_feasible");
    stats.basis_start_primal_violation =
        find_info_double(sol.info, "basis_start_primal_violation");
    stats.basis_start_dual_violation =
        find_info_double(sol.info, "basis_start_dual_violation");
    stats.phase1_status = find_info_string(sol.info, "phase1_status");
    stats.reason = find_info_string(sol.info, "reason");
    stats.note = find_info_string(sol.info, "note");
    stats.certificate = find_info_string(sol.info, "certificate");
    stats.dual_pricing = find_info_string(sol.info, "dual_pricing");
    stats.dual_bfrt_flips = find_info_int(sol.info, "dual_bfrt_flips");
    stats.degeneracy_streak = find_info_int(sol.info, "deg_streak");
    stats.degeneracy_total = find_info_int(sol.info, "deg_total");
    stats.suspected_cycle_length = find_info_int(sol.info, "cycle_len");
    stats.condition_estimate = find_info_double(sol.info, "cond_est");
    stats.degeneracy_threshold = find_info_double(sol.info, "deg_thresh");
    stats.degeneracy_epoch = find_info_int(sol.info, "deg_epoch");
    stats.farkas_has_cert = sol.farkas_has_cert;
    stats.primal_ray_has_cert = sol.primal_ray_has_cert;
    stats.trace_lines = static_cast<int>(sol.trace.size());
    stats.raw_info = sol.info;
    return stats;
}

class Var {
   public:
    Var() = default;

    Var(std::shared_ptr<ModelState> state, int index, std::uint64_t id)
        : state_(std::move(state)), index_(index), id_(id) {}

    const std::shared_ptr<ModelState>& state() const { return state_; }
    int index() const { return resolve_index_("index"); }

    std::string name() const {
        const int index = resolve_index_("name");
        return state_->vars[index].name;
    }

    double lower_bound() const {
        const int index = resolve_index_("lower_bound");
        return state_->vars[index].lb;
    }

    void set_lower_bound(double value) {
        const int index = resolve_index_("set_lower_bound");
        touch_state_();
        auto [lb, ub] =
            canonicalize_var_bounds(state_->vars[index].type, value, state_->vars[index].ub);
        state_->vars[index].lb = lb;
        state_->vars[index].ub = ub;
    }

    double upper_bound() const {
        const int index = resolve_index_("upper_bound");
        return state_->vars[index].ub;
    }

    void set_upper_bound(double value) {
        const int index = resolve_index_("set_upper_bound");
        touch_state_();
        auto [lb, ub] =
            canonicalize_var_bounds(state_->vars[index].type, state_->vars[index].lb, value);
        state_->vars[index].lb = lb;
        state_->vars[index].ub = ub;
    }

    VarType type() const {
        const int index = resolve_index_("type");
        return state_->vars[index].type;
    }

    void set_type(VarType value) {
        const int index = resolve_index_("set_type");
        touch_state_(true);
        auto [lb, ub] =
            canonicalize_var_bounds(value, state_->vars[index].lb, state_->vars[index].ub);
        state_->vars[index].lb = lb;
        state_->vars[index].ub = ub;
        state_->vars[index].type = value;
    }

    double objective_coefficient() const {
        const int index = resolve_index_("objective_coefficient");
        const auto it = state_->objective.coeffs.find(index);
        return it == state_->objective.coeffs.end() ? 0.0 : it->second;
    }

    void set_objective_coefficient(double value) {
        const int index = resolve_index_("set_objective_coefficient");
        touch_state_();
        set_coeff_value(state_->objective, index, value);
    }

    std::string repr() const {
        const int index = resolve_index_("repr");
        std::ostringstream oss;
        oss << "Var(name='" << state_->vars[index].name << "', lb="
            << format_number(state_->vars[index].lb) << ", ub=";
        if (std::isfinite(state_->vars[index].ub)) {
            oss << format_number(state_->vars[index].ub);
        } else {
            oss << "inf";
        }
        oss << ", type=";
        switch (state_->vars[index].type) {
            case VarType::Continuous:
                oss << "'continuous'";
                break;
            case VarType::Integer:
                oss << "'integer'";
                break;
            case VarType::Binary:
                oss << "'binary'";
                break;
        }
        oss << ", obj=" << format_number(objective_coefficient()) << ")";
        return oss.str();
    }

   private:
    void touch_state_(bool invalidate_basis = false) const {
        ++state_->revision;
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_constraint_pi.clear();
        if (invalidate_basis) state_->last_basis.reset();
    }

    int resolve_index_(const char* context) const {
        if (!state_) {
            throw std::invalid_argument(
                std::string("simplex: invalid variable in ") + context);
        }
        if (index_ >= 0 && index_ < static_cast<int>(state_->vars.size()) &&
            state_->vars[index_].id == id_) {
            return index_;
        }
        for (int i = 0; i < static_cast<int>(state_->vars.size()); ++i) {
            if (state_->vars[i].id == id_) {
                index_ = i;
                return index_;
            }
        }
        throw std::invalid_argument(
            std::string("simplex: invalid variable in ") + context);
    }

    std::shared_ptr<ModelState> state_;
    mutable int index_ = -1;
    std::uint64_t id_ = 0;
};

class LinearExpr {
   public:
    LinearExpr() = default;

    explicit LinearExpr(double constant) { data_.constant = constant; }

    LinearExpr(std::shared_ptr<ModelState> state, LinearExprData data = {})
        : state_(std::move(state)), data_(std::move(data)) {}

    const std::shared_ptr<ModelState>& state() const { return state_; }
    const LinearExprData& data() const { return data_; }

    std::string repr() const { return "LinearExpr(" + expr_repr(data_, state_) + ")"; }

   private:
    friend LinearExpr to_expr(const Var& var);
    friend LinearExpr make_constant_expr(const std::shared_ptr<ModelState>& state,
                                         double value);
    friend LinearExpr add_expr(const LinearExpr& lhs, const LinearExpr& rhs);
    friend LinearExpr sub_expr(const LinearExpr& lhs, const LinearExpr& rhs);
    friend LinearExpr scale_expr(const LinearExpr& expr, double scalar);
    friend class Model;
    friend class ConstraintSpec;

    std::shared_ptr<ModelState> state_;
    LinearExprData data_;
};

class ConstraintSpec {
   public:
    ConstraintSpec() = default;

    ConstraintSpec(std::shared_ptr<ModelState> state, LinearExprData expr,
                   ConstraintSense sense)
        : state_(std::move(state)), expr_(std::move(expr)), sense_(sense) {}

    const std::shared_ptr<ModelState>& state() const { return state_; }
    const LinearExprData& expr() const { return expr_; }
    ConstraintSense sense() const { return sense_; }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Constraint(" << expr_repr(expr_, state_) << " ";
        switch (sense_) {
            case ConstraintSense::LessEqual:
                oss << "<= 0";
                break;
            case ConstraintSense::Equal:
                oss << "== 0";
                break;
            case ConstraintSense::GreaterEqual:
                oss << ">= 0";
                break;
        }
        oss << ")";
        return oss.str();
    }

   private:
    std::shared_ptr<ModelState> state_;
    LinearExprData expr_;
    ConstraintSense sense_ = ConstraintSense::Equal;
};

LinearExpr to_expr(const Var& var) {
    LinearExprData data;
    add_coeff(data, var.index(), 1.0);
    return LinearExpr(var.state(), std::move(data));
}

LinearExpr make_constant_expr(const std::shared_ptr<ModelState>& state,
                              double value) {
    LinearExprData data;
    data.constant = value;
    return LinearExpr(state, std::move(data));
}

LinearExpr add_expr(const LinearExpr& lhs, const LinearExpr& rhs) {
    auto state = merge_model_state(lhs.state(), rhs.state(), "addition");
    return LinearExpr(state, add_expr_data(lhs.data(), rhs.data()));
}

LinearExpr sub_expr(const LinearExpr& lhs, const LinearExpr& rhs) {
    auto state = merge_model_state(lhs.state(), rhs.state(), "subtraction");
    return LinearExpr(state, add_expr_data(lhs.data(), rhs.data(), -1.0));
}

LinearExpr scale_expr(const LinearExpr& expr, double scalar) {
    return LinearExpr(expr.state(), scale_expr_data(expr.data(), scalar));
}

ConstraintSpec compare_exprs(const LinearExpr& lhs, const LinearExpr& rhs,
                             ConstraintSense sense) {
    auto state = merge_model_state(lhs.state(), rhs.state(), "constraint");
    return ConstraintSpec(state, add_expr_data(lhs.data(), rhs.data(), -1.0), sense);
}

std::vector<double> compute_constraint_duals(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& c, const LPSolution& raw,
    double objective_sign) {
    const int m = static_cast<int>(A.rows());
    std::vector<double> pi(m, std::numeric_limits<double>::quiet_NaN());
    if (m == 0) {
        return pi;
    }
    if (raw.dual_values.size() == m) {
        for (int i = 0; i < m; ++i) {
            const double value = objective_sign * raw.dual_values(i);
            pi[i] = std::abs(value) <= kCoeffTol ? 0.0 : value;
        }
        return pi;
    }
    if (raw.status != LPSolution::Status::Optimal ||
        static_cast<int>(raw.basis.size()) != m) {
        return pi;
    }

    Eigen::MatrixXd B(m, m);
    Eigen::VectorXd cB(m);
    for (int i = 0; i < m; ++i) {
        const int basis_index = raw.basis[i];
        if (basis_index < 0 || basis_index >= A.cols() || basis_index >= c.size()) {
            return pi;
        }
        B.col(i) = A.col(basis_index);
        cB(i) = c(basis_index);
    }

    Eigen::FullPivLU<Eigen::MatrixXd> lu(B.transpose());
    if (!(lu.rank() == m && lu.isInvertible())) {
        return pi;
    }

    const Eigen::VectorXd y = lu.solve(cB);
    for (int i = 0; i < m; ++i) {
        const double value = objective_sign * y(i);
        pi[i] = std::abs(value) <= kCoeffTol ? 0.0 : value;
    }
    return pi;
}

class ConstraintHandle {
   public:
    ConstraintHandle() = default;

    ConstraintHandle(std::shared_ptr<ModelState> state, int index, std::uint64_t id)
        : state_(std::move(state)), index_(index), id_(id) {}

    const std::shared_ptr<ModelState>& state() const { return state_; }

    double pi() const {
        const int index = resolve_index_("pi");
        if (state_->solved_revision != state_->revision) {
            throw std::runtime_error(
                "simplex: constraint duals are unavailable until the model is solved");
        }
        if (index < 0 || index >= static_cast<int>(state_->last_constraint_pi.size())) {
            throw std::out_of_range("simplex: constraint index out of range");
        }
        return state_->last_constraint_pi[index];
    }

    std::string name() const {
        const int index = resolve_index_("name");
        return state_->constraints[index].name;
    }

    double rhs() const {
        const int index = resolve_index_("rhs");
        return -state_->constraints[index].expr.constant;
    }

    void set_rhs(double value) {
        const int index = resolve_index_("set_rhs");
        touch_state_();
        state_->constraints[index].expr.constant = -value;
    }

    ConstraintSense sense() const {
        const int index = resolve_index_("sense");
        return state_->constraints[index].sense;
    }

    void set_sense(ConstraintSense value) {
        const int index = resolve_index_("set_sense");
        touch_state_(true);
        state_->constraints[index].sense = value;
    }

    double coefficient(const Var& var) const {
        const int index = resolve_index_("coefficient");
        if (!var.state() || var.state().get() != state_.get()) {
            throw std::invalid_argument(
                "simplex: variable does not belong to this constraint's model");
        }
        const auto it = state_->constraints[index].expr.coeffs.find(var.index());
        return it == state_->constraints[index].expr.coeffs.end() ? 0.0 : it->second;
    }

    void set_coefficient(const Var& var, double value) {
        const int index = resolve_index_("set_coefficient");
        if (!var.state() || var.state().get() != state_.get()) {
            throw std::invalid_argument(
                "simplex: variable does not belong to this constraint's model");
        }
        touch_state_();
        set_coeff_value(state_->constraints[index].expr, var.index(), value);
    }

    int index() const { return resolve_index_("index"); }

    std::string repr() const {
        const int index = resolve_index_("repr");
        std::ostringstream oss;
        oss << "ConstraintHandle(index=" << index;
        if (!state_->constraints[index].name.empty()) {
            oss << ", name='" << state_->constraints[index].name << "'";
        }
        oss << ")";
        return oss.str();
    }

   private:
    void touch_state_(bool invalidate_basis = false) const {
        ++state_->revision;
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_constraint_pi.clear();
        if (invalidate_basis) state_->last_basis.reset();
    }

    int resolve_index_(const char* context) const {
        if (!state_) {
            throw std::invalid_argument(
                std::string("simplex: invalid constraint handle in ") + context);
        }
        if (index_ >= 0 && index_ < static_cast<int>(state_->constraints.size()) &&
            state_->constraints[index_].id == id_) {
            return index_;
        }
        for (int i = 0; i < static_cast<int>(state_->constraints.size()); ++i) {
            if (state_->constraints[i].id == id_) {
                index_ = i;
                return index_;
            }
        }
        throw std::invalid_argument(
            std::string("simplex: invalid constraint handle in ") + context);
    }

    std::shared_ptr<ModelState> state_;
    mutable int index_ = -1;
    std::uint64_t id_ = 0;
};

class ModelSolution {
   public:
    ModelSolution() = default;

    ModelSolution(std::shared_ptr<ModelState> state, LPSolution raw,
                  Eigen::VectorXd primal, double objective)
        : state_(std::move(state)),
          raw_(std::move(raw)),
          primal_(std::move(primal)),
          objective_(objective) {
        if (state_) {
            for (int i = 0; i < primal_.size() && i < static_cast<int>(state_->vars.size());
                 ++i) {
                values_.emplace(state_->vars[i].name, primal_(i));
            }
        }
    }

    const LPSolution& raw() const { return raw_; }
    const Eigen::VectorXd& x() const { return primal_; }
    LPSolution::Status status() const { return raw_.status; }
    double objective() const { return objective_; }
    int iterations() const { return raw_.iters; }
    const std::unordered_map<std::string, double>& values() const { return values_; }
    const std::vector<std::string>& log_lines() const { return raw_.trace; }
    std::string log() const { return join_trace_lines(raw_.trace); }
    SolveStats stats() const { return build_solve_stats(raw_); }
    LPBasis basis() const { return rebuild_basis_from_solution(raw_); }

    double value(const Var& var) const {
        if (!state_ || !var.state() || state_.get() != var.state().get()) {
            throw std::invalid_argument(
                "simplex: variable does not belong to this solution's model");
        }
        const int index = var.index();
        if (index < 0 || index >= primal_.size()) {
            throw std::out_of_range("simplex: variable index out of range");
        }
        return primal_(index);
    }

    double value(const std::string& name) const {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::out_of_range("simplex: unknown variable name '" + name + "'");
        }
        return it->second;
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "ModelSolution(status='" << to_string(raw_.status) << "', obj="
            << format_number(objective_) << ")";
        return oss.str();
    }

   private:
    std::shared_ptr<ModelState> state_;
    LPSolution raw_;
    Eigen::VectorXd primal_;
    double objective_ = std::numeric_limits<double>::quiet_NaN();
    std::unordered_map<std::string, double> values_;
};

class MIPSolution {
   public:
    MIPSolution() = default;

    MIPSolution(std::shared_ptr<ModelState> state, simplex_bnb::SolveResult result,
                int original_vars)
        : state_(std::move(state)),
          status_(result.status),
          objective_(result.objective),
          best_bound_(result.best_bound),
          root_relaxation_objective_(result.root_relaxation_objective),
          root_presolve_tightened_bounds_(result.root_presolve_tightened_bounds),
          root_presolve_removed_rows_(result.root_presolve_removed_rows),
          root_presolve_removed_coeffs_(result.root_presolve_removed_coeffs),
          root_presolve_aggregations_(result.root_presolve_aggregations),
          node_count_(result.node_count),
          lp_iterations_(result.lp_iterations),
          incumbent_updates_(result.incumbent_updates),
          heuristic_lp_iterations_(result.heuristic_lp_iterations),
          heuristic_successes_(result.heuristic_successes),
          feasibility_jump_successes_(result.feasibility_jump_successes),
          feasibility_pump_successes_(result.feasibility_pump_successes),
          rens_successes_(result.rens_successes),
          rins_successes_(result.rins_successes),
          local_search_successes_(result.local_search_successes),
          local_branching_successes_(result.local_branching_successes),
          cuts_generated_(result.cuts_generated),
          cuts_applied_(result.cuts_applied),
          duplicate_cuts_(result.duplicate_cuts),
          cut_pool_size_(result.cut_pool_size),
          tree_nodes_(std::move(result.tree_nodes)) {
        primal_ = Eigen::VectorXd::Constant(original_vars, std::numeric_limits<double>::quiet_NaN());
        if (result.primal.size() >= original_vars) {
            primal_ = result.primal.head(original_vars);
        }
        has_solution_ = true;
        for (int i = 0; i < primal_.size(); ++i) {
            if (!std::isfinite(primal_(i))) {
                has_solution_ = false;
                break;
            }
        }
        if (state_ && has_solution_) {
            for (int i = 0; i < primal_.size() && i < static_cast<int>(state_->vars.size());
                 ++i) {
                values_.emplace(state_->vars[i].name, primal_(i));
            }
        }
    }

    MIPStatus status() const { return status_; }
    const Eigen::VectorXd& x() const { return primal_; }
    double objective() const { return objective_; }
    double best_bound() const { return best_bound_; }
    double root_relaxation_objective() const { return root_relaxation_objective_; }
    int root_presolve_tightened_bounds() const { return root_presolve_tightened_bounds_; }
    int root_presolve_removed_rows() const { return root_presolve_removed_rows_; }
    int root_presolve_removed_coeffs() const { return root_presolve_removed_coeffs_; }
    int root_presolve_aggregations() const { return root_presolve_aggregations_; }
    int node_count() const { return node_count_; }
    int lp_iterations() const { return lp_iterations_; }
    int incumbent_updates() const { return incumbent_updates_; }
    int heuristic_lp_iterations() const { return heuristic_lp_iterations_; }
    int heuristic_successes() const { return heuristic_successes_; }
    int feasibility_jump_successes() const { return feasibility_jump_successes_; }
    int feasibility_pump_successes() const { return feasibility_pump_successes_; }
    int rens_successes() const { return rens_successes_; }
    int rins_successes() const { return rins_successes_; }
    int local_search_successes() const { return local_search_successes_; }
    int local_branching_successes() const { return local_branching_successes_; }
    int cuts_generated() const { return cuts_generated_; }
    int cuts_applied() const { return cuts_applied_; }
    int duplicate_cuts() const { return duplicate_cuts_; }
    int cut_pool_size() const { return cut_pool_size_; }
    bool has_solution() const { return has_solution_; }
    const std::unordered_map<std::string, double>& values() const { return values_; }
    const std::vector<MIPTreeNode>& tree_nodes() const { return tree_nodes_; }

    std::optional<double> relative_gap() const {
        if (!has_solution_ || !std::isfinite(best_bound_) || !std::isfinite(objective_)) {
            return std::nullopt;
        }
        const double denom = std::max(1.0, std::abs(objective_));
        return std::abs(best_bound_ - objective_) / denom;
    }

    double value(const Var& var) const {
        if (!has_solution_ || !state_ || !var.state() || state_.get() != var.state().get()) {
            throw std::invalid_argument(
                "simplex: variable does not belong to this MIP solution's model");
        }
        const int index = var.index();
        if (index < 0 || index >= primal_.size()) {
            throw std::out_of_range("simplex: variable index out of range");
        }
        return primal_(index);
    }

    double value(const std::string& name) const {
        const auto it = values_.find(name);
        if (it == values_.end()) {
            throw std::out_of_range("simplex: unknown variable name '" + name + "'");
        }
        return it->second;
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "MIPSolution(status='" << simplex_bnb::to_string(status_) << "', obj=";
        if (std::isfinite(objective_)) {
            oss << format_number(objective_);
        } else {
            oss << "nan";
        }
        oss << ", best_bound=";
        if (std::isfinite(best_bound_)) {
            oss << format_number(best_bound_);
        } else if (std::isnan(best_bound_)) {
            oss << "nan";
        } else {
            oss << (best_bound_ > 0.0 ? "inf" : "-inf");
        }
        oss << ", nodes=" << node_count_ << ")";
        return oss.str();
    }

   private:
    std::shared_ptr<ModelState> state_;
    MIPStatus status_ = MIPStatus::Infeasible;
    Eigen::VectorXd primal_;
    double objective_ = std::numeric_limits<double>::quiet_NaN();
    double best_bound_ = std::numeric_limits<double>::quiet_NaN();
    double root_relaxation_objective_ = std::numeric_limits<double>::quiet_NaN();
    int root_presolve_tightened_bounds_ = 0;
    int root_presolve_removed_rows_ = 0;
    int root_presolve_removed_coeffs_ = 0;
    int root_presolve_aggregations_ = 0;
    int node_count_ = 0;
    int lp_iterations_ = 0;
    int incumbent_updates_ = 0;
    int heuristic_lp_iterations_ = 0;
    int heuristic_successes_ = 0;
    int feasibility_jump_successes_ = 0;
    int feasibility_pump_successes_ = 0;
    int rens_successes_ = 0;
    int rins_successes_ = 0;
    int local_search_successes_ = 0;
    int local_branching_successes_ = 0;
    int cuts_generated_ = 0;
    int cuts_applied_ = 0;
    int duplicate_cuts_ = 0;
    int cut_pool_size_ = 0;
    bool has_solution_ = false;
    std::unordered_map<std::string, double> values_;
    std::vector<MIPTreeNode> tree_nodes_;
};

struct ModelLPData {
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
    double objective_sign = 1.0;
    int original_vars = 0;
    int total_vars = 0;
    int rows = 0;
};

class Model {
   public:
    explicit Model(const RevisedSimplexOptions& options = {})
        : state_(std::make_shared<ModelState>()) {
        state_->options = options;
    }

    Var add_var(const std::optional<std::string>& name = std::nullopt, double lb = 0.0,
                double ub = std::numeric_limits<double>::infinity(),
                double obj = 0.0, VarType var_type = VarType::Continuous) {
        touch_(true);
        std::tie(lb, ub) = canonicalize_var_bounds(var_type, lb, ub);

        std::string resolved_name;
        if (name && !name->empty()) {
            resolved_name = *name;
        } else {
            resolved_name = next_auto_name_();
        }

        if (state_->name_to_index.contains(resolved_name)) {
            throw std::invalid_argument("simplex: duplicate variable name '" +
                                        resolved_name + "'");
        }

        const int index = static_cast<int>(state_->vars.size());
        const std::uint64_t id = state_->next_var_id++;
        state_->vars.push_back(VarData{id, resolved_name, lb, ub, var_type});
        state_->name_to_index.emplace(resolved_name, index);
        if (std::abs(obj) > kCoeffTol) {
            add_coeff(state_->objective, index, obj);
        }

        return Var(state_, index, id);
    }

    Var add_integer_var(const std::optional<std::string>& name = std::nullopt,
                        double lb = 0.0,
                        double ub = std::numeric_limits<double>::infinity(),
                        double obj = 0.0) {
        return add_var(name, lb, ub, obj, VarType::Integer);
    }

    Var add_binary_var(const std::optional<std::string>& name = std::nullopt,
                       double obj = 0.0) {
        return add_var(name, 0.0, 1.0, obj, VarType::Binary);
    }

    ConstraintHandle add_constr(const ConstraintSpec& constr,
                                const std::optional<std::string>& name = std::nullopt) {
        touch_(true);
        if (!constr.state() || constr.state().get() != state_.get()) {
            throw std::invalid_argument(
                "simplex: constraint does not belong to this model");
        }

        const std::uint64_t id = state_->next_constraint_id++;
        ConstraintData data{id, constr.expr(), constr.sense(), name.value_or("")};
        state_->constraints.push_back(std::move(data));
        return ConstraintHandle(state_, static_cast<int>(state_->constraints.size()) - 1,
                                id);
    }

    void set_objective(const LinearExpr& expr, const std::string& sense = "min") {
        touch_();
        if (expr.state() && expr.state().get() != state_.get()) {
            throw std::invalid_argument(
                "simplex: objective expression does not belong to this model");
        }

        std::string normalized = sense;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (normalized != "min" && normalized != "max") {
            throw std::invalid_argument(
                "simplex: objective sense must be 'min' or 'max'");
        }

        state_->objective = expr.data();
        state_->maximize = normalized == "max";
    }

    void minimize(const LinearExpr& expr) { set_objective(expr, "min"); }
    void maximize(const LinearExpr& expr) { set_objective(expr, "max"); }

    Var get_var(const std::string& name) const {
        const auto it = state_->name_to_index.find(name);
        if (it == state_->name_to_index.end()) {
            throw std::out_of_range("simplex: unknown variable name '" + name + "'");
        }
        return Var(state_, it->second, state_->vars[it->second].id);
    }

    int num_vars() const { return static_cast<int>(state_->vars.size()); }
    int num_constraints() const { return static_cast<int>(state_->constraints.size()); }

    RevisedSimplexOptions& options() { return state_->options; }
    const RevisedSimplexOptions& options() const { return state_->options; }

    double get_obj_coeff(const Var& var) const {
        ensure_same_model_(var.state(), "get_obj_coeff");
        return var.objective_coefficient();
    }

    void set_obj_coeff(const Var& var, double value) {
        ensure_same_model_(var.state(), "set_obj_coeff");
        touch_();
        set_coeff_value(state_->objective, var.index(), value);
    }

    double get_coeff(const ConstraintHandle& constr, const Var& var) const {
        ensure_same_model_(constr.state(), "get_coeff");
        ensure_same_model_(var.state(), "get_coeff");
        return constr.coefficient(var);
    }

    void set_coeff(const ConstraintHandle& constr, const Var& var, double value) {
        ensure_same_model_(constr.state(), "set_coeff");
        ensure_same_model_(var.state(), "set_coeff");
        touch_();
        set_coeff_value(state_->constraints[constr.index()].expr, var.index(), value);
    }

    void set_rhs(const ConstraintHandle& constr, double rhs) {
        ensure_same_model_(constr.state(), "set_rhs");
        touch_();
        state_->constraints[constr.index()].expr.constant = -rhs;
    }

    void delete_var(const Var& var) {
        ensure_same_model_(var.state(), "delete_var");
        const int removed_index = var.index();
        touch_(true);
        state_->vars.erase(state_->vars.begin() + removed_index);
        rebuild_name_to_index_();
        erase_and_reindex_coeffs(state_->objective, removed_index);
        for (auto& constr : state_->constraints) {
            erase_and_reindex_coeffs(constr.expr, removed_index);
        }
    }

    void delete_constr(const ConstraintHandle& constr) {
        ensure_same_model_(constr.state(), "delete_constr");
        touch_(true);
        state_->constraints.erase(state_->constraints.begin() + constr.index());
    }

    ModelSolution reoptimize(std::optional<LPBasis> warm_start = std::nullopt) const {
        return solve(std::move(warm_start));
    }

    ModelSolution solve(std::optional<LPBasis> warm_start = std::nullopt) const {
        const ModelLPData data = build_lp_data_();
        RevisedSimplex solver(state_->options);
        const LPBasis* effective_basis = nullptr;
        if (warm_start) {
            if (!basis_matches_dimensions(*warm_start, data.total_vars, data.rows)) {
                throw std::invalid_argument(
                    "simplex: warm-start basis does not match model dimensions");
            }
            effective_basis = &*warm_start;
        } else if (state_->last_basis &&
                   basis_matches_dimensions(*state_->last_basis, data.total_vars, data.rows)) {
            effective_basis = &*state_->last_basis;
        }

        auto raw = effective_basis
                       ? solver.solve(data.A, data.b, data.c, data.l, data.u, *effective_basis)
                       : solver.solve(data.A, data.b, data.c, data.l, data.u);
        state_->last_constraint_pi =
            compute_constraint_duals(data.A, data.c, raw, data.objective_sign);
        state_->solved_revision = state_->revision;
        const LPBasis rebuilt_basis = rebuild_basis_from_solution(raw);
        if (basis_matches_dimensions(rebuilt_basis, data.total_vars, data.rows)) {
            state_->last_basis = rebuilt_basis;
        }
        return make_model_solution_(std::move(raw), data);
    }

    MIPSolution solve_mip(
        const BranchAndBoundOptions& mip_options = BranchAndBoundOptions()) const {
        // Keep the LP option profile aligned with model.solve(): forcing a
        // different pricing/mode here led to incorrect infeasibility in some
        // cover-cut subproblems.
        RevisedSimplexOptions cold_lp_options = state_->options;
        RevisedSimplexOptions warm_lp_options = state_->options;
        state_->last_constraint_pi.clear();
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_basis.reset();
        simplex_bnb::Problem problem;
        const int original_vars = static_cast<int>(state_->vars.size());
        problem.lower_bounds = Eigen::VectorXd::Zero(original_vars);
        problem.upper_bounds = Eigen::VectorXd::Constant(
            original_vars, std::numeric_limits<double>::infinity());
        problem.maximize = state_->maximize;
        problem.objective_coefficients = Eigen::VectorXd::Zero(original_vars);
        problem.objective_constant = state_->objective.constant;
        problem.variable_types.assign(original_vars, VarType::Continuous);
        for (int j = 0; j < original_vars; ++j) {
            problem.lower_bounds(j) = state_->vars[j].lb;
            problem.upper_bounds(j) = state_->vars[j].ub;
            problem.variable_types[j] = state_->vars[j].type;
        }
        for (const auto& [index, coeff] : state_->objective.coeffs) {
            if (index >= 0 && index < original_vars) {
                problem.objective_coefficients(index) = coeff;
            }
        }
        problem.base_constraints = build_base_constraints_();

        const RootProblemPresolveResult root_presolve = presolve_mip_root_problem(problem);
        if (root_presolve.infeasible) {
            simplex_bnb::SolveResult infeasible_result;
            infeasible_result.status = simplex_bnb::Status::Infeasible;
            infeasible_result.primal = Eigen::VectorXd::Constant(
                original_vars, std::numeric_limits<double>::quiet_NaN());
            infeasible_result.objective = std::numeric_limits<double>::quiet_NaN();
            infeasible_result.best_bound = state_->maximize
                                               ? -std::numeric_limits<double>::infinity()
                                               : std::numeric_limits<double>::infinity();
            infeasible_result.root_presolve_tightened_bounds = root_presolve.tightened_bounds;
            infeasible_result.root_presolve_removed_rows = root_presolve.removed_rows;
            infeasible_result.root_presolve_removed_coeffs = root_presolve.removed_coeffs;
            infeasible_result.root_presolve_aggregations = root_presolve.aggregations;
            return MIPSolution(state_, std::move(infeasible_result), original_vars);
        }
        problem = root_presolve.problem;
        const ModelLPData data = build_lp_data_from_problem_(problem);
        problem.lower_bounds = data.l;
        problem.upper_bounds = data.u;
        problem.objective_coefficients.conservativeResize(data.total_vars);
        for (int j = original_vars; j < data.total_vars; ++j) {
            problem.objective_coefficients(j) = 0.0;
        }
        problem.variable_types.resize(data.total_vars, VarType::Continuous);

        struct ThreadLocalMIPLPContext {
            RevisedSimplex cold_solver;
            RevisedSimplex warm_solver;
            std::optional<RevisedSimplex> fallback_solver;
            std::unordered_map<std::string, ModelLPData> node_lp_cache;

            ThreadLocalMIPLPContext(const RevisedSimplexOptions& cold_options,
                                    const RevisedSimplexOptions& warm_options)
                : cold_solver(cold_options), warm_solver(warm_options) {
                if (cold_options.mode == SimplexMode::Dual) {
                    RevisedSimplexOptions fallback_options = cold_options;
                    fallback_options.mode = SimplexMode::Auto;
                    fallback_solver.emplace(fallback_options);
                } else if (cold_options.mode == SimplexMode::Auto) {
                    RevisedSimplexOptions fallback_options = cold_options;
                    fallback_options.mode = SimplexMode::Primal;
                    fallback_solver.emplace(fallback_options);
                }
            }
        };

        simplex_bnb::Solver bnb_solver(problem, mip_options);
        simplex_bnb::SolveResult result = bnb_solver.solve(
            [&](const Eigen::VectorXd& l_node, const Eigen::VectorXd& u_node,
                const LPBasis* basis,
                const std::vector<simplex_bnb::Cut>& cuts)
                -> simplex_bnb::RelaxationSolution {
                static thread_local std::unordered_map<const ModelState*,
                                                       std::unique_ptr<ThreadLocalMIPLPContext>>
                    thread_contexts;
                auto& context_ptr = thread_contexts[state_.get()];
                if (!context_ptr) {
                    context_ptr = std::make_unique<ThreadLocalMIPLPContext>(cold_lp_options,
                                                                            warm_lp_options);
                }
                ThreadLocalMIPLPContext& thread_context = *context_ptr;

                std::vector<simplex_bnb::Cut> presolve_only_cuts;
                std::vector<simplex_bnb::Cut> structural_cuts;
                presolve_only_cuts.reserve(cuts.size());
                structural_cuts.reserve(cuts.size());
                for (const auto& cut : cuts) {
                    if (cut.cut_type == "IncumbentCutoff") {
                        presolve_only_cuts.push_back(cut);
                    } else {
                        structural_cuts.push_back(cut);
                    }
                }

                Eigen::VectorXd node_l = data.l;
                Eigen::VectorXd node_u = data.u;
                if (l_node.size() == data.original_vars && u_node.size() == data.original_vars) {
                    node_l.head(data.original_vars) = l_node;
                    node_u.head(data.original_vars) = u_node;
                } else if (l_node.size() == data.total_vars && u_node.size() == data.total_vars) {
                    node_l = l_node;
                    node_u = u_node;
                }

                // Structural cuts stay in the LP as rows. Simplifying them
                // node-by-node was pruning feasible cover-cut branches.
                SimplifiedCutsResult simplified_structural_cuts;
                simplified_structural_cuts.cuts = structural_cuts;
                const SimplifiedCutsResult simplified_presolve_cuts =
                    simplify_cuts_for_bounds(presolve_only_cuts, node_l.head(data.total_vars),
                                             node_u.head(data.total_vars));
                if (simplified_presolve_cuts.infeasible) {
                    simplex_bnb::RelaxationSolution out;
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = state_->maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
                    return out;
                }
                const NodeBoundPresolveResult base_presolve = presolve_mip_node_bounds(
                    problem, node_l.head(data.original_vars), node_u.head(data.original_vars),
                    simplified_presolve_cuts.cuts);
                if (base_presolve.infeasible) {
                    simplex_bnb::RelaxationSolution out;
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = state_->maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
                    return out;
                }
                node_l.head(data.original_vars) = base_presolve.lower;
                node_u.head(data.original_vars) = base_presolve.upper;

                const ModelLPData* node_data = &data;
                if (!simplified_structural_cuts.cuts.empty()) {
                    const std::string cache_key =
                        cut_set_signature(simplified_structural_cuts.cuts);
                    auto it = thread_context.node_lp_cache.find(cache_key);
                    if (it == thread_context.node_lp_cache.end()) {
                        it = thread_context.node_lp_cache
                                 .emplace(cache_key,
                                          build_node_lp_data_from_base_(data,
                                                                       simplified_structural_cuts
                                                                           .cuts))
                                 .first;
                    }
                    node_data = &it->second;
                }

                Eigen::VectorXd solve_l = node_data->l;
                Eigen::VectorXd solve_u = node_data->u;
                solve_l.head(data.total_vars) = node_l;
                solve_u.head(data.total_vars) = node_u;

                std::vector<simplex_bnb::Cut> node_presolve_cuts =
                    simplified_presolve_cuts.cuts;
                const NodeBoundPresolveResult node_presolve = presolve_mip_node_bounds(
                    problem, solve_l.head(data.original_vars), solve_u.head(data.original_vars),
                    node_presolve_cuts);
                if (node_presolve.infeasible) {
                    simplex_bnb::RelaxationSolution out;
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = state_->maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
                    return out;
                }
                solve_l.head(data.original_vars) = node_presolve.lower;
                solve_u.head(data.original_vars) = node_presolve.upper;

                struct StandardizedNodeLP {
                    Eigen::MatrixXd A;
                    Eigen::VectorXd b;
                    Eigen::VectorXd c;
                    Eigen::VectorXd l;
                    Eigen::VectorXd u;
                    Eigen::VectorXd shift;
                    double objective_shift = 0.0;
                };

                auto standardize_node_lp = [&](const ModelLPData& lp_data,
                                               const Eigen::VectorXd& lower,
                                               const Eigen::VectorXd& upper) {
                    StandardizedNodeLP std_lp;
                    const int base_n = lp_data.total_vars;
                    const int base_m = lp_data.rows;

                    std_lp.shift = Eigen::VectorXd::Zero(base_n);
                    if (lower.size() >= base_n) {
                        std_lp.shift = lower.head(base_n);
                    }

                    std::vector<int> bounded_cols;
                    bounded_cols.reserve(base_n);
                    for (int j = 0; j < base_n && j < upper.size(); ++j) {
                        if (std::isfinite(upper(j))) bounded_cols.push_back(j);
                    }

                    const int extra_rows = static_cast<int>(bounded_cols.size());
                    const int extra_cols = extra_rows;
                    std_lp.A = Eigen::MatrixXd::Zero(base_m + extra_rows, base_n + extra_cols);
                    std_lp.b = Eigen::VectorXd::Zero(base_m + extra_rows);
                    std_lp.c = Eigen::VectorXd::Zero(base_n + extra_cols);
                    std_lp.l = Eigen::VectorXd::Zero(base_n + extra_cols);
                    std_lp.u = Eigen::VectorXd::Constant(base_n + extra_cols,
                                                         std::numeric_limits<double>::infinity());

                    std_lp.A.topLeftCorner(base_m, base_n) = lp_data.A;
                    std_lp.b.head(base_m) = lp_data.b - lp_data.A * std_lp.shift;
                    std_lp.c.head(base_n) = lp_data.c;
                    std_lp.objective_shift = lp_data.c.dot(std_lp.shift);

                    for (int k = 0; k < extra_rows; ++k) {
                        const int j = bounded_cols[k];
                        const int row = base_m + k;
                        const int slack = base_n + k;
                        const double rhs =
                            std::max(0.0, upper(j) - ((j < lower.size()) ? lower(j) : 0.0));
                        std_lp.A(row, j) = 1.0;
                        std_lp.A(row, slack) = 1.0;
                        std_lp.b(row) = rhs;
                    }
                    return std_lp;
                };

                const bool needs_standardized_node_lp =
                    (solve_l.size() >= node_data->total_vars &&
                     solve_u.size() >= node_data->total_vars) &&
                    (((solve_l.head(node_data->total_vars).array().abs() > 1e-12).any()) ||
                     (solve_u.head(node_data->total_vars).array().isFinite().any()));
                std::optional<StandardizedNodeLP> standardized_node_lp;
                if (needs_standardized_node_lp) {
                    standardized_node_lp =
                        standardize_node_lp(*node_data, solve_l, solve_u);
                }

                const LPBasis* effective_basis = nullptr;
                if (!standardized_node_lp.has_value() && basis &&
                    basis_matches_dimensions(*basis, node_data->total_vars, node_data->rows)) {
                    effective_basis = basis;
                }
                const bool use_fresh_solvers_for_this_lp =
                    !simplified_structural_cuts.cuts.empty();
                RevisedSimplex local_cold_solver(cold_lp_options);
                RevisedSimplex local_warm_solver(warm_lp_options);
                std::optional<RevisedSimplex> local_fallback_solver;
                if (use_fresh_solvers_for_this_lp) {
                    if (cold_lp_options.mode == SimplexMode::Dual) {
                        RevisedSimplexOptions fallback_options = cold_lp_options;
                        fallback_options.mode = SimplexMode::Auto;
                        local_fallback_solver.emplace(fallback_options);
                    } else if (cold_lp_options.mode == SimplexMode::Auto) {
                        RevisedSimplexOptions fallback_options = cold_lp_options;
                        fallback_options.mode = SimplexMode::Primal;
                        local_fallback_solver.emplace(fallback_options);
                    }
                }
                RevisedSimplex* warm_solver =
                    use_fresh_solvers_for_this_lp ? &local_warm_solver
                                                  : &thread_context.warm_solver;
                RevisedSimplex* cold_solver =
                    use_fresh_solvers_for_this_lp ? &local_cold_solver
                                                  : &thread_context.cold_solver;
                std::optional<RevisedSimplex>* fallback_solver =
                    use_fresh_solvers_for_this_lp ? &local_fallback_solver
                                                  : &thread_context.fallback_solver;
                const auto is_retryable_lu_failure = [](const std::runtime_error& err) {
                    const std::string_view msg(err.what());
                    return msg.find("MarkowitzLU: singular matrix") != std::string_view::npos ||
                           msg.find("MarkowitzLU: numerically singular pivot") !=
                               std::string_view::npos;
                };
                const auto try_lp_solve =
                    [&](RevisedSimplex& solver,
                        const LPBasis* basis_arg) -> std::optional<LPSolution> {
                    try {
                        if (standardized_node_lp.has_value()) {
                            const RevisedSimplex::SparseMatrix A_sparse =
                                standardized_node_lp->A.sparseView(kCoeffTol, 1.0);
                            return solver.solve(A_sparse, standardized_node_lp->b,
                                                standardized_node_lp->c,
                                                standardized_node_lp->l,
                                                standardized_node_lp->u);
                        }
                        const RevisedSimplex::SparseMatrix A_sparse =
                            node_data->A.sparseView(kCoeffTol, 1.0);
                        if (basis_arg) {
                            return solver.solve(A_sparse, node_data->b, node_data->c, solve_l,
                                                solve_u, *basis_arg);
                        }
                        return solver.solve(A_sparse, node_data->b, node_data->c, solve_l,
                                            solve_u);
                    } catch (const std::runtime_error& err) {
                        if (is_retryable_lu_failure(err)) {
                            return std::nullopt;
                        }
                        throw;
                    }
                };

                std::optional<LPSolution> raw_opt;
                if (effective_basis) {
                    raw_opt = try_lp_solve(*warm_solver, effective_basis);
                } else {
                    raw_opt = try_lp_solve(*cold_solver, nullptr);
                }

                if ((!raw_opt.has_value() || raw_opt->status == LPSolution::Status::Singular ||
                     raw_opt->status == LPSolution::Status::NeedPhase1) &&
                    effective_basis) {
                    raw_opt = try_lp_solve(*cold_solver, nullptr);
                }
                if ((!raw_opt.has_value() || raw_opt->status == LPSolution::Status::Singular ||
                     raw_opt->status == LPSolution::Status::NeedPhase1) &&
                    fallback_solver->has_value()) {
                    raw_opt = try_lp_solve(**fallback_solver, nullptr);
                }

                LPSolution raw;
                if (raw_opt.has_value()) {
                    raw = std::move(*raw_opt);
                } else {
                    raw.status = LPSolution::Status::Singular;
                    raw.x = Eigen::VectorXd::Constant(
                        node_data->total_vars, std::numeric_limits<double>::quiet_NaN());
                    raw.obj = std::numeric_limits<double>::quiet_NaN();
                }

                simplex_bnb::RelaxationSolution out;
                if (raw.status == LPSolution::Status::Optimal) {
                    out.status = simplex_bnb::RelaxationStatus::Optimal;
                } else if (raw.status == LPSolution::Status::Unbounded) {
                    out.status = simplex_bnb::RelaxationStatus::Unbounded;
                } else {
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                }
                if (standardized_node_lp.has_value()) {
                    out.primal = Eigen::VectorXd::Zero(node_data->total_vars);
                    if (raw.x.size() >= node_data->total_vars) {
                        out.primal = raw.x.head(node_data->total_vars);
                    }
                    out.primal += standardized_node_lp->shift;
                    out.objective = node_data->objective_sign *
                                        (raw.obj + standardized_node_lp->objective_shift) +
                                    state_->objective.constant;
                } else {
                    out.primal = raw.x;
                    out.objective =
                        node_data->objective_sign * raw.obj + state_->objective.constant;
                }
                out.iterations = raw.iters;
                out.lp_solution = raw;
                if (!standardized_node_lp.has_value()) {
                    const LPBasis rebuilt_basis = rebuild_basis_from_solution(raw);
                    if (basis_matches_dimensions(rebuilt_basis, node_data->total_vars,
                                                node_data->rows)) {
                        out.basis = rebuilt_basis;
                    }
                }
                return out;
            });
        result.root_presolve_tightened_bounds = root_presolve.tightened_bounds;
        result.root_presolve_removed_rows = root_presolve.removed_rows;
        result.root_presolve_removed_coeffs = root_presolve.removed_coeffs;
        result.root_presolve_aggregations = root_presolve.aggregations;
        return MIPSolution(state_, std::move(result), data.original_vars);
    }

    std::string repr() const {
        std::ostringstream oss;
        oss << "Model(num_vars=" << state_->vars.size()
            << ", num_constraints=" << state_->constraints.size() << ")";
        return oss.str();
    }

   private:
    void ensure_same_model_(const std::shared_ptr<ModelState>& other,
                            const char* context) const {
        if (!other || other.get() != state_.get()) {
            throw std::invalid_argument(std::string("simplex: object does not belong to "
                                                    "this model in ") +
                                        context);
        }
    }

    void rebuild_name_to_index_() {
        state_->name_to_index.clear();
        for (int i = 0; i < static_cast<int>(state_->vars.size()); ++i) {
            state_->name_to_index.emplace(state_->vars[i].name, i);
        }
    }

    ModelLPData build_lp_data_(
        const std::vector<simplex_bnb::Cut>& cuts = {}) const {
        const int n = static_cast<int>(state_->vars.size());
        int base_slack_count = 0;
        for (const auto& constr : state_->constraints) {
            if (constr.sense != ConstraintSense::Equal) {
                ++base_slack_count;
            }
        }
        int cut_slack_count = 0;
        for (const auto& cut : cuts) {
            if (cut.sense != simplex_bnb::LinearConstraintSense::Equal) {
                ++cut_slack_count;
            }
        }

        ModelLPData out;
        out.original_vars = n;
        out.total_vars = n + base_slack_count + cut_slack_count;
        out.rows = static_cast<int>(state_->constraints.size() + cuts.size());
        out.A = Eigen::MatrixXd::Zero(out.rows, out.total_vars);
        out.b = Eigen::VectorXd::Zero(out.rows);
        out.c = Eigen::VectorXd::Zero(out.total_vars);
        out.l = Eigen::VectorXd::Zero(out.total_vars);
        out.u = Eigen::VectorXd::Constant(out.total_vars,
                                          std::numeric_limits<double>::infinity());

        for (int j = 0; j < n; ++j) {
            out.l(j) = state_->vars[j].lb;
            out.u(j) = state_->vars[j].ub;
        }

        out.objective_sign = state_->maximize ? -1.0 : 1.0;
        for (const auto& [index, coeff] : state_->objective.coeffs) {
            if (index < 0 || index >= n) {
                throw std::out_of_range("simplex: objective references invalid variable");
            }
            out.c(index) = out.objective_sign * coeff;
        }

        int next_slack = n;
        int row = 0;
        for (; row < static_cast<int>(state_->constraints.size()); ++row) {
            const auto& constr = state_->constraints[row];
            for (const auto& [index, coeff] : constr.expr.coeffs) {
                if (index < 0 || index >= n) {
                    throw std::out_of_range(
                        "simplex: constraint references invalid variable");
                }
                out.A(row, index) = coeff;
            }
            out.b(row) = -constr.expr.constant;

            if (constr.sense == ConstraintSense::LessEqual) {
                out.A(row, next_slack++) = 1.0;
            } else if (constr.sense == ConstraintSense::GreaterEqual) {
                out.A(row, next_slack++) = -1.0;
            }
        }

        for (const auto& cut : cuts) {
            for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                            k < static_cast<int>(cut.values.size());
                 ++k) {
                const int index = cut.indices[k];
                if (index < 0 || index >= n + base_slack_count) {
                    throw std::out_of_range("simplex: cut references invalid base variable");
                }
                out.A(row, index) = cut.values[k];
            }
            out.b(row) = cut.rhs;
            if (cut.sense == simplex_bnb::LinearConstraintSense::LessEqual) {
                out.A(row, next_slack++) = 1.0;
            } else if (cut.sense == simplex_bnb::LinearConstraintSense::GreaterEqual) {
                out.A(row, next_slack++) = -1.0;
            }
            ++row;
        }

        return out;
    }

    ModelLPData build_lp_data_from_problem_(
        const simplex_bnb::Problem& problem,
        const std::vector<simplex_bnb::Cut>& cuts = {}) const {
        const int n = static_cast<int>(problem.lower_bounds.size());
        int base_slack_count = 0;
        for (const auto& row : problem.base_constraints) {
            if (row.sense != simplex_bnb::LinearConstraintSense::Equal) {
                ++base_slack_count;
            }
        }
        int cut_slack_count = 0;
        for (const auto& cut : cuts) {
            if (cut.sense != simplex_bnb::LinearConstraintSense::Equal) {
                ++cut_slack_count;
            }
        }

        ModelLPData out;
        out.original_vars = n;
        out.total_vars = n + base_slack_count + cut_slack_count;
        out.rows = static_cast<int>(problem.base_constraints.size() + cuts.size());
        out.A = Eigen::MatrixXd::Zero(out.rows, out.total_vars);
        out.b = Eigen::VectorXd::Zero(out.rows);
        out.c = Eigen::VectorXd::Zero(out.total_vars);
        out.l = Eigen::VectorXd::Zero(out.total_vars);
        out.u = Eigen::VectorXd::Constant(out.total_vars,
                                          std::numeric_limits<double>::infinity());
        out.objective_sign = problem.maximize ? -1.0 : 1.0;

        out.l.head(n) = problem.lower_bounds;
        out.u.head(n) = problem.upper_bounds;
        for (int j = 0; j < n && j < problem.objective_coefficients.size(); ++j) {
            out.c(j) = out.objective_sign * problem.objective_coefficients(j);
        }

        int next_slack = n;
        int row_index = 0;
        for (const auto& row : problem.base_constraints) {
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                const int index = row.indices[k];
                if (index < 0 || index >= n) {
                    throw std::out_of_range("simplex: sparse problem row references invalid variable");
                }
                out.A(row_index, index) = row.values[k];
            }
            out.b(row_index) = row.rhs;
            if (row.sense == simplex_bnb::LinearConstraintSense::LessEqual) {
                out.A(row_index, next_slack++) = 1.0;
            } else if (row.sense == simplex_bnb::LinearConstraintSense::GreaterEqual) {
                out.A(row_index, next_slack++) = -1.0;
            }
            ++row_index;
        }

        for (const auto& cut : cuts) {
            for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                            k < static_cast<int>(cut.values.size());
                 ++k) {
                const int index = cut.indices[k];
                if (index < 0 || index >= n + base_slack_count) {
                    throw std::out_of_range("simplex: cut references invalid base variable");
                }
                out.A(row_index, index) = cut.values[k];
            }
            out.b(row_index) = cut.rhs;
            if (cut.sense == simplex_bnb::LinearConstraintSense::LessEqual) {
                out.A(row_index, next_slack++) = 1.0;
            } else if (cut.sense == simplex_bnb::LinearConstraintSense::GreaterEqual) {
                out.A(row_index, next_slack++) = -1.0;
            }
            ++row_index;
        }

        return out;
    }

    ModelLPData build_node_lp_data_from_base_(
        const ModelLPData& base_data, const std::vector<simplex_bnb::Cut>& cuts) const {
        int cut_slack_count = 0;
        for (const auto& cut : cuts) {
            if (cut.sense != simplex_bnb::LinearConstraintSense::Equal) {
                ++cut_slack_count;
            }
        }

        ModelLPData out;
        out.original_vars = base_data.original_vars;
        out.objective_sign = base_data.objective_sign;
        out.rows = base_data.rows + static_cast<int>(cuts.size());
        out.total_vars = base_data.total_vars + cut_slack_count;
        out.A = Eigen::MatrixXd::Zero(out.rows, out.total_vars);
        out.b = Eigen::VectorXd::Zero(out.rows);
        out.c = Eigen::VectorXd::Zero(out.total_vars);
        out.l = Eigen::VectorXd::Zero(out.total_vars);
        out.u = Eigen::VectorXd::Constant(out.total_vars,
                                          std::numeric_limits<double>::infinity());

        out.A.topLeftCorner(base_data.rows, base_data.total_vars) = base_data.A;
        out.b.head(base_data.rows) = base_data.b;
        out.c.head(base_data.total_vars) = base_data.c;
        out.l.head(base_data.total_vars) = base_data.l;
        out.u.head(base_data.total_vars) = base_data.u;

        int next_slack = base_data.total_vars;
        int row = base_data.rows;
        for (const auto& cut : cuts) {
            for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                            k < static_cast<int>(cut.values.size());
                 ++k) {
                const int index = cut.indices[k];
                if (index < 0 || index >= base_data.total_vars) {
                    throw std::out_of_range("simplex: cut references invalid base variable");
                }
                out.A(row, index) = cut.values[k];
            }
            out.b(row) = cut.rhs;
            if (cut.sense == simplex_bnb::LinearConstraintSense::LessEqual) {
                out.A(row, next_slack++) = 1.0;
            } else if (cut.sense == simplex_bnb::LinearConstraintSense::GreaterEqual) {
                out.A(row, next_slack++) = -1.0;
            }
            ++row;
        }

        return out;
    }

    std::vector<simplex_bnb::SparseLinearConstraint> build_base_constraints_() const {
        std::vector<simplex_bnb::SparseLinearConstraint> out;
        out.reserve(state_->constraints.size());
        for (const auto& constr : state_->constraints) {
            simplex_bnb::SparseLinearConstraint row;
            row.sense = to_bnb_sense(constr.sense);
            row.rhs = -constr.expr.constant;
            for (const auto& [index, coeff] : constr.expr.coeffs) {
                if (index < 0 || index >= static_cast<int>(state_->vars.size())) {
                    throw std::out_of_range(
                        "simplex: constraint references invalid variable");
                }
                row.indices.push_back(index);
                row.values.push_back(coeff);
            }
            out.push_back(std::move(row));
        }
        return out;
    }

    ModelSolution make_model_solution_(LPSolution raw, const ModelLPData& data) const {
        Eigen::VectorXd primal = Eigen::VectorXd::Constant(
            data.original_vars, std::numeric_limits<double>::quiet_NaN());
        if (raw.x.size() >= data.original_vars) {
            primal = raw.x.head(data.original_vars);
        }

        const double objective =
            data.objective_sign * raw.obj + state_->objective.constant;
        return ModelSolution(state_, std::move(raw), std::move(primal), objective);
    }

    std::string next_auto_name_() const {
        std::string candidate;
        int next_index = static_cast<int>(state_->vars.size());
        do {
            candidate = "x" + std::to_string(next_index++);
        } while (state_->name_to_index.contains(candidate));
        return candidate;
    }

    void touch_(bool invalidate_basis = false) {
        ++state_->revision;
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_constraint_pi.clear();
        if (invalidate_basis) state_->last_basis.reset();
    }

    std::shared_ptr<ModelState> state_;
};

}  // namespace

PYBIND11_MODULE(simplinho, m) {
    m.doc() = "Bindings for the revised simplex solver";

    py::enum_<LPSolution::Status>(m, "LPStatus")
        .value("Optimal", LPSolution::Status::Optimal)
        .value("Unbounded", LPSolution::Status::Unbounded)
        .value("Infeasible", LPSolution::Status::Infeasible)
        .value("IterLimit", LPSolution::Status::IterLimit)
        .value("Singular", LPSolution::Status::Singular)
        .value("NeedPhase1", LPSolution::Status::NeedPhase1);

    py::enum_<LPBasisStatus>(m, "LPBasisStatus")
        .value("Basic", LPBasisStatus::Basic)
        .value("AtLower", LPBasisStatus::AtLower)
        .value("AtUpper", LPBasisStatus::AtUpper)
        .value("Fixed", LPBasisStatus::Fixed);

    py::enum_<VarType>(m, "VarType")
        .value("Continuous", VarType::Continuous)
        .value("Integer", VarType::Integer)
        .value("Binary", VarType::Binary);

    py::enum_<MIPStatus>(m, "MIPStatus")
        .value("Optimal", MIPStatus::Optimal)
        .value("Infeasible", MIPStatus::Infeasible)
        .value("Unbounded", MIPStatus::Unbounded)
        .value("NodeLimit", MIPStatus::NodeLimit);

    py::enum_<NodeSelectionStrategy>(m, "NodeSelectionStrategy")
        .value("DepthFirst", NodeSelectionStrategy::DepthFirst)
        .value("BreadthFirst", NodeSelectionStrategy::BreadthFirst)
        .value("BestBound", NodeSelectionStrategy::BestBound)
        .value("BestFirst", NodeSelectionStrategy::BestBound)
        .value("BestEstimate", NodeSelectionStrategy::BestEstimate)
        .value("Hybrid", NodeSelectionStrategy::Hybrid);

    py::enum_<BranchingStrategy>(m, "BranchingStrategy")
        .value("MostFractional", BranchingStrategy::MostFractional)
        .value("PseudoCost", BranchingStrategy::PseudoCost)
        .value("StrongBranching", BranchingStrategy::StrongBranching);

    py::enum_<DivingStrategy>(m, "DivingStrategy")
        .value("Disabled", DivingStrategy::Disabled)
        .value("Fractional", DivingStrategy::Fractional)
        .value("VectorLength", DivingStrategy::VectorLength)
        .value("ObjectiveValue", DivingStrategy::ObjectiveValue)
        .value("Coefficient", DivingStrategy::Coefficient)
        .value("Guided", DivingStrategy::Guided)
        .value("Adaptive", DivingStrategy::Adaptive);

    py::enum_<MIPTreeNodeStatus>(m, "MIPTreeNodeStatus")
        .value("Created", MIPTreeNodeStatus::Created)
        .value("Fractional", MIPTreeNodeStatus::Fractional)
        .value("Integral", MIPTreeNodeStatus::Integral)
        .value("Infeasible", MIPTreeNodeStatus::Infeasible)
        .value("Unbounded", MIPTreeNodeStatus::Unbounded)
        .value("PrunedByBound", MIPTreeNodeStatus::PrunedByBound)
        .value("Branched", MIPTreeNodeStatus::Branched)
        .value("Fathomed", MIPTreeNodeStatus::Fathomed);

    py::class_<MIPTreeNode>(m, "MIPTreeNode")
        .def_property_readonly("id", [](const MIPTreeNode& self) { return self.id; })
        .def_property_readonly("parent_id",
                               [](const MIPTreeNode& self) { return self.parent_id; })
        .def_property_readonly("depth", [](const MIPTreeNode& self) { return self.depth; })
        .def_property_readonly("order", [](const MIPTreeNode& self) { return self.order; })
        .def_property_readonly("status", [](const MIPTreeNode& self) { return self.status; })
        .def_property_readonly("bound", [](const MIPTreeNode& self) { return self.bound; })
        .def_property_readonly("estimate", [](const MIPTreeNode& self) { return self.estimate; })
        .def_property_readonly("branch_var",
                               [](const MIPTreeNode& self) { return self.branch_var; })
        .def_property_readonly("branch_value",
                               [](const MIPTreeNode& self) { return self.branch_value; })
        .def("__repr__", [](const MIPTreeNode& self) {
            std::ostringstream oss;
            oss << "MIPTreeNode(id=" << self.id << ", parent_id=" << self.parent_id
                << ", depth=" << self.depth << ", estimate=" << self.estimate
                << ", status='" << simplex_bnb::to_string(self.status) << "')";
            return oss.str();
        });

    py::class_<LPBasis>(m, "LPBasis")
        .def(py::init<>())
        .def_readwrite("column_status", &LPBasis::column_status)
        .def_property_readonly("num_columns", [](const LPBasis& self) {
            return static_cast<int>(self.column_status.size());
        })
        .def_property_readonly("basic_columns", [](const LPBasis& self) {
            std::vector<int> out;
            for (int j = 0; j < static_cast<int>(self.column_status.size()); ++j) {
                if (self.column_status[j] == LPBasisStatus::Basic) out.push_back(j);
            }
            return out;
        })
        .def("__repr__", [](const LPBasis& self) {
            int basics = 0;
            for (const auto status : self.column_status) {
                if (status == LPBasisStatus::Basic) ++basics;
            }
            std::ostringstream oss;
            oss << "LPBasis(num_columns=" << self.column_status.size()
                << ", basics=" << basics << ")";
            return oss.str();
        });

    py::class_<SolveStats>(m, "SolveStats")
        .def_property_readonly("status", [](const SolveStats& self) { return self.status; })
        .def_property_readonly("iterations",
                               [](const SolveStats& self) { return self.iterations; })
        .def_property_readonly("phase1_iterations",
                               [](const SolveStats& self) { return self.phase1_iterations; })
        .def_property_readonly("phase2_iterations",
                               [](const SolveStats& self) { return self.phase2_iterations; })
        .def_property_readonly("presolve_actions",
                               [](const SolveStats& self) { return self.presolve_actions; })
        .def_property_readonly("presolve_implied_bound_updates",
                               [](const SolveStats& self) {
                                   return self.presolve_implied_bound_updates;
                               })
        .def_property_readonly("reduced_rows",
                               [](const SolveStats& self) { return self.reduced_rows; })
        .def_property_readonly("reduced_cols",
                               [](const SolveStats& self) { return self.reduced_cols; })
        .def_property_readonly("objective_shift",
                               [](const SolveStats& self) { return self.objective_shift; })
        .def_property_readonly("input_upper_bounds_relaxed", [](const SolveStats& self) {
            return self.input_upper_bounds_relaxed;
        })
        .def_property_readonly("input_lower_bounds_relaxed", [](const SolveStats& self) {
            return self.input_lower_bounds_relaxed;
        })
        .def_property_readonly("basis_start",
                               [](const SolveStats& self) { return self.basis_start; })
        .def_property_readonly("basis_start_style", [](const SolveStats& self) {
            return self.basis_start_style;
        })
        .def_property_readonly("basis_start_attempt", [](const SolveStats& self) {
            return self.basis_start_attempt;
        })
        .def_property_readonly("basis_start_primal_feasible",
                               [](const SolveStats& self) {
                                   return self.basis_start_primal_feasible;
                               })
        .def_property_readonly("basis_start_dual_feasible",
                               [](const SolveStats& self) {
                                   return self.basis_start_dual_feasible;
                               })
        .def_property_readonly("basis_start_primal_violation",
                               [](const SolveStats& self) {
                                   return self.basis_start_primal_violation;
                               })
        .def_property_readonly("basis_start_dual_violation",
                               [](const SolveStats& self) {
                                   return self.basis_start_dual_violation;
                               })
        .def_property_readonly("phase1_status",
                               [](const SolveStats& self) { return self.phase1_status; })
        .def_property_readonly("reason",
                               [](const SolveStats& self) { return self.reason; })
        .def_property_readonly("note", [](const SolveStats& self) { return self.note; })
        .def_property_readonly("certificate",
                               [](const SolveStats& self) { return self.certificate; })
        .def_property_readonly("dual_pricing",
                               [](const SolveStats& self) { return self.dual_pricing; })
        .def_property_readonly("dual_bfrt_flips",
                               [](const SolveStats& self) { return self.dual_bfrt_flips; })
        .def_property_readonly("degeneracy_streak", [](const SolveStats& self) {
            return self.degeneracy_streak;
        })
        .def_property_readonly("degeneracy_total", [](const SolveStats& self) {
            return self.degeneracy_total;
        })
        .def_property_readonly("suspected_cycle_length", [](const SolveStats& self) {
            return self.suspected_cycle_length;
        })
        .def_property_readonly("condition_estimate", [](const SolveStats& self) {
            return self.condition_estimate;
        })
        .def_property_readonly("degeneracy_threshold", [](const SolveStats& self) {
            return self.degeneracy_threshold;
        })
        .def_property_readonly("degeneracy_epoch",
                               [](const SolveStats& self) { return self.degeneracy_epoch; })
        .def_property_readonly("farkas_has_cert",
                               [](const SolveStats& self) { return self.farkas_has_cert; })
        .def_property_readonly("primal_ray_has_cert", [](const SolveStats& self) {
            return self.primal_ray_has_cert;
        })
        .def_property_readonly("trace_lines",
                               [](const SolveStats& self) { return self.trace_lines; })
        .def_property_readonly("raw_info",
                               [](const SolveStats& self) { return self.raw_info; })
        .def("as_dict", &SolveStats::as_dict)
        .def("__repr__", [](const SolveStats& self) {
            std::ostringstream oss;
            oss << "SolveStats(status='" << self.status << "', iterations="
                << self.iterations << ", trace_lines=" << self.trace_lines << ")";
            return oss.str();
        });

    py::class_<BranchAndBoundOptions>(m, "BranchAndBoundOptions")
        .def(py::init<>())
        .def_readwrite("max_nodes", &BranchAndBoundOptions::max_nodes)
        .def_property(
            "node_limit",
            [](const BranchAndBoundOptions& self) { return self.max_nodes; },
            [](BranchAndBoundOptions& self, int value) { self.max_nodes = value; })
        .def_readwrite("parallel_workers", &BranchAndBoundOptions::parallel_workers)
        .def_readwrite("integrality_tol", &BranchAndBoundOptions::integrality_tol)
        .def_readwrite("verbose", &BranchAndBoundOptions::verbose)
        .def_readwrite("log_frequency", &BranchAndBoundOptions::log_frequency)
        .def_readwrite("node_selection", &BranchAndBoundOptions::node_selection)
        .def_readwrite("hybrid_depth_bias", &BranchAndBoundOptions::hybrid_depth_bias)
        .def_readwrite("branching_strategy", &BranchAndBoundOptions::branching_strategy)
        .def_readwrite("diving_strategy", &BranchAndBoundOptions::diving_strategy)
        .def_readwrite("strong_branching_candidates",
                       &BranchAndBoundOptions::strong_branching_candidates)
        .def_readwrite("strong_branching_max_depth",
                       &BranchAndBoundOptions::strong_branching_max_depth)
        .def_readwrite("pseudocost_reliability",
                       &BranchAndBoundOptions::pseudocost_reliability)
        .def_readwrite("max_dive_depth", &BranchAndBoundOptions::max_dive_depth)
        .def_readwrite("max_dive_lp_solves", &BranchAndBoundOptions::max_dive_lp_solves)
        .def_readwrite("heuristic_frequency", &BranchAndBoundOptions::heuristic_frequency)
        .def_readwrite("heuristic_max_depth", &BranchAndBoundOptions::heuristic_max_depth)
        .def_readwrite("use_rins", &BranchAndBoundOptions::use_rins)
        .def_readwrite("rins_fix_ratio", &BranchAndBoundOptions::rins_fix_ratio)
        .def_readwrite("rins_tolerance", &BranchAndBoundOptions::rins_tolerance)
        .def_readwrite("use_rens", &BranchAndBoundOptions::use_rens)
        .def_readwrite("rens_fix_ratio", &BranchAndBoundOptions::rens_fix_ratio)
        .def_readwrite("use_local_search", &BranchAndBoundOptions::use_local_search)
        .def_readwrite("local_search_iterations",
                       &BranchAndBoundOptions::local_search_iterations)
        .def_readwrite("local_search_max_free_vars",
                       &BranchAndBoundOptions::local_search_max_free_vars)
        .def_readwrite("use_local_branching",
                       &BranchAndBoundOptions::use_local_branching)
        .def_readwrite("local_branching_neighborhood_ratio",
                       &BranchAndBoundOptions::local_branching_neighborhood_ratio)
        .def_readwrite("local_branching_min_radius",
                       &BranchAndBoundOptions::local_branching_min_radius)
        .def_readwrite("local_branching_max_radius",
                       &BranchAndBoundOptions::local_branching_max_radius)
        .def_readwrite("local_branching_fix_agree_ratio",
                       &BranchAndBoundOptions::local_branching_fix_agree_ratio)
        .def_readwrite("local_branching_lp_agreement_tol",
                       &BranchAndBoundOptions::local_branching_lp_agreement_tol)
        .def_readwrite("use_feasibility_pump",
                       &BranchAndBoundOptions::use_feasibility_pump)
        .def_readwrite("feasibility_pump_iterations",
                       &BranchAndBoundOptions::feasibility_pump_iterations)
        .def_readwrite("feasibility_pump_fix_ratio",
                       &BranchAndBoundOptions::feasibility_pump_fix_ratio)
        .def_readwrite("use_feasibility_jump",
                       &BranchAndBoundOptions::use_feasibility_jump)
        .def_readwrite("feasibility_jump_iterations",
                       &BranchAndBoundOptions::feasibility_jump_iterations)
        .def_readwrite("feasibility_jump_max_free_vars",
                       &BranchAndBoundOptions::feasibility_jump_max_free_vars)
        .def_readwrite("feasibility_jump_objective_weight",
                       &BranchAndBoundOptions::feasibility_jump_objective_weight)
        .def_readwrite("heuristic_subproblem_max_nodes",
                       &BranchAndBoundOptions::heuristic_subproblem_max_nodes)
        .def_readwrite("use_cut_pool", &BranchAndBoundOptions::use_cut_pool)
        .def_readwrite("max_cut_rounds_per_node",
                       &BranchAndBoundOptions::max_cut_rounds_per_node)
        .def_readwrite("max_cuts_added_per_round",
                       &BranchAndBoundOptions::max_cuts_added_per_round)
        .def_readwrite("max_cut_pool_size", &BranchAndBoundOptions::max_cut_pool_size)
        .def_readwrite("min_cut_violation", &BranchAndBoundOptions::min_cut_violation)
        .def_readwrite("max_cut_age", &BranchAndBoundOptions::max_cut_age)
        .def_readwrite("use_gomory_cuts", &BranchAndBoundOptions::use_gomory_cuts)
        .def_readwrite("use_cover_cuts", &BranchAndBoundOptions::use_cover_cuts)
        .def_readwrite("use_implied_bound_cuts",
                       &BranchAndBoundOptions::use_implied_bound_cuts)
        .def_readwrite("use_clique_cuts", &BranchAndBoundOptions::use_clique_cuts)
        .def_readwrite("use_probing_implications",
                       &BranchAndBoundOptions::use_probing_implications)
        .def_readwrite("probing_max_candidates",
                       &BranchAndBoundOptions::probing_max_candidates)
        .def_readwrite("use_conflict_cuts", &BranchAndBoundOptions::use_conflict_cuts)
        .def_readwrite("max_conflict_cuts_per_round",
                       &BranchAndBoundOptions::max_conflict_cuts_per_round)
        .def_readwrite("max_cuts_per_type", &BranchAndBoundOptions::max_cuts_per_type)
        .def_readwrite("cut_max_parallelism", &BranchAndBoundOptions::cut_max_parallelism)
        .def("__repr__", [](const BranchAndBoundOptions& self) {
            std::ostringstream oss;
            oss << "BranchAndBoundOptions(max_nodes=" << self.max_nodes
                << ", integrality_tol=" << self.integrality_tol
                << ", verbose=" << (self.verbose ? "True" : "False")
                << ", log_frequency=" << self.log_frequency
                << ", node_selection='" << simplex_bnb::to_string(self.node_selection) << "'"
                << ", branching_strategy='" << simplex_bnb::to_string(self.branching_strategy)
                << "', diving_strategy='" << simplex_bnb::to_string(self.diving_strategy)
                << "')";
            return oss.str();
        });

    // -----------------------------------------------------------------------
    // LPSolution
    //
    // Attributes come in two groups:
    //
    //   Original space  — indexed over the columns/rows of the A matrix you
    //                     passed to solve().  Use these for sensitivity
    //                     analysis, warm starts, and Gomory cuts.
    //
    //   Internal space  — indexed over the presolve-reduced problem that the
    //                     simplex engine actually solved.  Column k in the
    //                     internal problem corresponds to original column
    //                     internal_column_labels[k].  Use these for
    //                     debugging, advanced cut generation from the
    //                     reduced tableau, or when presolve changes matter.
    // -----------------------------------------------------------------------
    py::class_<LPSolution>(m, "LPSolution")
        // ── Status / objective ────────────────────────────────────────────
        .def_readonly("status", &LPSolution::status,
            "Solve status (LPSolution.Status enum).")
        .def_readonly("obj", &LPSolution::obj,
            "Optimal objective value; NaN when infeasible or unbounded.")
        .def_readonly("iters", &LPSolution::iters,
            "Total simplex iterations (Phase I + Phase II).")

        // ── Primal solution (original space) ──────────────────────────────
        .def_readonly("x", &LPSolution::x,
            "Primal solution vector x, length n (original columns).\n"
            "NaN entries indicate infeasible or unbounded.")
        .def_readonly("basis", &LPSolution::basis,
            "List of basic column indices in the *original* problem.\n"
            "len(basis) == m when all basis variables map to original columns\n"
            "(always true for standard-form problems).  Use this together with\n"
            "the original A, b, c to reconstruct the original-space tableau\n"
            "for Gomory cuts or sensitivity analysis:\n"
            "    B = A[:, sol.basis]; T = np.linalg.solve(B, A)")

        // ── Dual / reduced costs (original space) ─────────────────────────
        .def_readonly("dual_values", &LPSolution::dual_values,
            "Dual variables y = B^{-T} c_B, length m (original rows).\n"
            "These are the shadow prices on the original equality constraints.\n"
            "Reduced costs in the original space: c - A^T @ dual_values.")

        // ── Certificates ──────────────────────────────────────────────────
        .def_readonly("farkas_y", &LPSolution::farkas_y,
            "Farkas infeasibility certificate in the original row space.\n"
            "Valid only when farkas_has_cert is True.")
        .def_readonly("farkas_has_cert", &LPSolution::farkas_has_cert,
            "True when a Farkas certificate of infeasibility is available.")
        .def_readonly("primal_ray", &LPSolution::primal_ray,
            "Primal unbounded ray in the original column space.\n"
            "Valid only when primal_ray_has_cert is True.")
        .def_readonly("primal_ray_has_cert", &LPSolution::primal_ray_has_cert,
            "True when a primal unbounded ray certificate is available.")

        // ── Warm-start basis ──────────────────────────────────────────────
        .def_property_readonly("basis_state", [](const LPSolution& self) {
            return rebuild_basis_from_solution(self);
        }, "LPBasis for warm-starting a subsequent solve on the same problem\n"
           "structure.  Stores Basic/AtLower/AtUpper/Fixed status per original\n"
           "column.  Pass to solver.solve(..., basis_state) or model.reoptimize(basis_state).")

        // ── Internal (reduced) space ───────────────────────────────────────
        // The solver applies presolve transformations and bound shifts before
        // running the simplex.  The fields below are expressed in terms of
        // the *reduced* problem, whose columns are a subset (and reordering)
        // of the original columns.  internal_column_labels[k] gives the
        // original column index for internal column k.
        .def_readonly("basis_internal", &LPSolution::basis_internal,
            "Basic column indices in the internal (presolve-reduced) problem.")
        .def_readonly("nonbasis_internal", &LPSolution::nonbasis_internal,
            "Nonbasic column indices in the internal problem.")
        .def_readonly("internal_column_labels", &LPSolution::internal_column_labels,
            "internal_column_labels[k] is the original column index for internal column k.\n"
            "Use to map internal-space results back to original variables.")
        .def_readonly("internal_row_labels", &LPSolution::internal_row_labels,
            "internal_row_labels[i] is the original row index for internal row i.")
        .def_readonly("tableau_internal", &LPSolution::tableau,
            "B^{-1} A in the internal problem, shape (m_int, n_int).\n"
            "tableau_internal[:, basis_internal] == I (identity on basis columns).\n"
            "Non-empty only when has_tableau is True.")
        .def_readonly("tableau_rhs_internal", &LPSolution::tableau_rhs,
            "B^{-1} b in the internal problem, length m_int.\n"
            "Gives the basic variable values in the shifted/reduced space.\n"
            "Non-empty only when has_tableau is True.")
        .def_readonly("reduced_costs_internal", &LPSolution::reduced_costs_internal,
            "Reduced costs c - A^T y in the internal problem, length n_int.")
        .def_readonly("dual_values_internal", &LPSolution::dual_values_internal,
            "Dual variables y = B^{-T} c_B in the internal problem, length m_int.")
        .def_property_readonly("has_tableau", [](const LPSolution& self) {
            return self.has_internal_tableau;
        }, "True when tableau_internal and tableau_rhs_internal are populated.\n"
           "They are empty for infeasible/unbounded solves.")
        .def_readonly("farkas_y_internal", &LPSolution::farkas_y_internal,
            "Farkas certificate in the internal row space (debug use).")
        .def_readonly("primal_ray_internal", &LPSolution::primal_ray_internal,
            "Primal ray in the internal column space (debug use).")

        // ── Diagnostics / logging ─────────────────────────────────────────
        .def_readonly("info", &LPSolution::info,
            "Raw key-value telemetry dict (string → string).  Use stats for\n"
            "a typed, stable interface.")
        .def_property_readonly("stats", [](const LPSolution& self) {
            return build_solve_stats(self);
        }, "SolveStats object with typed fields (iterations, basis_start, etc.).")
        .def_property_readonly("log_lines", [](const LPSolution& self) {
            return self.trace;
        }, "List of verbose trace lines emitted during the solve\n"
           "(populated only when options.verbose = True).")
        .def_property_readonly("log", [](const LPSolution& self) {
            return join_trace_lines(self.trace);
        }, "Verbose trace joined into a single newline-delimited string.")
        .def("__repr__", [](const LPSolution& self) {
            std::ostringstream oss;
            oss << "LPSolution(status=" << to_string(self.status)
                << ", obj=";
            if (std::isfinite(self.obj))
                oss << self.obj;
            else
                oss << (std::isnan(self.obj) ? "nan" : (self.obj > 0 ? "inf" : "-inf"));
            oss << ", iters=" << self.iters
                << ", basis_size=" << self.basis.size() << ")";
            return oss.str();
        });

    py::class_<RevisedSimplexOptions>(m, "RevisedSimplexOptions")
        .def(py::init<>())
        .def_readwrite("max_iters", &RevisedSimplexOptions::max_iters)
        .def_readwrite("tol", &RevisedSimplexOptions::tol)
        .def_readwrite("bland", &RevisedSimplexOptions::bland)
        .def_readwrite("svd_tol", &RevisedSimplexOptions::svd_tol)
        .def_readwrite("ratio_delta", &RevisedSimplexOptions::ratio_delta)
        .def_readwrite("ratio_eta", &RevisedSimplexOptions::ratio_eta)
        .def_readwrite("deg_step_tol", &RevisedSimplexOptions::deg_step_tol)
        .def_readwrite("epsilon_cost", &RevisedSimplexOptions::epsilon_cost)
        .def_readwrite("rng_seed", &RevisedSimplexOptions::rng_seed)
        .def_readwrite("refactor_every", &RevisedSimplexOptions::refactor_every)
        .def_readwrite("compress_every", &RevisedSimplexOptions::compress_every)
        .def_readwrite("lu_pivot_rel", &RevisedSimplexOptions::lu_pivot_rel)
        .def_readwrite("lu_abs_floor", &RevisedSimplexOptions::lu_abs_floor)
        .def_readwrite("alpha_tol", &RevisedSimplexOptions::alpha_tol)
        .def_readwrite("z_inf_guard", &RevisedSimplexOptions::z_inf_guard)
        .def_readwrite("basis_update", &RevisedSimplexOptions::basis_update)
        .def_readwrite("ft_bandwidth_cap", &RevisedSimplexOptions::ft_bandwidth_cap)
        .def_readwrite("devex_reset", &RevisedSimplexOptions::devex_reset)
        .def_readwrite("pricing_rule", &RevisedSimplexOptions::pricing_rule)
        .def_readwrite("adaptive_reset_freq", &RevisedSimplexOptions::adaptive_reset_freq)
        .def_readwrite("partial_pricing", &RevisedSimplexOptions::partial_pricing)
        .def_readwrite("dual_pricing", &RevisedSimplexOptions::dual_pricing)
        .def_readwrite("row_pricing_threshold",
                       &RevisedSimplexOptions::row_pricing_threshold)
        .def_readwrite("primal_edge_weight_strategy",
                       &RevisedSimplexOptions::primal_edge_weight_strategy)
        .def_readwrite("dual_edge_weight_strategy",
                       &RevisedSimplexOptions::dual_edge_weight_strategy)
        .def_readwrite("primal_steepest_edge_weight_log_error_threshold",
                       &RevisedSimplexOptions::
                           primal_steepest_edge_weight_log_error_threshold)
        .def_readwrite("dual_steepest_edge_weight_log_error_threshold",
                       &RevisedSimplexOptions::
                           dual_steepest_edge_weight_log_error_threshold)
        .def_readwrite("primal_simplex_cost_perturbation_multiplier",
                       &RevisedSimplexOptions::
                           primal_simplex_cost_perturbation_multiplier)
        .def_readwrite("dual_simplex_cost_perturbation_multiplier",
                       &RevisedSimplexOptions::
                           dual_simplex_cost_perturbation_multiplier)
        .def_readwrite("max_basis_rebuilds", &RevisedSimplexOptions::max_basis_rebuilds)
        .def_readwrite("crash_attempts", &RevisedSimplexOptions::crash_attempts)
        .def_readwrite("crash_markowitz_tol", &RevisedSimplexOptions::crash_markowitz_tol)
        .def_readwrite("crash_strategy", &RevisedSimplexOptions::crash_strategy)
        .def_readwrite("repair_mapped_basis", &RevisedSimplexOptions::repair_mapped_basis)
        .def_readwrite("dual_allow_bound_flip", &RevisedSimplexOptions::dual_allow_bound_flip)
        .def_readwrite("dual_flip_pivot_tol", &RevisedSimplexOptions::dual_flip_pivot_tol)
        .def_readwrite("dual_flip_rc_tol", &RevisedSimplexOptions::dual_flip_rc_tol)
        .def_readwrite("dual_flip_max_per_iter", &RevisedSimplexOptions::dual_flip_max_per_iter)
        .def_readwrite("verbose", &RevisedSimplexOptions::verbose)
        .def_readwrite("verbose_every", &RevisedSimplexOptions::verbose_every)
        .def_readwrite("verbose_include_basis", &RevisedSimplexOptions::verbose_include_basis)
        .def_readwrite("verbose_include_presolve", &RevisedSimplexOptions::verbose_include_presolve)
        .def_readwrite("mode", &RevisedSimplexOptions::mode);

    py::enum_<SimplexMode>(m, "SimplexMode")
        .value("Auto", SimplexMode::Auto)
        .value("Primal", SimplexMode::Primal)
        .value("Dual", SimplexMode::Dual);

    py::enum_<ConstraintSense>(m, "ConstraintSense")
        .value("LessEqual", ConstraintSense::LessEqual)
        .value("Equal", ConstraintSense::Equal)
        .value("GreaterEqual", ConstraintSense::GreaterEqual);

    py::class_<Var>(m, "Var")
        .def_property_readonly("name", &Var::name)
        .def_property("lb", &Var::lower_bound, &Var::set_lower_bound)
        .def_property("ub", &Var::upper_bound, &Var::set_upper_bound)
        .def_property("type", &Var::type, &Var::set_type)
        .def_property("obj", &Var::objective_coefficient,
                      &Var::set_objective_coefficient)
        .def("__repr__", &Var::repr)
        .def("__add__", [](const Var& self, const Var& other) {
            return add_expr(to_expr(self), to_expr(other));
        }, py::is_operator())
        .def("__add__", [](const Var& self, const LinearExpr& other) {
            return add_expr(to_expr(self), other);
        }, py::is_operator())
        .def("__add__", [](const Var& self, double other) {
            return add_expr(to_expr(self), make_constant_expr(self.state(), other));
        }, py::is_operator())
        .def("__radd__", [](const Var& self, double other) {
            return add_expr(make_constant_expr(self.state(), other), to_expr(self));
        }, py::is_operator())
        .def("__sub__", [](const Var& self, const Var& other) {
            return sub_expr(to_expr(self), to_expr(other));
        }, py::is_operator())
        .def("__sub__", [](const Var& self, const LinearExpr& other) {
            return sub_expr(to_expr(self), other);
        }, py::is_operator())
        .def("__sub__", [](const Var& self, double other) {
            return sub_expr(to_expr(self), make_constant_expr(self.state(), other));
        }, py::is_operator())
        .def("__rsub__", [](const Var& self, double other) {
            return sub_expr(make_constant_expr(self.state(), other), to_expr(self));
        }, py::is_operator())
        .def("__mul__", [](const Var& self, double scalar) {
            return scale_expr(to_expr(self), scalar);
        }, py::is_operator())
        .def("__rmul__", [](const Var& self, double scalar) {
            return scale_expr(to_expr(self), scalar);
        }, py::is_operator())
        .def("__neg__", [](const Var& self) { return scale_expr(to_expr(self), -1.0); },
             py::is_operator())
        .def("__le__", [](const Var& self, const Var& other) {
            return compare_exprs(to_expr(self), to_expr(other),
                                 ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__le__", [](const Var& self, const LinearExpr& other) {
            return compare_exprs(to_expr(self), other, ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__le__", [](const Var& self, double other) {
            return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                 ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__ge__", [](const Var& self, const Var& other) {
            return compare_exprs(to_expr(self), to_expr(other),
                                 ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__ge__", [](const Var& self, const LinearExpr& other) {
            return compare_exprs(to_expr(self), other, ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__ge__", [](const Var& self, double other) {
            return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                 ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__eq__", [](const Var& self, const Var& other) {
            return compare_exprs(to_expr(self), to_expr(other), ConstraintSense::Equal);
        }, py::is_operator())
        .def("__eq__", [](const Var& self, const LinearExpr& other) {
            return compare_exprs(to_expr(self), other, ConstraintSense::Equal);
        }, py::is_operator())
        .def("__eq__", [](const Var& self, double other) {
            return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                 ConstraintSense::Equal);
        }, py::is_operator());

    py::class_<LinearExpr>(m, "LinearExpr")
        .def(py::init<>())
        .def(py::init<double>(), py::arg("constant"))
        .def("__repr__", &LinearExpr::repr)
        .def("__add__", [](const LinearExpr& self, const LinearExpr& other) {
            return add_expr(self, other);
        }, py::is_operator())
        .def("__add__", [](const LinearExpr& self, const Var& other) {
            return add_expr(self, to_expr(other));
        }, py::is_operator())
        .def("__add__", [](const LinearExpr& self, double other) {
            return add_expr(self, make_constant_expr(self.state(), other));
        }, py::is_operator())
        .def("__radd__", [](const LinearExpr& self, double other) {
            return add_expr(make_constant_expr(self.state(), other), self);
        }, py::is_operator())
        .def("__sub__", [](const LinearExpr& self, const LinearExpr& other) {
            return sub_expr(self, other);
        }, py::is_operator())
        .def("__sub__", [](const LinearExpr& self, const Var& other) {
            return sub_expr(self, to_expr(other));
        }, py::is_operator())
        .def("__sub__", [](const LinearExpr& self, double other) {
            return sub_expr(self, make_constant_expr(self.state(), other));
        }, py::is_operator())
        .def("__rsub__", [](const LinearExpr& self, double other) {
            return sub_expr(make_constant_expr(self.state(), other), self);
        }, py::is_operator())
        .def("__mul__", [](const LinearExpr& self, double scalar) {
            return scale_expr(self, scalar);
        }, py::is_operator())
        .def("__rmul__", [](const LinearExpr& self, double scalar) {
            return scale_expr(self, scalar);
        }, py::is_operator())
        .def("__neg__", [](const LinearExpr& self) { return scale_expr(self, -1.0); },
             py::is_operator())
        .def("__le__", [](const LinearExpr& self, const LinearExpr& other) {
            return compare_exprs(self, other, ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__le__", [](const LinearExpr& self, const Var& other) {
            return compare_exprs(self, to_expr(other), ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__le__", [](const LinearExpr& self, double other) {
            return compare_exprs(self, make_constant_expr(self.state(), other),
                                 ConstraintSense::LessEqual);
        }, py::is_operator())
        .def("__ge__", [](const LinearExpr& self, const LinearExpr& other) {
            return compare_exprs(self, other, ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__ge__", [](const LinearExpr& self, const Var& other) {
            return compare_exprs(self, to_expr(other), ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__ge__", [](const LinearExpr& self, double other) {
            return compare_exprs(self, make_constant_expr(self.state(), other),
                                 ConstraintSense::GreaterEqual);
        }, py::is_operator())
        .def("__eq__", [](const LinearExpr& self, const LinearExpr& other) {
            return compare_exprs(self, other, ConstraintSense::Equal);
        }, py::is_operator())
        .def("__eq__", [](const LinearExpr& self, const Var& other) {
            return compare_exprs(self, to_expr(other), ConstraintSense::Equal);
        }, py::is_operator())
        .def("__eq__", [](const LinearExpr& self, double other) {
            return compare_exprs(self, make_constant_expr(self.state(), other),
                                 ConstraintSense::Equal);
        }, py::is_operator());

    py::implicitly_convertible<Var, LinearExpr>();

    py::class_<ConstraintSpec>(m, "Constraint")
        .def("__repr__", &ConstraintSpec::repr)
        .def("__bool__", [](const ConstraintSpec&) {
            throw std::runtime_error(
                "simplex: constraint objects cannot be used as booleans; "
                "add chained comparisons as separate constraints");
        });

    py::class_<ConstraintHandle>(m, "ConstraintHandle")
        .def_property_readonly("pi", &ConstraintHandle::pi)
        .def_property_readonly("name", &ConstraintHandle::name)
        .def_property("rhs", &ConstraintHandle::rhs, &ConstraintHandle::set_rhs)
        .def_property("sense", &ConstraintHandle::sense, &ConstraintHandle::set_sense)
        .def_property_readonly("index", &ConstraintHandle::index)
        .def("get_coeff", &ConstraintHandle::coefficient, py::arg("var"))
        .def("getCoeff", &ConstraintHandle::coefficient, py::arg("var"))
        .def("set_coeff", &ConstraintHandle::set_coefficient, py::arg("var"),
             py::arg("value"))
        .def("setCoeff", &ConstraintHandle::set_coefficient, py::arg("var"),
             py::arg("value"))
        .def("__repr__", &ConstraintHandle::repr);

    py::class_<ModelSolution>(m, "ModelSolution")
        .def_property_readonly("raw", &ModelSolution::raw,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("status", &ModelSolution::status)
        .def_property_readonly("x", &ModelSolution::x,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("obj", &ModelSolution::objective)
        .def_property_readonly("objective", &ModelSolution::objective)
        .def_property_readonly("iters", &ModelSolution::iterations)
        .def_property_readonly("values", &ModelSolution::values,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("stats", &ModelSolution::stats)
        .def_property_readonly("basis", &ModelSolution::basis)
        .def_property_readonly("log_lines", &ModelSolution::log_lines,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("log", &ModelSolution::log)
        .def("value", py::overload_cast<const Var&>(&ModelSolution::value, py::const_),
             py::arg("var"))
        .def("value",
             py::overload_cast<const std::string&>(&ModelSolution::value, py::const_),
             py::arg("name"))
        .def("__repr__", &ModelSolution::repr);

    py::class_<MIPSolution>(m, "MIPSolution")
        .def_property_readonly("status", &MIPSolution::status)
        .def_property_readonly("x", &MIPSolution::x,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("obj", &MIPSolution::objective)
        .def_property_readonly("objective", &MIPSolution::objective)
        .def_property_readonly("best_bound", &MIPSolution::best_bound)
        .def_property_readonly("root_relaxation_objective",
                               &MIPSolution::root_relaxation_objective)
        .def_property_readonly("root_presolve_tightened_bounds",
                               &MIPSolution::root_presolve_tightened_bounds)
        .def_property_readonly("root_presolve_removed_rows",
                               &MIPSolution::root_presolve_removed_rows)
        .def_property_readonly("root_presolve_removed_coeffs",
                               &MIPSolution::root_presolve_removed_coeffs)
        .def_property_readonly("root_presolve_aggregations",
                               &MIPSolution::root_presolve_aggregations)
        .def_property_readonly("node_count", &MIPSolution::node_count)
        .def_property_readonly("lp_iterations", &MIPSolution::lp_iterations)
        .def_property_readonly("incumbent_updates", &MIPSolution::incumbent_updates)
        .def_property_readonly("heuristic_lp_iterations",
                               &MIPSolution::heuristic_lp_iterations)
        .def_property_readonly("heuristic_successes",
                               &MIPSolution::heuristic_successes)
        .def_property_readonly("feasibility_jump_successes",
                               &MIPSolution::feasibility_jump_successes)
        .def_property_readonly("feasibility_pump_successes",
                               &MIPSolution::feasibility_pump_successes)
        .def_property_readonly("rens_successes", &MIPSolution::rens_successes)
        .def_property_readonly("rins_successes", &MIPSolution::rins_successes)
        .def_property_readonly("local_search_successes",
                               &MIPSolution::local_search_successes)
        .def_property_readonly("local_branching_successes",
                               &MIPSolution::local_branching_successes)
        .def_property_readonly("cuts_generated", &MIPSolution::cuts_generated)
        .def_property_readonly("cuts_applied", &MIPSolution::cuts_applied)
        .def_property_readonly("duplicate_cuts", &MIPSolution::duplicate_cuts)
        .def_property_readonly("cut_pool_size", &MIPSolution::cut_pool_size)
        .def_property_readonly("has_solution", &MIPSolution::has_solution)
        .def_property_readonly("values", &MIPSolution::values,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("tree_nodes", &MIPSolution::tree_nodes,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("relative_gap", &MIPSolution::relative_gap)
        .def("value", py::overload_cast<const Var&>(&MIPSolution::value, py::const_),
             py::arg("var"))
        .def("value",
             py::overload_cast<const std::string&>(&MIPSolution::value, py::const_),
             py::arg("name"))
        .def("__repr__", &MIPSolution::repr);

    py::class_<Model>(m, "Model")
        .def(py::init<const RevisedSimplexOptions&>(),
             py::arg("options") = RevisedSimplexOptions())
        .def("add_var", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("addVar", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("addvar", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("add_integer_var", &Model::add_integer_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("addIntegerVar", &Model::add_integer_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("add_binary_var", &Model::add_binary_var, py::arg("name") = py::none(),
             py::arg("obj") = 0.0)
        .def("addBinaryVar", &Model::add_binary_var, py::arg("name") = py::none(),
             py::arg("obj") = 0.0)
        .def("add_constr", &Model::add_constr, py::arg("constraint"),
             py::arg("name") = py::none())
        .def("addConstr", &Model::add_constr, py::arg("constraint"),
             py::arg("name") = py::none())
        .def("set_objective", &Model::set_objective, py::arg("expr"),
             py::arg("sense") = "min")
        .def("set_objective",
             [](Model& self, const Var& var, const std::string& sense) {
                 self.set_objective(to_expr(var), sense);
             },
             py::arg("expr"), py::arg("sense") = "min")
        .def("setObjective", &Model::set_objective, py::arg("expr"),
             py::arg("sense") = "min")
        .def("setObjective",
             [](Model& self, const Var& var, const std::string& sense) {
                 self.set_objective(to_expr(var), sense);
             },
             py::arg("expr"), py::arg("sense") = "min")
        .def("minimize", &Model::minimize, py::arg("expr"))
        .def("minimize",
             [](Model& self, const Var& var) { self.minimize(to_expr(var)); },
             py::arg("expr"))
        .def("maximize", &Model::maximize, py::arg("expr"))
        .def("maximize",
             [](Model& self, const Var& var) { self.maximize(to_expr(var)); },
             py::arg("expr"))
        .def("get_var", &Model::get_var, py::arg("name"))
        .def("getVar", &Model::get_var, py::arg("name"))
        .def("get_obj_coeff", &Model::get_obj_coeff, py::arg("var"))
        .def("getObjCoeff", &Model::get_obj_coeff, py::arg("var"))
        .def("set_obj_coeff", &Model::set_obj_coeff, py::arg("var"),
             py::arg("value"))
        .def("setObjCoeff", &Model::set_obj_coeff, py::arg("var"),
             py::arg("value"))
        .def("get_coeff", &Model::get_coeff, py::arg("constraint"),
             py::arg("var"))
        .def("getCoeff", &Model::get_coeff, py::arg("constraint"),
             py::arg("var"))
        .def("set_coeff", &Model::set_coeff, py::arg("constraint"),
             py::arg("var"), py::arg("value"))
        .def("setCoeff", &Model::set_coeff, py::arg("constraint"),
             py::arg("var"), py::arg("value"))
        .def("set_rhs", &Model::set_rhs, py::arg("constraint"), py::arg("rhs"))
        .def("setRhs", &Model::set_rhs, py::arg("constraint"), py::arg("rhs"))
        .def("delete_var", &Model::delete_var, py::arg("var"))
        .def("deleteVar", &Model::delete_var, py::arg("var"))
        .def("remove_var", &Model::delete_var, py::arg("var"))
        .def("removeVar", &Model::delete_var, py::arg("var"))
        .def("delete_constr", &Model::delete_constr, py::arg("constraint"))
        .def("deleteConstr", &Model::delete_constr, py::arg("constraint"))
        .def("remove_constr", &Model::delete_constr, py::arg("constraint"))
        .def("removeConstr", &Model::delete_constr, py::arg("constraint"))
        .def_property_readonly("num_vars", &Model::num_vars)
        .def_property_readonly("num_constraints", &Model::num_constraints)
        .def_property_readonly(
            "options",
            [](Model& self) -> RevisedSimplexOptions& { return self.options(); },
            py::return_value_policy::reference_internal)
        .def(
            "solve",
            [](const Model& self, py::object basis) {
                if (basis.is_none()) return self.solve();
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(basis.cast<LPBasis>());
                }
                throw std::invalid_argument(
                    "simplex: model.solve basis must be an LPBasis");
            },
            py::arg("basis") = py::none())
        .def(
            "reoptimize",
            [](const Model& self, py::object basis) {
                if (basis.is_none()) return self.reoptimize();
                if (py::isinstance<LPBasis>(basis)) {
                    return self.reoptimize(basis.cast<LPBasis>());
                }
                throw std::invalid_argument(
                    "simplex: model.reoptimize basis must be an LPBasis");
            },
            py::arg("basis") = py::none())
        .def("solve_mip", &Model::solve_mip,
             py::arg("options") = BranchAndBoundOptions())
        .def("__repr__", &Model::repr);

    py::class_<RevisedSimplex>(m, "RevisedSimplex")
        .def(py::init<const RevisedSimplexOptions&>(),
             py::arg("options") = RevisedSimplexOptions())
        .def("clear_basis_cache", &RevisedSimplex::clear_basis_cache)
        .def("clearBasisCache", &RevisedSimplex::clear_basis_cache)
        .def(
            "solve",
            [](RevisedSimplex& self, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c,
               const Eigen::VectorXd& l, const Eigen::VectorXd& u,
               py::object basis) {
                if (basis.is_none()) {
                    return self.solve(A, b, c, l, u);
                }
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(A, b, c, l, u, basis.cast<LPBasis>());
                }
                return self.solve(A, b, c, l, u,
                                  basis.cast<std::vector<int>>());
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"), py::arg("u"),
            py::arg("basis") = py::none(),
            "Solve LP: min c^T x s.t. Ax=b, l<=x<=u")
        .def(
            "solve",
            [](RevisedSimplex& self, const RevisedSimplex::SparseMatrix& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c,
               const Eigen::VectorXd& l, const Eigen::VectorXd& u,
               py::object basis) {
                if (basis.is_none()) {
                    return self.solve(A, b, c, l, u);
                }
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(A, b, c, l, u, basis.cast<LPBasis>());
                }
                return self.solve(A, b, c, l, u,
                                  basis.cast<std::vector<int>>());
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"), py::arg("u"),
            py::arg("basis") = py::none(),
            "Solve LP: min c^T x s.t. Ax=b, l<=x<=u");

    m.attr("SimplexModel") = m.attr("Model");
    m.def("status_to_string", [](LPSolution::Status status) {
        return std::string(to_string(status));
    });
    m.def("mip_status_to_string", [](MIPStatus status) {
        return std::string(simplex_bnb::to_string(status));
    });

    // HiGHS-inspired: cost perturbation utility for degeneracy handling
    m.def(
        "perturb_costs",
        [](Eigen::VectorXd c, double multiplier, int seed) {
            std::mt19937 rng(seed);
            degeneracy_helpers::perturbCosts(c, rng, multiplier);
            return c;
        },
        py::arg("costs"), py::arg("multiplier") = 1e-8, py::arg("seed") = 13,
        "Apply small random perturbations to cost vector to break degeneracy");
}
