#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#if defined(__unix__) || defined(__APPLE__)
#    include <unistd.h>
#endif

#include "bindings.h"
#include "simplex/bnb.h"
#include "simplex/simplex.h"
#include "solve_stats.h"

#ifndef SIMPLEX_PROJECT_VERSION
#    define SIMPLEX_PROJECT_VERSION "unknown"
#endif

#ifndef SIMPLEX_GIT_DESCRIBE
#    define SIMPLEX_GIT_DESCRIBE "unknown"
#endif

#ifndef SIMPLEX_GIT_BRANCH
#    define SIMPLEX_GIT_BRANCH "unknown"
#endif

namespace py = pybind11;
namespace simplex_bnb = simplex::bnb;

namespace {

constexpr double kCoeffTol = 1e-12;

bool log_colors_enabled_() {
    static const bool enabled = []() {
        if (std::getenv("NO_COLOR") != nullptr) {
            return false;
        }
        const char* term = std::getenv("TERM");
        if (term == nullptr || std::string_view(term) == "dumb") {
            return false;
        }
#if defined(__unix__) || defined(__APPLE__)
        return ::isatty(fileno(stdout)) != 0;
#else
        return false;
#endif
    }();
    return enabled;
}

std::string colorize_(std::string_view text, std::string_view ansi_code) {
    if (!log_colors_enabled_()) {
        return std::string(text);
    }
    std::string out;
    out.reserve(text.size() + ansi_code.size() + 10);
    out += "\033[";
    out += ansi_code;
    out += "m";
    out += text;
    out += "\033[0m";
    return out;
}

std::string accent_(std::string_view text) { return colorize_(text, "38;5;39"); }
std::string bold_(std::string_view text) { return colorize_(text, "1"); }
std::string dim_(std::string_view text) { return colorize_(text, "2"); }
std::string good_(std::string_view text) { return colorize_(text, "38;5;40"); }
std::string warn_(std::string_view text) { return colorize_(text, "38;5;214"); }

std::string rule_(char ch = '=') { return std::string(62, ch); }

void print_verbose_solver_banner() {
    std::cout << dim_(rule_()) << std::endl;
    std::cout << bold_("Simplinho") << " " << SIMPLEX_PROJECT_VERSION << "  " << dim_("git") << " "
              << SIMPLEX_GIT_DESCRIBE << "  " << dim_("branch") << " " << SIMPLEX_GIT_BRANCH
              << std::endl;
}

enum class ConstraintSense { LessEqual, Equal, GreaterEqual };
using VarType = simplex_bnb::VariableType;
using MIPStatus = simplex_bnb::Status;
using NodeSelectionStrategy = simplex_bnb::NodeSelectionStrategy;
using BranchingStrategy = simplex_bnb::BranchingStrategy;
using DivingStrategy = simplex_bnb::DivingStrategy;
using BranchAndBoundOptions = simplex_bnb::Options;
using MIPTreeNode = simplex_bnb::TreeNode;
using MIPTreeNodeStatus = simplex_bnb::TreeNodeStatus;
using SimplifiedCutsResult = simplex_bnb::presolve::SimplifiedCutsResult;
using NodeBoundPresolveResult = simplex_bnb::presolve::NodeBoundPresolveResult;
using RootProblemPresolveResult = simplex_bnb::presolve::RootProblemPresolveResult;
using simplex_bnb::presolve::cut_set_signature;
using simplex_bnb::presolve::presolve_mip_node_bounds;
using simplex_bnb::presolve::presolve_mip_root_problem;
using simplex_bnb::presolve::simplify_cuts_for_bounds;

std::string feature_token_(std::string_view name, bool enabled) {
    return enabled ? good_(name) : dim_(name);
}

template <typename Features> std::string join_feature_tokens_(const Features& features) {
    std::ostringstream oss;
    bool first = true;
    for (const auto& [name, enabled] : features) {
        if (!first) {
            oss << "  ";
        }
        first = false;
        oss << feature_token_(name, enabled);
    }
    return oss.str();
}

void print_verbose_solver_configuration(const BranchAndBoundOptions& options) {
    std::cout << accent_("MIP Search") << "   | node "
              << simplex_bnb::to_string(options.node_selection) << "  branch "
              << simplex_bnb::to_string(options.branching_strategy) << "  dive "
              << simplex_bnb::to_string(options.diving_strategy) << "  workers "
              << options.parallel_workers << "  async "
              << feature_token_("heuristics", options.use_async_heuristics) << std::endl;

    const std::vector<std::pair<std::string_view, bool>> heuristics = {
        {"rounding", options.use_rounding},
        {"diving", options.use_diving && options.diving_strategy != DivingStrategy::Disabled},
        {"feas-jump", options.use_feasibility_jump},
        {"feas-pump", options.use_feasibility_pump},
        {"RENS", options.use_rens},
        {"RINS", options.use_rins},
        {"local-search", options.use_local_search},
        {"local-branch", options.use_local_branching},
    };
    std::cout << accent_("MIP Heur") << "     | " << join_feature_tokens_(heuristics) << std::endl;

    const std::vector<std::pair<std::string_view, bool>> cuts = {
        {"pool", options.use_cut_pool},
        {"gomory", options.use_gomory_cuts},
        {"mir", options.use_mir_cuts},
        {"cover", options.use_cover_cuts},
        {"impl-bound", options.use_implied_bound_cuts},
        {"clique", options.use_clique_cuts},
        {"odd-cycle", options.use_odd_cycle_cuts},
        {"probing", options.use_probing_implications},
        {"conflict", options.use_conflict_cuts},
        {"dual-proof", options.use_dual_proof_cuts},
    };
    std::cout << accent_("MIP Cuts") << "     | " << join_feature_tokens_(cuts) << std::endl;
}

const char* simplex_mode_name(SimplexMode mode) {
    switch (mode) {
        case SimplexMode::Auto:
            return "auto";
        case SimplexMode::Primal:
            return "primal";
        case SimplexMode::Dual:
            return "dual";
    }
    return "unknown";
}

RevisedSimplexOptions tune_mip_lp_options(const RevisedSimplexOptions& base_options,
                                          const BranchAndBoundOptions& mip_options,
                                          bool warm_start_expected) {
    RevisedSimplexOptions tuned = base_options;
    if (!mip_options.use_lp_reoptimization_profile) {
        return tuned;
    }

    // BnB node LPs are dominated by small bound changes and repeated resolves.
    // Push them toward a HiGHS/SCIP-style dual reoptimization profile while
    // preserving the existing fallback solvers for robustness.
    tuned.mode = SimplexMode::Dual;
    tuned.pricing_rule = "adaptive";
    tuned.partial_pricing = true;
    tuned.dual_pricing = "switch";
    tuned.row_pricing_threshold = std::max(tuned.row_pricing_threshold, 40);

    tuned.basis_update = "hybrid";
    tuned.primal_edge_weight_strategy = "dense_diagonal";
    tuned.dual_edge_weight_strategy = "dense_diagonal";
    tuned.basis_refinement_steps = std::max(tuned.basis_refinement_steps, 3);
    tuned.basis_refinement_stall_progress_ratio =
        std::min(tuned.basis_refinement_stall_progress_ratio, 0.8);
    tuned.basis_refinement_stall_limit = std::max(tuned.basis_refinement_stall_limit, 3);
    tuned.min_dynamic_growth_tol = std::min(tuned.min_dynamic_growth_tol, 500.0);
    tuned.max_condition_estimate = std::min(tuned.max_condition_estimate, 1e13);
    tuned.basis_column_residual_tol = std::min(tuned.basis_column_residual_tol, 1e-8);
    tuned.basis_aggressive_residual_rebuild = true;

    tuned.primal_simplex_cost_perturbation_multiplier =
        std::max(tuned.primal_simplex_cost_perturbation_multiplier, 1.5);
    tuned.dual_simplex_cost_perturbation_multiplier =
        std::max(tuned.dual_simplex_cost_perturbation_multiplier, 2.0);
    tuned.dual_steepest_edge_weight_log_error_threshold =
        std::max(tuned.dual_steepest_edge_weight_log_error_threshold, 1.3862943611198906);

    tuned.adaptive_reset_freq = std::min(tuned.adaptive_reset_freq, 500);
    tuned.max_basis_rebuilds = std::max(tuned.max_basis_rebuilds, 5);
    tuned.crash_attempts = std::max(tuned.crash_attempts, 5);
    tuned.crash_markowitz_tol = std::min(tuned.crash_markowitz_tol, 0.15);
    if (warm_start_expected) {
        tuned.crash_attempts = std::min(tuned.crash_attempts, 1);
        tuned.crash_markowitz_tol = std::min(tuned.crash_markowitz_tol, 0.10);
    }
    tuned.devex_reset = std::max(50, std::min(tuned.devex_reset, 100));
    tuned.adaptive_reset_freq = std::min(tuned.adaptive_reset_freq, 350);

    return tuned;
}

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

double normalized_coeff(double value) { return std::abs(value) <= kCoeffTol ? 0.0 : value; }

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

std::shared_ptr<ModelState> merge_model_state(const std::shared_ptr<ModelState>& lhs,
                                              const std::shared_ptr<ModelState>& rhs,
                                              const char* context) {
    if (lhs && rhs && lhs.get() != rhs.get()) {
        throw std::invalid_argument(
            std::string("simplex: cannot combine objects from different models in ") + context);
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

struct ProblemDimensionSummary {
    int continuous = 0;
    int integer = 0;
    int binary = 0;
    int variables = 0;
    int constraints = 0;
};

ProblemDimensionSummary summarize_problem_dimensions(const simplex_bnb::Problem& problem) {
    ProblemDimensionSummary summary;
    summary.variables = static_cast<int>(problem.variable_types.size());
    summary.constraints = static_cast<int>(problem.base_constraints.size());
    for (VarType type : problem.variable_types) {
        switch (type) {
            case VarType::Continuous:
                ++summary.continuous;
                break;
            case VarType::Binary:
                ++summary.binary;
                break;
            case VarType::Integer:
                ++summary.integer;
                break;
        }
    }
    return summary;
}

void print_verbose_problem_summary(const char* label, const simplex_bnb::Problem& problem) {
    const ProblemDimensionSummary summary = summarize_problem_dimensions(problem);
    std::cout << accent_("MIP Model") << "    | " << std::left << std::setw(9) << label
              << std::right << " vars " << summary.variables << " (cont " << summary.continuous
              << ", int " << summary.integer << ", bin " << summary.binary << ") rows "
              << summary.constraints << std::endl;
}

std::string expr_repr(const LinearExprData& data, const std::shared_ptr<ModelState>& state) {
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

std::optional<std::string>
find_info_string(const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<int> find_info_int(const std::unordered_map<std::string, std::string>& info,
                                 const char* key) {
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

std::optional<double> find_info_double(const std::unordered_map<std::string, std::string>& info,
                                       const char* key) {
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

std::optional<bool> find_info_bool(const std::unordered_map<std::string, std::string>& info,
                                   const char* key) {
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

LPBasis parse_basis_state_from_info(const std::unordered_map<std::string, std::string>& info,
                                    const LPBasis& fallback = LPBasis{}) {
    const auto it = info.find("warm_start_basis_state");
    if (it == info.end()) {
        return fallback;
    }
    LPBasis out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty())
            continue;
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

std::optional<std::vector<double>>
parse_double_list_from_info(const std::unordered_map<std::string, std::string>& info,
                            const char* key) {
    const auto it = info.find(key);
    if (it == info.end())
        return std::nullopt;
    std::vector<double> out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty())
            continue;
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
        if (j < 0 || j >= sol.x.size() || chosen[j] || !eligible[j])
            return false;
        chosen[j] = 1;
        status[j] = 0;
        return true;
    };

    int chosen_count = 0;
    for (int j : sol.basis) {
        if (chosen_count == target)
            break;
        if (j < 0 || j >= sol.x.size())
            continue;
        const double l = (*maybe_l)[j];
        const double u = (*maybe_u)[j];
        const bool has_l = std::isfinite(l);
        const bool has_u = std::isfinite(u);
        const bool near_l = has_l && std::abs(sol.x(j) - l) <= tol;
        const bool near_u = has_u && std::abs(sol.x(j) - u) <= tol;
        if (!near_l && !near_u && choose_if(j))
            ++chosen_count;
    }
    for (int j = 0; j < sol.x.size() && chosen_count < target; ++j) {
        const double l = (*maybe_l)[j];
        const double u = (*maybe_u)[j];
        const bool has_l = std::isfinite(l);
        const bool has_u = std::isfinite(u);
        const bool near_l = has_l && std::abs(sol.x(j) - l) <= tol;
        const bool near_u = has_u && std::abs(sol.x(j) - u) <= tol;
        if (!near_l && !near_u && choose_if(j))
            ++chosen_count;
    }
    for (int j : sol.basis) {
        if (chosen_count == target)
            break;
        if (choose_if(j))
            ++chosen_count;
    }
    for (int j = 0; j < sol.x.size() && chosen_count < target; ++j) {
        if (choose_if(j))
            ++chosen_count;
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
        if (status == LPBasisStatus::Basic)
            ++basic_count;
    }
    return basic_count == rows;
}

std::optional<LPBasis> try_extend_basis(const LPBasis& basis, int total_vars, int rows) {
    if (static_cast<int>(basis.column_status.size()) > total_vars) {
        return std::nullopt;
    }
    LPBasis extended = basis;
    extended.column_status.resize(total_vars, LPBasisStatus::AtLower);
    if (basis_matches_dimensions(extended, total_vars, rows)) {
        return extended;
    }
    return std::nullopt;
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

using SolveStats = simplinho::bindings::SolveStats;

SolveStats build_solve_stats(const LPSolution& sol) {
    SolveStats stats;
    stats.status = to_string(sol.status);
    stats.iterations = sol.iters;
    stats.phase1_iterations = find_info_int(sol.info, "phase1_iters");
    stats.phase2_iterations = sol.iters - stats.phase1_iterations.value_or(0);
    stats.presolve_actions = find_info_int(sol.info, "presolve_actions");
    stats.presolve_implied_bound_updates =
        find_info_int(sol.info, "presolve_implied_bound_updates");
    stats.reduced_rows = find_info_int(sol.info, "reduced_m");
    stats.reduced_cols = find_info_int(sol.info, "reduced_n");
    stats.objective_shift = find_info_double(sol.info, "obj_shift");
    stats.input_upper_bounds_relaxed = find_info_int(sol.info, "input_upper_bounds_relaxed");
    stats.input_lower_bounds_relaxed = find_info_int(sol.info, "input_lower_bounds_relaxed");
    stats.basis_start = find_info_string(sol.info, "basis_start");
    stats.basis_start_style = find_info_string(sol.info, "basis_start_style");
    stats.basis_start_attempt = find_info_int(sol.info, "basis_start_attempt");
    stats.basis_start_primal_feasible = find_info_bool(sol.info, "basis_start_primal_feasible");
    stats.basis_start_dual_feasible = find_info_bool(sol.info, "basis_start_dual_feasible");
    stats.basis_start_primal_violation = find_info_double(sol.info, "basis_start_primal_violation");
    stats.basis_start_dual_violation = find_info_double(sol.info, "basis_start_dual_violation");
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
        oss << "Var(name='" << state_->vars[index].name
            << "', lb=" << format_number(state_->vars[index].lb) << ", ub=";
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
        if (invalidate_basis)
            state_->last_basis.reset();
    }

    int resolve_index_(const char* context) const {
        if (!state_) {
            throw std::invalid_argument(std::string("simplex: invalid variable in ") + context);
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
        throw std::invalid_argument(std::string("simplex: invalid variable in ") + context);
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
    friend LinearExpr make_constant_expr(const std::shared_ptr<ModelState>& state, double value);
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

    ConstraintSpec(std::shared_ptr<ModelState> state, LinearExprData expr, ConstraintSense sense)
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

LinearExpr make_constant_expr(const std::shared_ptr<ModelState>& state, double value) {
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

ConstraintSpec compare_exprs(const LinearExpr& lhs, const LinearExpr& rhs, ConstraintSense sense) {
    auto state = merge_model_state(lhs.state(), rhs.state(), "constraint");
    return ConstraintSpec(state, add_expr_data(lhs.data(), rhs.data(), -1.0), sense);
}

std::vector<double> compute_constraint_duals(const Eigen::MatrixXd& A, const Eigen::VectorXd& c,
                                             const LPSolution& raw, double objective_sign) {
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
    if (raw.status != LPSolution::Status::Optimal || static_cast<int>(raw.basis.size()) != m) {
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
            throw std::runtime_error("simplex: constraint duals are unavailable "
                                     "until the model is solved");
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
        if (invalidate_basis)
            state_->last_basis.reset();
    }

    int resolve_index_(const char* context) const {
        if (!state_) {
            throw std::invalid_argument(std::string("simplex: invalid constraint handle in ") +
                                        context);
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
        throw std::invalid_argument(std::string("simplex: invalid constraint handle in ") +
                                    context);
    }

    std::shared_ptr<ModelState> state_;
    mutable int index_ = -1;
    std::uint64_t id_ = 0;
};

class ModelSolution {
  public:
    ModelSolution() = default;

    ModelSolution(std::shared_ptr<ModelState> state, LPSolution raw, Eigen::VectorXd primal,
                  double objective)
        : state_(std::move(state)), raw_(std::move(raw)), primal_(std::move(primal)),
          objective_(objective) {
        if (state_) {
            for (int i = 0; i < primal_.size() && i < static_cast<int>(state_->vars.size()); ++i) {
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
    LPBasis basis() const {
        if (state_ && state_->last_basis.has_value()) {
            return *state_->last_basis;
        }
        if (!raw_.basis_state.column_status.empty()) {
            return raw_.basis_state;
        }
        return rebuild_basis_from_solution(raw_);
    }

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
        oss << "ModelSolution(status='" << to_string(raw_.status)
            << "', obj=" << format_number(objective_) << ")";
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
        : state_(std::move(state)), status_(result.status), objective_(result.objective),
          best_bound_(result.best_bound),
          root_relaxation_objective_(result.root_relaxation_objective),
          root_presolve_tightened_bounds_(result.root_presolve_tightened_bounds),
          root_presolve_removed_rows_(result.root_presolve_removed_rows),
          root_presolve_removed_coeffs_(result.root_presolve_removed_coeffs),
          root_presolve_aggregations_(result.root_presolve_aggregations),
          node_count_(result.node_count), relaxation_solve_count_(result.relaxation_solve_count),
          lp_iterations_(result.lp_iterations), incumbent_updates_(result.incumbent_updates),
          heuristic_lp_iterations_(result.heuristic_lp_iterations),
          heuristic_successes_(result.heuristic_successes),
          feasibility_jump_successes_(result.feasibility_jump_successes),
          feasibility_pump_successes_(result.feasibility_pump_successes),
          rens_successes_(result.rens_successes), rins_successes_(result.rins_successes),
          local_search_successes_(result.local_search_successes),
          local_branching_successes_(result.local_branching_successes),
          cuts_generated_(result.cuts_generated), cuts_applied_(result.cuts_applied),
          duplicate_cuts_(result.duplicate_cuts), cut_pool_size_(result.cut_pool_size),
          warm_start_relaxation_attempt_count_(result.warm_start_relaxation_attempt_count),
          warm_start_relaxation_accept_count_(result.warm_start_relaxation_accept_count),
          warm_start_cold_retry_count_(result.warm_start_cold_retry_count),
          warm_start_relaxation_solve_count_(result.warm_start_relaxation_solve_count),
          strong_branching_probe_count_(result.strong_branching_probe_count),
          strong_branching_probe_iterations_(result.strong_branching_probe_iterations),
          relaxation_core_solve_time_ns_(result.relaxation_core_solve_time_ns),
          relaxation_lp_assembly_time_ns_(result.relaxation_lp_assembly_time_ns),
          relaxation_lp_internal_presolve_ns_(result.relaxation_lp_internal_presolve_ns),
          relaxation_lp_internal_crash_ns_(result.relaxation_lp_internal_crash_ns),
          relaxation_lp_internal_iters_ns_(result.relaxation_lp_internal_iters_ns),
          relaxation_lp_internal_serialize_ns_(result.relaxation_lp_internal_serialize_ns),
          strong_branching_probe_core_solve_time_ns_(
              result.strong_branching_probe_core_solve_time_ns),
          strong_branching_probe_lp_assembly_time_ns_(
              result.strong_branching_probe_lp_assembly_time_ns),
          strong_branching_probe_lp_internal_presolve_ns_(
              result.strong_branching_probe_lp_internal_presolve_ns),
          strong_branching_probe_lp_internal_crash_ns_(
              result.strong_branching_probe_lp_internal_crash_ns),
          strong_branching_probe_lp_internal_iters_ns_(
              result.strong_branching_probe_lp_internal_iters_ns),
          strong_branching_probe_lp_internal_serialize_ns_(
              result.strong_branching_probe_lp_internal_serialize_ns),
          root_cut_generation_wall_ns_(result.root_cut_generation_wall_ns),
          root_cut_selection_wall_ns_(result.root_cut_selection_wall_ns),
          root_cut_activation_wall_ns_(result.root_cut_activation_wall_ns),
          root_cut_resolve_wall_ns_(result.root_cut_resolve_wall_ns),
          node_cut_generation_wall_ns_(result.node_cut_generation_wall_ns),
          node_cut_selection_wall_ns_(result.node_cut_selection_wall_ns),
          node_cut_resolve_wall_ns_(result.node_cut_resolve_wall_ns),
          rounding_heuristic_wall_ns_(result.rounding_heuristic_wall_ns),
          heuristics_wall_ns_(result.heuristics_wall_ns),
          feasibility_jump_wall_ns_(result.feasibility_jump_wall_ns),
          feasibility_pump_wall_ns_(result.feasibility_pump_wall_ns),
          diving_wall_ns_(result.diving_wall_ns), rens_wall_ns_(result.rens_wall_ns),
          rins_wall_ns_(result.rins_wall_ns), local_search_wall_ns_(result.local_search_wall_ns),
          local_branching_wall_ns_(result.local_branching_wall_ns),
          branching_wall_ns_(result.branching_wall_ns),
          child_processing_wall_ns_(result.child_processing_wall_ns),
          lp_profile_(std::move(result.lp_profile)), lp_mode_(std::move(result.lp_mode)),
          lp_partial_pricing_(result.lp_partial_pricing),
          lp_dual_pricing_(std::move(result.lp_dual_pricing)),
          warm_start_basis_state_used_(result.warm_start_basis_state_used),
          tree_nodes_(std::move(result.tree_nodes)) {
        primal_ =
            Eigen::VectorXd::Constant(original_vars, std::numeric_limits<double>::quiet_NaN());
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
            for (int i = 0; i < primal_.size() && i < static_cast<int>(state_->vars.size()); ++i) {
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
    int relaxation_solve_count() const { return relaxation_solve_count_; }
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
    int warm_start_relaxation_attempt_count() const { return warm_start_relaxation_attempt_count_; }
    int warm_start_relaxation_accept_count() const { return warm_start_relaxation_accept_count_; }
    int warm_start_cold_retry_count() const { return warm_start_cold_retry_count_; }
    int warm_start_relaxation_solve_count() const { return warm_start_relaxation_solve_count_; }
    int strong_branching_probe_count() const { return strong_branching_probe_count_; }
    int strong_branching_probe_iterations() const { return strong_branching_probe_iterations_; }
    std::uint64_t relaxation_core_solve_time_ns() const { return relaxation_core_solve_time_ns_; }
    std::uint64_t relaxation_lp_assembly_time_ns() const { return relaxation_lp_assembly_time_ns_; }
    std::uint64_t relaxation_lp_internal_presolve_ns() const {
        return relaxation_lp_internal_presolve_ns_;
    }
    std::uint64_t relaxation_lp_internal_crash_ns() const {
        return relaxation_lp_internal_crash_ns_;
    }
    std::uint64_t relaxation_lp_internal_iters_ns() const {
        return relaxation_lp_internal_iters_ns_;
    }
    std::uint64_t relaxation_lp_internal_serialize_ns() const {
        return relaxation_lp_internal_serialize_ns_;
    }
    std::uint64_t strong_branching_probe_core_solve_time_ns() const {
        return strong_branching_probe_core_solve_time_ns_;
    }
    std::uint64_t strong_branching_probe_lp_assembly_time_ns() const {
        return strong_branching_probe_lp_assembly_time_ns_;
    }
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns() const {
        return strong_branching_probe_lp_internal_presolve_ns_;
    }
    std::uint64_t strong_branching_probe_lp_internal_crash_ns() const {
        return strong_branching_probe_lp_internal_crash_ns_;
    }
    std::uint64_t strong_branching_probe_lp_internal_iters_ns() const {
        return strong_branching_probe_lp_internal_iters_ns_;
    }
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns() const {
        return strong_branching_probe_lp_internal_serialize_ns_;
    }
    std::uint64_t root_cut_generation_wall_ns() const { return root_cut_generation_wall_ns_; }
    std::uint64_t root_cut_selection_wall_ns() const { return root_cut_selection_wall_ns_; }
    std::uint64_t root_cut_activation_wall_ns() const { return root_cut_activation_wall_ns_; }
    std::uint64_t root_cut_resolve_wall_ns() const { return root_cut_resolve_wall_ns_; }
    std::uint64_t node_cut_generation_wall_ns() const { return node_cut_generation_wall_ns_; }
    std::uint64_t node_cut_selection_wall_ns() const { return node_cut_selection_wall_ns_; }
    std::uint64_t node_cut_resolve_wall_ns() const { return node_cut_resolve_wall_ns_; }
    std::uint64_t rounding_heuristic_wall_ns() const { return rounding_heuristic_wall_ns_; }
    std::uint64_t heuristics_wall_ns() const { return heuristics_wall_ns_; }
    std::uint64_t feasibility_jump_wall_ns() const { return feasibility_jump_wall_ns_; }
    std::uint64_t feasibility_pump_wall_ns() const { return feasibility_pump_wall_ns_; }
    std::uint64_t diving_wall_ns() const { return diving_wall_ns_; }
    std::uint64_t rens_wall_ns() const { return rens_wall_ns_; }
    std::uint64_t rins_wall_ns() const { return rins_wall_ns_; }
    std::uint64_t local_search_wall_ns() const { return local_search_wall_ns_; }
    std::uint64_t local_branching_wall_ns() const { return local_branching_wall_ns_; }
    std::uint64_t branching_wall_ns() const { return branching_wall_ns_; }
    std::uint64_t child_processing_wall_ns() const { return child_processing_wall_ns_; }
    const std::string& lp_profile() const { return lp_profile_; }
    const std::string& lp_mode() const { return lp_mode_; }
    bool lp_partial_pricing() const { return lp_partial_pricing_; }
    const std::string& lp_dual_pricing() const { return lp_dual_pricing_; }
    bool warm_start_basis_state_used() const { return warm_start_basis_state_used_; }
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
    int relaxation_solve_count_ = 0;
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
    int warm_start_relaxation_attempt_count_ = 0;
    int warm_start_relaxation_accept_count_ = 0;
    int warm_start_cold_retry_count_ = 0;
    int warm_start_relaxation_solve_count_ = 0;
    int strong_branching_probe_count_ = 0;
    int strong_branching_probe_iterations_ = 0;
    std::uint64_t relaxation_core_solve_time_ns_ = 0;
    std::uint64_t relaxation_lp_assembly_time_ns_ = 0;
    std::uint64_t relaxation_lp_internal_presolve_ns_ = 0;
    std::uint64_t relaxation_lp_internal_crash_ns_ = 0;
    std::uint64_t relaxation_lp_internal_iters_ns_ = 0;
    std::uint64_t relaxation_lp_internal_serialize_ns_ = 0;
    std::uint64_t strong_branching_probe_core_solve_time_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_assembly_time_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_crash_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_iters_ns_ = 0;
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns_ = 0;
    std::uint64_t root_cut_generation_wall_ns_ = 0;
    std::uint64_t root_cut_selection_wall_ns_ = 0;
    std::uint64_t root_cut_activation_wall_ns_ = 0;
    std::uint64_t root_cut_resolve_wall_ns_ = 0;
    std::uint64_t node_cut_generation_wall_ns_ = 0;
    std::uint64_t node_cut_selection_wall_ns_ = 0;
    std::uint64_t node_cut_resolve_wall_ns_ = 0;
    std::uint64_t rounding_heuristic_wall_ns_ = 0;
    std::uint64_t heuristics_wall_ns_ = 0;
    std::uint64_t feasibility_jump_wall_ns_ = 0;
    std::uint64_t feasibility_pump_wall_ns_ = 0;
    std::uint64_t diving_wall_ns_ = 0;
    std::uint64_t rens_wall_ns_ = 0;
    std::uint64_t rins_wall_ns_ = 0;
    std::uint64_t local_search_wall_ns_ = 0;
    std::uint64_t local_branching_wall_ns_ = 0;
    std::uint64_t branching_wall_ns_ = 0;
    std::uint64_t child_processing_wall_ns_ = 0;
    std::string lp_profile_;
    std::string lp_mode_;
    bool lp_partial_pricing_ = false;
    std::string lp_dual_pricing_;
    bool warm_start_basis_state_used_ = false;
    bool has_solution_ = false;
    std::unordered_map<std::string, double> values_;
    std::vector<MIPTreeNode> tree_nodes_;
};

struct ModelLPData {
    Eigen::MatrixXd A;
    RevisedSimplex::SparseMatrix A_sparse;
    Eigen::VectorXd b;
    Eigen::VectorXd c;
    Eigen::VectorXd l;
    Eigen::VectorXd u;
    double objective_sign = 1.0;
    int original_vars = 0;
    int total_vars = 0;
    int rows = 0;
    std::vector<int> warm_start_basis; // Basic column indices from root relaxation for warm-start
    std::optional<LPBasis> warm_start_basis_state;
};

class Model {
  public:
    explicit Model(const RevisedSimplexOptions& options = {})
        : state_(std::make_shared<ModelState>()) {
        state_->options = options;
    }

    Var add_var(const std::optional<std::string>& name = std::nullopt, double lb = 0.0,
                double ub = std::numeric_limits<double>::infinity(), double obj = 0.0,
                VarType var_type = VarType::Continuous) {
        touch_(true);
        std::tie(lb, ub) = canonicalize_var_bounds(var_type, lb, ub);

        std::string resolved_name;
        if (name && !name->empty()) {
            resolved_name = *name;
        } else {
            resolved_name = next_auto_name_();
        }

        if (state_->name_to_index.contains(resolved_name)) {
            throw std::invalid_argument("simplex: duplicate variable name '" + resolved_name + "'");
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

    Var add_integer_var(const std::optional<std::string>& name = std::nullopt, double lb = 0.0,
                        double ub = std::numeric_limits<double>::infinity(), double obj = 0.0) {
        return add_var(name, lb, ub, obj, VarType::Integer);
    }

    Var add_binary_var(const std::optional<std::string>& name = std::nullopt, double obj = 0.0) {
        return add_var(name, 0.0, 1.0, obj, VarType::Binary);
    }

    ConstraintHandle add_constr(const ConstraintSpec& constr,
                                const std::optional<std::string>& name = std::nullopt) {
        touch_(true);
        if (!constr.state() || constr.state().get() != state_.get()) {
            throw std::invalid_argument("simplex: constraint does not belong to this model");
        }

        const std::uint64_t id = state_->next_constraint_id++;
        ConstraintData data{id, constr.expr(), constr.sense(), name.value_or("")};
        state_->constraints.push_back(std::move(data));
        return ConstraintHandle(state_, static_cast<int>(state_->constraints.size()) - 1, id);
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
            throw std::invalid_argument("simplex: objective sense must be 'min' or 'max'");
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
        std::optional<LPBasis> extended_basis;
        auto try_extend_basis = [&](const LPBasis& basis) -> std::optional<LPBasis> {
            if (static_cast<int>(basis.column_status.size()) != data.original_vars) {
                return std::nullopt;
            }
            LPBasis extended = basis;
            extended.column_status.resize(data.total_vars, LPBasisStatus::AtLower);
            if (basis_matches_dimensions(extended, data.total_vars, data.rows)) {
                return extended;
            }
            return std::nullopt;
        };

        if (warm_start) {
            if (basis_matches_dimensions(*warm_start, data.total_vars, data.rows)) {
                effective_basis = &*warm_start;
            } else if (auto attempt = try_extend_basis(*warm_start)) {
                extended_basis = std::move(attempt);
                effective_basis = &*extended_basis;
            } else {
                throw std::invalid_argument(
                    "simplex: warm-start basis does not match model dimensions");
            }
        } else if (state_->last_basis &&
                   basis_matches_dimensions(*state_->last_basis, data.total_vars, data.rows)) {
            effective_basis = &*state_->last_basis;
        } else if (state_->last_basis) {
            if (auto attempt = try_extend_basis(*state_->last_basis)) {
                extended_basis = std::move(attempt);
                effective_basis = &*extended_basis;
            }
        }

        auto raw = effective_basis ? solver.solve(data.A_sparse, data.b, data.c, data.l, data.u,
                                                  *effective_basis)
                                   : solver.solve(data.A_sparse, data.b, data.c, data.l, data.u);
        state_->last_constraint_pi =
            compute_constraint_duals(data.A, data.c, raw, data.objective_sign);
        state_->solved_revision = state_->revision;
        if (!raw.basis_state.column_status.empty() &&
            basis_matches_dimensions(raw.basis_state, data.total_vars, data.rows)) {
            state_->last_basis = raw.basis_state;
        } else {
            const LPBasis rebuilt_basis = rebuild_basis_from_solution(raw);
            if (basis_matches_dimensions(rebuilt_basis, data.total_vars, data.rows)) {
                state_->last_basis = rebuilt_basis;
            }
        }
        return make_model_solution_(std::move(raw), data);
    }

    MIPSolution
    solve_mip(const BranchAndBoundOptions& mip_options = BranchAndBoundOptions()) const {
        // Keep cold/root LP solves on the model's baseline profile. The
        // aggressive BnB LP profile is reserved for warm-started
        // reoptimizations, which is where HiGHS/SCIP-style dual simplex tends
        // to pay off without destabilizing root or cut-added cold starts.
        RevisedSimplexOptions cold_lp_options = state_->options;
        RevisedSimplexOptions warm_lp_options =
            tune_mip_lp_options(state_->options, mip_options, true);
        state_->last_constraint_pi.clear();
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_basis.reset();
        simplex_bnb::Problem problem;
        const int original_vars = static_cast<int>(state_->vars.size());
        problem.lower_bounds = Eigen::VectorXd::Zero(original_vars);
        problem.upper_bounds =
            Eigen::VectorXd::Constant(original_vars, std::numeric_limits<double>::infinity());
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

        if (mip_options.verbose) {
            print_verbose_solver_banner();
            print_verbose_problem_summary("Original", problem);
            print_verbose_solver_configuration(mip_options);
        }

        const RootProblemPresolveResult root_presolve = presolve_mip_root_problem(problem);
        if (root_presolve.infeasible) {
            simplex_bnb::SolveResult infeasible_result;
            infeasible_result.status = simplex_bnb::Status::Infeasible;
            infeasible_result.primal =
                Eigen::VectorXd::Constant(original_vars, std::numeric_limits<double>::quiet_NaN());
            infeasible_result.objective = std::numeric_limits<double>::quiet_NaN();
            infeasible_result.best_bound = state_->maximize
                                               ? -std::numeric_limits<double>::infinity()
                                               : std::numeric_limits<double>::infinity();
            infeasible_result.root_presolve_tightened_bounds = root_presolve.tightened_bounds;
            infeasible_result.root_presolve_removed_rows = root_presolve.removed_rows;
            infeasible_result.root_presolve_removed_coeffs = root_presolve.removed_coeffs;
            infeasible_result.root_presolve_aggregations = root_presolve.aggregations;
            if (mip_options.verbose) {
                std::cout << accent_("MIP Presolve")
                          << " | Infeasible after presolve, tightened bounds "
                          << root_presolve.tightened_bounds << ", removed rows "
                          << root_presolve.removed_rows << ", removed coeffs "
                          << root_presolve.removed_coeffs << ", aggregations "
                          << root_presolve.aggregations << std::endl;
            }
            return MIPSolution(state_, std::move(infeasible_result), original_vars);
        }
        problem = root_presolve.problem;
        if (mip_options.verbose) {
            print_verbose_problem_summary("Presolved", problem);
            std::cout << accent_("MIP Presolve") << " | Tightened bounds "
                      << root_presolve.tightened_bounds << ", removed rows "
                      << root_presolve.removed_rows << ", removed coeffs "
                      << root_presolve.removed_coeffs << ", aggregations "
                      << root_presolve.aggregations << std::endl;
        }
        ModelLPData data = build_lp_data_from_problem_(problem);
        problem.lower_bounds = data.l;
        problem.upper_bounds = data.u;
        problem.objective_coefficients.conservativeResize(data.total_vars);
        for (int j = original_vars; j < data.total_vars; ++j) {
            problem.objective_coefficients(j) = 0.0;
        }
        problem.variable_types.resize(data.total_vars, VarType::Continuous);

        struct ThreadLocalMIPLPContext {
            struct RootWarmStartFallbackState {
                bool allow_root_basis_state = true;
                bool allow_parent_basis = true;
                bool root_basis_state_verified = false;
                bool parent_basis_verified = false;
                int root_basis_state_failures = 0;
                int parent_basis_failures = 0;
            };

            struct NodeLPCacheEntry {
                ModelLPData lp_data;
                RevisedSimplex cold_solver;
                RevisedSimplex warm_solver;
                std::optional<RevisedSimplex> fallback_solver;
                RootWarmStartFallbackState root_warm_start_fallback;

                NodeLPCacheEntry(ModelLPData data, const RevisedSimplexOptions& cold_options,
                                 const RevisedSimplexOptions& warm_options)
                    : lp_data(std::move(data)), cold_solver(cold_options),
                      warm_solver(warm_options) {
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

            struct NodeLPSolverView {
                const ModelLPData* lp_data;
                RevisedSimplex* cold_solver;
                RevisedSimplex* warm_solver;
                std::optional<RevisedSimplex>* fallback_solver;
                RootWarmStartFallbackState* root_warm_start_fallback;
            };

            static std::optional<std::vector<int>>
            build_slack_basis_guess(const ModelLPData& node_data) {
                if (node_data.rows <= 0 || node_data.total_vars <= node_data.original_vars) {
                    return std::nullopt;
                }
                std::vector<int> basis;
                basis.reserve(static_cast<std::size_t>(node_data.rows));
                for (int i = 0; i < node_data.rows; ++i) {
                    int slack_index = -1;
                    for (int j = node_data.original_vars; j < node_data.total_vars; ++j) {
                        const double coeff = node_data.A(i, j);
                        if (std::abs(coeff - 1.0) <= 1e-12 || std::abs(coeff + 1.0) <= 1e-12) {
                            slack_index = j;
                            break;
                        }
                    }
                    if (slack_index < 0) {
                        return std::nullopt;
                    }
                    basis.push_back(slack_index);
                }
                return basis;
            }

            std::function<ModelLPData(const ModelLPData&, const std::vector<simplex_bnb::Cut>&)>
                build_node_lp_data_;
            ModelLPData base_data;
            RevisedSimplex cold_solver;
            RevisedSimplex warm_solver;
            std::optional<RevisedSimplex> fallback_solver;
            RootWarmStartFallbackState root_warm_start_fallback;
            std::unordered_map<std::string, NodeLPCacheEntry> node_lp_cache;
            RevisedSimplexOptions cold_options_;
            RevisedSimplexOptions warm_options_;
            double objective_constant_ = 0.0;
            bool maximize_ = false;

            ThreadLocalMIPLPContext(
                std::function<ModelLPData(const ModelLPData&, const std::vector<simplex_bnb::Cut>&)>
                    build_node_lp_data,
                ModelLPData data, double objective_constant, bool maximize,
                const RevisedSimplexOptions& cold_options,
                const RevisedSimplexOptions& warm_options)
                : build_node_lp_data_(std::move(build_node_lp_data)), base_data(std::move(data)),
                  cold_solver(cold_options), warm_solver(warm_options), cold_options_(cold_options),
                  warm_options_(warm_options), objective_constant_(objective_constant),
                  maximize_(maximize) {
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

            static std::optional<LPBasis>
            adapt_warm_start_basis_state(const std::optional<LPBasis>& basis_state, int total_vars,
                                         int rows) {
                if (!basis_state.has_value()) {
                    return std::nullopt;
                }

                if (basis_matches_dimensions(*basis_state, total_vars, rows)) {
                    return basis_state;
                }

                LPBasis extended = *basis_state;
                if (static_cast<int>(extended.column_status.size()) > total_vars) {
                    return std::nullopt;
                }

                const std::size_t prior_size = extended.column_status.size();
                extended.column_status.resize(static_cast<std::size_t>(total_vars),
                                              LPBasisStatus::Basic);
                if (extended.column_status.size() > prior_size &&
                    basis_matches_dimensions(extended, total_vars, rows)) {
                    return extended;
                }

                return try_extend_basis(*basis_state, total_vars, rows);
            }

            void install_root_warm_start_basis_state(const std::optional<LPBasis>& basis_state) {
                base_data.warm_start_basis_state =
                    adapt_warm_start_basis_state(basis_state, base_data.total_vars, base_data.rows);
                root_warm_start_fallback = {};
                for (auto& [cache_key, entry] : node_lp_cache) {
                    (void)cache_key;
                    entry.lp_data.warm_start_basis_state = adapt_warm_start_basis_state(
                        basis_state, entry.lp_data.total_vars, entry.lp_data.rows);
                    entry.root_warm_start_fallback = {};
                }
            }

            NodeLPSolverView get_node_lp_entry(const std::vector<simplex_bnb::Cut>& cuts) {
                if (cuts.empty()) {
                    return NodeLPSolverView{&base_data, &cold_solver, &warm_solver,
                                            &fallback_solver, &root_warm_start_fallback};
                }
                const std::string cache_key = cut_set_signature(cuts);
                auto it = node_lp_cache.find(cache_key);
                if (it != node_lp_cache.end()) {
                    return NodeLPSolverView{&it->second.lp_data, &it->second.cold_solver,
                                            &it->second.warm_solver, &it->second.fallback_solver,
                                            &it->second.root_warm_start_fallback};
                }
                ModelLPData lp_data = build_node_lp_data_(base_data, cuts);
                auto inserted = node_lp_cache.emplace(
                    cache_key, NodeLPCacheEntry(std::move(lp_data), cold_options_, warm_options_));
                auto& entry = inserted.first->second;
                return NodeLPSolverView{&entry.lp_data, &entry.cold_solver, &entry.warm_solver,
                                        &entry.fallback_solver, &entry.root_warm_start_fallback};
            }

            std::optional<LPSolution> try_lp_solve(RevisedSimplex& solver,
                                                   const ModelLPData& node_data,
                                                   const Eigen::VectorXd& solve_l,
                                                   const Eigen::VectorXd& solve_u,
                                                   const LPBasis* basis_arg) {
                try {
                    if (basis_arg) {
                        return solver.solve(node_data.A_sparse, node_data.b, node_data.c, solve_l,
                                            solve_u, *basis_arg);
                    }
                    return solver.solve(node_data.A_sparse, node_data.b, node_data.c, solve_l,
                                        solve_u);
                } catch (const std::runtime_error& err) {
                    const std::string_view msg(err.what());
                    if (msg.find("MarkowitzLU: singular matrix") != std::string_view::npos ||
                        msg.find("MarkowitzLU: numerically singular pivot") !=
                            std::string_view::npos ||
                        msg.find("MarkowitzLU:") != std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: no pivot found") !=
                            std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: singular pivot") !=
                            std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: sparse fallback factorization failed") !=
                            std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: sparse transpose fallback factorization "
                                 "failed") != std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: fallback solve failed") !=
                            std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU: fallback transpose solve failed") !=
                            std::string_view::npos ||
                        msg.find("SparseForrestTomlinLU:") != std::string_view::npos ||
                        msg.find("FTBasis::refine_solve_B residual remained large") !=
                            std::string_view::npos ||
                        msg.find("FTBasis::refine_solve_BT residual remained large") !=
                            std::string_view::npos ||
                        msg.find("FTBasis::solve_B") != std::string_view::npos ||
                        msg.find("FTBasis::solve_BT") != std::string_view::npos ||
                        msg.find("Forrest-Tomlin:") != std::string_view::npos) {
                        return std::nullopt;
                    }
                    throw;
                }
            }

            static bool status_requires_cold_retry(LPSolution::Status status) {
                return status == LPSolution::Status::IterLimit ||
                       status == LPSolution::Status::Singular ||
                       status == LPSolution::Status::NeedPhase1;
            }

            static bool status_requires_warm_start_retry(LPSolution::Status status) {
                return status_requires_cold_retry(status);
            }

            static bool should_disable_warm_start(const std::optional<LPSolution>& sol) {
                return !sol.has_value() || sol->status == LPSolution::Status::IterLimit ||
                       sol->status == LPSolution::Status::Singular ||
                       sol->status == LPSolution::Status::NeedPhase1;
            }

            simplex_bnb::RelaxationSolution solve_node(const Eigen::VectorXd& node_l,
                                                       const Eigen::VectorXd& node_u,
                                                       const std::vector<simplex_bnb::Cut>& cuts,
                                                       const LPBasis* parent_basis) {
                const auto t0_assembly = std::chrono::steady_clock::now();
                const NodeLPSolverView entry = get_node_lp_entry(cuts);
                const auto t1_assembly = std::chrono::steady_clock::now();
                const ModelLPData& node_data = *entry.lp_data;
                Eigen::VectorXd solve_l = node_data.l;
                Eigen::VectorXd solve_u = node_data.u;
                solve_l.head(node_data.original_vars) = node_l;
                solve_u.head(node_data.original_vars) = node_u;

                enum class WarmStartSource {
                    None,
                    ParentBasis,
                    RootBasisState,
                };

                const LPBasis* effective_basis = nullptr;
                std::optional<LPBasis> effective_basis_state;
                WarmStartSource warm_start_source = WarmStartSource::None;
                auto attach_basis = [&](const LPBasis& basis) -> bool {
                    if (basis_matches_dimensions(basis, node_data.total_vars, node_data.rows)) {
                        effective_basis = &basis;
                        return true;
                    }
                    return false;
                };
                if (parent_basis && entry.root_warm_start_fallback->allow_parent_basis &&
                    attach_basis(*parent_basis)) {
                    warm_start_source = WarmStartSource::ParentBasis;
                } else if (parent_basis && entry.root_warm_start_fallback->allow_parent_basis) {
                    effective_basis_state =
                        try_extend_basis(*parent_basis, node_data.total_vars, node_data.rows);
                    if (effective_basis_state.has_value()) {
                        effective_basis = &*effective_basis_state;
                        warm_start_source = WarmStartSource::ParentBasis;
                    }
                } else if (entry.root_warm_start_fallback->allow_root_basis_state &&
                           node_data.warm_start_basis_state) {
                    if (attach_basis(*node_data.warm_start_basis_state)) {
                        warm_start_source = WarmStartSource::RootBasisState;
                    } else {
                        effective_basis_state =
                            try_extend_basis(*node_data.warm_start_basis_state,
                                             node_data.total_vars, node_data.rows);
                        if (effective_basis_state.has_value()) {
                            effective_basis = &*effective_basis_state;
                            warm_start_source = WarmStartSource::RootBasisState;
                        }
                    }
                }

                simplex_bnb::RelaxationSolution out;
                out.lp_assembly_time_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1_assembly - t0_assembly)
                        .count();
                out.attempted_warm_start_basis_state = effective_basis != nullptr;
                std::optional<LPSolution> raw_opt;
                bool used_warm_start_result = false;
                bool cold_retried_after_warm_start = false;
                std::uint64_t accumulated_core_solve_time_ns = 0;
                std::uint64_t accumulated_internal_presolve_ns = 0;
                std::uint64_t accumulated_internal_crash_ns = 0;
                std::uint64_t accumulated_internal_iters_ns = 0;
                std::uint64_t accumulated_internal_serialize_ns = 0;
                auto try_solver = [&](RevisedSimplex& solver, const LPBasis* basis_arg) {
                    const auto t0_solve = std::chrono::steady_clock::now();
                    std::optional<LPSolution> attempt =
                        try_lp_solve(solver, node_data, solve_l, solve_u, basis_arg);
                    const auto t1_solve = std::chrono::steady_clock::now();
                    accumulated_core_solve_time_ns +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_solve - t0_solve)
                            .count();
                    if (attempt.has_value()) {
                        accumulated_internal_presolve_ns += attempt->timing.presolve_ns;
                        accumulated_internal_crash_ns += attempt->timing.crash_ns;
                        accumulated_internal_iters_ns += attempt->timing.simplex_iters_ns;
                        accumulated_internal_serialize_ns += attempt->timing.serialization_ns;
                    }
                    return attempt;
                };
                const std::optional<std::vector<int>> slack_basis_guess =
                    build_slack_basis_guess(node_data);
                // Debugging helper: slack_basis_guess may be used for fallback solves.
                auto try_solver_with_basis_guess =
                    [&](RevisedSimplex& solver,
                        const std::optional<std::vector<int>>& basis_guess) {
                        const auto t0_solve = std::chrono::steady_clock::now();
                        std::optional<LPSolution> attempt;
                        if (basis_guess.has_value()) {
                            try {
                                attempt = solver.solve(node_data.A_sparse, node_data.b, node_data.c,
                                                       solve_l, solve_u, *basis_guess);
                            } catch (const std::runtime_error&) {
                                attempt.reset();
                            }
                        } else {
                            attempt = try_lp_solve(solver, node_data, solve_l, solve_u, nullptr);
                        }
                        const auto t1_solve = std::chrono::steady_clock::now();
                        accumulated_core_solve_time_ns +=
                            std::chrono::duration_cast<std::chrono::nanoseconds>(t1_solve -
                                                                                 t0_solve)
                                .count();
                        if (attempt.has_value()) {
                            accumulated_internal_presolve_ns += attempt->timing.presolve_ns;
                            accumulated_internal_crash_ns += attempt->timing.crash_ns;
                            accumulated_internal_iters_ns += attempt->timing.simplex_iters_ns;
                            accumulated_internal_serialize_ns += attempt->timing.serialization_ns;
                        }
                        return attempt;
                    };
                auto disable_warm_start_source = [&]() {
                    switch (warm_start_source) {
                        case WarmStartSource::ParentBasis:
                            entry.root_warm_start_fallback->allow_parent_basis = false;
                            ++entry.root_warm_start_fallback->parent_basis_failures;
                            break;
                        case WarmStartSource::RootBasisState:
                            entry.root_warm_start_fallback->allow_root_basis_state = false;
                            ++entry.root_warm_start_fallback->root_basis_state_failures;
                            break;
                        case WarmStartSource::None:
                            break;
                    }
                };
                if (effective_basis != nullptr) {
                    std::optional<LPSolution> warm_opt =
                        try_solver(*entry.warm_solver, effective_basis);
                    const bool warm_failed = !warm_opt.has_value() ||
                                             status_requires_warm_start_retry(warm_opt->status) ||
                                             warm_opt->status == LPSolution::Status::Infeasible ||
                                             warm_opt->status == LPSolution::Status::Unbounded;
                    if (!warm_failed) {
                        raw_opt = std::move(warm_opt);
                        used_warm_start_result = true;
                    } else {
                        if (warm_start_source != WarmStartSource::None &&
                            should_disable_warm_start(warm_opt)) {
                            disable_warm_start_source();
                        }
                        cold_retried_after_warm_start = true;
                        raw_opt = try_solver(*entry.cold_solver, nullptr);
                        if (warm_start_source != WarmStartSource::None && warm_opt.has_value() &&
                            (warm_opt->status == LPSolution::Status::Infeasible ||
                             warm_opt->status == LPSolution::Status::Unbounded) &&
                            raw_opt.has_value() && raw_opt->status != warm_opt->status) {
                            disable_warm_start_source();
                        }
                        if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                            entry.fallback_solver->has_value()) {
                            raw_opt = try_solver(**entry.fallback_solver, nullptr);
                        }
                        if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                            slack_basis_guess.has_value()) {
                            raw_opt =
                                try_solver_with_basis_guess(*entry.cold_solver, slack_basis_guess);
                        }
                        if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                            entry.fallback_solver->has_value() && slack_basis_guess.has_value()) {
                            raw_opt = try_solver_with_basis_guess(**entry.fallback_solver,
                                                                  slack_basis_guess);
                        }
                        used_warm_start_result = false;
                    }
                } else {
                    raw_opt = try_solver(*entry.cold_solver, nullptr);
                    if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                        entry.fallback_solver->has_value()) {
                        raw_opt = try_solver(**entry.fallback_solver, nullptr);
                    }
                    if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                        slack_basis_guess.has_value()) {
                        raw_opt =
                            try_solver_with_basis_guess(*entry.cold_solver, slack_basis_guess);
                    }
                    if ((!raw_opt.has_value() || status_requires_cold_retry(raw_opt->status)) &&
                        entry.fallback_solver->has_value() && slack_basis_guess.has_value()) {
                        raw_opt =
                            try_solver_with_basis_guess(**entry.fallback_solver, slack_basis_guess);
                    }
                }

                if (raw_opt.has_value()) {
                    const bool has_valid_primal = raw_opt->x.size() == node_data.total_vars &&
                                                  raw_opt->x.array().isFinite().all();
                    const bool terminal_optimal = raw_opt->status == LPSolution::Status::Optimal;
                    const bool terminal_unbounded =
                        raw_opt->status == LPSolution::Status::Unbounded;
                    out.status = raw_opt->status == LPSolution::Status::Optimal
                                     ? simplex_bnb::RelaxationStatus::Optimal
                                 : raw_opt->status == LPSolution::Status::Unbounded
                                     ? simplex_bnb::RelaxationStatus::Unbounded
                                     : simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal =
                        has_valid_primal
                            ? raw_opt->x
                            : Eigen::VectorXd::Constant(node_data.total_vars,
                                                        std::numeric_limits<double>::quiet_NaN());
                    if (terminal_optimal || terminal_unbounded) {
                        out.objective =
                            node_data.objective_sign * raw_opt->obj + objective_constant_;
                    } else {
                        out.objective = maximize_ ? -std::numeric_limits<double>::infinity()
                                                  : std::numeric_limits<double>::infinity();
                    }
                    out.iterations = raw_opt->iters;
                    out.lp_solution = *raw_opt;
                    if (terminal_optimal && !raw_opt->basis_state.column_status.empty() &&
                        basis_matches_dimensions(raw_opt->basis_state, node_data.total_vars,
                                                 node_data.rows)) {
                        out.basis = raw_opt->basis_state;
                    } else if (terminal_optimal) {
                        const LPBasis rebuilt_basis = rebuild_basis_from_solution(*raw_opt);
                        if (basis_matches_dimensions(rebuilt_basis, node_data.total_vars,
                                                     node_data.rows)) {
                            out.basis = rebuilt_basis;
                        }
                    }
                } else {
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        node_data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = maximize_ ? -std::numeric_limits<double>::infinity()
                                              : std::numeric_limits<double>::infinity();
                    out.iterations = 0;
                }
                out.used_warm_start_basis_state = used_warm_start_result;
                out.cold_retried_after_warm_start = cold_retried_after_warm_start;
                out.core_solve_time_ns = accumulated_core_solve_time_ns;
                out.lp_internal_presolve_ns = accumulated_internal_presolve_ns;
                out.lp_internal_crash_ns = accumulated_internal_crash_ns;
                out.lp_internal_iters_ns = accumulated_internal_iters_ns;
                out.lp_internal_serialize_ns = accumulated_internal_serialize_ns;
                return out;
            }
        };

        simplex_bnb::Solver bnb_solver(problem, mip_options);
        static std::atomic<std::uint64_t> next_solve_token{1};
        const std::uint64_t solve_token = next_solve_token.fetch_add(1, std::memory_order_relaxed);
        std::atomic<bool> any_warm_start_basis_state_used{false};
        simplex_bnb::SolveResult result = bnb_solver.solve(
            [&](const Eigen::VectorXd& l_node, const Eigen::VectorXd& u_node, const LPBasis* basis,
                const std::vector<simplex_bnb::Cut>& cuts) -> simplex_bnb::RelaxationSolution {
                static thread_local std::uint64_t context_owner = 0;
                static thread_local std::unique_ptr<ThreadLocalMIPLPContext> context_ptr;
                if (context_owner != solve_token || !context_ptr) {
                    context_ptr = std::make_unique<ThreadLocalMIPLPContext>(
                        [this](const ModelLPData& base, const std::vector<simplex_bnb::Cut>& cuts) {
                            return build_node_lp_data_from_base_(base, cuts);
                        },
                        data, state_->objective.constant, state_->maximize, cold_lp_options,
                        warm_lp_options);
                    context_owner = solve_token;
                }
                ThreadLocalMIPLPContext& thread_context = *context_ptr;
                if (!thread_context.base_data.warm_start_basis_state.has_value() &&
                    bnb_solver.root_warm_start_basis_state().has_value()) {
                    thread_context.install_root_warm_start_basis_state(
                        bnb_solver.root_warm_start_basis_state());
                }

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
                } else {
                    const int assign_vars =
                        std::min<int>({data.original_vars, static_cast<int>(l_node.size()),
                                       static_cast<int>(u_node.size())});
                    if (assign_vars > 0) {
                        node_l.head(assign_vars) = l_node.head(assign_vars);
                        node_u.head(assign_vars) = u_node.head(assign_vars);
                    }
                }

                // Structural cuts stay in the LP as rows. Simplifying them
                // node-by-node was pruning feasible cover-cut branches.
                SimplifiedCutsResult simplified_structural_cuts;
                simplified_structural_cuts.cuts = structural_cuts;
                const SimplifiedCutsResult simplified_presolve_cuts =
                    simplify_cuts_for_bounds(presolve_only_cuts, node_l, node_u);
                if (simplified_presolve_cuts.infeasible) {
                    simplex_bnb::RelaxationSolution out;
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = state_->maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
                    return out;
                }
                const NodeBoundPresolveResult node_presolve = presolve_mip_node_bounds(
                    problem, node_l.head(data.original_vars), node_u.head(data.original_vars),
                    simplified_presolve_cuts.cuts);
                if (node_presolve.infeasible) {
                    simplex_bnb::RelaxationSolution out;
                    out.status = simplex_bnb::RelaxationStatus::Infeasible;
                    out.primal = Eigen::VectorXd::Constant(
                        data.total_vars, std::numeric_limits<double>::quiet_NaN());
                    out.objective = state_->maximize ? -std::numeric_limits<double>::infinity()
                                                     : std::numeric_limits<double>::infinity();
                    return out;
                }
                simplex_bnb::RelaxationSolution relaxation =
                    thread_context.solve_node(node_presolve.lower, node_presolve.upper,
                                              simplified_structural_cuts.cuts, basis);
                if (relaxation.used_warm_start_basis_state) {
                    any_warm_start_basis_state_used.store(true, std::memory_order_relaxed);
                }
                return relaxation;
            });
        result.warm_start_basis_state_used =
            result.warm_start_basis_state_used ||
            any_warm_start_basis_state_used.load(std::memory_order_relaxed);
        result.lp_profile =
            mip_options.use_lp_reoptimization_profile ? "bnb_reoptimization" : "model_options";
        const RevisedSimplexOptions& reported_lp_options =
            mip_options.use_lp_reoptimization_profile ? warm_lp_options : cold_lp_options;
        result.lp_mode = simplex_mode_name(reported_lp_options.mode);
        result.lp_partial_pricing = reported_lp_options.partial_pricing;
        result.lp_dual_pricing = reported_lp_options.dual_pricing;
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
    void ensure_same_model_(const std::shared_ptr<ModelState>& other, const char* context) const {
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

    ModelLPData build_lp_data_(const std::vector<simplex_bnb::Cut>& cuts = {}) const {
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
        out.u = Eigen::VectorXd::Constant(out.total_vars, std::numeric_limits<double>::infinity());

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
                    throw std::out_of_range("simplex: constraint references invalid variable");
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

        out.A_sparse = out.A.sparseView(kCoeffTol, 1.0);
        return out;
    }

    ModelLPData build_lp_data_from_problem_(const simplex_bnb::Problem& problem,
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
        out.u = Eigen::VectorXd::Constant(out.total_vars, std::numeric_limits<double>::infinity());
        out.objective_sign = problem.maximize ? -1.0 : 1.0;

        out.l.head(n) = problem.lower_bounds;
        out.u.head(n) = problem.upper_bounds;
        out.warm_start_basis = problem.warm_start_basis;
        out.warm_start_basis_state = problem.warm_start_basis_state;
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
                    throw std::out_of_range(
                        "simplex: sparse problem row references invalid variable");
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

        out.A_sparse = out.A.sparseView(kCoeffTol, 1.0);
        return out;
    }

    ModelLPData build_node_lp_data_from_base_(const ModelLPData& base_data,
                                              const std::vector<simplex_bnb::Cut>& cuts) const {
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
        out.u = Eigen::VectorXd::Constant(out.total_vars, std::numeric_limits<double>::infinity());

        out.A.topLeftCorner(base_data.rows, base_data.total_vars) = base_data.A;
        out.b.head(base_data.rows) = base_data.b;
        out.c.head(base_data.total_vars) = base_data.c;
        out.l.head(base_data.total_vars) = base_data.l;
        out.u.head(base_data.total_vars) = base_data.u;
        out.warm_start_basis = base_data.warm_start_basis;
        out.warm_start_basis_state = base_data.warm_start_basis_state;

        int next_slack = base_data.total_vars;
        int row = base_data.rows;
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(base_data.A_sparse.nonZeros() + cuts.size() * 8 + cut_slack_count);
        for (int col = 0; col < base_data.A_sparse.outerSize(); ++col) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(base_data.A_sparse, col); it; ++it) {
                trips.emplace_back(it.row(), it.col(), it.value());
            }
        }
        for (const auto& cut : cuts) {
            for (int k = 0; k < static_cast<int>(cut.indices.size()) &&
                            k < static_cast<int>(cut.values.size());
                 ++k) {
                const int index = cut.indices[k];
                if (index < 0 || index >= base_data.total_vars) {
                    throw std::out_of_range("simplex: cut references invalid base variable");
                }
                out.A(row, index) = cut.values[k];
                trips.emplace_back(row, index, cut.values[k]);
            }
            out.b(row) = cut.rhs;
            if (cut.sense == simplex_bnb::LinearConstraintSense::LessEqual) {
                out.A(row, next_slack) = 1.0;
                trips.emplace_back(row, next_slack, 1.0);
                if (!out.warm_start_basis.empty()) {
                    out.warm_start_basis.push_back(next_slack);
                }
                if (out.warm_start_basis_state.has_value()) {
                    out.warm_start_basis_state->column_status.push_back(LPBasisStatus::Basic);
                }
                next_slack++;
            } else if (cut.sense == simplex_bnb::LinearConstraintSense::GreaterEqual) {
                out.A(row, next_slack) = -1.0;
                trips.emplace_back(row, next_slack, -1.0);
                if (!out.warm_start_basis.empty()) {
                    out.warm_start_basis.push_back(next_slack);
                }
                if (out.warm_start_basis_state.has_value()) {
                    out.warm_start_basis_state->column_status.push_back(LPBasisStatus::Basic);
                }
                next_slack++;
            }
            ++row;
        }

        out.A_sparse.resize(out.rows, out.total_vars);
        if (!trips.empty())
            out.A_sparse.setFromTriplets(trips.begin(), trips.end());
        else
            out.A_sparse.setZero();
        out.A_sparse.makeCompressed();
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
                    throw std::out_of_range("simplex: constraint references invalid variable");
                }
                row.indices.push_back(index);
                row.values.push_back(coeff);
            }
            out.push_back(std::move(row));
        }
        return out;
    }

    ModelSolution make_model_solution_(LPSolution raw, const ModelLPData& data) const {
        Eigen::VectorXd primal =
            Eigen::VectorXd::Constant(data.original_vars, std::numeric_limits<double>::quiet_NaN());
        if (raw.x.size() >= data.original_vars) {
            primal = raw.x.head(data.original_vars);
        }

        const double objective = data.objective_sign * raw.obj + state_->objective.constant;
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
        if (invalidate_basis)
            state_->last_basis.reset();
    }

    std::shared_ptr<ModelState> state_;
};

} // namespace

void bind_model_bindings(py::module_& m) {
    py::enum_<ConstraintSense>(m, "ConstraintSense")
        .value("LessEqual", ConstraintSense::LessEqual)
        .value("Equal", ConstraintSense::Equal)
        .value("GreaterEqual", ConstraintSense::GreaterEqual);

    py::class_<Var>(m, "Var")
        .def_property_readonly("name", &Var::name)
        .def_property("lb", &Var::lower_bound, &Var::set_lower_bound)
        .def_property("ub", &Var::upper_bound, &Var::set_upper_bound)
        .def_property("type", &Var::type, &Var::set_type)
        .def_property("obj", &Var::objective_coefficient, &Var::set_objective_coefficient)
        .def("__repr__", &Var::repr)
        .def(
            "__add__",
            [](const Var& self, const Var& other) {
                return add_expr(to_expr(self), to_expr(other));
            },
            py::is_operator())
        .def(
            "__add__",
            [](const Var& self, const LinearExpr& other) { return add_expr(to_expr(self), other); },
            py::is_operator())
        .def(
            "__add__",
            [](const Var& self, double other) {
                return add_expr(to_expr(self), make_constant_expr(self.state(), other));
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const Var& self, double other) {
                return add_expr(make_constant_expr(self.state(), other), to_expr(self));
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const Var& self, const Var& other) {
                return sub_expr(to_expr(self), to_expr(other));
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const Var& self, const LinearExpr& other) { return sub_expr(to_expr(self), other); },
            py::is_operator())
        .def(
            "__sub__",
            [](const Var& self, double other) {
                return sub_expr(to_expr(self), make_constant_expr(self.state(), other));
            },
            py::is_operator())
        .def(
            "__rsub__",
            [](const Var& self, double other) {
                return sub_expr(make_constant_expr(self.state(), other), to_expr(self));
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const Var& self, double scalar) { return scale_expr(to_expr(self), scalar); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const Var& self, double scalar) { return scale_expr(to_expr(self), scalar); },
            py::is_operator())
        .def(
            "__neg__", [](const Var& self) { return scale_expr(to_expr(self), -1.0); },
            py::is_operator())
        .def(
            "__le__",
            [](const Var& self, const Var& other) {
                return compare_exprs(to_expr(self), to_expr(other), ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__le__",
            [](const Var& self, const LinearExpr& other) {
                return compare_exprs(to_expr(self), other, ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__le__",
            [](const Var& self, double other) {
                return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                     ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const Var& self, const Var& other) {
                return compare_exprs(to_expr(self), to_expr(other), ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const Var& self, const LinearExpr& other) {
                return compare_exprs(to_expr(self), other, ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const Var& self, double other) {
                return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                     ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const Var& self, const Var& other) {
                return compare_exprs(to_expr(self), to_expr(other), ConstraintSense::Equal);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const Var& self, const LinearExpr& other) {
                return compare_exprs(to_expr(self), other, ConstraintSense::Equal);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const Var& self, double other) {
                return compare_exprs(to_expr(self), make_constant_expr(self.state(), other),
                                     ConstraintSense::Equal);
            },
            py::is_operator());

    py::class_<LinearExpr>(m, "LinearExpr")
        .def(py::init<>())
        .def(py::init<double>(), py::arg("constant"))
        .def("__repr__", &LinearExpr::repr)
        .def(
            "__add__",
            [](const LinearExpr& self, const LinearExpr& other) { return add_expr(self, other); },
            py::is_operator())
        .def(
            "__add__",
            [](const LinearExpr& self, const Var& other) { return add_expr(self, to_expr(other)); },
            py::is_operator())
        .def(
            "__add__",
            [](const LinearExpr& self, double other) {
                return add_expr(self, make_constant_expr(self.state(), other));
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const LinearExpr& self, double other) {
                return add_expr(make_constant_expr(self.state(), other), self);
            },
            py::is_operator())
        .def(
            "__sub__",
            [](const LinearExpr& self, const LinearExpr& other) { return sub_expr(self, other); },
            py::is_operator())
        .def(
            "__sub__",
            [](const LinearExpr& self, const Var& other) { return sub_expr(self, to_expr(other)); },
            py::is_operator())
        .def(
            "__sub__",
            [](const LinearExpr& self, double other) {
                return sub_expr(self, make_constant_expr(self.state(), other));
            },
            py::is_operator())
        .def(
            "__rsub__",
            [](const LinearExpr& self, double other) {
                return sub_expr(make_constant_expr(self.state(), other), self);
            },
            py::is_operator())
        .def(
            "__mul__",
            [](const LinearExpr& self, double scalar) { return scale_expr(self, scalar); },
            py::is_operator())
        .def(
            "__rmul__",
            [](const LinearExpr& self, double scalar) { return scale_expr(self, scalar); },
            py::is_operator())
        .def(
            "__neg__", [](const LinearExpr& self) { return scale_expr(self, -1.0); },
            py::is_operator())
        .def(
            "__le__",
            [](const LinearExpr& self, const LinearExpr& other) {
                return compare_exprs(self, other, ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__le__",
            [](const LinearExpr& self, const Var& other) {
                return compare_exprs(self, to_expr(other), ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__le__",
            [](const LinearExpr& self, double other) {
                return compare_exprs(self, make_constant_expr(self.state(), other),
                                     ConstraintSense::LessEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const LinearExpr& self, const LinearExpr& other) {
                return compare_exprs(self, other, ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const LinearExpr& self, const Var& other) {
                return compare_exprs(self, to_expr(other), ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__ge__",
            [](const LinearExpr& self, double other) {
                return compare_exprs(self, make_constant_expr(self.state(), other),
                                     ConstraintSense::GreaterEqual);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const LinearExpr& self, const LinearExpr& other) {
                return compare_exprs(self, other, ConstraintSense::Equal);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const LinearExpr& self, const Var& other) {
                return compare_exprs(self, to_expr(other), ConstraintSense::Equal);
            },
            py::is_operator())
        .def(
            "__eq__",
            [](const LinearExpr& self, double other) {
                return compare_exprs(self, make_constant_expr(self.state(), other),
                                     ConstraintSense::Equal);
            },
            py::is_operator());

    py::implicitly_convertible<Var, LinearExpr>();

    py::class_<ConstraintSpec>(m, "Constraint")
        .def("__repr__", &ConstraintSpec::repr)
        .def("__bool__", [](const ConstraintSpec&) {
            throw std::runtime_error("simplex: constraint objects cannot be used as booleans; "
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
        .def("set_coeff", &ConstraintHandle::set_coefficient, py::arg("var"), py::arg("value"))
        .def("setCoeff", &ConstraintHandle::set_coefficient, py::arg("var"), py::arg("value"))
        .def("__repr__", &ConstraintHandle::repr);

    py::class_<ModelSolution>(m, "ModelSolution")
        .def_property_readonly("raw", &ModelSolution::raw,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("status", &ModelSolution::status)
        .def_property_readonly("x", &ModelSolution::x, py::return_value_policy::reference_internal)
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
        .def("value", py::overload_cast<const std::string&>(&ModelSolution::value, py::const_),
             py::arg("name"))
        .def("__repr__", &ModelSolution::repr);

    py::class_<MIPSolution>(m, "MIPSolution")
        .def_property_readonly("status", &MIPSolution::status)
        .def_property_readonly("x", &MIPSolution::x, py::return_value_policy::reference_internal)
        .def_property_readonly("obj", &MIPSolution::objective)
        .def_property_readonly("objective", &MIPSolution::objective)
        .def_property_readonly("best_bound", &MIPSolution::best_bound)
        .def_property_readonly("root_relaxation_objective", &MIPSolution::root_relaxation_objective)
        .def_property_readonly("root_presolve_tightened_bounds",
                               &MIPSolution::root_presolve_tightened_bounds)
        .def_property_readonly("root_presolve_removed_rows",
                               &MIPSolution::root_presolve_removed_rows)
        .def_property_readonly("root_presolve_removed_coeffs",
                               &MIPSolution::root_presolve_removed_coeffs)
        .def_property_readonly("root_presolve_aggregations",
                               &MIPSolution::root_presolve_aggregations)
        .def_property_readonly("node_count", &MIPSolution::node_count)
        .def_property_readonly("relaxation_solve_count", &MIPSolution::relaxation_solve_count)
        .def_property_readonly("lp_iterations", &MIPSolution::lp_iterations)
        .def_property_readonly("incumbent_updates", &MIPSolution::incumbent_updates)
        .def_property_readonly("heuristic_lp_iterations", &MIPSolution::heuristic_lp_iterations)
        .def_property_readonly("heuristic_successes", &MIPSolution::heuristic_successes)
        .def_property_readonly("feasibility_jump_successes",
                               &MIPSolution::feasibility_jump_successes)
        .def_property_readonly("feasibility_pump_successes",
                               &MIPSolution::feasibility_pump_successes)
        .def_property_readonly("rens_successes", &MIPSolution::rens_successes)
        .def_property_readonly("rins_successes", &MIPSolution::rins_successes)
        .def_property_readonly("local_search_successes", &MIPSolution::local_search_successes)
        .def_property_readonly("local_branching_successes", &MIPSolution::local_branching_successes)
        .def_property_readonly("cuts_generated", &MIPSolution::cuts_generated)
        .def_property_readonly("cuts_applied", &MIPSolution::cuts_applied)
        .def_property_readonly("duplicate_cuts", &MIPSolution::duplicate_cuts)
        .def_property_readonly("cut_pool_size", &MIPSolution::cut_pool_size)
        .def_property_readonly("warm_start_relaxation_attempt_count",
                               &MIPSolution::warm_start_relaxation_attempt_count)
        .def_property_readonly("warm_start_relaxation_accept_count",
                               &MIPSolution::warm_start_relaxation_accept_count)
        .def_property_readonly("warm_start_cold_retry_count",
                               &MIPSolution::warm_start_cold_retry_count)
        .def_property_readonly("warm_start_relaxation_solve_count",
                               &MIPSolution::warm_start_relaxation_solve_count)
        .def_property_readonly("strong_branching_probe_count",
                               &MIPSolution::strong_branching_probe_count)
        .def_property_readonly("strong_branching_probe_iterations",
                               &MIPSolution::strong_branching_probe_iterations)
        .def_property_readonly("relaxation_core_solve_time_ns",
                               &MIPSolution::relaxation_core_solve_time_ns)
        .def_property_readonly("relaxation_lp_assembly_time_ns",
                               &MIPSolution::relaxation_lp_assembly_time_ns)
        .def_property_readonly("relaxation_lp_internal_presolve_ns",
                               &MIPSolution::relaxation_lp_internal_presolve_ns)
        .def_property_readonly("relaxation_lp_internal_crash_ns",
                               &MIPSolution::relaxation_lp_internal_crash_ns)
        .def_property_readonly("relaxation_lp_internal_iters_ns",
                               &MIPSolution::relaxation_lp_internal_iters_ns)
        .def_property_readonly("relaxation_lp_internal_serialize_ns",
                               &MIPSolution::relaxation_lp_internal_serialize_ns)
        .def_property_readonly("strong_branching_probe_core_solve_time_ns",
                               &MIPSolution::strong_branching_probe_core_solve_time_ns)
        .def_property_readonly("strong_branching_probe_lp_assembly_time_ns",
                               &MIPSolution::strong_branching_probe_lp_assembly_time_ns)
        .def_property_readonly("strong_branching_probe_lp_internal_presolve_ns",
                               &MIPSolution::strong_branching_probe_lp_internal_presolve_ns)
        .def_property_readonly("strong_branching_probe_lp_internal_crash_ns",
                               &MIPSolution::strong_branching_probe_lp_internal_crash_ns)
        .def_property_readonly("strong_branching_probe_lp_internal_iters_ns",
                               &MIPSolution::strong_branching_probe_lp_internal_iters_ns)
        .def_property_readonly("strong_branching_probe_lp_internal_serialize_ns",
                               &MIPSolution::strong_branching_probe_lp_internal_serialize_ns)
        .def_property_readonly("root_cut_generation_wall_ns",
                               &MIPSolution::root_cut_generation_wall_ns)
        .def_property_readonly("root_cut_selection_wall_ns",
                               &MIPSolution::root_cut_selection_wall_ns)
        .def_property_readonly("root_cut_activation_wall_ns",
                               &MIPSolution::root_cut_activation_wall_ns)
        .def_property_readonly("root_cut_resolve_wall_ns", &MIPSolution::root_cut_resolve_wall_ns)
        .def_property_readonly("node_cut_generation_wall_ns",
                               &MIPSolution::node_cut_generation_wall_ns)
        .def_property_readonly("node_cut_selection_wall_ns",
                               &MIPSolution::node_cut_selection_wall_ns)
        .def_property_readonly("node_cut_resolve_wall_ns", &MIPSolution::node_cut_resolve_wall_ns)
        .def_property_readonly("rounding_heuristic_wall_ns",
                               &MIPSolution::rounding_heuristic_wall_ns)
        .def_property_readonly("heuristics_wall_ns", &MIPSolution::heuristics_wall_ns)
        .def_property_readonly("feasibility_jump_wall_ns", &MIPSolution::feasibility_jump_wall_ns)
        .def_property_readonly("feasibility_pump_wall_ns", &MIPSolution::feasibility_pump_wall_ns)
        .def_property_readonly("diving_wall_ns", &MIPSolution::diving_wall_ns)
        .def_property_readonly("rens_wall_ns", &MIPSolution::rens_wall_ns)
        .def_property_readonly("rins_wall_ns", &MIPSolution::rins_wall_ns)
        .def_property_readonly("local_search_wall_ns", &MIPSolution::local_search_wall_ns)
        .def_property_readonly("local_branching_wall_ns", &MIPSolution::local_branching_wall_ns)
        .def_property_readonly("branching_wall_ns", &MIPSolution::branching_wall_ns)
        .def_property_readonly("child_processing_wall_ns", &MIPSolution::child_processing_wall_ns)
        .def_property_readonly("lp_profile", &MIPSolution::lp_profile)
        .def_property_readonly("lp_mode", &MIPSolution::lp_mode)
        .def_property_readonly("lp_partial_pricing", &MIPSolution::lp_partial_pricing)
        .def_property_readonly("lp_dual_pricing", &MIPSolution::lp_dual_pricing)
        .def_property_readonly("warm_start_basis_state_used",
                               &MIPSolution::warm_start_basis_state_used)
        .def_property_readonly("has_solution", &MIPSolution::has_solution)
        .def_property_readonly("values", &MIPSolution::values,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("tree_nodes", &MIPSolution::tree_nodes,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("relative_gap", &MIPSolution::relative_gap)
        .def("value", py::overload_cast<const Var&>(&MIPSolution::value, py::const_),
             py::arg("var"))
        .def("value", py::overload_cast<const std::string&>(&MIPSolution::value, py::const_),
             py::arg("name"))
        .def("__repr__", &MIPSolution::repr);

    py::class_<Model>(m, "Model")
        .def(py::init<const RevisedSimplexOptions&>(), py::arg("options") = RevisedSimplexOptions())
        .def("add_var", &Model::add_var, py::arg("name") = py::none(), py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(), py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("addVar", &Model::add_var, py::arg("name") = py::none(), py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(), py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("addvar", &Model::add_var, py::arg("name") = py::none(), py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(), py::arg("obj") = 0.0,
             py::arg("var_type") = VarType::Continuous)
        .def("add_integer_var", &Model::add_integer_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0, py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("addIntegerVar", &Model::add_integer_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0, py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("add_binary_var", &Model::add_binary_var, py::arg("name") = py::none(),
             py::arg("obj") = 0.0)
        .def("addBinaryVar", &Model::add_binary_var, py::arg("name") = py::none(),
             py::arg("obj") = 0.0)
        .def("add_constr", &Model::add_constr, py::arg("constraint"), py::arg("name") = py::none())
        .def("addConstr", &Model::add_constr, py::arg("constraint"), py::arg("name") = py::none())
        .def("set_objective", &Model::set_objective, py::arg("expr"), py::arg("sense") = "min")
        .def(
            "set_objective",
            [](Model& self, const Var& var, const std::string& sense) {
                self.set_objective(to_expr(var), sense);
            },
            py::arg("expr"), py::arg("sense") = "min")
        .def("setObjective", &Model::set_objective, py::arg("expr"), py::arg("sense") = "min")
        .def(
            "setObjective",
            [](Model& self, const Var& var, const std::string& sense) {
                self.set_objective(to_expr(var), sense);
            },
            py::arg("expr"), py::arg("sense") = "min")
        .def("minimize", &Model::minimize, py::arg("expr"))
        .def(
            "minimize", [](Model& self, const Var& var) { self.minimize(to_expr(var)); },
            py::arg("expr"))
        .def("maximize", &Model::maximize, py::arg("expr"))
        .def(
            "maximize", [](Model& self, const Var& var) { self.maximize(to_expr(var)); },
            py::arg("expr"))
        .def("get_var", &Model::get_var, py::arg("name"))
        .def("getVar", &Model::get_var, py::arg("name"))
        .def("get_obj_coeff", &Model::get_obj_coeff, py::arg("var"))
        .def("getObjCoeff", &Model::get_obj_coeff, py::arg("var"))
        .def("set_obj_coeff", &Model::set_obj_coeff, py::arg("var"), py::arg("value"))
        .def("setObjCoeff", &Model::set_obj_coeff, py::arg("var"), py::arg("value"))
        .def("get_coeff", &Model::get_coeff, py::arg("constraint"), py::arg("var"))
        .def("getCoeff", &Model::get_coeff, py::arg("constraint"), py::arg("var"))
        .def("set_coeff", &Model::set_coeff, py::arg("constraint"), py::arg("var"),
             py::arg("value"))
        .def("setCoeff", &Model::set_coeff, py::arg("constraint"), py::arg("var"), py::arg("value"))
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
            "options", [](Model& self) -> RevisedSimplexOptions& { return self.options(); },
            py::return_value_policy::reference_internal)
        .def(
            "solve",
            [](const Model& self, py::object basis) {
                if (basis.is_none())
                    return self.solve();
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(basis.cast<LPBasis>());
                }
                throw std::invalid_argument("simplex: model.solve basis must be an LPBasis");
            },
            py::arg("basis") = py::none())
        .def(
            "reoptimize",
            [](const Model& self, py::object basis) {
                if (basis.is_none())
                    return self.reoptimize();
                if (py::isinstance<LPBasis>(basis)) {
                    return self.reoptimize(basis.cast<LPBasis>());
                }
                throw std::invalid_argument("simplex: model.reoptimize basis must be an LPBasis");
            },
            py::arg("basis") = py::none())
        .def("solve_mip", &Model::solve_mip, py::arg("options") = BranchAndBoundOptions())
        .def("__repr__", &Model::repr);
}
