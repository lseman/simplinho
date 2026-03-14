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
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simplex/simplex.h"

namespace py = pybind11;

namespace {

constexpr double kCoeffTol = 1e-12;

enum class ConstraintSense { LessEqual, Equal, GreaterEqual };

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

struct SolveStats {
    std::string status;
    int iterations = 0;
    int phase2_iterations = 0;
    std::optional<int> phase1_iterations;
    std::optional<int> presolve_actions;
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
        if (std::isfinite(state_->vars[index].ub) && value > state_->vars[index].ub) {
            throw std::invalid_argument(
                "simplex: variable lower bound cannot exceed upper bound");
        }
        touch_state_();
        state_->vars[index].lb = value;
    }

    double upper_bound() const {
        const int index = resolve_index_("upper_bound");
        return state_->vars[index].ub;
    }

    void set_upper_bound(double value) {
        const int index = resolve_index_("set_upper_bound");
        if (std::isfinite(state_->vars[index].lb) && value < state_->vars[index].lb) {
            throw std::invalid_argument(
                "simplex: variable upper bound cannot be below lower bound");
        }
        touch_state_();
        state_->vars[index].ub = value;
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

class Model {
   public:
    explicit Model(const RevisedSimplexOptions& options = {})
        : state_(std::make_shared<ModelState>()) {
        state_->options = options;
    }

    Var add_var(const std::optional<std::string>& name = std::nullopt, double lb = 0.0,
                double ub = std::numeric_limits<double>::infinity(),
                double obj = 0.0) {
        touch_(true);
        if (std::isfinite(lb) && std::isfinite(ub) && ub < lb) {
            throw std::invalid_argument("simplex: add_var received ub < lb");
        }

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
        state_->vars.push_back(VarData{id, resolved_name, lb, ub});
        state_->name_to_index.emplace(resolved_name, index);
        if (std::abs(obj) > kCoeffTol) {
            add_coeff(state_->objective, index, obj);
        }

        return Var(state_, index, id);
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
        const int n = static_cast<int>(state_->vars.size());
        int slack_count = 0;
        for (const auto& constr : state_->constraints) {
            if (constr.sense != ConstraintSense::Equal) {
                ++slack_count;
            }
        }

        const int total_vars = n + slack_count;
        const int m = static_cast<int>(state_->constraints.size());

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m, total_vars);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(m);
        Eigen::VectorXd c = Eigen::VectorXd::Zero(total_vars);
        Eigen::VectorXd l = Eigen::VectorXd::Zero(total_vars);
        Eigen::VectorXd u =
            Eigen::VectorXd::Constant(total_vars, std::numeric_limits<double>::infinity());

        for (int j = 0; j < n; ++j) {
            l(j) = state_->vars[j].lb;
            u(j) = state_->vars[j].ub;
        }

        const double objective_sign = state_->maximize ? -1.0 : 1.0;
        for (const auto& [index, coeff] : state_->objective.coeffs) {
            if (index < 0 || index >= n) {
                throw std::out_of_range("simplex: objective references invalid variable");
            }
            c(index) = objective_sign * coeff;
        }

        int next_slack = n;
        for (int row = 0; row < m; ++row) {
            const auto& constr = state_->constraints[row];
            for (const auto& [index, coeff] : constr.expr.coeffs) {
                if (index < 0 || index >= n) {
                    throw std::out_of_range(
                        "simplex: constraint references invalid variable");
                }
                A(row, index) = coeff;
            }
            b(row) = -constr.expr.constant;

            if (constr.sense == ConstraintSense::LessEqual) {
                A(row, next_slack++) = 1.0;
            } else if (constr.sense == ConstraintSense::GreaterEqual) {
                A(row, next_slack++) = -1.0;
            }
        }

        RevisedSimplex solver(state_->options);
        const LPBasis* effective_basis = nullptr;
        if (warm_start) {
            if (!basis_matches_dimensions(*warm_start, total_vars, m)) {
                throw std::invalid_argument(
                    "simplex: warm-start basis does not match model dimensions");
            }
            effective_basis = &*warm_start;
        } else if (state_->last_basis &&
                   basis_matches_dimensions(*state_->last_basis, total_vars, m)) {
            effective_basis = &*state_->last_basis;
        }

        auto raw = effective_basis ? solver.solve(A, b, c, l, u, *effective_basis)
                                   : solver.solve(A, b, c, l, u);
        state_->last_constraint_pi =
            compute_constraint_duals(A, c, raw, objective_sign);
        state_->solved_revision = state_->revision;
        const LPBasis rebuilt_basis = rebuild_basis_from_solution(raw);
        if (basis_matches_dimensions(rebuilt_basis, total_vars, m)) {
            state_->last_basis = rebuilt_basis;
        }

        Eigen::VectorXd primal =
            Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
        if (raw.x.size() >= n) {
            primal = raw.x.head(n);
        }

        const double objective =
            objective_sign * raw.obj + state_->objective.constant;
        return ModelSolution(state_, std::move(raw), std::move(primal), objective);
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

    py::class_<LPSolution>(m, "LPSolution")
        .def_readonly("status", &LPSolution::status)
        .def_readonly("x", &LPSolution::x)
        .def_readonly("obj", &LPSolution::obj)
        .def_readonly("basis", &LPSolution::basis)
        .def_readonly("basis_internal", &LPSolution::basis_internal)
        .def_readonly("nonbasis_internal", &LPSolution::nonbasis_internal)
        .def_readonly("internal_column_labels", &LPSolution::internal_column_labels)
        .def_readonly("internal_row_labels", &LPSolution::internal_row_labels)
        .def_readonly("tableau", &LPSolution::tableau)
        .def_readonly("tableau_rhs", &LPSolution::tableau_rhs)
        .def_readonly("reduced_costs_internal", &LPSolution::reduced_costs_internal)
        .def_readonly("dual_values", &LPSolution::dual_values)
        .def_readonly("shadow_prices", &LPSolution::shadow_prices)
        .def_readonly("dual_values_internal", &LPSolution::dual_values_internal)
        .def_readonly("shadow_prices_internal", &LPSolution::shadow_prices_internal)
        .def_readonly("has_internal_tableau", &LPSolution::has_internal_tableau)
        .def_readonly("iters", &LPSolution::iters)
        .def_readonly("info", &LPSolution::info)
        .def_readonly("trace", &LPSolution::trace)
        .def_property_readonly("basis_state", [](const LPSolution& self) {
            return rebuild_basis_from_solution(self);
        })
        .def_property_readonly("stats", [](const LPSolution& self) {
            return build_solve_stats(self);
        })
        .def_property_readonly("log_lines", [](const LPSolution& self) {
            return self.trace;
        })
        .def_property_readonly("log", [](const LPSolution& self) {
            return join_trace_lines(self.trace);
        })
        .def_readonly("farkas_y", &LPSolution::farkas_y)
        .def_readonly("farkas_y_internal", &LPSolution::farkas_y_internal)
        .def_readonly("farkas_has_cert", &LPSolution::farkas_has_cert)
        .def_readonly("primal_ray", &LPSolution::primal_ray)
        .def_readonly("primal_ray_internal", &LPSolution::primal_ray_internal)
        .def_readonly("primal_ray_has_cert", &LPSolution::primal_ray_has_cert);

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

    py::class_<Model>(m, "Model")
        .def(py::init<const RevisedSimplexOptions&>(),
             py::arg("options") = RevisedSimplexOptions())
        .def("add_var", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("addVar", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
             py::arg("obj") = 0.0)
        .def("addvar", &Model::add_var, py::arg("name") = py::none(),
             py::arg("lb") = 0.0,
             py::arg("ub") = std::numeric_limits<double>::infinity(),
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
            "Solve LP: min c^T x s.t. Ax=b, l<=x<=u");

    m.attr("SimplexModel") = m.attr("Model");
    m.def("status_to_string", [](LPSolution::Status status) {
        return std::string(to_string(status));
    });
}
