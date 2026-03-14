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
    std::string name;
    double lb = 0.0;
    double ub = std::numeric_limits<double>::infinity();
};

struct ConstraintData {
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
    std::uint64_t revision = 0;
    std::uint64_t solved_revision = std::numeric_limits<std::uint64_t>::max();
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

class Var {
   public:
    Var() = default;

    Var(std::shared_ptr<ModelState> state, int index)
        : state_(std::move(state)), index_(index) {}

    const std::shared_ptr<ModelState>& state() const { return state_; }
    int index() const { return index_; }

    std::string name() const {
        ensure_valid_("name");
        return state_->vars[index_].name;
    }

    double lower_bound() const {
        ensure_valid_("lower_bound");
        return state_->vars[index_].lb;
    }

    double upper_bound() const {
        ensure_valid_("upper_bound");
        return state_->vars[index_].ub;
    }

    std::string repr() const {
        ensure_valid_("repr");
        std::ostringstream oss;
        oss << "Var(name='" << state_->vars[index_].name << "', lb="
            << format_number(state_->vars[index_].lb) << ", ub=";
        if (std::isfinite(state_->vars[index_].ub)) {
            oss << format_number(state_->vars[index_].ub);
        } else {
            oss << "inf";
        }
        oss << ")";
        return oss.str();
    }

   private:
    void ensure_valid_(const char* context) const {
        if (!state_ || index_ < 0 ||
            index_ >= static_cast<int>(state_->vars.size())) {
            throw std::invalid_argument(
                std::string("simplex: invalid variable in ") + context);
        }
    }

    std::shared_ptr<ModelState> state_;
    int index_ = -1;
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

    ConstraintHandle(std::shared_ptr<ModelState> state, int index)
        : state_(std::move(state)), index_(index) {}

    double pi() const {
        ensure_valid_("pi");
        if (state_->solved_revision != state_->revision) {
            throw std::runtime_error(
                "simplex: constraint duals are unavailable until the model is solved");
        }
        if (index_ < 0 || index_ >= static_cast<int>(state_->last_constraint_pi.size())) {
            throw std::out_of_range("simplex: constraint index out of range");
        }
        return state_->last_constraint_pi[index_];
    }

    std::string name() const {
        ensure_valid_("name");
        return state_->constraints[index_].name;
    }

    int index() const { return index_; }

    std::string repr() const {
        ensure_valid_("repr");
        std::ostringstream oss;
        oss << "ConstraintHandle(index=" << index_;
        if (!state_->constraints[index_].name.empty()) {
            oss << ", name='" << state_->constraints[index_].name << "'";
        }
        oss << ")";
        return oss.str();
    }

   private:
    void ensure_valid_(const char* context) const {
        if (!state_ || index_ < 0 ||
            index_ >= static_cast<int>(state_->constraints.size())) {
            throw std::invalid_argument(
                std::string("simplex: invalid constraint handle in ") + context);
        }
    }

    std::shared_ptr<ModelState> state_;
    int index_ = -1;
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
        touch_();
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
        state_->vars.push_back(VarData{resolved_name, lb, ub});
        state_->name_to_index.emplace(resolved_name, index);
        if (std::abs(obj) > kCoeffTol) {
            add_coeff(state_->objective, index, obj);
        }

        return Var(state_, index);
    }

    ConstraintHandle add_constr(const ConstraintSpec& constr,
                                const std::optional<std::string>& name = std::nullopt) {
        touch_();
        if (!constr.state() || constr.state().get() != state_.get()) {
            throw std::invalid_argument(
                "simplex: constraint does not belong to this model");
        }

        ConstraintData data{constr.expr(), constr.sense(), name.value_or("")};
        state_->constraints.push_back(std::move(data));
        return ConstraintHandle(state_, static_cast<int>(state_->constraints.size()) - 1);
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
        return Var(state_, it->second);
    }

    int num_vars() const { return static_cast<int>(state_->vars.size()); }
    int num_constraints() const { return static_cast<int>(state_->constraints.size()); }

    RevisedSimplexOptions& options() { return state_->options; }
    const RevisedSimplexOptions& options() const { return state_->options; }

    ModelSolution solve() const {
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
        auto raw = solver.solve(A, b, c, l, u);
        state_->last_constraint_pi =
            compute_constraint_duals(A, c, raw, objective_sign);
        state_->solved_revision = state_->revision;

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
    std::string next_auto_name_() const {
        std::string candidate;
        int next_index = static_cast<int>(state_->vars.size());
        do {
            candidate = "x" + std::to_string(next_index++);
        } while (state_->name_to_index.contains(candidate));
        return candidate;
    }

    void touch_() {
        ++state_->revision;
        state_->solved_revision = std::numeric_limits<std::uint64_t>::max();
        state_->last_constraint_pi.clear();
    }

    std::shared_ptr<ModelState> state_;
};

}  // namespace

PYBIND11_MODULE(simplex, m) {
    m.doc() = "Bindings for the revised simplex solver";

    py::enum_<LPSolution::Status>(m, "LPStatus")
        .value("Optimal", LPSolution::Status::Optimal)
        .value("Unbounded", LPSolution::Status::Unbounded)
        .value("Infeasible", LPSolution::Status::Infeasible)
        .value("IterLimit", LPSolution::Status::IterLimit)
        .value("Singular", LPSolution::Status::Singular)
        .value("NeedPhase1", LPSolution::Status::NeedPhase1);

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
        .def_readonly("farkas_y", &LPSolution::farkas_y)
        .def_readonly("farkas_has_cert", &LPSolution::farkas_has_cert);

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
        .def_property_readonly("lb", &Var::lower_bound)
        .def_property_readonly("ub", &Var::upper_bound)
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
        .def_property_readonly("index", &ConstraintHandle::index)
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
        .def_property_readonly("num_vars", &Model::num_vars)
        .def_property_readonly("num_constraints", &Model::num_constraints)
        .def_property_readonly(
            "options",
            [](Model& self) -> RevisedSimplexOptions& { return self.options(); },
            py::return_value_policy::reference_internal)
        .def("solve", &Model::solve)
        .def("__repr__", &Model::repr);

    py::class_<RevisedSimplex>(m, "RevisedSimplex")
        .def(py::init<const RevisedSimplexOptions&>(),
             py::arg("options") = RevisedSimplexOptions())
        .def(
            "solve",
            [](RevisedSimplex& self, const Eigen::MatrixXd& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c,
               const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
                return self.solve(A, b, c, l, u);
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"), py::arg("u"),
            "Solve LP: min c^T x s.t. Ax=b, l<=x<=u");

    m.attr("SimplexModel") = m.attr("Model");
    m.def("status_to_string", [](LPSolution::Status status) {
        return std::string(to_string(status));
    });
}
