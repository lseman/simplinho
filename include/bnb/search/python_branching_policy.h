#pragma once

#ifdef SIMPLEX_ENABLE_BNB

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "bnb/search/branching.h"
#include "bnb/search/branching_policy.h"

namespace py = pybind11;

namespace simplex::bnb {

/// Python-backed branching policy.
/// Delegates branching decisions to a Python callable, with strict
/// validation and fallback to the default (most-fractional) policy on
/// any Python-side error.
///
/// This class is header-only because it must be instantiated in the
/// simplinho module (which links to bnb::core) — the implementation
/// cannot live in simplinho_bnb because that module only links
/// bnb::core, not the other way around.
class PythonBranchingPolicy : public BranchingPolicy {
public:
    /// @param callback Python callable that receives observations dict and
    ///                 returns a variable index or None.
    explicit PythonBranchingPolicy(py::function callback)
        : callback_(std::move(callback)) {}

    [[nodiscard]]
    std::optional<int> choose_variable(
        const BranchingObservations& obs,
        const std::vector<detail::FractionalCandidate>& fractional) const override {
        auto start = std::chrono::steady_clock::now();

        // solve_mip releases the GIL while the native solver runs. Branching
        // may also happen on a solver worker, so every Python interaction in
        // this method must be protected by an explicit acquire.
        py::gil_scoped_acquire acquire;

        // Build observations dict.
        py::dict obs_dict = make_observations_dict_(obs);

        // Call the Python callback while holding the GIL.
        py::object result;
        bool exception_caught = false;

        try {
            result = callback_(obs_dict);
        } catch (py::error_already_set& error) {
            error.restore();
            PyErr_Clear();
            exception_caught = true;
        } catch (const std::exception&) {
            exception_caught = true;
        }

        auto end = std::chrono::steady_clock::now();
        uint64_t wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

        // Check timeout.
        bool timed_out = (telemetry_.timeout_ms > 0)
                         && (wall_ns >
                             static_cast<uint64_t>(telemetry_.timeout_ms) * 1'000'000ULL);

        if (exception_caught) {
            telemetry_.record_exception(wall_ns);
            return std::nullopt;
        }

        if (timed_out) {
            telemetry_.record_timeout(wall_ns);
            return std::nullopt;
        }

        std::optional<int> variable = parse_variable_(result, fractional);
        if (!variable.has_value()) {
            telemetry_.record_fallback(wall_ns);
            return std::nullopt;
        }

        telemetry_.record_success(wall_ns, *variable);
        return variable;
    }

    bool is_enabled() const override { return true; }

    [[nodiscard]] const CallbackTelemetry& telemetry() const override { return telemetry_; }
    [[nodiscard]] CallbackTelemetry& telemetry_mutable() override { return telemetry_; }

private:
    py::function callback_;
    CallbackTelemetry telemetry_;

    /// Build the observations dict for Python with batched NumPy arrays.
    py::dict make_observations_dict_(const BranchingObservations& obs) const {
        py::dict d;

        // Give Python-owned arrays to the callback so observations remain
        // valid when a learner retains them after the callback returns.
        auto copy_ints = [](const std::vector<int>& values) {
            py::array_t<int> out(values.size());
            std::copy(values.begin(), values.end(), out.mutable_data());
            return out;
        };
        auto copy_doubles = [](const Eigen::VectorXd& values) {
            py::array_t<double> out(values.size());
            if (values.size() > 0) {
                std::copy_n(values.data(), values.size(), out.mutable_data());
            }
            return out;
        };

        d["fractional_variables"] = copy_ints(obs.fractional_variables);
        d["lp_values"] = copy_doubles(obs.lp_values);
        d["fractionality"] = copy_doubles(obs.fractionality);
        d["down_distance"] = copy_doubles(obs.down_distance);
        d["up_distance"] = copy_doubles(obs.up_distance);

        // Node context.
        d["node_id"] = obs.node_id;
        d["depth"] = obs.depth;
        d["node_bound"] = obs.node_bound;
        d["lp_objective"] = obs.lp_objective;
        d["lower_bounds"] = copy_doubles(obs.lower_bounds);
        d["upper_bounds"] = copy_doubles(obs.upper_bounds);
        d["is_root"] = obs.is_root;
        d["maximize"] = obs.maximize;

        // LP status as string.
        d["lp_status"] = lp_status_string(obs.lp_status);

        // Search state.
        d["n_nodes_explored"] = obs.n_nodes_explored;
        d["incumbent"] = obs.incumbent;

        // Pseudocost data.
        py::list pc_list;
        for (const auto& pc : obs.pseudocost) {
            py::dict entry;
            entry["up_score"] = pc.up_score;
            entry["down_score"] = pc.down_score;
            entry["samples"] = pc.samples;
            pc_list.append(entry);
        }
        d["pseudocost"] = pc_list;

        return d;
    }

    /// Parse and validate the Python return value.
    static std::optional<int> parse_variable_(
        py::object result,
        const std::vector<detail::FractionalCandidate>& fractional) {
        // Check for None (explicit defer to default policy).
        if (result.is_none()) {
            return std::nullopt;
        }

        if (!py::isinstance<py::int_>(result) || PyBool_Check(result.ptr()) != 0) {
            return std::nullopt;
        }
        const int variable = result.cast<int>();
        const bool found = std::any_of(
            fractional.begin(), fractional.end(),
            [&](const detail::FractionalCandidate& f) { return f.variable == variable; });
        return found ? std::optional<int>(variable) : std::nullopt;
    }

    /// Convert LPSolutionStatus enum to Python string.
    static std::string lp_status_string(BranchingObservations::LPSolutionStatus status) {
        switch (status) {
            case BranchingObservations::LPSolutionStatus::Optimal:       return "optimal";
            case BranchingObservations::LPSolutionStatus::Infeasible:    return "infeasible";
            case BranchingObservations::LPSolutionStatus::Unbounded:     return "unbounded";
            case BranchingObservations::LPSolutionStatus::Numerical:     return "numerical";
        }
        return "unknown";
    }
};

/// Validate that a variable index is within the fractional candidate list.
bool validate_variable_in_fractional(int variable, const std::vector<detail::FractionalCandidate>& fractional) {
    for (const auto& f : fractional) {
        if (f.variable == variable) {
            return true;
        }
    }
    return false;
}

} // namespace simplex::bnb

#endif // SIMPLEX_ENABLE_BNB
