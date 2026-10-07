#pragma once

#ifdef SIMPLEX_ENABLE_BNB

#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
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
    ///                 returns a tuple (variable, down_bound, up_bound, score).
    explicit PythonBranchingPolicy(py::function callback)
        : callback_(std::move(callback)) {}

    [[nodiscard]]
    BranchDecisionPython decide(
        const BranchingObservations& obs,
        const std::vector<detail::FractionalCandidate>& fractional) const override {
        auto start = std::chrono::steady_clock::now();

        // Build observations dict.
        py::dict obs_dict = make_observations_dict_(obs);

        // Call Python callback, releasing GIL during Python execution.
        py::object result;
        bool exception_caught = false;

        {
            py::gil_scoped_release release;
            try {
                result = callback_(obs_dict);
            } catch (const py::error_already_set&) {
                exception_caught = true;
            } catch (const std::exception&) {
                exception_caught = true;
            }
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
            return BranchDecisionPython{};  // empty → fallback
        }

        if (timed_out) {
            telemetry_.record_timeout(wall_ns);
            return BranchDecisionPython{};  // empty → fallback
        }

        auto dec_opt = parse_decision_(result, fractional);
        if (!dec_opt.has_value()) {
            telemetry_.record_fallback(wall_ns);
            return BranchDecisionPython{};  // empty → fallback
        }

        telemetry_.record_success(wall_ns, dec_opt->variable);
        return *dec_opt;
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

        // Batched NumPy arrays (copy Eigen::VectorXd for py::array_t binding).
        Eigen::VectorXd lp_vals = obs.lp_values;
        Eigen::VectorXd frac_vals = obs.fractionality;
        Eigen::VectorXd down_vals = obs.down_distance;
        Eigen::VectorXd up_vals = obs.up_distance;

        d["fractional_variables"] = py::array_t<int>(
            obs.fractional_variables.size(),
            obs.fractional_variables.data());
        d["lp_values"] = py::array_t<double>(lp_vals.size(), lp_vals.data());
        d["fractionality"] = py::array_t<double>(frac_vals.size(), frac_vals.data());
        d["down_distance"] = py::array_t<double>(down_vals.size(), down_vals.data());
        d["up_distance"] = py::array_t<double>(up_vals.size(), up_vals.data());

        // Node context.
        d["node_id"] = obs.node_id;
        d["depth"] = obs.depth;
        d["node_bound"] = obs.node_bound;
        d["lp_objective"] = obs.lp_objective;
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
    static std::optional<BranchDecisionPython> parse_decision_(
        py::object result,
        const std::vector<detail::FractionalCandidate>& fractional) {
        // Check for None (explicit defer to default policy).
        if (result.is_none()) {
            return std::nullopt;
        }

        // Expect a tuple of 3 or 4 elements.
        if (!py::isinstance<py::tuple>(result)) {
            return std::nullopt;
        }

        auto tup = result.cast<py::tuple>();
        if (tup.size() < 3 || tup.size() > 4) {
            return std::nullopt;
        }

        BranchDecisionPython dec;

        // Variable index (must be a valid variable index).
        try {
            dec.variable = tup[0].cast<int>();
        } catch (const py::cast_error&) {
            return std::nullopt;
        }

        // Down bound (float).
        try {
            dec.down_bound = tup[1].cast<double>();
        } catch (const py::cast_error&) {
            return std::nullopt;
        }

        // Up bound (float).
        try {
            dec.up_bound = tup[2].cast<double>();
        } catch (const py::cast_error&) {
            return std::nullopt;
        }

        // Optional score (float, index 3).
        if (tup.size() >= 4) {
            try {
                dec.score = tup[3].cast<double>();
            } catch (const py::cast_error&) {
                dec.score = 0.0;
            }
        }

        // Validate variable is in the fractional list.
        bool found = false;
        for (const auto& f : fractional) {
            if (f.variable == dec.variable) {
                found = true;
                break;
            }
        }
        if (!found) {
            return std::nullopt;
        }

        return dec;
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
