#pragma once

#include <optional>
#include <cstdint>
#include <string>
#include <unordered_map>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace simplinho::bindings {

struct SolveStats {
    std::string status;
    int iterations = 0;
    int phase2_iterations = 0;
    int refactorizations = 0;
    int eta_stack_depth_entry = 0;
    int ft_updates = 0;
    int dual_pool_builds = 0;
    int primal_pool_builds = 0;
    int warm_start_attempted = 0;
    int warm_start_accepted = 0;
    int warm_start_cold_retry = 0;
    int warm_factorization_reused = 0;
    int warm_dual_weights_reused = 0;
    std::uint64_t lu_build_ns = 0;
    std::uint64_t pricing_build_ns = 0;
    std::uint64_t pivot_ns = 0;
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
        out["refactorizations"] = refactorizations;
        out["eta_stack_depth_entry"] = eta_stack_depth_entry;
        out["ft_updates"] = ft_updates;
        out["dual_pool_builds"] = dual_pool_builds;
        out["primal_pool_builds"] = primal_pool_builds;
        out["warm_start_attempted"] = warm_start_attempted;
        out["warm_start_accepted"] = warm_start_accepted;
        out["warm_start_cold_retry"] = warm_start_cold_retry;
        out["warm_factorization_reused"] = warm_factorization_reused;
        out["warm_dual_weights_reused"] = warm_dual_weights_reused;
        out["lu_build_ns"] = lu_build_ns;
        out["pricing_build_ns"] = pricing_build_ns;
        out["pivot_ns"] = pivot_ns;
        out["phase1_iterations"] = phase1_iterations ? py::cast(*phase1_iterations) : py::none();
        out["presolve_actions"] = presolve_actions ? py::cast(*presolve_actions) : py::none();
        out["presolve_implied_bound_updates"] =
            presolve_implied_bound_updates ? py::cast(*presolve_implied_bound_updates) : py::none();
        out["reduced_rows"] = reduced_rows ? py::cast(*reduced_rows) : py::none();
        out["reduced_cols"] = reduced_cols ? py::cast(*reduced_cols) : py::none();
        out["objective_shift"] = objective_shift ? py::cast(*objective_shift) : py::none();
        out["input_upper_bounds_relaxed"] =
            input_upper_bounds_relaxed ? py::cast(*input_upper_bounds_relaxed) : py::none();
        out["input_lower_bounds_relaxed"] =
            input_lower_bounds_relaxed ? py::cast(*input_lower_bounds_relaxed) : py::none();
        out["basis_start"] = basis_start ? py::cast(*basis_start) : py::none();
        out["basis_start_style"] = basis_start_style ? py::cast(*basis_start_style) : py::none();
        out["basis_start_attempt"] =
            basis_start_attempt ? py::cast(*basis_start_attempt) : py::none();
        out["basis_start_primal_feasible"] =
            basis_start_primal_feasible ? py::cast(*basis_start_primal_feasible) : py::none();
        out["basis_start_dual_feasible"] =
            basis_start_dual_feasible ? py::cast(*basis_start_dual_feasible) : py::none();
        out["basis_start_primal_violation"] =
            basis_start_primal_violation ? py::cast(*basis_start_primal_violation) : py::none();
        out["basis_start_dual_violation"] =
            basis_start_dual_violation ? py::cast(*basis_start_dual_violation) : py::none();
        out["phase1_status"] = phase1_status ? py::cast(*phase1_status) : py::none();
        out["reason"] = reason ? py::cast(*reason) : py::none();
        out["note"] = note ? py::cast(*note) : py::none();
        out["certificate"] = certificate ? py::cast(*certificate) : py::none();
        out["dual_pricing"] = dual_pricing ? py::cast(*dual_pricing) : py::none();
        out["dual_bfrt_flips"] = dual_bfrt_flips ? py::cast(*dual_bfrt_flips) : py::none();
        out["degeneracy_streak"] = degeneracy_streak ? py::cast(*degeneracy_streak) : py::none();
        out["degeneracy_total"] = degeneracy_total ? py::cast(*degeneracy_total) : py::none();
        out["suspected_cycle_length"] =
            suspected_cycle_length ? py::cast(*suspected_cycle_length) : py::none();
        out["condition_estimate"] = condition_estimate ? py::cast(*condition_estimate) : py::none();
        out["degeneracy_threshold"] =
            degeneracy_threshold ? py::cast(*degeneracy_threshold) : py::none();
        out["degeneracy_epoch"] = degeneracy_epoch ? py::cast(*degeneracy_epoch) : py::none();
        out["farkas_has_cert"] = farkas_has_cert;
        out["primal_ray_has_cert"] = primal_ray_has_cert;
        out["trace_lines"] = trace_lines;
        out["raw_info"] = raw_info;
        return out;
    }
};

} // namespace simplinho::bindings
