#pragma once

#include <cmath>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "simplex/simplex.h"
#include "solve_stats.h"

namespace simplinho::bindings {

inline std::string join_trace_lines(const std::vector<std::string>& trace) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < trace.size(); ++i) {
        if (i > 0) {
            oss << '\n';
        }
        oss << trace[i];
    }
    return oss.str();
}

inline std::optional<std::string>
find_info_string(const std::unordered_map<std::string, std::string>& info, const char* key) {
    const auto it = info.find(key);
    if (it == info.end()) {
        return std::nullopt;
    }
    return it->second;
}

inline std::optional<int> find_info_int(const std::unordered_map<std::string, std::string>& info,
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

inline std::optional<double>
find_info_double(const std::unordered_map<std::string, std::string>& info, const char* key) {
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

inline std::optional<bool> find_info_bool(const std::unordered_map<std::string, std::string>& info,
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

inline LPBasis parse_basis_state_from_info(const std::unordered_map<std::string, std::string>& info,
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
    out.basis_columns = fallback.basis_columns;
    out.warm_state = fallback.warm_state;
    return out;
}

inline std::optional<std::vector<double>>
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

inline LPBasis rebuild_basis_from_solution(const LPSolution& sol) {
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
    std::vector<int> indices(target);
    std::iota(indices.begin(), indices.end(), 0);
    for (int j = target; j < static_cast<int>(status.size()); ++j) {
        if (status[j] == 3) {
            indices.push_back(j);
            eligible[j] = 0;
        }
    }

    LPBasis result;
    for (int j = 0; j < static_cast<int>(indices.size()); ++j) {
        result.basis_columns.push_back(indices[j]);
    }
    result.column_status.resize(status.size());
    for (int j = 0; j < static_cast<int>(status.size()); ++j) {
        result.column_status[j] = static_cast<LPBasisStatus>(status[j]);
    }
    result.warm_state = sol.basis_state.warm_state;
    return result;
}

inline SolveStats build_solve_stats(const LPSolution& sol) {
    SolveStats stats;
    stats.status = to_string(sol.status);
    stats.iterations = sol.iters;
    stats.phase1_iterations = find_info_int(sol.info, "phase1_iters");
    stats.phase2_iterations = sol.iters - stats.phase1_iterations.value_or(0);
    stats.refactorizations = sol.solve_stats.refactorizations;
    stats.eta_stack_depth_entry = sol.solve_stats.eta_stack_depth_entry;
    stats.ft_updates = sol.solve_stats.ft_updates;
    stats.dual_pool_builds = sol.solve_stats.dual_pool_builds;
    stats.primal_pool_builds = sol.solve_stats.primal_pool_builds;
    stats.warm_start_attempted = sol.solve_stats.warm_start_attempted;
    stats.warm_start_accepted = sol.solve_stats.warm_start_accepted;
    stats.warm_start_cold_retry = sol.solve_stats.warm_start_cold_retry;
    stats.warm_factorization_reused = sol.solve_stats.warm_factorization_reused;
    stats.warm_dual_weights_reused = sol.solve_stats.warm_dual_weights_reused;
    stats.dual_row_price_calls = sol.solve_stats.dual_row_price_calls;
    stats.dual_col_price_calls = sol.solve_stats.dual_col_price_calls;
    stats.dual_price_switches = sol.solve_stats.dual_price_switches;
    stats.dual_row_ep_density = sol.solve_stats.dual_row_ep_density;
    stats.dual_row_ap_density = sol.solve_stats.dual_row_ap_density;
    stats.dual_col_aq_density = sol.solve_stats.dual_col_aq_density;
    stats.lu_build_ns = sol.solve_stats.lu_build_ns;
    stats.pricing_build_ns = sol.solve_stats.pricing_build_ns;
    stats.pivot_ns = sol.solve_stats.pivot_ns;
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

} // namespace simplinho::bindings
