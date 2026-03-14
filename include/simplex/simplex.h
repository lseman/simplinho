#pragma once

// -----------------------------------------------------------------------------
// Revised Simplex (header-only, drop-in compatible) — now with Dual Simplex
// Public API preserved: LPSolution, to_string, RevisedSimplexOptions,
//                       RevisedSimplex{ ctor, solve(...) }.
// Internals tidied without behavioral changes, plus a dual simplex phase:
//   - Options::mode = {Auto, Primal, Dual}
//   - Auto tries primal, and if primal reports negative basic variables,
//     falls back to dual before Phase I.
//   - You can force Dual by setting options.mode = SimplexMode::Dual.
// -----------------------------------------------------------------------------

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "presolver.h"      // presolve::LP, Presolver
#include "pricer.h"         // pricing + degeneracy helpers
#include "simplex_lu.h"     // FTBasis implementation (solve_B, solve_BT, replace_column, refactor)
#include "simplex_types.h"  // public result/status/options types

// Forward decls from degeneracy/pricer header (kept external)
static inline std::unordered_map<std::string, std::string> dm_stats_to_map(
    const DegeneracyManager::Stats& s) {
    std::unordered_map<std::string, std::string> info;
    info["deg_streak"] = std::to_string(s.degeneracy_streak);
    info["deg_total"] = std::to_string(s.degeneracy_total);
    info["cycle_len"] = std::to_string(s.suspected_cycling);
    info["cond_est"] = std::to_string(s.cond_est);
    info["deg_thresh"] = std::to_string(s.adaptive_deg_threshold);
    info["deg_epoch"] = std::to_string(s.epoch);
    return info;
}

// ============================================================================
// RevisedSimplex
// ============================================================================
class RevisedSimplexPrimalEngine;
class RevisedSimplexDualEngine;

class RevisedSimplex {
   public:
    using PhaseResult =
        std::tuple<LPSolution::Status, Eigen::VectorXd, std::vector<int>, int,
                   std::unordered_map<std::string, std::string>>;

    explicit RevisedSimplex(RevisedSimplexOptions opt = {})
        : opt_(std::move(opt)),
          rng_(opt_.rng_seed),
          degen_(opt_.rng_seed),
          adaptive_pricer_(1)  // initialized to a dummy size; rebuilt per solve
    {}

    struct SolveTraceScope {
        RevisedSimplex& self;
        bool root = false;

        explicit SolveTraceScope(RevisedSimplex& owner)
            : self(owner), root(owner.solve_depth_++ == 0) {
            if (root) self.trace_.clear();
        }

        ~SolveTraceScope() { --self.solve_depth_; }
    };

    // Main entry (drop-in compatible)
    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        const int n = static_cast<int>(A_in.cols());
        return solve(
            A_in, b_in, c_in, Eigen::VectorXd::Zero(n),
            Eigen::VectorXd::Constant(n, presolve::inf()), basis_opt);
    }

    LPSolution solve(const Eigen::MatrixXd& A_in, const Eigen::VectorXd& b_in,
                     const Eigen::VectorXd& c_in, const Eigen::VectorXd& l_in,
                     const Eigen::VectorXd& u_in,
                     std::optional<std::vector<int>> basis_opt = std::nullopt) {
        SolveTraceScope trace_scope(*this);
        const int n = static_cast<int>(A_in.cols());
        if (b_in.size() != A_in.rows()) {
            throw std::invalid_argument("simplex: b size mismatch with rows(A)");
        }
        if (c_in.size() != n || l_in.size() != n || u_in.size() != n) {
            throw std::invalid_argument(
                "simplex: c/l/u sizes must equal cols(A)");
        }

        trace_line_("[solve] start m=" + std::to_string(A_in.rows()) +
                    " n=" + std::to_string(n));

        const auto sanitized_bounds =
            canonicalize_inactive_huge_bounds_(A_in, b_in, l_in, u_in);
        const Eigen::VectorXd& l_use = sanitized_bounds.l;
        const Eigen::VectorXd& u_use = sanitized_bounds.u;
        if (sanitized_bounds.relaxed_upper > 0 ||
            sanitized_bounds.relaxed_lower > 0) {
            trace_line_("[solve] relaxed huge inactive bounds upper=" +
                        std::to_string(sanitized_bounds.relaxed_upper) +
                        " lower=" +
                        std::to_string(sanitized_bounds.relaxed_lower));
        }

        bool is_nonnegative_standard = true;
        for (int j = 0; j < n; ++j) {
            const bool l_is_zero =
                std::isfinite(l_use(j)) && std::abs(l_use(j)) <= opt_.tol;
            const bool u_is_inf = !std::isfinite(u_use(j));
            if (!l_is_zero || !u_is_inf) {
                is_nonnegative_standard = false;
                break;
            }
        }

        if (!is_nonnegative_standard) {
            struct ReformVar {
                int y = -1;
                int y_pos = -1;
                int y_neg = -1;
                int upper_slack = -1;
                double shift = 0.0;
                int sign = 1;
                bool uses_single_var = false;
                bool has_upper_row = false;
            };

            std::vector<ReformVar> map(n);
            int nv = 0;
            int upper_rows = 0;
            double obj_shift = 0.0;

            for (int j = 0; j < n; ++j) {
                const bool has_l = std::isfinite(l_use(j));
                const bool has_u = std::isfinite(u_use(j));

                if (has_l && has_u && u_use(j) < l_use(j) - opt_.tol) {
                    Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                        n, std::numeric_limits<double>::quiet_NaN());
                    return finalize_solution_(make_solution_(
                        LPSolution::Status::Infeasible, xnan,
                        std::numeric_limits<double>::infinity(), {}, 0,
                        {{"reason", "invalid_bounds"}}));
                }

                if (has_l) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = l_use(j);
                    map[j].sign = 1;
                    obj_shift += c_in(j) * l_use(j);
                    if (has_u) {
                        map[j].has_upper_row = true;
                        ++upper_rows;
                    }
                } else if (has_u) {
                    map[j].uses_single_var = true;
                    map[j].y = nv++;
                    map[j].shift = u_use(j);
                    map[j].sign = -1;
                    obj_shift += c_in(j) * u_use(j);
                } else {
                    map[j].y_pos = nv++;
                    map[j].y_neg = nv++;
                }
            }

            const int m_eq = static_cast<int>(A_in.rows());
            const int n_total = nv + upper_rows;
            const int m_total = m_eq + upper_rows;
            trace_line_("[solve] bound reformulation nv=" + std::to_string(nv) +
                        " upper_rows=" + std::to_string(upper_rows) +
                        " total_m=" + std::to_string(m_total) +
                        " total_n=" + std::to_string(n_total));

            Eigen::MatrixXd A_std = Eigen::MatrixXd::Zero(m_total, n_total);
            Eigen::VectorXd b_std = Eigen::VectorXd::Zero(m_total);
            Eigen::VectorXd c_std = Eigen::VectorXd::Zero(n_total);
            Eigen::VectorXd l_std = Eigen::VectorXd::Zero(n_total);
            Eigen::VectorXd u_std =
                Eigen::VectorXd::Constant(n_total, presolve::inf());

            for (int j = 0; j < n; ++j) {
                if (map[j].uses_single_var) {
                    c_std(map[j].y) += static_cast<double>(map[j].sign) * c_in(j);
                } else {
                    c_std(map[j].y_pos) += c_in(j);
                    c_std(map[j].y_neg) += -c_in(j);
                }
            }

            int row = 0;
            for (int i = 0; i < m_eq; ++i, ++row) {
                double rhs = b_in(i);
                for (int j = 0; j < n; ++j) {
                    const double aij = A_in(i, j);
                    if (std::abs(aij) <= 1e-16) continue;
                    if (map[j].uses_single_var) {
                        rhs -= aij * map[j].shift;
                        A_std(row, map[j].y) +=
                            static_cast<double>(map[j].sign) * aij;
                    } else {
                        A_std(row, map[j].y_pos) += aij;
                        A_std(row, map[j].y_neg) += -aij;
                    }
                }
                b_std(row) = rhs;
            }

            int upper_row = 0;
            for (int j = 0; j < n; ++j) {
                if (!map[j].has_upper_row) continue;
                const int slack = nv + upper_row;
                map[j].upper_slack = slack;
                A_std(row, map[j].y) = 1.0;
                A_std(row, slack) = 1.0;
                b_std(row) = u_use(j) - l_use(j);
                ++upper_row;
                ++row;
            }

            std::optional<std::vector<int>> basis_std = std::nullopt;
            if (basis_opt && !basis_opt->empty()) {
                std::vector<int> cand;
                cand.reserve(std::min(m_eq, (int)basis_opt->size()) + upper_rows);
                for (int jorig : *basis_opt) {
                    if (jorig < 0 || jorig >= n) continue;
                    if (map[jorig].uses_single_var) {
                        cand.push_back(map[jorig].y);
                    } else if (map[jorig].y_pos >= 0) {
                        cand.push_back(map[jorig].y_pos);
                    }
                    if ((int)cand.size() == m_eq) break;
                }
                for (int j = 0; j < n; ++j) {
                    if (map[j].upper_slack >= 0) cand.push_back(map[j].upper_slack);
                }
                if ((int)cand.size() == m_total) basis_std = std::move(cand);
            }

            LPSolution std_sol = solve(A_std, b_std, c_std, l_std, u_std, basis_std);

            Eigen::VectorXd x = Eigen::VectorXd::Constant(
                n, std::numeric_limits<double>::quiet_NaN());
            if (std_sol.x.size() == n_total && std_sol.x.array().isFinite().all()) {
                for (int j = 0; j < n; ++j) {
                    if (map[j].uses_single_var) {
                        x(j) = map[j].shift +
                               static_cast<double>(map[j].sign) * std_sol.x(map[j].y);
                    } else {
                        const double yp = std_sol.x(map[j].y_pos);
                        const double yn = std_sol.x(map[j].y_neg);
                        x(j) = yp - yn;
                    }
                }
            }

            std::vector<int> basis_out;
            std::vector<char> seen(n, 0);
            for (int idx : std_sol.basis) {
                for (int j = 0; j < n; ++j) {
                    const bool matches_single =
                        map[j].uses_single_var && map[j].y == idx;
                    const bool matches_split =
                        !map[j].uses_single_var &&
                        (map[j].y_pos == idx || map[j].y_neg == idx);
                    if ((matches_single || matches_split) && !seen[j]) {
                        seen[j] = 1;
                        basis_out.push_back(j);
                        break;
                    }
                }
            }

            auto info = std_sol.info;
            info["bound_reformulation"] = "1";
            if (sanitized_bounds.relaxed_upper > 0) {
                info["input_upper_bounds_relaxed"] =
                    std::to_string(sanitized_bounds.relaxed_upper);
            }
            if (sanitized_bounds.relaxed_lower > 0) {
                info["input_lower_bounds_relaxed"] =
                    std::to_string(sanitized_bounds.relaxed_lower);
            }
            const double obj =
                x.array().isFinite().all() ? c_in.dot(x)
                                           : (std::isfinite(std_sol.obj)
                                                  ? (std_sol.obj + obj_shift)
                                                  : std_sol.obj);
            auto sol = make_solution_(std_sol.status, std::move(x), obj,
                                      std::move(basis_out), std_sol.iters,
                                      std::move(info), std_sol.farkas_y,
                                      std_sol.farkas_has_cert,
                                      std_sol.primal_ray,
                                      std_sol.primal_ray_has_cert);
            sol.basis_internal = std_sol.basis_internal;
            sol.nonbasis_internal = std_sol.nonbasis_internal;
            sol.internal_column_labels = std_sol.internal_column_labels;
            sol.internal_row_labels = std_sol.internal_row_labels;
            sol.tableau = std_sol.tableau;
            sol.tableau_rhs = std_sol.tableau_rhs;
            sol.reduced_costs_internal = std_sol.reduced_costs_internal;
            if (std_sol.dual_values.size() >= m_eq) {
                sol.dual_values = std_sol.dual_values.head(m_eq);
                sol.shadow_prices = sol.dual_values;
            }
            sol.dual_values_internal = std_sol.dual_values_internal;
            sol.shadow_prices_internal = std_sol.shadow_prices_internal;
            sol.farkas_y_internal = std_sol.farkas_y_internal;
            sol.primal_ray_internal = std_sol.primal_ray_internal;
            if (std_sol.primal_ray_has_cert &&
                std_sol.primal_ray.size() == n_total) {
                Eigen::VectorXd ray = Eigen::VectorXd::Zero(n);
                for (int j = 0; j < n; ++j) {
                    if (map[j].uses_single_var) {
                        ray(j) = static_cast<double>(map[j].sign) *
                                 std_sol.primal_ray(map[j].y);
                    } else {
                        const double pos = (map[j].y_pos >= 0)
                                               ? std_sol.primal_ray(map[j].y_pos)
                                               : 0.0;
                        const double neg = (map[j].y_neg >= 0)
                                               ? std_sol.primal_ray(map[j].y_neg)
                                               : 0.0;
                        ray(j) = pos - neg;
                    }
                }
                sol.primal_ray = clip_small_(ray);
            }
            sol.has_internal_tableau = std_sol.has_internal_tableau;
            return finalize_solution_(std::move(sol));
        }

        Eigen::MatrixXd A_model = A_in;
        Eigen::VectorXd b_model = b_in;
        Eigen::VectorXd c_model = c_in;
        Eigen::VectorXd l_model = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd u_model = Eigen::VectorXd::Constant(n, presolve::inf());
        Eigen::VectorXd anchor = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd sign = Eigen::VectorXd::Ones(n);

        for (int j = 0; j < n; ++j) {
            const bool has_l = std::isfinite(l_use(j));
            const bool has_u = std::isfinite(u_use(j));
            if (!has_l && !has_u) {
                throw std::invalid_argument(
                    "simplex: free variables are unsupported in solve(A,b,c,l,u)");
            }

            if (has_l) {
                anchor(j) = l_use(j);
                l_model(j) = 0.0;
                u_model(j) = has_u ? (u_use(j) - l_use(j)) : presolve::inf();
            } else {
                anchor(j) = u_use(j);
                sign(j) = -1.0;
                l_model(j) = 0.0;
                u_model(j) = presolve::inf();
                A_model.col(j) = -A_model.col(j);
                c_model(j) = -c_model(j);
            }

            if (anchor(j) != 0.0) b_model.noalias() -= A_model.col(j) * anchor(j);
        }

        // ---- (0) Wrap into presolve LP: Ax=b, default bounds, costs=c ----
        presolve::LP lp;
        lp.A = A_model;
        lp.b = b_model;
        lp.sense.assign(static_cast<int>(A_in.rows()), presolve::RowSense::EQ);
        lp.c = c_model;
        lp.l = l_model;
        lp.u = u_model;
        lp.c0 = c_in.dot(anchor);

        // ---- (1) Presolve ----
        presolve::Presolver::Options popt;
        popt.enable_rowreduce = true;
        popt.enable_scaling = true;
        popt.enable_objective_probing = false;
        popt.non_destructive = true;
        popt.allow_structural_changes = false;
        popt.max_passes = 5;
        if (A_in.cols() > static_cast<int>(A_in.rows() * 1.2)) {
            popt.conservative_mode = true;
        }

        presolve::Presolver P(popt);
        const auto pres = P.run(lp);
        trace_presolve_(pres);

        if (pres.proven_infeasible) {
            return finalize_solution_(make_solution_(
                LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::infinity(), {}, 0,
                {{"presolve", "infeasible"}}));
        }
        if (pres.proven_unbounded) {
            Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                n, std::numeric_limits<double>::quiet_NaN());
            return finalize_solution_(make_solution_(
                LPSolution::Status::Unbounded, xnan,
                -std::numeric_limits<double>::infinity(), {}, 0,
                {{"presolve", "unbounded"}}));
        }

        const Eigen::MatrixXd& Atil = pres.reduced.A;
        const Eigen::VectorXd& btil = pres.reduced.b;
        const Eigen::VectorXd& ctil = pres.reduced.c;
        const Eigen::VectorXd& lred = pres.reduced.l;
        const Eigen::VectorXd& ured = pres.reduced.u;

        // ---- (2) m==0 fast path: optimize over bounds only ----
        if (Atil.rows() == 0) {
            Eigen::VectorXd vred =
                Eigen::VectorXd::Zero(static_cast<int>(ctil.size()));
            bool is_bounded = true;
            for (int j = 0; j < static_cast<int>(ctil.size()); ++j) {
                if (ctil(j) > opt_.tol) {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                } else if (ctil(j) < -opt_.tol) {
                    if (std::isfinite(ured(j)))
                        vred(j) = ured(j);
                    else {
                        is_bounded = false;
                        break;
                    }
                } else {
                    vred(j) = std::isfinite(lred(j)) ? lred(j) : 0.0;
                }
            }
            if (!is_bounded) {
                Eigen::VectorXd xnan = Eigen::VectorXd::Constant(
                    n, std::numeric_limits<double>::quiet_NaN());
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Unbounded, xnan,
                    -std::numeric_limits<double>::infinity(), {}, 0,
                    {{"presolve", "m=0 neg cost & +inf upper"}}));
            }
            auto [z_full, obj_corr] = P.postsolve(vred);
            z_full = repair_nan_primal_(A_model, b_model, l_model, u_model,
                                        std::move(z_full), opt_.tol);
            Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
            const double total_obj = c_in.dot(x_full);
            auto sol = make_solution_(LPSolution::Status::Optimal,
                                      std::move(x_full), total_obj, {}, 0,
                                      {{"presolve", "m=0 optimized over bounds"}});
            sol.dual_values = clip_small_vec_(
                P.postsolve_dual(Eigen::VectorXd::Zero(0)), opt_.tol);
            sol.shadow_prices = sol.dual_values;
            return finalize_solution_(std::move(sol));
        }

        // ---- (3) Solve reduced problem directly with explicit bounds ----
        const bool bypass_postsolve = false;
        Eigen::MatrixXd Ared = Atil;
        Eigen::VectorXd bred = btil;
        Eigen::VectorXd cred = ctil;
        std::vector<int> col_orig_map = pres.orig_col_index;
        std::vector<int> row_orig_map = pres.orig_row_index;
        const std::vector<std::string> internal_column_labels =
            make_internal_column_labels_(col_orig_map);
        const std::vector<std::string> internal_row_labels =
            make_internal_row_labels_(row_orig_map);
        const int m_eff = static_cast<int>(Ared.rows());
        const int n_eff = static_cast<int>(Ared.cols());

        // Effective bounds (reduced space)
        Eigen::VectorXd l_eff = lred;
        Eigen::VectorXd u_eff = ured;

        const auto postsolve_primal = [&](const Eigen::VectorXd& v) {
            if (bypass_postsolve) {
                return std::make_pair(v, 0.0);
            }
            auto out = P.postsolve(v);
            out.first = repair_nan_primal_(A_model, b_model, l_model, u_model,
                                           std::move(out.first), opt_.tol);
            return out;
        };

        // ---- (4) Map incoming basis into reduced space (optional) ----
        std::optional<std::vector<int>> red_basis_opt = std::nullopt;
        if (basis_opt && !basis_opt->empty()) {
            std::unordered_map<int, int> orig2red;
            orig2red.reserve(n_eff);
            for (int jr = 0; jr < n_eff; ++jr) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0) orig2red[jorig] = jr;
            }
            std::vector<int> cand;
            cand.reserve(std::min(m_eff, (int)basis_opt->size()));
            std::vector<char> seen_red(n_eff, 0);
            for (int jorig : *basis_opt) {
                auto it = orig2red.find(jorig);
                if (it != orig2red.end() && !seen_red[it->second]) {
                    seen_red[it->second] = 1;
                    cand.push_back(it->second);
                    if ((int)cand.size() == m_eff) break;
                }
            }
            if (!cand.empty()) red_basis_opt = std::move(cand);
        }

        // ---- (5) Try Phase II directly on reduced problem (Primal/Dual per
        // mode) ----
        std::vector<int> basis_guess;
        CrashSelection basis_choice = choose_initial_basis_(
            Ared, bred, cred, opt_,
            (red_basis_opt && !red_basis_opt->empty())
                ? std::optional<std::vector<int>>(*red_basis_opt)
                : std::nullopt);
        basis_guess = basis_choice.basis;
        const bool basis_guess_from_crash = (basis_choice.source == "crash");
        const bool basis_guess_from_warm_start =
            (basis_choice.source == "warm_start" ||
             basis_choice.source == "repaired_warm_start");

        const auto add_info =
            [&](std::unordered_map<std::string, std::string> info) {
                info["presolve_actions"] = std::to_string(pres.stack.size());
                info["reduced_m"] = std::to_string(m_eff);
                info["reduced_n"] = std::to_string(n_eff);
                info["obj_shift"] = std::to_string(pres.obj_shift);
                if (sanitized_bounds.relaxed_upper > 0) {
                    info["input_upper_bounds_relaxed"] =
                        std::to_string(sanitized_bounds.relaxed_upper);
                }
                if (sanitized_bounds.relaxed_lower > 0) {
                    info["input_lower_bounds_relaxed"] =
                        std::to_string(sanitized_bounds.relaxed_lower);
                }
                if (!basis_choice.source.empty() && basis_choice.source != "none") {
                    info["basis_start"] = basis_choice.source;
                    info["basis_start_style"] = basis_choice.style;
                    info["basis_start_attempt"] =
                        std::to_string(basis_choice.attempt);
                    info["basis_start_primal_feasible"] =
                        basis_choice.quality.primal_feasible ? "1" : "0";
                    info["basis_start_dual_feasible"] =
                        basis_choice.quality.dual_feasible ? "1" : "0";
                    info["basis_start_primal_violation"] =
                        std::to_string(basis_choice.quality.primal_violation);
                    info["basis_start_dual_violation"] =
                        std::to_string(basis_choice.quality.dual_violation);
                }
                return info;
            };

        const auto parse_serialized_vec =
            [](const std::unordered_map<std::string, std::string>& info,
               const char* key, int expected_dim)
            -> std::optional<Eigen::VectorXd> {
            auto it = info.find(key);
            if (it == info.end()) return std::nullopt;
            std::vector<double> vals;
            std::stringstream ss(it->second);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) vals.push_back(std::stod(tok));
            }
            if (expected_dim >= 0 && (int)vals.size() != expected_dim) {
                return std::nullopt;
            }
            if (vals.empty() && expected_dim == 0) return Eigen::VectorXd::Zero(0);
            if (vals.empty()) return std::nullopt;
            return Eigen::Map<const Eigen::VectorXd>(vals.data(),
                                                    static_cast<int>(vals.size()));
        };

        const bool basis_valid =
            ((int)basis_guess.size() == m_eff) && basis_choice.quality.valid;
        const bool allow_direct_primal =
            basis_valid &&
            (basis_choice.quality.primal_feasible || basis_guess_from_warm_start);
        const bool allow_direct_dual =
            basis_valid &&
            (basis_choice.quality.dual_feasible || basis_guess_from_warm_start);
        const bool allow_direct_from_guess =
            allow_direct_primal || allow_direct_dual;

        if (allow_direct_from_guess) {
            LPSolution::Status st;
            Eigen::VectorXd v2;
            std::vector<int> red_basis2;
            int it2;
            std::unordered_map<std::string, std::string> info2;

            auto run_primal = [&] {
                return phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };
            auto run_dual = [&] {
                return dual_phase_(Ared, bred, cred, basis_guess, l_eff, u_eff);
            };

            if (opt_.mode == SimplexMode::Dual) {
                if (allow_direct_dual) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                } else {
                    st = LPSolution::Status::NeedPhase1;
                    info2["reason"] = "no_dual_feasible_start_basis";
                }
            } else if (opt_.mode == SimplexMode::Primal) {
                if (allow_direct_primal) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                } else {
                    st = LPSolution::Status::NeedPhase1;
                    info2["reason"] = "no_primal_feasible_start_basis";
                }
            } else {
                if (allow_direct_primal) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_primal();
                } else if (allow_direct_dual) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
                if (allow_direct_dual &&
                    st == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") ==
                        std::string("negative_basic_vars")) {
                    std::tie(st, v2, red_basis2, it2, info2) = run_dual();
                }
            }

            if (st == LPSolution::Status::Optimal ||
                st == LPSolution::Status::Unbounded ||
                st == LPSolution::Status::IterLimit) {
                auto [z_full, obj_corr] = postsolve_primal(v2);
                Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
                const double total_obj = c_in.dot(x_full);
                const bool has_primal_ray =
                    info2.count("primal_ray_has_cert") &&
                    info2.at("primal_ray_has_cert") == "1";
                const auto primal_ray_internal =
                    has_primal_ray
                        ? parse_serialized_vec(info2, "primal_ray", n_eff)
                        : std::nullopt;

                std::vector<int> basis_full;
                basis_full.reserve(red_basis2.size());
                for (int jr : red_basis2) {
                    if (jr >= 0 && jr < (int)col_orig_map.size()) {
                        const int jorig = col_orig_map[jr];
                        if (jorig >= 0) basis_full.push_back(jorig);
                    }
                }
                auto info = add_info(std::move(info2));
                if (st == LPSolution::Status::Optimal &&
                    !primal_feasible_(A_in, b_in, x_full, l_in, u_in,
                                      opt_.tol)) {
                    info["reason"] = "invalid_returned_primal";
                    return finalize_solution_(attach_internal_basis_(
                        make_solution_(LPSolution::Status::Singular,
                                       std::move(x_full), total_obj, basis_full,
                                       it2, std::move(info)),
                        red_basis2, internal_column_labels));
                }
                return finalize_solution_(attach_mapped_primal_ray_(
                    attach_postsolved_farkas_(
                        attach_postsolved_row_duals_(
                            attach_internal_tableau_(
                                make_solution_(
                                    st, std::move(x_full), total_obj, basis_full,
                                    it2, std::move(info), std::nullopt,
                                    std::nullopt, primal_ray_internal,
                                    has_primal_ray),
                                Ared, bred, cred, red_basis2,
                                internal_column_labels, internal_row_labels,
                                opt_.tol),
                            P, opt_.tol),
                        P, opt_.tol),
                    col_orig_map, sign, A_model.cols(), opt_.tol));
            }
            if (st == LPSolution::Status::Singular) {
                auto info = add_info({});
                return finalize_solution_(make_solution_(
                    LPSolution::Status::Singular, Eigen::VectorXd::Zero(n),
                    std::numeric_limits<double>::quiet_NaN(), {}, 0,
                    std::move(info)));
            }
        }

        // ---- (6) Phase I on reduced problem ----
        auto [A1, b1, c1, basis1, n_orig_eff, m_rows] =
            make_phase1_(Ared, bred);
        auto [status1, v1, basis1_out, it1, info1] =
            phase_(A1, b1, c1, basis1, Eigen::VectorXd::Zero(A1.cols()),
                   Eigen::VectorXd::Constant(A1.cols(), presolve::inf()));
        if (status1 == LPSolution::Status::NeedPhase1 &&
            info1.count("reason") &&
            info1.at("reason") == std::string("negative_basic_vars")) {
            std::tie(status1, v1, basis1_out, it1, info1) =
                dual_phase_(A1, b1, c1, basis1_out.empty() ? basis1 : basis1_out,
                            Eigen::VectorXd::Zero(A1.cols()),
                            Eigen::VectorXd::Constant(A1.cols(),
                                                      presolve::inf()));
        }

        // If phase I fails or artificial cost > tol ⇒ infeasible
        if (status1 != LPSolution::Status::Optimal || c1.dot(v1) > opt_.tol) {
            auto info = add_info({{"phase1_status", to_string(status1)}});
            const auto s = degen_.get_stats();
            auto more = dm_stats_to_map(s);
            info.insert(more.begin(), more.end());
            return finalize_solution_(make_solution_(
                LPSolution::Status::Infeasible, Eigen::VectorXd::Zero(n),
                std::numeric_limits<double>::infinity(), {}, it1,
                std::move(info)));
        }

        // Warm-start Phase II basis by removing artificials
        std::vector<int> red_basis2;
        red_basis2.reserve(m_rows);
        for (int j : basis1_out)
            if (j < (int)n_orig_eff) red_basis2.push_back(j);

        // Basis completion if needed
        if ((int)red_basis2.size() < m_rows) {
            std::vector<int> fallback_basis = red_basis2;
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if ((int)red_basis2.size() == m_rows) break;
                if (std::find(red_basis2.begin(), red_basis2.end(), j) !=
                    red_basis2.end())
                    continue;
                std::vector<int> cand = red_basis2;
                cand.push_back(j);
                if ((int)cand.size() > m_rows) continue;
                const Eigen::MatrixXd Btest =
                    Ared(Eigen::all,
                         Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                if (!(lu.rank() == (int)cand.size() && lu.isInvertible())) {
                    continue;
                }
                if (basis_is_primal_feasible_(Ared, bred, cand, opt_.tol)) {
                    red_basis2 = std::move(cand);
                    continue;
                }
                if ((int)fallback_basis.size() < (int)cand.size()) {
                    fallback_basis = cand;
                }
            }
            if ((int)red_basis2.size() < m_rows &&
                (int)fallback_basis.size() == m_rows) {
                red_basis2 = std::move(fallback_basis);
            }
        }
        if ((int)red_basis2.size() == m_rows &&
            !basis_is_primal_feasible_(Ared, bred, red_basis2, opt_.tol)) {
            for (int j = 0; j < (int)n_orig_eff; ++j) {
                if (std::find(red_basis2.begin(), red_basis2.end(), j) !=
                    red_basis2.end()) {
                    continue;
                }
                bool improved = false;
                for (int r = 0; r < m_rows; ++r) {
                    std::vector<int> cand = red_basis2;
                    cand[r] = j;
                    const Eigen::MatrixXd Btest =
                        Ared(Eigen::all,
                             Eigen::VectorXi::Map(cand.data(), (int)cand.size()));
                    Eigen::FullPivLU<Eigen::MatrixXd> lu(Btest);
                    if (!(lu.rank() == m_rows && lu.isInvertible())) continue;
                    if (!basis_is_primal_feasible_(Ared, bred, cand, opt_.tol)) {
                        continue;
                    }
                    red_basis2 = std::move(cand);
                    improved = true;
                    break;
                }
                if (improved) break;
            }
        }

        // Final Phase II on reduced problem (respect mode)
        LPSolution::Status status2;
        Eigen::VectorXd v2;
        std::vector<int> red_basis_out;
        int it2 = 0;
        std::unordered_map<std::string, std::string> info2;

        if ((int)red_basis2.size() == m_rows) {
            if (opt_.mode == SimplexMode::Dual) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::Infeasible) {
                    auto it = info2.find("farkas_has_cert");
                    if (it != info2.end() && it->second == "1") {
                        auto yF = parse_serialized_vec(info2, "farkas_y", m_eff);
                        return finalize_solution_(attach_postsolved_farkas_(
                            attach_internal_basis_(
                                make_solution_(
                                    LPSolution::Status::Infeasible,
                                    Eigen::VectorXd::Zero(n),
                                    std::numeric_limits<double>::infinity(), {},
                                    it2, add_info(std::move(info2)), yF, true),
                                red_basis_out, internal_column_labels),
                            P, opt_.tol));
                    }
                }

            } else if (opt_.mode == SimplexMode::Primal) {
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
            } else {
                // Auto: primal first; if negative basics → dual
                std::tie(status2, v2, red_basis_out, it2, info2) =
                    phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                if (status2 == LPSolution::Status::NeedPhase1 &&
                    info2.count("reason") &&
                    info2.at("reason") == std::string("negative_basic_vars")) {
                    std::tie(status2, v2, red_basis_out, it2, info2) =
                        dual_phase_(Ared, bred, cred, red_basis2, l_eff, u_eff);
                }
            }
        } else {
            // Fall back to find a basis internally
            std::tie(status2, v2, red_basis_out, it2, info2) =
                phase_(Ared, bred, cred, std::nullopt, l_eff, u_eff);
            if (status2 == LPSolution::Status::NeedPhase1) {
                status2 = LPSolution::Status::Singular;
                info2["note"] = "reduced matrix cannot form a proper basis";
            }
        }

        const int total_iters = it1 + it2;
        auto merged_info = add_info(std::move(info2));
        merged_info.insert({"phase1_iters", std::to_string(it1)});
        const bool has_primal_ray =
            merged_info.count("primal_ray_has_cert") &&
            merged_info.at("primal_ray_has_cert") == "1";
        const auto primal_ray_internal =
            has_primal_ray
                ? parse_serialized_vec(merged_info, "primal_ray", n_eff)
                : std::nullopt;

        auto [z_full, obj_correction] = postsolve_primal(v2);
        Eigen::VectorXd x_full = anchor + sign.cwiseProduct(z_full);
        const double total_obj = c_in.dot(x_full);

        std::vector<int> basis_full;
        basis_full.reserve(red_basis_out.size());
        for (int jr : red_basis_out) {
            if (jr >= 0 && jr < (int)col_orig_map.size()) {
                const int jorig = col_orig_map[jr];
                if (jorig >= 0) basis_full.push_back(jorig);
            }
        }

        if (status2 == LPSolution::Status::Optimal &&
            !primal_feasible_(A_in, b_in, x_full, l_in, u_in, opt_.tol)) {
            merged_info["reason"] = "invalid_returned_primal";
            return finalize_solution_(attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(LPSolution::Status::Singular, x_full,
                                           total_obj, basis_full, total_iters,
                                           std::move(merged_info), std::nullopt,
                                           std::nullopt, primal_ray_internal,
                                           has_primal_ray),
                            Ared, bred, cred, red_basis_out,
                            internal_column_labels, internal_row_labels,
                            opt_.tol),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol));
        }

        if (status2 == LPSolution::Status::Optimal) {
            return finalize_solution_(attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(LPSolution::Status::Optimal, x_full,
                                           total_obj, basis_full, total_iters,
                                           std::move(merged_info), std::nullopt,
                                           std::nullopt, primal_ray_internal,
                                           has_primal_ray),
                            Ared, bred, cred, red_basis_out,
                            internal_column_labels, internal_row_labels,
                            opt_.tol),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol));
        }
        if (status2 == LPSolution::Status::Unbounded) {
            return finalize_solution_(attach_mapped_primal_ray_(
                attach_postsolved_farkas_(
                    attach_postsolved_row_duals_(
                        attach_internal_tableau_(
                            make_solution_(
                                LPSolution::Status::Unbounded, x_full,
                                -std::numeric_limits<double>::infinity(),
                                basis_full, total_iters, std::move(merged_info),
                                std::nullopt, std::nullopt, primal_ray_internal,
                                has_primal_ray),
                            Ared, bred, cred, red_basis_out,
                            internal_column_labels, internal_row_labels,
                            opt_.tol),
                        P, opt_.tol),
                    P, opt_.tol),
                col_orig_map, sign, A_model.cols(), opt_.tol));
        }

        const double obj_fallback =
            x_full.array().isFinite().all()
                ? total_obj
                : std::numeric_limits<double>::quiet_NaN();
        return finalize_solution_(attach_mapped_primal_ray_(
            attach_postsolved_farkas_(
                attach_postsolved_row_duals_(
                    attach_internal_tableau_(
                        make_solution_(status2, x_full, obj_fallback, basis_full,
                                       total_iters, std::move(merged_info),
                                       std::nullopt, std::nullopt,
                                       primal_ray_internal, has_primal_ray),
                        Ared, bred, cred, red_basis_out, internal_column_labels,
                        internal_row_labels, opt_.tol),
                    P, opt_.tol),
                P, opt_.tol),
            col_orig_map, sign, A_model.cols(), opt_.tol));
    }

   private:
    friend class RevisedSimplexPrimalEngine;
    friend class RevisedSimplexDualEngine;

    // =========================================================================
    // Helpers (private; signatures preserved where externally referenced)
    // =========================================================================

    static Eigen::VectorXd clip_small_(Eigen::VectorXd x, double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i)
            if (std::abs(x(i)) < tol) x(i) = 0.0;
        return x;
    }

    void trace_line_(const std::string& line) const {
        if (!opt_.verbose) return;
        trace_.push_back(line);
        std::cout << line << std::endl;
    }

    bool should_trace_iter_(int iter) const {
        if (!opt_.verbose) return false;
        const int freq = std::max(1, opt_.verbose_every);
        return iter <= 1 || (iter % freq) == 0;
    }

    static std::string format_basis_(const std::vector<int>& basis) {
        std::ostringstream oss;
        oss << "[";
        for (std::size_t i = 0; i < basis.size(); ++i) {
            if (i) oss << ", ";
            oss << basis[i];
        }
        oss << "]";
        return oss.str();
    }

    static std::string format_status_(LPSolution::Status status) {
        return std::string(to_string(status));
    }

    static std::vector<std::string> make_internal_column_labels_(
        const std::vector<int>& col_orig_map) {
        std::vector<std::string> labels;
        labels.reserve(col_orig_map.size());
        for (int jr = 0; jr < (int)col_orig_map.size(); ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig >= 0) {
                labels.push_back("x_orig_" + std::to_string(jorig));
            } else {
                labels.push_back("internal_" + std::to_string(jr));
            }
        }
        return labels;
    }

    static std::vector<std::string> make_internal_row_labels_(
        const std::vector<int>& row_orig_map) {
        std::vector<std::string> labels;
        labels.reserve(row_orig_map.size());
        for (int ir = 0; ir < (int)row_orig_map.size(); ++ir) {
            const int iorig = row_orig_map[ir];
            if (iorig >= 0) {
                labels.push_back("row_orig_" + std::to_string(iorig));
            } else {
                labels.push_back("internal_row_" + std::to_string(ir));
            }
        }
        return labels;
    }

    static std::vector<int> make_nonbasis_internal_(int n,
                                                    const std::vector<int>& basis) {
        std::vector<int> nonbasis;
        if (n <= 0) return nonbasis;
        std::vector<char> in_basis(n, 0);
        for (int j : basis) {
            if (j >= 0 && j < n) in_basis[j] = 1;
        }
        nonbasis.reserve(std::max(0, n - (int)basis.size()));
        for (int j = 0; j < n; ++j) {
            if (!in_basis[j]) nonbasis.push_back(j);
        }
        return nonbasis;
    }

    static Eigen::VectorXd clip_small_vec_(Eigen::VectorXd x,
                                           double tol = 1e-12) {
        for (int i = 0; i < x.size(); ++i) {
            if (std::abs(x(i)) < tol) x(i) = 0.0;
        }
        return x;
    }

    static Eigen::MatrixXd clip_small_mat_(Eigen::MatrixXd X,
                                           double tol = 1e-12) {
        for (int i = 0; i < X.rows(); ++i) {
            for (int j = 0; j < X.cols(); ++j) {
                if (std::abs(X(i, j)) < tol) X(i, j) = 0.0;
            }
        }
        return X;
    }

    static std::string describe_presolve_action_(
        const presolve::Action& action) {
        return std::visit(
            [](const auto& act) -> std::string {
                using T = std::decay_t<decltype(act)>;
                std::ostringstream oss;
                if constexpr (std::is_same_v<T, presolve::ActRowReduce>) {
                    oss << "row_reduce old_m=" << act.old_m
                        << " keep=" << act.keep.size();
                } else if constexpr (std::is_same_v<T, presolve::ActRemoveRow>) {
                    oss << "remove_row i=" << act.i << " rhs=" << act.rhs;
                } else if constexpr (std::is_same_v<T, presolve::ActRemoveCol>) {
                    oss << "remove_col j=" << act.j << " c=" << act.c_j;
                } else if constexpr (std::is_same_v<T, presolve::ActFixVar>) {
                    oss << "fix_var j=" << act.j << " x=" << act.x_fix;
                } else if constexpr (std::is_same_v<T, presolve::ActTightenBound>) {
                    oss << "tighten_bound j=" << act.j << " old_l=" << act.old_l
                        << " old_u=" << act.old_u;
                } else if constexpr (std::is_same_v<T, presolve::ActScaleRow>) {
                    oss << "scale_row i=" << act.i << " scale=" << act.scale;
                } else if constexpr (std::is_same_v<T, presolve::ActScaleCol>) {
                    oss << "scale_col j=" << act.j << " scale=" << act.scale;
                } else if constexpr (std::is_same_v<T, presolve::ActSingletonRowElim>) {
                    oss << "singleton_row_elim i=" << act.i << " j=" << act.j
                        << " rhs=" << act.rhs;
                } else if constexpr (std::is_same_v<T, presolve::ActSingletonColElim>) {
                    oss << "singleton_col_elim j=" << act.j << " i=" << act.i
                        << " aij=" << act.aij;
                } else if constexpr (std::is_same_v<T, presolve::ActDualFix>) {
                    oss << "dual_fix j=" << act.j << " x=" << act.x_fix;
                }
                return oss.str();
            },
            action);
    }

    void trace_presolve_(const presolve::PresolveResult& pres) const {
        if (!opt_.verbose || !opt_.verbose_include_presolve) return;
        trace_line_("[presolve] actions=" + std::to_string(pres.stack.size()) +
                    " reduced_m=" + std::to_string(pres.reduced.A.rows()) +
                    " reduced_n=" + std::to_string(pres.reduced.A.cols()) +
                    " infeasible=" +
                    std::string(pres.proven_infeasible ? "1" : "0") +
                    " unbounded=" +
                    std::string(pres.proven_unbounded ? "1" : "0"));
        for (std::size_t i = 0; i < pres.stack.size(); ++i) {
            trace_line_("[presolve] #" + std::to_string(i + 1) + " " +
                        describe_presolve_action_(pres.stack[i]));
        }
    }

    LPSolution finalize_solution_(LPSolution sol) const {
        if (opt_.verbose) sol.trace = trace_;
        return sol;
    }

    static LPSolution attach_postsolved_row_duals_(
        LPSolution sol, const presolve::Presolver& P, double tol) {
        if (sol.dual_values_internal.size() != P.result().reduced.A.rows()) {
            return sol;
        }
        Eigen::VectorXd y = P.postsolve_dual(sol.dual_values_internal);
        sol.dual_values = clip_small_vec_(std::move(y), tol);
        sol.shadow_prices = sol.dual_values;
        return sol;
    }

    static LPSolution attach_postsolved_farkas_(
        LPSolution sol, const presolve::Presolver& P, double tol) {
        if (!sol.farkas_has_cert) return sol;
        if (sol.farkas_y.size() != P.result().reduced.A.rows()) return sol;
        sol.farkas_y_internal = sol.farkas_y;
        sol.farkas_y = clip_small_vec_(P.postsolve_dual(sol.farkas_y_internal), tol);
        return sol;
    }

    static LPSolution attach_mapped_primal_ray_(
        LPSolution sol, const std::vector<int>& col_orig_map,
        const Eigen::VectorXd& sign, int original_num_cols, double tol) {
        if (!sol.primal_ray_has_cert) return sol;
        if (sol.primal_ray.size() != (int)col_orig_map.size()) return sol;
        sol.primal_ray_internal = sol.primal_ray;
        Eigen::VectorXd mapped = Eigen::VectorXd::Zero(original_num_cols);
        for (int jr = 0; jr < (int)col_orig_map.size(); ++jr) {
            const int jorig = col_orig_map[jr];
            if (jorig < 0 || jorig >= original_num_cols || jorig >= sign.size()) {
                continue;
            }
            mapped(jorig) += sign(jorig) * sol.primal_ray_internal(jr);
        }
        sol.primal_ray = clip_small_vec_(std::move(mapped), tol);
        return sol;
    }

    static LPSolution attach_internal_basis_(
        LPSolution sol, std::vector<int> basis_internal,
        std::vector<std::string> internal_column_labels) {
        sol.basis_internal = std::move(basis_internal);
        sol.internal_column_labels = std::move(internal_column_labels);
        return sol;
    }

    static LPSolution attach_internal_tableau_(
        LPSolution sol, const Eigen::MatrixXd& A_internal,
        const Eigen::VectorXd& b_internal, const Eigen::VectorXd& c_internal,
        std::vector<int> basis_internal,
        std::vector<std::string> internal_column_labels,
        std::vector<std::string> internal_row_labels, double tol) {
        sol.basis_internal = std::move(basis_internal);
        sol.internal_column_labels = std::move(internal_column_labels);
        sol.internal_row_labels = std::move(internal_row_labels);
        sol.nonbasis_internal =
            make_nonbasis_internal_(static_cast<int>(A_internal.cols()),
                                    sol.basis_internal);

        const int m = static_cast<int>(A_internal.rows());
        const int n = static_cast<int>(A_internal.cols());
        if (m == 0) {
            sol.tableau = Eigen::MatrixXd::Zero(0, n);
            sol.tableau_rhs = Eigen::VectorXd::Zero(0);
            sol.reduced_costs_internal = clip_small_vec_(c_internal, tol);
            sol.dual_values_internal = Eigen::VectorXd::Zero(0);
            sol.shadow_prices_internal = Eigen::VectorXd::Zero(0);
            sol.has_internal_tableau = true;
            return sol;
        }
        if ((int)sol.basis_internal.size() != m) return sol;

        Eigen::VectorXi basis_idx =
            Eigen::Map<const Eigen::VectorXi>(sol.basis_internal.data(), m);
        const Eigen::MatrixXd B = A_internal(Eigen::all, basis_idx);
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (!(lu.rank() == m && lu.isInvertible())) return sol;

        sol.tableau = clip_small_mat_(lu.solve(A_internal), tol);
        sol.tableau_rhs = clip_small_vec_(lu.solve(b_internal), tol);

        Eigen::VectorXd cB(m);
        for (int i = 0; i < m; ++i) cB(i) = c_internal(sol.basis_internal[i]);
        Eigen::FullPivLU<Eigen::MatrixXd> lu_t(B.transpose());
        if (lu_t.rank() == m && lu_t.isInvertible()) {
            const Eigen::VectorXd y = lu_t.solve(cB);
            sol.dual_values_internal = clip_small_vec_(y, tol);
            sol.shadow_prices_internal = sol.dual_values_internal;
            sol.reduced_costs_internal =
                clip_small_vec_(c_internal - A_internal.transpose() * y, tol);
        }

        sol.has_internal_tableau = true;
        return sol;
    }

    struct SanitizedBounds {
        Eigen::VectorXd l;
        Eigen::VectorXd u;
        int relaxed_lower = 0;
        int relaxed_upper = 0;
    };

    SanitizedBounds canonicalize_inactive_huge_bounds_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& l, const Eigen::VectorXd& u) const {
        SanitizedBounds out{l, u, 0, 0};
        if (A.rows() == 0 || A.cols() == 0) return out;

        double data_scale = 1.0;
        if (A.size() > 0) data_scale = std::max(data_scale, A.cwiseAbs().maxCoeff());
        if (b.size() > 0) data_scale = std::max(data_scale, b.cwiseAbs().maxCoeff());

        // Only relax bounds that are both numerically huge and provably much
        // looser than what the equality rows already imply.
        const double huge_bound = 1e6 * data_scale;
        const double relax_gap = 1e6;

        for (int j = 0; j < A.cols(); ++j) {
            if (std::isfinite(out.u(j)) && out.u(j) > huge_bound) {
                double implied_u = presolve::inf();
                bool has_implied_u = false;
                for (int i = 0; i < A.rows(); ++i) {
                    const double aij = A(i, j);
                    if (std::abs(aij) <= opt_.tol) continue;
                    const auto other = presolve::row_activity_range_excluding(
                        A.row(i), out.l, out.u, j, opt_.tol);
                    if (aij > 0.0 && other.min_finite) {
                        implied_u = std::min(implied_u, (b(i) - other.min_act) / aij);
                        has_implied_u = true;
                    } else if (aij < 0.0 && other.max_finite) {
                        implied_u = std::min(implied_u, (b(i) - other.max_act) / aij);
                        has_implied_u = true;
                    }
                }
                if (has_implied_u && std::isfinite(implied_u)) {
                    const double ref =
                        std::max({1.0, std::abs(implied_u), data_scale});
                    if (out.u(j) > implied_u + relax_gap * ref &&
                        out.u(j) > huge_bound) {
                        out.u(j) = presolve::inf();
                        ++out.relaxed_upper;
                    }
                }
            }

            if (std::isfinite(out.l(j)) && out.l(j) < -huge_bound) {
                double implied_l = presolve::ninf();
                bool has_implied_l = false;
                for (int i = 0; i < A.rows(); ++i) {
                    const double aij = A(i, j);
                    if (std::abs(aij) <= opt_.tol) continue;
                    const auto other = presolve::row_activity_range_excluding(
                        A.row(i), out.l, out.u, j, opt_.tol);
                    if (aij > 0.0 && other.max_finite) {
                        implied_l = std::max(implied_l, (b(i) - other.max_act) / aij);
                        has_implied_l = true;
                    } else if (aij < 0.0 && other.min_finite) {
                        implied_l = std::max(implied_l, (b(i) - other.min_act) / aij);
                        has_implied_l = true;
                    }
                }
                if (has_implied_l && std::isfinite(implied_l)) {
                    const double ref =
                        std::max({1.0, std::abs(implied_l), data_scale});
                    if (out.l(j) < implied_l - relax_gap * ref &&
                        -out.l(j) > huge_bound) {
                        out.l(j) = presolve::ninf();
                        ++out.relaxed_lower;
                    }
                }
            }
        }

        return out;
    }

    static bool primal_feasible_(const Eigen::MatrixXd& A,
                                 const Eigen::VectorXd& b,
                                 const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& l,
                                 const Eigen::VectorXd& u, double tol) {
        if (x.size() == 0 || !x.array().isFinite().all()) return false;
        if (A.rows() > 0) {
            const Eigen::VectorXd resid = A * x - b;
            if (resid.size() > 0 &&
                resid.lpNorm<Eigen::Infinity>() > 100.0 * tol) {
                return false;
            }
        }
        for (int j = 0; j < x.size(); ++j) {
            if (j < l.size() && std::isfinite(l(j)) &&
                x(j) < l(j) - 100.0 * tol) {
                return false;
            }
            if (j < u.size() && std::isfinite(u(j)) &&
                x(j) > u(j) + 100.0 * tol) {
                return false;
            }
        }
        return true;
    }

    static bool can_increase_from_lower_(int j, const Eigen::VectorXd& l,
                                         const Eigen::VectorXd& u,
                                         double tol) {
        const double x_at_bound =
            (j >= 0 && j < l.size() && std::isfinite(l(j))) ? l(j) : 0.0;
        if (j >= 0 && j < u.size() && std::isfinite(u(j))) {
            return (u(j) - x_at_bound) > tol;
        }
        return true;
    }

    FTBasis::Options make_basis_options_() const {
        FTBasis::Options bopt;
        bopt.refactor_every = opt_.refactor_every;
        bopt.compress_every = opt_.compress_every;
        bopt.pivot_rel = opt_.lu_pivot_rel;
        bopt.abs_floor = opt_.lu_abs_floor;
        bopt.alpha_tol = opt_.alpha_tol;
        bopt.z_inf_guard = opt_.z_inf_guard;
        bopt.ft_bandwidth_cap = opt_.ft_bandwidth_cap;

        std::string mode = opt_.basis_update;
        std::transform(mode.begin(), mode.end(), mode.begin(),
                       [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        if (mode == "eta" || mode == "eta_stack") {
            bopt.update_mode = FTBasis::Options::UpdateMode::EtaStack;
        } else {
            bopt.update_mode = FTBasis::Options::UpdateMode::ForrestTomlin;
        }

        return bopt;
    }

    static Eigen::VectorXd assemble_primal_(int n,
                                            const std::vector<int>& basis,
                                            const Eigen::VectorXd& xB,
                                            const Eigen::VectorXd& l,
                                            const Eigen::VectorXd& u,
                                            const std::vector<int>* sigma =
                                                nullptr) {
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        std::vector<char> inB(n, 0);
        for (int i = 0; i < (int)basis.size(); ++i) {
            const int j = basis[i];
            if (j >= 0 && j < n) {
                inB[j] = 1;
                if (i < xB.size()) x(j) = xB(i);
            }
        }

        for (int j = 0; j < n; ++j) {
            if (inB[j]) continue;

            const bool upper_view =
                sigma && j < (int)sigma->size() && (*sigma)[j] < 0;
            if (upper_view && j < u.size() && std::isfinite(u(j))) {
                x(j) = u(j);
            } else if (j < l.size() && std::isfinite(l(j))) {
                x(j) = l(j);
            } else {
                x(j) = 0.0;
            }
        }

        return clip_small_(x);
    }

    static Eigen::VectorXd repair_nan_primal_(const Eigen::MatrixXd& A,
                                              const Eigen::VectorXd& b,
                                              const Eigen::VectorXd& l,
                                              const Eigen::VectorXd& u,
                                              Eigen::VectorXd x,
                                              double tol = 1e-9) {
        if (x.size() == 0 || x.array().isFinite().all()) return x;

        std::vector<int> unknown;
        std::vector<int> known;
        unknown.reserve(x.size());
        known.reserve(x.size());
        for (int j = 0; j < x.size(); ++j) {
            if (std::isfinite(x(j)))
                known.push_back(j);
            else
                unknown.push_back(j);
        }

        if (!unknown.empty() && A.rows() > 0) {
            Eigen::VectorXd rhs = b;
            for (int j : known) rhs.noalias() -= A.col(j) * x(j);

            Eigen::MatrixXd AU(A.rows(), static_cast<int>(unknown.size()));
            for (int k = 0; k < (int)unknown.size(); ++k) {
                AU.col(k) = A.col(unknown[k]);
            }

            if (AU.size() > 0) {
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(AU);
                const Eigen::VectorXd z = qr.solve(rhs);
                if (z.size() == (int)unknown.size() && z.array().isFinite().all()) {
                    for (int k = 0; k < (int)unknown.size(); ++k) {
                        x(unknown[k]) = z(k);
                    }
                }
            }
        }

        for (int j = 0; j < x.size(); ++j) {
            if (!std::isfinite(x(j))) {
                if (j < l.size() && std::isfinite(l(j)))
                    x(j) = l(j);
                else if (j < u.size() && std::isfinite(u(j)))
                    x(j) = u(j);
                else
                    x(j) = 0.0;
            }
            if (j < l.size() && std::isfinite(l(j)) && x(j) < l(j) - tol) x(j) = l(j);
            if (j < u.size() && std::isfinite(u(j)) && x(j) > u(j) + tol) x(j) = u(j);
        }

        return clip_small_(x);
    }

    struct CrashCandidate {
        int col = -1;
        int pivot_row = -1;
        double score = -std::numeric_limits<double>::infinity();
    };

    enum class CrashStyle { Hybrid, Repair, Sprint, CrashII, CrashIII };

    struct CrashAttemptConfig {
        CrashStyle style = CrashStyle::Hybrid;
        std::string style_name = "hybrid";
        double markowitz_threshold = 0.2;
        double cost_penalty = 0.05;
        double rhs_bonus = 0.25;
        double dense_penalty = 0.5;
        double coverage_weight = 1.0;
        double seed_penalty = 0.0;
        double jitter = 0.0;
        int local_search_passes = 0;
        int max_swap_candidates = 8;
        bool prefer_seed_columns = false;
    };

    struct BasisQuality {
        bool valid = false;
        bool primal_feasible = false;
        bool dual_feasible = false;
        int rank = 0;
        double primal_violation = std::numeric_limits<double>::infinity();
        double dual_violation = std::numeric_limits<double>::infinity();
        double density = std::numeric_limits<double>::infinity();
    };

    struct CrashSelection {
        std::vector<int> basis;
        BasisQuality quality;
        std::string source = "none";
        std::string style = "none";
        int attempt = -1;
    };

    static double positive_violation_max_(const Eigen::VectorXd& x, double tol) {
        double worst = 0.0;
        for (int i = 0; i < x.size(); ++i) {
            worst = std::max(worst, x(i) - tol);
        }
        return worst;
    }

    static BasisQuality evaluate_basis_quality_(const Eigen::MatrixXd& A,
                                                const Eigen::VectorXd& b,
                                                const Eigen::VectorXd& c,
                                                const std::vector<int>& basis,
                                                double tol) {
        BasisQuality q;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        if ((int)basis.size() != m || m == 0) {
            if (m == 0 && basis.empty()) {
                q.valid = true;
                q.primal_feasible = true;
                q.dual_feasible = true;
                q.rank = 0;
                q.primal_violation = 0.0;
                q.dual_violation = 0.0;
                q.density = 0.0;
            }
            return q;
        }

        std::vector<char> in_basis(n, 0);
        for (int j : basis) {
            if (j < 0 || j >= n || in_basis[j]) return q;
            in_basis[j] = 1;
        }

        const Eigen::MatrixXd B =
            A(Eigen::all, Eigen::VectorXi::Map(basis.data(), m));
        q.density = ((B.array().abs() > 1e-12).cast<double>().sum()) /
                    std::max(1.0, static_cast<double>(m) * m);

        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        q.rank = lu.rank();
        if (q.rank != m || !lu.isInvertible()) return q;

        q.valid = true;
        const Eigen::VectorXd xB = lu.solve(b);
        q.primal_violation = positive_violation_max_(-xB, tol);
        q.primal_feasible = xB.allFinite() && q.primal_violation <= tol;

        Eigen::VectorXd cB(m);
        for (int i = 0; i < m; ++i) cB(i) = c(basis[i]);
        Eigen::FullPivLU<Eigen::MatrixXd> luT(B.transpose());
        const Eigen::VectorXd y = luT.solve(cB);
        if (!y.allFinite()) return q;

        Eigen::VectorXd neg_rc = Eigen::VectorXd::Zero(n - m);
        int k = 0;
        for (int j = 0; j < n; ++j) {
            if (in_basis[j]) continue;
            neg_rc(k++) = -(c(j) - A.col(j).dot(y));
        }
        q.dual_violation = positive_violation_max_(neg_rc, tol);
        q.dual_feasible = q.dual_violation <= tol;
        return q;
    }

    static bool better_basis_quality_(const CrashSelection& lhs,
                                      const CrashSelection& rhs,
                                      SimplexMode mode) {
        const BasisQuality& a = lhs.quality;
        const BasisQuality& b = rhs.quality;
        if (a.valid != b.valid) return a.valid;
        if (a.rank != b.rank) return a.rank > b.rank;

        if (mode == SimplexMode::Auto) {
            const int a_feasible =
                static_cast<int>(a.primal_feasible) + static_cast<int>(a.dual_feasible);
            const int b_feasible =
                static_cast<int>(b.primal_feasible) + static_cast<int>(b.dual_feasible);
            if (a_feasible != b_feasible) return a_feasible > b_feasible;

            const double a_best =
                std::min(a.primal_violation, a.dual_violation);
            const double b_best =
                std::min(b.primal_violation, b.dual_violation);
            if (std::abs(a_best - b_best) > 1e-12) return a_best < b_best;

            const double a_total = a.primal_violation + a.dual_violation;
            const double b_total = b.primal_violation + b.dual_violation;
            if (std::abs(a_total - b_total) > 1e-12) return a_total < b_total;
        }

        const bool a_primary =
            (mode == SimplexMode::Dual) ? a.dual_feasible : a.primal_feasible;
        const bool b_primary =
            (mode == SimplexMode::Dual) ? b.dual_feasible : b.primal_feasible;
        if (a_primary != b_primary) return a_primary;

        const bool a_secondary =
            (mode == SimplexMode::Dual) ? a.primal_feasible : a.dual_feasible;
        const bool b_secondary =
            (mode == SimplexMode::Dual) ? b.primal_feasible : b.dual_feasible;
        if (a_secondary != b_secondary) return a_secondary;

        const double a_primary_violation = (mode == SimplexMode::Dual)
                                               ? a.dual_violation
                                               : a.primal_violation;
        const double b_primary_violation = (mode == SimplexMode::Dual)
                                               ? b.dual_violation
                                               : b.primal_violation;
        if (std::abs(a_primary_violation - b_primary_violation) > 1e-12) {
            return a_primary_violation < b_primary_violation;
        }

        const double a_secondary_violation = (mode == SimplexMode::Dual)
                                                 ? a.primal_violation
                                                 : a.dual_violation;
        const double b_secondary_violation = (mode == SimplexMode::Dual)
                                                 ? b.primal_violation
                                                 : b.dual_violation;
        if (std::abs(a_secondary_violation - b_secondary_violation) > 1e-12) {
            return a_secondary_violation < b_secondary_violation;
        }

        if (std::abs(a.density - b.density) > 1e-12) return a.density < b.density;
        if (lhs.attempt != rhs.attempt) return lhs.attempt < rhs.attempt;
        return std::lexicographical_compare(lhs.basis.begin(), lhs.basis.end(),
                                            rhs.basis.begin(), rhs.basis.end());
    }

    static std::string lower_copy_(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        return value;
    }

    static CrashStyle parse_crash_style_(const std::string& strategy) {
        const std::string key = lower_copy_(strategy);
        if (key.empty() || key == "hybrid" || key == "auto") {
            return CrashStyle::Hybrid;
        }
        if (key == "repair" || key == "repair_warm_start") {
            return CrashStyle::Repair;
        }
        if (key == "sprint") return CrashStyle::Sprint;
        if (key == "crash_ii" || key == "crash-ii" || key == "crash2") {
            return CrashStyle::CrashII;
        }
        if (key == "crash_iii" || key == "crash-iii" || key == "crash3") {
            return CrashStyle::CrashIII;
        }
        return CrashStyle::Hybrid;
    }

    static const char* crash_style_name_(CrashStyle style) {
        switch (style) {
            case CrashStyle::Repair:
                return "repair";
            case CrashStyle::Sprint:
                return "sprint";
            case CrashStyle::CrashII:
                return "crash_ii";
            case CrashStyle::CrashIII:
                return "crash_iii";
            case CrashStyle::Hybrid:
            default:
                return "hybrid";
        }
    }

    static CrashAttemptConfig crash_attempt_config_(
        const RevisedSimplexOptions& opt, int attempt) {
        CrashAttemptConfig cfg;
        const double base = std::clamp(opt.crash_markowitz_tol, 1e-3, 0.95);
        CrashStyle style = parse_crash_style_(opt.crash_strategy);
        if (style == CrashStyle::Hybrid) {
            switch (attempt % 4) {
                case 0:
                    style = CrashStyle::Repair;
                    break;
                case 1:
                    style = CrashStyle::Sprint;
                    break;
                case 2:
                    style = CrashStyle::CrashII;
                    break;
                default:
                    style = CrashStyle::CrashIII;
                    break;
            }
        }

        cfg.style = style;
        cfg.style_name = crash_style_name_(style);
        switch (style) {
            case CrashStyle::Repair:
                cfg.markowitz_threshold = std::max(1e-3, 0.45 * base);
                cfg.cost_penalty = 0.02;
                cfg.rhs_bonus = 0.45;
                cfg.dense_penalty = 0.25;
                cfg.coverage_weight = 1.20;
                cfg.seed_penalty = 6.0;
                cfg.local_search_passes = 2;
                cfg.max_swap_candidates = 12;
                cfg.prefer_seed_columns = true;
                break;
            case CrashStyle::Sprint:
                cfg.markowitz_threshold = std::max(1e-3, 0.60 * base);
                cfg.cost_penalty = 0.02;
                cfg.rhs_bonus = 0.20;
                cfg.dense_penalty = 0.35;
                cfg.coverage_weight = 1.40;
                cfg.local_search_passes = 1;
                cfg.max_swap_candidates = 8;
                break;
            case CrashStyle::CrashII:
                cfg.markowitz_threshold = base;
                cfg.cost_penalty = 0.05;
                cfg.rhs_bonus = 0.25;
                cfg.dense_penalty = 0.50;
                cfg.coverage_weight = 1.00;
                cfg.local_search_passes = 1;
                cfg.max_swap_candidates = 10;
                break;
            case CrashStyle::CrashIII:
                cfg.markowitz_threshold = std::min(0.95, 1.6 * base);
                cfg.cost_penalty = 0.08;
                cfg.rhs_bonus = 0.15;
                cfg.dense_penalty = 0.65;
                cfg.coverage_weight = 0.80;
                cfg.local_search_passes = 2;
                cfg.max_swap_candidates = 14;
                break;
            case CrashStyle::Hybrid:
            default:
                break;
        }
        cfg.jitter = 1e-6 * static_cast<double>(attempt + 1);
        return cfg;
    }

    static void mark_pivot_row_(const Eigen::MatrixXd& A, int col,
                                int pivot_row_hint,
                                std::vector<char>& used_row) {
        if (pivot_row_hint >= 0 && pivot_row_hint < (int)used_row.size() &&
            !used_row[pivot_row_hint]) {
            used_row[pivot_row_hint] = 1;
            return;
        }

        int best_row = -1;
        double best_abs = 0.0;
        for (int i = 0; i < A.rows(); ++i) {
            if (used_row[i]) continue;
            const double aa = std::abs(A(i, col));
            if (aa > best_abs) {
                best_abs = aa;
                best_row = i;
            }
        }
        if (best_row >= 0) used_row[best_row] = 1;
    }

    static bool try_add_basis_column_(const Eigen::MatrixXd& A,
                                      std::vector<int>& basis,
                                      std::vector<char>& used_row,
                                      std::vector<char>& used_col,
                                      int& current_rank, int col,
                                      int pivot_row_hint) {
        const int n = static_cast<int>(A.cols());
        if (col < 0 || col >= n || used_col[col]) return false;
        used_col[col] = 1;

        std::vector<int> candidate = basis;
        candidate.push_back(col);
        const Eigen::MatrixXd Bcand =
            A(Eigen::all, Eigen::VectorXi::Map(candidate.data(),
                                              (int)candidate.size()));
        Eigen::FullPivLU<Eigen::MatrixXd> lu(Bcand);
        const int rank = lu.rank();
        if (rank <= current_rank) return false;

        basis.push_back(col);
        current_rank = rank;
        mark_pivot_row_(A, col, pivot_row_hint, used_row);
        return true;
    }

    static CrashCandidate choose_slack_like_column_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const std::vector<char>& used_row,
        const std::vector<char>& used_col) {
        CrashCandidate best;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        double c_scale = 1.0;
        if (c.size() > 0) c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

        for (int j = 0; j < n; ++j) {
            if (used_col[j]) continue;
            int pivot_row = -1;
            int nnz = 0;
            double pivot = 0.0;
            double off_sum = 0.0;
            for (int i = 0; i < m; ++i) {
                const double aij = A(i, j);
                if (std::abs(aij) <= 1e-12) continue;
                ++nnz;
                if (used_row[i]) {
                    off_sum += std::abs(aij);
                    continue;
                }
                if (std::abs(aij) > std::abs(pivot)) {
                    if (pivot_row >= 0) off_sum += std::abs(pivot);
                    pivot_row = i;
                    pivot = aij;
                } else {
                    off_sum += std::abs(aij);
                }
            }
            if (pivot_row < 0 || std::abs(pivot) <= 1e-12) continue;

            const bool exact_unit =
                (nnz == 1 && std::abs(std::abs(pivot) - 1.0) <= 1e-10);
            const bool slack_like = (nnz == 1) || (off_sum <= 1e-10);
            if (!slack_like) continue;

            double score = exact_unit ? 1e6 : 1e5;
            score += 1e3 / (1.0 + off_sum);
            score += 10.0 / (1.0 + std::abs(std::abs(pivot) - 1.0));
            if (b(pivot_row) >= -1e-10) score += 50.0;
            score -= std::abs(c(j)) / c_scale;
            score -= 0.01 * static_cast<double>(j);

            if (score > best.score) {
                best = {j, pivot_row, score};
            }
        }
        return best;
    }

    static CrashCandidate choose_sprint_column_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const std::vector<char>& used_row,
        const std::vector<char>& used_col, const CrashAttemptConfig& cfg) {
        CrashCandidate best;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        double c_scale = 1.0;
        if (c.size() > 0) c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

        for (int j = 0; j < n; ++j) {
            if (used_col[j]) continue;

            int pivot_row = -1;
            double pivot_abs = 0.0;
            double uncovered_sum = 0.0;
            int uncovered_nnz = 0;
            int total_nnz = 0;
            for (int i = 0; i < m; ++i) {
                const double aa = std::abs(A(i, j));
                if (aa <= 1e-12) continue;
                ++total_nnz;
                if (used_row[i]) continue;
                ++uncovered_nnz;
                uncovered_sum += aa;
                if (aa > pivot_abs) {
                    pivot_abs = aa;
                    pivot_row = i;
                }
            }
            if (pivot_row < 0 || pivot_abs <= 1e-12) continue;

            const double coverage =
                uncovered_sum / std::max(1e-12, A.col(j).lpNorm<1>());
            const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
            const double rhs_bonus =
                (b(pivot_row) >= -1e-10) ? cfg.rhs_bonus : 0.0;
            const double cost_penalty =
                cfg.cost_penalty * (std::abs(c(j)) / c_scale);
            const double jitter =
                cfg.jitter * std::cos(static_cast<double>((j + 1) * (pivot_row + 1)));
            const double score =
                90.0 * cfg.coverage_weight * coverage +
                25.0 * sparsity_bonus + 5.0 * pivot_abs + rhs_bonus -
                cost_penalty - 0.001 * static_cast<double>(j) + jitter;
            if (score > best.score) best = {j, pivot_row, score};
        }
        return best;
    }

    static CrashCandidate choose_triangular_column_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const std::vector<char>& used_row,
        const std::vector<char>& used_col, const CrashAttemptConfig& cfg) {
        CrashCandidate best;
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        double c_scale = 1.0;
        if (c.size() > 0) c_scale = std::max(1.0, c.cwiseAbs().maxCoeff());

        for (int j = 0; j < n; ++j) {
            if (used_col[j]) continue;

            int pivot_row = -1;
            double pivot_abs = 0.0;
            double uncovered_sum = 0.0;
            double covered_sum = 0.0;
            int uncovered_nnz = 0;
            int total_nnz = 0;
            for (int i = 0; i < m; ++i) {
                const double aij = A(i, j);
                const double aa = std::abs(aij);
                if (aa <= 1e-12) continue;
                ++total_nnz;
                if (used_row[i]) {
                    covered_sum += aa;
                    continue;
                }
                ++uncovered_nnz;
                uncovered_sum += aa;
                if (aa > pivot_abs) {
                    pivot_abs = aa;
                    pivot_row = i;
                }
            }
            if (pivot_row < 0 || pivot_abs <= 1e-12) continue;

            int row_nnz = 0;
            double row_max = 0.0;
            for (int jj = 0; jj < n; ++jj) {
                if (used_col[jj]) continue;
                const double aa = std::abs(A(pivot_row, jj));
                if (aa <= 1e-12) continue;
                ++row_nnz;
                row_max = std::max(row_max, aa);
            }
            if (row_max <= 1e-12) continue;
            if (pivot_abs + 1e-12 < cfg.markowitz_threshold * row_max) continue;

            const double dominance = pivot_abs / std::max(1e-12, uncovered_sum);
            const double triangularity =
                pivot_abs / std::max(1e-12, covered_sum + uncovered_sum);
            const double sparsity_bonus = 1.0 / (1.0 + total_nnz);
            const double rhs_bonus =
                (b(pivot_row) >= -1e-10) ? cfg.rhs_bonus : 0.0;
            const double cost_penalty =
                cfg.cost_penalty * (std::abs(c(j)) / c_scale);
            const double markowitz_penalty =
                static_cast<double>(std::max(0, row_nnz - 1) *
                                    std::max(0, uncovered_nnz - 1));
            const double markowitz_bonus = 1.0 / (1.0 + markowitz_penalty);
            const double dense_penalty =
                cfg.dense_penalty * (covered_sum / std::max(1e-12, pivot_abs));
            const double jitter =
                cfg.jitter * std::sin(static_cast<double>((j + 1) * (pivot_row + 1)));

            const double score =
                100.0 * dominance * cfg.coverage_weight +
                30.0 * triangularity +
                20.0 * markowitz_bonus + 10.0 * sparsity_bonus + rhs_bonus -
                cost_penalty - dense_penalty - 0.001 * static_cast<double>(j) +
                jitter;
            if (score > best.score) {
                best = {j, pivot_row, score};
            }
        }
        return best;
    }

    static std::vector<int> rank_remaining_columns_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& c,
        const std::vector<char>& used_col, const CrashAttemptConfig& cfg) {
        std::vector<int> ranked;
        ranked.reserve(A.cols());
        for (int j = 0; j < A.cols(); ++j) {
            if (!used_col[j]) ranked.push_back(j);
        }
        std::sort(ranked.begin(), ranked.end(), [&](int a, int b_idx) {
            const double nnz_a =
                (A.col(a).array().abs() > 1e-12).cast<double>().sum();
            const double nnz_b =
                (A.col(b_idx).array().abs() > 1e-12).cast<double>().sum();
            const double score_a =
                nnz_a + cfg.cost_penalty * std::abs(c(a)) +
                cfg.jitter * std::sin(static_cast<double>(a + 1));
            const double score_b =
                nnz_b + cfg.cost_penalty * std::abs(c(b_idx)) +
                cfg.jitter * std::sin(static_cast<double>(b_idx + 1));
            if (std::abs(score_a - score_b) > 1e-12) return score_a < score_b;
            return a < b_idx;
        });
        return ranked;
    }

    static std::vector<int> improve_basis_by_swaps_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, std::vector<int> basis,
        const CrashAttemptConfig& cfg, double tol, SimplexMode mode,
        std::optional<std::vector<int>> seed_basis = std::nullopt) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        if ((int)basis.size() != m || cfg.local_search_passes <= 0) {
            return basis;
        }

        std::vector<char> seeded(n, 0);
        if (seed_basis) {
            for (int j : *seed_basis) {
                if (j >= 0 && j < n) seeded[j] = 1;
            }
        }

        CrashSelection best;
        best.basis = basis;
        best.quality = evaluate_basis_quality_(A, b, c, basis, tol);

        for (int pass = 0; pass < cfg.local_search_passes; ++pass) {
            std::vector<char> in_basis(n, 0);
            for (int j : basis)
                if (j >= 0 && j < n) in_basis[j] = 1;

            std::vector<int> nonbasic;
            nonbasic.reserve(std::max(0, n - m));
            for (int j = 0; j < n; ++j)
                if (!in_basis[j]) nonbasic.push_back(j);
            std::sort(nonbasic.begin(), nonbasic.end(), [&](int a, int b_idx) {
                const double nnz_a =
                    (A.col(a).array().abs() > 1e-12).cast<double>().sum();
                const double nnz_b =
                    (A.col(b_idx).array().abs() > 1e-12).cast<double>().sum();
                const double score_a = nnz_a + cfg.cost_penalty * std::abs(c(a));
                const double score_b =
                    nnz_b + cfg.cost_penalty * std::abs(c(b_idx));
                if (std::abs(score_a - score_b) > 1e-12) return score_a < score_b;
                return a < b_idx;
            });
            if ((int)nonbasic.size() > cfg.max_swap_candidates) {
                nonbasic.resize(cfg.max_swap_candidates);
            }

            std::vector<int> positions(m);
            std::iota(positions.begin(), positions.end(), 0);
            std::stable_sort(positions.begin(), positions.end(),
                             [&](int lhs, int rhs) {
                                 const bool lhs_seed = seeded[basis[lhs]];
                                 const bool rhs_seed = seeded[basis[rhs]];
                                 if (cfg.prefer_seed_columns && lhs_seed != rhs_seed) {
                                     return !lhs_seed && rhs_seed;
                                 }
                                 return basis[lhs] < basis[rhs];
                             });

            bool improved = false;
            for (int entering : nonbasic) {
                for (int pos : positions) {
                    std::vector<int> cand = basis;
                    cand[pos] = entering;

                    CrashSelection trial;
                    trial.basis = std::move(cand);
                    trial.quality =
                        evaluate_basis_quality_(A, b, c, trial.basis, tol);
                    if (!trial.quality.valid) continue;
                    if (!better_basis_quality_(trial, best, mode)) continue;
                    basis = trial.basis;
                    best = std::move(trial);
                    improved = true;
                    break;
                }
                if (improved) break;
            }
            if (!improved) break;
        }

        return basis;
    }

    static std::vector<int> build_basis_attempt_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const CrashAttemptConfig& cfg,
        double tol, SimplexMode mode,
        std::optional<std::vector<int>> seed_basis = std::nullopt) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());
        if (m == 0) return {};
        if (n < m) return {};

        std::vector<int> basis;
        basis.reserve(m);
        std::vector<char> used_row(m, 0), used_col(n, 0);
        int current_rank = 0;

        if (seed_basis) {
            for (int j : *seed_basis) {
                if ((int)basis.size() == m) break;
                (void)try_add_basis_column_(A, basis, used_row, used_col,
                                            current_rank, j, -1);
            }
        }

        while ((int)basis.size() < m) {
            const CrashCandidate cand =
                choose_slack_like_column_(A, b, c, used_row, used_col);
            if (cand.col < 0) break;
            if (!try_add_basis_column_(A, basis, used_row, used_col,
                                       current_rank, cand.col,
                                       cand.pivot_row)) {
                continue;
            }
        }

        if (cfg.style == CrashStyle::Sprint) {
            while ((int)basis.size() < m) {
                const CrashCandidate cand =
                    choose_sprint_column_(A, b, c, used_row, used_col, cfg);
                if (cand.col < 0) break;
                if (!try_add_basis_column_(A, basis, used_row, used_col,
                                           current_rank, cand.col,
                                           cand.pivot_row)) {
                    continue;
                }
            }
        }

        while ((int)basis.size() < m) {
            const CrashCandidate cand =
                choose_triangular_column_(A, b, c, used_row, used_col, cfg);
            if (cand.col < 0) break;
            if (!try_add_basis_column_(A, basis, used_row, used_col,
                                       current_rank, cand.col,
                                       cand.pivot_row)) {
                continue;
            }
        }

        if ((int)basis.size() < m) {
            for (int j : rank_remaining_columns_(A, c, used_col, cfg)) {
                if ((int)basis.size() == m) break;
                (void)try_add_basis_column_(A, basis, used_row, used_col,
                                            current_rank, j, -1);
            }
        }

        if ((int)basis.size() != m) return {};
        return improve_basis_by_swaps_(A, b, c, std::move(basis), cfg, tol, mode,
                                       seed_basis);
    }

    static CrashSelection choose_initial_basis_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c, const RevisedSimplexOptions& opt,
        std::optional<std::vector<int>> seed_basis = std::nullopt) {
        CrashSelection best;

        auto consider = [&](std::vector<int> candidate, std::string source,
                            int attempt) {
            if (candidate.empty() && A.rows() != 0) return;
            CrashSelection sel;
            sel.basis = std::move(candidate);
            sel.quality = evaluate_basis_quality_(A, b, c, sel.basis, opt.tol);
            sel.source = std::move(source);
            sel.style = (attempt >= 0)
                            ? crash_attempt_config_(opt, attempt).style_name
                            : "mapped";
            sel.attempt = attempt;
            if (better_basis_quality_(sel, best, opt.mode)) best = std::move(sel);
        };

        if (seed_basis && !seed_basis->empty()) {
            if ((int)seed_basis->size() == A.rows()) {
                consider(*seed_basis, "warm_start", -1);
            }
            if (opt.repair_mapped_basis) {
                const int attempts = std::max(1, opt.crash_attempts);
                for (int k = 0; k < attempts; ++k) {
                    consider(build_basis_attempt_(A, b, c,
                                                 crash_attempt_config_(opt, k),
                                                 opt.tol,
                                                 opt.mode,
                                                 seed_basis),
                             "repaired_warm_start", k);
                }
            }
        }

        const int attempts = std::max(1, opt.crash_attempts);
        for (int k = 0; k < attempts; ++k) {
            consider(build_basis_attempt_(A, b, c, crash_attempt_config_(opt, k),
                                         opt.tol,
                                         opt.mode),
                     "crash", k);
        }
        return best;
    }

    static std::optional<std::vector<int>> find_initial_basis_(
        const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
        const Eigen::VectorXd& c,
        const RevisedSimplexOptions& opt = RevisedSimplexOptions{},
        std::optional<std::vector<int>> seed_basis = std::nullopt) {
        CrashSelection sel = choose_initial_basis_(A, b, c, opt, seed_basis);
        if (!sel.quality.valid) return std::nullopt;
        return sel.basis;
    }

    static bool basis_is_primal_feasible_(const Eigen::MatrixXd& A,
                                          const Eigen::VectorXd& b,
                                          const std::vector<int>& basis,
                                          double tol) {
        const int m = static_cast<int>(A.rows());
        if ((int)basis.size() != m) return false;
        if (m == 0) return true;
        const Eigen::MatrixXd B =
            A(Eigen::all, Eigen::VectorXi::Map(basis.data(), m));
        Eigen::FullPivLU<Eigen::MatrixXd> lu(B);
        if (lu.rank() != m || !lu.isInvertible()) return false;
        const Eigen::VectorXd xB = lu.solve(b);
        return xB.allFinite() && (xB.array() >= -tol).all();
    }

    static std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd,
                      std::vector<int>, std::size_t, int>
    make_phase1_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
        const int m = static_cast<int>(A.rows());
        const int n = static_cast<int>(A.cols());

        Eigen::MatrixXd A1 = A;
        Eigen::VectorXd b1 = b;
        for (int i = 0; i < m; ++i)
            if (b1(i) < 0) {
                A1.row(i) *= -1.0;
                b1(i) *= -1.0;
            }

        Eigen::MatrixXd A_aux(m, n + m);
        A_aux.leftCols(n) = A1;
        A_aux.rightCols(m) = Eigen::MatrixXd::Identity(m, m);

        Eigen::VectorXd c_aux(n + m);
        c_aux.setZero();
        c_aux.tail(m).setOnes();

        std::vector<int> basis(m);
        std::iota(basis.begin(), basis.end(), n);

        return {A_aux, b1, c_aux, basis, static_cast<std::size_t>(n), m};
    }

    // --------------------------- PRIMAL PHASE ---------------------------
    PhaseResult phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       const Eigen::VectorXd& c,
                       std::optional<std::vector<int>> basis_opt,
                       const Eigen::VectorXd& l, const Eigen::VectorXd& u);

    // --------------------------- DUAL PHASE ---------------------------
    PhaseResult dual_phase_(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                            const Eigen::VectorXd& c,
                            std::optional<std::vector<int>> basis_opt,
                            const Eigen::VectorXd& l,
                            const Eigen::VectorXd& u);

    // --------------------------- Utilities ---------------------------
    static LPSolution make_solution_(
        LPSolution::Status st, Eigen::VectorXd x, double obj,
        std::vector<int> basis, int iters,
        std::unordered_map<std::string, std::string> info,
        std::optional<Eigen::VectorXd> farkas_y = std::nullopt,
        std::optional<bool> farkas_has_cert = std::nullopt,
        std::optional<Eigen::VectorXd> primal_ray = std::nullopt,
        std::optional<bool> primal_ray_has_cert = std::nullopt) {
        LPSolution sol;
        sol.status = st;
        sol.x = std::move(x);
        sol.obj = obj;
        sol.basis = std::move(basis);
        sol.iters = iters;
        sol.info = std::move(info);
        sol.farkas_y =
            farkas_y ? std::move(*farkas_y) : Eigen::VectorXd{};
        sol.farkas_has_cert = farkas_has_cert.value_or(false);
        sol.primal_ray =
            primal_ray ? std::move(*primal_ray) : Eigen::VectorXd{};
        sol.primal_ray_has_cert = primal_ray_has_cert.value_or(false);
        return sol;
    }

   private:
    // Options and state
    RevisedSimplexOptions opt_;
    std::mt19937 rng_;

    // Degeneracy + pricing
    DegeneracyManager degen_;
    AdaptivePricer adaptive_pricer_{1};
    std::unique_ptr<PrimalPricingBridge<AdaptivePricer>> bridge_;
    mutable std::vector<std::string> trace_;
    mutable int solve_depth_ = 0;
};

#include "simplex_primal.h"
#include "simplex_dual.h"

inline RevisedSimplex::PhaseResult RevisedSimplex::phase_(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
    const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
    const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
    return RevisedSimplexPrimalEngine::run(*this, A, b, c, std::move(basis_opt),
                                           l, u);
}

inline RevisedSimplex::PhaseResult RevisedSimplex::dual_phase_(
    const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
    const Eigen::VectorXd& c, std::optional<std::vector<int>> basis_opt,
    const Eigen::VectorXd& l, const Eigen::VectorXd& u) {
    return RevisedSimplexDualEngine::run(*this, A, b, c, std::move(basis_opt),
                                         l, u);
}
