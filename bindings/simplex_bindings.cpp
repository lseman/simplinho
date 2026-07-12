#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bindings.h"
#include "bindings_helpers.h"
#include "simplex/engine/simplex.h"
#include "simplex/factorization/sparse_lu.h"
#include "solve_stats.h"

namespace py = pybind11;

namespace {

using SolveStats = simplinho::bindings::SolveStats;
using namespace simplinho::bindings;

void bind_solve_stats_type(py::module_& m) {
    py::class_<SolveStats>(m, "SolveStats")
        .def_property_readonly("status", [](const SolveStats& self) { return self.status; })
        .def_property_readonly("iterations", [](const SolveStats& self) { return self.iterations; })
        .def_property_readonly("phase1_iterations",
                               [](const SolveStats& self) { return self.phase1_iterations; })
        .def_property_readonly("phase2_iterations",
                               [](const SolveStats& self) { return self.phase2_iterations; })
        .def_property_readonly("refactorizations",
                               [](const SolveStats& self) { return self.refactorizations; })
        .def_property_readonly("eta_stack_depth_entry",
                               [](const SolveStats& self) { return self.eta_stack_depth_entry; })
        .def_property_readonly("ft_updates", [](const SolveStats& self) { return self.ft_updates; })
        .def_property_readonly("dual_pool_builds",
                               [](const SolveStats& self) { return self.dual_pool_builds; })
        .def_property_readonly("primal_pool_builds",
                               [](const SolveStats& self) { return self.primal_pool_builds; })
        .def_property_readonly("warm_start_attempted",
                               [](const SolveStats& self) { return self.warm_start_attempted; })
        .def_property_readonly("warm_start_accepted",
                               [](const SolveStats& self) { return self.warm_start_accepted; })
        .def_property_readonly("warm_start_cold_retry",
                               [](const SolveStats& self) { return self.warm_start_cold_retry; })
        .def_property_readonly(
            "warm_factorization_reused",
            [](const SolveStats& self) { return self.warm_factorization_reused; })
        .def_property_readonly("warm_dual_weights_reused",
                               [](const SolveStats& self) { return self.warm_dual_weights_reused; })
        .def_property_readonly("dual_row_price_calls",
                               [](const SolveStats& self) { return self.dual_row_price_calls; })
        .def_property_readonly("dual_col_price_calls",
                               [](const SolveStats& self) { return self.dual_col_price_calls; })
        .def_property_readonly("dual_price_switches",
                               [](const SolveStats& self) { return self.dual_price_switches; })
        .def_property_readonly("dual_row_ep_density",
                               [](const SolveStats& self) { return self.dual_row_ep_density; })
        .def_property_readonly("dual_row_ap_density",
                               [](const SolveStats& self) { return self.dual_row_ap_density; })
        .def_property_readonly("dual_col_aq_density",
                               [](const SolveStats& self) { return self.dual_col_aq_density; })
        .def_property_readonly("lu_build_ns",
                               [](const SolveStats& self) { return self.lu_build_ns; })
        .def_property_readonly("pricing_build_ns",
                               [](const SolveStats& self) { return self.pricing_build_ns; })
        .def_property_readonly("pivot_ns", [](const SolveStats& self) { return self.pivot_ns; })
        .def_property_readonly("presolve_actions",
                               [](const SolveStats& self) { return self.presolve_actions; })
        .def_property_readonly(
            "presolve_implied_bound_updates",
            [](const SolveStats& self) { return self.presolve_implied_bound_updates; })
        .def_property_readonly("reduced_rows",
                               [](const SolveStats& self) { return self.reduced_rows; })
        .def_property_readonly("reduced_cols",
                               [](const SolveStats& self) { return self.reduced_cols; })
        .def_property_readonly("objective_shift",
                               [](const SolveStats& self) { return self.objective_shift; })
        .def_property_readonly(
            "input_upper_bounds_relaxed",
            [](const SolveStats& self) { return self.input_upper_bounds_relaxed; })
        .def_property_readonly(
            "input_lower_bounds_relaxed",
            [](const SolveStats& self) { return self.input_lower_bounds_relaxed; })
        .def_property_readonly("basis_start",
                               [](const SolveStats& self) { return self.basis_start; })
        .def_property_readonly("basis_start_style",
                               [](const SolveStats& self) { return self.basis_start_style; })
        .def_property_readonly("basis_start_attempt",
                               [](const SolveStats& self) { return self.basis_start_attempt; })
        .def_property_readonly(
            "basis_start_primal_feasible",
            [](const SolveStats& self) { return self.basis_start_primal_feasible; })
        .def_property_readonly(
            "basis_start_dual_feasible",
            [](const SolveStats& self) { return self.basis_start_dual_feasible; })
        .def_property_readonly(
            "basis_start_primal_violation",
            [](const SolveStats& self) { return self.basis_start_primal_violation; })
        .def_property_readonly(
            "basis_start_dual_violation",
            [](const SolveStats& self) { return self.basis_start_dual_violation; })
        .def_property_readonly("phase1_status",
                               [](const SolveStats& self) { return self.phase1_status; })
        .def_property_readonly("reason", [](const SolveStats& self) { return self.reason; })
        .def_property_readonly("note", [](const SolveStats& self) { return self.note; })
        .def_property_readonly("certificate",
                               [](const SolveStats& self) { return self.certificate; })
        .def_property_readonly("dual_pricing",
                               [](const SolveStats& self) { return self.dual_pricing; })
        .def_property_readonly("dual_bfrt_flips",
                               [](const SolveStats& self) { return self.dual_bfrt_flips; })
        .def_property_readonly("degeneracy_streak",
                               [](const SolveStats& self) { return self.degeneracy_streak; })
        .def_property_readonly("degeneracy_total",
                               [](const SolveStats& self) { return self.degeneracy_total; })
        .def_property_readonly("suspected_cycle_length",
                               [](const SolveStats& self) { return self.suspected_cycle_length; })
        .def_property_readonly("condition_estimate",
                               [](const SolveStats& self) { return self.condition_estimate; })
        .def_property_readonly("degeneracy_threshold",
                               [](const SolveStats& self) { return self.degeneracy_threshold; })
        .def_property_readonly("degeneracy_epoch",
                               [](const SolveStats& self) { return self.degeneracy_epoch; })
        .def_property_readonly("farkas_has_cert",
                               [](const SolveStats& self) { return self.farkas_has_cert; })
        .def_property_readonly("primal_ray_has_cert",
                               [](const SolveStats& self) { return self.primal_ray_has_cert; })
        .def_property_readonly("trace_lines",
                               [](const SolveStats& self) { return self.trace_lines; })
        .def_property_readonly("raw_info", [](const SolveStats& self) { return self.raw_info; })
        .def("as_dict", &SolveStats::as_dict)
        .def("__repr__", [](const SolveStats& self) {
            std::ostringstream oss;
            oss << "SolveStats(status='" << self.status << "', iterations=" << self.iterations
                << ", trace_lines=" << self.trace_lines << ")";
            return oss.str();
        });
}

} // namespace

void bind_simplex_bindings(py::module_& m) {
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
        .def_property_readonly(
            "num_columns",
            [](const LPBasis& self) { return static_cast<int>(self.column_status.size()); })
        .def_property_readonly("basic_columns",
                               [](const LPBasis& self) {
                                   std::vector<int> out;
                                   for (int j = 0; j < static_cast<int>(self.column_status.size());
                                        ++j) {
                                       if (self.column_status[j] == LPBasisStatus::Basic)
                                           out.push_back(j);
                                   }
                                   return out;
                               })
        .def("__repr__", [](const LPBasis& self) {
            int basics = 0;
            for (const auto status : self.column_status) {
                if (status == LPBasisStatus::Basic)
                    ++basics;
            }
            std::ostringstream oss;
            oss << "LPBasis(num_columns=" << self.column_status.size() << ", basics=" << basics
                << ")";
            return oss.str();
        });

    bind_solve_stats_type(m);

    py::class_<LPSolution>(m, "LPSolution")
        .def_readonly("status", &LPSolution::status, "Solve status (LPSolution.Status enum).")
        .def_readonly("obj", &LPSolution::obj,
                      "Optimal objective value; NaN when infeasible or unbounded.")
        .def_readonly("iters", &LPSolution::iters, "Total simplex iterations (Phase I + Phase II).")
        .def_readonly("x", &LPSolution::x,
                      "Primal solution vector x, length n (original columns).\n"
                      "NaN entries indicate infeasible or unbounded.")
        .def_readonly("basis", &LPSolution::basis,
                      "List of basic column indices in the original problem.")
        .def_readonly("dual_values", &LPSolution::dual_values,
                      "Dual variables y = B^{-T} c_B in the original row space.")
        .def_readonly("farkas_y", &LPSolution::farkas_y,
                      "Farkas infeasibility certificate in the original row space.")
        .def_readonly("farkas_has_cert", &LPSolution::farkas_has_cert,
                      "True when a Farkas certificate of infeasibility is available.")
        .def_readonly("primal_ray", &LPSolution::primal_ray,
                      "Primal unbounded ray in the original column space.")
        .def_readonly("primal_ray_has_cert", &LPSolution::primal_ray_has_cert,
                      "True when a primal unbounded ray certificate is available.")
        .def_property_readonly(
            "basis_state", [](const LPSolution& self) { return rebuild_basis_from_solution(self); },
            "LPBasis for warm-starting a subsequent solve on the same problem structure.")
        .def_readonly("basis_internal", &LPSolution::basis_internal,
                      "Basic column indices in the internal presolve-reduced problem.")
        .def_readonly("nonbasis_internal", &LPSolution::nonbasis_internal,
                      "Nonbasic column indices in the internal problem.")
        .def_readonly("internal_column_labels", &LPSolution::internal_column_labels,
                      "Map from internal columns back to original columns.")
        .def_readonly("internal_row_labels", &LPSolution::internal_row_labels,
                      "Map from internal rows back to original rows.")
        .def_readonly("tableau_internal", &LPSolution::tableau, "B^{-1}A in the internal problem.")
        .def_readonly("tableau_rhs_internal", &LPSolution::tableau_rhs,
                      "B^{-1}b in the internal problem.")
        .def_readonly("reduced_costs_internal", &LPSolution::reduced_costs_internal,
                      "Reduced costs in the internal problem.")
        .def_readonly("dual_values_internal", &LPSolution::dual_values_internal,
                      "Dual variables in the internal problem.")
        .def_property_readonly(
            "has_tableau", [](const LPSolution& self) { return self.has_internal_tableau; },
            "True when tableau_internal and tableau_rhs_internal are populated.")
        .def_readonly("farkas_y_internal", &LPSolution::farkas_y_internal,
                      "Farkas certificate in the internal row space.")
        .def_readonly("primal_ray_internal", &LPSolution::primal_ray_internal,
                      "Primal ray in the internal column space.")
        .def_readonly("info", &LPSolution::info, "Raw key-value telemetry dict.")
        .def_property_readonly(
            "stats", [](const LPSolution& self) { return build_solve_stats(self); },
            "SolveStats object with typed telemetry.")
        .def_property_readonly(
            "log_lines", [](const LPSolution& self) { return self.trace; },
            "Verbose trace lines emitted during the solve.")
        .def_property_readonly(
            "log", [](const LPSolution& self) { return join_trace_lines(self.trace); },
            "Verbose trace joined into a single newline-delimited string.")
        .def("__repr__", [](const LPSolution& self) {
            std::ostringstream oss;
            oss << "LPSolution(status=" << to_string(self.status) << ", obj=";
            if (std::isfinite(self.obj))
                oss << self.obj;
            else
                oss << (std::isnan(self.obj) ? "nan" : (self.obj > 0 ? "inf" : "-inf"));
            oss << ", iters=" << self.iters << ", basis_size=" << self.basis.size() << ")";
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
        .def_readwrite("max_growth_tol", &RevisedSimplexOptions::max_growth_tol)
        .def_readwrite("min_dynamic_growth_tol", &RevisedSimplexOptions::min_dynamic_growth_tol)
        .def_readwrite("max_condition_estimate", &RevisedSimplexOptions::max_condition_estimate)
        .def_readwrite("basis_refinement_steps", &RevisedSimplexOptions::basis_refinement_steps)
        .def_readwrite("basis_residual_refactor_tol",
                       &RevisedSimplexOptions::basis_residual_refactor_tol)
        .def_readwrite("basis_refinement_stall_progress_ratio",
                       &RevisedSimplexOptions::basis_refinement_stall_progress_ratio)
        .def_readwrite("basis_refinement_stall_limit",
                       &RevisedSimplexOptions::basis_refinement_stall_limit)
        .def_readwrite("basis_max_eta_count", &RevisedSimplexOptions::basis_max_eta_count)
        .def_readwrite("basis_column_residual_tol",
                       &RevisedSimplexOptions::basis_column_residual_tol)
        .def_readwrite("basis_aggressive_residual_rebuild",
                       &RevisedSimplexOptions::basis_aggressive_residual_rebuild)
        .def_readwrite("basis_sparse_backend", &RevisedSimplexOptions::basis_sparse_backend)
        .def_readwrite("basis_sparse_equilibration",
                       &RevisedSimplexOptions::basis_sparse_equilibration)
        .def_readwrite("basis_sparse_rhs_density_threshold",
                       &RevisedSimplexOptions::basis_sparse_rhs_density_threshold)
        .def_readwrite("devex_reset", &RevisedSimplexOptions::devex_reset)
        .def_readwrite("pricing_rule", &RevisedSimplexOptions::pricing_rule)
        .def_readwrite("adaptive_reset_freq", &RevisedSimplexOptions::adaptive_reset_freq)
        .def_readwrite("partial_pricing", &RevisedSimplexOptions::partial_pricing)
        .def_readwrite("dual_pricing", &RevisedSimplexOptions::dual_pricing)
        .def_readwrite("row_pricing_threshold", &RevisedSimplexOptions::row_pricing_threshold)
        .def_readwrite("parallel_pricing_workers", &RevisedSimplexOptions::parallel_pricing_workers)
        .def_readwrite("parallel_pricing_min_cols",
                       &RevisedSimplexOptions::parallel_pricing_min_cols)
        .def_readwrite("dualization", &RevisedSimplexOptions::dualization)
        .def_readwrite("dualization_min_row_col_ratio",
                       &RevisedSimplexOptions::dualization_min_row_col_ratio)
        .def_readwrite("dualization_max_recovery_cols",
                       &RevisedSimplexOptions::dualization_max_recovery_cols)
        .def_readwrite("primal_edge_weight_strategy",
                       &RevisedSimplexOptions::primal_edge_weight_strategy)
        .def_readwrite("dual_edge_weight_strategy",
                       &RevisedSimplexOptions::dual_edge_weight_strategy)
        .def_readwrite("primal_steepest_edge_weight_log_error_threshold",
                       &RevisedSimplexOptions::primal_steepest_edge_weight_log_error_threshold)
        .def_readwrite("dual_steepest_edge_weight_log_error_threshold",
                       &RevisedSimplexOptions::dual_steepest_edge_weight_log_error_threshold)
        .def_readwrite("primal_simplex_cost_perturbation_multiplier",
                       &RevisedSimplexOptions::primal_simplex_cost_perturbation_multiplier)
        .def_readwrite("dual_simplex_cost_perturbation_multiplier",
                       &RevisedSimplexOptions::dual_simplex_cost_perturbation_multiplier)
        .def_readwrite("dual_warm_start_near_optimal",
                       &RevisedSimplexOptions::dual_warm_start_near_optimal)
        .def_readwrite("max_basis_rebuilds", &RevisedSimplexOptions::max_basis_rebuilds)
        .def_readwrite("crash_attempts", &RevisedSimplexOptions::crash_attempts)
        .def_readwrite("crash_markowitz_tol", &RevisedSimplexOptions::crash_markowitz_tol)
        .def_readwrite("crash_strategy", &RevisedSimplexOptions::crash_strategy)
        .def_readwrite("repair_mapped_basis", &RevisedSimplexOptions::repair_mapped_basis)
        .def_readwrite("use_quadratic_warm_start_repair",
                       &RevisedSimplexOptions::use_quadratic_warm_start_repair)
        .def_readwrite("dual_allow_bound_flip", &RevisedSimplexOptions::dual_allow_bound_flip)
        .def_readwrite("dual_flip_pivot_tol", &RevisedSimplexOptions::dual_flip_pivot_tol)
        .def_readwrite("dual_flip_rc_tol", &RevisedSimplexOptions::dual_flip_rc_tol)
        .def_readwrite("dual_flip_max_per_iter", &RevisedSimplexOptions::dual_flip_max_per_iter)
        .def_readwrite("verbose", &RevisedSimplexOptions::verbose)
        .def_readwrite("verbose_every", &RevisedSimplexOptions::verbose_every)
        .def_readwrite("verbose_include_basis", &RevisedSimplexOptions::verbose_include_basis)
        .def_readwrite("verbose_include_presolve", &RevisedSimplexOptions::verbose_include_presolve)
        .def_readwrite("disable_presolve", &RevisedSimplexOptions::disable_presolve)
        .def_readwrite("compute_tableau", &RevisedSimplexOptions::compute_tableau)
        .def_readwrite("compute_reduced_costs", &RevisedSimplexOptions::compute_reduced_costs)
        .def_readwrite("mode", &RevisedSimplexOptions::mode);

    py::enum_<SimplexMode>(m, "SimplexMode")
        .value("Auto", SimplexMode::Auto)
        .value("Primal", SimplexMode::Primal)
        .value("Dual", SimplexMode::Dual);

    py::class_<RevisedSimplex>(m, "RevisedSimplex")
        .def(py::init<const RevisedSimplexOptions&>(), py::arg("options") = RevisedSimplexOptions())
        .def("clear_basis_cache", &RevisedSimplex::clear_basis_cache)
        .def("clearBasisCache", &RevisedSimplex::clear_basis_cache)
        .def(
            "solve",
            [](RevisedSimplex& self, const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
               py::object basis) {
                if (basis.is_none()) {
                    return self.solve(A, b, c, l, u);
                }
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(A, b, c, l, u, basis.cast<LPBasis>());
                }
                return self.solve(A, b, c, l, u, basis.cast<std::vector<int>>());
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"), py::arg("u"),
            py::arg("basis") = py::none(), "Solve LP: min c^T x s.t. Ax=b, l<=x<=u")
        .def(
            "solve",
            [](RevisedSimplex& self, const RevisedSimplex::SparseMatrix& A,
               const Eigen::VectorXd& b, const Eigen::VectorXd& c, const Eigen::VectorXd& l,
               const Eigen::VectorXd& u, py::object basis) {
                if (basis.is_none()) {
                    return self.solve(A, b, c, l, u);
                }
                if (py::isinstance<LPBasis>(basis)) {
                    return self.solve(A, b, c, l, u, basis.cast<LPBasis>());
                }
                return self.solve(A, b, c, l, u, basis.cast<std::vector<int>>());
            },
            py::arg("A"), py::arg("b"), py::arg("c"), py::arg("l"), py::arg("u"),
            py::arg("basis") = py::none(), "Solve LP: min c^T x s.t. Ax=b, l<=x<=u");

    m.def("status_to_string",
          [](LPSolution::Status status) { return std::string(to_string(status)); });

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

namespace {

void bind_sparse_forrest_tomlin_lu_bindings(py::module_& m) {
    using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using UpdateStats = SparseForrestTomlinLU::UpdateStats;
    using UpdateFailureReason = SparseForrestTomlinLU::UpdateFailureReason;

    py::enum_<UpdateFailureReason>(m, "SparseLUUpdateFailureReason")
        .value("None", UpdateFailureReason::None)
        .value("BadDimensions", UpdateFailureReason::BadDimensions)
        .value("AlphaTooSmall", UpdateFailureReason::AlphaTooSmall)
        .value("NonFiniteInput", UpdateFailureReason::NonFiniteInput);

    py::enum_<SparseForrestTomlinLU::UpdateMethod>(m, "SparseLUUpdateMethod")
        .value("FT", SparseForrestTomlinLU::UpdateMethod::FT)
        .value("PF", SparseForrestTomlinLU::UpdateMethod::PF)
        .value("MPF", SparseForrestTomlinLU::UpdateMethod::MPF)
        .value("APF", SparseForrestTomlinLU::UpdateMethod::APF);

    py::class_<UpdateStats>(m, "SparseLUUpdateStats")
        .def_readonly("count", &UpdateStats::count)
        .def_readonly("max_z_inf", &UpdateStats::max_z_inf)
        .def_readonly("max_w_inf", &UpdateStats::max_w_inf)
        .def_readonly("avg_z_density", &UpdateStats::avg_z_density)
        .def_readonly("cumulative_z_inf", &UpdateStats::cumulative_z_inf)
        .def_readonly("norm_growth_estimate", &UpdateStats::norm_growth_estimate);

    py::class_<SparseForrestTomlinLU::Config>(m, "SparseLUConfig")
        .def(py::init<>())
        .def_readwrite("use_amd_ordering", &SparseForrestTomlinLU::Config::use_amd_ordering)
        .def_readwrite("fallback_to_legacy_symbolic",
                       &SparseForrestTomlinLU::Config::fallback_to_legacy_symbolic)
        .def_readwrite("diagonal_equilibration",
                       &SparseForrestTomlinLU::Config::diagonal_equilibration)
        .def_readwrite("equilibration_passes", &SparseForrestTomlinLU::Config::equilibration_passes)
        .def_readwrite("equilibration_floor", &SparseForrestTomlinLU::Config::equilibration_floor)
        .def_readwrite("iterative_refinement", &SparseForrestTomlinLU::Config::iterative_refinement)
        .def_readwrite("iterative_refinement_steps",
                       &SparseForrestTomlinLU::Config::iterative_refinement_steps)
        .def_readwrite("iterative_refinement_tol",
                       &SparseForrestTomlinLU::Config::iterative_refinement_tol)
        .def_readwrite("max_norm_growth_before_refactor",
                       &SparseForrestTomlinLU::Config::max_norm_growth_before_refactor)
        .def_readwrite("max_parallel_update_size",
                       &SparseForrestTomlinLU::Config::max_parallel_update_size)
        .def_readwrite("enable_hyper_sparse_rhs",
                       &SparseForrestTomlinLU::Config::enable_hyper_sparse_rhs)
        .def_readwrite("use_product_form_updates",
                       &SparseForrestTomlinLU::Config::use_product_form_updates)
        .def_readwrite("update_method", &SparseForrestTomlinLU::Config::update_method)
        .def_readwrite("force_eigen_sparse_lu",
                       &SparseForrestTomlinLU::Config::force_eigen_sparse_lu)
        .def_readwrite("enable_solve_oracle", &SparseForrestTomlinLU::Config::enable_solve_oracle)
        .def_readwrite("validate_solves", &SparseForrestTomlinLU::Config::validate_solves);

    py::class_<SparseForrestTomlinLU>(m, "SparseForrestTomlinLU")
        .def(py::init<>())
        .def(
            "factor",
            [](SparseForrestTomlinLU& self, py::object A_obj, double pivot_rel, double abs_floor,
               int refactor_rook_iters, py::object initial_row_perm_obj,
               py::object initial_col_perm_obj) {
                SparseMat A;
                // Check if it's a scipy sparse matrix by looking for 'data' attribute
                if (py::hasattr(A_obj, "data")) {
                    // Convert from scipy csc/csr
                    py::object csc =
                        py::hasattr(A_obj, "tocsr") ? A_obj.attr("tocsr")() : A_obj.attr("tocsc")();
                    py::object data = csc.attr("data");
                    py::object indices = csc.attr("indices");
                    py::object indptr = csc.attr("indptr");
                    py::tuple shape = csc.attr("shape").cast<py::tuple>();
                    int n_rows = (int)shape[0].cast<int>();
                    int n_cols = (int)shape[1].cast<int>();
                    A.resize(n_rows, n_cols);
                    std::vector<Eigen::Triplet<double>> triplets;
                    auto data_arr = data.cast<py::array_t<double>>();
                    auto indices_arr = indices.cast<py::array_t<int>>();
                    auto indptr_arr = indptr.cast<py::array_t<int>>();
                    auto data_ptr = data_arr.template unchecked<1>();
                    auto indices_ptr = indices_arr.template unchecked<1>();
                    auto indptr_ptr = indptr_arr.template unchecked<1>();
                    for (int col = 0; col < n_cols; ++col) {
                        int start = indptr_ptr(col);
                        int end = indptr_ptr(col + 1);
                        for (int k = start; k < end; ++k) {
                            int row = indices_ptr(k);
                            double val = data_ptr(k);
                            if (std::abs(val) > 1e-16)
                                triplets.emplace_back(row, col, val);
                        }
                    }
                    if (!triplets.empty())
                        A.setFromTriplets(triplets.begin(), triplets.end());
                    A.makeCompressed();
                } else {
                    A = py::cast<SparseMat>(A_obj);
                }
                std::vector<int> row_perm_vec;
                std::vector<int> col_perm_vec;
                std::vector<int>* row_perm = nullptr;
                std::vector<int>* col_perm = nullptr;
                if (!initial_row_perm_obj.is_none()) {
                    row_perm_vec = initial_row_perm_obj.cast<std::vector<int>>();
                    row_perm = &row_perm_vec;
                }
                if (!initial_col_perm_obj.is_none()) {
                    col_perm_vec = initial_col_perm_obj.cast<std::vector<int>>();
                    col_perm = &col_perm_vec;
                }
                self.factor(A, pivot_rel, abs_floor, refactor_rook_iters, 0, row_perm, col_perm);
            },
            py::arg("A"), py::arg("pivot_rel") = 1e-12, py::arg("abs_floor") = 1e-16,
            py::arg("refactor_rook_iters") = 2, py::arg("initial_row_perm") = py::none(),
            py::arg("initial_col_perm") = py::none(),
            "Factorize square sparse matrix A into LU decomposition")
        .def(
            "factor_with_config",
            [](SparseForrestTomlinLU& self, py::object A_obj, double pivot_rel, double abs_floor,
               int refactor_rook_iters, py::object initial_row_perm_obj,
               py::object initial_col_perm_obj, const SparseForrestTomlinLU::Config& config) {
                SparseMat A;
                // Check if it's a scipy sparse matrix by looking for 'data' attribute
                if (py::hasattr(A_obj, "data")) {
                    // Convert from scipy csc/csr
                    py::object csc =
                        py::hasattr(A_obj, "tocsr") ? A_obj.attr("tocsr")() : A_obj.attr("tocsc")();
                    py::object data = csc.attr("data");
                    py::object indices = csc.attr("indices");
                    py::object indptr = csc.attr("indptr");
                    py::tuple shape = csc.attr("shape").cast<py::tuple>();
                    int n_rows = (int)shape[0].cast<int>();
                    int n_cols = (int)shape[1].cast<int>();
                    A.resize(n_rows, n_cols);
                    std::vector<Eigen::Triplet<double>> triplets;
                    auto data_arr = data.cast<py::array_t<double>>();
                    auto indices_arr = indices.cast<py::array_t<int>>();
                    auto indptr_arr = indptr.cast<py::array_t<int>>();
                    auto data_ptr = data_arr.template unchecked<1>();
                    auto indices_ptr = indices_arr.template unchecked<1>();
                    auto indptr_ptr = indptr_arr.template unchecked<1>();
                    for (int col = 0; col < n_cols; ++col) {
                        int start = indptr_ptr(col);
                        int end = indptr_ptr(col + 1);
                        for (int k = start; k < end; ++k) {
                            int row = indices_ptr(k);
                            double val = data_ptr(k);
                            if (std::abs(val) > 1e-16)
                                triplets.emplace_back(row, col, val);
                        }
                    }
                    if (!triplets.empty())
                        A.setFromTriplets(triplets.begin(), triplets.end());
                    A.makeCompressed();
                } else {
                    A = py::cast<SparseMat>(A_obj);
                }
                std::vector<int> row_perm_vec;
                std::vector<int> col_perm_vec;
                std::vector<int>* row_perm = nullptr;
                std::vector<int>* col_perm = nullptr;
                if (!initial_row_perm_obj.is_none()) {
                    row_perm_vec = initial_row_perm_obj.cast<std::vector<int>>();
                    row_perm = &row_perm_vec;
                }
                if (!initial_col_perm_obj.is_none()) {
                    col_perm_vec = initial_col_perm_obj.cast<std::vector<int>>();
                    col_perm = &col_perm_vec;
                }
                self.factor(A, pivot_rel, abs_floor, refactor_rook_iters, 0, row_perm, col_perm,
                            config);
            },
            py::arg("A"), py::arg("pivot_rel") = 1e-12, py::arg("abs_floor") = 1e-16,
            py::arg("refactor_rook_iters") = 2, py::arg("initial_row_perm") = py::none(),
            py::arg("initial_col_perm") = py::none(), py::arg("config"), "Factorize with config")
        .def("supports_inplace_updates", &SparseForrestTomlinLU::supports_inplace_updates)
        .def("has_updates", &SparseForrestTomlinLU::has_updates)
        .def("synthetic_clock_says_refactor", &SparseForrestTomlinLU::synthetic_clock_says_refactor)
        .def("is_rank_deficient", &SparseForrestTomlinLU::is_rank_deficient)
        .def("rank_deficiency", &SparseForrestTomlinLU::rank_deficiency)
        .def("needs_refactor", &SparseForrestTomlinLU::needs_refactor)
        .def("clear_updates", &SparseForrestTomlinLU::clear_updates)
        .def(
            "append_forrest_tomlin_update",
            [](SparseForrestTomlinLU& self, int j, const Eigen::VectorXd& u,
               const Eigen::VectorXd& z, const Eigen::VectorXd& w, double alpha,
               double eps = 1e-14) {
                return self.append_forrest_tomlin_update(j, u, z, w, alpha, eps);
            },
            py::arg("j"), py::arg("u"), py::arg("z"), py::arg("w"), py::arg("alpha"),
            py::arg("eps") = 1e-14, "Append Forrest-Tomlin update")
        .def("update_stats", &SparseForrestTomlinLU::update_stats, "Get update statistics")
        .def("last_update_failure_reason", &SparseForrestTomlinLU::last_update_failure_reason)
        .def("last_update_failure_reason_message",
             &SparseForrestTomlinLU::last_update_failure_reason_message)
        .def(
            "solve",
            [](const SparseForrestTomlinLU& self, const Eigen::VectorXd& b,
               double expected_density) { return self.solve(b, expected_density); },
            py::arg("b"), py::arg("expected_density") = 1.0, "Solve B*x = b")
        .def(
            "solveT",
            [](const SparseForrestTomlinLU& self, const Eigen::VectorXd& c,
               double expected_density) { return self.solveT(c, expected_density); },
            py::arg("c"), py::arg("expected_density") = 1.0, "Solve B^T*y = c")
        .def(
            "solve_sparse",
            [](const SparseForrestTomlinLU& self, const std::vector<int>& seed_idx,
               const std::vector<double>& seed_val, double expected_density) {
                return self.solve_sparse(seed_idx, seed_val, expected_density);
            },
            py::arg("seed_idx"), py::arg("seed_val"), py::arg("expected_density") = 0.0,
            "Solve with sparse RHS")
        .def(
            "solveT_sparse",
            [](const SparseForrestTomlinLU& self, const std::vector<int>& seed_idx,
               const std::vector<double>& seed_val, double expected_density) {
                return self.solveT_sparse(seed_idx, seed_val, expected_density);
            },
            py::arg("seed_idx"), py::arg("seed_val"), py::arg("expected_density") = 0.0,
            "SolveT with sparse RHS")
        .def("last_solve_reach_original", &SparseForrestTomlinLU::last_solve_reach_original)
        .def("last_solve_pattern_valid", &SparseForrestTomlinLU::last_solve_pattern_valid);
}

} // namespace

void bind_sparse_lu_bindings(py::module_& m) { bind_sparse_forrest_tomlin_lu_bindings(m); }
