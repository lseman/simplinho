#include <pybind11/pybind11.h>

#include <sstream>

#include "bindings.h"
#include "simplex/bnb.h"

namespace simplex_bnb = simplex::bnb;

namespace {

using VarType = simplex_bnb::VariableType;
using MIPStatus = simplex_bnb::Status;
using NodeSelectionStrategy = simplex_bnb::NodeSelectionStrategy;
using BranchingStrategy = simplex_bnb::BranchingStrategy;
using DivingStrategy = simplex_bnb::DivingStrategy;
using BranchAndBoundOptions = simplex_bnb::Options;
using MIPTreeNode = simplex_bnb::TreeNode;
using MIPTreeNodeStatus = simplex_bnb::TreeNodeStatus;

} // namespace

void bind_bnb_bindings(py::module_& m) {
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
        .value("Hybrid", NodeSelectionStrategy::Hybrid)
        .value("BestFirstPlunging", NodeSelectionStrategy::BestFirstPlunging)
        .value("BestFirst_Plunging", NodeSelectionStrategy::BestFirstPlunging)
        .value("BestEstimatePlunging", NodeSelectionStrategy::BestEstimatePlunging)
        .value("BestEstimate_Plunging", NodeSelectionStrategy::BestEstimatePlunging)
        .value("InterleavedBestFirstBestEstimatePlunging",
               NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging)
        .value("Interleaved_BestFirst_BestEstimate_Plunging",
               NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging);

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
        .def_property_readonly("parent_id", [](const MIPTreeNode& self) { return self.parent_id; })
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
                << ", depth=" << self.depth << ", estimate=" << self.estimate << ", status='"
                << simplex_bnb::to_string(self.status) << "')";
            return oss.str();
        });

    py::class_<BranchAndBoundOptions>(m, "BranchAndBoundOptions")
        .def(py::init<>())
        .def_readwrite("max_nodes", &BranchAndBoundOptions::max_nodes)
        .def_property(
            "node_limit", [](const BranchAndBoundOptions& self) { return self.max_nodes; },
            [](BranchAndBoundOptions& self, int value) { self.max_nodes = value; })
        .def_readwrite("parallel_workers", &BranchAndBoundOptions::parallel_workers)
        .def_readwrite("integrality_tol", &BranchAndBoundOptions::integrality_tol)
        .def_readwrite("verbose", &BranchAndBoundOptions::verbose)
        .def_readwrite("log_frequency", &BranchAndBoundOptions::log_frequency)
        .def_readwrite("node_timing_log_path", &BranchAndBoundOptions::node_timing_log_path)
        .def_readwrite("node_selection", &BranchAndBoundOptions::node_selection)
        .def_readwrite("hybrid_depth_bias", &BranchAndBoundOptions::hybrid_depth_bias)
        .def_readwrite("plunging_bestfreq", &BranchAndBoundOptions::plunging_bestfreq)
        .def_readwrite("branching_strategy", &BranchAndBoundOptions::branching_strategy)
        .def_readwrite("diving_strategy", &BranchAndBoundOptions::diving_strategy)
        .def_readwrite("strong_branching_candidates",
                       &BranchAndBoundOptions::strong_branching_candidates)
        .def_readwrite("strong_branching_k", &BranchAndBoundOptions::strong_branching_k)
        .def_readwrite("strong_branching_max_depth",
                       &BranchAndBoundOptions::strong_branching_max_depth)
        .def_readwrite("pseudocost_reliability", &BranchAndBoundOptions::pseudocost_reliability)
        .def_readwrite("max_dive_depth", &BranchAndBoundOptions::max_dive_depth)
        .def_readwrite("max_dive_lp_solves", &BranchAndBoundOptions::max_dive_lp_solves)
        .def_readwrite("heuristic_frequency", &BranchAndBoundOptions::heuristic_frequency)
        .def_readwrite("heuristic_max_depth", &BranchAndBoundOptions::heuristic_max_depth)
        .def_readwrite("use_rounding", &BranchAndBoundOptions::use_rounding)
        .def_readwrite("use_diving", &BranchAndBoundOptions::use_diving)
        .def_readwrite("use_rins", &BranchAndBoundOptions::use_rins)
        .def_readwrite("rins_fix_ratio", &BranchAndBoundOptions::rins_fix_ratio)
        .def_readwrite("rins_tolerance", &BranchAndBoundOptions::rins_tolerance)
        .def_readwrite("use_rens", &BranchAndBoundOptions::use_rens)
        .def_readwrite("rens_fix_ratio", &BranchAndBoundOptions::rens_fix_ratio)
        .def_readwrite("use_local_search", &BranchAndBoundOptions::use_local_search)
        .def_readwrite("local_search_iterations", &BranchAndBoundOptions::local_search_iterations)
        .def_readwrite("local_search_max_free_vars",
                       &BranchAndBoundOptions::local_search_max_free_vars)
        .def_readwrite("use_local_branching", &BranchAndBoundOptions::use_local_branching)
        .def_readwrite("use_async_heuristics", &BranchAndBoundOptions::use_async_heuristics)
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
        .def_readwrite("use_feasibility_pump", &BranchAndBoundOptions::use_feasibility_pump)
        .def_readwrite("feasibility_pump_iterations",
                       &BranchAndBoundOptions::feasibility_pump_iterations)
        .def_readwrite("feasibility_pump_fix_ratio",
                       &BranchAndBoundOptions::feasibility_pump_fix_ratio)
        .def_readwrite("use_feasibility_jump", &BranchAndBoundOptions::use_feasibility_jump)
        .def_readwrite("feasibility_jump_iterations",
                       &BranchAndBoundOptions::feasibility_jump_iterations)
        .def_readwrite("feasibility_jump_max_free_vars",
                       &BranchAndBoundOptions::feasibility_jump_max_free_vars)
        .def_readwrite("feasibility_jump_objective_weight",
                       &BranchAndBoundOptions::feasibility_jump_objective_weight)
        .def_readwrite("heuristic_subproblem_max_nodes",
                       &BranchAndBoundOptions::heuristic_subproblem_max_nodes)
        .def_readwrite("use_cut_pool", &BranchAndBoundOptions::use_cut_pool)
        .def_readwrite("max_cut_rounds_per_node", &BranchAndBoundOptions::max_cut_rounds_per_node)
        .def_readwrite("max_cuts_added_per_round", &BranchAndBoundOptions::max_cuts_added_per_round)
        .def_readwrite("max_cut_pool_size", &BranchAndBoundOptions::max_cut_pool_size)
        .def_readwrite("min_cut_violation", &BranchAndBoundOptions::min_cut_violation)
        .def_readwrite("max_cut_age", &BranchAndBoundOptions::max_cut_age)
        .def_readwrite("use_gomory_cuts", &BranchAndBoundOptions::use_gomory_cuts)
        .def_readwrite("use_mir_cuts", &BranchAndBoundOptions::use_mir_cuts)
        .def_readwrite("use_cover_cuts", &BranchAndBoundOptions::use_cover_cuts)
        .def_readwrite("use_implied_bound_cuts", &BranchAndBoundOptions::use_implied_bound_cuts)
        .def_readwrite("use_clique_cuts", &BranchAndBoundOptions::use_clique_cuts)
        .def_readwrite("use_odd_cycle_cuts", &BranchAndBoundOptions::use_odd_cycle_cuts)
        .def_readwrite("use_probing_implications", &BranchAndBoundOptions::use_probing_implications)
        .def_readwrite("probing_max_candidates", &BranchAndBoundOptions::probing_max_candidates)
        .def_readwrite("use_conflict_cuts", &BranchAndBoundOptions::use_conflict_cuts)
        .def_readwrite("max_conflict_cuts_per_round",
                       &BranchAndBoundOptions::max_conflict_cuts_per_round)
        .def_readwrite("max_cuts_per_type", &BranchAndBoundOptions::max_cuts_per_type)
        .def_readwrite("cut_max_parallelism", &BranchAndBoundOptions::cut_max_parallelism)
        .def_readwrite("use_lp_reoptimization_profile",
                       &BranchAndBoundOptions::use_lp_reoptimization_profile)
        .def_readwrite("use_node_presolve", &BranchAndBoundOptions::use_node_presolve)
        .def("__repr__", [](const BranchAndBoundOptions& self) {
            std::ostringstream oss;
            oss << "BranchAndBoundOptions(max_nodes=" << self.max_nodes
                << ", integrality_tol=" << self.integrality_tol
                << ", verbose=" << (self.verbose ? "True" : "False")
                << ", log_frequency=" << self.log_frequency << ", node_selection='"
                << simplex_bnb::to_string(self.node_selection) << "'"
                << ", branching_strategy='" << simplex_bnb::to_string(self.branching_strategy)
                << "', diving_strategy='" << simplex_bnb::to_string(self.diving_strategy)
                << "', use_lp_reoptimization_profile="
                << (self.use_lp_reoptimization_profile ? "True" : "False") << ")";
            return oss.str();
        });

    m.def("mip_status_to_string",
          [](MIPStatus status) { return std::string(simplex_bnb::to_string(status)); });
}
