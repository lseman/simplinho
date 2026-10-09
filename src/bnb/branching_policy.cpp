#include "bnb/search/branching.h"
#include "bnb/search/branching_policy.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace simplex::bnb {

// ── Build BranchingObservations from solver-owned data ──
// The returned struct owns its Eigen vectors (copies) so the caller's
// data can be freed after the call returns.

BranchingObservations build_observations(
    const detail::ActiveNode& node,
    const RelaxationSolution& relaxation,
    const std::vector<detail::FractionalCandidate>& fractional,
    const std::vector<detail::PseudoCost>& pseudocosts,
    bool is_root,
    int n_nodes_explored,
    double incumbent,
    bool maximize) {

    BranchingObservations obs;

    // Fractional candidate data (copies into owned vectors, then views).
    obs.fractional_variables = [&fractional]() {
        std::vector<int> v;
        v.reserve(fractional.size());
        for (const auto& f : fractional) {
            v.push_back(f.variable);
        }
        return v;
    }();

    obs.lp_values = [&fractional]() {
        Eigen::VectorXd v(static_cast<int>(fractional.size()));
        for (int i = 0; i < static_cast<int>(fractional.size()); ++i) {
            v(i) = fractional[i].value;
        }
        return v;
    }();

    obs.fractionality = [&fractional]() {
        Eigen::VectorXd v(static_cast<int>(fractional.size()));
        for (int i = 0; i < static_cast<int>(fractional.size()); ++i) {
            v(i) = fractional[i].fractionality;
        }
        return v;
    }();

    obs.down_distance = [&fractional]() {
        Eigen::VectorXd v(static_cast<int>(fractional.size()));
        for (int i = 0; i < static_cast<int>(fractional.size()); ++i) {
            v(i) = fractional[i].down_distance;
        }
        return v;
    }();

    obs.up_distance = [&fractional]() {
        Eigen::VectorXd v(static_cast<int>(fractional.size()));
        for (int i = 0; i < static_cast<int>(fractional.size()); ++i) {
            v(i) = fractional[i].up_distance;
        }
        return v;
    }();

    // Node context
    obs.node_id = node.id;
    obs.depth = node.depth;
    obs.node_bound = node.bound;
    obs.lp_objective = relaxation.objective;
    obs.lower_bounds = node.lower_bounds;
    obs.upper_bounds = node.upper_bounds;
    obs.is_root = is_root;
    obs.maximize = maximize;

    // LP status
    switch (relaxation.status) {
        case RelaxationStatus::Optimal:
            obs.lp_status = BranchingObservations::LPSolutionStatus::Optimal;
            break;
        case RelaxationStatus::Infeasible:
            obs.lp_status = BranchingObservations::LPSolutionStatus::Infeasible;
            break;
        case RelaxationStatus::Unbounded:
            obs.lp_status = BranchingObservations::LPSolutionStatus::Unbounded;
            break;
    }

    // Search state
    obs.n_nodes_explored = n_nodes_explored;
    obs.incumbent = incumbent;

    // Candidate-aligned pseudocost data (skip at root).
    if (!is_root) {
        obs.pseudocost.reserve(fractional.size());
        for (const auto& candidate : fractional) {
            BranchingObservations::PseudoEntry entry;
            if (candidate.variable >= 0 &&
                candidate.variable < static_cast<int>(pseudocosts.size())) {
                const auto& pc = pseudocosts[candidate.variable];
                entry.up_score = pc.cost.is_reliable(4) ? pc.cost.up_value() : 0.0;
                entry.down_score = pc.cost.is_reliable(4) ? pc.cost.down_value() : 0.0;
                entry.samples = pc.cost.up_count + pc.cost.down_count;
            }
            obs.pseudocost.push_back(entry);
        }
    }

    return obs;
}

detail::BranchDecision convert_policy_variable(
    int variable,
    const std::vector<detail::FractionalCandidate>& fractional,
    const detail::ActiveNode& node) {

    detail::BranchDecision decision;

    // Find the fractional candidate matching the Python-decided variable.
    const detail::FractionalCandidate* candidate = nullptr;
    for (const auto& f : fractional) {
        if (f.variable == variable) {
            candidate = &f;
            break;
        }
    }

    if (candidate == nullptr) {
        // Variable not in fractional list — return empty (signals fallback).
        decision.variable = -1;
        return decision;
    }

    // Build decision using the same logic as build_decision_from_candidate.
    decision.variable = candidate->variable;
    decision.value = candidate->value;
    decision.down_child.state = detail::make_child_state(
        node, candidate->variable, false, candidate->value);
    decision.up_child.state = detail::make_child_state(
        node, candidate->variable, true, candidate->value);

    return decision;
}

} // namespace simplex::bnb
