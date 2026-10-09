#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "bnb/search/callback_telemetry.h"

namespace simplex::bnb {

// Forward declarations — full types defined in branching.h.
namespace detail {
    struct FractionalCandidate;
    struct ActiveNode;
    struct PseudoCost;
    struct BranchDecision;
}

/// Observation batch passed to an external branching policy.
/// Owns its vectors/matrices so it can be constructed and moved freely.
struct BranchingObservations {
    // ── Fractional candidate data (same length, indexed by position) ──
    std::vector<int>    fractional_variables;   // variable indices
    Eigen::VectorXd     lp_values;              // LP solution values
    Eigen::VectorXd     fractionality;          // min(|x - round(x)|)
    Eigen::VectorXd     down_distance;          // x - floor(x)
    Eigen::VectorXd     up_distance;            // ceil(x) - x

    // ── Node context ──
    int    node_id = 0;
    int    depth = 0;
    double node_bound = 0.0;
    double lp_objective = 0.0;
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    bool   is_root = false;
    bool   maximize = true;

    // ── LP solution status ──
    enum class LPSolutionStatus { Optimal, Infeasible, Unbounded, Numerical };
    LPSolutionStatus lp_status = LPSolutionStatus::Optimal;

    // ── Search state ──
    int    n_nodes_explored = 0;
    double incumbent = std::numeric_limits<double>::infinity();

    // ── Candidate-aligned pseudocost data (empty at root) ──
    struct PseudoEntry {
        double up_score = 0.0;
        double down_score = 0.0;
        int    samples = 0;
    };
    std::vector<PseudoEntry> pseudocost;
};

/// Abstract branching policy interface.
/// Implementations may be default (in-process), Python-backed, or any other strategy.
class BranchingPolicy {
public:
    virtual ~BranchingPolicy() = default;

    /// Decide which variable to branch on given the current observation batch.
    ///
    /// @param obs       Batched, solver-owned observation data.
    /// @param fractional Raw FractionalCandidate list (for converting the decision).
    /// @return The chosen variable index, or std::nullopt to defer.
    [[nodiscard]]
    virtual std::optional<int> choose_variable(
        const BranchingObservations& obs,
        const std::vector<detail::FractionalCandidate>& fractional) const = 0;

    /// Whether this policy is enabled. A disabled policy is treated as absent
    /// by the solver (no virtual dispatch overhead).
    [[nodiscard]] virtual bool is_enabled() const = 0;

    /// Telemetry for this policy instance (const access).
    [[nodiscard]] virtual const CallbackTelemetry& telemetry() const = 0;

    /// Non-const telemetry access (for Python bindings).
    [[nodiscard]] virtual CallbackTelemetry& telemetry_mutable() = 0;
};

/// Build BranchingObservations from solver-owned data structures.
/// The returned struct owns its Eigen vectors (copies) so the caller's
/// data can be freed after the call returns.
BranchingObservations build_observations(
    const detail::ActiveNode& node,
    const RelaxationSolution& relaxation,
    const std::vector<detail::FractionalCandidate>& fractional,
    const std::vector<detail::PseudoCost>& pseudocosts,
    bool is_root,
    int n_nodes_explored,
    double incumbent,
    bool maximize);

/// Convert an external variable choice to the internal branch representation.
/// Returns an empty decision if the variable is not a fractional candidate.
detail::BranchDecision convert_policy_variable(
    int variable,
    const std::vector<detail::FractionalCandidate>& fractional,
    const detail::ActiveNode& node);

} // namespace simplex::bnb
