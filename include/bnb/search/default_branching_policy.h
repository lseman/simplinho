#pragma once

#include "bnb/search/branching.h"
#include "bnb/search/branching_policy.h"

namespace simplex::bnb {

/// Default branching policy that delegates to existing free-function logic.
/// Used when no external policy is configured, or when an external policy
/// returns a fallback/defer decision.
class DefaultBranchingPolicy : public BranchingPolicy {
public:
    DefaultBranchingPolicy() = default;
    ~DefaultBranchingPolicy() override = default;

    [[nodiscard]]
    BranchDecisionPython decide(
        const BranchingObservations& obs,
        const std::vector<detail::FractionalCandidate>& fractional) const override;

    bool is_enabled() const override { return true; }

    [[nodiscard]] const CallbackTelemetry& telemetry() const override { return telemetry_; }
    [[nodiscard]] CallbackTelemetry& telemetry_mutable() override { return telemetry_; }

private:
    CallbackTelemetry telemetry_;
};

} // namespace simplex::bnb
