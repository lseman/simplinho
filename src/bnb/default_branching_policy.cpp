#include "bnb/search/branching.h"
#include "bnb/search/branching_policy.h"
#include "bnb/search/default_branching_policy.h"

#include <chrono>
#include <limits>
#include <vector>

namespace simplex::bnb {

BranchDecisionPython DefaultBranchingPolicy::decide(
    const BranchingObservations& obs,
    const std::vector<detail::FractionalCandidate>& fractional) const {

    auto start = std::chrono::steady_clock::now();

    // Delegate to the most-fractional heuristic (simplest default).
    if (fractional.empty()) {
        BranchDecisionPython dec;
        dec.variable = -1;
        telemetry_.record_fallback(0);
        return dec;
    }

    // Pick the most fractional candidate.
    const detail::FractionalCandidate* best = &fractional.front();
    for (const auto& f : fractional) {
        if (f.fractionality > best->fractionality) {
            best = &f;
        }
    }

    auto end = std::chrono::steady_clock::now();
    uint64_t wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());

    telemetry_.record_success(wall_ns, best->variable);

    BranchDecisionPython dec;
    dec.variable = best->variable;
    dec.down_bound = best->value;    // down child: upper bound = LP value
    dec.up_bound = best->value;      // up child: lower bound = LP value
    return dec;
}

} // namespace simplex::bnb
