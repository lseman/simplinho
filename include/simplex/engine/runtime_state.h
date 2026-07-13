#pragma once

#include <algorithm>
#include <cstdint>

namespace simplex::engine {

// Named recovery causes keep numerical control flow out of ad-hoc booleans.
// This follows the useful architectural idea in HiGHS' HEkk engines while
// remaining deliberately small and specific to Simplinho.
enum class RebuildReason : std::uint8_t {
    None,
    InitialFactorization,
    UpdateLimit,
    PossiblyOptimal,
    SingularBasis,
    PricingFailure,
    WeightDrift,
    NumericalTrouble,
};

struct RuntimeValidity {
    bool has_basis = false;
    bool has_factorization = false;
    bool has_fresh_factorization = false;
    bool has_primal_values = false;
    bool has_dual_values = false;
    bool has_pricing_weights = false;

    void invalidate_after_basis_change() noexcept {
        has_fresh_factorization = false;
        has_primal_values = false;
        has_dual_values = false;
    }

    void invalidate_factorization() noexcept {
        has_factorization = false;
        has_fresh_factorization = false;
        has_primal_values = false;
        has_dual_values = false;
        has_pricing_weights = false;
    }
};

// HiGHS uses smoothed vector densities to choose sparse/dense kernels. Keeping
// the estimator in shared runtime state prevents each engine from inventing a
// subtly different switching rule.
struct DensityHistory {
    double row_ep = 0.0;
    double row_ap = 0.0;
    double col_aq = 0.0;
    double row_dse = 0.0;

    static double update(double previous, double sample, double weight = 0.05) noexcept {
        sample = std::clamp(sample, 0.0, 1.0);
        weight = std::clamp(weight, 0.0, 1.0);
        return previous == 0.0 ? sample : (1.0 - weight) * previous + weight * sample;
    }
};

struct SimplexRuntimeState {
    RuntimeValidity validity;
    DensityHistory density;
    RebuildReason rebuild_reason = RebuildReason::None;
    int iterations_since_rebuild = 0;

    void request_rebuild(RebuildReason reason) noexcept {
        if (rebuild_reason == RebuildReason::None)
            rebuild_reason = reason;
    }

    void mark_rebuilt() noexcept {
        rebuild_reason = RebuildReason::None;
        iterations_since_rebuild = 0;
        validity.has_factorization = true;
        validity.has_fresh_factorization = true;
    }
};

} // namespace simplex::engine
