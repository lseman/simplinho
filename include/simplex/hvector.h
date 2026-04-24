#pragma once

// HVector: sparse-pattern-aware dense vector carrier.
//
// Inspired by HiGHS's HVector. Keeps a dense Eigen::VectorXd backing store
// (so arithmetic like .dot(), element access, and existing VectorXd APIs keep
// working unchanged) plus an optional list of nonzero row indices in the
// original (unpermuted) index space.
//
// When count < 0, the nonzero pattern is "unknown" and consumers must treat
// the vector as dense. When count >= 0, index[0..count) lists the rows where
// value may be nonzero — consumers can iterate just those entries and skip
// the O(m) scan over mostly-zero storage.
//
// The dense backing store is authoritative for values: consumers reading
// value(i) always get the correct number regardless of count. The index list
// is an *optimization hint* for iteration, not a uniqueness guarantee — it
// may contain duplicates or entries whose value is numerically zero, so
// consumers that do division or ratio tests must still check value(i) against
// their tolerance.

#include <Eigen/Dense>
#include <algorithm>
#include <utility>
#include <vector>

class HVector {
  public:
    Eigen::VectorXd value;
    std::vector<int> index;
    int count{-1}; // -1 means pattern unknown; treat as dense

    HVector() = default;

    // Dense construction: pattern is not computed. Implicit so existing code
    // returning/assigning Eigen::VectorXd keeps compiling.
    HVector(Eigen::VectorXd v) : value(std::move(v)), count(-1) {}

    // Sparse construction: caller provides the pattern.
    HVector(Eigen::VectorXd v, std::vector<int> idx)
        : value(std::move(v)), index(std::move(idx)),
          count(static_cast<int>(index.size())) {}

    // Pattern is known if count >= 0. Consumers that support sparse iteration
    // should test this and iterate index[0..count) instead of [0, m).
    bool has_pattern() const noexcept { return count >= 0; }

    // Implicit conversion to const VectorXd& so existing dense-consuming code
    // (ratio tests, dots, element access) keeps working. We intentionally only
    // provide the const-lvalue-ref conversion to avoid assignment ambiguity
    // with Eigen's const&/rvalue operator= overloads when the HVector is itself
    // an rvalue (e.g. `x = solver.solve_B(b);`).
    operator const Eigen::VectorXd&() const noexcept { return value; }

    // Forwarders for the most common VectorXd operations used across the
    // simplex code — reduces the number of call sites that need explicit
    // .value access.
    Eigen::Index size() const noexcept { return value.size(); }
    double operator()(Eigen::Index i) const noexcept { return value(i); }
    double& operator()(Eigen::Index i) noexcept { return value(i); }

    // Build an HVector from a dense result by scanning for nonzeros above
    // `tol`. O(m), but avoids duplicating the scan inside consumers.
    static HVector from_dense_with_pattern(Eigen::VectorXd v, double tol = 1e-14) {
        HVector out;
        out.value = std::move(v);
        out.index.reserve(static_cast<std::size_t>(out.value.size()));
        for (Eigen::Index i = 0; i < out.value.size(); ++i) {
            if (std::abs(out.value(i)) > tol)
                out.index.push_back(static_cast<int>(i));
        }
        out.count = static_cast<int>(out.index.size());
        return out;
    }

    // Mark the vector dense (pattern unknown); leaves value untouched.
    void drop_pattern() noexcept {
        index.clear();
        count = -1;
    }
};
