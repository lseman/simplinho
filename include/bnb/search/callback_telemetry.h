#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace simplex::bnb {

// Accumulates statistics for external policy callbacks (branching, cut selection, etc.).
// All timing is wall-clock nanoseconds; counts are per-solve.
struct CallbackTelemetry {
    mutable std::uint64_t call_count = 0;
    mutable std::uint64_t success_count = 0;
    mutable std::uint64_t fallback_count = 0;
    mutable std::uint64_t exception_count = 0;
    mutable std::uint64_t timeout_count = 0;
    mutable std::uint64_t total_wall_ns = 0;
    mutable std::uint64_t max_wall_ns = 0;
    mutable std::uint64_t min_wall_ns = std::numeric_limits<std::uint64_t>::max();

    // Per-solve variable selection histogram (key = variable index).
    mutable std::unordered_map<int, int> variable_selections;

    // Configurable timeout in milliseconds; 0 means no timeout.
    int timeout_ms = 0;

    CallbackTelemetry() = default;
    CallbackTelemetry(const CallbackTelemetry&) = default;
    CallbackTelemetry& operator=(const CallbackTelemetry&) = default;

    void record_success(std::uint64_t wall_ns, int variable) const {
        ++call_count;
        ++success_count;
        total_wall_ns += wall_ns;
        max_wall_ns = std::max(max_wall_ns, wall_ns);
        if (wall_ns < min_wall_ns) min_wall_ns = wall_ns;
        variable_selections[variable]++;
    }

    void record_fallback(std::uint64_t wall_ns) const {
        ++call_count;
        ++fallback_count;
        total_wall_ns += wall_ns;
        max_wall_ns = std::max(max_wall_ns, wall_ns);
        if (wall_ns < min_wall_ns) min_wall_ns = wall_ns;
    }

    void record_exception(std::uint64_t wall_ns) const {
        ++call_count;
        ++exception_count;
        total_wall_ns += wall_ns;
        max_wall_ns = std::max(max_wall_ns, wall_ns);
        if (wall_ns < min_wall_ns) min_wall_ns = wall_ns;
    }

    void record_timeout(std::uint64_t wall_ns) const {
        ++call_count;
        ++timeout_count;
        ++fallback_count; // timeouts count as fallbacks too
        total_wall_ns += wall_ns;
        max_wall_ns = std::max(max_wall_ns, wall_ns);
        if (wall_ns < min_wall_ns) min_wall_ns = wall_ns;
    }

    // Build a minimal human-readable string for logging.
    std::string summary() const {
        std::ostringstream oss;
        oss << "callbacks: calls=" << call_count
            << " success=" << success_count
            << " fallback=" << fallback_count
            << " exception=" << exception_count
            << " timeout=" << timeout_count;
        if (call_count > 0) {
            oss << " avg_ms=" << (total_wall_ns / call_count / 1'000'000);
        }
        return oss.str();
    }
};

} // namespace simplex::bnb
