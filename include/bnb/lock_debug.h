#pragma once

#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

namespace simplex::bnb::detail {

#ifdef SIMPLEX_BNB_DEBUG_LOCKS
inline void debug_log(std::string_view message) {
    static std::mutex debug_mutex;
    std::lock_guard<std::mutex> lock(debug_mutex);
    std::cout << message << std::endl;
}

template <typename... Args> inline std::string debug_line(Args&&... args) {
    std::ostringstream oss;
    (oss << ... << std::forward<Args>(args));
    return oss.str();
}

struct LockTrace {
    const char* name;
    std::chrono::steady_clock::time_point request_time;
    std::chrono::steady_clock::time_point acquire_time;
    std::chrono::steady_clock::duration held_duration{};
    bool acquired = false;
    bool owns_lock = false;

    explicit LockTrace(const char* name_) noexcept
        : name(name_), request_time(std::chrono::steady_clock::now()) {
        debug_log(debug_line("[LOCK] request ", std::this_thread::get_id(), " ", name));
    }

    void acquired_lock() noexcept {
        if (!acquired) {
            acquired = true;
            owns_lock = true;
            acquire_time = std::chrono::steady_clock::now();
            const auto wait_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(acquire_time - request_time)
                    .count();
            debug_log(debug_line("[LOCK] acquired ", std::this_thread::get_id(), " ", name,
                                 " wait=", wait_ms, "ms"));
        } else if (!owns_lock) {
            owns_lock = true;
            acquire_time = std::chrono::steady_clock::now();
        }
    }

    void released_lock() noexcept {
        if (acquired && owns_lock) {
            const auto now = std::chrono::steady_clock::now();
            held_duration += now - acquire_time;
            owns_lock = false;
        }
    }

    ~LockTrace() {
        if (acquired) {
            if (owns_lock) {
                released_lock();
            }
            const auto hold_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(held_duration).count();
            debug_log(debug_line("[LOCK] release ", std::this_thread::get_id(), " ", name,
                                 " hold=", hold_ms, "ms"));
        }
    }
};

#    ifdef SIMPLEX_BNB_DEBUG_TIMING
struct TimingTrace {
    const char* name;
    std::chrono::steady_clock::time_point start;

    explicit TimingTrace(const char* name_) noexcept
        : name(name_), start(std::chrono::steady_clock::now()) {
        debug_log(debug_line("[TIME] enter ", std::this_thread::get_id(), " ", name));
    }

    ~TimingTrace() {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        debug_log(debug_line("[TIME] exit  ", std::this_thread::get_id(), " ", name,
                             " elapsed=", elapsed_ms, "ms"));
    }
};
#    else
struct TimingTrace {
    explicit TimingTrace(const char*) noexcept {}
};
#    endif
#else
struct LockTrace {
    explicit LockTrace(const char*) noexcept {}
    void acquired_lock() noexcept {}
    void released_lock() noexcept {}
};

struct TimingTrace {
    explicit TimingTrace(const char*) noexcept {}
};
#endif

} // namespace simplex::bnb::detail
