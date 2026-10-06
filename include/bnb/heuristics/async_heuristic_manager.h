#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace simplex::bnb {
class Solver;

class AsyncHeuristicManager {
  public:
    static bool enabled(const Solver& solver) noexcept;
    static int worker_count(const Solver& solver) noexcept;
    static int max_tasks(const Solver& solver) noexcept;
    static std::uint64_t staleness_window() noexcept;

    static void reap(Solver& solver, bool wait_all = false);
    static void start_workers(Solver& solver);
    static void stop_workers(Solver& solver);

    template <typename Task>
    static void dispatch(Solver& solver, std::uint64_t launch_order, Task&& task);
};

} // namespace simplex::bnb
