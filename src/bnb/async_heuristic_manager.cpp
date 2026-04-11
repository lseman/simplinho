#include "bnb/async_heuristic_manager.h"
#include "bnb/core.h"

#include <algorithm>
#include <chrono>

namespace simplex::bnb {

bool AsyncHeuristicManager::enabled(const Solver& solver) noexcept {
    return solver.options_.use_async_heuristics;
}

int AsyncHeuristicManager::worker_count(const Solver& solver) noexcept {
    return std::clamp(solver.options_.parallel_workers + 2, 2, 16);
}

int AsyncHeuristicManager::max_tasks(const Solver& solver) noexcept {
    return enabled(solver) ? std::max(8, worker_count(solver) * 4) : 0;
}

std::uint64_t AsyncHeuristicManager::staleness_window() noexcept { return 64; }

void AsyncHeuristicManager::reap(Solver& solver, bool wait_all) {
    if (solver.async_heuristic_dispatcher_ == nullptr) {
        return;
    }
    if (!wait_all) {
        return;
    }
    if (solver.async_heuristic_dispatcher_->wait_for_all(std::chrono::seconds(5))) {
        return;
    }
    solver.async_heuristic_dispatcher_->request_stop();
    solver.async_heuristic_dispatcher_->wait_for_all(std::chrono::seconds(2));
}

void AsyncHeuristicManager::start_workers(Solver& solver) {
    stop_workers(solver);
    if (!enabled(solver)) {
        return;
    }
    solver.async_heuristic_dispatcher_ =
        std::make_unique<detail::AsyncTaskDispatcher>(worker_count(solver));
}

void AsyncHeuristicManager::stop_workers(Solver& solver) {
    if (solver.async_heuristic_dispatcher_ == nullptr) {
        return;
    }
    solver.async_heuristic_dispatcher_->request_stop();
    solver.async_heuristic_dispatcher_->wait_for_all(std::chrono::seconds(2));
    solver.async_heuristic_dispatcher_.reset();
}

} // namespace simplex::bnb
