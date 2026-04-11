template <typename Task>
void AsyncHeuristicManager::dispatch(Solver& solver, std::uint64_t launch_order, Task&& task) {
    auto work = [task = std::forward<Task>(task)]() mutable { std::invoke(task); };
    if (!enabled(solver)) {
        work();
        return;
    }

    if (solver.async_heuristic_dispatcher_ == nullptr) {
        start_workers(solver);
    }
    auto* dispatcher = solver.async_heuristic_dispatcher_.get();
    if (dispatcher == nullptr || dispatcher->stop_requested()) {
        work();
        return;
    }

    const bool dropped_stale = solver.current_search_order_() > launch_order + staleness_window();
    if (dropped_stale) {
        return;
    }

    if (dispatcher->pending_tasks() >= max_tasks(solver)) {
        return;
    }
    dispatcher->spawn(std::move(work));
}
