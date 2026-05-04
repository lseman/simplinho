#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "bnb/search.h"

namespace simplex::bnb::detail {

struct PopResult {
    std::optional<ActiveNode> node;
    bool terminated = false;
    int stolen_worker_id = -1; // -1 if node came from local queue
};

// Per-worker wait/notify state.  Allocated on the heap so it remains at a fixed
// address even when the outer vector is resized.
struct WorkerWaiter {
    std::mutex mutex;
    std::condition_variable cv;
    // Set to true by the pusher before notify_one; read by the waiter while
    // holding mutex to avoid missed wakeups between the queue-empty check and
    // the cv.wait call.
    bool wakeup_pending = false;
};

class SearchCoordinator {
  public:
    explicit SearchCoordinator(int worker_count = 1) : worker_count_(worker_count) {
        rebuild_per_worker_state_(worker_count);
    }

    void configure(int worker_count) {
        std::lock_guard<std::mutex> lock(coord_mutex_);
        worker_count_ = worker_count;
        rebuild_per_worker_state_locked_(worker_count);
        hybrid_counter_.store(0, std::memory_order_relaxed);
        active_workers_.store(0, std::memory_order_relaxed);
        hit_node_limit_.store(false, std::memory_order_release);
        found_unbounded_.store(false, std::memory_order_release);
        steal_attempts_.store(0, std::memory_order_relaxed);
        local_pops_.store(0, std::memory_order_relaxed);
        stolen_pops_.store(0, std::memory_order_relaxed);
    }

    void reset() {
        std::lock_guard<std::mutex> lock(coord_mutex_);
        rebuild_per_worker_state_locked_(worker_count_);
        hybrid_counter_.store(0, std::memory_order_relaxed);
        active_workers_.store(0, std::memory_order_relaxed);
        hit_node_limit_.store(false, std::memory_order_release);
        found_unbounded_.store(false, std::memory_order_release);
        steal_attempts_.store(0, std::memory_order_relaxed);
        local_pops_.store(0, std::memory_order_relaxed);
        stolen_pops_.store(0, std::memory_order_relaxed);
    }

    // Push a node to the designated worker's queue (or distribute round-robin if -1).
    // Hot path: takes only the target WLQ mutex, then notifies one waiter.
    void push(ActiveNode node, NodeSelectionStrategy strategy, bool maximize,
              int preferred_worker = -1) {
        int target_worker = preferred_worker;
        if (target_worker < 0) {
            const std::uint64_t idx = next_worker_index_.fetch_add(1, std::memory_order_relaxed);
            target_worker =
                static_cast<int>(idx % static_cast<std::uint64_t>(worker_queues_.size()));
        }
        if (target_worker >= static_cast<int>(worker_queues_.size())) {
            target_worker = static_cast<int>(worker_queues_.size()) - 1;
        }

        worker_queues_[target_worker].push(std::move(node), strategy, maximize);
        notify_worker_(target_worker);
    }

    // Try to pop a node from the worker's local queue (no waiting).
    std::optional<ActiveNode> try_pop(NodeSelectionStrategy strategy, bool maximize,
                                      int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                      int worker_id = -1) {
        int target_worker = worker_id < 0 ? 0 : worker_id;
        auto hc = hybrid_counter_.load(std::memory_order_relaxed);
        auto local_result = worker_queues_[target_worker].pop(strategy, maximize, hybrid_depth_bias,
                                                              plunging_bestfreq, &hc);
        hybrid_counter_.store(hc, std::memory_order_relaxed);
        if (local_result.has_value()) {
            local_pops_.fetch_add(1, std::memory_order_relaxed);
            return local_result;
        }
        return std::nullopt;
    }

    // Wait for a node from the local queue or steal from other workers.
    // Hot path: does NOT hold any global lock during the steal scan.
    PopResult wait_pop(NodeSelectionStrategy strategy, bool maximize, int hybrid_depth_bias = 5,
                       int plunging_bestfreq = 10, int worker_id = -1) {
        const int wid = worker_id < 0 ? 0 : worker_id;

        while (true) {
            if (found_unbounded_.load(std::memory_order_acquire) ||
                hit_node_limit_.load(std::memory_order_acquire)) {
                return {.node = std::nullopt, .terminated = true, .stolen_worker_id = -1};
            }

            // Try local queue first (takes only the per-queue WLQ mutex).
            auto hc = hybrid_counter_.load(std::memory_order_relaxed);
            auto local_result = worker_queues_[wid].pop(strategy, maximize, hybrid_depth_bias,
                                                        plunging_bestfreq, &hc);
            hybrid_counter_.store(hc, std::memory_order_relaxed);
            if (local_result.has_value()) {
                local_pops_.fetch_add(1, std::memory_order_relaxed);
                active_workers_.fetch_add(1, std::memory_order_acq_rel);
                return {.node = local_result, .terminated = false, .stolen_worker_id = -1};
            }

            // Attempt to steal from other workers.
            steal_attempts_.fetch_add(1, std::memory_order_relaxed);
            const int nw = static_cast<int>(worker_queues_.size());
            for (int round = 0; round < 3; ++round) {
                for (int other = 0; other < nw; ++other) {
                    if (other == wid)
                        continue;
                    auto hc2 = hybrid_counter_.load(std::memory_order_relaxed);
                    auto stolen = worker_queues_[other].steal(strategy, maximize, hybrid_depth_bias,
                                                              plunging_bestfreq, &hc2);
                    hybrid_counter_.store(hc2, std::memory_order_relaxed);
                    if (stolen.has_value()) {
                        stolen_pops_.fetch_add(1, std::memory_order_relaxed);
                        active_workers_.fetch_add(1, std::memory_order_acq_rel);
                        return {.node = stolen, .terminated = false, .stolen_worker_id = other};
                    }
                }
            }

            // Nothing available: check if we should terminate.
            if (active_workers_.load(std::memory_order_acquire) == 0 && !should_work_atomic_()) {
                return {.node = std::nullopt, .terminated = true, .stolen_worker_id = -1};
            }

            // Block on per-worker CV (short timeout to guard against missed wakeups).
            WorkerWaiter& waiter = *worker_waiters_[wid];
            std::unique_lock<std::mutex> lk(waiter.mutex);
            if (waiter.wakeup_pending) {
                waiter.wakeup_pending = false;
                // A notification arrived while we weren't waiting — re-check immediately.
                continue;
            }
            waiter.cv.wait_for(lk, std::chrono::microseconds(500));
            waiter.wakeup_pending = false;
        }
    }

    // Report that a worker has finished processing a node.
    void on_worker_finished() {
        const int prev = active_workers_.fetch_sub(1, std::memory_order_acq_rel);
        if (prev == 1) {
            // Last active worker — wake all so they can check termination.
            notify_all();
        }
    }

    void notify_all() {
        for (auto& w : worker_waiters_) {
            std::lock_guard<std::mutex> lk(w->mutex);
            w->wakeup_pending = true;
            w->cv.notify_one();
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(coord_mutex_);
        for (auto& q : worker_queues_) {
            q.clear();
        }
        active_workers_.store(0, std::memory_order_release);
        hit_node_limit_.store(false, std::memory_order_release);
        found_unbounded_.store(false, std::memory_order_release);
        steal_attempts_.store(0, std::memory_order_relaxed);
        local_pops_.store(0, std::memory_order_relaxed);
        stolen_pops_.store(0, std::memory_order_relaxed);
    }

    bool empty() const {
        for (const auto& q : worker_queues_) {
            if (!q.empty())
                return false;
        }
        return true;
    }

    int size() const {
        int total = 0;
        for (const auto& q : worker_queues_) {
            total += q.size();
        }
        return total;
    }

    bool should_work() const { return should_work_atomic_(); }

    bool should_terminate() const {
        return hit_node_limit_.load(std::memory_order_acquire) ||
               found_unbounded_.load(std::memory_order_acquire);
    }

    void mark_unbounded() {
        found_unbounded_.store(true, std::memory_order_release);
        notify_all();
    }

    void mark_node_limit_reached() {
        hit_node_limit_.store(true, std::memory_order_release);
        notify_all();
    }

    bool found_unbounded() const { return found_unbounded_.load(std::memory_order_acquire); }

    bool hit_node_limit() const { return hit_node_limit_.load(std::memory_order_acquire); }

    // Compute best bound across all local queues.
    double compute_best_bound(bool has_incumbent, double incumbent_objective, bool maximize,
                              const std::optional<double>& root_relaxation_objective) const {
        double best =
            has_incumbent ? incumbent_objective : std::numeric_limits<double>::quiet_NaN();

        for (const auto& q : worker_queues_) {
            const ActiveNode* best_node = q.peek_valid_node();
            if (best_node != nullptr) {
                if (!std::isfinite(best)) {
                    best = best_node->bound;
                } else {
                    best = maximize ? std::max(best, best_node->bound)
                                    : std::min(best, best_node->bound);
                }
            }
        }

        if (!std::isfinite(best) && root_relaxation_objective.has_value()) {
            best = *root_relaxation_objective;
        }

        return best;
    }

    template <typename Fn> void for_each_mutable_node(Fn&& fn) {
        for (auto& q : worker_queues_) {
            q.for_each_mutable(fn);
        }
    }

    struct WorkStatistics {
        int total_steal_attempts = 0;
        int local_pops = 0;
        int stolen_pops = 0;
        int total_nodes_processed = 0;
    };

    WorkStatistics get_work_statistics() const {
        WorkStatistics stats;
        stats.total_steal_attempts = steal_attempts_.load(std::memory_order_relaxed);
        stats.local_pops = local_pops_.load(std::memory_order_relaxed);
        stats.stolen_pops = stolen_pops_.load(std::memory_order_relaxed);
        return stats;
    }

    std::vector<int> get_queue_sizes() const {
        std::vector<int> sizes;
        sizes.reserve(worker_queues_.size());
        for (const auto& q : worker_queues_) {
            sizes.push_back(q.size());
        }
        return sizes;
    }

    int worker_count() const { return worker_count_; }

  private:
    bool should_work_atomic_() const {
        for (const auto& q : worker_queues_) {
            if (!q.empty())
                return true;
        }
        return false;
    }

    void notify_worker_(int worker_id) {
        WorkerWaiter& waiter = *worker_waiters_[worker_id];
        {
            std::lock_guard<std::mutex> lk(waiter.mutex);
            waiter.wakeup_pending = true;
        }
        waiter.cv.notify_one();
    }

    void rebuild_per_worker_state_(int count) {
        worker_queues_.clear();
        worker_waiters_.clear();
        for (int i = 0; i < count; ++i) {
            worker_queues_.emplace_back();
            worker_waiters_.push_back(std::make_unique<WorkerWaiter>());
        }
    }

    void rebuild_per_worker_state_locked_(int count) {
        // Must be called with coord_mutex_ held.
        worker_queues_.clear();
        worker_waiters_.clear();
        for (int i = 0; i < count; ++i) {
            worker_queues_.emplace_back();
            worker_waiters_.push_back(std::make_unique<WorkerWaiter>());
        }
    }

    mutable std::mutex coord_mutex_; // only for configure/reset

    std::vector<WorkerLocalQueue> worker_queues_;
    std::vector<std::unique_ptr<WorkerWaiter>> worker_waiters_;

    std::atomic<std::uint64_t> hybrid_counter_{0};
    std::atomic<int> active_workers_{0};
    int worker_count_ = 1;
    std::atomic<bool> hit_node_limit_{false};
    std::atomic<bool> found_unbounded_{false};
    std::atomic<int> steal_attempts_{0};
    std::atomic<int> local_pops_{0};
    std::atomic<int> stolen_pops_{0};
    std::atomic<std::uint64_t> next_worker_index_{0};
};

} // namespace simplex::bnb::detail
