#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
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

class SearchCoordinator {
  public:
    explicit SearchCoordinator(int worker_count = 1) : worker_count_(worker_count) {
        for (int i = 0; i < worker_count; ++i) {
            worker_queues_.emplace_back();
        }
    }

    void configure(int worker_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_count_ = worker_count;
        worker_queues_.clear();
        for (int i = 0; i < worker_count; ++i) {
            worker_queues_.emplace_back();
        }
        hybrid_counter_ = 0;
        active_workers_ = 0;
        hit_node_limit_ = false;
        found_unbounded_ = false;
        steal_attempts_ = 0;
        local_pops_ = 0;
        stolen_pops_ = 0;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_queues_.clear();
        for (int i = 0; i < worker_count_; ++i) {
            worker_queues_.emplace_back();
        }
        hybrid_counter_ = 0;
        active_workers_ = 0;
        hit_node_limit_ = false;
        found_unbounded_ = false;
        steal_attempts_ = 0;
        local_pops_ = 0;
        stolen_pops_ = 0;
    }

    // Push a node to the designated worker's queue (or distribute round-robin if -1)
    void push(ActiveNode node, NodeSelectionStrategy strategy, bool maximize,
              int preferred_worker = -1) {
        std::lock_guard<std::mutex> lock(mutex_);
        int target_worker = preferred_worker;
        if (target_worker < 0) {
            target_worker = next_worker_index_++ % static_cast<int>(worker_queues_.size());
        }
        // Ensure target_worker is valid
        if (target_worker >= static_cast<int>(worker_queues_.size())) {
            target_worker = worker_queues_.size() - 1;
        }

        worker_queues_[target_worker].push(std::move(node), strategy, maximize);
        cv_.notify_one();
    }

    // Try to pop a node from the worker's local queue (no waiting)
    std::optional<ActiveNode> try_pop(NodeSelectionStrategy strategy, bool maximize,
                                      int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                      int worker_id = -1) {
        std::lock_guard<std::mutex> lock(mutex_);
        int target_worker = worker_id;
        if (target_worker < 0) {
            target_worker = 0;
        }

        auto local_result = worker_queues_[target_worker].pop(strategy, maximize, hybrid_depth_bias,
                                                              plunging_bestfreq, &hybrid_counter_);
        if (local_result.has_value()) {
            ++local_pops_;
            return local_result;
        }

        return std::nullopt;
    }

    // Wait for a node from local queue or steal from other workers
    PopResult wait_pop(NodeSelectionStrategy strategy, bool maximize, int hybrid_depth_bias = 5,
                       int plunging_bestfreq = 10, int worker_id = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            if (found_unbounded_ || hit_node_limit_) {
                return {.node = std::nullopt, .terminated = true, .stolen_worker_id = -1};
            }

            // First, try local queue
            int target_worker = worker_id;
            if (target_worker < 0) {
                target_worker = 0;
            }
            auto local_result = worker_queues_[target_worker].pop(
                strategy, maximize, hybrid_depth_bias, plunging_bestfreq, &hybrid_counter_);
            if (local_result.has_value()) {
                ++local_pops_;
                ++active_workers_;
                return {.node = local_result, .terminated = false, .stolen_worker_id = -1};
            }

            // Local queue empty, attempt to steal from other workers
            ++steal_attempts_;
            int steal_rounds = 3;
            std::optional<ActiveNode> stolen_node;
            int best_stolen_worker = -1;

            while (steal_rounds-- > 0 && !stolen_node.has_value()) {
                for (int other_id = 0; other_id < static_cast<int>(worker_queues_.size());
                     ++other_id) {
                    if (other_id == worker_id) {
                        continue;
                    }
                    auto steal_result = worker_queues_[other_id].steal(
                        strategy, maximize, hybrid_depth_bias, plunging_bestfreq, &hybrid_counter_);
                    if (steal_result.has_value()) {
                        stolen_node = std::move(steal_result);
                        best_stolen_worker = other_id;
                        ++stolen_pops_;
                        break;
                    }
                }
            }

            if (stolen_node.has_value()) {
                ++active_workers_;
                return {.node = stolen_node,
                        .terminated = false,
                        .stolen_worker_id = best_stolen_worker};
            }

            if (active_workers_ == 0 && !should_work_locked_()) {
                return {.node = std::nullopt, .terminated = true, .stolen_worker_id = -1};
            }

            cv_.wait(lock);
        }
    }

    // Report that a worker has finished processing a node
    void on_worker_finished() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_workers_ > 0) {
            --active_workers_;
        }
        if (active_workers_ == 0) {
            cv_.notify_all();
        }
    }

    // Signal that all workers are idle (no more work)
    void notify_all() { cv_.notify_all(); }

    // Empty the coordinator
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_queues_.clear();
        for (int i = 0; i < worker_count_; ++i) {
            worker_queues_.emplace_back();
        }
        active_workers_ = 0;
        hit_node_limit_ = false;
        found_unbounded_ = false;
        steal_attempts_ = 0;
        local_pops_ = 0;
        stolen_pops_ = 0;
    }

    // Check if empty (all queues empty and no workers active)
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_locked_() == 0;
    }

    // Get the number of nodes across all queues
    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_locked_();
    }

    // Check if any node available
    bool should_work() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return should_work_locked_();
    }

    bool should_terminate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hit_node_limit_ || found_unbounded_;
    }

    void mark_unbounded() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            found_unbounded_ = true;
        }
        cv_.notify_all();
    }

    void mark_node_limit_reached() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            hit_node_limit_ = true;
        }
        cv_.notify_all();
    }

    bool found_unbounded() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return found_unbounded_;
    }

    bool hit_node_limit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return hit_node_limit_;
    }

    // Compute best bound across all local queues
    double compute_best_bound(bool has_incumbent, double incumbent_objective, bool maximize,
                              const std::optional<double>& root_relaxation_objective) const {
        std::lock_guard<std::mutex> lock(mutex_);
        double best = incumbent_objective;

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
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& q : worker_queues_) {
            q.for_each_mutable(fn);
        }
    }

    // Get statistics about work stealing
    struct WorkStatistics {
        int total_steal_attempts = 0;
        int local_pops = 0;
        int stolen_pops = 0;
        int total_nodes_processed = 0;
    };

    WorkStatistics get_work_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        WorkStatistics stats;
        stats.total_steal_attempts = steal_attempts_;
        stats.local_pops = local_pops_;
        stats.stolen_pops = stolen_pops_;
        // total_nodes_processed is tracked per worker in worker_stats_
        return stats;
    }

    // Get individual worker queue sizes for load balancing diagnostics
    std::vector<int> get_queue_sizes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<int> sizes;
        sizes.reserve(worker_queues_.size());
        for (const auto& q : worker_queues_) {
            sizes.push_back(q.size());
        }
        return sizes;
    }

    int worker_count() const { return worker_count_; }

  private:
    int size_locked_() const {
        int total = 0;
        for (const auto& q : worker_queues_) {
            total += q.size();
        }
        return total;
    }

    bool should_work_locked_() const {
        for (const auto& q : worker_queues_) {
            if (!q.empty()) {
                return true;
            }
        }
        return false;
    }

    mutable std::mutex mutex_;

    std::vector<WorkerLocalQueue> worker_queues_;
    std::uint64_t hybrid_counter_ = 0;
    int active_workers_ = 0;
    int worker_count_ = 1;
    bool hit_node_limit_ = false;
    bool found_unbounded_ = false;
    int steal_attempts_ = 0;
    int local_pops_ = 0;
    int stolen_pops_ = 0;
    std::uint64_t next_worker_index_ = 0;
    std::condition_variable cv_;
};

} // namespace simplex::bnb::detail
