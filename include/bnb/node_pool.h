#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "bnb/search.h"

namespace simplex::bnb::detail {

class NodePool {
  public:
    explicit NodePool(int worker_count = 1) { configure(worker_count); }

    void configure(int worker_count) {
        clear();
        worker_count_ = std::max(1, worker_count);
        local_queues_.clear();
        local_queues_.reserve(worker_count_);
        for (int i = 0; i < worker_count_; ++i) {
            local_queues_.push_back(std::make_unique<LocalQueue>());
        }
    }

    int worker_count() const noexcept { return worker_count_; }

    bool empty() const noexcept { return size() == 0; }

    int size() const noexcept {
        int total = 0;
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            total += global_queue_.size();
        }
        for (const auto& queue : local_queues_) {
            std::lock_guard<std::mutex> lock(queue->mutex);
            total += static_cast<int>(queue->nodes.size());
        }
        return total;
    }

    void clear() noexcept {
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_queue_.clear();
        }
        for (const auto& queue : local_queues_) {
            std::lock_guard<std::mutex> lock(queue->mutex);
            queue->nodes.clear();
        }
    }

    void push(ActiveNode node, NodeSelectionStrategy strategy, bool maximize,
              int preferred_worker = -1) {
        prepare_node_for_queue_(&node);
        if (uses_local_queue_(strategy) && preferred_worker >= 0 && preferred_worker < worker_count_) {
            push_local_(preferred_worker, std::move(node), strategy, maximize);
            return;
        }
        std::lock_guard<std::mutex> lock(global_mutex_);
        global_queue_.push(std::move(node), strategy, maximize);
    }

    std::optional<ActiveNode> pop(NodeSelectionStrategy strategy, bool maximize,
                                  int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                  std::uint64_t* hybrid_counter = nullptr, int worker_id = -1) {
        const bool local_first =
            worker_id >= 0 && worker_id < worker_count_ && uses_local_queue_(strategy);

        if (local_first) {
            if (std::optional<ActiveNode> local = pop_local_(worker_id); local.has_value()) {
                return local;
            }
        }

        if (std::optional<ActiveNode> global =
                pop_global_(strategy, maximize, hybrid_depth_bias, plunging_bestfreq,
                            hybrid_counter);
            global.has_value()) {
            return global;
        }

        if (worker_id >= 0 && worker_id < worker_count_) {
            if (std::optional<ActiveNode> stolen = steal_(worker_id); stolen.has_value()) {
                return stolen;
            }
            return pop_local_(worker_id);
        }

        return std::nullopt;
    }

    double compute_best_bound(bool has_incumbent, double incumbent_objective, bool maximize,
                              const std::optional<double>& root_relaxation_objective) const {
        double best = std::numeric_limits<double>::quiet_NaN();
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            best = global_queue_.compute_best_bound(false, incumbent_objective, maximize,
                                                   root_relaxation_objective);
        }

        for (const auto& queue : local_queues_) {
            std::lock_guard<std::mutex> lock(queue->mutex);
            for (const ActiveNode& node : queue->nodes) {
                if (!std::isfinite(node.bound)) {
                    continue;
                }
                if (!std::isfinite(best)) {
                    best = node.bound;
                    continue;
                }
                best = maximize ? std::max(best, node.bound) : std::min(best, node.bound);
            }
        }

        if (has_incumbent && std::isfinite(incumbent_objective)) {
            if (!std::isfinite(best)) {
                best = incumbent_objective;
            } else {
                best = maximize ? std::max(best, incumbent_objective)
                                : std::min(best, incumbent_objective);
            }
        }
        if (!std::isfinite(best) && root_relaxation_objective.has_value()) {
            best = *root_relaxation_objective;
        }
        return best;
    }

    template <typename Fn> void for_each_mutable(Fn&& fn) {
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            global_queue_.for_each_mutable(fn);
        }
        for (const auto& queue : local_queues_) {
            std::lock_guard<std::mutex> lock(queue->mutex);
            for (ActiveNode& node : queue->nodes) {
                fn(node);
            }
        }
    }

  private:
    struct LocalQueue {
        std::mutex mutex;
        std::deque<ActiveNode> nodes;
    };

    // Keep only one private plunge candidate per worker. In a binary tree,
    // retaining more siblings locally quickly makes the search behave almost
    // serial because workers hoard both children before the frontier has a
    // chance to spread work across threads.
    static constexpr int kLocalRetainCount_ = 1;

    static bool uses_local_queue_(NodeSelectionStrategy strategy) {
        return strategy == NodeSelectionStrategy::DepthFirst ||
               strategy == NodeSelectionStrategy::Hybrid || is_plunging_strategy(strategy);
    }

    static void prepare_node_for_queue_(ActiveNode* node) {
        if (node == nullptr) {
            return;
        }
        if (node->domain == nullptr &&
            has_materialized_bounds(node->lower_bounds, node->upper_bounds)) {
            node->domain = make_materialized_domain(node->lower_bounds, node->upper_bounds);
        }
        dematerialize_active_node(node);
    }

    void push_local_(int worker_id, ActiveNode node, NodeSelectionStrategy strategy,
                     bool maximize) {
        std::vector<ActiveNode> spill_nodes;
        {
            std::lock_guard<std::mutex> lock(local_queues_[worker_id]->mutex);
            local_queues_[worker_id]->nodes.push_back(std::move(node));
            while (static_cast<int>(local_queues_[worker_id]->nodes.size()) > kLocalRetainCount_) {
                spill_nodes.push_back(std::move(local_queues_[worker_id]->nodes.front()));
                local_queues_[worker_id]->nodes.pop_front();
            }
        }
        if (spill_nodes.empty()) {
            return;
        }
        std::lock_guard<std::mutex> global_lock(global_mutex_);
        for (ActiveNode& spill_node : spill_nodes) {
            global_queue_.push(std::move(spill_node), strategy, maximize);
        }
    }

    std::optional<ActiveNode> pop_local_(int worker_id) {
        if (worker_id < 0 || worker_id >= worker_count_) {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(local_queues_[worker_id]->mutex);
        if (local_queues_[worker_id]->nodes.empty()) {
            return std::nullopt;
        }
        ActiveNode node = std::move(local_queues_[worker_id]->nodes.back());
        local_queues_[worker_id]->nodes.pop_back();
        return node;
    }

    std::optional<ActiveNode> pop_global_(NodeSelectionStrategy strategy, bool maximize,
                                          int hybrid_depth_bias, int plunging_bestfreq,
                                          std::uint64_t* hybrid_counter) {
        std::lock_guard<std::mutex> lock(global_mutex_);
        if (global_queue_.empty()) {
            return std::nullopt;
        }
        return global_queue_.pop(strategy, maximize, hybrid_depth_bias, plunging_bestfreq,
                                 hybrid_counter);
    }

    std::optional<ActiveNode> steal_(int worker_id) {
        if (worker_id < 0 || worker_id >= worker_count_) {
            return std::nullopt;
        }
        for (int offset = 1; offset < worker_count_; ++offset) {
            const int donor_id = (worker_id + offset) % worker_count_;
            std::lock_guard<std::mutex> lock(local_queues_[donor_id]->mutex);
            if (local_queues_[donor_id]->nodes.size() <= 1) {
                continue;
            }
            ActiveNode node = std::move(local_queues_[donor_id]->nodes.front());
            local_queues_[donor_id]->nodes.pop_front();
            return node;
        }
        return std::nullopt;
    }

    int worker_count_ = 1;
    mutable std::mutex global_mutex_;
    OpenNodeQueue global_queue_;
    std::vector<std::unique_ptr<LocalQueue>> local_queues_;
};

inline void push_active_node(NodePool& active_nodes, ActiveNode node, NodeSelectionStrategy strategy,
                             bool maximize, int preferred_worker = -1) {
    active_nodes.push(std::move(node), strategy, maximize, preferred_worker);
}

inline std::optional<ActiveNode> pop_next_node(NodePool& active_nodes,
                                               NodeSelectionStrategy strategy, bool maximize,
                                               int hybrid_depth_bias = 5,
                                               int plunging_bestfreq = 10,
                                               std::uint64_t* hybrid_counter = nullptr,
                                               int worker_id = -1) {
    return active_nodes.pop(strategy, maximize, hybrid_depth_bias, plunging_bestfreq,
                            hybrid_counter, worker_id);
}

inline double compute_best_bound(const NodePool& active_nodes, bool has_incumbent,
                                 double incumbent_objective, bool maximize,
                                 const std::optional<double>& root_relaxation_objective) {
    return active_nodes.compute_best_bound(has_incumbent, incumbent_objective, maximize,
                                           root_relaxation_objective);
}

} // namespace simplex::bnb::detail
