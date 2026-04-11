#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bnb/types.h"

namespace simplex::bnb::detail {

struct ReasonLiteral {
    int variable = -1;
    bool is_lower = false;
    double value = std::numeric_limits<double>::quiet_NaN();
};

struct PropagationReason {
    int parent_literal = -1;
    int row_index = -1;
    std::vector<ReasonLiteral> antecedents;

    PropagationReason() = default;
    ~PropagationReason() = default;
    PropagationReason(const PropagationReason&) = default;
    PropagationReason& operator=(const PropagationReason&) = default;
    PropagationReason(PropagationReason&&) noexcept = default;
    PropagationReason& operator=(PropagationReason&&) noexcept = default;
};

using NodeReasonStore = std::vector<PropagationReason>;

struct DomainChange {
    int variable = -1;
    double lower_bound = std::numeric_limits<double>::quiet_NaN();
    double upper_bound = std::numeric_limits<double>::quiet_NaN();
};

struct NodeDomain {
    std::shared_ptr<const NodeDomain> parent;
    std::shared_ptr<const Eigen::VectorXd> root_lower_bounds;
    std::shared_ptr<const Eigen::VectorXd> root_upper_bounds;
    std::vector<DomainChange> changes;
    int variable_count = 0;
    int chain_depth = 0;
    int total_change_count = 0;
    mutable std::mutex materialization_mutex;
    mutable std::shared_ptr<const Eigen::VectorXd> cached_lower_bounds;
    mutable std::shared_ptr<const Eigen::VectorXd> cached_upper_bounds;
};

inline bool has_materialized_bounds(const Eigen::VectorXd& lower_bounds,
                                    const Eigen::VectorXd& upper_bounds) {
    return lower_bounds.size() > 0 && upper_bounds.size() > 0;
}

inline void clear_materialized_bounds(Eigen::VectorXd* lower_bounds,
                                      Eigen::VectorXd* upper_bounds) {
    if (lower_bounds != nullptr) {
        lower_bounds->resize(0);
    }
    if (upper_bounds != nullptr) {
        upper_bounds->resize(0);
    }
}

inline std::shared_ptr<const NodeDomain>
make_materialized_domain(const Eigen::VectorXd& lower_bounds, const Eigen::VectorXd& upper_bounds) {
    auto domain = std::make_shared<NodeDomain>();
    domain->root_lower_bounds = std::make_shared<Eigen::VectorXd>(lower_bounds);
    domain->root_upper_bounds = std::make_shared<Eigen::VectorXd>(upper_bounds);
    domain->variable_count = std::min<int>(lower_bounds.size(), upper_bounds.size());
    domain->chain_depth = 0;
    domain->total_change_count = 0;
    domain->cached_lower_bounds = domain->root_lower_bounds;
    domain->cached_upper_bounds = domain->root_upper_bounds;
    return domain;
}

inline void materialize_domain_bounds(const std::shared_ptr<const NodeDomain>& domain,
                                      Eigen::VectorXd* lower_bounds,
                                      Eigen::VectorXd* upper_bounds) {
    if (lower_bounds == nullptr || upper_bounds == nullptr) {
        return;
    }
    if (domain == nullptr) {
        lower_bounds->resize(0);
        upper_bounds->resize(0);
        return;
    }

    {
        std::lock_guard<std::mutex> lock(domain->materialization_mutex);
        if (domain->cached_lower_bounds != nullptr && domain->cached_upper_bounds != nullptr) {
            *lower_bounds = *domain->cached_lower_bounds;
            *upper_bounds = *domain->cached_upper_bounds;
            return;
        }
    }

    std::vector<const NodeDomain*> chain;
    for (const NodeDomain* current = domain.get(); current != nullptr;
         current = current->parent.get()) {
        chain.push_back(current);
    }
    if (chain.empty()) {
        lower_bounds->resize(0);
        upper_bounds->resize(0);
        return;
    }

    const NodeDomain* root = chain.back();
    if (root->root_lower_bounds == nullptr || root->root_upper_bounds == nullptr) {
        lower_bounds->resize(0);
        upper_bounds->resize(0);
        return;
    }

    *lower_bounds = *root->root_lower_bounds;
    *upper_bounds = *root->root_upper_bounds;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const NodeDomain* current = *it;
        for (const DomainChange& change : current->changes) {
            if (change.variable < 0 || change.variable >= lower_bounds->size() ||
                change.variable >= upper_bounds->size()) {
                continue;
            }
            (*lower_bounds)(change.variable) = change.lower_bound;
            (*upper_bounds)(change.variable) = change.upper_bound;
        }
    }

    auto cached_lower_bounds = std::make_shared<Eigen::VectorXd>(*lower_bounds);
    auto cached_upper_bounds = std::make_shared<Eigen::VectorXd>(*upper_bounds);
    {
        std::lock_guard<std::mutex> lock(domain->materialization_mutex);
        if (domain->cached_lower_bounds == nullptr || domain->cached_upper_bounds == nullptr) {
            domain->cached_lower_bounds = cached_lower_bounds;
            domain->cached_upper_bounds = cached_upper_bounds;
        }
        *lower_bounds = *domain->cached_lower_bounds;
        *upper_bounds = *domain->cached_upper_bounds;
    }
}

inline std::pair<double, double>
resolve_domain_variable_bounds(const std::shared_ptr<const NodeDomain>& domain, int variable) {
    double lower = std::numeric_limits<double>::quiet_NaN();
    double upper = std::numeric_limits<double>::quiet_NaN();
    if (domain == nullptr) {
        return {lower, upper};
    }
    const NodeDomain* root = domain.get();
    while (root->parent != nullptr) {
        root = root->parent.get();
    }
    if (root->root_lower_bounds == nullptr || root->root_upper_bounds == nullptr) {
        return {lower, upper};
    }
    if (variable < 0 || variable >= root->root_lower_bounds->size()) {
        return {lower, upper};
    }
    lower = (*root->root_lower_bounds)(variable);
    upper = (*root->root_upper_bounds)(variable);
    for (const NodeDomain* current = domain.get(); current != nullptr;
         current = current->parent.get()) {
        for (const DomainChange& change : current->changes) {
            if (change.variable == variable) {
                lower = change.lower_bound;
                upper = change.upper_bound;
            }
        }
    }
    return {lower, upper};
}

inline std::shared_ptr<const NodeDomain> compress_domain_from_reference(
    const std::shared_ptr<const NodeDomain>& reference_domain,
    const Eigen::VectorXd& reference_lower_bounds, const Eigen::VectorXd& reference_upper_bounds,
    const Eigen::VectorXd& lower_bounds, const Eigen::VectorXd& upper_bounds, double tol) {
    const int n =
        std::min<int>(std::min<int>(reference_lower_bounds.size(), reference_upper_bounds.size()),
                      std::min<int>(lower_bounds.size(), upper_bounds.size()));
    if (n == 0) {
        return make_materialized_domain(lower_bounds, upper_bounds);
    }

    std::vector<DomainChange> changes;
    changes.reserve(8);
    for (int j = 0; j < n; ++j) {
        const bool lower_changed = lower_bounds(j) > reference_lower_bounds(j) + tol;
        const bool upper_changed = upper_bounds(j) < reference_upper_bounds(j) - tol;
        if (!lower_changed && !upper_changed) {
            continue;
        }
        changes.push_back(DomainChange{j, lower_bounds(j), upper_bounds(j)});
    }

    if (changes.empty()) {
        return reference_domain != nullptr ? reference_domain
                                           : make_materialized_domain(lower_bounds, upper_bounds);
    }

    if (reference_domain == nullptr) {
        return make_materialized_domain(lower_bounds, upper_bounds);
    }

    constexpr int max_domain_chain_depth = 32;
    constexpr int max_domain_total_changes = 256;
    if (reference_domain->chain_depth >= max_domain_chain_depth ||
        reference_domain->total_change_count + static_cast<int>(changes.size()) >
            max_domain_total_changes) {
        return make_materialized_domain(lower_bounds, upper_bounds);
    }

    auto domain = std::make_shared<NodeDomain>();
    domain->parent = reference_domain;
    domain->changes = std::move(changes);
    domain->variable_count = reference_domain->variable_count;
    domain->chain_depth = reference_domain->chain_depth + 1;
    domain->total_change_count =
        reference_domain->total_change_count + static_cast<int>(domain->changes.size());
    return domain;
}

struct ActiveNode {
    int id = -1;
    int parent_id = -1;
    int depth = 0;
    int domain_change_count = 0;
    std::uint64_t order = 0;
    double bound = std::numeric_limits<double>::quiet_NaN();
    double estimate = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    std::shared_ptr<const NodeDomain> domain;
    bool bounds_presolved = false;
    std::uint64_t presolve_cuts_revision = 0;
    std::uint64_t presolve_conflicts_revision = 0;
    std::uint64_t presolve_implications_revision = 0;
    std::optional<LPBasis> basis;
    std::shared_ptr<NodeReasonStore> reasons;
};

inline void materialize_active_node(ActiveNode* node) {
    if (node == nullptr || has_materialized_bounds(node->lower_bounds, node->upper_bounds)) {
        return;
    }
    materialize_domain_bounds(node->domain, &node->lower_bounds, &node->upper_bounds);
}

inline void dematerialize_active_node(ActiveNode* node) {
    if (node == nullptr) {
        return;
    }
    clear_materialized_bounds(&node->lower_bounds, &node->upper_bounds);
}

inline int append_tree_node(std::vector<TreeNode>& tree_nodes, int parent_id, int depth,
                            std::uint64_t order) {
    TreeNode info;
    info.id = static_cast<int>(tree_nodes.size());
    info.parent_id = parent_id;
    info.depth = depth;
    info.order = order;
    tree_nodes.push_back(info);
    return info.id;
}

inline bool best_bound_worse(const ActiveNode& lhs, const ActiveNode& rhs, bool maximize) {
    const bool rhs_better =
        maximize ? (rhs.bound > lhs.bound + 1e-12) : (rhs.bound < lhs.bound - 1e-12);
    const bool tied = std::abs(lhs.bound - rhs.bound) <= 1e-12;
    if (!tied) {
        return rhs_better;
    }
    if (lhs.domain_change_count != rhs.domain_change_count) {
        return rhs.domain_change_count > lhs.domain_change_count;
    }
    return rhs.order < lhs.order;
}

inline bool best_estimate_worse(const ActiveNode& lhs, const ActiveNode& rhs, bool maximize) {
    const double lhs_value = std::isfinite(lhs.estimate) ? lhs.estimate : lhs.bound;
    const double rhs_value = std::isfinite(rhs.estimate) ? rhs.estimate : rhs.bound;
    const bool rhs_better =
        maximize ? (rhs_value > lhs_value + 1e-12) : (rhs_value < lhs_value - 1e-12);
    const bool tied = std::abs(lhs_value - rhs_value) <= 1e-12;
    if (!tied) {
        return rhs_better;
    }
    if (lhs.domain_change_count != rhs.domain_change_count) {
        return rhs.domain_change_count > lhs.domain_change_count;
    }
    return rhs.order < lhs.order;
}

inline double hybrid_node_score(const ActiveNode& node) {
    constexpr double estimate_weight = 0.1;
    constexpr double bound_weight = 1.0 - estimate_weight;
    const double estimate = std::isfinite(node.estimate) ? node.estimate : node.bound;
    const double bound = std::isfinite(node.bound) ? node.bound : estimate;
    return estimate_weight * estimate + bound_weight * bound;
}

inline bool hybrid_worse(const ActiveNode& lhs, const ActiveNode& rhs, bool maximize) {
    const double lhs_score = hybrid_node_score(lhs);
    const double rhs_score = hybrid_node_score(rhs);
    const bool rhs_better =
        maximize ? (rhs_score > lhs_score + 1e-12) : (rhs_score < lhs_score - 1e-12);
    const bool tied = std::abs(lhs_score - rhs_score) <= 1e-12;
    if (!tied) {
        return rhs_better;
    }
    if (lhs.domain_change_count != rhs.domain_change_count) {
        return rhs.domain_change_count > lhs.domain_change_count;
    }
    return rhs.order < lhs.order;
}

inline bool uses_best_bound_heap(NodeSelectionStrategy strategy) {
    return strategy == NodeSelectionStrategy::BestBound ||
           strategy == NodeSelectionStrategy::BestFirstPlunging;
}

inline bool uses_hybrid_heap(NodeSelectionStrategy strategy) {
    return strategy == NodeSelectionStrategy::Hybrid;
}

inline bool uses_best_estimate_heap(NodeSelectionStrategy strategy) {
    return strategy == NodeSelectionStrategy::BestEstimate ||
           strategy == NodeSelectionStrategy::BestEstimatePlunging ||
           strategy == NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging;
}

inline bool is_plunging_strategy(NodeSelectionStrategy strategy) {
    return strategy == NodeSelectionStrategy::Hybrid ||
           strategy == NodeSelectionStrategy::BestFirstPlunging ||
           strategy == NodeSelectionStrategy::BestEstimatePlunging ||
           strategy == NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging;
}

inline double node_selection_score(const ActiveNode& node, NodeSelectionStrategy strategy) {
    if (uses_best_bound_heap(strategy)) {
        return node.bound;
    }
    if (uses_hybrid_heap(strategy)) {
        return hybrid_node_score(node);
    }
    return std::isfinite(node.estimate) ? node.estimate : node.bound;
}

// ============================================================================
// Worker Local Queue
// ============================================================================
// A decentralized queue for worker-local node storage with support for
// remote stealing. Each worker maintains a local queue and can contribute
// nodes to a shared stealing heap when their local queue is empty.
// ============================================================================
class WorkerLocalQueue {
  public:
    WorkerLocalQueue() = default;
    WorkerLocalQueue(const WorkerLocalQueue&) = delete;
    WorkerLocalQueue& operator=(const WorkerLocalQueue&) = delete;

    WorkerLocalQueue(WorkerLocalQueue&& other) noexcept
        : nodes_(std::move(other.nodes_)), local_heap_(std::move(other.local_heap_)),
          stealing_heap_(std::move(other.stealing_heap_)), next_handle_(other.next_handle_),
          next_stamp_(other.next_stamp_),
          frontier_root_lower_bounds_(std::move(other.frontier_root_lower_bounds_)),
          frontier_root_upper_bounds_(std::move(other.frontier_root_upper_bounds_)),
          frontier_root_domain_(std::move(other.frontier_root_domain_)),
          lower_changed_counts_(std::move(other.lower_changed_counts_)),
          upper_changed_counts_(std::move(other.upper_changed_counts_)),
          lower_changed_values_(std::move(other.lower_changed_values_)),
          upper_changed_values_(std::move(other.upper_changed_values_)) {}

    WorkerLocalQueue& operator=(WorkerLocalQueue&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            nodes_ = std::move(other.nodes_);
            local_heap_ = std::move(other.local_heap_);
            stealing_heap_ = std::move(other.stealing_heap_);
            next_handle_ = other.next_handle_;
            next_stamp_ = other.next_stamp_;
            frontier_root_lower_bounds_ = std::move(other.frontier_root_lower_bounds_);
            frontier_root_upper_bounds_ = std::move(other.frontier_root_upper_bounds_);
            frontier_root_domain_ = std::move(other.frontier_root_domain_);
            lower_changed_counts_ = std::move(other.lower_changed_counts_);
            upper_changed_counts_ = std::move(other.upper_changed_counts_);
            lower_changed_values_ = std::move(other.lower_changed_values_);
            upper_changed_values_ = std::move(other.upper_changed_values_);
        }
        return *this;
    }

    friend class SearchCoordinator;

  private:
    static constexpr double kQueueDomainTol_ = 1e-12;

    struct NodeEntry {
        ActiveNode node;
        std::uint64_t stamp = 0;
        std::vector<DomainChange> root_domain_changes;
    };

    struct QueueKey {
        int handle = -1;
        std::uint64_t stamp = 0;
        double normalized_score = -std::numeric_limits<double>::infinity();
        int depth = 0;
        int domain_change_count = 0;
        std::uint64_t order = 0;
    };

    struct QueueKeyLess {
        bool operator()(const QueueKey& lhs, const QueueKey& rhs) const noexcept {
            if (std::abs(lhs.normalized_score - rhs.normalized_score) > 1e-12) {
                return lhs.normalized_score < rhs.normalized_score;
            }
            if (lhs.depth != rhs.depth) {
                return lhs.depth < rhs.depth;
            }
            if (lhs.domain_change_count != rhs.domain_change_count) {
                return lhs.domain_change_count < rhs.domain_change_count;
            }
            if (lhs.order != rhs.order) {
                return lhs.order > rhs.order;
            }
            return lhs.handle > rhs.handle;
        }
    };

    struct StealEntry {
        ActiveNode node;
        std::uint64_t stamp = 0;
    };

    struct StealKey {
        int handle = -1;
        std::uint64_t stamp = 0;
        double score = -std::numeric_limits<double>::infinity();
        int depth = 0;
        int domain_change_count = 0;
        std::uint64_t order = 0;
    };

    struct StealKeyLess {
        bool operator()(const StealKey& lhs, const StealKey& rhs) const noexcept {
            if (std::abs(lhs.score - rhs.score) > 1e-12) {
                return lhs.score < rhs.score;
            }
            if (lhs.depth != rhs.depth) {
                return lhs.depth < rhs.depth;
            }
            if (lhs.domain_change_count != rhs.domain_change_count) {
                return lhs.domain_change_count < rhs.domain_change_count;
            }
            if (lhs.order != rhs.order) {
                return lhs.order < rhs.order;
            }
            return lhs.stamp < rhs.stamp;
        }
    };

    std::unordered_map<int, NodeEntry> nodes_;
    std::priority_queue<QueueKey, std::vector<QueueKey>, QueueKeyLess> local_heap_;
    std::priority_queue<StealKey, std::vector<StealKey>, StealKeyLess> stealing_heap_;
    mutable std::mutex mutex_;
    int next_handle_ = 0;
    std::uint64_t next_stamp_ = 1;

  public:
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return nodes_.empty();
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(nodes_.size());
    }

    template <typename Fn> void for_each_mutable(Fn&& fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [handle, entry] : nodes_) {
            fn(entry.node);
        }
    }

    void push(ActiveNode node, NodeSelectionStrategy strategy, bool maximize) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int handle = next_handle_++;
        const std::uint64_t stamp = next_stamp_++;

        NodeEntry entry;
        entry.node = std::move(node);
        entry.stamp = stamp;

        if (entry.node.domain == nullptr &&
            has_materialized_bounds(entry.node.lower_bounds, entry.node.upper_bounds)) {
            entry.node.domain =
                make_materialized_domain(entry.node.lower_bounds, entry.node.upper_bounds);
        }
        dematerialize_active_node(&entry.node);

        initialize_frontier_root_(entry.node);
        entry.root_domain_changes =
            collect_root_domain_changes_(entry.node.domain, kQueueDomainTol_);
        update_frontier_summary_(entry.root_domain_changes, true);

        const ActiveNode& stored = entry.node;
        local_heap_.push(make_score_key_(handle, stamp, stored.bound, stored, maximize));

        const double estimate = std::isfinite(stored.estimate) ? stored.estimate : stored.bound;
        local_heap_.push(make_score_key_(handle, stamp, estimate, stored, maximize));

        local_heap_.push(
            make_score_key_(handle, stamp, hybrid_node_score(stored), stored, maximize));

        local_heap_.push(make_depth_key_(handle, stamp, stored, strategy, maximize));

        nodes_.emplace(handle, std::move(entry));
    }

    std::optional<ActiveNode> pop(NodeSelectionStrategy strategy, bool maximize,
                                  int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                  std::uint64_t* hybrid_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (nodes_.empty()) {
            return std::nullopt;
        }

        std::optional<ActiveNode> result;
        if (strategy == NodeSelectionStrategy::DepthFirst) {
            result = extract_valid_lifo_node_();
        } else if (strategy == NodeSelectionStrategy::BestBound) {
            result = extract_valid_node_(local_heap_);
        } else if (strategy == NodeSelectionStrategy::BestEstimate) {
            result = extract_valid_node_(local_heap_);
        } else if (strategy == NodeSelectionStrategy::Hybrid) {
            bool use_depth = true;
            if (hybrid_counter != nullptr) {
                ++(*hybrid_counter);
                use_depth =
                    ((*hybrid_counter) % static_cast<std::uint64_t>(hybrid_depth_bias + 1)) != 0;
            }
            result =
                use_depth ? extract_valid_node_(local_heap_) : extract_valid_node_(local_heap_);
        } else if (is_plunging_strategy(strategy)) {
            bool use_depth = true;
            if (hybrid_counter != nullptr) {
                ++(*hybrid_counter);
                use_depth =
                    ((*hybrid_counter) % static_cast<std::uint64_t>(hybrid_depth_bias + 1)) != 0;
            }
            if (use_depth) {
                result = extract_valid_node_(local_heap_);
            } else {
                bool use_best_bound = strategy == NodeSelectionStrategy::BestFirstPlunging;
                if (strategy == NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging &&
                    hybrid_counter != nullptr && plunging_bestfreq > 0) {
                    use_best_bound =
                        ((*hybrid_counter) % static_cast<std::uint64_t>(plunging_bestfreq)) == 0;
                }
                result = use_best_bound ? extract_valid_node_(local_heap_)
                                        : extract_valid_node_(local_heap_);
            }
        } else {
            result = extract_valid_node_(local_heap_);
        }

        if (result.has_value()) {
            update_frontier_summary_(collect_root_domain_changes_(result->domain, kQueueDomainTol_),
                                     false);
        }

        return result;
    }

    std::optional<ActiveNode> steal(NodeSelectionStrategy strategy, bool maximize,
                                    int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                    std::uint64_t* hybrid_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::optional<ActiveNode> result = extract_valid_node_(local_heap_);
        if (result.has_value()) {
            update_frontier_summary_(collect_root_domain_changes_(result->domain, kQueueDomainTol_),
                                     false);
            return result;
        }

        result = extract_valid_node_(stealing_heap_);
        if (result.has_value()) {
            update_frontier_summary_(collect_root_domain_changes_(result->domain, kQueueDomainTol_),
                                     false);
        }
        return result;
    }

    void contribute_to_stealing_heap(ActiveNode node, NodeSelectionStrategy strategy,
                                     bool maximize) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int handle = next_handle_++;
        const std::uint64_t stamp = next_stamp_++;

        StealKey key;
        key.handle = handle;
        key.stamp = stamp;
        const double score = node_selection_score(node, strategy);
        key.score = maximize ? score : -score;
        key.depth = node.depth;
        key.domain_change_count = node.domain_change_count;
        key.order = node.order;

        NodeEntry entry;
        entry.node = std::move(node);
        entry.stamp = stamp;
        if (entry.node.domain == nullptr &&
            has_materialized_bounds(entry.node.lower_bounds, entry.node.upper_bounds)) {
            entry.node.domain =
                make_materialized_domain(entry.node.lower_bounds, entry.node.upper_bounds);
        }
        dematerialize_active_node(&entry.node);

        initialize_frontier_root_(entry.node);
        entry.root_domain_changes =
            collect_root_domain_changes_(entry.node.domain, kQueueDomainTol_);
        update_frontier_summary_(entry.root_domain_changes, true);

        nodes_.emplace(handle, std::move(entry));
        stealing_heap_.push(key);
    }

    int stealing_heap_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(stealing_heap_.size());
    }

    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        nodes_.clear();
        while (!local_heap_.empty()) {
            local_heap_.pop();
        }
        while (!stealing_heap_.empty()) {
            stealing_heap_.pop();
        }
        next_handle_ = 0;
        next_stamp_ = 1;

        frontier_root_lower_bounds_.reset();
        frontier_root_upper_bounds_.reset();
        frontier_root_domain_.reset();
        lower_changed_counts_.clear();
        upper_changed_counts_.clear();
        lower_changed_values_.clear();
        upper_changed_values_.clear();
    }

    double compute_best_bound(bool has_incumbent, double incumbent_objective, bool maximize,
                              const std::optional<double>& root_relaxation_objective) const {
        std::lock_guard<std::mutex> lock(mutex_);
        double best =
            has_incumbent ? incumbent_objective : std::numeric_limits<double>::quiet_NaN();

        const ActiveNode* best_local_node = peek_valid_node_(local_heap_);
        if (best_local_node != nullptr) {
            if (!std::isfinite(best)) {
                best = best_local_node->bound;
            } else {
                best = maximize ? std::max(best, best_local_node->bound)
                                : std::min(best, best_local_node->bound);
            }
        }

        if (!std::isfinite(best) && root_relaxation_objective.has_value()) {
            best = *root_relaxation_objective;
        }
        return best;
    }

  private:
    static void resolve_root_bounds_(const std::shared_ptr<const NodeDomain>& domain,
                                     std::shared_ptr<const Eigen::VectorXd>* lower_bounds,
                                     std::shared_ptr<const Eigen::VectorXd>* upper_bounds) {
        if (lower_bounds == nullptr || upper_bounds == nullptr) {
            return;
        }
        lower_bounds->reset();
        upper_bounds->reset();
        if (domain == nullptr) {
            return;
        }
        const NodeDomain* root = domain.get();
        while (root->parent != nullptr) {
            root = root->parent.get();
        }
        if (root->root_lower_bounds == nullptr || root->root_upper_bounds == nullptr) {
            return;
        }
        *lower_bounds = root->root_lower_bounds;
        *upper_bounds = root->root_upper_bounds;
    }

    static std::vector<DomainChange>
    collect_root_domain_changes_(const std::shared_ptr<const NodeDomain>& domain, double tol) {
        if (domain == nullptr) {
            return {};
        }

        std::shared_ptr<const Eigen::VectorXd> root_lower_bounds;
        std::shared_ptr<const Eigen::VectorXd> root_upper_bounds;
        resolve_root_bounds_(domain, &root_lower_bounds, &root_upper_bounds);
        if (root_lower_bounds == nullptr || root_upper_bounds == nullptr) {
            return {};
        }

        std::vector<const NodeDomain*> chain;
        for (const NodeDomain* current = domain.get(); current != nullptr;
             current = current->parent.get()) {
            chain.push_back(current);
        }

        std::unordered_map<int, DomainChange> merged_changes;
        merged_changes.reserve(static_cast<std::size_t>(domain->total_change_count + 1));
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            const NodeDomain* current = *it;
            for (const DomainChange& change : current->changes) {
                if (change.variable < 0 || change.variable >= root_lower_bounds->size() ||
                    change.variable >= root_upper_bounds->size()) {
                    continue;
                }
                const auto existing = merged_changes.find(change.variable);
                if (existing == merged_changes.end() ||
                    change.lower_bound > existing->second.lower_bound ||
                    change.upper_bound < existing->second.upper_bound) {
                    merged_changes[change.variable] = change;
                }
            }
        }

        std::vector<DomainChange> result;
        result.reserve(merged_changes.size());
        for (auto& change_entry : merged_changes) {
            const DomainChange& change = change_entry.second;
            const double root_lower = (*root_lower_bounds)(change.variable);
            const double root_upper = (*root_upper_bounds)(change.variable);
            const bool lower_changed = change.lower_bound > root_lower + tol;
            const bool upper_changed = change.upper_bound < root_upper - tol;
            if (lower_changed || upper_changed) {
                result.push_back(change);
            }
        }
        return result;
    }

    void initialize_frontier_root_(const ActiveNode& node) {
        if (frontier_root_lower_bounds_ != nullptr && frontier_root_upper_bounds_ != nullptr &&
            frontier_root_domain_ != nullptr) {
            return;
        }

        resolve_root_bounds_(node.domain, &frontier_root_lower_bounds_,
                             &frontier_root_upper_bounds_);
        if (frontier_root_lower_bounds_ == nullptr || frontier_root_upper_bounds_ == nullptr) {
            return;
        }
        frontier_root_domain_ =
            make_materialized_domain(*frontier_root_lower_bounds_, *frontier_root_upper_bounds_);
        lower_changed_counts_.assign(frontier_root_lower_bounds_->size(), 0);
        upper_changed_counts_.assign(frontier_root_upper_bounds_->size(), 0);
    }

    void update_frontier_summary_(const std::vector<DomainChange>& changes, bool add) {
        if (frontier_root_lower_bounds_ == nullptr || frontier_root_upper_bounds_ == nullptr) {
            return;
        }
        for (const DomainChange& change : changes) {
            if (change.variable < 0 || change.variable >= frontier_root_lower_bounds_->size() ||
                change.variable >= frontier_root_upper_bounds_->size()) {
                continue;
            }
            const double root_lower = (*frontier_root_lower_bounds_)(change.variable);
            const double root_upper = (*frontier_root_upper_bounds_)(change.variable);
            const bool lower_changed = change.lower_bound > root_lower + kQueueDomainTol_;
            const bool upper_changed = change.upper_bound < root_upper - kQueueDomainTol_;
            if (lower_changed) {
                if (add) {
                    ++lower_changed_counts_[change.variable];
                    lower_changed_values_[change.variable][change.lower_bound] += 1;
                } else if (lower_changed_counts_[change.variable] > 0) {
                    --lower_changed_counts_[change.variable];
                    auto map_it = lower_changed_values_.find(change.variable);
                    if (map_it != lower_changed_values_.end()) {
                        auto value_it = map_it->second.find(change.lower_bound);
                        if (value_it != map_it->second.end()) {
                            if (--value_it->second == 0) {
                                map_it->second.erase(value_it);
                            }
                        }
                        if (map_it->second.empty()) {
                            lower_changed_values_.erase(map_it);
                        }
                    }
                }
            }
            if (upper_changed) {
                if (add) {
                    ++upper_changed_counts_[change.variable];
                    upper_changed_values_[change.variable][change.upper_bound] += 1;
                } else if (upper_changed_counts_[change.variable] > 0) {
                    --upper_changed_counts_[change.variable];
                    auto map_it = upper_changed_values_.find(change.variable);
                    if (map_it != upper_changed_values_.end()) {
                        auto value_it = map_it->second.find(change.upper_bound);
                        if (value_it != map_it->second.end()) {
                            if (--value_it->second == 0) {
                                map_it->second.erase(value_it);
                            }
                        }
                        if (map_it->second.empty()) {
                            upper_changed_values_.erase(map_it);
                        }
                    }
                }
            }
        }
    }

    static QueueKey make_score_key_(int handle, std::uint64_t stamp, double score,
                                    const ActiveNode& node, bool maximize) {
        QueueKey key;
        key.handle = handle;
        key.stamp = stamp;
        key.normalized_score = maximize ? score : -score;
        key.depth = node.depth;
        key.domain_change_count = node.domain_change_count;
        key.order = node.order;
        return key;
    }

    static QueueKey make_depth_key_(int handle, std::uint64_t stamp, const ActiveNode& node,
                                    NodeSelectionStrategy strategy, bool maximize) {
        QueueKey key;
        key.handle = handle;
        key.stamp = stamp;
        key.normalized_score =
            maximize ? node_selection_score(node, strategy) : -node_selection_score(node, strategy);
        key.depth = node.depth;
        key.domain_change_count = node.domain_change_count;
        key.order = node.order;
        return key;
    }

    template <typename Heap> const ActiveNode* peek_valid_node_(const Heap& heap) const {
        Heap scratch = heap;
        while (!scratch.empty()) {
            const auto key = scratch.top();
            scratch.pop();
            auto it = nodes_.find(key.handle);
            if (it == nodes_.end() || it->second.stamp != key.stamp) {
                continue;
            }
            return &it->second.node;
        }
        return nullptr;
    }

    template <typename Heap> std::optional<ActiveNode> extract_valid_node_(Heap& heap) {
        while (!heap.empty()) {
            const auto key = heap.top();
            heap.pop();

            auto it = nodes_.find(key.handle);
            if (it == nodes_.end() || it->second.stamp != key.stamp) {
                continue;
            }

            ActiveNode node = std::move(it->second.node);
            nodes_.erase(it);
            return node;
        }
        return std::nullopt;
    }

    std::optional<ActiveNode> extract_valid_lifo_node_() {
        while (!nodes_.empty()) {
            auto it = nodes_.begin();
            ActiveNode node = std::move(it->second.node);
            nodes_.erase(it);
            return node;
        }
        return std::nullopt;
    }

    const ActiveNode* peek_valid_node() const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto scratch = local_heap_;
        while (!scratch.empty()) {
            QueueKey key = scratch.top();
            scratch.pop();
            auto it = nodes_.find(key.handle);
            if (it == nodes_.end() || it->second.stamp != key.stamp) {
                continue;
            }
            return &it->second.node;
        }
        auto scratch_steal = stealing_heap_;
        while (!scratch_steal.empty()) {
            StealKey key = scratch_steal.top();
            scratch_steal.pop();
            auto it = nodes_.find(key.handle);
            if (it == nodes_.end() || it->second.stamp != key.stamp) {
                continue;
            }
            return &it->second.node;
        }
        return nullptr;
    }

    std::shared_ptr<const Eigen::VectorXd> frontier_root_lower_bounds_;
    std::shared_ptr<const Eigen::VectorXd> frontier_root_upper_bounds_;
    std::shared_ptr<const NodeDomain> frontier_root_domain_;
    std::vector<int> lower_changed_counts_;
    std::vector<int> upper_changed_counts_;
    std::unordered_map<int, std::map<double, int>> lower_changed_values_;
    std::unordered_map<int, std::map<double, int>> upper_changed_values_;
};

// ============================================================================
// Worker Local Queue (finalizing OpenNodeQueue for backward compatibility)
// ============================================================================
// The original OpenNodeQueue is kept for backward compatibility but will be
// gradually replaced by WorkerLocalQueue in new code.
// ============================================================================

} // namespace simplex::bnb::detail
