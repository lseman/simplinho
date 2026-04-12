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

    WorkerLocalQueue(WorkerLocalQueue&& other) noexcept {
        std::scoped_lock lock(other.mutex_);
        slots_ = std::move(other.slots_);
        free_handles_ = std::move(other.free_handles_);
        lifo_stack_ = std::move(other.lifo_stack_);
        best_heap_ = std::move(other.best_heap_);
        stealing_heap_ = std::move(other.stealing_heap_);
        next_stamp_ = other.next_stamp_;
        active_count_ = other.active_count_;

        frontier_root_lower_bounds_ = std::move(other.frontier_root_lower_bounds_);
        frontier_root_upper_bounds_ = std::move(other.frontier_root_upper_bounds_);
        frontier_root_domain_ = std::move(other.frontier_root_domain_);

        lower_changed_counts_ = std::move(other.lower_changed_counts_);
        upper_changed_counts_ = std::move(other.upper_changed_counts_);
        frontier_max_lower_bounds_ = std::move(other.frontier_max_lower_bounds_);
        frontier_min_upper_bounds_ = std::move(other.frontier_min_upper_bounds_);
        frontier_lower_dirty_ = std::move(other.frontier_lower_dirty_);
        frontier_upper_dirty_ = std::move(other.frontier_upper_dirty_);
    }

    WorkerLocalQueue& operator=(WorkerLocalQueue&& other) noexcept {
        if (this != &other) {
            std::scoped_lock lock(mutex_, other.mutex_);
            slots_ = std::move(other.slots_);
            free_handles_ = std::move(other.free_handles_);
            lifo_stack_ = std::move(other.lifo_stack_);
            best_heap_ = std::move(other.best_heap_);
            stealing_heap_ = std::move(other.stealing_heap_);
            next_stamp_ = other.next_stamp_;
            active_count_ = other.active_count_;

            frontier_root_lower_bounds_ = std::move(other.frontier_root_lower_bounds_);
            frontier_root_upper_bounds_ = std::move(other.frontier_root_upper_bounds_);
            frontier_root_domain_ = std::move(other.frontier_root_domain_);

            lower_changed_counts_ = std::move(other.lower_changed_counts_);
            upper_changed_counts_ = std::move(other.upper_changed_counts_);
            frontier_max_lower_bounds_ = std::move(other.frontier_max_lower_bounds_);
            frontier_min_upper_bounds_ = std::move(other.frontier_min_upper_bounds_);
            frontier_lower_dirty_ = std::move(other.frontier_lower_dirty_);
            frontier_upper_dirty_ = std::move(other.frontier_upper_dirty_);
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

    struct Slot {
        bool alive = false;
        NodeEntry entry;
    };

    struct HandleStamp {
        int handle = -1;
        std::uint64_t stamp = 0;
    };

    struct BestKey {
        int handle = -1;
        std::uint64_t stamp = 0;
        double normalized_score = -std::numeric_limits<double>::infinity();
        int depth = 0;
        int domain_change_count = 0;
        std::uint64_t order = 0;
    };

    struct BestKeyLess {
        bool operator()(const BestKey& lhs, const BestKey& rhs) const noexcept {
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

    struct StealKey {
        int handle = -1;
        std::uint64_t stamp = 0;
        double normalized_score = -std::numeric_limits<double>::infinity();
        int depth = 0;
        int domain_change_count = 0;
        std::uint64_t order = 0;
    };

    struct StealKeyLess {
        bool operator()(const StealKey& lhs, const StealKey& rhs) const noexcept {
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
                return lhs.order < rhs.order;
            }
            return lhs.stamp < rhs.stamp;
        }
    };

    std::vector<Slot> slots_;
    std::vector<int> free_handles_;
    std::vector<HandleStamp> lifo_stack_;
    std::priority_queue<BestKey, std::vector<BestKey>, BestKeyLess> best_heap_;
    std::priority_queue<StealKey, std::vector<StealKey>, StealKeyLess> stealing_heap_;
    mutable std::mutex mutex_;

    std::uint64_t next_stamp_ = 1;
    int active_count_ = 0;

  public:
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_count_ == 0;
    }

    int size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_count_;
    }

    template <typename Fn> void for_each_mutable(Fn&& fn) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& slot : slots_) {
            if (slot.alive) {
                fn(slot.entry.node);
            }
        }
    }

    void push(ActiveNode node, NodeSelectionStrategy strategy, bool maximize) {
        NodeEntry entry;
        entry.node = std::move(node);

        if (entry.node.domain == nullptr &&
            has_materialized_bounds(entry.node.lower_bounds, entry.node.upper_bounds)) {
            entry.node.domain =
                make_materialized_domain(entry.node.lower_bounds, entry.node.upper_bounds);
        }
        dematerialize_active_node(&entry.node);

        std::shared_ptr<const Eigen::VectorXd> init_lower;
        std::shared_ptr<const Eigen::VectorXd> init_upper;
        resolve_root_bounds_(entry.node.domain, &init_lower, &init_upper);

        entry.root_domain_changes =
            collect_root_domain_changes_(entry.node.domain, kQueueDomainTol_);

        std::lock_guard<std::mutex> lock(mutex_);

        initialize_frontier_root_locked_(init_lower, init_upper);

        const int handle = allocate_handle_locked_();
        const std::uint64_t stamp = next_stamp_++;
        entry.stamp = stamp;

        update_frontier_summary_locked_(entry.root_domain_changes, true);

        slots_[handle].alive = true;
        slots_[handle].entry = std::move(entry);
        ++active_count_;

        const ActiveNode& stored = slots_[handle].entry.node;
        lifo_stack_.push_back(HandleStamp{handle, stamp});
        best_heap_.push(make_best_key_(handle, stamp, stored, strategy, maximize));
    }

    std::optional<ActiveNode> pop(NodeSelectionStrategy strategy, bool maximize,
                                  int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                  std::uint64_t* hybrid_counter = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_count_ == 0) {
            return std::nullopt;
        }

        const bool prefer_depth =
            should_prefer_depth_(strategy, hybrid_depth_bias, plunging_bestfreq, hybrid_counter);

        if (strategy == NodeSelectionStrategy::DepthFirst) {
            return extract_valid_lifo_node_locked_();
        }

        if (prefer_depth && is_plunging_strategy(strategy)) {
            if (auto node = extract_valid_lifo_node_locked_()) {
                return node;
            }
            return extract_valid_best_node_locked_();
        }

        if (auto node = extract_valid_best_node_locked_()) {
            return node;
        }

        if (is_plunging_strategy(strategy)) {
            return extract_valid_lifo_node_locked_();
        }

        return std::nullopt;
    }

    std::optional<ActiveNode> steal(NodeSelectionStrategy strategy, bool maximize,
                                    int hybrid_depth_bias = 5, int plunging_bestfreq = 10,
                                    std::uint64_t* hybrid_counter = nullptr) {
        (void)strategy;
        (void)maximize;
        (void)hybrid_depth_bias;
        (void)plunging_bestfreq;
        (void)hybrid_counter;

        std::lock_guard<std::mutex> lock(mutex_);
        if (active_count_ == 0) {
            return std::nullopt;
        }

        if (auto node = extract_valid_steal_node_locked_()) {
            return node;
        }
        if (auto node = extract_valid_best_node_locked_()) {
            return node;
        }
        return extract_valid_lifo_node_locked_();
    }

    void contribute_to_stealing_heap(ActiveNode node, NodeSelectionStrategy strategy,
                                     bool maximize) {
        NodeEntry entry;
        entry.node = std::move(node);

        if (entry.node.domain == nullptr &&
            has_materialized_bounds(entry.node.lower_bounds, entry.node.upper_bounds)) {
            entry.node.domain =
                make_materialized_domain(entry.node.lower_bounds, entry.node.upper_bounds);
        }
        dematerialize_active_node(&entry.node);

        std::shared_ptr<const Eigen::VectorXd> init_lower;
        std::shared_ptr<const Eigen::VectorXd> init_upper;
        resolve_root_bounds_(entry.node.domain, &init_lower, &init_upper);

        entry.root_domain_changes =
            collect_root_domain_changes_(entry.node.domain, kQueueDomainTol_);

        std::lock_guard<std::mutex> lock(mutex_);

        initialize_frontier_root_locked_(init_lower, init_upper);

        const int handle = allocate_handle_locked_();
        const std::uint64_t stamp = next_stamp_++;
        entry.stamp = stamp;

        update_frontier_summary_locked_(entry.root_domain_changes, true);

        slots_[handle].alive = true;
        slots_[handle].entry = std::move(entry);
        ++active_count_;

        const ActiveNode& stored = slots_[handle].entry.node;
        stealing_heap_.push(make_steal_key_(handle, stamp, stored, strategy, maximize));
    }

    int stealing_heap_size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(stealing_heap_.size());
    }

    void clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex_);

        slots_.clear();
        free_handles_.clear();
        lifo_stack_.clear();
        while (!best_heap_.empty()) {
            best_heap_.pop();
        }
        while (!stealing_heap_.empty()) {
            stealing_heap_.pop();
        }

        next_stamp_ = 1;
        active_count_ = 0;

        frontier_root_lower_bounds_.reset();
        frontier_root_upper_bounds_.reset();
        frontier_root_domain_.reset();

        lower_changed_counts_.clear();
        upper_changed_counts_.clear();
        frontier_max_lower_bounds_.clear();
        frontier_min_upper_bounds_.clear();
        frontier_lower_dirty_.clear();
        frontier_upper_dirty_.clear();
    }

    const ActiveNode* peek_valid_node() const {
        std::lock_guard<std::mutex> lock(mutex_);

        const ActiveNode* best = peek_valid_best_node_locked_();
        if (best != nullptr) {
            return best;
        }
        return peek_valid_steal_node_locked_();
    }

    double compute_best_bound(bool has_incumbent, double incumbent_objective, bool maximize,
                              const std::optional<double>& root_relaxation_objective) const {
        std::lock_guard<std::mutex> lock(mutex_);

        double best =
            has_incumbent ? incumbent_objective : std::numeric_limits<double>::quiet_NaN();

        if (const ActiveNode* best_local_node = peek_valid_best_node_locked_()) {
            if (!std::isfinite(best)) {
                best = best_local_node->bound;
            } else {
                best = maximize ? std::max(best, best_local_node->bound)
                                : std::min(best, best_local_node->bound);
            }
        }

        if (const ActiveNode* best_steal_node = peek_valid_steal_node_locked_()) {
            if (!std::isfinite(best)) {
                best = best_steal_node->bound;
            } else {
                best = maximize ? std::max(best, best_steal_node->bound)
                                : std::min(best, best_steal_node->bound);
            }
        }

        if (!std::isfinite(best) && root_relaxation_objective.has_value()) {
            best = *root_relaxation_objective;
        }
        return best;
    }

  private:
    int allocate_handle_locked_() {
        if (!free_handles_.empty()) {
            const int handle = free_handles_.back();
            free_handles_.pop_back();
            return handle;
        }
        slots_.push_back(Slot{});
        return static_cast<int>(slots_.size()) - 1;
    }

    void release_handle_locked_(int handle) {
        if (handle >= 0) {
            free_handles_.push_back(handle);
        }
    }

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
                auto found = merged_changes.find(change.variable);
                if (found == merged_changes.end() ||
                    change.lower_bound > found->second.lower_bound ||
                    change.upper_bound < found->second.upper_bound) {
                    merged_changes[change.variable] = change;
                }
            }
        }

        std::vector<DomainChange> result;
        result.reserve(merged_changes.size());
        for (auto& kv : merged_changes) {
            const DomainChange& change = kv.second;
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

    void initialize_frontier_root_locked_(const std::shared_ptr<const Eigen::VectorXd>& lower,
                                          const std::shared_ptr<const Eigen::VectorXd>& upper) {
        if (frontier_root_lower_bounds_ != nullptr && frontier_root_upper_bounds_ != nullptr &&
            frontier_root_domain_ != nullptr) {
            return;
        }
        if (lower == nullptr || upper == nullptr) {
            return;
        }

        frontier_root_lower_bounds_ = lower;
        frontier_root_upper_bounds_ = upper;
        frontier_root_domain_ =
            make_materialized_domain(*frontier_root_lower_bounds_, *frontier_root_upper_bounds_);

        const int n =
            std::min<int>(frontier_root_lower_bounds_->size(), frontier_root_upper_bounds_->size());

        lower_changed_counts_.assign(n, 0);
        upper_changed_counts_.assign(n, 0);
        frontier_max_lower_bounds_.assign(n, -std::numeric_limits<double>::infinity());
        frontier_min_upper_bounds_.assign(n, std::numeric_limits<double>::infinity());
        frontier_lower_dirty_.assign(n, false);
        frontier_upper_dirty_.assign(n, false);
    }

    void update_frontier_summary_locked_(const std::vector<DomainChange>& changes, bool add) {
        if (frontier_root_lower_bounds_ == nullptr || frontier_root_upper_bounds_ == nullptr) {
            return;
        }

        for (const DomainChange& change : changes) {
            if (change.variable < 0 || change.variable >= frontier_root_lower_bounds_->size() ||
                change.variable >= frontier_root_upper_bounds_->size()) {
                continue;
            }

            const int j = change.variable;
            const double root_lower = (*frontier_root_lower_bounds_)(j);
            const double root_upper = (*frontier_root_upper_bounds_)(j);

            const bool lower_changed = change.lower_bound > root_lower + kQueueDomainTol_;
            const bool upper_changed = change.upper_bound < root_upper - kQueueDomainTol_;

            if (lower_changed) {
                if (add) {
                    ++lower_changed_counts_[j];
                    frontier_max_lower_bounds_[j] =
                        std::max(frontier_max_lower_bounds_[j], change.lower_bound);
                } else if (lower_changed_counts_[j] > 0) {
                    --lower_changed_counts_[j];
                    if (lower_changed_counts_[j] == 0) {
                        frontier_max_lower_bounds_[j] = -std::numeric_limits<double>::infinity();
                        frontier_lower_dirty_[j] = false;
                    } else if (std::abs(change.lower_bound - frontier_max_lower_bounds_[j]) <=
                               kQueueDomainTol_) {
                        frontier_lower_dirty_[j] = true;
                    }
                }
            }

            if (upper_changed) {
                if (add) {
                    ++upper_changed_counts_[j];
                    frontier_min_upper_bounds_[j] =
                        std::min(frontier_min_upper_bounds_[j], change.upper_bound);
                } else if (upper_changed_counts_[j] > 0) {
                    --upper_changed_counts_[j];
                    if (upper_changed_counts_[j] == 0) {
                        frontier_min_upper_bounds_[j] = std::numeric_limits<double>::infinity();
                        frontier_upper_dirty_[j] = false;
                    } else if (std::abs(change.upper_bound - frontier_min_upper_bounds_[j]) <=
                               kQueueDomainTol_) {
                        frontier_upper_dirty_[j] = true;
                    }
                }
            }
        }
    }

    void rebuild_frontier_extrema_locked_(int variable) const {
        if (frontier_root_lower_bounds_ == nullptr || frontier_root_upper_bounds_ == nullptr) {
            return;
        }
        if (variable < 0 || variable >= frontier_root_lower_bounds_->size() ||
            variable >= frontier_root_upper_bounds_->size()) {
            return;
        }

        const double root_lower = (*frontier_root_lower_bounds_)(variable);
        const double root_upper = (*frontier_root_upper_bounds_)(variable);

        double best_lower = -std::numeric_limits<double>::infinity();
        double best_upper = std::numeric_limits<double>::infinity();

        for (const auto& slot : slots_) {
            if (!slot.alive) {
                continue;
            }
            for (const DomainChange& change : slot.entry.root_domain_changes) {
                if (change.variable != variable) {
                    continue;
                }
                if (change.lower_bound > root_lower + kQueueDomainTol_) {
                    best_lower = std::max(best_lower, change.lower_bound);
                }
                if (change.upper_bound < root_upper - kQueueDomainTol_) {
                    best_upper = std::min(best_upper, change.upper_bound);
                }
            }
        }

        frontier_max_lower_bounds_[variable] = best_lower;
        frontier_min_upper_bounds_[variable] = best_upper;
        frontier_lower_dirty_[variable] = false;
        frontier_upper_dirty_[variable] = false;
    }

    double frontier_max_lower_bound_locked_(int variable) const {
        if (variable < 0 || variable >= static_cast<int>(frontier_max_lower_bounds_.size())) {
            return -std::numeric_limits<double>::infinity();
        }
        if (frontier_lower_dirty_[variable]) {
            rebuild_frontier_extrema_locked_(variable);
        }
        return frontier_max_lower_bounds_[variable];
    }

    double frontier_min_upper_bound_locked_(int variable) const {
        if (variable < 0 || variable >= static_cast<int>(frontier_min_upper_bounds_.size())) {
            return std::numeric_limits<double>::infinity();
        }
        if (frontier_upper_dirty_[variable]) {
            rebuild_frontier_extrema_locked_(variable);
        }
        return frontier_min_upper_bounds_[variable];
    }

    static bool should_prefer_depth_(NodeSelectionStrategy strategy, int hybrid_depth_bias,
                                     int plunging_bestfreq, std::uint64_t* hybrid_counter) {
        if (!is_plunging_strategy(strategy)) {
            return false;
        }

        if (hybrid_counter == nullptr) {
            return true;
        }

        ++(*hybrid_counter);

        if (strategy == NodeSelectionStrategy::InterleavedBestFirstBestEstimatePlunging) {
            if (plunging_bestfreq <= 0) {
                return true;
            }
            return ((*hybrid_counter) % static_cast<std::uint64_t>(plunging_bestfreq)) != 0;
        }

        return ((*hybrid_counter) % static_cast<std::uint64_t>(hybrid_depth_bias + 1)) != 0;
    }

    static double score_for_strategy_(const ActiveNode& node, NodeSelectionStrategy strategy) {
        if (strategy == NodeSelectionStrategy::BestBound ||
            strategy == NodeSelectionStrategy::BestFirstPlunging) {
            return node.bound;
        }
        if (strategy == NodeSelectionStrategy::Hybrid) {
            return hybrid_node_score(node);
        }
        return std::isfinite(node.estimate) ? node.estimate : node.bound;
    }

    static BestKey make_best_key_(int handle, std::uint64_t stamp, const ActiveNode& node,
                                  NodeSelectionStrategy strategy, bool maximize) {
        BestKey key;
        key.handle = handle;
        key.stamp = stamp;
        const double score = score_for_strategy_(node, strategy);
        key.normalized_score = maximize ? score : -score;
        key.depth = node.depth;
        key.domain_change_count = node.domain_change_count;
        key.order = node.order;
        return key;
    }

    static StealKey make_steal_key_(int handle, std::uint64_t stamp, const ActiveNode& node,
                                    NodeSelectionStrategy strategy, bool maximize) {
        StealKey key;
        key.handle = handle;
        key.stamp = stamp;
        const double score = score_for_strategy_(node, strategy);
        key.normalized_score = maximize ? score : -score;
        key.depth = node.depth;
        key.domain_change_count = node.domain_change_count;
        key.order = node.order;
        return key;
    }

    bool is_valid_locked_(int handle, std::uint64_t stamp) const {
        return handle >= 0 && handle < static_cast<int>(slots_.size()) && slots_[handle].alive &&
               slots_[handle].entry.stamp == stamp;
    }

    std::optional<ActiveNode> remove_slot_locked_(int handle) {
        if (handle < 0 || handle >= static_cast<int>(slots_.size()) || !slots_[handle].alive) {
            return std::nullopt;
        }

        Slot& slot = slots_[handle];
        update_frontier_summary_locked_(slot.entry.root_domain_changes, false);

        ActiveNode result = std::move(slot.entry.node);
        slot.alive = false;
        slot.entry = NodeEntry{};
        --active_count_;
        release_handle_locked_(handle);
        return result;
    }

    std::optional<ActiveNode> extract_valid_lifo_node_locked_() {
        while (!lifo_stack_.empty()) {
            const HandleStamp hs = lifo_stack_.back();
            lifo_stack_.pop_back();
            if (!is_valid_locked_(hs.handle, hs.stamp)) {
                continue;
            }
            return remove_slot_locked_(hs.handle);
        }
        return std::nullopt;
    }

    std::optional<ActiveNode> extract_valid_best_node_locked_() {
        while (!best_heap_.empty()) {
            const BestKey key = best_heap_.top();
            best_heap_.pop();
            if (!is_valid_locked_(key.handle, key.stamp)) {
                continue;
            }
            return remove_slot_locked_(key.handle);
        }
        return std::nullopt;
    }

    std::optional<ActiveNode> extract_valid_steal_node_locked_() {
        while (!stealing_heap_.empty()) {
            const StealKey key = stealing_heap_.top();
            stealing_heap_.pop();
            if (!is_valid_locked_(key.handle, key.stamp)) {
                continue;
            }
            return remove_slot_locked_(key.handle);
        }
        return std::nullopt;
    }

    const ActiveNode* peek_valid_best_node_locked_() const {
        auto& heap = const_cast<std::priority_queue<BestKey, std::vector<BestKey>, BestKeyLess>&>(
            best_heap_);
        while (!heap.empty()) {
            const BestKey& key = heap.top();
            if (!is_valid_locked_(key.handle, key.stamp)) {
                heap.pop();
                continue;
            }
            return &slots_[key.handle].entry.node;
        }
        return nullptr;
    }

    const ActiveNode* peek_valid_steal_node_locked_() const {
        auto& heap =
            const_cast<std::priority_queue<StealKey, std::vector<StealKey>, StealKeyLess>&>(
                stealing_heap_);
        while (!heap.empty()) {
            const StealKey& key = heap.top();
            if (!is_valid_locked_(key.handle, key.stamp)) {
                heap.pop();
                continue;
            }
            return &slots_[key.handle].entry.node;
        }
        return nullptr;
    }

    std::shared_ptr<const Eigen::VectorXd> frontier_root_lower_bounds_;
    std::shared_ptr<const Eigen::VectorXd> frontier_root_upper_bounds_;
    std::shared_ptr<const NodeDomain> frontier_root_domain_;

    std::vector<int> lower_changed_counts_;
    std::vector<int> upper_changed_counts_;

    mutable std::vector<double> frontier_max_lower_bounds_;
    mutable std::vector<double> frontier_min_upper_bounds_;
    mutable std::vector<bool> frontier_lower_dirty_;
    mutable std::vector<bool> frontier_upper_dirty_;
};
// ============================================================================
// Worker Local Queue (finalizing OpenNodeQueue for backward compatibility)
// ============================================================================
// The original OpenNodeQueue is kept for backward compatibility but will be
// gradually replaced by WorkerLocalQueue in new code.
// ============================================================================

} // namespace simplex::bnb::detail
