#pragma once

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
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
};

using NodeReasonStore = std::vector<PropagationReason>;

struct ActiveNode {
    int id = -1;
    int parent_id = -1;
    int depth = 0;
    std::uint64_t order = 0;
    double bound = std::numeric_limits<double>::quiet_NaN();
    double estimate = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    std::optional<LPBasis> basis;
    std::shared_ptr<NodeReasonStore> reasons;
};

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
    return rhs_better || (tied && rhs.order < lhs.order);
}

inline bool best_estimate_worse(const ActiveNode& lhs, const ActiveNode& rhs, bool maximize) {
    const double lhs_value = std::isfinite(lhs.estimate) ? lhs.estimate : lhs.bound;
    const double rhs_value = std::isfinite(rhs.estimate) ? rhs.estimate : rhs.bound;
    const bool rhs_better =
        maximize ? (rhs_value > lhs_value + 1e-12) : (rhs_value < lhs_value - 1e-12);
    const bool tied = std::abs(lhs_value - rhs_value) <= 1e-12;
    return rhs_better || (tied && rhs.order < lhs.order);
}

inline void push_active_node(std::vector<ActiveNode>& active_nodes, ActiveNode node,
                             NodeSelectionStrategy strategy, bool maximize) {
    active_nodes.push_back(std::move(node));
    if (strategy == NodeSelectionStrategy::BestBound) {
        std::push_heap(active_nodes.begin(), active_nodes.end(),
                       [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                           return best_bound_worse(lhs, rhs, maximize);
                       });
    } else if (strategy == NodeSelectionStrategy::BestEstimate ||
               strategy == NodeSelectionStrategy::Hybrid) {
        std::push_heap(active_nodes.begin(), active_nodes.end(),
                       [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                           return best_estimate_worse(lhs, rhs, maximize);
                       });
    }
}

inline auto deepest_node_iterator(std::vector<ActiveNode>& active_nodes, bool maximize) {
    return std::max_element(
        active_nodes.begin(), active_nodes.end(),
        [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
            if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
            const double lhs_estimate = std::isfinite(lhs.estimate) ? lhs.estimate : lhs.bound;
            const double rhs_estimate = std::isfinite(rhs.estimate) ? rhs.estimate : rhs.bound;
            const bool rhs_better = maximize ? (rhs_estimate > lhs_estimate + 1e-12)
                                             : (rhs_estimate < lhs_estimate - 1e-12);
            if (rhs_better) return true;
            if (std::abs(lhs_estimate - rhs_estimate) > 1e-12) return false;
            return lhs.order > rhs.order;
        });
}

inline ActiveNode pop_next_node(std::vector<ActiveNode>& active_nodes,
                                NodeSelectionStrategy strategy, bool maximize,
                                int hybrid_depth_bias = 5,
                                std::uint64_t* hybrid_counter = nullptr) {
    if (strategy == NodeSelectionStrategy::DepthFirst) {
        ActiveNode node = std::move(active_nodes.back());
        active_nodes.pop_back();
        return node;
    }
    if (strategy == NodeSelectionStrategy::BestBound) {
        std::pop_heap(active_nodes.begin(), active_nodes.end(),
                      [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                          return best_bound_worse(lhs, rhs, maximize);
                      });
        ActiveNode node = std::move(active_nodes.back());
        active_nodes.pop_back();
        return node;
    }
    if (strategy == NodeSelectionStrategy::BestEstimate) {
        std::pop_heap(active_nodes.begin(), active_nodes.end(),
                      [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                          return best_estimate_worse(lhs, rhs, maximize);
                      });
        ActiveNode node = std::move(active_nodes.back());
        active_nodes.pop_back();
        return node;
    }
    if (strategy == NodeSelectionStrategy::Hybrid) {
        bool use_depth = true;
        if (hybrid_counter != nullptr) {
            ++(*hybrid_counter);
            use_depth = ((*hybrid_counter) % static_cast<std::uint64_t>(hybrid_depth_bias + 1)) !=
                        0;
        }
        if (use_depth) {
            const auto it = deepest_node_iterator(active_nodes, maximize);
            ActiveNode node = std::move(*it);
            active_nodes.erase(it);
            std::make_heap(active_nodes.begin(), active_nodes.end(),
                           [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                               return best_estimate_worse(lhs, rhs, maximize);
                           });
            return node;
        }

        std::pop_heap(active_nodes.begin(), active_nodes.end(),
                      [maximize](const ActiveNode& lhs, const ActiveNode& rhs) {
                          return best_estimate_worse(lhs, rhs, maximize);
                      });
        ActiveNode node = std::move(active_nodes.back());
        active_nodes.pop_back();
        return node;
    }

    ActiveNode node = std::move(active_nodes.front());
    active_nodes.erase(active_nodes.begin());
    return node;
}

inline double compute_best_bound(const std::vector<ActiveNode>& active_nodes,
                                 bool has_incumbent, double incumbent_objective,
                                 bool maximize,
                                 const std::optional<double>& root_relaxation_objective) {
    double best = has_incumbent ? incumbent_objective
                                : std::numeric_limits<double>::quiet_NaN();
    for (const auto& node : active_nodes) {
        if (!std::isfinite(best)) {
            best = node.bound;
        } else {
            best = maximize ? std::max(best, node.bound) : std::min(best, node.bound);
        }
    }
    if (!std::isfinite(best) && root_relaxation_objective.has_value()) {
        best = *root_relaxation_objective;
    }
    return best;
}

}  // namespace simplex::bnb::detail
