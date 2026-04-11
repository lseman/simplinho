#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "bnb/conflict_graph.h"
#include "bnb/search.h"

namespace simplex::bnb::detail {

class ImplicationStore {
  public:
    void reset(int literal_count) {
        if (literal_count < 0) {
            literal_count = 0;
        }
        implications_.assign(static_cast<std::size_t>(literal_count), {});
    }

    int literal_count() const { return static_cast<int>(implications_.size()); }

    const std::vector<ReasonLiteral>& consequences(int literal) const {
        static const std::vector<ReasonLiteral> kEmpty;
        if (literal < 0 || literal >= static_cast<int>(implications_.size())) {
            return kEmpty;
        }
        return implications_[literal];
    }

    bool learn(int trigger_literal, const ReasonLiteral& consequence, double tol,
               std::size_t max_per_literal = 64) {
        if (trigger_literal < 0 || trigger_literal >= static_cast<int>(implications_.size()) ||
            consequence.variable < 0 || !std::isfinite(consequence.value)) {
            return false;
        }

        std::vector<ReasonLiteral>& stored = implications_[trigger_literal];
        for (ReasonLiteral& existing : stored) {
            if (existing.variable != consequence.variable ||
                existing.is_lower != consequence.is_lower) {
                continue;
            }
            if (consequence.is_lower) {
                if (consequence.value <= existing.value + tol) {
                    return false;
                }
                existing.value = consequence.value;
                return true;
            }
            if (consequence.value >= existing.value - tol) {
                return false;
            }
            existing.value = consequence.value;
            return true;
        }

        stored.push_back(consequence);
        if (stored.size() > max_per_literal) {
            stored.erase(stored.begin(), stored.begin() + static_cast<std::ptrdiff_t>(
                                                              stored.size() - max_per_literal));
        }
        return true;
    }

  private:
    std::vector<std::vector<ReasonLiteral>> implications_;
};

struct NodePropagationState {
    struct RowActivitySide {
        std::vector<double> min_contributions;
        double min_finite_sum = 0.0;
        int min_nonfinite = 0;
        bool initialized = false;
    };

    struct RowActivity {
        RowActivitySide positive;
        RowActivitySide negative;
    };

    std::vector<char> queued_literals;
    std::vector<int> literal_queue;
    int literal_queue_head = 0;

    std::vector<char> row_queued;
    std::vector<int> row_queue;
    int row_queue_head = 0;

    std::vector<int> changed_variables;
    std::vector<int> graph_changed_variables;
    std::vector<ReasonLiteral> row_conflict_literals;
    std::vector<RowActivity> row_activity;

    std::shared_ptr<NodeReasonStore> reasons;

    void reset(const ConflictGraph* graph, int row_count,
               const std::shared_ptr<NodeReasonStore>& initial_reasons = nullptr) {
        literal_queue.clear();
        literal_queue_head = 0;
        row_queue.clear();
        row_queue_head = 0;
        changed_variables.clear();
        graph_changed_variables.clear();
        row_conflict_literals.clear();

        if (graph != nullptr) {
            queued_literals.assign(static_cast<std::size_t>(graph->literal_count()), 0);
            if (initial_reasons != nullptr &&
                static_cast<int>(initial_reasons->size()) == graph->literal_count()) {
                reasons = initial_reasons;
            } else {
                reasons = std::make_shared<NodeReasonStore>(graph->literal_count());
            }
            literal_queue.reserve(std::max(4, graph->literal_count() / 4));
        } else {
            queued_literals.clear();
            reasons.reset();
        }

        row_queued.assign(static_cast<std::size_t>(std::max(0, row_count)), 0);
        row_queue.reserve(std::max(0, row_count));
        row_activity.assign(static_cast<std::size_t>(std::max(0, row_count)), {});
    }

    void seed_all_rows(int row_count) {
        row_queue.clear();
        row_queue_head = 0;
        row_queued.assign(static_cast<std::size_t>(std::max(0, row_count)), 1);
        row_queue.reserve(std::max(0, row_count));
        for (int row_index = 0; row_index < row_count; ++row_index) {
            row_queue.push_back(row_index);
        }
    }

    void enqueue_row(int row_index) {
        if (row_index < 0 || row_index >= static_cast<int>(row_queued.size()) ||
            row_queued[row_index]) {
            return;
        }
        row_queued[row_index] = 1;
        row_queue.push_back(row_index);
    }
};

} // namespace simplex::bnb::detail
