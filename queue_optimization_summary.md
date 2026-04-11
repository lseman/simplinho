# Queue Optimization Summary

## Executive Summary

The `WorkerLocalQueue` implementation has significant opportunities for improvement based on comparison with Highs' `HighsSplitDeque`. The main issues are:

1. **Redundant heap entries**: Each node is stored 4 times in separate priority queues
2. **Unbounded memory growth**: Uses `std::unordered_map` without size limits
3. **Poor cache utilization**: No explicit cache-line alignment
4. **Mutex contention**: Every operation acquires the same mutex

## Key Findings

### Current Memory Usage per Node

| Component | Size | Notes |
|-----------|------|-------|
| NodeEntry struct | ~200 bytes | node (200) + stamp (8) + vector (24) |
| nodes_ hash bucket | ~32 bytes | per entry |
| nodes_ pointer | 8 bytes | per entry |
| local_heap_ entries | 128 bytes | 4 heap pushes × 32 bytes |
| stealing_heap_ entries | 128 bytes | 4 heap pushes × 32 bytes |
| **Total** | **~528 bytes** | Plus heap allocations |

### Optimized Memory Usage (Phase 1)

| Component | Size | Notes |
|-----------|------|-------|
| NodeEntry struct | ~200 bytes | Same |
| nodes_ hash bucket | 0 bytes | Bounded array |
| nodes_ pointer | 0 bytes | Bounded array |
| heap_ entries | 32 bytes | Single heap |
| **Total** | **~232 bytes** | 56% reduction |

## Recommended Changes

### Phase 1: Memory Optimization (High Priority)

#### 1.1 Remove Redundant Heap Entries

**Current code (lines 1013-1020):**
```cpp
local_heap_.push(make_score_key_(handle, stamp, stored.bound, stored, maximize));
local_heap_.push(make_score_key_(handle, stamp, estimate, stored, maximize));
local_heap_.push(make_score_key_(handle, stamp, hybrid_node_score(stored), stored, maximize));
local_heap_.push(make_depth_key_(handle, stamp, stored, strategy, maximize));
```

**Problem:** Same node pushed to heap 4 times

**Solution:** Single heap + strategy selection at pop time

```cpp
// Single heap ordered by best bound
std::priority_queue<BestBoundKey, std::vector<BestBoundKey>, BestBoundKeyLess> heap_;

// Pop with strategy selection
ActiveNode pop(NodeSelectionStrategy strategy) {
    while (!heap_.empty()) {
        auto key = heap_.top();
        heap_.pop();
        
        auto it = nodes_.find(key.handle);
        if (it == nodes_.end() || it->second.stamp != key.stamp) {
            continue;  // Stale entry
        }
        
        ActiveNode node = std::move(it->second.node);
        nodes_.erase(it);
        return node;
    }
    return ActiveNode{};
}
```

**Benefits:**
- Memory: ~67% reduction (1 heap entry vs 4)
- Push: O(log n) × 1 (was O(log n) × 4)

#### 1.2 Add Cache-Line Alignment

```cpp
alignas(64) struct NodeEntry {
    std::uint64_t stamp = 0;
    std::int32_t handle = -1;
    ActiveNode node;
    std::vector<DomainChange> root_domain_changes;
};
```

**Benefits:** Better cache utilization, fewer cache misses

#### 1.3 Remove Per-Node Stamp

**Current:** `std::uint64_t stamp` per node (8 bytes)

**Solution:** Shared atomic counter

```cpp
alignas(64) std::atomic<std::uint64_t> next_stamp_{1};

std::uint64_t get_stamp() {
    return next_stamp_.fetch_add(1);
}
```

**Benefits:** 8 bytes saved per node

### Phase 2: Bounded Array (Medium Priority)

Replace `std::unordered_map` with bounded array:

```cpp
constexpr size_t kMaxQueueSize = 8192;

alignas(64) std::array<NodeEntry, kMaxQueueSize> taskArray_;
alignas(64) std::atomic<uint32_t> head_{0};
alignas(64) std::atomic<uint64_t> tailSplit_{0};
alignas(64) std::atomic<bool> allStolen_{true};

// Push: O(1)
void push(const ActiveNode& node) {
    uint32_t h = head_.fetch_add(1);
    if (h >= kMaxQueueSize) {
        execute_node_directly(node);
        return;
    }
    taskArray_[h] = create_entry(node);
}

// Steal: lock-free
ActiveNode steal() {
    uint64_t ts = tailSplit_.load(std::memory_order_relaxed);
    uint32_t t = ts >> 32;
    uint32_t s = ts & 0xFFFF_FFFFu;
    
    if (t >= s) return ActiveNode{};
    
    if (tailSplit_.compare_exchange_weak(
            ts, (ts & 0xFFFF_FFFFu) | (t + 1) << 32)) {
        return taskArray_[t];
    }
    return ActiveNode{};
}
```

**Benefits:**
- Memory: Bounded (8192 nodes × 200 bytes = ~1.6 MB per queue)
- No heap allocations per node
- Lock-free operations

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)

1. **Remove redundant heap entries**
   - Modify `push()` to push to single heap
   - Modify `pop()` to handle strategy selection at pop time
   - Add stale entry cleanup logic

2. **Add cache alignment**
   - Add `alignas(64)` to NodeEntry
   - Add `alignas(64)` to heap containers

3. **Remove per-node stamp**
   - Use shared atomic counter

### Phase 2: Bounded Array (2-3 days)

1. Implement bounded array structure
2. Add dynamic splitting logic
3. Implement lock-free push/pop/steal

### Phase 3: Advanced Features (3-5 days)

1. Add worker bunk mechanism
2. Implement randomized stealing
3. Add leapfrogging support

## Testing Required

1. **Correctness:**
   - Verify solution quality unchanged
   - Verify node ordering for different strategies
   - Verify frontier bounds correct

2. **Concurrent correctness:**
   - ThreadSanitizer for race conditions
   - Memory leak detection
   - Data race detection

3. **Performance benchmark:**
   - Compare against current implementation
   - Measure throughput and latency

## References

- Highs `HighsSplitDeque`: `/data/dev/simplinho/highs-sources/highs/parallel/HighsSplitDeque.h`
- Current `WorkerLocalQueue`: `/data/dev/simplinho/include/bnb/search.h` (lines 857-1426)
- Current `SearchCoordinator`: `/data/dev/simplinho/include/bnb/search_coordinator.h`
