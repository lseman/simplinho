# WorkerLocalQueue Analysis and Optimization Opportunities

## Current Implementation Analysis

### Line-by-Line Issues

#### Lines 965-967: Memory-Heavy Data Structures
```cpp
std::unordered_map<int, NodeEntry> nodes_;              // Pointer + bucket overhead per entry
std::priority_queue<QueueKey, std::vector<QueueKey>, QueueKeyLess> local_heap_;
std::priority_queue<StealKey, std::vector<StealKey>, StealKeyLess> stealing_heap_;
```

**Issues:**
1. `std::unordered_map` causes heap allocation per entry (pointer + hash table bucket)
2. Each `NodeEntry` is stored ONCE in the map but referenced in TWO heaps
3. `std::vector` as underlying container causes reallocations and poor cache locality

**Memory per node breakdown:**
- `nodes_` entry: `NodeEntry` (~200 bytes) + hash bucket overhead (~32 bytes) + pointer (8 bytes) = ~240 bytes
- `local_heap_` entry: `QueueKey` (32 bytes) per node
- `stealing_heap_` entry: `StealKey` (32 bytes) per node

**Total:** ~304 bytes per node, even before heap allocations

---

#### Lines 1013-1020: Redundant Heap Pushes
```cpp
// Push to local heap (using the same priority logic as OpenNodeQueue)
const ActiveNode& stored = entry.node;
local_heap_.push(make_score_key_(handle, stamp, stored.bound, stored, maximize));  // 1
local_heap_.push(make_score_key_(handle, stamp, estimate, stored, maximize));     // 2
local_heap_.push(make_score_key_(handle, stamp, hybrid_node_score(stored), stored, maximize));  // 3
local_heap_.push(make_depth_key_(handle, stamp, stored, strategy, maximize));     // 4
```

**Problem:** The SAME node is pushed to the heap FOUR times with different scoring functions.

**Impact:**
- Memory: 4x heap entries per node
- Time: 4x heap push operations (O(log n) each)
- Time: Up to 4x heap pop operations during steal (need to check 4 heaps)

---

#### Lines 1074-1079: Expensive Frontier Update
```cpp
if (result.has_value()) {
    // Update frontier summary after removal
    for (auto& [handle, entry] : nodes_) {
        update_frontier_summary_(entry.root_domain_changes, false);
    }
}
```

**Problem:** Iterates over ALL remaining nodes to update frontier summary

**Impact:** O(n) operation on every pop, where n is queue size

---

#### Lines 1084-1102: Sequential Stealing
```cpp
std::optional<ActiveNode> steal(NodeSelectionStrategy strategy, bool maximize,
                                int hybrid_depth_bias, int plunging_bestfreq,
                                std::uint64_t* hybrid_counter) {
    std::lock_guard<std::mutex> lock(mutex_);

    // First, try to steal from the local heap (highest priority)
    std::optional<ActiveNode> result = extract_valid_node_(local_heap_);
    if (result.has_value()) {
        update_frontier_summary_...
        return result;
    }

    // If local heap is exhausted, try the stealing heap
    return extract_valid_node_(stealing_heap_);
}
```

**Problem:** Sequential scan through heaps, no randomness

**Impact:** Can cause "thundering herd" when multiple workers steal simultaneously

---

## Optimization Opportunities

### Priority 1: Eliminate Redundant Heap Entries

**Current approach:** Push same node to 4 different heap priorities

**Optimized approach:** Single heap + strategy selection at pop time

```cpp
// Single heap ordered by best bound
struct BestBoundKey {
    int handle = -1;
    std::uint64_t stamp = 0;
    double score = -std::numeric_limits<double>::infinity();
    int depth = 0;
    std::uint64_t order = 0;
};

std::priority_queue<BestBoundKey, std::vector<BestBoundKey>, BestBoundKeyLess> heap_;

// Pop with strategy selection
ActiveNode pop(NodeSelectionStrategy strategy) {
    while (!heap_.empty()) {
        auto key = heap_.top();
        heap_.pop();
        
        auto it = nodes_.find(key.handle);
        if (it == nodes_.end() || it->second.stamp != key.stamp) {
            continue;  // Stale entry, skip
        }
        
        ActiveNode node = std::move(it->second.node);
        nodes_.erase(it);
        update_frontier_summary_(it->second.root_domain_changes, false);
        return node;
    }
    return ActiveNode{};
}
```

**Benefits:**
- Memory: ~67% reduction (1 heap entry vs 4)
- Push: O(log n) × 1 (was O(log n) × 4)
- Pop: O(log n) × 1 (with stale cleanup)

---

### Priority 2: Bounded Array + Cache Alignment

**Current:** Unbounded `std::unordered_map`

**Optimized:** Bounded array with cache alignment

```cpp
constexpr size_t kMaxQueueSize = 8192;

alignas(64) struct NodeEntry {
    std::uint64_t stamp = 0;
    std::int32_t handle = -1;
    ActiveNode node;
    std::vector<DomainChange> root_domain_changes;
};

alignas(64) std::array<NodeEntry, kMaxQueueSize> taskArray_;
alignas(64) std::atomic<uint32_t> head_{0};
alignas(64) std::atomic<uint64_t> tailSplit_{0};  // Tail + split in one atomic
alignas(64) std::atomic<bool> allStolen_{true};

// Push: O(1)
void push(ActiveNode node) {
    uint32_t h = head_.fetch_add(1);
    if (h >= kMaxQueueSize) {
        execute_node_directly(node);  // Queue full, execute immediately
        return;
    }
    taskArray_[h] = create_entry(node);
}

// Steal: lock-free
ActiveNode steal() {
    uint64_t ts = tailSplit_.load(std::memory_order_relaxed);
    uint32_t t = ts >> 32;
    uint32_t s = ts & 0xFFFF_FFFFu;
    
    if (t >= s) return ActiveNode{};  // Empty
    
    if (tailSplit_.compare_exchange_weak(
            ts, (ts & 0xFFFF_FFFFu) | (t + 1) << 32)) {
        return taskArray_[t];
    }
    return ActiveNode{};
}
```

**Benefits:**
- Memory: Bounded (8192 nodes × ~200 bytes = ~1.6 MB per queue)
- No heap allocations per node
- Lock-free operations (no mutex contention)

---

### Priority 3: Remove Per-Node Stamp

**Current:** `std::uint64_t stamp` per node (8 bytes)

**Optimized:** Shared atomic counter

```cpp
alignas(64) std::atomic<std::uint64_t> next_stamp_{1};

std::uint64_t get_stamp() {
    return next_stamp_.fetch_add(1);
}
```

**Benefits:**
- Memory: 8 bytes saved per node
- Centralized versioning

---

### Priority 4: Lazy Frontier Summary Update

**Current:** Update frontier summary on every pop (O(n))

**Optimized:** Update frontier summary lazily when needed

```cpp
// Don't update frontier summary on every pop
// Only update when computing best bound or when needed

double compute_best_bound() const {
    // Only scan nodes when actually needed
    for (auto& [handle, entry] : nodes_) {
        // Check if frontier summary is up to date
        if (!entry.frontier_summary_valid) {
            update_frontier_summary_for_entry(entry);
        }
    }
    // Compute best bound
}
```

**Benefits:**
- Pop: O(1) instead of O(n)
- Reduced memory bandwidth

---

## Detailed Memory Analysis

### Current Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| `NodeEntry` struct | ~200 bytes | node (200) + stamp (8) + vector (24) |
| `nodes_` hash bucket | ~32 bytes | per entry |
| `nodes_` pointer | 8 bytes | per entry |
| `local_heap_` entries | 32 bytes × 4 = 128 bytes | 4 heap pushes |
| `stealing_heap_` entries | 32 bytes × 4 = 128 bytes | 4 heap pushes |
| **Total per node** | **~528 bytes** | Plus heap allocations |

### Optimized Memory Usage

| Component | Size | Notes |
|-----------|------|-------|
| `NodeEntry` struct | ~200 bytes | Same |
| `nodes_` hash bucket | 0 bytes | Bounded array |
| `nodes_` pointer | 0 bytes | Bounded array |
| `heap_` entries | 32 bytes × 1 = 32 bytes | Single heap |
| **Total per node** | **~232 bytes** | 56% reduction |

### Bounded Array Approach

| Component | Size | Notes |
|-----------|------|-------|
| `taskArray_` | ~176 KB | 8192 × 200 bytes |
| Metadata | ~32 bytes | head, tailSplit, allStolen |
| **Total per queue** | **~176 KB** | Fixed size |

---

## Performance Analysis

### Current Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| push | O(log n) × 4 | 4 heap pushes |
| pop | O(log n) × 4 | 4 heap pops, O(n) frontier update |
| steal | O(log n) × 4 | Sequential heap scans |
| compute_best_bound | O(n) | Scan all nodes |

### Optimized Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| push | O(log n) | Single heap push |
| pop | O(log n) | Single heap pop, O(1) frontier |
| steal | O(log n) | Single heap scan |
| compute_best_bound | O(n) | Same |

---

## Implementation Plan

### Phase 1: Quick Wins (1-2 days)

1. **Remove redundant heap entries**
   - Modify `push()` to push to single heap
   - Modify `pop()` to handle strategy selection at pop time
   - Add stale entry cleanup

2. **Add cache alignment**
   - Add `alignas(64)` to `NodeEntry`
   - Add `alignas(64)` to heap containers

3. **Remove per-node stamp**
   - Use shared atomic counter

### Phase 2: Bounded Array (2-3 days)

1. **Implement bounded array**
   - Replace `std::unordered_map` with `std::array`
   - Add `head` and `tailSplit` atomics

2. **Implement lock-free push/pop**
   - Add `push()` with atomic head
   - Add `pop()` with atomic tail
   - Add `steal()` with CAS

3. **Remove mutex from push/pop**
   - Only keep mutex for coordination operations

### Phase 3: Advanced Features (3-5 days)

1. **Dynamic splitting**
   - Add lazy splitting when queue fills

2. **Randomized stealing**
   - Add random worker selection

3. **Worker bunk mechanism**
   - Add idle worker waiting

---

## Testing Considerations

### Regression Tests Needed

1. **Correctness:**
   - Verify solution quality unchanged
   - Verify node ordering for different strategies
   - Verify frontier bounds correct

2. **Concurrent correctness:**
   - Race condition detection (ThreadSanitizer)
   - Memory leak detection (AddressSanitizer)
   - Data race detection (Helgrind)

3. **Performance:**
   - Benchmark against current implementation
   - Memory usage verification
   - Throughput measurement

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Race conditions in lock-free code | Medium | High | Use ThreadSanitizer, conservative memory ordering |
| Performance regression | Medium | High | Benchmark before/after |
| Memory corruption | Low | High | AddressSanitizer, Valgrind |
| Incorrect node ordering | Medium | Medium | Thorough unit tests |
