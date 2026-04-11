# Queue Architecture Comparison: simplinho vs Highs

## Executive Summary

The actual queue used in simplinho's search path is **`WorkerLocalQueue`** (managed by `SearchCoordinator`), not `OpenNodeQueue`. Our `WorkerLocalQueue` implementation differs significantly from Highs' `HighsSplitDeque`.

---

## Current Architecture (simplinho)

```
SearchCoordinator
├── worker_queues_ (vector<WorkerLocalQueue>)
│   └── Each worker gets its own WorkerLocalQueue
├── mutex_ (central lock for coordination)
└── Statistics tracking
```

**WorkerLocalQueue internals:**
```cpp
std::unordered_map<int, NodeEntry> nodes_;              // 8 bytes + pointer per entry
std::priority_queue<QueueKey, std::vector<QueueKey>, QueueKeyLess> local_heap_;  // ~4x nodes stored
std::priority_queue<StealKey, std::vector<StealKey>, StealKeyLess> stealing_heap_; // ~4x nodes stored
mutable std::mutex mutex_;
```

### Key Issues in Current Implementation

1. **Multiple priority queues per node** (lines 1014-1020 in search.h):
   ```cpp
   // Push to local heap - pushes SAME node 4 times!
   local_heap_.push(make_score_key_(handle, stamp, stored.bound, stored, maximize));
   local_heap_.push(make_score_key_(handle, stamp, estimate, stored, maximize));
   local_heap_.push(make_score_key_(handle, stamp, hybrid_node_score(stored), stored, maximize));
   local_heap_.push(make_score_key_(handle, stamp, stored, strategy, maximize));  // depth
   ```
   **Problem:** Each node is stored 4 times in separate heaps → 4x memory overhead, 4x heap operations

2. **No bounded queue size**: Uses `std::unordered_map` which grows unbounded

3. **No cache alignment**: Poor cache utilization

4. **Mutex contention**: Every `push/pop/steal` acquires the same mutex

5. **Redundant stamp tracking**: `std::uint64_t` per node (8 bytes)

---

## Highs' HighsSplitDeque Approach

### Core Data Structure

```cpp
struct WorkerBunk;

struct OwnerData {
    cache_aligned::shared_ptr<WorkerBunk> workerBunk;
    cache_aligned::unique_ptr<HighsSplitDeque>* workers;
    HighsRandom randgen;              // For randomized stealing
    uint32_t head = 0;                // Push position
    uint32_t splitCopy = 0;           // Lazy split point
    int numWorkers = 0;
    int ownerId = -1;
    HighsTask* rootTask = nullptr;
    bool allStolenCopy = true;
};

struct StealerData {
    HighsBinarySemaphore semaphore{0};
    HighsTask* injectedTask{nullptr};
    std::atomic<uint64_t> ts{0};      // Tail + split in one atomic
    std::atomic<bool> allStolen{true};
};

// 8192 tasks per deque
std::array<HighsTask, kTaskArraySize> taskArray;
```

### Key Design Principles

| Feature | Highs Approach |
|---------|---------------|
| **Structure** | Single array with head/tail pointers |
| **Thread Safety** | Lock-free with atomics |
| **Cache Optimization** | 64-byte cache-line alignment |
| **Task Storage** | Inline task array (8192 tasks) |
| **Bounded** | Yes (8192 tasks max) |
| **Dynamic Splitting** | Lazy splitting when queue fills |

---

## Comparison Summary

| Aspect | simplinho (`WorkerLocalQueue`) | Highs (`HighsSplitDeque`) |
|--------|-------------------------------|---------------------------|
| **Data Structure** | `std::unordered_map` + 2 priority queues | Single bounded array |
| **Memory per Node** | ~500+ bytes (4 heap entries + map overhead) | ~64 bytes (inline task) |
| **Thread Safety** | `std::mutex` on every access | Lock-free with atomics |
| **Cache Alignment** | No explicit alignment | 64-byte aligned |
| **Queue Bounded** | Unbounded | 8192 tasks max |
| **Work Distribution** | Centralized with separate stealing heap | Decentralized dynamic splitting |
| **Push Complexity** | O(log n) × 4 (4 heap pushes) | O(1) (array push) |
| **Stealing** | Sequential scan of other queues | Randomized with worker bunk |

---

## Recommendations

### High Priority (Quick Wins)

#### 1. Remove Redundant Priority Queues

**Current:** 4 priority queues per node → 4x memory and 4x heap operations

**Solution:** Single heap + strategy selection at pop time:
```cpp
// Store nodes once
struct NodeEntry {
    ActiveNode node;
    std::uint64_t stamp;
    // Domain changes tracked separately
};
std::unordered_map<int, NodeEntry> nodes_;

// Single priority queue ordered by best bound
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
            continue;  // Stale entry
        }
        
        ActiveNode node = std::move(it->second.node);
        nodes_.erase(it);
        
        // Update frontier summary
        update_frontier_summary(it->second.root_domain_changes, false);
        return node;
    }
    return ActiveNode{};  // Empty
}
```

**Impact:**
- Memory: ~60% reduction (from 4 heap entries to 1)
- Push: O(log n) × 4 → O(log n)
- Pop: O(log n) × 1 (with stale entry cleanup)

#### 2. Add Cache-Line Alignment

**Current:** No explicit alignment

**Solution:**
```cpp
alignas(64) struct NodeEntry {
    std::uint64_t stamp;
    std::int32_t handle;
    ActiveNode node;  // ~200 bytes
    std::vector<DomainChange> root_domain_changes;  // Pointer + size
};

alignas(64) std::unordered_map<int, NodeEntry> nodes_;
```

**Impact:** Better cache utilization, fewer cache misses

### Medium Priority

#### 3. Bounded Queue with Dynamic Splitting

**Current:** Unbounded `std::unordered_map`

**Solution:** Bounded array with lazy splitting:
```cpp
constexpr size_t kMaxQueueSize = 8192;
constexpr double kSplitThreshold = 0.8;

alignas(64) std::array<NodeEntry, kMaxQueueSize> taskArray_;
alignas(64) std::atomic<uint32_t> head_{0};        // Push position
alignas(64) std::atomic<uint64_t> tailSplit_{0};   // Tail + split point
alignas(64) std::atomic<bool> allStolen_{true};    // All work stolen

void push(const ActiveNode& node) {
    uint32_t h = head_.fetch_add(1, std::memory_order_relaxed);
    
    if (h >= kMaxQueueSize * kSplitThreshold) {
        request_split();  // Lazy splitting
    }
    
    // Check if pushed successfully
    if (h >= kMaxQueueSize) {
        // Queue full, execute directly
        execute_node(node);
        return;
    }
    
    taskArray_[h] = create_entry(node);
}

void request_split() {
    // Request other workers to split the queue
    split_request_.store(true);
}

// Steal from another worker
ActiveNode steal() {
    uint64_t ts = tailSplit_.load(std::memory_order_relaxed);
    uint32_t t = ts >> 32;  // tail
    uint32_t s = ts & 0xFFFF_FFFFu;  // split
    
    if (t >= s) return ActiveNode{};  // Empty
    
    if (tailSplit_.compare_exchange_weak(
            ts, (ts & 0xFFFF_FFFFu) | (t + 1) << 32,
            std::memory_order_acq_rel, std::memory_order_relaxed)) {
        return taskArray_[t];
    }
    return ActiveNode{};
}
```

**Impact:**
- Memory: Bounded, predictable
- Work distribution: Automatic load balancing

#### 4. Remove Redundant Stamp Per Node

**Current:** `std::uint64_t stamp` per node (8 bytes)

**Solution:** Shared atomic counter:
```cpp
alignas(64) std::atomic<std::uint64_t> next_stamp_{1};

std::uint64_t get_stamp() {
    return next_stamp_.fetch_add(1);
}
```

**Impact:** 8 bytes saved per node

### Low Priority

#### 5. Remove `std::unordered_map`

**Current:** `std::unordered_map<int, NodeEntry>` causes heap allocation per entry

**Solution:** Bounded array (as shown above)

**Impact:** Significantly reduced heap allocations

#### 6. Add Worker Bunk Mechanism

Highs' worker bunk allows idle workers to wait efficiently for work:
```cpp
struct WorkerBunk {
    alignas(64) std::atomic<int> haveJobs{0};
    alignas(64) std::atomic<uint64_t> sleeperStack{0};
    
    void pushSleeper(HighsSplitDeque* deque) { /* ... */ }
    HighsSplitDeque* popSleeper(HighsSplitDeque* localDeque) { /* ... */ }
    void publishWork(HighsSplitDeque* localDeque) { /* ... */ }
    HighsTask* waitForNewTask(HighsSplitDeque* localDeque) { /* ... */ }
};
```

**Impact:** Better idle handling, reduced context switching

---

## Implementation Roadmap

### Phase 1: Memory Optimization (Quick Win - 1-2 days)
- [ ] Remove redundant priority queues (single heap)
- [ ] Add cache-line alignment to NodeEntry
- [ ] Remove per-node stamp, use shared atomic

### Phase 2: Work Stealing Enhancement (2-3 days)
- [ ] Implement bounded array (8192 tasks)
- [ ] Add dynamic splitting logic
- [ ] Implement lock-free push/pop
- [ ] Remove `std::unordered_map`

### Phase 3: Advanced Features (3-5 days)
- [ ] Add worker bunk mechanism
- [ ] Implement randomized stealing
- [ ] Add leapfrogging support
- [ ] Benchmark vs current implementation

---

## Expected Performance Impact

| Metric | Current | After Phase 1 | After Phase 2 |
|--------|---------|--------------|--------------|
| Memory per node | ~500 bytes | ~250 bytes | ~150 bytes |
| Push complexity | O(log n) × 4 | O(log n) | O(1) |
| Cache misses | High | Medium | Low |
| Thread contention | High (mutex) | Medium | Low (lock-free) |
| Queue depth | Unbounded | Unbounded | 8192 (bounded) |
| Heap allocs | 1 per node | 0 | 0 |

---

## References

- Highs `HighsSplitDeque`: `/data/dev/simplinho/highs-sources/highs/parallel/HighsSplitDeque.h`
- Current `WorkerLocalQueue`: `/data/dev/simplinho/include/bnb/search.h` (lines 857-1426)
- Current `SearchCoordinator`: `/data/dev/simplinho/include/bnb/search_coordinator.h`
