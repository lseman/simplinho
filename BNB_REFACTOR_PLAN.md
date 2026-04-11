# BnB Refactoring Plan: Making Simplinho More Highs-like

## Problem Statement
Current BnB results show gaps between simplinho and Highs optima:
- simplinho_obj: 1578.0
- pulp_obj: 1604.0
- Gap: ~1.6% (should reach true optimum)

Timing breakdown (wall-clock: 21.846s):
- Heuristics: 38.0% (25.96s) - too aggressive, not finding good incumbents
- Branching: 23.2% (15.86s) - strong branching overhead
- LP solves: 27.5% (15.85s) - child node LPs
- Cuts: 7.5% (5.14s) - cut generation and management

## Refactoring Goals

1. **Reach true optimum** - eliminate gaps between solver and reference results
2. **Reduce strong branching overhead** - Highs-style reduced strong branching
3. **Improve cut effectiveness** - MIR cuts, better pool management
4. **Tune heuristic balance** - find incumbents earlier without excessive cost

---

## Phase 1: Strong Branching Optimization (Highest Priority)

### Issue
Current strong branching solves full LP for each candidate, causing 15.86s overhead.

### Current Implementation (`src/bnb/branching.cpp`)
- `choose_strong_branching()` solves LPs for up to `candidate_limit` candidates
- Default limit is 6 candidates (from `options.strong_branching_candidates`)
- For each candidate, solves 2 LPs (down and up children)
- Total: up to 12 LP solves per node during strong branching

### Highs Approach
- **Reduced Strong Branching**: Only solve for k candidates (default k=2) at each step
- **Early termination**: Stop after finding a clearly inferior candidate
- **Bounded limit**: HiGHS uses `std::min(2, candidate_count)` for deep nodes

### Implementation Changes

#### 1.1 `src/bnb/branching.cpp` - `choose_strong_branching()`
Modify to use reduced strong branching (k=2) by default:
```cpp
// Change limit calculation
const int limited_limit = std::min(2, static_cast<int>(candidates.size()));
for (int i = 0; i < limited_limit; ++i) { ... }
```

#### 1.2 `src/bnb/branching.cpp` - `resolve_strong_branching_limit()`
Add a parameter to force reduced strong branching:
```cpp
[[nodiscard]] int resolve_strong_branching_limit(const ActiveNode& node, 
                                                  int candidate_limit,
                                                  int strong_branching_k,  // NEW
                                                  int parallel_workers,
                                                  std::size_t candidate_count) {
    if (candidate_limit > 0) {
        return std::min<int>(candidate_limit, static_cast<int>(candidate_count));
    }
    if (strong_branching_k > 0) {
        return std::min<int>(strong_branching_k, static_cast<int>(candidate_count));
    }
    return automatic_strong_branching_limit(node, parallel_workers, candidate_count);
}
```

#### 1.3 `src/bnb/branching.cpp` - `choose_pseudocost_branching()`
Use reduced strong branching (k=2) by default:
```cpp
const int evaluate_limit = resolve_strong_branching_limit(node, strong_branching_candidates,
                                                          2,  // Force k=2
                                                          parallel_workers, ranked.size());
```

#### 1.4 `include/bnb/types.h` - `Options` struct
Add new option for strong branching k:
```cpp
// Line 156-157: Add new option
int strong_branching_k = 2;  // New: number of candidates for reduced strong branching
int strong_branching_candidates = 6;  // Original: max candidates
```

#### 1.5 `src/bnb/branching.cpp` - `choose_branching_variable()`
Update to use new `strong_branching_k` option:
```cpp
if (options.strong_branching_k > 0) {
    return choose_strong_branching(node, relaxation, fractional, 
                                   options.strong_branching_k,
                                   options.parallel_workers, maximize, ...);
}
```

#### 1.6 Performance Target
Reduce strong branching time from 23.2% to ~8% (Highs achieves ~5-8%)

---

## Phase 2: Cut Generation Improvements

### Issue
Current cut generation uses Gomory cover cuts but MIR cuts are disabled.

### Current State
- `use_gomory_cuts = true`
- `use_mir_cuts = false`  // DISABLED
- `use_cover_cuts = true`

### Highs Approach
- **MIR cuts**: More powerful than Gomory cover
- **Lazy MIR generation**: Only generate when needed
- **Cut pool management**: Sophisticated pool with type limits

### Implementation Changes

#### 2.1 `include/bnb/types.h` - Enable MIR cuts
```cpp
bool use_mir_cuts = true;  // Enable MIR cuts (was false)
```

#### 2.2 `src/bnb/cuts.cpp` - Ensure MIR implementation exists
Verify `generate_mir_cuts()` is implemented and called.

#### 2.3 `src/bnb/cuts.cpp` - Cut Pool Management
Improve pool management:
- Add cut type limits (e.g., max 8 Gomory, max 4 MIR per type)
- Add age-based eviction
- Improve violation detection

#### 2.4 `src/bnb/cuts.cpp` - Cut Application
- Only generate cuts when fractional candidates exist
- Limit cut rounds per node (HiGHS uses 1-2 rounds)
- Early termination if no cuts improve bound

---

## Phase 3: Heuristic Tuning

### Issue
Heuristics consuming 38% of time without finding good incumbents early.

### Current Schedule
Based on the code in `include/bnb/core.h` (lines 1863-2091):
1. Rounding (always runs)
2. Feasibility jump
3. Feasibility pump
4. Diving
5. RENS (async)
6. RINS (async, after incumbent)
7. Local search (async, after incumbent)
8. Local branching (async, after incumbent)

### Highs Approach
- **Rounding first**: Fast, cheap incumbent finder
- **Skip expensive heuristics if rounding succeeds**
- **Limit diving depth and iterations**
- **Only run RINS/local branching after good incumbent found**

### Implementation Changes

#### 3.1 `src/bnb/heuristic.cpp` - `run_rounding_heuristic()`
Optimize rounding to find incumbents faster.

#### 3.2 `src/bnb/heuristic.cpp` - `run_feasibility_pump_heuristic()`
Add early exit if rounding already succeeded:
```cpp
if (rounding_heuristic_succeeded) {
    return NeighborhoodHeuristicResult{...successes=0...};
}
```

#### 3.3 `src/bnb/heuristic.cpp` - `run_diving_heuristic()`
Limit diving more aggressively:
- Reduce default `max_dive_depth` from 25 to 15
- Reduce default `max_dive_lp_solves` from 64 to 32

#### 3.4 `include/bnb/types.h` - Update Default Options
```cpp
int max_dive_depth = 15;  // Was 25
int max_dive_lp_solves = 32;  // Was 64
```

#### 3.5 `src/bnb/heuristic.cpp` - Heuristic Scheduling
Add parameter to skip heuristics if incumbent already found:
```cpp
NeighborhoodHeuristicResult
run_feasibility_pump_heuristic(const Problem& problem, const Options& options,
                               const RelaxationSolution& lp_relaxation,
                               const SubproblemSolveCallback& solve_submip,
                               bool skip_if_incumbent_exists = false);
```

---

## Phase 4: State Management and Allocations

### Issue
Unnecessary copying and allocations during node processing.

### Current Issues
- `ChildState` copying in `node_relaxation_solver` lambda (line 1945 in core.h)
- Frequent snapshot creation for shared state
- Excessive vector allocations in cut generation

### Highs Approach
- **Value types**: Avoid unnecessary heap allocations
- **Move semantics**: Pass by value where ownership transfers
- **Cache alignment**: Align hot data structures

### Implementation Changes

#### 4.1 `src/bnb/core.cpp` - Reduce ChildState copying
Pass `ChildState` by reference instead of copying:
```cpp
// Change from:
auto node_relaxation_solver = [&](detail::ChildState& state, const LPBasis* basis) {
    detail::ChildState local_state = state;  // Copy
    ...
};

// Change to:
auto node_relaxation_solver = [&](const detail::ChildState& state, const LPBasis* basis) {
    // Use state directly, no copy
    ...
};
```

#### 4.2 `src/bnb/core.cpp` - Snapshot Optimization
Use versioning more aggressively:
```cpp
// Instead of full snapshot copy:
shared.incumbent = incumbent_snapshot_versioned_();

// Use pointer-based access:
const auto& incumbent = *shared.incumbent.value;
```

#### 4.3 `src/bnb/cuts.cpp` - Cut Vector Optimization
Pre-allocate cut vectors:
```cpp
class CutPool {
    std::vector<Cut> cuts_;
    std::vector<std::vector<int>> cut_indices_pool_;  // Pre-allocate
    std::vector<std::vector<double>> cut_values_pool_;  // Pre-allocate
};
```

---

## Implementation Order

1. **Phase 1** (Strong Branching) - **COMPLETED** - Immediate impact on branching time
2. **Phase 2** (Cuts) - **PARTIALLY COMPLETED** - MIR cuts enabled, pool size reduced
3. **Phase 3** (Heuristics) - **PARTIALLY COMPLETED** - Diving limits reduced
4. **Phase 4** (State) - Pending - Cleanup and optimization

---

## Changes Made

### Phase 1: Strong Branching Optimization
- Added `strong_branching_k` option (default=2) for reduced strong branching
- Modified `resolve_strong_branching_limit()` to use `strong_branching_k` parameter
- Updated `choose_strong_branching()` and `choose_pseudocost_branching()` to use k=2 by default
- Root node uses full strong branching to seed pseudocosts
- Deep nodes use reduced strong branching (k=2) for efficiency

### Phase 2: Cut Generation Improvements
- Enabled `use_mir_cuts = true` (was disabled)
- Reduced `max_cut_pool_size` from 256 to 128
- Reduced `max_cut_rounds_per_node` from 2 to 1
- Reduced `max_cuts_added_per_round` from 8 to 4

### Phase 3: Heuristic Tuning
- Reduced `max_dive_depth` from 25 to 15
- Reduced `max_dive_lp_solves` from 64 to 32

---

## Files Modified

| File | Changes |
|------|---------|
| `include/bnb/types.h` | Add `strong_branching_k`, enable MIR cuts, update defaults |
| `src/bnb/branching.cpp` | Modify strong branching limits, add `strong_branching_k` support |

---

## Testing Strategy

1. **Benchmarks**: Compare against Highs on test instances
2. **Regression tests**: Ensure optimal solutions are still found
3. **Profiling**: Verify timing improvements after each phase

---

## Expected Improvements

| Metric | Before | After |
|--------|--------|-------|
| Strong branching | 23.2% | ~8% |
| Heuristics | 38.0% | ~15% |
| Cuts | 7.5% | ~5% |
| Total wall-clock | 21.8s | ~15s |
| Gap to optimum | ~1.6% | 0% |
