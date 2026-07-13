<p align="center">
  <img src="assets/simplinho-logo.svg" alt="simplinho logo" width="720">
</p>

# simplinho

`simplinho` is a C++ revised simplex and branch-and-bound solver exposed as a
small Python optimization package through `pybind11`.

The LP core focuses on repeated bounded LP solves, warm starts, and useful
diagnostics. The MIP layer builds on that core with branch-and-bound, presolve,
cut separation, primal heuristics, SOS constraints, and parallel work sharing
for the expensive parts of the search.

The project is still compact and hackable: the simplex core lives under
`include/simplex/`, the branch-and-bound layer under `include/bnb/`, and the
Python bindings/modeling API under `bindings/`.

## Component boundaries

The build treats simplex and branch-and-bound as separate components:

- `simplex_core` is the public, header-only LP dependency.
- `simplinho` contains only the simplex runtime and simplex/model bindings.
- `bnb::core` is an optional compiled library that links to `simplex_core`.
- `simplinho_bnb` contains the optional BnB bindings and imports `simplinho`
  when loaded.

Enable the downstream component with `-DSIMPLEX_ENABLE_BNB=ON`. Consumers of
the C++ MIP solver should include `<bnb/core.h>` and link `bnb::core`; the old
`<simplex/bnb.h>` reverse-dependency shim has been removed.

## Overview

`simplinho` is designed for experimentation with practical LP/MIP solver
machinery:

- fast revised simplex solves for bounded LPs
- reliable basis reuse across re-solves and branch-and-bound node relaxations
- a Python modeling API that stays close to common algebraic modeling syntax
- MIP search with configurable branching, node selection, cuts, and heuristics
- detailed telemetry for inspecting numerics, warm starts, cuts, heuristics, and
  branch-and-bound progress

## Highlights

- Primal and dual revised simplex with automatic mode selection
- Warm-started reoptimization through saved bases and per-solver basis caching
- Crash basis construction with hybrid, repair, sprint, Crash II, and Crash III
  strategies
- Presolve, scaling, singleton elimination, row/column reduction, and bound
  tightening
- Robust numerics with Markowitz LU, rook-style pivot quality controls,
  Forrest-Tomlin updates, and iterative refinement
- Branch-and-bound with flexible search policies and branching rules
- Cut generation for Gomory, MIR, cover, zero-half, implied-bound, clique,
  odd-cycle, conflict, and dual-proof cuts
- Dedicated root cut rounds, cut pool aging, duplicate filtering, and balanced
  cut selection by violation, efficacy, density, and type
- MIP heuristics including rounding, diving, feasibility pump, feasibility jump,
  RENS, RINS, local search, and local branching
- SOS1/SOS2 modeling support with dedicated feasibility checks and branching
- Parallel workers for node processing, cut separation, strong-branching probes,
  and asynchronous heuristics

## Python API

Two main solver interfaces are exposed:

1. `simplinho.RevisedSimplex` — low-level matrix LP solves
2. `simplinho.Model` — algebraic modeling and LP/MIP solving

The module also exposes options and enums such as:

- `RevisedSimplexOptions`
- `BranchAndBoundOptions`
- `SimplexMode`
- `NodeSelectionStrategy`
- `BranchingStrategy`
- `DivingStrategy`
- `VarType`
- `status_to_string(...)`
- `mip_status_to_string(...)`

## Solver Features

### Linear Programming

The LP engine is the foundation of the package and is meant to be useful both
directly and as the relaxation engine for MIP search.

- Revised simplex with primal, dual, and automatic modes
- Phase I fallback when a feasible initial basis is unavailable
- Bounded-variable handling, shifted/free variable support, and slack
  reformulation
- Dual bound-flip logic for faster progress on bound-heavy relaxations
- Markowitz-based crash basis construction with several strategy profiles
- Presolve reduction, scaling, singleton elimination, and bound tightening
- Warm starts from explicit `LPBasis` objects or the solver's cached last basis
- Diagnostics for tableau data, duals, reduced costs, shadow prices, Farkas
  certificates, primal rays, trace logs, and solve statistics

### Mixed-Integer Programming

The MIP layer solves integer and binary models through branch-and-bound, using
the simplex engine for node relaxations.

- Integer and binary variables through `VarType.Integer` and `VarType.Binary`
- SOS1 and SOS2 constraints through `add_sos1(...)` / `add_sos2(...)`
- Node selection strategies: depth-first, breadth-first, best-bound,
  best-estimate, hybrid, best-first plunging, best-estimate plunging, and
  interleaved best-first/best-estimate
- Branching rules: most-fractional, pseudo-cost, and reduced strong branching
- SOS-aware branching when SOS feasibility drives the split
- Diving strategies: fractional, vector-length, objective-value, coefficient,
  guided, and adaptive
- Root and node cut budgets with separate root cut limits
- Node presolve, relaxation warm starts, and LP reoptimization profiling
- Parallel workers and thread-local caches for reduced contention in larger
  searches
- Stopping controls for node limits, absolute MIP gap, and relative MIP gap

### Cut Engine

`simplinho` has a deliberately visible cut engine: each family can be toggled
from `BranchAndBoundOptions`, and the solution object reports generated,
applied, duplicate, and retained-pool counts.

- Gomory mixed-integer cuts from the LP tableau
- MIR cuts, including substitution choices for bounded variables
- Cover cuts for binary knapsack-like rows
- Zero-half cuts, available behind `use_zero_half_cuts`
- Implied-bound cuts
- Clique cuts, including graph-based clique separation and
  implication-strengthened cliques
- Odd-cycle cuts from binary conflict structure
- Conflict cuts with a HiGHS-style conflict pool, age limits, and per-round caps
- Dual-proof cuts
- Probing implications used by conflict and graph-based separators
- Cut pool scoring that balances violation, efficacy, density, age, parallelism,
  and per-family diversity

### Numerics and Stability

- Markowitz LU factorization and sparse/dense solver utilities
- Pivot quality controls and growth/condition guards
- Iterative refinement in forward and transpose solves
- Configurable refactor frequency and matrix compression
- Forrest-Tomlin updates by default, with eta-stack and hybrid update options
- Degeneracy management, cost perturbation, and anti-cycling support
- Pricing rules including adaptive, Devex, row pricing, most-infeasible, and
  most-negative style choices
- Optional quadratic warm-start repair paths for difficult reused bases

### Presolve and Diagnostics

- LP presolve with row/column simplification, scaling, singleton elimination,
  and bound tightening
- MIP root and node presolve with bound tightening, row removal, coefficient
  removal, and aggregation counters
- Early detection of infeasible and unbounded models in presolve
- Verbose tracing with optional basis and presolve diagnostics
- Internal tableau exposure, duals, reduced costs, and shadow prices for LPs
- MIP telemetry for root bounds, node counts, relaxation solves, heuristic
  successes, cut activity, strong-branching probes, warm-start reuse, and phase
  timing
- Farkas certificates and primal rays for LP proof/debug workflows

## Build

The CMake project currently builds a Python extension named `simplinho`.

### Requirements

- CMake 3.16+
- A C++23 compiler
- Python 3.13 development headers

CMake fetches Eigen, `pybind11`, `stdexec`, and `ankerl::unordered_dense` when
they are not already available through the configured local dependency paths.

### Build Commands

```bash
cmake -S . -B build-local
cmake --build build-local -j
```

That produces a shared object like `build-local/simplinho.cpython-313-...so`.

## Low-Level Example

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("build-local").resolve()))

import simplinho as simplex

A = np.array([
    [1.0, 1.0],
])
b = np.array([4.0])
c = np.array([1.0, 2.0])
l = np.array([0.0, 0.0])
u = np.array([np.inf, np.inf])

options = simplex.RevisedSimplexOptions()
options.mode = simplex.SimplexMode.Auto
options.pricing_rule = "adaptive"

solver = simplex.RevisedSimplex(options)
solution = solver.solve(A, b, c, l, u)

print(simplex.status_to_string(solution.status))
print("objective:", solution.obj)
print("x:", solution.x)
print("iterations:", solution.iters)
print("stats:", solution.stats.as_dict())
print("log:", solution.log)
print("basis:", solution.basis_state)
print("dual values:", solution.dual_values_internal)
print("reduced costs:", solution.reduced_costs_internal)
```

You can also reuse that basis as a warm start after bound changes:

```python
basis = solution.basis_state

u2 = np.array([1.5, np.inf])
warm = solver.solve(A, b, c, l, u2, basis=basis)
print(warm.stats.basis_start)
```

If you keep the same `RevisedSimplex` instance in `Dual` mode, it will also
automatically reuse the last valid basis on later `solve(...)` calls with the
same row/column dimensions. That makes repeated bound-fixing re-solves behave
more like a branch-and-bound node LP loop:

```python
options.mode = simplex.SimplexMode.Dual
solver = simplex.RevisedSimplex(options)

root = solver.solve(A, b, c, l, u)
u_node = np.array([1.5, np.inf])
node = solver.solve(A, b, c, l, u_node)
print(node.stats.basis_start)
```

## Modeling API Example

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("build-local").resolve()))

import simplinho as simplex

model = simplex.Model()

x = model.addVar("x", lb=0.0)
y = model.addVar("y", lb=0.0)

c1 = model.addConstr(x <= 3, name="cap_x")
c2 = model.addConstr(y <= 2, name="cap_y")
c3 = model.addConstr(x + 2 * y <= 6, name="mix_cap")

model.maximize(x + y)
solution = model.solve()

print("status   :", simplex.status_to_string(solution.status))
print("objective:", solution.obj)
print("x        :", solution.value(x))
print("y        :", solution.value("y"))
print("all vars :", solution.values)
print("stats    :", solution.stats.as_dict())
print("basis    :", solution.basis)
print("log      :", solution.log)
print("dual c1  :", c1.pi)
print("dual c2  :", c2.pi)
print("dual c3  :", c3.pi)
```

The modeling layer also supports live edits after construction:

```python
x.obj = 3.0
c3.rhs = 5.0
c3.set_coeff(y, 2.0)

solution = model.reoptimize()

model.deleteConstr(c1)
model.deleteVar(x)
solution = model.reoptimize()
```

When the model structure stays the same, `reoptimize()` will automatically try
to reuse the last valid basis after edits like bound changes, RHS updates,
coefficient changes, or objective changes. You can also pass an explicit saved
basis for fast dual re-solves:

```python
basis = solution.basis
x.ub = 1.5
model.options.mode = simplex.SimplexMode.Dual
solution = model.reoptimize(basis)
```

For mixed-integer models, mark variables as `Integer` or `Binary` and call
`solve_mip()`:

```python
import simplinho as simplex

model = simplex.Model()
x = model.add_binary_var("x", obj=4.0)
y = model.add_var("y", lb=0.0, ub=5.0, obj=3.0, var_type=simplex.VarType.Integer)

model.addConstr(2 * x + y <= 4, name="cap")
model.maximize(4 * x + 3 * y)

mip = simplex.BranchAndBoundOptions()
mip.node_selection = simplex.NodeSelectionStrategy.BestBound
mip.branching_strategy = simplex.BranchingStrategy.StrongBranching
mip.diving_strategy = simplex.DivingStrategy.ObjectiveValue
mip.use_feasibility_pump = True
mip.use_rens = True
mip.use_rins = True
mip.use_local_search = True
mip.use_cut_pool = True
mip.use_gomory_cuts = True
mip.use_cover_cuts = True

solution = model.solve_mip(mip)
print(simplex.mip_status_to_string(solution.status))
print(solution.obj, solution.values, solution.node_count)
print(solution.tree_nodes[0])
print(solution.heuristic_successes, solution.heuristic_lp_iterations)
print(solution.feasibility_pump_successes, solution.rens_successes)
print(solution.rins_successes, solution.local_search_successes)
print(solution.cuts_generated, solution.cuts_applied, solution.cut_pool_size)
```

## Useful Outputs

The low-level `LPSolution` object includes more than just the primal vector:

- `status`, `obj`, `x`, `iters`
- `stats` with typed solve telemetry and `as_dict()`
- `basis_state` / `basis` for reusable warm starts
- `log_lines` and `log` for verbose solver traces
- `basis`, `basis_internal`, `nonbasis_internal`
- `tableau`, `tableau_rhs`, `has_internal_tableau`
- `reduced_costs_internal`
- `dual_values_internal`
- `shadow_prices_internal`
- `trace`
- `info`
- `farkas_y`, `farkas_y_internal`, `farkas_has_cert`
- `primal_ray`, `primal_ray_internal`, `primal_ray_has_cert`

The modeling layer wraps that in `ModelSolution`, while still exposing the raw solve result as `solution.raw`.

## Repository Layout

- `include/simplex/`: revised simplex headers and solver internals
- `include/bnb/`: branch-and-bound core, branching rules, search policies, heuristics, and cut management
- `bindings/`: Python bindings and modeling API implementation
- `src/`: C++ implementation sources and helpers
- `tests/`: test coverage for solver and modeling behavior
- `CMakeLists.txt`: build configuration for the Python extension module
