<p align="center">
  <img src="assets/simplinho-logo.svg" alt="simplinho logo" width="720">
</p>

# simplinho

`simplinho` is a compact revised simplex LP solver with early branch-and-bound MIP support and Python bindings via `pybind11`.

The solver core is implemented as header-only C++ under `include/simplex/`.
The MIP branch-and-bound layer lives in `include/bnb/`, and the Python bindings are defined in `bindings/`.

## Overview

`simplinho` is designed to provide:

- a fast revised simplex engine for bounded LPs
- MIP search with branch-and-bound, cuts, and primal heuristics
- warm starts across repeated solves and node relaxations
- a small Python modeling API for algebraic model construction and solve control

## Highlights

- Primal and dual revised simplex with automatic mode selection
- Crash basis construction with multiple Markowitz-based strategies
- Presolve, scaling, singleton elimination, and bound tightening
- Robust numerics via rook pivoting and iterative refinement
- Branch-and-bound with flexible search policies and branching rules
- Cut generation for Gomory, MIR, cover, implied-bound, clique, odd-cycle, conflict, and dual-proof cuts
- Dedicated root cut repeat and cut pool aging for better bound tightening
- MIP heuristics including diving, feasibility pump, RINS, RENS, local search, and local branching

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

- Revised simplex with primal and dual pivoting
- Phase I fallback when a feasible initial basis is unavailable
- Support for bounded variables, shifted/free variable handling, and slack reformulation
- Bound-flip logic for better dual iterations
- Crash basis heuristics with Markowitz thresholding
- Presolve reduction, scaling, singleton elimination, and bound tightening
- Detailed solver diagnostics including tableau, duals, reduced costs, and Farkas certificates

### Mixed-Integer Programming

- Branch-and-bound on integer and binary variables
- Node selection strategies: depth-first, breadth-first, best-bound, best-estimate, hybrid, best-first plunging, best-estimate plunging, and interleaved best-first/best-estimate
- Branching rules: most-fractional, pseudo-cost, and strong branching
- Diving strategies: fractional, vector-length, objective-value, coefficient, guided, and adaptive
- Heuristics: LP rounding, diving heuristics, feasibility pump, RINS, RENS, local search, and local branching
- Configurable root and node cut budgets

### Cut Engine

- Implied-bound cuts
- Clique cuts and implication-strengthened cliques
- Odd-cycle cuts
- Gomory mixed-integer cuts
- MIR cuts
- Cover cuts
- Conflict cuts with age-based pool management
- Dual-proof cuts
- Cut pool selection tuned by violation, efficacy, density, and type balance

### Numerics and Stability

- Dense Markowitz LU factorization
- Rook pivoting and pivot quality control
- Iterative refinement in forward and transpose solves
- Configurable refactor frequency and matrix compression
- Forrest-Tomlin updates by default, optional eta updates available
- Degeneracy management and anti-cycling support
- Pricing rules: adaptive, Devex, and most-negative

### Presolve and Diagnostics

- Presolve with row/column simplification, scaling, singleton elimination, and bound tightening
- Early detection of infeasible and unbounded models in presolve
- Verbose tracing with optional basis and presolve diagnostics
- Internal tableau exposure, duals, reduced costs, and shadow prices
- Farkas certificate output for infeasibility proofs

## Build

The CMake project currently builds a Python extension named `simplinho`.

### Requirements

- CMake 3.16+
- A C++20 compiler
- Python 3.13 development headers

If local copies of Eigen or `pybind11` are not present in a nearby `_deps` directory, CMake will fetch them with `FetchContent`.

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
