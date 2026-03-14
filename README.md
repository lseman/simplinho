# simplinho

`simplinho` is a standalone revised simplex LP solver with a Python extension module and a small modeling API.

The core solver is header-only C++ in `include/simplex/`, the Python bindings live in `bindings/`, and the top-level build produces a `simplex` module via `pybind11`.

## Highlights

- Primal revised simplex and dual revised simplex in one solver
- Automatic mode selection with `Auto`, `Primal`, and `Dual` modes
- Direct Phase II attempts with Phase I fallback when a feasible basis is not available
- Explicit handling of lower and upper bounds, including automatic reformulation of nonstandard variable bounds
- Dual bound flipping with Beale-style bound-flip ratio logic
- Presolve passes for row reduction, scaling, singleton elimination, bound tightening, dual fixing, and early infeasible/unbounded detection
- Markowitz LU factorization with rook pivoting and iterative refinement
- Forrest-Tomlin basis updates, with eta-stack updates as an alternative
- Multi-attempt crash basis construction with Markowitz-threshold triangularization,
  rotating `hybrid`/`sprint`/`crash_ii`/`crash_iii` heuristics, and
  presolved warm-start repair
- Adaptive pricing, Devex pricing, and most-negative pricing in both primal
  and dual simplex
- Degeneracy management, anti-cycling support, and Harris-style ratio tests
- Rich solve outputs: basis data, reduced costs, dual values, shadow prices, internal tableau data, traces, and Farkas certificates
- A higher-level Python modeling layer with algebraic expressions, named variables, named constraints, and post-solve dual access

## What The Project Exposes

There are two main ways to use the solver:

1. `simplex.RevisedSimplex`
   Use the low-level matrix API directly:
   `solve(A, b, c, l, u)` for `min c^T x` subject to `Ax = b` and `l <= x <= u`.

2. `simplex.Model`
   Use the modeling API with variables, expressions, constraints, and `minimize(...)` / `maximize(...)`.

The Python module also exposes:

- `RevisedSimplexOptions`
- `SimplexMode`
- `LPStatus`
- `status_to_string(...)`

## Solver Features

### Algorithms

- Revised simplex implementation with both primal and dual pivoting
- Automatic fallback behavior when a direct solve needs Phase I work
- Support for bounded variables, free variables, shifted variables, and internally added slacks
- Bound-flip logic for dual iterations when enabled through `dual_allow_bound_flip`

### Numerics

- Dense Markowitz LU factorization
- Rook pivoting for more robust pivot selection
- Iterative refinement in both forward and transpose solves
- Configurable pivot tolerances, refactor frequency, and compression thresholds
- Forrest-Tomlin updates by default, or eta-stack updates via `basis_update = "eta"`

### Pricing And Stability

- `pricing_rule = "adaptive"` for steepest-edge on smaller dual bases and
  Devex on larger ones, with periodic weight resets
- `pricing_rule = "devex"` for Devex pricing
- `pricing_rule = "most_negative"` for a simple reduced-cost rule in primal and
  most-infeasible-row pricing in dual
- `crash_attempts`, `crash_markowitz_tol`, `crash_strategy`, and
  `repair_mapped_basis` tune the initial basis search and post-presolve basis
  repair
- Degeneracy tracking and anti-cycling support
- Harris-style primal and dual ratio tests
- Optional Bland rule toggle

### Presolve And Diagnostics

- Presolve with row/column simplifications, scaling, singleton eliminations, and bound tightening
- Early `infeasible` and `unbounded` detection in presolve when possible
- Verbose tracing with optional basis and presolve details
- Final internal tableau, reduced costs, dual values, and shadow prices on the solved internal model
- Farkas certificate output when infeasibility is proven through the dual path

## Build

The CMake project currently builds a Python extension named `simplex`.

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

That produces a shared object like `build-local/simplex.cpython-313-...so`.

## Low-Level Example

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path("build-local").resolve()))

import simplex

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
print("dual values:", solution.dual_values_internal)
print("reduced costs:", solution.reduced_costs_internal)
```

## Modeling API Example

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("build-local").resolve()))

import simplex

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
print("dual c1  :", c1.pi)
print("dual c2  :", c2.pi)
print("dual c3  :", c3.pi)
```

## Useful Outputs

The low-level `LPSolution` object includes more than just the primal vector:

- `status`, `obj`, `x`, `iters`
- `basis`, `basis_internal`, `nonbasis_internal`
- `tableau`, `tableau_rhs`, `has_internal_tableau`
- `reduced_costs_internal`
- `dual_values_internal`
- `shadow_prices_internal`
- `trace`
- `info`
- `farkas_y`, `farkas_has_cert`

The modeling layer wraps that in `ModelSolution`, while still exposing the raw solve result as `solution.raw`.

## Repository Layout

- `include/simplex/`: header-only solver core, presolve, LU, pricing, and degeneracy helpers
- `bindings/`: `pybind11` bindings and modeling API
- `src/`: reserved for future non-header sources
- `simplex_modeling_api_demo.ipynb`: notebook showing the modeling API in action
- `CMakeLists.txt`: standalone build for the Python module
