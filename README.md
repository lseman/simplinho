<p align="center">
  <img src="assets/simplinho-logo.svg" alt="simplinho logo" width="720">
</p>

# simplinho

`simplinho` is an experimental C++23 revised-simplex solver with Python
bindings. It is designed for fast repeated bounded LP solves, warm starts,
transparent numerical diagnostics, and research on practical simplex methods.

The optional branch-and-bound solver is a separate downstream component. It
depends on the simplex library; the simplex library never depends on BnB.

> [!NOTE]
> This is research software. The public API, numerical policies, and packaging
> may change while the solver is under active development.

## Highlights

- primal and dual revised simplex with automatic strategy selection
- native lower and upper bounds, free-variable reformulation, and Phase I
- sparse Markowitz factorization with Forrest–Tomlin update chains
- synthetic-clock reinversion, residual checks, and iterative refinement
- dual steepest-edge and Devex pricing with accuracy-driven switching
- Harris ratio tests and batched bound-flipping ratio tests (BFRT)
- crash bases, presolve, scaling, perturbation, and anti-cycling controls
- explicit and cached basis warm starts for repeated solves
- primal rays, Farkas certificates, dual values, reduced costs, and tableaux
- typed timing, factorization, pricing, pivot, and warm-start telemetry

## Components

The dependency direction is deliberately one-way:

```text
simplinho_bnb (Python) ──> simplinho (Python)
        │                       │
        v                       v
    bnb::core ───────────> simplex_core
```

| Component | Purpose |
| --- | --- |
| `simplex_core` | Public header-only simplex interface |
| `simplinho` | Simplex runtime, Python bindings, and LP modeling API |
| `bnb::core` | Optional compiled branch-and-bound library |
| `simplinho_bnb` | Optional, separate BnB Python bindings |

C++ simplex consumers include headers below `<simplex/...>`. BnB consumers
include `<bnb/core.h>` and link `bnb::core`. The former
`<simplex/bnb.h>` reverse-dependency shim no longer exists.

## Build

### Requirements

- CMake 3.16+
- a C++23 compiler
- Python 3.14 development files for the current CMake configuration

CMake obtains Eigen, pybind11, and `ankerl::unordered_dense` when local copies
are not configured. BnB builds additionally obtain `stdexec`.

### Simplex only

```bash
cmake -S . -B build-local \
  -DSIMPLEX_ENABLE_BNB=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j
```

### Simplex and BnB

```bash
cmake -S . -B build-local \
  -DSIMPLEX_ENABLE_BNB=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-local -j
```

This produces `simplinho*.so` and, when enabled, `simplinho_bnb*.so`.

Useful CMake options include:

| Option | Default | Meaning |
| --- | ---: | --- |
| `SIMPLEX_ENABLE_BNB` | `OFF` | Build the downstream BnB library/module |
| `SIMPLEX_ENABLE_PERF_FLAGS` | `ON` | Enable optimized compiler/linker flags |
| `SIMPLEX_ENABLE_CCACHE` | `ON` | Use ccache or sccache when available |
| `SIMPLEX_ENABLE_UNITY_BUILD` | `ON` | Enable unity builds |
| `SIMPLEX_ENABLE_PCH` | `OFF` | Enable precompiled headers |

## Low-level Python API

`RevisedSimplex` accepts dense NumPy arrays or SciPy sparse matrices.

```python
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path("build-local").resolve()))
import simplinho as sx

A = sp.csc_matrix([[1.0, 1.0]])
b = np.array([4.0])
c = np.array([1.0, 2.0])
lower = np.array([0.0, 0.0])
upper = np.array([np.inf, np.inf])

options = sx.RevisedSimplexOptions()
options.mode = sx.SimplexMode.Auto
options.pricing_rule = "adaptive"

solver = sx.RevisedSimplex(options)
solution = solver.solve(A, b, c, lower, upper)

print(sx.status_to_string(solution.status))
print(solution.obj, solution.x)
print(solution.stats.as_dict())
```

### Warm starts

An explicit `LPBasis` can be reused after compatible model changes:

```python
basis = solution.basis_state
upper2 = np.array([1.5, np.inf])
warm = solver.solve(A, b, c, lower, upper2, basis=basis)

print(warm.stats.basis_start)
print(warm.stats.warm_factorization_reused)
```

Keeping the same solver instance also enables its internal compatible-basis
cache. Dual mode is generally the natural choice for repeated bound changes.

## Modeling API

The `Model` interface provides algebraic LP construction and live edits:

```python
import simplinho as sx

model = sx.Model()
x = model.add_var("x", lb=0.0)
y = model.add_var("y", lb=0.0)

cap_x = model.add_constr(x <= 3.0, name="cap_x")
model.add_constr(y <= 2.0, name="cap_y")
model.add_constr(x + 2.0 * y <= 6.0, name="mix")
model.maximize(x + y)

solution = model.solve()
print(solution.obj, solution.value(x), solution.value("y"))
print(cap_x.pi)

# Compatible edits can reuse the previous basis.
x.ub = 1.5
solution = model.reoptimize(solution.basis)
```

The simplex module intentionally exposes LP solving only. MIP orchestration is
owned by the separate BnB component and should not be added back to the
`simplinho` module through conditional bindings.

## Solver outputs

`LPSolution` exposes:

- `status`, `obj`, `x`, and `iters`
- `basis_state`, internal basis/nonbasis indices, and warm-start metadata
- dual values, shadow prices, and reduced costs
- optional internal tableau and tableau RHS
- Farkas certificates for infeasibility
- primal rays for unboundedness
- trace lines and structured `info`
- typed `stats`, including LU builds, update depth, pricing frameworks, pivots,
  density estimates, and warm-state reuse

The modeling layer returns `ModelSolution` and keeps the low-level result in
`solution.raw`.

## Numerical strategy

The implementation takes inspiration from mature simplex solvers—especially
HiGHS—while retaining native data structures and independently tested logic.
Important policies include:

- update-age-dependent pivot eligibility
- batched BFRT updates with one FTRAN for all accepted bound flips
- sticky DSE-to-Devex framework switching
- O(rows) pricing-framework resets
- density-aware sparse/dense FTRAN and BTRAN selection
- synthetic-work reinversion with residual and growth safeguards
- cost perturbation cleanup before declaring optimality

The local HiGHS source under `third_party/highs-source/` is a design and
benchmark reference, not a linked runtime dependency of `simplex_core`.

## Testing and benchmarking

Run the project tests against a specific build artifact:

```bash
SIMPLINHO_BUILD_DIR="$PWD/build-local" python -m pytest -q tests
```

Run the quick parity/performance comparison with HiGHS:

```bash
SIMPLINHO_BUILD_DIR="$PWD/build-local" \
  python tests/bench_sparse_vs_highs.py --quick
```

The benchmark checks status and objective parity and reports iterations,
refactorizations, update count, and the LU/pricing/pivot timing split. Timings
are machine-dependent; parity and regression trends matter more than a single
headline number.

Optional diagnostics:

```bash
SIMPLINHO_TRACE_POOL_REBUILDS=1  # pricing framework rebuilds
SIMPLINHO_TRACE_REFACTORS=1      # factorization rebuild causes
```

## Repository layout

```text
include/simplex/        simplex API and implementation
  core/                 sparse vectors and shared utilities
  engine/               primal/dual engines and ratio/pricing logic
  factorization/        crash, LU, and update-chain implementations
  nla/                  numerical linear algebra orchestration
  presolve/             LP presolve and postsolve data
  types/                public options and result types
include/bnb/            independent branch-and-bound component
src/nla/                compiled simplex NLA implementation
src/bnb/                compiled BnB implementation
bindings/               separate simplex and BnB Python entry points
tests/                  correctness and HiGHS parity benchmarks
third_party/highs-source/ local algorithm/reference source tree
```

## Development principles

- preserve the dependency direction: BnB imports simplex
- measure before changing hot-path policy
- keep numerical fallbacks and exact verification paths
- compare status, objective, certificates, and iteration behavior—not only time
- keep performance decisions visible through typed telemetry

## License

See [LICENSE](LICENSE).
