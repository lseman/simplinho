# Bug Discoveries

## BUG-001: Doubly-bounded LP returns Singular via sparse bound-reformulation path

### Problem

`RevisedSimplex.solve(A, b, c, l, u)` returns `Singular` when **all variables have finite upper bounds** and the problem has **more than one constraint row**. The same problem without upper bounds solves correctly.

```
simplex status: Singular
simplex x:      [0. 0.]
pulp status:    Optimal
pulp obj:       -27.333...
pulp x:         [2.6667, 4.6667]
```

### Test case

```python
import numpy as np
import simplinho as splx

A = np.array([[1, 2],
              [2, 1]], dtype=float)
b = np.array([12.0, 10.0])
c = np.array([-3.0, -5.0])
l = np.array([0.0, 0.0])
u = np.array([10.0, 10.0])   # <-- finite upper bounds trigger the bug

solver = splx.RevisedSimplex()
sol = solver.solve(A, b, c, l, u)
print("simplex status:", sol.status)   # Singular (wrong)
print("simplex x:     ", sol.x)
```

### Comparison with PuLP

```python
import pulp

prob = pulp.LpProblem("bug001", pulp.LpMinimize)
x0 = pulp.LpVariable("x0", lowBound=0, upBound=10)
x1 = pulp.LpVariable("x1", lowBound=0, upBound=10)
prob += -3*x0 - 5*x1
prob += x0 + 2*x1 <= 12
prob += 2*x0 + x1 <= 10
prob.solve(pulp.PULP_CBC_CMD(msg=0))
print("pulp status:", pulp.LpStatus[prob.status])   # Optimal
print("pulp obj:   ", pulp.value(prob.objective))   # -27.333...
print("pulp x:     ", [pulp.value(x0), pulp.value(x1)])  # [2.667, 4.667]
```

### Key observations

1. **Without upper bounds** (`u = [inf, inf]`) → `Optimal, x=[2.667, 4.667]` ✓
2. **With upper bounds** (`u = [10, 10]`) → `Singular` ✗
3. Info shows `bound_reformulation_warm_start_valid=1` and `basis_start_style=mapped` — the mapped warm-start basis is accepted as primal+dual feasible but the inner primal solve still returns Singular.
4. The reformulated system is 4×4 (`reduced_m=4, reduced_n=4`) — two upper-bound slack rows added.
5. Solving the reformulated 4×4 system **directly** (bypassing the reformulation path) gives the correct answer.

### Root cause (suspected)

The sparse bound-reformulation path (`solve_impl_sparse_`) maps the warm-start `basis_state_opt` (size 2, original space) to the reformulated space (size 4). The mapped `basis_state_std` passes the quality check but causes the inner `RevisedSimplex` to fail with Singular. Likely a sign/index error in `map_reformulated_basis_state_` or `map_reformulated_basis_seed_state_` when `m_eq > 1` and all variables are doubly-bounded.

### Files involved

- `include/simplex/simplex.h` — `solve_impl_sparse_()` sparse bound-reformulation path (around line 3382–3400)
- `include/simplex/simplex.h` — `map_reformulated_basis_state_()` and `map_reformulated_basis_seed_state_()`
- `include/simplex/simplex_reformulation.h` — reformulation helpers

### Workaround

Convert finite upper bounds to explicit constraints before calling `solve()`:

```python
# Add x_i <= u_i as explicit rows
import scipy.sparse as sp

n = A.shape[1]
A_aug = np.vstack([A, np.eye(n)])
b_aug = np.concatenate([b, u])
l_aug = l
u_aug = np.full(n, np.inf)

sol = solver.solve(A_aug, b_aug, c, l_aug, u_aug)
```
