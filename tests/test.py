# %% [markdown]
# # Compare `simplex` vs PuLP + HiGHS
#
# This notebook exercises the locally built `simplex` extension in `build/` and compares it against PuLP with the HiGHS backend.
#
# Important: use `np.inf` for variables that are truly unbounded above. Passing a very large finite number like `1e20` does not mean infinity to this solver; it triggers the finite-bound reformulation path, which is currently numerically unstable for very large inactive bounds.

# %%
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

BUILD_DIR = Path.cwd()
# go up one dir
BUILD_DIR = BUILD_DIR.parent

print(BUILD_DIR)

# add /build to path
BUILD_DIR = BUILD_DIR / "build"
print(f"Looking for simplinho in {BUILD_DIR}")
if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))

import simplinho

print("build dir:", BUILD_DIR)
print("simplex module:", simplinho.__file__)
print("PuLP version:", pulp.__version__)

# %%
INF = np.inf
BIG_FINITE_BOUND = 1.0e20


def normalize_vector(x, *, dtype=float):
    return np.asarray(x, dtype=dtype).reshape(-1)


def normalize_matrix(A):
    return np.asarray(A, dtype=float)


def lp_case(name, A, b, c, l, u):
    A = normalize_matrix(A)
    b = normalize_vector(b)
    c = normalize_vector(c)
    l = normalize_vector(l)
    u = normalize_vector(u)
    assert A.shape[0] == b.shape[0], (A.shape, b.shape)
    assert A.shape[1] == c.shape[0], (A.shape, c.shape)
    assert l.shape == c.shape == u.shape
    return {
        "name": name,
        "A": A,
        "b": b,
        "c": c,
        "l": l,
        "u": u,
    }


def status_name(value):
    try:
        return value.name
    except AttributeError:
        return str(value)


def solve_with_simplex(case, *, mode="Auto"):
    options = simplinho.RevisedSimplexOptions()
    options.mode = getattr(simplinho.SimplexMode, mode)
    solver = simplinho.RevisedSimplex(options)
    sol = solver.solve(case["A"], case["b"], case["c"], case["l"], case["u"])
    return {
        "status": status_name(sol.status),
        "obj": float(sol.obj),
        "x": np.array(sol.x, dtype=float),
        "iters": int(sol.iters),
        "basis": list(sol.basis),
        "info": dict(sol.info),
    }


def solve_with_pulp_highs(case, *, msg=False):
    n = case["c"].shape[0]
    prob = pulp.LpProblem(case["name"], pulp.LpMinimize)
    xs = []

    for i in range(n):
        lb = None if np.isneginf(case["l"][i]) else float(case["l"][i])
        ub = None if np.isposinf(case["u"][i]) else float(case["u"][i])
        xs.append(pulp.LpVariable(f"x_{i}", lowBound=lb, upBound=ub))

    prob += pulp.lpSum(float(case["c"][i]) * xs[i] for i in range(n))

    for row_idx in range(case["A"].shape[0]):
        expr = pulp.lpSum(float(case["A"][row_idx, j]) * xs[j] for j in range(n))
        prob += expr == float(case["b"][row_idx]), f"eq_{row_idx}"

    solver = pulp.HiGHS(msg=msg)
    prob.solve(solver)

    return {
        "status": pulp.LpStatus[prob.status],
        "obj": float(pulp.value(prob.objective)),
        "x": np.array([pulp.value(v) for v in xs], dtype=float),
    }


def compare_case(case, *, mode="Auto", atol=1e-7):
    simplex_res = solve_with_simplex(case, mode=mode)
    pulp_res = solve_with_pulp_highs(case)

    x_diff = simplex_res["x"] - pulp_res["x"]
    obj_diff = simplex_res["obj"] - pulp_res["obj"]
    feasible_eq_resid = case["A"] @ simplex_res["x"] - case["b"]

    return {
        "case": case["name"],
        "simplex_status": simplex_res["status"],
        "pulp_status": pulp_res["status"],
        "simplex_obj": simplex_res["obj"],
        "pulp_obj": pulp_res["obj"],
        "obj_abs_diff": abs(obj_diff),
        "x_inf_diff": float(np.max(np.abs(x_diff))) if x_diff.size else 0.0,
        "simplex_eq_resid_inf": float(np.max(np.abs(feasible_eq_resid)))
        if feasible_eq_resid.size
        else 0.0,
        "simplex_iters": simplex_res["iters"],
        "matches_status": simplex_res["status"].lower() == pulp_res["status"].lower(),
        "close_obj": math.isfinite(simplex_res["obj"])
        and math.isfinite(pulp_res["obj"])
        and abs(obj_diff) <= atol,
        "close_x": np.allclose(simplex_res["x"], pulp_res["x"], atol=atol, rtol=0.0),
        "simplex_result": simplex_res,
        "pulp_result": pulp_res,
    }


def compare_big_bound_sweep(bounds):
    rows = []
    base = lp_case(
        "two_var_budget_big_bound",
        A=[[1.0, 1.0]],
        b=[1.0],
        c=[1.0, 2.0],
        l=[0.0, 0.0],
        u=[INF, INF],
    )

    for ub in bounds:
        case = dict(base)
        case["u"] = np.array([ub, ub], dtype=float)
        sx = solve_with_simplex(case)
        rows.append(
            {
                "ub": ub,
                "status": sx["status"],
                "obj": sx["obj"],
                "x0": sx["x"][0],
                "x1": sx["x"][1],
                "iters": sx["iters"],
            }
        )

    return pd.DataFrame(rows)


# %%
cases = [
    lp_case(
        "two_var_budget",
        A=[[1.0, 1.0]],
        b=[1.0],
        c=[1.0, 2.0],
        l=[0.0, 0.0],
        u=[INF, INF],
    ),
    lp_case(
        "transport_like",
        A=[
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ],
        b=[4.0, 3.0],
        c=[3.0, 1.0, 2.0],
        l=[0.0, 0.0, 0.0],
        u=[INF, INF, INF],
    ),
    lp_case(
        "bounded_box",
        A=[[1.0, -1.0, 1.0]],
        b=[2.0],
        c=[-1.0, 0.5, 2.0],
        l=[0.0, 0.0, 0.0],
        u=[3.0, 3.0, 3.0],
    ),
]

len(cases)

# %%
results = [compare_case(case) for case in cases]

summary = pd.DataFrame(
    [
        {k: v for k, v in item.items() if k not in {"simplex_result", "pulp_result"}}
        for item in results
    ]
)

summary

# %%
# Optional stress test: large finite inactive upper bounds eventually destabilize the current reformulation path.
compare_big_bound_sweep([1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e16, BIG_FINITE_BOUND])

# %%
for item in results:
    print("=" * 80)
    print(item["case"])
    print("simplex status:", item["simplex_status"])
    print("pulp status   :", item["pulp_status"])
    print("simplex obj   :", item["simplex_obj"])
    print("pulp obj      :", item["pulp_obj"])
    print("simplex x     :", item["simplex_result"]["x"])
    print("pulp x        :", item["pulp_result"]["x"])
    print("x inf diff    :", item["x_inf_diff"])
    print("eq resid inf  :", item["simplex_eq_resid_inf"])
    print("simplex info  :", item["simplex_result"]["info"])

# %% [markdown]
# ## Notes
#
# - For baseline comparisons, prefer `np.inf` in `simplex` and `None` bounds in PuLP rather than `1e20`-style big-M bounds.
# - A fresh shell repro shows the baseline cases match when the upper bounds are truly infinite.
# - The current solver starts drifting around `1e8` to `1e10` and can return `Singular` or even `Infeasible` once very large finite inactive upper bounds force the reformulation path.
# - If we want to harden the solver itself, the next step is either to canonicalize huge inactive bounds to `+inf` before presolve or to debug the finite-bound reformulation/presolve numerics directly.
