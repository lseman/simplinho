#!/usr/bin/env python3
"""Compare the local simplex solver against HiGHS on a batch of LPs.

The script:
1. Loads the locally built extension from `build*/`.
2. Loads `highspy` from the repo virtualenv when available.
3. Solves a mix of fixed and random LP instances with both solvers.
4. Compares solve status, objective value, and basic feasibility metrics.

Examples:
    ./.venv/bin/python testing.py
    ./.venv/bin/python testing.py --random-count 25 --seed 7 --verbose
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def add_repo_venv_to_path() -> None:
    """Make the repo venv importable when running under system Python."""
    candidates = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
    for candidate in candidates:
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def import_simplex_module():
    candidates = [
        ROOT / "build-local",
        ROOT / "build",
        ROOT / "build-verify",
    ]
    module_names = ["simplinho", "simplex"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        built_modules = list(candidate.glob("*.so"))
        if not built_modules:
            continue
        sys.path.insert(0, str(candidate))
        for module_name in module_names:
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
    raise ImportError(
        "Could not find a built solver extension in build-local/, build/, or "
        "build-verify/."
    )


@dataclass
class ProblemCase:
    name: str
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    l: np.ndarray
    u: np.ndarray
    expected: str | None = None


@dataclass
class SolverResult:
    name: str
    status: str
    objective: float
    x: np.ndarray
    primal_residual: float
    bound_violation: float
    iterations: int | None = None
    raw_status: str | None = None


def max_primal_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        return math.inf
    if A.size == 0:
        return 0.0
    return float(np.max(np.abs(A @ x - b))) if b.size else 0.0


def max_bound_violation(l: np.ndarray, u: np.ndarray, x: np.ndarray) -> float:
    if not np.all(np.isfinite(x)):
        return math.inf
    viol = 0.0
    for j, value in enumerate(x):
        if j < l.size and np.isfinite(l[j]):
            viol = max(viol, float(l[j] - value))
        if j < u.size and np.isfinite(u[j]):
            viol = max(viol, float(value - u[j]))
    return max(0.0, viol)


def normalize_simplex_status(status_name: str) -> str:
    key = status_name.strip().lower()
    if "optimal" in key:
        return "optimal"
    if "unbounded" in key:
        return "unbounded"
    if "infeasible" in key:
        return "infeasible"
    if "iter" in key:
        return "iterlimit"
    if "singular" in key:
        return "singular"
    if "phase1" in key:
        return "need_phase1"
    return key


def normalize_highs_status(status_name: str) -> str:
    key = status_name.strip().lower()
    if "optimal" in key:
        return "optimal"
    if "unbounded or infeasible" in key:
        return "ambiguous"
    if "unbounded" in key:
        return "unbounded"
    if "infeasible" in key:
        return "infeasible"
    if "iteration" in key or "time limit" in key:
        return "iterlimit"
    if "empty" in key:
        return "optimal"
    return key


def solve_with_simplex(simplex, case: ProblemCase, mode: str) -> SolverResult:
    options = simplex.RevisedSimplexOptions()
    options.mode = getattr(simplex.SimplexMode, mode)
    options.pricing_rule = "adaptive"
    options.crash_strategy = "hybrid"
    options.crash_attempts = 4

    solver = simplex.RevisedSimplex(options)
    sol = solver.solve(case.A, case.b, case.c, case.l, case.u)
    status_name = simplex.status_to_string(sol.status)
    x = np.array(sol.x, dtype=float, copy=True)
    return SolverResult(
        name=case.name,
        status=normalize_simplex_status(status_name),
        raw_status=status_name,
        objective=float(sol.obj),
        x=x,
        primal_residual=max_primal_residual(case.A, case.b, x),
        bound_violation=max_bound_violation(case.l, case.u, x),
        iterations=int(sol.iters),
    )


def build_highs_lp(highspy, case: ProblemCase):
    m, n = case.A.shape
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)

    mat = highspy.HighsSparseMatrix()
    mat.num_col_ = n
    mat.num_row_ = m
    mat.format_ = highspy.MatrixFormat.kColwise

    start = [0]
    index: list[int] = []
    value: list[float] = []
    for j in range(n):
        nz_rows = np.nonzero(np.abs(case.A[:, j]) > 1e-12)[0]
        index.extend(int(i) for i in nz_rows)
        value.extend(float(case.A[i, j]) for i in nz_rows)
        start.append(len(index))

    mat.start_ = start
    mat.index_ = index
    mat.value_ = value

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = [float(v) for v in case.c]
    lp.col_lower_ = [float(v) for v in case.l]
    lp.col_upper_ = [float(v) for v in case.u]
    lp.row_lower_ = [float(v) for v in case.b]
    lp.row_upper_ = [float(v) for v in case.b]
    lp.a_matrix_ = mat
    return highs, lp


def solve_with_highs(highspy, case: ProblemCase) -> SolverResult:
    highs, lp = build_highs_lp(highspy, case)
    highs.passModel(lp)
    highs.run()

    model_status = highs.getModelStatus()
    raw_status = highs.modelStatusToString(model_status)
    status = normalize_highs_status(raw_status)

    solution = highs.getSolution()
    x = np.array(solution.col_value, dtype=float, copy=True)
    info = highs.getInfo()

    return SolverResult(
        name=case.name,
        status=status,
        raw_status=raw_status,
        objective=float(highs.getObjectiveValue()),
        x=x,
        primal_residual=max_primal_residual(case.A, case.b, x),
        bound_violation=max_bound_violation(case.l, case.u, x),
        iterations=int(getattr(info, "simplex_iteration_count", 0)),
    )


def objective_close(lhs: float, rhs: float, tol: float) -> bool:
    if not (math.isfinite(lhs) and math.isfinite(rhs)):
        return lhs == rhs
    scale = max(1.0, abs(lhs), abs(rhs))
    return abs(lhs - rhs) <= tol * scale


def exercise_model_editing_api(simplex) -> list[str]:
    notes: list[str] = []

    model = simplex.Model()
    x = model.addVar("x", lb=0.0, obj=1.0)
    y = model.addVar("y", lb=0.0, ub=4.0, obj=2.0)

    c1 = model.addConstr(x + y <= 4.0, name="mix")
    c2 = model.addConstr(x <= 3.0, name="cap_x")
    model.maximize(x + 2.0 * y)

    sol1 = model.solve()
    if simplex.status_to_string(sol1.status) != "optimal":
        notes.append("initial model solve was not optimal")
    if not objective_close(sol1.obj, 8.0, 1e-8):
        notes.append(f"initial objective mismatch: expected 8.0, got {sol1.obj:.10g}")
    if not objective_close(sol1.value(y), 4.0, 1e-8):
        notes.append(f"initial y value mismatch: expected 4.0, got {sol1.value(y):.10g}")

    x.obj = 3.0
    if not objective_close(x.obj, 3.0, 1e-12):
        notes.append(f"x.obj property did not update: got {x.obj:.10g}")

    c1.set_coeff(y, 2.0)
    c1.rhs = 5.0
    if not objective_close(c1.get_coeff(y), 2.0, 1e-12):
        notes.append(f"constraint coefficient update failed: got {c1.get_coeff(y):.10g}")
    if not objective_close(c1.rhs, 5.0, 1e-12):
        notes.append(f"constraint rhs update failed: got {c1.rhs:.10g}")
    if not objective_close(model.getObjCoeff(x), 3.0, 1e-12):
        notes.append(
            f"model objective coefficient getter failed: got {model.getObjCoeff(x):.10g}"
        )

    sol2 = model.reoptimize()
    if simplex.status_to_string(sol2.status) != "optimal":
        notes.append("edited model reoptimize() was not optimal")
    if not objective_close(sol2.obj, 11.0, 1e-8):
        notes.append(f"edited objective mismatch: expected 11.0, got {sol2.obj:.10g}")
    if not objective_close(sol2.value(x), 3.0, 1e-8):
        notes.append(f"edited x value mismatch: expected 3.0, got {sol2.value(x):.10g}")
    if not objective_close(sol2.value(y), 1.0, 1e-8):
        notes.append(f"edited y value mismatch: expected 1.0, got {sol2.value(y):.10g}")

    model.deleteConstr(c1)
    if c2.index != 0:
        notes.append(f"surviving constraint did not reindex to 0, got {c2.index}")

    model.deleteVar(x)
    if model.num_vars != 1:
        notes.append(f"expected 1 variable after deletion, got {model.num_vars}")
    if y.name != "y":
        notes.append(f"surviving variable handle did not resolve back to y, got {y.name!r}")

    try:
        _ = x.lb
        notes.append("deleted variable handle remained usable")
    except Exception:
        pass

    sol3 = model.reoptimize()
    if simplex.status_to_string(sol3.status) != "optimal":
        notes.append("post-delete reoptimize() was not optimal")
    if not objective_close(sol3.obj, 8.0, 1e-8):
        notes.append(
            f"post-delete objective mismatch: expected 8.0, got {sol3.obj:.10g}"
        )
    if not objective_close(sol3.value(y), 4.0, 1e-8):
        notes.append(
            f"post-delete y value mismatch: expected 4.0, got {sol3.value(y):.10g}"
        )

    return notes


def exercise_solver_logging_api(simplex) -> list[str]:
    notes: list[str] = []

    options = simplex.RevisedSimplexOptions()
    options.verbose = True
    options.verbose_every = 1
    options.mode = simplex.SimplexMode.Auto

    model = simplex.Model(options)
    x = model.addVar("x", lb=0.0)
    y = model.addVar("y", lb=0.0)
    model.addConstr(x + y <= 4.0, name="cap")
    model.maximize(x + 2.0 * y)

    sol = model.solve()
    stats = sol.stats

    if simplex.status_to_string(sol.status) != "optimal":
        notes.append("logging model solve was not optimal")
    if stats.status != "optimal":
        notes.append(f"stats.status mismatch: expected 'optimal', got {stats.status!r}")
    if stats.iterations != sol.iters:
        notes.append(f"stats.iterations mismatch: expected {sol.iters}, got {stats.iterations}")
    if stats.phase2_iterations > stats.iterations:
        notes.append("phase2_iterations exceeded total iterations")
    if stats.trace_lines != len(sol.log_lines):
        notes.append(
            f"trace line count mismatch: stats={stats.trace_lines} "
            f"log_lines={len(sol.log_lines)}"
        )
    if stats.trace_lines <= 0:
        notes.append("expected verbose solve to emit at least one trace line")
    if "[solve] start" not in sol.log:
        notes.append("expected joined log text to contain '[solve] start'")
    if "raw_info" not in stats.as_dict():
        notes.append("SolveStats.as_dict() did not include raw_info")
    if not isinstance(stats.raw_info, dict):
        notes.append("SolveStats.raw_info was not exposed as a dict")

    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([4.0], dtype=float)
    c = np.array([1.0, 2.0], dtype=float)
    l = np.array([0.0, 0.0], dtype=float)
    u = np.array([np.inf, np.inf], dtype=float)
    solver = simplex.RevisedSimplex(options)
    raw = solver.solve(A, b, c, l, u)
    raw_stats = raw.stats
    if raw_stats.trace_lines != len(raw.log_lines):
        notes.append("LPSolution trace_lines did not match low-level log length")
    if "[solve] start" not in raw.log:
        notes.append("expected low-level joined log text to contain '[solve] start'")

    return notes


def exercise_basis_warm_start_api(simplex) -> list[str]:
    notes: list[str] = []

    options = simplex.RevisedSimplexOptions()
    options.mode = simplex.SimplexMode.Auto

    model = simplex.Model(options)
    x = model.addVar("x", lb=0.0, ub=5.0)
    y = model.addVar("y", lb=0.0, ub=5.0)
    model.addConstr(x + y <= 6.0, name="cap")
    model.maximize(4.0 * x + 3.0 * y)

    sol1 = model.solve()
    basis = sol1.basis
    if simplex.status_to_string(sol1.status) != "optimal":
        notes.append("initial warm-start model solve was not optimal")
        return notes
    if basis.num_columns != 3:
        notes.append(f"expected model basis to span 3 columns, got {basis.num_columns}")
    if len(basis.basic_columns) != 1:
        notes.append(
            f"expected 1 basic column for the single-row model, got {len(basis.basic_columns)}"
        )

    model.options.mode = simplex.SimplexMode.Dual
    x.ub = 1.5
    sol2 = model.reoptimize()
    if simplex.status_to_string(sol2.status) != "optimal":
        notes.append("automatic dual warm-start reoptimize() was not optimal")
    if sol2.value(x) > 1.5 + 1e-8:
        notes.append(f"automatic reoptimize() violated tightened upper bound: {sol2.value(x):.10g}")
    if sol2.stats.basis_start not in {"warm_start", "repaired_warm_start"}:
        notes.append(
            f"expected automatic warm-start basis source, got {sol2.stats.basis_start!r}"
        )

    x.ub = 1.0
    sol3 = model.reoptimize(basis)
    if simplex.status_to_string(sol3.status) != "optimal":
        notes.append("explicit dual warm-start reoptimize() was not optimal")
    if sol3.value(x) > 1.0 + 1e-8:
        notes.append(f"explicit basis reoptimize() violated tightened upper bound: {sol3.value(x):.10g}")
    if sol3.stats.basis_start not in {"warm_start", "repaired_warm_start"}:
        notes.append(
            f"expected explicit warm-start basis source, got {sol3.stats.basis_start!r}"
        )

    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([6.0], dtype=float)
    c = np.array([-4.0, -3.0], dtype=float)
    l = np.array([0.0, 0.0], dtype=float)
    u = np.array([5.0, 5.0], dtype=float)

    solver = simplex.RevisedSimplex(options)
    raw1 = solver.solve(A, b, c, l, u)
    if simplex.status_to_string(raw1.status) != "optimal":
        notes.append("initial low-level warm-start solve was not optimal")
        return notes

    dual_options = simplex.RevisedSimplexOptions()
    dual_options.mode = simplex.SimplexMode.Dual
    dual_solver = simplex.RevisedSimplex(dual_options)
    dual_raw1 = dual_solver.solve(A, b, c, l, u)
    if simplex.status_to_string(dual_raw1.status) != "optimal":
        notes.append("initial persistent dual solve was not optimal")
    else:
        u_auto = np.array([1.75, 5.0], dtype=float)
        dual_raw2 = dual_solver.solve(A, b, c, l, u_auto)
        if simplex.status_to_string(dual_raw2.status) != "optimal":
            notes.append("persistent dual auto-reuse solve was not optimal")
        if dual_raw2.stats.basis_start not in {"warm_start", "repaired_warm_start"}:
            notes.append(
                "expected persistent dual solve to reuse cached basis, "
                f"got {dual_raw2.stats.basis_start!r}"
            )
        if dual_raw2.x[0] > 1.75 + 1e-8:
            notes.append(
                f"persistent dual x[0] violated tightened upper bound: {dual_raw2.x[0]:.10g}"
            )

    u2 = np.array([1.5, 5.0], dtype=float)
    raw2 = solver.solve(A, b, c, l, u2, raw1.basis_state)
    if simplex.status_to_string(raw2.status) != "optimal":
        notes.append("low-level warm-start solve after bound change was not optimal")
    if raw2.stats.basis_start not in {"warm_start", "repaired_warm_start"}:
        notes.append(
            f"expected low-level warm-start basis source, got {raw2.stats.basis_start!r}"
        )
    if raw2.x[0] > 1.5 + 1e-8:
        notes.append(f"low-level x[0] violated tightened upper bound: {raw2.x[0]:.10g}")

    return notes


def compare_results(
    case: ProblemCase,
    ours: SolverResult,
    highs: SolverResult | None,
    tol: float,
) -> tuple[bool, list[str]]:
    notes: list[str] = []

    if case.expected and ours.status != case.expected:
        notes.append(
            f"simplex status {ours.status!r} != expected {case.expected!r}"
        )

    if highs is None:
        if ours.status == "optimal":
            if ours.primal_residual > 50.0 * tol:
                notes.append(
                    f"simplex primal residual too large ({ours.primal_residual:.3e})"
                )
            if ours.bound_violation > 50.0 * tol:
                notes.append(
                    f"simplex bound violation too large ({ours.bound_violation:.3e})"
                )
        return (len(notes) == 0), notes

    comparable_statuses = {ours.status, highs.status}
    if comparable_statuses == {"infeasible", "ambiguous"}:
        pass
    elif ours.status != highs.status:
        notes.append(
            f"status mismatch simplex={ours.status!r} highs={highs.status!r}"
        )

    if ours.status == "optimal" and highs.status == "optimal":
        if not objective_close(ours.objective, highs.objective, tol):
            notes.append(
                "objective mismatch "
                f"simplex={ours.objective:.10g} highs={highs.objective:.10g}"
            )
        if ours.primal_residual > 50.0 * tol:
            notes.append(
                f"simplex primal residual too large ({ours.primal_residual:.3e})"
            )
        if highs.primal_residual > 50.0 * tol:
            notes.append(
                f"HiGHS primal residual too large ({highs.primal_residual:.3e})"
            )
        if ours.bound_violation > 50.0 * tol:
            notes.append(
                f"simplex bound violation too large ({ours.bound_violation:.3e})"
            )
        if highs.bound_violation > 50.0 * tol:
            notes.append(
                f"HiGHS bound violation too large ({highs.bound_violation:.3e})"
            )

    return (len(notes) == 0), notes


def finite_or_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def deterministic_cases() -> list[ProblemCase]:
    inf = float("inf")
    cases: list[ProblemCase] = []

    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([4.0], dtype=float)
    c = np.array([1.0, 2.0], dtype=float)
    l = np.array([0.0, 0.0], dtype=float)
    u = np.array([inf, inf], dtype=float)
    cases.append(ProblemCase("simple_feasible", A, b, c, l, u, "optimal"))

    A = np.array([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]], dtype=float)
    b = np.array([1.0, 1.0], dtype=float)
    c = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
    l = np.zeros(4, dtype=float)
    u = np.array([inf, inf, inf, inf], dtype=float)
    cases.append(ProblemCase("degenerate_slack", A, b, c, l, u, "optimal"))

    A = np.array([[1.0, 2.0, -1.0], [0.0, 1.0, 1.0]], dtype=float)
    x_star = np.array([2.5, 1.5, 0.5], dtype=float)
    b = A @ x_star
    c = np.array([0.5, -1.0, 0.25], dtype=float)
    l = np.array([1.0, 0.5, 0.0], dtype=float)
    u = np.array([5.0, 2.5, inf], dtype=float)
    cases.append(ProblemCase("shifted_bounds", A, b, c, l, u, "optimal"))

    A = np.array([[1.0]], dtype=float)
    b = np.array([2.0], dtype=float)
    c = np.array([1.0], dtype=float)
    l = np.array([0.0], dtype=float)
    u = np.array([1.0], dtype=float)
    cases.append(ProblemCase("infeasible_bound_conflict", A, b, c, l, u, "infeasible"))

    A = np.array([[1.0, -1.0]], dtype=float)
    b = np.array([0.0], dtype=float)
    c = np.array([-1.0, -1.0], dtype=float)
    l = np.array([0.0, 0.0], dtype=float)
    u = np.array([inf, inf], dtype=float)
    cases.append(ProblemCase("unbounded_ray", A, b, c, l, u, "unbounded"))

    return cases


def sample_bounds(
    rng: np.random.Generator, n: int, require_finite_box: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    inf = float("inf")
    l = np.empty(n, dtype=float)
    u = np.empty(n, dtype=float)
    for j in range(n):
        if require_finite_box:
            lower = float(rng.uniform(-3.0, 2.0))
            width = float(rng.uniform(0.5, 6.0))
            l[j], u[j] = lower, lower + width
            continue

        mode = int(rng.integers(0, 5))
        if mode == 0:
            l[j], u[j] = 0.0, inf
        elif mode == 1:
            upper = float(rng.uniform(0.5, 6.0))
            l[j], u[j] = 0.0, upper
        elif mode == 2:
            lower = float(rng.uniform(-3.0, 2.0))
            l[j], u[j] = lower, inf
        elif mode == 3:
            lower = float(rng.uniform(-3.0, 2.0))
            width = float(rng.uniform(0.5, 5.0))
            l[j], u[j] = lower, lower + width
        else:
            l[j], u[j] = -inf, inf
    return l, u


def sample_feasible_point(
    rng: np.random.Generator, l: np.ndarray, u: np.ndarray
) -> np.ndarray:
    x = np.empty_like(l, dtype=float)
    for j in range(l.size):
        has_l = np.isfinite(l[j])
        has_u = np.isfinite(u[j])
        if has_l and has_u:
            span = max(1e-6, u[j] - l[j])
            x[j] = float(rng.uniform(l[j] + 0.15 * span, u[j] - 0.15 * span))
        elif has_l:
            x[j] = float(l[j] + abs(rng.normal(loc=1.0, scale=1.0)))
        elif has_u:
            x[j] = float(u[j] - abs(rng.normal(loc=1.0, scale=1.0)))
        else:
            x[j] = float(rng.normal())
    return x


def random_feasible_case(rng: np.random.Generator, idx: int) -> ProblemCase:
    for _ in range(100):
        m = int(rng.integers(2, 7))
        n = int(rng.integers(m + 1, m + 7))
        A = rng.normal(size=(m, n))
        mask = rng.random(size=(m, n)) < 0.35
        A[mask] = 0.0

        for i in range(m):
            if np.all(np.abs(A[i]) <= 1e-12):
                A[i, int(rng.integers(0, n))] = float(rng.normal())
        for j in range(n):
            if np.all(np.abs(A[:, j]) <= 1e-12):
                A[int(rng.integers(0, m)), j] = float(rng.normal())

        if np.linalg.matrix_rank(A) < m:
            continue

        l, u = sample_bounds(rng, n, require_finite_box=True)
        x_star = sample_feasible_point(rng, l, u)
        b = A @ x_star
        c = rng.normal(size=n)
        return ProblemCase(
            name=f"random_feasible_{idx:02d}",
            A=A.astype(float),
            b=b.astype(float),
            c=c.astype(float),
            l=l.astype(float),
            u=u.astype(float),
            expected="optimal",
        )
    raise RuntimeError("failed to generate a full-row-rank random feasible case")


def make_cases(random_count: int, seed: int) -> list[ProblemCase]:
    rng = np.random.default_rng(seed)
    cases = deterministic_cases()
    cases.extend(random_feasible_case(rng, i) for i in range(random_count))
    return cases


def format_result_line(
    case: ProblemCase,
    ours: SolverResult,
    highs: SolverResult | None,
    passed: bool,
) -> str:
    mark = "PASS" if passed else "FAIL"
    ours_obj = f"{ours.objective:.8g}" if math.isfinite(ours.objective) else str(ours.objective)
    if highs is None:
        return (
            f"{mark:4}  {case.name:24} "
            f"simplex={ours.status:12} obj={ours_obj:>12}"
        )
    highs_obj = (
        f"{highs.objective:.8g}" if math.isfinite(highs.objective) else str(highs.objective)
    )
    return (
        f"{mark:4}  {case.name:24} "
        f"simplex={ours.status:12} highs={highs.status:12} "
        f"obj=({ours_obj:>12}, {highs_obj:>12})"
    )


def run_suite(args: argparse.Namespace) -> int:
    add_repo_venv_to_path()
    simplex = import_simplex_module()

    try:
        import highspy  # type: ignore
    except ImportError:
        highspy = None

    cases = make_cases(args.random_count, args.seed)
    failures = 0

    print(
        f"Loaded simplex from {Path(simplex.__file__).resolve()}",
        flush=True,
    )
    if highspy is None:
        print("highspy not available; running simplex-only feasibility checks.", flush=True)
    else:
        print(f"Loaded highspy {getattr(highspy, '__version__', 'unknown')}", flush=True)
    edit_notes = exercise_model_editing_api(simplex)
    edit_passed = len(edit_notes) == 0
    print(
        f"{'PASS' if edit_passed else 'FAIL':4}  {'model_editing_api':24} "
        f"live edits, deletes, and reoptimize"
    )
    if args.verbose or not edit_passed:
        for note in edit_notes:
            print(f"      - {note}")
    if not edit_passed:
        return 1
    logging_notes = exercise_solver_logging_api(simplex)
    logging_passed = len(logging_notes) == 0
    print(
        f"{'PASS' if logging_passed else 'FAIL':4}  {'solver_logging_api':24} "
        f"typed stats and joined logs"
    )
    if args.verbose or not logging_passed:
        for note in logging_notes:
            print(f"      - {note}")
    if not logging_passed:
        return 1
    warm_start_notes = exercise_basis_warm_start_api(simplex)
    warm_start_passed = len(warm_start_notes) == 0
    print(
        f"{'PASS' if warm_start_passed else 'FAIL':4}  {'basis_warm_start_api':24} "
        f"basis export and dual reoptimize"
    )
    if args.verbose or not warm_start_passed:
        for note in warm_start_notes:
            print(f"      - {note}")
    if not warm_start_passed:
        return 1
    print(f"Running {len(cases)} problem(s) with mode={args.mode}", flush=True)
    print()

    for case in cases:
        ours = solve_with_simplex(simplex, case, args.mode)
        highs = solve_with_highs(highspy, case) if highspy is not None else None
        passed, notes = compare_results(case, ours, highs, args.tol)
        if not passed:
            failures += 1

        print(format_result_line(case, ours, highs, passed))
        if args.verbose or (not passed and notes):
            for note in notes:
                print(f"      - {note}")
            print(
                f"      - simplex residual={ours.primal_residual:.3e} "
                f"bound_violation={ours.bound_violation:.3e} "
                f"iters={ours.iterations}"
            )
            if highs is not None:
                print(
                    f"      - highs   residual={highs.primal_residual:.3e} "
                    f"bound_violation={highs.bound_violation:.3e} "
                    f"iters={highs.iterations}"
                )

    print()
    passed_cases = len(cases) - failures
    print(f"Summary: {passed_cases}/{len(cases)} passed")
    if failures:
        print(f"Failures: {failures}")
        return 1
    return 0


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the local simplex solver against HiGHS."
    )
    parser.add_argument(
        "--random-count",
        type=int,
        default=10,
        help="Number of random feasible LPs to add to the fixed test set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for generated LPs.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for objective comparisons.",
    )
    parser.add_argument(
        "--mode",
        choices=["Auto", "Primal", "Dual"],
        default="Auto",
        help="Simplex mode to use for the local solver.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print residuals and comparison notes for every case.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    return run_suite(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
