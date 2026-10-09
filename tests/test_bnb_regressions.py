"""Correctness regressions for cuts, warm starts, and parallel B&B."""

import os
import sys
from pathlib import Path

import numpy as np

BUILD_DIR = Path(os.environ.get("SIMPLINHO_BUILD_DIR", "build")).resolve()
sys.path.insert(0, str(BUILD_DIR))

import simplinho_bnb as snb  # noqa: E402
from simplinho import Model  # noqa: E402


def solve_binary_mip(c, A, b, *, workers=1, configure=None):
    model = Model()
    variables = [
        model.add_var(f"x{i}", lb=0.0, ub=1.0, var_type=snb.VarType.Binary)
        for i in range(len(c))
    ]
    for row, rhs in zip(A, b):
        model.add_constr(
            sum(float(coefficient) * variable for coefficient, variable in zip(row, variables))
            <= float(rhs)
        )
    model.maximize(
        sum(float(coefficient) * variable for coefficient, variable in zip(c, variables))
    )

    options = snb.BranchAndBoundOptions()
    options.parallel_workers = workers
    options.max_nodes = 10_000
    options.mip_abs_gap = 0.0
    options.mip_rel_gap = 0.0
    if configure is not None:
        configure(options)
    return model.solve_mip(options)


def assert_optimal_feasible(result, expected, A, b):
    assert result.status == snb.Status.Optimal
    assert result.objective == expected
    x = np.asarray(result.x)
    assert np.all(A @ x <= b + 1e-7)
    assert np.all(np.abs(x - np.rint(x)) <= 1e-7)


def test_gomory_bound_transform_preserves_integer_optimum():
    c = np.array([1.0, 3.0, 11.0, 6.0, 4.0, -1.0, -1.0, 12.0, 6.0])
    A = np.array(
        [
            [2.0, 0.0, 9.0, 8.0, 1.0, 4.0, -1.0, 7.0, 9.0],
            [7.0, 6.0, 4.0, 9.0, 8.0, 5.0, -2.0, 4.0, 8.0],
            [1.0, -1.0, 4.0, 7.0, 7.0, 9.0, -1.0, 3.0, 4.0],
            [-3.0, 1.0, 5.0, 5.0, 2.0, 9.0, -3.0, 9.0, -3.0],
        ]
    )
    b = np.array([21.0, 33.0, 25.0, 15.0])

    cut_flags = (
        "use_gomory_cuts",
        "use_mir_cuts",
        "use_cover_cuts",
        "use_zero_half_cuts",
        "use_implied_bound_cuts",
        "use_clique_cuts",
        "use_graph_clique_cuts",
        "use_odd_cycle_cuts",
        "use_conflict_cuts",
        "use_dual_proof_cuts",
    )

    def gomory_only(options):
        options.use_async_heuristics = False
        options.use_rounding = False
        options.use_diving = False
        options.use_node_presolve = False
        for flag in cut_flags:
            if hasattr(options, flag):
                setattr(options, flag, flag == "use_gomory_cuts")

    result = solve_binary_mip(c, A, b, configure=gomory_only)
    assert_optimal_feasible(result, 31.0, A, b)


def test_bound_only_lp_after_mip_presolve_does_not_factor_empty_basis():
    c = np.array([0.0, -2.0, 8.0, 6.0])
    A = np.array(
        [
            [-2.0, 2.0, 3.0, -3.0],
            [6.0, 8.0, 5.0, 2.0],
            [4.0, 1.0, 4.0, 3.0],
            [-1.0, 0.0, -1.0, 4.0],
            [5.0, 8.0, -2.0, 1.0],
            [5.0, 4.0, 7.0, 5.0],
            [2.0, -3.0, 0.0, 8.0],
        ]
    )
    b = np.array([5.0, 8.0, 8.0, 0.0, 13.0, 6.0, -3.0])

    result = solve_binary_mip(
        c, A, b, configure=lambda options: setattr(options, "use_cut_pool", False)
    )
    assert_optimal_feasible(result, -2.0, A, b)


def test_non_root_structural_cuts_use_the_global_domain():
    c = np.array([3.0, 4.0, -2.0, 6.0, 7.0, 9.0, 5.0, 0.0, 6.0, 5.0])
    A = np.array(
        [
            [-2.0, 3.0, 8.0, 0.0, 2.0, 6.0, 5.0, 7.0, 7.0, 1.0],
            [5.0, 1.0, 6.0, 5.0, 1.0, 2.0, -1.0, 2.0, -1.0, 3.0],
            [9.0, 2.0, -1.0, 0.0, -1.0, 4.0, 2.0, 8.0, -2.0, -2.0],
            [2.0, -3.0, 3.0, 0.0, 3.0, 9.0, 9.0, -1.0, 2.0, 9.0],
            [-3.0, -3.0, 6.0, 7.0, -3.0, 3.0, 1.0, 7.0, 0.0, -2.0],
            [0.0, 6.0, -1.0, -1.0, 7.0, 2.0, -3.0, 1.0, 3.0, -1.0],
        ]
    )
    b = np.array([19.0, 28.0, 17.0, 23.0, 12.0, 6.0])

    result = solve_binary_mip(c, A, b)
    assert_optimal_feasible(result, 30.0, A, b)


def test_parallel_cut_search_is_repeatably_correct():
    c = np.array([9.0, 4.0, 7.0, 1.0, 2.0, 8.0, 10.0, 12.0, 9.0, -4.0, 9.0])
    A = np.array(
        [
            [-1.0, 7.0, 8.0, 9.0, -2.0, 8.0, 5.0, 6.0, -2.0, 9.0, -1.0],
            [7.0, 0.0, 7.0, 7.0, -2.0, 5.0, 0.0, 9.0, 4.0, 8.0, 4.0],
            [5.0, -1.0, 7.0, 5.0, 6.0, -2.0, 1.0, -2.0, 5.0, 7.0, 5.0],
            [6.0, 6.0, 0.0, -1.0, 7.0, 4.0, -1.0, -3.0, 6.0, 5.0, 9.0],
            [2.0, -2.0, 7.0, -3.0, 6.0, 4.0, 7.0, 0.0, 0.0, 9.0, 2.0],
            [1.0, 1.0, 5.0, 8.0, 4.0, 0.0, 4.0, 9.0, 4.0, 6.0, -1.0],
            [8.0, 6.0, -3.0, -1.0, 9.0, -3.0, 1.0, 8.0, 8.0, 9.0, 5.0],
            [-2.0, -3.0, -2.0, -2.0, -3.0, 0.0, 8.0, -2.0, 9.0, -3.0, 2.0],
        ]
    )
    b = np.array([13.0, 23.0, 19.0, 27.0, 14.0, 22.0, 30.0, 6.0])

    for _ in range(30):
        result = solve_binary_mip(c, A, b, workers=2)
        assert_optimal_feasible(result, 40.0, A, b)


def test_parallel_basis_warm_starts_do_not_share_factorizations():
    c = np.array([11.0, 12.0, 8.0, 0.0, 6.0, -2.0, 8.0, 2.0, 10.0])
    A = np.array(
        [
            [5.0, -2.0, -2.0, 8.0, -1.0, 8.0, 5.0, 7.0, 7.0],
            [1.0, -1.0, 4.0, 7.0, -2.0, 3.0, 2.0, 5.0, 5.0],
            [-3.0, 2.0, -1.0, 6.0, 0.0, 6.0, -2.0, 0.0, 5.0],
            [3.0, 0.0, -2.0, 4.0, 6.0, 5.0, 6.0, 7.0, 0.0],
            [-2.0, 8.0, 7.0, -3.0, 0.0, 5.0, 6.0, -3.0, 0.0],
            [-2.0, 2.0, -3.0, 2.0, 0.0, 6.0, 6.0, 5.0, -3.0],
            [5.0, 8.0, 5.0, 9.0, -1.0, 5.0, 1.0, 0.0, -2.0],
        ]
    )
    b = np.array([39.0, 32.0, 17.0, 23.0, 4.0, 9.0, 29.0])

    def no_cuts(options):
        options.use_cut_pool = False
        options.use_async_heuristics = False

    for _ in range(30):
        result = solve_binary_mip(c, A, b, workers=4, configure=no_cuts)
        assert_optimal_feasible(result, 41.0, A, b)
