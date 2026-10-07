import numpy as np
import pytest
import scipy.sparse as sp

import simplinho as splx


@pytest.mark.parametrize("method", ["ft", "pf", "mpf", "apf", "eigen"])
def test_sparse_update_method_preserves_solution(method):
    rng = np.random.default_rng(17)
    rows, structural = 20, 45
    R = sp.random(rows, structural, density=0.12, random_state=rng, format="csc")
    A = sp.hstack([R, sp.eye(rows, format="csc")], format="csc")
    x = rng.random(A.shape[1])
    b = np.asarray(A @ x).ravel()
    c = rng.standard_normal(A.shape[1])
    lower = np.zeros(A.shape[1])
    upper = np.full(A.shape[1], 3.0)

    options = splx.RevisedSimplexOptions()
    options.basis_sparse_backend = method
    solution = splx.RevisedSimplex(options).solve(A, b, c, lower, upper)

    assert "Optimal" in str(solution.status)
    assert np.max(np.abs(np.asarray(A @ solution.x).ravel() - b)) < 1e-6


def test_primal_pricing_keeps_linear_solve_work_per_iteration():
    """Inactive pricing frameworks and exact repricing must stay off the hot path."""
    rng = np.random.default_rng(91)
    rows, structural = 30, 60
    R = sp.random(
        rows,
        structural,
        density=0.1,
        random_state=rng,
        data_rvs=lambda size: rng.standard_normal(size),
        format="csc",
    )
    A = sp.hstack([R, sp.eye(rows, format="csc")], format="csc")
    x_feasible = rng.random(A.shape[1]) * 2.0
    b = np.asarray(A @ x_feasible).ravel()

    # c = A.T @ y + s with s > 0 makes this nonnegative equality-form LP
    # bounded while still requiring enough pivots to exercise pricing rebuilds.
    y = rng.standard_normal(rows)
    c = np.asarray(A.T @ y).ravel() + rng.random(A.shape[1]) * 0.9 + 0.1

    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Primal
    options.disable_presolve = True
    solution = splx.RevisedSimplex(options).solve(
        A,
        b,
        c,
        np.zeros(A.shape[1]),
        np.full(A.shape[1], np.inf),
    )

    assert "Optimal" in str(solution.status)
    assert np.max(np.abs(np.asarray(A @ solution.x).ravel() - b)) < 1e-7
    assert solution.stats.iterations >= 20
    # A primal pivot needs one FTRAN for its pivotal column and one BTRAN for
    # its pivotal row. Allow a small fixed margin for startup/certification.
    assert solution.stats.ftran_calls <= solution.stats.iterations + 8
    assert solution.stats.btran_calls <= solution.stats.iterations + 8


def test_dual_dse_stays_stable_and_uses_sparse_price_support():
    rng = np.random.default_rng(900)
    rows, structural = 300, 600
    R = sp.random(
        rows,
        structural,
        density=0.02,
        random_state=rng,
        data_rvs=lambda size: rng.standard_normal(size),
        format="csc",
    )
    A = sp.hstack([R, sp.eye(rows, format="csc")], format="csc")
    x_feasible = rng.random(A.shape[1]) * 2.0
    b = np.asarray(A @ x_feasible).ravel()
    c = rng.standard_normal(A.shape[1])

    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.parallel_pricing_workers = 2
    options.parallel_pricing_min_cols = 1
    solution = splx.RevisedSimplex(options).solve(
        A,
        b,
        c,
        np.zeros(A.shape[1]),
        np.full(A.shape[1], 5.0),
    )

    assert "Optimal" in str(solution.status)
    assert np.max(np.abs(np.asarray(A @ solution.x).ravel() - b)) < 1e-7
    assert solution.stats.raw_info["dual_pricing"] == "dual_steepest_edge"
    assert solution.stats.dual_row_price_calls > 0
    assert solution.stats.iterations < 800


def test_sparse_lu_update_chain_matches_dense_reference():
    rng = np.random.default_rng(23)
    size = 18
    basis = rng.standard_normal((size, size))
    basis += np.diag(np.sum(np.abs(basis), axis=1) + 1.0)

    config = splx.SparseLUConfig()
    config.force_eigen_sparse_lu = True
    config.diagonal_equilibration = False
    config.iterative_refinement = False

    factor = splx.SparseForrestTomlinLU()
    factor.factor_with_config(sp.csc_matrix(basis), config=config)

    current = basis.copy()
    for pivot_row in (2, 11, 5, 16):
        new_column = rng.standard_normal(size)
        update_column = new_column - current[:, pivot_row]
        transformed_column = np.linalg.solve(current, update_column)
        pivot_row_inverse = np.linalg.solve(current.T, np.eye(size)[:, pivot_row])
        alpha = 1.0 + transformed_column[pivot_row]
        assert abs(alpha) > 1e-4
        assert factor.append_forrest_tomlin_update(
            pivot_row,
            update_column,
            transformed_column,
            pivot_row_inverse,
            alpha,
        )
        current[:, pivot_row] = new_column

    rhs = rng.standard_normal(size)
    transpose_rhs = rng.standard_normal(size)
    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(current, rhs), rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(
        factor.solveT(transpose_rhs),
        np.linalg.solve(current.T, transpose_rhs),
        rtol=2e-11,
        atol=2e-11,
    )


def test_mixed_hyper_dense_triangular_stages_clear_reach_marks():
    # Regression for stale reach marks when a hyper-sparse first triangular
    # stage switched to a dense second stage. The next sparse RHS used to skip
    # nodes visited by the previous solve and relied on residual fallback.
    basis = np.array(
        [
            [1.5167, -0.3224, 0.0, 0.0, 0.0],
            [-0.6365, 1.9531, 0.0, 0.0, -0.3166],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5420, 1.5420, 0.0],
            [0.0, 0.0, -1.5259, 0.0, 2.5259],
        ]
    )
    config = splx.SparseLUConfig()
    config.iterative_refinement = False
    config.enable_solve_oracle = False

    factor = splx.SparseForrestTomlinLU()
    factor.factor_with_config(sp.csc_matrix(basis), config=config)

    for transpose in (False, True):
        matrix = basis.T if transpose else basis
        solve = factor.solveT_sparse if transpose else factor.solve_sparse
        for column in range(basis.shape[0]):
            rhs = np.eye(basis.shape[0])[:, column]
            np.testing.assert_allclose(
                solve([column], [1.0], 0.0),
                np.linalg.solve(matrix, rhs),
                rtol=2e-11,
                atol=2e-11,
            )

    assert factor.ftran_sparse_reach_failures() == 0
    assert factor.btran_sparse_reach_failures() == 0
