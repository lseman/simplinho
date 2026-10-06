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
