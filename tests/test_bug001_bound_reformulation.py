import math

import numpy as np
import pytest
import simplinho as splx


def test_doubly_bounded_dense_reformulation_solves_square_inner_basis():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])
    b = np.array([12.0, 10.0])
    c = np.array([-3.0, -5.0])
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])

    sol = splx.RevisedSimplex().solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(sol.x, [8.0 / 3.0, 14.0 / 3.0], rtol=1e-9, atol=1e-9)
    assert math.isclose(sol.obj, -94.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)


def test_square_standard_form_full_warm_basis_is_optimal():
    A = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    b = np.array([12.0, 10.0, 10.0, 10.0])
    c = np.array([-3.0, -5.0, 0.0, 0.0])
    l = np.zeros(4)
    u = np.full(4, np.inf)

    sol = splx.RevisedSimplex().solve(A, b, c, l, u, [0, 1, 2, 3])

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(
        sol.x, [8.0 / 3.0, 14.0 / 3.0, 22.0 / 3.0, 16.0 / 3.0], rtol=1e-9, atol=1e-9
    )
    assert math.isclose(sol.obj, -94.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)


def test_doubly_bounded_sparse_reformulation_solves_square_inner_basis():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    A = scipy_sparse.csc_matrix(np.array([[1.0, 2.0], [2.0, 1.0]]))
    b = np.array([12.0, 10.0])
    c = np.array([-3.0, -5.0])
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])

    sol = splx.RevisedSimplex().solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(sol.x, [8.0 / 3.0, 14.0 / 3.0], rtol=1e-9, atol=1e-9)
    assert math.isclose(sol.obj, -94.0 / 3.0, rel_tol=1e-9, abs_tol=1e-9)
