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


def test_markowitz_column_permutation_phase1_regression():
    A = np.array(
        [
            [1.2637284581291104, -0.8706617379590857, -0.2591732349343976, -0.07534330701052097],
            [-0.740884652085609, -1.3677927017829434, 0.6488928021930399, 0.361058113054895],
            [-1.95286306301219, 2.347409654378852, 0.9684969057519236, -0.7593871804245066],
        ]
    )
    b = np.array([-2.3238793316808453, -2.9812142059633846, 6.8845243703828185])
    c = np.array([-0.29969851529910546, 0.9029193414250598, -1.6215827341822058, -0.15818926067687128])
    l = np.zeros(4)
    u = np.array([1.6712855491587846, 4.09287766317469, 7.205359435292624, 5.8821744991741545])

    sol = splx.RevisedSimplex().solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(
        sol.x, [1.6712855491587846, 3.76281473, 3.63196578, 2.89983701], rtol=1e-7, atol=1e-7
    )
    assert math.isclose(sol.obj, -3.4516196703614264, rel_tol=1e-9, abs_tol=1e-9)


def test_sparse_dual_reformulation_validates_mapped_primal_before_accepting_optimal():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    A = scipy_sparse.csc_matrix(
        np.array(
            [
                [0.14754746975086025, 0.20381642924179696, -0.40621884602135155, -2.2219723133128704, 0.3318116563956822],
                [-2.86054756497825, -1.5234494373617915, -0.47797031720486144, 0.41265723121444153, 0.8463928997612329],
            ]
        )
    )
    b = np.array([-1.7360987982545737, -3.428654334644951])
    c = np.array([1.4414901451598272, -0.563981532939405, 0.8934708274230185, -0.07425126800393837, -0.03460698643809272])
    l = np.zeros(5)
    u = np.array([4.03358316830722, 4.825126321719372, np.inf, 1.4547472272771973, 1.2702162069349718])
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual

    sol = splx.RevisedSimplex(options).solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(
        sol.x, [0.0, 3.30133415, 0.0, 1.27384012, 1.27021621], rtol=1e-7, atol=1e-7
    )
    assert math.isclose(sol.obj, -2.000434092862235, rel_tol=1e-9, abs_tol=1e-9)
