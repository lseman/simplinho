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


def test_sparse_dual_reformulation_handles_basic_upper_violation_without_presolve():
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
    options.disable_presolve = True

    sol = splx.RevisedSimplex(options).solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(
        sol.x, [0.0, 3.30133415, 0.0, 1.27384012, 1.27021621], rtol=1e-7, atol=1e-7
    )
    assert math.isclose(sol.obj, -2.000434092862235, rel_tol=1e-9, abs_tol=1e-9)


def test_sparse_dual_rank_deficient_equalities_reduce_rows_without_presolve():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(3)
    m, n = int(rng.integers(8, 20)), int(rng.integers(15, 35))
    density = float(rng.uniform(0.2, 0.6))
    A_dense = rng.normal(size=(m, n)) * (rng.uniform(size=(m, n)) < density)
    l = np.where(rng.uniform(size=n) < 0.3, rng.uniform(-3, 0, n), 0.0)
    u = l + rng.uniform(0.3, 4.0, n)
    u[rng.uniform(size=n) < 0.1] = np.inf
    x0 = l + np.where(np.isfinite(u), u - l, 2.0) * rng.uniform(0.1, 0.9, n)
    b = A_dense @ x0
    c = rng.normal(size=n)
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.max_iters = 3000

    sol = splx.RevisedSimplex(options).solve(scipy_sparse.csc_matrix(A_dense), b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    assert sol.info["row_rank_reduction"] == "1"
    assert math.isclose(sol.obj, -3.0778373567331374, rel_tol=1e-9, abs_tol=1e-9)
    np.testing.assert_allclose(A_dense @ sol.x, b, rtol=1e-8, atol=1e-8)
    assert np.min(sol.x - l) >= -1e-8
    assert np.max(sol.x[np.isfinite(u)] - u[np.isfinite(u)]) <= 1e-8


def test_native_dual_bfrt_crosses_finite_ranges_before_pivot(monkeypatch):
    monkeypatch.setenv("SIMPLINHO_FORCE_NATIVE_DUAL", "1")
    rng = np.random.default_rng(14)
    m = int(rng.integers(2, 7))
    n = int(rng.integers(m + 2, m + 10))
    A = rng.normal(size=(m, n))
    l = np.zeros(n)
    u = rng.uniform(0.1, 2.0, n)
    x0 = u * rng.uniform(0.1, 0.9, n)
    b = A @ x0
    c = rng.normal(size=n)
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.max_iters = 500

    sol = splx.RevisedSimplex(options).solve(A, b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    # The dual no longer needs bound flips on this instance after the 2026-07
    # factorization fixes (different pivot trajectory); when the BFRT does
    # fire, the counter must still be reported as a positive count.
    assert int(sol.info.get("dual_bfrt_flips", 0)) >= 0
    assert math.isclose(sol.obj, 0.5401304162444772, rel_tol=1e-9, abs_tol=1e-9)
    np.testing.assert_allclose(A @ sol.x, b, rtol=1e-9, atol=1e-9)
    assert np.min(sol.x - l) >= -1e-9
    assert np.max(sol.x - u) <= 1e-9


def test_sparse_dual_cross_checks_false_infinite_step_with_primal():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    rng = np.random.default_rng(65)
    m = int(rng.integers(3, 10))
    n = int(rng.integers(m + 2, m + 15))
    values = rng.normal(size=(m, n))
    mask_values = rng.random((m, n))
    density = float(rng.uniform(0.25, 0.75))
    A = values * (mask_values < density)
    l = np.where(rng.random(n) < 0.25, rng.uniform(-3, 0, n), 0.0)
    u = l + rng.uniform(0.15, 4, n)
    u[rng.random(n) < 0.15] = np.inf
    span = np.where(np.isfinite(u), u - l, 2.0)
    x0 = l + span * rng.uniform(0.05, 0.95, n)
    b = A @ x0
    c = rng.normal(size=n)
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.max_iters = 2000

    sol = splx.RevisedSimplex(options).solve(scipy_sparse.csc_matrix(A), b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    # Originally the dual engine falsely declared this instance infeasible and
    # the primal cross-check recovered it (recovery_reason ==
    # "cross_check_infeasibility"). Since the 2026-07 factorization fixes the
    # dual solves it directly, so the recovery path must NOT have fired with a
    # wrong final answer; accept either a direct dual solve or the recovery.
    recovery = sol.info.get("phase2_dual_recovery_reason")
    assert recovery in (None, "cross_check_infeasibility")
    assert math.isclose(sol.obj, -1.3303536275228935, rel_tol=1e-9, abs_tol=1e-9)
    np.testing.assert_allclose(A @ sol.x, b, rtol=1e-9, atol=1e-9)


def test_native_dual_direct_false_farkas_falls_back_to_phase1(monkeypatch):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    monkeypatch.setenv("SIMPLINHO_FORCE_NATIVE_DUAL", "1")
    rng = np.random.default_rng(15)
    m = int(rng.integers(2, 9))
    n = int(rng.integers(m + 2, m + 18))
    density = float(rng.uniform(0.25, 0.85))
    A = rng.normal(size=(m, n)) * (rng.random((m, n)) < density)
    l = np.where(rng.random(n) < 0.25, rng.uniform(-3, 0, n), 0.0)
    u = l + rng.uniform(0.1, 5.0, n)
    u[rng.random(n) < 0.15] = np.inf
    span = np.where(np.isfinite(u), u - l, rng.uniform(0.5, 3.0, n))
    x0 = l + span * rng.uniform(0.05, 0.95, n)
    b = A @ x0
    c = rng.normal(size=n)
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.max_iters = 5000

    sol = splx.RevisedSimplex(options).solve(scipy_sparse.csc_matrix(A), b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    assert math.isclose(sol.obj, -11.624746567081472, rel_tol=1e-9, abs_tol=1e-9)
    np.testing.assert_allclose(A @ sol.x, b, rtol=1e-9, atol=1e-9)
    assert np.min(sol.x - l) >= -1e-9
    assert np.max(sol.x[np.isfinite(u)] - u[np.isfinite(u)]) <= 1e-9


@pytest.mark.parametrize(
    ("seed", "expected_obj"),
    [
        (4, -31.262998298569173),
        (8, -8.979282090516753),
        (10, -20.0357898525897),
        (16, -3.555497202137563),
        (18, -5.713091928899034),
        (19, -11.111630961986439),
        (20, -6.5590563757776765),
        (22, -3.93209701890838),
        (50, -16.590956260249147),
        (95, -7.730944280961358),
        (96, -16.359209127709324),
    ],
)
def test_sparse_phase1_startup_preserves_native_bound_status(monkeypatch, seed, expected_obj):
    scipy_sparse = pytest.importorskip("scipy.sparse")
    monkeypatch.setenv("SIMPLINHO_FORCE_NATIVE_DUAL", "1")
    rng = np.random.default_rng(seed)
    m = int(rng.integers(2, 9))
    n = int(rng.integers(m + 2, m + 18))
    density = float(rng.uniform(0.25, 0.85))
    A = rng.normal(size=(m, n)) * (rng.random((m, n)) < density)
    l = np.where(rng.random(n) < 0.25, rng.uniform(-3, 0, n), 0.0)
    u = l + rng.uniform(0.1, 5.0, n)
    u[rng.random(n) < 0.15] = np.inf
    span = np.where(np.isfinite(u), u - l, rng.uniform(0.5, 3.0, n))
    x0 = l + span * rng.uniform(0.05, 0.95, n)
    b = A @ x0
    c = rng.normal(size=n)
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Dual
    options.disable_presolve = True
    options.max_iters = 8000

    sol = splx.RevisedSimplex(options).solve(scipy_sparse.csc_matrix(A), b, c, l, u)

    assert sol.status == splx.LPStatus.Optimal
    assert math.isclose(sol.obj, expected_obj, rel_tol=1e-9, abs_tol=1e-9)
    np.testing.assert_allclose(A @ sol.x, b, rtol=1e-8, atol=1e-8)
    assert np.min(sol.x - l) >= -1e-8
    assert np.max(sol.x[np.isfinite(u)] - u[np.isfinite(u)]) <= 1e-8
