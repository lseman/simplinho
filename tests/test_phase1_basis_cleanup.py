import numpy as np
import pytest

import simplinho as splx


def _problems():
    rng = np.random.RandomState(42)
    dense_a = rng.randn(20, 30)
    dense_x = np.abs(rng.randn(30)) + 1.0
    dense_b = dense_a @ dense_x
    dense_c = rng.randn(30)

    rng = np.random.RandomState(42)
    structured_a = np.zeros((50, 80))
    for block in range(5):
        start = block * 16
        structured_a[2 * block : 2 * block + 2, start : start + 16] = rng.randn(2, 16)
    structured_a[10:] = rng.randn(40, 80) * 0.1
    structured_x = np.abs(rng.randn(80)) + 1.0
    structured_b = structured_a @ structured_x
    structured_c = rng.randn(80)

    return (
        (dense_a, dense_b, dense_c, -62.01448908530322),
        (structured_a, structured_b, structured_c, -106.39424043357964),
    )


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("a,b,c,expected_obj", _problems())
def test_phase1_artificials_are_pivoted_out_without_losing_feasibility(
    a, b, c, expected_obj, as_sparse
):
    if as_sparse:
        scipy_sparse = pytest.importorskip("scipy.sparse")
        # Exercise the public sparse path used by scipy's canonical row-oriented
        # model assembly and by the original reproducer.
        a_input = scipy_sparse.csr_matrix(a)
    else:
        a_input = a

    options = splx.RevisedSimplexOptions()
    options.max_iters = 1_000
    solution = splx.RevisedSimplex(options).solve(
        a_input, b, c, np.zeros(a.shape[1]), np.full(a.shape[1], np.inf)
    )

    assert solution.status == splx.LPStatus.Optimal
    np.testing.assert_allclose(a @ np.asarray(solution.x), b, rtol=1e-8, atol=1e-8)
    assert np.min(solution.x) >= -1e-8
    assert solution.obj == pytest.approx(expected_obj, rel=1e-10, abs=1e-10)
    assert solution.iters < options.max_iters
