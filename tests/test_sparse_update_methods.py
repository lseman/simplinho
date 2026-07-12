import numpy as np
import pytest
import scipy.sparse as sp

import simplinho as splx


@pytest.mark.parametrize("method", ["ft", "pf", "mpf", "apf"])
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
