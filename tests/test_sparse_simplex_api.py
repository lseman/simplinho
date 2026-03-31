import math
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


ROOT = Path(__file__).resolve().parents[1]


def import_simplinho():
    for build_dir in ("build-local", "build", "build-verify"):
        candidate = ROOT / build_dir
        if not candidate.exists():
            continue
        if not any(candidate.glob("*.so")):
            continue
        sys.path.insert(0, str(candidate))
        try:
            import simplinho

            return simplinho
        except ImportError:
            continue
    raise ImportError("could not find a built simplinho module")


try:
    simplinho = import_simplinho()
    HAS_SIMPLINHO = True
except ImportError:
    simplinho = None
    HAS_SIMPLINHO = False


@unittest.skipUnless(HAS_SIMPLINHO and sp is not None,
                     "requires a locally built simplinho module and scipy")
class SparseSimplexApiTests(unittest.TestCase):
    def test_sparse_matrix_solves_and_marks_sparse_pipeline(self):
        A = sp.csc_matrix([[1.0, 1.0], [2.0, 1.0]])
        b = np.array([4.0, 5.0], dtype=float)
        c = np.array([1.0, 3.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Auto
        options.pricing_rule = "adaptive"
        options.primal_edge_weight_strategy = "dense_diagonal"
        options.primal_simplex_cost_perturbation_multiplier = 2.0

        solver = simplinho.RevisedSimplex(options)
        sol = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertTrue(math.isclose(sol.obj, 10.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(np.allclose(sol.x, np.array([1.0, 3.0]), atol=1e-8))
        self.assertEqual(sol.info.get("sparse_pipeline"), "1")

    def test_sparse_matrix_accepts_warm_start_basis(self):
        A = sp.csc_matrix([[1.0, 1.0]])
        b = np.array([4.0], dtype=float)
        c = np.array([1.0, 2.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)
        second = solver.solve(A, b, c, l, u, first.basis_state)

        self.assertEqual(simplinho.status_to_string(first.status), "optimal")
        self.assertEqual(simplinho.status_to_string(second.status), "optimal")
        self.assertEqual(second.info.get("sparse_pipeline"), "1")

    def test_dual_partial_pricing_uses_row_pricing_strategy(self):
        A = sp.csc_matrix([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
        b = np.array([4.0, 3.0], dtype=float)
        c = np.array([3.0, 1.0, 2.0], dtype=float)
        l = np.zeros(3, dtype=float)
        u = np.array([np.inf, np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual
        options.pricing_rule = "adaptive"
        options.partial_pricing = True
        options.dual_pricing = "switch"
        options.row_pricing_threshold = 10
        options.dual_edge_weight_strategy = "diagonal"
        options.dual_simplex_cost_perturbation_multiplier = 2.0

        solver = simplinho.RevisedSimplex(options)
        sol = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertTrue(math.isclose(sol.obj, 6.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(np.allclose(sol.x, np.array([1.0, 3.0, 0.0]), atol=1e-8))
        self.assertEqual(sol.info.get("dual_pricing"), "dual_row_pricing")
        self.assertEqual(sol.info.get("sparse_pipeline"), "1")


if __name__ == "__main__":
    unittest.main()
