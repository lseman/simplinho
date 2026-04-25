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


@unittest.skipUnless(
    HAS_SIMPLINHO and sp is not None,
    "requires a locally built simplinho module and scipy",
)
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

    def test_sparse_matrix_hyper_sparse_rhs_solve(self):
        rows = 20
        cols = 40
        data = []
        row_ind = []
        col_ptr = [0]
        for j in range(rows):
            data.append(1.0)
            row_ind.append(j)
            col_ptr.append(col_ptr[-1] + 1)
        for j in range(rows, cols):
            col_ptr.append(col_ptr[-1])

        A = sp.csc_matrix((data, row_ind, col_ptr), shape=(rows, cols))
        b = np.zeros(rows, dtype=float)
        b[0] = 1.0
        c = np.arange(cols, dtype=float)
        l = np.zeros(cols, dtype=float)
        u = np.full(cols, np.inf, dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Primal
        options.pricing_rule = "adaptive"
        options.primal_edge_weight_strategy = "dense_diagonal"

        solver = simplinho.RevisedSimplex(options)
        sol = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertTrue(np.allclose(sol.x[:rows], b, atol=1e-8))
        self.assertTrue(np.allclose(sol.x[rows:], 0.0, atol=1e-8))
        self.assertEqual(sol.info.get("sparse_pipeline"), "1")

    def test_sparse_matrix_warm_start_basis_after_feasible_bound_change(self):
        A = sp.csc_matrix([[1.0, 1.0]])
        b = np.array([4.0], dtype=float)
        c = np.array([1.0, 2.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)
        self.assertEqual(simplinho.status_to_string(first.status), "optimal")
        self.assertEqual(len(first.basis_state.column_status), 2)

        u2 = np.array([1.5, np.inf], dtype=float)
        second = solver.solve(A, b, c, l, u2, first.basis_state)

        self.assertEqual(simplinho.status_to_string(second.status), "optimal")
        self.assertEqual(len(second.basis_state.column_status), 2)
        self.assertTrue(math.isclose(second.obj, 6.5, rel_tol=0.0, abs_tol=1e-8))
        self.assertLessEqual(second.x[0], 1.5 + 1e-8)
        self.assertIn(second.stats.basis_start, {"warm_start", "repaired_warm_start"})
        self.assertEqual(second.info.get("bound_reformulation_initial_mode"), "dual")
        self.assertEqual(
            second.info.get("bound_reformulation_warm_start_dual_feasible"), "1"
        )
        self.assertNotIn("bound_reformulation_retry_mode", second.info)

    def test_sparse_matrix_reuses_bound_only_cache_after_same_orientation_bound_change(self):
        A = sp.csc_matrix([[1.0, 1.0]])
        b = np.array([4.0], dtype=float)
        c = np.array([1.0, 2.0], dtype=float)
        l = np.array([1.0, 0.0], dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(first.status), "optimal")
        self.assertEqual(first.info.get("sparse_pipeline"), "1")
        self.assertIsNone(first.info.get("sparse_bound_only_fast_path"))

        l2 = np.array([2.0, 0.0], dtype=float)
        second = solver.solve(A, b, c, l2, u)

        self.assertEqual(simplinho.status_to_string(second.status), "optimal")
        self.assertEqual(second.info.get("sparse_pipeline"), "1")
        self.assertEqual(second.info.get("sparse_bound_only_fast_path"), "1")

    def test_sparse_matrix_warm_start_reuses_internal_solver_basis_on_bound_change(self):
        A = sp.csc_matrix([[1.0, 1.0]])
        b = np.array([4.0], dtype=float)
        c = np.array([1.0, 2.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)
        self.assertEqual(simplinho.status_to_string(first.status), "optimal")

        u2 = np.array([1.5, np.inf], dtype=float)
        second = solver.solve(A, b, c, l, u2)

        self.assertEqual(simplinho.status_to_string(second.status), "optimal")
        self.assertIn(second.stats.basis_start, {"warm_start", "repaired_warm_start"})
        self.assertEqual(second.info.get("bound_reformulation_initial_mode"), "dual")
        self.assertEqual(
            second.info.get("bound_reformulation_warm_start_dual_feasible"), "1"
        )
        self.assertNotIn("bound_reformulation_retry_mode", second.info)

    def test_dual_adaptive_warm_start_near_optimal_switches_to_devex(self):
        A = sp.csc_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = np.array([7.0, 8.0], dtype=float)
        c = np.array([1.0, 1.0, 1.0], dtype=float)
        l = np.zeros(3, dtype=float)
        u = np.array([np.inf, np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual
        options.pricing_rule = "adaptive"
        options.row_pricing_threshold = 0
        options.dual_warm_start_near_optimal = True

        solver = simplinho.RevisedSimplex(options)
        sol = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertEqual(sol.info.get("dual_pricing"), "dual_devex")

    def test_sparse_matrix_warm_start_basis_after_upper_bound_branch(self):
        A = sp.csc_matrix([[1.0, 1.0]])
        b = np.array([2.0], dtype=float)
        c = np.array([-1.0, -1.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([1.0, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)
        self.assertEqual(simplinho.status_to_string(first.status), "optimal")
        self.assertTrue(math.isclose(first.x[0], 1.0, rel_tol=0.0, abs_tol=1e-8))

        u2 = np.array([0.5, np.inf], dtype=float)
        second = solver.solve(A, b, c, l, u2, first.basis_state)

        self.assertEqual(simplinho.status_to_string(second.status), "optimal")
        self.assertEqual(len(second.basis_state.column_status), 2)
        self.assertTrue(math.isclose(second.obj, -2.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(second.x[0], 0.5, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(second.x[1], 1.5, rel_tol=0.0, abs_tol=1e-8))
        self.assertIn(second.stats.basis_start, {"warm_start", "repaired_warm_start"})
        self.assertEqual(second.info.get("bound_reformulation_initial_mode"), "dual")
        self.assertEqual(
            second.info.get("bound_reformulation_warm_start_dual_feasible"), "1"
        )
        self.assertNotIn("bound_reformulation_retry_mode", second.info)

    def test_sparse_matrix_warm_start_basis_after_infeasible_bound_change(self):
        A = sp.csc_matrix([[1.0, 2.0], [3.0, 1.0]])
        b = np.array([5.0, 6.0], dtype=float)
        c = np.array([1.0, 4.0], dtype=float)
        l = np.zeros(2, dtype=float)
        u = np.array([np.inf, np.inf], dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual

        solver = simplinho.RevisedSimplex(options)
        first = solver.solve(A, b, c, l, u)
        self.assertEqual(simplinho.status_to_string(first.status), "optimal")

        u2 = np.array([1.0, np.inf], dtype=float)
        second = solver.solve(A, b, c, l, u2, first.basis_state)

        self.assertEqual(simplinho.status_to_string(second.status), "infeasible")
        self.assertIn(second.stats.basis_start, {"warm_start", "repaired_warm_start"})
        self.assertEqual(second.info.get("bound_reformulation_initial_mode"), "dual")
        self.assertEqual(
            second.info.get("bound_reformulation_warm_start_dual_feasible"), "1"
        )

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

    def test_dual_switch_pricing_uses_row_pricing_on_sparse_rows(self):
        # This matrix is very sparse in its nonbasic columns, so the dual
        # switch policy should select row pricing even when the basis is
        # larger than the threshold.
        rows = 20
        cols = 40
        data = []
        row_ind = []
        col_ptr = [0]
        for j in range(cols):
            if j < rows:
                data.append(1.0)
                row_ind.append(j)
            col_ptr.append(len(data))
        A = sp.csc_matrix((data, row_ind, col_ptr), shape=(rows, cols))
        b = np.ones(rows, dtype=float)
        c = np.zeros(cols, dtype=float)
        c[0] = 1.0
        l = np.zeros(cols, dtype=float)
        u = np.full(cols, np.inf, dtype=float)

        options = simplinho.RevisedSimplexOptions()
        options.mode = simplinho.SimplexMode.Dual
        options.pricing_rule = "adaptive"
        options.partial_pricing = False
        options.dual_pricing = "switch"
        options.row_pricing_threshold = 10

        solver = simplinho.RevisedSimplex(options)
        sol = solver.solve(A, b, c, l, u)

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertEqual(sol.info.get("dual_pricing"), "dual_row_pricing")


if __name__ == "__main__":
    unittest.main()
