import sys
import unittest
from pathlib import Path

import numpy as np


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


@unittest.skipUnless(HAS_SIMPLINHO, "requires a locally built simplinho module")
class PresolveImpliedBoundsTests(unittest.TestCase):
    def test_dense_presolve_reports_implied_bound_updates(self):
        A = np.array([[1.0, 1.0, 1.0]], dtype=float)
        b = np.array([1.0], dtype=float)
        c = np.array([-1.0, 0.0, 0.0], dtype=float)
        l = np.zeros(3, dtype=float)
        u = np.array([np.inf, np.inf, np.inf], dtype=float)

        solver = simplinho.RevisedSimplex()
        sol = solver.solve(A, b, c, l, u)
        stats = sol.stats

        self.assertEqual(simplinho.status_to_string(sol.status), "optimal")
        self.assertTrue(np.allclose(sol.x, np.array([1.0, 0.0, 0.0]), atol=1e-8))
        self.assertGreaterEqual(int(sol.info.get("presolve_implied_bound_updates", "0")), 2)
        self.assertGreaterEqual(int(sol.info.get("presolve_actions", "0")), 2)
        self.assertGreaterEqual(stats.presolve_implied_bound_updates or 0, 2)


if __name__ == "__main__":
    unittest.main()
