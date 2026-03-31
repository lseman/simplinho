import math
import unittest
from pathlib import Path

try:
    from gurobipy import GRB
    from bnb.solver.core import solve_lp_relaxation
    from bnb.solver.model_data import MIPData, normalize_lp_backend

    HAS_BNB_DEPS = True
except ModuleNotFoundError as exc:
    if exc.name != "gurobipy":
        raise
    GRB = None
    solve_lp_relaxation = None
    MIPData = None
    normalize_lp_backend = None
    HAS_BNB_DEPS = False


ROOT = Path(__file__).resolve().parents[1]


def _has_built_simplinho():
    for build_dir in ("build-local", "build", "build-verify"):
        candidate = ROOT / build_dir
        if not candidate.exists():
            continue
        if any(candidate.glob("simplinho*.so")) or any(candidate.glob("simplex*.so")):
            return True
    return False


@unittest.skipUnless(
    _has_built_simplinho() and HAS_BNB_DEPS,
    "requires a locally built simplinho module and gurobipy-backed bnb imports",
)
class SimplinhoBackendTests(unittest.TestCase):
    def test_backend_alias_normalization(self):
        self.assertEqual(normalize_lp_backend("simplex"), "simplinho")
        self.assertEqual(normalize_lp_backend("simplinho"), "simplinho")
        self.assertEqual(normalize_lp_backend("gurobi"), "gurobi")

    def test_lp_relaxation_solves_with_simplinho_backend(self):
        mip_data = MIPData(
            c={"x": 1.0, "y": 1.0},
            var_types={"x": GRB.BINARY, "y": GRB.BINARY},
            var_names=["x", "y"],
            sense=GRB.MAXIMIZE,
            constraints={
                "cap": {
                    "sense": GRB.LESS_EQUAL,
                    "rhs": 2.5,
                    "expr": {"x": 1.0, "y": 2.0},
                }
            },
            lb={"x": 0.0, "y": 0.0},
            ub={"x": 1.0, "y": 1.0},
        )

        (
            solution,
            obj_value,
            is_integer,
            int_vars,
            bound,
            _active_cuts,
            _fixed,
            _lower_bounds,
            _upper_bounds,
            relax_info,
        ) = solve_lp_relaxation(
            mip_data,
            fixed_vars={},
            is_maximization=True,
            lp_backend="simplinho",
        )

        self.assertIsNotNone(solution)
        self.assertFalse(is_integer)
        self.assertEqual(int_vars, {"x", "y"})
        self.assertTrue(math.isclose(obj_value, 1.75, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(bound, 1.75, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution["x"], 1.0, rel_tol=0.0, abs_tol=1e-8))
        self.assertTrue(math.isclose(solution["y"], 0.75, rel_tol=0.0, abs_tol=1e-8))
        self.assertEqual(relax_info["reduced_cost_fixings"], 0)


if __name__ == "__main__":
    unittest.main()
