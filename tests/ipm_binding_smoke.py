"""Smoke test for the pybind11 IPM binding.

Run from the repository root after building:

    PYTHONPATH=build python tests/ipm_binding_smoke.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import simplinho  # noqa: E402


def main() -> None:
    # min x0 + 2*x1
    # s.t. x0 + x1 = 1, x >= 0
    # Optimum is x = [1, 0], objective = 1.
    A = np.array([[1.0, 1.0]], dtype=float)
    b = np.array([1.0], dtype=float)
    c = np.array([1.0, 2.0], dtype=float)
    lb = np.zeros(2, dtype=float)
    ub = np.array([math.inf, math.inf], dtype=float)

    solution = simplinho.solve_ipm(A, b, c, lb, ub, sense=["="], tol=1e-8)

    print("status:", solution.status)
    print("objective:", solution.objective)
    print("x:", np.array(solution.x))
    print("duals:", np.array(solution.duals))

    if not math.isfinite(solution.objective):
        raise SystemExit("IPM smoke failed: objective is not finite")
    if not np.allclose(np.array(solution.x), np.array([1.0, 0.0]), atol=1e-5):
        raise SystemExit("IPM smoke failed: unexpected primal solution")
    if not math.isclose(solution.objective, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise SystemExit("IPM smoke failed: unexpected objective")


if __name__ == "__main__":
    main()
