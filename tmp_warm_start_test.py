import importlib.util
from pathlib import Path

import numpy as np


def load_simplinho():
    build_dir = Path("build")
    module_path = next(build_dir.glob("simplinho*.so"), None)
    if module_path is None:
        raise FileNotFoundError(
            "Could not find simplinho extension module in build directory"
        )
    spec = importlib.util.spec_from_file_location("simplinho", module_path)
    simplinho = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(simplinho)
    return simplinho


def print_basis_info(label, basis):
    if basis is None:
        print(f"{label}: no basis")
        return
    try:
        size = len(basis.column_status)
    except Exception:
        size = "unknown"
    print(f"{label}: basis available, column_status length = {size}")


def main():
    simplinho = load_simplinho()
    print("Loaded simplinho from build")

    # Problem: maximize x0 + 2*x1 subject to x0 + x1 = 5, x >= 0.
    # In minimization form, objective is min [-1, -2]^T x.
    A = np.array([[1.0, 1.0]])
    b = np.array([5.0])
    c = np.array([-1.0, -2.0])
    l = np.array([0.0, 0.0])
    u = np.array([np.inf, np.inf])

    options = simplinho.RevisedSimplexOptions()
    options.verbose = True
    options.verbose_every = 1
    solver = simplinho.RevisedSimplex(options)
    print("Solving initial LP...")
    sol1 = solver.solve(A, b, c, l, u)
    print("Initial status:", sol1.status)
    print("Initial objective:", sol1.obj)
    print("Initial primal:", sol1.x)
    print_basis_info("Initial basis", sol1.basis_state)
    if sol1.status != simplinho.LPStatus.Optimal:
        print("Initial solve not optimal; aborting warm-start test.")
        return

    old_basis = sol1.basis_state
    if old_basis is None:
        print("Initial basis_state is None; warm start cannot be tested.")
        return

    print("Updating bounds for x0 from [0, 10] to [1, 10].")
    l_warm = l.copy()
    u_warm = u.copy()
    l_warm[0] = 1.0

    print("Solving warm-started LP with previous basis...")
    sol2 = solver.solve(A, b, c, l_warm, u_warm, old_basis)
    print("Warm-start status:", sol2.status)
    print("Warm-start objective:", sol2.obj)
    print("Warm-start primal:", sol2.x)
    print_basis_info("Warm-start basis", sol2.basis_state)
    print(
        "Warm-start used basis_state?", getattr(sol2, "basis_state", None) is not None
    )
    print(
        "Warm-start info keys:",
        list(sol2.info.keys()) if hasattr(sol2, "info") else "<no info>",
    )

    print("Solving the same LP without providing basis for comparison...")
    sol3 = solver.solve(A, b, c, l_warm, u_warm)
    print("Cold solve status:", sol3.status)
    print("Cold solve objective:", sol3.obj)
    print("Cold solve primal:", sol3.x)
    print_basis_info("Cold solve basis", sol3.basis_state)

    print("Test complete.")


if __name__ == "__main__":
    main()
