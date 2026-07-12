"""Benchmark the sparse simplex path against HiGHS.

Runs synthetic sparse equality-form LPs and the example MPS instances (as LP
relaxations) through both solvers. Reports status/objective parity, wall time,
iteration counts, and simplinho factorization stats (refactorizations, FT
updates, time split between LU build / pricing / pivoting) so regressions in
the sparse factorization path show up as numbers, not vibes.

Usage:
    python bench_sparse_vs_highs.py [--quick]
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
requested_build = os.environ.get("SIMPLINHO_BUILD_DIR")
build_dirs = ((requested_build,) if requested_build else
              ("build-local", "build-local/build", "build", "build-verify"))
for d in build_dirs:
    if not os.path.isabs(d):
        d = os.path.join(ROOT, d)
    cand = d
    if glob.glob(os.path.join(cand, "simplinho*.so")):
        sys.path.insert(0, cand)
        break

import highspy
import simplinho as splx


# ---------------------------------------------------------------------------
# Problem generators
# ---------------------------------------------------------------------------

def gen_sparse_eq(m, n, density, seed, finite_ub):
    """Equality-form LP: min c'x, [R | I] x = b, l <= x <= u, full row rank."""
    rng = np.random.default_rng(seed)
    R = sp.random(m, n, density=density, random_state=rng,
                  data_rvs=lambda k: rng.standard_normal(k))
    A = sp.hstack([R, sp.identity(m)], format="csc")
    ntot = n + m
    x0 = rng.random(ntot) * 2.0
    b = A @ x0
    l = np.zeros(ntot)
    if finite_ub:
        u = np.full(ntot, 5.0)
        c = rng.standard_normal(ntot)
    else:
        u = np.full(ntot, np.inf)
        # Construct a dual-feasible c so the LP is bounded: c = A'y + s, s > 0.
        y = rng.standard_normal(m)
        c = np.asarray(A.T @ y).ravel() + rng.random(ntot) * 0.9 + 0.1
    return A, b, c, l, u


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solve_simplinho(A, b, c, l, u):
    opt = splx.RevisedSimplexOptions()
    solver = splx.RevisedSimplex(opt)
    t0 = time.perf_counter()
    sol = solver.solve(sp.csc_matrix(A), b, c, l, u)
    dt = time.perf_counter() - t0
    st = sol.stats
    total_ns = max(st.lu_build_ns + st.pricing_build_ns + st.pivot_ns, 1)
    return {
        "status": str(sol.status).split(".")[-1],
        "obj": sol.obj,
        "time": dt,
        "iters": st.iterations,
        "refactors": st.refactorizations,
        "ft_updates": st.ft_updates,
        "lu_pct": 100.0 * st.lu_build_ns / total_ns,
        "price_pct": 100.0 * st.pricing_build_ns / total_ns,
        "pivot_pct": 100.0 * st.pivot_ns / total_ns,
    }


def new_highs():
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "simplex")
    h.setOptionValue("threads", 1)
    h.setOptionValue("presolve", "off")
    return h


def solve_highs_eq(A, b, c, l, u):
    h = new_highs()
    A = sp.csc_matrix(A)
    m, n = A.shape
    inf = highspy.kHighsInf
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = c
    lp.col_lower_ = l
    lp.col_upper_ = np.where(np.isinf(u), inf, u)
    lp.row_lower_ = b
    lp.row_upper_ = b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    info = h.getInfo()
    return {
        "status": h.modelStatusToString(h.getModelStatus()),
        "obj": info.objective_function_value,
        "time": dt,
        "iters": info.simplex_iteration_count,
    }


def solve_highs_mps(path):
    h = new_highs()
    h.readModel(path)
    # LP relaxation: drop integrality.
    n = h.getNumCol()
    h.changeColsIntegrality(
        n, np.arange(n, dtype=np.int32),
        np.full(n, highspy.HighsVarType.kContinuous))
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    info = h.getInfo()
    return {
        "status": h.modelStatusToString(h.getModelStatus()),
        "obj": info.objective_function_value,
        "time": dt,
        "iters": info.simplex_iteration_count,
    }


# ---------------------------------------------------------------------------
# MPS -> simplinho Model (LP relaxation)
# ---------------------------------------------------------------------------

def parse_mps(path):
    rows, row_order, obj_row = {}, [], None
    cols, col_order, obj = {}, [], {}
    rhs, bounds = {}, {}
    section, in_int = None, False
    for raw in open(path):
        if not raw.strip() or raw.startswith("*"):
            continue
        if raw[0] not in " \t":
            section = raw.split()[0]
            continue
        f = raw.split()
        if section == "ROWS":
            sense, name = f[0], f[1]
            if sense == "N":
                obj_row = name
            else:
                rows[name] = sense
                row_order.append(name)
        elif section == "COLUMNS":
            if len(f) >= 3 and f[1] == "'MARKER'":
                in_int = f[2] == "'INTORG'"
                continue
            var = f[0]
            if var not in cols:
                cols[var] = {}
                col_order.append(var)
            for i in range(1, len(f), 2):
                r, v = f[i], float(f[i + 1])
                if r == obj_row:
                    obj[var] = obj.get(var, 0.0) + v
                else:
                    cols[var][r] = cols[var].get(r, 0.0) + v
        elif section == "RHS":
            for i in range(1, len(f), 2):
                rhs[f[i]] = float(f[i + 1])
        elif section == "BOUNDS":
            btype, var = f[0], f[2]
            val = float(f[3]) if len(f) > 3 else None
            bb = bounds.setdefault(var, [0.0, float("inf")])
            if btype in ("UP", "UI"):
                bb[1] = val
            elif btype in ("LO", "LI"):
                bb[0] = val
            elif btype == "FX":
                bb[0] = bb[1] = val
            elif btype == "FR":
                bb[0], bb[1] = float("-inf"), float("inf")
            elif btype == "MI":
                bb[0] = float("-inf")
            elif btype == "PL":
                bb[1] = float("inf")
            elif btype == "BV":
                bb[0], bb[1] = 0.0, 1.0
    return rows, row_order, cols, col_order, obj, rhs, bounds


def solve_simplinho_mps(path):
    rows, row_order, cols, col_order, obj, rhs, bounds = parse_mps(path)
    model = splx.Model()
    var = {}
    for name in col_order:
        lb, ub = bounds.get(name, [0.0, float("inf")])
        var[name] = model.add_var(name=name, lb=lb, ub=ub,
                                  obj=obj.get(name, 0.0))
    for rname in row_order:
        expr = splx.LinearExpr()
        for vname in col_order:
            coef = cols[vname].get(rname)
            if coef:
                expr += coef * var[vname]
        rb = rhs.get(rname, 0.0)
        sense = rows[rname]
        con = (expr == rb) if sense == "E" else (expr <= rb) if sense == "L" else (expr >= rb)
        model.add_constr(con, name=rname)
    model.minimize(sum(obj.get(n, 0.0) * var[n] for n in col_order))
    t0 = time.perf_counter()
    sol = model.solve()
    dt = time.perf_counter() - t0
    st = sol.stats
    total_ns = max(st.lu_build_ns + st.pricing_build_ns + st.pivot_ns, 1) if st else 1
    return {
        "status": str(sol.status).split(".")[-1],
        "obj": sol.obj,
        "time": dt,
        "iters": st.iterations if st else -1,
        "refactors": st.refactorizations if st else -1,
        "ft_updates": st.ft_updates if st else -1,
        "lu_pct": 100.0 * st.lu_build_ns / total_ns if st else 0.0,
        "price_pct": 100.0 * st.pricing_build_ns / total_ns if st else 0.0,
        "pivot_pct": 100.0 * st.pivot_ns / total_ns if st else 0.0,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def rel_diff(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    sizes = [(100, 200, 0.05), (300, 600, 0.02), (800, 1600, 0.01)]
    if args.quick:
        sizes = sizes[:2]

    cases = []
    for m, n, dens in sizes:
        for finite_ub in (False, True):
            tag = f"eq_{m}x{n+m}_d{dens}_{'ub' if finite_ub else 'inf'}"
            cases.append((tag, gen_sparse_eq(m, n, dens, seed=m + n, finite_ub=finite_ub)))

    hdr = (f"{'instance':30s} {'st_ok':5s} {'obj_ok':6s} "
           f"{'t_splx':>8s} {'t_highs':>8s} {'ratio':>7s} "
           f"{'it_splx':>7s} {'it_hi':>6s} {'refac':>5s} {'ft_up':>6s} "
           f"{'lu%':>5s} {'pr%':>5s} {'pv%':>5s}")
    print(hdr)
    print("-" * len(hdr))

    def report(tag, s, h):
        st_ok = ("Optimal" in s["status"]) == ("Optimal" in h["status"])
        obj_ok = st_ok and ("Optimal" not in s["status"]
                            or rel_diff(s["obj"], h["obj"]) < 1e-5)
        ratio = s["time"] / max(h["time"], 1e-9)
        print(f"{tag:30s} {str(st_ok):5s} {str(obj_ok):6s} "
              f"{s['time']:8.3f} {h['time']:8.3f} {ratio:7.1f} "
              f"{s['iters']:7d} {h['iters']:6d} {s.get('refactors', -1):5d} "
              f"{s.get('ft_updates', -1):6d} "
              f"{s.get('lu_pct', 0):5.1f} {s.get('price_pct', 0):5.1f} "
              f"{s.get('pivot_pct', 0):5.1f}")
        if not obj_ok:
            print(f"    !! splx: {s['status']} obj={s['obj']:.8g} | "
                  f"highs: {h['status']} obj={h['obj']:.8g}")
        return obj_ok

    all_ok = True
    for tag, (A, b, c, l, u) in cases:
        s = solve_simplinho(A, b, c, l, u)
        h = solve_highs_eq(A, b, c, l, u)
        all_ok &= report(tag, s, h)

    for mps in sorted(glob.glob(os.path.join(ROOT, "examples", "*.mps"))):
        tag = "mps_" + os.path.basename(mps).replace(".mps", "")
        s = solve_simplinho_mps(mps)
        h = solve_highs_mps(mps)
        all_ok &= report(tag, s, h)

    print("-" * len(hdr))
    print("PARITY:", "OK" if all_ok else "MISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
