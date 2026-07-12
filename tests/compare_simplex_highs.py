import numpy as np
import scipy.sparse as sp
import highspy
import simplinho as splx

np.random.seed(42)

def gen_dense_lp(m, n, seed=42):
    np.random.seed(seed)
    A = np.random.randn(m, n)
    x_true = np.abs(np.random.randn(n)) + 1.0
    b = A @ x_true
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    return A, b, c, lb, ub

def gen_structured_lp(m, n, seed=42):
    np.random.seed(seed)
    blocks = 5
    block_size = n // blocks
    A = np.zeros((m, n))
    for i in range(blocks):
        start, end = i * block_size, (i + 1) * block_size
        A[i*2:i*2+2, start:end] = np.random.randn(2, block_size)
    for i in range(m - 2*blocks):
        A[2*blocks + i] = np.random.randn(n) * 0.1
    x_true = np.abs(np.random.randn(n)) + 1.0
    b = A @ x_true
    c = np.random.randn(n)
    lb = np.zeros(n)
    ub = np.full(n, np.inf)
    return A, b, c, lb, ub

def solve_highs(A, b, c, lb, ub):
    m, n = A.shape
    highs = highspy.Highs()
    highs.clear()
    highs.setOptionValue("log_to_console", False)
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")  # Force simplex solver
    
    inf_val = highs.getInfinity()
    
    # Add variables with bounds
    for i in range(n):
        highs.addVar(lb[i] if np.isfinite(lb[i]) else -inf_val,
                     ub[i] if np.isfinite(ub[i]) else inf_val)
    
    # Add costs
    c_arr = np.array(c, dtype=np.float64)
    highs.changeColsCost(n, np.arange(n, dtype=np.int32), c_arr)
    
    # Add constraints
    for i in range(m):
        row_indices = []
        row_coeffs = []
        if isinstance(A, np.ndarray):
            for j in range(n):
                if A[i, j] != 0:
                    row_indices.append(j)
                    row_coeffs.append(A[i, j])
        else:
            for j, v in zip(A.indices[A.indptr[i]:A.indptr[i+1]], A.data[A.indptr[i]:A.indptr[i+1]]):
                row_indices.append(j)
                row_coeffs.append(v)
        
        row_indices_arr = np.array(row_indices, dtype=np.int32)
        row_coeffs_arr = np.array(row_coeffs, dtype=np.float64)
        highs.addRow(b[i], b[i], len(row_indices), row_indices_arr, row_coeffs_arr)
    
    highs.run()
    
    status_map = {
        highspy.HighsModelStatus.kOptimal: splx.LPStatus.Optimal,
        highspy.HighsModelStatus.kInfeasible: splx.LPStatus.Infeasible,
        highspy.HighsModelStatus.kUnboundedOrInfeasible: splx.LPStatus.Unbounded,
        highspy.HighsModelStatus.kUnbounded: splx.LPStatus.Unbounded,
    }
    
    status = highs.getModelStatus()
    sol_status = status_map.get(status, splx.LPStatus.Infeasible)
    
    obj = highs.getObjectiveValue() if status == highspy.HighsModelStatus.kOptimal else None
    x = np.array([highs.getSolution().col_value[i] for i in range(n)]) if status == highspy.HighsModelStatus.kOptimal else None
    
    # Get simplex iterations from highs info
    info = highs.getInfo()
    simplex_iters = info.simplex_iteration_count if hasattr(info, 'simplex_iteration_count') else 0
    
    return obj, x, sol_status, simplex_iters

def solve_simplinho_simplex(A, b, c, lb, ub):
    if isinstance(A, np.ndarray):
        A_sparse = sp.csr_matrix(A)
    else:
        A_sparse = A
    
    options = splx.RevisedSimplexOptions()
    options.mode = splx.SimplexMode.Auto
    options.max_iters = 10000
    
    sol = splx.RevisedSimplex(options).solve(A_sparse, b, c, lb, ub)
    
    obj = sol.obj if sol.status == splx.LPStatus.Optimal else None
    x = np.array(sol.x) if sol.status == splx.LPStatus.Optimal else None
    status = sol.status
    
    # Get iterations from solve_stats or info
    iters = sol.iters if hasattr(sol, "iters") else 0
    refactorizations = sol.solve_stats.refactorizations if hasattr(sol, "solve_stats") else 0
    
    return obj, x, status, iters, refactorizations

# Test problems
problems = [
    ("Dense Small", gen_dense_lp(20, 30)),
    ("Structured Medium", gen_structured_lp(50, 80)),
]

for prob_name, (A, b, c, lb, ub) in problems:
    print(f"\n{'='*60}")
    print(f"Problem: {prob_name}")
    print(f"A shape: {A.shape}")
    print(f"{'='*60}")
    
    # HiGHS
    print("Solving with HiGHS (simplex)...")
    highs_obj, highs_x, highs_status, highs_iters = solve_highs(A, b, c, lb, ub)
    print(f"  HiGHS Status: {highs_status}, Iters: {highs_iters}, Obj: {highs_obj if highs_obj is not None else 'N/A'}")
    
    # simplinho simplex
    print("Solving with simplinho RevisedSimplex...")
    simp_obj, simp_x, simp_status, simp_iters, simp_refactors = solve_simplinho_simplex(A, b, c, lb, ub)
    print(f"  Simplinho Status: {simp_status}, Iters: {simp_iters}, Refactors: {simp_refactors}, Obj: {simp_obj if simp_obj is not None else 'N/A'}")
    
    # Comparison
    if highs_obj is not None and simp_obj is not None:
        obj_diff = abs(simp_obj - highs_obj) / (abs(highs_obj) + 1e-10)
        print(f"\n  Objective diff: {obj_diff:.2e} (relative)")
        print(f"  HiGHS iters: {highs_iters}, Simplinho iters: {simp_iters}")
        if highs_iters > 0:
            print(f"  Iter ratio (simplinho/highs): {simp_iters/highs_iters:.2f}x")
