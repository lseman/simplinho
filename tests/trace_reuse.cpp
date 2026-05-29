// Focused trace: does RevisedSimplex reuse its FTBasis factorization across two
// solves of the SAME matrix when only bounds change (the BnB dual-reoptimize case)?
// Build: see command in chat. Run: ./trace_reuse
#include "../include/simplex/simplex.h"
#include <Eigen/Sparse>
#include <cstdio>

int main() {
    // Small standard-form LP: min c^T x, Ax = b, 0 <= x <= u.
    // 3 rows, give it slacks so it's a clean basis. We'll solve, then tighten a
    // bound and re-solve the SAME matrix to see if the factorization is reused.
    const int m = 3, n = 6; // 3 structural + 3 slack
    using Sp = RevisedSimplex::SparseMatrix;
    Sp A(m, n);
    std::vector<Eigen::Triplet<double>> T;
    // structural cols 0..2, slack cols 3..5 (identity)
    double a[3][3] = {{2, 1, 1}, {1, 3, 1}, {1, 1, 2}};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < 3; ++j)
            T.emplace_back(i, j, a[i][j]);
    for (int i = 0; i < m; ++i)
        T.emplace_back(i, 3 + i, 1.0);
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    Eigen::VectorXd b(m);
    b << 10, 12, 14;
    Eigen::VectorXd c(n);
    c << -3, -2, -4, 0, 0, 0; // minimize -> maximize 3x0+2x1+4x2
    Eigen::VectorXd l = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd u = Eigen::VectorXd::Constant(n, 1e30);

    RevisedSimplexOptions opt;
    opt.mode = SimplexMode::Dual;
    RevisedSimplex solver(opt);

    auto sol1 = solver.solve(A, b, c, l, u);
    std::printf("solve1: status=%d obj=%.4f iters=%d warm_attempted=%d warm_accepted=%d "
                "fact_reused=%d has_warm_state=%d\n",
                (int)sol1.status, sol1.obj, sol1.iters, sol1.solve_stats.warm_start_attempted,
                sol1.solve_stats.warm_start_accepted, sol1.solve_stats.warm_factorization_reused,
                (int)(bool)sol1.basis_state.warm_state);
    std::printf("  has_cached_basis_state(A)=%d has_cached_factorization_state(A)=%d\n",
                (int)solver.has_cached_basis_state(A),
                (int)solver.has_cached_basis_factorization_state(A));

    // Re-solve SAME matrix, NO bound change at all: pass the exact same problem
    // back. This is the cleanest possible reuse case -- the optimal basis is
    // unchanged, so reuse MUST be possible and the solve should take 0 iters.
    auto sol2 = solver.solve(A, b, c, l, u, sol1.basis_state);
    std::printf("solve2 (bound change, basis_state passed): status=%d obj=%.4f iters=%d "
                "warm_attempted=%d warm_accepted=%d fact_reused=%d\n",
                (int)sol2.status, sol2.obj, sol2.iters, sol2.solve_stats.warm_start_attempted,
                sol2.solve_stats.warm_start_accepted, sol2.solve_stats.warm_factorization_reused);

    // Re-solve SAME matrix, no basis_state passed (rely on internal cache, the
    // reformulated-solver hot path).
    Eigen::VectorXd u3 = u;
    u3(1) = 3.0;
    auto sol3 = solver.solve(A, b, c, l, u3);
    std::printf("solve3 (bound change, NO basis_state, internal cache): status=%d obj=%.4f "
                "iters=%d warm_attempted=%d warm_accepted=%d fact_reused=%d\n",
                (int)sol3.status, sol3.obj, sol3.iters, sol3.solve_stats.warm_start_attempted,
                sol3.solve_stats.warm_start_accepted, sol3.solve_stats.warm_factorization_reused);
    return 0;
}
