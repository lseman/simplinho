// Focused study of the dual warm-restart pivot count on a bounded-variable
// (reformulation-triggering) LP, isolated from BnB. Mirrors the knapsack repro
// shape: N binaries (0<=x<=1) with a few <= capacity rows, maximize profit.
// Solve once (Dual), then apply a branching bound change (fix one var) and
// re-solve passing the previous basis_state. Report iters on the warm re-solve.
//
// Goal: reproduce the ~196-iter warm-restart thrash in a standalone so the dual
// engine can be studied/fixed directly.
#include "../include/simplex/simplex.h"
#include <Eigen/Sparse>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : 100;
    using Sp = RevisedSimplex::SparseMatrix;
    // Rows: 3 knapsack capacity rows, all <= . Convert to == with slacks.
    // Variables: N structural (0<=x<=1) + 3 slacks (>=0).
    const int nrows = 3;
    const int n = N + nrows;
    std::vector<Eigen::Triplet<double>> T;
    Eigen::VectorXd c = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd b(nrows);
    double ws[3] = {0, 0, 0};
    for (int i = 0; i < N; ++i) {
        const double profit = double(((17 * i + 13) % 97) + 10);
        c(i) = -profit; // minimize -> maximize profit
        const double w1 = double(((11 * i + 7) % 29) + 1);
        const double w2 = double(((19 * i + 5) % 31) + 1);
        const double w3 = double(((23 * i + 3) % 37) + 1);
        ws[0] += w1; ws[1] += w2; ws[2] += w3;
        T.emplace_back(0, i, w1);
        T.emplace_back(1, i, w2);
        T.emplace_back(2, i, w3);
    }
    for (int r = 0; r < nrows; ++r)
        T.emplace_back(r, N + r, 1.0); // slack
    b << 0.35 * ws[0], 0.33 * ws[1], 0.31 * ws[2];
    Sp A(nrows, n);
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    Eigen::VectorXd l = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd u = Eigen::VectorXd::Constant(n, 1e30);
    for (int i = 0; i < N; ++i)
        u(i) = 1.0; // binaries relaxed to [0,1]

    RevisedSimplexOptions opt;
    opt.mode = (argc > 3 && std::string(argv[3]) == "auto") ? SimplexMode::Auto : SimplexMode::Dual;
    RevisedSimplex solver(opt);

    // Cold root: in Auto so it succeeds even if pure-Dual cold-solve is weak.
    RevisedSimplexOptions root_opt = opt;
    root_opt.mode = SimplexMode::Auto;
    RevisedSimplex root_solver(root_opt);
    auto root = root_solver.solve(A, b, c, l, u);
    std::printf("root(Auto): status=%d obj=%.4f iters=%d\n", (int)root.status, root.obj, root.iters);

    // Branching: walk down a chain, each time fixing the most-fractional var to 0
    // or 1, and re-solve from the parent basis_state (the BnB dual-reoptimize case).
    Eigen::VectorXd lb = l, ub = u;
    LPBasis basis = root.basis_state;
    long total_iters = 0;
    int depth = (argc > 2) ? std::atoi(argv[2]) : 30;
    for (int d = 0; d < depth; ++d) {
        // pick most-fractional structural var still free
        int jbest = -1;
        double bestfrac = 1e-6;
        for (int i = 0; i < N; ++i) {
            if (ub(i) < 0.99) continue; // already fixed/tightened
            const double xi = (i < root.x.size()) ? root.x(i) : 0.0;
            const double f = std::min(xi - std::floor(xi), std::ceil(xi) - xi);
            if (f > bestfrac) { bestfrac = f; jbest = i; }
        }
        if (jbest < 0) break;
        // Branch mode: argv[4]=="tighten" keeps the var bounded (ub=0.5) so the
        // reformulation STRUCTURE is unchanged (RHS-only) -- tests whether the
        // warm restart works cheaply when there's no structural cache miss.
        // Default: fix to 0 (structural change, the real binary-branch case).
        const bool tighten = (argc > 4 && std::string(argv[4]) == "tighten");
        if (tighten) ub(jbest) = 0.5; else ub(jbest) = 0.0;
        // Use the persistent root_solver (mirrors the BnB persistent solver +
        // reformulation cache) so warm restart can actually engage.
        auto child = root_solver.solve(A, b, c, lb, ub, basis);
        total_iters += child.iters;
        std::printf("depth=%2d fix x%d<=0 status=%d obj=%.4f iters=%d fact_reused=%d "
                    "warm_acc=%d\n",
                    d, jbest, (int)child.status, child.obj, child.iters,
                    child.solve_stats.warm_factorization_reused,
                    child.solve_stats.warm_start_accepted);
        if (child.status == LPSolution::Status::Optimal) {
            basis = child.basis_state;
            root = child; // update fractional reference
        } else {
            break;
        }
    }
    std::printf("total warm-restart iters over chain = %ld\n", total_iters);
    return 0;
}
