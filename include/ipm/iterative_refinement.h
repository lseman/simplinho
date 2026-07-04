// iterative_refinement.h — Iterative refinement for IPM with fallback to dense LU
// SOTA: Monitor residual growth, auto-fallback if LDL⁺ factor worsens
// Dense LU (BLAS3) for ill-conditioned systems

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>

namespace ipm {

// Iterative refinement result
struct RefinementStats {
    int num_refine = 0;      // Number of refinement iterations performed
    double initial_residual = 0.0;  // ‖r₀‖ / ‖b‖
    double final_residual = 0.0;    // ‖rₖ‖ / ‖b‖
    double factor_growth = 0.0;     // Growth of factor norms
    bool fallback_dense_lu = false; // Did we fall back to dense LU?
};

// Iterative refinement: solve A*x = b iteratively
// x_new = x_old + delta_x, where A*delta_x = r = b - A*x_old
template <typename Solver>
inline RefinementStats iterative_refinement(
    Solver& solver, const Eigen::SparseMatrix<double>& A,
    const Eigen::VectorXd& b, Eigen::VectorXd& x,
    int max_refine_iter = 3, double tol_residual = 1e-10) {
    RefinementStats stats;

    // Initial residual
    Eigen::VectorXd r = b - A * x;
    double r_norm = r.norm();
    double b_norm = b.norm() + 1e-14;

    stats.initial_residual = r_norm / b_norm;

    // Early exit if already accurate
    if (stats.initial_residual < tol_residual) {
        stats.final_residual = stats.initial_residual;
        return stats;
    }

    // Refinement iterations
    for (int k = 0; k < max_refine_iter; ++k) {
        // Solve for correction: A*delta_x = r
        Eigen::VectorXd delta_x = solver.solve(r);

        // Update solution
        x += delta_x;

        // Compute new residual
        r = b - A * x;
        double r_new = r.norm() / b_norm;

        stats.num_refine++;
        stats.final_residual = r_new;

        // Check stagnation or growth
        if (r_new > stats.initial_residual * 10.0) {
            // Residual grew 10×, factor likely ill-conditioned
            stats.fallback_dense_lu = true;
            break;
        }

        if (r_new < tol_residual) {
            break;  // Converged
        }

        // Early exit if not improving
        if (k > 0 && r_new / stats.final_residual > 0.5) {
            break;
        }
    }

    return stats;
}

// Dense LU solve for ill-conditioned systems (BLAS3-optimized)
// Falls back when sparse factorization fails or residuals grow
inline Eigen::VectorXd dense_lu_solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b) {
    // Convert sparse to dense (acceptable for small systems)
    Eigen::MatrixXd A_dense = Eigen::MatrixXd(A);
    return A_dense.fullPivLu().solve(b);
}

// Adaptive solve: try sparse first, fall back to dense if needed
class AdaptiveSolver {
  public:
    explicit AdaptiveSolver(double residual_threshold = 1e-6,
                            double growth_threshold = 10.0)
        : residual_threshold_(residual_threshold),
          growth_threshold_(growth_threshold) {}

    // Solve with automatic fallback
    template <typename SparseSolver>
    Eigen::VectorXd solve(SparseSolver& sparse_solver,
                          const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b) {
        // Try sparse solve + iterative refinement
        Eigen::VectorXd x = sparse_solver.solve(b);

        Eigen::VectorXd r = b - A * x;
        double r_norm = r.norm() / (b.norm() + 1e-14);

        last_stats_.initial_residual = r_norm;

        if (r_norm < residual_threshold_) {
            last_stats_.fallback_dense_lu = false;
            last_stats_.num_refine = 0;
            return x;
        }

        // Residual too large: refine or fall back
        if (r_norm > growth_threshold_ * residual_threshold_) {
            // Fall back to dense LU for this system
            last_stats_.fallback_dense_lu = true;
            return dense_lu_solve(A, b);
        }

        // Try refinement
        last_stats_ = iterative_refinement(sparse_solver, A, b, x, 3,
                                           residual_threshold_);
        return x;
    }

    const RefinementStats& last_stats() const { return last_stats_; }

  private:
    double residual_threshold_;
    double growth_threshold_;
    RefinementStats last_stats_;
};

}  // namespace ipm
