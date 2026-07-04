#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>

#ifdef GUROBI
#    include "gurobi_c++.h"
#    include "gurobi_c.h"
#endif

#include <iostream>
#include <new>
#include <stdexcept>
#include <vector>

#include "linear_system/supernodes.h"
#include "linear_system/ldlt.h"
#include "linear_system/schur_frontal_ldlt.h"
#include "ipm/ipm_linear_solver.h"
#include "ipm/diagonal_regularization.h"
#include "ipm/iterative_refinement.h"

/**
 * @struct OptimizationData
 * @brief A structure to hold data for an optimization problem.
 *
 * This structure contains the following members:
 * - As: A sparse matrix representing the constraint coefficients.
 * - bs: A vector representing the right-hand side of the constraints.
 * - cs: A vector representing the coefficients of the objective function.
 * - lo: A vector representing the lower bounds of the variables.
 * - hi: A vector representing the upper bounds of the variables.
 * - sense: A vector representing the sense of the constraints (e.g., equality,
 * inequality).
 */
struct OptimizationData {
    Eigen::SparseMatrix<double> As;
    Eigen::VectorXd bs;
    Eigen::VectorXd cs;
    Eigen::VectorXd lo;
    Eigen::VectorXd hi;
    Eigen::VectorXd sense;
};

/**
 * @struct Residuals
 * @brief A structure to hold various residual vectors and their norms.
 *
 * This structure contains the following members:
 * - rp: Residual vector for primal feasibility.
 * - ru: Residual vector for dual feasibility.
 * - rd: Residual vector for duality gap.
 * - rl: Residual vector for lower bounds.
 * - rpn: Norm of the primal feasibility residual.
 * - run: Norm of the dual feasibility residual.
 * - rdn: Norm of the duality gap residual.
 * - rgn: Norm of the gradient residual.
 * - rg: Gradient residual.
 * - rln: Norm of the lower bounds residual.
 */
struct Residuals {
    Eigen::VectorXd rp, ru, rd, rl;
    double rpn, run, rdn, rgn, rg, rln;
};

#ifdef GUROBI
OptimizationData extractOptimizationComponents(GRBModel& model);
#endif

/**
 * @class SparseSolver
 * @brief A class for solving sparse linear systems using various solver types.
 *
 * The SparseSolver class provides an interface for factorizing and solving
 * sparse linear systems. It supports different solver types, with the default
 * being CHOLMOD.
 *
 */
class SparseSolver {
  public:
    int n;
    int m;
    Eigen::VectorXd theta;
    Eigen::VectorXd regP;
    Eigen::VectorXd regD;
    Eigen::SparseMatrix<double> A;
    Eigen::SparseMatrix<double> S;
    Eigen::SparseMatrix<double> AD;
    Eigen::SparseMatrix<double> D;
    std::vector<Eigen::SparseMatrix<double>::StorageIndex> primalDiagPos;
    std::vector<Eigen::SparseMatrix<double>::StorageIndex> dualDiagPos;
    bool firstFactorization = true;

    enum SolverType {
        LDLT,
        SPARSE_LU,
    };

    SparseSolver(SolverType type = SPARSE_LU) {
        switch (type) {
            case LDLT:
                solver = new SolverWrapper<Eigen::CustomSimplicialLDLT<
                    Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>>>();
                break;
            case SPARSE_LU:
                solver = new SparseLUWrapper();
                break;
        }
    }

    ~SparseSolver() { delete solver; }

    void factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& matrix) {
        solver->factorizeMatrix(matrix);
    }

    void reset() { solver->reset(); }

    int info() { return solver->info(); }

    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) { return solver->solve(rhs); }

  private:
    SolverType solverType;

    struct SolverBase {
        virtual void
        factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& matrix) = 0;
        virtual Eigen::VectorXd solve(const Eigen::VectorXd& rhs) = 0;
        virtual ~SolverBase() = default;
        virtual void reset() = 0;
        virtual int info() = 0;
    };

    template <typename Solver> struct SolverWrapper : public SolverBase {
        Solver solver;
        void
        factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& matrix) override {
            solver.factorizeMatrix(matrix);
        }
        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override { return solver.solve(rhs); }
        void reset() override { solver.reset(); }
        int info() override { return solver.info(); }
    };

    struct SparseLUWrapper : public SolverBase {
        Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::ColMajor, int>,
                        Eigen::COLAMDOrdering<int>>
            solver;

        void
        factorizeMatrix(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& matrix) override {
            solver.compute(matrix);
        }
        Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override { return solver.solve(rhs); }
        void reset() override {}
        int info() override { return solver.info(); }
    };

    SolverBase* solver;
};

/**
 * @brief Performs Ruiz scaling on the optimization problem data.
 *
 * This function implements the Ruiz equilibration algorithm to improve the
 * numerical conditioning of the optimization problem. It iteratively scales the
 * rows and columns of the constraint matrix A and adjusts the corresponding
 * vectors accordingly.
 */
struct ScalingFactors {
    Eigen::VectorXd row_scaling; // Diagonal elements of row scaling matrix
    Eigen::VectorXd col_scaling; // Diagonal elements of column scaling matrix
};

/**
 * @class IPSolver
 * @brief A class for solving linear programming problems using an interior
 * point method.
 *
 * This class provides methods to convert linear programming problems to
 * standard form, update residuals, solve augmented systems, and run the
 * optimization process.
 *
 */
class IPSolver {
  public:
    Residuals res;
    std::unique_ptr<ipm::IPMLinearSolver> ls;  // Hybrid sparse/frontal solver
    double tau, kappa, tol;
    int max_iter;
    double infty = std::numeric_limits<double>::infinity();
    ipm::RegularizationControl reg_ctrl;  // Adaptive regularization
    ipm::AdaptiveSolver adaptive_solver;  // Iterative refinement + fallback

    // Create history of the old values
    Eigen::VectorXd x_old;
    Eigen::VectorXd lambda_old;
    Eigen::VectorXd s_old;
    Eigen::VectorXd v_old;
    Eigen::VectorXd w_old;
    double tau_old;
    double kappa_old;
    int n_slacks_old = 0;
    int n_slacks = 0;
    bool warm_start = false;

    std::vector<double> dual_vals;
    std::vector<double> primal_vals;
    double objVal;

    const std::vector<double>& getDualsRef() const { return dual_vals; }
    const std::vector<double>& getPrimalsRef() const { return primal_vals; }
    std::vector<double> getDuals() const { return dual_vals; }
    std::vector<double> getPrimals() const { return primal_vals; }
    double getObjective() const { return objVal; }

    IPSolver() {}
    Eigen::SparseMatrix<double> convertToSparseDiagonal(const Eigen::VectorXd& vec);

    void save_interior_solution(const Eigen::VectorXd& x, const Eigen::VectorXd& lambda,
                                const Eigen::VectorXd& s, const Eigen::VectorXd& v,
                                const Eigen::VectorXd& w, double tau, double kappa) {
        x_old = x;
        lambda_old = lambda;
        s_old = s;
        v_old = v;
        w_old = w;
        tau_old = tau;
        kappa_old = kappa;
    }

    /**
     * Convert a bounded LP into the equality form used by this IPM.
     *
     * `sense(i) == 1` marks row i as an equality. Other values are treated as
     * one-sided rows and receive a nonnegative slack. Lower bounds are shifted
     * into the right-hand side, upper-only variables are sign-flipped, and free
     * variables are split into positive and negative parts.
     */
    void convert_to_standard_form(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                                  const Eigen::VectorXd& c, const Eigen::VectorXd& lb,
                                  const Eigen::VectorXd& ub, const Eigen::VectorXd& sense,
                                  Eigen::SparseMatrix<double>& As, Eigen::VectorXd& bs,
                                  Eigen::VectorXd& cs);

    /**
     * Recompute homogeneous self-dual residuals for the current iterate.
     *
     * The residuals include finite upper-bound terms through `ubi/ubv` and are
     * reused by the predictor, centrality corrector, and higher-order correction
     * solves.
     */
    void update_residuals(Residuals& res, const Eigen::VectorXd& x, const Eigen::VectorXd& lambda,
                          const Eigen::VectorXd& s, const Eigen::VectorXd& v,
                          const Eigen::VectorXd& w, const Eigen::SparseMatrix<double>& A,
                          const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                          const Eigen::VectorXd& ubv, const Eigen::VectorXi& ubi, double tau,
                          double kappa);

    /**
     * Solve the regularized augmented KKT system for `(dx, dy)`.
     *
     * The factorization is owned by `ls`, so an IPM iteration can reuse it for
     * the predictor and all correction directions.
     */
    void solve_augmented_system(Eigen::VectorXd& dx, Eigen::VectorXd& dy,
                                ipm::IPMLinearSolver& ls, const Eigen::VectorXd& xi_p,
                                const Eigen::VectorXd& xi_d);

    /**
     * Solve the augmented system and recover the finite-upper-bound direction.
     *
     * `theta_vw` is the diagonal w/v ratio for variables with finite upper
     * bounds. `xi_u` is eliminated before the solve, then `delta_z` is recovered
     * from the resulting `delta_x`.
     */
    void solve_augsys(Eigen::VectorXd& delta_x, Eigen::VectorXd& delta_y, Eigen::VectorXd& delta_z,
                      ipm::IPMLinearSolver& ls, const Eigen::VectorXd& theta_vw,
                      const Eigen::VectorXi& ubi, const Eigen::VectorXd& xi_p,
                      const Eigen::VectorXd& xi_d, const Eigen::VectorXd& xi_u);

    /**
     * Compute one homogeneous self-dual Newton direction.
     *
     * The method solves the reduced augmented system, recovers `tau/kappa`, and
     * reconstructs the lower- and upper-bound complementarity directions.
     */
    void solve_newton_system(Eigen::VectorXd& Delta_x, Eigen::VectorXd& Delta_lambda,
                             Eigen::VectorXd& Delta_w, Eigen::VectorXd& Delta_s,
                             Eigen::VectorXd& Delta_v, double& Delta_tau, double& Delta_kappa,
                             ipm::IPMLinearSolver& ls, const Eigen::VectorXd& theta_vw,
                             const Eigen::VectorXd& b, const Eigen::VectorXd& c,
                             const Eigen::VectorXi& ubi, const Eigen::VectorXd& ubv,
                             const Eigen::VectorXd& delta_x, const Eigen::VectorXd& delta_y,
                             const Eigen::VectorXd& delta_w, double delta_0,
                             const Eigen::VectorXd& iter_x, const Eigen::VectorXd& iter_lambda,
                             const Eigen::VectorXd& iter_w, const Eigen::VectorXd& iter_s,
                             const Eigen::VectorXd& iter_v, double iter_tau, double iter_kappa,
                             const Eigen::VectorXd& xi_p, const Eigen::VectorXd& xi_u,
                             const Eigen::VectorXd& xi_d, double xi_g, const Eigen::VectorXd& xi_xs,
                             const Eigen::VectorXd& xi_vw, double xi_tau_kappa);

    /** Return the largest nonnegative step with `v + alpha * dv >= 0`. */
    double max_alpha_single(const Eigen::VectorXd& v, const Eigen::VectorXd& dv);

    /** Return the largest positivity-preserving step for all primal/dual blocks. */
    double max_alpha(const Eigen::VectorXd& x, const Eigen::VectorXd& dx, const Eigen::VectorXd& v,
                     const Eigen::VectorXd& dv, const Eigen::VectorXd& s, const Eigen::VectorXd& ds,
                     const Eigen::VectorXd& w, const Eigen::VectorXd& dw, double tau, double dtau,
                     double kappa, double dkappa);

    /** Solve an LP already expressed as `OptimizationData`. */
    void run_optimization(const OptimizationData& data, const double tol);

    /** Convenience wrapper used by Python bindings and smoke tests. */
    void solve(const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
               const Eigen::VectorXd& c, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
               const Eigen::VectorXd& sense, double tol);

#ifdef GUROBI
    // Method to extract optimization components from a Gurobi model
    OptimizationData extractOptimizationComponents(GRBModel& model);
#endif

    /**
     * Refresh the numeric values of the regularized augmented matrix.
     *
     * The sparsity pattern is fixed after `start_linear_solver`; only diagonal
     * values change across IPM iterations, so this writes directly to cached
     * diagonal slots before refactorization.
     */
    int update_linear_solver(ipm::IPMLinearSolver& ls, const Eigen::VectorXd& theta,
                             const Eigen::VectorXd& regP, const Eigen::VectorXd& regD);

    /**
     * Build and initialize the regularized augmented KKT matrix.
     *
     * The block form is `[-Theta-Rp, A^T; A, Rd]`. Its pattern is reused for all
     * later numeric factorizations in the IPM loop.
     */
    void start_linear_solver(ipm::IPMLinearSolver& ls, const Eigen::SparseMatrix<double>& A);

    ScalingFactors ruiz_scaling(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b,
                                Eigen::VectorXd& c, Eigen::VectorXd& lo, Eigen::VectorXd& hi);

    void unscale_solution(Eigen::VectorXd& x, Eigen::VectorXd& lambda,
                          const ScalingFactors& factors);
};
