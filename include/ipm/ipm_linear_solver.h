// ipm_linear_solver.h — Hybrid linear solver for IPM augmented systems
// Switches between sparse (LDLT/LU) and dense frontal (BLAS3-optimized)
// based on problem structure.

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

#include "linear_system/schur_frontal_ldlt.h"

namespace ipm {

enum class LinearSolverType {
    kSparseLDLT,   // Default: sparse LDLᵀ (existing)
    kSparseLU,     // Fallback: sparse LU
    kFrontal       // SOTA: Schur-complement frontal (BLAS3)
};

// Abstract interface for IPM augmented system solve
class IPMLinearSolver {
  public:
    virtual ~IPMLinearSolver() = default;

    // Factorize: (dx, dy) = solve_augsys(xi_p, xi_d)
    // where augsys is [-Theta-Rp, A^T; A, Rd]
    virtual void factorize(const Eigen::SparseMatrix<double>& A) = 0;

    // Solve augmented system
    virtual Eigen::VectorXd solve(const Eigen::VectorXd& rhs) = 0;

    // Query factorization status
    virtual bool is_factorized() const = 0;
    virtual std::string info() const = 0;
};

// Sparse solver wrapper (existing behavior)
class SparseIPMSolver : public IPMLinearSolver {
  public:
    SparseIPMSolver(LinearSolverType type = LinearSolverType::kSparseLDLT)
        : solver_type_(type) {}

    void factorize(const Eigen::SparseMatrix<double>& A) override;
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override;
    bool is_factorized() const override { return factorized_; }
    std::string info() const override;

  private:
    LinearSolverType solver_type_;
    bool factorized_ = false;
};

// Frontal solver wrapper (SOTA)
class FrontalIPMSolver : public IPMLinearSolver {
  public:
    FrontalIPMSolver() = default;

    void factorize(const Eigen::SparseMatrix<double>& A) override;
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override;
    bool is_factorized() const override { return ldlt_.factorized; }
    std::string info() const override;

  private:
    schur_frontal::FrontalLDLT ldlt_;
};

// Hybrid solver: auto-switches between sparse and frontal
class HybridIPMSolver : public IPMLinearSolver {
  public:
    HybridIPMSolver()
        : solver_(std::make_unique<SparseIPMSolver>(
              LinearSolverType::kSparseLDLT)) {}

    void factorize(const Eigen::SparseMatrix<double>& A) override;
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) override;
    bool is_factorized() const override {
        return solver_ && solver_->is_factorized();
    }
    std::string info() const override { return solver_->info(); }

    // Query which solver is active
    LinearSolverType active_solver_type() const {
        return active_type_;
    }

  private:
    std::unique_ptr<IPMLinearSolver> solver_;
    LinearSolverType active_type_ = LinearSolverType::kSparseLDLT;

    // Heuristic: estimate if problem is structured enough for frontal
    bool should_use_frontal(const Eigen::SparseMatrix<double>& A) const;
};

}  // namespace ipm
