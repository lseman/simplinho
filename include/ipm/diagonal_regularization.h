// diagonal_regularization.h — Adaptive diagonal regularization for IPM KKT systems
// SOTA: Scale regularization by problem norm (HiPO, ECOS style)
// Monitor ill-conditioning, auto-shift to preserve quasi-definiteness

#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>

namespace ipm {

// Regularization parameters + diagnostics
struct RegularizationControl {
    double reg_p = 1e-8;   // Primal regularization (for -Theta-Rp)
    double reg_d = 1e-8;   // Dual regularization (for Rd)
    double shift_p = 0.0;  // Primal shift (diagonal addition)
    double shift_d = 0.0;  // Dual shift
    int iter = 0;
    double mu = 1e-1;      // Complementarity measure
    double cond_est = 1.0; // Condition number estimate
};

// Compute regularization based on problem norm and complementarity
inline RegularizationControl compute_regularization(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& theta,
    const Eigen::VectorXd& reg_diag, double mu, int iter) {
    RegularizationControl rc;
    rc.mu = mu;
    rc.iter = iter;

    // Compute matrix norms
    double A_norm_sq = 0.0;
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(A, k); it;
             ++it) {
            A_norm_sq += it.value() * it.value();
        }
    }
    A_norm_sq = std::sqrt(A_norm_sq);

    int m = A.rows();
    int n = A.cols();

    // theta = diag(S * Z^-1) where S,Z are slack/dual variables
    double theta_mean = 0.0;
    double theta_max = 1e-16;
    for (int i = 0; i < (int)theta.size(); ++i) {
        if (theta[i] > 0) {
            theta_mean += theta[i];
            theta_max = std::max(theta_max, theta[i]);
        }
    }
    if (theta.size() > 0) {
        theta_mean /= theta.size();
    }

    // HiPO-style: scale regularization proportional to problem norm
    // and inversely to complementarity (tighter as we approach optimum)
    double scale_factor = std::max(1.0, A_norm_sq / (std::sqrt(m) + 1e-10));
    double mu_factor = std::clamp(1e-2 * mu, 1e-12, 1e-8);

    rc.reg_p = scale_factor * mu_factor;
    rc.reg_d = scale_factor * mu_factor;

    // Diagonal shift: if reg_diag has zeros or very small values,
    // add shift to ensure quasi-definiteness
    double min_diag = 1e16;
    for (int i = 0; i < (int)reg_diag.size(); ++i) {
        if (std::isfinite(reg_diag[i]) && reg_diag[i] > 0) {
            min_diag = std::min(min_diag, reg_diag[i]);
        }
    }

    if (min_diag < 1e-14) {
        rc.shift_p = 1e-10;
        rc.shift_d = 1e-10;
    } else if (min_diag < 1e-8) {
        rc.shift_p = 1e-12;
        rc.shift_d = 1e-12;
    }

    // Estimate conditioning: warn if Theta-Rp becomes ill-conditioned
    if (theta_max > 1e-10) {
        rc.cond_est = theta_max / (theta_mean + 1e-14);
    }

    return rc;
}

// Apply regularization to augmented KKT matrix blocks
inline void apply_regularization(Eigen::VectorXd& diag_theta_rp,
                                 Eigen::VectorXd& diag_rd,
                                 const RegularizationControl& rc) {
    // Primal block: -Theta - Rp
    for (int i = 0; i < (int)diag_theta_rp.size(); ++i) {
        diag_theta_rp[i] -= rc.reg_p + rc.shift_p;
    }

    // Dual block: Rd
    for (int i = 0; i < (int)diag_rd.size(); ++i) {
        diag_rd[i] += rc.reg_d + rc.shift_d;
    }
}

// Check if regularization is adequate (condition number acceptable)
inline bool regularization_adequate(const RegularizationControl& rc,
                                    double max_cond_allowed = 1e8) {
    return rc.cond_est <= max_cond_allowed;
}

}  // namespace ipm
