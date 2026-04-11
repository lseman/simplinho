#pragma once

#include <Eigen/Dense>

#include <string>
#include <vector>

#include "bnb/cuts.h"

namespace simplex::bnb::presolve {

inline constexpr double kCoeffTol = 1e-12;

struct SimplifiedCutsResult {
    bool infeasible = false;
    std::vector<Cut> cuts;
};

struct NodeBoundPresolveResult {
    bool infeasible = false;
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    int tightened_bounds = 0;
};

struct RootProblemPresolveResult {
    Problem problem;
    bool infeasible = false;
    int tightened_bounds = 0;
    int removed_rows = 0;
    int removed_coeffs = 0;
    int aggregations = 0;
    int fixed_variables = 0;
    int relaxed_huge_lower_bounds = 0;
    int relaxed_huge_upper_bounds = 0;
    int strengthened_coeffs = 0;
    int detected_components = 0;
};

SimplifiedCutsResult simplify_cuts_for_bounds(const std::vector<Cut>& cuts,
                                              const Eigen::VectorXd& lower,
                                              const Eigen::VectorXd& upper, double tol = 1e-9);

std::string cut_set_signature(const std::vector<Cut>& cuts);

NodeBoundPresolveResult presolve_mip_node_bounds(const Problem& problem,
                                                 const Eigen::VectorXd& lower_in,
                                                 const Eigen::VectorXd& upper_in,
                                                 const std::vector<Cut>& extra_cuts = {},
                                                 double tol = 1e-9, int max_passes = 2);

RootProblemPresolveResult presolve_mip_root_problem(const Problem& input, double tol = 1e-9,
                                                    int max_passes = 4);

} // namespace simplex::bnb::presolve
