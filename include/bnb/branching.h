#pragma once

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "bnb/diving.h"

namespace simplex::bnb::detail {

class ParallelDispatcher;

struct PseudoCostStats {
    double up_sum = 0.0;
    double down_sum = 0.0;
    int up_count = 0;
    int down_count = 0;

    void record_up(double gain, double distance);
    void record_down(double gain, double distance);
    [[nodiscard]] double up_value() const;
    [[nodiscard]] double down_value() const;
    [[nodiscard]] bool is_reliable(int reliability) const;
};

struct BranchSignalStats {
    double inference_up = 0.0;
    double inference_down = 0.0;
    double conflict_score_up = 0.0;
    double conflict_score_down = 0.0;
    double cutoff_up = 0.0;
    double cutoff_down = 0.0;

    void record_inference(bool branch_up, double amount);
    void record_conflict(bool branch_up, double amount);
    void record_cutoff(bool branch_up);
    void record_cutoff();
    [[nodiscard]] double up_score() const;
    [[nodiscard]] double down_score() const;
};

struct PseudoCost {
    PseudoCostStats cost;
    BranchSignalStats signal;

    void record_observation(bool branch_up, double parent_objective, double child_objective,
                            double parent_value, double child_value, bool maximize);
    void record_inference(bool branch_up, double amount);
    void record_conflict(bool branch_up, double amount);
    void record_cutoff(bool branch_up);
    void record_cutoff();
};

struct ChildEvaluation {
    ChildState state;
    std::optional<RelaxationSolution> relaxation;
    double score = -std::numeric_limits<double>::infinity();
    bool cutoff = false;
};

struct BranchDecision {
    int variable = -1;
    double value = std::numeric_limits<double>::quiet_NaN();
    ChildEvaluation down_child;
    ChildEvaluation up_child;
    int strong_branching_probe_count = 0;
    int strong_branching_probe_iterations = 0;
    std::uint64_t strong_branching_probe_core_solve_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_assembly_time_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_presolve_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_crash_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_iters_ns = 0;
    std::uint64_t strong_branching_probe_lp_internal_serialize_ns = 0;
};

struct RankedPseudoCostCandidate {
    FractionalCandidate candidate;
    double score = -std::numeric_limits<double>::infinity();
};

struct EvaluatedBranchCandidate {
    FractionalCandidate candidate;
    ChildEvaluation down_child;
    ChildEvaluation up_child;
    double score = -std::numeric_limits<double>::infinity();
};

struct PseudoCostAverages {
    // HiGHS-style averages over all candidates: each channel is averaged
    // separately so the weighted score in get_combined_pseudocost_score can
    // normalize per channel (cost, inference, conflict, cutoff).
    double cost = 1.0;
    double inference = 1.0;
    double conflict = 1.0;
    double cutoff = 1.0;
};

[[nodiscard]] double node_estimate(const RelaxationSolution& relaxation,
                                   const std::vector<VariableType>& variable_types,
                                   const std::vector<PseudoCost>& pseudocosts,
                                   double integrality_tol, bool maximize);

BranchDecision choose_branching_variable(const ActiveNode& node,
                                         const RelaxationSolution& relaxation,
                                         const std::vector<FractionalCandidate>& fractional,
                                         const Options& options, bool maximize,
                                         std::vector<PseudoCost>& pseudocosts,
                                         ParallelDispatcher* parallel_dispatcher,
                                         const RelaxationSolveCallback& relaxation_solver);

BranchDecision choose_sos_branching_constraint(const ActiveNode& node,
                                               const Eigen::VectorXd& primal,
                                               const std::vector<SOSConstraint>& sos_constraints,
                                               double feasibility_tol);

} // namespace simplex::bnb::detail
