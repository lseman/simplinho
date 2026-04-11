#pragma once

#include "bnb/core.h"

namespace simplex::bnb {

// Thin orchestration wrapper around the header-only `Solver` class.
// Keeps a single, small surface area to start splitting responsibilities.
class Manager {
  public:
    explicit Manager(Problem problem, Options options = {}, std::vector<Cut> initial_cuts = {})
        : solver_(std::move(problem), std::move(options), std::move(initial_cuts)) {}

    template <typename RelaxationSolver> SolveResult solve(RelaxationSolver&& relaxation_solver) {
        return solver_.solve(std::forward<RelaxationSolver>(relaxation_solver));
    }

    // Expose a few useful snapshots from the underlying solver.
    std::vector<LearnedConflict> learned_conflicts_snapshot() const {
        return solver_.learned_conflicts_snapshot_();
    }
    IncumbentSnapshot incumbent_snapshot() const { return solver_.incumbent_snapshot_(); }

  private:
    Solver solver_;
};

} // namespace simplex::bnb
