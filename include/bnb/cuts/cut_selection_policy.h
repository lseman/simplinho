#pragma once

#include <optional>
#include <string>
#include <vector>

namespace simplex::bnb {

// A self-contained batch of valid, violated cut candidates. Candidate order is
// stable only for the duration of score(). A policy returns one finite score
// per candidate or std::nullopt to defer to the built-in selector.
struct CutSelectionObservations {
    int node_id = -1;
    int depth = 0;
    int round = 0;
    int max_cuts = 0;
    bool is_root = false;

    std::vector<int> pool_indices;
    std::vector<double> violation;
    std::vector<double> efficacy;
    std::vector<double> active_efficacy;
    std::vector<double> density_adjusted_efficacy;
    std::vector<double> dynamism;
    std::vector<double> fractional_focus;
    std::vector<double> objective_parallelism;
    std::vector<double> strength;
    std::vector<int> age;
    std::vector<int> times_used;
    std::vector<int> nnz;
    std::vector<std::string> cut_type;
    std::vector<double> rhs;
    std::vector<int> sense;

    // Sparse cut coefficient rows in CSR form, aligned with the arrays above.
    std::vector<int> row_starts;
    std::vector<int> column_indices;
    std::vector<double> coefficients;
};

class CutSelectionPolicy {
  public:
    virtual ~CutSelectionPolicy() = default;

    [[nodiscard]] virtual std::optional<std::vector<double>>
    score(const CutSelectionObservations& observations) const = 0;

    [[nodiscard]] virtual bool is_enabled() const { return true; }
};

} // namespace simplex::bnb
