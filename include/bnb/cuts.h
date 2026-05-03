#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "bnb/conflict_graph.h"
#include "bnb/types.h"

namespace simplex::bnb::detail {

class ImplicationStore;

struct SeparatorContext {
    const Problem& problem;
    const RelaxationSolution& relaxation;
    const Options& options;
    const ImplicationStore* learned_implications = nullptr;
    const std::vector<Cut>* structural_cuts = nullptr;

    const LPSolution* lp_solution() const {
        return relaxation.lp_solution.has_value() ? &*relaxation.lp_solution : nullptr;
    }

    const LPBasis* basis() const {
        return relaxation.basis.has_value() ? &*relaxation.basis : nullptr;
    }

    const Eigen::VectorXd* reduced_costs_internal() const {
        const LPSolution* lp = lp_solution();
        return lp != nullptr ? &lp->reduced_costs_internal : nullptr;
    }

    const Eigen::VectorXd* dual_values() const {
        const LPSolution* lp = lp_solution();
        return lp != nullptr ? &lp->dual_values : nullptr;
    }

    bool has_internal_tableau() const {
        const LPSolution* lp = lp_solution();
        return lp != nullptr && lp->has_internal_tableau;
    }
};

enum class CutSeparatorPhase {
    ImpliedBound,
    Clique,
    OddCycle,
    LP,
    Proof,
};

class CutSeparator {
  public:
    virtual ~CutSeparator() = default;
    virtual std::string_view name() const = 0;
    virtual CutSeparatorPhase phase() const = 0;
    virtual bool enabled(const SeparatorContext& context) const = 0;
    virtual std::vector<Cut> separate(const SeparatorContext& context) const = 0;
};

struct CutSignature {
    std::uint64_t lo = 0x243f6a8885a308d3ULL;
    std::uint64_t hi = 0x13198a2e03707344ULL;

    bool operator==(const CutSignature&) const noexcept = default;
};

struct CutSignatureHash {
    std::size_t operator()(const CutSignature& signature) const noexcept;
};

double cut_violation(const Cut& cut, const Eigen::VectorXd& primal);

CutSignature cut_signature(const Cut& cut, int precision = 9);

bool canonicalize_cut(Cut* cut, double zero_tol = 1e-12);

Cut clique_cut_from_literals(const Problem& problem, const std::vector<int>& clique_literals,
                             const Options& options, const std::string& cut_type = "Clique");

double cut_parallelism(const Cut& lhs, const Cut& rhs);

class CutPool {
  public:
    explicit CutPool(const Options& options = {});

    bool add_cut(const Problem& problem, const Cut& cut);

    std::vector<Cut> select_violated_cuts(const Eigen::VectorXd& primal,
                                          const Eigen::VectorXd& lower_bounds,
                                          const Eigen::VectorXd& upper_bounds, int max_cuts,
                                          double density_penalty_scale = 1.0,
                                          const Eigen::VectorXd* objective = nullptr,
                                          bool maximize = false);

    void reset(const Options& options);

    void perform_aging();

    int cuts_generated() const { return cuts_generated_; }
    int cuts_applied() const { return cuts_applied_; }
    int duplicate_cuts() const { return duplicate_cuts_; }
    int size() const { return static_cast<int>(cuts_.size()); }
    const std::unordered_map<std::string, int>& generated_counts() const {
        return generated_counts_;
    }
    const std::unordered_map<std::string, int>& applied_counts() const { return applied_counts_; }

  public:
    // Accessor methods for cut pool contents
    const std::vector<Cut>& active_cuts() const { return cuts_; }
    const std::unordered_set<CutSignature, CutSignatureHash>& active_signatures() const {
        return signatures_;
    }

    // Thread-safe read access (for parallel workers)
    const std::vector<Cut>& active_cuts_read_only() const { return cuts_; }

  private:
    void manage_pool_size_();

    // Use shared_mutex for thread-safe read-heavy access
    mutable std::shared_mutex cuts_mutex_;

    int max_pool_size_ = 256;
    double min_violation_ = 1e-4;
    int max_age_ = 5;
    double cut_age_decay_ = 0.08;
    double cut_selection_age_bonus_ = 0.10;
    int max_cuts_per_type_ = 4;
    double max_parallelism_ = 0.98;
    double dynamism_weight_ = 0.15;
    std::vector<Cut> cuts_;
    std::vector<double> row_norms_;
    std::unordered_set<CutSignature, CutSignatureHash> signatures_;
    std::unordered_map<std::uint64_t, std::vector<int>> support_buckets_;
    std::unordered_map<std::string, int> generated_counts_;
    std::unordered_map<std::string, int> applied_counts_;
    std::unordered_map<std::string, int> type_usage_stats_;
    int cuts_generated_ = 0;
    int cuts_applied_ = 0;
    int duplicate_cuts_ = 0;
};

std::optional<int> parse_internal_label_index(const std::string& label);

double fractional_part(double value);

std::vector<Cut> generate_gomory_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                      const Options& options);

std::vector<Cut> generate_mir_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                   const Options& options);

std::vector<Cut> generate_cover_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                     const Options& options);

std::vector<Cut> generate_zero_half_cuts(const Problem& problem,
                                         const RelaxationSolution& relaxation,
                                         const Options& options);

std::vector<Cut> generate_implied_bound_cuts(const Problem& problem,
                                             const RelaxationSolution& relaxation,
                                             const Options& options);

std::vector<Cut> generate_clique_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                      const Options& options,
                                      const ImplicationStore* learned_implications = nullptr,
                                      const std::vector<Cut>* structural_cuts = nullptr);

std::vector<Cut> generate_odd_cycle_cuts(const Problem& problem,
                                         const RelaxationSolution& relaxation,
                                         const Options& options,
                                         const ImplicationStore* learned_implications = nullptr,
                                         const std::vector<Cut>* structural_cuts = nullptr);

std::vector<Cut> generate_dual_proof_cuts(const Problem& problem,
                                          const std::vector<Cut>& active_cuts,
                                          const RelaxationSolution& relaxation,
                                          const Eigen::VectorXd& node_lower_bounds,
                                          const Eigen::VectorXd& node_upper_bounds,
                                          const Options& options);

std::vector<Cut> generate_dual_proof_cuts(const Problem& problem,
                                          const RelaxationSolution& relaxation,
                                          const Options& options);

std::vector<Cut> generate_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                               const Options& options,
                               const ImplicationStore* learned_implications = nullptr,
                               const std::vector<Cut>* structural_cuts = nullptr);

std::vector<Cut> generate_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                               const Options& options, CutSeparatorPhase phase,
                               const ImplicationStore* learned_implications = nullptr,
                               const std::vector<Cut>* structural_cuts = nullptr);

} // namespace simplex::bnb::detail
