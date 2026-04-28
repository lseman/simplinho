#include "bnb/cuts.h"
#include "bnb/lock_debug.h"

#include "bnb/implications.h"
#include "bnb/parallel.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <thread>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace simplex::bnb::detail {

std::optional<int> parse_internal_label_index(const std::string& label);

namespace {

class FunctionCutSeparator final : public CutSeparator {
  public:
    using EnabledFn = std::function<bool(const SeparatorContext&)>;
    using SeparateFn = std::function<std::vector<Cut>(const SeparatorContext&)>;

    FunctionCutSeparator(std::string_view name, CutSeparatorPhase phase, EnabledFn enabled,
                         SeparateFn separate)
        : name_(name), phase_(phase), enabled_(std::move(enabled)), separate_(std::move(separate)) {
    }

    std::string_view name() const override { return name_; }
    CutSeparatorPhase phase() const override { return phase_; }
    bool enabled(const SeparatorContext& context) const override { return enabled_(context); }
    std::vector<Cut> separate(const SeparatorContext& context) const override {
        return separate_(context);
    }

  private:
    std::string name_;
    CutSeparatorPhase phase_;
    EnabledFn enabled_;
    SeparateFn separate_;
};

const std::vector<std::unique_ptr<CutSeparator>>& default_cut_separators_() {
    static const std::vector<std::unique_ptr<CutSeparator>> separators = [] {
        std::vector<std::unique_ptr<CutSeparator>> built;
        built.reserve(7);

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "ImpliedBound", CutSeparatorPhase::ImpliedBound,
            [](const SeparatorContext& context) { return context.options.use_implied_bound_cuts; },
            [](const SeparatorContext& context) {
                return generate_implied_bound_cuts(context.problem, context.relaxation,
                                                   context.options);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "Clique", CutSeparatorPhase::Clique,
            [](const SeparatorContext& context) { return context.options.use_clique_cuts; },
            [](const SeparatorContext& context) {
                return generate_clique_cuts(context.problem, context.relaxation, context.options,
                                            context.learned_implications, context.structural_cuts);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "OddCycle", CutSeparatorPhase::OddCycle,
            [](const SeparatorContext& context) { return context.options.use_odd_cycle_cuts; },
            [](const SeparatorContext& context) {
                return generate_odd_cycle_cuts(context.problem, context.relaxation, context.options,
                                               context.learned_implications,
                                               context.structural_cuts);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "GMI", CutSeparatorPhase::LP,
            [](const SeparatorContext& context) { return context.options.use_gomory_cuts; },
            [](const SeparatorContext& context) {
                return generate_gomory_cuts(context.problem, context.relaxation, context.options);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "MIR", CutSeparatorPhase::LP,
            [](const SeparatorContext& context) { return context.options.use_mir_cuts; },
            [](const SeparatorContext& context) {
                return generate_mir_cuts(context.problem, context.relaxation, context.options);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "Cover", CutSeparatorPhase::LP,
            [](const SeparatorContext& context) { return context.options.use_cover_cuts; },
            [](const SeparatorContext& context) {
                return generate_cover_cuts(context.problem, context.relaxation, context.options);
            }));

        built.push_back(std::make_unique<FunctionCutSeparator>(
            "DualProof", CutSeparatorPhase::Proof,
            [](const SeparatorContext& context) { return context.options.use_dual_proof_cuts; },
            [](const SeparatorContext& context) {
                return generate_dual_proof_cuts(context.problem, context.relaxation,
                                                context.options);
            }));

        return built;
    }();
    return separators;
}

std::uint64_t signature_mix_(std::uint64_t seed, std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    return seed;
}

void signature_combine_(CutSignature* signature, std::uint64_t value) noexcept {
    if (signature == nullptr)
        return;
    signature->lo = signature_mix_(signature->lo, value);
    signature->hi = signature_mix_(signature->hi, value ^ 0x9e3779b97f4a7c15ULL);
}

std::uint64_t signature_bits_(double value) noexcept {
    const double normalized = (value == 0.0) ? 0.0 : value;
    return std::bit_cast<std::uint64_t>(normalized);
}

double cut_norm_(const Cut& cut) {
    double norm_sq = 0.0;
    for (double value : cut.values)
        norm_sq += value * value;
    return norm_sq > 0.0 ? std::sqrt(norm_sq) : 0.0;
}

std::uint64_t support_signature_(const Cut& cut) noexcept {
    std::uint64_t seed = 0x84222325cbf29ce4ULL;
    for (int index : cut.indices)
        seed = signature_mix_(seed, static_cast<std::uint64_t>(index + 1));
    return seed;
}

bool same_support_(const Cut& lhs, const Cut& rhs) { return lhs.indices == rhs.indices; }

struct ActiveSupportStats {
    double norm = 0.0;
    int nnz = 0;
};

enum class MirSubstitution {
    Lower,
    Upper,
};

struct MirIntegerTerm {
    int variable = -1;
    double coeff = 0.0;
    double lp_value = 0.0;
    double upper = std::numeric_limits<double>::infinity();
    MirSubstitution substitution = MirSubstitution::Lower;
    double shift = 0.0;
};

struct MirContinuousTerm {
    int variable = -1;
    double coeff = 0.0;
    double lp_value = 0.0;
    MirSubstitution substitution = MirSubstitution::Lower;
    double shift = 0.0;
};

struct CanonicalMirRow {
    std::vector<MirIntegerTerm> integers;
    std::vector<MirContinuousTerm> continuous;
    double rhs = 0.0;
    bool valid = false;
};

struct MirRowData {
    std::vector<int> indices;
    std::vector<double> values;
    double rhs = 0.0;
};

Cut eliminate_base_row_slacks_from_cut_(const Problem& problem, Cut cut);
std::optional<int> parse_internal_label_index_impl_(const std::string& label);
std::vector<int> select_gmi_rows_(const Problem& problem, const RelaxationSolution& relaxation,
                                  const Options& options, const LPSolution& lp);
bool postprocess_gmi_cut_(const Problem& problem, const Options& options, Cut* cut);
double gmi_cut_quality_score_(const Cut& cut, const Eigen::VectorXd& primal, double violation);
CanonicalMirRow build_canonical_mir_row_from_leq_(const Problem& problem,
                                                  const RelaxationSolution& relaxation,
                                                  const std::vector<int>& indices,
                                                  const std::vector<double>& values, double rhs);
double mir_function_g_(double d, double f);
void map_transformed_mir_term_(Cut* cut, double coeff, int variable, MirSubstitution substitution,
                               double shift);
std::optional<Cut> build_mir_cut_from_canonical_row_(const Problem& problem,
                                                     const RelaxationSolution& relaxation,
                                                     const Options& options,
                                                     const CanonicalMirRow& row);
std::optional<MirRowData> aggregate_rows_for_mir_(const MirRowData& lhs, const MirRowData& rhs,
                                                  int pivot_index);
std::optional<int> choose_mir_aggregation_pivot_(const MirRowData& lhs, const MirRowData& rhs,
                                                 const Eigen::VectorXd& primal);
void add_implication_conflicts_(ConflictGraph* graph, const Problem& problem,
                                const ImplicationStore* learned_implications,
                                const Options& options);

ActiveSupportStats active_support_stats_(const Cut& cut, const Eigen::VectorXd& primal,
                                         const Eigen::VectorXd& lower_bounds,
                                         const Eigen::VectorXd& upper_bounds,
                                         double feasibility_tol) {
    ActiveSupportStats stats;
    double norm_sq = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        if (index < 0 || index >= primal.size() || index >= lower_bounds.size() ||
            index >= upper_bounds.size()) {
            continue;
        }

        const double coeff = cut.values[k];
        const double value = primal(index);
        const bool active = coeff > 0.0 ? value > lower_bounds(index) + feasibility_tol
                                        : value < upper_bounds(index) - feasibility_tol;
        if (!active)
            continue;

        norm_sq += coeff * coeff;
        ++stats.nnz;
    }

    stats.norm = norm_sq > 0.0 ? std::sqrt(norm_sq) : 0.0;
    return stats;
}

double cut_parallelism_with_norms_(const Cut& lhs, double lhs_norm, const Cut& rhs,
                                   double rhs_norm) {
    if (lhs_norm <= 1e-16 || rhs_norm <= 1e-16)
        return 0.0;

    double dot = 0.0;
    int i = 0;
    int j = 0;
    while (i < static_cast<int>(lhs.indices.size()) && j < static_cast<int>(rhs.indices.size())) {
        if (lhs.indices[i] == rhs.indices[j]) {
            dot += lhs.values[i] * rhs.values[j];
            ++i;
            ++j;
        } else if (lhs.indices[i] < rhs.indices[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return std::abs(dot) / (lhs_norm * rhs_norm);
}

double integrality_distance_(double value) {
    if (!std::isfinite(value))
        return 0.0;
    return std::abs(value - std::round(value));
}

double fractional_focus_(const Cut& cut, const Eigen::VectorXd& primal) {
    double weighted_sum = 0.0;
    double total_weight = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        if (index < 0 || index >= primal.size())
            continue;
        const double weight = std::abs(cut.values[k]);
        const double focus = std::min(1.0, 2.0 * integrality_distance_(primal(index)));
        weighted_sum += weight * focus;
        total_weight += weight;
    }
    return total_weight > 0.0 ? weighted_sum / total_weight : 0.0;
}

double density_adjusted_efficacy_(double violation, double norm, int nnz, double fractional_focus,
                                  double density_penalty_scale) {
    if (norm <= 1e-16)
        return 0.0;
    const double efficacy = violation / norm;
    const double density_penalty =
        density_penalty_scale * std::sqrt(static_cast<double>(std::max(1, nnz)));
    return efficacy * (0.7 + 0.3 * fractional_focus) / density_penalty;
}

double cut_dynamism_(const Cut& cut, const Eigen::VectorXd& primal, double violation, double norm) {
    if (norm <= 1e-16)
        return 0.0;

    double activity = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        if (index >= 0 && index < primal.size()) {
            activity += std::abs(cut.values[k] * primal(index));
        }
    }
    const double normalized_violation = std::min(1.0, violation / norm);
    const double focus = fractional_focus_(cut, primal);
    return normalized_violation * (0.5 + 0.5 * focus) / (1.0 + 0.1 * activity);
}

double cut_retention_score_(const Cut& cut, double norm, int type_usage, double age_decay_rate) {
    const double age_decay = std::exp(-age_decay_rate * static_cast<double>(cut.age));
    const double usage_reward = std::log1p(1.0 + static_cast<double>(cut.times_used));
    const double density_reward =
        1.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(1, cut.indices.size())));
    return age_decay * (0.70 * cut.strength + 0.15 * usage_reward + 0.10 * density_reward +
                        0.05 * static_cast<double>(type_usage)) +
           0.05 * norm;
}

struct CoverLiteralTerm {
    int literal = -1;
    double coeff = 0.0;
    double activity = 0.0;
};

struct MixedCoverContext {
    std::vector<CoverLiteralTerm> literals;
    double rhs = 0.0;
    bool valid = true;
    bool has_nonbinary_component = false;
};

struct CanonicalKnapsack {
    // ax <= b, all a_j > 0, x_j in {0,1}, with possible complementation
    std::vector<int> variables;     // original variable indices
    std::vector<double> coeffs;     // positive coefficients
    std::vector<bool> complemented; // true if x_j was complemented
    std::vector<double> lp_values;  // value in canonical space (0/1)
    double rhs = 0.0;
    bool valid = true;
    CanonicalKnapsack() = default;
};

struct BinaryCoverPartition {
    std::vector<int> cover_positions;
    std::vector<int> remainder_positions;
};

// Build canonical knapsack ax <= b from a row, only if all variables are binary
CanonicalKnapsack build_canonical_binary_knapsack_(const Problem& problem,
                                                   const RelaxationSolution& relaxation,
                                                   const SparseLinearConstraint& row) {
    CanonicalKnapsack kn;
    kn.rhs = row.rhs;
    for (int k = 0; k < (int)row.indices.size() && k < (int)row.values.size(); ++k) {
        int idx = row.indices[k];
        double coeff = row.values[k];
        if (idx < 0 || idx >= (int)problem.variable_types.size() ||
            idx >= (int)relaxation.primal.size() || idx >= (int)problem.upper_bounds.size() ||
            problem.variable_types[idx] != VariableType::Binary || std::abs(coeff) <= 1e-12) {
            kn.valid = false;
            break;
        }
        if (coeff > 0.0) {
            kn.variables.push_back(idx);
            kn.coeffs.push_back(coeff);
            kn.complemented.push_back(false);
            kn.lp_values.push_back(relaxation.primal(idx));
        } else if (std::isfinite(problem.upper_bounds(idx))) {
            kn.variables.push_back(idx);
            kn.coeffs.push_back(-coeff);
            kn.complemented.push_back(true);
            kn.lp_values.push_back(1.0 - relaxation.primal(idx));
            kn.rhs -= coeff * problem.upper_bounds(idx);
        } else {
            kn.valid = false;
            break;
        }
    }
    return kn;
}

std::optional<BinaryCoverPartition> find_lp_violated_minimal_cover_(const CanonicalKnapsack& kn,
                                                                    double tol = 1e-9) {
    const int n = static_cast<int>(kn.variables.size());
    if (n < 2 || !kn.valid || !std::isfinite(kn.rhs) || kn.rhs < 0.0) {
        return std::nullopt;
    }

    const double element_sum = std::accumulate(kn.coeffs.begin(), kn.coeffs.end(), 0.0);
    if (element_sum <= kn.rhs + tol) {
        return std::nullopt;
    }

    std::vector<int> ratio_order(n);
    std::iota(ratio_order.begin(), ratio_order.end(), 0);
    std::sort(ratio_order.begin(), ratio_order.end(), [&](int lhs, int rhs) {
        const double lhs_ratio = (1.0 - kn.lp_values[lhs]) / kn.coeffs[lhs];
        const double rhs_ratio = (1.0 - kn.lp_values[rhs]) / kn.coeffs[rhs];
        if (std::abs(lhs_ratio - rhs_ratio) > tol) {
            return lhs_ratio > rhs_ratio;
        }
        if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
            return kn.coeffs[lhs] > kn.coeffs[rhs];
        }
        return lhs < rhs;
    });

    const double transformed_rhs = element_sum - kn.rhs;
    int critical = 0;
    double prefix_sum = kn.coeffs[ratio_order.front()];
    while (critical + 1 < n && prefix_sum <= transformed_rhs - tol) {
        ++critical;
        prefix_sum += kn.coeffs[ratio_order[critical]];
    }

    double lp_cover_objective = 0.0;
    for (int i = critical; i < n; ++i) {
        lp_cover_objective += 1.0 - kn.lp_values[ratio_order[i]];
    }
    if (lp_cover_objective > 1.0 - tol) {
        return std::nullopt;
    }

    BinaryCoverPartition partition;
    partition.cover_positions.assign(ratio_order.begin() + critical, ratio_order.end());
    std::sort(partition.cover_positions.begin(), partition.cover_positions.end(),
              [&](int lhs, int rhs) {
                  if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
                      return kn.coeffs[lhs] > kn.coeffs[rhs];
                  }
                  return lhs < rhs;
              });

    double cover_sum = 0.0;
    for (int pos : partition.cover_positions) {
        cover_sum += kn.coeffs[pos];
    }
    if (cover_sum <= kn.rhs + tol) {
        return std::nullopt;
    }

    while (partition.cover_positions.size() > 1) {
        const int smallest = partition.cover_positions.back();
        if (cover_sum - kn.coeffs[smallest] <= kn.rhs + tol) {
            break;
        }
        cover_sum -= kn.coeffs[smallest];
        partition.cover_positions.pop_back();
    }
    if (partition.cover_positions.size() < 2 || cover_sum <= kn.rhs + tol) {
        return std::nullopt;
    }

    std::vector<char> in_cover(n, 0);
    for (int pos : partition.cover_positions) {
        in_cover[pos] = 1;
    }
    partition.remainder_positions.reserve(n - static_cast<int>(partition.cover_positions.size()));
    for (int pos = 0; pos < n; ++pos) {
        if (!in_cover[pos]) {
            partition.remainder_positions.push_back(pos);
        }
    }

    return partition;
}

// Sequence-independent lifting for pure binary knapsack cover (Gu-Nemhauser-Savelsbergh)
// Returns (indices, values, rhs) for the lifted cover cut
std::optional<std::tuple<std::vector<int>, std::vector<double>, double>>
lifted_binary_cover_cut_(const CanonicalKnapsack& kn, double tol = 1e-9) {
    const int n = static_cast<int>(kn.variables.size());
    if (n < 2 || !kn.valid || !std::isfinite(kn.rhs) || kn.rhs < 0.0)
        return std::nullopt;

    const std::optional<BinaryCoverPartition> partition = find_lp_violated_minimal_cover_(kn, tol);
    if (!partition.has_value()) {
        return std::nullopt;
    }

    const std::vector<int>& cover = partition->cover_positions;
    const std::vector<int>& remainder = partition->remainder_positions;
    const int cover_size = static_cast<int>(cover.size());

    double cover_sum = 0.0;
    for (int pos : cover) {
        cover_sum += kn.coeffs[pos];
    }
    const double lambda = cover_sum - kn.rhs;
    if (lambda <= tol) {
        return std::nullopt;
    }

    std::vector<double> mu(cover_size + 1, 0.0);
    std::vector<double> mu_minus_lambda(cover_size + 1, 0.0);
    mu_minus_lambda[0] = -lambda;
    for (int i = 1; i <= cover_size; ++i) {
        mu[i] = mu[i - 1] + kn.coeffs[cover[i - 1]];
        mu_minus_lambda[i] = mu[i] - lambda;
    }

    std::vector<double> canonical_coeffs(n, 0.0);
    for (int pos : cover) {
        canonical_coeffs[pos] = 1.0;
    }

    const bool use_superadditive_f =
        cover_size == 1 || mu_minus_lambda[1] >= kn.coeffs[cover[1]] - tol;
    if (use_superadditive_f) {
        for (int pos : remainder) {
            const double coeff = kn.coeffs[pos];
            if (coeff <= mu_minus_lambda[1] + tol) {
                continue;
            }

            bool found = false;
            for (int i = 2; i <= cover_size; ++i) {
                if (coeff <= mu_minus_lambda[i] + tol) {
                    canonical_coeffs[pos] = static_cast<double>(i - 1);
                    found = true;
                    break;
                }
            }
            if (!found) {
                return std::nullopt;
            }
        }
    } else {
        std::vector<double> rho(cover_size + 1, 0.0);
        rho[0] = lambda;
        rho[cover_size] = 0.0;
        for (int i = 1; i < cover_size; ++i) {
            rho[i] = std::max(0.0, kn.coeffs[cover[i]] - mu_minus_lambda[1]);
        }
        if (rho[1] <= tol) {
            return std::nullopt;
        }

        for (int pos : remainder) {
            const double coeff = kn.coeffs[pos];
            for (int i = 0; i < cover_size; ++i) {
                if (coeff <= mu_minus_lambda[i + 1] + tol) {
                    if (i > 0) {
                        canonical_coeffs[pos] = static_cast<double>(i);
                    }
                    break;
                }
                if (rho[i + 1] > tol && coeff < mu_minus_lambda[i + 1] + rho[i + 1]) {
                    const double lifted =
                        i + 1.0 - (mu_minus_lambda[i + 1] + rho[i + 1] - coeff) / rho[1];
                    if (std::abs(lifted) > tol) {
                        canonical_coeffs[pos] = lifted;
                    }
                    break;
                }
            }
        }
    }

    std::vector<int> indices;
    std::vector<double> values;
    indices.reserve(n);
    values.reserve(n);
    double cut_rhs = static_cast<double>(cover_size - 1);
    for (int pos = 0; pos < n; ++pos) {
        double coeff = canonical_coeffs[pos];
        if (std::abs(coeff) <= tol) {
            continue;
        }
        if (kn.complemented[pos]) {
            cut_rhs -= coeff;
            coeff = -coeff;
        }
        indices.push_back(kn.variables[pos]);
        values.push_back(coeff);
    }

    return std::make_tuple(indices, values, cut_rhs);
}

std::vector<int> minimize_binary_cover_positions_(std::vector<int> positions,
                                                  const CanonicalKnapsack& kn, double tol = 1e-9) {
    if (positions.size() < 2) {
        return {};
    }

    double total = 0.0;
    for (const int pos : positions) {
        if (pos < 0 || pos >= static_cast<int>(kn.coeffs.size())) {
            return {};
        }
        total += kn.coeffs[pos];
    }
    if (total <= kn.rhs + tol) {
        return {};
    }

    bool changed = true;
    while (changed && positions.size() > 1) {
        changed = false;
        std::sort(positions.begin(), positions.end(), [&](int lhs, int rhs) {
            if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
                return kn.coeffs[lhs] < kn.coeffs[rhs];
            }
            if (std::abs(kn.lp_values[lhs] - kn.lp_values[rhs]) > tol) {
                return kn.lp_values[lhs] < kn.lp_values[rhs];
            }
            return lhs < rhs;
        });
        for (auto it = positions.begin(); it != positions.end(); ++it) {
            const double coeff = kn.coeffs[*it];
            if (total - coeff > kn.rhs + tol) {
                total -= coeff;
                positions.erase(it);
                changed = true;
                break;
            }
        }
    }

    if (positions.size() < 2 || total <= kn.rhs + tol) {
        return {};
    }

    std::sort(positions.begin(), positions.end());
    positions.erase(std::unique(positions.begin(), positions.end()), positions.end());
    return positions;
}

enum class BinaryCoverOrdering { ActivityFirst, CoefficientFirst, RatioFirst };

std::vector<int> greedy_binary_cover_positions_(const CanonicalKnapsack& kn,
                                                BinaryCoverOrdering ordering, double tol = 1e-9) {
    const int n = static_cast<int>(kn.variables.size());
    if (!kn.valid || n < 2 || !std::isfinite(kn.rhs) || kn.rhs < 0.0) {
        return {};
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        const double lhs_ratio =
            kn.coeffs[lhs] > tol ? kn.lp_values[lhs] / kn.coeffs[lhs] : kn.lp_values[lhs];
        const double rhs_ratio =
            kn.coeffs[rhs] > tol ? kn.lp_values[rhs] / kn.coeffs[rhs] : kn.lp_values[rhs];
        switch (ordering) {
            case BinaryCoverOrdering::ActivityFirst:
                if (std::abs(kn.lp_values[lhs] - kn.lp_values[rhs]) > tol) {
                    return kn.lp_values[lhs] > kn.lp_values[rhs];
                }
                if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
                    return kn.coeffs[lhs] > kn.coeffs[rhs];
                }
                break;
            case BinaryCoverOrdering::CoefficientFirst:
                if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
                    return kn.coeffs[lhs] > kn.coeffs[rhs];
                }
                if (std::abs(kn.lp_values[lhs] - kn.lp_values[rhs]) > tol) {
                    return kn.lp_values[lhs] > kn.lp_values[rhs];
                }
                break;
            case BinaryCoverOrdering::RatioFirst:
                if (std::abs(lhs_ratio - rhs_ratio) > tol) {
                    return lhs_ratio > rhs_ratio;
                }
                if (std::abs(kn.lp_values[lhs] - kn.lp_values[rhs]) > tol) {
                    return kn.lp_values[lhs] > kn.lp_values[rhs];
                }
                if (std::abs(kn.coeffs[lhs] - kn.coeffs[rhs]) > tol) {
                    return kn.coeffs[lhs] > kn.coeffs[rhs];
                }
                break;
        }
        return lhs < rhs;
    });

    std::vector<int> cover;
    double total = 0.0;
    for (const int pos : order) {
        cover.push_back(pos);
        total += kn.coeffs[pos];
        if (total > kn.rhs + tol) {
            break;
        }
    }
    if (total <= kn.rhs + tol) {
        return {};
    }
    return minimize_binary_cover_positions_(std::move(cover), kn, tol);
}

std::vector<CoverLiteralTerm> canonical_binary_cover_terms_(const CanonicalKnapsack& kn) {
    std::vector<CoverLiteralTerm> terms;
    terms.reserve(kn.variables.size());
    for (int pos = 0; pos < static_cast<int>(kn.variables.size()); ++pos) {
        const int variable = kn.variables[pos];
        const bool complemented = kn.complemented[pos];
        terms.push_back(CoverLiteralTerm{ConflictGraph::literal_for(variable, !complemented),
                                         kn.coeffs[pos], kn.lp_values[pos]});
    }
    return terms;
}

std::vector<int> canonical_binary_cover_literals_(const CanonicalKnapsack& kn,
                                                  const std::vector<int>& positions) {
    std::vector<int> literals;
    literals.reserve(positions.size());
    for (const int pos : positions) {
        if (pos < 0 || pos >= static_cast<int>(kn.variables.size())) {
            continue;
        }
        literals.push_back(ConflictGraph::literal_for(kn.variables[pos], !kn.complemented[pos]));
    }
    std::sort(literals.begin(), literals.end());
    literals.erase(std::unique(literals.begin(), literals.end()), literals.end());
    return literals;
}

std::optional<CoverLiteralTerm> binary_cover_literal_(const Problem& problem,
                                                      const RelaxationSolution& relaxation,
                                                      int index, double coeff) {
    if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
        index >= static_cast<int>(relaxation.primal.size()) ||
        index >= static_cast<int>(problem.upper_bounds.size()) ||
        problem.variable_types[index] != VariableType::Binary || std::abs(coeff) <= 1e-12) {
        return std::nullopt;
    }

    if (coeff > 0.0) {
        return CoverLiteralTerm{ConflictGraph::literal_for(index, true), coeff,
                                relaxation.primal(index)};
    }
    if (std::isfinite(problem.upper_bounds(index))) {
        return CoverLiteralTerm{ConflictGraph::literal_for(index, false), -coeff,
                                1.0 - relaxation.primal(index)};
    }
    return std::nullopt;
}

MixedCoverContext build_mixed_cover_context_(const Problem& problem,
                                             const RelaxationSolution& relaxation,
                                             const SparseLinearConstraint& row) {
    MixedCoverContext context;
    context.rhs = row.rhs;

    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        const int index = row.indices[k];
        const double coeff = row.values[k];
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
            index >= static_cast<int>(problem.lower_bounds.size()) ||
            index >= static_cast<int>(problem.upper_bounds.size()) || std::abs(coeff) <= 1e-12) {
            continue;
        }

        if (problem.variable_types[index] == VariableType::Binary) {
            const std::optional<CoverLiteralTerm> literal =
                binary_cover_literal_(problem, relaxation, index, coeff);
            if (literal.has_value()) {
                context.literals.push_back(*literal);
            } else {
                context.valid = false;
            }
            continue;
        }

        const double anchor =
            coeff >= 0.0 ? problem.lower_bounds(index) : problem.upper_bounds(index);
        if (!std::isfinite(anchor)) {
            context.valid = false;
            continue;
        }
        context.rhs -= coeff * anchor;
        context.has_nonbinary_component = true;
    }

    return context;
}

std::vector<int> greedy_cover_literals_(std::vector<CoverLiteralTerm> terms, double rhs,
                                        bool prefer_activity, bool prefer_ratio = false) {
    if (terms.size() < 2 || !std::isfinite(rhs))
        return {};

    std::sort(terms.begin(), terms.end(),
              [&](const CoverLiteralTerm& lhs, const CoverLiteralTerm& rhs_term) {
                  if (prefer_ratio) {
                      const double lhs_ratio = lhs.activity / lhs.coeff;
                      const double rhs_ratio = rhs_term.activity / rhs_term.coeff;
                      if (std::abs(lhs_ratio - rhs_ratio) > 1e-12)
                          return lhs_ratio > rhs_ratio;
                      if (std::abs(lhs.coeff - rhs_term.coeff) > 1e-12)
                          return lhs.coeff > rhs_term.coeff;
                      return lhs.literal < rhs_term.literal;
                  }
                  if (prefer_activity && std::abs(lhs.activity - rhs_term.activity) > 1e-12)
                      return lhs.activity > rhs_term.activity;
                  if (std::abs(lhs.coeff - rhs_term.coeff) > 1e-12)
                      return lhs.coeff > rhs_term.coeff;
                  if (!prefer_activity && std::abs(lhs.activity - rhs_term.activity) > 1e-12)
                      return lhs.activity > rhs_term.activity;
                  return lhs.literal < rhs_term.literal;
              });

    double total = 0.0;
    std::vector<CoverLiteralTerm> cover_terms;
    for (const CoverLiteralTerm& term : terms) {
        cover_terms.push_back(term);
        total += term.coeff;
        if (total > rhs + 1e-9)
            break;
    }
    if (total <= rhs + 1e-9 || cover_terms.size() < 2)
        return {};

    bool changed = true;
    while (changed && cover_terms.size() > 1) {
        changed = false;
        std::sort(cover_terms.begin(), cover_terms.end(),
                  [](const CoverLiteralTerm& lhs, const CoverLiteralTerm& rhs_term) {
                      if (std::abs(lhs.coeff - rhs_term.coeff) > 1e-12)
                          return lhs.coeff < rhs_term.coeff;
                      return lhs.literal < rhs_term.literal;
                  });
        for (auto it = cover_terms.begin(); it != cover_terms.end(); ++it) {
            if (total - it->coeff > rhs + 1e-9) {
                total -= it->coeff;
                cover_terms.erase(it);
                changed = true;
                break;
            }
        }
    }

    std::vector<int> literals;
    literals.reserve(cover_terms.size());
    for (const CoverLiteralTerm& term : cover_terms)
        literals.push_back(term.literal);
    std::sort(literals.begin(), literals.end());
    return literals;
}

std::vector<int> extend_cover_literals_(const std::vector<CoverLiteralTerm>& all_terms,
                                        const std::vector<int>& base_cover, double rhs) {
    if (base_cover.size() < 2)
        return base_cover;

    std::unordered_set<int> in_cover(base_cover.begin(), base_cover.end());
    double cover_sum = 0.0;
    double cover_min_coeff = std::numeric_limits<double>::infinity();
    for (const CoverLiteralTerm& term : all_terms) {
        if (!in_cover.contains(term.literal))
            continue;
        cover_sum += term.coeff;
        cover_min_coeff = std::min(cover_min_coeff, term.coeff);
    }

    const double excess = cover_sum - rhs;
    if (!(excess > 1e-9) || cover_min_coeff <= 0.0)
        return base_cover;

    const double min_threshold = std::max(0.0, rhs - (cover_sum - cover_min_coeff));
    std::vector<int> lifted = base_cover;
    for (const CoverLiteralTerm& term : all_terms) {
        if (in_cover.contains(term.literal))
            continue;
        if (term.coeff > min_threshold + 1e-9)
            lifted.push_back(term.literal);
    }
    std::sort(lifted.begin(), lifted.end());
    lifted.erase(std::unique(lifted.begin(), lifted.end()), lifted.end());
    return lifted;
}

bool postprocess_cover_cut_(const Problem& problem, const Options& options, Cut* cut);

void maybe_add_cover_cut_(const Problem& problem, const RelaxationSolution& relaxation,
                          const Options& options, const std::vector<int>& literals,
                          const std::string& cut_type,
                          std::unordered_set<CutSignature, CutSignatureHash>* signatures,
                          std::vector<Cut>* cuts) {
    if (cuts == nullptr || literals.size() < 2)
        return;

    Cut cut;
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = static_cast<double>(literals.size() - 1);
    cut.cut_type = cut_type;
    for (int literal : literals) {
        const int variable = ConflictGraph::variable_of(literal);
        if (variable < 0 || variable >= static_cast<int>(problem.variable_types.size()))
            continue;
        cut.indices.push_back(variable);
        if (ConflictGraph::value_of(literal)) {
            cut.values.push_back(1.0);
        } else {
            cut.values.push_back(-1.0);
            cut.rhs -= 1.0;
        }
    }
    if (!postprocess_cover_cut_(problem, options, &cut))
        return;
    if (cut.indices.empty())
        return;

    const double violation = cut_violation(cut, relaxation.primal);
    if (violation <= options.min_cut_violation)
        return;

    cut.strength = violation;
    const CutSignature signature = cut_signature(cut);
    if (signatures != nullptr && !signatures->insert(signature).second)
        return;
    cuts->push_back(std::move(cut));
}

std::vector<int> make_minimal_cover_(std::vector<int> indices, const SparseLinearConstraint& row) {
    auto coeff_for = [&](int variable) {
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            if (row.indices[k] == variable)
                return row.values[k];
        }
        return 0.0;
    };

    double total = 0.0;
    for (int index : indices)
        total += coeff_for(index);
    if (total <= row.rhs + 1e-9)
        return {};

    bool changed = true;
    while (changed && indices.size() > 1) {
        changed = false;
        std::sort(indices.begin(), indices.end(),
                  [&](int lhs, int rhs) { return coeff_for(lhs) < coeff_for(rhs); });
        for (auto it = indices.begin(); it != indices.end(); ++it) {
            const double coeff = coeff_for(*it);
            if (total - coeff > row.rhs + 1e-9) {
                total -= coeff;
                indices.erase(it);
                changed = true;
                break;
            }
        }
    }
    return indices;
}

double clique_literal_activity_(const std::vector<int>& clique_literals,
                                const Eigen::VectorXd& primal) {
    double activity = 0.0;
    for (int literal : clique_literals)
        activity += ConflictGraph::literal_weight(primal, literal);
    return activity;
}

std::vector<int> intersect_sorted_(std::vector<int> lhs, const std::vector<int>& rhs) {
    std::vector<int> intersection;
    intersection.reserve(std::min(lhs.size(), rhs.size()));
    std::set_intersection(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                          std::back_inserter(intersection));
    return intersection;
}

ConflictGraph build_separator_conflict_graph_(const Problem& problem,
                                              const ImplicationStore* learned_implications,
                                              const Options& options,
                                              const std::vector<Cut>* structural_cuts) {
    ConflictGraph graph(problem);
    if (structural_cuts != nullptr) {
        for (const Cut& cut : *structural_cuts)
            graph.add_cut_cliques(cut);
    }
    add_implication_conflicts_(&graph, problem, learned_implications, options);
    return graph;
}

Cut odd_cycle_cut_from_literals_(const Problem& problem, const std::vector<int>& cycle_literals,
                                 const Options& options, const std::string& cut_type = "OddCycle") {
    Cut cut;
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = static_cast<double>(cycle_literals.size() / 2);
    cut.cut_type = cut_type;
    for (int literal : cycle_literals) {
        const int variable = ConflictGraph::variable_of(literal);
        if (variable < 0 || variable >= static_cast<int>(problem.variable_types.size()))
            continue;
        cut.indices.push_back(variable);
        if (ConflictGraph::value_of(literal)) {
            cut.values.push_back(1.0);
        } else {
            cut.values.push_back(-1.0);
            cut.rhs -= 1.0;
        }
    }
    if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3))
        cut.indices.clear();
    return cut;
}

bool cycle_has_unique_variables_(const std::vector<int>& cycle_literals) {
    std::unordered_set<int> variables;
    variables.reserve(cycle_literals.size());
    for (const int literal : cycle_literals) {
        const int variable = ConflictGraph::variable_of(literal);
        if (!variables.insert(variable).second)
            return false;
    }
    return true;
}

std::vector<int> reconstruct_odd_cycle_positions_(int lhs, int rhs, const std::vector<int>& parent,
                                                  const std::vector<int>& depth) {
    if (lhs < 0 || rhs < 0 || lhs >= static_cast<int>(parent.size()) ||
        rhs >= static_cast<int>(parent.size())) {
        return {};
    }

    std::vector<int> lhs_path;
    std::unordered_map<int, int> lhs_index;
    for (int node = lhs; node >= 0; node = parent[node]) {
        lhs_index.emplace(node, static_cast<int>(lhs_path.size()));
        lhs_path.push_back(node);
    }

    std::vector<int> rhs_path;
    int lca = -1;
    for (int node = rhs; node >= 0; node = parent[node]) {
        auto it = lhs_index.find(node);
        rhs_path.push_back(node);
        if (it != lhs_index.end()) {
            lca = node;
            break;
        }
    }
    if (lca < 0)
        return {};

    const int lhs_lca_pos = lhs_index[lca];
    const int rhs_lca_pos = static_cast<int>(rhs_path.size()) - 1;
    std::vector<int> cycle;
    cycle.reserve(lhs_lca_pos + rhs_lca_pos + 1);
    for (int i = 0; i <= lhs_lca_pos; ++i)
        cycle.push_back(lhs_path[i]);
    for (int i = rhs_lca_pos - 1; i >= 0; --i)
        cycle.push_back(rhs_path[i]);

    if (cycle.size() < 3 || cycle.size() % 2 == 0)
        return {};
    return cycle;
}

std::vector<int> find_weighted_odd_cycle_(const ConflictGraph& graph, const Eigen::VectorXd& primal,
                                          const Options& options, const std::vector<int>& vertices,
                                          int start_position,
                                          const std::unordered_map<int, int>& position_of) {
    const int n = static_cast<int>(vertices.size());
    std::vector<int> color(n, -1);
    std::vector<int> parent(n, -1);
    std::vector<int> depth(n, 0);
    std::vector<int> queue;
    queue.reserve(n);
    queue.push_back(start_position);
    color[start_position] = 0;

    for (int head = 0; head < static_cast<int>(queue.size()); ++head) {
        const int u = queue[head];
        for (const int neighbor_literal : graph.neighbors(vertices[u])) {
            const auto pos_it = position_of.find(neighbor_literal);
            if (pos_it == position_of.end())
                continue;
            const int v = pos_it->second;
            if (v == u)
                continue;
            if (ConflictGraph::literal_weight(primal, vertices[v]) <= options.integrality_tol)
                continue;

            if (color[v] < 0) {
                color[v] = color[u] ^ 1;
                parent[v] = u;
                depth[v] = depth[u] + 1;
                queue.push_back(v);
                continue;
            }

            if (color[v] != color[u] || parent[u] == v || parent[v] == u)
                continue;

            std::vector<int> cycle_positions =
                reconstruct_odd_cycle_positions_(u, v, parent, depth);
            if (cycle_positions.size() < 5)
                continue;

            std::vector<int> cycle_literals;
            cycle_literals.reserve(cycle_positions.size());
            for (const int pos : cycle_positions)
                cycle_literals.push_back(vertices[pos]);
            if (!cycle_has_unique_variables_(cycle_literals))
                continue;
            return cycle_literals;
        }
    }

    return {};
}

std::optional<int> binary_literal_from_reason_(const Problem& problem, const ReasonLiteral& literal,
                                               double integrality_tol) {
    if (literal.variable < 0 || literal.variable >= static_cast<int>(problem.variable_types.size()))
        return std::nullopt;
    if (problem.variable_types[literal.variable] != VariableType::Binary ||
        !std::isfinite(literal.value)) {
        return std::nullopt;
    }
    if (literal.is_lower && literal.value >= 1.0 - integrality_tol)
        return ConflictGraph::literal_for(literal.variable, true);
    if (!literal.is_lower && literal.value <= integrality_tol)
        return ConflictGraph::literal_for(literal.variable, false);
    return std::nullopt;
}

static std::vector<int> collect_implied_literals_(const Problem& problem,
                                                  const ImplicationStore* learned_implications,
                                                  const Options& options, int trigger_literal,
                                                  int max_literal) {
    std::vector<int> implied_literals;
    if (learned_implications == nullptr || trigger_literal < 0 || trigger_literal >= max_literal) {
        return implied_literals;
    }

    std::vector<char> visited(static_cast<std::size_t>(max_literal), 0);
    std::vector<int> stack;
    visited[trigger_literal] = 1;
    stack.push_back(trigger_literal);

    while (!stack.empty()) {
        const int literal = stack.back();
        stack.pop_back();
        for (const ReasonLiteral& consequence : learned_implications->consequences(literal)) {
            const std::optional<int> consequence_literal =
                binary_literal_from_reason_(problem, consequence, options.integrality_tol);
            if (!consequence_literal.has_value())
                continue;
            const int implied_literal = *consequence_literal;
            if (implied_literal < 0 || implied_literal >= max_literal)
                continue;
            if (visited[implied_literal])
                continue;
            visited[implied_literal] = 1;
            stack.push_back(implied_literal);
            implied_literals.push_back(implied_literal);
        }
    }
    return implied_literals;
}

void add_implication_conflicts_(ConflictGraph* graph, const Problem& problem,
                                const ImplicationStore* learned_implications,
                                const Options& options) {
    if (graph == nullptr || learned_implications == nullptr)
        return;
    const int literal_count =
        std::min(graph->literal_count(), learned_implications->literal_count());
    bool added = false;
    for (int trigger_literal = 0; trigger_literal < literal_count; ++trigger_literal) {
        const std::vector<int> implied_literals = collect_implied_literals_(
            problem, learned_implications, options, trigger_literal, literal_count);
        for (int consequence_literal : implied_literals) {
            graph->add_implication(trigger_literal, consequence_literal);
            added = true;
        }
    }
    if (added)
        graph->prepare_for_queries();
}

std::vector<std::vector<int>>
build_implication_seed_cliques_(const Problem& problem, const ConflictGraph& graph,
                                const ImplicationStore* learned_implications,
                                const Eigen::VectorXd& primal, const Options& options,
                                int max_seeds) {
    std::vector<std::vector<int>> seeds;
    if (learned_implications == nullptr || max_seeds <= 0)
        return seeds;

    const std::vector<int> fractional = graph.fractional_literals(primal, options.integrality_tol);
    if (fractional.empty())
        return seeds;

    struct SeedCandidate {
        std::vector<int> literals;
        double activity = 0.0;
    };
    std::vector<SeedCandidate> candidates;
    candidates.reserve(fractional.size());

    for (int trigger_literal : fractional) {
        std::vector<int> clique_literals = {trigger_literal};
        std::unordered_set<int> used_variables = {ConflictGraph::variable_of(trigger_literal)};
        std::vector<int> implication_candidates;
        const std::vector<int> implied_literals = collect_implied_literals_(
            problem, learned_implications, options, trigger_literal,
            std::min(graph.literal_count(), learned_implications->literal_count()));
        for (int implied_literal : implied_literals) {
            const int candidate_literal = ConflictGraph::complement_of(implied_literal);
            const int candidate_variable = ConflictGraph::variable_of(candidate_literal);
            if (!graph.has_literal(candidate_literal) ||
                used_variables.contains(candidate_variable))
                continue;
            implication_candidates.push_back(candidate_literal);
        }

        std::sort(implication_candidates.begin(), implication_candidates.end(),
                  [&](int lhs, int rhs) {
                      const double lhs_weight = ConflictGraph::literal_weight(primal, lhs);
                      const double rhs_weight = ConflictGraph::literal_weight(primal, rhs);
                      if (std::abs(lhs_weight - rhs_weight) > 1e-12)
                          return lhs_weight > rhs_weight;
                      const int lhs_degree = graph.degree(lhs);
                      const int rhs_degree = graph.degree(rhs);
                      if (lhs_degree != rhs_degree)
                          return lhs_degree > rhs_degree;
                      return lhs < rhs;
                  });
        implication_candidates.erase(
            std::unique(implication_candidates.begin(), implication_candidates.end()),
            implication_candidates.end());

        for (int candidate_literal : implication_candidates) {
            bool compatible = true;
            for (int chosen_literal : clique_literals) {
                if (!graph.are_conflicting(candidate_literal, chosen_literal)) {
                    compatible = false;
                    break;
                }
            }
            if (!compatible)
                continue;
            clique_literals.push_back(candidate_literal);
            used_variables.insert(ConflictGraph::variable_of(candidate_literal));
        }

        if (clique_literals.size() < 2)
            continue;
        const double activity = clique_literal_activity_(clique_literals, primal);
        if (activity <= 1.0 + options.min_cut_violation)
            continue;
        std::sort(clique_literals.begin(), clique_literals.end());
        clique_literals.erase(std::unique(clique_literals.begin(), clique_literals.end()),
                              clique_literals.end());
        candidates.push_back(SeedCandidate{std::move(clique_literals), activity});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const SeedCandidate& lhs, const SeedCandidate& rhs) {
                  if (std::abs(lhs.activity - rhs.activity) > 1e-12)
                      return lhs.activity > rhs.activity;
                  return lhs.literals.size() > rhs.literals.size();
              });
    if (candidates.size() > static_cast<std::size_t>(max_seeds))
        candidates.resize(max_seeds);

    seeds.reserve(candidates.size());
    for (SeedCandidate& candidate : candidates)
        seeds.push_back(std::move(candidate.literals));
    return seeds;
}

CliqueSignature clique_signature_from_literals_(const std::vector<int>& clique_literals) {
    CliqueSignature signature;
    clique_signature_combine_(&signature, static_cast<std::uint64_t>(clique_literals.size()));
    for (const int literal : clique_literals) {
        clique_signature_combine_(&signature, static_cast<std::uint64_t>(literal));
    }
    return signature;
}

double clique_candidate_score_(const ConflictGraph& graph, const Eigen::VectorXd& primal,
                               int literal) {
    return 1000.0 * ConflictGraph::literal_weight(primal, literal) +
           static_cast<double>(graph.degree(literal));
}

std::vector<std::vector<int>> build_weighted_seed_cliques_(const ConflictGraph& graph,
                                                           const Eigen::VectorXd& primal,
                                                           const Options& options,
                                                           const std::vector<int>& vertices,
                                                           int max_cliques) {
    std::vector<std::vector<int>> cliques;
    if (vertices.size() < 2 || max_cliques <= 0)
        return cliques;

    std::vector<char> is_fractional_literal(graph.literal_count(), 0);
    for (const int literal : vertices) {
        if (graph.has_literal(literal))
            is_fractional_literal[literal] = 1;
    }

    std::vector<int> seeds = vertices;
    std::sort(seeds.begin(), seeds.end(), [&](int lhs, int rhs) {
        const double lhs_score = clique_candidate_score_(graph, primal, lhs);
        const double rhs_score = clique_candidate_score_(graph, primal, rhs);
        if (std::abs(lhs_score - rhs_score) > 1e-12)
            return lhs_score > rhs_score;
        return lhs < rhs;
    });

    const int seed_limit =
        std::min<int>(static_cast<int>(seeds.size()), std::max(max_cliques * 4, 16));
    seeds.resize(seed_limit);

    std::unordered_set<CliqueSignature, CliqueSignatureHash> signatures;
    for (const int seed : seeds) {
        std::vector<int> clique = {seed};
        std::unordered_set<int> used_variables = {ConflictGraph::variable_of(seed)};
        std::vector<int> candidates = graph.neighbors(seed);
        while (true) {
            int best_literal = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            for (const int candidate : candidates) {
                if (candidate < 0 || candidate >= static_cast<int>(is_fractional_literal.size()) ||
                    !is_fractional_literal[candidate]) {
                    continue;
                }
                const int variable = ConflictGraph::variable_of(candidate);
                if (used_variables.contains(variable) ||
                    ConflictGraph::literal_weight(primal, candidate) <= options.integrality_tol) {
                    continue;
                }
                const double score = clique_candidate_score_(graph, primal, candidate);
                if (score > best_score + 1e-12 ||
                    (std::abs(score - best_score) <= 1e-12 && candidate < best_literal)) {
                    best_score = score;
                    best_literal = candidate;
                }
            }
            if (best_literal < 0)
                break;

            clique.push_back(best_literal);
            used_variables.insert(ConflictGraph::variable_of(best_literal));
            candidates = intersect_sorted_(std::move(candidates), graph.neighbors(best_literal));
        }

        if (clique.size() < 2)
            continue;
        if (clique_literal_activity_(clique, primal) <= 1.0 + options.min_cut_violation)
            continue;

        std::sort(clique.begin(), clique.end());
        clique.erase(std::unique(clique.begin(), clique.end()), clique.end());
        const CliqueSignature signature = clique_signature_from_literals_(clique);
        if (!signatures.insert(signature).second)
            continue;
        cliques.push_back(std::move(clique));
        if (static_cast<int>(cliques.size()) >= max_cliques)
            break;
    }

    return cliques;
}

std::vector<std::vector<int>> build_partition_cliques_(const ConflictGraph& graph,
                                                       const Eigen::VectorXd& primal,
                                                       const Options& options,
                                                       const std::vector<int>& vertices,
                                                       int max_cliques) {
    std::vector<std::vector<int>> cliques;
    if (vertices.size() < 2 || max_cliques <= 0)
        return cliques;

    std::vector<int> remaining = vertices;
    std::sort(remaining.begin(), remaining.end(), [&](int lhs, int rhs) {
        const double lhs_weight = ConflictGraph::literal_weight(primal, lhs);
        const double rhs_weight = ConflictGraph::literal_weight(primal, rhs);
        if (std::abs(lhs_weight - rhs_weight) > 1e-12)
            return lhs_weight > rhs_weight;
        const int lhs_degree = graph.degree(lhs);
        const int rhs_degree = graph.degree(rhs);
        if (lhs_degree != rhs_degree)
            return lhs_degree > rhs_degree;
        return lhs < rhs;
    });

    while (!remaining.empty() && static_cast<int>(cliques.size()) < max_cliques) {
        std::vector<int> clique;
        clique.reserve(8);
        clique.push_back(remaining.front());

        std::vector<int> next_remaining;
        next_remaining.reserve(remaining.size());
        for (int i = 1; i < static_cast<int>(remaining.size()); ++i) {
            const int literal = remaining[i];
            bool compatible = true;
            for (int chosen : clique) {
                if (!graph.are_conflicting(literal, chosen)) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) {
                clique.push_back(literal);
            } else {
                next_remaining.push_back(literal);
            }
        }

        if (clique.size() >= 2 &&
            clique_literal_activity_(clique, primal) > 1.0 + options.min_cut_violation) {
            cliques.push_back(clique);
        }

        remaining = std::move(next_remaining);
    }

    return cliques;
}

void lift_clique_literals_(const ConflictGraph& graph, const Eigen::VectorXd& primal,
                           const Options& options, std::vector<int>* clique_literals) {
    if (clique_literals == nullptr || clique_literals->size() < 2)
        return;

    std::sort(clique_literals->begin(), clique_literals->end());
    clique_literals->erase(std::unique(clique_literals->begin(), clique_literals->end()),
                           clique_literals->end());
    if (clique_literals->size() < 2)
        return;

    std::vector<int> candidates = graph.neighbors((*clique_literals)[0]);
    for (int i = 1; i < static_cast<int>(clique_literals->size()) && !candidates.empty(); ++i) {
        candidates =
            intersect_sorted_(std::move(candidates), graph.neighbors((*clique_literals)[i]));
    }

    std::unordered_set<int> used_variables;
    for (int literal : *clique_literals)
        used_variables.insert(ConflictGraph::variable_of(literal));

    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [&](int literal) {
                                        const int variable = ConflictGraph::variable_of(literal);
                                        return std::find(clique_literals->begin(),
                                                         clique_literals->end(),
                                                         literal) != clique_literals->end() ||
                                               used_variables.contains(variable) ||
                                               ConflictGraph::literal_weight(primal, literal) <=
                                                   options.integrality_tol;
                                    }),
                     candidates.end());

    std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs) {
        const double lhs_weight = ConflictGraph::literal_weight(primal, lhs);
        const double rhs_weight = ConflictGraph::literal_weight(primal, rhs);
        if (std::abs(lhs_weight - rhs_weight) > 1e-12)
            return lhs_weight > rhs_weight;
        const int lhs_degree = graph.degree(lhs);
        const int rhs_degree = graph.degree(rhs);
        if (lhs_degree != rhs_degree)
            return lhs_degree > rhs_degree;
        return lhs < rhs;
    });

    const int max_extra = std::max(8, options.max_cuts_added_per_round * 4);
    int added = 0;
    for (int literal : candidates) {
        if (added >= max_extra)
            break;
        bool compatible = true;
        for (int chosen : *clique_literals) {
            if (!graph.are_conflicting(literal, chosen)) {
                compatible = false;
                break;
            }
        }
        if (!compatible)
            continue;
        clique_literals->push_back(literal);
        used_variables.insert(ConflictGraph::variable_of(literal));
        ++added;
    }

    std::sort(clique_literals->begin(), clique_literals->end());
    clique_literals->erase(std::unique(clique_literals->begin(), clique_literals->end()),
                           clique_literals->end());
}

void append_clique_cut_candidate_(const Problem& problem, const RelaxationSolution& relaxation,
                                  const Options& options, const ConflictGraph* graph,
                                  std::vector<int> clique_literals, const std::string& cut_type,
                                  std::unordered_set<CutSignature, CutSignatureHash>* signatures,
                                  std::vector<Cut>* cuts) {
    if (cuts == nullptr || clique_literals.size() < 2)
        return;

    if (graph != nullptr)
        lift_clique_literals_(*graph, relaxation.primal, options, &clique_literals);

    Cut cut = clique_cut_from_literals(problem, clique_literals, options, cut_type);
    if (cut.indices.empty())
        return;

    const double violation = cut_violation(cut, relaxation.primal);
    if (violation <= options.min_cut_violation)
        return;

    cut.strength = violation;
    if (signatures != nullptr) {
        const CutSignature signature = cut_signature(cut);
        if (signatures->contains(signature))
            return;
        signatures->insert(signature);
    }
    cuts->push_back(std::move(cut));
}

void append_implied_bound_cuts_from_leq_(const Problem& problem, const std::vector<int>& indices,
                                         const std::vector<double>& values, double rhs,
                                         const RelaxationSolution& relaxation,
                                         const Options& options, std::vector<Cut>* cuts) {
    if (cuts == nullptr)
        return;

    for (int y_pos = 0;
         y_pos < static_cast<int>(indices.size()) && y_pos < static_cast<int>(values.size());
         ++y_pos) {
        const int y = indices[y_pos];
        const double y_coeff = values[y_pos];
        if (y < 0 || y >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[y] != VariableType::Binary || std::abs(y_coeff) <= 1e-9) {
            continue;
        }

        const double tight_y_value = y_coeff > 0.0 ? 1.0 : 0.0;
        for (int x_pos = 0;
             x_pos < static_cast<int>(indices.size()) && x_pos < static_cast<int>(values.size());
             ++x_pos) {
            if (x_pos == y_pos)
                continue;
            const int x = indices[x_pos];
            const double x_coeff = values[x_pos];
            if (x < 0 || x >= problem.lower_bounds.size() || x >= problem.upper_bounds.size() ||
                std::abs(x_coeff) <= 1e-9)
                continue;

            const bool tighten_upper = x_coeff > 0.0;
            const double x_bound =
                tighten_upper ? problem.upper_bounds(x) : problem.lower_bounds(x);
            if (!std::isfinite(x_bound))
                continue;

            double min_other_activity = 0.0;
            bool finite = true;
            for (int k = 0;
                 k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size()); ++k) {
                if (k == x_pos || k == y_pos)
                    continue;
                const int index = indices[k];
                const double coeff = values[k];
                if (index < 0 || index >= problem.lower_bounds.size() ||
                    index >= problem.upper_bounds.size() || std::abs(coeff) <= 1e-12) {
                    continue;
                }
                const double bound =
                    coeff >= 0.0 ? problem.lower_bounds(index) : problem.upper_bounds(index);
                if (!std::isfinite(bound)) {
                    finite = false;
                    break;
                }
                min_other_activity += coeff * bound;
            }
            if (!finite)
                continue;

            const double tightened_bound =
                (rhs - y_coeff * tight_y_value - min_other_activity) / x_coeff;
            if (!std::isfinite(tightened_bound))
                continue;

            Cut cut;
            cut.cut_type = "ImpliedBound";
            cut.indices = {x, y};

            if (tighten_upper) {
                if (tightened_bound >= x_bound - options.min_cut_violation)
                    continue;
                cut.sense = LinearConstraintSense::LessEqual;
                cut.rhs = tight_y_value > 0.5 ? x_bound : tightened_bound;
                cut.values = {1.0, tight_y_value > 0.5 ? (x_bound - tightened_bound)
                                                       : -(x_bound - tightened_bound)};
            } else {
                if (tightened_bound <= x_bound + options.min_cut_violation)
                    continue;
                cut.sense = LinearConstraintSense::GreaterEqual;
                const double lower = problem.lower_bounds(x);
                if (tight_y_value > 0.5) {
                    cut.rhs = lower;
                    cut.values = {1.0, -(tightened_bound - lower)};
                } else {
                    cut.rhs = tightened_bound;
                    cut.values = {1.0, tightened_bound - lower};
                }
            }

            const double violation = cut_violation(cut, relaxation.primal);
            if (violation <= options.min_cut_violation)
                continue;
            cut.strength = violation;
            cuts->push_back(std::move(cut));
        }
    }
}

void append_clique_cuts_from_leq_(const Problem& problem, const std::vector<int>& indices,
                                  const std::vector<double>& values, double rhs,
                                  const RelaxationSolution& relaxation, const Options& options,
                                  std::vector<Cut>* cuts) {
    if (cuts == nullptr)
        return;

    const std::vector<NormalizedCliqueRow> normalized =
        normalized_clique_rows(problem, indices, values, rhs, LinearConstraintSense::LessEqual);
    if (normalized.empty())
        return;

    std::unordered_set<CutSignature, CutSignatureHash> local_signatures;
    std::vector<Cut> local_cuts;
    const int max_local_cuts = std::max(2, options.max_cuts_added_per_round);
    for (const NormalizedCliqueRow& row : normalized) {
        std::vector<std::vector<int>> row_cliques;
        extract_binary_knapsack_cliques(row, &row_cliques, max_local_cuts * 4);
        for (std::vector<int>& clique_literals : row_cliques) {
            append_clique_cut_candidate_(problem, relaxation, options, nullptr,
                                         std::move(clique_literals), "Clique", &local_signatures,
                                         &local_cuts);
        }
    }

    std::sort(local_cuts.begin(), local_cuts.end(), [](const Cut& lhs, const Cut& rhs) {
        if (std::abs(lhs.strength - rhs.strength) > 1e-12)
            return lhs.strength > rhs.strength;
        if (lhs.indices.size() != rhs.indices.size())
            return lhs.indices.size() > rhs.indices.size();
        return lhs.indices < rhs.indices;
    });
    if (local_cuts.size() > static_cast<std::size_t>(max_local_cuts))
        local_cuts.resize(max_local_cuts);
    cuts->insert(cuts->end(), std::make_move_iterator(local_cuts.begin()),
                 std::make_move_iterator(local_cuts.end()));
}

std::optional<Cut> build_gmi_cut_from_row_(const Problem& problem,
                                           const RelaxationSolution& relaxation,
                                           const Options& options, const LPSolution& lp, int row) {
    const int basic_col = lp.basis_internal[row];
    if (basic_col < 0 || basic_col >= static_cast<int>(lp.internal_column_labels.size()))
        return std::nullopt;

    const auto basic_index = parse_internal_label_index_impl_(lp.internal_column_labels[basic_col]);
    if (!basic_index.has_value() || *basic_index < 0 ||
        *basic_index >= static_cast<int>(problem.variable_types.size()) ||
        problem.variable_types[*basic_index] == VariableType::Continuous) {
        return std::nullopt;
    }

    const double rhs = lp.tableau_rhs(row);
    const double f0 = fractional_part(rhs);
    if (std::min(f0, 1.0 - f0) <= options.min_cut_violation)
        return std::nullopt;

    Cut cut;
    cut.sense = LinearConstraintSense::GreaterEqual;
    cut.rhs = f0 + 1e-9;
    cut.cut_type = "GMI";

    for (int col = 0; col < lp.tableau.cols(); ++col) {
        if (col == basic_col)
            continue;
        const double tij = lp.tableau(row, col);
        if (std::abs(tij) <= 1e-10)
            continue;

        const auto mapped_index = parse_internal_label_index_impl_(lp.internal_column_labels[col]);
        if (!mapped_index.has_value() || *mapped_index < 0 ||
            *mapped_index >= static_cast<int>(problem.variable_types.size())) {
            return std::nullopt;
        }

        double coefficient = 0.0;
        if (problem.variable_types[*mapped_index] != VariableType::Continuous) {
            const double fj = fractional_part(tij);
            coefficient =
                (fj <= f0) ? fj
                           : ((std::abs(1.0 - f0) > 1e-10) ? (f0 * (1.0 - fj)) / (1.0 - f0) : 0.0);
        } else {
            coefficient = (tij > 0.0)
                              ? tij
                              : ((std::abs(1.0 - f0) > 1e-10) ? (-(f0 * tij) / (1.0 - f0)) : 0.0);
        }

        if (std::abs(coefficient) > 1e-8) {
            cut.indices.push_back(*mapped_index);
            cut.values.push_back(coefficient);
        }
    }

    if (cut.indices.empty())
        return std::nullopt;

    cut = eliminate_base_row_slacks_from_cut_(problem, std::move(cut));
    if (!postprocess_gmi_cut_(problem, options, &cut))
        return std::nullopt;
    if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3) || cut.indices.empty())
        return std::nullopt;

    const double violation = cut_violation(cut, relaxation.primal);
    if (violation <= options.min_cut_violation)
        return std::nullopt;
    cut.strength = gmi_cut_quality_score_(cut, relaxation.primal, violation);
    return cut;
}

Cut aggregate_gmi_cuts_(const Cut& lhs, const Cut& rhs, double rhs_scale,
                        const std::string& cut_type) {
    Cut aggregated;
    aggregated.sense = LinearConstraintSense::GreaterEqual;
    aggregated.rhs = lhs.rhs + rhs_scale * rhs.rhs;
    aggregated.cut_type = cut_type;
    aggregated.indices = lhs.indices;
    aggregated.values = lhs.values;
    aggregated.indices.insert(aggregated.indices.end(), rhs.indices.begin(), rhs.indices.end());
    for (double value : rhs.values)
        aggregated.values.push_back(rhs_scale * value);
    return aggregated;
}

std::vector<Cut> strengthen_gmi_candidates_(const Problem& problem,
                                            std::vector<Cut> base_candidates,
                                            const RelaxationSolution& relaxation,
                                            const Options& options) {
    std::vector<Cut> cuts;
    std::unordered_set<CutSignature, CutSignatureHash> signatures;

    auto maybe_add = [&](Cut cut) {
        if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3) || cut.indices.empty())
            return;
        const double violation = cut_violation(cut, relaxation.primal);
        if (violation <= options.min_cut_violation)
            return;
        cut.strength =
            std::max(cut.strength, gmi_cut_quality_score_(cut, relaxation.primal, violation));
        const CutSignature signature = cut_signature(cut);
        if (!signatures.insert(signature).second)
            return;
        cuts.push_back(std::move(cut));
    };

    for (Cut& cut : base_candidates)
        maybe_add(std::move(cut));

    std::sort(cuts.begin(), cuts.end(),
              [](const Cut& lhs, const Cut& rhs) { return lhs.strength > rhs.strength; });

    if (problem.variable_types.size() < 16 || cuts.size() < 4) {
        const int cap = std::max(2, options.max_cuts_added_per_round * 2);
        if (static_cast<int>(cuts.size()) > cap)
            cuts.resize(cap);
        return cuts;
    }

    const int top = std::min<int>(4, cuts.size());
    for (int i = 0; i < top; ++i) {
        const double lhs_norm = cut_norm_(cuts[i]);
        for (int j = i + 1; j < top; ++j) {
            const double rhs_norm = cut_norm_(cuts[j]);
            const double scale = (rhs_norm > 1e-12) ? lhs_norm / rhs_norm : 1.0;
            Cut aggregated = aggregate_gmi_cuts_(cuts[i], cuts[j], scale, "GMI-Agg2");
            const double raw_violation = cut_violation(aggregated, relaxation.primal);
            if (raw_violation <= std::max(3.0 * options.min_cut_violation, 5e-2))
                continue;
            maybe_add(std::move(aggregated));
        }
    }

    if (top >= 3) {
        Cut triple = aggregate_gmi_cuts_(cuts[0], cuts[1], 1.0, "GMI-Agg3");
        triple.indices.insert(triple.indices.end(), cuts[2].indices.begin(), cuts[2].indices.end());
        for (double value : cuts[2].values)
            triple.values.push_back(value);
        triple.rhs += cuts[2].rhs;
        const double raw_violation = cut_violation(triple, relaxation.primal);
        if (raw_violation > std::max(4.0 * options.min_cut_violation, 7.5e-2))
            maybe_add(std::move(triple));
    }

    std::sort(cuts.begin(), cuts.end(),
              [](const Cut& lhs, const Cut& rhs) { return lhs.strength > rhs.strength; });

    const int cap = std::max(2, options.max_cuts_added_per_round * 3);
    if (static_cast<int>(cuts.size()) > cap)
        cuts.resize(cap);
    return cuts;
}

namespace dual_proof_detail {

template <typename Row>
bool accumulate_standardized_row_(const Row& row, double multiplier, int original_vars,
                                  Eigen::VectorXd* aggregated, double* rhs) {
    if (aggregated == nullptr || rhs == nullptr)
        return false;

    double lambda = 0.0;
    double sign = 1.0;
    switch (row.sense) {
        case LinearConstraintSense::LessEqual:
            lambda = -multiplier;
            sign = 1.0;
            break;
        case LinearConstraintSense::GreaterEqual:
            lambda = multiplier;
            sign = -1.0;
            break;
        case LinearConstraintSense::Equal:
            lambda = std::abs(multiplier);
            sign = multiplier >= 0.0 ? 1.0 : -1.0;
            break;
    }

    if (!(lambda > 0.0) || !std::isfinite(lambda))
        return true;

    for (int k = 0;
         k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size()); ++k) {
        const int index = row.indices[k];
        if (index < 0 || index >= original_vars)
            return false;
        (*aggregated)(index) += lambda * sign * row.values[k];
    }
    *rhs += lambda * sign * row.rhs;
    return true;
}

std::optional<Cut> build_farkas_cut_(const Problem& problem, const std::vector<Cut>& active_cuts,
                                     const RelaxationSolution& relaxation,
                                     const Eigen::VectorXd& node_lower_bounds,
                                     const Eigen::VectorXd& node_upper_bounds,
                                     const Options& options) {
    if (!relaxation.lp_solution.has_value())
        return std::nullopt;

    const LPSolution& lp_sol = *relaxation.lp_solution;
    const int n = static_cast<int>(problem.variable_types.size());
    const int row_count = static_cast<int>(problem.base_constraints.size() + active_cuts.size());
    if (relaxation.status != RelaxationStatus::Infeasible || !lp_sol.farkas_has_cert ||
        problem.lower_bounds.size() != n || problem.upper_bounds.size() != n ||
        node_lower_bounds.size() < n || node_upper_bounds.size() < n) {
        return std::nullopt;
    }

    const Eigen::VectorXd* certificate = nullptr;
    if (lp_sol.farkas_y.size() == row_count) {
        certificate = &lp_sol.farkas_y;
    } else if (lp_sol.farkas_y_internal.size() == row_count) {
        certificate = &lp_sol.farkas_y_internal;
    } else {
        return std::nullopt;
    }

    Eigen::VectorXd aggregated = Eigen::VectorXd::Zero(n);
    double rhs = 0.0;
    int row_index = 0;
    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (!accumulate_standardized_row_(row, (*certificate)(row_index++), n, &aggregated, &rhs))
            return std::nullopt;
    }
    for (const Cut& row : active_cuts) {
        if (!accumulate_standardized_row_(row, (*certificate)(row_index++), n, &aggregated, &rhs))
            return std::nullopt;
    }

    Cut cut;
    cut.cut_type = "DualProof";
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = rhs;

    for (int j = 0; j < n; ++j) {
        const double coeff = aggregated(j);
        if (std::abs(coeff) <= options.feasibility_tol)
            continue;

        if (problem.variable_types[j] == VariableType::Continuous) {
            if (coeff > 0.0) {
                if (!std::isfinite(problem.upper_bounds(j)))
                    return std::nullopt;
                cut.rhs -= coeff * problem.upper_bounds(j);
            } else {
                if (!std::isfinite(problem.lower_bounds(j)))
                    return std::nullopt;
                cut.rhs -= coeff * problem.lower_bounds(j);
            }
            continue;
        }

        cut.indices.push_back(j);
        cut.values.push_back(coeff);
    }

    if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3) || cut.indices.empty())
        return std::nullopt;

    double node_min_lhs = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        const double coeff = cut.values[k];
        const double bound = coeff >= 0.0 ? node_lower_bounds(index) : node_upper_bounds(index);
        if (!std::isfinite(bound))
            return std::nullopt;
        node_min_lhs += coeff * bound;
    }

    const double violation = node_min_lhs - cut.rhs;
    if (!(violation > options.min_cut_violation))
        return std::nullopt;

    cut.strength = violation;
    return cut;
}

} // namespace dual_proof_detail

} // namespace

std::size_t CutSignatureHash::operator()(const CutSignature& signature) const noexcept {
    const std::uint64_t combined = signature.lo ^ (signature.hi + 0x9e3779b97f4a7c15ULL +
                                                   (signature.lo << 6) + (signature.lo >> 2));
    return static_cast<std::size_t>(combined);
}

double cut_violation(const Cut& cut, const Eigen::VectorXd& primal) {
    if (primal.size() == 0) {
        return 0.0;
    }
    double lhs = 0.0;
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        if (index >= 0 && index < primal.size())
            lhs += cut.values[k] * primal(index);
    }
    switch (cut.sense) {
        case LinearConstraintSense::LessEqual:
            return std::max(0.0, lhs - cut.rhs);
        case LinearConstraintSense::GreaterEqual:
            return std::max(0.0, cut.rhs - lhs);
        case LinearConstraintSense::Equal:
            return std::abs(lhs - cut.rhs);
    }
    return 0.0;
}

CutSignature cut_signature(const Cut& cut, int precision) {
    std::vector<std::pair<int, double>> terms;
    terms.reserve(std::min(cut.indices.size(), cut.values.size()));
    const double scale = std::pow(10.0, precision);
    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        if (std::abs(cut.values[k]) <= 1e-12)
            continue;
        const double rounded = std::round(cut.values[k] * scale) / scale;
        terms.emplace_back(cut.indices[k], rounded);
    }
    std::sort(terms.begin(), terms.end());

    CutSignature signature;
    signature_combine_(&signature, static_cast<std::uint64_t>(terms.size()));
    for (const auto& [index, coeff] : terms) {
        signature_combine_(&signature, static_cast<std::uint64_t>(index));
        signature_combine_(&signature, signature_bits_(coeff));
    }
    const double rounded_rhs = std::round(cut.rhs * scale) / scale;
    signature_combine_(&signature, signature_bits_(rounded_rhs));
    signature_combine_(&signature, static_cast<std::uint64_t>(cut.sense));
    return signature;
}

bool canonicalize_cut(Cut* cut, double zero_tol) {
    if (cut == nullptr || !std::isfinite(cut->rhs))
        return false;

    std::vector<std::pair<int, double>> merged;
    merged.reserve(std::min(cut->indices.size(), cut->values.size()));
    for (int k = 0;
         k < static_cast<int>(cut->indices.size()) && k < static_cast<int>(cut->values.size());
         ++k) {
        if (std::abs(cut->values[k]) <= zero_tol)
            continue;
        merged.emplace_back(cut->indices[k], cut->values[k]);
    }
    if (merged.empty())
        return false;

    std::sort(merged.begin(), merged.end());
    std::vector<int> indices;
    std::vector<double> values;
    indices.reserve(merged.size());
    values.reserve(merged.size());
    for (const auto& [index, value] : merged) {
        if (!indices.empty() && indices.back() == index) {
            values.back() += value;
        } else {
            indices.push_back(index);
            values.push_back(value);
        }
    }

    double max_abs = 0.0;
    std::vector<int> final_indices;
    std::vector<double> final_values;
    final_indices.reserve(indices.size());
    final_values.reserve(values.size());
    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        if (std::abs(values[i]) <= zero_tol)
            continue;
        final_indices.push_back(indices[i]);
        final_values.push_back(values[i]);
        max_abs = std::max(max_abs, std::abs(values[i]));
    }
    if (final_indices.empty() || max_abs <= zero_tol)
        return false;

    for (double& value : final_values)
        value /= max_abs;
    cut->rhs /= max_abs;
    cut->indices = std::move(final_indices);
    cut->values = std::move(final_values);
    return std::isfinite(cut->rhs);
}

Cut clique_cut_from_literals(const Problem& problem, const std::vector<int>& clique_literals,
                             const Options& options, const std::string& cut_type) {
    Cut cut;
    cut.sense = LinearConstraintSense::LessEqual;
    cut.rhs = 1.0;
    cut.cut_type = cut_type;
    for (int literal : clique_literals) {
        const int variable = ConflictGraph::variable_of(literal);
        if (variable < 0 || variable >= static_cast<int>(problem.variable_types.size()))
            continue;
        cut.indices.push_back(variable);
        if (ConflictGraph::value_of(literal)) {
            cut.values.push_back(1.0);
        } else {
            cut.values.push_back(-1.0);
            cut.rhs -= 1.0;
        }
    }
    if (!canonicalize_cut(&cut, options.min_cut_violation * 1e-3))
        cut.indices.clear();
    return cut;
}

double cut_parallelism(const Cut& lhs, const Cut& rhs) {
    return cut_parallelism_with_norms_(lhs, cut_norm_(lhs), rhs, cut_norm_(rhs));
}

CutPool::CutPool(const Options& options)
    : max_pool_size_(options.max_cut_pool_size), min_violation_(options.min_cut_violation),
      max_age_(options.max_cut_age), cut_age_decay_(options.cut_age_decay),
      cut_selection_age_bonus_(options.cut_selection_age_bonus),
      max_cuts_per_type_(options.max_cuts_per_type), max_parallelism_(options.cut_max_parallelism) {
}

void CutPool::reset(const Options& options) {
    const std::lock_guard<std::shared_mutex> lock(cuts_mutex_);
    max_pool_size_ = options.max_cut_pool_size;
    min_violation_ = options.min_cut_violation;
    max_age_ = options.max_cut_age;
    cut_age_decay_ = options.cut_age_decay;
    cut_selection_age_bonus_ = options.cut_selection_age_bonus;
    max_cuts_per_type_ = options.max_cuts_per_type;
    max_parallelism_ = options.cut_max_parallelism;
    cuts_.clear();
    row_norms_.clear();
    signatures_.clear();
    support_buckets_.clear();
    generated_counts_.clear();
    applied_counts_.clear();
    type_usage_stats_.clear();
    cuts_generated_ = 0;
    cuts_applied_ = 0;
}

namespace {

bool validate_cut_indices_(const Cut& cut, int num_variables) {
    if (cut.indices.size() != cut.values.size())
        return false;
    for (int index : cut.indices) {
        if (index < 0 || index >= num_variables)
            return false;
    }
    return true;
}

bool add_cut_locked_(const Problem& problem, const Cut& cut, double min_violation,
                     std::vector<Cut>* cuts, std::vector<double>* row_norms,
                     std::unordered_set<CutSignature, CutSignatureHash>* signatures,
                     std::unordered_map<std::uint64_t, std::vector<int>>* support_buckets,
                     std::unordered_map<std::string, int>* generated_counts,
                     std::unordered_map<std::string, int>* type_usage_stats, int* cuts_generated,
                     int* duplicate_cuts) {
    if (!validate_cut_indices_(cut, static_cast<int>(problem.variable_types.size()))) {
        return false;
    }

    Cut canonical = cut;
    if (!canonicalize_cut(&canonical, min_violation * 1e-3)) {
        return false;
    }

    const CutSignature signature = cut_signature(canonical);
    if (signatures->contains(signature)) {
        ++*duplicate_cuts;
        return false;
    }

    const std::uint64_t support_signature = support_signature_(canonical);
    if (auto bucket = support_buckets->find(support_signature); bucket != support_buckets->end()) {
        const double canonical_norm = cut_norm_(canonical);
        for (int existing_index : bucket->second) {
            if (existing_index < 0 || existing_index >= static_cast<int>(cuts->size())) {
                continue;
            }
            const Cut& existing = (*cuts)[existing_index];
            if (!same_support_(existing, canonical)) {
                continue;
            }
            const double existing_norm = existing_index < static_cast<int>(row_norms->size())
                                             ? (*row_norms)[existing_index]
                                             : cut_norm_(existing);
            const double parallelism =
                cut_parallelism_with_norms_(existing, existing_norm, canonical, canonical_norm);
            if (parallelism >= 1.0 - 1e-6 &&
                std::abs(existing.rhs - canonical.rhs) <= std::max(1e-9, min_violation)) {
                ++*duplicate_cuts;
                return false;
            }
        }
    }

    cuts->push_back(std::move(canonical));
    row_norms->push_back(cut_norm_(cuts->back()));
    signatures->insert(signature);
    (*support_buckets)[support_signature].push_back(static_cast<int>(cuts->size()) - 1);
    ++(*generated_counts)[cuts->back().cut_type];
    type_usage_stats->try_emplace(cuts->back().cut_type, 0);
    ++*cuts_generated;
    return true;
}

} // namespace

bool CutPool::add_cut(const Problem& problem, const Cut& cut) {
    detail::TimingTrace timing_trace("cutpool_add_cut");
    detail::LockTrace lock_trace("cuts_mutex_");
    std::unique_lock<std::shared_mutex> lock(cuts_mutex_);
    lock_trace.acquired_lock();
    const bool added = add_cut_locked_(problem, cut, min_violation_, &cuts_, &row_norms_,
                                       &signatures_, &support_buckets_, &generated_counts_,
                                       &type_usage_stats_, &cuts_generated_, &duplicate_cuts_);
    const bool should_manage =
        cuts_.size() > static_cast<std::size_t>(max_pool_size_ + std::max(8, max_pool_size_ / 8));
    lock.unlock();
    if (should_manage) {
        manage_pool_size_();
    }
    return added;
}

std::vector<Cut> CutPool::select_violated_cuts(const Eigen::VectorXd& primal,
                                               const Eigen::VectorXd& lower_bounds,
                                               const Eigen::VectorXd& upper_bounds, int max_cuts,
                                               double density_penalty_scale) {
    detail::TimingTrace timing_trace("cutpool_select_violated_cuts");
    detail::LockTrace lock_trace("cuts_mutex_");
    std::unique_lock<std::shared_mutex> lock(cuts_mutex_);
    lock_trace.acquired_lock();
    if (max_cuts <= 0 || cuts_.empty()) {
        return {};
    }

    struct Candidate {
        int index = -1;
        double violation = 0.0;
        double full_norm = 0.0;
        double efficacy = 0.0;
        double active_efficacy = 0.0;
        double density_adjusted_efficacy = 0.0;
        double dynamism = 0.0;
        double fractional_focus = 0.0;
        double strength = 0.0;
        double age_bonus = 0.0;
        bool marginal = false;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(cuts_.size());
    std::vector<char> violated(cuts_.size(), 0);
    for (int i = 0; i < static_cast<int>(cuts_.size()); ++i) {
        const double violation = cut_violation(cuts_[i], primal);
        if (violation <= min_violation_)
            continue;
        violated[i] = 1;
        const double full_norm =
            i < static_cast<int>(row_norms_.size()) ? row_norms_[i] : cut_norm_(cuts_[i]);
        const double age_bonus =
            std::exp(-cut_selection_age_bonus_ * static_cast<double>(cuts_[i].age));
        candidates.push_back(Candidate{i, violation, full_norm, 0.0, 0.0, 0.0, 0.0, 0.0,
                                       cuts_[i].strength, age_bonus,
                                       violation <= 2.5 * min_violation_});
    }

    if (candidates.empty()) {
        return {};
    }

    const int max_candidates = std::max(32, max_cuts * 8);
    if (static_cast<int>(candidates.size()) > max_candidates) {
        std::nth_element(candidates.begin(), candidates.begin() + max_candidates, candidates.end(),
                         [](const Candidate& lhs, const Candidate& rhs) {
                             const double lhs_ratio = lhs.full_norm > 1e-16
                                                          ? lhs.violation / lhs.full_norm
                                                          : lhs.violation;
                             const double rhs_ratio = rhs.full_norm > 1e-16
                                                          ? rhs.violation / rhs.full_norm
                                                          : rhs.violation;
                             return lhs_ratio > rhs_ratio;
                         });
        candidates.resize(max_candidates);
    }

    for (auto& candidate : candidates) {
        const Cut& cut = cuts_[candidate.index];
        const ActiveSupportStats active_stats =
            active_support_stats_(cut, primal, lower_bounds, upper_bounds, min_violation_);
        const double norm = active_stats.norm > 1e-16 ? active_stats.norm : candidate.full_norm;
        candidate.efficacy = norm > 1e-16 ? candidate.violation / norm : 0.0;
        candidate.active_efficacy = active_stats.norm > 1e-16
                                        ? candidate.violation / active_stats.norm
                                        : candidate.efficacy;
        candidate.fractional_focus = fractional_focus_(cut, primal);
        candidate.density_adjusted_efficacy = density_adjusted_efficacy_(
            candidate.violation, norm,
            active_stats.nnz > 0 ? active_stats.nnz : static_cast<int>(cut.indices.size()),
            candidate.fractional_focus, density_penalty_scale);
        candidate.dynamism = cut_dynamism_(cut, primal, candidate.violation, norm);
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        const double lhs_score = 0.45 * lhs.active_efficacy + 0.20 * lhs.efficacy +
                                 0.15 * lhs.density_adjusted_efficacy + 0.10 * lhs.dynamism +
                                 0.10 * lhs.fractional_focus;
        const double rhs_score = 0.45 * rhs.active_efficacy + 0.20 * rhs.efficacy +
                                 0.15 * rhs.density_adjusted_efficacy + 0.10 * rhs.dynamism +
                                 0.10 * rhs.fractional_focus;
        return lhs_score > rhs_score;
    });

    std::vector<Cut> selected;
    std::vector<int> selected_indices;
    std::unordered_map<std::string, int> local_type_counts;
    std::vector<char> chosen(cuts_.size(), 0);
    selected.reserve(static_cast<std::size_t>(std::max(0, max_cuts)));
    selected_indices.reserve(static_cast<std::size_t>(std::max(0, max_cuts)));

    while (selected.size() < static_cast<std::size_t>(max_cuts)) {
        double best_score = -std::numeric_limits<double>::infinity();
        int best_pos = -1;
        for (int pos = 0; pos < static_cast<int>(candidates.size()); ++pos) {
            const Candidate& candidate = candidates[pos];
            if (candidate.index < 0 || candidate.index >= static_cast<int>(cuts_.size()) ||
                chosen[candidate.index]) {
                continue;
            }
            const Cut& cut = cuts_[candidate.index];
            if (max_cuts_per_type_ > 0 && local_type_counts[cut.cut_type] >= max_cuts_per_type_) {
                continue;
            }

            double max_parallelism = 0.0;
            const double candidate_norm = row_norms_[candidate.index];
            for (int selected_index : selected_indices) {
                max_parallelism =
                    std::max(max_parallelism,
                             cut_parallelism_with_norms_(cut, candidate_norm, cuts_[selected_index],
                                                         row_norms_[selected_index]));
            }
            if (max_parallelism > max_parallelism_)
                continue;

            double score = 0.40 * candidate.active_efficacy + 0.15 * candidate.efficacy +
                           0.20 * (1.0 - max_parallelism) + 0.12 * candidate.age_bonus +
                           0.10 * candidate.density_adjusted_efficacy + 0.08 * candidate.strength +
                           dynamism_weight_ * candidate.dynamism +
                           0.05 * candidate.fractional_focus;
            if (candidate.marginal) {
                score +=
                    0.06 * candidate.density_adjusted_efficacy + 0.04 * candidate.fractional_focus;
            }
            score += 0.02 * std::log1p(static_cast<double>(type_usage_stats_[cut.cut_type]));
            if (score > best_score) {
                best_score = score;
                best_pos = pos;
            }
        }

        if (best_pos < 0)
            break;

        const int index = candidates[best_pos].index;
        chosen[index] = 1;
        selected.push_back(cuts_[index]);
        selected_indices.push_back(index);
        ++local_type_counts[cuts_[index].cut_type];
        ++applied_counts_[cuts_[index].cut_type];
        ++type_usage_stats_[cuts_[index].cut_type];
        ++cuts_[index].times_used;
        cuts_[index].age = 0;
        cuts_[index].strength = std::max(cuts_[index].strength, 0.90 * best_score);
    }

    for (int i = 0; i < static_cast<int>(cuts_.size()); ++i) {
        if (chosen[i])
            continue;
        ++cuts_[i].age;
        cuts_[i].strength *= violated[i] ? 0.998 : 0.97;
    }

    cuts_applied_ += static_cast<int>(selected.size());
    lock.unlock();
    manage_pool_size_();
    return selected;
}

void CutPool::manage_pool_size_() {
    detail::TimingTrace timing_trace("cutpool_manage_pool_size");
    detail::LockTrace lock_trace("cuts_mutex_");
    std::unique_lock<std::shared_mutex> lock(cuts_mutex_);
    lock_trace.acquired_lock();
    if (cuts_.empty())
        return;

    std::vector<int> keep_indices;
    keep_indices.reserve(cuts_.size());
    for (int i = 0; i < static_cast<int>(cuts_.size()); ++i) {
        if (cuts_[i].age <= max_age_ || cuts_[i].strength > 0.35 || cuts_[i].times_used > 0)
            keep_indices.push_back(i);
    }

    if (keep_indices.size() == cuts_.size() &&
        cuts_.size() <= static_cast<std::size_t>(max_pool_size_)) {
        return;
    }
    if (keep_indices.size() > static_cast<std::size_t>(max_pool_size_)) {
        std::sort(keep_indices.begin(), keep_indices.end(), [&](int lhs, int rhs) {
            return cut_retention_score_(cuts_[lhs], row_norms_[lhs],
                                        type_usage_stats_[cuts_[lhs].cut_type], cut_age_decay_) >
                   cut_retention_score_(cuts_[rhs], row_norms_[rhs],
                                        type_usage_stats_[cuts_[rhs].cut_type], cut_age_decay_);
        });
        keep_indices.resize(max_pool_size_);
    }

    std::vector<Cut> new_cuts;
    std::vector<double> new_norms;
    new_cuts.reserve(keep_indices.size());
    new_norms.reserve(keep_indices.size());
    for (int index : keep_indices) {
        new_cuts.push_back(std::move(cuts_[index]));
        new_norms.push_back(row_norms_[index]);
    }
    cuts_ = std::move(new_cuts);
    row_norms_ = std::move(new_norms);

    signatures_.clear();
    support_buckets_.clear();
    for (const Cut& cut : cuts_)
        signatures_.insert(cut_signature(cut));
    for (int i = 0; i < static_cast<int>(cuts_.size()); ++i)
        support_buckets_[support_signature_(cuts_[i])].push_back(i);
}

void CutPool::perform_aging() { manage_pool_size_(); }

namespace {

std::optional<int> parse_internal_label_index_impl_(const std::string& label) {
    constexpr const char* prefix = "x_orig_";
    if (!label.starts_with(prefix))
        return std::nullopt;
    try {
        return std::stoi(label.substr(std::char_traits<char>::length(prefix)));
    } catch (...) {
        return std::nullopt;
    }
}

} // namespace

std::optional<int> parse_internal_label_index(const std::string& label) {
    return parse_internal_label_index_impl_(label);
}

namespace {

Cut eliminate_base_row_slacks_from_cut_(const Problem& problem, Cut cut) {
    const int total_vars = static_cast<int>(problem.variable_types.size());
    const int slack_count = static_cast<int>(
        std::count_if(problem.base_constraints.begin(), problem.base_constraints.end(),
                      [](const SparseLinearConstraint& row) {
                          return row.sense != LinearConstraintSense::Equal;
                      }));
    const int original_vars = total_vars - slack_count;
    if (slack_count <= 0 || original_vars <= 0 || original_vars >= total_vars) {
        return cut;
    }

    std::vector<const SparseLinearConstraint*> slack_rows(static_cast<std::size_t>(slack_count),
                                                          nullptr);
    int next_slack = 0;
    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense == LinearConstraintSense::Equal)
            continue;
        if (next_slack >= slack_count)
            break;
        slack_rows[static_cast<std::size_t>(next_slack++)] = &row;
    }

    std::vector<double> coefficients(static_cast<std::size_t>(original_vars), 0.0);
    auto add_original_coeff = [&](int index, double value) {
        if (index < 0 || index >= original_vars || std::abs(value) <= 1e-12)
            return;
        coefficients[static_cast<std::size_t>(index)] += value;
    };

    for (int k = 0;
         k < static_cast<int>(cut.indices.size()) && k < static_cast<int>(cut.values.size()); ++k) {
        const int index = cut.indices[k];
        const double value = cut.values[k];
        if (index < 0 || index >= total_vars || std::abs(value) <= 1e-12)
            continue;
        if (index < original_vars) {
            add_original_coeff(index, value);
            continue;
        }

        const int slack_pos = index - original_vars;
        if (slack_pos < 0 || slack_pos >= slack_count)
            continue;
        const SparseLinearConstraint* row = slack_rows[static_cast<std::size_t>(slack_pos)];
        if (row == nullptr)
            continue;

        const double row_sign = row->sense == LinearConstraintSense::LessEqual ? -1.0 : 1.0;
        for (int t = 0;
             t < static_cast<int>(row->indices.size()) && t < static_cast<int>(row->values.size());
             ++t) {
            add_original_coeff(row->indices[t], row_sign * value * row->values[t]);
        }
        cut.rhs += row_sign * value * row->rhs;
    }

    cut.indices.clear();
    cut.values.clear();
    for (int j = 0; j < original_vars; ++j) {
        const double value = coefficients[static_cast<std::size_t>(j)];
        if (std::abs(value) <= 1e-12)
            continue;
        cut.indices.push_back(j);
        cut.values.push_back(value);
    }
    return cut;
}

double gmi_candidate_away_(const Options& options) {
    return std::max({10.0 * options.integrality_tol, 5.0 * options.min_cut_violation, 5e-4});
}

double gmi_cut_quality_score_(const Cut& cut, const Eigen::VectorXd& primal, double violation) {
    const double norm = cut_norm_(cut);
    if (norm <= 1e-16)
        return 0.0;
    const double focus = fractional_focus_(cut, primal);
    return density_adjusted_efficacy_(violation, norm, static_cast<int>(cut.indices.size()), focus,
                                      1.0);
}

bool scale_integral_support_gmi_cut_(const Problem& problem, const Options& options, Cut* cut) {
    if (cut == nullptr || cut->indices.empty())
        return false;

    bool integral_support = true;
    for (int index : cut->indices) {
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[index] == VariableType::Continuous) {
            integral_support = false;
            break;
        }
    }
    if (!integral_support)
        return false;

    constexpr std::array<int, 11> kCandidateScales = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    const double coeff_tol = std::max(1e-8, 50.0 * options.integrality_tol);
    for (const int scale : kCandidateScales) {
        bool good = true;
        double max_abs = 0.0;
        std::vector<double> scaled_values;
        scaled_values.reserve(cut->values.size());
        for (double value : cut->values) {
            const double scaled = static_cast<double>(scale) * value;
            const double rounded = std::round(scaled);
            if (std::abs(scaled - rounded) > coeff_tol) {
                good = false;
                break;
            }
            scaled_values.push_back(rounded);
            max_abs = std::max(max_abs, std::abs(rounded));
        }
        if (!good || max_abs > (1 << 20))
            continue;

        const double scaled_rhs = static_cast<double>(scale) * cut->rhs;
        cut->rhs = std::ceil(scaled_rhs - coeff_tol);
        cut->values = std::move(scaled_values);
        return true;
    }

    return false;
}

bool postprocess_gmi_cut_(const Problem& problem, const Options& options, Cut* cut) {
    if (cut == nullptr || cut->indices.empty())
        return false;

    const double max_abs = std::accumulate(
        cut->values.begin(), cut->values.end(), 0.0,
        [](double current, double value) { return std::max(current, std::abs(value)); });
    if (max_abs <= 1e-16)
        return false;

    const double drop_tol =
        std::max(1e-9, 25.0 * options.integrality_tol) * std::max(max_abs, 1e-3);

    std::vector<int> kept_indices;
    std::vector<double> kept_values;
    kept_indices.reserve(cut->indices.size());
    kept_values.reserve(cut->values.size());

    for (int k = 0;
         k < static_cast<int>(cut->indices.size()) && k < static_cast<int>(cut->values.size());
         ++k) {
        const int index = cut->indices[k];
        const double value = cut->values[k];
        if (index < 0 || index >= problem.lower_bounds.size() ||
            index >= problem.upper_bounds.size())
            return false;

        if (std::abs(value) <= drop_tol) {
            if (value > 0.0) {
                if (!std::isfinite(problem.lower_bounds(index)))
                    return false;
                cut->rhs -= value * problem.lower_bounds(index);
            } else {
                if (!std::isfinite(problem.upper_bounds(index)))
                    return false;
                cut->rhs -= value * problem.upper_bounds(index);
            }
            continue;
        }

        kept_indices.push_back(index);
        kept_values.push_back(value);
    }

    if (kept_indices.empty())
        return false;

    cut->indices = std::move(kept_indices);
    cut->values = std::move(kept_values);

    scale_integral_support_gmi_cut_(problem, options, cut);

    double min_abs = std::numeric_limits<double>::infinity();
    double new_max_abs = 0.0;
    for (double value : cut->values) {
        const double abs_value = std::abs(value);
        if (abs_value <= 1e-16)
            continue;
        min_abs = std::min(min_abs, abs_value);
        new_max_abs = std::max(new_max_abs, abs_value);
    }
    if (!(min_abs < std::numeric_limits<double>::infinity()) || new_max_abs <= 1e-16)
        return false;
    if (new_max_abs / min_abs > 1e6)
        return false;
    if (cut->indices.size() > 128)
        return false;

    return true;
}

bool strengthen_integral_cut_(const Problem& problem, const Options& options, Cut* cut) {
    if (cut == nullptr || cut->indices.empty())
        return false;

    for (int index : cut->indices) {
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()))
            return false;
        if (problem.variable_types[index] == VariableType::Continuous)
            return true;
    }

    scale_integral_support_gmi_cut_(problem, options, cut);
    return true;
}

bool postprocess_mir_cut_(const Problem& problem, const Options& options, Cut* cut) {
    if (cut == nullptr || cut->indices.empty())
        return false;
    if (!strengthen_integral_cut_(problem, options, cut))
        return false;
    if (!canonicalize_cut(cut, options.min_cut_violation * 1e-3) || cut->indices.empty())
        return false;
    return true;
}

bool postprocess_cover_cut_(const Problem& problem, const Options& options, Cut* cut) {
    if (cut == nullptr || cut->indices.empty())
        return false;
    if (!strengthen_integral_cut_(problem, options, cut))
        return false;
    if (!canonicalize_cut(cut, options.min_cut_violation * 1e-3) || cut->indices.empty())
        return false;
    return true;
}

std::vector<int> select_gmi_rows_(const Problem& problem, const RelaxationSolution& relaxation,
                                  const Options& options, const LPSolution& lp) {
    struct Candidate {
        int row = -1;
        double score = 0.0;
        double fractionality = 0.0;
        double norm = 0.0;
        std::vector<int> support_indices;
        std::vector<double> support_values;
    };

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<std::size_t>(lp.tableau.rows()));
    const double away = gmi_candidate_away_(options);

    auto sparse_parallelism = [](const Candidate& lhs, const Candidate& rhs) {
        if (lhs.norm <= 1e-16 || rhs.norm <= 1e-16)
            return 0.0;
        double dot = 0.0;
        int i = 0;
        int j = 0;
        while (i < static_cast<int>(lhs.support_indices.size()) &&
               j < static_cast<int>(rhs.support_indices.size())) {
            if (lhs.support_indices[i] == rhs.support_indices[j]) {
                dot += lhs.support_values[i] * rhs.support_values[j];
                ++i;
                ++j;
            } else if (lhs.support_indices[i] < rhs.support_indices[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        return std::abs(dot) / (lhs.norm * rhs.norm);
    };

    for (int row = 0; row < lp.tableau.rows(); ++row) {
        const int basic_col = lp.basis_internal[row];
        if (basic_col < 0 || basic_col >= static_cast<int>(lp.internal_column_labels.size()))
            continue;

        const auto basic_index =
            parse_internal_label_index_impl_(lp.internal_column_labels[basic_col]);
        if (!basic_index.has_value() || *basic_index < 0 ||
            *basic_index >= static_cast<int>(problem.variable_types.size()) ||
            problem.variable_types[*basic_index] == VariableType::Continuous) {
            continue;
        }

        const double rhs = lp.tableau_rhs(row);
        const double f0 = fractional_part(rhs);
        const double frac = std::min(f0, 1.0 - f0);
        if (frac <= away)
            continue;

        double norm_sq = 0.0;
        double min_abs = std::numeric_limits<double>::infinity();
        double max_abs = 0.0;
        int nnz = 0;
        std::vector<std::pair<int, double>> support_terms;
        support_terms.reserve(static_cast<std::size_t>(lp.tableau.cols()));
        for (int col = 0; col < lp.tableau.cols(); ++col) {
            if (col == basic_col)
                continue;
            const auto mapped_index =
                parse_internal_label_index_impl_(lp.internal_column_labels[col]);
            if (!mapped_index.has_value() || *mapped_index < 0 ||
                *mapped_index >= static_cast<int>(problem.variable_types.size())) {
                continue;
            }

            const double value = lp.tableau(row, col);
            const double abs_value = std::abs(value);
            if (abs_value <= 1e-10)
                continue;

            support_terms.emplace_back(*mapped_index, value);
            norm_sq += abs_value * abs_value;
            min_abs = std::min(min_abs, abs_value);
            max_abs = std::max(max_abs, abs_value);
            ++nnz;
        }

        if (nnz <= 1 || norm_sq <= 1e-16)
            continue;
        if (max_abs / std::max(min_abs, 1e-16) > 1e6)
            continue;

        const double score = frac * (1.0 - frac) / norm_sq;
        std::sort(support_terms.begin(), support_terms.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
        std::vector<int> support_indices;
        std::vector<double> support_values;
        support_indices.reserve(support_terms.size());
        support_values.reserve(support_terms.size());
        for (const auto& [index, value] : support_terms) {
            if (!support_indices.empty() && support_indices.back() == index) {
                support_values.back() += value;
            } else {
                support_indices.push_back(index);
                support_values.push_back(value);
            }
        }
        double merged_norm_sq = 0.0;
        for (double value : support_values)
            merged_norm_sq += value * value;
        if (merged_norm_sq <= 1e-16)
            continue;
        candidates.push_back(Candidate{row, score, frac, std::sqrt(merged_norm_sq),
                                       std::move(support_indices), std::move(support_values)});
    }

    std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
        if (std::abs(lhs.score - rhs.score) > 1e-12)
            return lhs.score > rhs.score;
        return lhs.fractionality > rhs.fractionality;
    });

    const int max_rows = std::min<int>(static_cast<int>(candidates.size()),
                                       std::max(8, 6 * options.max_cuts_added_per_round));
    std::vector<int> rows;
    rows.reserve(std::max(0, max_rows));
    std::vector<const Candidate*> accepted;
    accepted.reserve(std::max(0, max_rows));
    const double max_parallelism = 0.92;
    for (const Candidate& candidate : candidates) {
        bool too_parallel = false;
        for (const Candidate* accepted_candidate : accepted) {
            if (sparse_parallelism(candidate, *accepted_candidate) > max_parallelism) {
                too_parallel = true;
                break;
            }
        }
        if (too_parallel)
            continue;
        rows.push_back(candidate.row);
        accepted.push_back(&candidate);
        if (static_cast<int>(rows.size()) >= max_rows)
            break;
    }

    for (const Candidate& candidate : candidates) {
        if (static_cast<int>(rows.size()) >= max_rows)
            break;
        if (std::find(rows.begin(), rows.end(), candidate.row) != rows.end())
            continue;
        rows.push_back(candidate.row);
    }
    return rows;
}

std::vector<int> select_mir_rows_(const Problem& problem, const RelaxationSolution& relaxation,
                                  const Options& options) {
    struct Candidate {
        int row = -1;
        double score = 0.0;
    };

    std::vector<Candidate> candidates;
    if (relaxation.primal.size() != problem.lower_bounds.size())
        return {};

    for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
         ++row_index) {
        const SparseLinearConstraint& row = problem.base_constraints[row_index];
        if (row.indices.size() < 2)
            continue;

        double lhs = 0.0;
        int fractional_support = 0;
        int integer_support = 0;
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            const int col = row.indices[k];
            if (col < 0 || col >= static_cast<int>(relaxation.primal.size()))
                continue;
            const double value = relaxation.primal(col);
            lhs += row.values[k] * value;
            if (value > options.integrality_tol && value < 1.0 - options.integrality_tol)
                ++fractional_support;
            if (problem.variable_types[col] != VariableType::Continuous)
                ++integer_support;
        }

        double violation = 0.0;
        if (row.sense == LinearConstraintSense::LessEqual) {
            violation = lhs - row.rhs;
        } else if (row.sense == LinearConstraintSense::GreaterEqual) {
            violation = row.rhs - lhs;
        } else {
            violation = std::abs(lhs - row.rhs);
        }
        if (violation <= std::max(options.min_cut_violation, 1e-9))
            continue;

        const double density_penalty = 1.0 + 0.1 * static_cast<double>(row.indices.size());
        const double score = violation *
                             (1.0 + 0.05 * fractional_support + 0.04 * integer_support) /
                             density_penalty;
        candidates.push_back(Candidate{row_index, score});
    }

    if (candidates.empty())
        return {};

    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) { return lhs.score > rhs.score; });

    const int row_limit = std::max(16, 8 * options.max_cuts_added_per_round);
    if (static_cast<int>(candidates.size()) > row_limit)
        candidates.resize(static_cast<std::size_t>(row_limit));

    std::vector<int> rows;
    rows.reserve(candidates.size());
    for (const Candidate& candidate : candidates)
        rows.push_back(candidate.row);
    return rows;
}

CanonicalMirRow build_canonical_mir_row_from_leq_(const Problem& problem,
                                                  const RelaxationSolution& relaxation,
                                                  const std::vector<int>& indices,
                                                  const std::vector<double>& values, double rhs) {
    CanonicalMirRow row;
    row.valid = false;
    row.rhs = rhs;

    if (relaxation.primal.size() != problem.lower_bounds.size() ||
        problem.upper_bounds.size() != problem.lower_bounds.size()) {
        return row;
    }

    for (int k = 0; k < static_cast<int>(indices.size()) && k < static_cast<int>(values.size());
         ++k) {
        const int index = indices[k];
        const double coeff = values[k];
        if (index < 0 || index >= static_cast<int>(problem.variable_types.size()) ||
            index >= relaxation.primal.size() || std::abs(coeff) <= 1e-12) {
            continue;
        }

        const double lower = problem.lower_bounds(index);
        const double upper = problem.upper_bounds(index);
        const double primal = relaxation.primal(index);
        if (!std::isfinite(primal))
            return row;

        if (std::isfinite(lower) && std::isfinite(upper) && std::abs(upper - lower) <= 1e-8) {
            row.rhs -= coeff * lower;
            continue;
        }

        if (problem.variable_types[index] == VariableType::Continuous) {
            MirContinuousTerm term;
            term.variable = index;
            if (coeff > 0.0) {
                if (!std::isfinite(upper))
                    return row;
                term.coeff = -coeff;
                term.substitution = MirSubstitution::Upper;
                term.shift = upper;
                term.lp_value = upper - primal;
                row.rhs -= coeff * upper;
            } else {
                if (!std::isfinite(lower))
                    return row;
                term.coeff = coeff;
                term.substitution = MirSubstitution::Lower;
                term.shift = lower;
                term.lp_value = primal - lower;
                row.rhs -= coeff * lower;
            }
            if (!std::isfinite(term.lp_value) || term.lp_value < -1e-8)
                return row;
            term.lp_value = std::max(0.0, term.lp_value);
            row.continuous.push_back(term);
            continue;
        }

        MirIntegerTerm term;
        term.variable = index;
        if (coeff > 0.0) {
            if (!std::isfinite(lower))
                return row;
            term.coeff = coeff;
            term.substitution = MirSubstitution::Lower;
            term.shift = lower;
            term.lp_value = primal - lower;
            row.rhs -= coeff * lower;
            if (std::isfinite(upper))
                term.upper = upper - lower;
        } else {
            if (!std::isfinite(upper))
                return row;
            term.coeff = -coeff;
            term.substitution = MirSubstitution::Upper;
            term.shift = upper;
            term.lp_value = upper - primal;
            row.rhs -= coeff * upper;
            if (std::isfinite(lower))
                term.upper = upper - lower;
        }
        if (!std::isfinite(term.lp_value) || term.lp_value < -1e-8)
            return row;
        term.lp_value = std::max(0.0, term.lp_value);
        if (std::isfinite(term.upper) && term.upper < -1e-8)
            return row;
        if (std::isfinite(term.upper))
            term.upper = std::max(0.0, term.upper);
        row.integers.push_back(term);
    }

    row.valid = std::isfinite(row.rhs) && !row.integers.empty();
    return row;
}

double mir_function_g_(double d, double f) {
    const double base = std::floor(d);
    const double delta = d - base - f;
    if (delta > 1e-10) {
        return base + delta / std::max(1e-10, 1.0 - f);
    }
    return base;
}

void map_transformed_mir_term_(Cut* cut, double coeff, int variable, MirSubstitution substitution,
                               double shift) {
    if (cut == nullptr || variable < 0 || !std::isfinite(coeff) || std::abs(coeff) <= 1e-12) {
        return;
    }
    cut->indices.push_back(variable);
    if (substitution == MirSubstitution::Lower) {
        cut->values.push_back(coeff);
        cut->rhs += coeff * shift;
    } else {
        cut->values.push_back(-coeff);
        cut->rhs -= coeff * shift;
    }
}

std::optional<Cut> build_mir_cut_from_canonical_row_(const Problem& problem,
                                                     const RelaxationSolution& relaxation,
                                                     const Options& options,
                                                     const CanonicalMirRow& row) {
    if (!row.valid || row.integers.empty() || !std::isfinite(row.rhs)) {
        return std::nullopt;
    }

    std::vector<double> delta_candidates;
    delta_candidates.reserve(row.integers.size() * 8);
    const double away = gmi_candidate_away_(options);
    for (const MirIntegerTerm& term : row.integers) {
        if (term.coeff <= away)
            continue;
        delta_candidates.push_back(term.coeff);
        delta_candidates.push_back(0.5 * term.coeff);
        delta_candidates.push_back((2.0 / 3.0) * term.coeff);
        delta_candidates.push_back((3.0 / 4.0) * term.coeff);
        delta_candidates.push_back(0.25 * term.coeff);
        delta_candidates.push_back((1.0 / 3.0) * term.coeff);
        delta_candidates.push_back(0.125 * term.coeff);
        if (std::isfinite(term.upper) && term.upper > away) {
            delta_candidates.push_back(term.coeff / std::max(1.0, std::floor(term.upper)));
        }
    }
    std::sort(delta_candidates.begin(), delta_candidates.end());
    delta_candidates.erase(std::unique(delta_candidates.begin(), delta_candidates.end(),
                                       [&](double lhs, double rhs_value) {
                                           return std::abs(lhs - rhs_value) <= 1e-9;
                                       }),
                           delta_candidates.end());

    std::optional<Cut> best_cut;
    double best_score = 0.0;

    struct CSetMode {
        double threshold_ratio = 0.5;
        bool prefer_upper = true;
    };
    const std::array<CSetMode, 5> cset_modes = {{
        {0.0, false},
        {0.33, true},
        {0.50, true},
        {0.67, true},
        {0.25, false},
    }};

    for (const CSetMode& mode : cset_modes) {
        std::vector<char> in_c(row.integers.size(), 0);
        double numerator_beta = row.rhs;
        for (int i = 0; i < static_cast<int>(row.integers.size()); ++i) {
            const MirIntegerTerm& term = row.integers[i];
            if (!std::isfinite(term.upper) || term.upper <= 1e-8 || mode.threshold_ratio <= 0.0)
                continue;

            const bool choose_upper =
                term.lp_value >= mode.threshold_ratio * term.upper + options.integrality_tol;
            const bool choose_lower = term.lp_value <= (1.0 - mode.threshold_ratio) * term.upper -
                                                           options.integrality_tol;
            const bool select = mode.prefer_upper ? choose_upper : choose_lower;
            if (!select)
                continue;
            in_c[static_cast<std::size_t>(i)] = 1;
            numerator_beta -= term.coeff * term.upper;
        }

        for (double delta : delta_candidates) {
            if (!(delta > away))
                continue;
            const double beta = numerator_beta / delta;
            const double f = fractional_part(beta);
            if (f <= away || f >= 1.0 - away)
                continue;

            Cut cut;
            cut.cut_type = "MIR";
            cut.sense = LinearConstraintSense::LessEqual;
            cut.rhs = std::floor(beta);

            bool valid = true;
            for (int i = 0; i < static_cast<int>(row.integers.size()); ++i) {
                const MirIntegerTerm& term = row.integers[i];
                double coeff = 0.0;
                if (in_c[static_cast<std::size_t>(i)]) {
                    if (!std::isfinite(term.upper)) {
                        valid = false;
                        break;
                    }
                    const double g = mir_function_g_(-term.coeff / delta, f);
                    coeff = -g;
                    cut.rhs -= g * term.upper;
                } else {
                    coeff = mir_function_g_(term.coeff / delta, f);
                }
                map_transformed_mir_term_(&cut, coeff, term.variable, term.substitution,
                                          term.shift);
            }
            if (!valid)
                continue;

            const double s_coeff = 1.0 / (delta * (1.0 - f));
            for (const MirContinuousTerm& term : row.continuous) {
                map_transformed_mir_term_(&cut, s_coeff * term.coeff, term.variable,
                                          term.substitution, term.shift);
            }

            if (!postprocess_mir_cut_(problem, options, &cut) || cut.indices.empty()) {
                continue;
            }
            const double violation = cut_violation(cut, relaxation.primal);
            if (violation <= options.min_cut_violation)
                continue;

            cut.strength = density_adjusted_efficacy_(
                violation, cut_norm_(cut), static_cast<int>(cut.indices.size()),
                fractional_focus_(cut, relaxation.primal), 1.0);
            if (cut.strength <= best_score)
                continue;
            best_score = cut.strength;
            best_cut = std::move(cut);
        }
    }

    return best_cut;
}

std::optional<MirRowData> aggregate_rows_for_mir_(const MirRowData& lhs, const MirRowData& rhs,
                                                  int pivot_index) {
    auto coefficient_of = [&](const MirRowData& row) {
        for (int k = 0;
             k < static_cast<int>(row.indices.size()) && k < static_cast<int>(row.values.size());
             ++k) {
            if (row.indices[k] == pivot_index)
                return row.values[k];
        }
        return 0.0;
    };

    const double lhs_coeff = coefficient_of(lhs);
    const double rhs_coeff = coefficient_of(rhs);
    if (std::abs(lhs_coeff) <= 1e-12 || std::abs(rhs_coeff) <= 1e-12 ||
        lhs_coeff * rhs_coeff >= 0.0) {
        return std::nullopt;
    }

    const double lhs_scale = std::abs(rhs_coeff);
    const double rhs_scale = std::abs(lhs_coeff);
    MirRowData aggregated;
    aggregated.rhs = lhs_scale * lhs.rhs + rhs_scale * rhs.rhs;

    std::vector<std::pair<int, double>> merged;
    merged.reserve(lhs.indices.size() + rhs.indices.size());
    for (int k = 0;
         k < static_cast<int>(lhs.indices.size()) && k < static_cast<int>(lhs.values.size()); ++k) {
        merged.emplace_back(lhs.indices[k], lhs_scale * lhs.values[k]);
    }
    for (int k = 0;
         k < static_cast<int>(rhs.indices.size()) && k < static_cast<int>(rhs.values.size()); ++k) {
        merged.emplace_back(rhs.indices[k], rhs_scale * rhs.values[k]);
    }

    std::sort(merged.begin(), merged.end());
    for (const auto& [index, value] : merged) {
        if (!aggregated.indices.empty() && aggregated.indices.back() == index) {
            aggregated.values.back() += value;
        } else {
            aggregated.indices.push_back(index);
            aggregated.values.push_back(value);
        }
    }

    std::vector<int> final_indices;
    std::vector<double> final_values;
    final_indices.reserve(aggregated.indices.size());
    final_values.reserve(aggregated.values.size());
    double min_abs = std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    for (int k = 0; k < static_cast<int>(aggregated.indices.size()) &&
                    k < static_cast<int>(aggregated.values.size());
         ++k) {
        const double value = aggregated.values[k];
        if (std::abs(value) <= 1e-10)
            continue;
        if (aggregated.indices[k] == pivot_index)
            continue;
        final_indices.push_back(aggregated.indices[k]);
        final_values.push_back(value);
        min_abs = std::min(min_abs, std::abs(value));
        max_abs = std::max(max_abs, std::abs(value));
    }

    if (final_indices.size() < 2 || !std::isfinite(aggregated.rhs) ||
        !(min_abs < std::numeric_limits<double>::infinity()) || max_abs <= 1e-12) {
        return std::nullopt;
    }
    if (max_abs / std::max(min_abs, 1e-12) > 1e8)
        return std::nullopt;

    aggregated.indices = std::move(final_indices);
    aggregated.values = std::move(final_values);
    return aggregated;
}

std::optional<int> choose_mir_aggregation_pivot_(const MirRowData& lhs, const MirRowData& rhs,
                                                 const Eigen::VectorXd& primal) {
    std::unordered_map<int, double> rhs_coeffs;
    rhs_coeffs.reserve(rhs.indices.size());
    for (int k = 0;
         k < static_cast<int>(rhs.indices.size()) && k < static_cast<int>(rhs.values.size()); ++k) {
        if (std::abs(rhs.values[k]) <= 1e-12)
            continue;
        rhs_coeffs.emplace(rhs.indices[k], rhs.values[k]);
    }

    int best_pivot = -1;
    double best_score = -1.0;
    for (int k = 0;
         k < static_cast<int>(lhs.indices.size()) && k < static_cast<int>(lhs.values.size()); ++k) {
        const int index = lhs.indices[k];
        const double lhs_coeff = lhs.values[k];
        if (std::abs(lhs_coeff) <= 1e-12)
            continue;
        const auto rhs_it = rhs_coeffs.find(index);
        if (rhs_it == rhs_coeffs.end())
            continue;
        const double rhs_coeff = rhs_it->second;
        if (lhs_coeff * rhs_coeff >= 0.0)
            continue;

        const double abs_lhs = std::abs(lhs_coeff);
        const double abs_rhs = std::abs(rhs_coeff);
        const double balance = std::min(abs_lhs, abs_rhs) / std::max(abs_lhs, abs_rhs);
        double score = std::abs(lhs_coeff * rhs_coeff) * (0.6 + 0.4 * balance);
        if (index >= 0 && index < primal.size()) {
            const double value = primal(index);
            const double frac_dist = std::abs(value - std::round(value));
            score *= 1.0 + std::min(1.5, 2.0 * frac_dist);
        }
        if (score > best_score) {
            best_score = score;
            best_pivot = index;
        }
    }

    if (best_pivot < 0)
        return std::nullopt;
    return best_pivot;
}

} // namespace

double fractional_part(double value) {
    const double frac = value - std::floor(value);
    if (frac <= 1e-10)
        return 0.0;
    if (frac >= 1.0 - 1e-10)
        return 1.0;
    return frac;
}

std::vector<Cut> generate_gomory_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                      const Options& options) {
    if (!options.use_gomory_cuts || !relaxation.lp_solution.has_value())
        return {};

    const LPSolution& lp = *relaxation.lp_solution;
    if (!lp.has_internal_tableau || lp.tableau.rows() == 0 ||
        lp.tableau_rhs.size() != lp.tableau.rows()) {
        return {};
    }
    if (lp.internal_column_labels.size() != static_cast<std::size_t>(lp.tableau.cols()) ||
        lp.basis_internal.size() != static_cast<std::size_t>(lp.tableau.rows())) {
        return {};
    }

    const std::vector<int> candidate_rows = select_gmi_rows_(problem, relaxation, options, lp);
    std::vector<Cut> row_cuts;
    row_cuts.reserve(candidate_rows.empty() ? static_cast<std::size_t>(lp.tableau.rows())
                                            : candidate_rows.size());

    const auto emit_row = [&](int row) {
        std::optional<Cut> cut = build_gmi_cut_from_row_(problem, relaxation, options, lp, row);
        if (cut.has_value())
            row_cuts.push_back(std::move(*cut));
    };

    if (!candidate_rows.empty()) {
        for (int row : candidate_rows)
            emit_row(row);
    } else {
        for (int row = 0; row < lp.tableau.rows(); ++row)
            emit_row(row);
    }

    if (problem.variable_types.size() < 16 || row_cuts.size() < 4)
        return row_cuts;

    return strengthen_gmi_candidates_(problem, std::move(row_cuts), relaxation, options);
}

std::vector<Cut> generate_mir_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                   const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_mir_cuts || relaxation.primal.size() != problem.lower_bounds.size()) {
        return cuts;
    }

    std::vector<MirRowData> rows;
    rows.reserve(problem.base_constraints.size() * 2);
    const std::vector<int> selected_row_indices = select_mir_rows_(problem, relaxation, options);
    std::vector<int> row_indices = selected_row_indices;
    if (row_indices.empty()) {
        row_indices.resize(static_cast<int>(problem.base_constraints.size()));
        std::iota(row_indices.begin(), row_indices.end(), 0);
    }
    for (int row_index : row_indices) {
        const SparseLinearConstraint& row = problem.base_constraints[row_index];
        if (row.sense == LinearConstraintSense::LessEqual) {
            rows.push_back(MirRowData{row.indices, row.values, row.rhs});
        } else if (row.sense == LinearConstraintSense::GreaterEqual) {
            std::vector<double> negated = row.values;
            for (double& value : negated)
                value = -value;
            rows.push_back(MirRowData{row.indices, std::move(negated), -row.rhs});
        } else {
            rows.push_back(MirRowData{row.indices, row.values, row.rhs});
            std::vector<double> negated = row.values;
            for (double& value : negated)
                value = -value;
            rows.push_back(MirRowData{row.indices, std::move(negated), -row.rhs});
        }
    }

    std::unordered_set<CutSignature, CutSignatureHash> signatures;
    auto try_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                       double rhs, const char* cut_type = "MIR") {
        const CanonicalMirRow row =
            build_canonical_mir_row_from_leq_(problem, relaxation, indices, values, rhs);
        std::optional<Cut> cut =
            build_mir_cut_from_canonical_row_(problem, relaxation, options, row);
        if (!cut.has_value())
            return;
        cut->cut_type = cut_type;
        const CutSignature signature = cut_signature(*cut);
        if (!signatures.insert(signature).second)
            return;
        cuts.push_back(std::move(*cut));
    };

    for (const MirRowData& row : rows) {
        try_row(row.indices, row.values, row.rhs, "MIR");
    }

    const int max_pair_aggregations = std::max(4, 2 * options.max_cuts_added_per_round);
    int pair_aggregations = 0;
    std::vector<MirRowData> aggregated_rows;
    aggregated_rows.reserve(max_pair_aggregations);
    for (int i = 0; i < static_cast<int>(rows.size()) && pair_aggregations < max_pair_aggregations;
         ++i) {
        for (int j = i + 1;
             j < static_cast<int>(rows.size()) && pair_aggregations < max_pair_aggregations; ++j) {
            const std::optional<int> best_pivot =
                choose_mir_aggregation_pivot_(rows[i], rows[j], relaxation.primal);
            if (!best_pivot.has_value())
                continue;
            std::optional<MirRowData> aggregated =
                aggregate_rows_for_mir_(rows[i], rows[j], *best_pivot);
            if (!aggregated.has_value())
                continue;
            try_row(aggregated->indices, aggregated->values, aggregated->rhs, "CMIR");
            aggregated_rows.push_back(*aggregated);
            ++pair_aggregations;
        }
    }

    const int max_multi_aggregations = std::max(2, options.max_cuts_added_per_round);
    int multi_aggregations = 0;
    for (int i = 0; i < static_cast<int>(aggregated_rows.size()) &&
                    multi_aggregations < max_multi_aggregations;
         ++i) {
        for (int j = 0;
             j < static_cast<int>(rows.size()) && multi_aggregations < max_multi_aggregations;
             ++j) {
            const std::optional<int> best_pivot =
                choose_mir_aggregation_pivot_(aggregated_rows[i], rows[j], relaxation.primal);
            if (!best_pivot.has_value())
                continue;
            std::optional<MirRowData> aggregated =
                aggregate_rows_for_mir_(aggregated_rows[i], rows[j], *best_pivot);
            if (!aggregated.has_value())
                continue;
            try_row(aggregated->indices, aggregated->values, aggregated->rhs, "CMIR-3");
            ++multi_aggregations;
        }
    }

    if (cuts.empty())
        return cuts;

    std::sort(cuts.begin(), cuts.end(), [](const Cut& lhs, const Cut& rhs) {
        if (std::abs(lhs.strength - rhs.strength) > 1e-12)
            return lhs.strength > rhs.strength;
        return lhs.indices.size() < rhs.indices.size();
    });

    const int limit = std::max(options.max_cuts_added_per_round * 2, options.max_cuts_per_type);
    if (limit > 0 && static_cast<int>(cuts.size()) > limit) {
        cuts.resize(static_cast<std::size_t>(limit));
    }
    return cuts;
}

std::vector<Cut> generate_cover_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                     const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_cover_cuts)
        return cuts;

    std::unordered_set<CutSignature, CutSignatureHash> signatures;
    const bool large_binary_dense =
        problem.variable_types.size() >= 256 &&
        std::all_of(problem.variable_types.begin(), problem.variable_types.end(),
                    [](VariableType type) { return type == VariableType::Binary; });
    std::vector<int> selected_rows;
    struct RankedRow {
        int index = -1;
        double score = -std::numeric_limits<double>::infinity();
    };
    std::vector<RankedRow> ranked_rows;
    if (relaxation.primal.size() == problem.lower_bounds.size()) {
        ranked_rows.reserve(problem.base_constraints.size());
        const double min_cover_violation = std::max(options.min_cut_violation, 1e-9);
        for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
             ++row_index) {
            const SparseLinearConstraint& row = problem.base_constraints[row_index];
            if (row.sense != LinearConstraintSense::LessEqual || row.indices.size() < 2) {
                continue;
            }

            double lhs = 0.0;
            int fractional_support = 0;
            for (int k = 0; k < static_cast<int>(row.indices.size()) &&
                            k < static_cast<int>(row.values.size());
                 ++k) {
                const int col = row.indices[k];
                if (col < 0 || col >= relaxation.primal.size()) {
                    continue;
                }
                lhs += row.values[k] * relaxation.primal(col);
                const double value = relaxation.primal(col);
                if (value > options.integrality_tol && value < 1.0 - options.integrality_tol) {
                    ++fractional_support;
                }
            }
            const double violation = std::max(0.0, lhs - row.rhs);
            if (violation <= 0.25 * min_cover_violation && fractional_support < 3) {
                continue;
            }
            ranked_rows.push_back({row_index, violation + 0.02 * fractional_support});
        }
        std::sort(ranked_rows.begin(), ranked_rows.end(),
                  [](const RankedRow& lhs, const RankedRow& rhs) {
                      if (std::abs(lhs.score - rhs.score) > 1e-12) {
                          return lhs.score > rhs.score;
                      }
                      return lhs.index < rhs.index;
                  });
        const int row_limit = std::max(16, 4 * options.max_cuts_added_per_round);
        if (static_cast<int>(ranked_rows.size()) > row_limit) {
            ranked_rows.resize(static_cast<std::size_t>(row_limit));
        }
    }

    if (!ranked_rows.empty()) {
        selected_rows.reserve(ranked_rows.size());
        for (const RankedRow& ranked : ranked_rows) {
            selected_rows.push_back(ranked.index);
        }
    }

    auto handle_row_index = [&](int row_index) {
        const SparseLinearConstraint& row = problem.base_constraints[row_index];
        if (row.sense != LinearConstraintSense::LessEqual)
            return;

        // Try canonical pure binary knapsack lifting first
        CanonicalKnapsack kn = build_canonical_binary_knapsack_(problem, relaxation, row);
        if (kn.valid && kn.variables.size() >= 2 && !std::isfinite(kn.rhs) == false &&
            kn.rhs >= 0.0) {
            const std::vector<CoverLiteralTerm> canonical_terms = canonical_binary_cover_terms_(kn);
            std::vector<std::vector<int>> candidate_literal_covers;
            auto queue_literal_cover = [&](std::vector<int> cover_literals) {
                if (cover_literals.size() < 2) {
                    return;
                }
                std::sort(cover_literals.begin(), cover_literals.end());
                cover_literals.erase(std::unique(cover_literals.begin(), cover_literals.end()),
                                     cover_literals.end());
                if (cover_literals.size() < 2) {
                    return;
                }
                if (std::find(candidate_literal_covers.begin(), candidate_literal_covers.end(),
                              cover_literals) == candidate_literal_covers.end()) {
                    candidate_literal_covers.push_back(std::move(cover_literals));
                }
            };

            if (const std::optional<BinaryCoverPartition> partition =
                    find_lp_violated_minimal_cover_(kn)) {
                queue_literal_cover(
                    canonical_binary_cover_literals_(kn, partition->cover_positions));
            }

            queue_literal_cover(canonical_binary_cover_literals_(
                kn, greedy_binary_cover_positions_(kn, BinaryCoverOrdering::ActivityFirst)));
            queue_literal_cover(canonical_binary_cover_literals_(
                kn, greedy_binary_cover_positions_(kn, BinaryCoverOrdering::CoefficientFirst)));
            queue_literal_cover(canonical_binary_cover_literals_(
                kn, greedy_binary_cover_positions_(kn, BinaryCoverOrdering::RatioFirst)));

            for (const std::vector<int>& cover_literals : candidate_literal_covers) {
                maybe_add_cover_cut_(problem, relaxation, options, cover_literals, "Cover",
                                     &signatures, &cuts);
            }
            if (!candidate_literal_covers.empty()) {
                return;
            }
        }

        // Fallback: old literal-based cover for mixed or non-binary
        MixedCoverContext context = build_mixed_cover_context_(problem, relaxation, row);
        if (!context.valid || context.literals.size() < 2 || !std::isfinite(context.rhs) ||
            context.rhs < 0.0) {
            return;
        }
        const std::vector<int> activity_cover =
            greedy_cover_literals_(context.literals, context.rhs, true);
        const std::vector<int> coeff_cover =
            greedy_cover_literals_(context.literals, context.rhs, false);
        const std::vector<int> ratio_cover =
            greedy_cover_literals_(context.literals, context.rhs, false, true);
        if (activity_cover.size() < 2 && coeff_cover.size() < 2 && ratio_cover.size() < 2)
            return;
        const std::string base_type =
            context.has_nonbinary_component ? "MixedBinaryCover" : "Cover";
        maybe_add_cover_cut_(problem, relaxation, options, activity_cover, base_type, &signatures,
                             &cuts);
        maybe_add_cover_cut_(problem, relaxation, options, coeff_cover, base_type, &signatures,
                             &cuts);
        maybe_add_cover_cut_(problem, relaxation, options, ratio_cover, base_type, &signatures,
                             &cuts);
        const std::vector<int> lifted_activity =
            extend_cover_literals_(context.literals, activity_cover, context.rhs);
        const std::vector<int> lifted_coeff =
            extend_cover_literals_(context.literals, coeff_cover, context.rhs);
        maybe_add_cover_cut_(problem, relaxation, options, lifted_activity,
                             context.has_nonbinary_component ? "LiftedMixedBinaryCover"
                                                             : "LiftedCoverLite",
                             &signatures, &cuts);
        maybe_add_cover_cut_(problem, relaxation, options, lifted_coeff,
                             context.has_nonbinary_component ? "LiftedMixedBinaryCover"
                                                             : "LiftedCoverLite",
                             &signatures, &cuts);
    };

    if (!selected_rows.empty()) {
        for (const int row_index : selected_rows) {
            handle_row_index(row_index);
        }
    } else {
        for (int row_index = 0; row_index < static_cast<int>(problem.base_constraints.size());
             ++row_index) {
            handle_row_index(row_index);
        }
    }

    return cuts;
}

std::vector<Cut> generate_implied_bound_cuts(const Problem& problem,
                                             const RelaxationSolution& relaxation,
                                             const Options& options) {
    std::vector<Cut> cuts;
    if (!options.use_implied_bound_cuts ||
        relaxation.primal.size() != problem.lower_bounds.size() ||
        problem.upper_bounds.size() != problem.lower_bounds.size()) {
        return cuts;
    }

    auto handle_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                          double rhs) {
        append_implied_bound_cuts_from_leq_(problem, indices, values, rhs, relaxation, options,
                                            &cuts);
    };

    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense == LinearConstraintSense::LessEqual) {
            handle_row(row.indices, row.values, row.rhs);
        } else if (row.sense == LinearConstraintSense::GreaterEqual) {
            std::vector<double> negated = row.values;
            for (double& value : negated)
                value = -value;
            handle_row(row.indices, negated, -row.rhs);
        } else {
            handle_row(row.indices, row.values, row.rhs);
            std::vector<double> negated = row.values;
            for (double& value : negated)
                value = -value;
            handle_row(row.indices, negated, -row.rhs);
        }
    }

    return cuts;
}

std::vector<Cut> generate_clique_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                                      const Options& options,
                                      const ImplicationStore* learned_implications,
                                      const std::vector<Cut>* structural_cuts) {
    const bool large_binary_dense =
        problem.variable_types.size() >= 256 &&
        std::all_of(problem.variable_types.begin(), problem.variable_types.end(),
                    [](VariableType type) { return type == VariableType::Binary; });
    auto generate_graph_clique_cuts = [&]() {
        std::vector<Cut> cuts;
        if (!options.use_clique_cuts || !options.use_graph_clique_cuts ||
            relaxation.primal.size() != problem.lower_bounds.size())
            return cuts;

        ConflictGraph graph(problem);
        if (structural_cuts != nullptr) {
            for (const Cut& cut : *structural_cuts)
                graph.add_cut_cliques(cut);
        }
        add_implication_conflicts_(&graph, problem, learned_implications, options);
        std::unordered_set<CutSignature, CutSignatureHash> signatures;
        const int max_found =
            std::max(options.max_cuts_added_per_round * 6, options.max_cuts_per_type);

        struct StoredCliqueCandidate {
            int index = -1;
            double activity = 0.0;
        };
        std::vector<StoredCliqueCandidate> stored_candidates;
        stored_candidates.reserve(graph.cliques().size());
        const int worker_count = std::max(
            1,
            static_cast<int>(std::floor(std::thread::hardware_concurrency() *
                                       options.cut_max_parallelism)));
        if (worker_count <= 1) {
            for (int i = 0; i < static_cast<int>(graph.cliques().size()); ++i) {
                const auto& clique_literals = graph.cliques()[i];
                if (clique_literals.size() < 2)
                    continue;
                const double activity = clique_literal_activity_(clique_literals, relaxation.primal);
                const double seed_threshold =
                    clique_literals.size() >= 3 ? 1.0 + 0.5 * options.min_cut_violation
                                                : 1.0 - std::max(0.05, 8.0 * options.min_cut_violation);
                if (activity <= seed_threshold)
                    continue;
                stored_candidates.push_back(StoredCliqueCandidate{i, activity});
            }
        } else {
            std::mutex stored_mutex;
            ParallelDispatcher dispatcher(worker_count);
            dispatcher.run(static_cast<int>(graph.cliques().size()), [&](int i) {
                const auto& clique_literals = graph.cliques()[i];
                if (clique_literals.size() < 2)
                    return;
                const double activity = clique_literal_activity_(clique_literals, relaxation.primal);
                const double seed_threshold =
                    clique_literals.size() >= 3 ? 1.0 + 0.5 * options.min_cut_violation
                                                : 1.0 - std::max(0.05, 8.0 * options.min_cut_violation);
                if (activity <= seed_threshold)
                    return;
                const StoredCliqueCandidate candidate{i, activity};
                std::lock_guard<std::mutex> lock(stored_mutex);
                stored_candidates.push_back(candidate);
            });
        }
        std::sort(stored_candidates.begin(), stored_candidates.end(),
                  [&](const StoredCliqueCandidate& lhs, const StoredCliqueCandidate& rhs) {
                      if (std::abs(lhs.activity - rhs.activity) > 1e-12)
                          return lhs.activity > rhs.activity;
                      return graph.cliques()[lhs.index].size() > graph.cliques()[rhs.index].size();
                  });
        if (stored_candidates.size() > static_cast<std::size_t>(max_found * 6))
            stored_candidates.resize(max_found * 6);
        for (const StoredCliqueCandidate& candidate : stored_candidates) {
            append_clique_cut_candidate_(problem, relaxation, options, &graph,
                                         graph.cliques()[candidate.index], "Clique", &signatures,
                                         &cuts);
            if (static_cast<int>(cuts.size()) >= max_found)
                return cuts;
        }

        std::vector<std::vector<int>> implication_seeds = build_implication_seed_cliques_(
            problem, graph, learned_implications, relaxation.primal, options, max_found * 2);
        for (std::vector<int>& clique_literals : implication_seeds) {
            append_clique_cut_candidate_(problem, relaxation, options, &graph,
                                         std::move(clique_literals), "ImplicationClique",
                                         &signatures, &cuts);
            if (static_cast<int>(cuts.size()) >= max_found)
                return cuts;
        }

        if (large_binary_dense) {
            return cuts;
        }

        const std::vector<int> vertices =
            graph.fractional_literals(relaxation.primal, options.integrality_tol);
        if (vertices.size() < 2)
            return cuts;

        std::vector<std::vector<int>> weighted_seeds = build_weighted_seed_cliques_(
            graph, relaxation.primal, options, vertices, max_found * 2);
        for (std::vector<int>& clique_literals : weighted_seeds) {
            append_clique_cut_candidate_(problem, relaxation, options, &graph,
                                         std::move(clique_literals), "WeightedClique", &signatures,
                                         &cuts);
            if (static_cast<int>(cuts.size()) >= max_found)
                return cuts;
        }

        std::vector<std::vector<int>> partition_cliques =
            build_partition_cliques_(graph, relaxation.primal, options, vertices, max_found * 2);
        for (std::vector<int>& clique_literals : partition_cliques) {
            append_clique_cut_candidate_(problem, relaxation, options, &graph,
                                         std::move(clique_literals), "CliquePartition", &signatures,
                                         &cuts);
            if (static_cast<int>(cuts.size()) >= max_found)
                return cuts;
        }

        std::vector<double> vertex_weights(vertices.size(), 0.0);
        for (int i = 0; i < static_cast<int>(vertices.size()); ++i)
            vertex_weights[i] = ConflictGraph::literal_weight(relaxation.primal, vertices[i]);

        std::vector<std::vector<char>> adjacent(vertices.size(),
                                                std::vector<char>(vertices.size(), 0));
        for (int i = 0; i < static_cast<int>(vertices.size()); ++i) {
            adjacent[i][i] = 1;
            for (int j = i + 1; j < static_cast<int>(vertices.size()); ++j) {
                if (graph.are_conflicting(vertices[i], vertices[j])) {
                    adjacent[i][j] = 1;
                    adjacent[j][i] = 1;
                }
            }
        }

        constexpr int kMaxCliqueCalls = 10000;
        int calls = 0;
        std::vector<std::vector<int>> found;

        std::function<void(std::vector<int>, std::vector<int>, std::vector<int>, double)> search;
        search = [&](std::vector<int> clique, std::vector<int> candidates,
                     std::vector<int> excluded, double clique_weight) {
            if (calls++ >= kMaxCliqueCalls || static_cast<int>(found.size()) >= max_found)
                return;

            double upper_bound = clique_weight;
            for (int v : candidates)
                upper_bound += vertex_weights[v];
            if (upper_bound <= 1.0 + options.min_cut_violation)
                return;

            if (candidates.empty() && excluded.empty()) {
                if (clique_weight > 1.0 + options.min_cut_violation)
                    found.push_back(std::move(clique));
                return;
            }

            int pivot = -1;
            int best_neighbors = -1;
            std::vector<int> pivot_pool = candidates;
            pivot_pool.insert(pivot_pool.end(), excluded.begin(), excluded.end());
            for (int u : pivot_pool) {
                int count = 0;
                for (int v : candidates)
                    count += adjacent[u][v] ? 1 : 0;
                if (count > best_neighbors) {
                    best_neighbors = count;
                    pivot = u;
                }
            }

            std::vector<int> expand = candidates;
            if (pivot >= 0) {
                expand.erase(std::remove_if(expand.begin(), expand.end(),
                                            [&](int v) { return adjacent[pivot][v] != 0; }),
                             expand.end());
            }

            for (int v : expand) {
                std::vector<int> next_clique = clique;
                next_clique.push_back(v);
                std::vector<int> next_candidates;
                std::vector<int> next_excluded;
                next_candidates.reserve(candidates.size());
                next_excluded.reserve(excluded.size());
                for (int u : candidates) {
                    if (u != v && adjacent[v][u])
                        next_candidates.push_back(u);
                }
                for (int u : excluded) {
                    if (adjacent[v][u])
                        next_excluded.push_back(u);
                }

                search(std::move(next_clique), std::move(next_candidates), std::move(next_excluded),
                       clique_weight + vertex_weights[v]);

                candidates.erase(std::remove(candidates.begin(), candidates.end(), v),
                                 candidates.end());
                excluded.push_back(v);
            }
        };

        std::vector<int> all_vertices(vertices.size(), 0);
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        std::sort(all_vertices.begin(), all_vertices.end(), [&](int lhs, int rhs) {
            if (std::abs(vertex_weights[lhs] - vertex_weights[rhs]) > 1e-12)
                return vertex_weights[lhs] > vertex_weights[rhs];
            const int lhs_degree = graph.degree(vertices[lhs]);
            const int rhs_degree = graph.degree(vertices[rhs]);
            if (lhs_degree != rhs_degree)
                return lhs_degree > rhs_degree;
            return vertices[lhs] < vertices[rhs];
        });
        search({}, all_vertices, {}, 0.0);

        for (std::vector<int>& clique_pos : found) {
            std::vector<int> clique_literals;
            clique_literals.reserve(clique_pos.size());
            for (int pos : clique_pos)
                clique_literals.push_back(vertices[pos]);
            append_clique_cut_candidate_(problem, relaxation, options, &graph,
                                         std::move(clique_literals), "Clique", &signatures, &cuts);
        }

        return cuts;
    };

    std::vector<Cut> cuts;
    if (!options.use_clique_cuts || relaxation.primal.size() != problem.lower_bounds.size())
        return cuts;

    auto handle_row = [&](const std::vector<int>& indices, const std::vector<double>& values,
                          double rhs) {
        append_clique_cuts_from_leq_(problem, indices, values, rhs, relaxation, options, &cuts);
    };

    for (const SparseLinearConstraint& row : problem.base_constraints) {
        if (row.sense == LinearConstraintSense::LessEqual) {
            handle_row(row.indices, row.values, row.rhs);
        } else if (row.sense == LinearConstraintSense::Equal) {
            handle_row(row.indices, row.values, row.rhs);
            std::vector<double> negated = row.values;
            for (double& value : negated)
                value = -value;
            handle_row(row.indices, negated, -row.rhs);
        }
    }

    std::vector<Cut> graph_cuts = generate_graph_clique_cuts();
    cuts.insert(cuts.end(), std::make_move_iterator(graph_cuts.begin()),
                std::make_move_iterator(graph_cuts.end()));
    return cuts;
}

std::vector<Cut> generate_odd_cycle_cuts(const Problem& problem,
                                         const RelaxationSolution& relaxation,
                                         const Options& options,
                                         const ImplicationStore* learned_implications,
                                         const std::vector<Cut>* structural_cuts) {
    std::vector<Cut> cuts;
    if (!options.use_odd_cycle_cuts || relaxation.primal.size() != problem.lower_bounds.size())
        return cuts;

    ConflictGraph graph =
        build_separator_conflict_graph_(problem, learned_implications, options, structural_cuts);
    const std::vector<int> vertices =
        graph.fractional_literals(relaxation.primal, options.integrality_tol);
    if (vertices.size() < 5)
        return cuts;

    std::unordered_map<int, int> position_of;
    position_of.reserve(vertices.size());
    for (int i = 0; i < static_cast<int>(vertices.size()); ++i)
        position_of.emplace(vertices[i], i);

    std::vector<int> start_positions(vertices.size(), 0);
    std::iota(start_positions.begin(), start_positions.end(), 0);
    std::sort(start_positions.begin(), start_positions.end(), [&](int lhs, int rhs) {
        const double lhs_weight = ConflictGraph::literal_weight(relaxation.primal, vertices[lhs]);
        const double rhs_weight = ConflictGraph::literal_weight(relaxation.primal, vertices[rhs]);
        if (std::abs(lhs_weight - rhs_weight) > 1e-12)
            return lhs_weight > rhs_weight;
        const int lhs_degree = graph.degree(vertices[lhs]);
        const int rhs_degree = graph.degree(vertices[rhs]);
        if (lhs_degree != rhs_degree)
            return lhs_degree > rhs_degree;
        return vertices[lhs] < vertices[rhs];
    });

    std::unordered_set<CutSignature, CutSignatureHash> signatures;
    const int max_found = std::max(options.max_cuts_added_per_round * 4, options.max_cuts_per_type);
    for (const int start_pos : start_positions) {
        const std::vector<int> cycle_literals = find_weighted_odd_cycle_(
            graph, relaxation.primal, options, vertices, start_pos, position_of);
        if (cycle_literals.size() < 5 || cycle_literals.size() % 2 == 0)
            continue;

        Cut cut = odd_cycle_cut_from_literals_(problem, cycle_literals, options, "OddCycle");
        if (cut.indices.empty())
            continue;
        const double violation = cut_violation(cut, relaxation.primal);
        if (violation <= options.min_cut_violation)
            continue;
        cut.strength = density_adjusted_efficacy_(violation, cut_norm_(cut),
                                                  static_cast<int>(cut.indices.size()),
                                                  fractional_focus_(cut, relaxation.primal), 1.0);
        const CutSignature signature = cut_signature(cut);
        if (!signatures.insert(signature).second)
            continue;
        cuts.push_back(std::move(cut));
        if (static_cast<int>(cuts.size()) >= max_found)
            break;
    }

    std::sort(cuts.begin(), cuts.end(), [](const Cut& lhs, const Cut& rhs) {
        if (std::abs(lhs.strength - rhs.strength) > 1e-12)
            return lhs.strength > rhs.strength;
        return lhs.indices.size() < rhs.indices.size();
    });
    return cuts;
}

std::vector<Cut> generate_dual_proof_cuts(const Problem& problem,
                                          const std::vector<Cut>& active_cuts,
                                          const RelaxationSolution& relaxation,
                                          const Eigen::VectorXd& node_lower_bounds,
                                          const Eigen::VectorXd& node_upper_bounds,
                                          const Options& options) {
    std::vector<Cut> cuts;
    std::optional<Cut> cut = dual_proof_detail::build_farkas_cut_(
        problem, active_cuts, relaxation, node_lower_bounds, node_upper_bounds, options);
    if (cut.has_value())
        cuts.push_back(std::move(*cut));
    return cuts;
}

std::vector<Cut> generate_dual_proof_cuts(const Problem& problem,
                                          const RelaxationSolution& relaxation,
                                          const Options& options) {
    return generate_dual_proof_cuts(problem, {}, relaxation, problem.lower_bounds,
                                    problem.upper_bounds, options);
}

static std::vector<Cut> filter_generated_cuts_(std::vector<Cut> cuts, const Problem& problem,
                                               const RelaxationSolution& relaxation,
                                               const Options& options) {
    if (cuts.empty())
        return cuts;

    const Eigen::VectorXd& primal = relaxation.primal;
    std::vector<std::pair<double, int>> scored;
    scored.reserve(cuts.size());
    for (int i = 0; i < static_cast<int>(cuts.size()); ++i) {
        const Cut& cut = cuts[i];
        if (cut.indices.empty())
            continue;
        const double violation = cut_violation(cut, primal);
        if (violation <= options.min_cut_violation)
            continue;
        const double norm = cut_norm_(cut);
        if (norm <= 1e-16)
            continue;
        const ActiveSupportStats active_stats = active_support_stats_(
            cut, primal, problem.lower_bounds, problem.upper_bounds, options.min_cut_violation);
        const double efficacy = violation / norm;
        const double active_efficacy =
            active_stats.norm > 1e-16 ? violation / active_stats.norm : efficacy;
        const double fractional_focus = fractional_focus_(cut, primal);
        const double density_adjusted_efficacy = density_adjusted_efficacy_(
            violation, norm,
            active_stats.nnz > 0 ? active_stats.nnz : static_cast<int>(cut.indices.size()),
            fractional_focus, 1.0);
        const double dynamism = cut_dynamism_(cut, primal, violation, norm);
        const double base_score = 0.40 * active_efficacy + 0.25 * efficacy +
                                  0.20 * density_adjusted_efficacy + 0.10 * dynamism +
                                  0.05 * fractional_focus;
        const std::string_view type = cut.cut_type;
        const double type_bonus =
            (type == "OddCycle" || type == "Conflict" || type == "ConflictClique" ||
             type == "Clique" || type == "ImplicationClique" || type == "WeightedClique" ||
             type == "CliquePartition")
                ? 1.08
                : 1.0;
        const double score = base_score * type_bonus;
        if (cut.indices.size() > 128 && score < std::max(0.02, options.min_cut_violation * 2.0))
            continue;
        scored.emplace_back(score, i);
    }

    if (scored.empty())
        return {};

    std::sort(scored.begin(), scored.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first > rhs.first; });

    const int cap = std::max(16, options.max_cuts_added_per_round * 6);
    std::unordered_map<std::string, int> per_type_count;
    std::vector<Cut> filtered;
    filtered.reserve(std::min<int>(static_cast<int>(scored.size()), cap));

    for (const auto& [score, index] : scored) {
        if (index < 0 || index >= static_cast<int>(cuts.size()))
            continue;
        Cut& cut = cuts[index];
        if (options.max_cuts_per_type > 0 &&
            per_type_count[cut.cut_type] >= options.max_cuts_per_type) {
            continue;
        }
        filtered.push_back(std::move(cut));
        ++per_type_count[filtered.back().cut_type];
        if (static_cast<int>(filtered.size()) >= cap)
            break;
    }
    return filtered;
}

std::vector<Cut> generate_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                               const Options& options, const ImplicationStore* learned_implications,
                               const std::vector<Cut>* structural_cuts) {
    const SeparatorContext context{
        problem, relaxation, options, learned_implications, structural_cuts,
    };

    const auto& separators = default_cut_separators_();
    std::vector<Cut> cuts;
    const std::array<CutSeparatorPhase, 5> phase_order = {
        CutSeparatorPhase::ImpliedBound, CutSeparatorPhase::Clique, CutSeparatorPhase::OddCycle,
        CutSeparatorPhase::LP,           CutSeparatorPhase::Proof,
    };

    for (CutSeparatorPhase phase : phase_order) {
        std::vector<const CutSeparator*> phase_separators;
        for (const auto& separator : separators) {
            if (separator == nullptr || separator->phase() != phase ||
                !separator->enabled(context)) {
                continue;
            }
            phase_separators.push_back(separator.get());
        }

        if (phase_separators.empty())
            continue;

        if (options.parallel_workers > 1 && phase_separators.size() > 1) {
            const int worker_count =
                std::min<int>(options.parallel_workers, phase_separators.size());
            detail::ParallelDispatcher dispatcher(worker_count);
            std::vector<std::vector<Cut>> family_cuts(phase_separators.size());
            dispatcher.run(static_cast<int>(phase_separators.size()), [&](int index) {
                family_cuts[index] = phase_separators[index]->separate(context);
            });
            for (auto& separator_cuts : family_cuts) {
                cuts.insert(cuts.end(), std::make_move_iterator(separator_cuts.begin()),
                            std::make_move_iterator(separator_cuts.end()));
            }
        } else {
            for (const CutSeparator* separator : phase_separators) {
                std::vector<Cut> family_cuts = separator->separate(context);
                cuts.insert(cuts.end(), std::make_move_iterator(family_cuts.begin()),
                            std::make_move_iterator(family_cuts.end()));
            }
        }
    }
    return filter_generated_cuts_(std::move(cuts), problem, relaxation, options);
}

std::vector<Cut> generate_cuts(const Problem& problem, const RelaxationSolution& relaxation,
                               const Options& options, CutSeparatorPhase phase,
                               const ImplicationStore* learned_implications,
                               const std::vector<Cut>* structural_cuts) {
    const SeparatorContext context{problem, relaxation, options, learned_implications,
                                   structural_cuts};
    const auto& separators = default_cut_separators_();
    std::vector<Cut> cuts;
    std::vector<const CutSeparator*> phase_separators;
    for (const auto& separator : separators) {
        if (separator == nullptr || separator->phase() != phase || !separator->enabled(context)) {
            continue;
        }
        phase_separators.push_back(separator.get());
    }

    if (phase_separators.empty())
        return {};

    if (options.parallel_workers > 1 && phase_separators.size() > 1) {
        const int worker_count = std::min<int>(options.parallel_workers, phase_separators.size());
        ParallelDispatcher dispatcher(worker_count);
        std::vector<std::vector<Cut>> family_cuts(phase_separators.size());
        dispatcher.run(static_cast<int>(phase_separators.size()), [&](int index) {
            family_cuts[index] = phase_separators[index]->separate(context);
        });
        for (auto& separator_cuts : family_cuts) {
            cuts.insert(cuts.end(), std::make_move_iterator(separator_cuts.begin()),
                        std::make_move_iterator(separator_cuts.end()));
        }
    } else {
        for (const CutSeparator* separator : phase_separators) {
            std::vector<Cut> family_cuts = separator->separate(context);
            cuts.insert(cuts.end(), std::make_move_iterator(family_cuts.begin()),
                        std::make_move_iterator(family_cuts.end()));
        }
    }
    return filter_generated_cuts_(std::move(cuts), problem, relaxation, options);
}

} // namespace simplex::bnb::detail
