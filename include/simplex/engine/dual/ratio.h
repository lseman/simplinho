#pragma once

#include "extern/pdqsort/pdqsort.h"
#include "simplex/engine/common/utils.h"
#include "simplex/types/simplex_types.h"

namespace simplex::engine {

class DualRatioTest : public BoundUtilities {
  public:
    enum class BoundView { Lower, Upper, Fixed };

    struct DualChoose {
        std::optional<int> e_rel;
        double tau = std::numeric_limits<double>::infinity();
    };

    struct DualBFRTDecision {
        std::optional<int> pivot_rel;
        double tau = std::numeric_limits<double>::infinity();
        std::vector<int> flip_rels;
    };

    struct DualBFRTWorkspace {
        struct Candidate {
            int rel;
            double alpha;
            double dual;
            double range;
        };

        std::vector<Candidate> candidates;
        std::vector<unsigned char> selected;
        std::vector<int> group_indices;
        std::vector<std::pair<int, int>> group_ranges;
    };

    static DualChoose dual_harris_choose(const Eigen::VectorXd& rN, const Eigen::VectorXd& pN,
                                         double delta, double eta,
                                         double pivot_threshold = 0.0,
                                         const std::vector<int>* candidate_rels = nullptr) {
        const double eligibility = std::max(delta, pivot_threshold);
        double tau_star = std::numeric_limits<double>::infinity();
        auto scan = [&](auto&& visit) {
            if (candidate_rels) {
                for (const int k : *candidate_rels)
                    if (k >= 0 && k < pN.size())
                        visit(k);
            } else {
                for (int k = 0; k < pN.size(); ++k)
                    visit(k);
            }
        };
        scan([&](int k) {
            if (pN(k) < -eligibility)
                tau_star = std::min(tau_star, rN(k) / -pN(k));
        });
        if (!std::isfinite(tau_star))
            return {};
        const double window = std::max(eta, eta * std::abs(tau_star));

        int best = -1;
        double best_pivot = 0.0;
        scan([&](int k) {
            if (!(pN(k) < -eligibility))
                return;
            if (rN(k) / -pN(k) > tau_star + window)
                return;
            const double pivot = std::abs(pN(k));
            if (best < 0 || pivot > best_pivot + 1e-16 ||
                (std::abs(pivot - best_pivot) <= 1e-16 && k < best)) {
                best = k;
                best_pivot = pivot;
            }
        });
        if (best < 0)
            return {};
        return {best, std::max(0.0, rN(best) / -pN(best))};
    }

    static void dual_bfrt_decide(
        const RevisedSimplexOptions& options, const Eigen::VectorXd& rN,
        const Eigen::VectorXd& pN, const std::vector<int>& nonbasis,
        const std::vector<BoundView>& view, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        double primal_delta, int max_flips, int basis_update_count, DualBFRTWorkspace& workspace,
        DualBFRTDecision& out, const std::vector<int>* candidate_rels = nullptr) {
        out.pivot_rel.reset();
        out.tau = std::numeric_limits<double>::infinity();
        out.flip_rels.clear();
        const double pivot_threshold = basis_update_count < 10   ? 1e-9
                                       : basis_update_count < 20 ? 3e-8
                                                                 : 1e-6;
        const double eligibility = std::max(options.ratio_delta, pivot_threshold);
        const DualChoose harris = dual_harris_choose(
            rN, pN, options.ratio_delta, options.ratio_eta, pivot_threshold, candidate_rels);
        out.pivot_rel = harris.e_rel;
        out.tau = harris.tau;
        if (!harris.e_rel || !std::isfinite(harris.tau) || max_flips <= 0 ||
            !(primal_delta > options.tol)) {
            return;
        }

        auto& candidates = workspace.candidates;
        candidates.clear();
        candidates.reserve(candidate_rels ? candidate_rels->size() : nonbasis.size());
        auto add_candidate = [&](int k) {
            if (k < 0 || k >= static_cast<int>(nonbasis.size()))
                return;
            if (!(pN(k) < -eligibility))
                return;
            const int j = nonbasis[k];
            if (view[j] == BoundView::Fixed)
                return;
            const double alpha = -pN(k);
            const double dual = std::max(0.0, rN(k));
            if (std::isfinite(alpha) && std::isfinite(dual))
                candidates.push_back({k, alpha, dual, bound_range(j, l, u)});
        };
        if (candidate_rels) {
            for (const int k : *candidate_rels)
                add_candidate(k);
        } else {
            for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k)
                add_candidate(k);
        }
        if (candidates.empty())
            return;

        const double dual_tol = std::max(options.tol, options.ratio_eta);
        auto& selected = workspace.selected;
        selected.assign(candidates.size(), 0);
        auto& group_indices = workspace.group_indices;
        auto& group_ranges = workspace.group_ranges;
        group_indices.clear();
        group_ranges.clear();
        group_indices.reserve(candidates.size());
        group_ranges.reserve(candidates.size());
        double total_change = 0.0;
        double select_theta = std::numeric_limits<double>::infinity();
        for (const auto& candidate : candidates)
            select_theta =
                std::min(select_theta, (candidate.dual + dual_tol) / candidate.alpha);

        while (group_indices.size() < candidates.size() && std::isfinite(select_theta)) {
            const int group_begin = static_cast<int>(group_indices.size());
            double next_theta = std::numeric_limits<double>::infinity();
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                if (selected[i])
                    continue;
                const auto& candidate = candidates[i];
                const double tight_limit = select_theta * candidate.alpha;
                const double roundoff =
                    1e-14 * (1.0 + std::abs(candidate.dual) + std::abs(tight_limit));
                if (candidate.dual <= tight_limit + roundoff) {
                    selected[i] = 1;
                    group_indices.push_back(i);
                    total_change = std::isfinite(candidate.range)
                                       ? total_change +
                                             candidate.alpha * std::max(0.0, candidate.range)
                                       : std::numeric_limits<double>::infinity();
                } else {
                    next_theta =
                        std::min(next_theta, (candidate.dual + dual_tol) / candidate.alpha);
                }
            }
            const int group_end = static_cast<int>(group_indices.size());
            if (group_begin == group_end)
                break;
            pdqsort(group_indices.begin() + group_begin, group_indices.begin() + group_end,
                    [&](int a, int b) { return candidates[a].rel < candidates[b].rel; });
            group_ranges.emplace_back(group_begin, group_end);
            if (total_change >= primal_delta)
                break;
            select_theta = next_theta;
        }
        if (group_ranges.empty())
            return;

        double max_alpha = 0.0;
        for (const auto [begin, end] : group_ranges)
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_indices[pos];
                max_alpha = std::max(max_alpha, candidates[i].alpha);
            }
        const double alpha_threshold = std::min(0.1 * max_alpha, 1.0);

        int pivot_group = static_cast<int>(group_ranges.size()) - 1;
        int pivot_index = group_indices[group_ranges.back().first];
        for (int g = static_cast<int>(group_ranges.size()) - 1; g >= 0; --g) {
            const auto [begin, end] = group_ranges[g];
            int best = group_indices[begin];
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_indices[pos];
                if (candidates[i].alpha > candidates[best].alpha + 1e-16 ||
                    (std::abs(candidates[i].alpha - candidates[best].alpha) <= 1e-16 &&
                     candidates[i].rel < candidates[best].rel)) {
                    best = i;
                }
            }
            if (candidates[best].alpha > alpha_threshold) {
                pivot_group = g;
                pivot_index = best;
                break;
            }
        }

        auto& flips = out.flip_rels;
        flips.reserve(candidates.size());
        for (int g = 0; g < pivot_group; ++g)
            for (int pos = group_ranges[g].first; pos < group_ranges[g].second; ++pos) {
                const int i = group_indices[pos];
                if (std::isfinite(candidates[i].range) && candidates[i].range > options.tol)
                    flips.push_back(candidates[i].rel);
            }

        const auto& pivot = candidates[pivot_index];
        const double theta = pivot.dual / pivot.alpha;
        if (theta > 0.0) {
            const auto [begin, end] = group_ranges[pivot_group];
            for (int pos = begin; pos < end; ++pos) {
                const int i = group_indices[pos];
                if (i == pivot_index)
                    continue;
                const auto& candidate = candidates[i];
                const double new_dual = candidate.dual - theta * candidate.alpha;
                if (new_dual < -dual_tol && std::isfinite(candidate.range) &&
                    candidate.range > options.tol) {
                    flips.push_back(candidate.rel);
                }
            }
        }
        if (theta <= 0.0)
            flips.clear();
        if (static_cast<int>(flips.size()) > max_flips) {
            flips.clear();
            return;
        }

        pdqsort(flips.begin(), flips.end());
        out.pivot_rel = pivot.rel;
        out.tau = theta;
    }
};

} // namespace simplex::engine
