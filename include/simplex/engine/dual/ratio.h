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

    static DualChoose dual_harris_choose(const Eigen::VectorXd& rN, const Eigen::VectorXd& pN,
                                         double delta, double eta) {
        std::vector<int> eligible;
        for (int k = 0; k < pN.size(); ++k)
            if (pN(k) < -delta)
                eligible.push_back(k);
        if (eligible.empty())
            return {};

        double tau_star = std::numeric_limits<double>::infinity();
        for (int k : eligible)
            tau_star = std::min(tau_star, rN(k) / -pN(k));
        const double window = std::max(eta, eta * std::abs(tau_star));

        int best = -1;
        double best_pivot = 0.0;
        for (int k : eligible) {
            if (rN(k) / -pN(k) > tau_star + window)
                continue;
            const double pivot = std::abs(pN(k));
            if (best < 0 || pivot > best_pivot + 1e-16 ||
                (std::abs(pivot - best_pivot) <= 1e-16 && k < best)) {
                best = k;
                best_pivot = pivot;
            }
        }
        if (best < 0)
            return {};
        return {best, std::max(0.0, rN(best) / -pN(best))};
    }

    static DualBFRTDecision dual_bfrt_decide(
        const RevisedSimplexOptions& options, const Eigen::VectorXd& rN,
        const Eigen::VectorXd& pN, const std::vector<int>& nonbasis,
        const std::vector<BoundView>& view, const Eigen::VectorXd& l, const Eigen::VectorXd& u,
        double primal_delta, int max_flips) {
        DualBFRTDecision out;
        const DualChoose harris =
            dual_harris_choose(rN, pN, options.ratio_delta, options.ratio_eta);
        out.pivot_rel = harris.e_rel;
        out.tau = harris.tau;
        if (!harris.e_rel || !std::isfinite(harris.tau) || max_flips <= 0 ||
            !(primal_delta > options.tol)) {
            return out;
        }

        struct Candidate {
            int rel;
            double alpha;
            double dual;
            double range;
        };
        std::vector<Candidate> candidates;
        candidates.reserve(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            if (!(pN(k) < -options.ratio_delta))
                continue;
            const int j = nonbasis[k];
            if (view[j] == BoundView::Fixed)
                continue;
            const double alpha = -pN(k);
            const double dual = std::max(0.0, rN(k));
            if (std::isfinite(alpha) && std::isfinite(dual))
                candidates.push_back({k, alpha, dual, bound_range(j, l, u)});
        }
        if (candidates.empty())
            return out;

        const double dual_tol = std::max(options.tol, options.ratio_eta);
        std::vector<char> selected(candidates.size(), 0);
        std::vector<std::vector<int>> groups;
        double total_change = 0.0;
        double select_theta = std::numeric_limits<double>::infinity();
        for (const Candidate& candidate : candidates)
            select_theta =
                std::min(select_theta, (candidate.dual + dual_tol) / candidate.alpha);

        while (groups.size() < candidates.size() && std::isfinite(select_theta)) {
            std::vector<int> group;
            double next_theta = std::numeric_limits<double>::infinity();
            for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
                if (selected[i])
                    continue;
                const Candidate& candidate = candidates[i];
                const double tight_limit = select_theta * candidate.alpha;
                const double roundoff =
                    1e-14 * (1.0 + std::abs(candidate.dual) + std::abs(tight_limit));
                if (candidate.dual <= tight_limit + roundoff) {
                    selected[i] = 1;
                    group.push_back(i);
                    total_change = std::isfinite(candidate.range)
                                       ? total_change +
                                             candidate.alpha * std::max(0.0, candidate.range)
                                       : std::numeric_limits<double>::infinity();
                } else {
                    next_theta =
                        std::min(next_theta, (candidate.dual + dual_tol) / candidate.alpha);
                }
            }
            if (group.empty())
                break;
            pdqsort(group.begin(), group.end(),
                    [&](int a, int b) { return candidates[a].rel < candidates[b].rel; });
            groups.push_back(std::move(group));
            if (total_change >= primal_delta)
                break;
            select_theta = next_theta;
        }
        if (groups.empty())
            return out;

        double max_alpha = 0.0;
        for (const auto& group : groups)
            for (int i : group)
                max_alpha = std::max(max_alpha, candidates[i].alpha);
        const double alpha_threshold = std::min(0.1 * max_alpha, 1.0);

        int pivot_group = static_cast<int>(groups.size()) - 1;
        int pivot_index = groups.back().front();
        for (int g = static_cast<int>(groups.size()) - 1; g >= 0; --g) {
            int best = groups[g].front();
            for (int i : groups[g]) {
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

        std::vector<int> flips;
        for (int g = 0; g < pivot_group; ++g)
            for (int i : groups[g])
                if (std::isfinite(candidates[i].range) && candidates[i].range > options.tol)
                    flips.push_back(candidates[i].rel);

        const Candidate& pivot = candidates[pivot_index];
        const double theta = pivot.dual / pivot.alpha;
        if (theta > 0.0) {
            for (int i : groups[pivot_group]) {
                if (i == pivot_index)
                    continue;
                const Candidate& candidate = candidates[i];
                const double new_dual = candidate.dual - theta * candidate.alpha;
                if (new_dual < -dual_tol && std::isfinite(candidate.range) &&
                    candidate.range > options.tol) {
                    flips.push_back(candidate.rel);
                }
            }
        }
        if (theta <= 0.0)
            flips.clear();
        if (static_cast<int>(flips.size()) > max_flips)
            return out;

        pdqsort(flips.begin(), flips.end());
        out.pivot_rel = pivot.rel;
        out.tau = theta;
        out.flip_rels = std::move(flips);
        return out;
    }
};

} // namespace simplex::engine
