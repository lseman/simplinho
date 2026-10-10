#pragma once
// LP model and scaling for the HiGHS-style ("EKK") simplex engine.
//
// The model follows HiGHS (third_party/highs-source, MIT licence):
//   minimise c'x + offset  s.t.  row_lower <= A x <= row_upper,
//                                col_lower <= x  <= col_upper.
// The simplex works on [A | I] with one logical per row, x_{n+i} = -(A x)_i,
// so the logical bounds are [-row_upper, -row_lower] and the system is
// [A | I] x = 0. Bounds are handled natively; nothing is reformulated.

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace simplex::ekk {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

// Column-wise sparse LP. Rows and columns keep their own bounds.
struct LpData {
    int num_col = 0;
    int num_row = 0;
    std::vector<int> a_start{0}; // CSC, size num_col + 1
    std::vector<int> a_index;
    std::vector<double> a_value;
    std::vector<double> col_cost;
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    std::vector<double> row_lower;
    std::vector<double> row_upper;
    double offset = 0.0;

    void validate() const {
        if (static_cast<int>(a_start.size()) != num_col + 1 ||
            static_cast<int>(col_cost.size()) != num_col ||
            static_cast<int>(col_lower.size()) != num_col ||
            static_cast<int>(col_upper.size()) != num_col ||
            static_cast<int>(row_lower.size()) != num_row ||
            static_cast<int>(row_upper.size()) != num_row ||
            a_index.size() != a_value.size() ||
            static_cast<int>(a_index.size()) != a_start[num_col])
            throw std::invalid_argument("ekk::LpData: inconsistent dimensions");
        for (int idx : a_index)
            if (idx < 0 || idx >= num_row)
                throw std::invalid_argument("ekk::LpData: row index out of range");
    }

    static LpData from_sparse(const Eigen::SparseMatrix<double, Eigen::ColMajor, int>& A,
                              std::vector<double> cost, std::vector<double> col_lower,
                              std::vector<double> col_upper, std::vector<double> row_lower,
                              std::vector<double> row_upper, double offset = 0.0) {
        LpData lp;
        lp.num_col = static_cast<int>(A.cols());
        lp.num_row = static_cast<int>(A.rows());
        lp.a_start.assign(lp.num_col + 1, 0);
        for (int j = 0; j < lp.num_col; ++j) {
            for (Eigen::SparseMatrix<double, Eigen::ColMajor, int>::InnerIterator it(A, j); it;
                 ++it) {
                if (it.value() == 0.0)
                    continue;
                lp.a_index.push_back(static_cast<int>(it.row()));
                lp.a_value.push_back(it.value());
            }
            lp.a_start[j + 1] = static_cast<int>(lp.a_index.size());
        }
        lp.col_cost = std::move(cost);
        lp.col_lower = std::move(col_lower);
        lp.col_upper = std::move(col_upper);
        lp.row_lower = std::move(row_lower);
        lp.row_upper = std::move(row_upper);
        lp.offset = offset;
        lp.validate();
        return lp;
    }
};

// Row-wise copy of A, used for hyper-sparse PRICE.
struct RowwiseMatrix {
    std::vector<int> start{0};
    std::vector<int> index;
    std::vector<double> value;

    void build(const LpData& lp) {
        start.assign(lp.num_row + 1, 0);
        for (int k = 0; k < lp.a_start[lp.num_col]; ++k)
            ++start[lp.a_index[k] + 1];
        for (int i = 0; i < lp.num_row; ++i)
            start[i + 1] += start[i];
        index.assign(lp.a_index.size(), 0);
        value.assign(lp.a_value.size(), 0.0);
        std::vector<int> next(start.begin(), start.end() - 1);
        for (int j = 0; j < lp.num_col; ++j) {
            for (int k = lp.a_start[j]; k < lp.a_start[j + 1]; ++k) {
                const int pos = next[lp.a_index[k]]++;
                index[pos] = j;
                value[pos] = lp.a_value[k];
            }
        }
    }
};

// Power-of-two row/column scale factors: x = col * x_scaled, (row-scaled A)
// = diag(row) A diag(col). Powers of two keep scaling free of round-off.
struct LpScale {
    bool active = false;
    std::vector<double> col;
    std::vector<double> row;
};

inline double nearest_power_of_two(double value) {
    if (!(value > 0.0) || !std::isfinite(value))
        return 1.0;
    return std::exp2(std::floor(std::log2(value) + 0.5));
}

// HiGHS equilibrationScaleMatrix (HighsLpUtils.cpp): geometric-mean
// equilibration over up to `passes` alternating column/row sweeps, costs
// included when the smallest nonzero cost is below 0.1, factors clamped to
// 2^+-allowed_factor and rounded to powers of two. Scaling is skipped when all
// |a_ij| are already in [0.2, 5] and kept only if it improves equilibration.
inline LpScale compute_equilibration_scale(const LpData& lp, int passes = 6,
                                           int allowed_factor = 20) {
    LpScale scale;
    const int n = lp.num_col;
    const int m = lp.num_row;
    if (n == 0 || m == 0 || lp.a_value.empty())
        return scale;
    double min_abs = kInf, max_abs = 0.0;
    for (double v : lp.a_value) {
        const double a = std::abs(v);
        if (a == 0.0)
            continue;
        min_abs = std::min(min_abs, a);
        max_abs = std::max(max_abs, a);
    }
    if (min_abs >= 0.2 && max_abs <= 5.0)
        return scale;

    double min_nonzero_cost = kInf;
    for (double c : lp.col_cost)
        if (c != 0.0)
            min_nonzero_cost = std::min(min_nonzero_cost, std::abs(c));
    const bool include_cost = min_nonzero_cost < 0.1;
    const double max_allow = std::exp2(static_cast<double>(allowed_factor));
    const double min_allow = 1.0 / max_allow;
    constexpr double kFiniteInf = 1e200;

    std::vector<double> col_scale(n, 1.0), row_scale(m, 1.0);
    std::vector<double> row_min(m, kFiniteInf), row_max(m, 1.0 / kFiniteInf);
    for (int pass = 0; pass < passes; ++pass) {
        for (int j = 0; j < n; ++j) {
            double cmin = kFiniteInf, cmax = 1.0 / kFiniteInf;
            const double abs_cost = std::abs(lp.col_cost[j]);
            if (include_cost && abs_cost != 0.0) {
                cmin = std::min(cmin, abs_cost);
                cmax = std::max(cmax, abs_cost);
            }
            for (int k = lp.a_start[j]; k < lp.a_start[j + 1]; ++k) {
                const double v = std::abs(lp.a_value[k]) * row_scale[lp.a_index[k]];
                cmin = std::min(cmin, v);
                cmax = std::max(cmax, v);
            }
            col_scale[j] = std::clamp(1.0 / std::sqrt(cmin * cmax), min_allow, max_allow);
            for (int k = lp.a_start[j]; k < lp.a_start[j + 1]; ++k) {
                const int i = lp.a_index[k];
                const double v = std::abs(lp.a_value[k]) * col_scale[j];
                row_min[i] = std::min(row_min[i], v);
                row_max[i] = std::max(row_max[i], v);
            }
        }
        for (int i = 0; i < m; ++i)
            row_scale[i] =
                std::clamp(1.0 / std::sqrt(row_min[i] * row_max[i]), min_allow, max_allow);
        std::fill(row_min.begin(), row_min.end(), kFiniteInf);
        std::fill(row_max.begin(), row_max.end(), 1.0 / kFiniteInf);
    }
    for (double& s : col_scale)
        s = nearest_power_of_two(s);
    for (double& s : row_scale)
        s = nearest_power_of_two(s);

    // Keep the scaling only if it improves the matrix value ratio and the
    // column/row equilibration (product of improvement factors > 1).
    auto equilibration = [&](bool scaled, double* ratio, double* geo_col, double* geo_row,
                             double* extreme) {
        std::vector<double> rmin(m, kFiniteInf), rmax(m, 1.0 / kFiniteInf);
        double vmin = kFiniteInf, vmax = 0.0, cmin_e = kFiniteInf, cmax_e = 0.0, csum = 0.0;
        int ccount = 0;
        for (int j = 0; j < n; ++j) {
            double cmin = kFiniteInf, cmax = 1.0 / kFiniteInf;
            for (int k = lp.a_start[j]; k < lp.a_start[j + 1]; ++k) {
                const int i = lp.a_index[k];
                const double v = std::abs(lp.a_value[k]) *
                                 (scaled ? col_scale[j] * row_scale[i] : 1.0);
                if (v == 0.0)
                    continue;
                cmin = std::min(cmin, v);
                cmax = std::max(cmax, v);
                rmin[i] = std::min(rmin[i], v);
                rmax[i] = std::max(rmax[i], v);
            }
            if (cmax <= 1.0 / kFiniteInf)
                continue;
            vmin = std::min(vmin, cmin);
            vmax = std::max(vmax, cmax);
            const double e = 1.0 / std::sqrt(cmin * cmax);
            cmin_e = std::min(cmin_e, e);
            cmax_e = std::max(cmax_e, e);
            csum += std::log(e);
            ++ccount;
        }
        double rmin_e = kFiniteInf, rmax_e = 0.0, rsum = 0.0;
        int rcount = 0;
        for (int i = 0; i < m; ++i) {
            if (rmax[i] <= 1.0 / kFiniteInf)
                continue;
            const double e = 1.0 / std::sqrt(rmin[i] * rmax[i]);
            rmin_e = std::min(rmin_e, e);
            rmax_e = std::max(rmax_e, e);
            rsum += std::log(e);
            ++rcount;
        }
        *ratio = vmax / vmin;
        const double gc = std::exp(csum / std::max(1, ccount));
        const double gr = std::exp(rsum / std::max(1, rcount));
        *geo_col = std::max(gc, 1.0 / gc);
        *geo_row = std::max(gr, 1.0 / gr);
        *extreme = cmax_e / cmin_e + rmax_e / rmin_e;
    };
    double r0, gc0, gr0, x0, r1, gc1, gr1, x1;
    equilibration(false, &r0, &gc0, &gr0, &x0);
    equilibration(true, &r1, &gc1, &gr1, &x1);
    const double improvement = (x0 / x1) * std::sqrt((gc0 * gr0) / (gc1 * gr1)) * (r0 / r1);
    if (!(improvement > 1.0))
        return scale;
    scale.active = true;
    scale.col = std::move(col_scale);
    scale.row = std::move(row_scale);
    return scale;
}

// Apply scaling in place: A_s = R A C, c_s = C c, col bounds / C, row bounds * R.
inline void apply_scale(LpData& lp, const LpScale& scale) {
    if (!scale.active)
        return;
    for (int j = 0; j < lp.num_col; ++j) {
        const double cj = scale.col[j];
        for (int k = lp.a_start[j]; k < lp.a_start[j + 1]; ++k)
            lp.a_value[k] *= cj * scale.row[lp.a_index[k]];
        lp.col_cost[j] *= cj;
        lp.col_lower[j] /= cj;
        lp.col_upper[j] /= cj;
    }
    for (int i = 0; i < lp.num_row; ++i) {
        lp.row_lower[i] *= scale.row[i];
        lp.row_upper[i] *= scale.row[i];
    }
}

// HiGHS HVector: dense array plus the indices of its nonzeros (count < 0:
// pattern unknown), workspace for hyper-sparse solves, and a packed copy
// captured mid-solve for the Forrest-Tomlin update.
struct WorkVector {
    int size = 0;
    int count = 0;
    std::vector<int> index;
    std::vector<double> array;
    std::vector<char> cwork;
    std::vector<int> iwork;
    int pack_count = 0;
    std::vector<int> pack_index;
    std::vector<double> pack_value;
    bool pack_flag = false;
    double synthetic_tick = 0.0;

    void setup(int size_) {
        size = size_;
        count = 0;
        index.assign(size, 0);
        array.assign(size, 0.0);
        cwork.assign(size + 6400, 0);
        iwork.assign(size * 4, 0);
        pack_count = 0;
        pack_index.assign(size, 0);
        pack_value.assign(size, 0.0);
        pack_flag = false;
        synthetic_tick = 0.0;
    }
    void clear_scalars() {
        pack_flag = false;
        count = 0;
        synthetic_tick = 0.0;
    }
    void clear() {
        if (count < 0 || count > size * 0.3) {
            std::fill(array.begin(), array.end(), 0.0);
        } else {
            for (int i = 0; i < count; ++i)
                array[index[i]] = 0.0;
        }
        clear_scalars();
    }
    // Set a single entry of a cleared vector.
    void set_unit(int i, double v = 1.0) {
        array[i] = v;
        index[0] = i;
        count = 1;
    }
    // Zero entries below 1e-14 in magnitude, maintaining the index.
    void tight() {
        constexpr double kTiny = 1e-14;
        if (count < 0) {
            for (double& v : array)
                if (std::abs(v) < kTiny)
                    v = 0.0;
            return;
        }
        int kept = 0;
        for (int i = 0; i < count; ++i) {
            const int j = index[i];
            if (std::abs(array[j]) >= kTiny)
                index[kept++] = j;
            else
                array[j] = 0.0;
        }
        count = kept;
    }
    void pack() {
        if (!pack_flag)
            return;
        pack_flag = false;
        pack_count = 0;
        for (int i = 0; i < count; ++i) {
            const int j = index[i];
            pack_index[pack_count] = j;
            pack_value[pack_count++] = array[j];
        }
    }
    // Rebuild the index from scratch unless it is known and sparse.
    void re_index() {
        if (count >= 0 && count <= size * 0.1)
            return;
        count = 0;
        for (int i = 0; i < size; ++i)
            if (array[i] != 0.0)
                index[count++] = i;
    }
    double norm2() const {
        double s = 0.0;
        for (int i = 0; i < count; ++i)
            s += array[index[i]] * array[index[i]];
        return s;
    }
};

} // namespace simplex::ekk
