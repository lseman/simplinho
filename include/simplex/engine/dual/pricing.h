#pragma once

#include "simplex/core/thread_pool.h"
#include "simplex/engine/dual/bounds.h"

namespace simplex::engine {

class DualPricingOperations : public DualBoundModel {
  public:
    struct SparsePricingWorkspace {
        std::vector<int> rel_of_col;
        std::vector<unsigned int> marks;
        unsigned int stamp = 1;

        void prepare(int num_cols, const std::vector<int>& nonbasis) {
            if (static_cast<int>(rel_of_col.size()) < num_cols) {
                rel_of_col.resize(num_cols, -1);
                marks.resize(num_cols, 0);
            }
            if (++stamp == 0) {
                std::fill(marks.begin(), marks.end(), 0);
                stamp = 1;
            }
            for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
                const int col = nonbasis[k];
                if (col >= 0 && col < num_cols) {
                    rel_of_col[col] = k;
                    marks[col] = stamp;
                }
            }
        }

        int lookup(int col) const {
            return col >= 0 && col < static_cast<int>(marks.size()) && marks[col] == stamp
                       ? rel_of_col[col]
                       : -1;
        }
    };

    struct Telemetry {
        double row_ep_density = 0.0;
        double row_ap_density = 0.0;
        double col_aq_density = 0.0;
        int row_price_calls = 0;
        int col_price_calls = 0;
        int price_switches = 0;
        bool last_used_column_price = false;
        bool has_last_price_mode = false;

        static void update_density(double local_density, double& density) {
            constexpr double multiplier = 0.05;
            if (std::isfinite(local_density)) {
                local_density = std::clamp(local_density, 0.0, 1.0);
                density = (1.0 - multiplier) * density + multiplier * local_density;
            }
        }

        void record_price_mode(bool use_columns) {
            use_columns ? ++col_price_calls : ++row_price_calls;
            if (has_last_price_mode && last_used_column_price != use_columns)
                ++price_switches;
            last_used_column_price = use_columns;
            has_last_price_mode = true;
        }
    };

    static void compute_pricing_products(const Eigen::MatrixXd& Ahat,
                                         const std::vector<int>& nonbasis,
                                         const Eigen::VectorXd& pivot_row,
                                         const Eigen::VectorXd& dual, const Eigen::VectorXd& costs,
                                         Eigen::VectorXd& row_price,
                                         Eigen::VectorXd& reduced_cost) {
        row_price.resize(nonbasis.size());
        reduced_cost.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            const int j = nonbasis[k];
            const Eigen::VectorXd column = Ahat.col(j);
            row_price(k) = pivot_row.dot(column);
            reduced_cost(k) = costs(j) - column.dot(dual);
        }
    }

    static void compute_row_price(const Eigen::MatrixXd& Ahat,
                                  const std::vector<int>& nonbasis,
                                  const Eigen::VectorXd& pivot_row,
                                  Eigen::VectorXd& row_price) {
        row_price.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k)
            row_price(k) = pivot_row.dot(Ahat.col(nonbasis[k]));
    }

    static void compute_pricing_products(const RevisedSimplex::SparseMatrix& Ahat,
                                         const SparseRowMatrix& row_matrix,
                                         const std::vector<int>& nonbasis,
                                         const Eigen::VectorXd& pivot_row,
                                         const Eigen::VectorXd& dual, const Eigen::VectorXd& costs,
                                         Eigen::VectorXd& row_price,
                                         Eigen::VectorXd& reduced_cost) {
        row_price = Eigen::VectorXd::Zero(nonbasis.size());
        reduced_cost.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k)
            reduced_cost(k) = costs(nonbasis[k]);
        thread_local SparsePricingWorkspace workspace;
        workspace.prepare(Ahat.cols(), nonbasis);
        for (int i = 0; i < row_matrix.rows(); ++i) {
            const double wi = i < pivot_row.size() ? pivot_row(i) : 0.0;
            const double yi = i < dual.size() ? dual(i) : 0.0;
            if (wi == 0.0 && yi == 0.0)
                continue;
            for (SparseRowMatrix::InnerIterator it(row_matrix, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel < 0)
                    continue;
                row_price(rel) += wi * it.value();
                reduced_cost(rel) -= yi * it.value();
            }
        }
    }

    static void compute_row_price(const RevisedSimplex::SparseMatrix& Ahat,
                                  const SparseRowMatrix& row_matrix,
                                  const std::vector<int>& nonbasis,
                                  const HVector& pivot_row,
                                  Eigen::VectorXd& row_price) {
        if (!pivot_row.has_pattern()) {
            row_price.resize(nonbasis.size());
            for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
                double price = 0.0;
                for (RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, nonbasis[k]); it; ++it)
                    price += pivot_row.value(it.row()) * it.value();
                row_price(k) = price;
            }
            return;
        }
        row_price = Eigen::VectorXd::Zero(nonbasis.size());
        thread_local SparsePricingWorkspace workspace;
        workspace.prepare(Ahat.cols(), nonbasis);
        for (int k = 0; k < pivot_row.count; ++k) {
            const int row = pivot_row.index[k];
            if (row < 0 || row >= row_matrix.rows())
                continue;
            const double value = pivot_row.value(row);
            if (value == 0.0)
                continue;
            for (SparseRowMatrix::InnerIterator it(row_matrix, row); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel >= 0)
                    row_price(rel) += value * it.value();
            }
        }
    }

    static void compute_pricing_products(const RevisedSimplex::SparseMatrix& Ahat,
                                         const SparseRowMatrix& row_matrix,
                                         const std::vector<int>& nonbasis, const HVector& pivot_row,
                                         const Eigen::VectorXd& dual, const Eigen::VectorXd& costs,
                                         Eigen::VectorXd& row_price,
                                         Eigen::VectorXd& reduced_cost) {
        if (!pivot_row.has_pattern()) {
            compute_pricing_products(Ahat, row_matrix, nonbasis, pivot_row.value, dual, costs,
                                     row_price, reduced_cost);
            return;
        }
        row_price = Eigen::VectorXd::Zero(nonbasis.size());
        reduced_cost.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k)
            reduced_cost(k) = costs(nonbasis[k]);
        thread_local SparsePricingWorkspace workspace;
        workspace.prepare(Ahat.cols(), nonbasis);
        for (int k = 0; k < pivot_row.count; ++k) {
            const int i = pivot_row.index[k];
            const double wi = pivot_row.value(i);
            if (wi == 0.0 || i >= row_matrix.rows())
                continue;
            for (SparseRowMatrix::InnerIterator it(row_matrix, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel >= 0)
                    row_price(rel) += wi * it.value();
            }
        }
        for (int i = 0; i < row_matrix.rows(); ++i) {
            const double yi = i < dual.size() ? dual(i) : 0.0;
            if (yi == 0.0)
                continue;
            for (SparseRowMatrix::InnerIterator it(row_matrix, i); it; ++it) {
                const int rel = workspace.lookup(it.col());
                if (rel >= 0)
                    reduced_cost(rel) -= yi * it.value();
            }
        }
    }

    static void
    compute_pricing_products_by_column(const RevisedSimplex::SparseMatrix& Ahat,
                                       const std::vector<int>& nonbasis, const HVector& pivot_row,
                                       const Eigen::VectorXd& dual, const Eigen::VectorXd& costs,
                                       Eigen::VectorXd& row_price, Eigen::VectorXd& reduced_cost) {
        row_price.resize(nonbasis.size());
        reduced_cost.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            const int j = nonbasis[k];
            double price = 0.0;
            double cost = costs(j);
            for (RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, j); it; ++it) {
                price += it.value() * pivot_row(it.row());
                cost -= it.value() * dual(it.row());
            }
            row_price(k) = price;
            reduced_cost(k) = cost;
        }
    }

    static void compute_row_price_by_column(const RevisedSimplex::SparseMatrix& Ahat,
                                            const std::vector<int>& nonbasis,
                                            const HVector& pivot_row,
                                            Eigen::VectorXd& row_price) {
        row_price.resize(nonbasis.size());
        for (int k = 0; k < static_cast<int>(nonbasis.size()); ++k) {
            double price = 0.0;
            for (RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, nonbasis[k]); it; ++it)
                price += it.value() * pivot_row(it.row());
            row_price(k) = price;
        }
    }

    static void compute_pricing_products_parallel(
        const RevisedSimplex::SparseMatrix& Ahat, const std::vector<int>& nonbasis,
        const HVector& pivot_row, const Eigen::VectorXd& dual, const Eigen::VectorXd& costs,
        Eigen::VectorXd& row_price, Eigen::VectorXd& reduced_cost, int workers, int min_cols) {
        const int total = nonbasis.size();
        workers = std::max(1, workers);
        if (workers <= 1 || total < std::max(1, min_cols)) {
            compute_pricing_products(Ahat, SparseRowMatrix(Ahat), nonbasis, pivot_row, dual, costs,
                                     row_price, reduced_cost);
            return;
        }
        row_price = Eigen::VectorXd::Zero(total);
        reduced_cost.resize(total);
        const int worker_count = std::min(workers, total);

        // Use the process-wide thread pool instead of std::async — avoids
        // spawning a new thread per pricing call, which is the dominant
        // overhead when BnB solves many node LPs in parallel.
        auto worker_fn = [&](int tid) {
            // Distribute tasks evenly across workers (same chunking as the
            // old std::async loop, but driven by the pool's internal thread id).
            const int begin = tid * total / worker_count;
            const int end = (tid + 1) * total / worker_count;
            for (int k = begin; k < end; ++k) {
                const int j = nonbasis[k];
                double price = 0.0;
                double cost = costs(j);
                for (RevisedSimplex::SparseMatrix::InnerIterator it(Ahat, j); it; ++it) {
                    price += it.value() * pivot_row(it.row());
                    cost -= it.value() * dual(it.row());
                }
                row_price(k) = price;
                reduced_cost(k) = cost;
            }
        };
        ThreadPool::instance().submit(worker_count, std::move(worker_fn));
    }
};

} // namespace simplex::engine
