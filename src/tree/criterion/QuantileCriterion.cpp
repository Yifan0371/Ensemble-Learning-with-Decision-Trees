// QuantileCriterion.cpp
#include "criterion/QuantileCriterion.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>

double QuantileCriterion::nodeMetric(const std::vector<double>& y,
                                     const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    size_t n = idx.size();
    // Parallel copy of selected values
    std::vector<double> vals(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        vals[i] = y[idx[i]];
    }

    // Compute τ-quantile serially
    size_t k = static_cast<size_t>(tau_ * (n - 1));
    std::nth_element(vals.begin(), vals.begin() + k, vals.end());
    double q = vals[k];

    // Parallel pinball loss computation
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        double v = vals[i] - q;
        loss += (v < 0 ? (tau_ - 1.0) * v : tau_ * v);
    }

    return loss / n;
}
