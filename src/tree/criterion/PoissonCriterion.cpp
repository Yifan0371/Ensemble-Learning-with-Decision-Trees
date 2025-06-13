// PoissonCriterion.cpp
#include "criterion/PoissonCriterion.hpp"
#include <cmath>
#include <limits>
#include <omp.h>

double PoissonCriterion::nodeMetric(const std::vector<double>& y,
                                    const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    int n = static_cast<int>(idx.size());
    // Parallel sum for mean calculation
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int t = 0; t < n; ++t) {
        sum += y[idx[t]];
    }
    double mu = std::max(sum / n, 1e-12);

    // Parallel Poisson loss computation
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (int t = 0; t < n; ++t) {
        double yi = std::max(y[idx[t]], 1e-12);
        loss += mu - yi * std::log(mu);
    }

    return loss / n;
}
