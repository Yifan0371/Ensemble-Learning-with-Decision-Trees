// LogCoshCriterion.cpp
#include "criterion/LogCoshCriterion.hpp"
#include <cmath>
#include <numeric>
#include <omp.h>

double LogCoshCriterion::nodeMetric(const std::vector<double>& y,
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
    double mu = sum / n;

    // Parallel log-cosh loss computation
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (int t = 0; t < n; ++t) {
        double r = y[idx[t]] - mu;
        loss += std::log(std::cosh(r));
    }

    return loss / n;
}
