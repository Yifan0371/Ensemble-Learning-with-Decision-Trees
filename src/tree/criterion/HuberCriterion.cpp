#include "criterion/HuberCriterion.hpp"
#include <cmath>
#include <numeric>
#include <omp.h>

double HuberCriterion::nodeMetric(const std::vector<double>& y,
                                  const std::vector<int>& idx) const
{
    if (idx.empty()) return 0.0;

    const double d = delta_;
    const int n = static_cast<int>(idx.size());

    // Parallel sum for mean calculation
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int t = 0; t < n; ++t) {
        sum += y[idx[t]];
    }
    double mu = sum / n;

    // Parallel Huber loss computation
    double loss = 0.0;
    #pragma omp parallel for reduction(+:loss) schedule(static)
    for (int t = 0; t < n; ++t) {
        double r = y[idx[t]] - mu;
        double abs_r = std::abs(r);
        if (abs_r <= d) {
            loss += 0.5 * r * r;
        } else {
            loss += d * (abs_r - 0.5 * d);
        }
    }

    return loss / n;
}
