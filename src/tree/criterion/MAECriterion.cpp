// src/tree/criterion/MAECriterion.cpp - OpenMP Parallel Version
#include "criterion/MAECriterion.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

// Utility: compute median of a subset in parallel
static double subsetMedianParallel(const std::vector<double>& y,
                                   const std::vector<int>& idx)
{
    std::vector<double> v;
    v.reserve(idx.size());
    
    // Parallel copy for large subsets
    if (idx.size() > 1000) {
        v.resize(idx.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < idx.size(); ++i) {
            v[i] = y[idx[i]];
        }
    } else {
        // Serial copy for small subsets
        for (int i : idx) {
            v.push_back(y[i]);
        }
    }

    size_t n = v.size();
    auto mid_it = v.begin() + n/2;
    std::nth_element(v.begin(), mid_it, v.end());

    if (n % 2 == 1) {
        return *mid_it;
    } else {
        // average two middle elements
        auto left_it = std::max_element(v.begin(), mid_it);
        return 0.5 * (*left_it + *mid_it);
    }
}

// Parallel MAE computation per node
double MAECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const
{
    if (indices.empty()) return 0.0;

    double med = subsetMedianParallel(labels, indices);
    double sumAbs = 0.0;
    
    // Parallel sum of absolute deviations
    #pragma omp parallel for reduction(+:sumAbs) schedule(static) if(indices.size() > 1000)
    for (size_t i = 0; i < indices.size(); ++i) {
        sumAbs += std::abs(labels[indices[i]] - med);
    }
    
    return sumAbs / indices.size();
}
