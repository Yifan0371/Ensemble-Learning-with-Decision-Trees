// src/tree/criterion/MSECriterion.cpp - OpenMP Parallel Version
#include "criterion/MSECriterion.hpp"
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

double MSECriterion::nodeMetric(const std::vector<double>& labels,
                                const std::vector<int>& indices) const {
    if (indices.empty()) return 0.0;
    
    size_t n = indices.size();
    
    // Parallel reduction for sum and sum of squares
    double sum = 0.0;
    double sumSq = 0.0;
    #pragma omp parallel for reduction(+:sum,sumSq) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        double y = labels[indices[i]];
        sum += y;
        sumSq += y * y;
    }
    
    // MSE = E[y^2] - (E[y])^2
    double mean = sum / n;
    double mse = sumSq / n - mean * mean;
    
    // Ensure non-negative due to precision
    return std::max(0.0, mse);
}
