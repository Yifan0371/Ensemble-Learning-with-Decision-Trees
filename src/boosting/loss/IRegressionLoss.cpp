// =============================================================================
// src/boosting/loss/IRegressionLoss.cpp - OpenMP Parallel Optimized Version
// =============================================================================
#include "boosting/loss/IRegressionLoss.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

void IRegressionLoss::computeGradientsHessians(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients,
    std::vector<double>& hessians) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    hessians.resize(n);
    
    // **Parallel Optimization 1: Parallel computation of gradients and Hessians**
    // This is a performance-critical path for second-order methods like XGBoost
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = gradient(y_true[i], y_pred[i]);
        hessians[i] = hessian(y_true[i], y_pred[i]);
    }
}

double IRegressionLoss::computeBatchLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **Parallel Optimization 2: Parallel reduction for batch loss computation**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += loss(y_true[i], y_pred[i]);
    }
    
    return totalLoss / n;
}

void IRegressionLoss::computeBatchGradients(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    
    // **Parallel Optimization 3: Parallel gradient computation (commonly used in GBDT)**
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        gradients[i] = gradient(y_true[i], y_pred[i]);
    }
}



void IRegressionLoss::computeGradientsVectorized(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& gradients) const {
    
    size_t n = y_true.size();
    gradients.resize(n);
    
    // **Parallel Optimization 4: SIMD-friendly vectorized version**
    // Use larger block size to enable better SIMD optimization by the compiler
    #pragma omp parallel for schedule(static, 2048) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        // Simplified memory access pattern to benefit SIMD
        double yt = y_true[i];
        double yp = y_pred[i];
        gradients[i] = gradient(yt, yp);
    }
}



double IRegressionLoss::computeBatchLossWithTiming(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    double& computeTimeMs) const {
    
    auto start = std::chrono::high_resolution_clock::now();
    
    double result = computeBatchLoss(y_true, y_pred);
    
    auto end = std::chrono::high_resolution_clock::now();
    computeTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    return result;
}
