// =============================================================================
// src/boosting/strategy/GradientRegressionStrategy.cpp - OpenMP Parallel Optimized Version
// =============================================================================
#include "boosting/strategy/GradientRegressionStrategy.hpp"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

void GradientRegressionStrategy::updateTargets(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    std::vector<double>& targets) const {
    
    size_t n = y_true.size();
    targets.resize(n);
    
    // **Parallel Optimization 1: Parallel computation of residuals/gradients**
    // Gradient computation for each sample is fully independent, ideal for parallelization
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        targets[i] = lossFunc_->gradient(y_true[i], y_pred[i]);
    }
}

void GradientRegressionStrategy::updatePredictions(
    const std::vector<double>& tree_pred,
    double learning_rate,
    std::vector<double>& y_pred) const {
    
    size_t n = y_pred.size();
    
    // **Parallel Optimization 2: Parallel update of predictions**
    // Prediction updates for each sample are fully independent
    #pragma omp parallel for schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        y_pred[i] += learning_rate * tree_pred[i];
    }
}

double GradientRegressionStrategy::computeTotalLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **Parallel Optimization 3: Parallel reduction for loss computation**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 1024) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += lossFunc_->loss(y_true[i], y_pred[i]);
    }
    
    return totalLoss / n;
}

double GradientRegressionStrategy::computeOptimalLearningRate(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const std::vector<double>& tree_pred) const {
    
    // Simple golden-section line search (parallel optimized version)
    double low = 0.0, high = 1.0;
    const double phi = 0.618033988749;
    const int maxIter = 10;
    const double tol = 1e-3;
    
    for (int iter = 0; iter < maxIter; ++iter) {
        double mid1 = low + (1 - phi) * (high - low);
        double mid2 = low + phi * (high - low);
        
        // **Parallel Optimization 4: Parallelized loss evaluation during line search**
        double loss1 = evaluateLoss(y_true, y_pred, tree_pred, mid1);
        double loss2 = evaluateLoss(y_true, y_pred, tree_pred, mid2);
        
        if (loss1 < loss2) {
            high = mid2;
        } else {
            low = mid1;
        }
        
        if (std::abs(high - low) < tol) break;
    }
    
    return (low + high) * 0.5;
}

double GradientRegressionStrategy::evaluateLoss(
    const std::vector<double>& y_true,
    const std::vector<double>& y_pred,
    const std::vector<double>& tree_pred,
    double lr) const {
    
    size_t n = y_true.size();
    double totalLoss = 0.0;
    
    // **Parallel Optimization 5: Parallel loss evaluation during line search**
    #pragma omp parallel for reduction(+:totalLoss) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        double newPred = y_pred[i] + lr * tree_pred[i];
        totalLoss += lossFunc_->loss(y_true[i], newPred);
    }
    
    return totalLoss / n;
}
