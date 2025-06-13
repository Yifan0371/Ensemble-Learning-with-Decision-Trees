// =============================================================================
// src/lightgbm/feature/FeatureBundler.cpp - Optimized Version (avoiding vector<vector>)
// =============================================================================
#include "lightgbm/feature/FeatureBundler.hpp"
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <memory>

// Optimized conflict matrix - uses a 1D array to simulate a 2D matrix
class OptimizedConflictMatrix {
private:
    std::vector<double> data_;
    int size_;
    
public:
    explicit OptimizedConflictMatrix(int size) : size_(size) {
        data_.resize(size * size, 0.0);
    }
    
    double& operator()(int i, int j) {
        return data_[i * size_ + j];
    }
    
    const double& operator()(int i, int j) const {
        return data_[i * size_ + j];
    }
    
    int size() const { return size_; }
};

void FeatureBundler::createBundles(const std::vector<double>& data,
                                  int rowLength,
                                  size_t sampleSize,
                                  std::vector<FeatureBundle>& bundles) const {
    bundles.clear();
    
    // Optimization 1: Fast sparsity detection
    std::vector<double> sparsity(rowLength);
    constexpr double EPS = 1e-12;
    
    // Parallel sparsity calculation
    #pragma omp parallel for schedule(static) if(rowLength > 20)
    for (int f = 0; f < rowLength; ++f) {
        int nonZeroCount = 0;
        const size_t checkSize = std::min(sampleSize, size_t(5000)); // Sample check to reduce computation
        
        for (size_t i = 0; i < checkSize; ++i) {
            if (std::abs(data[i * rowLength + f]) > EPS) {
                ++nonZeroCount;
            }
        }
        sparsity[f] = 1.0 - static_cast<double>(nonZeroCount) / checkSize;
    }
    
    // Optimization 2: Only bind sparse features
    std::vector<int> sparseFeatures;
    std::vector<int> denseFeatures;
    constexpr double SPARSITY_THRESHOLD = 0.8; // 80% sparsity threshold
    
    sparseFeatures.reserve(rowLength);
    denseFeatures.reserve(rowLength);
    
    for (int f = 0; f < rowLength; ++f) {
        if (sparsity[f] > SPARSITY_THRESHOLD) {
            sparseFeatures.push_back(f);
        } else {
            denseFeatures.push_back(f);
        }
    }
    
    // Dense features are treated as individual bundles
    bundles.reserve(denseFeatures.size() + sparseFeatures.size() / 2);
    for (int f : denseFeatures) {
        FeatureBundle bundle;
        bundle.features.push_back(f);
        bundle.offsets.push_back(0.0);
        bundle.totalBins = maxBin_;
        bundles.push_back(std::move(bundle));
    }
    
    if (sparseFeatures.size() < 2) {
        // If too few sparse features, treat them individually
        for (int f : sparseFeatures) {
            FeatureBundle bundle;
            bundle.features.push_back(f);
            bundle.offsets.push_back(0.0);
            bundle.totalBins = maxBin_;
            bundles.push_back(std::move(bundle));
        }
        return;
    }
    
    // Optimization 3: Use an optimized conflict matrix (1D array simulating 2D)
    const int numSparse = static_cast<int>(sparseFeatures.size());
    OptimizedConflictMatrix conflictMatrix(numSparse);
    
    // Parallel conflict matrix calculation
    #pragma omp parallel for schedule(dynamic) if(numSparse > 10)
    for (int i = 0; i < numSparse; ++i) {
        for (int j = i + 1; j < numSparse; ++j) {
            const double conflict = calculateConflictRateOptimized(
                data, rowLength, sampleSize, sparseFeatures[i], sparseFeatures[j]);
            conflictMatrix(i, j) = conflictMatrix(j, i) = conflict;
        }
    }
    
    // Optimization 4: Improved greedy bundling algorithm
    std::vector<bool> used(numSparse, false);
    
    // Sort by sparsity (more sparse first)
    std::vector<std::pair<double, int>> sparsityWithIndex;
    sparsityWithIndex.reserve(numSparse);
    for (int i = 0; i < numSparse; ++i) {
        sparsityWithIndex.emplace_back(sparsity[sparseFeatures[i]], i);
    }
    std::sort(sparsityWithIndex.begin(), sparsityWithIndex.end(), std::greater<>());
    
    for (const auto& [sp, i] : sparsityWithIndex) {
        if (used[i]) continue;
        
        FeatureBundle bundle;
        bundle.features.push_back(sparseFeatures[i]);
        bundle.offsets.push_back(0.0);
        used[i] = true;
        
        double currentOffset = maxBin_;
        
        // Greedily add compatible features
        for (const auto& [sp2, j] : sparsityWithIndex) {
            if (used[j] || conflictMatrix(i, j) > maxConflictRate_) continue;
            
            // Check compatibility with all features in the bundle
            bool compatible = true;
            for (int bundledFeature : bundle.features) {
                const int bundledIdx = std::find(sparseFeatures.begin(), sparseFeatures.end(), bundledFeature) 
                                     - sparseFeatures.begin();
                if (conflictMatrix(j, bundledIdx) > maxConflictRate_) {
                    compatible = false;
                    break;
                }
            }
            
            if (compatible && currentOffset + maxBin_ <= 65536) {
                bundle.features.push_back(sparseFeatures[j]);
                bundle.offsets.push_back(currentOffset);
                used[j] = true;
                currentOffset += maxBin_;
            }
        }
        
        bundle.totalBins = static_cast<int>(currentOffset);
        bundles.push_back(std::move(bundle));
    }
}

// Optimized conflict rate calculation
double FeatureBundler::calculateConflictRateOptimized(const std::vector<double>& data,
                                                     int rowLength, 
                                                     size_t sampleSize,
                                                     int feat1, 
                                                     int feat2) const {
    constexpr double EPS = 1e-12;
    size_t conflicts = 0;
    size_t validPairs = 0;
    
    // Optimization: Sample calculation to reduce computation
    const size_t checkSize = std::min(sampleSize, size_t(2000)); // Sample check
    const size_t step = std::max(size_t(1), sampleSize / checkSize);
    
    for (size_t i = 0; i < sampleSize; i += step) {
        const double val1 = data[i * rowLength + feat1];
        const double val2 = data[i * rowLength + feat2];
        
        const bool nonZero1 = std::abs(val1) > EPS;
        const bool nonZero2 = std::abs(val2) > EPS;
        
        if (nonZero1 || nonZero2) {
            ++validPairs;
            if (nonZero1 && nonZero2) {
                ++conflicts;
            }
        }
    }
    
    return validPairs > 0 ? static_cast<double>(conflicts) / validPairs : 0.0;
}

// Retain original method but use optimized implementation
double FeatureBundler::calculateConflictRate(const std::vector<double>& data,
                                           int rowLength, size_t sampleSize,
                                           int feat1, int feat2) const {
    return calculateConflictRateOptimized(data, rowLength, sampleSize, feat1, feat2);
}

void FeatureBundler::buildConflictGraph(const std::vector<double>& data,
                                      int rowLength, size_t sampleSize,
                                      std::vector<std::vector<double>>& conflictMatrix) const {
    // Optimization: Use flattened conflict matrix for calculation
    const int numFeatures = rowLength;
    OptimizedConflictMatrix optimizedMatrix(numFeatures);
    
    // Parallel computation
    #pragma omp parallel for schedule(dynamic) if(numFeatures > 10)
    for (int i = 0; i < numFeatures; ++i) {
        for (int j = i + 1; j < numFeatures; ++j) {
            const double conflict = calculateConflictRateOptimized(data, rowLength, sampleSize, i, j);
            optimizedMatrix(i, j) = optimizedMatrix(j, i) = conflict;
        }
    }
    
    // Convert back to original format (for compatibility)
    conflictMatrix.assign(numFeatures, std::vector<double>(numFeatures, 0.0));
    for (int i = 0; i < numFeatures; ++i) {
        for (int j = 0; j < numFeatures; ++j) {
            conflictMatrix[i][j] = optimizedMatrix(i, j);
        }
    }
}

std::pair<int, double> FeatureBundler::transformFeature(int originalFeature,
                                                       double value,
                                                       const std::vector<FeatureBundle>& bundles) const {
    // Optimization: Use precomputed lookup table (if called frequently)
    for (size_t bundleIdx = 0; bundleIdx < bundles.size(); ++bundleIdx) {
        const auto& bundle = bundles[bundleIdx];
        
        // Optimization: Linear search for small bundles, binary search for large ones
        if (bundle.features.size() <= 8) {
            // Linear search
            for (size_t pos = 0; pos < bundle.features.size(); ++pos) {
                if (bundle.features[pos] == originalFeature) {
                    return transformFeatureValue(value, bundle.offsets[pos]);
                }
            }
        } else {
            // Binary search (if features are sorted)
            auto it = std::lower_bound(bundle.features.begin(), bundle.features.end(), originalFeature);
            if (it != bundle.features.end() && *it == originalFeature) {
                const size_t pos = std::distance(bundle.features.begin(), it);
                return transformFeatureValue(value, bundle.offsets[pos]);
            }
        }
    }
    
    // Should not happen, return original value
    return {originalFeature, value};
}

// New auxiliary method
std::pair<int, double> FeatureBundler::transformFeatureValue(double value, double offset) const {
    constexpr double EPS = 1e-12;
    
    double transformedValue;
    if (std::abs(value) < EPS) {
        // Handle zero values
        transformedValue = offset;
    } else {
        // Optimization: More precise value transformation
        // Use a more stable mapping function
        const int binIndex = static_cast<int>(std::abs(value) * maxBin_ / 1000.0) % maxBin_;
        transformedValue = offset + binIndex + 1; // +1 to avoid conflict with zero value
    }
    
    return {0, transformedValue}; // bundleIdx determined upstream
}