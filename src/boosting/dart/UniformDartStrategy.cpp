// =============================================================================
// src/boosting/dart/UniformDartStrategy.cpp - Deep Parallel Optimized Version
// =============================================================================
#include "boosting/dart/UniformDartStrategy.hpp"
#include <algorithm>
#include <random>
#include <iostream>
#include <cmath>
#include <execution>  // C++17 parallel algorithms
#ifdef _OPENMP
#include <omp.h>
#endif

std::vector<int> UniformDartStrategy::selectDroppedTrees(
    int totalTrees, double dropRate, std::mt19937& gen) const {
    
    if (totalTrees <= 0 || dropRate <= 0.0 || dropRate >= 1.0) {
        return {};
    }
    
    std::vector<int> droppedTrees;
    const int expectedDrops = static_cast<int>(std::ceil(totalTrees * dropRate));
    droppedTrees.reserve(expectedDrops + 5);  // Pre-allocate space
    
    // **Optimization 1: Batch random number generation**
    std::vector<double> randomValues(totalTrees);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Batch generate random numbers (more efficient)
    for (int i = 0; i < totalTrees; ++i) {
        randomValues[i] = dist(gen);
    }
    
    // **Optimization 2: Vectorized selection process**
    for (int i = 0; i < totalTrees; ++i) {
        if (randomValues[i] < dropRate) {
            droppedTrees.push_back(i);
        }
    }
    
    // **Optimization 3: Ensure at least one tree is dropped (if expected >= 1)**
    if (droppedTrees.empty() && expectedDrops >= 1 && totalTrees > 0) {
        std::uniform_int_distribution<int> treeDist(0, totalTrees - 1);
        droppedTrees.push_back(treeDist(gen));
    }
    
    return droppedTrees;
}

double UniformDartStrategy::computeDropoutPrediction(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    // **Skip dropout during prediction (used in final prediction)**
    if (skipDropForPrediction_) {
        return computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    }
    
    // **Exclude dropped trees during training**
    return computeDropoutPredictionOptimized(trees, droppedIndices, sample, rowLength, baseScore);
}

// **Optimization 1: Efficient full prediction (no dropout)**
double UniformDartStrategy::computeFullPredictionOptimized(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    double prediction = baseScore;
    const size_t numTrees = trees.size();
    size_t i = 0;

    // **Unroll loop for the first few trees**
    for (; i + 3 < numTrees; i += 4) {
        prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+1], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+2], sample, rowLength);
        prediction += computeSingleTreeContribution(trees[i+3], sample, rowLength);
    }

    // **Handle remaining trees**
    for (; i < numTrees; ++i) {
        prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
    }
    
    return prediction;
}

// **Optimization 2: Efficient dropout prediction**
double UniformDartStrategy::computeDropoutPredictionOptimized(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    if (droppedIndices.empty()) {
        return computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    }
    
    double prediction = baseScore;

    // **Optimization: Use exclusion if only a few trees are dropped**
    if (droppedIndices.size() <= 5) {
        return computeDropoutByExclusion(trees, droppedIndices, sample, rowLength, baseScore);
    }

    // **Optimization: Use inclusion if many trees are dropped**
    return computeDropoutByInclusion(trees, droppedIndices, sample, rowLength, baseScore);
}

// **Exclusion method: compute full then subtract dropped contributions**
double UniformDartStrategy::computeDropoutByExclusion(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    double fullPrediction = computeFullPredictionOptimized(trees, sample, rowLength, baseScore);
    
    for (int idx : droppedIndices) {
        if (idx >= 0 && idx < static_cast<int>(trees.size())) {
            fullPrediction -= computeSingleTreeContribution(trees[idx], sample, rowLength);
        }
    }
    
    return fullPrediction;
}

// **Inclusion method: only sum retained trees**
double UniformDartStrategy::computeDropoutByInclusion(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const double* sample,
    int rowLength,
    double baseScore) const {
    
    static thread_local std::vector<bool> droppedMask;
    const size_t numTrees = trees.size();

    if (droppedMask.size() != numTrees) {
        droppedMask.resize(numTrees);
    }
    std::fill(droppedMask.begin(), droppedMask.end(), false);

    for (int idx : droppedIndices) {
        if (idx >= 0 && idx < static_cast<int>(numTrees)) {
            droppedMask[idx] = true;
        }
    }

    double prediction = baseScore;
    for (size_t i = 0; i < numTrees; ++i) {
        if (!droppedMask[i]) {
            prediction += computeSingleTreeContribution(trees[i], sample, rowLength);
        }
    }
    
    return prediction;
}

// **Inline: single tree contribution**
inline double UniformDartStrategy::computeSingleTreeContribution(
    const RegressionBoostingModel::RegressionTree& tree,
    const double* sample,
    int rowLength) const {
    
    const Node* cur = tree.tree.get();
    while (cur && !cur->isLeaf) {
        const int featIdx = cur->getFeatureIndex();
        const double threshold = cur->getThreshold();
        cur = (sample[featIdx] <= threshold) ? cur->getLeft() : cur->getRight();
    }
    
    const double treePred = cur ? cur->getPrediction() : 0.0;
    return tree.learningRate * tree.weight * treePred;
}

void UniformDartStrategy::updateTreeWeights(
    std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    int newTreeIndex,
    double learningRate) const {
    
    if (!normalizeWeights_ || trees.empty()) {
        return;
    }
    
    const double k = static_cast<double>(droppedIndices.size());
    if (k == 0.0) return;

    // **Optimization: Batch weight updates**
    switch (weightStrategy_) {
        case DartWeightStrategy::NONE:
            // No weight adjustment
            break;
            
        case DartWeightStrategy::MILD: {
            // Mild strategy: slightly increase new tree weight
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                const double adjustmentFactor = 1.0 + 0.05 * k;
                trees[newTreeIndex].weight = learningRate * std::min(adjustmentFactor, 1.2);
            }
            break;
        }
        
        case DartWeightStrategy::ORIGINAL: {
            // Original DART method: aggressive adjustment
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                trees[newTreeIndex].weight = learningRate * (k + 1.0);
            }

            // **Optimization: parallel weight update for dropped trees**
            if (droppedIndices.size() > 10) {
                #pragma omp parallel for schedule(static) if(droppedIndices.size() > 50)
                for (size_t i = 0; i < droppedIndices.size(); ++i) {
                    const int idx = droppedIndices[i];
                    if (idx >= 0 && idx < static_cast<int>(trees.size())) {
                        trees[idx].weight *= (k + 1.0) / k;
                    }
                }
            } else {
                for (int idx : droppedIndices) {
                    if (idx >= 0 && idx < static_cast<int>(trees.size())) {
                        trees[idx].weight *= (k + 1.0) / k;
                    }
                }
            }
            break;
        }
        
        case DartWeightStrategy::EXPERIMENTAL: {
            // Experimental: adaptive weight adjustment
            if (newTreeIndex >= 0 && newTreeIndex < static_cast<int>(trees.size())) {
                const double totalTrees = static_cast<double>(trees.size());
                const double dropRatio = k / totalTrees;
                const double adaptiveFactor = 1.0 + dropRatio * 0.5;
                trees[newTreeIndex].weight = learningRate * adaptiveFactor;
                
                // **Experimental: dynamic learning rate decay**
                const double decayFactor = std::max(0.95, 1.0 - dropRatio * 0.1);
                trees[newTreeIndex].learningRate *= decayFactor;
            }
            break;
        }
    }
}

// **Optimized tree lookup**
bool UniformDartStrategy::isTreeDropped(int treeIndex, 
                                       const std::vector<int>& droppedIndices) const {
    
    // **Optimization: linear search for small lists**
    if (droppedIndices.size() <= 8) {
        for (int idx : droppedIndices) {
            if (idx == treeIndex) return true;
        }
        return false;
    }

    // **Optimization: binary search for large lists**
    // Note: droppedIndices must be sorted for this to work
    return std::binary_search(droppedIndices.begin(), droppedIndices.end(), treeIndex);
}

// **New: Batch dropout prediction (for training)**
void UniformDartStrategy::computeDropoutPredictionsBatch(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    const std::vector<int>& droppedIndices,
    const std::vector<double>& X,
    int rowLength,
    double baseScore,
    std::vector<double>& predictions) const {
    
    const size_t n = predictions.size();
    
    if (droppedIndices.empty()) {
        // **No dropout: batch full prediction**
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            predictions[i] = computeFullPredictionOptimized(
                trees, &X[i * rowLength], rowLength, baseScore);
        }
    } else {
        // **Dropout applied: batch dropout prediction**
        #pragma omp parallel for schedule(static, 256) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            predictions[i] = computeDropoutPredictionOptimized(
                trees, droppedIndices, &X[i * rowLength], rowLength, baseScore);
        }
    }
}

// **New: Adaptive dropout strategy (drop based on tree importance)**
std::vector<int> UniformDartStrategy::selectDroppedTreesAdaptive(
    const std::vector<RegressionBoostingModel::RegressionTree>& trees,
    double dropRate,
    std::mt19937& gen) const {
    
    const int totalTrees = static_cast<int>(trees.size());
    if (totalTrees <= 0 || dropRate <= 0.0) {
        return {};
    }
    
    // **Compute importance weights for each tree**
    std::vector<double> treeWeights(totalTrees);
    for (int i = 0; i < totalTrees; ++i) {
        treeWeights[i] = std::abs(trees[i].weight * trees[i].learningRate);
    }
    
    // **Create weighted distribution**
    std::discrete_distribution<int> dist(treeWeights.begin(), treeWeights.end());
    
    const int numToDrop = static_cast<int>(std::ceil(totalTrees * dropRate));
    std::vector<int> droppedTrees;
    std::vector<bool> alreadyDropped(totalTrees, false);
    
    // **Randomly select trees to drop based on importance**
    for (int i = 0; i < numToDrop && droppedTrees.size() < static_cast<size_t>(totalTrees); ++i) {
        int candidate = dist(gen);
        if (!alreadyDropped[candidate]) {
            droppedTrees.push_back(candidate);
            alreadyDropped[candidate] = true;
        }
    }
    
    return droppedTrees;
}
