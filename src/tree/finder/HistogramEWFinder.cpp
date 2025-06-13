#include "finder/HistogramEWFinder.hpp"

// Include header for precomputed histograms
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <memory>
#include <iostream>
#include <numeric>   // For std::iota

#include <ostream>
#ifdef _OPENMP
#include <omp.h>
#endif

// Static thread-local histogram manager (thread-safe singleton pattern)
static thread_local std::unique_ptr<PrecomputedHistograms> g_histogramManager = nullptr;
static thread_local bool g_histogramInitialized = false;

// Access function for the histogram manager
static PrecomputedHistograms* getHistogramManager(int numFeatures) {
    if (!g_histogramInitialized) {
        g_histogramManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_histogramInitialized = true;
    }
    return g_histogramManager.get();
}

std::tuple<int, double, double>
HistogramEWFinder::findBestSplit(const std::vector<double>& X,
                                 int                        D,
                                 const std::vector<double>& y,
                                 const std::vector<int>&    idx,
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const {
    
    if (idx.size() < 2) return {-1, 0.0, 0.0};

    // **Core Optimization 1: Use the precomputed histogram manager**
    PrecomputedHistograms* histManager = getHistogramManager(D);
    
    // **Optimization 2: Precompute all histograms on the first call**
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        // Create full dataset indices for precomputation
        std::vector<int> allIndices(y.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // Precompute equal-width histograms for all features at once
        histManager->precompute(X, D, y, allIndices, "equal_width", bins_);
        isFirstCall = false;
        
        std::cout << "HistogramEW: Precomputed histograms for " << D 
                  << " features with " << bins_ << " bins" << std::endl;
    }
    
    // **Optimization 3: Use fast split finding to avoid re-calculating histograms**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        X, D, y, idx, parentMetric);
    
    // If fast lookup fails, fall back to the traditional (but still optimized) method
    if (bestFeat < 0) {
        return findBestSplitTraditionalOptimized(X, D, y, idx, parentMetric, crit);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **Optimized Traditional Method**: Retained as a fallback, but still optimized
std::tuple<int, double, double>
HistogramEWFinder::findBestSplitTraditionalOptimized(const std::vector<double>& X,
                                                     int D,
                                                     const std::vector<double>& y,
                                                     const std::vector<int>& idx,
                                                     double parentMetric,
                                                     const ISplitCriterion& crit) const {

    const size_t N = idx.size();
    int globalBestFeat = -1;
    double globalBestThr = 0.0;
    double globalBestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12; // Epsilon for floating point comparisons

    // **Optimization 4: Smart parallelization strategy - reduces thread creation overhead**
    const bool useParallel = (N > 1000 && D > 4);
    
    if (useParallel) {
        #pragma omp parallel
        {
            // Thread-local variables for best split
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();
            
            // **Optimization 5: Thread-local histogram buffers (avoids false sharing)**
            std::vector<int> histCnt(bins_);
            std::vector<double> histSum(bins_);
            std::vector<double> histSumSq(bins_);
            std::vector<int> prefixCnt(bins_);
            std::vector<double> prefixSum(bins_);
            std::vector<double> prefixSumSq(bins_);

            #pragma omp for schedule(dynamic) nowait // Dynamic scheduling, no implicit barrier
            for (int f = 0; f < D; ++f) {
                // **Optimization 6: Fast feature range calculation (avoids multiple passes)**
                double vMin = std::numeric_limits<double>::infinity();
                double vMax = -vMin;
                
                for (int i : idx) {
                    double v = X[i * D + f];
                    vMin = std::min(vMin, v);
                    vMax = std::max(vMax, v);
                }
                
                if (std::abs(vMax - vMin) < EPS) continue; // Skip if feature has no variance

                const double binW = (vMax - vMin) / bins_;

                // **Optimization 7: Fast histogram building (single pass)**
                std::fill(histCnt.begin(), histCnt.end(), 0);
                std::fill(histSum.begin(), histSum.end(), 0.0);
                std::fill(histSumSq.begin(), histSumSq.end(), 0.0);

                for (int i : idx) {
                    const double v = X[i * D + f];
                    int b = static_cast<int>((v - vMin) / binW);
                    if (b == bins_) b--; // Clamp to the last bin if value is max
                    const double lbl = y[i];

                    histCnt[b] += 1;
                    histSum[b] += lbl;
                    histSumSq[b] += lbl * lbl;
                }

                // **Optimization 8: Vectorized prefix sum calculation**
                prefixCnt[0] = histCnt[0];
                prefixSum[0] = histSum[0];
                prefixSumSq[0] = histSumSq[0];
                
                for (int b = 1; b < bins_; ++b) {
                    prefixCnt[b] = prefixCnt[b-1] + histCnt[b];
                    prefixSum[b] = prefixSum[b-1] + histSum[b];
                    prefixSumSq[b] = prefixSumSq[b-1] + histSumSq[b];
                }

                // **Optimization 9: Fast split evaluation (avoids redundant calculations)**
                const double totalSum = prefixSum[bins_-1];
                const double totalSumSq = prefixSumSq[bins_-1];
                
                for (int b = 0; b < bins_ - 1; ++b) { // Iterate through possible split points
                    const int leftCnt = prefixCnt[b];
                    const int rightCnt = static_cast<int>(N) - leftCnt;
                    if (leftCnt == 0 || rightCnt == 0) continue; // Must have samples in both children

                    const double leftSum = prefixSum[b];
                    const double leftSumSq = prefixSumSq[b];
                    const double rightSum = totalSum - leftSum;
                    const double rightSumSq = totalSumSq - leftSumSq;

                    // **Optimization 10: Inlined MSE calculation (reduces function calls)**
                    const double leftMean = leftSum / leftCnt;
                    const double rightMean = rightSum / rightCnt;
                    const double leftMSE = leftSumSq / leftCnt - leftMean * leftMean;
                    const double rightMSE = rightSumSq / rightCnt - rightMean * rightMean;
                    
                    const double gain = parentMetric - (leftMSE * leftCnt + rightMSE * rightCnt) / N;

                    if (gain > localBestGain) {
                        localBestGain = gain;
                        localBestFeat = f;
                        // Threshold is the midpoint of the bin boundary
                        localBestThr = vMin + (b + 0.5) * binW; 
                    }
                }
            }
            
            // **Optimization 11: Minimize critical section time**
            #pragma omp critical // Protect global variables during update
            {
                if (localBestGain > globalBestGain) {
                    globalBestGain = localBestGain;
                    globalBestFeat = localBestFeat;
                    globalBestThr = localBestThr;
                }
            }
        }
    } else {
        // **Serial Version - Optimized for smaller datasets**
        std::vector<int> histCnt(bins_);
        std::vector<double> histSum(bins_);
        std::vector<double> histSumSq(bins_);
        
        for (int f = 0; f < D; ++f) {
            // Calculate feature range
            auto [minIt, maxIt] = std::minmax_element(idx.begin(), idx.end(),
                [&](int a, int b) { return X[a * D + f] < X[b * D + f]; });
            
            double vMin = X[*minIt * D + f];
            double vMax = X[*maxIt * D + f];
            
            if (std::abs(vMax - vMin) < EPS) continue;

            const double binW = (vMax - vMin) / bins_;

            // Build histogram
            std::fill(histCnt.begin(), histCnt.end(), 0);
            std::fill(histSum.begin(), histSum.end(), 0.0);
            std::fill(histSumSq.begin(), histSumSq.end(), 0.0);

            for (int i : idx) {
                const double v = X[i * D + f];
                int b = static_cast<int>((v - vMin) / binW);
                if (b == bins_) b--;
                const double lbl = y[i];

                histCnt[b] += 1;
                histSum[b] += lbl;
                histSumSq[b] += lbl * lbl;
            }

            // Evaluate split points
            double leftSum = 0.0, leftSumSq = 0.0;
            int leftCnt = 0;
            
            for (int b = 0; b < bins_ - 1; ++b) {
                leftSum += histSum[b];
                leftSumSq += histSumSq[b];
                leftCnt += histCnt[b];
                
                const int rightCnt = static_cast<int>(N) - leftCnt;
                if (leftCnt == 0 || rightCnt == 0) continue;
                
                double rightSum = 0.0, rightSumSq = 0.0;
                for (int rb = b + 1; rb < bins_; ++rb) { // Sum from remaining bins
                    rightSum += histSum[rb];
                    rightSumSq += histSumSq[rb];
                }

                const double leftMSE = leftSumSq / leftCnt - std::pow(leftSum / leftCnt, 2);
                const double rightMSE = rightSumSq / rightCnt - std::pow(rightSum / rightCnt, 2);
                const double gain = parentMetric - (leftMSE * leftCnt + rightMSE * rightCnt) / N;

                if (gain > globalBestGain) {
                    globalBestGain = gain;
                    globalBestFeat = f;
                    globalBestThr = vMin + (b + 0.5) * binW;
                }
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}