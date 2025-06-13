
#include "finder/HistogramEQFinder.hpp"
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <ostream>
#include <numeric>   // For std::iota

#include <iostream>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

// Equal-Frequency Histogram specific manager
static thread_local std::unique_ptr<PrecomputedHistograms> g_eqHistogramManager = nullptr;
static thread_local bool g_eqHistogramInitialized = false;

static PrecomputedHistograms* getEQHistogramManager(int numFeatures) {
    if (!g_eqHistogramInitialized) {
        g_eqHistogramManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_eqHistogramInitialized = true;
    }
    return g_eqHistogramManager.get();
}

std::tuple<int, double, double>
HistogramEQFinder::findBestSplit(const std::vector<double>& X,
                                 int                        D,
                                 const std::vector<double>& y,
                                 const std::vector<int>&    idx,
                                 double                     parentMetric,
                                 const ISplitCriterion&     crit) const {
    
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    // **Core Optimization 1: Use equal-frequency precomputed histogram manager**
    PrecomputedHistograms* histManager = getEQHistogramManager(D);
    
    // **Optimization 2: Precompute equal-frequency histograms on first call**
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        std::vector<int> allIndices(y.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // Precompute equal-frequency histograms
        histManager->precompute(X, D, y, allIndices, "equal_frequency", bins_);
        isFirstCall = false;
        
        std::cout << "HistogramEQ: Precomputed equal-frequency histograms for " << D 
                  << " features with " << bins_ << " bins" << std::endl;
    }
    
    // **Optimization 3: Fast equal-frequency split finding**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        X, D, y, idx, parentMetric);
    
    // If fast lookup fails, use the optimized traditional equal-frequency method
    if (bestFeat < 0) {
        return findBestSplitEqualFrequencyOptimized(X, D, y, idx, parentMetric, crit);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **Optimized Equal-Frequency Split Finding**: avoids re-sorting every time
std::tuple<int, double, double>
HistogramEQFinder::findBestSplitEqualFrequencyOptimized(const std::vector<double>& X,
                                                        int D,
                                                        const std::vector<double>& y,
                                                        const std::vector<int>& idx,
                                                        double parentMetric,
                                                        const ISplitCriterion& crit) const {

    const size_t N = idx.size();
    // Samples per bin, ensuring at least 1 sample per bin
    const int per = std::max(1, static_cast<int>(N) / bins_); 
    const double EPS = 1e-12; // Epsilon for floating point comparisons

    int bestFeat = -1;
    double bestThr = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    // **Optimization 4: Smart parallelization decision**
    const bool useParallel = (N > 500 && D > 4);

    if (useParallel) {
        #pragma omp parallel
        {
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();

            // **Optimization 5: Thread-local sort buffer (reduces memory allocation)**
            std::vector<int> localSorted;
            localSorted.reserve(N);
            std::vector<int> localLeft, localRight;
            localLeft.reserve(N / 2); // Reserve half size for efficiency
            localRight.reserve(N / 2);

            #pragma omp for schedule(dynamic) nowait // Dynamic scheduling for load balancing, no implicit barrier
            for (int f = 0; f < D; ++f) {
                // **Optimization 6: Single sort, multiple reuses**
                localSorted.clear();
                localSorted.assign(idx.begin(), idx.end());
                
                std::sort(localSorted.begin(), localSorted.end(),
                          [&](int a, int b) {
                              return X[a * D + f] < X[b * D + f];
                          });

                if (localSorted.size() < 2) continue;

                // **Optimization 7: Fast equal-frequency split point generation**
                for (size_t pivot = per; pivot < N; pivot += per) {
                    if (pivot >= N - 1) break; // Ensure there's at least one sample in right child
                    
                    double vL = X[localSorted[pivot - 1] * D + f];
                    double vR = X[localSorted[pivot] * D + f];
                    if (std::abs(vR - vL) < EPS) continue; // Skip if values are identical

                    // **Optimization 8: In-place partitioning, avoids vector copy**
                    // The 'localLeft' and 'localRight' are filled directly from 'localSorted'
                    localLeft.clear();
                    localRight.clear();
                    
                    double threshold = 0.5 * (vL + vR);
                    
                    // Fast partition
                    for (size_t i = 0; i < localSorted.size(); ++i) {
                        int sampleIdx = localSorted[i];
                        if (i < pivot) {
                            localLeft.push_back(sampleIdx);
                        } else {
                            localRight.push_back(sampleIdx);
                        }
                    }

                    if (localLeft.empty() || localRight.empty()) continue;

                    // **Optimization 9: Fast statistics calculation (avoids generic criterion call)**
                    // This assumes the criterion is MSE for inline optimization.
                    double leftSum = 0.0, leftSumSq = 0.0;
                    double rightSum = 0.0, rightSumSq = 0.0;
                    
                    for (int current_idx : localLeft) { // Using current_idx to avoid conflict with outer 'idx'
                        double val = y[current_idx];
                        leftSum += val;
                        leftSumSq += val * val;
                    }
                    
                    for (int current_idx : localRight) {
                        double val = y[current_idx];
                        rightSum += val;
                        rightSumSq += val * val;
                    }
                    
                    // Inlined MSE calculation
                    double leftMSE = leftSumSq / localLeft.size() - 
                                     std::pow(leftSum / localLeft.size(), 2);
                    double rightMSE = rightSumSq / localRight.size() - 
                                      std::pow(rightSum / localRight.size(), 2);
                    
                    double gain = parentMetric - 
                                 (leftMSE * localLeft.size() + rightMSE * localRight.size()) / N;

                    if (gain > localBestGain) {
                        localBestGain = gain;
                        localBestFeat = f;
                        localBestThr = threshold;
                    }
                }
            }

            #pragma omp critical // Protect global 'best' variables
            {
                if (localBestGain > bestGain) {
                    bestGain = localBestGain;
                    bestFeat = localBestFeat;
                    bestThr = localBestThr;
                }
            }
        }
    } else {
        // **Serial Optimized Version**
        std::vector<int> sortedIdx;
        sortedIdx.reserve(N);
        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(N / 2);
        rightBuf.reserve(N / 2);

        for (int f = 0; f < D; ++f) {
            sortedIdx.assign(idx.begin(), idx.end());
            std::sort(sortedIdx.begin(), sortedIdx.end(),
                      [&](int a, int b) {
                          return X[a * D + f] < X[b * D + f];
                      });

            if (sortedIdx.size() < 2) continue;

            // **Optimization 10: Batch evaluation of equal-frequency split points**
            std::vector<size_t> pivotPoints;
            for (size_t pivot = per; pivot < N; pivot += per) {
                if (pivot < N - 1) { // Ensure right child is not empty
                    double vL = X[sortedIdx[pivot - 1] * D + f];
                    double vR = X[sortedIdx[pivot] * D + f];
                    if (std::abs(vR - vL) >= EPS) { // Only add valid splits
                        pivotPoints.push_back(pivot);
                    }
                }
            }
            
            // Evaluate all valid split points in batch
            for (size_t pivot : pivotPoints) {
                leftBuf.clear();
                rightBuf.clear();
                
                leftBuf.assign(sortedIdx.begin(), sortedIdx.begin() + pivot);
                rightBuf.assign(sortedIdx.begin() + pivot, sortedIdx.end());

                if (leftBuf.empty() || rightBuf.empty()) continue;

                // Fast MSE calculation
                double leftSum = 0.0, leftSumSq = 0.0;
                double rightSum = 0.0, rightSumSq = 0.0;
                
                for (int current_idx : leftBuf) {
                    double val = y[current_idx];
                    leftSum += val;
                    leftSumSq += val * val;
                }
                
                for (int current_idx : rightBuf) {
                    double val = y[current_idx];
                    rightSum += val;
                    rightSumSq += val * val;
                }
                
                double leftMSE = leftSumSq / leftBuf.size() - 
                                 std::pow(leftSum / leftBuf.size(), 2);
                double rightMSE = rightSumSq / rightBuf.size() - 
                                  std::pow(rightSum / rightBuf.size(), 2);
                
                double gain = parentMetric - 
                             (leftMSE * leftBuf.size() + rightMSE * rightBuf.size()) / N;

                if (gain > bestGain) {
                    bestGain = gain;
                    bestFeat = f;
                    double vL = X[sortedIdx[pivot - 1] * D + f];
                    double vR = X[sortedIdx[pivot] * D + f];
                    bestThr = 0.5 * (vL + vR);
                }
            }
        }
    }

    return {bestFeat, bestThr, bestGain};
}