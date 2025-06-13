// =============================================================================
// src/tree/finder/AdaptiveEWFinder.cpp - Precomputed Histogram Optimized Version
// =============================================================================
#include "finder/AdaptiveEWFinder.hpp"
#include "histogram/PrecomputedHistograms.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <iostream>

#include <ostream>
#include <memory>
#ifdef _OPENMP
#include <omp.h>
#endif

// Adaptive Equal-Width Histogram Manager
static thread_local std::unique_ptr<PrecomputedHistograms> g_adaptiveEWManager = nullptr;
static thread_local bool g_adaptiveEWInitialized = false;

static PrecomputedHistograms* getAdaptiveEWManager(int numFeatures) {
    if (!g_adaptiveEWInitialized) {
        g_adaptiveEWManager = std::make_unique<PrecomputedHistograms>(numFeatures);
        g_adaptiveEWInitialized = true;
    }
    return g_adaptiveEWManager.get();
}

// **Optimized Utility Function**
static double calculateIQRFast(std::vector<double>& values) {
    if (values.size() < 4) return 0.0;
    
    // **Optimization: Use nth_element instead of full sort**
    const size_t n = values.size();
    const size_t q1_pos = n / 4;
    const size_t q3_pos = 3 * n / 4;
    
    std::nth_element(values.begin(), values.begin() + q1_pos, values.end());
    double q1 = values[q1_pos];
    
    std::nth_element(values.begin() + q1_pos + 1, values.begin() + q3_pos, values.end());
    double q3 = values[q3_pos];
    
    return q3 - q1;
}

std::tuple<int, double, double>
AdaptiveEWFinder::findBestSplit(const std::vector<double>& data,
                                int                       rowLen,
                                const std::vector<double>&labels,
                                const std::vector<int>&   idx,
                                double                    parentMetric,
                                const ISplitCriterion&    criterion) const {
    
    const size_t N = idx.size();
    if (N < 2) return {-1, 0.0, 0.0};

    // **Core Optimization 1: Use adaptive equal-width precomputed histograms**
    PrecomputedHistograms* histManager = getAdaptiveEWManager(rowLen);
    
    static thread_local bool isFirstCall = true;
    if (isFirstCall) {
        std::vector<int> allIndices(labels.size());
        std::iota(allIndices.begin(), allIndices.end(), 0);
        
        // **Optimization 2: Precompute adaptive equal-width histograms**
        histManager->precompute(data, rowLen, labels, allIndices, "adaptive_ew", 0);
        isFirstCall = false;
        
        std::cout << "AdaptiveEW: Precomputed adaptive equal-width histograms for " 
                  << rowLen << " features" << std::endl;
    }
    
    // **Optimization 3: Fast adaptive split finding**
    auto [bestFeat, bestThr, bestGain] = histManager->findBestSplitFast(
        data, rowLen, labels, idx, parentMetric);
    
    // Fallback optimized method
    if (bestFeat < 0) {
        return findBestSplitAdaptiveEWOptimized(data, rowLen, labels, idx, parentMetric, criterion);
    }
    
    return {bestFeat, bestThr, bestGain};
}

// **Optimized Adaptive Equal-Width Method**
std::tuple<int, double, double>
AdaptiveEWFinder::findBestSplitAdaptiveEWOptimized(const std::vector<double>& data,
                                                   int rowLen,
                                                   const std::vector<double>& labels,
                                                   const std::vector<int>& idx,
                                                   double parentMetric,
                                                   const ISplitCriterion& criterion) const {

    const size_t N = idx.size();
    int globalBestFeat = -1;
    double globalBestThr = 0.0;
    double globalBestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // **Optimization 4: Smart parallel strategy**
    const bool useParallel = (N > 1000 && rowLen > 4);

    if (useParallel) {
        #pragma omp parallel
        {
            int localBestFeat = -1;
            double localBestThr = 0.0;
            double localBestGain = -std::numeric_limits<double>::infinity();

            // **Optimization 5: Thread-local buffers (reduced memory allocation and false sharing)**
            std::vector<double> values;
            std::vector<std::vector<int>> buckets;
            std::vector<int> leftBuf, rightBuf; // RightBuf is not used in the optimized loop, but kept for consistency
            values.reserve(N);
            buckets.reserve(128);  // Max number of bins
            leftBuf.reserve(N);
            rightBuf.reserve(N); // Kept for consistency, but often not explicitly filled

            #pragma omp for schedule(dynamic) nowait
            for (int f = 0; f < rowLen; ++f) {
                // **Optimization 6: Single pass to collect feature values**
                values.clear();
                for (int i : idx) {
                    values.emplace_back(data[i * rowLen + f]);
                }

                if (values.empty()) continue;

                // **Optimization 7: Fast optimal bin calculation**
                int optimalBins = calculateOptimalBinsFast(values);
                if (optimalBins < 2) continue;

                auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
                double vMin = *vMinIt;
                double vMax = *vMaxIt;
                if (std::abs(vMax - vMin) < EPS) continue;

                double binW = (vMax - vMin) / optimalBins;

                // **Optimization 8: Fast bucketing (avoids recalculation)**
                buckets.clear();
                buckets.resize(optimalBins);
                
                for (int i : idx) {
                    double val = data[i * rowLen + f];
                    int b = static_cast<int>((val - vMin) / binW);
                    if (b == optimalBins) b--; // Clamp to last bin if value is max
                    buckets[b].push_back(i);
                }

                // **Optimization 9: Batch split evaluation (reduced loop overhead)**
                leftBuf.clear();
                
                for (int b = 0; b < optimalBins - 1; ++b) {
                    // Accumulate left-side buckets
                    leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
                    if (leftBuf.empty()) continue;

                    size_t leftN = leftBuf.size();
                    size_t rightN = N - leftN;
                    if (rightN == 0) break; // All samples are in left, no split possible

                    // **Optimization 10: Fast MSE calculation (inlined, avoids function call)**
                    // This is a direct calculation for MSE, potentially faster than generic criterion
                    double leftSum = 0.0, leftSumSq = 0.0;
                    double rightSum = 0.0, rightSumSq = 0.0;
                    
                    for (int current_idx : leftBuf) { // Renamed 'idx' to 'current_idx' to avoid conflict with outer 'idx'
                        double val = labels[current_idx];
                        leftSum += val;
                        leftSumSq += val * val;
                    }
                    
                    // Fast right-side statistics calculation (avoids building rightBuf explicitly)
                    for (int k = b + 1; k < optimalBins; ++k) {
                        for (int current_idx : buckets[k]) {
                            double val = labels[current_idx];
                            rightSum += val;
                            rightSumSq += val * val;
                        }
                    }
                    
                    if (rightN > 0) {
                        double leftMSE = leftSumSq / leftN - std::pow(leftSum / leftN, 2);
                        double rightMSE = rightSumSq / rightN - std::pow(rightSum / rightN, 2);
                        double gain = parentMetric - (leftMSE * leftN + rightMSE * rightN) / N;

                        if (gain > localBestGain) {
                            localBestGain = gain;
                            localBestFeat = f;
                            localBestThr = vMin + binW * (b + 1);
                        }
                    }
                }
            }

            #pragma omp critical
            {
                if (localBestGain > globalBestGain) {
                    globalBestGain = localBestGain;
                    globalBestFeat = localBestFeat;
                    globalBestThr = localBestThr;
                }
            }
        }
    } else {
        // **Serial Optimized Version**
        std::vector<double> values;
        std::vector<std::vector<int>> buckets;
        values.reserve(N);
        buckets.reserve(128);

        for (int f = 0; f < rowLen; ++f) {
            values.clear();
            for (int i : idx) {
                values.emplace_back(data[i * rowLen + f]);
            }

            if (values.empty()) continue;

            int optimalBins = calculateOptimalBinsFast(values);
            if (optimalBins < 2) continue;

            auto [vMinIt, vMaxIt] = std::minmax_element(values.begin(), values.end());
            double vMin = *vMinIt;
            double vMax = *vMaxIt;
            if (std::abs(vMax - vMin) < EPS) continue;

            double binW = (vMax - vMin) / optimalBins;

            // Bucketing
            buckets.clear();
            buckets.resize(optimalBins);
            for (int i : idx) {
                double val = data[i * rowLen + f];
                int b = static_cast<int>((val - vMin) / binW);
                if (b == optimalBins) b--;
                buckets[b].push_back(i);
            }

            // Evaluate splits
            std::vector<int> leftBuf;
            leftBuf.reserve(N);

            for (int b = 0; b < optimalBins - 1; ++b) {
                leftBuf.insert(leftBuf.end(), buckets[b].begin(), buckets[b].end());
                if (leftBuf.empty()) continue;

                size_t leftN = leftBuf.size();
                size_t rightN = N - leftN;
                if (rightN == 0) break;

                // Fast statistics calculation
                double leftSum = 0.0, leftSumSq = 0.0;
                double rightSum = 0.0, rightSumSq = 0.0;
                
                for (int current_idx : leftBuf) {
                    double val = labels[current_idx];
                    leftSum += val;
                    leftSumSq += val * val;
                }
                
                for (int k = b + 1; k < optimalBins; ++k) {
                    for (int current_idx : buckets[k]) {
                        double val = labels[current_idx];
                        rightSum += val;
                        rightSumSq += val * val;
                    }
                }
                
                if (rightN > 0) {
                    double leftMSE = leftSumSq / leftN - std::pow(leftSum / leftN, 2);
                    double rightMSE = rightSumSq / rightN - std::pow(rightSum / rightN, 2);
                    double gain = parentMetric - (leftMSE * leftN + rightMSE * rightN) / N;

                    if (gain > globalBestGain) {
                        globalBestGain = gain;
                        globalBestFeat = f;
                        globalBestThr = vMin + binW * (b + 1);
                    }
                }
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}

// **Optimized Optimal Bin Calculation**
int AdaptiveEWFinder::calculateOptimalBinsFast(const std::vector<double>& values) const {
    const int n = static_cast<int>(values.size());
    if (n <= 1) return 1;

    int bins = minBins_;

    if (rule_ == "sturges") {
        bins = static_cast<int>(std::ceil(std::log2(n))) + 1;
    } else if (rule_ == "rice") {
        bins = static_cast<int>(std::ceil(2.0 * std::cbrt(n)));
    } else if (rule_ == "sqrt") {
        bins = static_cast<int>(std::ceil(std::sqrt(n)));
    } else if (rule_ == "freedman_diaconis") {
        // **Optimization: Use fast IQR calculation**
        std::vector<double> valuesCopy = values;  // Needs a modifiable copy
        double iqr = calculateIQRFast(valuesCopy);
        if (iqr > 0.0) {
            auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
            double h = 2.0 * iqr / std::cbrt(n);
            bins = static_cast<int>(std::ceil((*maxIt - *minIt) / h));
        }
    }

    return std::clamp(bins, minBins_, maxBins_);
}

// **Compatibility for old interface**
double AdaptiveEWFinder::calculateIQR(std::vector<double> values) const {
    return calculateIQRFast(values);
}