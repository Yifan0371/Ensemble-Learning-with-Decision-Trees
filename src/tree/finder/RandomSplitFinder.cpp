// src/tree/finder/RandomSplitFinder.cpp
#include "finder/RandomSplitFinder.hpp"
#include <limits>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double>
RandomSplitFinder::findBestSplit(const std::vector<double>& X,
                                 int                          D,
                                 const std::vector<double>&   y,
                                 const std::vector<int>&      idx,
                                 double                       parentMetric, // Parent node's impurity metric (e.g., MSE)
                                 const ISplitCriterion&       crit) const // Not directly used for gain calculation in this optimized version
{
    const int nIdx = static_cast<int>(idx.size()); // Number of samples in the current node
    if (nIdx < 2) {
        return {-1, 0.0, 0.0}; // Not enough samples to split
    }

    // Adaptive threshold: Use serial processing for small nodes
    const int PARALLEL_THRESHOLD = 1000;
    bool useParallel = (nIdx >= PARALLEL_THRESHOLD);

    int    globalBestFeat  = -1;
    double globalBestThr   = 0.0;
    double globalBestGain  = -std::numeric_limits<double>::infinity();

    // Prepare seeds for each thread for parallel execution
    int maxThreads = 1;
#ifdef _OPENMP
    maxThreads = omp_get_max_threads();
#endif
    std::vector<uint32_t> threadSeeds(maxThreads);
    {
        // Serially generate unique seeds for each thread
        std::mt19937 seedGen(gen_()); // Use the finder's base generator as initial seed
        std::uniform_int_distribution<uint32_t> seedDist(0, 0xFFFFFFFF);
        for (int t = 0; t < maxThreads; ++t) {
            threadSeeds[t] = seedDist(seedGen);
        }
    }

    // Thread-local storage for best split found by each thread
    std::vector<int>    bestFeatPerThread(maxThreads, -1);
    std::vector<double> bestThrPerThread(maxThreads, 0.0);
    std::vector<double> bestGainPerThread(maxThreads,
                                          -std::numeric_limits<double>::infinity());

    // Lambda function to encapsulate the logic for processing a single feature
    // This will be called by each thread (or serially)
    auto processFeature = [&](int f, int tid) {
        // 1) Extract feature values and corresponding labels for current node samples
        static thread_local std::vector<std::pair<double,double>> vals; // Thread-local buffer
        vals.clear();
        vals.reserve(nIdx);
        for (int i = 0; i < nIdx; ++i) {
            int sampleIdx = idx[i];
            double xv = X[sampleIdx * D + f];
            vals.emplace_back(xv, y[sampleIdx]);
        }

        // 2) Sort by feature value
        std::sort(vals.begin(), vals.end(),
                  [](auto &a, auto &b) { return a.first < b.first; });

        // 3) Construct prefix sum arrays:
        //    prefixSum[i] = sum of labels up to index i-1
        //    prefixSumSq[i] = sum of squared labels up to index i-1
        //    sortedX stores the sorted feature values
        static thread_local std::vector<double> prefixSum, prefixSumSq, sortedX; // Thread-local buffers
        prefixSum.resize(nIdx + 1);
        prefixSumSq.resize(nIdx + 1);
        sortedX.resize(nIdx);
        prefixSum[0]   = 0.0;
        prefixSumSq[0] = 0.0;
        for (int i = 0; i < nIdx; ++i) {
            sortedX[i] = vals[i].first;
            double yi  = vals[i].second;
            prefixSum[i+1]   = prefixSum[i]   + yi;
            prefixSumSq[i+1] = prefixSumSq[i] + yi * yi;
        }

        // 4) Perform k_ random threshold trials based on parentMetric (MSE)
        std::mt19937 localGen(threadSeeds[tid]); // Thread-specific random generator
        std::uniform_real_distribution<double> uni01(0.0, 1.0); // For generating random thresholds

        double vMin = sortedX.front();
        double vMax = sortedX.back();
        if (vMax - vMin < 1e-12) {
            return; // Feature has only one unique value, cannot split
        }

        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        for (int r = 0; r < k_; ++r) {
            double thr = vMin + uni01(localGen) * (vMax - vMin); // Generate random threshold within range
            
            // Binary search to find position 'pos' (first element > thr)
            int pos = int(std::upper_bound(sortedX.begin(), sortedX.end(), thr) - sortedX.begin());
            if (pos == 0 || pos == nIdx) {
                // Split results in an empty child, skip
                continue;
            }
            
            // Left child: [0, pos), Right child: [pos, nIdx)
            double sumL   = prefixSum[pos];
            double sumSqL = prefixSumSq[pos];
            double nL     = static_cast<double>(pos);
            double mL     = sumL / nL;
            double varL   = (sumSqL / nL) - (mL * mL); // Variance (MSE) of left child

            double sumTotal   = prefixSum[nIdx];
            double sumSqTotal = prefixSumSq[nIdx];
            double sumR       = sumTotal - sumL;
            double sumSqR     = sumSqTotal - sumSqL;
            double nR         = static_cast<double>(nIdx - pos);
            double mR         = sumR / nR;
            double varR       = (sumSqR / nR) - (mR * mR); // Variance (MSE) of right child

            // Gain calculation: parent MSE - weighted average of child MSEs
            double msel = varL;
            double mser = varR;
            double gain = parentMetric - (msel * nL + mser * nR) / (double)nIdx;

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = thr;
            }
        }

        // Update thread-local best result for the current feature
        if (localBestGain > bestGainPerThread[tid]) {
            bestGainPerThread[tid] = localBestGain;
            bestFeatPerThread[tid] = f;
            bestThrPerThread[tid]  = localBestThr;
        }
    };

    // **Parallel or serial iteration over features**
    if (useParallel) {
        #pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num(); // Get current thread ID
#endif
            #pragma omp for schedule(dynamic) // Dynamic scheduling for better load balancing
            for (int f = 0; f < D; ++f) {
                processFeature(f, tid); // Each feature is processed by a thread
            }
        }
    } else {
        // Serial iteration for all features
        for (int f = 0; f < D; ++f) {
            processFeature(f, /*tid=*/0); // Pass 0 for the single thread
        }
    }

    // **Reduce local bests from all threads to find the global best**
    for (int t = 0; t < maxThreads; ++t) {
        double gain = bestGainPerThread[t];
        if (gain > globalBestGain) {
            globalBestGain  = gain;
            globalBestFeat  = bestFeatPerThread[t];
            globalBestThr   = bestThrPerThread[t];
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}