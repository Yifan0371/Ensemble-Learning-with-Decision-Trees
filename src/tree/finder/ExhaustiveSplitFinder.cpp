// src/tree/finder/ExhaustiveSplitFinder.cpp 
#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <vector>
#include <cmath>
#include <tuple>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

std::tuple<int, double, double>
ExhaustiveSplitFinder::findBestSplit(const std::vector<double>& data,
                                     int                       rowLength,
                                     const std::vector<double>& labels,
                                     const std::vector<int>&    indices,
                                     double /*currentMetric*/, // Not used in this implementation (assuming MSE based gain)
                                     const ISplitCriterion&     /*criterion*/) const // Not used in this implementation
{
    const size_t N = indices.size();
    if (N < 2) return {-1, 0.0, 0.0};

    
    double totalSum   = 0.0;
    double totalSumSq = 0.0;
    
    // Choose whether to use parallelization based on data size
    bool useParallel = N > 1000;
    
    if (useParallel) {
        #pragma omp parallel for reduction(+:totalSum,totalSumSq) schedule(static)
        for (size_t i = 0; i < N; ++i) {
            const double y = labels[indices[i]];
            totalSum   += y;
            totalSumSq += y * y;
        }
    } else {
        // For small datasets, use serial computation
        for (size_t i = 0; i < N; ++i) {
            const double y = labels[indices[i]];
            totalSum   += y;
            totalSumSq += y * y;
        }
    }
    
    const double parentMean = totalSum / static_cast<double>(N);
    const double parentMSE  = totalSumSq / static_cast<double>(N) - parentMean * parentMean;

    int    globalBestFeat = -1;
    double globalBestThr  = 0.0;
    double globalBestGain = 0.0;
    constexpr double EPS = 1e-12; // Small epsilon for floating point comparisons

    if (useParallel) {
        // Parallel version for medium to large datasets
        #pragma omp parallel
        {
            // Thread-local variables for best split
            int    localBestFeat = -1;
            double localBestThr  = 0.0;
            double localBestGain = 0.0;
            
            // Thread-local buffer (avoids repeated allocation)
            std::vector<int> localSortedIdx(N);
            
            #pragma omp for schedule(dynamic) nowait // Dynamic scheduling for load balancing, no barrier here
            for (int f = 0; f < rowLength; ++f) {
                /* --- Copy current indices and sort by feature value --- */
                std::copy(indices.begin(), indices.end(), localSortedIdx.begin());
                std::sort(localSortedIdx.begin(), localSortedIdx.end(),
                          [&](int a, int b) {
                              return data[a * rowLength + f] < data[b * rowLength + f];
                          });

                /* --- Single loop to accumulate left subset statistics and evaluate splits immediately --- */
                double leftSum   = 0.0;
                double leftSumSq = 0.0;

                for (size_t i = 0; i < N - 1; ++i) {
                    const int    idx = localSortedIdx[i];
                    const double y   = labels[idx];
                    leftSum   += y;
                    leftSumSq += y * y;

                    /* Check if adjacent samples have different feature values to allow a split */
                    const double currentVal = data[idx * rowLength + f];
                    const double nextVal    = data[localSortedIdx[i + 1] * rowLength + f];

                    if (currentVal + EPS < nextVal) { // Only consider splits between distinct feature values
                        const size_t leftCnt  = i + 1;
                        const size_t rightCnt = N - leftCnt;

                        /* Right subset statistics can be derived from total minus left subset */
                        const double rightSum   = totalSum   - leftSum;
                        const double rightSumSq = totalSumSq - leftSumSq;

                        /* Calculate variance for left and right subsets (MSE as variance) */
                        const double leftMean  = leftSum  / static_cast<double>(leftCnt);
                        const double rightMean = rightSum / static_cast<double>(rightCnt);

                        const double leftMSE  = leftSumSq  / static_cast<double>(leftCnt)  - leftMean  * leftMean;
                        const double rightMSE = rightSumSq / static_cast<double>(rightCnt) - rightMean * rightMean;

                        /* Information Gain (Reduction in MSE) */
                        const double gain = parentMSE -
                                             (leftMSE * static_cast<double>(leftCnt) +
                                              rightMSE * static_cast<double>(rightCnt)) / static_cast<double>(N);

                        if (gain > localBestGain) {
                            localBestGain = gain;
                            localBestFeat = f;
                            localBestThr  = 0.5 * (currentVal + nextVal); // Midpoint as threshold
                        }
                    }
                }
            }
            
            /* --- Thread reduction: Update global best result --- */
            #pragma omp critical // Protect global variables during update
            {
                if (localBestGain > globalBestGain) {
                    globalBestGain = localBestGain;
                    globalBestFeat = localBestFeat;
                    globalBestThr  = localBestThr;
                }
            }
        }
    } else {
        // Serial version for small datasets
        std::vector<int> sortedIdx(N);
        
        for (int f = 0; f < rowLength; ++f) {
            /* --- Copy current indices and sort by feature value --- */
            std::copy(indices.begin(), indices.end(), sortedIdx.begin());
            std::sort(sortedIdx.begin(), sortedIdx.end(),
                      [&](int a, int b) {
                          return data[a * rowLength + f] < data[b * rowLength + f];
                      });

            /* --- Single loop to accumulate left subset statistics and evaluate splits immediately --- */
            double leftSum   = 0.0;
            double leftSumSq = 0.0;

            for (size_t i = 0; i < N - 1; ++i) {
                const int    idx = sortedIdx[i];
                const double y   = labels[idx];
                leftSum   += y;
                leftSumSq += y * y;

                /* Check if adjacent samples have different feature values to allow a split */
                const double currentVal = data[idx * rowLength + f];
                const double nextVal    = data[sortedIdx[i + 1] * rowLength + f];

                if (currentVal + EPS < nextVal) {
                    const size_t leftCnt  = i + 1;
                    const size_t rightCnt = N - leftCnt;

                    /* Right subset statistics can be derived from total minus left subset */
                    const double rightSum   = totalSum   - leftSum;
                    const double rightSumSq = totalSumSq - leftSumSq;

                    /* Calculate variance for left and right subsets (MSE as variance) */
                    const double leftMean  = leftSum  / static_cast<double>(leftCnt);
                    const double rightMean = rightSum / static_cast<double>(rightCnt);

                    const double leftMSE  = leftSumSq  / static_cast<double>(leftCnt)  - leftMean  * leftMean;
                    const double rightMSE = rightSumSq / static_cast<double>(rightCnt) - rightMean * rightMean;

                    /* Information Gain (Reduction in MSE) */
                    const double gain = parentMSE -
                                         (leftMSE * static_cast<double>(leftCnt) +
                                          rightMSE * static_cast<double>(rightCnt)) / static_cast<double>(N);

                    if (gain > globalBestGain) {
                        globalBestGain = gain;
                        globalBestFeat = f;
                        globalBestThr  = 0.5 * (currentVal + nextVal);
                    }
                }
            }
        }
    }

    return {globalBestFeat, globalBestThr, globalBestGain};
}