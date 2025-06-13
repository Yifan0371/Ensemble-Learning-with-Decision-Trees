// AdaptiveEQFinder.cpp
#include "finder/AdaptiveEQFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>   // Added OpenMP header

static double coeffOfVariation(const std::vector<double>& v)
{
    if (v.size() <= 1) return 0.0;
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    double var = 0.0;
    for (double x : v) var += (x - mean) * (x - mean);
    var /= v.size();
    return std::sqrt(var) / (std::fabs(mean) + 1e-12);
}

std::pair<int,int>
AdaptiveEQFinder::calculateOptimalFrequencyParams(const std::vector<double>& v) const
{
    const int n  = static_cast<int>(v.size());
    const double cv = coeffOfVariation(v);

    int bins = (cv < variabilityThreshold_)
             ? std::max(4, std::min(16, static_cast<int>(std::sqrt(n) / 2)))
             : std::max(8, std::min(maxBins_, static_cast<int>(std::sqrt(n))));
    bins = std::clamp(bins, 2, n / std::max(1, minSamplesPerBin_));  // At least 2 bins

    int perBin = std::max(minSamplesPerBin_, n / bins);
    return {bins, perBin};
}

std::tuple<int,double,double>
AdaptiveEQFinder::findBestSplit(const std::vector<double>& data,
                                int                       rowLen,
                                const std::vector<double>&labels,
                                const std::vector<int>&   idx,
                                double                    parentMetric,
                                const ISplitCriterion&    criterion) const
{
    const size_t N = idx.size();
    if (N < static_cast<size_t>(2 * minSamplesPerBin_))
        return {-1, 0.0, 0.0};

    // Initialize global best split
    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const double EPS = 1e-12;

    // Iterate over each feature 'f' in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < rowLen; ++f) {
        // Each thread maintains its own local best
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // 1. Collect current feature values into local vectors in parallel
        std::vector<double> values;
        values.reserve(N);
        for (int i : idx) {
            values.push_back(data[i * rowLen + f]);
        }

        // 2. Calculate adaptive equal-frequency parameters (perBin, bins)
        const auto [bins, perBin] = calculateOptimalFrequencyParams(values);
        if (N < static_cast<size_t>(2 * perBin)) continue;  // Skip if too few samples for this feature

        // 3. Sort indices to get sortedIdx
        std::vector<int> sortedIdx = idx;  // Direct copy
        std::sort(sortedIdx.begin(), sortedIdx.end(),
                  [&](int a, int b) {
                      return data[a * rowLen + f] < data[b * rowLen + f];
                  });

        // 4. Enumerate equal-frequency split points
        for (size_t pivot = perBin; pivot <= N - perBin; pivot += perBin) {
            double vL = data[sortedIdx[pivot - 1] * rowLen + f];
            double vR = data[sortedIdx[pivot]     * rowLen + f];
            if (std::fabs(vR - vL) < EPS) 
                continue;  // Invalid split if values are identical

            // Populate left and right child index buffers
            std::vector<int> leftBuf, rightBuf;
            leftBuf.reserve(pivot);
            rightBuf.reserve(N - pivot);
            leftBuf.assign(sortedIdx.begin(),          sortedIdx.begin() + pivot);
            rightBuf.assign(sortedIdx.begin() + pivot, sortedIdx.end());

            // Ensure each child node has enough samples
            if (leftBuf.size() < static_cast<size_t>(minSamplesPerBin_) ||
                rightBuf.size() < static_cast<size_t>(minSamplesPerBin_))
            {
                continue;
            }

            // Calculate metric for left and right children (criterion.nodeMetric can be parallelized internally)
            double mL = criterion.nodeMetric(labels, leftBuf);
            double mR = criterion.nodeMetric(labels, rightBuf);
            double gain = parentMetric -
                          (mL * leftBuf.size() + mR * rightBuf.size())
                          / static_cast<double>(N);

            // Update local best
            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = 0.5 * (vL + vR);
            }
        }

        // Acquire lock to update global best
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeat = f;
                bestThr  = localBestThr;
            }
        }
    } // End of parallel for loop

    return {bestFeat, bestThr, bestGain};
}