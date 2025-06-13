
#include "finder/QuartileSplitFinder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <omp.h>   
std::tuple<int, double, double>
QuartileSplitFinder::findBestSplit(const std::vector<double>& X,   // Feature matrix (row-major)
                                   int                        D,   // Number of features per row
                                   const std::vector<double>& y,   // Labels
                                   const std::vector<int>&    idx, // Current sample indices
                                   double                     parentMetric,
                                   const ISplitCriterion&     crit) const
{
    if (idx.size() < 4) return {-1, 0.0, 0.0};   // Return if insufficient data

    int    bestFeat = -1;
    double bestThr  = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();

    const size_t N = idx.size(); // Number of samples in current node
    const double EPS = 1e-12;    // Epsilon for floating-point comparisons

    /* Iterate over each feature 'f' in parallel */
    #pragma omp parallel for schedule(dynamic)
    for (int f = 0; f < D; ++f) {
        // Each thread maintains its own local best split
        double localBestGain = -std::numeric_limits<double>::infinity();
        double localBestThr  = 0.0;

        // ---- Thread-private buffers ----
        std::vector<double> vals;
        vals.reserve(N);

        std::vector<int> leftBuf, rightBuf;
        leftBuf.reserve(N);
        rightBuf.reserve(N);

        /* -------- Collect current feature values -------- */
        for (int i : idx) {
            vals.emplace_back(X[i * D + f]);
        }
        if (vals.size() < 4) {
            continue;  // Re-check: skip if too few values
        }

        /* -------- Sort once to get quartiles directly -------- */
        std::sort(vals.begin(), vals.end());
        const size_t nVals    = vals.size();
        const double q1       = vals[static_cast<size_t>(0.25 * (nVals - 1))];
        const double q2       = vals[static_cast<size_t>(0.50 * (nVals - 1))];
        const double q3       = vals[static_cast<size_t>(0.75 * (nVals - 1))];

        /* -------- Organize unique thresholds -------- */
        double thrList[3];
        int    thrCnt = 0;
        thrList[thrCnt++] = q1;
        if (std::fabs(q2 - q1) > EPS)  thrList[thrCnt++] = q2;
        if (std::fabs(q3 - q2) > EPS && std::fabs(q3 - q1) > EPS)  thrList[thrCnt++] = q3;

        /* -------- Evaluate each threshold -------- */
        for (int t = 0; t < thrCnt; ++t) {
            const double thr = thrList[t];

            leftBuf.clear();
            rightBuf.clear();
            for (int i : idx) {
                if (X[i * D + f] <= thr)
                    leftBuf.emplace_back(i);
                else
                    rightBuf.emplace_back(i);
            }
            if (leftBuf.empty() || rightBuf.empty()) continue; // Skip if a child node is empty

            const double mL = crit.nodeMetric(y, leftBuf);
            const double mR = crit.nodeMetric(y, rightBuf);
            const double gain = parentMetric -
                                (mL * leftBuf.size() + mR * rightBuf.size()) / static_cast<double>(N);

            if (gain > localBestGain) {
                localBestGain = gain;
                localBestThr  = thr;
            }
        }

        /* Update global best (with lock) */
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