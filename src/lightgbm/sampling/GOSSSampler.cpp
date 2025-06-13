// =============================================================================
// src/lightgbm/sampling/GOSSSampler.cpp
// OpenMP Deep Parallel Optimization Version (with header additions and parallel reduction fixes)
// =============================================================================
#include "lightgbm/sampling/GOSSSampler.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <chrono>    // Fix: Include std::chrono
#ifdef _OPENMP
#include <omp.h>
#endif

void GOSSSampler::sample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // If parameters are invalid, perform full sampling
    if (!validateParameters()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
        return;
    }
    // Parallel only if sample size is large enough, otherwise serial
    if (n >= getParallelThreshold()) {
        sampleParallel(gradients, sampleIndices, sampleWeights);
    } else {
        sampleSerial(gradients, sampleIndices, sampleWeights);
    }
}

void GOSSSampler::sampleParallel(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // Parallel construction of (|grad|, idx) pairs
    std::vector<std::pair<double, int>> gradWithIndex(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex[i] = {std::abs(gradients[i]), static_cast<int>(i)};
    }

    // Parallel sort: Can use std::sort(std::execution::par, ...) if C++17 parallel algorithms are supported.
    // Using serial std::sort for now, executed only when n is large.
    std::sort(gradWithIndex.begin(), gradWithIndex.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Calculate sample counts
    size_t topNum = static_cast<size_t>(std::floor(n * topRate_));
    size_t smallGradNum = n - topNum;
    size_t randNum = static_cast<size_t>(std::floor(smallGradNum * otherRate_));
    topNum = std::min(topNum, n);
    randNum = std::min(randNum, smallGradNum);

    sampleIndices.clear();
    sampleWeights.clear();
    sampleIndices.reserve(topNum + randNum);
    sampleWeights.reserve(topNum + randNum);

    // Parallel collection of large gradient sample indices
    std::vector<int> topIndices(topNum);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < topNum; ++i) {
        topIndices[i] = gradWithIndex[i].second;
    }
    sampleIndices.insert(sampleIndices.end(), topIndices.begin(), topIndices.end());
    sampleWeights.insert(sampleWeights.end(), topNum, 1.0);

    // Randomly sample small gradient samples (serial for randomness)
    if (randNum > 0 && smallGradNum > 0) {
        std::vector<int> smallGradPool;
        smallGradPool.reserve(smallGradNum);
        for (size_t i = topNum; i < n; ++i) {
            smallGradPool.push_back(gradWithIndex[i].second);
        }
        std::shuffle(smallGradPool.begin(), smallGradPool.end(), gen_);
        double smallWeight = (1.0 - topRate_) / otherRate_;
        for (size_t i = 0; i < randNum; ++i) {
            sampleIndices.push_back(smallGradPool[i]);
            sampleWeights.push_back(smallWeight);
        }
    }

    // If sampling results in empty, fallback to full sampling
    if (sampleIndices.empty()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

void GOSSSampler::sampleSerial(const std::vector<double>& gradients,
                               std::vector<int>& sampleIndices,
                               std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    std::vector<std::pair<double, int>> gradWithIndex;
    gradWithIndex.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        gradWithIndex.emplace_back(std::abs(gradients[i]), static_cast<int>(i));
    }
    std::sort(gradWithIndex.begin(), gradWithIndex.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    size_t topNum = static_cast<size_t>(std::floor(n * topRate_));
    size_t smallGradNum = n - topNum;
    size_t randNum = static_cast<size_t>(std::floor(smallGradNum * otherRate_));
    topNum = std::min(topNum, n);
    randNum = std::min(randNum, smallGradNum);

    sampleIndices.clear();
    sampleWeights.clear();
    sampleIndices.reserve(topNum + randNum);
    sampleWeights.reserve(topNum + randNum);

    // Keep large gradients
    for (size_t i = 0; i < topNum; ++i) {
        sampleIndices.push_back(gradWithIndex[i].second);
        sampleWeights.push_back(1.0);
    }
    // Randomly sample small gradients
    if (randNum > 0 && smallGradNum > 0) {
        std::vector<int> smallGradPool;
        smallGradPool.reserve(smallGradNum);
        for (size_t i = topNum; i < n; ++i) {
            smallGradPool.push_back(gradWithIndex[i].second);
        }
        std::shuffle(smallGradPool.begin(), smallGradPool.end(), gen_);
        double smallWeight = (1.0 - topRate_) / otherRate_;
        for (size_t i = 0; i < randNum; ++i) {
            sampleIndices.push_back(smallGradPool[i]);
            sampleWeights.push_back(smallWeight);
        }
    }
    if (sampleIndices.empty()) {
        sampleIndices.resize(n);
        sampleWeights.assign(n, 1.0);
        std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
    }
}

void GOSSSampler::sampleWithTiming(const std::vector<double>& gradients,
                                   std::vector<int>& sampleIndices,
                                   std::vector<double>& sampleWeights,
                                   double& samplingTimeMs) const {
    auto start = std::chrono::high_resolution_clock::now();
    sample(gradients, sampleIndices, sampleWeights);
    auto end = std::chrono::high_resolution_clock::now();
    samplingTimeMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void GOSSSampler::adaptiveSample(const std::vector<double>& gradients,
                                 std::vector<int>& sampleIndices,
                                 std::vector<double>& sampleWeights) const {
    size_t n = gradients.size();
    // Calculate mean and std dev first, serial for n < 10000
    double meanGrad = 0.0, stdGrad = 0.0;
    if (n >= 10000) {
        #pragma omp parallel for reduction(+:meanGrad) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            meanGrad += std::abs(gradients[i]);
        }
        meanGrad /= n;
        #pragma omp parallel for reduction(+:stdGrad) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(gradients[i]) - meanGrad;
            stdGrad += diff * diff;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            meanGrad += std::abs(gradients[i]);
        }
        meanGrad /= n;
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(gradients[i]) - meanGrad;
            stdGrad += diff * diff;
        }
    }
    stdGrad = std::sqrt(stdGrad / n);

    double adaptiveTopRate = topRate_;
    double adaptiveOtherRate = otherRate_;
    double cv = (meanGrad > 0) ? stdGrad / meanGrad : 1.0;
    if (cv > 2.0) {
        adaptiveTopRate = std::min(0.5, topRate_ * 1.5);
        adaptiveOtherRate = std::max(0.05, otherRate_ * 0.8);
    } else if (cv < 0.5) {
        adaptiveTopRate = std::max(0.1, topRate_ * 0.8);
        adaptiveOtherRate = std::min(0.3, otherRate_ * 1.2);
    }
    GOSSSampler adaptiveSampler(adaptiveTopRate, adaptiveOtherRate);
    adaptiveSampler.sample(gradients, sampleIndices, sampleWeights);
}

GOSSSampler::SamplingStats GOSSSampler::getSamplingStats(
    const std::vector<double>& gradients,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights) const {
    SamplingStats stats;
    stats.totalSamples = gradients.size();
    stats.selectedSamples = sampleIndices.size();
    stats.samplingRatio = static_cast<double>(stats.selectedSamples) / stats.totalSamples;

    // Initialize stats
    stats.effectiveWeightSum = 0.0;
    stats.maxGradient = 0.0;
    stats.minGradient = std::numeric_limits<double>::max();

    size_t m = sampleIndices.size();
    if (m == 0) {
        stats.effectiveWeightSum = 0.0;
        stats.maxGradient = 0.0;
        stats.minGradient = 0.0;
        return stats;
    }

    // Parallel calculation of effective weight sum
    double localWeightSum = 0.0;
    if (m >= 2000) {
        #pragma omp parallel for reduction(+:localWeightSum) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            localWeightSum += sampleWeights[i];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            localWeightSum += sampleWeights[i];
        }
    }
    stats.effectiveWeightSum = localWeightSum;

    // Parallel calculation of max gradient
    double localMaxGrad = 0.0;
    if (m >= 2000) {
        #pragma omp parallel for reduction(max:localMaxGrad) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad > localMaxGrad) {
                localMaxGrad = grad;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad > localMaxGrad) {
                localMaxGrad = grad;
            }
        }
    }
    stats.maxGradient = localMaxGrad;

    // Parallel calculation of min gradient
    double localMinGrad = std::numeric_limits<double>::max();
    if (m >= 2000) {
        #pragma omp parallel for reduction(min:localMinGrad) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad < localMinGrad) {
                localMinGrad = grad;
            }
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            int idx = sampleIndices[i];
            double grad = std::abs(gradients[idx]);
            if (grad < localMinGrad) {
                localMinGrad = grad;
            }
        }
    }
    stats.minGradient = localMinGrad;

    return stats;
}