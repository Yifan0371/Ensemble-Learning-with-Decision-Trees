// =============================================================================
// include/lightgbm/sampling/GOSSSampler.hpp
// Deep OpenMP parallel optimization version (threshold adjustment, reduced repeated allocation, conditional parallel sorting)
// =============================================================================
#pragma once

#include <vector>
#include <random>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

/** GOSS (Gradient-based One-Side Sampling) sampler */
class GOSSSampler {
public:
    explicit GOSSSampler(double topRate = 0.2, double otherRate = 0.1, uint32_t seed = 42)
        : topRate_(topRate), otherRate_(otherRate), gen_(seed) {}

    /** 
     * Execute GOSS sampling
     * @param gradients Gradient array
     * @param sampleIndices Output: sampled sample indices
     * @param sampleWeights Output: sampling weights (small gradient samples need amplified weights)
     */
    void sample(const std::vector<double>& gradients,
                std::vector<int>& sampleIndices,
                std::vector<double>& sampleWeights) const;

    /** 
     * GOSS sampling with performance monitoring
     */
    void sampleWithTiming(const std::vector<double>& gradients,
                          std::vector<int>& sampleIndices,
                          std::vector<double>& sampleWeights,
                          double& samplingTimeMs) const;

    /** 
     * Adaptive GOSS sampling
     */
    void adaptiveSample(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;

    /** Sampling statistics */
    struct SamplingStats {
        size_t totalSamples;        // Total sample count
        size_t selectedSamples;     // Selected sample count
        double samplingRatio;       // Sampling ratio
        double effectiveWeightSum;  // Effective weight sum
        double maxGradient;         // Maximum gradient value
        double minGradient;         // Minimum gradient value
    };

    /** 
     * Get sampling statistics
     */
    SamplingStats getSamplingStats(const std::vector<double>& gradients,
                                   const std::vector<int>& sampleIndices,
                                   const std::vector<double>& sampleWeights) const;

    /** Get current parameters */
    double getTopRate() const { return topRate_; }
    double getOtherRate() const { return otherRate_; }

    /** Update sampling parameters */
    void updateRates(double topRate, double otherRate) {
        topRate_ = topRate;
        otherRate_ = otherRate;
    }

    /** Get recommended parallel threshold */
    static size_t getParallelThreshold() { return 10000; }
    /** Calculate theoretical sampling ratio */
    double getTheoreticalSamplingRatio() const {
        return topRate_ + (1.0 - topRate_) * otherRate_;
    }

private:
    double topRate_;      // Large gradient retention ratio
    double otherRate_;    // Small gradient sampling ratio  
    mutable std::mt19937 gen_;

    /** Parallel GOSS sampling for large datasets */
    void sampleParallel(const std::vector<double>& gradients,
                        std::vector<int>& sampleIndices,
                        std::vector<double>& sampleWeights) const;

    /** Serial GOSS sampling for small datasets */
    void sampleSerial(const std::vector<double>& gradients,
                      std::vector<int>& sampleIndices,
                      std::vector<double>& sampleWeights) const;

    /** Validate sampling parameters */
    bool validateParameters() const {
        return topRate_ > 0.0 && topRate_ < 1.0 &&
               otherRate_ > 0.0 && otherRate_ < 1.0 &&
               (topRate_ + otherRate_) <= 1.0;
    }
};