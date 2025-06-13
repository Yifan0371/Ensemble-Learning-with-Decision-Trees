// =============================================================================
// include/histogram/PrecomputedHistograms.hpp - Precomputed histogram optimization
// =============================================================================
#pragma once

#include <vector>
#include <string>
#include <algorithm> 
#include <memory>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * Histogram bin structure - optimized memory layout
 */
struct HistogramBin {
    std::vector<int> sampleIndices;     // Sample indices in this bin
    double sum = 0.0;                   // Sum of label values
    double sumSq = 0.0;                 // Sum of squared label values
    int count = 0;                      // Number of samples
    double binStart = 0.0;              // Bin start value
    double binEnd = 0.0;                // Bin end value
    
    // Fast statistics update
    void addSample(int idx, double label) {
        sampleIndices.push_back(idx);
        sum += label;
        sumSq += label * label;
        ++count;
    }
    
    void removeSample(int idx, double label) {
        auto it = std::find(sampleIndices.begin(), sampleIndices.end(), idx);
        if (it != sampleIndices.end()) {
            sampleIndices.erase(it);
            sum -= label;
            sumSq -= label * label;
            --count;
        }
    }
    
    double getMSE() const {
        if (count == 0) return 0.0;
        double mean = sum / count;
        return sumSq / count - mean * mean;
    }
};

/**
 * Feature histogram - all bins for a single feature
 */
struct FeatureHistogram {
    int featureIndex;
    std::vector<HistogramBin> bins;
    std::vector<double> binBoundaries;  // Bin boundary values
    std::string binningType;            // "equal_width", "equal_frequency", "adaptive_ew", "adaptive_eq"
    
    // Prefix sum arrays (for fast range queries)
    std::vector<double> prefixSum;
    std::vector<double> prefixSumSq; 
    std::vector<int> prefixCount;
    
    void updatePrefixArrays() {
        int numBins = static_cast<int>(bins.size());
        prefixSum.resize(numBins + 1, 0.0);
        prefixSumSq.resize(numBins + 1, 0.0);
        prefixCount.resize(numBins + 1, 0);
        
        for (int i = 0; i < numBins; ++i) {
            prefixSum[i + 1] = prefixSum[i] + bins[i].sum;
            prefixSumSq[i + 1] = prefixSumSq[i] + bins[i].sumSq;
            prefixCount[i + 1] = prefixCount[i] + bins[i].count;
        }
    }
    
    // Fast computation of range [startBin, endBin) statistics
    void getRangeStats(int startBin, int endBin, double& sum, double& sumSq, int& count) const {
        sum = prefixSum[endBin] - prefixSum[startBin];
        sumSq = prefixSumSq[endBin] - prefixSumSq[startBin];
        count = prefixCount[endBin] - prefixCount[startBin];
    }
};

/**
 * Precomputed histogram manager - core optimization class
 */
class PrecomputedHistograms {
public:
    explicit PrecomputedHistograms(int numFeatures) : numFeatures_(numFeatures) {
        histograms_.resize(numFeatures);
    }
    
    /**
     * Preprocessing phase: one-time computation of all feature histograms
     */
    void precompute(const std::vector<double>& data,
                    int rowLength,
                    const std::vector<double>& labels,
                    const std::vector<int>& sampleIndices,
                    const std::string& defaultBinningType = "equal_width",
                    int defaultBins = 64);
    
    /**
     * Set custom binning parameters for specific feature
     */
    void setFeatureBinning(int featureIndex, 
                          const std::string& binningType, 
                          int numBins,
                          const std::vector<double>& customBoundaries = {});
    
    /**
     * Fast split finding - based on precomputed histograms
     */
    std::tuple<int, double, double> findBestSplitFast(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,
        const std::vector<int>& nodeIndices,
        double parentMetric,
        const std::vector<int>& candidateFeatures = {}) const;
    
    /**
     * Fast child histogram update - core optimization
     */
    void updateChildHistograms(int featureIndex,
                              double splitThreshold,
                              const std::vector<int>& parentIndices,
                              std::vector<int>& leftIndices,
                              std::vector<int>& rightIndices,
                              FeatureHistogram& leftHist,
                              FeatureHistogram& rightHist) const;
    
    /**
     * Get feature histogram
     */
    const FeatureHistogram& getFeatureHistogram(int featureIndex) const {
        return histograms_[featureIndex];
    }
    
    FeatureHistogram& getFeatureHistogram(int featureIndex) {
        return histograms_[featureIndex];
    }
    
    /**
     * Memory usage statistics
     */
    size_t getMemoryUsage() const;
    
    /**
     * Performance statistics
     */
    struct PerformanceStats {
        double precomputeTimeMs = 0.0;
        double splitFindTimeMs = 0.0;
        double histogramUpdateTimeMs = 0.0;
        int totalSplitQueries = 0;
        int totalHistogramUpdates = 0;
    };
    
    const PerformanceStats& getPerformanceStats() const { return stats_; }
    void resetPerformanceStats() { stats_ = PerformanceStats{}; }

private:
    int numFeatures_;
    std::vector<FeatureHistogram> histograms_;
    mutable PerformanceStats stats_;
    
    // Internal helper methods
    void computeEqualWidthBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              int numBins);
    
    void computeEqualFrequencyBins(int featureIndex,
                                  const std::vector<double>& featureValues,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int numBins);
    
    void computeAdaptiveEWBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              const std::string& rule = "sturges");
    
    void computeAdaptiveEQBins(int featureIndex,
                              const std::vector<double>& featureValues,
                              const std::vector<double>& labels,
                              const std::vector<int>& indices,
                              int minSamplesPerBin = 5,
                              double variabilityThreshold = 0.1);
    
    // Bin allocation helper functions
    int findBin(const FeatureHistogram& hist, double value) const;
    
    // Parallel optimization helper functions
    void parallelBinConstruction(int featureIndex,
                                const std::vector<double>& featureValues,
                                const std::vector<double>& labels,
                                const std::vector<int>& indices,
                                const std::vector<double>& boundaries);
};

/**
 * Histogram cache manager - for node-level caching
 */
class HistogramCache {
public:
    explicit HistogramCache(int maxCacheSize = 1000) : maxCacheSize_(maxCacheSize) {}
    
    bool hasHistogram(const std::vector<int>& nodeIndices, int featureIndex) const;
    
    const FeatureHistogram& getHistogram(const std::vector<int>& nodeIndices, int featureIndex) const;
    
    void cacheHistogram(const std::vector<int>& nodeIndices, 
                       int featureIndex,
                       const FeatureHistogram& histogram);
    
    void clear() { cache_.clear(); }
    
    size_t size() const { return cache_.size(); }

private:
    int maxCacheSize_;
    mutable std::unordered_map<std::string, FeatureHistogram> cache_;
    
    std::string generateKey(const std::vector<int>& nodeIndices, int featureIndex) const;
    void evictOldEntries();
};