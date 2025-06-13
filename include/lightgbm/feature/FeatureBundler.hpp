// =============================================================================
// include/lightgbm/feature/FeatureBundler.hpp - Optimized version
// =============================================================================
#pragma once

#include <vector>
#include <unordered_set>

struct FeatureBundle {
    std::vector<int> features;        
    std::vector<double> offsets;      
    int totalBins;                    
};

class FeatureBundler {
public:
    explicit FeatureBundler(int maxBin = 255, double maxConflictRate = 0.0)
        : maxBin_(maxBin), maxConflictRate_(maxConflictRate) {}
    
    // Main methods
    void createBundles(const std::vector<double>& data,
                      int rowLength, 
                      size_t sampleSize,
                      std::vector<FeatureBundle>& bundles) const;
    
    std::pair<int, double> transformFeature(int originalFeature, 
                                           double value,
                                           const std::vector<FeatureBundle>& bundles) const;

    // Compatibility methods (retain old interface)
    double calculateConflictRate(const std::vector<double>& data,
                               int rowLength, size_t sampleSize,
                               int feat1, int feat2) const;
    
    void buildConflictGraph(const std::vector<double>& data,
                          int rowLength, size_t sampleSize,
                          std::vector<std::vector<double>>& conflictMatrix) const;

private:
    int maxBin_;
    double maxConflictRate_;
    
    // Optimized internal methods
    double calculateConflictRateOptimized(const std::vector<double>& data,
                                        int rowLength, size_t sampleSize,
                                        int feat1, int feat2) const;
    
    std::pair<int, double> transformFeatureValue(double value, double offset) const;
};