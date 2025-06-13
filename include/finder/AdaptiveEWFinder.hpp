#pragma once

#include "tree/ISplitFinder.hpp"
#include <string>

class AdaptiveEWFinder : public ISplitFinder {
public:
    explicit AdaptiveEWFinder(int minBins = 8, int maxBins = 128, 
                             const std::string& rule = std::string("sturges"))
        : minBins_(minBins), maxBins_(maxBins), rule_(rule) {}
    
    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const override;

private:
    int minBins_;
    int maxBins_;
    std::string rule_;
    
    // Enhanced: Optimized adaptive equal-width method
    std::tuple<int, double, double> findBestSplitAdaptiveEWOptimized(
        const std::vector<double>& data,
        int rowLen,
        const std::vector<double>& labels,
        const std::vector<int>& idx,
        double parentMetric,
        const ISplitCriterion& criterion) const;
    
    // Optimized: Fast optimal bin calculation
    int calculateOptimalBinsFast(const std::vector<double>& values) const;
    
    // Retained: Compatibility method
    double calculateIQR(std::vector<double> values) const;
};