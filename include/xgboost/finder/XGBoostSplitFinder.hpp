#pragma once

#include "tree/ISplitFinder.hpp"
#include "xgboost/criterion/XGBoostCriterion.hpp"
#include <vector>
#include <tuple>

class XGBoostSplitFinder : public ISplitFinder {
public:
    explicit XGBoostSplitFinder(double gamma = 0.0, int minChildWeight = 1)
        : gamma_(gamma), minChildWeight_(minChildWeight) {}

    std::tuple<int, double, double> findBestSplit(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& labels,
        const std::vector<int>& indices,
        double currentMetric,
        const ISplitCriterion& criterion) const override;

    // XGBoost-specific split finding
    std::tuple<int, double, double> findBestSplitXGB(
        const std::vector<double>& data,
        int rowLength,
        const std::vector<double>& gradients,
        const std::vector<double>& hessians,
        const std::vector<char>& nodeMask,
        const std::vector<std::vector<int>>& sortedIndicesAll,
        const XGBoostCriterion& xgbCriterion) const;

    double getGamma() const { return gamma_; }
    int getMinChildWeight() const { return minChildWeight_; }

private:
    double gamma_;          // Minimum split gain
    int minChildWeight_;    // Minimum child weight
};