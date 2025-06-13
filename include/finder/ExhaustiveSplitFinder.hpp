#pragma once

#include "../tree/ISplitFinder.hpp"
#include <vector>
#include <tuple>

class ExhaustiveSplitFinder : public ISplitFinder {
public:
    // Find best split by checking all possible split points
    std::tuple<int, double, double>
    findBestSplit(const std::vector<double>& data,
                  int                         rowLength,
                  const std::vector<double>&  labels,
                  const std::vector<int>&     indices,
                  double                      currentMetric,
                  const ISplitCriterion&      criterion) const override;
};