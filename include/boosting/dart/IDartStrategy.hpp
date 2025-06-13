#pragma once

#include "boosting/model/RegressionBoostingModel.hpp"
#include <vector>
#include <memory>
#include <random>

class IDartStrategy {
public:
    virtual ~IDartStrategy() = default;
    
    // Select trees to drop during training
    virtual std::vector<int> selectDroppedTrees(
        int totalTrees, 
        double dropRate,
        std::mt19937& gen) const = 0;
    
    // Compute prediction with dropout
    virtual double computeDropoutPrediction(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const = 0;
    
    // Update tree weights after dropout
    virtual void updateTreeWeights(
        std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        int newTreeIndex,
        double learningRate) const = 0;
    
    // Get strategy name
    virtual std::string name() const = 0;
};