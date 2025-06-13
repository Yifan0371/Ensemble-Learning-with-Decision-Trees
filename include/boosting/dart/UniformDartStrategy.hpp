#pragma once

#include "IDartStrategy.hpp"
#include <unordered_set>

enum class DartWeightStrategy {
    NONE,           
    MILD,           
    ORIGINAL,       
    EXPERIMENTAL    
};

class UniformDartStrategy : public IDartStrategy {
public:
    explicit UniformDartStrategy(bool normalizeWeights = true, 
                                bool skipDropForPrediction = false,
                                DartWeightStrategy weightStrategy = DartWeightStrategy::MILD)
        : normalizeWeights_(normalizeWeights), 
          skipDropForPrediction_(skipDropForPrediction),
          weightStrategy_(weightStrategy) {}
    
    // Original interface methods
    std::vector<int> selectDroppedTrees(int totalTrees, double dropRate, 
                                       std::mt19937& gen) const override;
    
    double computeDropoutPrediction(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const override;
    
    void updateTreeWeights(
        std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        int newTreeIndex,
        double learningRate) const override;
    
    std::string name() const override { return "uniform_dart"; }
    
    // Configuration methods
    void setNormalizeWeights(bool normalize) { normalizeWeights_ = normalize; }
    void setSkipDropForPrediction(bool skip) { skipDropForPrediction_ = skip; }

    // Enhanced prediction methods
    double computeFullPredictionOptimized(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    double computeDropoutPredictionOptimized(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    // Smart dropout strategy methods
    double computeDropoutByExclusion(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    double computeDropoutByInclusion(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const double* sample,
        int rowLength,
        double baseScore) const;
    
    // Helper methods
    double computeSingleTreeContribution(
        const RegressionBoostingModel::RegressionTree& tree,
        const double* sample,
        int rowLength) const;
    
    // Batch processing methods
    void computeDropoutPredictionsBatch(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        const std::vector<int>& droppedIndices,
        const std::vector<double>& X,
        int rowLength,
        double baseScore,
        std::vector<double>& predictions) const;
    
    // Adaptive dropout selection
    std::vector<int> selectDroppedTreesAdaptive(
        const std::vector<RegressionBoostingModel::RegressionTree>& trees,
        double dropRate,
        std::mt19937& gen) const;

private:
    bool normalizeWeights_;
    bool skipDropForPrediction_;
    DartWeightStrategy weightStrategy_;
    
    void updateTreeWeightsStrategy(std::vector<RegressionBoostingModel::RegressionTree>& trees,
                                  const std::vector<int>& droppedIndices,
                                  int newTreeIndex,
                                  double learningRate) const;
    
    bool isTreeDropped(int treeIndex, const std::vector<int>& droppedIndices) const;
};