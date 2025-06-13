#pragma once

#include "../loss/IRegressionLoss.hpp"
#include <memory>
#include <vector>
#include <string>

class GradientRegressionStrategy {
public:
    explicit GradientRegressionStrategy(
        std::unique_ptr<IRegressionLoss> lossFunc,
        double baseLearningRate = 0.1,
        bool useLineSearch = false)
        : lossFunc_(std::move(lossFunc)),
          baseLearningRate_(baseLearningRate),
          useLineSearch_(useLineSearch) {}

    // Update targets (gradients) for next tree
    void updateTargets(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& targets) const;

    // Compute learning rate (with optional line search)
    double computeLearningRate(
        int ,
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const {
        if (!useLineSearch_) {
            return baseLearningRate_;
        }
        
        return computeOptimalLearningRate(y_true, y_pred, tree_pred);
    }

    // Update predictions with new tree
    void updatePredictions(
        const std::vector<double>& tree_pred,
        double learning_rate,
        std::vector<double>& y_pred) const;

    std::string name() const { return "gradient_regression"; }

    const IRegressionLoss* getLossFunction() const { return lossFunc_.get(); }

    // Compute total loss
    double computeTotalLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const;

private:
    std::unique_ptr<IRegressionLoss> lossFunc_;
    double baseLearningRate_;
    bool useLineSearch_;

    // Optimal learning rate via line search
    double computeOptimalLearningRate(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred) const;

    // Evaluate loss for given learning rate
    double evaluateLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        const std::vector<double>& tree_pred,
        double lr) const;
};