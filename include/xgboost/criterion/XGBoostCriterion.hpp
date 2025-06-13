#pragma once

#include "tree/ISplitCriterion.hpp"
#include <vector>

class XGBoostCriterion : public ISplitCriterion {
public:
    explicit XGBoostCriterion(double lambda = 1.0) : lambda_(lambda) {}
    
    double nodeMetric(const std::vector<double>& labels,
                      const std::vector<int>& indices) const override {
        // Not used in XGBoost - using structure score instead
        return 0.0;
    }
    
    // Compute structure score for XGBoost
    double computeStructureScore(double G, double H) const {
        // Structure score = -0.5 * G^2 / (H + lambda)
        return 0.5 * (G * G) / (H + lambda_);
    }
    
    // Compute split gain for XGBoost
    double computeSplitGain(double Gl, double Hl,
                                double Gr, double Hr,
                                double Gp, double Hp,
                                double gamma) const {
        double gain =
            computeStructureScore(Gl, Hl) +
            computeStructureScore(Gr, Hr) -
            computeStructureScore(Gp, Hp);
        return gain - gamma;
    }

    // Compute optimal leaf weight
    double computeLeafWeight(double G, double H) const {
        return -G / (H + lambda_);
    }
    
    double getLambda() const { return lambda_; }

private:
    double lambda_;  // L2 regularization parameter
};