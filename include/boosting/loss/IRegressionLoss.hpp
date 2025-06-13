#pragma once

#include <vector>
#include <string>
#include <chrono>

class IRegressionLoss {
public:
    virtual ~IRegressionLoss() = default;
    
    // Basic loss function interface
    virtual double loss(double y_true, double y_pred) const = 0;
    virtual double gradient(double y_true, double y_pred) const = 0;
    virtual double hessian(double y_true, double y_pred) const = 0;
    
    // Batch computation methods
    virtual void computeGradientsHessians(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients,
        std::vector<double>& hessians) const;
    
    virtual std::string name() const = 0;
    virtual bool supportsSecondOrder() const { return false; }
    
    // Enhanced batch processing methods
    virtual double computeBatchLoss(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred) const;
    
    virtual void computeBatchGradients(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients) const;
    
    virtual void computeGradientsVectorized(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        std::vector<double>& gradients) const;
    
    virtual double computeBatchLossWithTiming(
        const std::vector<double>& y_true,
        const std::vector<double>& y_pred,
        double& computeTimeMs) const;
    
    // Performance estimation methods
    virtual size_t estimateComputeOps(size_t sampleCount) const {
        return sampleCount; 
    }
    
    virtual size_t getParallelThreshold() const {
        return 2000; 
    }
};