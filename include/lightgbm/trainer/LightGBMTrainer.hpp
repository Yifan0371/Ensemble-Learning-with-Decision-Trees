// =============================================================================
// include/lightgbm/trainer/LightGBMTrainer.hpp
// Deep OpenMP parallel optimization version (raised thresholds, pre-allocated member variables)
// =============================================================================
#pragma once

#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/model/LightGBMModel.hpp"
#include "lightgbm/sampling/GOSSSampler.hpp"
#include "lightgbm/feature/FeatureBundler.hpp"
#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include "boosting/loss/IRegressionLoss.hpp"
#include "tree/ITreeTrainer.hpp"
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration of optimized data structures
struct OptimizedFeatureBundles;

/** LightGBM trainer - deep OpenMP parallel optimization version */
class LightGBMTrainer : public ITreeTrainer {
public:
    explicit LightGBMTrainer(const LightGBMConfig& config);

    // ITreeTrainer interface
    void train(const std::vector<double>& data,
               int rowLength,
               const std::vector<double>& labels) override;

    double predict(const double* sample, int rowLength) const override;

    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& mse,
                  double& mae) override;

    // LightGBM specific methods
    const LightGBMModel* getLGBModel() const { return &model_; }
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }

    std::vector<double> getFeatureImportance(int numFeatures) const {
        return calculateFeatureImportance(numFeatures);
    }

private:
    LightGBMConfig config_;
    LightGBMModel model_;
    std::unique_ptr<IRegressionLoss> lossFunction_;
    std::unique_ptr<GOSSSampler> gossSampler_;
    std::unique_ptr<FeatureBundler> featureBundler_;
    std::unique_ptr<LeafwiseTreeBuilder> treeBuilder_;

    // Training data structures
    std::vector<double> trainingLoss_;
    std::vector<FeatureBundle> featureBundles_;

    // Memory pool (pre-allocated, avoid repeated allocation)
    mutable std::vector<double> gradients_;
    mutable std::vector<int> sampleIndices_;
    mutable std::vector<double> sampleWeights_;

    // Private methods
    void preprocessFeaturesOptimized(const std::vector<double>& data,
                                    int rowLength,
                                    size_t sampleSize,
                                    OptimizedFeatureBundles& bundles);
    
    double computeLossOptimized(const std::vector<double>& labels,
                               const std::vector<double>& predictions) const;
    
    void computeGradientsOptimized(const std::vector<double>& labels,
                                  const std::vector<double>& predictions);
    
    void computeAbsGradients(std::vector<double>& absGradients) const;
    
    void normalizeWeights(size_t n);
    
    void prepareFullSample(size_t n);
    
    void updatePredictionsOptimized(const std::vector<double>& data,
                                   int rowLength,
                                   const Node* tree,
                                   std::vector<double>& predictions,
                                   size_t n) const;
    
    bool checkEarlyStop(int currentIter) const;
    void initializeComponents();
    void preprocessFeatures(const std::vector<double>& data,
                            int rowLength,
                            size_t sampleSize);
    void preprocessFeaturesSerial(const std::vector<double>& data,
                                  int rowLength,
                                  size_t sampleSize);
    double computeBaseScore(const std::vector<double>& y) const;
    double computeLossSerial(const std::vector<double>& labels,
                             const std::vector<double>& predictions) const;
    void computeGradientsSerial(const std::vector<double>& labels,
                                const std::vector<double>& predictions);
    std::vector<double> calculateFeatureImportance(int numFeatures) const;

    std::unique_ptr<ISplitCriterion> createCriterion() const;
    std::unique_ptr<ISplitFinder> createOptimalSplitFinder() const;
    std::unique_ptr<ISplitFinder> createHistogramFinder() const;

    double predictSingleTree(const Node* tree,
                             const double* sample,
                             int rowLength) const;
};