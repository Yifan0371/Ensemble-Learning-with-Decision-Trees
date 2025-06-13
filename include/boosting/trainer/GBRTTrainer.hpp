// =============================================================================
// include/boosting/trainer/GBRTTrainer.hpp - Deep parallel optimization version
// =============================================================================
#pragma once

#include "../model/RegressionBoostingModel.hpp"
#include "../strategy/GradientRegressionStrategy.hpp"
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "../dart/IDartStrategy.hpp"
#include <memory>
#include <iostream>
#include <vector>
#include <random>

struct GBRTConfig {
    // Basic parameters
    int numIterations = 100;           
    double learningRate = 0.1;         
    int maxDepth = 6;                  
    int minSamplesLeaf = 1;            
    
    // Tree building parameters
    std::string criterion = "mse";     
    std::string splitMethod = "exhaustive";  
    std::string prunerType = "none";   
    double prunerParam = 0.0;          
    
    // Training control parameters
    bool verbose = true;               
    int earlyStoppingRounds = 0;       
    double tolerance = 1e-7;           
    
    // Sampling parameters
    double subsample = 1.0;
    
    // Line search parameters
    bool useLineSearch = false;
    
    // DART parameters
    bool enableDart = false;
    double dartDropRate = 0.1;
    bool dartNormalize = true;
    bool dartSkipDropForPrediction = false;
    std::string dartStrategy = "uniform";
    uint32_t dartSeed = 42;
    std::string dartWeightStrategy = "mild";
    
    // Enhanced: Parallel optimization parameters
    int parallelThreshold = 1000;      // Minimum samples for parallel execution
    int chunkSize = 2048;              // Parallel chunk size
    bool enableVectorization = true;   // Enable vectorization optimization
    bool enableMemoryPool = true;      // Enable memory pool
};

class GBRTTrainer {
public:
    explicit GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy);
    
    void train(const std::vector<double>& X,
               int rowLength,
               const std::vector<double>& y);
    
    double predict(const double* sample, int rowLength) const;
    
    std::vector<double> predictBatch(
        const std::vector<double>& X, int rowLength) const;
    
    void evaluate(const std::vector<double>& X,
                  int rowLength,
                  const std::vector<double>& y,
                  double& loss,
                  double& mse,
                  double& mae);
    
    const RegressionBoostingModel* getModel() const { return &model_; }
    std::string name() const { return "GBRT_Optimized"; }
    
    const std::vector<double>& getTrainingLoss() const { return trainingLoss_; }
    
    std::vector<double> getFeatureImportance(int numFeatures) const {
        return model_.getFeatureImportance(numFeatures);
    }
    
    void setValidationData(const std::vector<double>& X_val,
                          const std::vector<double>& y_val,
                          int rowLength) {
        X_val_ = X_val;
        y_val_ = y_val;
        valRowLength_ = rowLength;
        hasValidation_ = true;
    }

private:
    GBRTConfig config_;
    std::unique_ptr<GradientRegressionStrategy> strategy_;
    RegressionBoostingModel model_;
    std::vector<double> trainingLoss_;
    std::vector<double> validationLoss_;
    
    // Validation data
    std::vector<double> X_val_;
    std::vector<double> y_val_;
    int valRowLength_;
    bool hasValidation_ = false;
    
    // DART components
    std::unique_ptr<IDartStrategy> dartStrategy_;
    mutable std::mt19937 dartGen_;
    
    // Core optimization methods
    void trainStandardOptimized(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y);
    
    void trainWithDartOptimized(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y);
    
    // Parallel computation methods
    double computeBaseScoreParallel(const std::vector<double>& y) const;
    
    double computeTotalLossParallel(const std::vector<double>& y,
                                   const std::vector<double>& pred) const;
    
    void computeResidualsParallel(const std::vector<double>& y,
                                 const std::vector<double>& pred,
                                 std::vector<double>& residuals) const;
    
    void batchTreePredictOptimized(const SingleTreeTrainer* trainer,
                                  const std::vector<double>& X,
                                  int rowLength,
                                  std::vector<double>& predictions) const;
    
    void updatePredictionsVectorized(const std::vector<double>& treePred,
                                    double lr,
                                    std::vector<double>& predictions) const;
    
    // DART-specific parallel methods
    void computeDartPredictionsParallel(const std::vector<double>& X,
                                       int rowLength,
                                       const std::vector<int>& droppedTrees,
                                       std::vector<double>& predictions) const;
    
    void recomputeFullPredictionsParallel(const std::vector<double>& X,
                                         int rowLength,
                                         std::vector<double>& predictions) const;
    
    // Optimization helper methods
    inline double predictSingleTreeFast(const Node* tree, const double* sample) const;
    
    std::unique_ptr<Node> cloneTreeOptimized(const Node* original) const;
    
    bool shouldEarlyStop(const std::vector<double>& losses, int patience) const;
    
    // Original methods retained
    std::unique_ptr<SingleTreeTrainer> createTreeTrainer() const;
    std::unique_ptr<IDartStrategy> createDartStrategy() const;
    double computeValidationLoss(const std::vector<double>& predictions) const;
    
    // Deprecated old methods - retained for compatibility
    void sampleData(const std::vector<double>& /* X */, int /* rowLength */,
                   const std::vector<double>& /* targets */,
                   std::vector<double>& /* sampledX */,
                   std::vector<double>& /* sampledTargets */) const {}
    
    std::unique_ptr<Node> cloneTree(const Node* original) const {
        return cloneTreeOptimized(original);
    }
};