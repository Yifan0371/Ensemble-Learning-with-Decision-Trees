// =============================================================================
// src/boosting/trainer/GBRTTrainer.cpp - 深度并行优化版本
// =============================================================================
#include "boosting/trainer/GBRTTrainer.hpp"
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"
#include "pruner/NoPruner.hpp"
#include "boosting/dart/UniformDartStrategy.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <memory>
#include <execution>  // C++17并行算法
#ifdef _OPENMP
#include <omp.h>
#endif

GBRTTrainer::GBRTTrainer(const GBRTConfig& config,
                        std::unique_ptr<GradientRegressionStrategy> strategy)
    : config_(config), strategy_(std::move(strategy)), dartGen_(config.dartSeed) {
    
    if (config_.enableDart) {
        dartStrategy_ = createDartStrategy();
        if (config_.verbose) {
            std::cout << "DART enabled with strategy: " << dartStrategy_->name() 
                      << ", drop rate: " << config_.dartDropRate << std::endl;
        }
    }
    
    #ifdef _OPENMP
   
    omp_set_dynamic(1);
    omp_set_max_active_levels(2);  
    if (config_.verbose) {
        std::cout << "GBRT initialized with OpenMP support (" 
                  << omp_get_max_threads() << " threads)" << std::endl;
    }
    #endif
}

void GBRTTrainer::train(const std::vector<double>& X,
                       int rowLength,
                       const std::vector<double>& y) {
    
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    if (config_.enableDart) {
        trainWithDartOptimized(X, rowLength, y);
    } else {
        trainStandardOptimized(X, rowLength, y);
    }
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalEnd - totalStart);
    
    if (config_.verbose) {
        std::cout << "GBRT training completed in " << totalTime.count() 
                  << "ms with " << model_.getTreeCount() << " trees" << std::endl;
        
        #ifdef _OPENMP
        std::cout << "Parallel efficiency: " << std::fixed << std::setprecision(1)
                  << (static_cast<double>(y.size() * config_.numIterations) 
                      / (totalTime.count() * omp_get_max_threads())) 
                  << " samples/(ms*thread)" << std::endl;
        #endif
    }
}


void GBRTTrainer::trainStandardOptimized(const std::vector<double>& X,
                                         int rowLength,
                                         const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training optimized GBRT with " << config_.numIterations 
                  << " iterations..." << std::endl;
    }
    
    const size_t n = y.size();
    const int numThreads = omp_get_max_threads();
    
  
    double baseScore = computeBaseScoreParallel(y);
    model_.setBaseScore(baseScore);
    
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);
  
    std::vector<std::vector<double>> threadLocalPreds(numThreads);
    for (auto& buf : threadLocalPreds) {
        buf.resize(n);
    }
    
    trainingLoss_.reserve(config_.numIterations);
    
 
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
     
        double currentLoss = computeTotalLossParallel(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
        computeResidualsParallel(y, currentPred, residuals);
        
     
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
       
        batchTreePredictOptimized(treeTrainer.get(), X, rowLength, treePred);
        
       
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
     
        updatePredictionsVectorized(treePred, lr, currentPred);
        
      
        auto rootCopy = cloneTreeOptimized(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss 
                      << " | LR: " << lr
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
      
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "Early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
}


void GBRTTrainer::trainWithDartOptimized(const std::vector<double>& X,
                                         int rowLength,
                                         const std::vector<double>& y) {
    
    if (config_.verbose) {
        std::cout << "Training optimized DART GBRT (" << config_.numIterations 
                  << " iterations, drop rate: " << config_.dartDropRate << ")..." << std::endl;
    }
    
    const size_t n = y.size();
    
   
    double baseScore = computeBaseScoreParallel(y);
    model_.setBaseScore(baseScore);
    
    std::vector<double> currentPred(n, baseScore);
    std::vector<double> residuals(n);
    std::vector<double> treePred(n);
    

    std::vector<double> predBeforeDrop(n);
    std::vector<double> predAfterDrop(n);
    
    trainingLoss_.reserve(config_.numIterations);
    

    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();
        
       
        std::vector<int> droppedTrees;
        if (model_.getTreeCount() > 0) {
            droppedTrees = dartStrategy_->selectDroppedTrees(
                static_cast<int>(model_.getTreeCount()), 
                config_.dartDropRate, 
                dartGen_);
        }
        
        if (config_.verbose && iter % 10 == 0 && !droppedTrees.empty()) {
            std::cout << "DART Iter " << iter << ": Dropping " << droppedTrees.size() 
                      << " trees" << std::endl;
        }
        
       
        if (!droppedTrees.empty()) {
            computeDartPredictionsParallel(X, rowLength, droppedTrees, currentPred);
        }
        
        
        double currentLoss = computeTotalLossParallel(y, currentPred);
        trainingLoss_.push_back(currentLoss);
        
      
        computeResidualsParallel(y, currentPred, residuals);
        
      
        auto treeTrainer = createTreeTrainer();
        treeTrainer->train(X, rowLength, residuals);
        
     
        batchTreePredictOptimized(treeTrainer.get(), X, rowLength, treePred);
        
      
        double lr = strategy_->computeLearningRate(iter, y, currentPred, treePred);
        
        auto rootCopy = cloneTreeOptimized(treeTrainer->getRoot());
        model_.addTree(std::move(rootCopy), 1.0, lr);
        
      
        int newTreeIndex = static_cast<int>(model_.getTreeCount()) - 1;
        dartStrategy_->updateTreeWeights(model_.getTrees(), droppedTrees, newTreeIndex, lr);
        
      
        recomputeFullPredictionsParallel(X, rowLength, currentPred);
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);
        
        if (config_.verbose && iter % 10 == 0) {
            std::cout << "DART Iter " << iter 
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Dropped: " << droppedTrees.size() << " trees"
                      << " | Time: " << iterTime.count() << "ms" << std::endl;
        }
        
        
        if (config_.earlyStoppingRounds > 0 && 
            shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) {
            if (config_.verbose) {
                std::cout << "DART early stopping at iteration " << iter << std::endl;
            }
            break;
        }
    }
}


double GBRTTrainer::computeBaseScoreParallel(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static, 2048) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}


double GBRTTrainer::computeTotalLossParallel(const std::vector<double>& y,
                                            const std::vector<double>& pred) const {
    const size_t n = y.size();
    double totalLoss = 0.0;
    
    
    #pragma omp parallel for reduction(+:totalLoss) schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        totalLoss += strategy_->getLossFunction()->loss(y[i], pred[i]);
    }
    
    return totalLoss / n;
}


void GBRTTrainer::computeResidualsParallel(const std::vector<double>& y,
                                          const std::vector<double>& pred,
                                          std::vector<double>& residuals) const {
    const size_t n = y.size();
    
   
    #pragma omp parallel for schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        residuals[i] = strategy_->getLossFunction()->gradient(y[i], pred[i]);
    }
}


void GBRTTrainer::batchTreePredictOptimized(const SingleTreeTrainer* trainer,
                                           const std::vector<double>& X,
                                           int rowLength,
                                           std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    
    #pragma omp parallel for schedule(static, 1024) if(n > 500)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = trainer->predict(&X[i * rowLength], rowLength);
    }
}


void GBRTTrainer::updatePredictionsVectorized(const std::vector<double>& treePred,
                                              double lr,
                                              std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
   
    #pragma omp parallel for schedule(static, 4096) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] += lr * treePred[i];
    }
}


void GBRTTrainer::computeDartPredictionsParallel(const std::vector<double>& X,
                                                 int rowLength,
                                                 const std::vector<int>& droppedTrees,
                                                 std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    if (droppedTrees.empty()) return;
    
    
    if (droppedTrees.size() <= 3) {
        for (int treeIdx : droppedTrees) {
            if (treeIdx >= 0 && treeIdx < static_cast<int>(model_.getTrees().size())) {
                const auto& tree = model_.getTrees()[treeIdx];
                
                #pragma omp parallel for schedule(static, 1024) if(n > 500)
                for (size_t i = 0; i < n; ++i) {
                    double treePred = predictSingleTreeFast(tree.tree.get(), &X[i * rowLength]);
                    predictions[i] -= tree.learningRate * tree.weight * treePred;
                }
            }
        }
    } else {
        
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), droppedTrees, sample, rowLength, model_.getBaseScore());
        }
    }
}

void GBRTTrainer::recomputeFullPredictionsParallel(const std::vector<double>& X,
                                                   int rowLength,
                                                   std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 512) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        predictions[i] = model_.predict(&X[i * rowLength], rowLength);
    }
}


inline double GBRTTrainer::predictSingleTreeFast(const Node* tree, const double* sample) const {
    const Node* cur = tree;
    while (cur && !cur->isLeaf) {
        const int featIdx = cur->getFeatureIndex();
        const double threshold = cur->getThreshold();
        cur = (sample[featIdx] <= threshold) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}


std::unique_ptr<Node> GBRTTrainer::cloneTreeOptimized(const Node* original) const {
    if (!original) return nullptr;
    
    
    auto root = std::make_unique<Node>();
    
    
    std::function<void(Node*, const Node*)> cloneNode = [&](Node* dest, const Node* src) {
        dest->isLeaf = src->isLeaf;
        dest->samples = src->samples;
        dest->metric = src->metric;
        
        if (src->isLeaf) {
            dest->makeLeaf(src->getPrediction(), src->getNodePrediction());
        } else {
            dest->makeInternal(src->getFeatureIndex(), src->getThreshold());
            if (src->getLeft()) {
                dest->leftChild = std::make_unique<Node>();
                cloneNode(dest->leftChild.get(), src->getLeft());
            }
            if (src->getRight()) {
                dest->rightChild = std::make_unique<Node>();
                cloneNode(dest->rightChild.get(), src->getRight());
            }
        }
    };
    
    cloneNode(root.get(), original);
    return root;
}


std::vector<double> GBRTTrainer::predictBatch(
    const std::vector<double>& X, int rowLength) const {
    
    const size_t n = X.size() / rowLength;
    std::vector<double> predictions;
    predictions.reserve(n);
    
    if (config_.enableDart && dartStrategy_) {
       
        predictions.resize(n);
        #pragma omp parallel for schedule(static, 512) if(n > 1000)
        for (size_t i = 0; i < n; ++i) {
            const double* sample = &X[i * rowLength];
            predictions[i] = dartStrategy_->computeDropoutPrediction(
                model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
        }
    } else {
       
        predictions = model_.predictBatch(X, rowLength);
    }
    
    return predictions;
}


void GBRTTrainer::evaluate(const std::vector<double>& X,
                          int rowLength,
                          const std::vector<double>& y,
                          double& loss,
                          double& mse,
                          double& mae) {
    auto predictions = predictBatch(X, rowLength);
    const size_t n = y.size();
    
   
    loss = strategy_->computeTotalLoss(y, predictions);
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 2048) if(n > 2000)
    for (size_t i = 0; i < n; ++i) {
        double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}


bool GBRTTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    
    const double currentLoss = losses.back();
    const auto recentStart = losses.end() - patience - 1;
    const auto recentEnd = losses.end() - 1;
    
    const double bestLoss = *std::min_element(recentStart, recentEnd);
    return currentLoss >= bestLoss - config_.tolerance;
}


std::unique_ptr<SingleTreeTrainer> GBRTTrainer::createTreeTrainer() const {
    auto criterion = std::make_unique<MSECriterion>();
    auto finder = std::make_unique<ExhaustiveSplitFinder>();
    auto pruner = std::make_unique<NoPruner>();
    
    return std::make_unique<SingleTreeTrainer>(
        std::move(finder), std::move(criterion), std::move(pruner),
        config_.maxDepth, config_.minSamplesLeaf);
}

std::unique_ptr<IDartStrategy> GBRTTrainer::createDartStrategy() const {
    if (config_.dartStrategy == "uniform") {
        return std::make_unique<UniformDartStrategy>(
            config_.dartNormalize, 
            config_.dartSkipDropForPrediction);
    } else {
        throw std::invalid_argument("Unsupported DART strategy: " + config_.dartStrategy);
    }
}

double GBRTTrainer::predict(const double* sample, int rowLength) const {
    if (config_.enableDart && dartStrategy_) {
        return dartStrategy_->computeDropoutPrediction(
            model_.getTrees(), {}, sample, rowLength, model_.getBaseScore());
    } else {
        return model_.predict(sample, rowLength);
    }
}