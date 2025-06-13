// =============================================================================
// src/tree/ensemble/BaggingTrainer.cpp - Optimized Version (avoiding vector copy and new)
// =============================================================================
#include "ensemble/BaggingTrainer.hpp"

// Criteria
#include "criterion/MSECriterion.hpp"
#include "criterion/MAECriterion.hpp"
#include "criterion/HuberCriterion.hpp"
#include "criterion/QuantileCriterion.hpp"
#include "criterion/LogCoshCriterion.hpp"
#include "criterion/PoissonCriterion.hpp"

// Split Finders
#include "finder/ExhaustiveSplitFinder.hpp"
#include "finder/RandomSplitFinder.hpp"
#include "finder/QuartileSplitFinder.hpp"
#include "finder/HistogramEWFinder.hpp"
#include "finder/HistogramEQFinder.hpp"
#include "finder/AdaptiveEWFinder.hpp"
#include "finder/AdaptiveEQFinder.hpp"

// Pruners
#include "pruner/NoPruner.hpp"
#include "pruner/MinGainPrePruner.hpp"
#include "pruner/CostComplexityPruner.hpp"
#include "pruner/ReducedErrorPruner.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include <functional>
#include <atomic>
#ifdef _OPENMP
#include <omp.h>
#endif

BaggingTrainer::BaggingTrainer(int numTrees,
                               double sampleRatio,
                               int maxDepth,
                               int minSamplesLeaf,
                               const std::string& criterion,
                               const std::string& splitMethod,
                               const std::string& prunerType,
                               double prunerParam,
                               uint32_t seed)
    : numTrees_(numTrees),
      sampleRatio_(sampleRatio),
      maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf),
      criterion_(criterion),
      splitMethod_(splitMethod),
      prunerType_(prunerType),
      prunerParam_(prunerParam),
      gen_(seed) {
    
    trees_.reserve(numTrees_);
    oobIndices_.reserve(numTrees_);
}

std::unique_ptr<ISplitFinder> BaggingTrainer::createSplitFinder() const {
    const std::string& method = splitMethod_;
    
    if (method == "exhaustive" || method == "exact") {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
    else if (method == "random" || method.find("random:") == 0) {
        int k = 10;
        const auto pos = method.find(':');
        if (pos != std::string::npos) {
            k = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<RandomSplitFinder>(k);
    }
    else if (method == "quartile") {
        return std::make_unique<QuartileSplitFinder>();
    }
    else if (method == "histogram_ew" || method.find("histogram_ew:") == 0) {
        int bins = 64;
        const auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEWFinder>(bins);
    }
    else if (method == "histogram_eq" || method.find("histogram_eq:") == 0) {
        int bins = 64;
        const auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEQFinder>(bins);
    }
    else if (method == "adaptive_ew" || method.find("adaptive_ew:") == 0) {
        std::string rule = "sturges";
        const auto pos = method.find(':');
        if (pos != std::string::npos) {
            rule = method.substr(pos + 1);
        }
        return std::make_unique<AdaptiveEWFinder>(8, 128, rule);
    }
    else if (method == "adaptive_eq") {
        return std::make_unique<AdaptiveEQFinder>(5, 64, 0.1);
    }
    else {
        return std::make_unique<ExhaustiveSplitFinder>();
    }
}

std::unique_ptr<ISplitCriterion> BaggingTrainer::createCriterion() const {
    const std::string& crit = criterion_;
    
    if (crit == "mae")
        return std::make_unique<MAECriterion>();
    else if (crit == "huber")
        return std::make_unique<HuberCriterion>();
    else if (crit.rfind("quantile", 0) == 0) {
        double tau = 0.5;
        const auto pos = crit.find(':');
        if (pos != std::string::npos)
            tau = std::stod(crit.substr(pos + 1));
        return std::make_unique<QuantileCriterion>(tau);
    }
    else if (crit == "logcosh")
        return std::make_unique<LogCoshCriterion>();
    else if (crit == "poisson")
        return std::make_unique<PoissonCriterion>();
    else
        return std::make_unique<MSECriterion>();
}

std::unique_ptr<IPruner> BaggingTrainer::createPruner(const std::vector<double>& X_val,
                                                     int rowLength,
                                                     const std::vector<double>& y_val) const {
    if (prunerType_ == "mingain") {
        return std::make_unique<MinGainPrePruner>(prunerParam_);
    }
    else if (prunerType_ == "cost_complexity") {
        return std::make_unique<CostComplexityPruner>(prunerParam_);
    }
    else if (prunerType_ == "reduced_error") {
        if (X_val.empty() || y_val.empty()) {
            return std::make_unique<NoPruner>();
        }
        return std::make_unique<ReducedErrorPruner>(X_val, rowLength, y_val);
    }
    else {
        return std::make_unique<NoPruner>();
    }
}

// Optimized Bootstrap sampling: avoids vector copies
void BaggingTrainer::bootstrapSample(int dataSize,
                                     std::vector<int>& sampleIndices,
                                     std::vector<int>& oobIndices,
                                     std::mt19937& localGen) const {
    const int sampleSize = static_cast<int>(dataSize * sampleRatio_);
    
    // Pre-allocate to exact size to prevent reallocations
    sampleIndices.clear();
    sampleIndices.reserve(sampleSize);
    
    // Use a bitmap for efficient sampling status tracking
    std::vector<bool> sampledBits(dataSize, false);
    
    // Perform bootstrap sampling (sampling with replacement)
    std::uniform_int_distribution<int> dist(0, dataSize - 1);
    for (int i = 0; i < sampleSize; ++i) {
        const int idx = dist(localGen);
        sampleIndices.push_back(idx);
        sampledBits[idx] = true;
    }
    
    // Identify out-of-bag samples
    oobIndices.clear();
    oobIndices.reserve(dataSize - sampleSize);
    for (int i = 0; i < dataSize; ++i) {
        if (!sampledBits[i]) {
            oobIndices.push_back(i);
        }
    }
}

// Zero-copy data passing: uses references and move semantics
void BaggingTrainer::extractSubsetOptimized(const std::vector<double>& originalData,
                                           int rowLength,
                                           const std::vector<double>& originalLabels,
                                           const std::vector<int>& indices,
                                           std::vector<double>& subData,
                                           std::vector<double>& subLabels) const {
    
    const int featCount = rowLength;
    const size_t totalSize = indices.size() * featCount;
    
    // Pre-allocate exact size to prevent reallocations
    subData.resize(totalSize);
    subLabels.resize(indices.size());
    
    // Optimized bulk copy for better cache locality
    #pragma omp parallel for schedule(static) if(indices.size() > 500)
    for (size_t i = 0; i < indices.size(); ++i) {
        const int idx = indices[i];
        
        // Use contiguous memory copy optimization
        const double* srcStart = &originalData[idx * featCount];
        double* dstStart = &subData[i * featCount];
        std::copy(srcStart, srcStart + featCount, dstStart);
        
        subLabels[i] = originalLabels[idx];
    }
}

void BaggingTrainer::train(const std::vector<double>& data,
                          int rowLength,
                          const std::vector<double>& labels) {
    trees_.clear();
    oobIndices_.clear();
    
    const int dataSize = static_cast<int>(labels.size());
    
    // Data validation
    if (dataSize == 0 || data.empty() || rowLength <= 0) {
        std::cerr << "Error: Invalid training data (empty dataset)" << std::endl;
        return;
    }
    
    if (static_cast<int>(data.size()) != dataSize * rowLength) {
        std::cerr << "Error: Data size mismatch" << std::endl;
        return;
    }
    
    #ifdef _OPENMP
    const int numThreads = omp_get_max_threads();
    std::cout << "Training " << numTrees_ << " trees with " << numThreads 
              << " OpenMP threads..." << std::endl;
    std::cout << "Dataset: " << dataSize << " samples, " << rowLength 
              << " features" << std::endl;
    #else
    std::cout << "Training " << numTrees_ << " trees (no OpenMP)..." << std::endl;
    #endif
    
    // Important: Pre-allocate all containers for thread safety
    trees_.resize(numTrees_);
    oobIndices_.resize(numTrees_);
    
    // Atomic counter for thread-safe progress tracking
    std::atomic<int> completedTrees(0);
    
    // Core: Parallel training of multiple trees, avoiding vector copies
    #pragma omp parallel if(numTrees_ > 1)
    {
        // Thread-local random number generator
        thread_local std::mt19937 localGen;
        thread_local bool initialized = false;
        if (!initialized) {
            #ifdef _OPENMP
            const int threadId = omp_get_thread_num();
            localGen.seed(gen_() + threadId * 1000);
            #else
            localGen.seed(gen_());
            #endif
            initialized = true;
        }
        
        // Thread-local data buffers to avoid repeated allocations
        std::vector<int> sampleIndices, oobIndices;
        std::vector<double> subData, subLabels;
        
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < numTrees_; ++t) {
            // Bootstrap sampling
            bootstrapSample(dataSize, sampleIndices, oobIndices, localGen);
            
            // Efficient data extraction, avoiding unnecessary copies
            extractSubsetOptimized(data, rowLength, labels, sampleIndices, 
                                  subData, subLabels);
            
            // Create a single tree using smart pointers for memory management
            auto tree = std::make_unique<SingleTreeTrainer>(
                createSplitFinder(),
                createCriterion(),
                createPruner({}, rowLength, {}), // Pruner needs to be created without validation data here
                maxDepth_,
                minSamplesLeaf_
            );
            
            tree->train(subData, rowLength, subLabels);
            
            // Thread-safe storage of results
            trees_[t] = std::move(tree);
            oobIndices_[t] = std::move(oobIndices);
            
            // Thread-safe progress output
            const int completed = ++completedTrees;
            if (completed % std::max(1, numTrees_ / 10) == 0) {
                #pragma omp critical(progress_output)
                {
                    std::cout << "Completed " << completed << "/" << numTrees_ 
                              << " trees (" << std::fixed << std::setprecision(1) 
                              << 100.0 * completed / numTrees_ << "%)" << std::endl;
                    std::cout.flush();
                }
            }
        }
    }
    
    std::cout << "Bagging training completed!" << std::endl;
    
    #ifdef _OPENMP
    std::cout << "Used " << omp_get_max_threads() << " threads for parallel training" << std::endl;
    #endif
}

double BaggingTrainer::predict(const double* sample, int rowLength) const {
    if (trees_.empty()) return 0.0;
    
    double sum = 0.0;
    
    // Parallel prediction
    #pragma omp parallel for reduction(+:sum) schedule(static) if(trees_.size() > 10)
    for (size_t i = 0; i < trees_.size(); ++i) {
        if (trees_[i]) {
            sum += trees_[i]->predict(sample, rowLength);
        }
    }
    
    return sum / trees_.size();
}

void BaggingTrainer::evaluate(const std::vector<double>& X,
                             int rowLength,
                             const std::vector<double>& y,
                             double& mse,
                             double& mae) {
    const size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    // Parallel evaluation
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double pred = predict(&X[i * rowLength], rowLength);
        const double diff = y[i] - pred;
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

// Efficient feature importance calculation: avoids recursion and vector copies
std::vector<double> BaggingTrainer::getFeatureImportance(int numFeatures) const {
    std::vector<double> importance(numFeatures, 0.0);
    
    // Parallel computation of feature importance
    #pragma omp parallel if(trees_.size() > 10)
    {
        // Thread-local importance accumulator
        std::vector<double> localImportance(numFeatures, 0.0);
        
        #pragma omp for schedule(static) nowait
        for (size_t t = 0; t < trees_.size(); ++t) {
            if (!trees_[t]) continue;
            
            const Node* root = trees_[t]->getRoot();
            if (!root || root->isLeaf) continue;
            
            // Non-recursive traversal using a stack to avoid recursion overhead
            std::vector<const Node*> nodeStack;
            nodeStack.reserve(1000);
            nodeStack.push_back(root);
            
            while (!nodeStack.empty()) {
                const Node* node = nodeStack.back();
                nodeStack.pop_back();
                
                if (!node || node->isLeaf) continue;
                
                const int feat = node->getFeatureIndex();
                if (feat >= 0 && feat < numFeatures) {
                    localImportance[feat] += 1.0;
                }
                
                if (node->getLeft()) nodeStack.push_back(node->getLeft());
                if (node->getRight()) nodeStack.push_back(node->getRight());
            }
        }
        
        // Reduction to global importance
        #pragma omp critical(importance_reduction)
        {
            for (int i = 0; i < numFeatures; ++i) {
                importance[i] += localImportance[i];
            }
        }
    }
    
    // Normalize importance scores
    const double total = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (total > 0) {
        const double invTotal = 1.0 / total;
        #pragma omp parallel for schedule(static) if(numFeatures > 100)
        for (int i = 0; i < numFeatures; ++i) {
            importance[i] *= invTotal;
        }
    }
    
    return importance;
}

// Batch OOB computation: avoids vector copies
double BaggingTrainer::getOOBError(const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels) const {
    if (trees_.empty() || oobIndices_.empty()) return 0.0;
    
    const int dataSize = static_cast<int>(labels.size());
    std::vector<double> oobPredictions(dataSize, 0.0);
    std::vector<int> oobCounts(dataSize, 0);
    
    // Parallel computation of all OOB predictions
    #pragma omp parallel for schedule(static) if(trees_.size() > 10)
    for (size_t t = 0; t < trees_.size(); ++t) {
        if (!trees_[t]) continue;
        
        const auto& oobSet = oobIndices_[t];
        
        // Predict for the current tree's OOB samples
        for (const int idx : oobSet) {
            const double pred = trees_[t]->predict(&data[idx * rowLength], rowLength);
            
            // Thread-safe accumulation
            #pragma omp atomic
            oobPredictions[idx] += pred;
            
            #pragma omp atomic
            oobCounts[idx]++;
        }
    }
    
    // Calculate OOB error
    double oobMSE = 0.0;
    int validCount = 0;
    
    #pragma omp parallel for reduction(+:oobMSE,validCount) schedule(static)
    for (int i = 0; i < dataSize; ++i) {
        if (oobCounts[i] > 0) {
            const double avgPred = oobPredictions[i] / oobCounts[i];
            const double diff = labels[i] - avgPred;
            oobMSE += diff * diff;
            validCount++;
        }
    }
    
    return validCount > 0 ? oobMSE / validCount : 0.0;
}

// Compatibility method (retains old interface)
void BaggingTrainer::bootstrapSample(int dataSize,
                                     std::vector<int>& sampleIndices,
                                     std::vector<int>& oobIndices) const {
    thread_local std::mt19937 localGen(gen_());
    bootstrapSample(dataSize, sampleIndices, oobIndices, localGen);
}