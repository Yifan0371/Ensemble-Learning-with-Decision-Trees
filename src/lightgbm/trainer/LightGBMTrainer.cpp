// =============================================================================
// src/lightgbm/trainer/LightGBMTrainer.cpp - Optimized Version (avoiding vector<vector>)
// =============================================================================
#include "lightgbm/trainer/LightGBMTrainer.hpp"
#include "boosting/loss/SquaredLoss.hpp"
#include "criterion/MSECriterion.hpp"
#include "finder/HistogramEWFinder.hpp"
#include "finder/HistogramEQFinder.hpp"
#include "finder/AdaptiveEWFinder.hpp"
#include "finder/AdaptiveEQFinder.hpp"
#include "finder/ExhaustiveSplitFinder.hpp"
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

// Optimized feature bundle storage structure - replaces vector<FeatureBundle>
struct OptimizedFeatureBundles {
    std::vector<int> featureToBundle;     // Maps feature index to bundle index
    std::vector<double> featureOffsets;   // Offset value for each feature within its bundle
    std::vector<int> bundleSizes;         // Size of each bundle
    int numBundles = 0;
    
    OptimizedFeatureBundles(int numFeatures) 
        : featureToBundle(numFeatures), featureOffsets(numFeatures, 0.0) {
        // Simple initialization: each feature gets its own bundle
        for (int i = 0; i < numFeatures; ++i) {
            featureToBundle[i] = i;
        }
        bundleSizes.resize(numFeatures, 1);
        numBundles = numFeatures;
    }
    
    std::pair<int, double> transformFeature(int originalFeature, double value) const {
        return {featureToBundle[originalFeature], value + featureOffsets[originalFeature]};
    }
};

LightGBMTrainer::LightGBMTrainer(const LightGBMConfig& config)
    : config_(config) {
    initializeComponents();
    // Pre-allocate memory pools
    gradients_.reserve(50000);
    sampleIndices_.reserve(50000);
    sampleWeights_.reserve(50000);
    trainingLoss_.reserve(config_.numIterations);

    #ifdef _OPENMP
    if (config_.verbose) {
        std::cout << "LightGBM Initialized, OpenMP threads: "
                  << omp_get_max_threads() << std::endl;
    }
    #endif
}

void LightGBMTrainer::initializeComponents() {
    lossFunction_ = std::make_unique<SquaredLoss>();
    if (config_.enableGOSS) {
        gossSampler_ = std::make_unique<GOSSSampler>(config_.topRate, config_.otherRate);
    }
    if (config_.enableFeatureBundling) {
        featureBundler_ = std::make_unique<FeatureBundler>(config_.maxBin, config_.maxConflictRate);
    }
    treeBuilder_ = std::make_unique<LeafwiseTreeBuilder>(
        config_, createOptimalSplitFinder(), createCriterion());
}

void LightGBMTrainer::train(const std::vector<double>& data,
                            int rowLength,
                            const std::vector<double>& labels) {
    const size_t n = labels.size();
    if (config_.verbose) {
        std::cout << "LightGBM Enhanced: " << n << " samples, " << rowLength << " features" << std::endl;
        std::cout << "Split Method: " << config_.splitMethod << std::endl;
        std::cout << "GOSS: " << (config_.enableGOSS ? "Enabled" : "Disabled") << std::endl;
        std::cout << "Feature Bundling: " << (config_.enableFeatureBundling ? "Enabled" : "Disabled") << std::endl;
    }

    // Optimization 1: Use optimized feature bundling structure
    OptimizedFeatureBundles optimizedBundles(rowLength);
    
    if (config_.enableFeatureBundling && rowLength >= 100) {
        preprocessFeaturesOptimized(data, rowLength, n, optimizedBundles);
    } else {
        // Simple handling: each feature is independent
        if (config_.verbose) {
            std::cout << "Feature Bundling (simple): " << rowLength << " -> "
                      << rowLength << " bundles" << std::endl;
        }
    }

    // Initialize predictions and gradients
    const double baseScore = computeBaseScore(labels);
    model_.setBaseScore(baseScore);
    std::vector<double> predictions(n, baseScore);
    gradients_.assign(n, 0.0);

    // Boosting iterations
    for (int iter = 0; iter < config_.numIterations; ++iter) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // Compute loss and update gradients
        const double currentLoss = computeLossOptimized(labels, predictions);
        trainingLoss_.push_back(currentLoss);
        computeGradientsOptimized(labels, predictions);

        // GOSS sampling or full sample
        if (config_.enableGOSS) {
            std::vector<double> absGradients(n);
            computeAbsGradients(absGradients);
            gossSampler_->sample(absGradients, sampleIndices_, sampleWeights_);
            normalizeWeights(n);
        } else {
            prepareFullSample(n);
        }

        // Build a tree
        auto tree = treeBuilder_->buildTree(
            data, rowLength, labels, gradients_,
            sampleIndices_, sampleWeights_, featureBundles_);

        if (!tree) {
            if (config_.verbose) {
                std::cout << "Iteration " << iter << ": No valid split found, stopping training." << std::endl;
            }
            break;
        }

        // Optimization 2: Efficient prediction update
        updatePredictionsOptimized(data, rowLength, tree.get(), predictions, n);
        model_.addTree(std::move(tree), config_.learningRate);

        auto iterEnd = std::chrono::high_resolution_clock::now();
        auto iterTime = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart);

        if (config_.verbose && iter % 10 == 0) {
            std::cout << "Iter " << iter
                      << " | Loss: " << std::fixed << std::setprecision(6) << currentLoss
                      << " | Samples: " << sampleIndices_.size()
                      << " | Time: " << iterTime.count() << " ms" << std::endl;
        }

        // Early stopping check
        if (config_.earlyStoppingRounds > 0 && iter >= config_.earlyStoppingRounds) {
            if (checkEarlyStop(iter)) {
                if (config_.verbose) {
                    std::cout << "Early stopping at iteration " << iter << std::endl;
                }
                break;
            }
        }
    }

    if (config_.verbose) {
        std::cout << "LightGBM Enhanced training complete, " << model_.getTreeCount() << " trees built." << std::endl;
    }
}

// Optimized method implementations

void LightGBMTrainer::preprocessFeaturesOptimized(const std::vector<double>& data,
                                                  int rowLength,
                                                  size_t sampleSize,
                                                  OptimizedFeatureBundles& bundles) {
    // Simplified feature bundling based on sparsity analysis
    std::vector<double> sparsity(rowLength);
    constexpr double EPS = 1e-12;
    
    // Parallel computation of feature sparsity
    #pragma omp parallel for schedule(static) if(rowLength > 50)
    for (int f = 0; f < rowLength; ++f) {
        int nonZeroCount = 0;
        const size_t checkSize = std::min(sampleSize, size_t(10000)); // Sample check
        
        for (size_t i = 0; i < checkSize; ++i) {
            if (std::abs(data[i * rowLength + f]) > EPS) {
                ++nonZeroCount;
            }
        }
        sparsity[f] = 1.0 - static_cast<double>(nonZeroCount) / checkSize;
    }
    
    // Reorganize bundles based on sparsity
    constexpr double SPARSITY_THRESHOLD = 0.8;
    int bundleId = 0;
    
    for (int f = 0; f < rowLength; ++f) {
        if (sparsity[f] > SPARSITY_THRESHOLD) {
            // High sparsity features can be bundled
            bundles.featureToBundle[f] = bundleId;
            bundles.featureOffsets[f] = 0.0; // Simplified
        } else {
            // Low sparsity features are independent
            bundles.featureToBundle[f] = bundleId;
            bundles.featureOffsets[f] = 0.0;
        }
        ++bundleId;
    }
    
    bundles.numBundles = bundleId;
    bundles.bundleSizes.resize(bundleId, 1);
    
    if (config_.verbose) {
        std::cout << "Feature Bundling (optimized): " << rowLength << " -> "
                  << bundles.numBundles << " bundles" << std::endl;
    }
}

double LightGBMTrainer::computeLossOptimized(const std::vector<double>& labels,
                                            const std::vector<double>& predictions) const {
    const size_t n = labels.size();
    double loss = 0.0;
    
    #pragma omp parallel for reduction(+:loss) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        loss += lossFunction_->loss(labels[i], predictions[i]);
    }
    
    return loss / n;
}

void LightGBMTrainer::computeGradientsOptimized(const std::vector<double>& labels,
                                               const std::vector<double>& predictions) {
    const size_t n = labels.size();
    
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        gradients_[i] = labels[i] - predictions[i];
    }
}

void LightGBMTrainer::computeAbsGradients(std::vector<double>& absGradients) const {
    const size_t n = gradients_.size();
    
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        absGradients[i] = std::abs(gradients_[i]);
    }
}

void LightGBMTrainer::normalizeWeights(size_t n) {
    const double totalWeight = std::accumulate(sampleWeights_.begin(), sampleWeights_.end(), 0.0);
    if (totalWeight > 0.0) {
        const double normFactor = static_cast<double>(n) / totalWeight;
        
        #pragma omp parallel for schedule(static) if(sampleWeights_.size() > 1000)
        for (size_t i = 0; i < sampleWeights_.size(); ++i) {
            sampleWeights_[i] *= normFactor;
        }
    }
}

void LightGBMTrainer::prepareFullSample(size_t n) {
    sampleIndices_.resize(n);
    sampleWeights_.assign(n, 1.0);
    std::iota(sampleIndices_.begin(), sampleIndices_.end(), 0);
}

void LightGBMTrainer::updatePredictionsOptimized(const std::vector<double>& data,
                                                 int rowLength,
                                                 const Node* tree,
                                                 std::vector<double>& predictions,
                                                 size_t n) const {
    #pragma omp parallel for schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        const double treePred = predictSingleTree(tree, sample, rowLength);
        predictions[i] += config_.learningRate * treePred;
    }
}

bool LightGBMTrainer::checkEarlyStop(int currentIter) const {
    const int patience = config_.earlyStoppingRounds;
    if (static_cast<int>(trainingLoss_.size()) < patience + 1) {
        return false;
    }
    
    const auto recentStart = trainingLoss_.end() - patience - 1;
    const auto recentEnd = trainingLoss_.end() - 1;
    const double bestLoss = *std::min_element(recentStart, recentEnd);
    const double currentLoss = trainingLoss_.back();
    
    return currentLoss >= bestLoss - config_.tolerance;
}

// Other necessary methods

double LightGBMTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}

void LightGBMTrainer::evaluate(const std::vector<double>& X,
                               int rowLength,
                               const std::vector<double>& y,
                               double& mse,
                               double& mae) {
    const auto predictions = model_.predictBatch(X, rowLength);
    const size_t n = y.size();
    
    mse = 0.0;
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        const double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n;
    mae /= n;
}

double LightGBMTrainer::computeBaseScore(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 5000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    
    return sum / n;
}

std::vector<double> LightGBMTrainer::calculateFeatureImportance(int numFeatures) const {
    return model_.getFeatureImportance(numFeatures);
}

std::unique_ptr<ISplitCriterion> LightGBMTrainer::createCriterion() const {
    return std::make_unique<MSECriterion>();
}

std::unique_ptr<ISplitFinder> LightGBMTrainer::createOptimalSplitFinder() const {
    const std::string& method = config_.splitMethod;
    
    if (method == "histogram_ew" || method.find("histogram_ew:") == 0) {
        int bins = config_.histogramBins;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEWFinder>(bins);
    } else if (method == "histogram_eq" || method.find("histogram_eq:") == 0) {
        int bins = config_.histogramBins;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            bins = std::stoi(method.substr(pos + 1));
        }
        return std::make_unique<HistogramEQFinder>(bins);
    } else if (method == "adaptive_ew" || method.find("adaptive_ew:") == 0) {
        std::string rule = config_.adaptiveRule;
        auto pos = method.find(':');
        if (pos != std::string::npos) {
            rule = method.substr(pos + 1);
        }
        return std::make_unique<AdaptiveEWFinder>(8, config_.maxAdaptiveBins, rule);
    } else if (method == "adaptive_eq") {
        return std::make_unique<AdaptiveEQFinder>(
            config_.minSamplesPerBin,
            config_.maxAdaptiveBins,
            config_.variabilityThreshold);
    } else if (method == "exhaustive") {
        return std::make_unique<ExhaustiveSplitFinder>();
    } else {
        return std::make_unique<HistogramEWFinder>(config_.histogramBins);
    }
}

std::unique_ptr<ISplitFinder> LightGBMTrainer::createHistogramFinder() const {
    return std::make_unique<HistogramEWFinder>(config_.histogramBins);
}

double LightGBMTrainer::predictSingleTree(const Node* tree,
                                          const double* sample,
                                          int /* rowLength */) const {
    const Node* cur = tree;
    while (cur && !cur->isLeaf) {
        const int featureIndex = cur->getFeatureIndex();
        const double value = sample[featureIndex];
        cur = (value <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
    }
    return cur ? cur->getPrediction() : 0.0;
}

// Compatibility method (retains old interface)
void LightGBMTrainer::preprocessFeaturesSerial(const std::vector<double>& /* data */,
                                               int rowLength,
                                               size_t /* sampleSize */) {
    // Serial simple bundling (each feature gets its own bundle)
    featureBundles_.clear();
    featureBundles_.reserve(rowLength);
    for (int i = 0; i < rowLength; ++i) {
        FeatureBundle bundle;
        bundle.features.push_back(i);
        bundle.offsets.push_back(0.0);
        bundle.totalBins = config_.maxBin;
        featureBundles_.push_back(std::move(bundle));
    }
    if (config_.verbose) {
        std::cout << "Feature Bundling (serial): " << rowLength << " -> "
                  << featureBundles_.size() << " bundles" << std::endl;
    }
}