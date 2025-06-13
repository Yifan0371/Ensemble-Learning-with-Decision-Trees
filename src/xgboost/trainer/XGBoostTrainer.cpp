#include "xgboost/trainer/XGBoostTrainer.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

XGBoostTrainer::XGBoostTrainer(const XGBoostConfig& config) : config_(config) {
    lossFunction_ = XGBoostLossFactory::create(config_.objective);
    xgbCriterion_ = std::make_unique<XGBoostCriterion>(config_.lambda);
    trainingLoss_.reserve(config_.numRounds);
}

void XGBoostTrainer::train(const std::vector<double>& data, int rowLength, const std::vector<double>& labels) {
    const size_t n = labels.size();
    
    
    ColumnData columnData(rowLength, n);
    
   
    #pragma omp parallel for schedule(dynamic) if(rowLength > 4)
    for (int f = 0; f < rowLength; ++f) {
        columnData.sortedIndices[f].resize(n);
        std::iota(columnData.sortedIndices[f].begin(), columnData.sortedIndices[f].end(), 0);
        std::sort(columnData.sortedIndices[f].begin(), columnData.sortedIndices[f].end(),
                  [&](int a, int b) { return data[a * rowLength + f] < data[b * rowLength + f]; });
    }
    
   
    columnData.values = data;

  
    const double baseScore = computeBaseScore(labels);
    model_.setGlobalBaseScore(baseScore);

    std::vector<double> predictions(n, baseScore);
    std::vector<double> gradients(n), hessians(n);
    std::vector<char> rootMask(n, 1);

    
    for (int round = 0; round < config_.numRounds; ++round) {
        
        const double currentLoss = lossFunction_->computeBatchLoss(labels, predictions);
        trainingLoss_.push_back(currentLoss);

     
        lossFunction_->computeGradientsHessians(labels, predictions, gradients, hessians);

       
        if (config_.subsample < 1.0) {
            const size_t sampleSize = static_cast<size_t>(n * config_.subsample);
            thread_local std::mt19937 gen(std::random_device{}());
            thread_local std::vector<int> indices(n);
            
            if (indices.size() != n) {
                indices.resize(n);
                std::iota(indices.begin(), indices.end(), 0);
            }
            
            std::shuffle(indices.begin(), indices.end(), gen);
            
            std::fill(rootMask.begin(), rootMask.end(), 0);
            for (size_t i = 0; i < sampleSize; ++i) {
                rootMask[indices[i]] = 1;
            }
        } else {
            std::fill(rootMask.begin(), rootMask.end(), 1);
        }

       
        auto tree = trainSingleTree(columnData, gradients, hessians, rootMask);
        if (!tree) break;

        
        updatePredictions(data, rowLength, tree.get(), predictions);
        model_.addTree(std::move(tree), config_.eta);

      
        if (hasValidation_ && config_.earlyStoppingRounds > 0) {
            if (shouldEarlyStop(trainingLoss_, config_.earlyStoppingRounds)) break;
        }
    }
}

std::unique_ptr<Node> XGBoostTrainer::trainSingleTree(const ColumnData& columnData,
                                                     const std::vector<double>& gradients,
                                                     const std::vector<double>& hessians,
                                                     const std::vector<char>& rootMask) const {
    auto root = std::make_unique<Node>();
    buildXGBNode(root.get(), columnData, gradients, hessians, rootMask, 0);
    return root;
}

void XGBoostTrainer::buildXGBNode(Node* node, 
                                  const ColumnData& columnData,
                                  const std::vector<double>& gradients,
                                  const std::vector<double>& hessians,
                                  const std::vector<char>& nodeMask, 
                                  int depth) const {
    const size_t n = nodeMask.size();

    
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    #pragma omp parallel for reduction(+:G_parent,H_parent,sampleCount) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    node->samples = sampleCount;
    const double leafWeight = xgbCriterion_->computeLeafWeight(G_parent, H_parent);

 
    if (depth >= config_.maxDepth || sampleCount < 2 || H_parent < config_.minChildWeight) {
        node->makeLeaf(leafWeight);
        return;
    }

    auto [bestFeature, bestThreshold, bestGain] = findBestSplitXGB(columnData, gradients, hessians, nodeMask);

    if (bestFeature < 0 || bestGain <= config_.gamma) {
        node->makeLeaf(leafWeight);
        return;
    }

   
    node->makeInternal(bestFeature, bestThreshold);

 
    std::vector<char> leftMask(n, 0), rightMask(n, 0);
    
    #pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        if (!nodeMask[i]) continue;
        const double val = columnData.values[i * columnData.numFeatures + bestFeature];
        if (val <= bestThreshold) {
            leftMask[i] = 1;
        } else {
            rightMask[i] = 1;
        }
    }

    
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();

   
    if (depth <= 2 && sampleCount > 5000) {
        #pragma omp parallel sections
        {
            #pragma omp section
            buildXGBNode(node->leftChild.get(), columnData, gradients, hessians, leftMask, depth + 1);
            #pragma omp section
            buildXGBNode(node->rightChild.get(), columnData, gradients, hessians, rightMask, depth + 1);
        }
    } else {
        
        buildXGBNode(node->leftChild.get(), columnData, gradients, hessians, leftMask, depth + 1);
        buildXGBNode(node->rightChild.get(), columnData, gradients, hessians, rightMask, depth + 1);
    }
}

std::tuple<int, double, double> XGBoostTrainer::findBestSplitXGB(
    const ColumnData& columnData,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<char>& nodeMask) const {

    const size_t n = nodeMask.size();

  
    double G_parent = 0.0, H_parent = 0.0;
    int sampleCount = 0;
    
    for (size_t i = 0; i < n; ++i) {
        if (nodeMask[i]) {
            G_parent += gradients[i];
            H_parent += hessians[i];
            ++sampleCount;
        }
    }
    
    if (sampleCount < 2 || H_parent < config_.minChildWeight) {
        return {-1, 0.0, 0.0};
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGain = -std::numeric_limits<double>::infinity();
    constexpr double EPS = 1e-12;

  
    #pragma omp parallel if(columnData.numFeatures > 4)
    {
        int localBestFeature = -1;
        double localBestThreshold = 0.0;
        double localBestGain = -std::numeric_limits<double>::infinity();
      
        std::vector<int> nodeSorted;
        nodeSorted.reserve(sampleCount);
        
        #pragma omp for schedule(dynamic) nowait
        for (int f = 0; f < columnData.numFeatures; ++f) {
          
            nodeSorted.clear();
            const std::vector<int>& featureIndices = columnData.sortedIndices[f];
            
            for (const int idx : featureIndices) {
                if (nodeMask[idx]) {
                    nodeSorted.push_back(idx);
                }
            }
            
            if (nodeSorted.size() < 2) continue;

       
            double G_left = 0.0, H_left = 0.0;
            
            for (size_t i = 0; i + 1 < nodeSorted.size(); ++i) {
                const int idx = nodeSorted[i];
                G_left += gradients[idx];
                H_left += hessians[idx];

                const int nextIdx = nodeSorted[i + 1];
                const double currentVal = columnData.values[idx * columnData.numFeatures + f];
                const double nextVal = columnData.values[nextIdx * columnData.numFeatures + f];

                if (std::abs(nextVal - currentVal) < EPS) continue;

                const double G_right = G_parent - G_left;
                const double H_right = H_parent - H_left;

           
                if (H_left < config_.minChildWeight || H_right < config_.minChildWeight) continue;

              
                const double gain = xgbCriterion_->computeSplitGain(
                    G_left, H_left, G_right, H_right, G_parent, H_parent, config_.gamma);

                if (gain > localBestGain) {
                    localBestGain = gain;
                    localBestFeature = f;
                    localBestThreshold = 0.5 * (currentVal + nextVal);
                }
            }
        }
        
  
        #pragma omp critical
        {
            if (localBestGain > bestGain) {
                bestGain = localBestGain;
                bestFeature = localBestFeature;
                bestThreshold = localBestThreshold;
            }
        }
    }

    return {bestFeature, bestThreshold, bestGain};
}

void XGBoostTrainer::updatePredictions(const std::vector<double>& data, int rowLength,
                                      const Node* tree, std::vector<double>& predictions) const {
    const size_t n = predictions.size();
    
    #pragma omp parallel for schedule(static, 256) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double* sample = &data[i * rowLength];
        const Node* cur = tree;
        
  
        while (cur && !cur->isLeaf) {
            const double val = sample[cur->getFeatureIndex()];
            cur = (val <= cur->getThreshold()) ? cur->getLeft() : cur->getRight();
        }
        
        if (cur) {
            predictions[i] += config_.eta * cur->getPrediction();
        }
    }
}

void XGBoostTrainer::evaluate(const std::vector<double>& X, int rowLength,
                              const std::vector<double>& y, double& mse, double& mae) {
    const auto predictions = model_.predictBatch(X, rowLength);
    const size_t n = y.size();

    mse = 0.0; 
    mae = 0.0;
    
    #pragma omp parallel for reduction(+:mse,mae) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double diff = y[i] - predictions[i];
        mse += diff * diff;
        mae += std::abs(diff);
    }
    
    mse /= n; 
    mae /= n;
}

double XGBoostTrainer::computeBaseScore(const std::vector<double>& y) const {
    const size_t n = y.size();
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum) schedule(static) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        sum += y[i];
    }
    return sum / n;
}

bool XGBoostTrainer::shouldEarlyStop(const std::vector<double>& losses, int patience) const {
    if (static_cast<int>(losses.size()) < patience + 1) return false;
    const double bestLoss = *std::min_element(losses.end() - patience - 1, losses.end() - 1);
    return losses.back() >= bestLoss - config_.tolerance;
}

double XGBoostTrainer::computeValidationLoss() const {
    if (!hasValidation_) return 0.0;
    const auto predictions = model_.predictBatch(X_val_, valRowLength_);
    return lossFunction_->computeBatchLoss(y_val_, predictions);
}

double XGBoostTrainer::predict(const double* sample, int rowLength) const {
    return model_.predict(sample, rowLength);
}