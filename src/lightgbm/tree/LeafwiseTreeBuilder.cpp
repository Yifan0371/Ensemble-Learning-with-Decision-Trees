// =============================================================================
// src/lightgbm/tree/LeafwiseTreeBuilder.cpp
// OpenMP Deep Parallel Optimization Version (reduced lock contention, increased thresholds, pre-allocated buffers)
// =============================================================================
#include "lightgbm/tree/LeafwiseTreeBuilder.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

std::unique_ptr<Node> LeafwiseTreeBuilder::buildTree(
    const std::vector<double>& data,
    int rowLength,
    const std::vector<double>& /* labels */,
    const std::vector<double>& targets,
    const std::vector<int>& sampleIndices,
    const std::vector<double>& sampleWeights,
    const std::vector<FeatureBundle>& /* bundles */) {

    // Clear the priority queue
    while (!leafQueue_.empty()) leafQueue_.pop();

    // Initialize root node
    auto root = std::make_unique<Node>();
    root->samples = sampleIndices.size();

    // Calculate root node prediction (weighted average). Parallel for n >= 2000.
    double weightedSum = 0.0, totalWeight = 0.0;
    size_t n = sampleIndices.size();
    if (n >= 2000) {
        #pragma omp parallel for reduction(+:weightedSum, totalWeight) schedule(static)
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double w = sampleWeights[i];
            weightedSum += targets[idx] * w;
            totalWeight += w;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            int idx = sampleIndices[i];
            double w = sampleWeights[i];
            weightedSum += targets[idx] * w;
            totalWeight += w;
        }
    }
    double rootPrediction = (totalWeight > 0.0) ? (weightedSum / totalWeight) : 0.0;

    // Attempt to split root node
    LeafInfo rootInfo;
    rootInfo.node = root.get();
    rootInfo.sampleIndices = sampleIndices;
    if (n < static_cast<size_t>(config_.minDataInLeaf) * 2) {
        // Too few samples, make it a leaf
        root->makeLeaf(rootPrediction);
        return root;
    }
    if (n >= 2000) {
        if (!findBestSplitParallel(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
            root->makeLeaf(rootPrediction);
            return root;
        }
    } else {
        if (!findBestSplitSerial(data, rowLength, targets, rootInfo.sampleIndices, sampleWeights, rootInfo)) {
            root->makeLeaf(rootPrediction);
            return root;
        }
    }
    leafQueue_.push(rootInfo);

    int currentLeaves = 1;
    while (!leafQueue_.empty() && currentLeaves < config_.numLeaves) {
        // Get the leaf with the best gain
        LeafInfo bestLeaf = leafQueue_.top();
        leafQueue_.pop();

        // If not enough to continue splitting
        size_t m = bestLeaf.sampleIndices.size();
        if (bestLeaf.splitGain <= config_.minSplitGain ||
            m < static_cast<size_t>(config_.minDataInLeaf) * 2) {
            // Calculate leaf prediction (parallel/serial) and make it a leaf
            double leafPred = (m >= 500)
                              ? computeLeafPredictionParallel(bestLeaf.sampleIndices, targets, sampleWeights)
                              : computeLeafPredictionSerial(bestLeaf.sampleIndices, targets, sampleWeights);
            bestLeaf.node->makeLeaf(leafPred);
            continue;
        }

        // Perform split (parallel or serial)
        if (m >= 2000) {
            splitLeafParallel(bestLeaf, data, rowLength, targets, sampleWeights);
        } else {
            splitLeafSerial(bestLeaf, data, rowLength, targets, sampleWeights);
        }
        currentLeaves++;
    }

    // Process all remaining nodes: serial or parallel
    if (!leafQueue_.empty()) {
        if (leafQueue_.size() >= 4) {
            processRemainingLeavesParallel(targets, sampleWeights);
        } else {
            processRemainingLeavesSerial(targets, sampleWeights);
        }
    }

    return root;
}

// Serial find best split
bool LeafwiseTreeBuilder::findBestSplitSerial(const std::vector<double>& data,
                                              int rowLength,
                                              const std::vector<double>& targets,
                                              const std::vector<int>& indices,
                                              const std::vector<double>& weights,
                                              LeafInfo& leafInfo) {
    if (indices.size() < static_cast<size_t>(config_.minDataInLeaf) * 2) return false;
    double currentMetric = criterion_->nodeMetric(targets, indices);
    auto [f, thresh, gain] =
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    leafInfo.bestFeature = f;
    leafInfo.bestThreshold = thresh;
    leafInfo.splitGain = gain;
    return f >= 0 && gain > 0;
}

// Parallel find best split (conceptually parallel, implementation is still serial search for best split here)
bool LeafwiseTreeBuilder::findBestSplitParallel(const std::vector<double>& data,
                                                int rowLength,
                                                const std::vector<double>& targets,
                                                const std::vector<int>& indices,
                                                const std::vector<double>& weights,
                                                LeafInfo& leafInfo) {
    if (indices.size() < static_cast<size_t>(config_.minDataInLeaf) * 2) return false;
    double currentMetric = criterion_->nodeMetric(targets, indices);
    auto [f, thresh, gain] =
        finder_->findBestSplit(data, rowLength, targets, indices, currentMetric, *criterion_);
    leafInfo.bestFeature = f;
    leafInfo.bestThreshold = thresh;
    leafInfo.splitGain = gain;
    return f >= 0 && gain > 0;
}

// Serial split leaf
void LeafwiseTreeBuilder::splitLeafSerial(LeafInfo& leafInfo,
                                          const std::vector<double>& data,
                                          int rowLength,
                                          const std::vector<double>& targets,
                                          const std::vector<double>& sampleWeights) {
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();

    leftIndices_.clear();
    rightIndices_.clear();
    leftWeights_.clear();
    rightWeights_.clear();

    for (size_t i = 0; i < leafInfo.sampleIndices.size(); ++i) {
        int idx = leafInfo.sampleIndices[i];
        double value = data[idx * rowLength + leafInfo.bestFeature];
        double w = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
        if (value <= leafInfo.bestThreshold) {
            leftIndices_.push_back(idx);
            leftWeights_.push_back(w);
        } else {
            rightIndices_.push_back(idx);
            rightWeights_.push_back(w);
        }
    }

    // Left child node
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitSerial(data, rowLength, targets, leftInfo.sampleIndices, leftWeights_, leftInfo)) {
            leafQueue_.push(leftInfo);
        } else {
            double leftPred = computeLeafPredictionSerial(leftIndices_, targets, leftWeights_);
            leftInfo.node->makeLeaf(leftPred);
        }
    } else {
        double leftPred = computeLeafPredictionSerial(leftIndices_, targets, leftWeights_);
        leafInfo.node->leftChild->makeLeaf(leftPred);
    }

    // Right child node (logic same as left)
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitSerial(data, rowLength, targets, rightInfo.sampleIndices, rightWeights_, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightPred = computeLeafPredictionSerial(rightIndices_, targets, rightWeights_);
            rightInfo.node->makeLeaf(rightPred);
        }
    } else {
        double rightPred = computeLeafPredictionSerial(rightIndices_, targets, rightWeights_);
        leafInfo.node->rightChild->makeLeaf(rightPred);
    }
}

// Parallel split leaf, reduced lock contention: use thread-local buffers to collect child nodes, then merge serially
void LeafwiseTreeBuilder::splitLeafParallel(LeafInfo& leafInfo,
                                            const std::vector<double>& data,
                                            int rowLength,
                                            const std::vector<double>& targets,
                                            const std::vector<double>& sampleWeights) {
    leafInfo.node->makeInternal(leafInfo.bestFeature, leafInfo.bestThreshold);
    leafInfo.node->leftChild = std::make_unique<Node>();
    leafInfo.node->rightChild = std::make_unique<Node>();

    size_t m = leafInfo.sampleIndices.size();
    leftIndices_.clear();
    rightIndices_.clear();
    leftWeights_.clear();
    rightWeights_.clear();
    leftIndices_.reserve(m / 2 + 1);
    rightIndices_.reserve(m / 2 + 1);
    leftWeights_.reserve(m / 2 + 1);
    rightWeights_.reserve(m / 2 + 1);

    // Parallel processing to local buffers
    int bestFeat = leafInfo.bestFeature;
    double bestThresh = leafInfo.bestThreshold;
    #pragma omp parallel
    {
        std::vector<int> localLeftIdx, localRightIdx;
        std::vector<double> localLeftW, localRightW;
        localLeftIdx.reserve(m / 4 + 1);
        localRightIdx.reserve(m / 4 + 1);
        localLeftW.reserve(m / 4 + 1);
        localRightW.reserve(m / 4 + 1);

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < m; ++i) {
            int idx = leafInfo.sampleIndices[i];
            double value = data[idx * rowLength + bestFeat];
            double w = (i < sampleWeights.size()) ? sampleWeights[i] : 1.0;
            if (value <= bestThresh) {
                localLeftIdx.push_back(idx);
                localLeftW.push_back(w);
            } else {
                localRightIdx.push_back(idx);
                localRightW.push_back(w);
            }
        }

        #pragma omp critical
        {
            leftIndices_.insert(leftIndices_.end(), localLeftIdx.begin(), localLeftIdx.end());
            leftWeights_.insert(leftWeights_.end(), localLeftW.begin(), localLeftW.end());
            rightIndices_.insert(rightIndices_.end(), localRightIdx.begin(), localRightIdx.end());
            rightWeights_.insert(rightWeights_.end(), localRightW.begin(), localRightW.end());
        }
    }

    // Left child node
    if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo leftInfo;
        leftInfo.node = leafInfo.node->leftChild.get();
        leftInfo.sampleIndices = leftIndices_;
        leftInfo.node->samples = leftIndices_.size();
        if (leftIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitParallel(data, rowLength, targets, leftInfo.sampleIndices, leftWeights_, leftInfo)) {
            // Thread-safe insertion: performed outside parallel region since findBestSplitParallel only determines, no insertion conflict
            leafQueue_.push(leftInfo);
        } else {
            double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights_);
            leftInfo.node->makeLeaf(leftPred);
        }
    } else {
        double leftPred = computeLeafPredictionParallel(leftIndices_, targets, leftWeights_);
        leafInfo.node->leftChild->makeLeaf(leftPred);
    }

    // Right child node
    if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf)) {
        LeafInfo rightInfo;
        rightInfo.node = leafInfo.node->rightChild.get();
        rightInfo.sampleIndices = rightIndices_;
        rightInfo.node->samples = rightIndices_.size();
        if (rightIndices_.size() >= static_cast<size_t>(config_.minDataInLeaf) * 2 &&
            findBestSplitParallel(data, rowLength, targets, rightInfo.sampleIndices, rightWeights_, rightInfo)) {
            leafQueue_.push(rightInfo);
        } else {
            double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights_);
            rightInfo.node->makeLeaf(rightPred);
        }
    } else {
        double rightPred = computeLeafPredictionParallel(rightIndices_, targets, rightWeights_);
        leafInfo.node->rightChild->makeLeaf(rightPred);
    }
}

double LeafwiseTreeBuilder::computeLeafPredictionSerial(
    const std::vector<int>& indices,
    const std::vector<double>& targets,
    const std::vector<double>& weights) const {
    if (indices.empty()) return 0.0;
    double sum = 0.0, wsum = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        sum += targets[indices[i]] * weights[i];
        wsum += weights[i];
    }
    return (wsum > 0.0) ? (sum / wsum) : 0.0;
}

double LeafwiseTreeBuilder::computeLeafPredictionParallel(
    const std::vector<int>& indices,
    const std::vector<double>& targets,
    const std::vector<double>& weights) const {
    if (indices.empty()) return 0.0;
    double sum = 0.0, wsum = 0.0;
    size_t m = indices.size();
    if (m >= 1000) {
        #pragma omp parallel for reduction(+:sum, wsum) schedule(static)
        for (size_t i = 0; i < m; ++i) {
            sum += targets[indices[i]] * weights[i];
            wsum += weights[i];
        }
    } else {
        for (size_t i = 0; i < m; ++i) {
            sum += targets[indices[i]] * weights[i];
            wsum += weights[i];
        }
    }
    return (wsum > 0.0) ? (sum / wsum) : 0.0;
}

// Serial processing of remaining leaves
void LeafwiseTreeBuilder::processRemainingLeavesSerial(
    const std::vector<double>& targets,
    const std::vector<double>& sampleWeights) {
    while (!leafQueue_.empty()) {
        LeafInfo leaf = leafQueue_.top();
        leafQueue_.pop();
        double leafPred = computeLeafPredictionSerial(leaf.sampleIndices, targets, sampleWeights);
        leaf.node->makeLeaf(leafPred);
    }
}

// Parallel processing of remaining leaves: using parallel for
void LeafwiseTreeBuilder::processRemainingLeavesParallel(
    const std::vector<double>& targets,
    const std::vector<double>& sampleWeights) {
    // Collect all remaining leaves into a temporary array
    std::vector<LeafInfo> rem;
    rem.reserve(leafQueue_.size());
    while (!leafQueue_.empty()) {
        rem.push_back(leafQueue_.top());
        leafQueue_.pop();
    }
    size_t m = rem.size();
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < m; ++i) {
        double leafPred = computeLeafPredictionParallel(rem[i].sampleIndices, targets, sampleWeights);
        rem[i].node->makeLeaf(leafPred);
    }
}