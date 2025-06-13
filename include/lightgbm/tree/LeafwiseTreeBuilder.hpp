// =============================================================================
// include/lightgbm/tree/LeafwiseTreeBuilder.hpp
//
// =============================================================================
#pragma once

#include "tree/Node.hpp"
#include "tree/ISplitFinder.hpp"
#include "tree/ISplitCriterion.hpp"
#include "lightgbm/core/LightGBMConfig.hpp"
#include "lightgbm/sampling/GOSSSampler.hpp"
#include "lightgbm/feature/FeatureBundler.hpp"
#include <queue>
#include <memory>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

/** Leaf node information (for priority queue) */
struct LeafInfo {
    Node* node;
    std::vector<int> sampleIndices;
    double splitGain;
    int bestFeature;
    double bestThreshold;
    // Comparator: sort by splitGain size
    bool operator<(const LeafInfo& other) const {
        return splitGain < other.splitGain;
    }
};

/** Leaf-wise tree builder - deep OpenMP parallel optimization version */
class LeafwiseTreeBuilder {
public:
    LeafwiseTreeBuilder(const LightGBMConfig& config,
                        std::unique_ptr<ISplitFinder> finder,
                        std::unique_ptr<ISplitCriterion> criterion)
        : config_(config), finder_(std::move(finder)), criterion_(std::move(criterion)) {
        tempIndices_.reserve(10000);
        leftIndices_.reserve(5000);
        rightIndices_.reserve(5000);
        leftWeights_.reserve(5000);
        rightWeights_.reserve(5000);
    }

    /**
     * Build single tree
     * @param data  Training data
     * @param rowLength Number of features
     * @param labels Original labels (unused, for compatibility only)
     * @param targets Residuals/gradients
     * @param sampleIndices GOSS sampled sample indices
     * @param sampleWeights GOSS sampled weights
     * @param bundles Feature bundling information
     */
    std::unique_ptr<Node> buildTree(const std::vector<double>& data,
                                    int rowLength,
                                    const std::vector<double>& /* labels */,
                                    const std::vector<double>& targets,
                                    const std::vector<int>& sampleIndices,
                                    const std::vector<double>& sampleWeights,
                                    const std::vector<FeatureBundle>& bundles);

private:
    const LightGBMConfig& config_;
    std::unique_ptr<ISplitFinder> finder_;
    std::unique_ptr<ISplitCriterion> criterion_;

    // Max-heap: current leaves to be split
    std::priority_queue<LeafInfo> leafQueue_;

    // Pre-allocated memory pool
    std::vector<int> tempIndices_;
    std::vector<int> leftIndices_, rightIndices_;
    std::vector<double> leftWeights_, rightWeights_;

    // Single split local buffer, avoid multiple allocations in parallel
    std::vector<LeafInfo> localNewLeafInfos_;

    // Serial version: retain original interface
    bool findBestSplitSerial(const std::vector<double>& data,
                             int rowLength,
                             const std::vector<double>& targets,
                             const std::vector<int>& indices,
                             const std::vector<double>& weights,
                             LeafInfo& leafInfo);

    void splitLeafSerial(LeafInfo& leafInfo,
                         const std::vector<double>& data,
                         int rowLength,
                         const std::vector<double>& targets,
                         const std::vector<double>& sampleWeights);

    // Parallel version: called when needed
    bool findBestSplitParallel(const std::vector<double>& data,
                               int rowLength,
                               const std::vector<double>& targets,
                               const std::vector<int>& indices,
                               const std::vector<double>& weights,
                               LeafInfo& leafInfo);

    void splitLeafParallel(LeafInfo& leafInfo,
                           const std::vector<double>& data,
                           int rowLength,
                           const std::vector<double>& targets,
                           const std::vector<double>& sampleWeights);

    double computeLeafPredictionSerial(const std::vector<int>& indices,
                                       const std::vector<double>& targets,
                                       const std::vector<double>& weights) const;

    double computeLeafPredictionParallel(const std::vector<int>& indices,
                                         const std::vector<double>& targets,
                                         const std::vector<double>& weights) const;

    void processRemainingLeavesSerial(const std::vector<double>& targets,
                                      const std::vector<double>& sampleWeights);

    void processRemainingLeavesParallel(const std::vector<double>& targets,
                                        const std::vector<double>& sampleWeights);
};