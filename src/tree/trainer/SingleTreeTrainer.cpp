// =============================================================================
// src/tree/trainer/SingleTreeTrainer.cpp - Task Queue Strategy Optimized Version
// =============================================================================
#include "tree/trainer/SingleTreeTrainer.hpp"
#include "tree/Node.hpp"
#include "pruner/MinGainPrePruner.hpp" // For pre-pruning check
#include <numeric>     // For std::iota
#include <cmath>       // For std::abs
#include <iostream>    // For std::cout
#include <algorithm>   // For std::min, std::max, std::partition
#include <memory>      // For std::unique_ptr
#include <chrono>      // For timing
#include <iomanip>     // For output formatting
#include <queue>       // For std::queue
#include <mutex>       // For std::mutex
#include <condition_variable> // For std::condition_variable
#include <atomic>      // For std::atomic
#ifdef _OPENMP
#include <omp.h>       // For OpenMP pragmas and functions
#endif

// **Task Queue Data Structure**
struct SplitTask {
    Node* node;
    std::vector<int> indices;
    int depth;
    
    // Constructor using move semantics for efficiency
    SplitTask(Node* n, std::vector<int>&& idx, int d) 
        : node(n), indices(std::move(idx)), depth(d) {}
};

// **Thread-Safe Task Queue**
class TaskQueue {
private:
    std::queue<std::unique_ptr<SplitTask>> tasks_; // Queue of split tasks
    mutable std::mutex mutex_;                     // Mutex for protecting queue access
    std::condition_variable condition_;             // Condition variable for signaling task availability
    std::atomic<bool> finished_{false};             // Atomic flag to signal completion
    
public:
    // Pushes a task to the queue and notifies one waiting thread
    void push(std::unique_ptr<SplitTask> task) {
        std::lock_guard<std::mutex> lock(mutex_); // Acquire lock
        tasks_.push(std::move(task));             // Add task
        condition_.notify_one();                  // Notify one consumer
    }
    
    // Pops a task from the queue, waits if empty until a task is available or finished signal is set
    std::unique_ptr<SplitTask> pop() {
        std::unique_lock<std::mutex> lock(mutex_); // Acquire unique lock
        // Wait until tasks are available or all workers are finished
        condition_.wait(lock, [this] { return !tasks_.empty() || finished_; });
        
        if (tasks_.empty()) return nullptr; // If finished and queue is empty, return nullptr
        
        auto task = std::move(tasks_.front()); // Get task
        tasks_.pop();                          // Remove task
        return task;                           // Return task
    }
    
    // Checks if the queue is empty (thread-safe)
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.empty();
    }
    
    // Signals that no more tasks will be added and notifies all waiting threads
    void finish() {
        finished_ = true;
        condition_.notify_all();
    }
    
    // Returns the current size of the queue (thread-safe)
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return tasks_.size();
    }
};

// Constructor for SingleTreeTrainer
SingleTreeTrainer::SingleTreeTrainer(std::unique_ptr<ISplitFinder> finder,
                                     std::unique_ptr<ISplitCriterion> criterion,
                                     std::unique_ptr<IPruner> pruner,
                                     int maxDepth,
                                     int minSamplesLeaf)
    : maxDepth_(maxDepth),
      minSamplesLeaf_(minSamplesLeaf),
      finder_(std::move(finder)),        // Initialize split finder
      criterion_(std::move(criterion)),  // Initialize split criterion
      pruner_(std::move(pruner))         // Initialize pruner
{}

// Main training method for the decision tree
void SingleTreeTrainer::train(const std::vector<double>& data,
                              int rowLength,
                              const std::vector<double>& labels) {
    
    auto trainStart = std::chrono::high_resolution_clock::now(); // Start timing tree building
    
    root_ = std::make_unique<Node>(); // Initialize the root node
    
    // **Removed omp_set_num_threads call, now controlled by environment variable**
    int numThreads = 1;
    #ifdef _OPENMP
    numThreads = omp_get_max_threads(); // Get maximum threads available from OpenMP runtime
    std::cout << "Using " << numThreads << " OpenMP threads (controlled by OMP_NUM_THREADS)" << std::endl;
    #endif
    
    // Pre-allocate and fill root indices
    std::vector<int> rootIndices(labels.size());
    std::iota(rootIndices.begin(), rootIndices.end(), 0); // Fill with 0, 1, 2, ... N-1
    
    // **Professor's suggested task queue/thread pool pattern**
    // Use task queue for large datasets and multiple threads
    const bool useTaskQueue = (labels.size() > 1000 && numThreads > 1);
    
    if (useTaskQueue) {
        std::cout << "Large dataset detected, using task queue strategy" << std::endl;
        buildTreeWithTaskQueue(data, rowLength, labels, std::move(rootIndices));
    } else {
        std::cout << "Small dataset, using optimized recursive strategy" << std::endl;
        // Use optimized recursive split for smaller datasets
        splitNodeOptimized(root_.get(), data, rowLength, labels, rootIndices, 0);
    }
    
    auto splitEnd = std::chrono::high_resolution_clock::now(); // End timing tree building
    
    // Post-pruning phase
    auto pruneStart = std::chrono::high_resolution_clock::now(); // Start timing pruning
    pruner_->prune(root_); // Apply the chosen pruner
    auto pruneEnd = std::chrono::high_resolution_clock::now();   // End timing pruning
    
    auto trainEnd = std::chrono::high_resolution_clock::now(); // End timing total training
    
    // Calculate and print performance statistics
    auto splitTime = std::chrono::duration_cast<std::chrono::milliseconds>(splitEnd - trainStart);
    auto pruneTime = std::chrono::duration_cast<std::chrono::milliseconds>(pruneEnd - pruneStart);
    auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
    
    int treeDepth = 0, leafCount = 0;
    calculateTreeStats(root_.get(), 0, treeDepth, leafCount); // Calculate tree depth and leaf count
    
    std::cout << "Tree training completed:" << std::endl;
    std::cout << "  Depth: " << treeDepth << " | Leaves: " << leafCount << std::endl;
    std::cout << "  Split time: " << splitTime.count() << "ms" 
              << " | Prune time: " << pruneTime.count() << "ms"
              << " | Total: " << totalTime.count() << "ms" << std::endl;
}

// **New method: Task queue driven tree building**
void SingleTreeTrainer::buildTreeWithTaskQueue(const std::vector<double>& data,
                                               int rowLength,
                                               const std::vector<double>& labels,
                                               std::vector<int>&& rootIndices) {
    
    TaskQueue taskQueue; // Create the shared task queue
    std::atomic<int> activeWorkers{0}; // Count of workers currently processing tasks
    std::atomic<int> totalTasks{0};    // Total tasks ever pushed to queue
    
    // Create and push the root task to the queue
    auto rootTask = std::make_unique<SplitTask>(root_.get(), std::move(rootIndices), 0);
    taskQueue.push(std::move(rootTask));
    totalTasks++; // Increment total tasks counter
    
    const int numWorkers = std::min(omp_get_max_threads(), 8); // Limit maximum worker threads
    
    #pragma omp parallel num_threads(numWorkers) // Create a team of threads
    {
        // const int threadId = omp_get_thread_num(); // Thread ID (not used in this simplified version)
        
        while (true) {
            auto task = taskQueue.pop(); // Try to pop a task
            if (!task) break; // If nullptr is returned, queue is finished and empty
            
            activeWorkers++; // Increment active worker count
            
            // Process the current task
            processTask(data, rowLength, labels, std::move(task), taskQueue, totalTasks);
            
            activeWorkers--; // Decrement active worker count
            
            // Check if all work is completed (no active workers and queue is empty)
            if (activeWorkers == 0 && taskQueue.empty()) {
                taskQueue.finish(); // Signal all threads to exit
                break;
            }
        }
    }
    
    std::cout << "Task queue processing completed. Total tasks: " << totalTasks.load() << std::endl;
}

// **Task processing method (called by worker threads)**
void SingleTreeTrainer::processTask(const std::vector<double>& data,
                                    int rowLength,
                                    const std::vector<double>& labels,
                                    std::unique_ptr<SplitTask> task,
                                    TaskQueue& taskQueue,
                                    std::atomic<int>& totalTasks) {
    
    Node* node = task->node;
    const auto& indices = task->indices;
    const int depth = task->depth;
    
    if (indices.empty()) {
        node->makeLeaf(0.0); // Handle empty node (e.g., if no samples reach it)
        return;
    }
    
    // Calculate node's impurity metric and sample count
    node->metric = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();
    
    // **Efficiently calculate node prediction value (mean of labels)**
    double sum = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
        sum += labels[indices[i]];
    }
    const double nodePrediction = sum / indices.size();
    
    // **Stopping condition checks**
    if (depth >= maxDepth_ ||                           // Max depth reached
        indices.size() < 2 * static_cast<size_t>(minSamplesLeaf_) || // Not enough samples to split into two valid leaves
        indices.size() < 2) {                           // Less than 2 samples (cannot split)
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **Find the best split for the current node**
    auto [bestFeat, bestThr, bestGain] =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    // If no valid split found (bestFeat < 0) or no gain (bestGain <= 0)
    if (bestFeat < 0 || bestGain <= 0) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // **Pre-pruning check (using MinGainPrePruner if available)**
    if (auto* prePruner = dynamic_cast<const MinGainPrePruner*>(pruner_.get())) {
        if (bestGain < prePruner->minGain()) {
            node->makeLeaf(nodePrediction, nodePrediction);
            return;
        }
    }

    // **In-place partitioning of indices for child nodes**
    std::vector<int> leftIndices, rightIndices;
    leftIndices.reserve(indices.size());  // Reserve capacity to reduce reallocations
    rightIndices.reserve(indices.size());
    
    for (int idx_val : indices) { // Renamed 'idx' to 'idx_val' to avoid conflict with 'idx' parameter
        if (data[idx_val * rowLength + bestFeat] <= bestThr) {
            leftIndices.push_back(idx_val);
        } else {
            rightIndices.push_back(idx_val);
        }
    }
    
    // Check if both child nodes meet the minimum sample leaf requirement
    if (leftIndices.size() < static_cast<size_t>(minSamplesLeaf_) || 
        rightIndices.size() < static_cast<size_t>(minSamplesLeaf_)) {
        node->makeLeaf(nodePrediction, nodePrediction); // If not, make current node a leaf
        return;
    }

    // Create child nodes and set internal node information
    node->makeInternal(bestFeat, bestThr);
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // **Crucial: Add child node tasks to the queue**
    if (!leftIndices.empty()) {
        auto leftTask = std::make_unique<SplitTask>(
            node->leftChild.get(), std::move(leftIndices), depth + 1);
        taskQueue.push(std::move(leftTask));
        totalTasks++; // Increment total tasks counter
    }
    
    if (!rightIndices.empty()) {
        auto rightTask = std::make_unique<SplitTask>(
            node->rightChild.get(), std::move(rightIndices), depth + 1);
        taskQueue.push(std::move(rightTask));
        totalTasks++; // Increment total tasks counter
    }
}

// **Optimized recursive node splitting (retained for smaller datasets)**
void SingleTreeTrainer::splitNodeOptimized(Node* node,
                                           const std::vector<double>& data,
                                           int rowLength,
                                           const std::vector<double>& labels,
                                           std::vector<int>& indices, // Indices are mutable for in-place partition
                                           int depth) {
    if (indices.empty()) {
        node->makeLeaf(0.0);
        return;
    }
    
    // Calculate node's impurity metric and sample count
    node->metric = criterion_->nodeMetric(labels, indices);
    node->samples = indices.size();
    
    // **Efficiently calculate node prediction value (mean of labels)**
    double sum = 0.0;
    const size_t numSamples = indices.size();
    
    // **Professor's suggestion: Avoid parallel overhead for small datasets**
    if (numSamples > 1000) {
        #pragma omp parallel for reduction(+:sum) schedule(static) num_threads(4) // Use 4 threads for sum reduction
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    } else {
        for (size_t i = 0; i < numSamples; ++i) {
            sum += labels[indices[i]];
        }
    }
    const double nodePrediction = sum / numSamples;
    
    // Stopping condition checks (similar to task queue version)
    if (depth >= maxDepth_ || 
        indices.size() < 2 * static_cast<size_t>(minSamplesLeaf_) ||
        indices.size() < 2) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // Find the best split
    auto [bestFeat, bestThr, bestGain] =
        finder_->findBestSplit(data, rowLength, labels, indices,
                               node->metric, *criterion_);

    if (bestFeat < 0 || bestGain <= 0) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // Pre-pruning check
    if (auto* prePruner = dynamic_cast<const MinGainPrePruner*>(pruner_.get())) {
        if (bestGain < prePruner->minGain()) {
            node->makeLeaf(nodePrediction, nodePrediction);
            return;
        }
    }

    // **In-place partition strategy**
    // Rearranges elements in 'indices' such that elements satisfying the predicate
    // are moved to the beginning. 'partitionPoint' points to the first element
    // of the second group (elements for right child).
    auto partitionPoint = std::partition(indices.begin(), indices.end(),
        [&](int idx_val) { // Renamed 'idx' to 'idx_val'
            return data[idx_val * rowLength + bestFeat] <= bestThr;
        });
    
    const size_t leftSize = std::distance(indices.begin(), partitionPoint);
    const size_t rightSize = indices.size() - leftSize;
    
    // Check min samples per leaf after partitioning
    if (leftSize < static_cast<size_t>(minSamplesLeaf_) || 
        rightSize < static_cast<size_t>(minSamplesLeaf_)) {
        node->makeLeaf(nodePrediction, nodePrediction);
        return;
    }

    // Create child nodes and set internal node information
    node->makeInternal(bestFeat, bestThr);
    node->leftChild = std::make_unique<Node>();
    node->rightChild = std::make_unique<Node>();
    node->info.internal.left = node->leftChild.get();
    node->info.internal.right = node->rightChild.get();

    // Create child index vectors from the partitioned 'indices'
    // This involves copying, which could be optimized further if 'indices' was passed by value
    // and then moved into children, but current design copies to new vectors.
    std::vector<int> leftIndices(indices.begin(), partitionPoint);
    std::vector<int> rightIndices(partitionPoint, indices.end());
    
    // **Careful parallel recursion (only for the first few levels)**
    // This uses OpenMP sections for splitting the recursive calls.
    const bool useParallelRecursion = (depth <= 2) &&           // Only parallelize at shallow depths
                                     (indices.size() > 2000) && // Only for larger nodes
                                     (leftIndices.size() > 500 && rightIndices.size() > 500); // Both children are substantial
    
    if (useParallelRecursion) {
        #pragma omp parallel sections num_threads(2) // Create 2 sections (threads)
        {
            #pragma omp section // First section for left child
            {
                splitNodeOptimized(node->leftChild.get(), data, rowLength, 
                                  labels, leftIndices, depth + 1);
            }
            #pragma omp section  // Second section for right child
            {
                splitNodeOptimized(node->rightChild.get(), data, rowLength, 
                                  labels, rightIndices, depth + 1);
            }
        }
    } else {
        // Serial recursive processing for deeper levels or smaller nodes
        splitNodeOptimized(node->leftChild.get(), data, rowLength, 
                          labels, leftIndices, depth + 1);
        splitNodeOptimized(node->rightChild.get(), data, rowLength, 
                          labels, rightIndices, depth + 1);
    }
}

// Predicts the label for a single sample by traversing the tree
double SingleTreeTrainer::predict(const double* sample, int /* rowLength */) const {
    const Node* cur = root_.get(); // Start from the root
    while (cur && !cur->isLeaf) { // Traverse until a leaf or null node
        const double v = sample[cur->getFeatureIndex()]; // Get feature value for split
        cur = (v <= cur->getThreshold()) ? cur->getLeft() : cur->getRight(); // Move to appropriate child
    }
    return cur ? cur->getPrediction() : 0.0; // Return prediction or 0.0 if tree is empty/null
}

// Evaluates the tree's performance on a given dataset (X, y)
void SingleTreeTrainer::evaluate(const std::vector<double>& X,
                                 int rowLength,
                                 const std::vector<double>& y,
                                 double& mse, // Output: Mean Squared Error
                                 double& mae) { // Output: Mean Absolute Error
    const size_t n = y.size();
    mse = 0.0;
    mae = 0.0;
    
    // **Parallel prediction and error calculation, using num_threads clause**
    // Apply OpenMP parallel for with reduction for mse and mae, static scheduling
    // Use 4 threads if dataset size 'n' is greater than 1000
    #pragma omp parallel for reduction(+:mse,mae) schedule(static, 256) num_threads(4) if(n > 1000)
    for (size_t i = 0; i < n; ++i) {
        const double pred = predict(&X[i * rowLength], rowLength); // Predict for current sample
        const double diff = y[i] - pred;                            // Calculate difference
        mse += diff * diff;                                         // Accumulate squared difference
        mae += std::abs(diff);                                      // Accumulate absolute difference
    }
    
    mse /= n; // Calculate average MSE
    mae /= n; // Calculate average MAE
}

// Recursively calculates tree depth and leaf count
void SingleTreeTrainer::calculateTreeStats(const Node* node, int currentDepth, 
                                           int& maxDepth, int& leafCount) const {
    if (!node) return; // Base case: null node
    
    maxDepth = std::max(maxDepth, currentDepth); // Update max depth if current depth is greater
    
    if (node->isLeaf) {
        leafCount++; // Increment leaf count for leaf nodes
    } else {
        // Recurse for internal nodes
        calculateTreeStats(node->getLeft(), currentDepth + 1, maxDepth, leafCount);
        calculateTreeStats(node->getRight(), currentDepth + 1, maxDepth, leafCount);
    }
}

// Compatibility method: Converts const ref to mutable copy for splitNodeOptimized
void SingleTreeTrainer::splitNode(Node* node,
                                  const std::vector<double>& data,
                                  int rowLength,
                                  const std::vector<double>& labels,
                                  const std::vector<int>& indices,
                                  int depth) {
    std::vector<int> mutableIndices = indices; // Create a mutable copy
    splitNodeOptimized(node, data, rowLength, labels, mutableIndices, depth); // Call optimized method
}

// Compatibility method: Direct call to splitNodeOptimized (assuming indices are mutable)
void SingleTreeTrainer::splitNodeInPlace(Node* node,
                                         const std::vector<double>& data,
                                         int rowLength,
                                         const std::vector<double>& labels,
                                         std::vector<int>& indices, // Indices are already mutable
                                         int depth) {
    splitNodeOptimized(node, data, rowLength, labels, indices, depth);
}

// Compatibility method: Direct call to splitNodeOptimized (assuming indices are mutable)
void SingleTreeTrainer::splitNodeInPlaceParallel(Node* node,
                                                 const std::vector<double>& data,
                                                 int rowLength,
                                                 const std::vector<double>& labels,
                                                 std::vector<int>& indices, // Indices are already mutable
                                                 int depth) {
    splitNodeOptimized(node, data, rowLength, labels, indices, depth);
}