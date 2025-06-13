#include "pruner/ReducedErrorPruner.hpp"
#include <cmath>

// Validates the Mean Squared Error (MSE) of the tree (or subtree) on the validation set.
double ReducedErrorPruner::validate(Node* n) const {
    double mse = 0.0;
    for (size_t i = 0; i < yv_.size(); ++i) {
        const double* sample = &Xv_[i * D_]; // Get validation sample features
        Node* cur = n; // Start traversal from the given node
        // Traverse down the tree until a leaf node is reached
        while (!cur->isLeaf) {
            cur = (sample[cur->getFeatureIndex()] <= cur->getThreshold())
                    ? cur->getLeft() : cur->getRight();
        }
        // Calculate squared difference between actual and predicted value
        double diff = yv_[i] - cur->getPrediction();
        mse += diff * diff;
    }
    return mse / yv_.size(); // Return average MSE
}

// Recursively prunes the tree based on Reduced Error Pruning logic.
void ReducedErrorPruner::pruneRec(std::unique_ptr<Node>& n) const {
    if (!n || n->isLeaf) return; // Base case: if node is null or already a leaf, do nothing
    
    // First, recursively prune the child subtrees
    pruneRec(n->leftChild);
    pruneRec(n->rightChild);
    
    // Backup the current node's state before attempting to prune
    bool oldIsLeaf = n->isLeaf;
    double oldPred = n->isLeaf ? n->getPrediction() : 0.0; // Backup prediction if it was a leaf
    
    // Temporarily move child nodes to backups to allow modification of 'n'
    auto leftBackup  = std::move(n->leftChild);
    auto rightBackup = std::move(n->rightChild);

    // Attempt to transform the current node into a leaf node
    // Calculate the prediction value if it were a leaf (e.g., mean of samples in this node)
    double leafPrediction = n->getNodePrediction(); // Assumes this method computes the mean for the node's samples
    if (leafPrediction == 0.0 && !oldIsLeaf) { // Fallback if getNodePrediction is not implemented or returns 0.0 for internal nodes
        // If no node prediction stored, use existing prediction if it was already a leaf, or default 0.0
        // This part needs careful consideration based on actual Node class implementation.
        // For a regression tree, typically `getNodePrediction` for an internal node would calculate the mean of its training labels.
        leafPrediction = 0.0; 
    }
    
    // Make the current node a leaf with the calculated prediction
    n->makeLeaf(leafPrediction, leafPrediction); // Assuming two arguments for prediction

    // Validate the MSE of the tree *after* pruning this specific node (and its already pruned children)
    double msePruned = validate(n.get());
    
    // Restore the node's original subtree state to calculate its original error
    n->isLeaf = oldIsLeaf; // Restore leaf status
    n->leftChild  = std::move(leftBackup);  // Restore left child
    n->rightChild = std::move(rightBackup); // Restore right child
    
    // Update internal pointers if it was an internal node
    if (!n->isLeaf) {
        n->info.internal.left = n->leftChild.get();
        n->info.internal.right = n->rightChild.get();
    } else {
        // Restore leaf prediction if it was originally a leaf (shouldn't happen here if `oldIsLeaf` is true from start)
        n->info.leaf.prediction = oldPred;
    }
    
    // Validate the MSE of the tree *before* pruning this specific node (original subtree)
    double mseOriginal = validate(n.get());

    // If pruning this node (making it a leaf) does not increase validation MSE (or decreases it), keep it pruned.
    if (msePruned <= mseOriginal) {
        // Finalize pruning: make the node a leaf
        n->makeLeaf(leafPrediction, leafPrediction);
    }
    // Otherwise, the node's state has already been restored to its original (unpruned) form by the backup/restore steps.
}

// Entry point for pruning the entire tree.
void ReducedErrorPruner::prune(std::unique_ptr<Node>& root) const {
    pruneRec(root);
}