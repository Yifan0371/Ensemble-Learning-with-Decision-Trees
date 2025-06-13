#include "pruner/CostComplexityPruner.hpp"
#include <cmath>

// Helper function: Counts the number of leaf nodes in a subtree
static int countLeaves(const Node* node) {
    if (!node || node->isLeaf) return 1; // A null node or a leaf is 1 leaf
    return countLeaves(node->getLeft()) + countLeaves(node->getRight());
}

// Recursively prunes a subtree and returns its total error (cost without alpha)
double CostComplexityPruner::pruneRec(Node* n) const {
    if (n->isLeaf) {
        // For a leaf node, return its total error (error * number of samples)
        // Use n->metric, which represents the impurity (e.g., MSE) for this node.
        return n->metric * n->samples;
    }
    
    // Recursively prune child subtrees
    double errLeft  = pruneRec(n->getLeft());
    double errRight = pruneRec(n->getRight());
    
    // Total error of the current subtree (sum of errors from its children)
    double subtreeError = errLeft + errRight;
    
    // Calculate the number of leaves in the current subtree
    int subtreeLeaves = countLeaves(n->getLeft()) + countLeaves(n->getRight());
    
    // CART complexity comparison:
    // Cost if this node becomes a leaf: (current node's error * samples) + alpha * 1 (for 1 leaf)
    double leafCost = n->metric * n->samples + alpha_;
    
    // Cost if the subtree is kept: (sum of children's errors) + alpha * (number of leaves in subtree)
    double subtreeCost = subtreeError + alpha_ * subtreeLeaves;
    
    // If making this node a leaf is cheaper or equal in cost, prune it
    if (leafCost <= subtreeCost) {
        // The prediction value after pruning this internal node into a leaf
        // For an internal node being pruned, its prediction becomes the mean of its samples.
        // getNodePrediction() should provide this mean.
        double nodePred = n->getNodePrediction(); 
        
        // Transform the current node into a leaf node
        n->makeLeaf(nodePred, nodePred); // Assuming second parameter is also prediction for consistency
        
        // Return the total error of the new single leaf node
        return n->metric * n->samples;  
    }
    
    // If keeping the subtree is cheaper, return the total error of the subtree
    return subtreeError;
}

// Entry point for pruning the entire tree
void CostComplexityPruner::prune(std::unique_ptr<Node>& root) const {
    if (root) pruneRec(root.get()); // Only prune if the root exists
}