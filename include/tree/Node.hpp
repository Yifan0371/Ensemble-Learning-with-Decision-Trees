#pragma once

#include <memory>
#include <cstddef>

struct Node {
    bool   isLeaf      = false;
    size_t samples     = 0;
    double metric      = 0.0;      
    
    // Union for memory efficiency
    union NodeInfo {
        struct InternalNode { 
            int featureIndex;
            double threshold;
            Node* left;
            Node* right;
        } internal;
        
        struct LeafNode {
            double prediction;
            double nodePrediction;  
        } leaf;
        
        // Default constructor
        NodeInfo() {
            // Initialize as internal node
            internal.featureIndex = -1;
            internal.threshold = 0.0;
            internal.left = nullptr;
            internal.right = nullptr;
        }
        
        // Destructor (union members don't auto-destruct)
        ~NodeInfo() {
            // Manual cleanup not needed for POD types
        }
    } info;
    
    // Smart pointers for automatic memory management
    std::unique_ptr<Node> leftChild  = nullptr;
    std::unique_ptr<Node> rightChild = nullptr;
    
    Node() : isLeaf(false), samples(0), metric(0.0) {
        // Default initialization
    }
    
    // Make this node a leaf
    void makeLeaf(double prediction, double nodePrediction = 0.0) {
        isLeaf = true;
        info.leaf.prediction = prediction;
        info.leaf.nodePrediction = (nodePrediction != 0.0) ? nodePrediction : prediction;
        
        leftChild.reset();
        rightChild.reset();
    }
    
    // Make this node an internal node
    void makeInternal(int featureIndex, double threshold) {
        isLeaf = false;
        info.internal.featureIndex = featureIndex;
        info.internal.threshold = threshold;
        info.internal.left = nullptr;
        info.internal.right = nullptr;
    }
    
    // Accessor methods
    int getFeatureIndex() const { 
        return isLeaf ? -1 : info.internal.featureIndex; 
    }
    
    double getThreshold() const { 
        return isLeaf ? 0.0 : info.internal.threshold; 
    }
    
    double getPrediction() const { 
        return isLeaf ? info.leaf.prediction : 0.0; 
    }
    
    double getNodePrediction() const {
        return isLeaf ? info.leaf.nodePrediction : 0.0;
    }
    
    Node* getLeft() const { 
        return isLeaf ? nullptr : leftChild.get(); 
    }
    
    Node* getRight() const { 
        return isLeaf ? nullptr : rightChild.get(); 
    }
    
    // Legacy accessor methods
    int featureIndex() const { return getFeatureIndex(); }
    double threshold() const { return getThreshold(); }
    double prediction() const { return getPrediction(); }
    Node* left() const { return getLeft(); }
    Node* right() const { return getRight(); }
};