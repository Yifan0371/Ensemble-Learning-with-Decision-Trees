#pragma once

#include <string>

struct LightGBMConfig {
    // Model parameters
    int numIterations = 100;          
    double learningRate = 0.1;        
    int maxDepth = -1;                
    int numLeaves = 31;               
    int minDataInLeaf = 20;           
    
    // GOSS parameters
    double topRate = 0.2;             
    double otherRate = 0.1;           
    
    // Histogram parameters
    int maxBin = 255;                 
    double maxConflictRate = 0.0;     
    
    // Training control
    bool verbose = true;              
    int earlyStoppingRounds = 0;      
    double tolerance = 1e-7;          
    
    // Regularization
    double lambda = 0.0;              
    double minSplitGain = 0.0;        
    
    // Optimization features
    bool enableFeatureBundling = true; 
    bool enableGOSS = true;           
    int histPoolSize = 16384;         
    
    // Objective function
    std::string objective = "regression"; 

    // Split method configuration
    std::string splitMethod = "histogram_ew";  
    int histogramBins = 255;           
    std::string adaptiveRule = "sturges"; 
    int minSamplesPerBin = 5;          
    int maxAdaptiveBins = 128;         
    double variabilityThreshold = 0.1; 
    bool enableSIMD = true;            
};