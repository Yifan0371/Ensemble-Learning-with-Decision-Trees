#pragma once

#include <string> 
#include <vector> 
#include <memory> 
#include <tuple>  
#include <iostream> 

struct XGBoostConfig {
    // Basic parameters
    int numRounds = 100;              
    double eta = 0.3;                 
    int maxDepth = 6;                 
    int minChildWeight = 1;           
    
    // Regularization parameters
    double lambda = 1.0;              
    double gamma = 0.0;               
    double alpha = 0.0;               
    
    // Sampling parameters
    double subsample = 1.0;           
    double colsampleByTree = 1.0;     
    
    // Training control
    bool verbose = true;              
    int earlyStoppingRounds = 0;      
    double tolerance = 1e-7;          
    
    // Split method
    bool useApproxSplit = false;      
    int maxBins = 256;                
    
    // Objective function
    std::string objective = "reg:squarederror";  
};