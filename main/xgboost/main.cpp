

#include "xgboost/app/XGBoostApp.hpp"
#include <iostream>
#include <string>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [OPTIONS]\n\n";
    std::cout << "Required:\n";
    std::cout << "  --data PATH           Training data CSV file\n\n";
    std::cout << "Model Parameters:\n";
    std::cout << "  --objective STR       Objective function (default: reg:squarederror)\n";
    std::cout << "  --num-rounds INT      Boosting rounds (default: 100)\n";
    std::cout << "  --eta FLOAT           Learning rate (default: 0.3)\n";
    std::cout << "  --max-depth INT       Maximum tree depth (default: 6)\n";
    std::cout << "  --lambda FLOAT        L2 regularization (default: 1.0)\n";
    std::cout << "  --gamma FLOAT         Minimum loss reduction (default: 0.0)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << programName << " --data data.csv\n";
    std::cout << "  " << programName << " --data data.csv --num-rounds 200 --eta 0.1\n";
}

bool parseArguments(int argc, char** argv, XGBoostAppOptions& opts) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") return false;
        else if (arg == "--data" && i + 1 < argc) opts.dataPath = argv[++i];
        else if (arg == "--objective" && i + 1 < argc) opts.objective = argv[++i];
        else if (arg == "--num-rounds" && i + 1 < argc) opts.numRounds = std::stoi(argv[++i]);
        else if (arg == "--eta" && i + 1 < argc) opts.eta = std::stod(argv[++i]);
        else if (arg == "--max-depth" && i + 1 < argc) opts.maxDepth = std::stoi(argv[++i]);
        else if (arg == "--min-child-weight" && i + 1 < argc) opts.minChildWeight = std::stoi(argv[++i]);
        else if (arg == "--lambda" && i + 1 < argc) opts.lambda = std::stod(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc) opts.gamma = std::stod(argv[++i]);
        else if (arg == "--subsample" && i + 1 < argc) opts.subsample = std::stod(argv[++i]);
        else if (arg == "--colsample-bytree" && i + 1 < argc) opts.colsampleByTree = std::stod(argv[++i]);
        else if (arg == "--early-stopping" && i + 1 < argc) opts.earlyStoppingRounds = std::stoi(argv[++i]);
        else if (arg == "--verbose") opts.verbose = true;
        else if (arg == "--quiet") opts.verbose = false;
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    return !opts.dataPath.empty();
}

int main(int argc, char** argv) {
    // Default options
    XGBoostAppOptions opts;
    opts.dataPath = "";
    opts.objective = "reg:squarederror";
    opts.numRounds = 100;
    opts.eta = 0.3;
    opts.maxDepth = 6;
    opts.minChildWeight = 1;
    opts.lambda = 1.0;
    opts.gamma = 0.0;
    opts.subsample = 1.0;
    opts.colsampleByTree = 1.0;
    opts.verbose = true;
    opts.earlyStoppingRounds = 0;
    opts.tolerance = 1e-7;
    opts.valSplit = 0.2;
    opts.useApproxSplit = false;
    opts.maxBins = 256;
    
    if (!parseArguments(argc, argv, opts)) {
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        runXGBoostApp(opts);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}