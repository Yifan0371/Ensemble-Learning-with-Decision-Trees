

#include "app/BaggingApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    // Default parameters
    BaggingOptions opts;
    opts.dataPath       = "../data/data_clean/cleaned_data.csv";
    opts.numTrees       = 10;
    opts.sampleRatio    = 1.0;
    opts.maxDepth       = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion      = "mse";
    opts.splitMethod    = "exhaustive";
    opts.prunerType     = "none";
    opts.prunerParam    = 0.01;
    opts.seed           = 42;

    // Parse arguments
    if (argc >= 2)  opts.dataPath = argv[1];
    if (argc >= 3)  opts.numTrees = std::stoi(argv[2]);
    if (argc >= 4)  opts.sampleRatio = std::stod(argv[3]);
    if (argc >= 5)  opts.maxDepth = std::stoi(argv[4]);
    if (argc >= 6)  opts.minSamplesLeaf = std::stoi(argv[5]);
    if (argc >= 7)  opts.criterion = argv[6];
    if (argc >= 8)  opts.splitMethod = argv[7];
    if (argc >= 9)  opts.prunerType = argv[8];
    if (argc >= 10) opts.prunerParam = std::stod(argv[9]);
    if (argc >= 11) opts.seed = static_cast<uint32_t>(std::stoi(argv[10]));
    
    // Run bagging
    runBaggingApp(opts);
    return 0;
}