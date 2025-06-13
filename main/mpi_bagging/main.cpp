
#include "ensemble/MPIBaggingTrainer.hpp"
#include "functions/io/DataIO.hpp"
#include "pipeline/DataSplit.hpp"
#include <mpi.h>
#include <iostream>
#include <chrono>
#include <algorithm>

struct MPIBaggingOptions {
    std::string dataPath;
    int numTrees;
    double sampleRatio;
    int maxDepth;
    int minSamplesLeaf;
    std::string criterion;
    std::string splitMethod;
    std::string prunerType;
    double prunerParam;
    uint32_t seed;
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Default parameters
    MPIBaggingOptions opts;
    opts.dataPath = "../data/data_clean/cleaned_data.csv";
    opts.numTrees = 100;
    opts.sampleRatio = 1.0;
    opts.maxDepth = 800;
    opts.minSamplesLeaf = 2;
    opts.criterion = "mse";
    opts.splitMethod = "exhaustive";
    opts.prunerType = "none";
    opts.prunerParam = 0.01;
    opts.seed = 42;
    
    // Parse arguments
    if (argc >= 2) opts.dataPath = argv[1];
    if (argc >= 3) opts.numTrees = std::stoi(argv[2]);
    if (argc >= 4) opts.sampleRatio = std::stod(argv[3]);
    if (argc >= 5) opts.maxDepth = std::stoi(argv[4]);
    if (argc >= 6) opts.minSamplesLeaf = std::stoi(argv[5]);
    if (argc >= 7) opts.criterion = argv[6];
    if (argc >= 8) opts.splitMethod = argv[7];
    if (argc >= 9) opts.prunerType = argv[8];
    if (argc >= 10) opts.prunerParam = std::stod(argv[9]);
    if (argc >= 11) opts.seed = static_cast<uint32_t>(std::stoi(argv[10]));
    
    try {
        // Data loading and distribution
        std::vector<double> trainX, trainY, testX, testY;
        int numFeatures = 0;
        
        // Master process loads data
        if (mpiRank == 0) {
            DataIO io;
            int rawRowLength = 0;
            auto [X, y] = io.readCSV(opts.dataPath, rawRowLength);
            
            if (X.empty() || y.empty()) {
                std::cerr << "Error: Failed to load data" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            numFeatures = rawRowLength - 1;
            
            DataParams dp;
            if (!splitDataset(X, y, rawRowLength, dp)) {
                std::cerr << "Failed to split dataset" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            trainX = std::move(dp.X_train);
            trainY = std::move(dp.y_train);
            testX = std::move(dp.X_test);
            testY = std::move(dp.y_test);
        }
        
        // Broadcast and distribute data
        MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int trainSize = 0;
        if (mpiRank == 0) trainSize = static_cast<int>(trainY.size());
        MPI_Bcast(&trainSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpiRank != 0) {
            trainX.resize(trainSize * numFeatures);
            trainY.resize(trainSize);
        }
        
        MPI_Bcast(trainX.data(), trainSize * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(trainY.data(), trainSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Create and train MPI Bagging trainer
        MPIBaggingTrainer trainer(
            opts.numTrees, opts.sampleRatio, opts.maxDepth, opts.minSamplesLeaf,
            opts.criterion, opts.splitMethod, opts.prunerType, opts.prunerParam, opts.seed
        );
        
        auto trainStart = std::chrono::high_resolution_clock::now();
        trainer.train(trainX, numFeatures, trainY);
        auto trainEnd = std::chrono::high_resolution_clock::now();
        
        // Distribute test data
        int testSize = 0;
        if (mpiRank == 0) testSize = static_cast<int>(testY.size());
        MPI_Bcast(&testSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (mpiRank != 0) {
            testX.resize(testSize * numFeatures);
            testY.resize(testSize);
        }
        
        MPI_Bcast(testX.data(), testSize * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(testY.data(), testSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Evaluation
        double mse = 0.0, mae = 0.0;
        trainer.evaluate(testX, numFeatures, testY, mse, mae);
        
        if (mpiRank == 0) {
            auto trainTime = std::chrono::duration_cast<std::chrono::milliseconds>(trainEnd - trainStart);
            std::cout << "Training time: " << trainTime.count() << "ms" << std::endl;
            std::cout << "Final MSE: " << mse << std::endl;
            std::cout << "Final MAE: " << mae << std::endl;
            std::cout << "Total Trees: " << opts.numTrees << " (distributed across " << mpiSize << " processes)" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Process " << mpiRank << " error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}
