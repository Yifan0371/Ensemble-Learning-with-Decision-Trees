// =============================================================================
// include/functions/io/DataIO.hpp - Optimized version
// =============================================================================
#pragma once

#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <algorithm>

class DataIO {
public:
    // Core methods
    std::pair<std::vector<double>, std::vector<double>>
    readCSV(const std::string& filename, int& rowLength);

    void writeResults(const std::vector<double>& results,
                      const std::string& filename);

    // Enhanced: Batch processing methods (for large files)
    bool readCSVBatch(const std::string& filename, 
                      std::vector<double>& flattenedFeatures,
                      std::vector<double>& labels,
                      int& rowLength,
                      size_t batchSize = 10000,
                      size_t skipRows = 0);

    // Enhanced: Parallel writing method
    void writeResultsParallel(const std::vector<double>& results,
                              const std::string& filename,
                              size_t chunkSize = 10000);

    // Enhanced: Memory-mapped reading method (for very large files)
    bool readCSVMemoryMapped(const std::string& filename,
                             std::vector<double>& flattenedFeatures,
                             std::vector<double>& labels,
                             int& rowLength);

    // Enhanced: Data validation method
    bool validateData(const std::vector<double>& flattenedFeatures,
                      const std::vector<double>& labels,
                      int rowLength);

    // Enhanced: Get file statistics
    struct FileStats {
        size_t totalRows;
        size_t totalFeatures;
        size_t estimatedMemoryMB;
        bool hasHeader;
    };

    FileStats getFileStats(const std::string& filename) const;

    // Enhanced: Streaming reader interface (for huge datasets)
    class CSVReader {
    public:
        explicit CSVReader(const std::string& filename);
        ~CSVReader();
        
        bool hasNext() const;
        bool readNext(std::vector<double>& features, double& label);
        void reset();
        
    private:
        class Impl;
        std::unique_ptr<Impl> pImpl_;
    };

    std::unique_ptr<CSVReader> createReader(const std::string& filename);
};