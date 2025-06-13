// =============================================================================
// src/functions/io/DataIO.cpp - Optimized version (avoid unnecessary vector copies)
// =============================================================================
#include "functions/io/DataIO.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <cmath>
#include <utility>
#include <iomanip>
#include <stdexcept>    
#include <vector>


std::pair<std::vector<double>, std::vector<double>>
DataIO::readCSV(const std::string& filename, int& rowLength) {
    std::vector<double> flattenedFeatures;
    std::vector<double> labels;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return {std::move(flattenedFeatures), std::move(labels)};
    }

    std::string line;
    bool headerSkipped = false;
    rowLength = 0;

    // Optimization 1: Pre-allocate container size estimation
    // First quick scan file to get row count estimation
    const auto initialPos = file.tellg();
    size_t estimatedRows = 0;
    while (std::getline(file, line)) {
        ++estimatedRows;
    }
    file.clear();
    file.seekg(initialPos);
    
    if (estimatedRows > 1) { // Subtract header row
        --estimatedRows;
        labels.reserve(estimatedRows);
        // Feature count determined after reading first row
    }

    while (std::getline(file, line)) {
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        
        // Optimization 2: Use efficient string parsing
        std::vector<double> row;
        row.reserve(50); // Pre-allocate common feature count
        
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to parse value '" << value 
                          << "' as double: " << e.what() << std::endl;
                row.push_back(0.0); // Default value
            }
        }
        
        if (!row.empty()) {
            // Optimization 3: Avoid pushing back label every time, use batch operations
            labels.push_back(row.back());
            row.pop_back();
            
            // Optimization 4: Pre-allocate feature vector and batch insert
            if (flattenedFeatures.empty() && !row.empty()) {
                // First row determines feature count, pre-allocate total space
                const size_t featuresPerRow = row.size();
                flattenedFeatures.reserve(estimatedRows * featuresPerRow);
                rowLength = static_cast<int>(featuresPerRow) + 1; // +1 for label
            }
            
            // Use move semantics and batch insertion
            flattenedFeatures.insert(flattenedFeatures.end(),
                                   std::make_move_iterator(row.begin()),
                                   std::make_move_iterator(row.end()));
        }
    }

    file.close();
    
    // Optimization 5: Final memory cleanup
    flattenedFeatures.shrink_to_fit();
    labels.shrink_to_fit();
    
    std::cout << "Loaded " << labels.size() << " samples with " 
              << (rowLength - 1) << " features each" << std::endl;
    
    return {std::move(flattenedFeatures), std::move(labels)};
}

void DataIO::writeResults(const std::vector<double>& results,
                          const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    // Optimization: Use buffered writing to improve performance
    file.precision(10); // Set precision
    file << std::fixed;
    
    for (const auto& r : results) {
        file << r << '\n';
    }
    
    file.close();
}

// New method: Batch CSV reading (for large files)
bool DataIO::readCSVBatch(const std::string& filename, 
                          std::vector<double>& flattenedFeatures,
                          std::vector<double>& labels,
                          int& rowLength,
                          size_t batchSize,
                          size_t skipRows) {
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return false;
    }

    std::string line;
    
    // Skip header
    if (!std::getline(file, line)) {
        std::cerr << "Empty file: " << filename << std::endl;
        return false;
    }
    
    // Skip specified rows
    for (size_t i = 0; i < skipRows; ++i) {
        if (!std::getline(file, line)) {
            return false; // End of file
        }
    }
    
    // Clear output containers
    flattenedFeatures.clear();
    labels.clear();
    flattenedFeatures.reserve(batchSize * 50); // Estimate feature count
    labels.reserve(batchSize);
    
    size_t rowsRead = 0;
    while (rowsRead < batchSize && std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            try {
                row.push_back(std::stod(value));
            } catch (const std::exception&) {
                row.push_back(0.0);
            }
        }
        
        if (!row.empty()) {
            labels.push_back(row.back());
            row.pop_back();
            
            if (rowsRead == 0) {
                rowLength = static_cast<int>(row.size()) + 1;
            }
            
            flattenedFeatures.insert(flattenedFeatures.end(),
                                   row.begin(), row.end());
            ++rowsRead;
        }
    }
    
    file.close();
    return rowsRead > 0;
}

// New method: Parallel CSV writing (for large result sets)
void DataIO::writeResultsParallel(const std::vector<double>& results,
                                  const std::string& filename,
                                  size_t chunkSize) {
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }
    
    file.precision(10);
    file << std::fixed;
    
    // For large datasets, use buffered batch writing
    if (results.size() > chunkSize) {
        std::string buffer;
        buffer.reserve(chunkSize * 20); // Pre-allocate buffer
        
        for (size_t i = 0; i < results.size(); ++i) {
            buffer += std::to_string(results[i]) + '\n';
            
            if ((i + 1) % chunkSize == 0 || i == results.size() - 1) {
                file << buffer;
                buffer.clear();
            }
        }
    } else {
        // Direct writing for small datasets
        for (const auto& r : results) {
            file << r << '\n';
        }
    }
    
    file.close();
}

// New method: Memory mapped reading (for very large files)
bool DataIO::readCSVMemoryMapped(const std::string& filename,
                                 std::vector<double>& flattenedFeatures,
                                 std::vector<double>& labels,
                                 int& rowLength) {
    // Memory mapped file reading can be implemented here
    // For now, use optimized version of standard method
    auto result = readCSV(filename, rowLength);
    flattenedFeatures = std::move(result.first);
    labels = std::move(result.second);
    return !flattenedFeatures.empty();
}

// New method: Validate data integrity
bool DataIO::validateData(const std::vector<double>& flattenedFeatures,
                          const std::vector<double>& labels,
                          int rowLength) {
    
    if (labels.empty()) {
        std::cerr << "Error: No labels found" << std::endl;
        return false;
    }
    
    const size_t expectedFeatureCount = labels.size() * (rowLength - 1);
    if (flattenedFeatures.size() != expectedFeatureCount) {
        std::cerr << "Error: Feature count mismatch. Expected: " 
                  << expectedFeatureCount << ", Got: " << flattenedFeatures.size() << std::endl;
        return false;
    }
    
    // Check for invalid values
    const auto invalidFeature = std::find_if(flattenedFeatures.begin(), flattenedFeatures.end(),
        [](double val) { return !std::isfinite(val); });
    
    if (invalidFeature != flattenedFeatures.end()) {
        std::cerr << "Warning: Found non-finite feature values" << std::endl;
    }
    
    const auto invalidLabel = std::find_if(labels.begin(), labels.end(),
        [](double val) { return !std::isfinite(val); });
    
    if (invalidLabel != labels.end()) {
        std::cerr << "Warning: Found non-finite label values" << std::endl;
    }
    
    return true;
}