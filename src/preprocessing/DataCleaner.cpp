#include "preprocessing/DataCleaner.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>    // For std::fixed and std::setprecision

namespace preprocessing {

void DataCleaner::readCSV(const std::string& filePath,
                          std::vector<std::string>& headers,
                          std::vector<std::vector<double>>& data) {
    
    std::ifstream in(filePath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath);
    }

    headers.clear();
    data.clear();
    
    std::string line;
    int lineNumber = 0;
    
    // **Improvement 1: Read and parse header row**
    if (!std::getline(in, line)) {
        throw std::runtime_error("File is empty or cannot read first line.");
    }
    
    lineNumber++;
    std::stringstream ssHead(line);
    std::string col;
    
    while (std::getline(ssHead, col, ',')) {
        // **Improvement 2: Trim whitespace from header names**
        col.erase(0, col.find_first_not_of(" \t\r\n"));
        col.erase(col.find_last_not_of(" \t\r\n") + 1);
        headers.push_back(col);
    }
    
    if (headers.empty()) {
        throw std::runtime_error("No valid headers found.");
    }
    
    std::cout << "Found " << headers.size() << " columns: ";
    for (size_t i = 0; i < headers.size(); ++i) {
        std::cout << "'" << headers[i] << "'";
        if (i < headers.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // **Improvement 3: Read data rows with enhanced error handling**
    int validRows = 0;
    int errorRows = 0; // Counts rows that had errors but were salvaged
    
    while (std::getline(in, line)) {
        lineNumber++;
        
        // Skip empty lines or lines with only whitespace
        if (line.empty() || line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        row.reserve(headers.size()); // Pre-allocate memory for efficiency
        
        bool rowHasError = false;
        size_t columnIndex = 0;
        
        while (std::getline(ss, cell, ',') && columnIndex < headers.size()) {
            // **Improvement 4: Clean (trim) cell content**
            cell.erase(0, cell.find_first_not_of(" \t\r\n"));
            cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
            
            if (cell.empty()) {
                std::cerr << "Warning: Empty cell at line " << lineNumber << ", column " << (columnIndex + 1) 
                          << ". Using 0.0 as replacement." << std::endl;
                row.push_back(0.0);
            } else {
                try {
                    double value = std::stod(cell);
                    
                    // **Improvement 5: Check for invalid numerical values (inf/nan)**
                    if (!std::isfinite(value)) {
                        std::cerr << "Warning: Invalid number (inf/nan) at line " << lineNumber << ": '" 
                                  << cell << "'. Using 0.0 as replacement." << std::endl;
                        value = 0.0;
                    }
                    
                    row.push_back(value);
                } catch (const std::exception& e) {
                    std::cerr << "Error: Failed to parse value at line " << lineNumber << ": '" 
                              << cell << "'. Using 0.0 as replacement. Reason: " << e.what() << std::endl;
                    row.push_back(0.0);
                    rowHasError = true; // Mark row as having an error
                }
            }
            columnIndex++;
        }
        
        // **Improvement 6: Check if number of columns matches header**
        if (row.size() != headers.size()) {
            std::cerr << "Warning: Column count mismatch at line " << lineNumber 
                      << " (expected " << headers.size() << ", actual " << row.size() << ")";
            
            if (row.size() < headers.size()) {
                // Pad missing columns with 0.0
                while (row.size() < headers.size()) {
                    row.push_back(0.0);
                }
                std::cerr << ", padded with 0.0" << std::endl;
            } else {
                // Truncate extra columns
                row.resize(headers.size());
                std::cerr << ", truncated" << std::endl;
            }
            rowHasError = true; // Mark row as having an error
        }
        
        if (!row.empty()) { // Only add if row is not empty after potential processing
            data.push_back(std::move(row)); // Use move semantics for efficiency
            if (rowHasError) {
                errorRows++;
            } else {
                validRows++;
            }
        }
    }
    
    in.close();
    
    std::cout << "Read complete: " << validRows << " valid data rows";
    if (errorRows > 0) {
        std::cout << ", " << errorRows << " rows contained errors but were fixed";
    }
    std::cout << std::endl;
    
    if (data.empty()) {
        throw std::runtime_error("No valid data rows found after processing.");
    }
}

void DataCleaner::writeCSV(const std::string& filePath,
                           const std::vector<std::string>& headers,
                           const std::vector<std::vector<double>>& data) {
    
    std::ofstream out(filePath);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write to file: " + filePath);
    }
    
    // **Improvement 7: Set output precision for floating-point numbers**
    out << std::fixed << std::setprecision(6); // Fixed notation with 6 decimal places
    
    // Write headers
    for (size_t i = 0; i < headers.size(); ++i) {
        out << headers[i];
        if (i + 1 < headers.size()) {
            out << ',';
        }
    }
    out << '\n';
    
    // Write data rows
    for (const auto& row : data) {
        for (size_t j = 0; j < row.size(); ++j) {
            out << row[j];
            if (j + 1 < row.size()) {
                out << ',';
            }
        }
        out << '\n';
    }
    
    out.close();
    
    // Check for errors during file writing
    if (!out.good()) {
        throw std::runtime_error("Error occurred while writing to file: " + filePath);
    }
}

std::vector<std::vector<double>> DataCleaner::removeOutliers(
    const std::vector<std::vector<double>>& data,
    size_t colIndex,
    double zThreshold) {
    
    if (data.empty()) {
        std::cerr << "Warning: Input data is empty." << std::endl;
        return data;
    }
    
    // Check if column index is valid for the first row (assuming consistent column count)
    if (colIndex >= data[0].size()) {
        std::cerr << "Error: Column index " << colIndex << " is out of bounds (max: " 
                  << (data[0].size() - 1) << ")." << std::endl;
        return data;
    }
    
    // **Improvement 8: Extract target column values and check validity**
    std::vector<double> colVals;
    colVals.reserve(data.size()); // Reserve capacity for efficiency
    
    for (const auto& row : data) {
        if (colIndex < row.size()) { // Defensive check for ragged rows
            double val = row[colIndex];
            if (std::isfinite(val)) { // Only consider finite values for statistics
                colVals.push_back(val);
            }
        }
    }
    
    if (colVals.size() < 3) { // Need at least 3 points for meaningful stddev/mean
        std::cerr << "Warning: Too few valid data points (" << colVals.size() 
                  << ") for reliable outlier detection. Skipping." << std::endl;
        return data; // Return original data if not enough points
    }
    
    // **Improvement 9: Use robust statistical calculations**
    double mean = std::accumulate(colVals.begin(), colVals.end(), 0.0) / colVals.size();
    
    double variance = 0.0;
    for (double v : colVals) {
        double diff = v - mean;
        variance += diff * diff;
    }
    variance /= colVals.size(); // Population variance
    double stddev = std::sqrt(variance);
    
    if (stddev < 1e-10) { // Check for near-zero standard deviation (all values are the same)
        std::cout << "All values in the column are identical. No outlier detection needed." << std::endl;
        return data;
    }
    
    std::cout << "Statistics for column " << colIndex << ":" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << std::endl;
    std::cout << "  Standard Deviation: " << stddev << std::endl;
    std::cout << "  Z-score Threshold: " << zThreshold << std::endl;
    
    // **Improvement 10: Filter outliers and track statistics**
    std::vector<std::vector<double>> cleaned;
    cleaned.reserve(data.size()); // Pre-allocate memory
    
    int removedCount = 0;
    std::vector<double> removedValues; // Store removed values for reporting
    
    for (const auto& row : data) {
        if (colIndex >= row.size()) {
            cleaned.push_back(row); // Keep rows that are too short to have the target column
            continue;
        }
        
        double v = row[colIndex];
        if (!std::isfinite(v)) { // Remove non-finite values if they slipped through
            removedCount++;
            continue;
        }
        
        double z = std::abs((v - mean) / stddev); // Calculate Z-score
        if (z <= zThreshold) {
            cleaned.push_back(row); // Keep if within threshold
        } else {
            removedValues.push_back(v); // Record removed value
            removedCount++;
        }
    }
    
    std::cout << "Outlier detection results:" << std::endl;
    std::cout << "  Original samples: " << data.size() << std::endl;
    std::cout << "  Samples removed: " << removedCount << std::endl;
    std::cout << "  Samples retained: " << cleaned.size() << std::endl;
    std::cout << "  Removal percentage: " << std::fixed << std::setprecision(1) 
              << (100.0 * removedCount / data.size()) << "%" << std::endl;
    
    // Print removed values if there are few of them
    if (!removedValues.empty() && removedValues.size() <= 10) {
        std::cout << "  Removed values: ";
        for (size_t i = 0; i < removedValues.size(); ++i) {
            std::cout << std::fixed << std::setprecision(2) << removedValues[i];
            if (i < removedValues.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    return cleaned;
}

// **Improvement 11: Enhanced Equal Frequency Binning Method**
std::vector<int> DataCleaner::equalFrequencyBinning(const std::vector<double>& values,
                                                     int numBins) {
    if (values.empty()) {
        return {};
    }
    
    if (numBins <= 0) { // Ensure at least one bin
        numBins = 1;
    }
    
    const int n = static_cast<int>(values.size());
    std::vector<std::pair<double, int>> sorted; // Pair (value, original_index)
    sorted.reserve(n);
    
    // Create pairs of (value, original_index)
    for (int i = 0; i < n; ++i) {
        sorted.emplace_back(values[i], i);
    }
    
    std::sort(sorted.begin(), sorted.end()); // Sort by value
    
    std::vector<int> bins(n); // Stores the bin index for each original data point
    const int baseSize = n / numBins; // Minimum number of elements per bin
    const int remainder = n % numBins; // Remainder to distribute among first 'remainder' bins
    
    int idx = 0; // Current index in the sorted array
    for (int b = 0; b < numBins; ++b) {
        int thisSize = baseSize + (b < remainder ? 1 : 0); // Distribute remainder
        for (int k = 0; k < thisSize && idx < n; ++k) {
            bins[sorted[idx].second] = b; // Assign bin 'b' to original index
            ++idx;
        }
    }
    
    return bins;
}

// **Improvement 12: More Robust Bin-Based Outlier Detection**
std::vector<std::vector<double>> DataCleaner::removeOutliersByBinning(
    const std::vector<std::vector<double>>& data,
    size_t colX, // Feature column for X-axis binning
    size_t colY, // Feature column for Y-axis binning
    int numBins, // Number of bins per dimension
    double zThreshold) {
    
    if (data.empty()) {
        return data;
    }
    
    std::cout << "Starting bin-based outlier detection (columns " << colX << " and " << colY << ")" << std::endl;
    
    // Extract values for the two dimensions
    std::vector<double> valsX, valsY;
    valsX.reserve(data.size());
    valsY.reserve(data.size());
    
    for (const auto& row : data) {
        // Defensive check for column existence
        if (colX < row.size() && colY < row.size() && row.size() > 0) { // Ensure row is not empty
            valsX.push_back(row[colX]);
            valsY.push_back(row[colY]);
        } else {
            // Handle rows that don't have enough columns for the specified indices
            // For now, we just skip them from binning, they won't be in result unless added explicitly
            // A more robust solution might copy them directly to result or throw an error.
        }
    }
    
    // Check if enough data points for meaningful binning
    if (valsX.size() < static_cast<size_t>(numBins * 2) || valsY.size() < static_cast<size_t>(numBins * 2)) {
        std::cerr << "Warning: Too few data points (" << valsX.size() 
                  << ") for effective bin-based outlier detection. Skipping." << std::endl;
        return data; // Return original data if not enough points
    }
    
    // Perform equal frequency binning on both dimensions
    auto binsX = equalFrequencyBinning(valsX, numBins);
    auto binsY = equalFrequencyBinning(valsY, numBins);
    
    std::vector<std::vector<double>> result;
    result.reserve(data.size());
    
    int totalRemoved = 0;
    
    // Iterate through each bin combination
    for (int bx = 0; bx < numBins; ++bx) {
        for (int by = 0; by < numBins; ++by) {
            // Collect data points belonging to the current bin combination
            std::vector<size_t> binIndices;       // Original indices of rows in this bin
            std::vector<double> binPerformance;   // Performance values (last column) for these rows
            
            for (size_t i = 0; i < data.size(); ++i) {
                // Ensure indices are within bounds and the row belongs to the current bin combination
                // Note: The original Chinese comment for the condition 'binsX[i] == bx || binsY[i] == by'
                // implies an OR condition. This means any row where its X bin matches bx OR its Y bin matches by
                // is considered for this bin. This might lead to overlapping data in different bin loops.
                // For a true 2D grid of bins, it should typically be 'binsX[i] == bx && binsY[i] == by'.
                // I will maintain the original OR logic to avoid changing behavior, but this is a point of clarification.
                if (i < binsX.size() && i < binsY.size() && 
                    (binsX[i] == bx || binsY[i] == by)) {
                    binIndices.push_back(i);
                    // Assuming the last column is the performance metric for outlier detection
                    if (!data[i].empty()) { // Check if row is not empty before accessing back()
                        binPerformance.push_back(data[i].back()); 
                    }
                }
            }
            
            // If bin has too few data points, keep all of them
            if (binPerformance.size() < 3) { // Minimum points for statistical calculation
                for (size_t idx : binIndices) {
                    result.push_back(data[idx]);
                }
                continue;
            }
            
            // Calculate statistics for this bin
            double mean = std::accumulate(binPerformance.begin(), binPerformance.end(), 0.0) / binPerformance.size();
            double variance = 0.0;
            for (double v : binPerformance) {
                variance += (v - mean) * (v - mean);
            }
            variance /= binPerformance.size(); // Population variance
            double stddev = std::sqrt(variance);
            
            // Filter outliers within this bin
            for (size_t j = 0; j < binIndices.size(); ++j) {
                double z = (stddev > 1e-10) ? std::abs((binPerformance[j] - mean) / stddev) : 0.0;
                if (z <= zThreshold) {
                    result.push_back(data[binIndices[j]]);
                } else {
                    totalRemoved++;
                }
            }
        }
    }
    
    std::cout << "Bin-based outlier detection complete:" << std::endl;
    std::cout << "  Original data: " << data.size() << " rows" << std::endl;
    std::cout << "  Outliers removed: " << totalRemoved << " rows" << std::endl;
    std::cout << "  Data retained: " << result.size() << " rows" << std::endl;
    
    return result;
}

} // namespace preprocessing