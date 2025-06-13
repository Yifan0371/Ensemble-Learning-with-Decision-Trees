// =============================================================================
// main/data_clean/main.cpp - Fixed Version
// =============================================================================
#include "preprocessing/DataCleaner.hpp"
#include <filesystem>
#include <iostream>
#include <string>
#include <stdexcept>

namespace fs = std::filesystem;

int main() {
    // **Fix 1: Use relative paths and ensure paths exist**
    const std::string inDir  = "../data/data_base";
    const std::string outDir = "../data/data_clean";

    std::cout << "=== Data Cleaning Tool ===" << std::endl;
    std::cout << "Input directory: " << inDir << std::endl;
    std::cout << "Output directory: " << outDir << std::endl;

    // **Fix 2: Check if input directory exists**
    if (!fs::exists(inDir)) {
        std::cerr << "Error: Input directory does not exist: " << inDir << std::endl;
        std::cerr << "Please create the directory and place CSV files in it." << std::endl;
        return 1;
    }

    // **Fix 3: Handle exceptions when creating output directory**
    try {
        fs::create_directories(outDir);
        std::cout << "Output directory created/verified: " << outDir << std::endl;
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Error creating output directory: " << e.what() << std::endl;
        return 1;
    }

    std::vector<std::string> headers;
    std::vector<std::vector<double>> data, cleaned;
    int filesProcessed = 0;
    int filesWithErrors = 0;

    // **Fix 4: Check if directory is empty of CSV files**
    bool hasCSVFiles = false;
    for (const auto& entry : fs::directory_iterator(inDir)) {
        if (entry.path().extension() == ".csv") {
            hasCSVFiles = true;
            break;
        }
    }

    if (!hasCSVFiles) {
        std::cerr << "Warning: No CSV files found in " << inDir << std::endl;
        std::cout << "Please place CSV files in the input directory." << std::endl;
        return 0;
    }

    // **Fix 5: Add detailed error handling and progress information**
    for (const auto& entry : fs::directory_iterator(inDir)) {
        if (entry.path().extension() != ".csv") {
            continue; // Skip non-CSV files
        }

        std::string filename = entry.path().filename().string();
        std::string inPath  = entry.path().string();
        std::string outPath = outDir + "/cleaned_" + filename;

        std::cout << "\nProcessing: " << filename << std::endl;

        try {
            // **Fix 6: Check file size and readability before reading CSV**
            if (!fs::is_regular_file(entry)) {
                std::cerr << "  Skipping: Not a regular file" << std::endl;
                continue;
            }

            auto fileSize = fs::file_size(entry);
            if (fileSize == 0) {
                std::cerr << "  Skipping: Empty file" << std::endl;
                continue;
            }

            std::cout << "  File size: " << fileSize << " bytes" << std::endl;

            // Read CSV
            std::cout << "  Reading CSV..." << std::endl;
            preprocessing::DataCleaner::readCSV(inPath, headers, data);
            
            if (data.empty()) {
                std::cerr << "  Warning: No data rows found" << std::endl;
                continue;
            }

            std::cout << "  Loaded: " << data.size() << " rows, " 
                      << headers.size() << " columns" << std::endl;

            // **Fix 7: Check for sufficient columns for outlier detection**
            if (headers.empty()) {
                std::cerr << "  Error: No headers found" << std::endl;
                filesWithErrors++;
                continue;
            }

            size_t targetColumn = headers.size() - 1; // Last column
            std::cout << "  Removing outliers from column: " << headers[targetColumn] 
                      << " (index " << targetColumn << ")" << std::endl;

            // **Fix 8: Add safety checks for outlier detection**
            if (data.size() < 10) { // Arbitrary threshold for reliable outlier detection
                std::cerr << "  Warning: Too few samples (" << data.size() 
                          << ") for reliable outlier detection" << std::endl;
                cleaned = data; // Copy data without outlier processing
            } else {
                // Perform outlier detection (Z-score threshold 3.0)
                size_t originalSize = data.size();
                cleaned = preprocessing::DataCleaner::removeOutliers(data, targetColumn, 3.0);
                size_t removedCount = originalSize - cleaned.size();
                
                std::cout << "  Outliers removed: " << removedCount 
                          << " (" << (100.0 * removedCount / originalSize) << "%)" << std::endl;
                std::cout << "  Remaining samples: " << cleaned.size() << std::endl;
            }

            // **Fix 9: Check if data remains after cleaning**
            if (cleaned.empty()) {
                std::cerr << "  Warning: All data removed as outliers! Keeping original data." << std::endl;
                cleaned = data; // Revert to original data if all samples were removed
            }

            // Write cleaned data
            std::cout << "  Writing cleaned data..." << std::endl;
            preprocessing::DataCleaner::writeCSV(outPath, headers, cleaned);
            
            std::cout << "  ✓ Successfully cleaned: " << filename 
                      << " -> " << outPath << std::endl;
            
            filesProcessed++;

        } catch (const std::invalid_argument& e) {
            std::cerr << "  ✗ Data format error in " << filename << ": " << e.what() << std::endl;
            filesWithErrors++;
        } catch (const std::out_of_range& e) {
            std::cerr << "  ✗ Data range error in " << filename << ": " << e.what() << std::endl;
            filesWithErrors++;
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error processing " << filename << ": " << e.what() << std::endl;
            filesWithErrors++;
        }
    }

    // **Fix 10: Provide a complete processing summary**
    std::cout << "\n=== Processing Summary ===" << std::endl;
    std::cout << "Files processed successfully: " << filesProcessed << std::endl;
    std::cout << "Files with errors: " << filesWithErrors << std::endl;
    std::cout << "Total files attempted: " << (filesProcessed + filesWithErrors) << std::endl;

    if (filesProcessed > 0) {
        std::cout << "\n✓ Data cleaning completed successfully!" << std::endl;
        std::cout << "Cleaned files are available in: " << outDir << std::endl;
    } else {
        std::cerr << "\n✗ No files were processed successfully." << std::endl;
        return 1; // Indicate failure if no files were processed successfully
    }

    return 0;
}