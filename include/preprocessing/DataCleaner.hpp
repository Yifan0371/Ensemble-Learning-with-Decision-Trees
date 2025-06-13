#pragma once

#include <string>
#include <vector>

namespace preprocessing {

class DataCleaner {
public:
    /**
     * Read CSV file, return data matrix (row-major) and output header column names
     * @param filePath File path
     * @param headers Output: column name list
     * @param data Output: 2D data matrix
     */
    static void readCSV(const std::string& filePath,
                        std::vector<std::string>& headers,
                        std::vector<std::vector<double>>& data);

    /**
     * Write data matrix to CSV file, first row as header
     * @param filePath Output file path
     * @param headers Column name list
     * @param data Data matrix
     */
    static void writeCSV(const std::string& filePath,
                         const std::vector<std::string>& headers,
                         const std::vector<std::vector<double>>& data);

    /**
     * Remove outliers in specified column based on Z-score
     * @param data Input data matrix
     * @param colIndex Column index to check
     * @param zThreshold Z-score threshold
     * @return New matrix after removing outlier rows
     */
    static std::vector<std::vector<double>> removeOutliers(const std::vector<std::vector<double>>& data,
                                                            size_t colIndex,
                                                            double zThreshold = 3.0);

    /**
     * Equal frequency binning
     * @param values Input value vector
     * @param numBins Number of bins
     * @return Bin index for each value
     */
    static std::vector<int> equalFrequencyBinning(const std::vector<double>& values,
                                                   int numBins);

    /**
     * Bin in two dimensions first, then remove outliers
     * @param data Input data matrix
     * @param colX First binning dimension column index
     * @param colY Second binning dimension column index
     * @param numBins Number of bins
     * @param zThreshold Z-score threshold
     * @return New matrix after removing outliers
     */
    static std::vector<std::vector<double>> removeOutliersByBinning(const std::vector<std::vector<double>>& data,
                                                                    size_t colX,
                                                                    size_t colY,
                                                                    int numBins,
                                                                    double zThreshold = 3.0);
};

} // namespace preprocessing