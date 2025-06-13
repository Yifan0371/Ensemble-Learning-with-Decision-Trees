
#include "boosting/app/RegressionBoostingApp.hpp"
#include <iostream>

int main(int argc, char** argv) {
    try {
        RegressionBoostingOptions opts = parseRegressionCommandLine(argc, argv);
        runRegressionBoostingApp(opts);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}