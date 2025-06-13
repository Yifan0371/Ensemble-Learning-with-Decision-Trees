

add_library(RegressionBoosting_lib

    loss/IRegressionLoss.cpp
    

    strategy/GradientRegressionStrategy.cpp
    

    trainer/GBRTTrainer.cpp

    dart/UniformDartStrategy.cpp

    app/RegressionBoostingApp.cpp
)

target_include_directories(RegressionBoosting_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(RegressionBoosting_lib PUBLIC
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
)
