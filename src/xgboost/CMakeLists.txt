
add_library(XGBoost_lib
    
    loss/XGBoostLossFactory.cpp
    
    finder/XGBoostSplitFinder.cpp
    
   
    trainer/XGBoostTrainer.cpp
    
   
    app/XGBoostApp.cpp
)

target_include_directories(XGBoost_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(XGBoost_lib PUBLIC
    DecisionTree_lib
    RegressionBoosting_lib  
    DataIO_lib
    DataSplit_lib
)
