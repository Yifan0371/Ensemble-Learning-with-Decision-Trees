

if(ENABLE_MPI AND MPI_CXX_FOUND)
    # MPI Bagging library as STATIC library
    add_library(MPIBagging_lib STATIC
        ensemble/MPIBaggingTrainer.cpp
    )
    
    target_include_directories(MPIBagging_lib PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${MPI_CXX_INCLUDE_DIRS}
    )
    
    # Link dependencies
    target_link_libraries(MPIBagging_lib PUBLIC
        DecisionTree_lib          # Includes BaggingTrainer
        HistogramOptimized_lib
        MPI::MPI_CXX
    )
    
    # OpenMP support for hybrid parallelism
    if(OpenMP_CXX_FOUND)
        target_link_libraries(MPIBagging_lib PUBLIC OpenMP::OpenMP_CXX)
    endif()
    
    # Compiler flags for MPI
    target_compile_options(MPIBagging_lib PRIVATE
        ${MPI_CXX_COMPILE_FLAGS}
    )
    
    # Performance optimizations
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(MPIBagging_lib PRIVATE
            -O3
            -march=native
            -funroll-loops
            -ftree-vectorize
        )
    endif()
    
    # Install MPI library
    install(TARGETS MPIBagging_lib
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    
    message(STATUS "MPI Bagging support enabled")
endif()

add_library(DecisionTree_lib

    criterion/MSECriterion.cpp
    criterion/MAECriterion.cpp
    criterion/HuberCriterion.cpp
    criterion/QuantileCriterion.cpp
    criterion/LogCoshCriterion.cpp
    criterion/PoissonCriterion.cpp
    
  
    finder/ExhaustiveSplitFinder.cpp
    finder/RandomSplitFinder.cpp
    finder/QuartileSplitFinder.cpp
    finder/HistogramEWFinder.cpp       
    finder/HistogramEQFinder.cpp        
    finder/AdaptiveEWFinder.cpp         
    finder/AdaptiveEQFinder.cpp         
    
    
    pruner/NoPruner.cpp
    pruner/MinGainPrePruner.cpp
    pruner/CostComplexityPruner.cpp
    pruner/ReducedErrorPruner.cpp
    
    
    trainer/SingleTreeTrainer.cpp
    
    
    ensemble/BaggingTrainer.cpp
)

target_include_directories(DecisionTree_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)


target_link_libraries(DecisionTree_lib PUBLIC
    HistogramOptimized_lib             
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(DecisionTree_lib PUBLIC OpenMP::OpenMP_CXX)

endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(DecisionTree_lib PRIVATE
        -O3                   
        -march=native         
        -funroll-loops         
        -ftree-vectorize      
        -ffast-math           
        # 移除verbose优化报告
    )
endif()


if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(DecisionTree_lib PRIVATE
        /O2                   
        /arch:AVX2            
        /fp:fast              
    )
endif()

