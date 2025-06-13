
add_library(HistogramOptimized_lib
    PrecomputedHistograms.cpp
)

target_include_directories(HistogramOptimized_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(HistogramOptimized_lib PUBLIC OpenMP::OpenMP_CXX)
   
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HistogramOptimized_lib PRIVATE
        -O3                   
        -march=native         
        -funroll-loops        
        -ffast-math          
        -ftree-vectorize     
        -fno-tree-loop-distribute-patterns 
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(HistogramOptimized_lib PRIVATE
        /O2                    
        /arch:AVX2            
        /fp:fast              
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(HistogramOptimized_lib PRIVATE
        -Wall                  
        -Wno-unused-parameter  
        -Wno-unused-variable   
        -Wno-sign-compare     
    )
endif()


option(ENABLE_HISTOGRAM_PROFILING "Enable histogram performance profiling" OFF)
if(ENABLE_HISTOGRAM_PROFILING)
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_PROFILING)
endif()

option(ENABLE_HISTOGRAM_MEMORY_OPTIMIZATION "Enable aggressive memory optimization" ON)
if(ENABLE_HISTOGRAM_MEMORY_OPTIMIZATION)
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_MEMORY_OPT)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(HistogramOptimized_lib PUBLIC HISTOGRAM_DEBUG)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(HistogramOptimized_lib PRIVATE -g -O1)
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(HistogramOptimized_lib PRIVATE -flto)
        target_link_options(HistogramOptimized_lib PRIVATE -flto)
    endif()
endif()

target_compile_features(HistogramOptimized_lib PUBLIC cxx_std_17)