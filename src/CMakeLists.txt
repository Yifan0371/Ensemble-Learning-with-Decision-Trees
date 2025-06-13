# src/CMakeLists.txt

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Module subdirectories
add_subdirectory(preprocessing)
add_subdirectory(functions/io)
add_subdirectory(pipeline)
add_subdirectory(tree)
add_subdirectory(boosting)  
add_subdirectory(xgboost)   
add_subdirectory(lightgbm)  
add_subdirectory(app)
add_subdirectory(histogram) 