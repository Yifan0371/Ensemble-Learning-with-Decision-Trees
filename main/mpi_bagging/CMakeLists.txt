
# =============================================================================
# main/mpi_bagging/CMakeLists.txt
# =============================================================================
find_package(MPI REQUIRED)

add_executable(MPIBaggingMain
    main.cpp
    ${PROJECT_SOURCE_DIR}/src/tree/ensemble/MPIBaggingTrainer.cpp
)

target_include_directories(MPIBaggingMain PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(MPIBaggingMain PRIVATE
    DecisionTree_lib
    DataIO_lib
    DataSplit_lib
    MPI::MPI_CXX
)

target_compile_options(MPIBaggingMain PRIVATE
    ${MPI_CXX_COMPILE_FLAGS}
)
set_target_properties(MPIBaggingMain PROPERTIES
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
)

install(TARGETS MPIBaggingMain RUNTIME DESTINATION bin)