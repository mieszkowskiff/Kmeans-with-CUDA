# Minimum required version of CMake
cmake_minimum_required(VERSION 3.18)

# Project name and specify that it's a CUDA project
project(KMeans LANGUAGES CUDA)

# Set the CUDA standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    main.cu
    ./lib/data_generator.cu
    ./lib/data_generator.h
)



# Add the CUDA source file
add_executable(KMeans ${SOURCES})

# Specify any required CUDA options if needed
set_target_properties(KMeans PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
