cmake_minimum_required(VERSION 3.10)

project(GeluOMP LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


add_executable(gelu_cuda
    main.cpp
    gelu_cuda.cu
)

target_include_directories(gelu_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
