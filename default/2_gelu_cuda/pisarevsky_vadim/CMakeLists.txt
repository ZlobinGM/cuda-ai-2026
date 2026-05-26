cmake_minimum_required(VERSION 3.18)
project(2_gelu_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(mimalloc CONFIG)

set(the_target 2_gelu_cuda)

add_executable(${the_target} gelu_cuda.cu gelu_cuda.h)

set_target_properties(${the_target} PROPERTIES
    CUDA_SEPARATE_COMPILATION ON
)

target_compile_definitions(${the_target} PRIVATE ADD_TEST)

if (mimalloc_FOUND)
    target_compile_definitions(${the_target} PRIVATE HAVE_MIMALLOC)
    target_link_libraries(${the_target} PRIVATE mimalloc)
endif()
