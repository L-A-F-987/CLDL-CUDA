
cmake_minimum_required(VERSION 3.10)
project(CLDL LANGUAGES CXX CUDA)


add_subdirectory(gpu_tests)

add_subdirectory(ecg_tests)

enable_testing()

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
        -rdc=true;)

add_library(CLDL STATIC
        lib/Neuron.cu
        lib/Layer.cu
        lib/Net.cu)


set_target_properties(
		CLDL
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON)

target_include_directories(CLDL
        PUBLIC
        include/)



find_package(Threads)

target_link_libraries(CLDL curand ${CMAKE_THREAD_LIBS_INIT})

