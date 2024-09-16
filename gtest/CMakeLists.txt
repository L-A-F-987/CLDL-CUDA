cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(gtest LANGUAGES CUDA CXX)

add_subdirectory(lib)
add_executable(Google_Tests_run
        allTests.cu
        NeuronTest.cu
        LayerTest.cu
        NetTest.cu
        )

target_link_libraries(Google_Tests_run PRIVATE CLDL gtest gtest_main)

set(gtest_force_shared_crt ON FORCE)

set_property(
        TARGET
        Google_Tests_run
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 'lib' is the folder with Google Test sources
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})



