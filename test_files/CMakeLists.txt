cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CUDA CXX C)

add_executable(ecg_test
        ecg_test.cu
        )

add_executable(gpu_testing
        gpu_tests.cu
)

target_link_libraries(ecg_test PRIVATE CLDL)
target_link_libraries(gpu_testing PRIVATE CLDL)



set(tests_force_shared_crt ON FORCE)

set_property(
        TARGET
        gpu_testing
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

        set_property(
                TARGET
                ecg_test
                PROPERTY
                CUDA_RESOLVE_DEVICE_SYMBOLS ON)

