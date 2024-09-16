cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CUDA CXX C)


add_executable(ecg_test
        ecg_test.cu
)


target_link_libraries(ecg_test PRIVATE CLDL)


set(tests_force_shared_crt ON FORCE)

set_property(
        TARGET
        ecg_test
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

