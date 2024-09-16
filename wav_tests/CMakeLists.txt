cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CUDA CXX C)


add_executable(wav_test
        wav_tests.cu
)

target_link_libraries(wav_test PRIVATE CLDL)

set(tests_force_shared_crt ON FORCE)

set_property(
        TARGET
        gpu_testing
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

