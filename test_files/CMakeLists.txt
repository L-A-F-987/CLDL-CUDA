cmake_minimum_required(VERSION 3.10)
# 'Google_test' is the subproject name
project(tests LANGUAGES CUDA CXX C)

add_executable(tests
        t.cu
        )


target_link_libraries(tests PRIVATE CLDL)


target_link_libraries(tests PRIVATE Pins)


set(tests_force_shared_crt ON FORCE)

set_property(
        TARGET
        tests
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)

