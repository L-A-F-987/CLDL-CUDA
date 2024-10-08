cmake_minimum_required(VERSION 3.10)

project(tests LANGUAGES CUDA CXX C)


add_executable(gpu_testing
        gpu_tests.cu
)

add_executable(testing_input_prop
        testing_individual_elements_of_gpu_code.cu
)

add_executable(gpu_testing_different_sizes
        gpu_tests_different_sizes.cu
)

add_executable(cublas_testing
        cublas_testing.cu
)


target_link_libraries(gpu_testing PRIVATE CLDL)
target_link_libraries(testing_input_prop PRIVATE CLDL)
target_link_libraries(gpu_testing_different_sizes PRIVATE CLDL)
target_link_libraries(cublas_testing -lcublas -lcurand)


set(tests_force_shared_crt ON FORCE)

set_property(
        TARGET
        gpu_testing
        PROPERTY
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)



